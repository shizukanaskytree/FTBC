#!/usr/bin/env python3
"""Bounded Route C E0: frozen-source vs BN-only proxy adaptation.

No FTBC fitting, protected data access, parameter selection or job resubmission.
Run only in the explicitly allowed Slurm allocation. --cpu-check checks imports,
identities and checkpoint loading without reading images or initializing CUDA.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def state_digest(model, names=None):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if names is not None and name not in names:
            continue
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def source_model(config):
    family = config["family"]
    root = Path(config["source_snapshot"][family]).resolve()
    sys.path.insert(0, str(root))
    source = config["source"]
    path = Path(source["checkpoint"])
    if not path.is_absolute():
        path = Path(config["workspace"]) / path
    if sha256(path) != source["checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if family == "parallel":
        modules = importlib.import_module("modules")
        model = importlib.import_module("models.ResNet_QCFS").resnet34_qcfs(
            source["quantization_level"], False, source["num_classes"])
        classes = {"activation": modules.QCFS, "spike": modules.ParaInfNeuron}
        renamed = {}
        for key, value in state.items():
            key = key.removeprefix("module.").replace(".act.", ".relu.")
            if key.endswith(".thresh"):
                key = key[:-len(".thresh")] + ".up"
            if key in renamed:
                raise ValueError("Checkpoint key collision")
            renamed[key] = value
    elif family == "serial":
        layer = importlib.import_module("Models.layer")
        model = importlib.import_module("Models.ResNet").resnet34(source["num_classes"])
        model.set_L(source["quantization_level"])
        classes = {"activation": layer.IF, "spike": layer.IF,
                   "expand": layer.ExpandTemporalDim}
        renamed = {}
        for key, value in state.items():
            key = key.removeprefix("module.").replace(".relu.", ".act.")
            if key.endswith(".up"):
                key = key[:-len(".up")] + ".thresh"
            if key in renamed:
                raise ValueError("Checkpoint key collision")
            renamed[key] = value
    else:
        raise ValueError(f"Unsupported family {family}")
    expected = model.state_dict()
    if set(expected) != set(renamed):
        raise ValueError("Strict source key mismatch")
    for name, value in renamed.items():
        if expected[name].shape != value.shape or expected[name].dtype != value.dtype:
            raise ValueError(f"Source shape/dtype mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite source tensor: {name}")
    model.load_state_dict(renamed, strict=True)
    if any(not torch.equal(model.state_dict()[k], v) for k, v in renamed.items()):
        raise ValueError("Source exact load failed")
    activations = [(name, m) for name, m in model.named_modules()
                   if isinstance(m, classes["activation"])]
    if len(activations) != 33:
        raise ValueError(f"Expected 33 QCFS interfaces, got {len(activations)}")
    for name, module in activations:
        threshold = module.up if family == "parallel" else module.thresh
        if threshold.numel() != 1 or not (threshold > 0).all():
            raise ValueError(f"Invalid scalar source threshold: {name}")
    return freeze(model), classes


def make_proxy(source, config, classes):
    proxy = copy.deepcopy(source)
    for module in proxy.modules():
        if isinstance(module, classes["activation"]):
            if config["family"] == "parallel":
                module.t = config["timesteps"]
            else:
                module.L = config["timesteps"]
    return freeze(proxy)


def make_snn(proxy, config, classes):
    model = copy.deepcopy(proxy).cpu()
    T = config["timesteps"]
    if config["family"] == "parallel":
        def replace(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, classes["activation"]):
                    replacement = classes["spike"](T, th=child.up.item(), init_mem=0.5, dim=child.dim)
                    # Pre-shape the standard zero tc correction so strict reload is exact.
                    replacement.output_bias_correction = torch.zeros(T, 1, child.in_ch, 1, 1)
                    parent._modules[name] = replacement
                else:
                    replace(child)
        replace(model)
    else:
        model.T = T
        for module in model.modules():
            if isinstance(module, (classes["activation"], classes["expand"])):
                module.T = T
            if isinstance(module, classes["activation"]) and hasattr(module, "time_based_bias"):
                del module.time_based_bias
    return freeze(model).cuda()


def logits(model, images, config, spiking=False):
    T = config["timesteps"]
    if spiking and config["family"] == "parallel":
        out = model(images.repeat(T, 1, 1, 1)).reshape(T, images.shape[0], -1).mean(0)
    elif spiking:
        out = model(images).mean(0)
    else:
        out = model(images)
    if out.shape != (images.shape[0], 1000) or not torch.isfinite(out).all():
        raise ValueError("Invalid or nonfinite logits")
    return out


class WhitelistImages(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        row = self.rows[position]
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Image identity mismatch for training ID {row['index']}")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), int(row["index"])


def loader(rows, config, batch_size, shuffle=False):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(WhitelistImages(rows), batch_size=batch_size, shuffle=shuffle,
                      num_workers=config.get("num_workers", 4), pin_memory=True,
                      generator=generator, drop_last=False)


def gpu_guard(config):
    job = os.environ.get("SLURM_JOB_ID")
    cgroup = Path("/proc/self/cgroup").read_text()
    if not job or "slurm" not in cgroup.lower():
        raise RuntimeError("This runner requires a Slurm cgroup and SLURM_JOB_ID")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one Slurm-visible GPU")
    torch.cuda.set_device(0)
    token = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    query = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"
    ], text=True)
    own = [line.split(",")[1].strip() for line in query.splitlines()
           if line.split(",")[0].strip() == str(os.getpid())]
    if set(own) != {config["allowed_uuid"]}:
        raise RuntimeError(f"PID GPU UUID mismatch: observed {own}")
    props = torch.cuda.get_device_properties(0)
    del token
    return {"slurm_job_id": job, "pid": os.getpid(), "cgroup": cgroup,
            "gpu_uuid": own[0], "name": props.name, "total_memory": props.total_memory,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


@torch.no_grad()
def evaluate(model, rows, config, spiking=False):
    model.eval()
    outputs, targets, identities = [], [], []
    for images, labels, ids in loader(rows, config, config.get("eval_batch_size", 4)):
        outputs.append(logits(model, images.cuda(non_blocking=True), config, spiking).cpu())
        targets.append(labels)
        identities.append(ids)
    outputs, targets, identities = torch.cat(outputs), torch.cat(targets), torch.cat(identities)
    predictions = outputs.argmax(1)
    return {"ids": identities, "labels": targets, "logits": outputs, "predictions": predictions,
            "metrics": {"n": len(rows), "correct": int((predictions == targets).sum()),
                        "accuracy_percent": float((predictions == targets).double().mean() * 100),
                        "cross_entropy": float(nn.functional.cross_entropy(outputs.double(), targets))}}


def bn_names(model):
    mutable, parameters = set(), []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            mutable.update(name + "." + key for key in module.state_dict())
            parameters.extend(module.parameters(recurse=False))
    return mutable, parameters


def bn_gamma_counts(model):
    per_layer = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.affine:
            gamma = module.weight.detach().cpu()
            per_layer[name] = {"total": gamma.numel(), "nonpositive": int((gamma <= 0).sum()),
                               "negative": int((gamma < 0).sum()), "zero": int((gamma == 0).sum())}
    return {"total": sum(v["total"] for v in per_layer.values()),
            "nonpositive": sum(v["nonpositive"] for v in per_layer.values()),
            "per_layer": per_layer}


def lr_at(step, total, settings):
    warmup = settings["warmup_steps"]
    if step < warmup:
        factor = settings["warmup_start_factor"] + (1 - settings["warmup_start_factor"]) * step / (warmup - 1)
    else:
        fraction = (step - warmup + 1) / (total - warmup)
        factor = settings["final_factor"] + (1 - settings["final_factor"]) * (1 + math.cos(math.pi * fraction)) / 2
    return settings["learning_rate"] * factor


def fit_bn(proxy, rows, config, output, progress):
    freeze(proxy)
    mutable, params = bn_names(proxy)
    immutable = set(proxy.state_dict()) - mutable
    initial_immutable = state_digest(proxy, immutable)
    initial_mutable = state_digest(proxy, mutable)
    initial_gammas = bn_gamma_counts(proxy)
    for module in proxy.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train()
            module.requires_grad_(True)
    settings = config["fit"]
    optimizer = torch.optim.AdamW(params, lr=settings["learning_rate"],
                                 betas=tuple(settings["betas"]), eps=settings["eps"],
                                 weight_decay=settings["weight_decay"])
    batches = loader(rows, config, settings["batch_size"], shuffle=False)
    records, seen = [], []
    for step, (images, labels, ids) in enumerate(batches):
        lr = lr_at(step, len(batches), settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        prediction = logits(proxy, images.cuda(non_blocking=True), config)
        loss = nn.functional.cross_entropy(prediction, labels.cuda(non_blocking=True))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(params, settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params):
            raise ValueError("Nonfinite BN parameter")
        seen.extend(ids.tolist())
        record = {"step": step + 1, "n": len(ids), "loss": float(loss),
                  "learning_rate": lr, "preclip_gradient_norm": float(norm)}
        records.append(record)
        if (step + 1) % 8 == 0 or step + 1 == len(batches):
            progress("fit_bn", completed_steps=step + 1, total_steps=len(batches), last=record)
            atomic_json(output / "training-trace.json", records)
    if seen != [int(r["index"]) for r in rows] or len(set(seen)) != len(rows):
        raise ValueError("Fit did not cover the exact whitelist once in declared order")
    freeze(proxy)
    if state_digest(proxy, immutable) != initial_immutable:
        raise ValueError("Frozen non-BN model state changed")
    atomic_torch(output / "training-state.pt", {"optimizer": optimizer.state_dict(),
                 "seen_ids": seen, "steps": len(records), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate()})
    return {"steps": len(records), "images_seen": len(seen), "one_pass_verified": True,
            "bn_parameters": sum(p.numel() for p in params), "mutable_state_entries": sorted(mutable),
            "immutable_state_sha256": initial_immutable, "immutable_state_unchanged": True,
            "bn_state_before_sha256": initial_mutable, "bn_state_after_sha256": state_digest(proxy, mutable),
            "bn_gamma_before": initial_gammas, "bn_gamma_after": bn_gamma_counts(proxy)}


@torch.no_grad()
def capture(model, rows, config, classes, spiking=False):
    captured = {}
    handles = []
    selected_class = classes["spike"] if spiking else classes["activation"]
    T = config["timesteps"] if spiking else 1
    for name, module in model.named_modules():
        if not isinstance(module, selected_class):
            continue
        def hook(layer, inputs, result, name=name):
            x, y = inputs[0].detach(), result.detach()
            if x.ndim != 4 or y.shape != x.shape:
                raise ValueError(f"Unexpected interface shape at {name}")
            B, C, H, W = x.shape[0] // T, *x.shape[1:]
            channels = torch.linspace(0, C - 1, min(C, config["diagnostics"]["channels"])).long()
            positions = torch.linspace(0, H * W - 1, min(H * W, config["diagnostics"]["positions"])).long()
            def sample(value):
                value = value.reshape(T, B, C, H * W)
                return value.index_select(2, channels.to(value.device)).index_select(
                    3, positions.to(value.device)).permute(0, 1, 3, 2).reshape(T, -1, len(channels)).cpu()
            item = captured.setdefault(name, {"input": [], "output": [], "channels": channels,
                                              "spatial_positions": positions, "spatial_shape": [H, W]})
            item["input"].append(sample(x))
            item["output"].append(sample(y))
            threshold = (layer.v_threshold if config["family"] == "parallel" and spiking
                         else layer.up if config["family"] == "parallel" else layer.thresh)
            threshold = torch.as_tensor(threshold).detach().cpu().reshape(-1)
            if threshold.numel() != 1:
                raise ValueError("E0 expects a fixed scalar amplitude per interface")
            item["theta"] = threshold.expand(len(channels)).clone()
        handles.append(module.register_forward_hook(hook))
    try:
        for images, _, _ in loader(rows, config, config.get("eval_batch_size", 4)):
            logits(model, images.cuda(non_blocking=True), config, spiking)
    finally:
        for handle in handles:
            handle.remove()
    for item in captured.values():
        item["input"] = torch.cat(item["input"], dim=1)
        item["output"] = torch.cat(item["output"], dim=1)
        if not torch.isfinite(item["input"]).all() or not torch.isfinite(item["output"]).all():
            raise ValueError("Nonfinite diagnostic interface")
    return captured


def save_diagnostics(source_capture, proxy_capture, snn_capture, rows, config, branch, path):
    if set(source_capture) != set(proxy_capture) or set(source_capture) != set(snn_capture):
        raise ValueError("Diagnostic interfaces differ")
    layers = {}
    for name in source_capture:
        A, P, S = source_capture[name], proxy_capture[name], snn_capture[name]
        for field in ("channels", "spatial_positions", "theta"):
            if not torch.equal(A[field], P[field]) or not torch.equal(A[field], S[field]):
                raise ValueError(f"Diagnostic alignment mismatch: {name}/{field}")
        theta = S["theta"]
        layers[name] = {"teacher": A["output"][0], "proxy": P["output"][0],
                        "teacher_input": A["input"][0], "proxy_input": P["input"][0],
                        "spikes": S["output"], "snn_input": S["input"],
                        "alpha": theta, "theta": theta, "initial_mem": 0.5 * theta,
                        "bias": torch.zeros(config["timesteps"], len(theta)),
                        "channels": S["channels"], "spatial_positions": S["spatial_positions"],
                        "spatial_shape": S["spatial_shape"]}
    atomic_torch(path, {"metadata": {"schema_version": 1, "family": config["family"],
                  "interface": config["family"], "T": config["timesteps"], "branch": branch,
                  "spike_encoding": "amplitude", "source_quantization_level": config["source"]["quantization_level"],
                  "proxy_quantization_level": config["timesteps"], "teacher": "frozen_original_Q8",
                  "sample_order": "image_then_flat_spatial_position", "image_ids": [r["index"] for r in rows],
                  "config_sha256": config["_config_sha256"], "source_checkpoint_sha256": config["source"]["checkpoint_sha256"]},
                  "layers": layers})


def runtime_biases(model, config, classes):
    if config["family"] == "parallel":
        return {name: m.output_bias_correction.detach().cpu().clone()
                for name, m in model.named_modules() if isinstance(m, classes["spike"])}
    return {name: [b.detach().cpu().clone() for b in m.time_based_bias]
            for name, m in model.named_modules() if isinstance(m, classes["spike"])}


@torch.no_grad()
def checkpoint_and_reload(model, proxy, images, config, classes, path):
    before = logits(model, images, config, True).cpu()
    biases = runtime_biases(model, config, classes)
    if any(torch.count_nonzero(b) for values in biases.values()
           for b in (values if isinstance(values, list) else [values])):
        raise ValueError("E0 must contain zero temporal calibration corrections")
    thresholds = {name: float(m.v_threshold if config["family"] == "parallel" else m.thresh.detach().item())
                  for name, m in model.named_modules() if isinstance(m, classes["spike"])}
    atomic_torch(path, {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       "runtime_biases": biases, "source": config["source"],
                       "interface_thresholds": thresholds,
                       "family": config["family"], "timesteps": config["timesteps"],
                       "config_sha256": config["_config_sha256"]})
    saved = torch.load(path, map_location="cpu", weights_only=False)
    restored = make_snn(proxy, config, classes)
    restored.load_state_dict(saved["state_dict"], strict=True)
    if config["family"] == "parallel":
        modules = dict(restored.named_modules())
        for name, threshold in saved["interface_thresholds"].items():
            modules[name].v_threshold = threshold
    if config["family"] == "serial":
        modules = dict(restored.named_modules())
        for name, values in saved["runtime_biases"].items():
            modules[name].time_based_bias = [value.cuda() for value in values]
    after = logits(restored, images, config, True).cpu()
    if not torch.equal(before, after):
        raise ValueError(f"Exact deployment reload failed: max delta {(before - after).abs().max()}")
    return {"checkpoint_sha256": sha256(path), "strict_state_reload": True,
            "logits_bitwise_equal": True, "max_logit_delta": 0.0,
            "zero_correction_verified": True, "model_state_sha256": state_digest(restored)}


def validate_config(config):
    if config.get("freeze_manifest"):
        frozen = json.loads(Path(config["freeze_manifest"]).read_text())
        identities = frozen["files_sha256"]
        if not identities:
            raise ValueError("Empty protocol freeze")
        workspace = Path(config["workspace"]).resolve()
        for relative, expected in identities.items():
            path = (workspace / relative).resolve()
            if not path.is_relative_to(workspace) or sha256(path) != expected:
                raise ValueError(f"Frozen file identity mismatch: {relative}")
    if (config["family"], config["timesteps"]) not in (("parallel", 2), ("serial", 4)):
        raise ValueError("E0 setting outside bounded protocol")
    if config["source"]["quantization_level"] != 8 or config["source"]["num_classes"] != 1000:
        raise ValueError("E0 source mismatch")
    path = Path(config["dataset_manifest"])
    if config.get("dataset_manifest_sha256") and sha256(path) != config["dataset_manifest_sha256"]:
        raise ValueError("Input manifest hash mismatch")
    roles = json.loads(path.read_text())
    all_ids = []
    for role, n in (("fit", 4096), ("development", 512), ("diagnostic", 64)):
        rows = roles[role]
        if len(rows) != n:
            raise ValueError(f"Expected exactly {n} {role} images")
        for row in rows:
            if not Path(row["path"]).is_absolute() or not 0 <= int(row["label"]) < 1000:
                raise ValueError("Invalid whitelisted training image")
            all_ids.append(int(row["index"]))
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Input data roles overlap")
    return roles


def run(config, cpu_check=False):
    started = time.perf_counter()
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
    roles = validate_config(config)
    source, classes = source_model(config)
    original_digest = state_digest(source)
    if cpu_check:
        print(json.dumps({"cpu_check": "passed", "source_state_sha256": original_digest,
                          "interfaces": 33, "data_roles": {k: len(roles[k]) for k in ("fit", "development", "diagnostic")},
                          "images_read": False, "cuda_initialized": torch.cuda.is_initialized()}))
        return
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("progress.json", "result.json", "failure.json")):
        raise RuntimeError("Output contains a previous attempt; use a new run ID/output for a retry")
    atomic_json(output / "config.json", {k: v for k, v in config.items() if not k.startswith("_")})
    stages = {}
    def progress(stage, **extra):
        value = {"stage": stage, "elapsed_seconds": time.perf_counter() - started, **extra}
        atomic_json(output / "progress.json", value)
        print(json.dumps(value, allow_nan=False), flush=True)
    try:
        progress("gpu_guard")
        hardware = gpu_guard(config)
        atomic_json(output / "hardware.json", hardware)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        source = source.cuda()
        proxy = make_proxy(source, config, classes)
        first = next(iter(loader(roles["development"], config, config.get("eval_batch_size", 4))))[0].cuda()
        progress("teacher_development_and_diagnostic")
        t0 = time.perf_counter()
        predictions = {"teacher": evaluate(source, roles["development"], config)}
        source_capture = capture(source, roles["diagnostic"], config, classes)
        torch.cuda.synchronize()
        stages["teacher_evaluation"] = time.perf_counter() - t0
        checks, fit_checks = {}, None
        for branch in ("plain", "spatial"):
            if branch == "spatial":
                progress("fit_bn")
                t0 = time.perf_counter()
                fit_checks = fit_bn(proxy, roles["fit"], config, output, progress)
                torch.cuda.synchronize()
                stages["bn_fit"] = time.perf_counter() - t0
            progress(branch + "_development_and_diagnostic")
            t0 = time.perf_counter()
            predictions[branch + "_proxy"] = evaluate(proxy, roles["development"], config)
            proxy_capture = capture(proxy, roles["diagnostic"], config, classes)
            snn = make_snn(proxy, config, classes)
            predictions[branch + "_snn"] = evaluate(snn, roles["development"], config, True)
            snn_capture = capture(snn, roles["diagnostic"], config, classes, True)
            save_diagnostics(source_capture, proxy_capture, snn_capture, roles["diagnostic"], config,
                             branch, output / ("diagnostics-" + branch + ".pt"))
            checks[branch] = checkpoint_and_reload(snn, proxy, first, config, classes,
                                                   output / ("deployment-" + branch + ".pt"))
            atomic_torch(output / ("proxy-" + branch + ".pt"), {
                "state_dict": {k: v.detach().cpu() for k, v in proxy.state_dict().items()},
                "proxy_quantization_level": config["timesteps"], "family": config["family"],
                "config_sha256": config["_config_sha256"]})
            del snn, proxy_capture, snn_capture
            torch.cuda.synchronize()
            stages[branch + "_evaluation_reload"] = time.perf_counter() - t0
            atomic_torch(output / "predictions.pt", predictions)
        source_unchanged = state_digest(source) == original_digest
        if not source_unchanged:
            raise ValueError("Frozen teacher state changed")
        torch.cuda.synchronize()
        result = {"status": "completed", "scope": "exploratory_C1_pre_E0_BN_only_no_FTBC",
                  "family": config["family"], "T": config["timesteps"], "hardware": hardware,
                  "config_sha256": config["_config_sha256"], "metrics": {k: v["metrics"] for k, v in predictions.items()},
                  "fit_checks": fit_checks, "deployment_checks": checks,
                  "source_state_sha256": original_digest, "source_state_unchanged": source_unchanged,
                  "source_quantization_level": 8, "proxy_quantization_level": config["timesteps"],
                  "data_roles": {k: len(roles[k]) for k in ("fit", "development", "diagnostic")},
                  "protected_evaluation": False, "stages_seconds": stages,
                  "elapsed_seconds_including_setup_and_preprocessing": time.perf_counter() - started,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "preprocessing": "Resize256 bilinear antialias / CenterCrop224 / RGB ToTensor / ImageNet normalize",
                  "fit_precision": "FP32; TF32 disabled; deterministic algorithms; no autocast",
                  "inference_timing_claim": False}
        atomic_json(output / "result.json", result)
        artifacts = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                     for p in output.iterdir() if p.is_file() and p.name not in ("artifacts.json", "progress.json")}
        atomic_json(output / "artifacts.json", artifacts)
        progress("completed", result="result.json")
    except BaseException as exc:
        atomic_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - started,
                    "retained_partial_artifacts": True, "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["_config_sha256"] = sha256(args.config)
    run(config, args.cpu_check)


if __name__ == "__main__":
    main()
