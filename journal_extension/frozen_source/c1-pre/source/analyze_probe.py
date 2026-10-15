#!/usr/bin/env python3
"""CPU-only, fixed-input diagnostics for the Route C development probe.

Inputs are trusted project-produced captures, never datasets or checkpoints.
The discrete ordering bound is a relaxation over the complete count grid. It is
not the old support-aware J_order, an optimum of FTBC, or a classification bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1
ROUNDING_TOLERANCE = 2e-6
GAP_ROUNDING_TOLERANCE = 1e-12


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name}: analysis accepts CPU tensors only")
        value = value.detach().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: non-finite capture")
    return result


def channel_vector(value: Any, channels: int, name: str) -> np.ndarray:
    value = array(value, name).reshape(-1)
    if value.size == 1:
        value = np.repeat(value, channels)
    if value.shape != (channels,):
        raise ValueError(f"{name}: expected scalar or [{channels}], got {value.shape}")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Non-JSON metadata type: {type(value).__name__}")


def normalize_family(metadata: dict) -> str:
    family = str(metadata.get("interface", metadata.get("family", ""))).lower()
    if family.startswith("parallel"):
        return "parallel"
    if family.startswith("serial"):
        return "serial"
    raise ValueError(f"Unknown interface/family {family!r}")


def order_relation(inputs: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray]:
    """Return i <= j relation and its defining coordinates, in stored-input FP64."""
    if family == "parallel":
        coordinates = inputs.mean(axis=0, keepdims=True)
    elif family == "serial":
        coordinates = inputs.cumsum(axis=0)
    else:
        raise ValueError(family)
    relation = (coordinates[:, :, None] <= coordinates[:, None, :]).all(axis=0)
    np.fill_diagonal(relation, False)
    return relation, coordinates


def pair_bound(target: np.ndarray, T: int, relation: np.ndarray) -> dict:
    """An explicitly non-optimal greedy disjoint matching; O(N^2 T)."""
    n = len(target)
    grid = np.arange(T + 1, dtype=np.float64) / T
    costs = (target[:, None] - grid[None, :]) ** 2
    individual = costs.min(axis=1)
    # suffix_cost[j,k] = min_{l>=k} (l/T-a_j)^2.
    suffix_cost = np.minimum.accumulate(costs[:, ::-1], axis=1)[:, ::-1]
    constrained = (costs[:, None, :] + suffix_cost[None, :, :]).min(axis=2)
    raw_penalties = constrained - individual[:, None] - individual[None, :]
    min_raw = float(raw_penalties.min())
    if min_raw < -GAP_ROUNDING_TOLERANCE:
        raise ValueError(f"Negative pair relaxation gap {min_raw}; no silent clipping")
    # A tiny negative subtraction error is recorded below before being zeroed.
    penalties = np.maximum(raw_penalties, 0.0)
    conflict = relation & (target[:, None] > target[None, :])
    positive = conflict & (penalties > GAP_ROUNDING_TOLERANCE)
    row, col = np.nonzero(positive)
    edges = sorted(zip(row.tolist(), col.tolist()),
                   key=lambda ij: (-float(penalties[ij]), ij[0], ij[1]))
    used: set[int] = set()
    selected = []
    for i, j in edges:
        if i in used or j in used:
            continue
        used.update((i, j))
        selected.append({"i": i, "j": j, "penalty_sum_squared": float(penalties[i, j])})
    total = sum(item["penalty_sum_squared"] for item in selected)
    comparable = relation | relation.T
    tied = relation & relation.T
    possible = n * (n - 1) // 2
    ordered_n = int(np.triu(comparable, 1).sum())
    tied_n = int(np.triu(tied, 1).sum())
    return {
        "independent_grid_floor": float(individual.mean()),
        "greedy_disjoint_pair_excess_lower_bound": float(total / n),
        "grid_plus_pair_lower_bound": float(individual.mean() + total / n),
        "pair_selection": "descending weight, then i,j; greedy matching, not maximum matching",
        "possible_unordered_pairs": possible,
        "comparable_unordered_pairs": ordered_n,
        "comparable_pair_fraction": ordered_n / possible if possible else None,
        "equal_coordinate_unordered_pairs": tied_n,
        "teacher_order_conflict_directed_pairs": int(conflict.sum()),
        "positive_grid_penalty_directed_pairs": int(positive.sum()),
        "positive_grid_penalty_equal_coordinate_pairs": int((positive & tied).sum()),
        "selected_disjoint_pairs": selected,
        "matched_observations": len(used),
        "matching_observation_fraction": len(used) / n,
        "minimum_unclipped_pair_gap": min_raw,
        "tiny_negative_pair_gap_count": int((raw_penalties < 0).sum()),
        "gap_rounding_tolerance": GAP_ROUNDING_TOLERANCE,
        "relaxation": "all counts 0..T; shared amplitude; no reachable-support restriction",
    }


def floating_point_audit(inputs: np.ndarray, bits: np.ndarray, relation: np.ndarray,
                         family: str, theta: float, initial: float,
                         bias: np.ndarray) -> dict:
    """Audit observed counts rather than assuming the exact-arithmetic order holds."""
    T, n = inputs.shape
    cumulative = bits.cumsum(axis=0)
    audit_counts = cumulative if family == "serial" else cumulative[-1:]
    violations = relation[None, :, :] & (audit_counts[:, :, None] > audit_counts[:, None, :])
    entries = np.argwhere(violations)
    if family == "serial":
        previous = np.concatenate((np.zeros((1, n)), cumulative[:-1]), axis=0)
        margins = initial + (inputs - bias[:, None]).cumsum(axis=0) - theta * (previous + 1)
    else:
        ranks = np.arange(1, T + 1, dtype=np.float64)[:, None]
        margins = (T * inputs.mean(axis=0)[None, :] + initial) / ranks + bias[:, None] - theta
    fp64_bits = margins >= 0
    mismatches = fp64_bits != bits
    # This envelope is a reporting heuristic, not a floating-point error proof.
    scale = max(1.0, abs(initial) + float(np.abs(inputs).sum(axis=0).max())
                + float(np.abs(bias).sum()) + T * abs(theta))
    envelope = 16 * np.finfo(np.float32).eps * scale
    near = np.abs(margins) <= envelope
    witnesses = []
    for step, i, j in entries[:8]:
        witnesses.append({"prefix": int(step + 1) if family == "serial" else T,
                          "i": int(i), "j": int(j),
                          "count_i": int(audit_counts[step, i]),
                          "count_j": int(audit_counts[step, j]),
                          "min_abs_threshold_margin_i": float(np.abs(margins[:, i]).min()),
                          "min_abs_threshold_margin_j": float(np.abs(margins[:, j]).min()),
                          "either_observation_near_boundary": bool(near[:, [i, j]].any())})
    return {
        "observed_order_violation_count": int(violations.sum()),
        "observed_order_violation_witnesses": witnesses,
        "observed_output_satisfies_checked_relation": not bool(violations.any()),
        "fp64_threshold_reconstruction_mismatch_count": int(mismatches.sum()),
        "fp64_mismatch_near_boundary_count": int((mismatches & near).sum()),
        "minimum_abs_reconstructed_threshold_margin": float(np.abs(margins).min()),
        "near_boundary_fraction": float(near.mean()),
        "near_boundary_heuristic_absolute_tolerance": float(envelope),
        "boundary_tolerance_is_formal_error_bound": False,
        "reconstruction_arithmetic": "FP64 on captured local inputs and actual previous spike counts",
        "scope": "Compatibility of this observed output; no certification for all future FP32 corrections",
    }


def analyze_layer(layer: dict, family: str, T: int) -> dict:
    teacher = array(layer["teacher"], "teacher")
    proxy = array(layer["proxy"], "proxy")
    spikes = array(layer["spikes"], "spikes")
    inputs = array(layer["snn_input"], "snn_input")
    if teacher.ndim != 2:
        raise ValueError(f"teacher must be [N,C], got {teacher.shape}")
    n, c = teacher.shape
    if n < 1 or c < 1 or proxy.shape != (n, c) or spikes.shape != (T, n, c) or inputs.shape != (T, n, c):
        raise ValueError("teacher/proxy/input/spike dimensions do not agree")
    alpha = channel_vector(layer["alpha"], c, "alpha")
    theta = channel_vector(layer["theta"], c, "theta")
    initial = channel_vector(layer["initial_mem"], c, "initial_mem")
    bias = array(layer["bias"], "bias")
    if bias.shape != (T, c):
        raise ValueError(f"bias must be [T,C], got {bias.shape}")
    if (alpha <= 0).any() or (theta <= 0).any():
        raise ValueError("positive frozen spike amplitude and threshold required")
    if family == "serial" and not np.array_equal(alpha, theta):
        raise ValueError("this serial theorem requires subtractive reset by the emitted theta amplitude")
    scaled_spikes = spikes / alpha[None, None, :]
    bits = np.rint(scaled_spikes)
    bit_error = float(np.abs(scaled_spikes - bits).max())
    if bit_error > ROUNDING_TOLERANCE or (bits < 0).any() or (bits > 1).any():
        raise ValueError(f"spikes must be 0 or alpha per step; maximum rounding error {bit_error}")
    a, p, s = teacher / alpha, proxy / alpha, spikes.mean(axis=0) / alpha
    ap, ps = p - a, s - p
    mse = float(np.square(s - a).mean())
    ap_mse = float(np.square(ap).mean())
    ps_mse = float(np.square(ps).mean())
    cross = float(2 * (ap * ps).mean())
    channel_results = []
    channel_ids = json_safe(layer.get("channels", list(range(c))))
    if len(channel_ids) != c:
        raise ValueError("channels identity length differs from sampled channel count")
    for ch in range(c):
        relation, _ = order_relation(inputs[:, :, ch], family)
        result = pair_bound(a[:, ch], T, relation)
        audit = floating_point_audit(inputs[:, :, ch], bits[:, :, ch], relation,
                                     family, theta[ch], initial[ch], bias[:, ch])
        observed_mse = float(np.square(s[:, ch] - a[:, ch]).mean())
        result.update({"sampled_channel_index": ch, "original_channel": channel_ids[ch],
                       "alpha": float(alpha[ch]), "theta": float(theta[ch]),
                       "normalized_local_mse": observed_mse,
                       "normalized_signed_mean_residual": float((s[:, ch] - a[:, ch]).mean()),
                       "normalized_proxy_teacher_mse": float(np.square(ap[:, ch]).mean()),
                       "normalized_spike_proxy_mse": float(np.square(ps[:, ch]).mean()),
                       "normalized_twice_cross_term": float(2 * (ap[:, ch] * ps[:, ch]).mean()),
                       "observed_mse_minus_grid_and_pair_bound": observed_mse - result["grid_plus_pair_lower_bound"],
                       "floating_point_audit": audit})
        result["lower_bound_applies_to_this_observed_count_vector"] = bool(
            audit["observed_output_satisfies_checked_relation"]
            and observed_mse + 1e-10 >= result["grid_plus_pair_lower_bound"])
        channel_results.append(result)
    mean_bound = float(np.mean([x["greedy_disjoint_pair_excess_lower_bound"] for x in channel_results]))
    mean_floor = float(np.mean([x["independent_grid_floor"] for x in channel_results]))
    return {
        "status": "valid_capture", "observations": n, "sampled_channels": c,
        "spatial_positions": json_safe(layer.get("spatial_positions")),
        "normalization": "each channel divided by its positive frozen alpha; equal observation/channel weight",
        "normalized_local_mse": mse,
        "normalized_proxy_teacher_mse": ap_mse,
        "normalized_spike_proxy_mse": ps_mse,
        "normalized_twice_cross_term": cross,
        "mse_identity_residual": mse - ap_mse - ps_mse - cross,
        "normalized_signed_mean_residual": float((s - a).mean()),
        "channel_abs_mean_residual_average": float(np.abs((s - a).mean(axis=0)).mean()),
        "independent_grid_floor": mean_floor,
        "greedy_disjoint_pair_excess_lower_bound": mean_bound,
        "grid_plus_pair_lower_bound": mean_floor + mean_bound,
        "comparable_pair_fraction": float(np.mean([x["comparable_pair_fraction"] for x in channel_results])) if n > 1 else None,
        "observed_order_violation_count": sum(x["floating_point_audit"]["observed_order_violation_count"] for x in channel_results),
        "channels_compatible_with_observed_lower_bound": sum(x["lower_bound_applies_to_this_observed_count_vector"] for x in channel_results),
        "maximum_spike_to_binary_rounding_error": bit_error,
        "channels": channel_results,
    }


AGGREGATE_KEYS = ("normalized_local_mse", "normalized_proxy_teacher_mse",
                  "normalized_spike_proxy_mse", "normalized_twice_cross_term",
                  "normalized_signed_mean_residual", "channel_abs_mean_residual_average",
                  "independent_grid_floor", "greedy_disjoint_pair_excess_lower_bound",
                  "grid_plus_pair_lower_bound")


def analyze_capture(capture: dict) -> dict:
    metadata = capture["metadata"]
    family = normalize_family(metadata)
    T = int(metadata["T"])
    if T < 1 or T != metadata["T"]:
        raise ValueError("positive integer T required")
    if metadata.get("spike_encoding", "amplitude") != "amplitude":
        raise ValueError("capture spikes must store emitted amplitudes")
    layers = {}
    for name, layer in capture["layers"].items():
        try:
            layers[name] = analyze_layer(layer, family, T)
        except (ValueError, KeyError, IndexError) as exc:
            layers[name] = {"status": "invalid_capture", "error": str(exc)}
    valid = [x for x in layers.values() if x["status"] == "valid_capture"]
    total = sum(x["observations"] * x["sampled_channels"] for x in valid)
    aggregate = {k: sum(x[k] * x["observations"] * x["sampled_channels"] for x in valid) / total
                 for k in AGGREGATE_KEYS} if total else {}
    aggregate["weighting"] = "each saved observation/channel equally weighted across valid layers"
    aggregate["valid_layer_count"] = len(valid)
    aggregate["invalid_layer_count"] = len(layers) - len(valid)
    aggregate["observed_order_violation_count"] = sum(x["observed_order_violation_count"] for x in valid)
    aggregate["not_a_network_global_lower_bound"] = True
    return {"metadata": json_safe(metadata), "family": family, "T": T,
            "status": "complete" if layers and len(valid) == len(layers) else "incomplete",
            "aggregate": aggregate, "layers": layers}


def paired_comparison(plain: dict, spatial: dict, plain_result: dict, spatial_result: dict) -> dict:
    errors = []
    for key in ("image_ids", "sample_axes", "sample_order", "source_checkpoint_sha256",
                "source_quantization_level", "proxy_quantization_level"):
        if json_safe(plain["metadata"].get(key)) != json_safe(spatial["metadata"].get(key)):
            errors.append(f"metadata.{key} differs")
    if (plain_result["T"], plain_result["family"]) != (spatial_result["T"], spatial_result["family"]):
        errors.append("T or interface differs")
    if set(plain["layers"]) != set(spatial["layers"]):
        errors.append("interface names differ")
    differences = {}
    for name in sorted(set(plain["layers"]) & set(spatial["layers"])):
        p, s = plain["layers"][name], spatial["layers"][name]
        for key in ("teacher", "alpha", "theta", "initial_mem", "bias"):
            if not np.array_equal(array(p[key], key), array(s[key], key)):
                errors.append(f"{name}.{key} differs")
        for key in ("channels", "spatial_positions"):
            if json_safe(p.get(key)) != json_safe(s.get(key)):
                errors.append(f"{name}.{key} differs")
        pr, sr = plain_result["layers"][name], spatial_result["layers"][name]
        if pr["status"] != "valid_capture" or sr["status"] != "valid_capture":
            errors.append(f"{name} has invalid capture")
        else:
            differences[name] = {key: sr[key] - pr[key] for key in AGGREGATE_KEYS}
    if errors:
        return {"status": "unpaired_or_invalid", "errors": errors, "deltas_withheld": True}
    return {"status": "paired", "difference_direction": "spatial minus plain",
            "aggregate": {key: spatial_result["aggregate"][key] - plain_result["aggregate"][key]
                          for key in AGGREGATE_KEYS}, "layers": differences,
            "interpretation": "Matched exploratory within-checkpoint comparison; no causal/classification conclusion from local loss"}


def load_capture(path: Path) -> dict:
    # weights_only rejects untrusted pickle classes; the runner writes tensor/primitive dictionaries.
    capture = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(capture, dict) or not {"metadata", "layers"} <= capture.keys():
        raise ValueError(f"{path}: expected metadata and layers dictionaries")
    return capture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--spatial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = {"plain": args.plain}
    if args.spatial:
        paths["spatial"] = args.spatial
    captures = {name: load_capture(path) for name, path in paths.items()}
    results = {name: analyze_capture(capture) for name, capture in captures.items()}
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_source_sha256": file_sha256(Path(__file__).resolve()),
        "inputs": {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in paths.items()},
        "status": "complete" if all(x["status"] == "complete" for x in results.values()) else "incomplete",
        "arithmetic": "CPU FP64 analysis of saved FP32 observations; actual captured spikes audited",
        "scope": "Exploratory fixed-local-input activation diagnostics; not independent error sources, not classifier performance attribution",
        "branches": results,
    }
    if "spatial" in captures:
        report["paired_comparison"] = paired_comparison(captures["plain"], captures["spatial"], results["plain"], results["spatial"])
        if report["paired_comparison"]["status"] != "paired":
            report["status"] = "incomplete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Preserve existing analysis: choose a new output path: {args.output}")
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "branches": {name: value["aggregate"] for name, value in results.items()}}, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
