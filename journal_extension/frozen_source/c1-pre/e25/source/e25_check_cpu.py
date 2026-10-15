"""CPU check of E25 (EXPERIMENT_PLAN.md sections 7-8; adapted from the E24 CPU check). No CUDA. Writes cache/cpu-check.json.
Checks: (1) the eight configs: one key set; per tag the dataset, architecture, checkpoint (hash), level, interfaces and class count; main and
sanity sizes; the frozen solver settings; the A100 allowlist; (2) the inputs: counts, disjointness of dev-L and C1-C3 by ID and content,
the exclusions recomputed from the sources, the two halves of dev-L both cover every class; (3) the E17-E22 frozen files are unchanged; (4)
the job driver with stand-in children (chain order, the sanity gate, a failing sanity run stops the chain, the sampler writes); (5) the dry
freeze resolves every entry and the staged manifest covers every workspace entry; (6) gpurun --dry-run accepts the four submission commands."""
import hashlib, importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; C1 = E.parent
A100 = ['GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894']
H100 = ['GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211']
TAGS = {'c10-vgg16': ('cifar10', 'vgg16', 10, 4, 15), 'c10-resnet20': ('cifar10', 'resnet20', 10, 4, 19),
        'c100-vgg16': ('cifar100', 'vgg16', 100, 8, 15), 'c100-resnet20': ('cifar100', 'resnet20', 100, 8, 19)}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


rep = {'checks': {}}
# (1) configs
keys = None
for tag, (ds, arch, ncls, level, n_if) in TAGS.items():
    for which in ['sanity', 'main']:
        c = json.loads((E / f'config/{tag}-{which}.json').read_text()); keys = keys or set(c); assert set(c) == keys, (tag, which, set(c) ^ keys)
        assert (c['dataset'], c['arch'], c['num_classes'], c['source_level'], c['interfaces']) == (ds, arch, ncls, level, n_if), tag
        assert sha(c['source_checkpoint']) == c['source_checkpoint_sha256'] and c['checkpoint_key'] == ('model' if tag == 'c100-resnet20' else None), tag
        assert c['allowed_uuids'] == A100 + H100 and c['freeze_manifest'] == str(E / 'protocol-freeze.json') and c['sanity'] == (which == 'sanity')
        assert c['alphas'] == [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2] and c['Ts'] == [2, 4, 8] and c['score_roles'] == ['fresh']
        assert (c['bound_ratio'], c['bisection_steps'], c['ftbc_alpha'], c['release_alpha'], c['release_seed']) == (0.5, 32, 0.4, 0.4, 2026101030)
        assert c['gpu_mem_cap_gib'] == 15                                         # initial review N-e: below the 16 GiB shard (checked against GPU_MEM below)
        assert c['output'] == f'/data7/projects/ftbc/runs/c1pre-e25-20261010/{tag}-{which}-v1' and not Path(c['output']).exists()
        assert c['rows_manifest'] == str(E / f'input/{ds}-rows.json') and sha(c['rows_manifest']) == c['rows_manifest_sha256'] and sha(c['preparation']) == c['preparation_sha256']
        want = (10000, 600, 96, 32, [32, 96, 256], 96, 600, 200) if which == 'main' else (16, 24, 12, 4, [4, 12, 16], 12, 12, 4)
        assert tuple(c[k] for k in ['dev_count', 'draw_n', 'small_n', 'ftbc_batch', 'p0_sizes', 'm0_size', 'release_n', 'release_batch']) == want, (tag, which)
        for k in ['e13_runner', 'e18_runner', 'step_solver', 'serial_solver', 'calibration_export', 'calibration_export_manifest']:
            assert Path(c[k]).is_relative_to(C1) and Path(c[k]).exists(), k
    rep['checks'][f'config_{tag}'] = 'ok'
pol = json.loads(Path('/etc/slurm-manager/policy.json').read_text())
types = {g['uuid']: g['type'] for node in pol['nodes'].values() for g in node['gpus']}; assert all(types[u] == 'a100' for u in A100) and all(types[u] == 'h100' for u in H100)
# (2) inputs
prep = json.loads((E / 'input/preparation.json').read_text()); assert prep['passed'] and prep['sizes'] == {'development': 10000, 'C1': 600, 'C2': 600, 'C3': 600}
for p, h in prep['exclusion_sources'].items():
    assert sha(p) == h, p
A = load('chk_e25_accept', E / 'source/e25_accept.py')
for ds in ['cifar10', 'cifar100']:
    man = json.loads((E / f'input/{ds}-rows.json').read_text()); rows = man['development'] + [r for k in ['C1', 'C2', 'C3'] for r in man[k]]
    assert [len(man[k]) for k in ['development', 'C1', 'C2', 'C3']] == [10000, 600, 600, 600]
    assert len({r['index'] for r in rows}) == len(rows) == len({r['sha256'] for r in rows})
    ex = A.excluded_ids(ds, list(prep['exclusion_sources'])); assert not {r['index'] for r in rows} & ex and len(ex) == prep['datasets'][ds]['excluded_ids']
    assert not {r['sha256'] for r in rows} & A.excluded_content(ds, prep, ex)
    ncls = 10 if ds == 'cifar10' else 100; dev = man['development']
    assert len({r['label'] for r in dev[:5000]}) == ncls == len({r['label'] for r in dev[5000:]})
    rep['checks'][f'inputs_{ds}'] = {'excluded_ids': len(ex), 'rows': len(rows)}
# (2b) follow-up 1 R4: the CIFAR-100 development role of calibration-sample-stability-freeze.json (2000 images, not an exclusion source) is the
# cifar100_r20_t2 tuning list of protocol-inputs.json (an exclusion source), checked by content, and it is disjoint from every E25 row.
def stability_development_role():
    import pickle
    import numpy as np
    W = Path('/data3/snn_prjs/ftbc_neural_networks')
    stab = json.loads((W / 'ccfa-workfiles/experiments/ftbc-neural-networks/calibration-sample-stability-freeze.json').read_text())['settings']['cifar100']['development_identity']
    pin = W / 'ccfa-workfiles/experiments/ftbc-neural-networks/protocol-inputs.json'; assert str(pin) in prep['exclusion_sources']
    tuning = json.loads(pin.read_text())['settings']['cifar100_r20_t2']['tuning']
    rel, h = next(iter(prep['datasets']['cifar100']['train_files'].items())); f = W / 'code/parallel-conversion/downloaded_dataset' / rel; assert sha(f) == h['sha256']
    with f.open('rb') as fh:
        data = np.asarray(pickle.load(fh, encoding='bytes')[b'data'], dtype=np.uint8)
    assert data.shape == (50000, 3072)
    chw = [hashlib.sha256(data[i].tobytes()).hexdigest() for i in tuning]
    hwc = [hashlib.sha256(np.ascontiguousarray(data[i].reshape(3, 32, 32).transpose(1, 2, 0)).tobytes()).hexdigest() for i in tuning]
    ph = stab['pixel_hashes_in_id_order']; assert len(ph) == 2000 == len(tuning) == len(set(tuning)) == stab['images']
    layout = 'CHW' if set(chw) == set(ph) else ('HWC' if set(hwc) == set(ph) else None); assert layout, 'the stability development role is not the tuning list by content'
    same_order = (chw if layout == 'CHW' else hwc) == ph
    man = json.loads((E / 'input/cifar100-rows.json').read_text()); ours = [r for k in ['development', 'C1', 'C2', 'C3'] for r in man[k]]
    assert not {r['sha256'] for r in ours} & set(chw) and not {int(r['index']) for r in ours} & set(tuning)       # E25 row hashes are CHW
    return {'images': len(ph), 'pixel_hash_layout': layout, 'same_order_as_tuning_list': same_order, 'disjoint_from_e25_rows_by_id_and_content': True}
rep['checks']['stability_freeze_cifar100_development_role'] = stability_development_role()
# (3) frozen files unchanged
for name in ['e17', 'e18', 'e19', 'e20', 'e21', 'e22']:
    fl = json.loads((C1 / name / 'protocol-freeze.json').read_text())['files_sha256']
    for p, h in fl.items():
        assert sha(p) == h, f'{name}: {p}'
    rep['checks'][f'frozen_unchanged_{name}'] = len(fl)
assert not (E / 'protocol-freeze.json').exists()
# (4) job driver with stand-in children
J = load('chk_e25_job', E / 'source/e25_job.py')
for tag in TAGS:
    ch = J.chain(tag); assert [x[0] for x in ch] == [f'{tag}-sanity', f'{tag}-sanity-acceptance', f'{tag}-main']
    assert ch[0][1][2:] == [str(E / 'source/e25_runner.py'), '--config', str(E / f'config/{tag}-sanity.json')]
    assert ch[1][1][2:] == [str(E / 'source/e25_accept.py'), tag, 'sanity'] and ch[2][1][2:] == [str(E / 'source/e25_runner.py'), '--config', str(E / f'config/{tag}-main.json')]
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    tmp = Path(tmp); (tmp / 'slurm').mkdir(); py = sys.executable; real_chain = J.chain
    def standin(passed, first_code=0):
        def ch(tag):
            acc = f"import json; open('{tmp}/{tag}-sanity-acceptance.json','w').write(json.dumps({{'passed': {passed}, 'cuda_initialized': False, 'scope': 'sanity only'}}))"
            return [(f'{tag}-sanity', [py, '-c', f'import sys; print("sanity"); sys.exit({first_code})']), (f'{tag}-sanity-acceptance', [py, '-c', acc]),
                    (f'{tag}-main', [py, '-c', 'print("main ran")'])]
        return ch
    J.E = tmp; outcomes = {}
    for job, (passed, first, want) in {'ok': (True, 0, 0), 'gate': (False, 0, 3), 'fail': (True, 5, 5)}.items():
        for x in tmp.glob('*-sanity-acceptance.json'): x.unlink()
        J.chain = standin(passed, first); os.environ['SLURM_JOB_ID'] = job
        try:
            J.main('c10-vgg16'); code = 0
        except SystemExit as e:
            code = int(str(e).rsplit(' ', 1)[-1])
        log = (tmp / f'slurm/c10-vgg16-{job}.out').read_text()
        assert code == want and (('main ran' in log) == (want == 0)), (job, code, log)
        gm = (tmp / f'slurm/gpumem-c10-vgg16-{job}.csv').read_text().splitlines(); outcomes[job] = {'returncode': code, 'main_ran': 'main ran' in log, 'device_samples': len(gm)}
    del os.environ['SLURM_JOB_ID']; J.chain = real_chain
    rep['checks']['job_driver_standin'] = outcomes
# (5) dry freeze
FS = load('chk_e25_freeze_submit', E / 'source/e25_freeze_submit.py')
assert all(json.loads((E / f'config/{t}-{w}.json').read_text())['gpu_mem_cap_gib'] < FS.GPU_MEM for t in TAGS for w in ['sanity', 'main'])
rep['checks']['manuscript_snapshot_current'] = FS.manuscript_current()                           # recorded here; asserted at freeze (N-g)
f = FS.collect(with_review=False)
w = Path('/data3/snn_prjs/ftbc_neural_networks')
inside = [p for p in f if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w)]
rep['checks']['dry_freeze'] = {'entries': len(f), 'staged': len(inside), 'not_staged_outside_workspace': len(f) - len(inside)}
# (6) gpurun dry runs
dry_out = {}
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    m = Path(tmp) / 'slurm-source-manifest.json'; m.write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): f[p] for p in inside}}) + '\n')
    for tag in TAGS:
        for kind in FS.GPU[tag][0]:                      # every card type allowed for the job (chosen at submission)
            cmd = FS.command(tag, kind); i = cmd.index('--source-manifest'); cmd[i + 1] = str(m); dry = cmd[:1] + ['--dry-run'] + cmd[1:]
            r = subprocess.run(dry, capture_output=True, text=True); assert r.returncode == 0, (tag, kind, r.stdout, r.stderr)
            h, mi, _ = map(int, FS.TIMES[tag].split(':'))
            assert all((u in cmd) == (u in FS.TYPE[kind]) for u in A100 + H100), (tag, kind)       # only the chosen type's UUIDs are passed
            assert f'--gres=shard:{kind}:{FS.GPU_MEM}' in r.stdout and '--cpus-per-task=16' in r.stdout and '--mem=128G' in r.stdout and f'--time={60 * h + mi}' in r.stdout, (tag, kind, r.stdout[-800:])
            dry_out[f'{tag}-{kind}'] = {'command': dry, 'stdout_tail': r.stdout[-600:]}
rep['checks']['gpurun_dry_run'] = dry_out
import torch
names = ['source/e25_runner.py', 'source/e25_accept.py', 'source/cifar_e0.py', 'source/e25_job.py', 'source/e25_freeze_submit.py', 'source/e25_check_cpu.py',
         'source/prepare_inputs.py', 'record_costs.py', 'verify_preservation.py', 'EXPERIMENT_PLAN.md', 'preservation-before.json', 'tools/cpu_dryrun.py',
         'tools/make_configs.py'] + [f'config/{t}-{p}.json' for t in TAGS for p in ['sanity', 'main']]
rep.update(passed=True, cuda_initialized=torch.cuda.is_initialized(), python=sys.version, torch=torch.__version__, files_sha256={str(E / n): sha(E / n) for n in names})
assert not rep['cuda_initialized']
(E / 'cache').mkdir(exist_ok=True); (E / 'cache/cpu-check.json').write_text(json.dumps(rep, indent=2) + '\n')
print(json.dumps({k: v for k, v in rep['checks'].items() if k != 'gpurun_dry_run'}, indent=1))
