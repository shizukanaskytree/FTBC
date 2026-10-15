"""CPU check of E23 (EXPERIMENT_PLAN.md sections 6-7; adapted from the E22 CPU check). No CUDA. Writes cache/cpu-check.json.
Checks: (1) the four child configs differ from their E22 templates only in the intended keys (inputs, outputs, freeze, earlier splits, counts,
sizes, the release keys for R34; E22's seeds and tpp_neuron removed) and keep the H100 UUID; (2) the inputs: counts, the exclusion chain on E22
and disjointness by ID from the earlier splits listed as exclusion sources, including the release rows; (3) the E17-E22 frozen files are
unchanged; (4) the job driver: chain commands, the R34 chain then the VGG chain (never concurrent), the sanity gate, one chain's failure leaves
the other running, the environment reaches both chains, the samplers write (gpumem with seven fields); (5) the memwatch exit hook; (6)
check_memory's descriptive rule (initial review B3: OOMs compared with the recorded release-default OOM arms); (7) the dry freeze resolves every
entry and the staged manifest covers every workspace entry; (8) gpurun --dry-run accepts the submission command; (9) the runner and the
acceptance use the same release-default T list and the same library out-of-memory signatures, and the VGG runner has none; (10)
tools/unit_checks.py (timing harness, the out-of-memory basis and the acceptance helpers) passes in a subprocess without CUDA."""
import ast, hashlib, importlib.util, json, os, subprocess, sys, tempfile, time
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; C1 = E.parent
H100 = 'GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


rep = {'checks': {}}
# (1) configs
INTENDED = {'calibration_manifest', 'calibration_manifest_sha256', 'fresh_manifest', 'fresh_manifest_sha256', 'freeze_manifest', 'output',
            'previous_fresh_manifests', 'previous_calibration_manifests', 'fresh_count', 'dev_count', 'sizes', 'seeds', 'tpp_neuron',
            'release_n', 'release_batch', 'release_seed', 'small_n', 'ftbc_batch', 'draw_n'}
for model in ['r34', 'vgg']:
    for which in ['sanity', 'main']:
        a = json.loads((C1 / f'e22/config/{model}-{which}.json').read_text()); b = json.loads((E / f'config/{model}-{which}.json').read_text())
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}; assert diff <= INTENDED, sorted(diff - INTENDED)
        assert 'seeds' not in b and 'tpp_neuron' not in b and b['fresh_count'] == 16 and b['dev_count'] == 8
        assert b['allowed_uuids'] == [H100] and b['freeze_manifest'] == str(E / 'protocol-freeze.json')
        assert b['output'] == f'/data7/projects/ftbc/runs/c1pre-e23-20261010/{model}-{which}-v1' and not Path(b['output']).exists()
        assert b['sizes'] == ([32, 96] if which == 'main' else [4, 12]) and b['sizes'][0] < b['sizes'][1] == b['small_n'] == 3 * b['ftbc_batch'] < b['draw_n'] == (128 if which == 'main' else 16)
        assert b['fresh_manifest'] == str(E / 'input/fresh-development.json') and b['calibration_manifest'] == str(E / 'input/calibration-rows.json')
        assert b['fresh_manifest_sha256'] == sha(b['fresh_manifest']) and b['calibration_manifest_sha256'] == sha(b['calibration_manifest'])
        assert b['previous_fresh_manifests'] == a['previous_fresh_manifests'] + [str(C1 / 'e22/input/fresh-development.json')]
        assert b['previous_calibration_manifests'] == a['previous_calibration_manifests'] + [str(C1 / 'e22/input/calibration-rows.json')]
        if model == 'r34':
            assert b['release_n'] == 3 and b['release_batch'] == (200 if which == 'main' else 4) and isinstance(b['release_seed'], int)
        else:
            assert not any(k.startswith('release') for k in b) and b['eccv_n'] == b['eccv_batch'] * b['eccv_batches'] < b['draw_n']
        rep['checks'][f'config_{model}_{which}'] = sorted(diff)
pol = json.loads(Path('/etc/slurm-manager/policy.json').read_text())
assert {g['uuid']: g['type'] for node in pol['nodes'].values() for g in node['gpus']}[H100] == 'h100'
# (2) inputs
prep = json.loads((E / 'input/preparation.json').read_text()); prev22 = json.loads((C1 / 'e22/input/preparation.json').read_text())
assert prep['passed'] and prep['sizes'] == {'fresh': 16, 'C1': 600, 'C2': 128, 'C3': 128}
assert prep['excluded_ids'] == prev22['excluded_ids'] + sum(prev22['sizes'].values()), (prep['excluded_ids'], prev22['excluded_ids'], prev22['sizes'])
for p in [str(C1 / 'e22/input/fresh-development.json'), str(C1 / 'e22/input/calibration-rows.json')]:
    assert prep['exclusion_sources'][p] == sha(p)
fresh = json.loads((E / 'input/fresh-development.json').read_text())['development']; cal = json.loads((E / 'input/calibration-rows.json').read_text())
new = fresh + [r for k in ['C1', 'C2', 'C3'] for r in cal[k]]
old = []
for p in prep['exclusion_sources']:
    d = json.loads(Path(p).read_text())
    if isinstance(d, dict) and ('development' in d or 'C1' in d):
        old += [r for v in d.values() for r in v if isinstance(r, dict) and 'index' in r]
assert len({r['index'] for r in new}) == len(new) and not {r['index'] for r in new} & {r['index'] for r in old}
rep['checks']['inputs'] = {'fresh': len(fresh), 'calibration': {k: len(v) for k, v in cal.items()}, 'excluded_ids': prep['excluded_ids']}
# (3) frozen files unchanged
for name in ['e17', 'e18', 'e19', 'e20', 'e21', 'e22']:
    fl = json.loads((C1 / name / 'protocol-freeze.json').read_text())['files_sha256']
    for p, h in fl.items():
        assert sha(p) == h, f'{name}: {p}'
    rep['checks'][f'frozen_unchanged_{name}'] = len(fl)
assert not (E / 'protocol-freeze.json').exists()
for m in ['r34', 'vgg']:
    for p in ['sanity', 'main']:
        assert not (E / f'{m}-{p}-acceptance.json').exists()
# (4) job driver with stand-in chains
J = load('chk_e23_job', E / 'source/e23_job.py')
for m in ['r34', 'vgg']:
    ch = J.chain(m); assert [x[0] for x in ch] == [f'{m}-sanity', f'{m}-sanity-acceptance', f'{m}-main']
    assert ch[0][1][2:] == [str(E / f'source/e23_{m}_runner.py'), '--config', str(E / f'config/{m}-sanity.json')]
    assert ch[1][1][2:] == [str(E / f'source/e23_{m}_accept.py'), 'sanity'] and ch[2][1][2:] == [str(E / f'source/e23_{m}_runner.py'), '--config', str(E / f'config/{m}-main.json')]
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    tmp = Path(tmp); (tmp / 'slurm').mkdir(); (tmp / 'memwatch').mkdir()
    (tmp / 'memwatch/sitecustomize.py').write_bytes((E / 'memwatch/sitecustomize.py').read_bytes())
    py = sys.executable
    def standin(m):
        stamp = f"import os,time,json,sys; print(json.dumps({{'m': '{m}', 'start': time.time(), 'alloc': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), 'memstats': os.environ.get('PAR_MEMSTATS'), 'site': 'sitecustomize' in sys.modules}}), flush=True)"
        acc = (f"import json; open('{tmp}/{m}-sanity-acceptance.json','w').write(json.dumps({{'passed': True, 'cuda_initialized': False, 'scope': 'sanity only'}}))"
               if m == 'r34' else "pass")                     # the VGG stand-in leaves no acceptance file: its main must be skipped
        noise = "; print(json.dumps({'stage': 'scored', 'arm': 'x'}), flush=True); print('{\"stage\": \"scor', flush=True)"   # runner-like and cut progress lines
        return [(f'{m}-sanity', [py, '-c', stamp + noise + '; time.sleep(2)']), (f'{m}-sanity-acceptance', [py, '-c', acc]),
                (f'{m}-main', [py, '-c', stamp + '; time.sleep(1)'])]
    J.E = tmp; J.chain = standin; os.environ['SLURM_JOB_ID'] = 'chk'
    try:
        J.main(); raise AssertionError('e23_job did not report the skipped VGG main')
    except SystemExit as e:
        assert "'vgg': 3" in str(e) and "'r34': 0" in str(e), e
    def standin2(m):                                   # r34: the sanity run exits nonzero; vgg: the acceptance exists but did not pass
        acc = f"import json; open('{tmp}/{m}-sanity-acceptance.json','w').write(json.dumps({{'passed': False, 'cuda_initialized': False, 'scope': 'sanity only'}}))"
        first = [py, '-c', 'import sys; sys.exit(5)'] if m == 'r34' else [py, '-c', 'pass']
        return [(f'{m}-sanity', first), (f'{m}-sanity-acceptance', [py, '-c', acc if m == 'vgg' else 'raise SystemExit(9)']), (f'{m}-main', [py, '-c', 'raise SystemExit(7)'])]
    for x in [tmp / 'r34-sanity-acceptance.json', tmp / 'vgg-sanity-acceptance.json']:
        x.unlink(missing_ok=True)
    J.chain = standin2; os.environ['SLURM_JOB_ID'] = 'chk2'
    try:
        J.main(); raise AssertionError('e23_job did not report the failed chains')
    except SystemExit as e:
        assert "'r34': 5" in str(e) and "'vgg': 3" in str(e), e
    l2 = (tmp / 'slurm/r34-chk2.out').read_text(); assert 'r34-sanity-acceptance' not in l2 and 'r34-main' not in l2, l2
    assert 'vgg-main' not in (tmp / 'slurm/vgg-chk2.out').read_text()
    def standin3(m):                                   # both chains pass; VGG starts only after R34 finished
        stamp = lambda w: f"import json,time; print(json.dumps({{'m': '{m}', 'w': '{w}', 'start': time.time()}}), flush=True); time.sleep(1); print(json.dumps({{'m': '{m}', 'w': '{w}', 'end': time.time()}}), flush=True)"
        acc = f"import json; open('{tmp}/{m}-sanity-acceptance.json','w').write(json.dumps({{'passed': True, 'cuda_initialized': False, 'scope': 'sanity only'}}))"
        return [(f'{m}-sanity', [py, '-c', stamp('sanity')]), (f'{m}-sanity-acceptance', [py, '-c', acc]), (f'{m}-main', [py, '-c', stamp('main')])]
    for x in [tmp / 'r34-sanity-acceptance.json', tmp / 'vgg-sanity-acceptance.json']:
        x.unlink(missing_ok=True)
    J.chain = standin3; os.environ['SLURM_JOB_ID'] = 'chk3'
    J.main()                                           # sequential by default; both chains pass: no SystemExit
    ev = {m: [json.loads(l) for l in (tmp / f'slurm/{m}-chk3.out').read_text().splitlines() if l.startswith('{"m"')] for m in ['r34', 'vgg']}
    r34_end = max(x['end'] for x in ev['r34'] if 'end' in x); vgg_start = min(x['start'] for x in ev['vgg'] if 'start' in x)
    assert len(ev['r34']) == len(ev['vgg']) == 4 and vgg_start >= r34_end, ev
    del os.environ['SLURM_JOB_ID']
    # check_memory on the real chain logs of the first stand-in run, with records written as the hook would write them
    CMr = load('chk_e23_check_memory_real', E / 'source/check_memory.py')
    real = [{'pid': 1, 'argv': [str(E / f'source/e23_{m}_runner.py'), '--config', str(E / f'config/{m}-{w}.json')], 'unix_time': 0, 'cuda_initialized': True, 'num_ooms': 0,
             'num_alloc_retries': 0, 'max_memory_reserved': 2 ** 30, 'max_memory_allocated': 2 ** 29, 'alloc_conf': 'expandable_segments:True', 'slurm_job_id': 'chk'}
            for m, w in [('r34', 'sanity'), ('r34', 'main'), ('vgg', 'sanity')]]
    (tmp / 'slurm/memstats-chk.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in real))
    rr = CMr.check('chk', slurm=tmp / 'slurm'); assert rr['memory_state_known'] and rr['clean'] and rr['started_runners'] == ['r34-main', 'r34-sanity', 'vgg-sanity'], rr
    lr, lv = (tmp / 'slurm/r34-chk.out').read_text().splitlines(), (tmp / 'slurm/vgg-chk.out').read_text().splitlines()
    sr = [json.loads(l) for l in lr if l.startswith('{"m"')]; sv = [json.loads(l) for l in lv if l.startswith('{"m"')]
    assert len(sr) == 2 and len(sv) == 1, (lr, lv)                           # r34: sanity and main ran; vgg: only sanity
    assert sv[0]['start'] >= sr[1]['start'] + 0.9 and all(x['alloc'] == 'expandable_segments:True' and x['site'] and x['memstats'] == str(tmp / 'slurm/memstats-chk.jsonl') for x in sr + sv)   # sequential: VGG after R34's main
    gm = (tmp / 'slurm/gpumem-chk.csv').read_text().splitlines(); assert gm and gm[0].split(',')[1].strip() == H100 and len(gm[0].split(',')) == 7, gm[:2]
    full = [l.split(',') for l in gm]; full = [f for f in full if len(f) == 7 and f[2].strip().isdigit() and f[3].strip().isdigit()]   # follow-up 1 B6: a cut last line is skipped
    assert rr['device_samples'] == len(full) > 0 and rr['device_total_gib'], (rr['device_samples'], len(full))
    ga = (tmp / 'slurm/gpuapps-chk.csv').read_text().splitlines(); assert not any('Field' in l and 'is not a valid' in l for l in ga), ga[:2]
    rep['checks']['job_driver_standin'] = {'vgg_start_after_r34_main_start_s': sv[0]['start'] - sr[1]['start'], 'vgg_main_skipped_without_acceptance': True, 'device_samples': len(gm),
                                           'nonzero_sanity_stops_its_chain_only': True, 'failed_acceptance_skips_main': True, 'sequential_mode_vgg_after_r34': True,
                                           'check_memory_on_real_logs': {'started': rr['started_runners'], 'known': rr['memory_state_known'], 'clean': rr['clean']}}
# (5) memwatch hook
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    tmp = Path(tmp); out = tmp / 'memstats.jsonl'
    env = dict(os.environ, PYTHONPATH=str(E / 'memwatch'), PAR_MEMSTATS=str(out), CUDA_VISIBLE_DEVICES='')
    mock = ("import torch; torch.cuda.is_initialized = lambda: True; torch.cuda.memory_stats = lambda: {'num_ooms': 0, 'num_alloc_retries': 2}; "
            "torch.cuda.max_memory_reserved = lambda: 5; torch.cuda.max_memory_allocated = lambda: 3")
    r1 = subprocess.run([sys.executable, '-B', '-c', mock, '--config', 'x.json'], env=env, capture_output=True, text=True); assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run([sys.executable, '-B', '-c', 'import torch'], env=env, capture_output=True, text=True); assert r2.returncode == 0, r2.stderr
    stub = tmp / 'e23_r34_runner.py'                   # a runner that never initializes CUDA, with a forked child (as a DataLoader worker)
    stub.write_text("import multiprocessing as mp, torch\nif __name__ == '__main__':\n    mp.set_start_method('fork'); p = mp.Process(target=print); p.start(); p.join()\n")
    r3 = subprocess.run([sys.executable, '-B', str(stub), '--config', 'y.json'], env=env, capture_output=True, text=True); assert r3.returncode == 0, r3.stderr
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 2 and recs[0]['cuda_initialized'] and recs[0]['num_ooms'] == 0 and recs[0]['num_alloc_retries'] == 2 and recs[0]['argv'][1:] == ['--config', 'x.json'], recs
    assert recs[1]['cuda_initialized'] is False and recs[1]['num_ooms'] is None and recs[1]['argv'] == [str(stub), '--config', 'y.json'], recs
    rep['checks']['memwatch_hook'] = {'records_for_cuda_process': 1, 'records_for_non_cuda_process': 0, 'records_for_non_cuda_runner': 1, 'records_for_forked_child': 0}
# (6) check_memory rule (descriptive; initial review B3, follow-up 1 B5)
CM = load('chk_e23_check_memory', E / 'source/check_memory.py')
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    tmp = Path(tmp)
    def rec(m, which, ooms, retries=0, cuda=True):
        return {'pid': 1, 'argv': [str(E / f'source/e23_{m}_runner.py'), '--config', str(E / f'config/{m}-{which}.json')], 'unix_time': 0, 'cuda_initialized': cuda,
                'num_ooms': ooms if cuda else None, 'num_alloc_retries': retries if cuda else None, 'max_memory_reserved': 2 ** 30 if cuda else None,
                'max_memory_allocated': 2 ** 29 if cuda else None, 'alloc_conf': 'expandable_segments:True', 'slurm_job_id': 'x'}
    def logs(job, started):
        for m in ['r34', 'vgg']:
            (tmp / f'{m}-{job}.out').write_text(''.join(json.dumps({'stage': s}) + '\n' for s in started if s.startswith(m)))
    ALL = ['r34-sanity', 'r34-main', 'vgg-sanity', 'vgg-main']
    inc = lambda oom=None, done=None: {'cost_json': True, 'oom_arms': oom or {}, 'completed_arms': done or {}}
    two = {'r34-main': inc({'F0rel-T8-r1': 1, 'F0rel-T8-r2': 3})}                     # an allocator OOM and a library failure after 3 caught ones
    clean4 = lambda **o: [rec(m, w, o.get(f'{m}-{w}', 0)) for m in ['r34', 'vgg'] for w in ['sanity', 'main']]
    cases = [('a', ALL, clean4(), {}, (True, True)),                                                                     # all clean
             ('b', ALL[:3], [rec('r34', 'sanity', 0), rec('r34', 'main', 0), rec('vgg', 'sanity', 0)], {}, (True, True)),     # vgg main never started
             ('c', ALL, [rec('r34', 'sanity', 0), rec('r34', 'main', 0), rec('vgg', 'sanity', 0)], {}, (False, False)),       # started, no record
             ('d', ALL, clean4(**{'r34-main': 4}), two, (True, True)),                                                    # OOMs = the recorded per-arm increases
             ('e', ALL, clean4(**{'r34-main': 5}), two, (True, False)),                                                   # one OOM outside every arm
             ('f', ALL, clean4(**{'r34-main': 2}), {'r34-main': inc({'F0rel-T8-r1': 0}, {'P0-96-T8-r2': 2})}, (True, False)),  # OOM arm without count; completed arm with OOMs
             ('g', ALL, clean4(**{'vgg-main': 1}), {}, (True, False)),                                                    # an OOM with nothing recorded
             ('h', ALL, [rec(m, w, 0, retries=3) for m in ['r34', 'vgg'] for w in ['sanity', 'main']], {}, (True, True)),    # retries do not decide (near_full set)
             ('i', ALL, [rec(m, w, 0, cuda=(m, w) != ('vgg', 'main')) for m in ['r34', 'vgg'] for w in ['sanity', 'main']], {}, (True, True)),  # never used the GPU
             ('j', ALL[:3], [rec('r34', 'sanity', 0), rec('r34', 'main', 0), rec('r34', 'main', 0), rec('vgg', 'sanity', 0)], {}, (False, False)),  # duplicate record
             ('k', ALL, clean4() + [dict(rec('r34', 'main', 0), argv=['/x/other.py', '--config', 'y.json'])], {}, (False, False))]           # unexpected record
    for job, started, recs_, recorded, want in cases:
        logs(job, started); (tmp / f'memstats-{job}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in recs_))
        res = CM.check(job, slurm=tmp, recorded=recorded); assert (res['memory_state_known'], res['clean']) == want, (job, res)
        if job == 'd': assert not res['ooms_outside_arms'] and not res['completed_arms_with_ooms'] and not res['oom_arms_without_allocator_count'], res
        if job == 'e': assert res['ooms_outside_arms'] == {'r34-main': 1}, res
        if job == 'f': assert res['oom_arms_without_allocator_count'] == {'r34-main': ['F0rel-T8-r1']} and res['completed_arms_with_ooms'] == {'r34-main': {'P0-96-T8-r2': 2}}, res
        if job == 'g': assert res['ooms_outside_arms'] == {'vgg-main': 1}, res
        if job == 'h': assert res['near_full'], res
        if job == 'i': assert res['gpu_used']['vgg-main'] is False and 'vgg-main' not in res['num_ooms'], res
    cfg = tmp / 'cfg.json'; outd = tmp / 'run'; outd.mkdir(); cfg.write_text(json.dumps({'output': str(outd)}))
    assert CM.recorded_ooms(cfg) == {'cost_json': False, 'oom_arms': {}, 'completed_arms': {}}                          # no cost.json yet
    (outd / 'cost.json').write_text(json.dumps({'P0-32-T8-r1': {'T': 8, 'num_ooms_increase': 0}, 'P0-96-T8-r1': {'T': 8, 'num_ooms_increase': 2},
                                                'F0rel-T8-r1': {'status': 'out_of_memory', 'num_ooms_increase': 1}, 'F0rel-T4-r1': {'status': 'completed', 'num_ooms_increase': 0}}))
    assert CM.recorded_ooms(cfg) == {'cost_json': True, 'oom_arms': {'F0rel-T8-r1': 1}, 'completed_arms': {'P0-96-T8-r1': 2}}
    rep['checks']['check_memory_rule'] = {k: v for k, v in zip('abcdefghijk', ['all clean', 'unstarted main ignored', 'started without record: unknown',
                                                                            'OOMs equal to the recorded per-arm increases: clean', 'an OOM outside every arm flagged',
                                                                            'an OOM arm without allocator count and a completed arm with OOMs flagged', 'an OOM with nothing recorded flagged',
                                                                            'retries reported only (near_full set)', 'runner that never used the GPU outside the rule',
                                                                            'duplicate record: unknown', 'unexpected record: unknown'])} | {'recorded_ooms_from_cost_json': True}
# (7) dry freeze
FS = load('chk_e23_freeze_submit', E / 'source/e23_freeze_submit.py')
f = FS.collect(with_review=False)
w = Path('/data3/snn_prjs/ftbc_neural_networks')
inside = [p for p in f if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w)]
rep['checks']['dry_freeze'] = {'entries': len(f), 'staged': len(inside), 'not_staged_outside_workspace': len(f) - len(inside)}
# (8) gpurun dry run
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    m = Path(tmp) / 'slurm-source-manifest.json'; m.write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): f[p] for p in inside}}) + '\n')
    cmd = FS.command(); i = cmd.index('--source-manifest'); cmd[i + 1] = str(m); dry = cmd[:1] + ['--dry-run'] + cmd[1:]
    r = subprocess.run(dry, capture_output=True, text=True); assert r.returncode == 0, (r.stdout, r.stderr)
assert '--gres=gpu:h100:1' in r.stdout and '--cpus-per-task=16' in r.stdout and '--mem=128G' in r.stdout and f"--time={int(FS.TIME.split(':')[0]) * 60 + int(FS.TIME.split(':')[1])}" in r.stdout, r.stdout[-800:]
rep['checks']['gpurun_dry_run'] = {'command': dry, 'stdout_tail': r.stdout[-1500:]}
# (9) the release-default T list
def const(path, name):
    tree = ast.parse(Path(path).read_text())
    vals = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    return vals
assert const(E / 'source/e23_r34_runner.py', 'RELEASE_TS') == const(E / 'source/e23_r34_accept.py', 'RELEASE_TS') == [[2, 4, 8]]
assert const(E / 'source/e23_vgg_runner.py', 'RELEASE_TS') == [] and const(E / 'source/e23_vgg_accept.py', 'RELEASE_TS') == []
lib = const(E / 'source/e23_r34_runner.py', 'LIB_OOM'); assert lib == const(E / 'source/e23_r34_accept.py', 'LIB_OOM') and len(lib) == 1 and len(lib[0]) == 4, lib   # follow-up 1 B5
rep['checks']['release_ts'] = {'r34_runner_and_acceptance': [2, 4, 8], 'vgg': None, 'library_oom_signatures_equal': list(lib[0])}
# (10) unit checks in a subprocess (CUDA stand-ins stay in that process)
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    u = subprocess.run([sys.executable, '-B', str(E / 'tools/unit_checks.py'), tmp], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert u.returncode == 0, (u.stdout[-2000:], u.stderr[-3000:])
    rep['checks']['unit_checks'] = json.loads(u.stdout.strip().splitlines()[-1])
import torch
names = ['source/e23_r34_runner.py', 'source/e23_vgg_runner.py', 'source/e23_r34_accept.py', 'source/e23_vgg_accept.py', 'source/e23_job.py', 'source/check_memory.py',
         'source/e23_timing.py', 'source/e23_freeze_submit.py', 'source/e23_check_cpu.py', 'source/prepare_inputs.py', 'memwatch/sitecustomize.py', 'record_costs.py',
         'verify_preservation.py', 'EXPERIMENT_PLAN.md', 'preservation-before.json', 'tools/cpu_dryrun.py', 'tools/unit_checks.py'] + [f'config/{m}-{p}.json' for m in ['r34', 'vgg'] for p in ['sanity', 'main']]
rep.update(passed=True, cuda_initialized=torch.cuda.is_initialized(), python=sys.version, torch=torch.__version__, files_sha256={str(E / n): sha(E / n) for n in names})
assert not rep['cuda_initialized']
(E / 'cache').mkdir(exist_ok=True); (E / 'cache/cpu-check.json').write_text(json.dumps(rep, indent=2) + '\n')
print(json.dumps({k: v for k, v in rep['checks'].items() if k != 'gpurun_dry_run'}, indent=1))
