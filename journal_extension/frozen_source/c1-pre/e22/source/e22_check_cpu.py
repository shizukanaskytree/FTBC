"""CPU check of E22 (EXPERIMENT_PLAN.md sections 7-8; adapted from the E20 CPU check). No CUDA. Writes cache/cpu-check.json.
Checks: (1) the four child configs differ from their E20 templates only in the intended keys (inputs, outputs, freeze, earlier splits, draw_n,
seeds, tpp_neuron; E20's sizes and diag_n removed) and keep the H100 UUID; (2) the inputs: counts, the exclusion chain on E21 and disjointness
by ID from the earlier splits listed as exclusion sources (content hashes are checked by prepare_inputs.py, the runners and the acceptances);
(3) the E17-E20 frozen files are unchanged (E21's too once it is frozen); (4) the job driver: chain commands, concurrency, the sanity gate,
one chain's failure leaves the other running, the environment reaches both chains, the samplers write; (5) the memwatch exit hook; (6)
check_memory's rule; (7) the dry freeze resolves every entry and the staged manifest covers every workspace entry; (8) gpurun --dry-run
accepts the submission command; (9) TPP (section 4): the downloaded layer.py and receipt hashes, the copied prob branch against the
downloaded source text (read as text, never imported or executed), tpp_prob_forward against an independent re-implementation on the
project's serial IF class (bitwise, same seed), swap and restore (also on error), seeded reproducibility, and that the swapped forward ignores
the per-step biases."""
import hashlib, importlib.util, json, os, subprocess, sys, tempfile, time
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
            'previous_fresh_manifests', 'previous_calibration_manifests', 'draw_n', 'seeds', 'tpp_neuron', 'sizes', 'diag_n'}
for model in ['r34', 'vgg']:
    for which in ['sanity', 'main']:
        a = json.loads((C1 / f'e20/config/{model}-{which}.json').read_text()); b = json.loads((E / f'config/{model}-{which}.json').read_text())
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}; assert diff <= INTENDED, sorted(diff - INTENDED)
        assert 'sizes' not in b and 'diag_n' not in b and b['seeds'] == [0, 1, 2, 3, 4] and b['tpp_neuron'] == str(E / 'source/tpp_neuron.py')
        assert b['allowed_uuids'] == [H100] and b['freeze_manifest'] == str(E / 'protocol-freeze.json')
        assert b['output'] == f'/data7/projects/ftbc/runs/c1pre-e22-20261010/{model}-{which}-v1' and not Path(b['output']).exists()
        assert b['draw_n'] == (256 if which == 'main' else 16) and b['small_n'] == 3 * b['ftbc_batch'] == (96 if which == 'main' else 12) < b['draw_n']
        assert b['fresh_manifest'] == str(E / 'input/fresh-development.json') and b['calibration_manifest'] == str(E / 'input/calibration-rows.json')
        assert b['fresh_manifest_sha256'] == sha(b['fresh_manifest']) and b['calibration_manifest_sha256'] == sha(b['calibration_manifest'])
        assert b['previous_fresh_manifests'] == a['previous_fresh_manifests'] + [str(C1 / f'e{k}/input/fresh-development.json') for k in (20, 21)]
        assert b['previous_calibration_manifests'] == a['previous_calibration_manifests'] + [str(C1 / f'e{k}/input/calibration-rows.json') for k in (20, 21)]
        if model == 'vgg':
            assert b['e18_runner'] == str(C1 / 'e18/source/e18_runner.py') and b['eccv_n'] == b['eccv_batch'] * b['eccv_batches'] < b['draw_n']
        else:
            assert b['e17_runner'] == str(C1 / 'e17/source/e17_runner.py') and b['ftbc_alpha'] == 0.4
        rep['checks'][f'config_{model}_{which}'] = sorted(diff)
pol = json.loads(Path('/etc/slurm-manager/policy.json').read_text())
assert {g['uuid']: g['type'] for node in pol['nodes'].values() for g in node['gpus']}[H100] == 'h100'
# (2) inputs
prep = json.loads((E / 'input/preparation.json').read_text()); prev21 = json.loads((C1 / 'e21/input/preparation.json').read_text())
assert prep['passed'] and prep['sizes'] == {'fresh': 2048, 'C1': 256, 'C2': 256, 'C3': 256}
assert prep['excluded_ids'] == prev21['excluded_ids'] + sum(prev21['sizes'].values()), (prep['excluded_ids'], prev21['excluded_ids'], prev21['sizes'])
for p in [str(C1 / 'e21/input/fresh-development.json'), str(C1 / 'e21/input/calibration-rows.json')]:
    assert prep['exclusion_sources'][p] == sha(p)
fresh = json.loads((E / 'input/fresh-development.json').read_text())['development']; cal = json.loads((E / 'input/calibration-rows.json').read_text())
new = fresh + [r for k in ['C1', 'C2', 'C3'] for r in cal[k]]
old = []
for p in prep['exclusion_sources']:
    d = json.loads(Path(p).read_text())
    if isinstance(d, dict) and ('development' in d or 'C1' in d):
        old += [r for v in d.values() for r in v if isinstance(r, dict) and 'index' in r]
assert len({r['index'] for r in new}) == len(new) and not {r['index'] for r in new} & {r['index'] for r in old}
rep['checks']['inputs'] = {'fresh': len(fresh), 'calibration': {k: len(v) for k, v in cal.items()}, 'excluded_ids': prep['excluded_ids'], 'fresh_distinct_labels': prep['fresh_distinct_labels']}
# (3) frozen files unchanged
for name in ['e17', 'e18', 'e19', 'e20'] + (['e21'] if (C1 / 'e21/protocol-freeze.json').exists() else []):
    fl = json.loads((C1 / name / 'protocol-freeze.json').read_text())['files_sha256']
    for p, h in fl.items():
        assert sha(p) == h, f'{name}: {p}'
    rep['checks'][f'frozen_unchanged_{name}'] = len(fl)
assert not (E / 'protocol-freeze.json').exists()
for m in ['r34', 'vgg']:
    for p in ['sanity', 'main']:
        assert not (E / f'{m}-{p}-acceptance.json').exists()
# (4) job driver with stand-in chains
J = load('chk_e22_job', E / 'source/e22_job.py')
for m in ['r34', 'vgg']:
    ch = J.chain(m); assert [x[0] for x in ch] == [f'{m}-sanity', f'{m}-sanity-acceptance', f'{m}-main']
    assert ch[0][1][2:] == [str(E / f'source/e22_{m}_runner.py'), '--config', str(E / f'config/{m}-sanity.json')]
    assert ch[1][1][2:] == [str(E / f'source/e22_{m}_accept.py'), 'sanity'] and ch[2][1][2:] == [str(E / f'source/e22_{m}_runner.py'), '--config', str(E / f'config/{m}-main.json')]
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
        J.main(); raise AssertionError('e22_job did not report the skipped VGG main')
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
        J.main(); raise AssertionError('e22_job did not report the failed chains')
    except SystemExit as e:
        assert "'r34': 5" in str(e) and "'vgg': 3" in str(e), e
    l2 = (tmp / 'slurm/r34-chk2.out').read_text(); assert 'r34-sanity-acceptance' not in l2 and 'r34-main' not in l2, l2
    assert 'vgg-main' not in (tmp / 'slurm/vgg-chk2.out').read_text()
    def standin3(m):                                   # --sequential (section 8 fallback): both chains pass; VGG starts only after R34 finished
        stamp = lambda w: f"import json,time; print(json.dumps({{'m': '{m}', 'w': '{w}', 'start': time.time()}}), flush=True); time.sleep(1); print(json.dumps({{'m': '{m}', 'w': '{w}', 'end': time.time()}}), flush=True)"
        acc = f"import json; open('{tmp}/{m}-sanity-acceptance.json','w').write(json.dumps({{'passed': True, 'cuda_initialized': False, 'scope': 'sanity only'}}))"
        return [(f'{m}-sanity', [py, '-c', stamp('sanity')]), (f'{m}-sanity-acceptance', [py, '-c', acc]), (f'{m}-main', [py, '-c', stamp('main')])]
    for x in [tmp / 'r34-sanity-acceptance.json', tmp / 'vgg-sanity-acceptance.json']:
        x.unlink(missing_ok=True)
    J.chain = standin3; os.environ['SLURM_JOB_ID'] = 'chk3'
    J.main(sequential=True)                            # both chains pass: no SystemExit
    ev = {m: [json.loads(l) for l in (tmp / f'slurm/{m}-chk3.out').read_text().splitlines() if l.startswith('{"m"')] for m in ['r34', 'vgg']}
    r34_end = max(x['end'] for x in ev['r34'] if 'end' in x); vgg_start = min(x['start'] for x in ev['vgg'] if 'start' in x)
    assert len(ev['r34']) == len(ev['vgg']) == 4 and vgg_start >= r34_end, ev
    del os.environ['SLURM_JOB_ID']
    # check_memory on the real chain logs of the first stand-in run, with records written as the hook would write them
    CMr = load('chk_e22_check_memory_real', E / 'source/check_memory.py')
    real = [{'pid': 1, 'argv': [str(E / f'source/e22_{m}_runner.py'), '--config', str(E / f'config/{m}-{w}.json')], 'unix_time': 0, 'cuda_initialized': True, 'num_ooms': 0,
             'num_alloc_retries': 0, 'max_memory_reserved': 2 ** 30, 'max_memory_allocated': 2 ** 29, 'alloc_conf': 'expandable_segments:True', 'slurm_job_id': 'chk'}
            for m, w in [('r34', 'sanity'), ('r34', 'main'), ('vgg', 'sanity')]]
    (tmp / 'slurm/memstats-chk.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in real))
    rr = CMr.check('chk', slurm=tmp / 'slurm'); assert rr['valid_for_both_models'] and rr['started_runners'] == ['r34-main', 'r34-sanity', 'vgg-sanity'], rr
    lr, lv = (tmp / 'slurm/r34-chk.out').read_text().splitlines(), (tmp / 'slurm/vgg-chk.out').read_text().splitlines()
    sr = [json.loads(l) for l in lr if l.startswith('{"m"')]; sv = [json.loads(l) for l in lv if l.startswith('{"m"')]
    assert len(sr) == 2 and len(sv) == 1, (lr, lv)                           # r34: sanity and main ran; vgg: only sanity
    assert abs(sr[0]['start'] - sv[0]['start']) < 1.5 and all(x['alloc'] == 'expandable_segments:True' and x['site'] and x['memstats'] == str(tmp / 'slurm/memstats-chk.jsonl') for x in sr + sv)
    gm = (tmp / 'slurm/gpumem-chk.csv').read_text().splitlines(); assert gm and gm[0].split(',')[1].strip() == H100, gm[:2]
    ga = (tmp / 'slurm/gpuapps-chk.csv').read_text().splitlines(); assert not any('Field' in l and 'is not a valid' in l for l in ga), ga[:2]
    rep['checks']['job_driver_standin'] = {'start_gap_s': abs(sr[0]['start'] - sv[0]['start']), 'vgg_main_skipped_without_acceptance': True, 'device_samples': len(gm),
                                           'nonzero_sanity_stops_its_chain_only': True, 'failed_acceptance_skips_main': True, 'sequential_mode_vgg_after_r34': True,
                                           'check_memory_on_real_logs': {'started': rr['started_runners'], 'valid': rr['valid_for_both_models']}}
# (5) memwatch hook
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    tmp = Path(tmp); out = tmp / 'memstats.jsonl'
    env = dict(os.environ, PYTHONPATH=str(E / 'memwatch'), PAR_MEMSTATS=str(out), CUDA_VISIBLE_DEVICES='')
    mock = ("import torch; torch.cuda.is_initialized = lambda: True; torch.cuda.memory_stats = lambda: {'num_ooms': 0, 'num_alloc_retries': 2}; "
            "torch.cuda.max_memory_reserved = lambda: 5; torch.cuda.max_memory_allocated = lambda: 3")
    r1 = subprocess.run([sys.executable, '-B', '-c', mock, '--config', 'x.json'], env=env, capture_output=True, text=True); assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run([sys.executable, '-B', '-c', 'import torch'], env=env, capture_output=True, text=True); assert r2.returncode == 0, r2.stderr
    stub = tmp / 'e22_r34_runner.py'                   # a runner that never initializes CUDA, with a forked child (as a DataLoader worker)
    stub.write_text("import multiprocessing as mp, torch\nif __name__ == '__main__':\n    mp.set_start_method('fork'); p = mp.Process(target=print); p.start(); p.join()\n")
    r3 = subprocess.run([sys.executable, '-B', str(stub), '--config', 'y.json'], env=env, capture_output=True, text=True); assert r3.returncode == 0, r3.stderr
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 2 and recs[0]['cuda_initialized'] and recs[0]['num_ooms'] == 0 and recs[0]['num_alloc_retries'] == 2 and recs[0]['argv'][1:] == ['--config', 'x.json'], recs
    assert recs[1]['cuda_initialized'] is False and recs[1]['num_ooms'] is None and recs[1]['argv'] == [str(stub), '--config', 'y.json'], recs
    rep['checks']['memwatch_hook'] = {'records_for_cuda_process': 1, 'records_for_non_cuda_process': 0, 'records_for_non_cuda_runner': 1, 'records_for_forked_child': 0}
# (6) check_memory rule
CM = load('chk_e22_check_memory', E / 'source/check_memory.py')
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    tmp = Path(tmp)
    def rec(m, which, ooms, retries=0, cuda=True):
        return {'pid': 1, 'argv': [str(E / f'source/e22_{m}_runner.py'), '--config', str(E / f'config/{m}-{which}.json')], 'unix_time': 0, 'cuda_initialized': cuda,
                'num_ooms': ooms if cuda else None, 'num_alloc_retries': retries if cuda else None, 'max_memory_reserved': 2 ** 30 if cuda else None,
                'max_memory_allocated': 2 ** 29 if cuda else None, 'alloc_conf': 'expandable_segments:True', 'slurm_job_id': 'x'}
    def logs(job, started):
        for m in ['r34', 'vgg']:
            (tmp / f'{m}-{job}.out').write_text(''.join(json.dumps({'stage': s}) + '\n' for s in started if s.startswith(m)))
    cases = [('a', ['r34-sanity', 'r34-main', 'vgg-sanity', 'vgg-main'], [rec(m, w, 0) for m in ['r34', 'vgg'] for w in ['sanity', 'main']], True),
             ('b', ['r34-sanity', 'r34-main', 'vgg-sanity'], [rec('r34', 'sanity', 0), rec('r34', 'main', 0), rec('vgg', 'sanity', 0)], True),     # vgg main never started
             ('c', ['r34-sanity', 'r34-main', 'vgg-sanity', 'vgg-main'], [rec('r34', 'sanity', 0), rec('r34', 'main', 0), rec('vgg', 'sanity', 0)], False),  # started, no record
             ('d', ['r34-sanity', 'r34-main', 'vgg-sanity', 'vgg-main'], [rec(m, w, 1 if (m, w) == ('vgg', 'main') else 0) for m in ['r34', 'vgg'] for w in ['sanity', 'main']], False),
             ('e', ['r34-sanity', 'r34-main', 'vgg-sanity', 'vgg-main'], [rec(m, w, 0, retries=3) for m in ['r34', 'vgg'] for w in ['sanity', 'main']], True),   # retries do not decide
             ('f', ['r34-sanity', 'r34-main', 'vgg-sanity', 'vgg-main'], [rec(m, w, 0, cuda=(m, w) != ('vgg', 'main')) for m in ['r34', 'vgg'] for w in ['sanity', 'main']], True),  # never used the GPU
             ('g', ['r34-sanity', 'r34-main', 'vgg-sanity'], [rec('r34', 'sanity', 0), rec('r34', 'main', 0), rec('r34', 'main', 0), rec('vgg', 'sanity', 0)], False),  # duplicate record
             ('h', ['r34-sanity', 'r34-main', 'vgg-sanity', 'vgg-main'], [rec(m, w, 0) for m in ['r34', 'vgg'] for w in ['sanity', 'main']]
              + [dict(rec('r34', 'main', 0), argv=['/x/other.py', '--config', 'y.json'])], False)]                                                      # unexpected record
    for job, started, recs_, want in cases:
        logs(job, started); (tmp / f'memstats-{job}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in recs_))
        res = CM.check(job, slurm=tmp); assert res['valid_for_both_models'] == want, (job, res)
        if job == 'e': assert res['near_full'], res
        if job == 'f': assert res['gpu_used']['vgg-main'] is False and 'vgg-main' not in res['num_ooms'], res
    rep['checks']['check_memory_rule'] = {k: v for k, v in zip('abcdefgh', ['all clean', 'unstarted main ignored', 'started without record invalid', 'one OOM invalid',
                                                                          'retries reported only (near_full set)', 'runner that never used the GPU outside the rule', 'duplicate record invalid',
                                                                          'unexpected record invalid'])}
# (7) dry freeze
FS = load('chk_e22_freeze_submit', E / 'source/e22_freeze_submit.py')
f = FS.collect(with_review=False)
w = Path('/data3/snn_prjs/ftbc_neural_networks')
inside = [p for p in f if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w)]
rep['checks']['dry_freeze'] = {'entries': len(f), 'staged': len(inside), 'not_staged_outside_workspace': len(f) - len(inside)}
# (8) gpurun dry run
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    m = Path(tmp) / 'slurm-source-manifest.json'; m.write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): f[p] for p in inside}}) + '\n')
    cmd = FS.command(); i = cmd.index('--source-manifest'); cmd[i + 1] = str(m); dry = cmd[:1] + ['--dry-run'] + cmd[1:]
    r = subprocess.run(dry, capture_output=True, text=True); assert r.returncode == 0, (r.stdout, r.stderr)
assert '--gres=gpu:h100:1' in r.stdout and '--cpus-per-task=48' in r.stdout and '--mem=256G' in r.stdout and f"--time={int(FS.TIME.split(':')[0]) * 60 + int(FS.TIME.split(':')[1])}" in r.stdout, r.stdout[-800:]
rep['checks']['gpurun_dry_run'] = {'command': dry, 'stdout_tail': r.stdout[-1500:]}
# (9) TPP (section 4)
import torch
TP = E / 'third_party/tpp-drive-20261010'; receipt = json.loads((E / 'third_party/tpp-drive-20261010-receipt.json').read_text())
assert not receipt['executed'] and all(sha(TP / f) == v['sha256'] for f, v in receipt['files'].items()), 'downloaded TPP files changed'
TN = load('chk_e22_tpp', E / 'source/tpp_neuron.py')
assert sha(TP / 'Models/layer.py') == TN.TPP_LAYER_SHA256 == sha(C1.parent / 'evidence/serial-tpp-Models-layer.py')
text = (TP / 'Models/layer.py').read_text().splitlines()          # the third-party file is read as text only
start = text.index("            elif self.mode == 'prob':"); stop = start + 1
while not text[stop].strip().startswith('x = self.merge(x)'): stop += 1
branch = [l.strip() for l in text[start + 1:stop + 1] if l.strip()]
import inspect
copied = inspect.getsource(TN.tpp_prob_forward).splitlines()
c0 = next(k for k, l in enumerate(copied) if l.strip() == 'thre = self.thresh.data'); c1 = next(k for k, l in enumerate(copied) if l.strip() == 'x = self.merge(x)')
mine = [l.strip() for l in copied[c0:c1 + 1] if l.strip()]
assert [l.replace('self.mem_pot', 'mem_pot') for l in branch] == mine, (branch, mine)
assert sum(l.count('self.mem_pot') for l in branch) == 4 and sum(l.count('mem_pot') for l in mine) == 4 and not any('self.mem_pot' in l for l in mine)
layer = load('chk_e22_serial_layer', C1 / 'source-snapshot/serial/Models/layer.py')        # the project's serial IF (R34 and VGG-16 chains)
def reference(m, x):
    """Independent re-implementation of TPP's prob mode (section 4): sum the T input currents, then T Bernoulli steps with
    p = clamp(m / ((T - t) theta), 0, 1) and subtraction of theta per spike."""
    T, th = m.T, float(m.thresh); xs = x.view(T, x.shape[0] // T, *x.shape[1:]); pot = xs.sum(0); out = []
    for t in range(T):
        s = torch.bernoulli(torch.clamp(pot / (T - t) / th, 0, 1)) * th; pot = pot - s; out.append(s)
    return torch.stack(out).flatten(0, 1)
net = torch.nn.Sequential(layer.IF(T=0, L=4, thresh=1.3), torch.nn.Conv2d(3, 3, 1), layer.IF(T=0, L=4, thresh=0.7))
for m in net.modules():
    if isinstance(m, layer.IF):
        m.T = 4; m.expand.T = 4; m.merge.T = 4
x = torch.randn(4 * 5, 3, 6, 6) * 2
ifs = [m for m in net.modules() if isinstance(m, layer.IF)]
for m in ifs:
    m.time_based_bias = [torch.full((3,), 0.25 * (t + 1)) for t in range(4)]       # TPP ignores the per-step biases
with TN.tpp_mode(net, layer.IF) as n_swapped:
    assert n_swapped == 2 and all('forward' in vars(m) for m in ifs)
    torch.manual_seed(7); y1 = ifs[0](x)
    torch.manual_seed(7); y2 = reference(ifs[0], x)
    torch.manual_seed(7); y3 = ifs[0](x)
    for m in ifs: m.time_based_bias = [torch.zeros(3) for _ in range(4)]
    torch.manual_seed(7); y4 = ifs[0](x)
    torch.manual_seed(11); y5 = net(x)
    torch.manual_seed(11); y6 = net(x)
assert torch.equal(y1, y2) and torch.equal(y1, y3) and torch.equal(y1, y4) and torch.equal(y5, y6)
assert all('forward' not in vars(m) for m in ifs)
cnt = (y1.view(4, 5, 3, 6, 6) / float(ifs[0].thresh)).sum(0); assert ((cnt >= 0) & (cnt <= 4)).all() and torch.equal(cnt, cnt.round())
try:
    with TN.tpp_mode(net, layer.IF):
        raise RuntimeError('probe')
except RuntimeError:
    pass
assert all('forward' not in vars(m) for m in ifs)
try:
    with TN.tpp_mode(net, layer.IF):
        with TN.tpp_mode(net, layer.IF):
            pass
    raise AssertionError('a nested swap was not refused')
except AssertionError as e:
    assert 'already swapped' in str(e), e
assert all('forward' not in vars(m) for m in ifs)
y7 = ifs[0](x[:4 * 2]); assert y7.shape == x[:8].shape                                            # class forward back (normal mode with biases)
assert TN.seed_for(3, 1) == 3001 and TN.seed_for(0, 9) == 9
rep['checks']['tpp'] = {'download_unchanged': len(receipt['files']), 'copied_branch_lines': len(mine), 'equal_to_reference': True, 'reproducible': True,
                        'biases_ignored': True, 'restored_after_exit_and_error': True, 'nested_swap_refused': True}
names = ['source/e22_r34_runner.py', 'source/e22_vgg_runner.py', 'source/e22_r34_accept.py', 'source/e22_vgg_accept.py', 'source/e22_job.py', 'source/check_memory.py',
         'source/e22_freeze_submit.py', 'source/e22_check_cpu.py', 'source/prepare_inputs.py', 'source/tpp_neuron.py', 'memwatch/sitecustomize.py', 'record_costs.py', 'verify_preservation.py',
         'EXPERIMENT_PLAN.md', 'preservation-before.json', 'tools/cpu_dryrun.py'] + [f'config/{m}-{p}.json' for m in ['r34', 'vgg'] for p in ['sanity', 'main']]
rep.update(passed=True, cuda_initialized=torch.cuda.is_initialized(), python=sys.version, torch=torch.__version__, files_sha256={str(E / n): sha(E / n) for n in names})
assert not rep['cuda_initialized']
(E / 'cache').mkdir(exist_ok=True); (E / 'cache/cpu-check.json').write_text(json.dumps(rep, indent=2) + '\n')
print(json.dumps({k: v for k, v in rep['checks'].items() if k != 'gpurun_dry_run'}, indent=1))
