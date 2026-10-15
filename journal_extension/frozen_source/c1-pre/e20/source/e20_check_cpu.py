"""CPU check of E20 (EXPERIMENT_PLAN.md sections 7-8). No CUDA. Writes cache/cpu-check.json.
Checks: (1) the four child configs differ from their templates (E19 for R34, E18 for VGG-16) only in the intended keys, the sizes and the
H100 UUID; (2) the inputs: counts, the exclusion chain on E19 and disjointness by ID from the earlier splits listed as exclusion sources (content hashes are
checked by prepare_inputs.py, the runners and the acceptances); (3) the E17, E18 and E19 frozen
files are unchanged; (4) the job driver: chain commands, concurrency, the sanity gate (main is skipped when the sanity acceptance is missing),
one chain's failure (a skipped main, a nonzero exit, a failed sanity acceptance) leaves the other running, the environment reaches both chains, the samplers write;
(5) the memwatch exit hook (CUDA processes and runner processes, forked children excluded); (6) check_memory's rule: started runners only, num_ooms decides
for processes that used the GPU, a runner that never used the GPU does not enter the rule, a missing or duplicate record is invalid, retries do not decide; also
on the real chain logs of (4); (7) the dry freeze resolves every entry and the staged
manifest covers every workspace entry; (8) gpurun --dry-run accepts the submission command (it does not compare the manifest with the
workspace; the TaskProlog does at start)."""
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
            'previous_fresh_manifests', 'previous_calibration_manifests', 'sizes', 'draw_n'}
for model, tpl, extra in [('r34', C1 / 'e19/config', {'ftbc_variants', 'eccv_alpha', 'eccv_batch', 'eccv_batches', 'eccv_n', 'diag_n'}), ('vgg', C1 / 'e18/config', {'e18_runner'})]:
    for which in ['sanity', 'main']:
        a = json.loads((tpl / f'{which}.json').read_text()); b = json.loads((E / f'config/{model}-{which}.json').read_text())
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}; assert diff <= INTENDED | extra, sorted(diff - INTENDED - extra)
        assert b['allowed_uuids'] == [H100] and b['freeze_manifest'] == str(E / 'protocol-freeze.json')
        assert b['output'] == f'/data7/projects/ftbc/runs/c1pre-e20-20261010/{model}-{which}-v1' and not Path(b['output']).exists()
        sizes = b['sizes']; assert sizes == ([8, 16, 32, 96] if which == 'main' else [2, 4, 8, 12]) and sizes[-1] == b['small_n'] == 3 * b['ftbc_batch'] < b['draw_n']
        assert b['fresh_manifest'] == str(E / 'input/fresh-development.json') and b['calibration_manifest'] == str(E / 'input/calibration-rows.json')
        assert b['fresh_manifest_sha256'] == sha(b['fresh_manifest']) and b['calibration_manifest_sha256'] == sha(b['calibration_manifest'])
        prev = set(b['previous_fresh_manifests']); assert str(C1 / 'e19/input/fresh-development.json') in prev and str(C1 / 'e18/input/fresh-development.json') in prev
        prevc = set(b['previous_calibration_manifests']); assert str(C1 / 'e19/input/calibration-rows.json') in prevc and str(C1 / 'e18/input/calibration-rows.json') in prevc
        if model == 'r34':
            assert b['diag_n'] <= b['fresh_count'] and 'e17_runner' in b
        else:
            assert b['e18_runner'] == str(C1 / 'e18/source/e18_runner.py') and b['eccv_n'] == b['eccv_batch'] * b['eccv_batches'] < b['draw_n']
        rep['checks'][f'config_{model}_{which}'] = sorted(diff)
pol = json.loads(Path('/etc/slurm-manager/policy.json').read_text())
assert {g['uuid']: g['type'] for node in pol['nodes'].values() for g in node['gpus']}[H100] == 'h100'
# (2) inputs
prep = json.loads((E / 'input/preparation.json').read_text()); prev19 = json.loads((C1 / 'e19/input/preparation.json').read_text())
assert prep['passed'] and prep['sizes'] == {'fresh': 2048, 'C1': 128, 'C2': 128, 'C3': 128} and prep['excluded_ids'] == prev19['excluded_ids'] + 2048 + 3 * 256
for p in [str(C1 / 'e19/input/fresh-development.json'), str(C1 / 'e19/input/calibration-rows.json')]:
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
for name in ['e17', 'e18', 'e19']:
    fl = json.loads((C1 / name / 'protocol-freeze.json').read_text())['files_sha256']
    for p, h in fl.items():
        assert sha(p) == h, f'{name}: {p}'
    rep['checks'][f'frozen_unchanged_{name}'] = len(fl)
assert not (E / 'protocol-freeze.json').exists()
for m in ['r34', 'vgg']:
    for p in ['sanity', 'main']:
        assert not (E / f'{m}-{p}-acceptance.json').exists()
# (4) job driver with stand-in chains
J = load('chk_e20_job', E / 'source/e20_job.py')
for m in ['r34', 'vgg']:
    ch = J.chain(m); assert [x[0] for x in ch] == [f'{m}-sanity', f'{m}-sanity-acceptance', f'{m}-main']
    assert ch[0][1][2:] == [str(E / f'source/e20_{m}_runner.py'), '--config', str(E / f'config/{m}-sanity.json')]
    assert ch[1][1][2:] == [str(E / f'source/e20_{m}_accept.py'), 'sanity'] and ch[2][1][2:] == [str(E / f'source/e20_{m}_runner.py'), '--config', str(E / f'config/{m}-main.json')]
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
        J.main(); raise AssertionError('e20_job did not report the skipped VGG main')
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
        J.main(); raise AssertionError('e20_job did not report the failed chains')
    except SystemExit as e:
        assert "'r34': 5" in str(e) and "'vgg': 3" in str(e), e
    l2 = (tmp / 'slurm/r34-chk2.out').read_text(); assert 'r34-sanity-acceptance' not in l2 and 'r34-main' not in l2, l2
    assert 'vgg-main' not in (tmp / 'slurm/vgg-chk2.out').read_text()
    del os.environ['SLURM_JOB_ID']
    # check_memory on the real chain logs of the first stand-in run, with records written as the hook would write them
    CMr = load('chk_e20_check_memory_real', E / 'source/check_memory.py')
    real = [{'pid': 1, 'argv': [str(E / f'source/e20_{m}_runner.py'), '--config', str(E / f'config/{m}-{w}.json')], 'unix_time': 0, 'cuda_initialized': True, 'num_ooms': 0,
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
                                           'nonzero_sanity_stops_its_chain_only': True, 'failed_acceptance_skips_main': True,
                                           'check_memory_on_real_logs': {'started': rr['started_runners'], 'valid': rr['valid_for_both_models']}}
# (5) memwatch hook
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    tmp = Path(tmp); out = tmp / 'memstats.jsonl'
    env = dict(os.environ, PYTHONPATH=str(E / 'memwatch'), PAR_MEMSTATS=str(out), CUDA_VISIBLE_DEVICES='')
    mock = ("import torch; torch.cuda.is_initialized = lambda: True; torch.cuda.memory_stats = lambda: {'num_ooms': 0, 'num_alloc_retries': 2}; "
            "torch.cuda.max_memory_reserved = lambda: 5; torch.cuda.max_memory_allocated = lambda: 3")
    r1 = subprocess.run([sys.executable, '-B', '-c', mock, '--config', 'x.json'], env=env, capture_output=True, text=True); assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run([sys.executable, '-B', '-c', 'import torch'], env=env, capture_output=True, text=True); assert r2.returncode == 0, r2.stderr
    stub = tmp / 'e20_r34_runner.py'                   # a runner that never initializes CUDA, with a forked child (as a DataLoader worker)
    stub.write_text("import multiprocessing as mp, torch\nif __name__ == '__main__':\n    mp.set_start_method('fork'); p = mp.Process(target=print); p.start(); p.join()\n")
    r3 = subprocess.run([sys.executable, '-B', str(stub), '--config', 'y.json'], env=env, capture_output=True, text=True); assert r3.returncode == 0, r3.stderr
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 2 and recs[0]['cuda_initialized'] and recs[0]['num_ooms'] == 0 and recs[0]['num_alloc_retries'] == 2 and recs[0]['argv'][1:] == ['--config', 'x.json'], recs
    assert recs[1]['cuda_initialized'] is False and recs[1]['num_ooms'] is None and recs[1]['argv'] == [str(stub), '--config', 'y.json'], recs
    rep['checks']['memwatch_hook'] = {'records_for_cuda_process': 1, 'records_for_non_cuda_process': 0, 'records_for_non_cuda_runner': 1, 'records_for_forked_child': 0}
# (6) check_memory rule
CM = load('chk_e20_check_memory', E / 'source/check_memory.py')
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    tmp = Path(tmp)
    def rec(m, which, ooms, retries=0, cuda=True):
        return {'pid': 1, 'argv': [str(E / f'source/e20_{m}_runner.py'), '--config', str(E / f'config/{m}-{which}.json')], 'unix_time': 0, 'cuda_initialized': cuda,
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
             ('g', ['r34-sanity', 'r34-main', 'vgg-sanity'], [rec('r34', 'sanity', 0), rec('r34', 'main', 0), rec('r34', 'main', 0), rec('vgg', 'sanity', 0)], False)]  # duplicate record
    for job, started, recs_, want in cases:
        logs(job, started); (tmp / f'memstats-{job}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in recs_))
        res = CM.check(job, slurm=tmp); assert res['valid_for_both_models'] == want, (job, res)
        if job == 'e': assert res['near_full'], res
        if job == 'f': assert res['gpu_used']['vgg-main'] is False and 'vgg-main' not in res['num_ooms'], res
    rep['checks']['check_memory_rule'] = {k: v for k, v in zip('abcdefg', ['all clean', 'unstarted main ignored', 'started without record invalid', 'one OOM invalid',
                                                                         'retries reported only (near_full set)', 'runner that never used the GPU outside the rule', 'duplicate record invalid'])}
# (7) dry freeze
FS = load('chk_e20_freeze_submit', E / 'source/e20_freeze_submit.py')
f = FS.collect(with_review=False)
w = Path('/data3/snn_prjs/ftbc_neural_networks')
inside = [p for p in f if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w)]
rep['checks']['dry_freeze'] = {'entries': len(f), 'staged': len(inside), 'not_staged_outside_workspace': len(f) - len(inside)}
# (8) gpurun dry run
with tempfile.TemporaryDirectory(dir='/data5/FPGA/cache/tmp') as tmp:
    m = Path(tmp) / 'slurm-source-manifest.json'; m.write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): f[p] for p in inside}}) + '\n')
    cmd = FS.command(); i = cmd.index('--source-manifest'); cmd[i + 1] = str(m); dry = cmd[:1] + ['--dry-run'] + cmd[1:]
    r = subprocess.run(dry, capture_output=True, text=True); assert r.returncode == 0, (r.stdout, r.stderr)
assert '--gres=gpu:h100:1' in r.stdout and '--cpus-per-task=48' in r.stdout and '--mem=256G' in r.stdout and '--time=330' in r.stdout, r.stdout[-800:]
rep['checks']['gpurun_dry_run'] = {'command': dry, 'stdout_tail': r.stdout[-1500:]}
import torch
names = ['source/e20_r34_runner.py', 'source/e20_vgg_runner.py', 'source/e20_r34_accept.py', 'source/e20_vgg_accept.py', 'source/e20_job.py', 'source/check_memory.py',
         'source/e20_freeze_submit.py', 'source/e20_check_cpu.py', 'source/prepare_inputs.py', 'memwatch/sitecustomize.py', 'record_costs.py', 'verify_preservation.py',
         'EXPERIMENT_PLAN.md', 'preservation-before.json', 'tools/cpu_dryrun.py'] + [f'config/{m}-{p}.json' for m in ['r34', 'vgg'] for p in ['sanity', 'main']]
rep.update(passed=True, cuda_initialized=torch.cuda.is_initialized(), python=sys.version, torch=torch.__version__, files_sha256={str(E / n): sha(E / n) for n in names})
assert not rep['cuda_initialized']
(E / 'cache').mkdir(exist_ok=True); (E / 'cache/cpu-check.json').write_text(json.dumps(rep, indent=2) + '\n')
print(json.dumps({k: v for k, v in rep['checks'].items() if k != 'gpurun_dry_run'}, indent=1))
