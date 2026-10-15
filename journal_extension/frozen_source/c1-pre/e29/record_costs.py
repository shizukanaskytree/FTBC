"""Account E29's actual allocation once; preserve existing ledger rows (adapted from E23). Charged seconds are the sum of every allocation
segment of the job in /var/log/slurm/jobcomp.log; scontrol RunTime is recorded for cross-checking only. E29 has one job in which the R34 and
VGG-16 chains run one after the other (EXPERIMENT_PLAN.md section 7): one row, both chains' validity flags and main readings (initial review
B2: a valid run whose reading is 'fail' or 'not decided' is completed; follow-up 1 N1: so is a chain that stopped at a failed sanity gate). A
job whose chains are not both complete is still accounted (its allocation was used), with status 'not valid'."""
import json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path
E = Path(__file__).resolve().parent; W = Path('/data3/snn_prjs/ftbc_neural_networks')
ledger_path = W / 'ccfa-workfiles/experiments/ftbc-neural-networks/submission-budget-ledger.json'
raw = ledger_path.read_bytes(); ledger = json.loads(raw); added = []
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
stamp = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
receipt = E / 'slurm/job-submit.json'
if not receipt.exists():
    raise AssertionError('the E29 job was never submitted')
accs = {f'{m}-{p}': E / f'{m}-{p}-acceptance.json' for m in ['r34', 'vgg'] for p in ['sanity', 'main']}
acc_flags = {k: a.exists() and json.loads(a.read_text())['valid'] for k, a in accs.items()}
import sys; sys.path.insert(0, str(E)); import verify_preservation as V           # follow-up 1 N1: a chain that stopped at a failed gate is complete
loaded = {k: json.loads(a.read_text()) for k, a in accs.items() if a.exists()}
chains = {m: V.complete(loaded, m) for m in ['r34', 'vgg']}; accepted = all(chains.values())
readings = {m: V.model_reading(loaded, m) for m in ['r34', 'vgg']}
rec = json.loads(receipt.read_text()); job = json.loads(rec['stdout'])['job_id']; run_id = rec['command'][rec['command'].index('--run-id') + 1]
segs = [l for l in Path('/var/log/slurm/jobcomp.log').read_text().splitlines() if l.startswith(f'JobId={job} ')]
(E / 'slurm/job-jobcomp.txt').write_text('\n'.join(segs) + '\n')
seconds = 0; states = []
for l in segs:
    st = re.search(r'JobState=(\S+)', l).group(1); states.append(st)
    a, b = re.search(r'StartTime=(\S+)', l).group(1), re.search(r'EndTime=(\S+)', l).group(1)
    if re.search(r'NodeCnt=(\d+)', l).group(1) != '0' and a != 'Unknown' and b != 'Unknown':
        seconds += int((stamp(b) - stamp(a)).total_seconds())
assert segs and states[-1] in TERMINAL, states
try:
    s = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True); (E / 'slurm/job-final.txt').write_text(s)
    h, m, sec = map(int, re.search(r'RunTime=(\d+):(\d+):(\d+)', s).groups()); runtime = 3600 * h + 60 * m + sec
except subprocess.CalledProcessError:
    runtime = None                                             # record already purged from slurmctld; jobcomp is authoritative
state = states[-1]; status = 'completed' if state == 'COMPLETED' and accepted else f'{state.lower()}{"" if accepted else ", not valid"}'
row = {'id': run_id, 'phase': 'route_c_C1pre_E29_chunked_release_default_equivalence_r34_vgg16_sequential', 'status': status, 'slurm_state': state,
       'charged_seconds': seconds, 'segments': len(segs), 'scontrol_runtime_seconds': runtime, 'slurm_job_id': job, 'valid_phases_count': sum(acc_flags.values()),
       'valid_phases': acc_flags, 'chains_complete': chains, 'readings': readings,
       'accounting': 'sum of jobcomp allocation segments, single H100, the two chains (R34, VGG-16) one after the other; charged once',
       'acceptance': {k: str(v) if v.exists() else None for k, v in accs.items()}, 'jobcomp_records': str(E / 'slurm/job-jobcomp.txt')}
if not any(v['id'] == row['id'] for v in ledger['runs']):
    ledger['runs'].append(row); ledger['charged_gpu_seconds'] += seconds; added.append(row)
(E / 'cost.json').write_text(json.dumps({'rows': [row], 'total_allocation_seconds': seconds, 'GPU_hours': seconds / 3600, 'not_formal_latency': True}, indent=2) + '\n')
if added:
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); (E / 'cache').mkdir(exist_ok=True)
    (E / f'cache/ledger-before-{ts}.json').write_bytes(raw)
    assert ledger_path.read_bytes() == raw, 'Concurrent ledger update; re-read and merge'
    temp = ledger_path.with_name(ledger_path.name + '.e29.tmp'); temp.write_text(json.dumps(ledger, indent=2) + '\n'); temp.replace(ledger_path)
print(json.dumps({'rows': [row], 'new_rows': len(added)}))
