"""Account E18's actual allocation once; preserve existing ledger rows (adapted from E15/E17).
Charged seconds are the sum of every allocation segment of the job in /var/log/slurm/jobcomp.log (a requeued job has several);
scontrol RunTime is recorded for cross-checking only. E18 has one job (sanity, sanity acceptance and main in one allocation, EXPERIMENT_PLAN.md
section 8). A job whose sanity or main is not accepted is still accounted (its allocation was used), with status 'not accepted'."""
import json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path
E = Path(__file__).resolve().parent; W = Path('/data3/snn_prjs/ftbc_neural_networks')
ledger_path = W / 'ccfa-workfiles/experiments/ftbc-neural-networks/submission-budget-ledger.json'
raw = ledger_path.read_bytes(); ledger = json.loads(raw); rows = []; added = []
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
stamp = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
receipt = E / 'slurm/job-submit.json'
if not receipt.exists():
    raise AssertionError('the E18 job was never submitted')
accs = {phase: E / f'{phase}-acceptance.json' for phase in ['sanity', 'main']}
acc_flags = {k: a.exists() and json.loads(a.read_text())['passed'] for k, a in accs.items()}; accepted = all(acc_flags.values())
job = json.loads(json.loads(receipt.read_text())['stdout'])['job_id']
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
state = states[-1]; status = 'completed' if state == 'COMPLETED' and accepted else f'{state.lower()}{"" if accepted else ", not accepted"}'
row = {'id': 'c1pre-e18-job-v1', 'phase': 'route_c_C1pre_E18_vgg16_second_model_screen', 'status': status, 'slurm_state': state,
       'charged_seconds': seconds, 'segments': len(segs), 'scontrol_runtime_seconds': runtime, 'slurm_job_id': job, 'accepted_children': sum(acc_flags.values()),
       'accepted_phases': acc_flags,
       'accounting': 'sum of jobcomp allocation segments, single GPU; sanity, its CPU acceptance and main run in sequence in one allocation',
       'acceptance': {k: str(v) if v.exists() else None for k, v in accs.items()}, 'jobcomp_records': str(E / 'slurm/job-jobcomp.txt')}
rows.append(row)
if not any(v['id'] == row['id'] for v in ledger['runs']):
    ledger['runs'].append(row); ledger['charged_gpu_seconds'] += seconds; added.append(row)
(E / 'cost.json').write_text(json.dumps({'rows': rows, 'total_allocation_seconds': seconds, 'GPU_hours': seconds / 3600, 'not_formal_latency': True}, indent=2) + '\n')
if added:
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); (E / 'cache').mkdir(exist_ok=True)
    (E / f'cache/ledger-before-{ts}.json').write_bytes(raw)
    assert ledger_path.read_bytes() == raw, 'Concurrent ledger update; re-read and merge'
    temp = ledger_path.with_name(ledger_path.name + '.e18.tmp'); temp.write_text(json.dumps(ledger, indent=2) + '\n'); temp.replace(ledger_path)
print(json.dumps({'rows': rows, 'new_rows': len(added)}))
