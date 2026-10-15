"""Account E16's actual allocations once; preserve existing ledger rows (adapted from E15).
Charged seconds are the sum of every allocation segment of the job in /var/log/slurm/jobcomp.log (a requeued job has several);
scontrol RunTime is recorded for cross-checking only. sanity and main must be accepted; release and extra are charged in any
terminal state and their acceptance is recorded (EXPERIMENT_PLAN.md section 8). E16 has no extra job."""
import json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path
E = Path(__file__).resolve().parent; W = Path('/data3/snn_prjs/ftbc_neural_networks')
ledger_path = W / 'ccfa-workfiles/experiments/ftbc-neural-networks/submission-budget-ledger.json'
raw = ledger_path.read_bytes(); ledger = json.loads(raw); rows = []; added = []
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
stamp = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
for phase in ['sanity', 'main', 'release']:
    receipt = E / f'slurm/{phase}-submit.json'
    if not receipt.exists():
        assert phase == 'release', f'{phase} was never submitted'; continue
    acc = E / f'{phase}-acceptance.json'; accepted = acc.exists() and json.loads(acc.read_text())['passed']
    if phase in ('sanity', 'main'): assert accepted, f'{phase} must be accepted before accounting'
    job = json.loads(json.loads(receipt.read_text())['stdout'])['job_id']
    segs = [l for l in Path('/var/log/slurm/jobcomp.log').read_text().splitlines() if l.startswith(f'JobId={job} ')]
    (E / f'slurm/{phase}-jobcomp.txt').write_text('\n'.join(segs) + '\n')
    seconds = 0; states = []
    for l in segs:
        st = re.search(r'JobState=(\S+)', l).group(1); states.append(st)
        a, b = re.search(r'StartTime=(\S+)', l).group(1), re.search(r'EndTime=(\S+)', l).group(1)
        if re.search(r'NodeCnt=(\d+)', l).group(1) != '0' and a != 'Unknown' and b != 'Unknown':
            seconds += int((stamp(b) - stamp(a)).total_seconds())
    assert segs and states[-1] in TERMINAL, (phase, states)
    try:
        s = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True); (E / f'slurm/{phase}-final.txt').write_text(s)
        h, m, sec = map(int, re.search(r'RunTime=(\d+):(\d+):(\d+)', s).groups()); runtime = 3600 * h + 60 * m + sec
    except subprocess.CalledProcessError:
        runtime = None                                         # record already purged from slurmctld; jobcomp is authoritative
    state = states[-1]; status = 'completed' if state == 'COMPLETED' and accepted else f'{state.lower()}{"" if accepted else ", not accepted"}'
    row = {'id': f'c1pre-e16-{phase}-v1', 'phase': 'route_c_C1pre_E16_two_stage_serial', 'status': status, 'slurm_state': state,
           'charged_seconds': seconds, 'segments': len(segs), 'scontrol_runtime_seconds': runtime, 'slurm_job_id': job, 'accepted_children': int(accepted),
           'accounting': 'sum of jobcomp allocation segments, single GPU, single process', 'acceptance': str(acc) if acc.exists() else None,
           'jobcomp_records': str(E / f'slurm/{phase}-jobcomp.txt')}
    rows.append(row)
    if not any(v['id'] == row['id'] for v in ledger['runs']):
        ledger['runs'].append(row); ledger['charged_gpu_seconds'] += seconds; added.append(row)
(E / 'cost.json').write_text(json.dumps({'rows': rows, 'total_allocation_seconds': sum(r['charged_seconds'] for r in rows),
                                         'GPU_hours': sum(r['charged_seconds'] for r in rows) / 3600, 'not_formal_latency': True}, indent=2) + '\n')
if added:
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); (E / 'cache').mkdir(exist_ok=True)
    (E / f'cache/ledger-before-{ts}.json').write_bytes(raw)
    assert ledger_path.read_bytes() == raw, 'Concurrent ledger update; re-read and merge'
    temp = ledger_path.with_name(ledger_path.name + '.e16.tmp'); temp.write_text(json.dumps(ledger, indent=2) + '\n'); temp.replace(ledger_path)
print(json.dumps({'rows': rows, 'new_rows': len(added)}))
