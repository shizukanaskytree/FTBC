"""Account E14's actual finished allocations once; preserve existing ledger rows (adapted from E13).
Every submitted phase is charged its Slurm allocation seconds in any terminal state; sanity and main must be accepted,
extra is descriptive and its failure (e.g. OOM) is recorded and charged without invalidating main (EXPERIMENT_PLAN.md section 8)."""
import json, re, subprocess
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parent; W = Path('/data3/snn_prjs/ftbc_neural_networks')
ledger_path = W / 'ccfa-workfiles/experiments/ftbc-neural-networks/submission-budget-ledger.json'
raw = ledger_path.read_bytes(); ledger = json.loads(raw); rows = []; added = []
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
for phase in ['sanity', 'main', 'extra']:
    receipt = E / f'slurm/{phase}-submit.json'
    if not receipt.exists():
        assert phase == 'extra', f'{phase} was never submitted'; continue
    acc = E / f'{phase}-acceptance.json'; accepted = acc.exists() and json.loads(acc.read_text())['passed']
    if phase in ('sanity', 'main'): assert accepted, f'{phase} must be accepted before accounting'
    job = json.loads(json.loads(receipt.read_text())['stdout'])['job_id']
    s = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True)
    state = re.search(r'JobState=(\S+)', s).group(1); assert state in TERMINAL, state
    (E / f'slurm/{phase}-final.txt').write_text(s)
    h, m, sec = map(int, re.search(r'RunTime=(\d+):(\d+):(\d+)', s).groups()); seconds = 3600 * h + 60 * m + sec
    status = 'completed' if state == 'COMPLETED' and accepted else f'{state.lower()}{"" if accepted else ", not accepted"}'
    row = {'id': f'c1pre-e14-{phase}-v2', 'phase': 'route_c_C1pre_E14_per_step_ftbc_screen', 'status': status, 'slurm_state': state,
           'charged_seconds': seconds, 'slurm_job_id': job, 'accepted_children': int(accepted),
           'accounting': 'actual single GPU allocation window, single process', 'acceptance': str(acc) if acc.exists() else None,
           'final_slurm': str(E / f'slurm/{phase}-final.txt')}
    rows.append(row)
    if not any(v['id'] == row['id'] for v in ledger['runs']):
        ledger['runs'].append(row); ledger['charged_gpu_seconds'] += seconds; added.append(row)
(E / 'cost.json').write_text(json.dumps({'rows': rows, 'total_allocation_seconds': sum(r['charged_seconds'] for r in rows),
                                         'GPU_hours': sum(r['charged_seconds'] for r in rows) / 3600, 'not_formal_latency': True}, indent=2) + '\n')
if added:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); (E / 'cache').mkdir(exist_ok=True)
    (E / f'cache/ledger-before-{stamp}.json').write_bytes(raw)
    assert ledger_path.read_bytes() == raw, 'Concurrent ledger update; re-read and merge'
    temp = ledger_path.with_name(ledger_path.name + '.e14.tmp'); temp.write_text(json.dumps(ledger, indent=2) + '\n'); temp.replace(ledger_path)
print(json.dumps({'rows': rows, 'new_rows': len(added)}))
