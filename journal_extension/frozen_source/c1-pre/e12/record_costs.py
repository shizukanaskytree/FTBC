"""Account E12's actual finished allocations once; preserve existing ledger rows (adapted from E11)."""
import json, re, subprocess
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parent; W = Path('/data3/snn_prjs/ftbc_neural_networks')
ledger_path = W / 'ccfa-workfiles/experiments/ftbc-neural-networks/submission-budget-ledger.json'
raw = ledger_path.read_bytes(); ledger = json.loads(raw); rows = []; added = []
for phase in ['sanity', 'main']:
    assert json.loads((E / f'{phase}-acceptance.json').read_text())['passed']
    text = json.loads((E / f'slurm/{phase}-submit.json').read_text())['stdout']
    job = json.loads(text)['job_id']
    s = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True)
    assert 'JobState=COMPLETED ' in s and 'ExitCode=0:0 ' in s
    (E / f'slurm/{phase}-final.txt').write_text(s)
    h, m, sec = map(int, re.search(r'RunTime=(\d+):(\d+):(\d+)', s).groups()); seconds = 3600 * h + 60 * m + sec
    row = {'id': f'c1pre-e12-{phase}-v1', 'phase': 'route_c_C1pre_E12_combination_control', 'status': 'completed', 'charged_seconds': seconds,
           'slurm_job_id': job, 'accepted_children': 1, 'accounting': 'actual single GPU allocation window, single process',
           'acceptance': str(E / f'{phase}-acceptance.json'), 'final_slurm': str(E / f'slurm/{phase}-final.txt')}
    rows.append(row)
    if not any(v['id'] == row['id'] for v in ledger['runs']):
        ledger['runs'].append(row); ledger['charged_gpu_seconds'] += seconds; added.append(row)
(E / 'cost.json').write_text(json.dumps({'rows': rows, 'total_allocation_seconds': sum(r['charged_seconds'] for r in rows),
                                         'GPU_hours': sum(r['charged_seconds'] for r in rows) / 3600, 'not_formal_latency': True}, indent=2) + '\n')
if added:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); (E / 'cache').mkdir(exist_ok=True)
    (E / f'cache/ledger-before-{stamp}.json').write_bytes(raw)
    assert ledger_path.read_bytes() == raw, 'Concurrent ledger update; re-read and merge'
    temp = ledger_path.with_name(ledger_path.name + '.e12.tmp'); temp.write_text(json.dumps(ledger, indent=2) + '\n'); temp.replace(ledger_path)
print(json.dumps({'rows': rows, 'new_rows': len(added)}))
