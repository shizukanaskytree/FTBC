"""Account each locked-evaluation v2 job's actual allocation once; preserve existing ledger rows (adapted from E25). Usage: record_costs.py <job tag>. Charged
seconds are the sum of every allocation segment of the job in /var/log/slurm/jobcomp.log; scontrol RunTime is recorded for cross-checking
only. One row per job (EXPERIMENT_PLAN.md section 8); a job whose sanity or main is not accepted is still accounted, with status 'not accepted'.
Whole-card jobs (ImageNet) are charged as one whole GPU (initial review B4). When the submission receipt holds an error but the job was
accepted by slurmctld (a socket timeout, as for E29 job 1000942), the job id is read from slurm/job-submit-<tag>-reconciliation.json, which
must name the same run id (initial review N7)."""
import json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
E = Path(__file__).resolve().parent; W = Path('/data3/snn_prjs/ftbc_neural_networks'); tag = sys.argv[1]; assert tag in [f'{m}-t{T}' for m in ['r34', 'vgg'] for T in (2, 4, 8)] + ['c10-vgg16', 'c10-resnet20', 'c100-vgg16', 'c100-resnet20']
ledger_path = W / 'ccfa-workfiles/experiments/ftbc-neural-networks/submission-budget-ledger.json'
raw = ledger_path.read_bytes(); ledger = json.loads(raw); added = []
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
stamp = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
receipt = E / f'slurm/job-submit-{tag}.json'
if not receipt.exists():
    raise AssertionError(f'the locked-evaluation {tag} job was never submitted')
accs = {f'{tag}-{p}': E / f'{tag}-{p}-acceptance.json' for p in ['sanity', 'main']}
acc_flags = {k: a.exists() and json.loads(a.read_text())['passed'] for k, a in accs.items()}; accepted = all(acc_flags.values())
rec = json.loads(receipt.read_text()); run_id = rec['command'][rec['command'].index('--run-id') + 1]
if rec['returncode'] == 0:
    job = json.loads(rec['stdout'])['job_id']
else:
    recon = json.loads((E / f'slurm/job-submit-{tag}-reconciliation.json').read_text()); assert recon['run_id'] == run_id, recon
    job = int(recon['job_id'])
segs = [l for l in Path('/var/log/slurm/jobcomp.log').read_text().splitlines() if l.startswith(f'JobId={job} ')]
(E / f'slurm/job-{tag}-jobcomp.txt').write_text('\n'.join(segs) + '\n')
seconds = 0; states = []
for l in segs:
    st = re.search(r'JobState=(\S+)', l).group(1); states.append(st)
    a, b = re.search(r'StartTime=(\S+)', l).group(1), re.search(r'EndTime=(\S+)', l).group(1)
    if re.search(r'NodeCnt=(\d+)', l).group(1) != '0' and a != 'Unknown' and b != 'Unknown':
        seconds += int((stamp(b) - stamp(a)).total_seconds())
assert segs and states[-1] in TERMINAL, states
try:
    s = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True); (E / f'slurm/job-{tag}-final.txt').write_text(s)
    h, m, sec = map(int, re.search(r'RunTime=(\d+):(\d+):(\d+)', s).groups()); runtime = 3600 * h + 60 * m + sec
except subprocess.CalledProcessError:
    runtime = None                                             # record already purged from slurmctld; jobcomp is authoritative
state = states[-1]; status = 'completed' if state == 'COMPLETED' and accepted else f'{state.lower()}{"" if accepted else ", not accepted"}'
whole = '--gpu-mem' not in rec['command']
share = None if whole else int(rec['command'][rec['command'].index('--gpu-mem') + 1])   # GiB of the 76-shard card budget
row = {'id': run_id, 'phase': f'route_c_locked_evaluation_v2_{tag}', 'status': status, 'slurm_state': state,
       'charged_seconds': seconds, 'segments': len(segs), 'scontrol_runtime_seconds': runtime, 'slurm_job_id': job, 'accepted_phases': acc_flags,
       'gpu_share_gib': share, 'card_shards': 76, 'share_weighted_gpu_seconds': seconds if whole else seconds * share / 76,
       'accounting': ('sum of jobcomp allocation segments of one whole-GPU allocation, one child chain' if whole else
                      'sum of jobcomp allocation segments of one shared-GPU allocation (a memory budget of gpu_share_gib on one card, compute '
                      'time-shared with other jobs), one child chain; charged_seconds counts the allocation as one GPU, the conservative upper bound; '
                      'share_weighted_gpu_seconds is the same time weighted by the share of the 76-shard card budget, for reference'),
       'acceptance': {k: str(v) if v.exists() else None for k, v in accs.items()}, 'jobcomp_records': str(E / f'slurm/job-{tag}-jobcomp.txt')}
if not any(v['id'] == row['id'] for v in ledger['runs']):
    ledger['runs'].append(row); ledger['charged_gpu_seconds'] += seconds; added.append(row)
cost = E / 'cost.json'; prev = json.loads(cost.read_text()) if cost.exists() else {'rows': []}
rows = [r for r in prev['rows'] if r['id'] != row['id']] + [row]
cost.write_text(json.dumps({'rows': rows, 'total_allocation_seconds': sum(r['charged_seconds'] for r in rows), 'GPU_hours': sum(r['charged_seconds'] for r in rows) / 3600,
                            'not_formal_latency': True}, indent=2) + '\n')
if added:
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); (E / 'cache').mkdir(exist_ok=True)
    (E / f'cache/ledger-before-{tag}-{ts}.json').write_bytes(raw)
    assert ledger_path.read_bytes() == raw, 'Concurrent ledger update; re-read and merge'
    temp = ledger_path.with_name(ledger_path.name + '.l1.tmp'); temp.write_text(json.dumps(ledger, indent=2) + '\n'); temp.replace(ledger_path)
print(json.dumps({'row': row, 'new_rows': len(added)}))
