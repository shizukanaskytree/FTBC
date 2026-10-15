"""Final integrity (EXPERIMENT_PLAN.md section 6 items 1 and 7, section 7): every frozen file unchanged; manuscript Git HEAD, index entries and staged changes unchanged
(status recorded only); sanity and main accepted for both chains (r34, vgg; run one after the other, adapted from E22). The memory record (slurm/memory-check.json, section 7)
is descriptive in E23 and is copied here, not asserted (initial review B3)."""
import json, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parent
manifest = json.loads((E / 'protocol-freeze.json').read_text())
for p, h in manifest['files_sha256'].items():
    d = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            d.update(b)
    assert d.hexdigest() == h, p
M = Path('/data3/snn_prjs/ftbc_neural_networks/manuscript')
before = json.loads((E / 'preservation-before.json').read_text())
git = lambda *a: subprocess.check_output(['git', '--no-optional-locks', '-C', str(M), *a], text=True)
current = {'HEAD': git('rev-parse', 'HEAD'), 'index_entries_sha256': hashlib.sha256(git('ls-files', '--stage').encode()).hexdigest(),
           'staged': git('diff', '--cached', '--name-status'), 'status': git('status', '--porcelain=v1', '--untracked-files=all'),
           'index_sha256': hashlib.sha256((M / '.git/index').read_bytes()).hexdigest()}
assert all(current[k] == before[k] for k in ['HEAD', 'index_entries_sha256', 'staged']), 'Manuscript HEAD, index entries or staged changes differ; investigate'
status_changes = {'added': sorted(set(current['status'].splitlines()) - set(before['status'].splitlines())),
                  'removed': sorted(set(before['status'].splitlines()) - set(current['status'].splitlines()))}
for phase in ['r34-sanity', 'r34-main', 'vgg-sanity', 'vgg-main']:
    assert json.loads((E / f'{phase}-acceptance.json').read_text())['passed'], phase
mem = json.loads((E / 'slurm/memory-check.json').read_text())     # descriptive in E23 (sequential, solo): OOMs are compared and reported, not decisive
optional = {}
for phase in []:
    f = E / f'{phase}-acceptance.json'
    if f.exists(): assert json.loads(f.read_text())['passed']
    optional[phase] = True if f.exists() else 'absent (job not accepted)'
r = {'passed': True, 'frozen_files': len(manifest['files_sha256']), 'all_frozen_unchanged': True, 'manuscript_head_index_entries_staged_unchanged': True, 'manuscript_status_changes_recorded_only': status_changes,
     'manuscript_index_file_unchanged': current['index_sha256'] == before['index_sha256'],
     'memory_record': {k: mem.get(k) for k in ['memory_state_known', 'clean', 'num_ooms', 'recorded_increases', 'ooms_outside_arms', 'completed_arms_with_ooms',
                                               'oom_arms_without_allocator_count', 'num_alloc_retries', 'near_full']},
     'numeric_acceptance_passed': {'r34-sanity': True, 'r34-main': True, 'vgg-sanity': True, 'vgg-main': True, **optional},
     'protected_evaluation': False, 'checked_utc': datetime.now(timezone.utc).isoformat()}
(E / 'integrity-check.json').write_text(json.dumps(r, indent=2) + '\n'); print(json.dumps(r))
