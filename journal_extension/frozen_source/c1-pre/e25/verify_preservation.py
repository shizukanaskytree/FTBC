"""Final integrity of E25 (adapted from E22): every frozen file unchanged; manuscript Git HEAD, index entries and staged changes unchanged
(status recorded only); for every job tag that has run, sanity and main accepted. Usage: verify_preservation.py <job tag> [<job tag> ...]."""
import json, hashlib, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parent; tags = sys.argv[1:]; assert tags and all(t in ('c10-vgg16', 'c10-resnet20', 'c100-vgg16', 'c100-resnet20') for t in tags)
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
for t in tags:
    for phase in ['sanity', 'main']:
        assert json.loads((E / f'{t}-{phase}-acceptance.json').read_text())['passed'], (t, phase)
r = {'passed': True, 'tags': tags, 'frozen_files': len(manifest['files_sha256']), 'all_frozen_unchanged': True, 'manuscript_head_index_entries_staged_unchanged': True,
     'manuscript_status_changes_recorded_only': status_changes, 'manuscript_index_file_unchanged': current['index_sha256'] == before['index_sha256'],
     'numeric_acceptance_passed': {f'{t}-{p}': True for t in tags for p in ['sanity', 'main']}, 'protected_evaluation': False,
     'checked_utc': datetime.now(timezone.utc).isoformat()}
out = E / ('integrity-check-' + '-'.join(tags) + '.json'); out.write_text(json.dumps(r, indent=2) + '\n'); print(json.dumps(r))
