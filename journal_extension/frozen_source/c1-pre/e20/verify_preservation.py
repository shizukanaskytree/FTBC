"""Final integrity: every frozen file unchanged; manuscript Git HEAD/status/index unchanged; sanity and main accepted; the pre-registered memory rule
held (slurm/memory-check.json valid, EXPERIMENT_PLAN.md section 7 item 12). E20: both chains (r34, vgg), sanity and main (section 8)."""
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
current = {k: subprocess.check_output(['git', '-C', str(M), *a], text=True) for k, a in [('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]}
current['index_sha256'] = hashlib.sha256((M / '.git/index').read_bytes()).hexdigest()
assert current == before, 'Manuscript Git changed; investigate rather than overwrite'
for phase in ['r34-sanity', 'r34-main', 'vgg-sanity', 'vgg-main']:
    assert json.loads((E / f'{phase}-acceptance.json').read_text())['passed'], phase
mem = json.loads((E / 'slurm/memory-check.json').read_text()); assert mem['valid_for_both_models'], 'memory rule failed: both models are infrastructure failures'
optional = {}
for phase in []:
    f = E / f'{phase}-acceptance.json'
    if f.exists(): assert json.loads(f.read_text())['passed']
    optional[phase] = True if f.exists() else 'absent (job not accepted)'
r = {'passed': True, 'frozen_files': len(manifest['files_sha256']), 'all_frozen_unchanged': True, 'manuscript_head_status_index_unchanged': True,
     'memory_rule_valid': True, 'near_full': mem['near_full'],
     'numeric_acceptance_passed': {'r34-sanity': True, 'r34-main': True, 'vgg-sanity': True, 'vgg-main': True, **optional},
     'protected_evaluation': False, 'checked_utc': datetime.now(timezone.utc).isoformat()}
(E / 'integrity-check.json').write_text(json.dumps(r, indent=2) + '\n'); print(json.dumps(r))
