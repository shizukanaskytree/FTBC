"""Locked evaluation v2 final integrity (plan sections 2 and 6; adapted from E25/E29): every frozen file unchanged; the approval names the plan
version that states this freeze; manuscript Git HEAD, index entries and staged changes unchanged (status recorded only); sanity and main
accepted for every job that ran. Usage: python3 verify_preservation.py <job tag>... Run it right after each job's main acceptance (plan
section 5). When the three ImageNet jobs of a model are all given, the source-ANN rows are compared across them (plan section 3, follow-up 1
P7): the per-image predictions must agree between jobs on the same card type; any difference is recorded, never used to choose a row."""
import json, hashlib, re, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parent; PLAN = E.parent / 'innovation/official-evaluation-plan-v2.md'


def sha(p):
    d = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            d.update(b)
    return d.hexdigest()


manifest = json.loads((E / 'protocol-freeze.json').read_text())
for p, h in manifest['files_sha256'].items():
    assert sha(p) == h, p
a = json.loads((E / 'approval.json').read_text()); assert a['plan_sha256'] == sha(PLAN) and a['author_reply'].strip() == '开启评价'
m = re.search(r'protocol-freeze\.json` 的 SHA-256：`([0-9a-f]{64})`', PLAN.read_text()); assert m and m.group(1) == sha(E / 'protocol-freeze.json')
M = Path('/data3/snn_prjs/ftbc_neural_networks/manuscript')
before = json.loads((E / 'preservation-before.json').read_text())
git = lambda *x: subprocess.check_output(['git', '--no-optional-locks', '-C', str(M), *x], text=True)
current = {'HEAD': git('rev-parse', 'HEAD'), 'index_entries_sha256': hashlib.sha256(git('ls-files', '--stage').encode()).hexdigest(),
           'staged': git('diff', '--cached', '--name-status'), 'status': git('status', '--porcelain=v1', '--untracked-files=all'),
           'index_sha256': hashlib.sha256((M / '.git/index').read_bytes()).hexdigest()}
assert all(current[k] == before[k] for k in ['HEAD', 'index_entries_sha256', 'staged']), 'Manuscript HEAD, index entries or staged changes differ; investigate'
status_changes = {'added': sorted(set(current['status'].splitlines()) - set(before['status'].splitlines())),
                  'removed': sorted(set(before['status'].splitlines()) - set(current['status'].splitlines()))}
tags = sys.argv[1:]; assert tags
for tag in tags:
    for phase in ['sanity', 'main']:
        assert json.loads((E / f'{tag}-{phase}-acceptance.json').read_text())['passed'], (tag, phase)
import torch
ann = {}
for model in ['r34', 'vgg']:                                         # plan section 3: the ANN row is reported from the T=2 job
    jobs = [f'{model}-t{T}' for T in (2, 4, 8)]
    if all(j in tags for j in jobs):
        recs = {j: json.loads((E / f'config/{j}-main.json').read_text()) for j in jobs}
        outs = {j: Path(recs[j]['output']) for j in jobs}
        card = {j: json.loads((outs[j] / 'result.json').read_text())['hardware']['name'] for j in jobs}
        pred = {j: torch.load(outs[j] / 'eval-ANN.pt', map_location='cpu', weights_only=False)['predictions'] for j in jobs}
        ann[model] = {'reported_from': jobs[0], 'cards': card,
                      'differing_images_vs_T2': {j: int((pred[j] != pred[jobs[0]]).sum()) for j in jobs[1:]},
                      'same_card_type_and_equal': {j: card[j] == card[jobs[0]] and bool(torch.equal(pred[j], pred[jobs[0]])) for j in jobs[1:]}}
r = {'passed': True, 'jobs': tags, 'frozen_files': len(manifest['files_sha256']), 'all_frozen_unchanged': True, 'plan_sha256': a['plan_sha256'],
     'ann_rows_across_jobs': ann,
     'freeze_sha256': sha(E / 'protocol-freeze.json'), 'manuscript_head_index_entries_staged_unchanged': True, 'manuscript_status_changes_recorded_only': status_changes,
     'manuscript_index_file_unchanged': current['index_sha256'] == before['index_sha256'], 'protected_evaluation': True, 'checked_utc': datetime.now(timezone.utc).isoformat()}
out = E / f'integrity-check-{"-".join(tags)}.json'; out.write_text(json.dumps(r, indent=2) + '\n'); print(json.dumps(r))
