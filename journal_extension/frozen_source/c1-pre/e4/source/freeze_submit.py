"""Persist scientific identity and submit through gpurun; no direct GPU work."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def freeze(e, objective):
    c = json.loads((e / ('config/objective.json' if objective else 'config/scale.json')).read_text())
    target = Path(c['freeze_manifest'])
    assert not target.exists()
    disposition = json.loads((e / ('review/objective-disposition.json' if objective else 'review/disposition.json')).read_text())
    assert disposition['release'] and not disposition['unresolved_blocking']
    files = dict(json.loads(Path(c['prior_freeze']).read_text())['files_sha256'])
    names = ['source/common.py', 'source/freeze_submit.py', 'source/accept_and_summarize.py', 'research_contract.md', 'EXPERIMENT_PLAN.md']
    if objective:
        names += ['source/run_objective_probe.py', 'OBJECTIVE_PROTOCOL.md', 'config/objective.json', 'cache/cpu-objective.json', 'review/objective-disposition.json', 'review/objective-final.json']
    else:
        names += ['source/run_interventions.py', 'config/sanity.json', 'config/scale.json', 'config/layer.json', 'cache/cpu-scale.json', 'cache/cpu-layer.json', 'review/disposition.json', 'review/final.json']
    paths = [e / n for n in names] + [Path(c[k]) for k in ['prior_freeze','parent_proxy','parent_deployment','parent_predictions','calibrated_deployment','calibrated_predictions']]
    for p in paths: files[str(p)] = digest(p)
    for p, h in files.items(): assert digest(p) == h, p
    target.write_text(json.dumps({'schema': 'ftbc.e4.objective.v1' if objective else 'ftbc.e4.interventions.v1', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'files_sha256': files, 'authorized_development_execution': True, 'protected_evaluation': False, 'review': disposition}, indent=2)+'\n')
    w = Path(c['workspace'])
    staged = {str(Path(p).relative_to(w)): h for p,h in files.items() if Path(p).is_relative_to(w)}
    staged[str(target.relative_to(w))] = digest(target)
    out = e / ('objective-slurm-manifest.json' if objective else 'slurm-source-manifest.json')
    out.write_text(json.dumps({'files_sha256': staged}, indent=2)+'\n')
    print(json.dumps({'frozen': str(target), 'files': len(files)}))


def submit(e, phase):
    c = json.loads((e / f'config/{phase}.json').read_text())
    manifest = json.loads(Path(c['freeze_manifest']).read_text())
    assert manifest['authorized_development_execution'] and not manifest['protected_evaluation']
    for p,h in manifest['files_sha256'].items(): assert digest(p) == h, p
    if phase != 'sanity':
        a = json.loads((e / 'sanity-acceptance.json').read_text()); assert a['passed']
    receipt = e / f'slurm/{phase}-submit.json'
    assert not receipt.exists(), 'Existing receipt: reconcile job, do not duplicate'
    objective = phase == 'objective'
    duration = '00:30:00' if phase == 'sanity' else '03:00:00' if objective else '02:00:00'
    command = ['gpurun', '--project', 'ftbc-neural-networks', '--run-id', f'c1pre-diagnosis-e4-{phase}-v1', '--gpu', 'a100', '--gpus', '1', '--cpus', '16', '--mem', '128G', '--time', duration, '--allowed-uuid', c['allowed_uuid'], '--chdir', c['workspace'], '--output', str(e / f'slurm/{phase}-%j.out'), '--source-manifest', str(e / ('objective-slurm-manifest.json' if objective else 'slurm-source-manifest.json')), '--', '/home/wxf/miniconda3/envs/snn/bin/python', '-B', str(e / ('source/run_objective_probe.py' if objective else 'source/run_interventions.py')), '--config', str(e / f'config/{phase}.json')]
    r = subprocess.run(command, capture_output=True, text=True)
    receipt.write_text(json.dumps({'command': command, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}, indent=2)+'\n')
    print(r.stdout); print(r.stderr); assert r.returncode == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze','submit']); p.add_argument('phase', choices=['interventions','sanity','scale','layer','objective']); a = p.parse_args()
    e = Path(__file__).resolve().parents[1]
    if a.action == 'freeze': freeze(e, a.phase == 'objective')
    else: submit(e, a.phase)
