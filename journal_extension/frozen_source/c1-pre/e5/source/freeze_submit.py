"""E5 freeze and persistent Slurm submission; never direct GPU execution."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
E=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def freeze():
    target=E/'protocol-freeze.json'; assert not target.exists()
    disposition=json.loads((E/'review/disposition.json').read_text())
    assert disposition['release'] and not disposition['unresolved_blocking']
    files={}
    for n in ['protocol-freeze.json','objective-freeze.json']:
        p=E.parent/'e4'/n
        if p.exists(): files.update(json.loads(p.read_text())['files_sha256']); files[str(p)]=sha(p)
    paths=list((E/'source').glob('*.py'))+list((E/'config').glob('*.json'))
    paths += [E/n for n in ['EXPERIMENT_PLAN.md','research_contract.md','input-roles.json','parent-config.json','input-preparation.json','exclusion-audit.json','cache/cpu-check.json','review/disposition.json','review/final.json']]
    c=json.loads((E/'config/S.json').read_text())
    paths += [Path(c['e4_objective_output'])/f'bias-{n}.pt' for n in ['kl-lr0.001-e4','rate256-e1','rate256-e4','rate256-e8']]
    for p in paths: files[str(p)]=sha(p)
    for p,h in files.items(): assert sha(p)==h,p
    target.write_text(json.dumps({'schema':'ftbc.e5.v1','frozen_at':datetime.now(timezone.utc).isoformat(),'authorized_development_execution':True,'protected_evaluation':False,'files_sha256':files,'review':disposition},indent=2)+'\n')
    w=Path(c['workspace']); staged={str(Path(p).relative_to(w)):h for p,h in files.items() if Path(p).is_relative_to(w)}
    staged[str(target.relative_to(w))]=sha(target)
    (E/'slurm-source-manifest.json').write_text(json.dumps({'files_sha256':staged},indent=2)+'\n')
    print(json.dumps({'files':len(files),'manifest':str(target)}))

def submit(phase):
    f=json.loads((E/'protocol-freeze.json').read_text())
    for p,h in f['files_sha256'].items(): assert sha(p)==h,p
    if phase!='sanity':
        a=json.loads((E/'sanity-acceptance.json').read_text()); assert a['passed'] and a['sum_process_peak_reserved_bytes']<48*2**30
    receipt=E/f'slurm/{phase}-submit.json'; receipt.parent.mkdir(parents=True,exist_ok=True); assert not receipt.exists(),'Reconcile existing receipt instead of duplicate submission'
    configs=['sanity-'+m for m in ['S','G','L']] if phase=='sanity' else ['S','G','L'] if phase=='fitting' else ['vector']
    c=json.loads((E/f'config/{configs[0]}.json').read_text())
    command=['gpurun','--project','ftbc-neural-networks','--run-id',f'c1pre-diagnosis-e5-{phase}-v1','--gpu','a100','--gpus','1','--cpus','24' if phase!='vector' else '16','--mem','128G','--time',{'sanity':'00:30:00','fitting':'02:00:00','vector':'01:00:00'}[phase],'--allowed-uuid',c['allowed_uuid'],'--chdir',c['workspace'],'--output',str(E/f'slurm/{phase}-%j.out'),'--source-manifest',str(E/'slurm-source-manifest.json'),'--','/home/wxf/miniconda3/envs/snn/bin/python','-B',str(E/'source/run_packed.py'),'--configs']+[str(E/f'config/{n}.json') for n in configs]
    r=subprocess.run(command,capture_output=True,text=True)
    receipt.write_text(json.dumps({'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
    print(r.stdout); print(r.stderr); assert r.returncode==0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','sanity','fitting','vector']);a=p.parse_args()
    freeze() if a.action=='freeze' else submit(a.action)
