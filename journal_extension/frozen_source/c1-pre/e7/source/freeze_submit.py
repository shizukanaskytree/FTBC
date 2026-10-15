import argparse,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
E=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def freeze():
 p=E/'protocol-freeze.json';assert not p.exists()
 d=json.loads((E/'review/disposition.json').read_text());assert d['release'] and not d['unresolved_blocking']
 for x,h in json.loads((E/'cache/cpu-check.json').read_text())['files_sha256'].items():assert sha(x)==h,'CPU check stale: '+x
 parent=E.parent/'e6/protocol-freeze.json';files=dict(json.loads(parent.read_text())['files_sha256']);files[str(parent)]=sha(parent)
 paths=list((E/'source').glob('*.py'))+list((E/'config').glob('*.json'))
 paths += [E/'EXPERIMENT_PLAN.md',E/'cache/cpu-check.json',E/'review/disposition.json']
 paths += list((E/'review').glob('*-final.json'))
 for c in (E/'config').glob('*.json'):
  for ref in json.loads(c.read_text())['references'].values():paths += [Path(ref['bias']),Path(ref['predictions'])]
 for x in paths:files[str(x)]=sha(x)
 for x,h in files.items():assert sha(x)==h,x
 p.write_text(json.dumps({'schema':'ftbc.e7.forward-diagnostic.v1','created_utc':datetime.now(timezone.utc).isoformat(),'protected_evaluation':False,'files_sha256':files},indent=2)+'\n')
 W=Path(json.loads((E/'config/main.json').read_text())['workspace']);staged={str(Path(x).relative_to(W)):h for x,h in files.items() if Path(x).is_relative_to(W)}
 staged[str(p.relative_to(W))]=sha(p)
 (E/'slurm-source-manifest.json').write_text(json.dumps({'files_sha256':staged},indent=2)+'\n');print('Frozen',len(files),'entries')
def submit(which):
 for p,h in json.loads((E/'protocol-freeze.json').read_text())['files_sha256'].items():assert sha(p)==h,p
 if which=='main':assert json.loads((E/'sanity-acceptance.json').read_text())['passed']
 path=E/f'slurm/{which}-submit.json';assert not path.exists(),'Reconcile previous receipt first'
 c=json.loads((E/'config'/f'{which}.json').read_text())
 cmd=['gpurun','--project','ftbc-neural-networks','--run-id',f'c1pre-e7-{which}-v1','--gpu','a100','--gpus','1','--cpus','24','--mem','128G','--time','02:00:00','--allowed-uuid',c['allowed_uuid'],'--chdir',c['workspace'],'--output',str(E/f'slurm/{which}-%j.out'),'--source-manifest',str(E/'slurm-source-manifest.json'),'--','/home/wxf/miniconda3/envs/snn/bin/python','-B',str(E/'source/run_diagnostic.py'),'--config',str(E/'config'/f'{which}.json')]
 r=subprocess.run(cmd,capture_output=True,text=True);path.write_text(json.dumps({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');print(r.stdout,r.stderr);assert r.returncode==0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','sanity','main']);a=p.parse_args()
 freeze() if a.action=='freeze' else submit(a.action)
