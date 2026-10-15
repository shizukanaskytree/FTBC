"""Additive E6 freeze and finite gpurun submission. Never edits a previous freeze."""
import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
E=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def freeze():
 target=E/'protocol-freeze.json';assert not target.exists()
 d=json.loads((E/'review/disposition.json').read_text());assert d['release'] and not d['unresolved_blocking']
 parent=E.parent/'e5-measurement/protocol-freeze.json'
 files=dict(json.loads(parent.read_text())['files_sha256']);files[str(parent)]=sha(parent)
 paths=list((E/'source').glob('*.py'))+list((E/'configs').glob('*.json'))
 paths += [E/p for p in ['EXPERIMENT_PLAN.md','cache/cpu-math.json','cache/cpu-integration.json','cache/cpu-real-block.json','cache/cpu-rank-structure.json','cache/cpu-gate.json','review/disposition.json','review/code-final.json','review/code-revision-final.json','review/concept-final.json','review/concept-revision-final.json']]
 for path in paths:files[str(path)]=sha(path)
 for path,h in files.items():assert sha(path)==h,path
 target.write_text(json.dumps({'schema':'ftbc.e6.method.v1','frozen_at':datetime.now(timezone.utc).isoformat(),'training':True,'protected_evaluation':False,'files_sha256':files},indent=2)+'\n')
 w=Path(json.loads((E/'configs/comp.json').read_text())['workspace'])
 staged={str(Path(p).relative_to(w)):h for p,h in files.items() if Path(p).is_relative_to(w)}
 staged[str(target.relative_to(w))]=sha(target)
 (E/'slurm-source-manifest.json').write_text(json.dumps({'files_sha256':staged},indent=2)+'\n')
 print('Frozen',len(files))

def submit(phase):
 for path,h in json.loads((E/'protocol-freeze.json').read_text())['files_sha256'].items():assert sha(path)==h,path
 parallel=1;names=['sanity']
 if phase=='main':
  s=json.loads((E/'sanity-acceptance.json').read_text());assert s['passed'] and s['gate_passed']
  parallel=max(1,min(3,int(.8*s['total_memory']/s['max_memory_reserved'])))
  names=['comp','stop','free-ce','free-kl','beta-ce']
 receipt=E/f'slurm/{phase}-submit.json';assert not receipt.exists(),'Reconcile existing receipt before any retry.'
 c=json.loads((E/'configs'/f'{names[0]}.json').read_text())
 cmd=['gpurun','--project','ftbc-neural-networks','--run-id',f'c1pre-e6-{phase}-v1','--gpu','a100','--gpus','1','--cpus','24','--mem','128G','--time','02:00:00','--allowed-uuid',c['allowed_uuid'],'--chdir',c['workspace'],'--output',str(E/f'slurm/{phase}-%j.out'),'--source-manifest',str(E/'slurm-source-manifest.json'),'--','/home/wxf/miniconda3/envs/snn/bin/python','-B',str(E/'source/run_packed.py'),'--parallel',str(parallel),'--configs']+[str(E/'configs'/f'{n}.json') for n in names]
 r=subprocess.run(cmd,capture_output=True,text=True)
 receipt.write_text(json.dumps({'command':cmd,'parallel':parallel,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
 print(r.stdout,r.stderr);assert r.returncode==0

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','sanity','main']);a=p.parse_args()
 freeze() if a.action=='freeze' else submit(a.action)
