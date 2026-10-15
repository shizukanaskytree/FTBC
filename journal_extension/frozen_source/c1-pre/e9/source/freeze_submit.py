import argparse,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
E=Path(__file__).resolve().parents[1]
MODES=['root','permuted','constant','full-kl','full-ce']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def freeze():
 target=E/'protocol-freeze.json';assert not target.exists()
 d=json.loads((E/'review/disposition.json').read_text());assert d['release'] and not d['unresolved_blocking']
 cpu=json.loads((E/'cache/cpu-check.json').read_text());assert cpu['passed']
 for p,h in cpu['files_sha256'].items():assert sha(p)==h,p
 old=E.parent/'e8/protocol-freeze.json';files=dict(json.loads(old.read_text())['files_sha256']);files[str(old)]=sha(old)
 paths=list((E/'source').glob('*.py'))+list((E/'config').glob('*.json'))+[E/f for f in ['EXPERIMENT_PLAN.md','analyze.py','cache/cpu-check.json','review/disposition.json','review/code-final.json','review/method-final.json']]
 for p in paths:files[str(p)]=sha(p)
 for p,h in files.items():assert sha(p)==h,p
 target.write_text(json.dumps({'schema':'ftbc.e9.direction-screen.v1','created_utc':datetime.now(timezone.utc).isoformat(),'protected_evaluation':False,'files_sha256':files},indent=2)+'\n')
 w=Path(json.loads((E/'config/main-root.json').read_text())['workspace']);staged={str(Path(p).relative_to(w)):h for p,h in files.items() if Path(p).is_relative_to(w)};staged[str(target.relative_to(w))]=sha(target)
 (E/'slurm-source-manifest.json').write_text(json.dumps({'files_sha256':staged},indent=2)+'\n');print('Frozen',len(files))
def submit(phase):
 for p,h in json.loads((E/'protocol-freeze.json').read_text())['files_sha256'].items():assert sha(p)==h,p
 parallel=3
 if phase=='main':
  a=json.loads((E/'sanity-acceptance.json').read_text());assert a['passed']
  # Main adds 240 cached fit images (<0.2GiB) and more CPU-saved records, no larger batch.
  parallel=min(3,max(1,int(.8*a['total_memory']/(max(a['max_memory_reserved'])+.5*2**30))))
 receipt=E/f'slurm/{phase}-submit.json';assert not receipt.exists(),'Reconcile previous receipt, do not duplicate'
 receipt.parent.mkdir(parents=True,exist_ok=True)
 configs=[E/'config'/f'{phase}-{m}.json' for m in MODES];c=json.loads(configs[0].read_text())
 cmd=['gpurun','--project','ftbc-neural-networks','--run-id',f'c1pre-e9-{phase}-v1','--gpu','a100','--gpus','1','--cpus','24','--mem','128G','--time','01:00:00','--allowed-uuid',c['allowed_uuid'],'--chdir',c['workspace'],'--output',str(E/f'slurm/{phase}-%j.out'),'--source-manifest',str(E/'slurm-source-manifest.json'),'--','/home/wxf/miniconda3/envs/snn/bin/python','-B',str(E/'source/run_packed.py'),'--parallel',str(parallel),'--configs']+[str(p) for p in configs]
 r=subprocess.run(cmd,capture_output=True,text=True);receipt.write_text(json.dumps({'command':cmd,'parallel':parallel,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');print(r.stdout,r.stderr);assert r.returncode==0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','sanity','main']);a=p.parse_args()
 freeze() if a.action=='freeze' else submit(a.action)
