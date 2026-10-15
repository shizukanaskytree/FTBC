"""Freeze additive E5-M measurements and submit through gpurun only."""
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
 target=E/'protocol-freeze.json';assert not target.exists()
 d=json.loads((E/'review/disposition.json').read_text());assert d['release'] and not d['unresolved_blocking']
 parent=E.parent/'e5/protocol-freeze.json';files=dict(json.loads(parent.read_text())['files_sha256']);files[str(parent)]=sha(parent)
 paths=list((E/'source').glob('*.py'))+list((E/'configs').glob('*.json'))+[E/p for p in ['EXPERIMENT_PLAN.md','research_contract.md','cache/cpu-check.json','cache/cpu-G.json','cache/cpu-L.json','cache/static-scope-check.json','review/disposition.json','review/final.json']]
 for c in (E/'configs').glob('*.json'):paths += [Path(p) for p in json.loads(c.read_text())['states'].values()]
 for p in paths:files[str(p)]=sha(p)
 for p,h in files.items():assert sha(p)==h,p
 target.write_text(json.dumps({'schema':'ftbc.e5.measurement.v1','frozen_at':datetime.now(timezone.utc).isoformat(),'posthoc':True,'fit_only':True,'training':False,'protected_evaluation':False,'files_sha256':files},indent=2)+'\n')
 w=Path(json.loads((E/'configs/S.json').read_text())['workspace']);staged={str(Path(p).relative_to(w)):h for p,h in files.items() if Path(p).is_relative_to(w)};staged[str(target.relative_to(w))]=sha(target)
 (E/'slurm-source-manifest.json').write_text(json.dumps({'files_sha256':staged},indent=2)+'\n');print('Frozen',len(files))

def submit(phase):
 for p,h in json.loads((E/'protocol-freeze.json').read_text())['files_sha256'].items():assert sha(p)==h,p
 if phase=='main':
  a=json.loads((E/'sanity-acceptance.json').read_text());assert a['passed'] and a['sum_process_peak_reserved_bytes']<48*2**30
 receipt=E/f'slurm/{phase}-submit.json';assert not receipt.exists(),'Reconcile existing receipt; never duplicate.'
 names=[('sanity-' if phase=='sanity' else '')+g for g in 'SGL'];c=json.loads((E/'configs'/f'{names[0]}.json').read_text())
 cmd=['gpurun','--project','ftbc-neural-networks','--run-id',f'c1pre-e5-measurement-{phase}-v1','--gpu','a100','--gpus','1','--cpus','24','--mem','128G','--time','00:30:00','--allowed-uuid',c['allowed_uuid'],'--chdir',c['workspace'],'--output',str(E/f'slurm/{phase}-%j.out'),'--source-manifest',str(E/'slurm-source-manifest.json'),'--','/home/wxf/miniconda3/envs/snn/bin/python','-B',str(E/'source/run_packed.py'),'--configs']+[str(E/'configs'/f'{n}.json') for n in names]
 r=subprocess.run(cmd,capture_output=True,text=True);receipt.write_text(json.dumps({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');print(r.stdout,r.stderr);assert r.returncode==0

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','sanity','main']);a=p.parse_args();freeze() if a.action=='freeze' else submit(a.action)
