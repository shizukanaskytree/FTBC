"""Finite CPU dispatcher inside a single, verified project Slurm allocation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--configs',nargs='+',required=True);p.add_argument('--parallel',type=int,required=True);a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID') and 'slurm' in Path('/proc/self/cgroup').read_text().lower()
assert 1<=a.parallel<=3
for filename in a.configs:
 out=Path(json.loads(Path(filename).read_text())['output'])
 assert not any((out/n).exists() for n in ['child-console.txt','progress.json','result.json','failure.json'])
pending=list(a.configs);active=[];finished=[]
while pending or active:
 while pending and len(active)<a.parallel:
  filename=pending.pop(0);out=Path(json.loads(Path(filename).read_text())['output']);out.mkdir(parents=True,exist_ok=True)
  log=(out/'child-console.txt').open('x')
  child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('run_intervention.py')),'--config',filename],stdout=log,stderr=subprocess.STDOUT)
  active.append((child,log,filename))
  print(json.dumps({'started':filename,'pid':child.pid}),flush=True)
 remaining=[]
 for child,log,filename in active:
  rc=child.poll()
  if rc is None:remaining.append((child,log,filename))
  else:
   log.close();record={'config':filename,'pid':child.pid,'returncode':rc};finished.append(record);print(json.dumps(record),flush=True)
 active=remaining
 if active:time.sleep(2)
print(json.dumps({'job':os.environ['SLURM_JOB_ID'],'children':finished}),flush=True)
sys.exit(int(any(r['returncode'] for r in finished)))
