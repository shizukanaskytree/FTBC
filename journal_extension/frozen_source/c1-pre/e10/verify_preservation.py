import json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
E=Path(__file__).resolve().parent
manifest=json.loads((E/'protocol-freeze.json').read_text())
for p,h in manifest['files_sha256'].items():
 d=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):d.update(b)
 assert d.hexdigest()==h,p
before=json.loads((E/'preservation-before.json').read_text())
current={key:subprocess.check_output(['git','-C',str(Path('/data3/snn_prjs/ftbc_neural_networks/manuscript')),*args],text=True) for key,args in [('HEAD',['rev-parse','HEAD']),('status',['status','--porcelain=v1','--untracked-files=all'])]}
current['index_sha256']=hashlib.sha256(Path('/data3/snn_prjs/ftbc_neural_networks/manuscript/.git/index').read_bytes()).hexdigest()
assert current==before,'Manuscript Git changed; investigate rather than overwrite'
accept=json.loads((E/'main-acceptance.json').read_text());assert accept['passed']
r={'passed':True,'frozen_files':len(manifest['files_sha256']),'all_frozen_unchanged':True,'manuscript_head_status_index_unchanged':True,'numeric_acceptance_passed':True,'protected_evaluation':False,'checked_utc':datetime.now(timezone.utc).isoformat()}
(E/'integrity-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
