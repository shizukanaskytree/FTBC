"""Run the unchanged archived numerical test; CPU only, no benchmark data."""
from pathlib import Path
import runpy
import shutil
import torch

ROOT = Path(__file__).resolve().parents[1]
stage = ROOT / 'frozen_source/c1-pre/e14'
assert not torch.cuda.is_initialized()
(stage / 'cache').mkdir(exist_ok=True)
runpy.run_path(str(stage / 'source/check_step_solver.py'), run_name='__main__')
assert not torch.cuda.is_initialized()
(ROOT / 'generated').mkdir(exist_ok=True)
shutil.copy2(stage / 'cache/step-solver-check.json', ROOT / 'generated/step-solver-check.json')
