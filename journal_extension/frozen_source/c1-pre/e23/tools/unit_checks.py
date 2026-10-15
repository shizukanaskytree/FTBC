"""E23 unit checks on CPU (initial review B1, B2, B4, N10), run by e23_check_cpu.py in a subprocess; never initializes CUDA (the CUDA timing and
allocator calls are replaced by stand-ins in this process only). Usage: unit_checks.py <scratch_dir>. Prints one JSON line.
(a) e23_timing.timed: exclusive part times with nesting (target -> write -> hash), a materialized loader with unchanged items, the remainder
fit_seconds, the baseline increments, the monotonic window and the allocator counter increases, every wrap undone (module attributes restored,
instance overrides deleted), also when the call raises.
(b) e23_r34_runner.oom_basis (follow-up 1 B5): an allocator OOM; a library signature with num_ooms grown; with less than 1 GiB free; a
signature without evidence (None); an AssertionError carrying a signature (None); an unrelated RuntimeError (None). cuda_alive on CPU stand-ins.
(c) e23_r34_accept helpers: timing_sane accepts consistent records and rejects overlapping monotonic windows, reserved below allocated and
parts that do not add up; cost_table's two time calibers (the release-default data step added to the wall), forward counts and out-of-memory
rows; exclusivity readings from synthesized nvidia-smi rows (covered without other processes, a gap above GAP s, a foreign PID whose used
memory reads [N/A], a cut last line, '[N/A]' power, a missing log)."""
import importlib.util, json, sys, time, types
from pathlib import Path
E = Path(__file__).resolve().parents[1]; SCRATCH = Path(sys.argv[1]); SCRATCH.mkdir(parents=True, exist_ok=True)
import torch
REAL_IS_INITIALIZED = torch.cuda.is_initialized
torch.cuda.synchronize = lambda *a, **k: None; torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
torch.cuda.memory_allocated = lambda *a, **k: 1 << 30; torch.cuda.memory_reserved = lambda *a, **k: 2 << 30
torch.cuda.max_memory_allocated = lambda *a, **k: 3 << 30; torch.cuda.max_memory_reserved = lambda *a, **k: 4 << 30
STATS = {'num_ooms': 5, 'num_alloc_retries': 1}; FREE = [40 << 30]
torch.cuda.memory_stats = lambda *a, **k: dict(STATS); torch.cuda.mem_get_info = lambda *a, **k: (FREE[0], 80 << 30)
torch.Tensor.cuda = lambda self, *a, **k: self


def load(n, p):
    s = importlib.util.spec_from_file_location(n, str(p)); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m


out = {}
# (a) timing harness
TM = load('e23_timing', E / 'source/e23_timing.py')
class R:
    def free(self): pass
    def target(self): time.sleep(0.10); return self.e0.atomic_torch()
mod = types.ModuleType('fake_e0')
def loader(n):
    for i in range(n):
        time.sleep(0.05); yield i
def atomic_torch(): time.sleep(0.05); return mod.tensor_hash()
def tensor_hash(): time.sleep(0.03); return 7
mod.loader, mod.atomic_torch, mod.tensor_hash = loader, atomic_torch, tensor_hash
r = R(); r.e0 = mod
W = [(mod, 'loader', 'data', True), (r, 'target', 'target', False), (mod, 'atomic_torch', 'write', False), (mod, 'tensor_hash', 'hash', False)]
def fn():
    assert list(r.e0.loader(3)) == [0, 1, 2]
    v = r.target(); time.sleep(0.2); STATS['num_ooms'] += 2; STATS['num_alloc_retries'] += 1; return v      # a caught allocation failure inside the call
res, rec = TM.timed(r, fn, W); p = rec['parts_seconds']
assert res == 7 and abs(p['data'] - 0.15) < 0.04 and abs(p['target'] - 0.10) < 0.04 and abs(p['write'] - 0.05) < 0.04 and abs(p['hash'] - 0.03) < 0.03, p
assert abs(rec['fit_seconds'] - 0.2) < 0.05 and abs(rec['seconds'] - rec['fit_seconds'] - sum(p.values())) < 1e-9, rec
assert rec['peak_allocated_increment_bytes'] == 2 << 30 and rec['peak_reserved_increment_bytes'] == 2 << 30 and rec['unix_start'] < rec['unix_end']
assert rec['perf_start'] < rec['perf_end'] and abs(rec['perf_end'] - rec['perf_start'] - rec['seconds']) < 1e-9
assert rec['num_ooms_increase'] == 2 and rec['num_alloc_retries_increase'] == 1, rec
assert mod.loader is loader and mod.atomic_torch is atomic_torch and mod.tensor_hash is tensor_hash and 'target' not in vars(r) and r.target.__func__ is R.target
try:
    TM.timed(r, lambda: (r.target(), 1 / 0), W); raise AssertionError('the exception was swallowed')
except ZeroDivisionError:
    pass
assert mod.loader is loader and mod.atomic_torch is atomic_torch and 'target' not in vars(r)
out['timing'] = {'parts_seconds': p, 'fit_seconds': rec['fit_seconds'], 'wraps_restored': True, 'restored_after_exception': True, 'counter_increases': [2, 1]}
# (b) the out-of-memory basis of the release-default arms
RN = load('e23_r34_runner', E / 'source/e23_r34_runner.py'); before = STATS['num_ooms']
cases = {}
def basis(exc, grow=0, free=40 << 30):
    STATS['num_ooms'] = before + grow; FREE[0] = free; b = RN.oom_basis(exc, before); STATS['num_ooms'] = before; return b
cases['allocator'] = basis(torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB'))
cases['signature_ooms_grew'] = basis(RuntimeError('GET was unable to find an engine to execute this computation'), grow=3)
cases['signature_low_free'] = basis(RuntimeError('cuDNN error: CUDNN_STATUS_INTERNAL_ERROR'), free=512 << 20)
cases['signature_no_evidence'] = basis(RuntimeError('CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)'))
cases['assertion_with_signature'] = basis(AssertionError('CUDNN_STATUS_INTERNAL_ERROR'), grow=3, free=512 << 20)
cases['unrelated_runtime_error'] = basis(RuntimeError('mat1 and mat2 shapes cannot be multiplied'), grow=3, free=512 << 20)
assert cases['allocator'] == ('exception', None, None) and cases['signature_ooms_grew'] == ('library', 3, 40 << 30), cases
assert cases['signature_low_free'] == ('library', 0, 512 << 20) and cases['signature_no_evidence'] == (None, None, None), cases
assert cases['assertion_with_signature'] == (None, None, None) and cases['unrelated_runtime_error'] == (None, None, None), cases
assert RN.cuda_alive()
out['oom_basis'] = {k: v[0] for k, v in cases.items()}
# (c) acceptance helpers
sys.path.insert(0, str(E / 'source')); A = __import__('e23_r34_accept')
def arm(t0, dur, method='P0', **kw):
    d = {'T': 2, 'method': method, 'seconds': dur, 'parts_seconds': {'data': 0.1, 'write': 0.05}, 'fit_seconds': dur - 0.15, 'unix_start': t0, 'unix_end': t0 + dur,
         'perf_start': t0 - P, 'perf_end': t0 - P + dur, 'num_ooms_increase': 0, 'num_alloc_retries_increase': 0,
         'peak_allocated_bytes': 3 << 30, 'peak_reserved_bytes': 4 << 30, 'baseline_allocated_bytes': 1 << 30, 'baseline_reserved_bytes': 2 << 30,
         'peak_allocated_increment_bytes': 2 << 30, 'peak_reserved_increment_bytes': 2 << 30, 'fit_images': 4}
    d.update(kw); return d
t = time.time() - 1000; P = 1e5; calls = {'ann': 99, 'snn': 99, 'updates': {}}   # P: the perf clock runs at an arbitrary offset
cost = {'P0-4-T2-r1': arm(t, 20, layer_seconds_sum=10, captures=33, capture_batches=1, cache_bytes_max=1),
        'F0reg1-T2-r1': arm(t + 21, 30, method='FTBC registered, 1 pass', batches=3, batch_size=4, calls=calls),
        'F0rel-T2-r1': {'T': 2, 'method': 'FTBC release default', 'status': 'out_of_memory', 'stage': 'update', 'exception_type': 'OutOfMemoryError', 'peak_allocated_bytes': 1,
                        'peak_reserved_bytes': 2, 'card_total_bytes': 80 << 30, 'data_seconds': 2.0, 'attempt_unix_start': t + 52, 'attempt_unix_end': t + 70, 'leftover_hooks_dropped': 1,
                        'attempt_perf_start': t + 52 - P, 'attempt_perf_end': t + 70 - P, 'oom_basis': 'exception', 'num_ooms_increase': 1, 'device_free_bytes': None},
        'F0rel-T2-r2': arm(t + 75, 40, method='FTBC release default', batches=3, batch_size=4, calls=calls, data_seconds=5.0, attempt_unix_start=t + 71, status='completed',
                           attempt_perf_start=t + 71 - P, parts_seconds={}, fit_seconds=40)}
done = ['P0-4-T2-r1', 'F0reg1-T2-r1', 'F0rel-T2-r2']
A.timing_sane(cost, done, 80 << 30)
A.timing_sane(dict(cost, **{'F0reg1-T2-r1': dict(cost['F0reg1-T2-r1'], unix_start=t + 5, unix_end=t + 6)}), done, 80 << 30)   # a unix step (NTP) overlapping P0's unix window is not an overlap
for bad in [dict(cost['F0reg1-T2-r1'], perf_start=t + 15 - P), dict(cost['F0reg1-T2-r1'], peak_reserved_bytes=1 << 30), dict(cost['F0reg1-T2-r1'], fit_seconds=1.0)]:
    try:
        A.timing_sane(dict(cost, **{'F0reg1-T2-r1': bad}), done, 80 << 30); raise RuntimeError('timing_sane accepted an inconsistent record')
    except AssertionError:
        pass
tab = A.cost_table(cost, list(cost), 8)
assert tab['F0rel-T2']['r2']['wall_seconds'] == 45 and tab['F0rel-T2']['r2']['wall_minus_data_seconds'] == 40 and tab['F0rel-T2']['r2']['data_seconds'] == 5
assert tab['F0rel-T2']['r1']['status'] == 'out_of_memory' and 'r1_minus_r2' not in tab['F0rel-T2']
assert tab['P0-4-T2']['r1']['forwards'] == {'teacher': 1, 'prefix_capture': 33, 'full_replay': 1, 'batch_size': 8}
assert tab['F0reg1-T2']['r1']['forwards'] == {'ann_full': 99, 'snn_full': 99, 'batch_size': 4} and abs(tab['P0-4-T2']['r1']['wall_minus_data_seconds'] - 19.9) < 1e-9
slurm = SCRATCH / 'slurm'; slurm.mkdir(exist_ok=True)
st = lambda x: time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(x)) + f'.{int((x % 1) * 1000):03d}'
rows = [f'{st(x)}, U, 111, 2274' for x in [t - 5 + i for i in range(130)] if not (t + 25 < x < t + 45)] + [f'{st(t + 80)}, U, 999, [N/A]', f'{st(t + 30)}, U, 222, 10']
(slurm / 'memstats-9.jsonl').write_text(json.dumps({'pid': 222}) + '\n')                    # 222: another runner of this job, not external
(slurm / 'gpuapps-9.csv').write_text('\n'.join(sorted(rows)) + '\n2026/1')
(slurm / 'gpumem-9.csv').write_text('\n'.join(f'{st(t - 5 + i / 2)}, U, 2297, 81559, 1980, [N/A], 58' for i in range(260)) + '\n')
ex = A.exclusivity(9, 111, cost, slurm); rd = {a: v['reading'] for a, v in ex['arms'].items()}
assert rd == {'P0-4-T2-r1': 'no_other_process_seen', 'F0reg1-T2-r1': 'unknown', 'F0rel-T2-r1': 'no_other_process_seen', 'F0rel-T2-r2': 'possibly_affected'}, rd
assert ex['other_compute_pids'] == [999] and ex['arms']['P0-4-T2-r1']['device']['power_w_max'] is None and ex['arms']['P0-4-T2-r1']['device']['sm_clock_mhz_median'] == 1980
assert A.exclusivity(10, 111, cost, slurm)['arms_unknown'] == sorted(cost)
out['accept_helpers'] = {'readings': rd, 'timing_sane_rejects': ['monotonic overlap', 'reserved below allocated', 'parts not adding up'], 'unix_step_not_overlap': True,
                         'foreign_pid_with_na_memory_detected': True, 'calibers': tab['F0rel-T2']['r2']['wall_seconds']}
out['cuda_initialized'] = REAL_IS_INITIALIZED(); assert not out['cuda_initialized']
print(json.dumps(out))
