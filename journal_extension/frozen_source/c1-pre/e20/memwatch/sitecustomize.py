"""E20 instrumentation (EXPERIMENT_PLAN.md section 8; adapted from par-e17e19). e20_job.py puts this directory on PYTHONPATH, so every Python process of the job imports
this module at startup. It only registers an exit hook. At exit it appends one record to the file named by PAR_MEMSTATS for every process that initialized CUDA and for
every E20 runner process (e20_r34_runner.py, e20_vgg_runner.py), whether or not it used the GPU; the record carries cuda_initialized, and for a CUDA process the
caching-allocator counters (num_ooms, num_alloc_retries) and peaks. It reads statistics only and never raises, so it cannot change any computation. A process killed
by a signal leaves no record. num_ooms counts the out-of-memory errors the CUDA caching allocator raised, the author's criterion (no OOM). It does not see failures
outside the allocator, e.g. lazy module loading, which can also make cuDNN switch algorithms (EXPERIMENT_PLAN.md section 8)."""
import atexit, json, os, sys, time
RUNNERS = ('e20_r34_runner.py', 'e20_vgg_runner.py')
_PID = os.getpid()                                  # a forked child (e.g. a DataLoader worker) never writes the parent's record


def _report():
    try:
        path = os.environ.get('PAR_MEMSTATS')
        if not path or os.getpid() != _PID:
            return
        torch = sys.modules.get('torch'); cuda = bool(torch is not None and torch.cuda.is_initialized())
        runner = bool(sys.argv) and os.path.basename(sys.argv[0]) in RUNNERS
        if not cuda and not runner:
            return
        rec = {'pid': os.getpid(), 'argv': sys.argv, 'unix_time': time.time(), 'cuda_initialized': cuda, 'num_ooms': None, 'num_alloc_retries': None,
               'max_memory_reserved': None, 'max_memory_allocated': None, 'alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
        if cuda:
            s = torch.cuda.memory_stats()
            rec.update(num_ooms=s.get('num_ooms'), num_alloc_retries=s.get('num_alloc_retries'), max_memory_reserved=torch.cuda.max_memory_reserved(),
                       max_memory_allocated=torch.cuda.max_memory_allocated())
        with open(path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    except Exception as e:                              # never affect the process; leave a trace instead
        try:
            sys.stderr.write(f'E20 memstats hook failed: {e!r}\n')
        except Exception:
            pass


atexit.register(_report)
