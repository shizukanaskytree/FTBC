"""E23 timing harness (EXPERIMENT_PLAN.md section 4; initial review B1, B2, B4).

`timed(r, fn, wraps)` empties the allocator cache, resets the peak counters and reads the resident baseline, then times `fn` with a GPU
synchronization at both ends. While `fn` runs, every wrapped callable (object, attribute, part, materialize) is timed with a synchronization
before and after; its exclusive time (its own time minus that of the wrapped calls it makes) is added to its part, so no second is counted
twice. A wrapped loader is materialized inside its own timing, so reading and decoding count as data. Every loader call reachable from a
timed arm consumes the loader to the end (list(...) or a full loop: E15 use_rows and registered_batches, E16 use_rows_target, E18 use_rows and
ftbc_batches; the two next(iter(...)) calls are in the E13 and E18 constructors, outside every timed call), so the batches and their order
are unchanged. Every wrap is undone in a finally block. The frozen code is not modified; only attribute lookups are redirected for the
duration of one call. Not wrapped, so counted in the remainder: the method's own work, the progress prints, and the zero-start hash of
T x channels floats in both FTBC paths (inline hashlib in E16, bias_hash in E18).

The record: the wall seconds, the parts, the remainder `fit_seconds` (wall minus all parts: the method's own capture, solve and update work
plus whatever is not wrapped), the monotonic start and end (perf_counter; the arm-window overlap check, follow-up 1 c), the absolute start and
end (unix; only to align with the nvidia-smi rows), the increase of the allocator's num_ooms and num_alloc_retries counters during the call
(follow-up 1 B5: an arm that completed after a caught allocation failure, e.g. a cuDNN algorithm fallback, is flagged), and the peak and
baseline allocator memory with their difference (the resident networks are in the baseline)."""
import time
import torch


def counters():
    s = torch.cuda.memory_stats()                      # {} before CUDA is initialized
    return s.get('num_ooms', 0), s.get('num_alloc_retries', 0)


def timed(r, fn, wraps=()):
    r.free(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base_alloc, base_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved(); ooms0, retries0 = counters()
    parts, stack, undo = {}, [], []

    def wrapper(orig, part, materialize):
        def f(*a, **k):
            torch.cuda.synchronize(); t0 = time.perf_counter(); stack.append(0.0)
            try:
                out = orig(*a, **k)
                if materialize:
                    out = iter(list(out))
                torch.cuda.synchronize()
            finally:
                elapsed = time.perf_counter() - t0; child = stack.pop()
                parts[part] = parts.get(part, 0.0) + elapsed - child
                if stack:
                    stack[-1] += elapsed
            return out
        return f

    try:
        for obj, name, part, materialize in wraps:
            own = name in vars(obj); orig = getattr(obj, name); undo.append((obj, name, orig, own))
            setattr(obj, name, wrapper(orig, part, materialize))
        unix_start = time.time(); t0 = time.perf_counter()
        out = fn(); torch.cuda.synchronize(); t1 = time.perf_counter(); seconds = t1 - t0; unix_end = time.time()
    finally:
        for obj, name, orig, own in reversed(undo):
            if own:
                setattr(obj, name, orig)
            else:
                delattr(obj, name)
    pa, pr = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(); ooms1, retries1 = counters()
    rec = {'seconds': seconds, 'parts_seconds': dict(parts), 'fit_seconds': seconds - sum(parts.values()), 'perf_start': t0, 'perf_end': t1,
           'unix_start': unix_start, 'unix_end': unix_end, 'num_ooms_increase': ooms1 - ooms0, 'num_alloc_retries_increase': retries1 - retries0,
           'peak_allocated_bytes': pa, 'peak_reserved_bytes': pr, 'baseline_allocated_bytes': base_alloc, 'baseline_reserved_bytes': base_reserved,
           'peak_allocated_increment_bytes': pa - base_alloc, 'peak_reserved_increment_bytes': pr - base_reserved}
    return out, rec
