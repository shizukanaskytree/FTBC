"""E22 post-run memory check (EXPERIMENT_PLAN.md section 8; the E20 check with the E22 runner names), on CPU. Usage: check_memory.py <slurm job id>. Writes slurm/memory-check.json.
Pre-registered rule: failures are first classified by each model's own protocol (chain logs). Every runner process that started (its stage line
is in the chain log) must have left exactly one exit record. A record of a process that used the GPU (cuda_initialized) must have num_ooms == 0
(the author's criterion: no OOM); a record of a runner that never initialized CUDA does not enter the memory rule (its failure is classified by its
model's protocol); a started runner without a record (e.g. killed) leaves the memory state unknown. Any OOM or unknown state makes both models
infrastructure failures, rerun separately. num_alloc_retries and the device samples (peak used, minimum free) are reported, not decision rules;
retries > 0 or a minimum free below 1 GiB sets near_full, which the results document must flag next to the decision table."""
import csv, json, sys
from pathlib import Path
E = Path(__file__).resolve().parents[1]
GIB = 2 ** 30


def stages(log):
    out = []
    for line in log.read_text(errors='replace').splitlines() if log.exists() else []:
        if line.startswith('{"stage"'):
            try:
                out.append(json.loads(line)['stage'])
            except (ValueError, KeyError):              # a runner line cut by a kill; never a chain stage line
                pass
    return out


def expected(job, slurm):
    out = {}
    for model in ['r34', 'vgg']:
        st = stages(slurm / f'{model}-{job}.out')
        for which in ['sanity', 'main']:
            if f'{model}-{which}' in st:
                out[f'{model}-{which}'] = (E / f'source/e22_{model}_runner.py', E / f'config/{model}-{which}.json')
    return out


def check(job, slurm=E / 'slurm'):
    f = slurm / f'memstats-{job}.jsonl'
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    exp = expected(job, slurm); found = {}
    for key, (runner, cfg) in exp.items():
        found[key] = [r for r in recs if len(r['argv']) >= 3 and Path(r['argv'][0]).resolve() == runner.resolve() and r['argv'][1:3] == ['--config', str(cfg)]]
    unexpected = [r for r in recs if not any(r in v for v in found.values())]
    complete = all(len(v) == 1 for v in found.values())
    one = {k: v[0] for k, v in found.items() if len(v) == 1}
    used = {k: r.get('cuda_initialized', True) for k, r in one.items()}
    ooms = {k: r['num_ooms'] for k, r in one.items() if used[k]}
    valid = bool(exp) and complete and all(n == 0 for n in ooms.values()) and not unexpected
    dev, total = [], None
    g = slurm / f'gpumem-{job}.csv'
    if g.exists():
        for row in csv.reader(g.read_text().splitlines()):
            if len(row) == 4 and row[2].strip().isdigit():
                dev.append(int(row[2])); total = int(row[3])
    retries = {k: r['num_alloc_retries'] for k, r in one.items() if used[k]}
    min_free = (total - max(dev)) / 1024 if dev else None
    return {'job': str(job), 'started_runners': sorted(exp), 'valid_for_both_models': valid, 'records_complete': complete,
            'missing_or_duplicate_records': sorted(k for k, v in found.items() if len(v) != 1), 'gpu_used': used, 'num_ooms': ooms, 'num_alloc_retries': retries,
            'process_peak_reserved_gib': {k: one[k]['max_memory_reserved'] / GIB for k in ooms}, 'process_peak_allocated_gib': {k: one[k]['max_memory_allocated'] / GIB for k in ooms},
            'alloc_conf': sorted({str(r.get('alloc_conf')) for r in recs}), 'unexpected_records': unexpected,
            'device_samples': len(dev), 'device_peak_used_gib': max(dev) / 1024 if dev else None, 'device_min_free_gib': min_free, 'device_total_gib': total / 1024 if total else None,
            'near_full': any((x or 0) > 0 for x in retries.values()) or (min_free is not None and min_free < 1),
            'rule': 'valid only if every started runner left one exit record, every record of a process that used the GPU has num_ooms == 0, and no record is unexpected; '
                    'retries and device samples are descriptive; near_full (retries > 0 or minimum free < 1 GiB) must be flagged in the results'}


if __name__ == '__main__':
    out = check(sys.argv[1]); dst = E / 'slurm/memory-check.json'; assert not dst.exists(); dst.write_text(json.dumps(out, indent=2) + '\n'); print(json.dumps(out, indent=1))
