"""Final integrity and the cross-model reading (EXPERIMENT_PLAN.md section 4 and section 6 items 1 and 6; adapted from E23). Integrity: every
frozen file unchanged; manuscript Git HEAD, index entries and staged changes unchanged (status recorded only); the sanity and main
chains (r34, vgg) are complete ('passed'): the sanity run is valid and either its gate held and the main run is valid, or the gate did not
hold and main was not run (follow-up 1 N1). Validity and results are separate (initial review B2): a valid 'fail' or 'not decided'
reading is a result, not an integrity failure. The cross-model reading of section 4.3 is written in every case: a model is
adopted for the chunked implementation only if its reading is 'pass' and no model's reading is 'fail'; the locked-evaluation release-default
cell of each model and T then follows the mapping of section 4.3 (chunked, original or infeasible)."""
import json, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parent


def model_reading(acc, m):
    """Section 4.3 for one model from its acceptances (None if absent): 'invalid' if the sanity run is missing or invalid, or the gate held
    and the main run is missing or invalid; a valid sanity run whose gate did not hold reads 'fail' if a T failed there, else 'not decided';
    otherwise the main reading."""
    s, a = acc.get(f'{m}-sanity'), acc.get(f'{m}-main')
    if not s or not s['valid']:
        return 'invalid'
    if not s['gate']:
        return 'fail' if s['any_T_fail'] else 'not decided'
    return a['reading'] if a and a['valid'] else 'invalid'


def complete(acc, m):
    """Follow-up 1 N1: a model's chain is complete when its sanity run is valid and either the gate held and the main run is valid, or the
    gate did not hold and main was not run (a pre-registered result)."""
    s, a = acc.get(f'{m}-sanity'), acc.get(f'{m}-main')
    return bool(s and s['valid'] and ((s['gate'] and a is not None and a['valid']) or (not s['gate'] and a is None)))


def combined_reading(acc):
    """Section 4.3: a model is adopted only if its reading is 'pass' and no model's reading is 'fail'. Each release-default cell of a model
    with a valid main run is chunked (adopted and CHS feasible at that T), original (REL completed at that T) or infeasible; a model without
    a valid main run follows plan v2's approach (a): the original function where E23 showed it fits (R34 T=2 and T=4), infeasible elsewhere."""
    readings = {m: model_reading(acc, m) for m in ['r34', 'vgg']}
    adopted = {m: readings[m] == 'pass' and 'fail' not in readings.values() for m in readings}
    cells = {}
    for m in readings:
        a = acc.get(f'{m}-main')
        for T in ['T2', 'T4', 'T8']:
            if a and a['valid']:
                rel = a['per_T'][T]['arms']['REL']['status'] == 'completed'
                cells[f'{m}-{T}'] = 'chunked' if adopted[m] and a['feasible_CHS'][T] else 'original' if rel else 'infeasible (显存不可行，未测)'
            else:
                cells[f'{m}-{T}'] = 'original' if (m, T) in (('r34', 'T2'), ('r34', 'T4')) else 'infeasible (显存不可行，未测)'
    return readings, adopted, cells


def main():
    manifest = json.loads((E / 'protocol-freeze.json').read_text())
    for p, h in manifest['files_sha256'].items():
        d = hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda: f.read(1048576), b''):
                d.update(b)
        assert d.hexdigest() == h, p
    M = Path('/data3/snn_prjs/ftbc_neural_networks/manuscript')
    before = json.loads((E / 'preservation-before.json').read_text())
    git = lambda *a: subprocess.check_output(['git', '--no-optional-locks', '-C', str(M), *a], text=True)
    current = {'HEAD': git('rev-parse', 'HEAD'), 'index_entries_sha256': hashlib.sha256(git('ls-files', '--stage').encode()).hexdigest(),
               'staged': git('diff', '--cached', '--name-status'), 'status': git('status', '--porcelain=v1', '--untracked-files=all'),
               'index_sha256': hashlib.sha256((M / '.git/index').read_bytes()).hexdigest()}
    assert all(current[k] == before[k] for k in ['HEAD', 'index_entries_sha256', 'staged']), 'Manuscript HEAD, index entries or staged changes differ; investigate'
    status_changes = {'added': sorted(set(current['status'].splitlines()) - set(before['status'].splitlines())),
                      'removed': sorted(set(before['status'].splitlines()) - set(current['status'].splitlines()))}
    acc = {}
    for m in ['r34', 'vgg']:
        for p in ['sanity', 'main']:
            f = E / f'{m}-{p}-acceptance.json'
            if f.exists():
                acc[f'{m}-{p}'] = json.loads(f.read_text()); assert not acc[f'{m}-{p}']['cuda_initialized'], f
    valid = {k: k in acc and acc[k]['valid'] for k in [f'{m}-{p}' for m in ['r34', 'vgg'] for p in ['sanity', 'main']]}
    chains = {m: complete(acc, m) for m in ['r34', 'vgg']}
    readings, adopted, cells = combined_reading(acc)
    r = {'passed': all(chains.values()), 'frozen_files': len(manifest['files_sha256']), 'all_frozen_unchanged': True,
         'manuscript_head_index_entries_staged_unchanged': True, 'manuscript_status_changes_recorded_only': status_changes,
         'manuscript_index_file_unchanged': current['index_sha256'] == before['index_sha256'], 'phases_valid': valid, 'chains_complete': chains,
         'readings': readings, 'adopt_chunked': adopted, 'locked_evaluation_release_default': cells,
         'protected_evaluation': False, 'checked_utc': datetime.now(timezone.utc).isoformat()}
    (E / 'integrity-check.json').write_text(json.dumps(r, indent=2) + '\n'); print(json.dumps(r))
    if not r['passed']:
        raise SystemExit('E29: a chain is not complete (its sanity run or, after a held gate, its main run is missing or invalid); the readings above follow section 4.3')


if __name__ == '__main__':
    main()
