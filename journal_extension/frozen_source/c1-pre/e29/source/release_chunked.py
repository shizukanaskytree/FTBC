"""E29 (EXPERIMENT_PLAN.md sections 2-4): the released FTBC update with its forward passes split into chunks of images, the per-call pairing
of implementations under one state, and the counted fit.

The released function is bias_corr_step_by_step of the E1 export (../e1/calibration-export/serial.py), an unmodified copy of
code/ftbc-eccv/QCFS/ANN_SNN_QCFS/main_cali.py. For one interface and one batch it captures the ANN output and the SNN input of the module
with the released GetLayerInputOutput hook on the whole batch, replays the T timesteps element by element, and adds curr_t_alpha times the
channel mean of `bias` (batch and spatial dimensions, or the batch dimension of a vector interface) to time_based_bias[t].

chunked_step(inherited, chunk) returns a drop-in replacement with the same signature. It captures the same tensors chunk by chunk with the
same released hook (looked up in the export at call time, as the released function does), runs the same element-wise replay (the
expressions below are copied line by line), accumulates the per-timestep channel sums of `bias` in float64 together with the element count,
and applies the same update after the last chunk. In the released function the batch enters the update only through that channel mean and
the replay state is element-wise; time_based_bias of the module only affects the module's own output, not the captured input, and the call
writes it only after all captures. The update is therefore the released one up to floating point (convolution algorithms chosen per batch
size, summation order). The float64 sum copies one timestep of `bias` (memory only; initial review N7).

paired_step (initial review B1) runs several implementations of one call from the same saved state and keeps the released result;
release_fit (B3) counts the update calls per interface and the whole-network ANN and SNN forward passes."""
import torch


def chunked_step(inherited, chunk):
    assert isinstance(chunk, int) and chunk > 0, chunk

    def bias_corr_step_by_step_chunked(ann, module_ann, snn, module_snn, T, train_data, curr_t_alpha=0.5):
        thre = module_snn.thresh.data
        sums, count = None, 0
        for start in range(0, train_data.shape[0], chunk):
            part_data = train_data[start:start + chunk]
            ann_get_out = inherited.GetLayerInputOutput(ann, module_ann)
            ann_out = ann_get_out(part_data)[1]
            ann_get_out.data_saver.reset()

            snn_get_out = inherited.GetLayerInputOutput(snn, module_snn)
            snn_in = snn_get_out(part_data)[0]
            snn_get_out.data_saver.reset()

            snn_in = snn_in.view(T, -1, *snn_in.shape[1:])

            v_t_1 = 0.5 * thre
            part_sums = []
            for t in range(module_snn.T):
                x_l_1_t = snn_in[t, ...]
                mem_t = v_t_1 + x_l_1_t
                spike = module_snn.act(mem_t - thre, module_snn.gama) * thre

                bias = spike - ann_out
                mem_t_updated = mem_t - bias

                spike_updated = module_snn.act(mem_t_updated - thre, module_snn.gama) * thre
                v_t_updated = mem_t_updated - spike_updated

                if len(bias.shape) == 4:
                    part_sums.append(bias.double().sum(dim=[0, 2, 3]))
                elif len(bias.shape) == 2:
                    part_sums.append(bias.double().sum(dim=0))
                else:
                    raise AssertionError(f'the released update defines no channel mean for shape {tuple(bias.shape)}')
                v_t_1 = v_t_updated
            count += bias.numel() // bias.shape[1]
            sums = part_sums if sums is None else [a + b for a, b in zip(sums, part_sums)]
            del ann_out, snn_in, x_l_1_t, mem_t, spike, bias, mem_t_updated, spike_updated, v_t_updated, v_t_1, part_sums, part_data
        assert sums is not None and len(sums) == module_snn.T and count > 0
        for t in range(module_snn.T):
            bias_mean = (sums[t] / count).to(module_snn.time_based_bias[t].dtype)
            module_snn.time_based_bias[t] = curr_t_alpha * bias_mean + module_snn.time_based_bias[t]

    bias_corr_step_by_step_chunked.chunk = chunk
    return bias_corr_step_by_step_chunked


def fingerprint(x, words=1 << 22):
    """Exact-equality fingerprint of a captured tensor on its own device (B1): its raw bits as 16-bit words, weighted by two position-weight
    sets in [1, 65521] and summed exactly in int64 per slice (each product < 2^32, at most 2^22 products per slice, so no overflow), then across
    slices in Python integers. Equal tensors give equal fingerprints; changed bits change them unless the weighted changes cancel exactly in
    both sums. Returns [shape, dtype, sum1, sum2]."""
    assert x.is_contiguous(), 'fingerprint of a non-contiguous tensor'
    w16 = x.detach().reshape(-1).view(torch.int16)
    sums = [0, 0]
    for s in range(0, w16.numel(), words):
        v = w16[s:s + words].to(torch.int64) & 0xFFFF
        pos = torch.arange(s, s + v.numel(), device=v.device, dtype=torch.int64)
        sums[0] += int((v * (pos % 65521 + 1)).sum())
        sums[1] += int((v * ((pos * 40503) % 65519 + 1)).sum())
        del v, pos
    return [list(x.shape), str(x.dtype)] + sums


def paired_step(inherited, implementations, log, names):
    """B1: a step for the released bias_corr_model that runs every implementation of one call from the same state. implementations is a list
    of (name, step, fingerprinted), the released step first under the name 'REL'. For each implementation in order, the module's
    time_based_bias is restored (clones) to the state saved at the start of the call, the step runs, with the captures of the released hook
    fingerprinted if requested, and its result is kept. Afterwards the module carries the released result, so the run follows the released
    trajectory. names maps id(IF module) to its interface name. Appends one record per call to log: interface, batch index, the shape of the
    released capture of the SNN input, the fingerprints, the allocator's num_ooms before and after each implementation (follow-up 1 R1:
    evidence for a forward that changed its convolution plan under memory pressure), and max and mean of |b_a - b_b| / theta over timesteps
    and channels for every later implementation a against REL and for every pair of later implementations (keys 'CHB_vs_REL', 'CHS_vs_REL',
    'CHS_vs_CHB')."""
    assert implementations[0][0] == 'REL' and len({n for n, _, _ in implementations}) == len(implementations)
    seen = {}

    def bias_corr_step_by_step_paired(ann, module_ann, snn, module_snn, T, train_data, curr_t_alpha=0.5):
        key = names[id(module_snn)]; seen[key] = seen.get(key, 0) + 1
        theta = float(module_snn.thresh.data)
        saved = [b.detach().clone() for b in module_snn.time_based_bias]
        results, prints, ooms, base = {}, {}, {}, inherited.GetLayerInputOutput
        for name, step, fingerprinted in implementations:
            for t in range(len(saved)):
                module_snn.time_based_bias[t] = saved[t].clone()
            captured = []; ooms[name] = [torch.cuda.memory_stats().get('num_ooms', 0)]   # {} on a CPU without CUDA
            if fingerprinted:
                class GetLayerInputOutputFingerprinted(base):
                    def __call__(self, input):
                        out = super().__call__(input)
                        captured.append([fingerprint(x) for x in out])
                        return out
                inherited.GetLayerInputOutput = GetLayerInputOutputFingerprinted
            try:
                step(ann, module_ann, snn, module_snn, T, train_data, curr_t_alpha=curr_t_alpha)
            finally:
                inherited.GetLayerInputOutput = base
            results[name] = [b.detach().clone() for b in module_snn.time_based_bias]
            ooms[name].append(torch.cuda.memory_stats().get('num_ooms', 0))
            if fingerprinted:
                prints[name] = captured
        for t in range(len(saved)):
            module_snn.time_based_bias[t] = results['REL'][t]
        rec = {'interface': key, 'batch': seen[key] - 1, 'theta': theta, 'fingerprints': prints, 'compare': {}, 'num_ooms': ooms,
               'snn_input_shape': prints['REL'][1][0][0] if 'REL' in prints else None}
        later = [n for n, _, _ in implementations[1:]]
        for a, b in [(n, 'REL') for n in later] + [(later[j], later[i]) for i in range(len(later)) for j in range(i + 1, len(later))]:
            d = torch.stack([(x.double() - y.double()).abs() for x, y in zip(results[a], results[b])]) / theta
            rec['compare'][f'{a}_vs_{b}'] = {'max': float(d.max()), 'mean': float(d.mean()), 'argmax_timestep': int(d.amax(dim=1).argmax())}
        log.append(rec)

    bias_corr_step_by_step_paired.paired = [n for n, _, _ in implementations]
    return bias_corr_step_by_step_paired


def release_fit(inherited, ann, snn, T, batches, alpha, step, interfaces):
    """Run the released bias_corr_model with `step` (the released function, a chunked one or a paired one) in place of the export's
    bias_corr_step_by_step, wrapped to count one call per batch and interface, with forward pre-hooks counting the whole-network ANN and SNN
    forward passes (B3); restore the original and remove the hooks in finally. `interfaces` are the names of the IF modules of `snn` (the
    caller's net table). Returns {'updates': calls per interface, 'ann_forwards': n, 'snn_forwards': n}."""
    original = inherited.bias_corr_step_by_step
    names = {id(m): n for n, m in snn.named_modules() if isinstance(m, inherited.IF)}
    assert sorted(names.values()) == sorted(interfaces), 'the IF modules differ from the net table'
    count = {'updates': {}, 'ann_forwards': 0, 'snn_forwards': 0}

    def counted(*args, **kw):
        key = names[id(args[3])]; count['updates'][key] = count['updates'].get(key, 0) + 1
        return step(*args, **kw)
    hooks = [ann.register_forward_pre_hook(lambda *a: count.__setitem__('ann_forwards', count['ann_forwards'] + 1)),
             snn.register_forward_pre_hook(lambda *a: count.__setitem__('snn_forwards', count['snn_forwards'] + 1))]
    inherited.bias_corr_step_by_step = counted
    try:
        with torch.no_grad():
            inherited.bias_corr_model(ann, snn, T, batches, curr_t_alpha=alpha, num_cali_sample_batches=len(batches))
    finally:
        inherited.bias_corr_step_by_step = original
        for h in hooks:
            h.remove()
    assert set(count['updates']) == set(interfaces) and all(v == len(batches) for v in count['updates'].values()), count
    return count


def forwards_per_call(mode, batch, chunk):
    """B3: whole-network forward passes of the ANN (and of the SNN) per update call: REL one; a chunked step ceil(batch / chunk); the paired
    step REL + CHB (one chunk) + CHS."""
    k = -(-batch // chunk)
    return {'REL': 1, 'CHS': k, 'PAIR': 2 + k}[mode]
