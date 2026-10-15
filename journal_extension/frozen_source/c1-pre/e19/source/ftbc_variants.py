"""E19 ablation variants of the released serial FTBC per-interface update (EXPERIMENT_PLAN.md section 3). Ablation tools, not published methods.

Both keep the signature of the released bias_corr_step_by_step (E1 export of code/ftbc-eccv/QCFS/ANN_SNN_QCFS/main_cali.py), so the released
bias_corr_model calls them batch by batch and interface by interface, and both use the export's GetLayerInputOutput capture and the
interface's own spike function. b_t is the stored bias that IF.forward subtracts (mem = mem - b_t; mem = mem + x_t).
  FA: line for line the released update, except that the local replay includes the interface's current stored bias:
      mem_t = v_t_1 - b_t + x_t (the released replay uses mem_t = v_t_1 + x_t).
  FD: deployed-trajectory update. At step t: spike from the deployed IF with the current b_t; b_t += alpha * ChMean(spike - ann_out)
      (the released sign); then advance the membrane with the updated b_t by the deployed IF dynamics (update, then advance).
`record` (FD only, keyword, default None) collects the advanced spikes for the CPU check against the deployed IF."""
import torch


def make(inherited):
    G = inherited.GetLayerInputOutput

    def capture(ann, module_ann, snn, module_snn, T, train_data):
        ann_get_out = G(ann, module_ann); ann_out = ann_get_out(train_data)[1]; ann_get_out.data_saver.reset()
        snn_get_out = G(snn, module_snn); snn_in = snn_get_out(train_data)[0]; snn_get_out.data_saver.reset()
        return ann_out, snn_in.view(T, -1, *snn_in.shape[1:])

    def shaped(b, x):
        return b.view(1, -1, 1, 1) if x.dim() == 4 else b.view(1, -1)

    def chmean(r):
        return r.mean(dim=[0, 2, 3]) if r.dim() == 4 else r.mean(dim=0)

    def fa_step(ann, module_ann, snn, module_snn, T, train_data, curr_t_alpha=0.5):
        ann_out, snn_in = capture(ann, module_ann, snn, module_snn, T, train_data)
        thre = module_snn.thresh.data
        v_t_1 = 0.5 * thre
        for t in range(module_snn.T):
            x_l_1_t = snn_in[t, ...]
            mem_t = v_t_1 - shaped(module_snn.time_based_bias[t], x_l_1_t) + x_l_1_t      # the only change from the released update
            spike = module_snn.act(mem_t - thre, module_snn.gama) * thre
            bias = spike - ann_out
            mem_t_updated = mem_t - bias
            spike_updated = module_snn.act(mem_t_updated - thre, module_snn.gama) * thre
            v_t_updated = mem_t_updated - spike_updated
            module_snn.time_based_bias[t] = curr_t_alpha * chmean(bias) + module_snn.time_based_bias[t]
            v_t_1 = v_t_updated

    def fd_step(ann, module_ann, snn, module_snn, T, train_data, curr_t_alpha=0.5, record=None):
        ann_out, snn_in = capture(ann, module_ann, snn, module_snn, T, train_data)
        thre = module_snn.thresh.data
        v = 0.5 * thre
        for t in range(module_snn.T):
            x = snn_in[t, ...]
            mem = v - shaped(module_snn.time_based_bias[t], x) + x                           # deployed IF with the current b_t
            spike = module_snn.act(mem - thre, module_snn.gama) * thre
            module_snn.time_based_bias[t] = curr_t_alpha * chmean(spike - ann_out) + module_snn.time_based_bias[t]
            mem = v - shaped(module_snn.time_based_bias[t], x) + x                           # advance with the updated b_t
            spike = module_snn.act(mem - thre, module_snn.gama) * thre
            v = mem - spike
            if record is not None:
                record.append(spike.detach().clone())

    return {'FA': fa_step, 'FD': fd_step}
