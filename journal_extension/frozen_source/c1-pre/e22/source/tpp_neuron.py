"""TPP's probabilistic integrate-and-fire neuron (EXPERIMENT_PLAN.md section 4), for E22 only.

`tpp_prob_forward` is the `mode == 'prob'` branch of `IF.forward` in TPP's Models/layer.py, downloaded on 2026-10-10 into
../third_party/tpp-drive-20261010/Models/layer.py (sha256 265642cd973fae0ad3c4fc5366612ed74f9f0c7774b5d4886cc7b5c6f797111f, equal to the
2026-10-08 cache ../../../evidence/serial-tpp-Models-layer.py). Copied line for line; the only change is that the running potential is a local
variable instead of the module attribute `self.mem_pot`, which does not change the computation and keeps the module's attributes untouched.
The third-party file itself is never imported or executed.

`tpp_mode(net)` swaps this forward into every IF module of `net` (the project's serial IF class, whose normal forward has the same structure as
TPP's normal mode plus FTBC's per-step biases) and restores the class forward on exit; TPP ignores the per-step biases, as in its code."""
import contextlib, types
import torch

TPP_LAYER_SHA256 = '265642cd973fae0ad3c4fc5366612ed74f9f0c7774b5d4886cc7b5c6f797111f'


def tpp_prob_forward(self, x):
    if self.T > 0:
        thre = self.thresh.data
        x = self.expand(x)
        # mem = 0.5 * thre
        z = x.sum(0)
        mem_pot = z

        spike_pot = []
        for t in range(self.T):
            mem_temp = torch.clamp(mem_pot / (self.T - t) / thre, 0, 1)
            spike = torch.bernoulli(mem_temp) * thre
            mem_pot = mem_pot - spike
            spike_pot.append(spike)

        x = torch.stack(spike_pot, dim=0)
        x = self.merge(x)
        return x
    raise AssertionError('TPP prob mode is used only for the spiking network (T > 0)')


@contextlib.contextmanager
def tpp_mode(net, spike_class):
    """Swap tpp_prob_forward into every module of class `spike_class` in `net`; restore the class forward afterwards, even on error."""
    mods = [m for m in net.modules() if isinstance(m, spike_class)]
    assert mods and all('forward' not in vars(m) for m in mods), 'a forward was already swapped'
    for m in mods:
        m.forward = types.MethodType(tpp_prob_forward, m)
    try:
        yield len(mods)
    finally:
        for m in mods:
            del m.forward
        assert all('forward' not in vars(m) for m in mods)


def seed_for(seed, role_index):
    """Section 4: one seed per TPP arm and evaluation role."""
    return 1000 * seed + role_index


def set_seed(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


@torch.no_grad()
def score_tpp(r, arm, T, seed, spike_class):
    """Score one TPP arm with a runner `r` of either E22 child (the E17 chain for R34, the E18 Run for VGG-16): the deployed T-step network with
    zero per-step biases (ignored by TPP), the TPP forward swapped in, seed 1000*seed + role index before each evaluation role, one Bernoulli
    sample per image. Reproducibility check: the same seed gives bitwise the same logits on the four strict-reload images twice. The non-bias
    state of the network must be unchanged afterwards. TPP arms have no biases to save, so there is no strict reload (section 4). The record's
    copied_from_layer_sha256 is the declared source of the copy; the acceptance compares it with the frozen layer.py read as bytes."""
    N = r.net(T); r.set_biases(T, r.zero_biases(T))
    with tpp_mode(N['snn'], spike_class) as swapped:
        p = {}
        for i, role in enumerate(r.c['score_roles']):
            set_seed(seed_for(seed, i)); p[role] = r.e0.evaluate(N['snn'], r.rows[role], N['parent'], True)
        set_seed(seed_for(seed, 9)); a = r.e0.logits(N['snn'], r.test_images, N['parent'], True).cpu()
        set_seed(seed_for(seed, 9)); b = r.e0.logits(N['snn'], r.test_images, N['parent'], True).cpu()
    assert torch.equal(a, b), 'TPP is not reproducible under a fixed seed: ' + arm
    assert r.e0.state_digest(N['snn']) == N['immutable_digest'], 'non-bias SNN state changed: ' + arm
    r.predictions[arm] = p
    r.records[arm] = {'arm': arm, 'T': T, 'metrics': {k: v['metrics'] for k, v in p.items()}, 'tpp': {'mode': 'prob', 'seed': seed, 'role_seeds': {role: seed_for(seed, i) for i, role in enumerate(r.c['score_roles'])},
                      'swapped_modules': swapped, 'copied_from_layer_sha256': TPP_LAYER_SHA256, 'reproducible_on_strict_reload_images': True}}
    r.e0.atomic_json(r.out / 'metrics.json', r.records); r.e0.atomic_torch(r.out / 'predictions.pt', r.predictions)
    r.progress('scored', arm=arm, T=T, **{role: p[role]['metrics'] for role in p})
