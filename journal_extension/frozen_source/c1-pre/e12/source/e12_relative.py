"""E12 pure functions: relative target, pre-solve descriptors, B1 rule, B2 dev metric, bias geometry.

CPU-only, FP64. Undefined ratios/cosines are None, never 0. No selection logic beyond the
pre-registered rule in EXPERIMENT_PLAN.md sections 5-7.
"""
import math
import torch

CONTROLS = ('bn-only', 'kl-e4', 'Cabs', 'Transfer')
CANDIDATE = 'Crel'
MARGIN = 3
RHO_DOUBLE_CORRECTION = 0.5
KAPPA_LOW_INFORMATION = 0.1


def means_from_counts(measured):
    """Same expression as E7 solver mean(): count.double()*theta/n."""
    return {n: v['count'].double() * v['theta'] / v['observations_per_channel'] for n, v in measured.items()}


def relative_target(a, p0, p1):
    assert list(a) == list(p0) == list(p1)
    delta = {n: p1[n] - p0[n] for n in a}
    e = {n: a[n] - p0[n] for n in a}
    m = {n: a[n] + delta[n] for n in a}
    for n in a:
        for x in (a[n], p0[n], p1[n], delta[n], e[n], m[n]):
            assert x.dtype == torch.float64 and torch.isfinite(x).all()
    return delta, e, m


def _dot(x, y):
    return float((x * y).sum())


def _ratio(num, den):
    return None if den == 0 else num / den


def _cos(x, y):
    nx, ny = math.sqrt(_dot(x, x)), math.sqrt(_dot(y, y))
    return None if nx == 0 or ny == 0 else _dot(x, y) / (nx * ny)


def descriptors(a, p0, p1, thetas):
    """Pre-solve descriptors of Delta=p1-p0 against e=a-p0 (section 5)."""
    delta, e, m = relative_target(a, p0, p1)
    layers, gd, ge = {}, [], []
    outside_total = 0
    for n in a:
        th = float(thetas[n])
        d, s = delta[n], e[n]
        outside = int(((m[n] < 0) | (m[n] > th)).sum())
        outside_total += outside
        layers[n] = {'channels': len(s), 'theta': th,
                     'rho': _ratio(_dot(d, s), _dot(s, s)), 'cos': _cos(d, s),
                     'kappa': _ratio(math.sqrt(_dot(d, d)), math.sqrt(_dot(s, s))),
                     'mean_abs_delta_over_theta': float(d.abs().mean() / th),
                     'mean_abs_e_over_theta': float(s.abs().mean() / th),
                     'm_outside_0_theta': outside, 'm_outside_0_theta_fraction': outside / len(s)}
        gd.append(d / th); ge.append(s / th)
    gd, ge = torch.cat(gd), torch.cat(ge)
    channels = len(gd)
    rho_g = _ratio(_dot(gd, ge), _dot(ge, ge))
    kappa_g = _ratio(math.sqrt(_dot(gd, gd)), math.sqrt(_dot(ge, ge)))
    out = {'layers': layers, 'channels': channels,
           'global_theta_normalized': {'rho': rho_g, 'cos': _cos(gd, ge), 'kappa': kappa_g},
           'm_outside_0_theta': outside_total, 'm_outside_0_theta_fraction': outside_total / channels,
           'layers_rho_ge_0.5': sum(1 for v in layers.values() if v['rho'] is not None and v['rho'] >= RHO_DOUBLE_CORRECTION),
           'rules': {'R1_double_correction_applies': rho_g is not None and rho_g >= RHO_DOUBLE_CORRECTION,
                     'R2_low_information_applies': kappa_g is not None and kappa_g < KAPPA_LOW_INFORMATION}}
    return out, m


def decide(dev):
    """B1 exploratory stop rule. dev[state] = {'correct': int, 'ce': float}."""
    best = max(dev[k]['correct'] for k in CONTROLS)
    lowest = min(dev[k]['ce'] for k in CONTROLS)
    passed = dev[CANDIDATE]['correct'] >= best + MARGIN and dev[CANDIDATE]['ce'] <= lowest
    return {'controls': list(CONTROLS), 'best_control_correct': best,
            'best_control_states': [k for k in CONTROLS if dev[k]['correct'] == best],
            'lowest_control_ce': lowest, 'lowest_control_state': min(CONTROLS, key=lambda k: dev[k]['ce']),
            'candidate_correct': dev[CANDIDATE]['correct'], 'candidate_ce': dev[CANDIDATE]['ce'],
            'margin_required': MARGIN, 'exploratory_rule_passed': passed,
            'meaning': ('exploratory signal on exposed development only; separate frozen confirmation is the author\'s decision'
                        if passed else 'stop: close the reference-target family (E2+E12) for this setting and scale')}


def rule_r1(desc, dev):
    if not desc['rules']['R1_double_correction_applies']:
        return {'applies': False}
    held = dev['Crel']['ce'] > dev['Cabs']['ce'] and dev['Crel']['correct'] <= dev['Cabs']['correct']
    return {'applies': True, 'prediction': 'Crel dev CE > Cabs and Crel correct <= Cabs', 'held': held}


def binomial_two_sided(b, c):
    """Exact two-sided sign test on discordant pairs; descriptive only. Undefined (None) without discordant pairs."""
    n, k = b + c, min(b, c)
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(labels, pred_x, pred_y):
    x, y = pred_x == labels, pred_y == labels
    b, c = int((x & ~y).sum()), int((y & ~x).sum())
    return {'x_only_correct': b, 'y_only_correct': c, 'net': b - c,
            'prediction_disagreements': int((pred_x != pred_y).sum()),
            'exact_two_sided_sign_p_descriptive': binomial_two_sided(b, c)}


def edev(mu, thetas, arms=('Crel', 'Transfer', 'Cabs')):
    """B2: mean_c |(mu_s - mu_C0) - (p1_dev - p0_dev)| / theta, on dev means of the same images/batch."""
    out = {}
    for s in arms:
        chans, per_layer = [], {}
        for n in mu['C0']:
            z = ((mu[s][n] - mu['C0'][n]) - (mu['bn-only'][n] - mu['p0-zero'][n])).abs() / float(thetas[n])
            chans.append(z); per_layer[n] = float(z.mean())
        z = torch.cat(chans)
        out[s] = {'channel_equal': float(z.mean()), 'layer_equal': sum(per_layer.values()) / len(per_layer),
                  'channels': len(z), 'per_layer': per_layer}
    return out


def teacher_residual(mu, target, thetas):
    chans, per_layer = [], {}
    for n in target:
        z = (mu[n] - target[n]).abs() / float(thetas[n]); chans.append(z); per_layer[n] = float(z.mean())
    z = torch.cat(chans)
    return {'channel_equal': float(z.mean()), 'layer_equal': sum(per_layer.values()) / len(per_layer), 'per_layer': per_layer}


def geometry(x, y, thetas):
    """Relative distance ||x-y||/||y|| and cos(x,y) per layer on the time-shared bias (time slot 0)."""
    layers, gx, gy = {}, [], []
    for n in y:
        a, b = x[n][0].flatten().double(), y[n][0].flatten().double()
        nb = math.sqrt(_dot(b, b))
        layers[n] = {'relative_distance': None if nb == 0 else math.sqrt(_dot(a - b, a - b)) / nb, 'cos': _cos(a, b)}
        th = float(thetas[n]); gx.append(a / th); gy.append(b / th)
    gx, gy = torch.cat(gx), torch.cat(gy)
    ny = math.sqrt(_dot(gy, gy))
    return {'layers': layers, 'global_theta_normalized': {
        'relative_distance': None if ny == 0 else math.sqrt(_dot(gx - gy, gx - gy)) / ny, 'cos': _cos(gx, gy)}}


def bias_summary(b, thetas):
    raw, norm = [], []
    for n, v in b.items():
        z = v[0].flatten().double(); raw.append(z); norm.append(z / float(thetas[n]))
    raw, norm = torch.cat(raw), torch.cat(norm)
    return {'rms': float(raw.square().mean().sqrt()), 'rms_over_theta': float(norm.square().mean().sqrt()),
            'mean_abs_over_theta': float(norm.abs().mean()), 'max_abs_over_theta': float(norm.abs().max()),
            'zero_channels': int((raw == 0).sum()), 'channels': len(raw)}
