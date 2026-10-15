"""Derived E12 report from the accepted main run; no state selection, no new statistics beyond acceptance."""
import csv, hashlib, json
from pathlib import Path
E = Path(__file__).resolve().parent
a = json.loads((E / 'main-acceptance.json').read_text()); assert a['passed']
S = ['p0-zero', 'C0', 'bn-only', 'kl-e4', 'Cabs', 'Transfer', 'Crel']
LABEL = {'p0-zero': 'P0（plain零偏置）', 'C0': 'C0（P0求解a）', 'bn-only': 'BN（P1零偏置）', 'kl-e4': 'KL-e4（历史选优重放）',
         'Cabs': 'Cabs（P1求解a）', 'Transfer': 'Transfer（P1+δ0）', 'Crel': 'Crel（P1求解m）'}
m = a['metrics']; pf = a['pairs_fit_in_sample_vs_bn']; pdv = a['pairs_development']
d = a['decision_B1']; g = a['descriptors']['global_theta_normalized']; r1 = a['rule_R1']; ft = a['E_fit_Transfer_descriptive']


def f(x, nd=6):
    return '未定义' if x is None else f'{x:.{nd}f}'


rows = ['| 状态 | fit正确/256（样本内） | fit CE | dev正确/512 | dev Top1 (%) | dev CE | dev相对BN修复/损伤 |', '|---|---:|---:|---:|---:|---:|---:|']
for s in S:
    fm, dm = m[s]['fit'], m[s]['development']
    if s == 'bn-only':
        rd = '—'
    else:
        x = pdv[f'{s}-vs-bn-only']; rd = f"{x['x_only_correct']}/{x['y_only_correct']}"
    rows.append(f"| {LABEL[s]} | {fm['correct']} | {fm['ce']:.6f} | {dm['correct']} | {dm['top1']:.4f} | {dm['ce']:.6f} | {rd} |")
pair = pdv['Crel-vs-Transfer']; geo = a['bias_geometry']['Crel-vs-Transfer(delta_r vs delta_0)']['global_theta_normalized']
b2 = a['B2_E_dev']; arms = a['solved_arms']
lines = []
for s in ['C0', 'Cabs', 'Crel']:
    x = arms[s]; p = x['residual_partition']
    lines.append(f"| {s} | {x['target']} | {x['M_own_target_channel_equal']:.3e} | {100 * x['within_quantum_fraction']:.3f}% | {x['selected_zero']} | {x['selected_boundary']} | "
                 f"{x['support']['below_range']}/{x['support']['above_range']} | {p['outside_bound_support']['channels']} | {p['bracketed_adjacent_count_jump']['channels']} | {p['other_unresolved']['channels']} |")
bs = a['bias_summary']
decision_text = ('**探索性停止规则：通过。** 这只是已暴露开发集上的探索信号；是否冻结一次独立确认由作者决定，本轮不开启，也不构成方法或新颖性主张。'
                 if d['exploratory_rule_passed'] else '**探索性停止规则：未通过。** 按协议停止，关闭本设置（R34 Q8→T2、E0 BN、fit256、E7求解器）下参照目标族（E2+E12）；不追加学习率、系数、剂量、种子、模型或样本。这只关闭本次具体测试，不证明所有相对参照方案都不可能有效。')
r1_text = ('R1不适用（ρ_g<0.5），本轮不作方向性预测。' if not r1['applies'] else
           f"R1适用（ρ_g≥0.5，预测Crel dev CE高于Cabs且正确数不多于Cabs）：预测{'成立' if r1['held'] else '不成立'}。单次结果只是一个数据点，不证明双重校正机制。")
r2 = a['descriptors']['rules']['R2_low_information_applies']
text = f'''# E12 结果：源校准与BN适应的普通组合对照（2026-10-09）

状态：GPU sanity/main 与 CPU 验收通过；官方结果审查见 `review/`。本轮是普通组合控制诊断，不是方法候选或期刊主结果；dev512 已在 E4–E11 多次暴露，fit256 对 P1 系各臂是样本内。

## 同卡同状态比较（A100 PCIe，评分 batch4）

''' + '\n'.join(rows) + f'''

{decision_text}

- 控制集 K={{BN, KL-e4, Cabs, Transfer}}：最强正确数 {d['best_control_correct']}（{', '.join(d['best_control_states'])}），最低 CE {d['lowest_control_ce']:.6f}（{d['lowest_control_state']}）；Crel 为 {d['candidate_correct']}/512、CE {d['candidate_ce']:.6f}；通过需正确数 ≥{d['best_control_correct'] + 3} 且 CE ≤ 最低值。
- **首要配对 Crel−Transfer（dev）**：仅 Crel 正确 {pair['x_only_correct']}、仅 Transfer 正确 {pair['y_only_correct']}，净 {pair['net']:+d}；预测类别不同 {pair['prediction_disagreements']} 张；discordant 精确双侧符号检验 p={f(pair['exact_two_sided_sign_p_descriptive'], 4)}（仅描述，不作判据）。
- 等价读数：θ归一化全网 ‖δr−δ0‖/‖δ0‖={f(geo['relative_distance'], 4)}，cos(δr,δ0)={f(geo['cos'], 4)}；逐层值见 `main-acceptance.json`。

## 求解前描述量与事前规则

全网 θ 归一化：ρ_g={f(g['rho'], 4)}，cos_g={f(g['cos'], 4)}，κ_g={f(g['kappa'], 4)}；m 落在 [0,θ] 之外 {a['descriptors']['m_outside_0_theta']}/7616 通道；ρ_l≥0.5 的层 {a['descriptors']['layers_rho_ge_0.5']}/33。m 与描述量在 Crel 求解前落盘，哈希见 `target-manifest.json`。

{r1_text} R2（κ_g<0.1 低信息量声明）{'适用：本测试区分 Crel 与 Cabs 的信息量低，差异照报不解释。' if r2 else '不适用。'}

Δ=p1−p0 与 e=a−p0 的投影只是描述：E0 BN 在同域 T2 上用标签训练，本身就在修复转换误差，不能先验称 Δ 为“有用适应”、e 为“待移除转换误差”。

## 机制读数（B2，仅 dev 测量）

E_dev(s)=mean_c|(μ^dev_s−μ^dev_C0)−(p1^dev−p0^dev)|/θ（7616 通道等权 / 33 层等权）：
- Crel：{b2['Crel']['channel_equal']:.6e} / {b2['Crel']['layer_equal']:.6e}
- Transfer：{b2['Transfer']['channel_equal']:.6e} / {b2['Transfer']['layer_equal']:.6e}
- Cabs：{b2['Cabs']['channel_equal']:.6e} / {b2['Cabs']['layer_equal']:.6e}

它衡量各臂能否在未参与拟合的 dev 图上重现 C0 在 P0 上造成的偏置诱导均值变化，不是分类因果。dev 均值未进入任何拟合、目标或选择。fit 侧恒等式最大偏差 {a['fit_identity_solver_execution_check_max_abs']:.2e}，只是求解执行检查（由残差定义直接成立）。

fit 侧 Transfer（未拟合，实测量）：E_fit(Transfer)={ft['channel_equal']:.6e} / {ft['layer_equal']:.6e}，即 D_1(δ0)=D_0(δ0) 这一等价条件在 fit 上的偏离程度。

## 求解充分性与可达性

| 臂 | 目标 | M（自身目标，通道等权） | 一计数量子内 | 选零 | 选边界 | 低于/高于可达 | 超出有界支持 | 包围但浮点相邻跳变 | 其余 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
''' + '\n'.join(lines) + f'''

偏置幅度（θ 归一化 RMS / 最大）：C0 {bs['C0']['rms_over_theta']:.4f}/{bs['C0']['max_abs_over_theta']:.4f}，Cabs {bs['Cabs']['rms_over_theta']:.4f}/{bs['Cabs']['max_abs_over_theta']:.4f}，Crel {bs['Crel']['rms_over_theta']:.4f}/{bs['Crel']['max_abs_over_theta']:.4f}，KL-e4 {bs['kl-e4']['rms_over_theta']:.4f}/{bs['kl-e4']['max_abs_over_theta']:.4f}。幅度差异本身会影响结果（E7/E9），不能把臂间差异单独归因于“保持适应差”。

## 有效性

教师目标、BN 与 KL 的 fit 计数及 logits、Cabs 的偏置/计数/logits 均与 E7 逐位相同；Transfer 偏置与 C0 逐位相同；三个求解臂的完整网络 fit 重放计数等于求解记录；全部状态真实偏置保存与 strict 重载通过；教师和非偏置状态不变。资源：{a['resources']['gpu']}（{a['resources']['uuid']}），Slurm {a['resources']['slurm_job_id']}，进程 {a['resources']['seconds_total']:.1f} 秒，峰值 reserved {a['resources']['max_memory_reserved'] / 2 ** 30:.2f} GiB；分配成本见 `cost.json`，不是正式推理延迟。

## 解释边界

- 本轮无论通过与否都不构成期刊方法创新（概念审查主观新颖性 2/10）；m=p1+(a−p0) 是把源网络测得的均值转换误差作加性校正搬到适应后网络。
- “取 δref=δ0 时相对目标等于 Transfer 响应”只在源求解残差为零时成立；“均值相等则逐位置计数相等”只在同一固定局部输入下成立，本轮不作这两条推论。
- 审查把本目标类比为 PTQ 偏置校正（如 DFQ）只作背景，写入论文前须查原文。
- 单一开发抽样、单一设置；描述性 p 值和停止门槛都不是独立确认。
'''
(E / 'results.md').write_text(text)
flat = []
for s, x in arms.items():
    for n, v in x['layers'].items():
        flat.append({'arm': s, 'layer': n, 'M_own_target': v['M'], 'within_quantum': v['within_quantum'], 'selected_zero': v['selected_zero'],
                     'selected_boundary': v['selected_boundary'], 'outside_bound_support': v['residual_partition']['outside_bound_support']['channels']})
with (E / 'layer-results.csv').open('w') as fh:
    w = csv.DictWriter(fh, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
(E / 'analysis-provenance.json').write_text(json.dumps({'derived_report_generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                       'frozen_analyzer': 'source/e12_accept.py', 'acceptance_sha256': hashlib.sha256((E / 'main-acceptance.json').read_bytes()).hexdigest()}, indent=2) + '\n')
print('E12 report generated')
