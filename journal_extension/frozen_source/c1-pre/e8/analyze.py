"""Derived reporting only; no fitting/selection and no GPU."""
import json
from pathlib import Path
import torch
torch.set_num_threads(4)
E=Path(__file__).resolve().parent
a=json.loads((E/'main-acceptance.json').read_text());assert a['passed']
pred={}
for mode in ['reference','global','layer']:
 c=json.loads((E/'config'/f'main-{mode}.json').read_text())
 for name,p in torch.load(Path(c['output'])/'predictions.pt',map_location='cpu',weights_only=False).items():
  if name in pred:
   for role in ['fit','development']:assert torch.equal(pred[name][role]['logits'],p[role]['logits'])
  pred[name]=p
paired={}
for name in ['root-global-Gnorm','root-layer-Gnorm']:
 paired[name]={}
 for role in ['fit','development']:
  x=pred[name][role];y=pred['G-e8'][role];assert torch.equal(x['ids'],y['ids']) and torch.equal(x['labels'],y['labels'])
  correct=x['predictions']==x['labels'];old=y['predictions']==y['labels']
  paired[name][role]={'repair_vs_G':int((correct&~old).sum()),'damage_vs_G':int((~correct&old).sum()),'prediction_flips_vs_G':int((x['predictions']!=y['predictions']).sum())}
order=['bn-only','G-e8','finite-solve','kl-e4','root-global-Gnorm','root-layer-Gnorm']
lines=['# E8固定幅度干预：实际结果','',
 '状态：GPU产物已通过CPU独立数值验收；官方结果到主张审查状态另见review/。单一已暴露开发设置，不是新方法或确认结果。','',
 '| 状态 | fit正确/256 | dev正确/512 | fit Top1 | dev Top1 | dev CE | fit M |',
 '|---|---:|---:|---:|---:|---:|---:|']
for name in order:
 s=a['states'][name];f=s['metrics']['fit'];d=s['metrics']['development']
 lines.append(f"| {name} | {f['correct']} | {d['correct']} | {f['top1']:.4f}% | {d['top1']:.4f}% | {d['ce']:.6f} | {s['M']:.8f} |")
lines+=['','## 固定判读','',f"E8-C1描述档：{a['E8_C1_descriptive']}。主要臂是root-layer-Gnorm；事前要求fit/dev都恢复至少一半G→BN缺口（21/51张）。global仅为固定辅助，不能择优替代主要臂。",'']
for name in paired:
 for role,d in paired[name].items():
  ratio=a['recovered_G_to_BN_gap'][name][role]
  lines.append(f"- {name}/{role}：恢复比例{ratio:.6f}；相对G修复{d['repair_vs_G']}、损坏{d['damage_vs_G']}，预测类别改变{d['prediction_flips_vs_G']}。")
lines+=['','## 范围','',
 '缩放由保存偏置/θ计算，无标签、训练、扫描或开发选优。global匹配总L2、layer匹配每层L2；两者仍改变方向、实际计数和均值，不能解释成固定目标残差或固定总脉冲量。目标方向来自E7另一套完整上游，参数空间角度不是某层修对了的保证。',
 '若layer修复说明这些终点在固定逐层范数下的整体方向干预可以改变损失；若未修复也不证明范数是唯一原因。没有定位单层损伤、证明继续G迭代会到R，或验证泛化。正确数/CE与M分别报告，不把量化目标变好当作任务收益。',
 'CPU验收核对所有实际bias重新构造、原数据角色、历史锚点logits/计数、非bias冻结、strict部署重载、三个进程BN一致。冻结文件未改；GPU分配成本见cost.json（生成后有效），不是正式延迟。']
(E/'results.md').write_text('\n'.join(lines)+'\n')
(E/'analysis.json').write_text(json.dumps({'acceptance':a,'paired_vs_G':paired,'posthoc_selected_hypothesis_from_E7':True,'new_data_selection':False,'candidate_method':False},indent=2)+'\n')
assert not torch.cuda.is_initialized()
print(json.dumps({'E8_C1_descriptive':a['E8_C1_descriptive'],'gap_recovery':a['recovered_G_to_BN_gap'],'paired':paired}))
