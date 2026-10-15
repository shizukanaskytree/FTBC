"""Describe the predeclared finite ordinary-control screen, no endpoint selection."""
import json
from pathlib import Path
E=Path(__file__).resolve().parent
a=json.loads((E/'main-acceptance.json').read_text());assert a['passed']
rows=['| 反向规则 | fit正确/256 | dev正确/512 | dev Top1 | dev CE | dev教师KL | 对native修复/损伤 | 有梯度接口 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for name in ['bn-only','native','relative-area','relative-peak']:
 m=a['states'][name]['metrics'];d=m['development'];f=m['fit'];pair=d['paired_vs_native']
 rows.append(f"| {name} | {f['correct']} | {d['correct']} | {d['top1']:.4f}% | {d['ce']:.6f} | {d['kl']:.6f} | {pair['repair']}/{pair['damage']} | {a['states'][name].get('nonzero_gradient_interfaces','-')} |")
decision=('预指定relative-area达到本次有限开发门槛，后继可在native旁增加此普通控制候选；尚非独立增益。' if a['C1'] else '预指定relative-area未达到本次有限开发门槛；停止扩大本梯度尺度检查。')
text='# E10普通KL控制结果｜2026-10-09\n\n'+ '\n'.join(rows)+'\n\n'+decision+'\n\n'
text+='三个臂的全部终点均报告，未以辅助relative-peak救主要判据；仅一个重复暴露开发集、一个固定学习率和末次epoch，不是种子复现或最优优化器比较。native的真实bias与fit/dev logits位级重现E9，只算复现控制，不算独立支持。\n\n'
text+='此实验只涉及普通KL任务微调的反向规则，不能解释无梯度FTBC旧降点；不证明新方法、硬函数精确梯度、稳定泛化或后续可扩大权限。保持父C1/保护评价及已关闭候选关闭。\n\n'
text+='执行身份与实际bias重构、严格原生重载、冻结teacher/非bias状态、逐图指标重算见 `main-acceptance.json`。成本见 `cost.json`；同卡并发成本为allocation时长，不能当正式推理延迟。独立语义审查状态见 `review/`。\n'
(E/'results.md').write_text(text)
(E/'analysis.json').write_text(json.dumps({'C1':a['C1'],'result_scope':'one fixed ordinary-control development screen','novel_method':False,'protected_evaluation':False,'stop_this_screen':not a['C1'],'states':a['states']},indent=2)+'\n')
(E/'CLAIMS_FROM_RESULTS.md').write_text('# E10结果到主张\n\nE10-C1='+('yes' if a['C1'] else 'no')+'，只针对冻结有限开发标准。'+decision+'\n\n无新颖性主张；反向尺度已有直接近邻。即使开发阳性，也只是更强普通控制候选，不能写期刊主结果或重开旧候选。\n\n官方独立结果审查待回。原始逐图结果、实际bias、复现、执行与成本见 `results.md` 和 `main-acceptance.json`。\n')
print(json.dumps({'C1':a['C1'],'novel_method':False}))
