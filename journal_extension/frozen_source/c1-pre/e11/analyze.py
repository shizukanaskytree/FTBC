import json
from pathlib import Path
E=Path(__file__).resolve().parent
a=json.loads((E/'main-acceptance.json').read_text());assert a['passed']
lines=['# E11｜固定状态的batch形状回放','', '状态：完整执行与CPU原始张量验收通过；官方结果复审待完成。所有数字为旧拟合图上的操作性诊断，不是方法收益、开发选择或独立评价。','',
'|冻结状态|batch4正确/256|batch8正确/256|预测类别改变|最大logit差|首个计数变化接口|变化接口数/33|', '|---|---:|---:|---:|---:|---|---:|']
layer_tables=[]
for state,r in a['states'].items():
 lines.append(f"|{state}|{r['correct']['b4']}|{r['correct']['b8']}|{r['prediction_disagreements']}|{r['max_logit_diff']:.8g}|{r['first_count_difference']}|{r['layers_with_count_differences']}|")
 c=json.loads((E/f'config/main-{state}.json').read_text());d=json.loads((Path(c['output'])/'layer-diagnostics.json').read_text())
 layer_tables.extend([f'## {state}：固定前8图全部接口','', '|接口|输入最大差|计数元素改变|受影响图像|输入相同而计数改变|原生重构/共同输入一致|','|---|---:|---:|---:|---:|---|'])
 for name,x in d.items():
  valid=all(x[k] for k in ['local_b4_reconstruct','local_b8_reconstruct','common_input_b4_b8_equal','repeated_mean_bitwise'])
  layer_tables.append(f"|{name}|{x['input_max_abs']:.8g}|{x['count_changed_elements']}|{x['count_affected_images']}|{x['count_changed_with_equal_input_elements']}|{valid}|")
 layer_tables.append('')
lines+=['','## 支持范围','',f"E11-C1={'yes' if a['E11-C1'] else 'no'}：同一设备/软件、同一冻结模型与FP32输入，仅改变4/8分批，是否出现预测差异。所有batch4重复及历史锚点位级一致。三状态全部列出，零差异状态也保留。",'',
'C2是固定前8图的定位读数：所有接口记录真实量化前输入或神经元时间均值输入、计数，并执行原生局部重放。统计在layer-diagnostics.md，原始张量在各worker输出。共同输入局部一致时，层间计数变化与入射数值变化相容；不能唯一定位卷积/归约算子，也不能推断256图的全部首次变化位置。','',
'不能据此说batch4或8更正确、所有旧结果无效、FTBC损伤全部来自数值误差，或已获得新算法。无新训练，不消费开发集/确认集/官方评价；不存在新种子独立性。','',
'## 验证与成本','', '同A100 PCIe三worker；实际bias保存/strict重载，原教师与模型完整state digest不变，原始逐图CE、预测、33层计数差异重算。冻结/稿件Git另见integrity-check.json，Slurm总占卡成本见cost.json；成本不是正式推理延迟。','',
'## 下一步边界','', '结束此有限4/8诊断，不追加batch/模型/卡型网格。后续若设计数值一致性措施，需把它定位为实现控制，先证明是否影响之前的科学判断；单纯固定batch不会构成期刊新方法。没有独立新颖性与匹配分类优势之前，不扩大父C1或写论文主结果。']
(E/'results.md').write_text('\n'.join(lines)+'\n');(E/'layer-diagnostics.md').write_text('\n'.join(layer_tables)+'\n')
(E/'analysis.json').write_text(json.dumps({'scope':'fixed-input/state fit-only numerical diagnostic','claim_supported_local':a['E11-C1'],'review_status':'pending','states':a['states'],'new_method':False},indent=2)+'\n')
print(json.dumps(a['states']))
