import json
from pathlib import Path
E=Path(__file__).resolve().parent
d=json.loads((E/'main-acceptance.json').read_text());assert d['passed'];s=d['states']
modes=['root','permuted','constant','full-kl','full-ce']
selected={m:min([n for n in s if n.startswith(m+'-lr')],key=lambda n:(-s[n]['metrics']['development']['correct'],s[n]['metrics']['development']['ce'],s[n]['detail']['lr'])) for m in modes}
controls=['bn-only','finite-solve','kl-e4']+[selected[m] for m in modes if m!='root']
winner=max(controls,key=lambda n:s[n]['metrics']['development']['correct']);root=selected['root'];r=s[root]['metrics']['development']
c1=r['correct']>=s[winner]['metrics']['development']['correct']+3 and r['ce']<=s['bn-only']['metrics']['development']['ce']
c2={str(lr):all(s[f'root-lr{lr}']['metrics']['development']['correct']>s[f'{m}-lr{lr}']['metrics']['development']['correct'] and s[f'root-lr{lr}']['metrics']['development']['kl']<=s[f'{m}-lr{lr}']['metrics']['development']['kl'] for m in ['permuted','constant']) for lr in [.001,.01]}
disturbance={}
for lr in [.001,.01]:
 a=s[f'root-lr{lr}']['metrics']['development']
 disturbance[str(lr)]={m:all(a[k]<s[f'{m}-lr{lr}']['metrics']['development'][k] for k in ['prediction_changes_vs_bn','kl_BN_to_candidate']) for m in ['permuted','constant']}
result={'selected':selected,'strongest_control_by_correct':winner,'E9_C1_development_investment':c1,'E9_C2_both_rates':all(c2.values()),'C2_per_rate':c2,'smaller_output_perturbation':disturbance,'paired_vs_strongest':d['pairwise_development'][root][winner],'stop_expansion':not(c1 and all(c2.values())),'independent_validation':False,'novel_method_established':False,'states':s}
(E/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# E9 开发筛选结果','', '全部为已暴露fit256/dev512，单次开发筛选，非独立确认。普通链式法则/低维训练不声明创新。','', '|终点|fit正确/256|dev正确/512|devTop1|devCE|devKL|均值残差M|','|---|---:|---:|---:|---:|---:|---:|']
for n,z in s.items():
 f=z['metrics']['fit'];v=z['metrics']['development'];lines.append(f'|{n}|{f["correct"]}|{v["correct"]}|{v["top1"]:.4f}|{v["ce"]:.6f}|{v["kl"]:.6f}|{z["M"]:.8g}|')
lines+=['','full-ce为标签监督；其余新训练族为冻结Q8教师KL，不使用标签训练，均使用开发标签选择。',f'事前每族选择：{selected}。',f'E9-C1开发投入判据：{c1}；E9-C2两学习率方向区分：{all(c2.values())}（{c2}）。',f'最强控制正确数：{winner}；本轮不据选优读数宣称统计显著、泛化稳定或期刊主方法。','', '所有学习率终点保留；仅末轮4epoch，未事后选轮/扩网格。',f'与最强控制的配对：{result["paired_vs_strongest"]}。',f'R是否两个输出扰动量均较小：{disturbance}（即便不是也不代表逐层响应匹配）。']
(E/'results.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='states'}))
