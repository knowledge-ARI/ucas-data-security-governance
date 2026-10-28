"""从实际逐次记录生成图表、配对统计与完整性审计；不硬编码实验结果。"""
from __future__ import annotations
import csv
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import font_manager
from experiment import ROOT, SEEDS, Setting, load_data, corrupt, sha256, DEFAULT_DATA, save_csv

OUT = ROOT / 'results'
FIG = OUT / 'figures'
COLORS = ['#215d91', '#cc643d', '#389183', '#8064a2', '#a08034']
font_path = Path('C:/Windows/Fonts/msyh.ttc')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'figure.facecolor':'white', 'savefig.facecolor':'white'})


def main():
    global OUT, FIG
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.out.resolve();FIG=OUT/'figures'
    FIG.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'runs').glob('*.json'))]
    meta = json.loads((OUT/'metadata.json').read_text(encoding='utf-8'))
    assert meta['n_runs'] == 65 and not meta['smoke']
    assert len(records) == 65
    assert all(r['fingerprint']==meta['fingerprint'] for r in records)
    data = load_data(args.data_dir)
    assert data['source_hashes'] == meta['data_hashes']
    rows = [r['metrics'] for r in records]
    baseline = {r['seed']:r for r in rows if r['model']=='mlp' and r['attack']=='indiscriminate' and r['rate']==0}
    model_baselines = {(r['model'],r['seed']):r for r in rows if r['rate']==0}
    checks = []
    for row in rows:
        config = Setting(**{k:row[k] for k in ['attack','rate','seed','source','target','model','epochs']})
        Xp, yp, chosen, pool_size = corrupt(data['X'], data['y'], config)
        saved = np.load(OUT/'runs'/(row['key']+'.npz'))
        assert np.array_equal(chosen, saved['poisoned_local_indices'])
        assert np.array_equal(yp[chosen], saved['poisoned_labels'])
        assert np.array_equal(saved['ytest'],data['ytest'])
        assert len(saved['pred']) == 10000 and saved['confusion'].sum() == 10000
        assert row['n_poison'] == len(chosen) and row['pool_size']==pool_size
        assert np.count_nonzero(yp!=data['y']) == row['n_poison']
        if row['attack'] != 'indiscriminate':
            assert np.all(data['y'][chosen] == row['source'])
        source_mask = data['ytest']==row['source']
        target_error = np.sum(saved['pred'][source_mask]==row['target'])
        assert target_error == row['target_errors']
        assert np.isclose(target_error/source_mask.sum(),row['asr'])
        base = model_baselines[(row['model'],row['seed'])]
        bp = np.load(OUT/'runs'/(base['key']+'.npz'))['pred']
        row['accuracy_drop'] = base['accuracy']-row['accuracy']
        base_asr = float(np.mean(bp[source_mask]==row['target']))
        row['asr_increase'] = row['asr']-base_asr
        strict = source_mask & (bp==data['ytest'])
        row['induced_asr'] = float(np.mean(saved['pred'][strict]==row['target']))
        checks.append(row['key'])
    save_csv(OUT/'paired_runs.csv',rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'],row['attack'],row['source'],row['target'],row['rate'])].append(row)
    summaries=[]
    metrics=['accuracy','macro_f1','asr','source_recall','other_accuracy',
             'accuracy_drop','asr_increase','induced_asr']
    for key, group in groups.items():
        assert sorted(g['seed'] for g in group)==sorted(SEEDS)
        item=dict(model=key[0],attack=key[1],source=key[2],target=key[3],rate=key[4],
                  n_poison=group[0]['n_poison'],pool_size=group[0]['pool_size'],
                  global_rate=group[0]['global_rate'],repeats=len(group))
        for metric in metrics:
            values=[g[metric] for g in group]
            item[metric+'_mean']=float(np.mean(values))
            item[metric+'_sd']=float(np.std(values,ddof=1))
        summaries.append(item)
    summaries.sort(key=lambda x:(x['model'],x['attack'],x['source'],x['rate']))
    save_csv(OUT/'summary.csv',summaries)
    (OUT/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps(dict(passed=True,validated_runs=len(checks),
        data_hashes_match=True,train_validation_disjoint=True,test_count=10000,
        checks=['IDX headers and lengths','original data SHA256','training-only corruption',
                'exact corruption counts','source-only targeting','saved predictions and ASR denominators',
                'confusion matrix totals','5 paired seeds in each group'],
        source_train=int(np.sum(data['y']==3)),source_test=int(np.sum(data['ytest']==3))),indent=2),encoding='utf-8')

    def get(attack,rate,model='mlp',source=3):
        return next(s for s in summaries if s['attack']==attack and s['rate']==rate and s['model']==model and s['source']==source)

    def save(fig,name):
        fig.savefig(FIG/name,dpi=200,bbox_inches='tight')
        plt.close(fig)

    def curve(ax, items, metric, label, color, xkey='rate'):
        ax.errorbar([100*s[xkey] for s in items],[100*s[metric+'_mean'] for s in items],
                    yerr=[100*s[metric+'_sd'] for s in items], marker='o',capsize=4,
                    linewidth=1.8,color=color,label=label)
        ax.grid(alpha=.2)

    # 按原题重新绘制计数图：菱形为实际异常，圆点为实际正常。
    from matplotlib.patches import Circle
    fig,ax=plt.subplots(figsize=(8.5,2.65))
    ax.set(xlim=(0,9),ylim=(0,3));ax.set_aspect('equal');ax.axis('off')
    for cx,label in [(2.3,'预测为异常（正类）'),(6.7,'预测为正常（负类）')]:
        ax.add_patch(Circle((cx,1.6),1.05,fill=False,linewidth=1.5,color='#333333'))
        ax.text(cx,2.82,label,ha='center',fontsize=11)
    anomaly=[(1.65,1.7),(2.6,1.9),(2.5,1.35),(2.0,.9),(6.4,.9)]
    normal=[(2.3,2.25),(2.2,1.55),(6.5,2.25),(6.1,1.55),(7.0,1.5)]
    ax.scatter(*zip(*anomaly),marker='D',s=65,facecolors='white',edgecolors='#333333',linewidths=1.2,label='实际异常行为')
    ax.scatter(*zip(*normal),marker='o',s=65,c='#333333',label='实际正常行为')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.13),ncol=2,frameon=False,fontsize=10)
    fig.tight_layout();save(fig,'00_detection.png')

    fig,ax=plt.subplots(figsize=(9.1,3.0))
    ax.axis('off')
    steps=[('真实 MNIST','核验IDX与文件哈希'),('固定分层抽样','10000训练 / 2000验证'),
           ('仅污染训练集','随机改标 / 定向改标'),('独立重新训练','5个配对种子 / 固定60轮'),('保留官方测试集','10000样本一次评估')]
    for i,(title,subtitle) in enumerate(steps):
        x=.02+i*.198
        ax.text(x,.61,title,ha='left',va='center',fontsize=11,fontweight='bold',transform=ax.transAxes,color=COLORS[0])
        ax.text(x,.38,subtitle.replace(' / ','\n'),ha='left',va='center',fontsize=8.5,transform=ax.transAxes)
        if i<4: ax.annotate('',xy=(x+.18,.6),xytext=(x+.153,.6),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#738391'})
    ax.text(.02,.05,'训练数据污染 → 学习到错误监督 → 在真实测试数据上测量性能变化',transform=ax.transAxes,fontsize=11)
    save(fig,'01_workflow.png')

    fig,axes=plt.subplots(1,6,figsize=(10,2.0))
    for i,ax in enumerate(axes):
        idx=np.flatnonzero(data['ytest']==3)[i]
        ax.imshow(data['Xtest'][idx].reshape(14,14),cmap='gray',vmin=0,vmax=1)
        ax.axis('off');ax.set_title(f'真实标签3 / 索引{idx}',fontsize=9)
    fig.tight_layout();save(fig,'02_samples.png')

    un=[get('indiscriminate',r) for r in [0,.07,.1,.2,.4,.6]]
    tg=[get('indiscriminate',0)]+[get('targeted',r) for r in [.1,.3,.5,.7]]
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    curve(axes[0],un,'accuracy','整体准确率',COLORS[0]); curve(axes[0],un,'macro_f1','宏平均F1',COLORS[2])
    axes[0].set(xlabel='全部训练集投毒率（%）',ylabel='指标（%）',ylim=(60,100),title='任意投毒：性能退化')
    curve(axes[1],un,'accuracy_drop','配对准确率下降',COLORS[1])
    axes[1].set(xlabel='全部训练集投毒率（%）',ylabel='下降幅度（百分点）',title='与同种子干净模型比较')
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout(); save(fig,'03_indiscriminate.png')

    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    curve(axes[0],tg,'asr','3→8 攻击成功率',COLORS[1]); curve(axes[0],tg,'source_recall','数字3召回率',COLORS[0])
    curve(axes[1],tg,'accuracy','整体准确率',COLORS[0]); curve(axes[1],tg,'other_accuracy','非源类准确率',COLORS[2])
    for ax in axes:
        ax.set(xlabel='数字3类内投毒率（%）',ylabel='比例（%）'); ax.legend(fontsize=8)
    axes[0].set(title='目标投毒：攻击与源类损失',ylim=(0,102)); axes[1].set(title='整体指标掩盖局部损害',ylim=(80,100))
    fig.tight_layout(); save(fig,'04_targeted.png')

    selected=[baseline[17],next(r for r in rows if r['key']=='mlp_indiscriminate_3to8_0.6_s17_e60'),
              next(r for r in rows if r['key']=='mlp_targeted_3to8_0.7_s17_e60')]
    for i,row in enumerate(selected):
        matrix=np.load(OUT/'runs'/(row['key']+'.npz'))['confusion']
        fig,ax=plt.subplots(figsize=(6,5.0))
        image=ax.imshow(matrix/matrix.sum(axis=1,keepdims=True),cmap='Blues',vmin=0,vmax=1)
        for y in range(10):
            for x in range(10):
                ax.text(x,y,str(matrix[y,x]),ha='center',va='center',fontsize=7.5,
                        color='white' if matrix[y,x]/matrix[y].sum()>.55 else '#252525')
        ax.set(xticks=range(10),yticks=range(10),xlabel='预测标签',ylabel='真实标签',
               title=['干净基线','任意投毒 60%','目标投毒 70%（数字3类内）'][i]+' / seed=17')
        fig.colorbar(image,ax=ax,label='按真实类别归一化的比例',fraction=.045,pad=.03)
        fig.tight_layout(); save(fig,f'0{5+i}_confusion.png')

    # 同一类别的学习过程只使用独立验证集，无按测试ASR挑选轮数的过程。
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for setting,label,color in [('mlp_indiscriminate_3to8_0_s17_e60','干净基线',COLORS[0]),
                                ('mlp_targeted_3to8_0.5_s17_e60','目标投毒50%',COLORS[1])]:
        h=next(r['history'] for r in records if r['metrics']['key']==setting)
        axes[0].plot([v['epoch'] for v in h],[v['train_loss'] for v in h],label=label,color=color)
        axes[1].plot([v['epoch'] for v in h],[100*v['val_asr'] for v in h],label=label,color=color)
    axes[0].set(xlabel='训练轮数',ylabel='训练交叉熵',title='对各自训练标签计算损失')
    axes[1].set(xlabel='训练轮数',ylabel='验证集 3→8 成功率（%）',title='独立干净验证集 / seed=17')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); save(fig,'08_history.png')

    comparisons=[('MLP 3→8',get('targeted',.5)),('MLP 0→1',get('targeted',.5,source=0)),
                 ('逻辑回归 3→8',get('targeted',.5,model='logistic'))]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    x=np.arange(len(comparisons))
    for ax,metric,title in [(axes[0],'asr','50%源类投毒的目标误判率'),(axes[1],'accuracy','对应整体准确率')]:
        ax.bar(x,[100*s[metric+'_mean'] for _,s in comparisons],
               yerr=[100*s[metric+'_sd'] for _,s in comparisons],capsize=4,color=COLORS[:3],width=.55)
        ax.set(xticks=x,xticklabels=[label for label,_ in comparisons],ylabel='比例（%）',title=title,ylim=(0,100))
        ax.grid(axis='y',alpha=.2)
    fig.tight_layout(); save(fig,'09_factors.png')

    # 展示预先固定位置的前6个源类测试样本，包含失败样例，不挑选成功个例。
    row=selected[2]; pred=np.load(OUT/'runs'/(row['key']+'.npz'))['pred']
    bp=np.load(OUT/'runs'/(selected[0]['key']+'.npz'))['pred']
    fig,axes=plt.subplots(1,6,figsize=(10,2.2))
    for ax,idx in zip(axes,np.flatnonzero(data['ytest']==3)[:6]):
        ax.imshow(data['Xtest'][idx].reshape(14,14),cmap='gray',vmin=0,vmax=1); ax.axis('off')
        ax.set_title(f'测试索引 {idx}\n真实3 / 基线{bp[idx]}\n投毒后 {pred[idx]}',fontsize=9,
                     color=COLORS[1] if pred[idx]==8 else COLORS[0])
    fig.tight_layout(); save(fig,'11_predictions.png')
    print('Verified 65 runs; wrote summary, paired records, audit and 11 figures.')
    for s in summaries:
        print(f'{s["model"]} {s["attack"]} {s["source"]}->{s["target"]} {s["rate"]:.0%} n={s["n_poison"]} '
              f'ACC={s["accuracy_mean"]:.4%} +/- {s["accuracy_sd"]:.4%} '
              f'ASR={s["asr_mean"]:.4%} +/- {s["asr_sd"]:.4%}')


if __name__=='__main__':
    main()
