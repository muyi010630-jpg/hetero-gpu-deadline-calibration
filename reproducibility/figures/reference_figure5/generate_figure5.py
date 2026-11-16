"""Render Figure 5 from the archived C2 risk ablation, without new experiments."""
from pathlib import Path
import csv, hashlib, json, os, sys, warnings

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]
DEPS = ROOT / '_scratch' / 'figure1_1_bilingual_20261011' / 'python-deps'
sys.path.insert(0, str(DEPS))
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont

SOURCE = ROOT / '修补实验2_deadline重设_20260907' / 'experiment_C2_replay_deadline_fixed' / 'per_seed_results.csv'
SCENARIO = ['cluster_size','device_mix','load','load_per_device','deadline_setting','deadline_scale','strict_setting','strict_qos_prob']
KEYS = SCENARIO + ['seed']
COLORS = ['#0072B2', '#D55E00']
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path, data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def prepare_data():
    source_hash = digest(SOURCE)
    frame = pd.read_csv(SOURCE)
    assert len(frame)==32400 and frame['strategy'].nunique()==12
    oracle = frame.loc[frame.strategy=='oracle', KEYS+['qos_bound_coverage']].copy()
    assert len(oracle)==2700 and not oracle.duplicated(KEYS).any()
    oracle['coverage_defined'] = oracle.qos_bound_coverage > 0
    assert int((~oracle.coverage_defined).sum())==12
    rows=[]
    for strategy in ['full_qos','full_no_risk']:
        values=frame.loc[frame.strategy==strategy].copy()
        assert len(values)==2700 and not values.duplicated(KEYS).any()
        values=values.merge(oracle[KEYS+['coverage_defined']],on=KEYS,validate='one_to_one')
        defined=values.loc[values.coverage_defined]
        means=values.groupby(SCENARIO,dropna=False)[['prediction_mape','sla_violation_rate','false_safe_rate']].mean()
        coverage=defined.groupby(SCENARIO,dropna=False)['qos_bound_coverage'].mean()
        assert len(means)==len(coverage)==135
        row=dict(strategy=strategy,scenarios=135,total_cells=2700,coverage_defined_cells=len(defined),
                 coverage_pct=float(coverage.mean()*100),false_safe_pct=float(means.false_safe_rate.mean()*100),
                 mape_pct=float(means.prediction_mape.mean()*100),sla_pct=float(means.sla_violation_rate.mean()*100))
        rows.append(row)
        means.join(coverage.rename('qos_bound_coverage')).reset_index().to_csv(HERE/(strategy+'_scenario_means.csv'),index=False,encoding='utf-8-sig')
    assert [round(r['coverage_pct'],2) for r in rows]==[95.95,63.65]
    assert [round(r['false_safe_pct'],2) for r in rows]==[2.75,4.33]
    assert abs(rows[1]['mape_pct']-rows[0]['mape_pct'])<.001
    assert abs(rows[1]['sla_pct']-rows[0]['sla_pct'])<.004
    pd.DataFrame(rows).to_csv(HERE/'figure5_plot_data.csv',index=False,encoding='utf-8-sig')
    validation=dict(source=str(SOURCE),source_sha256=source_hash,source_rows=len(frame),
        compared_strategies=['full_qos','full_no_risk'],seed_range=[601,620],scenario_count=135,
        aggregation='Within-scenario mean over seeds, then equal weight over 135 scenarios.',
        coverage='Coverage is defined for strict-QoS tasks. Exclude 12 cells with no strict task, identified by zero Oracle coverage, before each scenario mean.',
        false_safe='Retain all 2700 cells; stored false_safe_rate is conditional on predicted-safe tasks and can have a nonzero denominator without strict tasks.',
        excluded_coverage_cells=12,descriptive_only=True,new_experiments=False,confidence_intervals_added=False,
        rows=rows,delta_no_upper_minus_full_pp={key:rows[1][key]-rows[0][key] for key in ['coverage_pct','false_safe_pct','mape_pct','sla_pct']})
    dump(HERE/'figure5_data_validation.json',validation)
    assert digest(SOURCE)==source_hash
    return rows, validation

def render(rows, language):
    zh = language=='中文'
    family='Microsoft YaHei' if zh else 'Arial'
    font_path=findfont(FontProperties(family=family),fallback_to_default=False)
    plt.rcParams.update({'font.family':family,'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                         'xtick.labelsize':9,'ytick.labelsize':8.5,'axes.unicode_minus':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
                         'axes.linewidth':.8,'savefig.facecolor':'white'})
    fig, axes=plt.subplots(1,2,figsize=(180/25.4,85/25.4))
    fig.subplots_adjust(left=.085,right=.985,bottom=.15,top=.90,wspace=.36)
    labels=['完整风险','移除上界'] if zh else ['Full risk','No upper bound']
    ylabels=['上界覆盖率（%）','假安全率（%）'] if zh else ['Upper-bound coverage (%)','False-safe rate (%)']
    for ax, field, ymax, panel, ylabel in zip(axes,['coverage_pct','false_safe_pct'],[108,5.5],['(a)','(b)'],ylabels):
        values=[r[field] for r in rows]
        bars=ax.bar([0,1],values,width=.58,color=COLORS,edgecolor='white',linewidth=.6,zorder=3)
        bars[1].set_hatch('//')
        ax.set_xticks([0,1],labels)
        ax.set_xlim(-.62,1.62)
        ax.set_ylim(0,ymax)
        ax.set_ylabel(ylabel)
        ax.set_title(panel,loc='left',pad=8,fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#DDE1E5',linewidth=.55,zorder=0)
        ax.tick_params(axis='both',length=3,width=.7)
        if field=='coverage_pct': ax.set_yticks([0,20,40,60,80,100])
        else: ax.set_yticks([0,1,2,3,4,5])
        for b,value in zip(bars,values):
            ax.annotate(f'{value:.2f}%',(b.get_x()+b.get_width()/2,value),xytext=(0,5),
                        textcoords='offset points',ha='center',va='bottom',fontsize=10,fontweight='bold')
    stem=OUT/('图5_覆盖率与假安全率_'+language)
    outputs=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for extension in ['png','pdf','svg']:
            path=stem.with_suffix('.'+extension)
            fig.savefig(path,dpi=600,facecolor='white')
            outputs.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)))
        glyph=[str(w.message) for w in caught if 'Glyph' in str(w.message)]
        assert not glyph,glyph
    dump(HERE/('figure5_render_'+language+'.json'),dict(language=language,font=family,font_path=font_path,
            figure_size_mm=[180,85],png_dpi=600,missing_glyph_warnings=[],outputs=outputs))
    plt.close(fig)
    return outputs

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    rows, validation=prepare_data()
    outputs=[]
    for language in ['中文','英文']: outputs.extend(render(rows,language))
    captions={
        'zh':'图5 风险上界的覆盖—假安全权衡。（a）完整风险与移除上界配置的上界覆盖率；（b）相应假安全率。数据来自风险组件对照，135个场景、每场景20个种子（601—620）；先在场景内对种子取均值，再对场景等权平均。覆盖率仅对存在严格QoS任务的运行有定义，12个无严格任务的场景—种子单元不参与覆盖率场景内平均；假安全率以预测安全任务为条件，保留全部2700单元。条形为描述性均值。移除上界后，MAPE由12.281644%变为12.280687%，SLA违约率由19.621786%变为19.617898%，总体近乎不变，不表示严格相等。风险识别改善不等同于实际SLA收益或概率保证。',
        'en':'Figure 5. Coverage–false-safe tradeoff of the risk upper bound. (a) Upper-bound coverage and (b) false-safe rate for the full-risk and no-upper-bound configurations. The risk-component ablation covers 135 scenarios with 20 seeds per scenario (601–620). Seeds are averaged within each scenario, followed by equal weighting across scenarios. Coverage is defined only for runs containing strict-QoS tasks; 12 scenario–seed cells without such tasks are excluded from within-scenario coverage means. The false-safe rate is conditional on predicted-safe tasks and retains all 2,700 cells. Bars show descriptive means. Removing the upper bound changes MAPE from 12.281644% to 12.280687% and the SLA violation rate from 19.621786% to 19.617898%; both overall means remain nearly unchanged, rather than exactly equal. Better risk identification does not imply an observed SLA benefit or a probabilistic guarantee.'}
    dump(HERE/'captions.json',captions)
    caption='# 图5 数据来源与图注\n\n中文图注：'+captions['zh']+'\n\nEnglish caption: '+captions['en']+'\n\n插入位置：第5.4节“风险组件对照”说明之后。中文/英文版数据和布局相同。PNG为600dpi，PDF/SVG为矢量格式。\n\n原始数据：`'+str(SOURCE)+'`。\n程序：`generate_figure5.py`。数据：`figure5_plot_data.csv`；各策略场景均值及完整数值核查见同目录CSV/JSON。\n'
    (HERE/'Figure5_Sources_and_Caption.md').write_text(caption,encoding='utf-8')
    dump(HERE/'figure5_manifest.json',dict(source_sha256=validation['source_sha256'],script_sha256=digest(Path(__file__)),captions_sha256=digest(HERE/'captions.json'),outputs=outputs))
    print(json.dumps(dict(figure=5,outputs=len(outputs),rows=rows),ensure_ascii=True))

if __name__=='__main__': main()
