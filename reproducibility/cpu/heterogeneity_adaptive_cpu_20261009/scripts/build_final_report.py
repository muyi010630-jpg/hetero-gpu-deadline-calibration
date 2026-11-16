"""Describe all completed conditions; no result-dependent experiment selection."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NAMES={'full_original':'原分层Full','local_ema':'局部族步长EMA','pool_ema':'组级池化EMA',
       'gated_point':'首反馈后停止点借用','local_window':'局部窗口均值','adaptive_pool_local':'自适应部分池化'}
def read(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def jread(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def link(rel,label=None):return f'[{label or Path(rel).name}]({(ROOT/rel).as_posix()})'
def fmt(v):return f'{float(v):.3f}'
def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]
def main():
    qa=jread('qa/completed_data_checks.json');assert qa['passed']
    frozen=jread('00_protocol_sources/final_freeze.json')
    static=read('03_independent_prediction_test/condition_summary.csv');infer=read('03_independent_prediction_test/primary_paired_inference.csv')
    closed=read('04_closed_loop_scheduling/full/analysis/summary.csv');ci=read('04_closed_loop_scheduling/full/analysis/paired_image_inference.csv')
    drift=read('05_drift_tests/condition_summary.csv');di=read('05_drift_tests/exploratory_paired_inference.csv')
    pilot=read('01_mechanism_diagnostics/condition_summary.csv')
    rows=[]
    for mode in NAMES:
        s=[r for r in static if r['mode']==mode];c=[r for r in closed if r['mode']==mode]
        rows.append((NAMES[mode],fmt(np.mean([float(r['overall_macro_mape']) for r in s])),
                     fmt(100*np.mean([float(r['sla_violation_rate']) for r in c])),
                     fmt(100*np.mean([float(r['panel_median_mape']) for r in c]))))
    effects={}
    for label,data,baseline_key,pkey,dkey in [('static',infer,'comparator','p_holm','candidate_minus_comparator_pp'),
                     ('closed', [r for r in ci if r['metric']=='sla_violation_rate'],'baseline','p_holm_primary','mean_delta')]:
        effects[label]={}
        for base in ('pool_ema','local_window','full_original'):
            subset=[r for r in data if r[baseline_key]==base]
            effects[label][base]=dict(n_conditions=len(subset),significantly_better=sum(float(r[pkey])<.05 and float(r[dkey])<0 for r in subset),
                                     significantly_worse=sum(float(r[pkey])<.05 and float(r[dkey])>0 for r in subset),
                                     no_detected_difference=sum(float(r[pkey])>=.05 for r in subset))
    lines=['# 异质性控制与自适应校准：CPU补实验总报告','',
           '本轮已完成固定设计的机制诊断、独立参数验证、正式预测测试、闭环调度与漂移敏感性实验。全部正式条件与候选参数在测试开始前锁定；旧组级池化补实验与论文正文保持原样。',
           '', '## 主要结果概览','',
           '下表是各阶段预定条件的等权描述均值。固定反馈误差为观测时长的上下文宏平均MAPE；闭环SLA为实际任务违约率；公共面板误差使用同一45个代表单位的无噪声中位时长。三个指标评估对象不同，不能直接相减。', '']
    lines+=table(['方法','固定反馈MAPE (%)','闭环SLA违约率 (%)','闭环公共面板MAPE (%)'],rows)
    lines+=['','候选与三项强基线的正式条件比较如下。负差表示候选更好；“未检出”不表示等效。固定反馈和闭环各自30个主检验共同Holm校正，不能把条件获胜数解释为普遍胜率。','']
    count_rows=[]
    for stage,label in [('static','固定反馈：10个ρ×流条件'),('closed','闭环：10个ρ×负载条件')]:
        for base,counts in effects[stage].items():count_rows.append((label,NAMES[base],counts['significantly_better'],counts['significantly_worse'],counts['no_detected_difference']))
    lines+=table(['阶段','比较基线','显著更好','显著更差','未检出差异'],count_rows)
    sfull=effects['static']['full_original'];cfull=effects['closed']['full_original'];cwindow=effects['closed']['local_window']
    lines+=['',f"相对原Full，候选在固定反馈10个条件中有{sfull['significantly_better']}个显著更好、{sfull['significantly_worse']}个显著更差；在闭环10个条件中只有{cfull['significantly_better']}个SLA显著更好。相对LocalWindow，只有{cwindow['significantly_better']}个闭环条件显著更好。这些结果应作为条件性的证据，不能把全条件平均改善表述为普遍QoS优势。"]
    swindow=effects['static']['local_window']
    avg_adaptive=np.mean([float(r['overall_macro_mape']) for r in static if r['mode']=='adaptive_pool_local'])
    avg_window=np.mean([float(r['overall_macro_mape']) for r in static if r['mode']=='local_window'])
    lines+=['',f"最关键的强基线结果是：候选与LocalWindow的全条件平均MAPE差仅{avg_adaptive-avg_window:.6f}个百分点（描述均值，未另做全条件优势检验），逐条件为{swindow['significantly_better']}个显著更好、{swindow['significantly_worse']}个显著更差、{swindow['no_detected_difference']}个未检出差异。相对原Full的改善不能全部归因于新增自适应共享；局部窗口估计本身解释了很大部分收益。共享结构与反馈稀疏度需同时交代。"]
    lines+=['', '补充的探索性全条件画像块95%CI为[−0.031137, +0.013687]个百分点，跨越0；没有新增正式p检验。'+link('03_independent_prediction_test/exploratory_aggregate_CI.csv','全条件探索性CI')+'。']
    lines+=['','## 实验如何控制原审稿质疑','',
       '从原来8个Worker的实测画像中去除型号克隆，保留45个唯一型号×上下文。型号有RTX5070、T4和A100；任务族为compute、memory和inference。对数初始失配的总体均值固定为0，标准差固定为0.20。以ρ=0、0.25、0.50、0.75、1.00控制实现的组间方差占比：ρ=1为组内常数，ρ=0为只有组内上下文差异。每个画像在所有ρ使用相同底层向量。这样不会把不同总体失配强度混入共享结构的比较。',
       '', 'ρ描述这45维受控画像的组间方差比例，不是i.i.d.上下文相关系数。画像是统计推断单位；同一画像内的任务种子先平均。20个正式画像为独立随机失配方向，并不是20台真实GPU，也不是45个独立上下文。',
       '', '固定反馈使用balanced与late_sparse两种流。后者预先用SHA256排序选定22个上下文，前40%不出现，之后每个单位的抽样权重为其余单位的1/4；不根据实验效果挑选稀疏单位。每个事件所有方法先预测，再接收相同带噪声时长。first-k按目标实际已反馈次数统计，区别于全流前20%的cold_phase。',
       '', '所有模式保留原族级EMA步长（compute 0.35、memory 0.45、inference 0.22）、创新裁剪±0.35、状态裁剪±0.70、窗口24以及原Full风险规则。PoolEMA直接维护每个型号×任务族的共享残差状态。局部窗口均值单独列为强基线，用于判断自适应共享收益是否只是更换局部估计器。各方法的漂移检测都关闭。',
       '', '## 机制诊断（开发阶段，描述性）','',
       '5画像×3ρ×3任务种子×两种流=90流，每流600事件，13方法共702000预测。下列为该开发矩阵的等权宏MAPE；画像数、ρ和事件长度与正式测试不同，不能直接将两阶段均值作差。它用于诊断机制，没有进入正式测试参数选择。', '']
    diagnostic_names={'hierarchy_common_scope':'分层仅借同型号同族（匹配Pool范围）','pool_matched_dtype':'Pool增加precision分组','full_no_neighbor_shrink':'去除邻居二次收缩',
                      'full_fresh_neighbors':'分层邻居加年龄权重','pool_exposure_matched':'Pool按已知局部单元数降低单次步长','adaptive_fixed_mix':'固定混合比例（反馈数规则）','adaptive_no_freshness':'默认候选关闭年龄衰减'}
    lines+=table(['方法/diagnostics','开发矩阵宏MAPE (%)'],[(NAMES.get(m,diagnostic_names.get(m,m)),fmt(np.mean([float(r['overall_macro_mape']) for r in pilot if r['mode']==m]))) for m in sorted({r['mode'] for r in pilot})])
    lines+=['', '去除邻居二次收缩和调整借用范围的均值变化，不能单独解释全部估计差距；直接Pool、Window和候选的结果同时受共享结构及局部估计方式影响。因此保留各正式强基线，而不把开发诊断当成消融因果证明。'+link('01_mechanism_diagnostics/condition_summary.csv','按ρ/流的全部开发结果')+'。',
       '', '## 候选方法与参数选择','',
       '候选以组级EMA和目标24项局部反馈窗口均值为两个端点，根据历史窗口中的组内差异代理与局部采样方差决定局部权重。重复Worker在异质性统计中按唯一型号×上下文合并，不当作多个独立上下文。该代理是描述性量，本轮不声称其无偏性或风险概率保证。',
       '', '验证网格只包含收缩系数0.5/1/2与年龄半衰期1.5/6/10⁹秒，共9项；基线不依据验证或测试改参数。验证使用6个独立画像、5个ρ、两种流和3个任务种子，每流600事件，按上下文宏MAPE等权选一个配置。',
       '', f"冻结配置为 `{json.dumps(frozen['selected_params'],ensure_ascii=False)}`。验证目标值为{fmt(frozen['validation_objective'])}%。10⁹秒在12秒实验中等效关闭年龄衰减，不能据此宣称反馈年龄衰减已被验证为必要。候选是部分池化思想的在线实现，不能仅凭混合权重宣称新的统计估计原理。",'',
       link('00_protocol_sources/final_freeze.json','正式冻结记录')+'；'+link('00_protocol_sources/候选方法定义.md')+'；'+link('00_protocol_sources/设计与创新定位说明.md')+'。',
       '', '## 正式固定反馈结果','', '20画像×5ρ×5任务种子×2流=1000流，每流1800事件，6种方法共10,800,000项预测。主指标先在每个流的已观测上下文内算MAPE，再对上下文等权平均；不为尚未出现的上下文虚填误差。已观测数量保存在逐流结果。', '']
    index={(float(r['rho']),r['flow_kind'],r['mode']):r for r in static}
    lines+=table(['ρ','流','原Full','PoolEMA','LocalWindow','候选'],[
        (fmt(rho),'均衡' if flow=='balanced' else '延迟稀疏',*[fmt(index[(rho,flow,m)]['overall_macro_mape']) for m in ('full_original','pool_ema','local_window','adaptive_pool_local')])
        for rho in (0.,.25,.5,.75,1.) for flow in ('balanced','late_sparse')])
    lines+=['', '完整主比较（差值单位：百分点；95%CI为边际画像块percentile bootstrap区间）：','']
    lines+=table(['ρ','流','基线','候选−基线','95%CI','Holm p'],[
        (fmt(r['rho']),r['flow_kind'],NAMES[r['comparator']],fmt(r['candidate_minus_comparator_pp']),f"[{fmt(r['ci_lower_pp'])}, {fmt(r['ci_upper_pp'])}]",f"{float(r['p_holm']):.5f}") for r in infer])
    lines+=['',link('03_independent_prediction_test/primary_paired_inference.csv','固定反馈主检验CSV')+'；'+link('03_independent_prediction_test/condition_summary.csv','所有方法与冷启动/反馈次数分段结果')+'。',
        '', '## 闭环调度与预测—QoS边界','',
        '20画像×5ρ×5任务种子×2负载=1000配对单元，6种方法共6000运行。8个Worker由2个RTX5070、3个T4、3个A100实例构成；仿真到达窗口12秒，负载参数为每设备1.25与1.70。截止期以资源池实测中位时长为锚点，所有方法使用同一任务、用户、潜在设备时长与截止期；哈希逐单元核对。运行等待全部实际任务完成。',
        '', '固定8个时刻在校准状态的深拷贝上评估45个唯一型号×上下文，代表Worker为各型号__01。这些被动面板不回写反馈，也不改变实际完成时刻、队列或吞吐量统计终点。面板不是全部Worker或实际任务频率的替代指标。闭环风险规则一致，但风险数值因内生反馈而可能不同。', '']
    cindex={(float(r['rho']),float(r['load']),r['mode']):r for r in closed}
    lines+=table(['ρ','负载','原Full SLA(%)','Pool SLA(%)','LocalWindow SLA(%)','候选 SLA(%)'],[
        (fmt(rho),fmt(load),*[fmt(100*float(cindex[(rho,load,m)]['sla_violation_rate'])) for m in ('full_original','pool_ema','local_window','adaptive_pool_local')])
        for rho in (0.,.25,.5,.75,1.) for load in (1.25,1.70)])
    lines+=['','完整SLA主比较（百分点）：','']
    lines+=table(['ρ','负载','基线','候选−基线','95%CI','Holm p'],[
        (fmt(r['rho']),fmt(r['load']),NAMES[r['baseline']],fmt(r['mean_delta']),f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]",f"{float(r['p_holm_primary']):.5f}") for r in ci if r['metric']=='sla_violation_rate'])
    lines+=['',link('04_closed_loop_scheduling/full/analysis/summary.csv','闭环全部指标')+'；'+link('04_closed_loop_scheduling/full/analysis/paired_image_inference.csv','SLA与公共面板配对比较')+'。面板比较是探索性结果；其未校正p不能替代SLA的30项主检验。',
        '', '## 漂移敏感性','',
        '预定的10个正式画像子集×3个初始ρ×3任务种子×两种流×两类漂移=360流，每流900事件，6方法共1,944,000项预测。漂移在中点加入独立生成方向，幅度为对数RMS 0.15，分别为组级或组内上下文变化。未据静态结果重新挑选漂移画像。检测器均关闭；该结果用于检查估计结构的响应，不是最优漂移检测比较。', '',
        '漂移方向未强制与初始失配正交，因此漂移后的总体失配标准差和组间方差比例可能变化；此处ρ指初始共享比例。两类型匹配的是添加漂移RMS，而非最终总失配幅度，不能将类型间差异解释为严格等最终幅度的因果对照。漂移后半期结果另存探索性CI，用于避免前半期静态数据稀释跟踪劣势。', '']
    dindex={(float(r['rho']),r['flow_kind'],r['drift_type'],r['mode']):r for r in drift}
    lines+=table(['ρ','流','漂移类型','Pool','LocalWindow','候选','候选漂移后MAPE'],[
        (fmt(rho),flow,kind,*[fmt(dindex[(rho,flow,kind,m)]['overall_macro_mape']) for m in ('pool_ema','local_window','adaptive_pool_local')],fmt(dindex[(rho,flow,kind,'adaptive_pool_local')]['post_drift_macro_mape']))
        for rho in (0.,.5,1.) for flow in ('balanced','late_sparse') for kind in ('group','context')])
    lines+=['', '候选的漂移后半期等权平均MAPE为10.80887%，高于原Full的10.35144%和局部EMA的10.15907%，低于Window的11.25466%。在初始ρ=1、均衡流、上下文级漂移时，候选相对Window后半期恶化1.197858个百分点，边际95%CI[1.155365,1.243941]；相对原Full恶化3.574983个百分点，CI[3.475000,3.669679]。在同一初始ρ的组级漂移均衡流中，相对Window则改善3.492332个百分点，CI[−3.684568,−3.275962]。两种方向均保留，不能用静态收益声称更强的普遍漂移跟踪能力。', '',
        '36项全程画像块比较作为单独的探索性Holm族。10个画像块的双侧精确signflip最小p为2/2¹⁰≈0.001953，大于第一步Holm阈值0.05/36≈0.001389；该设计缺少通过第一关的分辨率，不能把未拒绝零假设解读为无效应或等效。后半期只报告未做同时校正的探索性边际CI。全部劣势和无差异条件一并保留：'+link('05_drift_tests/exploratory_paired_inference.csv','漂移全程差值、CI及校正p')+'；'+link('05_drift_tests/exploratory_post_phase_CI.csv','后半期探索性CI')+'；'+link('05_drift_tests/drift_effective_profile_moments.csv','漂移前后实际画像矩')+'。',
        '', '## 完整性、复现与限制','',
        '冻结前实现核查通过：155画像矩误差小于10⁻¹²；Pool直接更新与手算一致；13模式在同反馈下原局部状态和风险σ完全一致；原Full点估计和guard与冻结原版相同。缓存优化逐步数值误差0。被动面板和不含面板的Full闭环与冻结原版29个数值指标误差均为0。',
        '', 'Windows并行缓存写入竞争导致首个验证attempt缺1流；该attempt完整保留，随后修复文件写入方式并重跑全部180流。与原179个成功流的1,503,600预测及共同NPZ内容精确一致。它不参与验证选择或正式统计。IO修复未改变公式、种子或条件。',
        '', '正式数据逐文件SHA、全部预测的正值与有限值、同反馈更新次数与风险σ、Pool递推、MAPE独立重算、闭环任务配对、全部任务排空与面板指标重算均通过。'+link('qa/completed_data_checks.json','完成后数据核验')+'；'+link('qa/闭环独立审查.md')+'。',
        '', '统计区间按画像块配对bootstrap 5000次；signflip 100000次并做+1校正。区间为边际95%区间，不是同时区间；signflip依赖零假设下块差值可交换/符号对称。任务流数不会被当成独立画像重复。结论条件于该实测基础画像、受控生成机制和共用任务种子，不能直接推广为所有部署环境的概率保证。',
        '', '本轮为本地CPU实验，没有重新执行三GPU真实在线校准。也没有测量候选的最小部署开销；所有模式为诊断共同维护新增历史。原论文若采用候选的新机制，需要明示其与旧Full的区别，并补真实部署验证后才能写真实系统收益。',
        '', '## 论文如何使用','',
        '保留旧组级常数失配下PoolEMA的优势，正式加入组级直接池化对照。将本轮受控ρ实验用于解释共享结构的适用边界，避免把族级步长调整收益合并算作分层借用创新。主表同时列PoolEMA、LocalWindow、原Full及候选，附画像块CI与校正p。',
        '', '候选不能悄然替换原Full并继承旧方法名称。若采纳，在3.3节分层借用之后增加组内差异估计、局部窗口、混合权重和更新顺序，给候选独立名称；引言贡献与实验描述同步修改。年龄衰减的必要性未建立，当前冻结实现应如实披露近乎关闭。',
        '', '把机制贡献限定到冻结测试支持的条件，把经验贡献落到池化/局部估计的边界以及预测与QoS的关系。对于未检出或显著不利的条件应正面说明，不能删除原实验或只报告获胜条件。首反馈门控与原Full的既有相近结果仍须保留，不能据新候选结果反推持续分层融合必要。',
        '', '当前证据不支持把候选升级为“全面优于简单局部窗口”的主方法。更合适的论文定位是受控共享结构、稀疏反馈和QoS收益边界的经验研究；如以新增估计机制为主要创新，还需要独立于本轮测试的理论或部署证据。',
        '', '当前报告的比较基于冻结结果，论文结论仍需结合真实在线验证和相关工作定位。']
    report=ROOT/'report/实验总报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary=dict(completed=True,selected_params=frozen['selected_params'],overview=[dict(method=r[0],fixed_macro_mape_pct=float(r[1]),closed_sla_pct=float(r[2]),panel_median_mape_pct=float(r[3])) for r in rows],primary_condition_counts=effects,
                 formal_fixed_predictions=10800000,formal_drift_predictions=1944000,formal_closed_runs=6000,
                 raw_data_verified=True,real_gpu_rerun=False,all_formal_conditions_retained=True)
    (ROOT/'report/主要结果汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    readme=['# 本地CPU补实验：异质性控制与自适应校准','',
       '状态：全部预定CPU实验完成，正式数据与汇总核验通过。旧补实验和论文正文保持原样。', '',
       link('report/实验总报告.md','从实验总报告开始阅读')+'；'+link('report/主要结果汇总.json','主要结果机器可读摘要')+'。','',
       '目录：00为设计/来源/冻结；01为13方法机制诊断；02为9格参数验证；03为正式共用反馈预测；04为匹配调度器闭环；05为漂移敏感性；scripts为复现代码；qa为实现及数据核验；logs为运行日志。', '',
       '关键原始数据：03/raw和05/raw下每流NPZ保存全部预测、反馈、更新前计数与诊断；04/full/raw保存6000运行汇总、任务计划、1000被动面板和代表执行轨迹。02/attempt_01_shared_cache_write保存首个不完整验证attempt，不用于结论。', '',
       '复现请使用已保存的final_freeze.json和同版本脚本；runner会拒绝修改代码/参数后的原目录续跑。详见报告中的冻结文件和run_config。所有本轮结果都是CPU模拟与受控回放，没有重采三台真实GPU。']
    (ROOT/'README.md').write_text('\n'.join(readme)+'\n',encoding='utf-8')
    manifest=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.suffix in ('.py','.md','.json','.csv') and '__pycache__' not in str(path) and path.name!='结果文件索引.json':
            manifest.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    (ROOT/'report/结果文件索引.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
