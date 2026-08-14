# 预注册:E-lf 长表 matched-firm swap + C-anon 实体匿名化臂(prereg-ea-v1.0)

日期:2026-07-15。先于两项分析的任何统计量提交并打 tag。承诺(家族惯例):全部分支结果
无论方向进论文;单发纪律;修订先于对应统计量并记录;不以结果为条件调口径。

## E-lf — 长表 matched-firm swap(ml rank-1 的直接应对)

**动机(审稿人原话)**:现稿的 swap 证据(ED:杀掉残差 84–93% = content;38 个 HAR-genuine
格中位保留 31%)与标题的 identity 叙事之间的张力未调和;"a reviewer who wants to kill this
paper will use the authors' own swap numbers to do it"。本实验把 swap 推广到**长表面板的
逐模型读数**,直接回答:长表的 HAR-genuine 增量在 correspondence 摧毁后保留多少。

**设计**:机制逐字复用 committed `scripts/analysis/matched_firm_swap.py`(within-day 验证期
RV 配对、level 保留、correspondence 摧毁;配对规则、随机种子 2026、重绘次数与 committed 版
完全一致)。新增:长表文档交换后,对**冻结的**长表模型重推理(无重训):C2 FinBERT-S1、
C5(frozen embed + 已拟合头)、B2 TF-IDF(CPU)。
【v1.2 修订(2026-07-16,先于任何 E-lf 统计量):C2 的三个 horizon checkpoint 已物理灭失
(仅存于箱、遭磁盘清理;本地从未持有 .pt)。重训替身违背本节零训练原则,故 **C2 臂预登记
为"工件灭失、不执行"**,读数覆盖降级为 B2 + C5 两臂并在论文中如实披露(深度微调臂缺席);
C5 的已拟合回归头若文件缺失,按 recipe 在原始 train 上确定性重建并对 committed 预测过 1e-8
复现闸(与 B2 同一机制)——重建即复现,非训练自由度。【v1.3 一词修正(先于 E-lf 统计量,
沿 v1.1 笔误先例):初稿误写"ridge 头";committed C5 配方实为 MLP VolatilityHead(hidden 128,
AdamW 1e-4,早停 patience 3,env.json 在案)。机制不变;1e-8 复现闸依赖同型 GPU 确定性,
闸开火则 C5 臂如实退场。】】每个长表 HAR-genuine 格报告
retention = swap 后增量 / 原增量,day-clustered DM。
**预声明读数**:长表 genuine 格的 retention 中位数与四分位;与 committed ED/全格读数并表。
**分支**:(a) retention 低(中位 <50%)→ 长表增量主要是 content(与 ED 残差一致),
"who spoke" 措辞按既有 bracket 框架限定,swap 张力在正文用一段显式调和;
(b) retention 高(中位 ≥50%)→ level-channel 主导,identity 叙事获得直接支持;
(c) 混合 → 逐格如实。**三分支都必须写进正文的"调和段"**——本实验的目的就是让那段话
有数字可写,不管方向。
**Gates**:G1 committed matched_firm_swap 表机器精度复现(同码路);G2 swap 后输入的
level-保留断言(配对 RV 差异分布与 committed 相同);G3 无重训断言(checkpoint 哈希不变)。

## C-anon — 实体匿名化臂(bound → estimate)

**动机**:ac/ml——"标题的中心量(identity share)从未被点估计;reference interval 只给界"。
掩码-未掩码增量差 = identity share 的直接点估计,与 matched-swap 互补(swap 摧毁
correspondence、保留 level;anon 摧毁 identity 线索、保留全部内容)。

**设计**:NER 掩码(公司名/ticker/高管人名/产品名/CIK;spaCy en_core_web_lg + 规则表,
掩码率与样例披露)应用于**事件驱动面板**(残差所在渠道;长表作 stretch,仅当 ED 完成)。
重跑三臂:C6 prompted(masked excerpts,与原 C6 逐字同协议、同口径——**v1.1 修订:committed
C6 的实际口径为 Qwen3-32B bf16、TP=2(config.json 在案),初稿"AWQ 单卡"系笔误;G1 逐位对照
按构造要求 committed 权重,量化差异会污染 masked/unmasked 比值,故 masked 与 control 均用
bf16 TP=2**;单 seed 2026)、
C2 FinBERT-S1(masked 重训,rung-2 配方 = 固定 recipe,种子 2026,与原臂同法)、
B2 TF-IDF(CPU)。每臂 vs 双参考(单一重校准 HAR / firm-identity)M1 增量,day-clustered DM,
各臂预声明 Holm(6)(3 horizons × 2 refs)。
**预声明点估计**:identity share^anon = 1 − (masked 增量 / unmasked 增量),逐格 + 汇总;
与 reference-interval 的 bound 及 swap 的 retention 三角对照。
**分支**:(a) masked 增量 ≈ 0(share→1)→ identity 主导,标题获得点估计支持;
(b) masked 增量保留大半且仍被 firmID 参考吸收 → "genuinely firm-stable content"叙事,
标题措辞软化(修复即减分,承诺执行);(c) masked 增量保留且**不再**被 firmID 吸收 →
掩码本身破坏了 firm-stable 通道的对齐——如实报告为方法学发现。
**Gates**:G1 未掩码对照重跑与 committed 预测逐位一致(管线不变性);G2 掩码质量抽检
(100 份人工规则核对,漏掩率披露);G3 excerpt 构建器在掩码文本上的截断统计与原版可比。
**范围披露**:2 GPU box;ED 优先;长表 stretch;C5x 不进本轮(GPU 预算,预登记为未做)。
**v1.1 打分前操作化补登(先于任何 anon 统计量)**:分支阈值 share 中位数 ≥0.75 → (a)、
≤0.50 且 masked 增量仍被 firmID 吸收 → (b)、否则按 (c)/混合逐格;汇总聚合 = 中位数(与 swap
retention 惯例一致);unmasked 增量非正的格 share 记 n/a 不剔除;G1 对 GPU 臂按
exact-match 率 + max|diff| 报告,非逐位一致需 --record-g1-deviation 显式记录后方可进入打分
(bf16 批式推理非位确定,REVIEW_BLINDSPOTS 在案)。

## v1.4 修订(2026-07-16,先于任何长表-anon 统计量):LF stretch 操作化 — B2-only

v1.0 的执行门"长表作 stretch,仅当 ED 完成"**在此显式修订**为打分时序门:ED anon 打分表
(`anon_arm.{csv,md}`)必须先于任何 LF 统计量落盘;LF 掩码构建与 B2-lf 两臂运行可与 ED C6
推理并行——因为本节把 LF 的全部决策自由度先行封死,且不读任何 ED anon 统计量(ED 打分
尚未开火,anon_arm 表尚不存在;results/anon/ 现存 *_smoke 产物为非打分工程件)。v1.4
提交+tag 先于任何非 smoke LF 运行。以下决定**仅基于 committed 表格数字**。

**臂集 = B2-lf 单臂**(TF-IDF ridge,长表;masked = 匿名化 store 上固定配方重训,种子 2026,
与 ED B2 臂同一估计量)。**估计目标**:committed 主表 B2-lf 增量的 identity share —
vol_rel_impr_pct = +3.3305/+3.4823/+5.9199%(h=5/10/20,m1_ensemble_primary;B2 单种子,
即打分分母)——其中 h=20 即正文 "up to 5.9%" 的头条格。

**share 估计面收缩(构造性,引 committed 数字)**:B2-lf 对 firm-identity 参考的无掩码
增量为 −0.615/−3.892/−8.089%(h=5/10/20,committed firm_identity_control.csv)——全非正,
故按 v1.1 n/a 规则,firmID 侧 3 格 share **先于执行即 n/a**;share 中位数 = **有定义格
(3 个 HAR 格)的中位数**(3 格时即 h=10 格;偶数格时取中间两值均值——通用登记)。
**v1.1 n/a 规则义域澄清(两渠道同义,均先于打分)**:n/a 只作用于 share,不作用于检验——
share=n/a 的**已执行**格仍保留其 day-clustered DM 并留在该臂 Holm(6) 家族内(3 horizons ×
2 参考,v1.0 原文);只有 status=not-executed 的臂(C2/C5/C6-lf)不携带检验。

**分支量化操作化(两渠道同用,均先于两表开火;打分脚本与本文逐字一致并随本 tag 提交)**:
- **(a)** ⇔ 有定义格 share 中位数 ≥0.75 **且** masked 臂对 HAR 参考 0/3 格 Holm 显著为正;
- **(b)** ⇔ 中位数 ≤0.50 **且** masked 增量仍被 firmID 吸收,吸收 = masked 臂对 firmID
  参考 Holm 显著为正的格 ≤1/3;
- **(c)** ⇔ 其余(含中位数落 (0.50,0.75)、或 firmID Holm 显著 ≥2/3 = 吸收破坏),混合逐格。
- LF 退化性如实登记:B2-lf 的**无掩码**增量已被 firmID 全吸收(上引 committed 数字),
  故 (b) 的吸收子句对 LF 预期为真、区分力弱——LF 的 (b)/(c) 判别实际落在中位数与
  firmID-Holm 计数上,论文措辞按此弱化。

**排除(全部先于统计量)**:
- **C2-lf**:工件灭失先例(v1.2)+ 重训边际成本 ~20–30 GPU-h,不执行。
- **C5-lf(HAR 侧):构造性 n/a**。committed m1_multiseed.csv 长表 C5_qwen3 seed-2026
  rel_impr_pct = −1.0347/−3.1346/−6.6467(主引,与打分时单种子口径同基);deployable_combiner
  FIXED mean rel% = −0.85/−2.48/−5.97 佐证(3 种子基)。全负 → HAR 侧 share 格先于执行即空。
- **C5-lf(firmID 侧)**:committed 表中无该增量 → 不援引构造性 n/a,与 C6-lf 同以 GPU
  预算/范围排除;且其执行与否不能改变分支判定(中位数只取 HAR 格)。代码路径保留,不执行。
- **C6-lf:前提更正 + 预算排除**。此前内部工作假设"C6 未跑过长表"**有误**——committed
  `C6_llmtext_full_long_form_seed2026` 真实存在(11,907/11,907 长表文档全覆盖,parse_fail 0)。
  故 C6-lf masked 臂**良定义**且比 ED C6 便宜(11,907 vs 39,322 文档);排除理由只能是也仅是
  GPU 预算与范围(ED 渠道才是残差所在),不得写成"不可行"。
- 不重训任何文本模型配方(B2 固定配方重训与 ED 同法,非调参)。

**产物与时序**:姊妹表 `results/tables/anon_arm_lf.{csv,md}`,write-once,与 ED 表分文件
(单发守卫按文件生效;ED 表先开火)。**三角对照列**(操作化):= E-lf **B2 臂**同 horizon
的 document-swap retention,取自 committed `swap_longform.csv`;该格缺席或 E-lf G1–G3
任一未过则记 n/a 并注明原因。LF 打分硬性前置:`swap_longform.csv` 与 `anon_arm.csv` 均已存在。
**Gates 映射**:**G1(CPU 臂,无偏差逃生门)**= control 复现过 **1e-8 全面板复现闸**
(沿 E-lf 已登记的 CPU 惯例"与 B2 同一机制";v1.1 的 exact-match 率 + --record-g1-deviation
流程仅限 GPU 臂)——超过 1e-8 即管线不变性失败,**臂如实退场**(报 G1-fail,无 share 估计),
不得走偏差记录路线。G2 = mask_stats_lf.json + 与 ED 同协议的 100 份审计抽检。G3(C6 截断
统计)无 C6-lf 臂即 n/a,md 中说明。
**工程登记**:anon_mask_build.py 新增 --panel lf(输出改名 *_lf)与 --batch-docs 原子
分片续跑——纯工程,无统计效应;默认 0 = 原单发路径,ED 已跑路径不受影响。

## v1.5 修订(2026-07-16,先于 LF 表落盘;E-lf 已依自身闸全员退役,其统计量从未产生)

**事实记录**:E-lf 三臂全部依已登记规则退场——C2 工件灭失(v1.2);B2 重建复现闸
max rel diff 1.402e+00(v1.2"与 B2 同一机制"的闸);C5 重建复现闸 1.062e-05 > 1e-8
(v1.3:"闸开火则 C5 臂如实退场")。swap_longform.csv 永不存在;E-lf 预声明读数无法产生。
退役记录 = `results/tables/swap_longform_retirement.md`(非统计表,门开火数字 + 诊断链)。

**调和段义务保持**:v1.0"三分支都必须写进正文的调和段"改由退役事实 + 已 committed 的
ED matched-swap 与 C-anon 表执行(anon share 0.51 vs ED swap 隐含 0.71 的间隙 = 名义身份
vs level-alignment 通道的分解)——不引入任何新统计量。

**LF 表前置处置**:v1.4 硬性前置 "swap_longform.csv 已存在" 因退役不可满足。鉴于 LF 表
依既有裁决不含任何统计量(1 行 g1-fail + 3 行 not-executed,判定 undefined,三角列本就
登记"缺席即 n/a 注明原因"),前置修订为:"swap_longform.csv **或** swap_longform_retirement.md
存在";三角列 n/a 原因 = E-lf retired at its own gates。本修订不改变表中任何单元格取值
(全部由已 committed 的门记录决定)。

## v1.6 修订(2026-07-16,先于任何 share 置信区间统计量):identity share 的 day-block bootstrap CI

**动机**:R12 skeptic 原话("Bootstrap confidence intervals on the per-horizon identity shares
(0.51/0.56/0.71)")。纯不确定性量化,无新判定、无分支——分支判定已由 anon_arm 表(已开火)
永久锁定,本分析不得也不能改动它。

**设计**:对已提交的 ED anon 预测(ctrl/masked 六跑,commit a721b8b)做 **day-block bootstrap**:
重采样单位 = effective trading day(与 day-clustered DM 同单位),各格在其 test 面板内有放回
重采样天,B = 2000,种子 2026;combiner 权重按 committed(val 拟合、test 冻结)**不重拟合**;
每次抽样重算 unmasked/masked M1 增量(与 anon_score 同机器)与 share = 1 − masked/unmasked。
**登记读数**:C6 HAR 侧 3 格 share、C2 HAR 侧 3 格 share、两臂中位、六格池化中位的
**percentile 95% CI**;某抽样中 unmasked 增量 ≤0 → 该抽样 share 无定义(沿 v1.1 n/a 规则),
CI 取自有定义抽样并**披露无定义比例**(比例 >20% 的格,CI 旁标注不稳定)。
**产物**:`results/tables/anon_share_ci.{csv,md}`,write-once 单发。**散文规则**:论文引用点估计
处附 CI(摘要视版面可豁免);CI 不改变任何已锁定分支措辞。

## 边界

两项均不触碰 test 标签选择;E-lf 零训练;C-anon 的 FinBERT 重训用固定 recipe(非调参)。
与 prereg-cd(D 在跑)共箱不共卡:D 占 CPU 核,E-lf/C-anon 占 GPU。
