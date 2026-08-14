# 预注册:8-K 残差的家族鲁棒性审计 + item-code 控制 + omnibus 检验(R9 队列 B/D)

日期:2026-07-15。本文件在任何下述统计量被计算**之前**提交并打 tag(`prereg-rfa-v1.0`)。
动机:R9 面板 4/4 rank-3 concern(唯一正结果卡在 p=.049 边界格);quant 审稿人:
"the paper catches the literature's shortcut and misses its own"(item-code 控制不跑的代价)。
承诺:**所有分支的结果无论方向一律进论文;不得以结果为条件选择口径。**

## 冻结输入(已在盘,本文件提交时未被下述任何检验读取过)

- `results/runs/C6_llmtext_llama70_full_event_driven_seed2026/predictions.parquet`(单种子,已入 committed 表)
- `results/runs/C6_llmtext_llama70_s2027_full_event_driven_seed2026/predictions.parquet`
- `results/runs/C6_llmtext_llama70_s2028_full_event_driven_seed2026/predictions.parquet`
- `results/runs/C6_llmtext_llama70ens_full_event_driven_seed2026/predictions.parquet`(3 种子 ensemble)
- 对照锚:`results/tables/crossfamily_llama70.csv`(committed,单种子 Holm(6) 读数)

## B0 — Llama-70B 三种子 ensemble 重打分(零 GPU)

**检验批**:与 `scripts/analysis/crossfamily_llama70.py` 的 M1 块逐字相同——log-space 组合器
val 拟合、test 冻结;参考 (a) 单一重校准 HAR(A2),(b) firm-identity 增强参考(val 窗公司均值 spec);
day-clustered DM,HAC lag h−1,HLN 修正。**Holm 施加于新的预声明 6 检验族**
(ensemble 的 3 horizons × 2 references),与单种子的 Holm(6) 平行,不合并。

**Sanity gates(任一失败即中止,不出表)**:
- G1′:单种子 llama70 行按同一代码路径复算,与 committed `crossfamily_llama70.csv` 机器精度一致;
- G5(v1.2 修订):ensemble 预测逐行等于三种子预测的**算术均值**(`mean(pred_seed)`,rtol 1e-6);
  若不等,先查 ensemble 的生成脚本再议,**不得**用"近似成立"继续。
  【修订记录 2026-07-15:v1.1 误写为 log 空间均值(错抄 HPO seed_validation 的口径);闸门按规定
  中止后核查生成脚本,确认冻结 artifact 的口径是算术均值——row15 launch.sh 刻意对齐论文
  seed-ensemble primary 的 `m1_ensemble_primary.ensemble_text` 惯例并在 config.json 披露。
  与 C 模型 primary 基准做跨家族比较,算术均值才是一致口径。修订发生在任何 ensemble 行的
  M1/Holm 统计量被计算之前(G5 在 M1 块之前执行,中止时未触及下游)。】

**预声明判定阶梯(ensemble 行)**——沿用现行脚本的分级文字,阈值不变:
- REPLICATES:vs firmID 参考 3/3 horizons Holm<.05 且 DM<0;
- DIRECTIONALLY REPLICATES:3/3 rel_firm>0 且(≥2/3 raw p<.05 firmID 或 ≥1/3 Holm vs HAR);
- DOES NOT REPLICATE:0/3 正号且 0/3 raw 显著;
- 其余:PARTIAL/MIXED,按数报告。
单种子行**保留**在表中,ensemble 行并列;禁止用 ensemble 行替换单种子行后重述历史。

## B1 — 第三家族(模型待核验后修订本文件再启动)

家族集合 F = {Qwen3-32B(primary,单 seed,已披露)、Llama-3.1-70B-AWQ(3 种子 ensemble)、第三家族(3 种子 ensemble)}。
第三家族必须非 Qwen/Llama/Yi/Phi 谱系(Qwen2.5-72B 与 primary 同宗,**不合格**;
Llama-3.3 与复制臂同族,**不合格**)。

**第三家族选定(核验后修订,2026-07-15,生成任何预测之前)**:
`mistralai/Mistral-Small-24B-Instruct-2501`(Mistral 谱系;apache-2.0,非 gated,镜像可直下)。
服务方案:vLLM offline batch,bf16,TP2,`--max-model-len 8192`,与 C6/llama70 逐字相同的
prompt / guided-JSON / clip[0.03,3.0] / retry 协议(`scripts/experiments/e1_llm_forecast`)。
**"3 种子"语义与 llama70 复制臂完全一致**:temperature=0 下 seeds 2026/2027/2028 仅通过
kernel 非确定性产生差异(reproducibility-jitter ensemble,launch.sh 已披露);论文措辞沿用
该口径,不得称随机解码 ensemble。备选(仅当 Mistral 下载/加载失败时启用,需追加修订):
`google/gemma-2-27b-it`(gated=manual,需已获批 token;chat template 无 system role,需 shim)。
检验批与 B0 完全相同;每家族各自的预声明 Holm(6)。

**跨家族主张规则(预声明)**:
- 家族 STRONG 通过:≥2/3 horizons Holm<.05 且 DM<0 vs 单一重校准 HAR(族内 Holm(6));
- 家族 WEAK 通过:达到 B0 阶梯的 DIRECTIONALLY REPLICATES 或以上;
- 论文措辞:≥2/3 家族 STRONG →"replicates across families";
  ≥2/3 家族 ≥WEAK(含 primary)→"sign-robust across families, significance attenuated";
  否则 →"does not replicate beyond the primary family"(残差段落相应降级;
  按 FACTS.md 既定规则,不得写成 family-specific 的证明)。

## IC — item-code / earnings-window 控制(零 GPU;数据 = predictions.parquet 自带 `item_subtype`,0% 缺失)

**Primary spec**:firm-identity 参考的 log-space 组合器中追加一个二元项
`has_202 = 1[item_subtype 含 "2.02"]`(对 log 变换取 `L(x)=log(clip(x+ε))` 不适用于哑元——
哑元**直接线性进入**设计矩阵,不做 log)。val 拟合、test 冻结,与现行组合器同法。
**Secondary spec**(仅报告,不参与判定):追加 train+val 频次 top-8 的 item 二元指示。
对象:C6_llmtext(Qwen3-32B,**单 seed**——C6 以近确定性解码单 seed 进入,paper 已披露,
无 3 种子 ensemble;本行修正于任何 IC 统计量计算之前,修正原因:上稿误写为 ensemble),
event-driven,3 horizons。**预声明 Holm(3) 族**(3 horizons × 1 个增强参考)。

**判定**:
- 残差在 firmID+has_202 参考下 ≥2/3 horizons DM<0 且 Holm<.05 →"not an earnings-window artefact";
- 否则 → 论文改写:8-K 残差(部分)是 earnings-window 效应,残差段落降级,摘要的
  "what survives" 句相应弱化。**两分支均承诺进正文。**

## D — 跨格 omnibus 联合检验 + 功效校准(零 GPU)

- 统计量:69 格 primary 家族(seed-ensemble 基准,vs 单一重校准 HAR)的逐日损失差
  (QLIKE(f_R) − QLIKE(f_U)),先在 (day, cell) 上取格内当日均值,再对 day 取跨格均值,
  得单一逐日序列;对其做 day-clustered DM(HAC lag = max(h)−1 天,HLN)。
  预声明子族:long-form 格、event-driven 格、全 69 格,共 3 个 omnibus p 值,Holm(3)。
- Secondary(仅报告):对参考集合做 SPA/MCS 一次。
- 功效校准:用现有 signal-injection 管线,在 {0.1, 0.2, 0.3, 0.5, 1.0}% firm-orthogonal
  注入网格上估计该 omnibus 的检出率,报告 80% 功效对应的 MDE。
- 判定语言(预声明):omnibus 不拒绝且 MDE ≤ 0.3% →"经功效背书的界";
  omnibus 拒绝 → 与 detectable≠attributable≠bankable 三分法一致地写入
  (检出的是跨格系统性微增量,归因与可实现性不变);功效不足 → 如实报 MDE,不升级措辞。

## 与在跑 HPO 的边界

T1c 种子重训与单次 test 评估(tuning-artefact 句的更正)由 `configs/hpo_arm.yaml`
(tag `hpo-prereg-v1.0`)管辖,不属本文件;本文件不新增任何对 test split 的读取——
上述所有检验只触碰既有 predictions.parquet 的既有 test 预测,不训练任何新模型(B1 除外,
B1 只生成新的 test 预测、走与 C6 完全相同的既定协议,不做任何按 test 表现的选择)。


## v1.3 修订(2026-07-16,先于任何 B2 统计量):B2 = 第四跨家族探针(Gemma)+ 70B 零内容探针

**动机(审稿人原话,R11/R12 全票收敛)**:ml "a second healthy non-Qwen cross-family probe …
with the forecaster-health screen and the replication decision rule pre-declared — this is the
single experiment that moves the residual from 'directional' to real or kills it, and either
outcome raises my P";skeptic "would remove my strongest live objection and add ~0.10 to my P";
R12 major #2(头条点估计单种子单家族)。

**B2 设计**:机制逐字复用 committed `crossfamily_mistral24.py`(3 个复现性抖动种子 + 算术
集成、ED 面板、同一 excerpts、同一 prompt 逐字、G1''/G1q/G5/G3'' 机器精度门),仅换模型:
**Gemma-3-27B-it,bf16,TP=2**(Google 家族,matched-class 对 Qwen3-32B,非量化——顺带
消解 70B 的 int4 混杂之一)。权重经 ModelScope(box 无 HF egress 的既有惯例)。
**降级序列(仅在硬失败时推进,绝不因结果推进)**:下载不可得或试点健康失败 → GLM-4-32B
(Zhipu 家族)同协议;至多两次试点,全部试点无论生死都进论文的 Stress Tests 叙述。

**试点健康门(先于全量,预登记)**:val 切片 ~2,000 文档,种子 2026 单遍;健康公式 =
committed Yi/Phi 判据逐字:**max 方差单位 QLIKE < 4 且 max modal share(round(pred,2))<
60%**(pred_sd、R² 作诊断列照报)。试点失败 → 该模型 instrument-dead-at-pilot,如实报告,
不产生推断,探针分母句更新为"of five probes"并标注 pilot-gated;试点通过 → 全量放行
(3 种子 × 39,322 ED 文档 + 集成)。

**复制判定规则(预声明,ml 原话要求;全部分支进论文)**:健康(全量同公式)为前置;
- **(a) Holm-robust 复制** ⇔ vs firm-identity 参考,≥2/3 horizon 满足 clustered DM<0 且
  Holm(3)<.05 → 残差措辞升级为 "family-robust(两个健康家族 Holm 显著)",摘要删
  "only partly family-robust";
- **(b) 方向性复制** ⇔ 3/3 DM<0 但 Holm 达标 <2 → 与 llama70 同级,"三个健康探针同号"
  入正文,Holm-稳健性措辞不变;
- **(c) 不复制** ⇔ 健康但非 (a)(b) → 残差在摘要+06+07 降级为 "Qwen-conditional"
  (修复即减分,承诺执行);
- **(d) instrument-dead** ⇔ 健康公式失败(试点或全量)→ 照 Mistral 先例入表,无推断。
读数与 B1 同式:rel% vs firmID(主)与 vs HAR、逐 h clustered DM、每参考 Holm(3)、
STRONG/WEAK/NONE 逐字沿 committed 判定公式;集成为主基,单种子作 robustness。
**产物**:`results/tables/crossfamily_gemma27.{csv,md}`(降级则按实际模型命名),
write-once 单发。

**搭车:70B 零内容探针(quant 原话:"date+ticker term inside its reference, reconciling
the Table 6 149/103% probe cell with the replication claim")**:对已 committed 的
llama70-ens 面板,以同一 date+ticker 零内容 prompt(与 C6 contamination 臂逐字同模板)
过 llama70(int4,与被对照的 committed 运行同精度以保内部一致),单种子;读数 =
probe rel% 与 fulltext rel% 并列 + text-beyond-identity(f_datefirm 联合参考下 fulltext
是否仍加)——**描述性,无分支**,供 Table 6 探针格与复制主张的和解句使用。
产物:`results/tables/crossfamily_llama70_probe.{csv,md}`,单发。

**边界**:不重训任何模型;prompt 零改动;C6/llama70 的 committed 读数不重算(锚定门
照 G1'' 惯例);时间戳 = 本 tag(建议同步 OSF 存证)。
