# 预注册:M1 — Kogan 原语料上的级联审计(prereg-kc-v1.0)

日期:2026-07-17。先于任何 M1 统计量提交并打 tag。承诺(家族惯例):全部分支无论方向进论文;
单发纪律;修订先于统计量。本实验**不使用** /Volumes/Z 上的任何数据,不占 GPU(CPU-only),
与 prereg-ef(M2,占箱)并行。

## 动机(审稿人原话,R14 area chair,major #1)

"唯一能打动我的路线是把协议从'我们的实例 + 2 个演示'扩为 **把审计施加到领域已发表结果的规模**
——在几个有公开代码的已发表 disclosure-NLP 结果上跑级联并报存活率。"

## 重要的范围更正(先于统计量,基于可核查的事实)

**AC 字面要求的"N 个独立已发表结果"在本领域不存在。** 侦察(2026-07-17,逐仓核查)确认:
disclosure-text→波动率的**公开证据基座只有三个语料**——(i) Kogan/FIN10K 10-K 语料、
(ii) MDRM/EC 财报电话语料、(iii) MAEC。HTML、NumHTML、VolTAGE、KeFVP、ECHO-GL **全部**
构建于 MDRM 之上(各仓 README 自述),故"审计 N 个模型"= N 个模型跑在 1 个语料上 =
**伪重复**,其存活率的分母是假的。
**因此本文改以普查(census)而非抽样立论**:三个语料中,能合法取得数据者全部跑完级联;
不能取得者公开披露理由。**MDRM 已由 `prereg_maec_audit.md` §9 裁定 cite-only**(文本与音频
捆绑分卷、无 licence,不再分发)——该裁决在先,不因本实验改变。MAEC 已审计(FACTS §13g)。
**本实验补上第三个:Kogan 语料。**

## 与现有 Kogan 段的关系(关键区分,防自我混淆)

committed `kogan_dissolve.md` 是**把 Kogan 的 evaluation DESIGN 移植到我们的面板**;FACTS 已
绑定其诚实读法("the published-style design yields no transferable positive on a modern panel";
**不得**声称复现了已发表正结果)。**本实验相反:在 Kogan 自己的语料上跑我们的级联**——
若成功,它才是"复现已发表正结果、再由协议重新定价"的真外部审计(与 MAEC 同型)。
两者并存:前者答"design 是否转移到现代面板",后者答"协议施于原语料的已发表结果时发生什么"。

## 数据(全部公开,实测可达 2026-07-17)

Kogan et al. (2009) 10-K 语料 `http://www.cs.cmu.edu/~ark/10K/`:每年(1996–2006)提供
`meta.txt`(key | filing date yyyymmdd | EDGAR URL | company | **CIK**)、`tok.tgz`(分词文本)、
`logvol.+12.txt`(**前向 12 月 log 波动率 = 标签**)、`logvol.-12.txt`(**过去 12 月 log 波动率
= 重校准所需的价格基线**)。无 licence 条款,仅要求引用。**自给自足:不需 WRDS/CRSP/GPU。**
G-K0:六文件 SHA-256 记入产物;行数与 key 空间一致性断言。

## 级联(逐级,机制复用 committed `maec_protocol.py` 的同型阶梯)

- **L0 published convention**:Kogan 的口径——文本特征(TF-IDF)+ `logvol.-12` 作为控制,
  对 `logvol.+12` 回归,**朴素 obs 级推断**,他们的年度 OOS 划分(train ≤ y,test = y+1)。
  读数 = 文本臂 vs 仅 `logvol.-12` 臂的 MSE 改善率(他们报的量)。
- **L1 recalibrated baseline**:基线改为**重校准**的 `logvol.-12`(OLS 截距+斜率,
  在训练年拟合、测试年冻结)——本文协议的第一条实践。
- **L2 firm-identity reference**:参考再加**同一 CIK 的训练期均值 log 波动率**(零文本项)。
- **L3 clustered inference**:按 **filing date** 聚类(冲击共享单元),HAC + HLN,取代朴素 obs-t。
- **L4 Holm(预声明家族)**:L3 的逐年 p 值族内 Holm。
- **L5 conjunction**:L1∧L2∧L4 同时满足方为存活。
- **placebo**:标签置换(5 seeds),|DM|<2 为闸,与主协议同式。

## 分支承诺(全部预登记,全部进论文)

- **(a) 复现后溶解**(预期):L0 复现已发表量级的正向文本效应,L1–L5 逐级消解至不存活
  → 普查主张成立:"三个语料中我们能取得数据的全部跑完;apparent gain 复现 k/k,存活 0/k";
  07 的现有 Kogan 段**原地替换**为本读数(页面自筹)。
- **(b) 文本在 Kogan 语料上存活全级联** → **协议认证了一个真实的已发表正结果**:这正是
  R11/R14 反复要求的 **real-world positive control**(证明协议不只会杀);如实报告并据此
  软化"near-null"的普遍性措辞——`shortcut 的大小是面板与基线的属性,不是常数`
  (FACTS §11/§13g 既有框架直接吸收);**对本文是好消息,照写**。
- **(c) L0 无法复现已发表正结果**(如因未用其精确 SVR 超参/特征口径)→ 报告复现失败本身,
  不进一步推断;普查表中该语料标 "published reading not reproduced",理由公开。
- **(d) 数据取得失败**(链接失效等)→ 如实登记为未执行,普查降为 2/3 语料。

## Gates

- **G-K1**:L0 读数与 Kogan et al. (2009) 论文所报数量级一致性检查(同符号、同数量级);
  不一致即触发分支 (c),**不得**调参使其一致(单发纪律)。
- **G-K2**:无 look-ahead 断言——L1 起所有拟合只用训练年;L0 的朴素口径按其原样复现并标注
  (与 kogan_dissolve.md 的 look-ahead 披露惯例一致)。
- **G-K3**:CIK 覆盖与跨年重现率报告(firm-identity 参考的前提)。

## 产物

`results/tables/kogan_corpus_audit.{csv,md}`(单发守卫)+ `scripts/analysis/kogan_corpus_audit.py`
+ 抓取脚本(数据不再分发,仅管线)。CPU-only,预计 2–3 小时机时。
时间戳 = 本 tag(建议同步 OSF 存证)。


## v1.1 记录(2026-07-17,**在统计量产生之后**——故这是记录,不是修改)

单发已开火(`kogan_corpus_audit.{csv,md}`)。以下两处 prereg 缺陷由执行暴露,**两处的两种读法
均被无条件报告,无一按结果择取**;本节记录裁决及其理由,供审稿人核验。

**(1) split 归属误述(prereg 的事实错误,规则本身无歧义)**。本文件称 `train ≤ y, test = y+1`
为"**他们的**年度 OOS 划分"——**不是他们的**。Kogan et al. §6 明文用**5 年滚动窗口**、
test 2001–2006(其 Table 4 变动窗长为 1/2/5 年,**从不**用扩张窗)。prereg 的*规则*可执行且
无歧义,错的只是*归属*。若只跑 prereg 规则,G-K1 无法作答(扩张窗读数对不上他们的滚动窗
发表数);若只跑发表惯例,则违反已 tag 的 prereg。**故两臂在任何统计量之前即在脚本中声明并
双双无条件报告**;G-K1 的比较对象取 `L0_pub`(唯一与其 Table 2 可通约者)。

**(2) L2 自含性沉默(决定分支的歧义)**。prereg 只说"同一 CIK 的**训练期均值**",未言明训练行
自身的标签可否进入其自己的 CIK 均值。**裁决:`loo`(留一)为 primary**,理由是结构性的、
非结果驱动的:
- prereg 规定 L2 **强化**参考(它是文本必须击败的 firm-identity 控制)。`incl` 下拟合系数被
  推向 1.0(特征**部分即标签**),参考过拟合,其 **test MSE 10/10 差于 L1** ——该 rung **弱化**
  了参考,机械**膨胀**文本增益并可凭空制造 survival。弱化参考的 rung 不可能是 prereg 描述的
  那个 rung。
- committed 模板同向:`maec_protocol.py` 的实体均值控制(STPEV)是 `shift(1)` 的 PIT 扩张
  先验均值——当前行标签按构造排除;自含固定均值在该模板中本就被降级为 robustness。
- 铁证:test 1997(train=1996 单年)中 99.1% 训练行是其 CIK 唯一 filing,"firm mean" 即该行
  标签,拟合 β = **+1.000**。
- **诚实声明**:`incl` 给 **(a)**(文本溶解,**对本文有利**),`loo` 给 **(b)**(文本存活,
  **对本文不利**)。裁决取对本文不利者;`incl` 的全部读数照登于产物,**但任何 incl 的 L2
  数字不得在正文引用**。

**(3) G-K1 的量级读法(如实标注,不改门)**。门写"同符号、同数量级"。实测 L0_pub **+9.54%**
vs 发表 **+1.21%**:同符号,比值 **7.91×**,在"≤10×"操作化内但**不宽裕**;若按更严的
"同一个十的幂"读法则不过门、触发 (c)。**两种读法均在产物中写明**;判 PASS 的支持证据是
**逐年结构一致**(6 个 test 年中 4 年同号,2001 最差、2004 最佳在两边一致——其 Sarbanes-Oxley
模式),即同一效应在不同量级,而非另一个效应。

**(4) placebo 的可移植性边界(方法学发现,须进正文限定)**。Kogan 的 **joint-arm** 惯例
(文本与波动率控制同回归)使"置换文本"**不会**令该臂坍缩到参考(与 maec 的 combiner 惯例不同),
故移植来的 |DM|<2 闸在此设计下混淆了**文本信号**与**函数形式差异**:50 次抽样中 **13 次**
置换后的文本仍显著击败参考。因此 6/10 的未过闸存活里,**1997 与 2005 不可归因于文本**,
1998/2000/2003 清楚可归因 → **3/10 为 placebo-gated 存活**。正文引用 (b) 时必须同时给出
该限定与 3/10 这个数,不得写成"文本在 Kogan 语料上存活"。
