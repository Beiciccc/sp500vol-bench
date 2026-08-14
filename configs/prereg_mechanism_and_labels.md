# 预注册:C 机制检验(捷径大小可预测)+ D range-based 标签稳健性(prereg-cd-v1.0)

日期:2026-07-15。本文件先于两项分析的任何统计量提交并打 tag。承诺(家族惯例):
**全部分支结果无论方向进论文;不得以结果为条件调口径;单发纪律;修订须先于对应统计量并记录。**

## C — 机制检验:身份控制的符号与大小是"基线实体编码度"的函数

**主张升级对象**:现稿 "the shortcut's size is a property of the panel and its baseline, not a
constant" 是事后描述;C 把它变成**带被预测量的诊断**。

**被预测量 y(每 cell)**:零文本身份控制对参考的相对增益
y = 100·[QLIKE(f_R) − QLIKE(f_Re)]/QLIKE(f_R)(f_Re = 参考 + 实体均值项;正 = 控制帮助参考)。
全部取自**已冻结产物**:SEC 6 cells(2 渠道 × 3 horizons,firm_identity_ensemble 的
zero-text-vs-f_R 行)、Yelp 2 cells(chronological,yelp_cascade 的 entity-mean 行,负值)、
MAEC 8 cells(4 horizons × 2 参考,protocol json 的 row4/f_Re 读数)= **16 点**。

**预测量 x(每 cell,唯一新计算)**:基线预测的实体编码度
x = R²[在 val 上把 f_R 的 log 预测对实体哑元回归](panel 内同 horizon 同参考)。
禁止触碰 test 标签;x 只用 val 预测与实体 id。

**预声明统计量**:(i) 16 点的 Spearman 秩相关 ρ(x, y),预测 **ρ < 0**(基线编码实体越多,
控制越无益乃至有害);置换 p(cell 标签置换 10,000 次,seed 2026);(ii) 符号检验:x 高于中位的
cells 中 y≤0 的比例 vs x 低于中位的(Fisher 精确);(iii) 披露:panel 聚类(16 点来自 3 个
panel,点非独立——置换按 panel 内置换 + 全局置换双报,措辞取保守者)。
**证伪线**:ρ ≥ 0 或双置换 p 均 >.10 → 机制主张不成立,论文保留描述性措辞,不升级;
此结果同样入 FACTS 与正文(一句诚实披露)。
**成立线**:ρ < 0 且保守置换 p < .05 → Discussion 段升级 + 一张 16 点散点(或紧凑表),
措辞上限:"the audit's identity term is a *predictable* correction: its sign and size track how
much of the entity the baseline already encodes"。**不得**声称因果或普适(ac 预埋的反驳
——"3 panels = 3 点太薄"——以 16 cells + panel-聚类置换正面应对,并在 Limitations 补一句)。

## D — range-based RV 标签稳健性(Parkinson / Garman–Klass)

**数据**:`/Volumes/Z/sp500vol-data/market/full_ohlcv.parquet`(已核验 = CRSP DlyHigh/DlyLow,
面板窗口覆盖 100%)。标签:前瞻窗内 Parkinson σ̂²_P = (1/(4n·ln2))Σ ln(H/L)²、
Garman–Klass 标准式,annualise 同现行惯例(√(252/H)·RMS 形式对齐 volatility.py);
**Parkinson 为 primary,GK 同表**。price 侧一致性:A-block 参考(A2 HAR 等)的过去-RV 特征
**同步换算为同一估计量**(特征窗全部止于 filing 前,现行防泄漏审计照跑);文本臂**不重训**
——其预测为标签无关的冻结产物,combiner/recalibration 在 val 上按新标签重拟合(log 空间
重校准吸收尺度差);此"预测冻结、标签更换"设计**作为首要 Limitation 披露**(文本臂曾按
close-to-close 目标优化,读数对文本侧偏保守)。

**范围**:69-cell 网格全链条重算(primary → firm-identity → maximal pool → conjunction),
day-clustered DM + 各族 Holm + placebo 门 + **逐格 MDE 与注入恢复率**(机制逐字复用现行
signal_injection 管线)。输出 `results/tables/rangebased_cascade.{csv,md}`,单发。

**分支承诺**:
- **(a) null 保持 + MDE 收缩**(预期):conjunction 仍 0/69 且 MDE 中位数显著低于现行
  0.82%(收缩 ≥30%)→ 措辞升级:"under a ~5× lower-variance label proxy the near-null
  persists with materially smaller MDEs——'selection device' 部分升级为 evidence of absence
  at the observed effect sizes";qf 的 rank-1(噪声代理)退役。
- **(b) null 保持 + MDE 不缩**:如实报告,限定句不变。
- **(c) 文本对身份参考 Holm 显著**:诚实反转——残差章节重写,摘要相应调整(不隐藏)。
- **(d) 参考侧排序大变**(HAR 不再强)→ 先查标签构建(G2 式量级门:Parkinson 与
  close-to-close 标签的秩相关须 >0.8,否则中止查错),不得带病出表。

**Sanity gates**:G1 现行标签重算复现 committed 38/69 机器精度(同码路);G2 新旧标签秩相关
>0.8(逐 horizon);G3 泄漏断言与现行审计同套;G4 placebo 照跑。

## 边界

C 只读已冻结预测与 val 标签;D 不触碰任何模型训练;两者与既有预注册互不越界。
