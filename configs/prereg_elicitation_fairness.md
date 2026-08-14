# 预注册:M2 — elicitation-protocol 公平性检验(prereg-ef-v1.0)

日期:2026-07-17。先于任何 M2 统计量提交并打 tag。承诺(家族惯例):全部分支无论方向进论文;
单发纪律;修订先于统计量;不以结果为条件调口径。

## 动机(审稿人原话,R14 area chair)

"若 5 个指令微调家族中 4 个在此 elicitation 下坍缩,最经济的解释是 **prompt 协议脆弱且是在
Qwen3-32B 上开发的**——筛子可能在选 **elicitation-protocol fit**,而它 fit 的那个家族正是
管线所构建者。C5x 控制了 curation,**不控制 elicitation-protocol tuning**。"(R14 ac, major #2)

本实验直接检验该混杂:**给每个已死家族公平的 prompt 适配机会**,看健康是否恢复。

## 设计

**臂集(2 家族)**:Mistral-Small-24B(bf16)与 Gemma-3-27B-it(bf16)。
**选择理由(先于统计量,只引已 committed 事实)**:二者均 bf16(与 primary 同精度,消除
llama70 的 int4 混杂)、均为 24–27B(matched-class 对 32B)、权重均已在箱;Yi-1.5-34B 与
Phi-4-14B 排除——它们的失败已由 committed 的 capability-floor 读数解释(Yi 方差 QLIKE
7.60–8.19 = 能力地板;Phi 14B 远小于 matched class),且 GPU 预算有限,如实披露。

**prompt 变体(K=3,内容先于任何运行冻结于本文件)**:
- **V0** = committed C6 prompt 逐字(基线,即现有 instrument-dead 读数所用);
- **V1 = few-shot**:V0 + 2 个**验证集**示例(定值:val 面板 canonical sort 的第 1、第 2 篇,
  连同其真实 RV 标签),示例对所有家族**逐字相同**;
- **V2 = format-hardened**:V0 + 显式数值域提示("annualised RV, typically 0.05--1.50")
  + JSON schema 重申 + 一次 re-ask on parse failure(与 committed retry 同机制)。
V1/V2 的文本在本 tag 中固定;**不得因任何运行结果修改**。家族间除 chat template 折叠
(Gemma 无 system role,沿 prereg-rfa v1.3 已披露的 fold)外逐字相同。

**选择规则(只读 val,永不读 test)**:每家族 × 每变体在 **2,000 份 val 文档**
(确定性:canonical sort 前 2,000,与 v1.3 pilot 同一集合)跑单遍,计算 committed 健康公式
(max 方差单位 QLIKE < 4 **且** max modal share(round(pred,2)) < 60%)。
**每家族选 val modal share 最低的变体**(并列取 V0 → V1 → V2 序)。该变体若**健康**
→ 放行全量(3 抖动种子 × 39,322 ED 文档 + 算术集成,与 B1/B2 逐字同协议);
若该家族三变体**全部不健康** → 该家族登记为 elicitation-robust instrument-dead,不进全量。
**test 只被健康放行的家族触碰一次。**

## 分支承诺(全部预登记,全部进论文)

- **(a) 适配后健康且 Holm-robust 复制**(≥2/3 horizon,vs firm-identity,clustered DM<0 且
  Holm(3)<.05)→ **AC 的混杂被证实**:此前的"仪器死"确有 prompt-protocol 成分;残差措辞
  在 06/07 升级为 family-robust(摘要不动,见 v1.1 修订);并如实写明该结论由 prompt 适配
  而非模型能力驱动。
- **(b) 适配后健康、方向性复制**(3/3 DM<0,Holm<2)→ 与 llama70 同级并表;
  "三个健康探针同号";Holm-稳健性措辞不变;混杂部分成立(健康可由 prompt 恢复)。
- **(c) 适配后健康但不复制** → **残差降级为 Qwen-conditional**(**06 + 07**,修复即减分,
  承诺执行);混杂成立且对本文不利——如实报告。
  【v1.1 修订(2026-07-17,先于任何 M2 统计量——pilot 未跑,无任何读数存在):后果的修改地点
  由 "摘要 + 06 + 07" 收窄为 "**06 + 07**"。**理由是外部死线,不是结果**:目标会场[脱敏]的摘要于
  2026-07-21 冻结,其后实质性改动按 Paper Modification Guidelines 可致 "rejected without
  review";M2 落地于 07-21 与全文截止 07-28 之间。为使 (c) 可执行而不违反冻结,摘要**此刻
  即预先承载最保守读数**——加入 "only partly family-robust"(该限定在 (a)(b)(c)(d) 全部分支
  下均为真:(a) 下仅属低估,(b)(c)(d) 下精确),故 (c) 的降级可完整落在 06/07 而摘要无须改动。
  对称地,**(a) 分支登记的"摘要删 only partly family-robust"同样收窄为不动摘要**——升级
  只在 06/07 表述。本修订不改变任何分支的判定条件与统计口径。】
- **(d) 三变体全不健康(两家族均)** → **AC 的混杂被证伪**:公平的 prompt 适配无法救活它们,
  故健康筛并非在选 elicitation-protocol fit,而是在量能力地板;正文获得对该质疑的直接答复,
  且 capability-floor 主张升格为"经 prompt 适配检验"。
- 混合(一家族 (a)/(b)/(c),另一 (d))→ 逐家族如实,措辞取**较保守**的那支。

## v1.1 修订(2026-07-18,先于任何 M2 统计量——pilot 未跑,无任何读数存在):硬件变更 TP=2 → TP=1

原计划的 box(2×A100-40G)不可用;实际 box 为 **单卡 A100-80GB**,故 **TP=2 → TP=1**(24B/27B
bf16 均装得下 80GB,无需张量并行)。这是**外部硬件约束,非结果驱动**;修订先于 pilot,无任何
统计量存在。三处随之调整:

1. **推理 TP=1**;每个产物、pilot json、哨兵、日志记录实际 TP(`tp_effective`)。
2. **G-E1 由复现闸降级为 TP 不变性诊断**(诚实理由):committed 的 crossfamily 读数产自
   TP=2,而 bf16 批式推理**非位确定**(论文已披露 repeat-decode 仅 94–97% 逐位相同),故
   V0@TP=1 本就不可能逐位复现 committed——即便 TP=2 亦然。因此 G-E1 改为:V0@TP=1 的 val
   健康列与 committed(TP=2)**并列报差**,脚本继续(不中止);**仅当某家族 healthy/dead
   判定在 TP=2-committed 与 TP=1-V0 间翻转时高声报警**——翻转即意味着 committed 的
   instrument-dead 判定带 TP 限定,须如实报告(对本文不利,但照报)。
3. **TP 混杂的预登记处置**:M2 的核心估计目标(每家族内 V0/V1/V2 对照,判「公平适配能否
   救活健康」)在 TP 常数下不受影响——TP 只影响与 committed 的锚定。分支判定与统计口径
   **不因 TP 改变**。论文如实披露 TP=2→1 及 TP 不变性诊断结果;若诊断 PASS(判定不翻转,
   预期,因 modal share 门 60% 远离 Mistral 87.9%/Gemma 71.4%),则附带证明 committed 的
   instrument-dead 判定非 TP 假象——对论文为正面。

本修订不改变任何变体文本(V0/V1/V2 哈希不变)、任何分支条件、任何健康公式、任何 val-only
纪律。

## Gates

- **G-E1**:V0 重跑对 committed 读数机器精度复现(同码路、同权重、同种子)——若不复现,
  本实验中止并报管线漂移(不得以新数覆盖旧读数)。
- **G-E2**:val-only 断言——pilot 阶段任何 test 行读取即致命错误(代码级断言 + 日志)。
- **G-E3**:变体文本哈希对本 tag 固定值一致(防运行中改 prompt)。
- **G-E4**:全量健康公式复核(与 v1.3 同,集成基)。

## 产物与边界

`results/tables/elicitation_fairness.{csv,md}`(单发守卫):每家族 × 每变体的 val 健康列、
选中变体、全量读数(若放行)、分支裁定。**不重训任何模型;不改 combiner;不触碰 C6/llama70
的 committed 读数**。GPU 预算:val pilot 6 × ~0.5h ≈ 3h;每放行家族全量 ≈ 1.5 箱日。
时间戳 = 本 tag(建议投稿前 OSF 存证,双盲版)。


## v1.2 记录(2026-07-18,**在 pilot 统计量产生之后**——故这是记录 + deviation 登记,不是修改)

pilot 单发已跑(6 格,2000 val 文档,TP=1)。执行暴露一处 **selection rule 缺陷**,经对抗核查
(2 镜头:hostile-reviewer + methodologist,两者均判「实质结论稳健、但脚本的字面分支标签
self-serving 须修正」)。本节记录裁决 + 全部披露,**结论未按有利方向择取**。

### 事实(逐格 val 健康,健康门 = max 方差QLIKE<4 且 max modal<60%)

- **Mistral24**:V0 dead(qlike 4.34,modal 57.7%)、V1 dead(qlike 49.8)、V2 dead(qlike 5.03,
  modal 34.2%)——**三变体全 dead,无 val 健康变体**。
- **Gemma27**:**V0 HEALTHY(qlike 3.69,modal 45.4%<60%)**、V1 dead(qlike 10.3)、V2 dead
  (qlike 6.03)——V0 健康,适配变体 V1/V2 dead。

### selection rule 缺陷(登记为 deviation)

规则「选 val modal 最低的变体;若健康→放行;三变体全不健康→instrument-dead」**对 Gemma 失效**:
modal 最低者是 V2(40.8%)但 V2 **QLIKE-dead**(6.03>4);唯一健康的 V0 **不**使 modal 最小。
「最低 modal」与「健康」分道,故字面规则**返回未定义 gap**——子句1 选中 dead 的 V2、子句2 的
「三变体全不健康」前提**为假**(V0 健康)。脚本触发的字面 (d)(两家族全不健康)**对 Gemma
事实错误**,不得采用。

### 裁决:purposive/mixed 读法(prereg 的「mixed」子句管辖:逐家族如实,取较保守措辞)

估计目标是「**公平适配(V1 few-shot / V2 format-hardened)能否恢复健康**」,非「是否存在任一
val 健康变体」。逐家族:
- **Mistral24 = 真字面 (d)**:无 val 健康变体,适配无效。
- **Gemma27 = 非字面 (d)**:有 val 健康变体 V0,但——(i) **V0 = committed C6 原 prompt**,
  其 committed 全量跑(crossfamily_gemma27)在 **test 坍缩 71.4%(instrument-dead)**;
  val 健康**从来不是** instrument-dead 判据(该判定是全量/test 属性),故 val 健康的 V0
  与 committed 的 instrument-dead 判定**从不矛盾**;(ii) **适配 V1/V2 比 V0 更糟**
  (qlike 10.3/6.03 vs 3.69)——适配是**伤害**不是帮助。
- **联合结论**(每个诚实读法下都成立):公平的 prompt 适配无法恢复任一家族的全量健康 →
  健康筛量的是**能力地板**,非 Qwen-elicitation-protocol fit → **R14 AC major #2 证伪**。

### deviation 登记(两项)

1. **Gemma V0 未按 registered 3-seed 协议重跑全量**:接受 committed crossfamily_gemma27
   (单跑 / TP=2,test 71.4% dead)代替。理由:V0 = committed prompt 逐字,重跑只会得同样的
   test 坍缩,且 pilot 已证 V1/V2 val-dead(无新变体够格放行);零新 test 预算(无 val/test
   double-dip)。
2. **selection「最低 modal」判据的缺陷**如上,登记;不改变结论(两家族适配变体均 dead)。

### 强制披露(核查要求,进 md 与论文正文,逐字)

(a) Gemma V0 val-HEALTHY 逐字(qlike 3.69,modal 45.4%);**绝不写/暗示「所有 Gemma 变体
val 不健康」**;(b) selection 缺陷(最低 modal V2 与健康 V0 分道);(c) 路径是 purposive/mixed,
非字面 all-dead;(d) instrument-dead 是 committed 全量/test 属性,val 健康 V0 不与之矛盾;
(e) 适配伤害不帮助(V1 10.3 / V2 6.03 vs V0 3.69);(f) Mistral 真字面 (d);(g) TP 诊断零翻转
(Mistral Δmodal 0.05%、Gemma 0.20% → committed 判定非 TP 假象);(h) Gemma V0 deviation
如上。

### TP 不变性诊断(v1.1 登记)

V0@TP1 vs committed@TP2 判定**零翻转**:Mistral(dead↔dead,Δqlike 0.011、Δmodal 0.05%)、
Gemma(healthy↔healthy,Δqlike 0.028、Δmodal 0.20%)。→ committed 的 instrument-dead 判定
**在 TP=2→TP=1 下稳定,非 TP 假象**;box2/TP=1 选择被验证无害(附带正面证据)。
