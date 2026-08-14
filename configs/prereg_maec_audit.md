# 预注册:MAEC 财报电话会波动率基准的量化审计(实验 A)

日期:2026-07-15。状态:**FROZEN(prereg-maec-v1.0)**。全部 9 项 OPEN 由 lead 于
2026-07-15 裁决完毕(裁决内联于各节,原【OPEN】编号保留以便追溯);本文件先于任何
MAEC 标签构建、任何价格统计量、任何模型运行提交并打 tag。
截至本稿,仅读取过 `maec_manifest.parquet` 的元数据(按日期/字符数的行计数,
见 §3 披露)与 `ACQUISITION_NOTES.md`;transcripts、价格、任何标签均未被触碰。

承诺(与 prereg-rfa-v1.2 同款):**所有分支的结果无论方向一律进论文;不得以结果
为条件选择口径、单位、族结构、split 或臂。** 修订须在对应统计量计算之前落文件、
注明日期与原因(修订记录格式沿用 prereg-rfa G5 先例)。

## 0. 定位、先行工作与禁述条款(reframe mandate,必读)

**先行工作(动机,非对手)**:Yu, Liu & He, "Same Company, Same Signal"
(Findings of ACL 2025, arXiv:2412.18029;建议 bib key `yu2025samecompany`,按
FACTS §9 规则以 `% NEWBIB:` 申报)已在 EC(MDRM)/MAEC-15/16 上证明:同 ticker 的
历史财报后波动率均值("STPEV")在表示相似性层面追平所有文本模型(其 MSE 0.257
vs Gemini 0.258)。**他们缺**:(a) 任何重校准的 past-vol 参考(其原始 V_past
MSE 1.12,被弃用);(b) 组合问题(文本在校准后的价格/历史基线之上是否有增量);
(c) 任何显著性检验(无 p 值、无聚类——而 call 日期高度聚簇);(d) bracketing/
归因阶梯;(e) 经济显著性。

**本实验的唯一定位**:对 Yu et al. 的表示层发现做**带正式推断的量化重定价**
(quantified repricing with formal inference)——把本文的协议(log-space 组合器
val 拟合 test 冻结、重校准 AR 参考、entity-mean 控制、call-date-clustered DM、
placebo、MDE 功效校准)第三次移植到新领域(第一次 SEC 8-K,第二次 Yelp),
STPEV 作为**阶梯内的一臂**被纳入并定价。

**禁述条款(binding on the prose)**:
1. **不得**把 A 卖成 discovery——论文不得声称首次发现财报电话会基准中的
   identity 效应;Yu et al. 的优先权在 A 的所有文字里以专段引用作为动机。
2. **不得**复活 split 规则对比(entity-disjoint / group-wise CV):Action-A 已被
   本项目自己的预注册 falsifier 证伪并回滚(FACTS §12,DO-NOT-REVIVE)。
   entity-disjointness 问题**整体超出本实验范围**。
3. **不得**写跨领域 "the same shortcut"(2026-07-14 已撤回);唯一许可框架:
   **"shortcut 的大小是面板及其基线的属性,不是常数"**(Yelp 先例)。
4. MDRM 仅引用(结构性阻断:5 分卷 zip 捆绑音频,见 §9);不得对 MDRM 做任何
   数值主张。不得声称与 Yu et al. 的数字直接可比(其面板为 MAEC-15/16 子集,
   我们用完整 3,443-call 发布版;见 G2 的量级门)。

## 1. 冻结输入(已在盘,integrity-checked,本文件冻结时未被任何标签/统计读取)

- 转录文本:`/Volumes/Z/second-domain/earnings_calls/MAEC/MAEC_Dataset/`
  (3,443 个 `YYYYMMDD_TICKER` 目录,各含 `text.txt`;clone commit
  `65a109f5b1a8cb4c96e8337b749ce3db41f2c210`;license CC BY-SA 4.0,副本在
  `LICENSE_MAEC_CC-BY-SA-4.0.txt`;引用 Li et al., CIKM 2020)。
- Manifest:`/Volumes/Z/second-domain/earnings_calls/maec_manifest.parquet`
  (3,443 行;call_id, ticker, call_date, n_chars, n_sentences, path;
  1,213 tickers;2015-02-25..2018-06-21;(ticker, call_date) 无重复)。
- 价格(700 个非 S&P500 tickers):
  `/Volumes/Z/second-domain/earnings_calls/crsp_sp1500_daily_2014_2019.parquet`
  (958,071 行,706 (ticker, permno) 对,OHLC+DlyRet+facpr,2014-01-02..2019-06-28,
  union 交易日历 1,382 天)。
- 价格(其余 513 tickers):既有 `/Volumes/Z/sp500vol-data/market/full_ohlcv.parquet`。
- Ticker→PERMNO 点时映射:`ticker_permno_map.parquet`(700/700 解析,0 未解析;
  6 个 ambiguous tickers 已标旗,处理规则见 §3.3)。
- 覆盖表:`maec_price_coverage_by_ticker.csv`(44/700 tickers 窗口内覆盖 <90%,
  均为期中 IPO/退市,预期内,见 §9)。

## 2. 任务定义(标签、估计量、horizon、对齐)

### 2.1 估计量(MAEC/MDRM 自有的 Eq.-1 惯例,被审计对象的口径)

对 call i(ticker→permno p,call 日 τ),取 CRSP 总回报 r_t = DlyRet(p, t)
(**简单收益、含股利与拆股**——facpr/除权已内建于 DlyRet,不需再从 close 价
自行调整;这与被审计基准的 adjusted-price 惯例一致)。n 日已实现波动率标签:

    v_[a,b] = ln( sqrt( (1/n) · Σ_{t∈window} (r_t − r̄_window)² ) )

即**对数日波动率**(demeaned RMS of daily returns)。标签、预测、损失全部在
v(log-vol)单位——被审计文献的 MSE(其 0.257 / 1.12)正是此单位;log-space
组合器在此单位下退化为对 v 的线性 OLS(与 yelp_protocol 的 exp/log 移植代数
恒等,更简单)。稳健性附注(零成本、仅并列报告):r_t = ln(1+DlyRet) 的
log-return 变体。

### 2.2 Horizons

n ∈ {3, 7, 15, 30} 个交易日(MDRM/MAEC 系列的公开惯例族)。四个 horizon 全部
进入 §6 的 Holm 族;**审计已发表主张的臂以这四个公开惯例 horizon 为 primary**。
【OPEN-8 已裁决:保留族内】n=30 留在 Holm 族(不做事后剔除);其 HAC lag/簇数比
在冻结时披露,且 n=30 cell 的确证读数为 §6.4 的日期块 bootstrap CI(预声明)。

### 2.3 日期对齐(MAEC 无时间戳——day-0 歧义,预声明)

MAEC 不含 call 的 before/afterMarket 时间戳。Yu et al. 报告 64–69% 的 call 在
盘前。记 t_0 = 最后一个 ≤ call_date 的交易日,t_1 = 第一个 > call_date 的交易日。

- **PRIMARY 对齐**:标签窗 = {t_1, …, t_n}(严格 call 后,任何盘前/盘后场景下
  都无 day-0 泄漏;代价:对盘前多数派少算一天反应日)。
- **SENSITIVITY 臂(整体平移一天)**:标签窗 = {t_0, …, t_{n−1}}(day-0 计入,
  匹配盘前多数派)。此臂**零 GPU**:所有文本臂预测不变,仅标签与组合器 val
  重拟合变;完整跑一遍 §5 阶梯,只报告判定档位是否改变。
- 若 primary 与 sensitivity 的判定档位(§8)不一致:**primary 管辖论文措辞**,
  sensitivity 差异如实披露并限定主张。【OPEN-2 已裁决:维持严格 call 后窗为 primary】
  零泄漏是本协议的 DNA(SEC 侧 effective-trading-day 纪律的移植)。
  【v1.1 依 OPEN-1 核验更正】公开惯例对齐即 day-1-start(未调整),与 PRIMARY 同向;
  day-0-inclusive 仅保留为平移 sensitivity 臂。

### 2.4 past-vol 特征(参考臂的原料)

与标签同一估计量,窗口全部**止于 t_0**(primary 对齐)或 t_0−1(sensitivity
对齐,避免与 day-0 标签窗重叠):
- V_past^(n):匹配窗 {t_0−n+1, …, t_0}——即 Yu et al. 弃用的那个 raw 基线;
- HAR 式三窗:V_past^(5), V_past^(22), V_past^(66)。

## 3. 样本构建与排除规则(全部先于标签构建定死)

以下计数来自 manifest 元数据与 ACQUISITION_NOTES(引用为既得事实,非新统计):

### 3.1 stub 转录排除

n_chars < 100 的 32 条(多为 11 字符空壳)**从所有臂一律排除**(文本臂与非文本
臂同删,保证各臂行集恒等),per-split 计数在 build 脚本输出中披露(按 §4 的
split,test 内为 2 条)。< 500 字符共 95 条:第 100–500 区间**保留**,其计数
(63)作为 Limitations 披露。【OPEN-9 已裁决:100】排除规则取最小(n=32);100–500 字符段保留、计数披露,
内容质量差异由臂自己消化。

### 3.2 价格覆盖排除(每 horizon 独立判定,计数逐项披露)

- 标签窗必须 100% 完整(n 个交易日的 DlyRet 全在);任一缺失 → 该 (call, n)
  行删除,计数披露。permno 在标签窗内退市(CRSP 末日落窗内)→ 删除并计数
  (不做 delisting-return 插补,作为 Limitations 披露)。
- past 窗:V_past^(n) 匹配窗要求 ≥80% 天数在场,按在场天数计算;不足 → 该
  (call, n) 行删除。HAR 三窗同规则。
- 每 horizon 通过全部闸门的行数、以及 44 个低覆盖 tickers 贡献的删除数,
  进 build 报告。

### 3.3 ambiguous tickers(6 个,permno-keyed,预声明规则)

- 双股类(GEF 83233/83264、HVT 10294/41217、WSO 46068/66376):每 call 取
  **窗口内 median 日成交额(|DlyClose|×DlyVol)较高的股类**,规则在 build 时
  执行一次、选择结果连同两条线的成交额中位数一并披露。【OPEN-11 已裁决:确认成交额中位数 tie-break】选择结果与两股类中位数一并披露。
- 窗口内 ticker 复用(ENR、FLOW、TIVO):按 call_date ∈
  [SecInfoStartDt, SecInfoEndDt] 点时消歧;落双窗或零窗 → 该 call 删除并计数。
- 断言:消歧后每 call 恰好一个 permno(G5)。

### 3.4 键与合并纪律

行键 = (permno, call_date, horizon);(ticker, call_date) 已验证无重复。所有臂
inner-merge 于同一行集;merge 后 label 逐行断言一致(yelp_protocol 同款)。

## 4. 切分(dates pinned;entity-disjointness 超范围,见 §0-2)

**PRIMARY = 按 call_date 时间切分,70/10/20(按 call 计数),边界日期钉死**:
- train:call_date ≤ **2017-02-23**(2,436 calls)
- val:2017-02-24 .. **2017-05-09**(333 calls)
- test:2017-05-10 .. 2018-06-21(674 calls,143 个不同 call 日,463 tickers)
边界由 manifest 行计数的 70%/80% 分位日期得出(见 §0 披露;标签未触碰)。
断言:max(train date) < min(val date) < min(test date);val ≥ 100 行、
test ≥ 30 行的下限(MIN_VAL/MIN_TEST,Yelp 同款)每 horizon 检查。
边界重叠披露:val 末端 h−1 个交易日内的 call 其结果窗跨入 test 期——计数
报告,`--embargo-val` 作为 robustness 杆(yelp_protocol 同款)。

**published-convention 臂(只为复现被审计的公开式读数)【v1.1 修订,依 OPEN-1 文献核验】**:
split = MAEC(CIKM 2020)Table 5 的**按年三面板、面板内 chronological 7:1:2**
(2015:train ≤2015-10-22 / val ≤2015-10-28 / test ≤2015-12-17;2016:≤2016-08-03 /
≤2016-08-12 / ≤2016-11-15;2017–18:≤2017-11-07 / ≤2018-02-15 / ≤2018-06-21;
逐年独立拟合,与原文"different models for different years"一致);对齐 = day-1-start
未调整(Yu et al. Table 3 脚注明言其打分未调整 beforeAfterMarket)。读数按公开式:
raw V_past^(n)(不重校准)vs 各文本臂 standalone,MSE(v),无聚类推断。
MDRM 系的 80/20 切分不采用(MDRM cite-only)。
【OPEN-1 核验完成,2026-07-15(v1.1)】核验结论:公开惯例 = 按年三面板 7:1:2
(上方已按 Table 5 钉死);公开对齐 = **day-1-start 未调整**(与本预注册 PRIMARY 同向,
故 §2.3 中"day-0-inclusive 归属 published-convention 臂"一句作废,day-0-inclusive 仅
保留为平移 sensitivity);标签公式/horizons/单位与 §2.1–2.2 相符;"days"歧义
(MDRM 写 calendar days、MAEC 未指明、Yu et al. 按交易日)按交易日执行并披露。
primary split/对齐不动。修订发生在该臂任何打分之前(panel 构建亦未触及该臂 split)。

## 5. 臂阶梯(每臂 vs 重校准 past-vol 参考;组合权重一律 val 拟合、test 冻结)

组合器 = 对 v 的 OLS(§2.1;= yelp_protocol `log_ols_frozen` 的恒等移植),
预测裁剪到 σ_daily ∈ [1e-4, 1.0] 对应的 v ∈ [ln 1e-4, 0]。

**参考(两个,同入 Holm 族,mirror prereg-rfa 的双参考先例)**:
- **R-AR(匹配窗)**:f_R = OLS[1, V_past^(n)]——被审计文献弃用的 raw 基线的
  重校准版,审计叙事的主参考("他们扔掉的基线,校准后值多少");
- **R-HAR**:f_R' = OLS[1, V_past^(5), V_past^(22), V_past^(66)]——本文
  reference-interval 纪律的强参考端。
【OPEN-3 已裁决:R-AR 为 headline 参考】审计叙事优先("他们弃用的基线,校准后
值多少");R-HAR 为族内保守端,永远同表并列。

**STPEV entity-mean 控制(阶梯内臂,Yu et al. 的对象)**:
STPEV_i(τ) = 该 ticker **在 τ 时点已实现**的历史 call 标签均值(仅计
τ′+n 个交易日 ≤ τ 的先前 call,即标签窗已完整落在 τ 之前;点时、expanding)。
无先前 call 的行 → 回退 train+val 全局均值;覆盖率逐 horizon 披露(manifest
计数给出下界:test 有 75/674 行其 ticker 在 train+val 内无任何 call,11.1%;
expanding 口径还会计入 test 期内更早的已实现 call,实际覆盖以 build 披露为准)。控制参考:f_Re = OLS[1, V_past·, STPEV]
(对两参考各一版);零文本 STPEV-only 行同表报告(descriptive)。
【OPEN-4 已裁决:点时 expanding 为 primary】(亦更贴近 Yu et al. 的 STPEV 原义);
Yelp-port 固定均值口径为 robustness 行,两者都算。

**文本臂**:
1. **TF-IDF ridge**(fitted):word 1–2 gram,min_df=5,max_features 50,000,
   sublinear-tf;ridge α ∈ {1e-2 … 1e3}(对数网格)train 拟合、val 选择;
   目标 = v。
2. **Prompted LLM**(Qwen3-32B-AWQ,单卡,temperature 0,单 seed——与论文 C6
   primary 同口径并照此披露):复用 `scripts/experiments/e1_llm_forecast` 的
   prompt/guided-JSON/clip/retry 机制;要价 = 未来 n 个交易日年化波动率 %,
   裁剪 [3, 300]%,换算 σ_daily = (ann%/100)/√252 → v̂ = ln σ_daily。转录截断:
   头部 12,000 tokens(max-model-len 16,384;median 转录 ≈ 10.8k 字符,截断
   触发数披露)。prompt 全文在冻结前写入
   `scripts/experiments/second_domain/maec_prompt.py` 并随 tag 冻结。
   【OPEN-12 已裁决:head-only】截断触发数披露(median 转录 ≈2.7k tokens,预计罕见)。
3. **零内容 identity probe**(prompted 臂的镜像,对应论文的 date+ticker probe
   与 Yelp 的 name+city probe):prompt 只给 ticker + call 日期,**无转录**,
   其余与臂 2 逐字相同。probe 是诊断行(§6.2),其"再现 fulltext 组合增益的
   份额"是预声明的报告量。【OPEN-7 已裁决:ticker + CRSP comnam 公司名 + call 日期】probe 的职责是最大化
   引出 identity 先验——S&P 1500 小盘名的 ticker 单独可能引不出先验,弱 probe 会
   低估 identity 份额、美化文本。与 SEC probe(date+ticker)和 Yelp probe
   (name+city+categories+month)的载荷差异在论文中披露。
4. 【OPEN-5 已裁决:纳入】**frozen-embedding ridge**(Qwen3-Emb-8B mean-pool +
   ridge,ridge 网格与 TF-IDF 同,val 选择)——镜像 Yelp 三臂设计,补上
   fitted-representation 一格。其 F1/F2 Holm 族随 §6.2 同构建立(见该节)。
5. 【OPEN-6 已裁决:不纳入】第二 prompted 家族不进本审计(单家族;跨家族
   稳健性问题由 SEC 侧 B1 承载);若未来加做,须以修订预登记。

每臂读数(对每个参考 X ∈ {R-AR, R-HAR}):
- text-alone:重校准文本 standalone vs f_X(descriptive + raw DM);
- **组合增量(row-3 类比)**:f_U = OLS[1, V_past·, v_text] vs f_X;
- STPEV 控制(row-4 类比):f_Xe vs f_X(descriptive + raw DM);
- **identity-controlled 残差(row-5 类比)**:f_Ue = OLS[1, V_past·, STPEV,
  v_text] vs f_Xe;
- identity share:(组合增量被 STPEV 控制吸收的份额)= d4/d3(yelp_protocol
  同名量),probe 份额 = probe 组合增量 / fulltext 组合增量。

## 6. 统计与推断(预声明,一次执行,全部进表)

### 6.1 聚类 DM

- **PRIMARY:call-date-clustered DM**(day → call-date 的移植):损失差先按
  call 日期取等权均值(test 共 143 个日期簇),对日期序列做 DM,HAC lag
  **L_n = 冻结 test 日期格上、任一日期之后落在 n−1 个交易日内的后续不同
  call 日期数的最大值**(从 manifest 日期格一次算出、随冻结披露——只依赖
  日期元数据,不依赖标签),HLN 小样本修正,t(#dates−1)。
- **co-primary 稳健性:date × ticker 两向 CGM**(yelp_protocol `dm_test_2way`
  的移植,entity = permno),同表并列。
- 日期格非连续(财报季簇状):`monthly_mean` 的连续性断言按日期格放宽为
  "有序、去重",L_n 定义已按真实格覆盖重叠,披露 n=30 时 L_n/#dates 比值。

### 6.2 Holm 族(预声明;族一经冻结不增不减)

对每个 headline 文本臂 a ∈ {TF-IDF, frozen-embedding(Qwen3-Emb-8B), prompted-Qwen}
(OPEN-5 已纳入,OPEN-6 已排除;三臂,族结构同构):
- **族 F1(a)"组合增量"**:4 horizons × 2 参考 = **Holm(8)**;
- **族 F2(a)"identity-controlled 残差"**:同构 **Holm(8)**。
probe、text-alone、STPEV-only、published-convention 读数均为 descriptive /
诊断行(raw p 或无 p),**不进 Holm、不得进"win"表述**。

### 6.3 placebo 闸门(任何进入"win"表述的 cell 必须通过)

- **PRIMARY:label-shuffle**(文本预测全行置换,val+test 同置换、权重 val
  重拟合),20 seeds(1000–1019,Yelp 同款);
- **诊断:within-date text-swap**(同 call 日内置换;单 call 日期不换,有效
  置换行占比披露),5 seeds(2000–2004)。
判定沿用 Yelp 先例:shuffle 为 G4 主闸,swap 为 G4b 诊断;swap 边缘脏
(如 Yelp h=3)→ 该 cell 不进 prose 主张。

### 6.4 功效与 CI

- **分析 MDE(80% power)**:(1.96+0.84)·SE_date/MSE_ref·100,SE_date 来自
  日期均值损失差的 HAC(L_n) 方差;每 stage(AR 段 / entity 段)× horizon 报告。
- **oracle signal-injection**(yelp_protocol row-1 移植,s = within-permno
  demeaned test 残差,entity-orthogonal;目标 {0.5, 1.0, 2.0}% + 自适应
  max(2, 1.5·MDE, real+1)):机制检出门;披露语原文照录("ORACLE injection —
  power calibration only, never citable as forecast performance")。
- **CI**:凡进入 prose 的残差 rel%,配 date-block moving bootstrap CI
  (block = 5 个 call 日期,2,000 draws,seed 2026;Yelp/omnibus 的
  block-bootstrap 先例在 6 个 test 月的格上退化,故以日期块代月块,披露)。
- **null 的措辞纪律**:任何"absorbed/无残差"结论必须并排 MDE;若 MDE 大于
  被重定价的公开式增益(换算到同单位),措辞降为
  "underpowered to rule out",不得写成干净的零。

### 6.5 单发纪律

`maec_protocol.py`(yelp_protocol 的移植,entity=permno、cluster=call-date、
组合器=v-空间 OLS)对每个 (臂 × 对齐) 组合**只运行一次**;所有数字无论方向进
`results/second_domain/maec/protocol_<arm>.json` +
`results/tables/maec_audit.{csv,md}` + FACTS.md。重跑仅限脚本 bug,须在本文件
修订记录中记 diff 与原因。

## 7. Sanity gates(任一失败即中止,不出表;G 编号沿用家族惯例)

- **G1(公开式读数的符号复现,审计前置)**:published-convention 臂下,至少
  一个文本臂 text-alone 的 MSE(v) 低于 raw V_past^(n)(镜像 Yu et al. 的
  0.257 vs 1.12 排序)。若**全部**文本臂失败:替身臂太弱、无法承载"重定价
  文本增益"的主张 → 审计范围降级为"仅基线重校准审计"(§8 分支 D),
  不得中途换臂或调 prompt 重试。
- **G2(量级门)**:我们的 raw V_past MSE(v) 与 Yu et al. 报告的 1.12 同量级
  (比值 ∈ [1/3, 3];面板不同——他们 MAEC-15/16 子集、我们全量 3,443——
  故只做量级门不做等值门)。失败 → 先查标签构建(§2)再议,不得带病出表。
- **G3(泄漏断言)**:split 边界断言(§4);组合器权重 val-fit test-frozen 的
  代码断言;STPEV 点时断言(每个贡献标签的窗末 ≤ 当前 call 的 τ);
  within-entity 注入信号 mean-zero 断言(yelp_protocol 同款)。
- **G4/G4b**:placebo 闸(§6.3)。
- **G5(键与合并)**:消歧后每 call 恰一 permno;各臂行集恒等;merge 后
  label 逐行一致;(permno, call_date, horizon) 无重复。
- **G6(排除审计)**:stub 排除恰为预声明规则(n_chars<100,总数 32),
  per-split 计数与 §3.2/§3.3 各排除计数逐项披露、总量对账
  (3,443 − 排除 = 各臂行数)。

## 8. 判定阶梯与分支承诺(三分支现在写死,binding on the prose)

判定对象 = 每个 headline 文本臂的族 F2(identity-controlled 残差,8 cells):

- **(a) FULLY ABSORBED**:0/8 cells 同时满足 DM<0、Holm<.05、G4 通过;且
  identity share(d4/d3)≥ 100% 或组合增量本身不显著。
  → 论文措辞:"在 MAEC 上,把被弃用的 past-vol 基线重校准、再加同 ticker 的
  STPEV 均值,公开式文本增益被完全吸收;call-date-clustered 推断(文献从未
  提供)在 MDE=X% 的功效下确认了 Yu et al. 的表示层发现在预测层同样成立——
  第三个领域,同一测量仪器。"(必须并排 MDE,§6.4。)
- **(b) PARTIALLY ABSORBED**:≥1/8 cells 过 Holm+placebo,且这些 horizon 上
  identity share ≥ 50%。
  → "重校准 + identity 吸收公开式增益的 X–Y%;一个有界、placebo-clean、
  功效校准的残差存活——与 SEC 面板的有界残差同构,大小由面板及其基线定价。"
- **(c) SURVIVES**:≥4/8 cells 过 Holm+placebo 且 identity share < 50%。
  → "MAEC 上文本臂的增量不是 identity 伪影:该基准的问题在基线失校准而非
  identity;我们给出该基准上首个聚类显著性与功效校准。Yu et al. 的表示层
  相似并未在组合层转化为预测冗余。"(诚实分支;fallback 框架仍是
  "shortcut 的大小是面板及其基线的属性"——Yelp 先例。)
- **其余组合**:MIXED,逐 cell 如实报告,措辞取最弱可辩护形式。
- **(d) G1 降级分支**:替身臂全弱 → 只发表基线重校准审计(raw V_past 1.12 类
  读数 vs 重校准后读数 + STPEV 定价),文本增益的重定价主张整体撤下。

两个 headline 臂判定不同档 → 逐臂报告,不合并措辞(Yelp 先例:prompted 与
fitted 臂的 identity/content 拆分本身就是发现)。§2.3 的对齐 sensitivity 改判
档位 → primary 管辖 + 差异披露(OPEN-2 已裁决维持,此句即终版)。

## 9. 预先披露的偏离与局限(写进论文 Limitations 的底稿)

1. **text-only 审计一个多模态基准**:59 GB 音频特征未取(per 采集决定);
   正当性:Yu et al. 的分析同样以文本侧为主,且 AMA-LSTM 先例表明音频通道的
   增量边际且不稳。审计对象是"文本增益"主张,不是完整多模态栈。
2. **MDRM cite-only**:全量数据被 5 分卷 zip 捆绑音频结构性阻断
   (ACQUISITION_NOTES §2);无 license,不再分发。
3. **无时间戳**:day-0 歧义按 §2.3 预声明处理(primary + 平移 sensitivity)。
4. **44/700 tickers 窗口覆盖 <90%**(期中 IPO/退市):按 §3.2 规则逐行排除并
   计数;非缺口,是上市窗效应。
5. **面板小**(3,443 calls,test 674 行 / 143 日期簇):功效以 MDE 明码标价
   (§6.4);n=30 的 HAC lag 相对簇数偏大(OPEN-8:留族内,bootstrap CI 为确证读数)。
6. **prompted 臂单 seed**(temperature 0,与 C6 primary 同口径);"多 seed"
   若做,措辞必须按 FACTS §13c:reproducibility jitter,不得称随机解码
   ensemble。
7. **污染披露**:MAEC 2015–2018 深处 LLM 预训练窗内;probe 臂(§5-3)正是
   为定价 memorization/identity 先验而设,论文按 SEC 侧 llm_contamination
   的既定框架披露。
8. 本 DRAFT 起草时读过 manifest 的日期/字符计数(§0、§4 边界即由此来);
   标签、价格、转录内容未读。此披露随冻结版保留。

## 10. 计算与实现

- 硬件:4×A100-40GB 单节点;Qwen3-32B-AWQ 单卡(vLLM offline batch,与
  e1_llm_forecast 逐字同协议);3,443 calls ≈ 数 GPU 时,成本低。fitted 臂
  CPU(遵守本机 ≤半数核规则;box 上按 cgroup 顶满)。
- 新脚本(冻结前入库):`scripts/experiments/second_domain/maec_build_panel.py`
  (标签+特征+排除审计)、`maec_baseline_text.py`(R-AR/R-HAR/TF-IDF 臂)、
  `maec_prompt.py`(prompt 冻结)、`maec_protocol.py`(yelp_protocol 移植)。
  产出:`results/second_domain/maec/` + `results/tables/maec_audit.{csv,md}`。
- 与在跑实验的边界:本实验不读 SEC 面板的任何 test split;B1(Mistral)与
  HPO 单发评估各自受其预注册管辖;A 的一切统计只触碰 §1 冻结输入。

## 11. 修订记录

- **v1.2(2026-07-15)**:两项 build 期澄清/修订,均先于任何臂打分(panel 构建的
  gates 已跑,协议/臂统计量零计算):(1) **收益源澄清**——§1 钉的 full_ohlcv 无收益列,
  且验证其 adj_close = 未调整 DlyClose(548,223 重叠日 1.36% 差异 >1bp,XRX 反向拆股日
  差 301%),按 §2.1 估计量优先改用同一 ingest 的 `market/crsp/market_returns.parquet`
  (log1p(DlyRet) 经 expm1 精确还原,两侧同构);(2) **S&P500 侧成员期缺口补齐**——
  S&P500 缓存仅覆盖指数成员期窗口,致 ~307-311 条 call(全部 S&P500 侧,含 49 ticker
  的 122 call 零行)被 §3.2 价格闸删除;此为缓存结构伪影而非数据不可得,追加冻结输入
  `crsp_sp500side_gapfill_2014_2019.parquet`(从本地 CRSP 全宇宙 raw zip 提取,机制与
  700-ticker 提取一致),panel 以三源重建,排除对账随 build_report 更新。
- **v1.1(2026-07-15)**:OPEN-1 文献核验完成(CIKM 2020 Table 5 + Yu et al. Table 1/3 +
  MDRM §6.2 原文引证,报告存会话档案)。修订:published-convention 臂 split 改为按年
  三面板 7:1:2(Table 5 日期钉死)、对齐改为 day-1-start 未调整;§2.3 的对应句更正。
  证明:修订时 MAEC 面板尚在构建、published-convention 臂无任何统计量;primary 不动。
