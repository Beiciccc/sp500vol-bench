# 预注册:H 公开价格变体全量级联(prereg-h-v1.0)

日期:2026-07-16。先于任何 H 统计量提交并打 tag。承诺(家族惯例):全部分支结果无论方向
进论文;单发纪律;修订先于统计量。上游:label_parity(ROW 10,committed)已给出
verdict-preservation 三面板先验(standalone 18/18、combo genuine A=8 vs C=7)与 trade 量化
(clean coverage 80.19%,exit 公司行覆盖 31.9%,test split 覆盖 95.4%)。

## 设计

**公开源**:Yahoo Finance v8 日频 adjusted close(与 label_parity Stage 1 同 fetcher、同
total-return 口径;缓存已灭失须重抓,**抓取漂移披露**:重抓相关性/覆盖率与 committed
label_parity 表并列,漂移超过 0.5pp 覆盖或 0.001 相关性须在产物中显著标注)。Yahoo 数据
禁再分发——发布物为**重建管线 + 抓取脚本**,非数据本身(与 label_parity.md 口径一致)。

**三面板(label_parity Stage 5 的级联版)**:
- **A** = 全 panel + CRSP 标签(锚,G1 复现门);
- **B** = 公开覆盖行 ∩ panel + CRSP 标签(隔离幸存者效应);
- **C** = 公开覆盖行 + 公开标签/特征(= 可发布变体本体)。
A−B 之差 = 幸存者损失;B−C 之差 = 标签源噪声。

**范围**:leaderboard 全臂 standalone 判定(day-clustered DM vs A2,variance-unit,
与 committed 同法)+ 69-cell 级联全链(primary → firm-identity → maximal pool → conjunction,
文本预测冻结、combiner val 重拟合,机器 = rangebased_cascade 的 est 钩子扩展)+ 逐格 MDE 与
注入恢复(verbatim)。A-block:A2/A6 在公开特征+标签上 refit(A6 的 RS± 用公开收益重建符号
分解);A3/A4/A5 冻结 + val 重校准(range-based 先例)。

**Gates**:G1 面板 A 复现 committed 级联机器精度(counts + stats);G2 covered 行标签
Pearson ≥0.99(先验 0.998;低于即中止查抓取);G3 覆盖对账(80.19% ± 抓取漂移,
逐 split/逐 exit-status 分解并与 committed parity 表并列);G4 placebo 照跑。

**分支承诺**:
- **(a) 判定保持**(预期):B、C 面板的 standalone 判定与 conjunction(Holm=0)与 A 一致
  → licence-free 变体升格为**正式发布物**:12_reproducibility 从 "withheld" 改写为
  "shipped variant + 量化幸存者代价"(train 丢 27%、test 丢 4.6% 明码标价);
  07 的一句话升级为全级联数字;贡献 1 增加变体条目。
- **(b) 构成移动、判定保持**(range-based 先例):同 (a) 升级,构成差异如实并表。
- **(c) 判定翻转**(conjunction>0 或出现 standalone 胜者):诚实报告;变体照发,
  翻转本身作为幸存者/标签源效应的发现,A vs B vs C 分解定位来源。
- **蓄意剔除**:不做 Yahoo 之外的第二公开源(Stooq 等留作未来);长表/事件驱动不拆分处理
  (同一管线);不重训任何文本模型。

**输出**:`results/tables/public_variant_cascade.{csv,md}`(单发守卫)+
`scripts/analysis/public_variant_{labels,cascade}.py` + 抓取脚本。全程本地 CPU(≤5 核)。
