# JCIM 投稿前严格审查报告

**审查对象：** `E:\fullproject\project\CU-MOBO\docs` 及其全部子目录  
**目标期刊：** *Journal of Chemical Information and Modeling*（JCIM，ACS Publications）  
**审查日期：** 2026-07-20（Asia/Shanghai）  
**审查方式：** 只读检查原始文件；未修改、覆盖或删除任何原稿。为版面核查生成的临时 PDF/PNG 位于 Codex 审查缓存目录，不属于原稿。  
**规则基准：** JCIM 作者指南网页标注最后更新于 2026-07-03；本报告于 2026-07-20 查询。

---

## 1. 执行摘要

### 1.1 最适合投稿的文件

当前最接近 JCIM 投稿稿的是：

`E:\fullproject\project\CU-MOBO\docs\paper_jcim_clean.docx`

SHA-256：`9C10D8DE528074AEC982B8224760B486FCCC00A3A770EC5E54DD9C45A6044B93`

该文件包含完整英文正文、8 幅正文图和 48 条编号参考文献，且删除了工作稿中的复现实验命令附录。目录中另有 6 个与它逐字节完全相同的别名文件，不能视为不同版本：

- `paper_jcim_clean.english_tmp.docx`
- `paper_jcim_clean.namefix_out.docx`
- `paper_jcim_clean.tmp_600dpi.docx`
- `paper_jcim_clean_600dpi.docx`
- `paper_jcim_clean_600dpi.namefix_out.docx`
- `paper_jcim_clean_english_figures.docx`

### 1.2 总体结论

**结论等级：需要重大修改后再投稿。**

主题属于 JCIM 所覆盖的化学信息学、分子设计、机器学习与计算方法范围，稿件类型宜选择 **Article**。但是当前稿件尚缺若干提交必需内容，并存在会被编辑或审稿人集中质疑的科学问题：

1. 通讯作者邮箱仍是占位符；缺 Data Availability Statement、利益冲突声明、代码/数据持久链接及必要投稿附件。
2. 摘要约 392 词、11 句且分成 3 段，不符合 JCIM 当前明确要求的简洁 3–4 句摘要。
3. 核心统计解释尚不充分：把外部预测器的 `σ²` 直接作为 GP 的 `train_Yvar`，并据此声称后验均值“向真实活性去噪”，需要证明 `σ²` 确实代表观测 `μ` 相对目标真值的条件误差方差，而不是一般 epistemic uncertainty 或排序分数。
4. 合成实验中 oracle error 与 `σ` 按仿射关系构造，Spearman 相关为 1；这验证了理想情形，却不足以支持对弱校准、失配或错误不确定度的广泛结论。
5. 真实数据验证仍有限：核心真实任务主要是一个 kcat 数据集上的相似性恢复/代理排序，不是对生成分子的实际 kcat 测量；3–5 个随机种子不足以支撑多处“significantly”或稳定泛化表述。
6. 数据清洗、任务单位、酶–底物关系、分组划分和防泄漏细节不足，尤其是“约 13 个 log10 单位”的 kcat 范围需要追溯、单位统一和异常值说明。
7. 48 条文献均有可靠存在证据，**未发现确认虚假或高度疑似虚假的引用**；但第 6、15、47 等文献不能充分支持相应正文表述，且多条会议论文、arXiv 和软件引用不完整。

因此，不建议以现状直接提交；但在补齐强制项、收敛论文主线、加强统计与真实任务验证并修正引用后，文章仍具有 JCIM 投稿潜力。

---

## 2. 审查范围、方法与限制

### 2.1 文件范围

递归发现 **127 个文件**：39 个 XML、39 个 PNG、15 个 DOCX、9 个 Python 脚本、7 个 Markdown、2 个 TeX、1 个 PDF、1 个 BibTeX、1 个 CSL、1 个 Lua 及若干构建中间文件。大量 XML/关系文件属于已解包的 DOCX 结构，不是独立投稿材料。

### 2.2 使用的核查证据

- DOCX：提取正文和文档结构，并将最优候选稿导出为 34 页 PDF 后逐页渲染、检查图表、分页、字体和引用表。
- 文本：以 `paper_clean.md` 作为可定位源文本，对照 `paper_jcim_clean.docx`；源文件约 5604 个正文词，不含参考文献。
- 参考文献：对 48 条逐条检查；有 DOI 的条目通过 DOI/Crossref 与出版商元数据核对，无 DOI 的条目通过 NeurIPS、PMLR、ICLR、arXiv、RDKit/Zenodo 等官方页面核对。
- 引用对应：程序化核对 48 个 BibTeX 键；正文独立引用键 48 个，缺失 0，未引用 BibTeX 条目 0。
- 图像：直接读取候选 DOCX 内嵌 8 张 PNG；均为 600 dpi，像素尺寸从 3286×2947 到 9000×2280。对彩色/灰度图满足 ACS 300/600 dpi 门槛；若某图被归类为纯黑白线稿，则 ACS 要求 1200 dpi，仍应以最终图型确认。

### 2.3 限制

- 本审查未访问付费 Web of Science/Scopus 账户；但所有条目均能由 DOI、出版商、PubMed、官方会议或预印本页面核验，未因此留下“无法核实”的条目。
- “正文论述是否被文献支持”依据标题、摘要、官方元数据和可获取正文判断；需要全文付费内容时采用保守结论，不把未确认事项断言为虚假。
- 未重新运行全部实验，因此数值一致性检查限于正文、表格、图注和目录内可见结果文件的交叉阅读；报告中的科学质疑不是对原始实验数据造假的认定。
- 页码来自审查时由 Word 导出的 34 页版本；作者重新排版后页码会变化，因此同时提供章节和 `paper_clean.md` 行号。

---

## 3. 目录文件与版本识别

| 文件/组 | 角色判断 | 完整性及差异 | JCIM 适配判断 |
|---|---|---|---|
| `paper_jcim_clean.docx` + 6 个同哈希别名 | 当前英文清洁投稿候选 | 34 页、8 图、48 条编号参考文献；无工作命令附录 | **最优候选，但尚不可直接投稿** |
| `paper_jcim.docx`、`paper_jcim.namefix_out.docx` | 较早英文 JCIM 版 | 两文件同哈希；参考文献后仍含 `Reproducibility` 和原始命令 | 不宜作为正文直接提交 |
| `paper_draft.docx`、`paper_draft.md` | 工作稿/旧稿 | 结构及参考文献渲染不如 clean 版；保留开发性内容 | 不推荐 |
| `paper_clean.md` | 最优候选的英文源文本 | 内容与 clean DOCX 主体对应；参考文献由 Pandoc/BibTeX 渲染 | 适合继续修订，不宜单独作为 Word 投稿稿 |
| `paper_clean_zh.docx`、`paper_clean_zh.md` | 中文翻译稿 | 不是 JCIM 英文主稿 | 不适合作为主文，可供作者内部校对 |
| `paper_jcim_clean.before_*` | 历史备份 | 600 dpi、英文图替换、名称修复前版本 | 备份，不提交 |
| `references.bib`、`references_links.md`、CSL/Lua/TeX | 引文数据库及构建资源 | 支撑主稿生成 | 不作为主文；BibTeX 可在修订时继续使用 |
| `cu_mobo_method.md`、`cu_mobo_improvements.md`、`validation_protocol_research.md` | 方法说明、扩展笔记、验证研究记录 | 部分内容已进入正文；仍含研发记录 | 可整理为 SI，但当前不是可提交的 SI |
| `figures/`、`english_600dpi/`、PNG、绘图脚本 | 图和生成资产 | 候选稿实际嵌图为 600 dpi | 原始图可作为修订资源；投稿时遵循文件指定 |
| `docx_media_check*`、`docx_update_*` | DOCX 解包/构建中间目录 | OOXML 内部资源 | 不提交 |

### 3.1 版本判断依据

1. 文件名明确含 `jcim_clean`，且修改时间晚于 draft。
2. 内容为英文，章节、图、表和参考文献最完整。
3. 相比 `paper_jcim.docx`，删除了不适合留在正文末尾的命令行复现段。
4. 内嵌图替换成了 600 dpi 版本。
5. 七个文件哈希完全一致，说明“600dpi”“english_figures”“namefix”后缀是同一最终产物的复制，而不是可比较的不同稿。

建议作者只保留一个明确的提交候选名称，例如 `CU_MOBO_JCIM_Article_v1.docx`，但本次审查未替作者重命名或删除文件。

---

## 4. 最新 JCIM/ACS 官方要求与逐项对照

### 4.1 官方来源

查询日期均为 2026-07-20：

1. [JCIM Author Guidelines（标注 2026-07-03 更新）](https://researcher-resources.acs.org/publish/author_guidelines?coden=jcisd8)
2. [JCIM 2015 编辑指南，DOI 10.1021/acs.jcim.5b00180](https://pubs.acs.org/doi/10.1021/acs.jcim.5b00180)
3. [JCIM 数据与软件可用性编辑要求，DOI 10.1021/acs.jcim.0c01389](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01389)
4. [ACS Research Data Policy](https://researcher-resources.acs.org/publish/data_policy)
5. [ACS AI Best Practices and Policies](https://researcher-resources.acs.org/publish/aipolicy)
6. [ACS Ethical Guidelines](https://researcher-resources.acs.org/publish/ethical_guidelines)
7. [ACS TOC/Abstract Graphic Specifications](https://pubs.acs.org/paragonplus/submission/toc_abstract_graphics_guidelines.pdf)

### 4.2 对照表

| 项目 | 规则性质 | 当前状态 | 结论与措施 |
|---|---|---|---|
| 期刊范围 | 官方范围 | 方法涉及化学信息学、分子设计、ML 和优化 | **符合主题范围**；但实验验证强度偏弱 |
| 稿件类型 | 官方分类 | 完整方法论文，约 5604 正文词 | 选择 **Article**；不是 Letter/Application Note |
| 稿件完整、无批注、组件编号 | 官方强制（Fast Format） | 主体完整，无可见批注；缺投稿声明和附件 | 不符合“完整提交” |
| 标题 | 官方建议“preferably 12 words or fewer” | 约 12 个词，信息明确 | 基本符合；不是硬性字数违规 |
| 摘要 | 官方强制 3–4 句且简洁 | 约 392 词、11 句、3 段，跨第 1–2 页 | **不符合**；重写为单段 3–4 句 |
| 关键词 | Article 页面未明确硬性数量；Perspective/Review 明确约 8–10 个 | 无 | 作为投稿元数据和检索优化建议补 5–8 个；不要误称 Article 硬性 8–10 个 |
| 正文篇幅 | Article 当前指南未给明确词数上限 | 约 5604 词、34 页单栏 | 无硬性超限；但主线过长、扩展 E1–E4 分散重点 |
| 图表嵌入正文 | 官方强制/快速格式 | 已嵌入相应位置 | 符合 |
| 图分辨率 | 官方强制规格：线稿 1200、灰度 600、彩色 300 dpi | 8 图均 600 dpi | 彩色/灰度满足；纯线稿需确认是否应 1200 dpi |
| 图尺寸、字体、线宽、色盲可及性 | 官方规格/建议 | 分辨率足够；缩小后部分字体和颜色可读性待检查 | 按最终 3.33 或 4.167–7 in 宽复核 ≥4.5 pt、线宽 ≥0.5 pt，不能只靠颜色编码 |
| 图注 | 官方强制自足、符号一致 | 多数有图注；部分图注含宣传性结论 | 技术上基本具备；压缩并补充统计定义/误差带含义 |
| 表格 | 官方要求简洁、短标题、避免拆分/合并单元格 | 多表列宽过窄；QED 表跨页断裂 | 需重排；表题/脚注应自足 |
| TOC Graphic | 当前 JCIM 指南链接官方规格；对 Article 的文字未像 Viewpoint 那样在类型段明确写“must” | 未发现 | 按投稿包准备原创 TOC Graphic；在最终提交系统确认文件指定。不得用 AI 生成图像 |
| Supporting Information | 有 SI 时须与主稿同时作为独立文件，并在正文给含扩展名的非句式说明 | 无正式 SI；方法细节散落在开发笔记/仓库路径 | 建议生成独立 SI PDF/ZIP，并在 Associated Content 中说明 |
| Data Availability Statement | **JCIM Level 2，投稿时强制** | 缺失 | **必须补**；给公开仓库、DOI/URL、版本和访问方法 |
| 数据/软件公开 | JCIM 官方编辑要求；指南强烈鼓励公开并正确引用 | 只有本地路径，没有公共链接/版本 DOI | 属重大风险；至少提供可供编辑/审稿人验证的代码和数据，最好 Zenodo DOI + GitHub release |
| 数据引用 | 官方要求公开数据在正文和参考文献中正式引用 | 本地 CSV 未形成数据引用 | 补数据集/软件引用、许可证和版本 |
| Cover letter | **官方强制，每稿必须** | 目录中未发现 | 必须准备；说明范围适配、创新、数据代码、预印本/既往投稿情况 |
| 通讯作者及联系信息 | 投稿系统/作者信息要求 | 邮箱是 `to be provided` | **致命缺项**；提交前必须替换 |
| Conflict of Interest | ACS 官方要求声明有无利益冲突 | 缺失 | 必须补，例如无冲突时使用 ACS 标准声明 |
| Author Contributions | ACS 作者责任政策；通常以 CRediT 声明提交 | 缺失 | 建议/投稿系统很可能要求；补充并由全体作者确认 |
| Acknowledgment/Funding | ACS 要求披露基金与潜在偏倚来源 | 缺失 | 有基金必须完整列号；无致谢也应确认 |
| AI 工具披露 | 使用 AI 文本/图像生成时官方强制透明披露 | 稿件无声明，实际使用情况未知 | 作者必须确认；若使用，在 Acknowledgments 说明何时、如何使用，实质性方法使用写入 Methods |
| AI 生成 TOC 图 | ACS 明确禁止 | 当前无 TOC 图 | 制作时不得使用 AI 生成图像 |
| 化学结构/命名/单位 | ACS 图和化学表达规范 | 本文以计算方法为主；未展示新化合物表征 | 不是主要缺陷；但 kcat 单位、log 变换和边界需补 |
| 实验化合物表征 | 有新实验活性化合物时需结构纯度证据 | 无湿实验、新化合物活性声明 | 当前不适用；不可把计算候选写成已验证活性化合物 |
| 参考文献初投稿样式 | Fast Format：可用任意一致样式，但必须完整并含标题 | 编号连续，含标题；若干条目不完整 | 初投稿不因未用 CASSI 自动违规；但缺 arXiv ID、会议页码/官方链接、软件版本仍需修 |
| 正式 ACS/JCIM 参考文献样式 | 官方 ACS Style | 期刊名多为全称，会议格式不统一 | 投稿前最好统一为 ACS，期刊名改 CASSI 缩写 |
| 多任务/真实问题验证、数据整理章节 | JCIM 2015 编辑选择标准，不等同机械格式硬性项 | 一个主要 kcat 任务 + 理想合成任务 + QED 控制；缺独立 Data Curation 章节 | **显著 desk-rejection 风险**；增加真实任务/外部验证并建立数据整理章节 |

---

## 5. 参考文献真实性与内容支持核查

### 5.1 总结

- 总条目：48。
- 已通过可靠来源确认真实：48。
- 确认虚假：0。
- 高度疑似虚假：0。
- 无法核实：0。
- DOI 指向错误论文：0。
- 重复 DOI/重复条目：0。
- 正文有引用但参考文献表缺失：0。
- 参考文献表有条目但正文未引用：0。
- Boltz-1 仍是 bioRxiv 预印本；当前稿正确标为预印本，没有冒充同行评审期刊论文。
- RDKit DOI `10.5281/zenodo.591637` 是 Zenodo 概念 DOI，Crossref 不收录并不表示虚假；应改为实际使用版本的 DOI/版本号。

“内容支持情况”针对稿件实际使用语境，而不是评价文献本身质量。“格式风险”与“真实性”分开。

### 5.2 逐条核查表

| 序号 | 正文引用位置 | 文献信息 | 核验结果 | DOI/官方链接 | 内容支持情况 | 格式问题 | 风险等级 | 建议 |
|---:|---|---|---|---|---|---|---|---|
| 1 | 引言、§6 | Li et al., DLKcat, 2022 | 已确认真实且信息正确 | [10.1038/s41929-022-00798-z](https://doi.org/10.1038/s41929-022-00798-z) | 支持 kcat 预测器示例；不单独证明“校准 σ” | 期刊名应统一缩写 | 低 | 保留；不要把其扩展成已校准不确定度证据 |
| 2 | 引言、§6 | Yu et al., UniKP, 2023 | 已确认真实且信息正确 | [10.1038/s41467-023-44113-1](https://doi.org/10.1038/s41467-023-44113-1) | 支持酶动力学预测器示例 | 同上 | 低 | 保留 |
| 3 | 引言 | Kroll et al., turnover prediction, 2023 | 已确认真实且信息正确 | [10.1038/s41467-023-39840-4](https://doi.org/10.1038/s41467-023-39840-4) | 支持 kcat/turnover 预测 | 同上 | 低 | 保留 |
| 4 | 引言、§6 | Wohlwend et al., Boltz-1 | 已确认真实且信息正确 | [10.1101/2024.11.19.624167](https://doi.org/10.1101/2024.11.19.624167) | 支持结构预测模型示例；不证明其输出是本文所需的“校准 σ” | 应显式保留 `[Preprint]`/bioRxiv 版本日期 | 中 | 把“well-calibrated … Boltz”改为条件性示例或补校准证据 |
| 5 | 引言 | Jumper et al., AlphaFold, 2021 | 真实但书目信息有误 | [10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) | 支持结构预测；不支持通用不确定度校准声明 | `others` 不是规范作者列法 | 中 | 按 ACS 长作者名单规则重排 |
| 6 | 引言、相关工作、§4.6 | Gao et al., PMO, 2022 | 真实但引用不能充分支持正文论述 | [NeurIPS 官方页](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html) | 支持样本效率和 oracle budget；不直接证明“高预测分、低真实值”或秒到分钟成本 | 缺卷/页码或官方 URL | **高** | 拆分主张；为 reward hacking 另引直接研究 |
| 7 | 引言、§4.4 | Lakshminarayanan et al., Deep Ensembles | 已确认真实且信息正确 | [NeurIPS 官方检索](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | 支持集成不确定度估计；“calibrated”需限定 | 会议格式统一 | 低 | 保留，避免宣称自动校准 |
| 8 | 引言、§4.4 | Angelopoulos & Bates, conformal tutorial | 真实但书目信息有误 | [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) | 支持 conformal UQ 概念 | 当前缺 arXiv ID、URL | 中 | 补 `arXiv:2107.07511` 和链接 |
| 9 | 引言 | Amini et al., Deep Evidential Regression | 已确认真实且信息正确 | [NeurIPS 官方页](https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html) | 支持 evidential regression | 会议格式统一 | 低 | 保留 |
| 10 | 引言 | Amodei et al., Concrete Problems in AI Safety | 真实但书目信息有误 | [arXiv:1606.06565](https://arxiv.org/abs/1606.06565) | 支持 reward hacking 广义概念；不是分子设计实证 | 缺 arXiv ID/URL | 中 | 补 ID，并配分子领域直接证据 |
| 11 | 引言 | Skalse et al., reward gaming | 已确认真实且信息正确 | [NeurIPS 官方检索](https://proceedings.neurips.cc/paper_files/paper/2022/) | 支持 reward gaming 定义 | 应补精确官方页面 | 低 | 保留 |
| 12 | 引言 | Shahriari et al., BO review | 已确认真实且信息正确 | [10.1109/JPROC.2015.2494218](https://doi.org/10.1109/JPROC.2015.2494218) | 支持 BO 框架 | 期刊缩写统一 | 低 | 保留 |
| 13 | 引言 | Frazier, BO tutorial | 真实但书目信息有误 | [arXiv:1807.02811](https://arxiv.org/abs/1807.02811) | 支持 BO 教程性陈述 | 缺 arXiv ID/URL | 中 | 补 ID/URL |
| 14 | 相关工作 | Daulton et al., qEHVI | 真实但书目信息有误 | [NeurIPS 官方页](https://proceedings.neurips.cc/paper/2020/hash/6fec24eac8f18ed793f5eaad3dd7977c-Abstract.html) | 支持 qEHVI；不应与 qNEHVI 合并宣称都假设同方差 | 当前缺页码 9851–9864 | 中 | 补页码；拆分 qEHVI/qNEHVI 性质 |
| 15 | 相关工作、方法 | Daulton et al., qNEHVI | 真实但引用不能充分支持正文论述 | [NeurIPS 官方页](https://proceedings.neurips.cc/paper/2021/hash/11704817e347269b7254e744b5e22dac-Abstract.html) | 原文明确处理 noisy objectives；与“assume noiseless or homoscedastic”表述冲突 | 缺页码/官方 URL | **高** | 改为“未默认消费外部校准误差方差”，不要说 qNEHVI 假设无噪声 |
| 16 | 相关工作、§4.1 | Knowles, ParEGO | 已确认真实且信息正确 | [10.1109/TEVC.2005.851274](https://doi.org/10.1109/TEVC.2005.851274) | 支持 ParEGO 基线 | 期刊缩写统一 | 低 | 保留 |
| 17 | 相关工作、§5 | Trabucco et al., Design-Bench | 已确认真实且信息正确 | [PMLR 162 官方页](https://proceedings.mlr.press/v162/trabucco22a.html) | 支持离线 MBO 独立 oracle 评估 | 会议大小写、出版信息统一 | 低 | 保留 |
| 18 | 相关工作、§5 | Trabucco et al., COMs | 已确认真实且信息正确 | [PMLR 139 官方页](https://proceedings.mlr.press/v139/trabucco21a.html) | 支持 proxy over-optimization 退化 | 同上 | 低 | 保留 |
| 19 | 相关工作、§5 | Yoshizawa et al., DyRAMO | 已确认真实且信息正确 | [10.1038/s41467-025-57582-3](https://doi.org/10.1038/s41467-025-57582-3) | 支持多目标分子设计中 reward hacking/applicability gating | 期刊缩写统一 | 低 | 保留 |
| 20 | 方法、§4.6 | Bickerton et al., QED | 已确认真实且信息正确 | [10.1038/nchem.1243](https://doi.org/10.1038/nchem.1243) | 支持 QED 定义；不支持“QED near-bijective in fingerprints” | 期刊缩写统一 | 中 | 删除/弱化 near-bijective 主张，或提供实验和引用 |
| 21 | 方法 | Ertl & Schuffenhauer, SA score | 已确认真实且信息正确 | [10.1186/1758-2946-1-8](https://doi.org/10.1186/1758-2946-1-8) | 支持 SA score | 期刊缩写统一 | 低 | 保留 |
| 22 | 方法 | Lipinski et al., Ro5 | 已确认真实且信息正确 | [10.1016/S0169-409X(96)00423-1](https://doi.org/10.1016/S0169-409X(96)00423-1) | 支持规则定义 | 页码连接号/缩写统一 | 低 | 保留 |
| 23 | 方法 | Baell & Holloway, PAINS | 已确认真实且信息正确 | [10.1021/jm901137j](https://doi.org/10.1021/jm901137j) | 支持 PAINS 过滤 | 期刊缩写统一 | 低 | 保留 |
| 24 | 方法 | RDKit software | 真实但书目信息有误 | [10.5281/zenodo.591637](https://doi.org/10.5281/zenodo.591637) | 支持软件来源 | 概念 DOI，无实际版本、年份、访问日期；`others` 不规范 | **高** | 引用实验所用 RDKit release 的版本 DOI |
| 25 | 方法 | Balandat et al., BoTorch | 真实但书目信息有误 | [NeurIPS 官方页](https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html) | 支持实现框架 | 缺页码/官方 URL | 中 | 补完整会议元数据 |
| 26 | §4.1 | Deb et al., NSGA-II | 已确认真实且信息正确 | [10.1109/4235.996017](https://doi.org/10.1109/4235.996017) | 支持算法 | 期刊缩写统一 | 低 | 保留 |
| 27 | §4.1 | Blank & Deb, pymoo | 已确认真实且信息正确 | [10.1109/ACCESS.2020.2990567](https://doi.org/10.1109/ACCESS.2020.2990567) | 支持软件实现 | 建议另报实际软件版本 | 低 | 保留并在 Methods 加版本 |
| 28 | §4.3 | Truchon & Bayly, BEDROC | 已确认真实且信息正确 | [10.1021/ci600426e](https://doi.org/10.1021/ci600426e) | 支持 BEDROC 指标 | 期刊缩写统一 | 低 | 保留 |
| 29 | §4.3 | Bemis & Murcko, scaffolds | 已确认真实且信息正确 | [10.1021/jm9602928](https://doi.org/10.1021/jm9602928) | 支持 scaffold 定义 | 期刊缩写统一 | 低 | 保留 |
| 30 | §4.3 | Wu et al., MoleculeNet | 已确认真实且信息正确 | [10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A) | 支持 scaffold/OOD benchmark rationale | 期刊缩写统一 | 低 | 保留 |
| 31 | §4.3 | Tran-Nguyen et al., LIT-PCBA | 已确认真实且信息正确 | [10.1021/acs.jcim.0c00155](https://doi.org/10.1021/acs.jcim.0c00155) | 支持虚拟筛选偏差/泄漏动机 | 期刊缩写统一 | 低 | 保留 |
| 32 | §4.4 | Levi et al., ENCE | 已确认真实且信息正确 | [10.3390/s22155540](https://doi.org/10.3390/s22155540) | 支持回归 UQ 校准指标 | 格式统一即可 | 低 | 保留 |
| 33 | §4.4 | Kuleshov et al., calibrated regression | 已确认真实且信息正确 | [PMLR 80 官方页](https://proceedings.mlr.press/v80/kuleshov18a.html) | 支持校准回归/可靠性图 | 会议大小写统一 | 低 | 保留 |
| 34 | §4.4 | Hirschfeld et al., molecular UQ | 已确认真实且信息正确 | [10.1021/acs.jcim.0c00502](https://doi.org/10.1021/acs.jcim.0c00502) | 支持分子性质 UQ | 期刊缩写统一 | 低 | 保留 |
| 35 | §4.5 | Polykovskiy et al., MOSES | 已确认真实且信息正确 | [10.3389/fphar.2020.565644](https://doi.org/10.3389/fphar.2020.565644) | 支持生成分子评估 | 期刊缩写统一 | 低 | 保留 |
| 36 | §4.5 | Brown et al., GuacaMol | 已确认真实且信息正确 | [10.1021/acs.jcim.8b00839](https://doi.org/10.1021/acs.jcim.8b00839) | 支持生成模型 benchmark | 期刊缩写统一 | 低 | 保留 |
| 37 | §4.5 | Xie et al., #Circles | 真实但书目信息有误 | [ICLR 官方页](https://iclr.cc/virtual/2023/poster/11769) | 支持 chemical space coverage 指标 | 缺官方 URL/会议出版信息 | 中 | 补 ICLR 官方链接 |
| 38 | §4.6 | Huang et al., TDC | 真实但书目信息有误 | [NeurIPS Datasets and Benchmarks 官方站](https://datasets-benchmarks-proceedings.neurips.cc/) | 支持 TDC 平台；具体 QED oracle 最好另引 TDC 文档/版本 | 缺官方 URL/页码 | 中 | 补 TDC 版本和具体 API/任务引用 |
| 39 | §4.6 | Olivecrona et al., REINVENT | 已确认真实且信息正确 | [10.1186/s13321-017-0235-x](https://doi.org/10.1186/s13321-017-0235-x) | 支持 RL 分子设计 | 期刊缩写统一 | 低 | 保留 |
| 40 | §4.6 | Rogers & Hahn, ECFP | 已确认真实且信息正确 | [10.1021/ci100050t](https://doi.org/10.1021/ci100050t) | 支持 Morgan/ECFP 表示 | 期刊缩写统一 | 低 | 保留 |
| 41 | §4.6 | Breiman, Random Forests | 已确认真实且信息正确 | [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) | 支持 RF | 期刊缩写统一 | 低 | 保留 |
| 42 | §5 E1 | Eriksson et al., TuRBO | 真实但书目信息有误 | [NeurIPS 官方检索](https://proceedings.neurips.cc/paper/2019/) | 支持 trust-region BO | 缺页码/精确 URL | 中 | 补完整官方元数据 |
| 43 | §5 E3 | Gómez-Bombarelli et al., molecular VAE | 已确认真实且信息正确 | [10.1021/acscentsci.7b00572](https://doi.org/10.1021/acscentsci.7b00572) | 支持连续分子表示 | 期刊缩写统一 | 低 | 保留 |
| 44 | §5 E3 | Maus et al., LOL-BO | 真实但书目信息有误 | [NeurIPS 官方页](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ded98d28f82342a39f371c013dfb3058-Abstract-Conference.html) | 支持局部潜空间 BO | 缺页码/官方 URL | 中 | 补完整元数据 |
| 45 | §5 E3 | Lee et al., CoBO | 真实但书目信息有误 | [NeurIPS 2023 官方站](https://proceedings.neurips.cc/paper_files/paper/2023) | 支持相关潜空间 BO | 缺卷/页码/精确 URL | 中 | 补官方 BibTeX 信息 |
| 46 | §5 E4 | Liu et al., Rectified Flow | 真实但书目信息有误 | [ICLR 官方页](https://iclr.cc/virtual/2023/poster/11266) | 支持 rectified flow | 缺官方 URL | 中 | 补 ICLR 官方链接 |
| 47 | §5 E4 | Dhariwal & Nichol, classifier guidance | 真实但引用不能充分支持正文论述 | [NeurIPS 官方检索](https://proceedings.neurips.cc/paper/2021/) | 支持图像扩散的 classifier guidance；不能直接证明分子潜空间 rectified-flow guidance 的有效性 | 会议格式统一 | 中 | 加更直接的分子/flow guidance 文献，并将此文献限定为启发来源 |
| 48 | §5 E4 | Gruver et al., guided discrete diffusion | 真实但书目信息有误 | [NeurIPS 2023 官方站](https://proceedings.neurips.cc/paper_files/paper/2023) | 支持蛋白质引导离散扩散；对小分子潜空间仅为邻近工作 | 缺卷/页码/精确 URL | 中 | 补官方元数据并限定适用语境 |

---

## 6. 参考文献格式问题与安全修订建议

### 6.1 适用于全部条目的统一规则

1. 初投稿 Fast Format 允许任意一致样式，但每条必须完整且含标题；因此“期刊名未缩写”本身不是初投致命问题。
2. 若按 ACS/JCIM 定稿样式，期刊名应使用 CASSI 缩写，例如 `J. Chem. Inf. Model.`、`Nat. Commun.`、`Nat. Catal.`、`J. Med. Chem.`、`Chem. Sci.`。
3. 年份通常加粗、卷号斜体、页码/文章号完整；DOCTYPE 内现有样式总体一致，但会议文献明显不完整。
4. DOI 统一写成 `https://doi.org/...` 或 `DOI: ...`，不要同一列表混用多种形式。
5. 软件、数据、网页必须有版本、发布者/组织、日期、持久链接和访问日期；数据应作为正式参考文献引用。
6. 不应使用 `others` 作为显示作者；按 ACS 当前长作者名单规则由引文管理器正确生成。

### 6.2 问题条目的当前写法与建议完整格式

以下格式仅使用已核实信息；作者应从官方 BibTeX 补齐仍缺的页码，不在本报告中猜造。

| 条目 | 当前核心写法 | 具体问题 | 建议写法 |
|---:|---|---|---|
| 5 | `Jumper ...; others. ... Nature 2021...` | `others` 不规范 | `Jumper, J.; Evans, R.; Pritzel, A.; et al. Highly Accurate Protein Structure Prediction with AlphaFold. Nature 2021, 596, 583–589. https://doi.org/10.1038/s41586-021-03819-2.` |
| 6 | `In Advances in neural information processing systems ...; 2022.` | 会议大小写、卷、官方链接不完整 | `Gao, W.; Fu, T.; Sun, J.; Coley, C. W. Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization. In Advances in Neural Information Processing Systems, Vol. 35, NeurIPS 2022, Datasets and Benchmarks Track; 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html.` |
| 8 | 仅标题与 2021 | 缺预印本编号/链接 | `Angelopoulos, A. N.; Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv 2021, arXiv:2107.07511. https://arxiv.org/abs/2107.07511 (accessed 2026-07-20).` |
| 10 | 仅标题与 2016 | 缺预印本编号/链接 | `Amodei, D.; Olah, C.; Steinhardt, J.; Christiano, P.; Schulman, J.; Mané, D. Concrete Problems in AI Safety. arXiv 2016, arXiv:1606.06565. https://arxiv.org/abs/1606.06565 (accessed 2026-07-20).` |
| 13 | 仅标题与 2018 | 缺预印本编号/链接 | `Frazier, P. I. A Tutorial on Bayesian Optimization. arXiv 2018, arXiv:1807.02811. https://arxiv.org/abs/1807.02811 (accessed 2026-07-20).` |
| 14 | NeurIPS 2020, Vol. 33，无页码 | 缺页码 | `Daulton, S.; Balandat, M.; Bakshy, E. Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. In Advances in Neural Information Processing Systems, Vol. 33; 2020; pp 9851–9864.` |
| 15 | NeurIPS 2021, Vol. 34，无页码/URL | 不完整 | `Daulton, S.; Balandat, M.; Bakshy, E. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. In Advances in Neural Information Processing Systems, Vol. 34; 2021. https://proceedings.neurips.cc/paper/2021/hash/11704817e347269b7254e744b5e22dac-Abstract.html.` |
| 17 | `International conference ... PMLR 162` | 大小写/系列写法不统一 | `Trabucco, B.; Geng, X.; Kumar, A.; Levine, S. Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization. In Proceedings of the 39th International Conference on Machine Learning; PMLR 162; 2022; pp 21658–21676. https://proceedings.mlr.press/v162/trabucco22a.html.` |
| 18 | `International conference ... PMLR 139` | 同上 | `Trabucco, B.; Kumar, A.; Geng, X.; Levine, S. Conservative Objective Models for Effective Offline Model-Based Optimization. In Proceedings of the 38th International Conference on Machine Learning; PMLR 139; 2021; pp 10358–10368. https://proceedings.mlr.press/v139/trabucco21a.html.` |
| 24 | `Landrum, G.; others. RDKit...` + 概念 DOI | 无版本、年份、访问日期 | 安全模板：`Landrum, G.; RDKit Contributors. RDKit: Open-Source Cheminformatics, version [实际版本]. Zenodo, [年份]. [该版本 DOI] (accessed 2026-07-20).` 作者必须从实际运行环境确定版本，不得沿用未知版本占位符提交。 |
| 25 | NeurIPS 2020, Vol. 33，无页码 | 不完整 | `Balandat, M.; Karrer, B.; Jiang, D. R.; et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. In Advances in Neural Information Processing Systems, Vol. 33; 2020. https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html.` |
| 33 | `International conference ... PMLR 80` | 大小写/系列写法 | `Kuleshov, V.; Fenner, N.; Ermon, S. Accurate Uncertainties for Deep Learning Using Calibrated Regression. In Proceedings of the 35th International Conference on Machine Learning; PMLR 80; 2018; pp 2796–2804. https://proceedings.mlr.press/v80/kuleshov18a.html.` |
| 37 | ICLR 2023，无 URL | 不完整 | `Xie, Y.; Xu, Z.; Ma, J.; Mei, Q. How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules. In International Conference on Learning Representations; 2023. https://iclr.cc/virtual/2023/poster/11769.` |
| 38 | TDC NeurIPS 2021，无 URL/版本 | 具体平台版本不明 | 保留论文条目并加官方 proceedings URL；Methods 另列 TDC 软件版本、oracle 名称、参数和访问日期。 |
| 42 | TuRBO NeurIPS 2019，无页码/精确 URL | 不完整 | 从 NeurIPS 官方 BibTeX 导入完整页码和 URL；在未确认前不要自行填页码。 |
| 44 | LOL-BO NeurIPS 2022，无页码/URL | 不完整 | `Maus, N.; Jones, H. T.; Moore, J. S.; Kusner, M. J.; Bradshaw, J.; Gardner, J. R. Local Latent Space Bayesian Optimization over Structured Inputs. In Advances in Neural Information Processing Systems, Vol. 35; 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/ded98d28f82342a39f371c013dfb3058-Abstract-Conference.html.` |
| 45 | CoBO NeurIPS 2023 | 缺卷/页码/精确 URL | 从 NeurIPS 2023 官方 BibTeX 导入；本报告不猜造页码。 |
| 46 | ICLR 2023，无 URL | 不完整 | `Liu, X.; Gong, C.; Liu, Q. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In International Conference on Learning Representations; 2023. https://iclr.cc/virtual/2023/poster/11266.` |
| 48 | LaMBO-2/论文标题为 Protein Design... | 当前正文称 LaMBO-2，参考标题未体现别名；会议信息不完整 | 以官方论文标题为主，并从 NeurIPS 2023 BibTeX 补卷、页码和精确 URL；正文首次出现时写 `LaMBO-2 (Protein Design with Guided Discrete Diffusion)`。 |

此外，所有 DOI 期刊论文应批量用 ACS CSL 重新渲染，主要变化是期刊缩写和标点，不需要改变已经核实的作者、标题、年份、卷页或 DOI。

---

## 7. 全文科学性、完整性与逻辑审查

### 7.1 研究范围与创新性

论文的问题设置与 JCIM 匹配：利用不确定度降低 learned-oracle exploitation，属于分子设计和化学机器学习方法研究。创新点“把 oracle 的 σ 同时作为 Pareto 目标和 GP 观测噪声”表达清楚，2×2 消融也围绕该点组织。

主要风险是创新性表述超过证据边界。将 `σ²` 写入 `train_Yvar` 本身在 GP/异方差建模中并非新原则；文章真正可辩护的创新应限定为：在特定的 inverse molecular design 与 externally calibrated oracle 场景中，将经过验证的 oracle error variance 作为 BO surrogate 的已知观测方差，并与风险目标组合。需要补相关异方差 GP/heteroscedastic BO 文献检索，避免 `to our knowledge` 建立在不完整综述上。

### 7.2 方法与统计前提

核心等式需要明确区分：

- 观测值：`y(x)=μ(x)`；
- 目标真值：`a_true(x)`；
- oracle 误差：`e(x)=μ(x)-a_true(x)`；
- 输入给 GP 的 `train_Yvar` 应代表 `Var[e(x)|x]` 或有明确定标关系的估计。

当前文本把一般的 `σ(x)`、epistemic uncertainty、tree variance、OOF residual predictor 混在一起，然后把 `σ²` 当成绝对误差方差。排序校准高并不等于量纲和方差校准正确；即使高 σ 点被下权，也不能自动推出后验均值“toward true activity”。应增加：校准映射、覆盖率/标准化残差检验、误差均值偏置、σ 缩放敏感性、错误 σ/反相关 σ 的负控制，以及对固定噪声 GP 假设的推导。

### 7.3 合成验证

合成基准中 `μ=a_true+trap(x)`、`σ=σ0+γ·trap(x)`，因此 σ 与 oracle error 基本由同一个函数决定，并在 §4.4 明示 Spearman ρ=1。该设计适合做机制 sanity check，但不足以证明现实稳健性。建议至少增加：

- ρ 从负值、0、0.2、0.5 到 1 的系统扫描；
- σ 存在全局缩放误差、局部失配、偏置或随机噪声；
- 高真值区域本身高 σ 的场景；
- trap 数量、宽度、幅度和维度变化；
- 非仿射和异方差误差；
- 明确失败边界，并将 E2 纳入核心而不是“preliminary”。

### 7.4 真实 kcat 数据与数据泄漏

“4.3k unique substrates”不足以说明样本单位。kcat 本质上依赖酶–底物对；如果同一底物出现在多个酶、同源酶跨分割或同一反应条件的近重复跨分割，按底物 canonicalization 后随机分组仍可能泄漏。必须说明：

- 原始数据来源、许可证和下载日期；
- kcat 原始单位、转换规则、条件和重复测量聚合；
- 酶序列/EC 号、底物、酶–底物对的唯一性定义；
- 同源蛋白、底物 scaffold、反应/酶家族级分组；
- 训练/验证/测试中所有实体交叠统计；
- “log10 kcat spanning ~13 units”的极端值、检测下限、单位混用与异常值处理。

当前 scaffold split 主要控制底物结构相似性，不能自动排除蛋白序列或酶–底物对层面的泄漏。

### 7.5 评价指标和统计报告

- 核心合成实验 5 seeds、真实生成和 QED 仅 3 seeds。Cliff's δ 在 n=5 时非常离散，不能替代置信区间或预先规定的统计检验。
- “significantly enriched”目前主要依据 3 个 seed 都大于 1 和 mean±sd，不足以构成严格显著性声明。
- 多个方法、多个指标、多个预算被同时比较，未说明多重比较、paired design、置信区间构造和 bootstrap 层级。
- AUC 0.651 的 CI 排除 0.5 可以支持代理模型有弱泛化信号，但不能直接证明生成方法产生高 kcat 分子。
- QED 是解析、平滑且较易拟合的属性，适合作为 no-regret 对照，不应称为强外部验证或“decisive evidence”。

建议核心结果至少 20–30 个独立 seeds，预先定义 primary endpoint，报告 paired bootstrap/permutation 或合适的层级模型、95% CI、校正后的 p 值，并公开每个 seed 的原始结果。

### 7.6 基线公平性与可复现性

需补足：所有方法是否使用同一初始设计、相同真实 oracle 查询数、相同候选生成次数、相同约束过滤、相同推荐 front/top-K 定义和相同超参数调优预算。qParEGO 表现异常低，必须核查参考点、目标缩放、采集优化重启数和实现是否公平；NSGA-II 的计算/候选预算也不能只按 oracle 次数笼统比较。

论文当前只给本地路径，例如 `algorithms/cu_mobo.py`、`results/*.csv`、`unified_train_mixed_phys.csv`，读者和审稿人无法访问。应冻结版本并公开：代码、环境锁文件、随机种子、原始/处理后数据、split 文件、每次运行日志、图表数据和复现命令。

### 7.7 结论边界与表述

优点是 §6 明确承认 in-silico 不等于实验验证。仍需修正以下过度表述：

- “real experimental kcat labels”只说明评估数据库标签是真实测量；生成候选本身没有 kcat 实测。
- “denoises toward true activity”没有充分统计前提。
- “QED is near-bijective in fingerprints”过强且无引文/证明。
- “dominant failure mode”“SOTA”“among the worst”“significantly”应有直接证据或降调。
- “well-calibrated learned oracles like KcatMoE/Boltz”缺引用支持；这些模型的论文并不自动证明输出满足本文需要的 σ 校准。
- “genuinely generalizes to unseen scaffolds”应改为“shows above-chance scaffold-split ranking on this dataset”，避免从 AUC 0.651 泛化过度。

---

## 8. 按严重程度排序的问题清单

| 严重度 | 文件及定位 | 原文/准确定位 | 问题类型 | 解释 | 可执行建议 |
|---|---|---|---|---|---|
| **致命** | `paper_jcim_clean.docx` p1；`paper_clean.md` L7 | `E-mail: to be provided` | 投稿完整性 | 通讯作者信息未完成，不能正式提交 | 填入有效机构邮箱并核对所有作者联系信息 |
| **致命** | 全文末尾，Conclusion 后直接 References | 无 Data Availability Statement | JCIM 强制项 | JCIM Level 2 要求投稿时提供 | 在 Associated Content 下加入可访问数据/代码的持久链接、版本和限制说明 |
| **致命** | 全文/仓库引用 | 仅有本地脚本和 CSV 路径 | 可验证性/软件数据要求 | 编辑和审稿人无法核验结果 | 发布匿名审稿仓库或公共仓库；接受前给 DOI/release |
| **重大** | Abstract p1–2；L13–17 | 约 392 词、11 句、3 段 | 官方格式 | 与 JCIM 强制 3–4 句简洁摘要不符 | 重写为单段 3–4 句，保留问题、方法、两项核心结果、边界 |
| **重大** | Related Work p4；L43 | `qEHVI / qNEHVI ... assume noiseless or homoscedastic` | 事实/引用错误 | qNEHVI 原文明确针对 noisy objectives | 区分 qEHVI/qNEHVI，并准确限定外部校准 σ 的差别 |
| **重大** | Abstract p1–2、Method p7；L15、66 | `posterior ... denoises toward true activity` | 方法有效性 | `train_Yvar` 的统计含义未建立，不能保证朝真值去噪 | 给生成模型、误差分解、校准映射、偏置假设和失配敏感性 |
| **重大** | §4.1 p8–11；L80、131 | `σ = σ0 + γ·trap(x)`、ρ=1 | 验证设计 | 结果在理想校准下几乎按构造成立 | 增加校准强度/失配/反相关/多 trap/高维鲁棒性实验 |
| **重大** | §4.3 p12–15；L107–127 | `leakage-controlled`、`unique substrates` | 数据泄漏 | 未说明酶–底物、蛋白同源性、单位和重复测量分组 | 独立 Data Curation 节；多实体 group/scaffold split 与交叠审计 |
| **重大** | §4.3、§4.6；L111–115、168–178 | 3 seeds；`significantly enriched` | 统计 | n 太小，未给正式检验/CI | 增加 seeds，定义 primary endpoint，paired CI/检验和多重比较 |
| **重大** | §4.3；L107–115 | `real experimental kcat labels` | 证据边界 | 实际是相似性恢复/代理排序，不是生成分子 kcat 测量 | 全文改为“retrospective label-based/similarity recovery”；不要暗示湿实验 |
| **重大** | §4.4 p15–17；L133 | `log10 kcat spanning ~13 units` | 数据质量 | 范围极端，可能涉及单位/异常值混合 | 列原始单位、清洗、截断、异常值、重复实验和分布图 |
| **重大** | §4.6 p18–21；L164 | `QED is near-bijective in fingerprints` | 无支持事实 | 无文献或定量定义，技术上过强 | 删除“near-bijective”，报告 RF 拟合误差/冲突率即可 |
| **重大** | §4.1–4.6 | qParEGO 很差、预算定义简略 | 基线公平性 | 可能受目标缩放/参考点/采集优化预算影响 | 发布统一配置和每方法预算；对 qParEGO 复核及敏感性 |
| **重大** | 2015 JCIM 编辑标准对照 | 单一主要真实任务，无独立 data curation | desk-rejection 风险 | JCIM 明确把多任务 benchmark 和 data curation 作为重要选择标准 | 增加至少一个非平凡真实任务/外部数据；建立 Data Curation 节 |
| **重大** | 标题页至 References | 缺 COI、基金、作者贡献、AI 使用确认 | 投稿/伦理完整性 | 至少 COI 为 ACS 发表所需；AI 使用需按实际披露 | 补 Associated Content、Author Information、Acknowledgments/COI |
| **中等** | §5 p21–24；L185 | 标题 `D1–D4`，正文 `E1–E4` | 内部不一致 | 版本命名错误 | 统一为 E1–E4 或 D1–D4 |
| **中等** | §5 p21–24 | 四个 preliminary extensions | 结构 | 分散核心故事，且 E3 AUC=1.000 可能显得自证/标签泄漏 | 主文保留最相关 E2，其他移 SI 或 Future Work |
| **中等** | §5 E3；L193 | 直接监督 latent，probe AUC 1.000 | 实验解释 | 直接监督同一标签后满分不等于下游 BO 泛化 | 严格独立 split、嵌套训练；降低论断强度 |
| **中等** | §5 E4；L195 | `independent real oracle`、2.3× | 方法描述 | 独立 oracle 的任务、数据和校准信息不足 | 明确是否 DRD2、来源、版本、训练交叠和统计 |
| **中等** | §6 p25–26；L210 | Design-Bench bucket 失效的操作细节 | 叙事/篇幅 | 日期、安装修复和 bucket 错误过多占据主文 | 移至 SI/复现说明；主文一两句即可 |
| **中等** | References p27–34 | 多条会议/arXiv/软件引用不完整 | 参考文献 | Fast Format 仍要求完整 | 依 §6 修订并用 ACS CSL 统一渲染 |
| **中等** | 表格 p9、p14–16、p19–20 | 列窄、换行严重，QED 表跨页断裂 | 版面 | 降低可读性，可能在转换中产生误读 | 横向表、缩短列名、拆表或移 SI；禁止跨页切断数据组 |
| **中等** | p3 Contributions | 项目符号渲染成行内短横线 | 版面 | Word 样式未保留真正列表 | 使用 Word 列表样式并重新导出检查 |
| **中等** | 全文 | 英美拼写混用 | 语言一致性 | `optimisation/optimization`、`generalises/generalizes` 等并存 | 统一美式英语，符合 ACS 常用风格 |
| **轻微** | 全文 | `SOTA`、粗体过多、宣传性措辞 | 写作风格 | 影响客观语气 | 减少粗体和排行榜式语言，突出可复核数据 |
| **轻微** | 候选 PDF 34 页 | 无页码/页眉 | 审稿便利性 | Fast Format 未明确强制，但定位不便 | 开启连续页码和行号 |

---

## 9. 投稿前必须完成的修改

1. 填写通讯作者真实邮箱，核对全体作者姓名、单位、ORCID 和投稿系统信息。
2. 把摘要改为简洁的 3–4 句单段摘要；避免把所有扩展实验塞入摘要。
3. 增加 `Associated Content`，至少包括 Data Availability Statement；提供代码、数据、split、原始结果和环境的公开/匿名可访问链接及版本。
4. 补利益冲突声明；确认并补作者贡献、基金、致谢和 AI 使用披露。
5. 修正 qNEHVI 事实错误以及“Boltz/KcatMoE 已校准 σ”等无支持表述。
6. 为 `train_Yvar=σ²` 建立严格误差模型和校准证据；把“向真值去噪”改成有条件的表述。
7. 增加 σ 失配/弱校准鲁棒性实验，并将失败边界作为核心结果。
8. 建立独立 Data Curation 章节；说明 kcat 单位、清洗、异常值、酶–底物任务定义和所有防泄漏分组。
9. 增加随机种子和正式统计；删除未被检验支撑的 “significantly”。
10. 明确真实 kcat 结果是回顾性相似性恢复/代理排序，不是生成分子的实验 kcat 验证。
11. 审计所有基线的相同预算、目标缩放、参考点、初始样本和采集优化设置。
12. 补齐所有会议、arXiv、软件和数据引用；统一参考文献格式。
13. 修正跨页表格、列表渲染和小字体；重新导出 PDF 全页检查。
14. 准备 cover letter；按投稿系统要求准备原创 TOC Graphic 和独立 SI。

---

## 10. 建议完成但不必单独阻碍投稿的修改

- 将 E1、E3、E4 和 Design-Bench 基础设施说明移入 SI，主文集中于 CU-MOBO、E2、合成鲁棒性和真实验证。
- 为稿件加入 5–8 个检索关键词。
- 将 `Honest boundaries` 改为更常见的 `Limitations`，减少反复使用 “honest”。
- 加一张“σ 校准强度—CU-MOBO 收益”总图，比多个零散扩展更能支撑主线。
- 报告推荐分子的结构、有效性、合成可行性分布和失败例，但明确仅为计算候选。
- 在图中同时使用颜色和线型/符号，确保色觉缺陷可读。
- 使用美式英语并进行专业语言校对；加入页码和行号方便审稿。

---

## 11. 尚需作者确认的信息

1. 通讯作者有效邮箱、ORCID、基金项目和利益冲突。
2. 是否使用过 ChatGPT、Copilot、Grammarly、生成式图像或其他 AI 工具；具体用途和范围。
3. 计划公开的代码/数据仓库、许可证、冻结版本和 DOI。
4. kcat 数据的原始来源、单位、下载日期、许可证、清洗脚本及 13-log-unit 范围的成因。
5. split 是否按酶序列/EC、底物 scaffold、酶–底物对同时隔离，而不只是底物 canonicalization。
6. E4 中“independent real oracle”的确切名称、训练数据、版本和是否与生成/代理训练数据重叠。
7. RDKit、BoTorch、PyTorch、pymoo、TDC 等实际软件版本。
8. 是否已有预印本或此前向 JCIM/其他期刊投稿；若有，需在 cover letter 说明。
9. 是否准备正式 SI；当前 `cu_mobo_improvements.md` 等开发文档不能直接视为 SI。
10. 是否能增加第二个非平凡真实任务或湿实验/外部合作验证。

---

## 12. 最终结论

**需要重大修改后再投稿。**

文章主题和稿件类型适合 JCIM，方法主线也具有潜在价值；参考文献经逐条核验后未发现虚假引用。当前阻碍不是单纯的 ACS 排版，而是“强制提交组件缺失 + 核心统计前提未证 + 理想化合成验证 + 真实任务与统计强度不足”的组合。若直接提交，存在较高的编辑退回或 desk rejection 风险。

优先级应是：先补投稿强制项和可访问代码数据；随后修正 `train_Yvar` 的统计论证、σ 失配实验、kcat 数据整理/防泄漏和正式统计；最后再做参考文献 ACS 化、TOC 图和版面优化。完成这些工作后，再根据新增真实验证强度判断是否达到 JCIM 的外审门槛。

