# BigVul 自建增强数据集：下一步计划与费用估算（只跑一份）

## 1. 目标与范围

- 目标：复现论文中 BigVul 在 VulnSC 框架下的增强流程。
- 当前约束：论文未开源 BigVul 增强版，需要自行构建。
- 本次执行范围（默认）：
  - 仅做 **1 份增强数据**（DeepSeek，`basic` 提示策略）
  - 仅生成一套可训练数据（不做多提示词、多 LLM 对照）
  - 后续先在一个 baseline 上验证（建议先 CodeBERT）

---

## 2. 下一步执行清单（按顺序）

### Step A. 获取 BigVul 原始数据

1. 下载 BigVul 原始数据与元信息（项目名、commit、函数级样本、标签）。
2. 放置到建议目录：
   - `data/origin/bigvul/`
3. 统一字段（至少包含）：
   - `id`
   - `project`
   - `commit_id`
   - `func_before` 或模型输入字段
   - `target`（0/1）

> 验收：统计样本总数、正负样本分布、缺失字段比例。

### Step B. 调用链检索与可调用函数提取

1. 依据 `project + commit_id` checkout 代码版本。
2. 对每个样本函数构建调用关系（直接/间接调用，建议设最大深度防止爆炸）。
3. 将被调函数源码整理为中间产物：
   - `data/intermediate/bigvul_called_funcs.jsonl`

> 验收：检索成功率（可检索样本占比）、平均每样本被调函数数。

### Step C. 用 DeepSeek 生成摘要（只跑 basic）

1. 使用固定 system/user prompt（basic 版，不加 behavior/oneshot/cot）。
2. 对每个被调函数生成结构化摘要：
   - `input: ... | output: ... | behavior: ...`
3. 失败重试策略：最多 2~3 次，超限记录失败样本。
4. 输出文件：
   - `data/enhance/bigvul/deepseek/basic/all.jsonl`

> 验收：
> - 生成成功率
> - 空摘要率
> - 平均输出 token

### Step D. 组装增强训练集并划分

1. 将摘要以注释形式拼接回原函数样本。
2. 生成 train/valid/test 三份（沿用论文风格 8:1:1）。
3. 建议路径：
   - `data/enhance/bigvul/deepseek/basic/train.jsonl`
   - `data/enhance/bigvul/deepseek/basic/valid.jsonl`
   - `data/enhance/bigvul/deepseek/basic/test.jsonl`

> 验收：样本数与标签分布和原始集一致（剔除失败样本需单独记录）。

### Step E. 先跑一个 baseline 验证闭环

1. 建议优先 CodeBERT（最快确认 pipeline）。
2. 对比：`none-enhance` vs `enhance-basic`。
3. 输出：`*_result.csv` + `*_time.csv`。

> 验收：至少得到一组可复现实验结果与日志。

---

## 3. 费用估算（DeepSeek，1 份增强）

## 3.1 估算公式

总费用约为：

`Cost = (InputTokens / 1e6) * P_in + (OutputTokens / 1e6) * P_out`

其中：
- `P_in`：输入单价（每百万 token）
- `P_out`：输出单价（每百万 token）
- 实际价格以 DeepSeek 当日官方计费为准。

## 3.2 只跑一份的量级假设

按 BigVul 增强后常见样本规模量级估算（约 `62,016`）：

- 样本数 `N = 62,016`
- 平均输入 token `Tin = 1,200 ~ 2,000`
- 平均输出 token `Tout = 80 ~ 160`

则总 token：
- 输入：`N * Tin ≈ 74.4M ~ 124.0M`
- 输出：`N * Tout ≈ 5.0M ~ 9.9M`

## 3.3 金额区间（人民币，粗估）

给出两档参考（你可按当日单价替换）：

- **低价档假设**：`P_in=¥1/M`，`P_out=¥2/M`
  - 费用约：`¥84 ~ ¥144`

- **中价档假设**：`P_in=¥4/M`，`P_out=¥8/M`
  - 费用约：`¥347 ~ ¥575`

> 结论：只跑一份增强集，预算可先按 **¥200~¥600** 预留；若平均输入更长或重试率高，预算上浮 20%~40%。

---

## 4. 风险与控制

- 调用检索失败：会导致样本被剔除或仅保留原函数。
- 长函数截断：可能丢失关键信息，影响增强质量。
- API 抖动与速率限制：需要重试与断点续跑。
- 成本失控：需先小样本试跑（如 500 条）估算真实 token/条。

---

## 5. 建议的最小执行策略（你现在就可以用）

1. 先抽样 `500` 条做试跑。
2. 记录真实 `avg_input_tokens/avg_output_tokens/成功率`。
3. 用真实值回填成本公式，再决定是否全量跑 `62k`。
4. 全量只做 `deepseek + basic` 一份，避免多分支烧钱。

---

## 6. 本文档的默认“只跑一份”定义

- 数据集：BigVul
- LLM：DeepSeek
- 提示策略：basic
- 产物：一套增强数据（train/valid/test）
- 实验：先在 1 个 baseline 上做 `none-enhance` 与 `enhance-basic` 对比
