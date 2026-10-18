# 在 ShoppingBench 上跑 harness

代码：[`bench/`](../src/shopping_agent/bench/)。目的是把模拟的 39 个商品换成真实环境：[ShoppingBench](https://github.com/yjwjy/ShoppingBench)（AAAI 2026，Apache-2.0）提供的 250 万 Lazada 商品沙箱和 750 条带标注的测试任务（product / shop / voucher 各 250 条）。

## 对应关系

| 本项目 | ShoppingBench |
| --- | --- |
| `search_products` | 沙箱 `find_product`：价格区间、服务（LazMall / 包邮 / 货到付款 / 闪购）、店铺直接下推给沙箱过滤 |
| `get_product_detail` | 沙箱 `view_product_information`：SKU 选项、属性 |
| `verify_evidence` / `filter_by_attributes` | 本地逐项核验属性词，区分精确命中 / 只在标题里出现 / 缺失 |
| 硬约束（价格、品牌……） | 价格、服务、同店、用券后预算 |
| 重规划 | 换检索策略：补缺失属性词 → 翻页 → 缩短 query |
| 新增 `plan_basket` | 多件商品 + 优惠券的组合求解 |
| 推荐 | `recommend_product` + `terminate`，写成官方 rollout 格式 |

`compare_products` 在这三类任务里用不到，没有接入。

和主链路的一处区别：主链路重规划是**放宽软约束**，这里是**换检索策略、不放宽**。ShoppingBench 按属性逐项打分，放宽一个属性就直接丢分，所以宁可多搜几次。

## 子 agent、并行和循环护栏

```text
主 agent：understand ──> dispatch ──> align_shop ──> plan_basket ──> respond
                           │
              ┌────────────┼────────────┐      每件商品一个 ItemFinder 子 agent
         item-1         item-2       item-3    （retrieve -> gather_evidence -> judge -> replan）
              └────────────┼────────────┘      并行执行；共用沙箱和 CallGuard
                           ▼
               只交回候选列表 + 一行摘要（调用数、停止原因、最佳候选）
```

- **隔离**：每个子 agent 有自己的 trace 和 rollout 缓冲区，看不到别的商品的检索过程；主 agent 只拿候选列表做同店对齐和组合求解。trace 里每条事件带 `agent` 字段（`main` / `item-N`），失败归因能定位到具体哪件商品。
- **并行但可复现**：rollout step 先记成 (think, calls)，最后按商品顺序统一渲染，所以 `--workers 4` 和 `--workers 1` 产出的 rollout 完全一致（单测覆盖）。
- **循环护栏**（`guard.py`）：
  - 同一条任务里参数完全相同的沙箱调用直接复用结果，不再发请求，也不写新的 rollout step；
  - 重规划拿不出新的 (query, page) 就停，`judge` 里记 `stop_reason=no_new_strategy`。典型情况是 query 本来就不超过 4 个词，“缩短 query”之后还是原 query；
  - 每条任务真实调用沙箱的次数有上限（默认 40），用完就用已有候选收尾，`stop_reason=tool_budget_exhausted`；
  - 参数先过 `tools.py` 的 pydantic 校验（页码越界、空 query），不合法就在本地拦下。
- **关掉护栏 = 改造前的行为**：`--no-guard --workers 1 --max-tool-calls 0` 产出的 rollout 和改造前逐字一致，消融对比是干净的。

消融（离线夹具 7 个商品、9 条任务，解析结果用手写的 `tests/fixtures/bench_parse_cache.json` 模拟 LLM 抽出的属性词，沙箱模拟 30ms 时延）：

```bash
python eval/run_bench_ablation.py --demo --latency-ms 30
```

| 配置 | product 平均调用 | voucher 平均调用 | shop p50 | voucher p50 |
| --- | --- | --- | --- | --- |
| 完整 | 2.6 | 6.5 | 123 ms | 93 ms |
| 去掉护栏 | 3.2 | 9.5 | 122 ms | 153 ms |
| 去掉并行 | 2.6 | 6.5 | 213 ms | 202 ms |
| 都去掉（改造前） | 3.2 | 9.5 | 213 ms | 289 ms |

四组的推荐结果完全一致。夹具太小、没有标注，只能说明“省了多少调用和等待、没有改变决策”；真实效果要在 ShoppingBench 上用同一份 `--parse-cache` 跑四组，比较 ASR 和调用数。

## 评测模式（非交互、可复现）

| 参数 | 作用 |
| --- | --- |
| `--only N` | 只跑第 N 条（1 起，和日志编号一致），排查 bad case 用；产出文件名带 `_N` |
| `--offset` / `--limit` | 分段跑 |
| `--parse-cache FILE` | LLM 解析结果落盘，之后的重跑直接复用。改检索策略做消融时，各组用的是同一份解析，差异只来自被改的模块 |
| `--seed` | 传给 LLM 解析器（temperature 本来就是 0；不支持 seed 的服务会忽略） |
| `--max-tool-calls` | 每条任务的沙箱调用上限，0 = 不限 |
| `--workers` | 子 agent 并行数，1 = 串行 |
| `--no-guard` | 关掉去重和无进展停止 |
| `--latency-ms` | 给每次沙箱调用加固定时延，压测并行效果 |

summary 里新增 `efficiency`（平均 / 最大调用数、去重命中、预算拦截、p50 / p95 时延、子 agent 停止原因分布）和 `run`（git commit、开始时间、完整参数、解析缓存命中）。没有标注的任务不报 ASR，免得 0.0 被误读成全部失败。

## 运行

```bash
# 1. 准备 ShoppingBench（需要 JDK 21；商品库约 1.4GB，解压后 4.8GB）
git clone https://github.com/yjwjy/ShoppingBench ../ShoppingBench
cd ../ShoppingBench && git lfs pull
gunzip -c resources/documents.jsonl.gz > resources/documents.jsonl
./init_env.sh            # 建索引并在 127.0.0.1:5631 启动搜索服务
cd -

# 2. 跑 harness（建议先各 50 条）；解析结果落盘，之后的消融复用同一份
export LLM_API_KEY=...   # 任务解析用 LLM；不设就用规则解析
python -m shopping_agent.bench.run --bench-dir ../ShoppingBench --task product --limit 50 --llm \
    --parse-cache eval/results/shoppingbench/parse_cache_product.json
python -m shopping_agent.bench.run --bench-dir ../ShoppingBench --task voucher --limit 50 --llm \
    --parse-cache eval/results/shoppingbench/parse_cache_voucher.json

# 消融：护栏、并行（同一份解析）
python eval/run_bench_ablation.py --bench-dir ../ShoppingBench --task voucher --limit 50 \
    --parse-cache eval/results/shoppingbench/parse_cache_voucher.json

# 3. 用官方脚本复核（在 ShoppingBench 目录、它的环境里）
python src/agent/run_evaluate.py ../evidence-shopping-agent/eval/results/shoppingbench/official_config_product.json
```

不装 Java 时，可以用 `--offline resources/documents.jsonl --offline-limit 200000` 加载一个子集做离线沙箱。它只能用来验证链路，**数字和官方不可比**，因为候选池小了。

## 产出

`eval/results/shoppingbench/`：

| 文件 | 内容 |
| --- | --- |
| `rollout_<task>.jsonl` | 官方格式，每行是一条任务的 step 列表，可直接交给官方 `run_evaluate.py` |
| `official_config_<task>.json` | 对应的官方评测配置 |
| `trace_<task>.jsonl` | 解析结果、推荐、组合、逐节点 trace、得分、失败归因 |
| `summary_<task>.json` | ASR、GT 命中率、分字段匹配、失败阶段分布 |

本地评分（`metrics.py`）的口径和官方 `rewards/orm.py`、`run_evaluate.py` 一致，只有标题相似度一项不同：官方用 Qwen3-Embedding-0.6B；装了 `sentence-transformers` 就用同一个模型，否则退化成词级 Jaccard，并在 summary 里标为 `jaccard(approx)`。**最终数字以官方脚本为准。**

trace 和主链路同一格式，可以用 `shopping_agent.trace.replay` 回放：

```python
import json
from shopping_agent.trace import replay
rec = json.loads(open("eval/results/shoppingbench/trace_product.jsonl").readline())
print(rec["attribution"]); print(replay(rec["trace"]))
```

## 失败归因

`run.py` 的 `attribute()` 把每条失败任务归到一个阶段，和论文的失败类型对得上：

| 归因 | 含义 | 对应论文里的失败类型 |
| --- | --- | --- |
| `understanding(item_split)` | 多商品需求拆错了件数 | 意图理解 |
| `retrieval(no_candidate)` | 某件商品没有通过硬约束的候选 | 检索 |
| `evidence(attribute_mismatch)` | 推荐的商品还有属性没核验上 | 属性失配 |
| `constraint(same_shop)` / `constraint(budget)` | 同店 / 用券后超预算 | 约束未满足 |
| `decision(wrong_product)` | 都满足了但不是评测认可的商品（多半是标题相似度不够） | |

## 已验证到什么程度

| 验证 | 结果 |
| --- | --- |
| 优惠券规则解析（250 条 voucher 测试任务，和标注逐字段比对） | 250 / 250 |
| 价格约束解析（product 任务中带价格标注的） | 180 / 183 |
| 服务约束解析（product 任务中带服务标注的） | 89 / 92 |
| 多商品拆分（规则兜底，件数和标注一致） | shop 198 / 250，voucher 148 / 250 |
| 全部 750 条真实测试 query 跑通链路（夹具沙箱） | 0 异常，0 节点报错 |
| 产出的每个 rollout step 过官方 `Message` + `format_reward` | 全部 1.0 |
| 券后价格计算与官方 `eval_voucher` 一致 | 单测覆盖满减、折扣封顶、跨店不能用店铺券、原价已在预算内 |

**还没有真实的 ASR。** 商品库需要 Git LFS 下载 1.4GB，再用 Java 建索引，开发环境里跑不了。第一批数字需要按上面的步骤在本地跑。拿到数字后，可以和论文主结果表里 GPT-4.1 等模型在对应意图上的 ASR 对照（以论文为准）。

规则兜底在多商品拆分上只有 59%–79%，所以 shop / voucher 任务一定要带 `--llm` 跑。

## 预计最先暴露的问题

1. **属性同义改写**：用户说 “crimson”，商品属性写 “red”；用户说 “XL”，SKU 写 “extra large”。现在的核验是字符串匹配。看 trace 里 `gather_evidence.best.missing` 就能统计这类失败占比。
2. **只在标题里命中**：`weak` 命中在官方评测里不得分。如果这类占比高，说明应该优先选属性字段命中的候选，而不是标题相似的。
3. **同店检索的预算**：`align_shop` 最多试 3 个店铺，每个店铺每件商品一次检索，4 件商品时调用次数会明显上升，需要看时延。

这几条都是跑完第一批数字之后、按 trace 统计出来再决定修哪个，修完做消融表（比如“+ 属性同义词表：sku & attrs 匹配 x → y，ASR a → b”）。
