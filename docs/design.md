# 设计说明

## 1. 状态设计

所有节点共享一份 `AgentState`（[state.py](../src/shopping_agent/state.py)），分三层：

| 层 | 字段 | 生命周期 |
| --- | --- | --- |
| 会话层 | `turn`、`messages`、`trace` | 整个会话，追加写（reducer = `operator.add`） |
| 任务层 | `constraints`、`shown`、`evidence` | 跨轮保留，按规则合并或覆盖 |
| 本轮 | `intent`、`focus`、`candidates`、`relaxed`、`replan_count`、`route`、`notes` | 每轮在 `understand` 里重置 |

`constraints` 是这个 Agent 的核心。它的合并规则见 `merge_constraints()`：

- 切换品类时整体重置（旧预算、品牌对新品类没有意义）；
- 预算本轮给了就覆盖；
- 排除品牌取并集，指定品牌覆盖，指定后从排除里移除；
- 属性取并集。

`shown` 记录上一次展示给用户的商品，是“第二个”“这两个”“更便宜的”这类指代的依据。

## 2. 路由

`judge` 只根据状态和证据做决定，不调用模型：

| 条件 | 路由 |
| --- | --- |
| 没有品类，或指代找不到商品 | `clarify` |
| 用户要对比 / 追问某个商品，且证据齐全 | `compare` / `detail` |
| 没有可用候选，且还有软约束可放宽、次数未用完 | `replan` |
| 没有可用候选，且无法再放宽 | `fallback`（给出忽略预算后最接近的商品） |
| 有可用候选，前 3 个的平均置信度 ≥ 0.75 | `recommend` |
| 有可用候选，但置信度低 | `caveat`（推荐，但明确提示信息可能不准） |

## 3. 证据与置信度

`verify_evidence` 为每个候选生成一条证据记录（[evidence.py](../src/shopping_agent/evidence.py)）：

- `verified_price`：详情页价格优先；
- `issues`：`price_conflict`、`out_of_stock`、`stale`（超过 30 天未更新）、`missing_seller_rating`；
- `coverage`：价格、库存、商家评分、属性、更新时间五个字段的覆盖率；
- `confidence = coverage − 0.3 × 价格冲突 − 0.3 × 信息过期`；
- `violations`：按详情页价格和库存复核硬约束。只要有一项违反，`usable = False`。

## 4. 重规划

只放宽软约束，顺序是 `features` → `brands_include`，每次放宽一个，最多 2 次。放宽只作用于本轮检索（`relaxed` 字段），不改用户的原始约束，下一轮会重新按原始约束来。每次放宽都会写进 `notes`，在回复里告诉用户。

## 5. 记忆

| | 会话记忆 | 用户画像 |
| --- | --- | --- |
| 实现 | LangGraph checkpointer（Redis / 内存） | LangGraph Store |
| key | `thread_id` | `("profiles", user_id)` |
| 存什么 | 整个 `AgentState` | 按品类：不喜欢的品牌、预算、偏好属性 |
| 什么时候读 | 每轮自动 | 新会话第一次出现某品类时 |
| 什么时候写 | 每个节点之后自动 | 每轮结束（`update_memory`） |

为什么会话状态放 Redis：每轮都要读写，对延迟敏感；可以设置 TTL，让过期会话自动清理。会话状态丢了，最多是用户再说一遍需求；用户画像这类长期数据不应该只放在 Redis 里。

## 6. 可观测性

`guarded()` 给每个节点写一条 trace：节点名、耗时、状态、关键输出（解析器、意图、约束、候选、被拒原因、路由、置信度）。每次 skill 调用也写一条：参数、成功与否、错误、耗时。

`trace.attribute()` 根据评测失败项和轨迹，把失败归到 `understanding` / `retrieval` / `tool_call` / `evidence` / `replan` / `routing` 中的某一阶段，评测结束后按阶段统计。

## 7. 为什么需求理解有两套解析器

- 规则解析器：确定、零成本、零延迟，作为基线，也是 LLM 失败时的兜底；
- LLM 解析器：覆盖口语表达。用“提示词 + JSON + pydantic 校验”而不是原生 function calling，是为了兼容任何 OpenAI 接口的模型；解析失败会把错误回传给模型重试一次。

两者输出同一个 `TurnParse`，后面的合并、指代消解、决策逻辑完全一致，所以可以用同一套评测直接对比。
