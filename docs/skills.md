# Skills：契约和下游服务

代码：[`skills.py`](../src/shopping_agent/skills.py)。skill 是给模型用的一层薄封装，真正的能力在下游服务里。

## 统一契约

所有 skill 都经过 `call_skill()`：

| 环节 | 做法 | 为什么 |
| --- | --- | --- |
| 参数 | pydantic 模型校验，失败直接返回 `invalid args`，不调下游 | 模型生成的参数不可信；参数错了重试也没用 |
| 超时 | 每次调用 `skill_timeout_ms`（默认 2s） | 下游慢不能拖死整条链路 |
| 重试 | 只读 skill（`idempotent=True`）超时或异常时重试 `skill_retries` 次；业务错误（商品不存在）不重试 | 加购、下单这类写操作重试会重复执行 |
| 返回 | `{"ok", "data" / "error", "latency_ms", "attempts"}`，失败不抛异常 | 编排层统一决定重规划还是兜底 |
| 观测 | 每次调用写一条 trace：参数、成败、错误、耗时 | bad case 回放、按 skill 统计时延和失败率 |
| 暴露给模型 | `Skill.schema()` 导出 function calling 格式 | 规则编排和 LLM 编排用同一套 skill |
| 契约基类 | `tooling.ToolSpec`：名字、描述、参数模型、是否只读、`schema()`、`validate()`。主链路 skill 和 ShoppingBench 沙箱工具（`bench/tools.py`）都继承它 | 加新工具只写参数模型和实现；校验、schema 导出、去重（只读才去重）不用重复写 |

## 每个 skill 对应的下游服务

| skill | 本项目的实现 | 生产环境的下游服务 | 关键设计 |
| --- | --- | --- | --- |
| `search_products` | 规则过滤 + 打分 | 商品搜索服务（倒排 / 向量召回 + 排序）、推荐服务 | 硬约束（品类、预算、排除品牌）下推到检索层过滤，不靠模型挑 |
| `filter_by_attributes` | 本地过滤 | SKU / 属性服务 | 属性值归一化（“XL” = “extra large”） |
| `get_product_detail` | 读商品库（可模拟 RPC 时延） | 商品详情服务（PDP） | 多个候选并行拉取；只返回白名单字段 |
| `verify_evidence` | 证据聚合 + 硬约束复核 | 价格服务（实时价、促销价）、库存服务、店铺服务（评分）、内容时效 | 实时字段（价格、库存）以详情为准，静态字段可以缓存 |
| `compare_products` | 按证据对比 | 详情 + 评论 / UGC 服务 + 物流时效服务 | 价格按详情价，不按列表价；至少两个商品 |
| `get_review_evidence` | 模拟评论上的 BM25 + 向量混合检索（RRF） | 评论 / UGC 服务、方面抽取（aspect）和情感模型 | 按卖点检索，好评差评聚合成 supported / disputed / no_evidence；返回片段 ID，推荐理由必须引用。见 [review_evidence.md](review_evidence.md) |

电商导购场景里一般还会有：

| 类别 | 下游服务 |
| --- | --- |
| 用户 | 用户画像服务、订单 / 浏览历史、购物车（加购是写操作，不能自动重试） |
| 营销 | 优惠券 / 营销服务（券后价，见 ShoppingBench 的 `plan_basket`） |
| 平台 | 内容安全 / 风控、特征平台、配置中心、日志埋点 |

这些服务一般用 Go / C++ 写，Python 的 agent 是入口层，负责理解、编排和决策。

## `compare_products` 的设计细节

1. **参数**：`product_ids` 至少 2 个，校验失败直接返回，`attempts = 0`，不调下游。
2. **价格口径**：调 `build_evidence()`，按详情页价格对比。列表价和详情价冲突时，以详情价为准，并在 `issues` 里标出 `price_conflict`。
3. **输出**：每个商品的价格、评分、商家评分、属性、库存、证据问题，外加“更便宜”和“评分更高”两个结论。结论由代码计算，不让模型自己比大小。
4. **失败**：某个商品不存在时整个 skill 返回 `ok: False`，编排层把路由改成 `fallback`，不会拿半份数据做对比。
5. **回复**：对比结果进模板回复。如果接了 LLM 润色（`rewriter`），润色后的回复要通过 `fact_check`，里面的每个价格都必须是证据层核验过的，对不上就退回模板。

## 迁移到地图场景

同一套契约可以直接套用：

| 导购 skill | 地图对应 | 下游服务 |
| --- | --- | --- |
| `search_products` | POI 检索 | 地点检索、周边检索 |
| `get_product_detail` | POI 详情 | 详情、营业状态、评论 |
| `verify_evidence` | 实时性校验 | 实时营业状态、排队、路况 |
| `compare_products` | 方案对比 | 路线规划（驾车 / 公交 / 步行）、ETA |
| — | 位置理解 | 地理编码 / 逆地理编码、绑路 |
