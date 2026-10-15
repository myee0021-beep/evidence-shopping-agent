# 长期记忆：L0–L3 分层

代码：[`agent_memory/`](../src/shopping_agent/agent_memory/)。分层思路参考了 [TencentDB Agent Memory](https://github.com/Tencent/TencentDB-Agent-Memory) 的“语义金字塔”，按购物场景做了取舍。没有使用它的代码。

## 原来的问题

原来的画像是“每轮把完整约束并进去”：

- **一次性需求污染画像**：“这次给我爸买，不要耐克”写进画像后，用户自己买鞋时也看不到耐克了；
- **沿用的约束被当成又说了一遍**：约束跨轮继承，每轮合并一次，无法区分“用户确认了两次”和“上一轮的约束还在”；
- **无法追溯**：画像里有“不喜欢小米”，但说不清是哪句话、什么时候说的。

## 四层结构

| 层 | 内容 | 怎么产生 | 存在哪 |
| --- | --- | --- | --- |
| L0 原始对话 | 用户原话 | 每轮归档（`archive.py`，`visibility="user"`） | `("archive", tenant, user)` |
| L1 原子事实 | “蓝牙耳机：不要小米” | 从**本轮约束 diff** 规则抽取 | `("memory", tenant, user, "facts")` |
| L2 场景块 | 每个品类一块 Markdown | 从 L1 派生 | 按需计算 |
| L3 用户画像 | 按品类：不喜欢的品牌、预算、属性 | 从 L1 聚合，达到确认阈值才进入 | `("profiles",)`，harness 预填约束时读 |

每条 L1 事实带：

```json
{
  "id": "f_67b0883c0c", "kind": "brand_dislike", "key": "brand_dislike|蓝牙耳机|小米",
  "value": true, "text": "蓝牙耳机：不要小米", "category": "蓝牙耳机",
  "scope": "long_term",           // 或 session：一次性需求
  "source_refs": ["utterance_…"], // 指向 L0 原话，可能多条（多次确认）
  "confirmations": 1,
  "status": "active",             // 或 superseded
  "supersedes": null              // 覆盖链
}
```

## 关键设计

**1. 只从 diff 里抽事实。** `understand` 节点合并约束后，立刻算 `diff_constraints(prev, merged)`，这时还没做画像预填和“更便宜的”推导。所以：
- 画像预填进来的品牌不算用户又说了一遍；
- “有没有更便宜的”推出来的预算不是用户原话，不进记忆。

**2. 作用范围。** 句子里有“给我爸 / 送人 / 这次 / 礼物”这类词时，本轮事实的 `scope = session`：只在本次会话可见，不进画像，换个会话就看不到。

**3. 确认阈值，防画像漂移。**

| 事实类型 | 进入画像需要的确认次数 | 理由 |
| --- | --- | --- |
| 明确不要某品牌 | 1 | 显式的否定偏好，说一次就该尊重 |
| 预算、属性、想要某品牌 | 2 | 可能只是这一次的需求 |

**4. 覆盖而不删除。** 同一个 key：值相同则确认次数 +1、追加 source_ref；值不同则新事实生效，旧事实标为 `superseded` 并保留，`trace_memory` 能沿覆盖链一路追溯到每一次的原话。“不要小米”之后又说“还是看看小米吧”，会生成一条 `value=false` 的事实覆盖它，画像里的“不喜欢小米”随之消失。

**5. 两路召回。**
- 请求前主动预取：新会话第一次出现品类时，读 L3 画像预填约束，同时 `search_memory` 取相关的 L1 / L2，记进 trace；
- 模型按需检索：`MEMORY_TOOLS` 里导出了 `search_memory`、`trace_memory`、`recall_context` 三个 function calling schema。

**6. 检索：BM25 + 向量，RRF 融合。** 品牌、型号这类精确词靠 BM25（中文按字和二元组切，不依赖分词器），“适合跑步的”这类模糊语义靠向量。RRF 只看名次，不用调两路分数的权重。默认向量是字符 n-gram 哈希，零依赖、结果确定；接真实 embedding 模型时传 `embed=` 即可。

## 和 TencentDB Agent Memory 的取舍

| | TencentDB Agent Memory | 本项目 |
| --- | --- | --- |
| L1 → L2 → L3 聚合 | LLM 抽取和聚合 | 规则聚合：零写入成本、确定、可测；购物偏好的结构是固定的（品牌 / 预算 / 属性） |
| 画像漂移 | 文档里谈得不多 | 作用范围 + 确认阈值 + 覆盖链 |
| 可追溯 | 不做不可逆压缩 | 同样：L3 字段 → L1 事实（provenance）→ L0 原话（source_refs） |
| Mermaid 任务画布 | 有，针对 20+ 次工具调用、有并行分支的长任务 | 不做：购物链路是固定几步，`AgentState` 已经承担了结构化导航 |
| 上下文卸载 | 有 | 有，见 [context_compression.md](context_compression.md)，和 L0 共用一套归档 |

它报告的提升数字（WideSearch、PersonaMem）是它自己的结果，和本项目无关。

## 局限

- 一次性需求的识别是关键词规则，“帮同事看看”能识别，“我老婆的生日快到了”识别不了。接 LLM 解析器后可以让它直接输出 scope；
- 预算的“相同值”是精确匹配，500 和 499 算两个值；
- 没有时间衰减。半年前的偏好和昨天的权重一样，后续可以在 L3 聚合时按 `updated_at` 衰减。
