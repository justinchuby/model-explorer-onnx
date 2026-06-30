# ONNX 高层问答设计（Prompt API + 本地检索）

## 1. 目标

在现有 `web/` 静态站点基础上，新增一个**本地问答面板**，支持：

- 模型高层结构总结（架构层次、主要子图）
- 参数量统计（总量、按 namespace/op 分布）
- 节点来源与去向（`node from where` / lineage）

约束：

- 模型通常很大，不能直接把全量 onnx text 喂给 LLM
- 默认本地处理，不上传模型

---

## 2. 总体架构

采用三层设计：

1. **事实层（确定性计算）**
   - 从 ONNX/IR 图构建结构化索引（节点、边、initializer、namespace）
   - 预计算统计指标（参数量、节点数、算子分布、拓扑摘要）

2. **检索层（本地）**
   - 根据问题类型召回相关证据（Top-K chunk + 结构化结果）
   - 证据来源包括：模型摘要、namespace 摘要、节点 lineage 子图

3. **解释层（Prompt API）**
   - 只接收“问题 + 证据”，负责生成可读答案
   - 输出必须带引用（graph id / node id / namespace）

> 结论：LLM 不负责“全图计算”，只负责“解释”，可在大模型场景稳定运行。

---

## 3. 数据与索引设计

## 3.1 核心数据结构

- `ModelSummary`
  - `total_params`
  - `graph_count`
  - `node_count`
  - `op_type_histogram`
  - `namespace_stats`
- `NodeIndex`
  - `node_id -> {graph_id, op_type, namespace, attrs, inputs, outputs}`
- `EdgeIndex`
  - `producer_map` / `consumer_map`
- `ChunkStore`
  - 面向问答的文本 chunk（每块 0.5~2KB）
  - 记录 `chunk_id -> refs[]`

## 3.2 参数量计算

- 以 initializer/constant tensor 为主
- 支持按维度乘积计算元素数量
- 输出维度：
  - 全局总参数量
  - 按 namespace 聚合
  - 按 op 类型近似分摊（可选）

## 3.3 Lineage 计算

- `upstream(node_id, k_hops)`
- `downstream(node_id, k_hops)`
- 默认 `k=2`，防止结果爆炸

---

## 4. 问答流程

1. 用户提问
2. 规则路由（意图分类）：
   - `summary` / `params` / `lineage` / `compare`
3. 执行对应确定性查询
4. 组装 Prompt 输入（问题 + 证据 + 输出格式约束）
5. 调用 Chrome Prompt API 生成回答
6. 回答中引用 node/graph，可点击跳转 visualizer `selectNode(...)`

---

## 5. Prompt 约束（避免幻觉）

系统约束建议：

- 仅依据提供证据回答
- 若证据不足，明确说“不确定”
- 输出固定结构：
  1. 结论
  2. 证据引用（`graph_id`, `node_id`）
  3. 建议下一步（可选）

---

## 6. MVP 范围

第一阶段仅做三类问题：

1. `Summarize this model`
2. `How many parameters?`（总量 + Top namespace）
3. `Where does node X come from?`（2-hop upstream）

UI：

- 右侧 Chat 面板
- 问答引用可点击定位节点
- 展示“证据条数”和“是否使用 Prompt API”

---

## 7. 性能与可扩展

- 预计算索引放在 Worker，主线程只做展示
- 对超大模型采用分层摘要（graph -> namespace -> node）
- 限制单次 Prompt 的 token 预算（仅传 Top-K 证据）
- 后续可加：
  - 多轮上下文记忆
  - Agent 编排（复杂问题自动拆解）
  - Electron 本地持久化索引（IndexedDB/SQLite）

---

## 8. 里程碑建议

- **M1**：索引与统计 API（无 LLM）
- **M2**：Prompt API 接入 + 三类 MVP 问题
- **M3**：可点击引用、错误处理、性能优化
- **M4**：迁移到 Electron，支持本地文件长期索引
