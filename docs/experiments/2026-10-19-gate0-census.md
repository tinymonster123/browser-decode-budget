# 2026-10-19 — Gate 0：q4f16 解码图普查

## 问题

`gemma-translate` 的 decode 实验把瓶颈定位到「每算子 dispatch / 同步」。
在写任何融合代码之前，先回答一个更便宜的问题：

> 这张图里还有多少节点是 GQA 可以吸收的？如果不到 10%，这个项目就不值得开始。

## 方法

- 读取 `decoder_model_merged_q4f16.onnx`，`load_external_data=False`，只分析图结构。
- 按节点名里的 `/model/layers.<n>/` 归层；按 `attn/GroupQueryAttention`、`attn/manual/`、
  `attn/shared/` 判定每层 attention 的实现方式。
- **保守可删** = `attn/manual/` 或 `attn/shared/` 下的节点，减去 `q_norm` / `k_norm`
  （保留在 GQA 外面），再减去 1（新的 GQA 节点本身）。
- **激进可删** = 不保留 q/k norm，全部折进 GQA；另外计入已有 GQA 层里可折叠的
  q/k norm + rotary。激进值只是上限，不作为决策依据。
- 时间投影用实测基线做线性外推，不在这里跑 benchmark。

模型：`onnx-community/gemma-4-E2B-it-ONNX`，`decoder_model_merged_q4f16.onnx`，
sha256 `73c0f1fe04f9a3a0…`，opset `"" 21` + `com.microsoft 1`。

## 结果

### 图规模

| 项 | 数值 |
|---|---:|
| 节点总数 | **1289** |
| shape-only 节点 | 85 |
| dispatch 候选（总数 − shape-only） | 1204 |
| `past_key_values.*` 输入 | 15 组（layers 0–14） |
| 顶层算子 | `MatMulNBits` 242、`SimplifiedLayerNormalization` 242、`Reshape` 153、`Mul` 109、`Add` 106、`Transpose` 71、`Gelu` 70、`RotaryEmbedding` 50 |

### 逐层分类

| 类型 | 层数 | 层号 |
|---|---:|---|
| 已融合 GQA | 12 | 0–3、5–8、10–13（全部是 sliding） |
| 手工 full-attention | 3 | 4、9、14 |
| shared-KV | 20 | 15–34 |
| final norm（非 block） | 1 | 35 |

### 可删节点

| 层组 | 每层核心节点 | 保留的 norm | 每层可删 | 小计 |
|---|---:|---:|---:|---:|
| manual full（4/9/14） | 27 / 26 / 26 | 6 | 20 / 19 / 19 | **58** |
| shared sliding（16 层） | 13 | 3 | 9 | **144** |
| shared full（19/24/29/34） | 11 | 3 | 7 | **28** |
| 合计（保守） | | | | **230（17.8%）** |

激进估计 404（31.3%），多出来的部分来自：

- 3 个 manual 层折叠 q/k norm：+18；
- 20 个 shared 层折叠 q/k norm：+60；
- 12 个已有 GQA 层折叠 q/k norm + rotary：+96。

### shared-KV 的捐赠关系

这是 Gate 2 的关键事实：

| 共享层 | K 来源 | V 来源 |
|---|---|---|
| 15–18、20–23、25–28、30–33（16 个 sliding） | `k_transpose(present.13.key)` | `present.13.value` |
| 19、24、29、34（4 个 full） | `layers.14/attn/manual/k_scores_transpose` | `present.14.value` |

也就是说，35 层里只有 15 层有自己的 KV cache；后 20 层分别复用 layer 13（sliding）
和 layer 14（full）的 KV。这个结构来自 Gemma 4 的 KV cache sharing（E2B 的共享比例
是 20/35）。

### 时间投影

基线：decode 8.50s（27.87 ms/token × 305 tokens）、prefill 2.60s、固定开销 0.20s，
合计 11.30s。隐含单节点成本 21.6µs。

| 场景 | 可删 | 新节点数 | decode | 端到端 | decode Δ | 端到端 Δ |
|---|---:|---:|---:|---:|---:|---:|
| 保守 | 230 | 1059 | 6.98s | 9.78s | −17.8% | −13.4% |
| 激进 | 404 | 885 | 5.84s | 8.64s | −31.3% | −23.6% |
| 10% 阈值 | 128 | 1161 | 7.66s | 10.46s | −9.9% | −7.5% |

## 这意味着什么

1. **保守估计通过 kill criterion**：17.8% > 10%，值得进入 Gate 1。
2. **收益几乎全部来自 shared-KV 层**（172/230，75%）。只融 3 个 manual 层的话，
   可删 58 个节点，约 decode −4.5%、端到端 −3.4%，基本在噪声内。
   所以 Gate 1 的价值主要是**验证融合器是否正确**，不是拿收益。
3. **Gate 2 才是决定性的**。如果 shared-KV 无法映射到 GQA，这个项目大概率只能得到
   一个负结果。
4. 激进值依赖 runtime 支持 q/k norm 折叠。GQA 文档明确说 JSEP WebGPU/JS 必须拒绝
   `q_norm_weight` / `k_norm_weight`，所以不要把 31.3% 当预期。

## 这不能证明什么

- **线性模型是假设**：`T ∝ N` 只在「每节点成本恒定」时成立。融合后的 GQA 可能更快，
  也可能因为 kernel 更复杂而更慢；Gate 3 才能回答。
- **shape-only 节点可能不 dispatch**：如果 85 个节点在 CPU 上执行，真实的单次 dispatch
  成本会比 21.6µs 高，可删比例也会更高。这里用总节点数做分母，是保守的。
- **shared-KV 的 GQA 映射未验证**：GQA 的 `key`/`value` 输入能接受外部张量，但它的
  cache 语义是按「每层自己的 past → present」设计的；把「已含当前 token 的共享 K/V」
  喂进去是否等价，Gate 2 的 prototype 才能回答。
- **浏览器 runtime 版本未测**：Python ORT 的 op 支持不等于 ort-web 的 op 支持。

## 决策

**PROCEED to Gate 1**，但把 Gate 1 的定位写清楚：

- 目标不是拿 4% 收益，而是产出一个**可验证的融合器**和一套**等价性检查**；
- 先在 layers 4/9/14 上跑通（它们有自己的 KV cache，语义最简单）；
- 产物必须过 `onnx.checker`，并在浏览器里与原始模型做同 prompt 输出对比；
- Gate 1 通过后，Gate 2 用 `/prototype` 先回答 shared-KV 的语义问题，再决定是否写
  完整融合器。

## 复现

```bash
make gate0
# 等价于：
# .venv/bin/python tools/inspect_graph.py \
#   --model ~/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_q4f16.onnx \
#   --config ~/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX/config.json \
#   --json runs/2026-10-19-gate0-census.json
# .venv/bin/python tools/dispatch_cost.py --census runs/2026-10-19-gate0-census.json
```

原始 census（含每层明细、算子计数、sha256）在
[`runs/2026-10-19-gate0-census.json`](../../runs/2026-10-19-gate0-census.json)。
