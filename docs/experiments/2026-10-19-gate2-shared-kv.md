# 2026-10-19 — Gate 2：20 层 shared-KV → GQA

## 问题

Gate 1 之后，图里还剩 20 层 shared-KV attention。它们没有自己的 KV cache：

| 共享层 | K / V 来源 |
|---|---|
| 15–18、20–23、25–28、30–33（16 个 sliding） | `present.13.key` / `present.13.value` |
| 19、24、29、34（4 个 full） | `present.14.key` / `present.14.value` |

问题是：GQA 的 `past_key` + `key` 语义能不能表达「复用一个已经包含当前 token 的
共享 cache」？

## 方法

对每个 donor 只做一次 cache 切分：

```text
new  = cache[:, :, -S:, :]   -> reshape (B, S, D)   # GQA 的 key / value
past = cache[:, :, :-S, :]                          # GQA 的 past_key / past_value
S    = Shape(q)[1]
```

- 切分由 9 个节点组成（Shape、Gather、Neg、4×Slice、2×Reshape），**16 / 4 个层共用**；
- 每层只额外加 1 个 GQA；
- GQA 输入：`q_rotary` 输出、`new_k`、`new_v`、`past_k`、`past_v`、共享的
  `seqlens_k` / `total_sequence_length`、`position_ids`、`attention_bias`；
- GQA 属性：`num_heads=8`、`kv_num_heads=1`、`scale=1.0`、`causal=1`、
  `local_window_size=512`（sliding）或 `-1`（full）；
- 融合完 4 个 full 共享层后，layer 14 的 `fused_k_transpose` 不再被消费，删掉。

## 关键发现：GQA 的 present 输出

**CPU 端的 GQA kernel 只有声明了可选的 `present_key` / `present_value` 输出，
才会读取 `past_key` / `past_value`。** 只声明 `output` 时，它会静默忽略 past，
只算当前 token。

证据来自一个 tiny synthetic GQA（float32，past=1，new=1）：

| 节点输出声明 | 与手工 `past+new` attention 的 cosine |
|---|---:|
| 只有 `out` | 0.7669（实际退化成“只看 new”） |
| `out` + `present_key` + `present_value` | **1.0000**（max_abs 1.2e-7） |

因此 shared 层的 GQA 节点虽然不需要把 cache 暴露给图外，也必须声明两个 present
输出（名字唯一、不被消费即可）。

**这个行为是 CPU EP 的实测结果，WebGPU EP 必须重新验证**——这是 Gate 3 的第一项。

## 结果

### 产物

```text
models/derived/gemma4-e2b-q4f16-gate2/
├─ manifest.json
└─ onnx/decoder_model_merged_q4f16.onnx   # sha256 10bbfa636336997b…
```

- 节点数：**1233 → 1078（−155）**；相对原始 1289 是 **−211（−16.4%）**
- 层分类：**35 GQA / 0 manual / 0 shared-KV**
- 结构检查：无悬空输入、无重复生产者、ORT session 可创建

节点账：

| 组 | 层数 | 删除 | 新增 | 净省 |
|---|---:|---:|---:|---:|
| sliding（donor 13） | 16 | 160 | 25（16 GQA + 9 切分） | **135** |
| full（donor 14） | 4 | 32 | 13（4 GQA + 9 切分） | **19** |
| 删除未使用的 layer-14 transpose | | | | **1** |
| 合计 | 20 | 192 | 38 | **155** |

### 数值等价

`tools/compare_models.py`，CPU ORT：

| 对比 | 场景 | cosine | rmse/std | top5 | argmax |
|---|---|---:|---:|---:|---|
| Gate1 → Gate2 | decode S=1, past=4, seed=1 | **0.99999893** | 0.00378 | 5/5 | 一致 |
| Gate1 → Gate2 | prefill S=4, past=0, seed=7 | **0.99999261** | 0.00678 | 5/5 | 一致 |
| 原始 → Gate2 | decode S=1, past=4, seed=1 | **0.99999672** | 0.00981 | 5/5 | 一致 |

额外扫过的组合（Gate1 → Gate2）：S=1 past=0/2/4、S=2 past=0/2、S=4 past=0，
多个 seed 全部 PASS。唯一一个 top-1 互换出现在 S=4 seed=3，但 base 的 top-2
差距只有 0.023，top5 仍有 4/5，判定为 fp16 近并列。

### 累计投影

| 项 | 原始 | Gate 1 | Gate 2 |
|---|---:|---:|---:|
| 节点数 | 1289 | 1233（−4.3%） | **1078（−16.4%）** |
| decode 投影 | 8.50s | 8.13s | **7.11s（−16.4%）** |
| 端到端投影 | 11.30s | 10.93s | **9.91s（−12.3%）** |

Gate 0 的保守估计是删 230 个节点、decode −17.8%。Gate 2 实际删了 211 个，
达到估计的 **92%**；差的 19 个是切分节点（18）和 layer-14 transpose（1）。

## 这不能证明什么

- **CPU 等价 ≠ WebGPU 等价**。尤其是上面的 present 输出行为，必须重新验证。
- **投影不是计时**。切分节点是 2 个 donor × 9，真实 WebGPU 上有 dispatch 成本；
  Gate 3 才知道净收益。
- **翻译质量没测**。这里只比了 logits 和 top5，没有跑真实译文。
- **fp16 差异会累积**。三层融合时 cosine 0.999998，全部 35 层 GQA 后是 0.999997；
  仍在同一量级，但真实长文本需要 Gate 3 的 20 段对照。

## 决策

**Gate 2 PASS（CPU 数值等价）**，进入 Gate 3。

Gate 3 的顺序：

1. **浏览器等价性**：base / gate1 / gate2 三版在 transformers.js + WebGPU 下跑同一组
   prompt，先确认 WebGPU 的 GQA 也遵守 present 输出行为；
2. **配对计时**：同一 offscreen 会话，交替 A/B，4 轮中位数；
3. **kill criterion**：decode 提升 < 10% → 写负结果，停止。

## 复现

```bash
make gate2        # 从 Gate 1 artifact 生成 Gate 2
make gate2-check  # Gate1→Gate2 + 原始→Gate2 等价性
```

原始 ledger：

- [`runs/2026-10-19-gate2-census.json`](../../runs/2026-10-19-gate2-census.json)
- [`runs/2026-10-19-gate2-equivalence-decode.json`](../../runs/2026-10-19-gate2-equivalence-decode.json)
- [`runs/2026-10-19-gate2-equivalence-prefill.json`](../../runs/2026-10-19-gate2-equivalence-prefill.json)
- [`runs/2026-10-19-gate2-equivalence-cumulative.json`](../../runs/2026-10-19-gate2-equivalence-cumulative.json)
- [`runs/2026-10-19-gate2-manifest.json`](../../runs/2026-10-19-gate2-manifest.json)
