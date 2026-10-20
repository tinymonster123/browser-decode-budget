# 2026-10-20 — Gate 8：小 matmul 的带宽效率

## 判据（先于结果）

1. 至少 **90%** 的 MatMulNBits 时间必须能对上已知的权重指纹（一组 initializer
   的字节大小）；否则本 gate 判为「无法按权重归因」，写负结果。
2. 两次独立运行；每个权重档位报 **GB/s 范围**，两次差 >10% 的档位必须标出来。
3. 一个 dispatch 对上 0 个或多个指纹都算未归因，不猜。

## 引用索引

| ID | Ledger | 用到的字段 |
|---|---|---|
| L1 | `runs/2026-10-20-exp8-matmul-efficiency.json` | `analyses.base.classes[]` |
| L2 | `runs/2026-10-20-exp8-matmul-efficiency-run2.json` | 同上 |
| L3 | `runs/2026-10-20-exp7-weight-bytes-by-op.json` | `fingerprints.entries[]` |
| L4 | `runs/2026-10-20-exp6-bandwidth.json` | `plateau.gbps` = 109.06 |
| L5 | `runs/2026-10-19-exp2-dispatch-count.json` | 每 token 27.225 ms |
| L6 | `runs/2026-10-20-exp7-op-attribution.json` | MatMulNBits 18.10 ms、`loadavg` 2.72 |
| L7 | `runs/2026-10-20-exp7-op-attribution-run2.json` | MatMulNBits 17.76 ms、`loadavg` 3.48 |

## 问题

Gate 7 把时间拆到 op class：MatMulNBits 占 71–75%，有效带宽只有峰值的
67–68%。但 242 次 MatMulNBits 不是同一种 kernel：LM head 读 232,783,872 B，
最小的只有 0.22 MiB。本轮回答：**有效带宽是否随每次 dispatch 读的权重
字节数变化；小 matmul 是不是拉低平均值的原因。**

## 方法

### 每次 dispatch 读了哪些 buffer

`op-attribution.html` 现在在 `createBindGroup` 时记下每个 bind group 绑定的
buffer 大小（storage 和 uniform 都记）；一 dispatch 一 pass 的 `flush` 把这些
大小按 index 展开（同一个 index 重绑只算一次，不同 group 保留重数）。

### 权重指纹

`tools/weight_bytes_by_op.py --op MatMulNBits` 对每个 MatMulNBits 节点收集它
消费的 initializer 字节，排序成指纹。726 个 initializer、242 个节点只产生
**9 个不同指纹**：

| 权重字节 | 节点数 | 权重 shape | 示例节点 |
|---:|---:|---|---|
| 232,783,872 | 1 | `[262144, 48, 16]` | `/lm_head/MatMul_Quant` |
| 21,823,488 | 20 | `[24576, 48, 16]` | `/model/layers.15/mlp/gate_up_proj/MatMul_Quant` |
| 10,911,744 | 35 | `[12288, 48, 16]` | `/model/layers.0/mlp/gate_up_proj/MatMul_Quant` |
| 7,956,480 | 1 | `[8960, 48, 16]` | `/model/per_layer_projection/MatMul_Quant` |
| 5,455,872 | 15 | `[1536, 192, 16]` | `/model/layers.0/mlp/down_proj/MatMul_Quant` |
| 3,637,248 | 14 | `[4096, 48, 16]` | `/model/layers.4/attn/q_proj/MatMul_Quant` |
| 1,818,624 | 56 | `[2048, 48, 16]` | `/model/layers.0/attn/q_proj/MatMul_Quant` |
| 454,656 | 6 | `[512, 48, 16]` | `/model/layers.4/attn/k_proj/MatMul_Quant` |
| 227,328 | 94 | `[256, 48, 16]` | `/model/layers.0/attn/k_proj/MatMul_Quant` |

出处：`runs/2026-10-20-exp7-weight-bytes-by-op.json` 的
`fingerprints.entries[]`（`bytes` / `nodes` / `dims` / `sample`）。

### 归因

一个 dispatch 的 buffer 大小多重集必须**包含**某个指纹的全部大小。
恰好一个字节数 → 归因；0 个 → unmatched；多个不同字节数 → ambiguous。

## 结果

base，4 个 decode step，242 dispatch/step，两次运行：

| 权重 MiB | 节点 | dispatch/step | run1 µs | run1 GB/s | run2 µs | run2 GB/s | GB/s 范围 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20.81 | gate_up（20 层） | 20.0 | 6051.8 | **72.1** | 6582.0 | **66.3** | 66.3–72.1 |
| 10.41 | gate_up（35 层） | 35.0 | 5256.0 | **72.7** | 5712.8 | **66.9** | 66.9–72.7 |
| 222.00 | LM head | 1.0 | 3505.0 | **66.4** | 3274.6 | **71.1** | 66.4–71.1 |
| 1.73 | q_proj | 56.0 | 2146.2 | **47.5** | 2163.9 | **47.1** | 47.1–47.5 |
| **0.22** | **k_proj** | **94.0** | 1995.4 | **10.7** | 1407.5 | **15.2** | **10.7–15.2** |
| 5.20 | down_proj | 15.0 | 1292.3 | 63.3 | 1278.9 | 64.0 | 63.3–64.0 |
| 3.47 | q_proj | 14.0 | 788.5 | 64.6 | 865.8 | 58.8 | 58.8–64.6 |
| 7.59 | per-layer projection | 1.0 | 114.8 | 69.3 | 142.3 | 55.9 | 55.9–69.3 |
| 0.43 | k_proj | 6.0 | 110.1 | 24.8 | 87.2 | 31.3 | 24.8–31.3 |

两次运行的 MatMulNBits 总时间：21.26 / 21.52 ms；总字节
1,317,820,416 B/step（1.318 GB），整体有效带宽 **62.0 / 61.3 GB/s**。

## 交叉检查

- **覆盖率**：两次都是 100%（0 unmatched、0 ambiguous）。
  判据 ≥90% 通过。归因不是按时间近似，而是每次 dispatch 的 buffer 大小
  对上指纹。
- **两次一致性**：9 个档位里 6 个的 GB/s 差 ≤10%（最大 9.8%）；
  1.73 MiB 档差 0.8%。**0.22 MiB 档差 42%**（10.7 vs 15.2），
  7.59 MiB 档差 24%，0.43 MiB 档差 26%。最小的档位噪声最大，但两次都
  远低于其余档位（10.7–15.2 vs 47–73）。
- **系统负载**：两次 loadavg 是 5.04 / 5.90 [L1、L2]。Gate 7 同一天的两轮是
  2.72 / 3.48 [L6、L7]，MatMulNBits 总时间是 18.10 / 17.76 ms；本轮是
  21.26 / 21.52 ms（高 17–21%）。**绝对值的差异与负载方向一致，但没有
  单独控制变量**，所以本轮只用同一轮内档位之间的相对关系，不跨轮比较绝对值。
- **和 Gate 5 的 LM head 不同**：Gate 5 在 batched pass 下测到 86.18 GB/s，
  本轮 split-pass 下是 66.4 / 71.1 GB/s。测量模式不同，绝对值不可直接比较；
  本轮只用同一模式下各档位之间的差距。

## 结论

**实测到的是：有效带宽随每次 dispatch 的权重字节数下降。**

- 权重 ≥3.47 MiB 的档位都在 **55.9–72.7 GB/s**；
- 1.73 MiB 的 q_proj 降到 **47.1–47.5 GB/s**；
- 0.22 MiB 的 k_proj 只有 **10.7–15.2 GB/s**，和最大档位差 **3.7–6.8 倍**。

这 94 次 k_proj 占 matmul 时间的 6.5–9.4%（1.4–2.0 ms），加上 56 次
q_proj（10.1%，2.15 ms），两个小档位是 150/242 次 dispatch。
**如果这两个档位达到 66 GB/s，可省约 1.70–2.27 ms（token 的 6.3–8.4%）**
（按 0.22 MiB 档 + 1.73 MiB 档分别重算，不含 0.43 MiB 档）。
这是推论，不是已选路线。

机制上，这更像**并行度/延迟受限**而不是带宽受限：0.22 MiB 的权重在
21.2 / 15.0 µs 内只搬了 0.22 MiB，远低于带宽墙，说明大部分时间不在等内存。
**但这只是解释，不是实测结论**：本轮没有测 workgroup 数、occupancy 或
kernel 内部结构。

## 未回答

1. **为什么小 matmul 只有 10–15 GB/s**：没有记录 workgroup 数/占用率；
   是并行度不足、固定启动成本，还是 grid 太小，未测。
2. **合并/批处理这 94 次 k_proj 是否有效**：这是下一步的候选，但没有测。
3. **0.22 MiB 档位的 42% 运行间波动**：需要更多重复才能给出可信区间。

## 复现

```bash
make bench-prepare
make weight-bytes
make bench-matmul-efficiency
# 第二次运行，保留第一份 ledger
MODELS=base STEPS=4 OUT=runs/2026-10-20-exp8-matmul-efficiency-run2.json \
  node bench/browser/matmul-efficiency.mjs
```

页面：`bench/browser/op-attribution.html`。
分析：`bench/analysis/ledger.mjs` 的 `attributeMatmuls`（纯函数，有单测）。
