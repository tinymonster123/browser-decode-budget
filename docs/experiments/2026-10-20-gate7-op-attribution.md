# 2026-10-20 — Gate 7：把 decode 时间按 op class 拆开

## 判据（先于结果）

1. 归因覆盖率必须 **≥95%**（按 GPU pass 时间计）；否则本 gate 判为
   「无法归因」，写负结果，不硬凑分类。
2. 每个数字必须来自 ledger；同一个 op 在 4 个 decode step 里的 share
   必须给出范围，不能只报一个数。
3. 实测、推论分开写；减法得到的差值不能当实测。

## 引用索引

| ID | Ledger | 本报告用到的字段 |
|---|---|---|
| L1 | `runs/2026-10-20-exp7-op-attribution.json` | `attributions.base.classes[]`、`results[0].runs[]` |
| L2 | `runs/2026-10-20-exp7-op-attribution-run2.json` | 同上（第二次运行） |
| L3 | `runs/2026-10-20-exp7-weight-bytes-by-op.json` | `rows[opType=MatMulNBits].bytes`、`sharedBytes` |
| L4 | `runs/2026-10-20-exp6-bandwidth.json` | `plateau.gbps` = 109.06 |
| L5 | `runs/2026-10-19-exp4-pass-profile.json` | LM head 2.701 ms |
| L6 | `runs/2026-10-19-exp3-dispatch-cost.json` | pass 边界 0.88 µs（wall-clock） |
| L7 | `runs/2026-10-19-exp2-dispatch-count.json` | 每 token 27.225 ms、1112 dispatch |

## 问题

Gate 6 之后剩下的一句话：**地板之上约 13 ms 到底是什么**。
Gate 5 有逐 pass 时间，但 pass 和节点没有名字；Gate 4 证明调度不是大头。
本轮只回答一个问题：这 27 ms 里，每个 op class 各占多少。

## 方法

### 为什么不能用 pass 内时间戳

`GPUComputePassEncoder.writeTimestamp()` 仍然需要
`chromium-experimental-timestamp-query-inside-passes`，本机 Chrome 155
的 adapter feature 列表里没有它；调用会抛
`TypeError ... requires the 'chromium-experimental-timestamp-query-inside-passes' feature`。
所以一个 pass 内部的多个 dispatch 无法逐个计时。

### 一 dispatch 一 pass

拦截 `beginComputePass`，返回一个代理 pass：

- `setPipeline` / `setBindGroup` 记下当前状态并转发；
- `dispatchWorkgroups` 转发后，**结束当前 pass、开一个新 pass**，把 pipeline 和
  bind group 回放进去；
- 每个 pass 的描述符带标准 `timestampWrites`。

于是得到 **1 pass = 1 dispatch**，pass 的起止时间戳就是这次 dispatch 的
GPU 时间。ORT 看到的仍是一个普通 pass encoder。

代价：多出 1112 个 pass 边界。按 Gate 4 的 pass 边界成本 `0.88 µs`
逐 dispatch 扣掉，得到一个修正值（这是估计，见交叉检查）；原始值和修正值
都写进 ledger。

### op class 从哪里来

- pipeline 描述符的 `label`：**空**。
- shader module 描述符的 `label`：**是 op 名**（`MatMulNBits`、`LayerNorm`、
  `Add`、`FlashAttentionDecodeQKV` …）。

所以 identity 用 shader module label；两者都没有时才退回 shader 代码哈希。
本轮 25 个 identity 全部是有名字的 op，覆盖率 **100%**。

## 结果

两次独立运行，base 模型，每次 4 个 decode step，每步 1112 个 dispatch：

- `runs/2026-10-20-exp7-op-attribution.json`（run1）
- `runs/2026-10-20-exp7-op-attribution-run2.json`（run2）

| op class | run1 µs/step | run1 share | run2 µs/step | run2 share | run1 逐步 share |
|---|---:|---:|---:|---:|---:|
| **MatMulNBits** | **18103.3** | **71.3%** | **17757.4** | **75.3%** | 66.5–76.7% |
| LayerNorm | 1825.2 | 7.2% | 1490.1 | 6.3% | 5.0–8.5% |
| MatMul | 1104.8 | 4.4% | 1029.7 | 4.4% | 3.3–4.9% |
| Mul | 663.5 | 2.6% | 441.9 | 1.9% | 1.5–3.7% |
| Add | 650.5 | 2.6% | 421.5 | 1.8% | 2.0–3.3% |
| Where | 473.2 | 1.9% | 417.9 | 1.8% | 1.8–2.0% |
| FlashAttentionDecodeQKV | 449.3 | 1.8% | 430.7 | 1.8% | 1.6–2.0% |
| Gelu | 401.2 | 1.6% | 287.2 | 1.2% | 0.9–2.1% |
| RotaryEmbedding | 388.0 | 1.5% | 307.6 | 1.3% | 1.1–1.8% |
| Transpose | 386.8 | 1.5% | 278.4 | 1.2% | 0.8–1.9% |
| 其余 15 类 | 934 | 3.7% | 707 | 3.0% | |

两次运行的 raw pass 时间之和：run1 **25.38 ms**、run2 **23.57 ms**。
按 0.88 µs/pass 扣掉边界后是 24.40 / 22.59 ms——这是估计，不是实测（见下）。

非 `MatMulNBits` 合计：run1 **7.28 ms（28.7%）**、run2 **5.81 ms（24.7%）**。

## 交叉检查

- **覆盖率**：两次运行都是 25 个 op identity、1112/1112 dispatch 有名字、
  0 个 mixed pass，named op 覆盖率 **100%**。判据 ≥95% 通过。
  shader 哈希/pipeline 序号只算 identity，不算命名，不计入覆盖率。
- **两次运行的一致性**：MatMulNBits 的时间 **18.10 vs 17.76 ms（差 1.9%）**；
  share 差 4 个百分点（71.3% vs 75.3%），因为小 kernel 的时间在两次之间波动
  （LayerNorm 1.83 vs 1.49，Mul 0.66 vs 0.44）。**排序两次完全相同**。
- **和 Gate 5 对撞**：本题 pass 时间之和 23.57–25.38 ms，Gate 5 是 20.3–26.5 ms，
  同一区间。逐 dispatch 拆分没有把总时间抬出原先的量级。
- **profiling 的 wall 不可用**：拆分 pass 后 wall 是 55–72 ms（pass 之间
  CPU/GPU 同步），所以只使用 pass 时间，不使用 wall。
- **逐步范围**：MatMulNBits 在 run1 是 66.5–76.7%，run2 是 70.5–79.7%；
  MatMul 两次都是 4.4%；attention 两次都是 1.8%。第一名很宽，第二名以后很稳。
- **pass 边界的扣除是估计**：0.88 µs 来自 Gate 4 的 **wall-clock** 斜率
  （one-pass 1.3225 vs n-pass 2.2025 µs），不是 GPU 时间戳。把它从 GPU
  时间戳区间里减掉混了两种时钟，所以 headline 用原始值；扣除后的值只用来
  说明小 kernel 被边界抬高了多少。它不会改变前两名的排序。
- **带宽对撞**：MatMulNBits 的 726 个唯一 initializer 共 **1.3178 GB**
  （`runs/2026-10-20-exp7-weight-bytes-by-op.json`），raw 时间
  18.10 / 17.76 ms → **72.8 / 74.2 GB/s = 109.06 GB/s 的 66.8% / 68.0%**。
  Gate 6 单独测的 LM head 是 **86.18 GB/s（79.0%）**。
  同一个 op、同一个模型，效率差约 11–12 个百分点。
- **权重口径（修正 Gate 6 的 roofline）**：`decoder_model_merged_q4f16.onnx`
  的 initializer 共 1,519,728,816 B，其中 MatMulNBits 消费 **1,317,820,416 B**
  （726 个唯一张量，没有共享）；差值约 201.9 MB 是 4 个共享的 rotary
  cos/sin cache（20 个共享张量，201,326,876 B）。
  出处：`runs/2026-10-20-exp7-weight-bytes-by-op.json`，工具
  `tools/weight_bytes_by_op.py`。
  RotaryEmbedding 实测只有 0.31–0.39 ms，说明它按位置 gather，不是每 token 全读。
  所以 Gate 6 按「全部 initializer」算出的 13.93 ms 地板偏高；按 matmul
  实际权重算，地板是 `1.3178 GB / 109.06 GB/s = 12.08 ms`（token 的 44.4%）。

## 结论

**这 27 ms 不是「13 ms 权重 + 13 ms 其他」。** 按 GPU pass 时间（两次运行）：

```
MatMulNBits（q4 权重矩阵乘）  18.10 / 17.76 ms   71.3% / 75.3%
其余所有 kernel               7.28 / 5.81 ms    28.7% / 24.7%
第二名：LayerNorm 1.83 / 1.49 ms（7.2% / 6.3%）
第三名：MatMul     1.10 / 1.03 ms（4.4% / 4.4%）
```

- MatMulNBits 是唯一的大头：**71–75%**，每次运行内部逐步 66.5–79.7%。
  这个区间的波动来自 GPU 状态和小 kernel 的调度，不来自 matmul 本身
  （两次运行的 matmul 时间只差 1.9%）。
- Gate 6 的 roofline 按全部 initializer 算「权重至少 13.93 ms」；按 matmul
  实际消费的 1.3178 GB 修正后是 **12.08 ms**。实测 matmul 用了
  **17.76–18.10 ms**，说明问题不是「多出 13 ms 的别的计算」，而是
  **matmul 自己只跑到带宽上限的 67–68%**。
- LayerNorm 是第二大 op（6.3–7.2%），242 次 dispatch；MatMul 4.4%，没有波动。
- 当前短上下文下 attention 相关约 3.2%（QKV 1.8% + Softmax 0.7% +
  CopyKVCache 0.4% + VxReduce 0.3%，各项独立相加）。上下文变长时这部分会增长，
  本轮没测。
- 27.2 ms 与 23.6–25.4 ms 之间约 2–4 ms 的差，是 pass 之间没有被时间戳覆盖的
  空隙/CPU 侧工作（跨运行相减，量级参考，不是实测）。

## 推论（不是本 gate 的路线选择）

- 如果所有 MatMulNBits 都达到 LM head 的 86.18 GB/s：1.3178 GB → 15.29 ms，
  按 raw 18.10/17.76 ms 算省约 **2.5–2.8 ms（token 的 9–10%）**。
- 如果达到硬件峰值 109.06 GB/s：12.08 ms，省约 **5.7–6.0 ms（21–22%）**。
- 两条都要求改 kernel/布局，不是改图；Gate 6 已证明图手术碰不到这里。

## 未回答

1. **为什么 242 次 matmul 的效率不一样**：没有逐节点核对 tile / workgroup /
   shape；只知道平均值和 LM head 的对比。
2. **长上下文**：benchmark 是短 prompt，KV cache 和 attention 的占比会变；
   本轮的数字不能外推到长上下文。
3. **`MatMul`（非量化）46 次 dispatch** 对应哪些图节点，没有确认。

## 复现

```bash
make bench-prepare
MODELS=base STEPS=4 node bench/browser/op-attribution.mjs
make weight-bytes
```

页面：`bench/browser/op-attribution.html`。
分析：`bench/analysis/ledger.mjs` 的 `attributeOps`（纯函数，有单测）。
