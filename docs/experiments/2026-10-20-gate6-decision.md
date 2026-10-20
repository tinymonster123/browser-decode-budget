# 2026-10-20 — Gate 6 决策报告

## 判据（先于结果）

在看到本轮任何结果之前确定：

1. 一条路线只有在 **token 级实测空间 ≥ 10%** 时才进入下一轮（#1 的 gate）。
2. 每个数字必须来自 ledger；同一次测量两次运行差 > 10% 的数字不用于决策。
3. 测不到就写测不到，不用估算替代。
4. 没有任何路线过门槛 → 报告是负结果，**本项目停止这条线**，不选路线。

## 结论

**没有路线过 10%。三条路线全部关闭。选中路线：无。Gate 6 是负结果。**

## 引用索引

| ID | Ledger | 本报告用到的字段 |
|---|---|---|
| L1 | `runs/2026-10-19-exp2-dispatch-count.json` | `results[].variants[].runs[]` 的 `decodeMsPerStep`、`perToken` |
| L2 | `runs/2026-10-19-gate0-census.json` | `graph.total_nodes`（节点计数也在 L1 的 `nodeCounts`） |
| L3 | `runs/2026-10-19-gate3-base-vs-gate2.json` 等 | `summary.<model>.medianMsPerToken` |
| L4 | `runs/2026-10-19-exp4-pass-profile.json` | `results[].runs[].segments[]`、`sumUs` |
| L5 | `runs/2026-10-20-exp5-serialized-dispatch.json`（+ `-run2`、`-run3`） | `comparison`、`ceilings`、`agreement` |
| L6 | `runs/2026-10-20-exp6-bandwidth.json`（+ `-run2`） | `plateau`、`agreement` |
| L7 | `runs/2026-10-20-lm-head-weights*.json` | `totalBytes`、`graphInitializerBytes` |
| L8 | `runs/2026-10-20-exp6-lm-head-utilization.json` | `results[].medianUs`、`gbps`、`percent` |
| G2 | GitHub issue #1 | 先前 spec 记录的约 7.7 µs / removed dispatch 反推值（不是本轮实测） |

## 路线清单

| 路线 | 状态 | 决定数字 | 引用 |
|---|---|---|---|
| 减少 dispatch 数（节点融合 / pass 合并） | **关闭** | 实测 base→gate2 **−1.9%**；GPU 侧 dispatch 成本上界 7.0%（全部 1112 次）；实际 93 次 0.6% | L1、L2、L3、L5 |
| LM head 带宽优化 | **关闭** | 已占读带宽 **79.0–82.3%**；整个 pass 占 token **9.9%**；剩余 **1.7–2.1%** | L4、L6、L7、L8 |
| dtype 比较（q4f16 → fp16） | **关闭（未运行）** | LM head 剩余约 **2.1%** token；fp16 字节数是记录值的 **3.46×** | L4、L7、L8 |

没有路线被选中。下面的“实测事实”和“推论”分开写，推论不参与选路。

## 实测事实

**F1. 一个 decode token 的基线。** base 模型每 token `27.225 ms`、
`1112` 次 `pass.dispatchWorkgroups`、`266` 个 compute pass、`86` 次 submit。[L1]

**F2. Gate 2 删掉的是节点，不是 dispatch。** base → gate2：节点 `1289 → 1078`
（`211` 个），但每 token dispatch 只从 `1112` 降到 `1019`（`93` 次）。[L1、L2]

**F3. Gate 3 的实测 decode 变化。** base → gate1 `−4.3%`，gate1 → gate2 `−2.5%`，
base → gate2 `−1.9%`（`+64` token 的 A/B 中位数）。[L3]

**F4. 串行 dispatch 的真实成本。** `1.708 µs / dispatch`，独立基线 `1.028 µs`，
比值 `1.66×`；两次 sweep 分别差 `6.8%` / `6.9%`；三次独立进程为
`1.649 / 1.708 / 1.800 µs`（最大两两差 `8.8%`）。[L5]

**F5. 按 F4 重算的融合天花板。** `211` 次 dispatch → `0.360 ms`；
全部 `1112` 次 → `1.899 ms`。[L5]

**F6. 实测读带宽上限。** `109.06 GB/s`（两次 sweep `109.03 / 109.09`，差 `0.06%`）；
复现运行 `108.52 GB/s`（差 `0.5%`）。扫描 `4–256 MB` 七档；
10% 带起点 canonical `8 MB`、复现 `16 MB`；平台值在 `108.5–109.1 GB/s` 之间。[L6]

**F7. LM head 的权重字节。** `232,783,872 bytes`（packed q4 `201,326,592`
+ scales `25,165,824` + zero points `6,291,456`）；base / gate1 / gate2 三个
artifact 分别记录，数值相同。整模型 initializer：base/gate1 `1,519,728,816 bytes`，
gate2 `1,519,728,944 bytes`（多 `128 bytes`）。[L7]

**F8. LM head 的实测时间与利用率。** 中位数 `2701.3 µs`（base）；
有效带宽 `86.18 GB/s`，占 `109.06 GB/s` 的 **`79.0%`**；
gate1 `82.3%`、gate2 `82.2%`。[L4、L8]

**F9. LM head 在 token 里的占比。** `2701.3 / 27225 = 9.92%`。[L4、L1]

**F10. Gate 5 的 pass 分布。** 每个 decode step 有 `17–28` 个 pass 超过 `300 µs`，
它们占该 step pass 时间总和的 `48–70%`。[L4]

**F11. fp16 的 LM head 字节数（由 F7 的 shape 直接算术）。**
`262144 × 1536 × 2 = 805,306,368 bytes`，是记录的 `232,783,872` 的 `3.46×`，
是 packed q4 `201,326,592` 的 `4×`。[L7、F7]

## 推论（不是实测，不参与选路）

**I1. Gate 2 的 7.7 µs 不是 GPU 串行成本。** 本轮实测串行 dispatch 是
`1.708 µs`（F4），比 issue #1 记录的约 `7.7 µs` 反推值低约 `4.5` 倍。[G2、L5]
差的约 `6 µs` 更可能来自 ORT 每个节点的 CPU 侧工作（bind group、参数上传、
命令编码），或 Gate 2/3 对比里没有剥离的其他差异。本轮**没有测**这部分。

**I2. 整模型的权重流量 roofline。** 如果每个 token 把全部 initializer 读一遍，
`1.5197 GB / 109.06 GB/s = 13.93 ms`，占 `27.225 ms` 的 **`51.2%`**。[L6、L7、L1]
这是**下限**：每 token 至少要读一遍权重，实际时间只会更高；
所以 51.2% 是权重流量的下界占比，不是上界。

**I3. 剩下的约 13 ms 不是权重流量。** 它分布在 attention/KV、200 多个小 pass、
pass 间隙和 CPU 喂数据里；Gate 5 只给出了逐 pass 时间，没有按 op class 归因。[L4]

**I4. fp16 不可能赢。** 由 F11，fp16 的 LM head 权重是 `805 MB`，
是记录值的 `3.46×`；其他层权重也会按同样比例增加。
LM head 已经在读带宽上限的 `79.0–82.3%`（F8），增加字节数的方向相反。[F11、L8]

**I5. 唯一能同时绕过“权重地板”和“非权重一半”的杠杆是减少 step 数**
（speculative decoding），但 #1 明确把它列为 out of scope。[I2、I3]

## 每条路线为什么关闭

### R1 — 减少 dispatch 数：关闭

- **实测路线结果**：`211` 个节点只对应 `93` 次 dispatch（F2）；
  Gate 3 的 base → gate2 实测是 `−1.9%`（F3）。这是 #1 要求的“实测空间”，
  低于 10%，路线关闭。
- **交叉检查（GPU 侧）**：按实测串行成本 `1.708 µs`，实际删掉的 `93` 次值
  `0.159 ms = 0.6%`；即使假设删掉全部 1112 次 dispatch，GPU 侧也只有
  `1.899 ms = 7.0%`（F4、F5）。
- **7.0% 是 GPU 侧 dispatch 成本的上界，不是总时间的上界。**
  I1 说的约 `6 µs/dispatch` CPU 侧差额没有测；如果它存在且可移除，
  理论上限会更高。但它没有形成任何实测路线；能用于选路的仍是 F3 的 `−1.9%`。

### R2 — LM head 带宽：关闭

- LM head 整个 pass 是 `2701.3 µs`，占 token 的 `9.92%`（F9）。
- 它已经用掉实测读带宽的 `79.0%`（F8）；剩余理论空间是
  `21.0% × 2.701 ms = 0.567 ms = token 的 2.1%`；gate1/gate2 约 `1.7%`。
- 即使 kernel 免费，token 也只能减少 `9.92%`，仍低于 10%。**这条路线同样在
  算术上就过不了门槛**，不是“优化得不够好”的问题。

### R3 — dtype 比较：关闭（未运行）

- #5 的条件是“LM head 显示出带宽余量”。F8 显示它已在 `79.0–82.3%`，
  token 级剩余约 `2.1%`，低于 10% 门槛。
- fp16 的 LM head 权重是 `805,306,368 bytes`，是记录值的 `3.46×`（F11），
  在已经接近带宽上限的 kernel 上只会更慢。
- 因此没有下载或运行 fp16 artifact。决定数字见 [L7、L8]；
  #5 的关闭记录在 issue tracker #5，不在 ledger 里。

## 选中路线

**无。**

没有路线达到 10% 的 token 级实测空间。按照 #1 的 gate，Gate 6 到此停止，
不进入下一轮优化。R1 的实测结果是 `−1.9%`（GPU 侧上界 `7.0%`），
R2 的 token 上界是 `9.92%`；两者都低于门槛，不是“还没调好”。
继续在同一批文件上做图手术或 kernel 微调不会改变结论。

## 未测到的部分（不是路线，不被选中）

这些是本轮明确没有回答的问题，记录在这里是为了不让它们被误当成已选路线：

1. **7.7 µs 的来源**：CPU 侧每节点开销没有单独测量（I1）。
2. **非权重约 13 ms 的归因**：没有按 op class 或 CPU/GPU 间隙拆开（I3）。
3. **prefill、speculative decoding、更换 runtime**：#1 明确 out of scope（I5）。

如果未来要开新 gate，它应该先测上面某一项，并重新写判据；本报告不替它选路。

## 附录：复算方式

- F1/F2：`node bench/browser/count.mjs` 的 ledger；`perToken` 是每 token 计数。
- F3：`node bench/browser/run.mjs` 的 A/B 中位数。
- F4/F5：`node bench/browser/gpu-microbench.mjs`；斜率取 4000→8000 的 p05 wall，
  天花板 = `usPerDispatch × dispatches / 1000`。
- F6：`node bench/browser/bandwidth.mjs`；平台 = 两次 sweep 的均值。
- F7：`.venv/bin/python tools/lm_head_weights.py`；每个 artifact 一个 JSON。
- F8/F9：`node bench/browser/lm-head-utilization.mjs`；LM head = 倒数第三个 pass，
  每模型取三个 step 的中位数。
- F10：Gate 5 ledger 的 `segments[].us` 按 `>300 µs` 求和，除以 `sumUs`。
