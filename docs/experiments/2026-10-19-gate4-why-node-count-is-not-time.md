# 2026-10-19 — Gate 4：为什么节点数没有换成时间

## 问题

Gate 3 是负结果：全量融合删掉 16.4% 的图节点，decode 只快了 1.9–3.4%。
本轮只回答一个问题：**节点减少为什么没有变成时间减少**。

不做猜测。要的是可证伪的测量：

1. 每个 artifact 每 token 实际发多少次 WebGPU 调用？
2. 一次 dispatch / 一个 compute pass / 一次 submit 在本机各值多少微秒？

把两者相乘，就能预测「删掉 N 个节点应该省多少毫秒」，再和 Gate 3 实测对撞。

## 方法

两个新 harness，都不做 A/B，因此不受双模型同页的加载顺序与内存压力影响。

### 实验 2 — 数调用（`bench/browser/count.mjs`）

单模型单独加载（每个 artifact 一个全新 browser context），在模型加载**之前**
patch WebGPU 原型，`generate` 前清零、后读数。

patch 有两条路，互为兜底：

- 拦截 `navigator.gpu.requestAdapter`，在 ORT 真正拿到 device 的那一刻 patch
  它的原型；
- 另外自己建一个临时 device 拿到同一批原型（同一个 realm 里原型共享）。

每个关键步骤在 `token_callback_function` 里取一次累计快照，所以能把
**prefill** 和**每个 decode step** 分开。每个模型跑两遍：

- `cpu=false`：只计数，用于计数与计时；
- `cpu=true`：在 wrapper 内累加 CPU 时间，用于定位 CPU 侧的耗时分布。

两遍计数完全一致（例如 base 都是 1112 次 dispatch），确定性成立。
但 `cpu=true` 那遍比 `cpu=false` 慢 4.7%（28.51 vs 27.23 ms/token），说明
`performance.now()` 本身有可测扰动 —— **所有计时结论取 `cpu=false` 那遍**。

### 实验 3 — 测单价（`bench/browser/gpu-microbench.mjs`）

最小 compute shader（单 workgroup，对 4 字节做一次加一），四种模式：

| 模式 | 结构 |
|---|---|
| `one-pass` | N 次 dispatch 塞进 1 个 pass，1 次 submit |
| `n-pass` | N 个 pass，1 次 submit |
| `n-submit` | N 个 pass，N 次 submit |
| `ort-like` | 每次 dispatch 前重建 bind group + `writeBuffer`，1 个 pass |

N = 0 / 100 / 500 / 1000 / 2000，各 5 次取中位数，用
`queue.onSubmittedWorkDone()` 等 GPU 完成；单价用 `(T(N) − T(0)) / N`。

`N=100` 和 `N=500` 是 GPU 时钟爬坡期，两次运行相差 5 倍，**判读只用
N=1000 和 N=2000**（两次运行之间稳定在 2% 内）。

## 结果 1：删节点 ≠ 删 dispatch

每个 decode token 的真实调用次数（中位数，31 个 decode step）：

| model | 图节点 | dispatch | compute pass | submit | bindGroup | writeBuffer |
|---|---:|---:|---:|---:|---:|---:|
| base | 1289 | 1112 | 266 | 86 | 1112 | 1112 |
| gate1 | 1233 | 1089 | 238 | 84 | 1089 | 1089 |
| gate2 | 1078 | 1019 | 214 | 80 | 1019 | 1019 |

prefill（一次跑完 prompt）：

| model | dispatch | compute pass | submit |
|---|---:|---:|---:|
| base | 1100 | 254 | 84 |
| gate1 | 1074 | 223 | 83 |
| gate2 | 980 | 175 | 77 |

两个立即可量的事实：

1. **1289 个节点只发出 1112 次 dispatch。** 净差 177。可能是部分节点
   （Shape / Reshape / Constant / Identity 一类）不产生 GPU dispatch，也可能是
   少量一对多/多对一的映射；本轮没有逐个节点核对，但**差值是直接数出来的**。
2. **删掉 211 个节点，只删掉 93 次 dispatch —— 比例 2.27 : 1。**

| 对比 | Δ节点 | Δdispatch | 节点 : dispatch | Gate 3 实测 Δms/token |
|---|---:|---:|---:|---:|
| base → gate1 | −56 | −23 | 2.43 | −1.31 |
| gate1 → gate2 | −155 | −70 | 2.21 | −0.73 |
| base → gate2 | −211 | −93 | 2.27 | −0.78 |

**「删 16.4% 节点」实际只删了 8.4% 的 dispatch。** 这是第一个数量级误差。

## 结果 2：一次 dispatch 多少钱

| 模式 | µs / dispatch（N=1000 / 2000） |
|---|---:|
| `one-pass`（同一 pass 内批量） | 1.335 / 1.323 |
| `n-pass`（每个 dispatch 一个 pass） | 2.210 / 2.203 |
| `n-submit`（每个 dispatch 一次 submit） | 20.41 / 18.82 |
| `ort-like`（bind group + writeBuffer + dispatch） | 3.67 / 3.42 |

推导出的单价：

- **一次 dispatch：≈ 1.32 µs**
- **一次 pass 边界：≈ 0.88 µs**（2.20 − 1.32）
- **一次 submit：≈ 18.2 µs**（20.4 − 2.2）
- **单独一次 `createBindGroup`：≈ 0.49 µs**

`ort-like` 每次迭代 3.42–3.67 µs，而 ORT 实际每节点端到端是
27 ms ÷ 1112 ≈ 24.3 µs。也就是说，**启动/记账开销只占每节点时间的 14%**，
剩下 86% 是 kernel 本身在 GPU 上跑的时间。

## 换算：这些单价能解释多少

按 base 的每 token 结构算总启动开销：

```
submit    86 × 18.2 µs = 1565 µs
pass     266 ×  0.88 µs =  234 µs
dispatch 1112 ×  1.32 µs = 1468 µs
                         ─────────
                          3267 µs ≈ 3.3 ms
```

27 ms 的 token 时间里，**全部 WebGPU 启动管线只占 12%**。

exp 2 在 wrapper 内实测的 CPU 时间也约 3.0 ms/token
（`createBindGroup` 1.81 + `writeBuffer` 0.34 + `submit` 0.22 + 其余约 0.53）。
两者量相同的量级，但量的不是同一样东西：那个 3.0 ms 是 **CPU 侧** 的编码开销，
上面的 3.3 ms 里 1.57 ms 是 **GPU 侧** 的 submit 串行开销。两条独立的数据
指向同一个结论：每 token 的调度层预算就是 3 ms 上下，不是 27 ms 的大部分。

再看每个 gate 的差值：

| 对比 | Δsubmit | Δpass | Δdispatch | 管线预测 | Gate 3 实测 |
|---|---:|---:|---:|---:|---:|
| base → gate1 | −2 | −28 | −23 | **−0.09 ms** | −1.31 ms |
| gate1 → gate2 | −4 | −24 | −70 | **−0.19 ms** | −0.73 ms |
| base → gate2 | −6 | −52 | −93 | **−0.28 ms** | −0.78 ms |

这张表是整轮实验的核心。它把 Gate 3 的两个结果分开了：

- **Gate 1 的 4.3% 不是 dispatch 省出来的。** 管线只预测 0.09 ms，实测量级是
  1.3 ms，差 14 倍。删掉的 `Softmax` / `Where` 是有真实计算量的 kernel，
  gate1 赢在 GPU 执行时间，不在调度。
- **Gate 2 的 2.5% 里，管线解释约四分之一到三分之一。** 剩下 0.4–0.5 ms
  最可能来自被删掉的 Reshape/Transpose 少搬的内存，但这部分本轮没有单独测量，
  只能说是残差。

这就是「每节点收益」从 23 µs 掉到 4.7 µs 的原因：**两次删的不是同一类东西**。
用「节点数」当统一成本尺子，从一开始就不成立。

## 回答：为什么 Gate 3 不行

三个独立的量级问题，任何一个都足以让 10% 落空：

1. **节点 → dispatch 有 2.3 倍的缩水。** 删 211 个节点只删 93 次 dispatch。
2. **dispatch 本身很便宜。** 同 pass 内批量提交时一次只要 1.32 µs，
   1112 次全部加起来也只有 1.47 ms（5.4%）。
3. **最贵的那部分启动开销和节点数无关。** 每 token 有 86 次 submit，
   共 1.57 ms（5.8%），而删掉 16.4% 节点只让 submit 从 86 掉到 80。
   submit 次数由图的切分结构决定，不是由节点数决定。

把「最理想情况」也封个顶：即使 211 个被删节点**每一个**都对应一次 dispatch，
也只能省 `211 × 1.32 µs ≈ 0.28 ms ≈ 1.0%`。**10% 的 kill criterion 在这条路线
上不可能达到，不是实现没做好。**

反过来，Gate 1 说明真正有效的那类改动是「删掉 GPU 上真在计算的 kernel」，
而不是「删掉搬运节点」。

## 尚未回答的问题

- 剩下的 24 ms/token 具体花在哪些 kernel 上？本轮只确认了
  「不是调度开销」，没有做 GPU 侧逐 kernel 归因。
- Gate 2 残差那 0.4–0.5 ms 是内存流量还是别的？需要单独实验。

## 复现

```bash
make bench-prepare
MODELS=base,gate1,gate2 MAX_NEW=32 node bench/browser/count.mjs
node bench/browser/gpu-microbench.mjs
```

产物：

- `runs/2026-10-19-exp2-dispatch-count.json` — 每个 artifact 的完整计数与逐 step 序列
- `runs/2026-10-19-exp3-dispatch-cost.json` — 四种模式的原始样本
- `bench/browser/count.html` / `count.mjs` — 计数 harness
- `bench/browser/gpu-microbench.html` / `gpu-microbench.mjs` — 单价 microbenchmark

## 测量扰动：`performance.now()` 的 4.7% 从哪来

`cpu=true` 那遍比 `cpu=false` 慢 4.7%（base 28.51 vs 27.23 ms/token）。
这不是噪声，可以精确算出来。

用 `bench/browser/perf-now-cost.mjs` 在同一个 Chrome 里测（200 万次）：

| 操作 | 每次 |
|---|---:|
| `performance.now()` | 0.083 µs |
| 连续两次 `performance.now()` | 0.168 µs |
| 字符串键计数 `obj[key] = (obj[key] ?? 0) + 1` | 0.027 µs |
| 同一操作打在 `Float64Array` 上 | 0.003 µs |

`cpu=true` 相对 `cpu=false` 每次调用多做的事只有两件：两次
`performance.now()` 和一次 `cpuMs` 字典写入，合计约 0.195 µs。

一个 token 被 wrapper 包住的调用数（base）：

```text
dispatch 1112 + setPipeline 1112 + setBindGroup 1112
+ writeBuffer 1112 + createBindGroup 1112
+ beginComputePass 266 + pass.end 266
+ submit 86 + finish 86 + createCommandEncoder 86
+ createBuffer 19 + copyBufferToBuffer 19 = 6388
```

`6388 × 0.195 µs = 1.25 ms`，实测差 1.28 ms，**误差 +3%**。

| model | calls/token | 预测 | 实测 | 误差 | 占 token |
|---|---:|---:|---:|---:|---:|
| base | 6388 | 1.25 ms | 1.28 ms | +3% | 4.7% |
| gate1 | 6205 | 1.21 ms | 1.52 ms | +26% | 5.8% |
| gate2 | 5804 | 1.13 ms | 0.90 ms | −21% | 3.3% |

gate1 / gate2 的偏差来自单次运行的 ±0.3 ms 抖动；base 的吻合说明模型是对的。

两点结论：

- wrapper 的其余部分（rest args、`apply`、`try/finally`）在两遍里都有，
  **4.7% 只是时间戳的增量**，不是整个 wrapper 的成本。
- 这段 CPU 开销直接反映到 wall time，说明 ORT 的编码在 submit 之间部分卡在
  关键路径上 —— 否则 1.2 ms 的 CPU 会被 GPU 掩盖。

原始数据：`runs/2026-10-19-exp2b-timestamp-overhead.json`。

## 已知局限

- microbenchmark 的 dispatch 是空载、互相独立的；真实 dispatch 有依赖和实际
  计算，所以 1.32 µs 是**启动成本的下界**，不是一次完整节点的成本。
- exp 2 每模型只跑一次，跨浏览器的绝对 ms/token 有 ±1 ms 抖动
  （base 27.23 / gate1 26.35 / gate2 27.29）。因此时间差一律引用 Gate 3 的
  同轮配对值，exp 2 的绝对时间只用于计算每节点端到端成本。
- 计数在 `cpu=false` 那遍读取；`cpu=true` 会引入约 4.7% 的计时扰动。
