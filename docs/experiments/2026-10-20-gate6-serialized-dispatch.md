# 2026-10-20 — Gate 6：串行 dispatch 的真实成本

## 问题

Gate 3–5 关掉了节点融合这条路，但留下一个数没测准：一次**串行** dispatch 到底要多久。

现有两个数对不上：

- exp3 的独立 dispatch 微基准：**1.32 µs / dispatch**（同 pass 内 N 次 dispatch，GPU 可以重叠）。
- Gate 2 从“删掉 93 次 dispatch、省了 0.72 ms”反推：**约 7.7 µs / removed dispatch**。

差了约 6 倍。如果 7.7 µs 是串行 dispatch 的真实成本，节点融合的天花板要按新的数重算；
如果不是，就得解释 7.7 µs 从哪来。

本轮只回答一个问题：**同 harness、同 device、同计时方法下，一次串行 dispatch 的斜率是多少。**

## 先写判据

在看到任何数字之前定下：

1. 两次完整测量的斜率差异 **> 10%** 的数字不用于决策。
2. 自检：dependent 斜率必须 **大于** independent 斜率；串行链的最终值必须命中递推期望值。
3. 负结果优先：如果串行成本不足以支撑任何 ≥10% 的路线，就写负结果并停止。
4. 测不到就写测不到，不用估算替代。

## 方法

`bench/browser/gpu-microbench.mjs` + `gpu-microbench.html`，扩展原来的 dispatch 微基准。

### 两种编码，同一个 device、同一个计时方法

| 模式 | 编码 | 能否重叠 |
|---|---|---|
| `one-pass`（独立） | 一个 compute pass 里 N 次 `dispatchWorkgroups` | GPU 可以重叠 |
| `dependent`（串行） | N 个 compute pass，每个 pass 一次 dispatch，全部读写同一个元素 | pass 之间有隐式内存屏障，不能重叠 |

WebGPU 保证前一个 pass 的写对后一个 pass 可见，所以“一个 dispatch 一个 pass”是
在 WebGPU 里表达串行依赖的最小单位。

`n-pass` 保留为历史对照：它和 `dependent` 走同一个 `encodePassPerDispatch`，
编码完全相同；`dependent` 只是多了链值验证。因此历史 `n-pass` 本来就是串行的——
这是结构上的同一，不是从两者之间很小的差异推断出来的。

计时方法沿用 exp3：`wall = encode + submit + onSubmittedWorkDone`。

- `gpuMs` 记录的是 **submit 返回到完成** 的时间，包含队列和驱动延迟；
  它不是 `timestamp-query` 的 GPU 执行时间。本轮没有启用 `timestamp-query`，
  因为 #3 要求的计时方法与 exp3 的独立数字一致，而 exp3 用的是 wall。
- 决策只用 wall 的 p05 通道，`gpuMs` 作为同向性的交叉检查。

### 串行链是可验证的

所有模式共用一个 shader，把元素做一遍顺序相关的递推：

```wgsl
out[0] = out[0] * 1664525u + 1013904223u;
```

递推常数在页面里只定义一次，同时用于 shader 和期望值计算，避免两边漂移。
每次 dependent 测量前把元素清零，测量后读回，和 `lcg(0, N)` 比较。
如果 dispatch 被并发执行、丢了更新，结果就不会命中。本轮 **600 / 600 个链点命中**。

### 两次测量怎么跑

- 扫 `0, 1000, 2000, 4000, 8000` 五个 count，斜率取最大的两个（4000 → 8000）。
- 60 次 repeat × 2 个 sweep，repeat-major：每一轮 repeat 完整跑一遍两个 sweep，
  sweep 顺序每轮翻转。这样 4000 和 8000 两个斜率点、以及两次 sweep，都在同一段
  机器状态下采样。
- 正式采样前先连续压 GPU **1 秒**并丢弃，避免测量期间时钟还在爬升。
- 每次采样取 **5 分位数（p05）** 作为决策通道：外部 GPU 负载只会让时间变长，
  所以快尾是污染最少的估计；p05 同时排掉极少数异常快的样本
  （严格最小值偶尔会抓到一次时钟抬升或驱动重叠的样本）。
  median 和 min 一并写进 ledger，便于看到干扰。
- `SWEEPS` 被限定为 exactly 2：agreement 检查比较的就是这两次，更多 sweep 会被忽略。

## 结果

机器：Apple M4（10 核 GPU），Chrome 155，headed，Metal 3。
canonical 运行开始时 loadavg 为 1.72（本轮最后一次运行；前面几次在 3–7 之间，
结果都记录在复现表里）。

### 斜率（µs / dispatch，两个最大 count：4000 → 8000）

| 模式 | sweep 0 | sweep 1 | 均值 |
|---|---:|---:|---:|
| `one-pass`（独立） | 1.064 | 0.992 | **1.028** |
| `n-pass`（历史对照） | 1.722 | 1.656 | 1.689 |
| `dependent`（串行） | 1.766 | 1.650 | **1.708** |

- 串行 / 独立 = **1.66×**，差值 **+0.680 µs / dispatch**。
- 两次 sweep 的一致性：dependent **6.8%**，independent **6.9%**，都在 10% 以内。
- `n-pass` 与 `dependent` 相差约 1%，与“两者编码相同”一致。

### 曲线是线性的

canonical 的 p05 原始点（ms，sweep 0）：

| count | one-pass | dependent |
|---:|---:|---:|
| 1000 | 1.18 | 2.02 |
| 2000 | 2.24 | 4.01 |
| 4000 | 4.18 | 7.28 |
| 8000 | 8.44 | 14.35 |

dependent 的平均成本：8000 处 14.35 / 8000 = **1.79 µs**，4000 处 7.28 / 4000 = **1.82 µs**，
与两点斜率 1.77 µs 吻合。斜率不是非线性曲线造成的假象。

### 跨三次独立进程的复现

| 运行 | dependent 均值 | independent 均值 | 比值 | 211 dispatch | 1112 dispatch |
|---|---:|---:|---:|---:|---:|
| canonical | 1.708 µs | 1.028 µs | 1.66× | 0.360 ms | 1.899 ms |
| run2 | 1.649 µs | 1.053 µs | 1.57× | 0.348 ms | 1.834 ms |
| run3 | 1.800 µs | 1.040 µs | 1.73× | 0.380 ms | 2.002 ms |

- 每次运行内的两个 sweep 都通过 10% 检查（dependent：6.8%、1.9%、1.1%；
  independent：6.9%、3.9%、5.5%）。
- 跨进程：dependent 在 **1.649–1.800 µs** 之间，跨度 **8.8%**；
  independent 在 **1.028–1.053 µs** 之间，跨度 **2.4%**。
  三次之间两两比较的最大差异：dependent **8.8%**，independent **2.4%**，全部 <10%。
- 比值因此报告为 **1.57–1.73×** 的区间，而不是一位小数。

## 重算节点融合天花板

取 canonical 的串行成本 1.708 µs / dispatch：

| 假设 | dispatch 数 | 上限 | 占 27.2 ms / token |
|---|---:|---:|---:|
| Gate 2 删掉的节点，**假设每个节点都是 dispatch** | 211 | **0.360 ms** | 1.3% |
| 一整个 decode step 的全部 dispatch | 1112 | **1.899 ms** | 7.0% |

**实测**：Gate 2 实际删掉的是 **93 次 dispatch**，不是 211 次：

```
93 × 1.708 µs = 0.159 ms ≈ 0.6% of 27.2 ms
```

这解释了 Gate 3 为什么拿不到 10%：实际删掉的那 93 次只值 0.6%。
即使把 1112 次 dispatch 全部抹掉，假设性上限也只有 7.0%——
这是“每个 dispatch 都值 1.708 µs”的上界，不是任何已实现的优化。

## 7.7 µs 不是串行 dispatch 的成本

**实测（本轮）**：串行 dispatch 约 1.7 µs，是独立 dispatch 的 1.6–1.7 倍，
不是 6 倍。211 次按 7.7 µs 算应该是 1.62 ms，实测上限只有 0.36 ms，低约 4.5 倍。

**推论（不是实测）**：Gate 2 反推出来的 7.7 µs / removed dispatch 不能解释为
GPU 的串行调度成本。它更可能来自 ORT 每个节点的 CPU 侧工作
（bind group 创建、参数上传、命令编码），或 Gate 2/3 对比里没有剥离的其他差异。
本轮没有单独测量这部分，所以这里只作为推论记录，不作为结论。

## 结论

1. **节点融合的上限被重新钉死**：211 次 dispatch 最多值 0.36 ms（1.3%）；
   实际删掉的 93 次值 0.16 ms（0.6%）。远低于 10% 的选择门槛。
2. **串行 dispatch 的真实成本约 1.6–1.8 µs**，不是 7.7 µs；
   独立与串行的差距约 0.6–0.8 µs / dispatch。
3. **负结果**：实测层面，基于“减少 dispatch 数”的路线（节点融合、pass 合并）
   拿不到 10%——实际删掉的 93 次只值 0.6%；假设删掉全部 1112 次的上界也只有 7.0%。
   下一轮不应再在 dispatch 数量上做文章。

## 不确定性与边界

- 机器在测量时有外部负载（本轮几次运行的 loadavg 在 1.7–7.1 之间，另有
  Chrome renderer 和 WindowServer 使用 GPU）。三次独立运行的 dependent
  结果跨度为 8.8%；报告给的是区间，不是一位小数。
- **独立基线不是无竞争基准**：`one-pass` 的 N 次 dispatch 在同一个 pass 内
  并发读写同一个地址，存在内存竞争。这是 exp3 独立基线的原始形状，本轮故意保留
  以便同形状对照；它适合回答“同样的单地址访问形状下，可重叠的 dispatch 有多快”，
  不是纯粹的无竞争独立内存基准。串行数字本身由链值验证，不依赖这个基线。
- **shader 变了**：所有模式的递推式从 `+1` 换成 LCG，为的是让串行链可验证。
  一次整数乘加相对一个 pass 边界可以忽略，但这意味着本轮的独立基线与 exp3 的
  1.32 µs 不是逐位可比的。exp3 的 1.32 µs 也只有 5 个 repeat、取 median，
  原始样本波动很大；本轮在同一 harness 内重新测了独立基线，比值用的是同一轮的两个数。
- 5 分位数是估计量，不是物理常数。它的作用是排除干扰和异常快样本，
  不代表 GPU 在空载时的理论峰值。
- 没有测 CPU 侧的每节点开销，所以“7.7 µs 来自哪里”仍然是一个待测问题。

## 复现

```bash
node bench/browser/gpu-microbench.mjs

# 调整参数
COUNTS=0,1000,2000,4000,8000 REPEATS=60 SWEEPS=2 \
  OUT=runs/2026-10-20-exp5-serialized-dispatch.json \
  node bench/browser/gpu-microbench.mjs
```

必须 headed（`HEADLESS=1` 可能拿不到 WebGPU）。端口默认 8125。

产物：

- `runs/2026-10-20-exp5-serialized-dispatch.json`：canonical，
  含每个 count、每个 sweep、每次 repeat 的原始样本（600 个链点）。
- `runs/2026-10-20-exp5-serialized-dispatch-run2.json`、`-run3.json`：两次复现。
- 分析面：`bench/analysis/ledger.mjs`（`dispatchSlope` / `agreement` /
  `fusionCeiling` / `dispatchModeComparison`）。
- 测试：`make test-js`。
