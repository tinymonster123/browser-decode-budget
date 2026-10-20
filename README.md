# browser-decode-budget

浏览器里 LLM decode 的时间账。项目从「减少 WebGPU dispatch 能不能把 decode 压下来」
这个问题开始；Gate 6 用实测负结果关闭了这条路线（tag `gate6-closed`）。
Gate 7 把 27.2 ms 按 op class 拆开：**71–75% 是 q4 矩阵乘，其余 kernel 25–29%**。
Gate 8 再按权重字节拆：**带宽随权重变小而下降**，0.22 MiB 的 k_proj 只有
10.7–15.2 GB/s，把 matmul 的平均值拉到 62 GB/s。
Gate 9 修正了这个读法：同一个字节数混合了不同形状；按
`(字节, workgroup)` 拆开后，两个候选（小 kernel 并行度、大 kernel 带宽）
的**上界都低于 10%**，优化线到此关闭。每次 dispatch 有
**12.4–13.1 µs 固定成本**（两轮拟合范围），小 kernel 的慢主要是摊不薄它。

从 `gemma-translate` 的 decode 实验中抽出。那边看到的是 29.9 ms/token；本项目
先证伪了「瓶颈在每算子 dispatch」这个假设，再转向成本归因。

## 状态

| Gate | 问题 | 状态 | 结论 |
|---|---|---|---|
| **0** | 还有多少 dispatch 可删？ | ✅ 2026-10-19 | **PROCEED**：保守可删 230/1289 节点（17.8%） |
| **1** | 3 层 full-attention 能否换成 GQA？ | ✅ 2026-10-19 | **PASS**：1289→1233，logits cosine 0.999998 |
| **2** | 20 层 shared-KV 能否映射到 GQA？ | ✅ 2026-10-19 | **PASS**：1233→1078，35/35 层全 GQA |
| **3** | 浏览器里真的更快吗？ | ❌ 2026-10-19 | **FAIL**：只快 1.9–3.4%，低于 10% 门槛；按 kill criterion 停止 |
| **4** | 删节点为什么没换成时间？ | ✅ 2026-10-19 | 一次 dispatch 1.32 µs；1112 次全删也只有 5.4% |
| **5** | 27 ms 到底花在哪？ | ✅ 2026-10-19 | 最贵的 1 个 pass 占 13%；LM head 2.6 ms 只发 1 次 dispatch |
| **6** | 还有没有 ≥10% 的路线？ | ❌ 2026-10-20 | **三条路线全关，无 frontier ticket**；见 `gate6-closed` |
| **7** | 27 ms 按 op class 怎么分？ | ✅ 2026-10-20 | **MatMulNBits 71–75%**；其余所有 kernel 合计 25–29% |
| **8** | 242 次 matmul 效率为什么不同？ | ✅ 2026-10-20 | 带宽随权重字节下降：0.22 MiB 的 k_proj 只有 **10.7–15.2 GB/s** |
| **9** | 哪个缺口是真的？ | ❌ 2026-10-20 | **两个候选都 <10%，优化线关闭**；每次 dispatch 固定成本 **12.4–13.1 µs** |

Gate 0 的完整报告见 [`docs/experiments/2026-10-19-gate0-census.md`](docs/experiments/2026-10-19-gate0-census.md)，
Gate 1 见 [`docs/experiments/2026-10-19-gate1-manual-fusion.md`](docs/experiments/2026-10-19-gate1-manual-fusion.md)，
Gate 2 见 [`docs/experiments/2026-10-19-gate2-shared-kv.md`](docs/experiments/2026-10-19-gate2-shared-kv.md)，
Gate 3 见 [`docs/experiments/2026-10-19-gate3-browser.md`](docs/experiments/2026-10-19-gate3-browser.md)。

## 假设与 kill criterion

**假设**

```
T_token ≈ N_dispatch × C_dispatch + T_compute
```

batch=1 decode 时 `T_compute` 很小；`C_dispatch` 在 Metal 上是 32–71µs（公开测量）。
如果用 GQA 把一层里十几个小算子合成一个 kernel，时间应随节点数近似线性下降。

**Kill criterion（写在最前面，避免实验室变成长期项目）**

- Gate 0：可删节点 < 10% → 停止；
- Gate 1：图非法，或浏览器输出发散 → 停止；
- Gate 2：GQA 语义无法表达 shared-KV → 只保留 Gate 1 的收益；
- Gate 3：decode 提升 < 10% → 写负结果，停止。

## 基线

| 项 | 值 | 来源 |
|---|---:|---|
| 模型 | Gemma 4 E2B，35 层，hidden 1536，8 Q heads / 1 KV head，head_dim 256 | `config.json` |
| 图节点 | **1289**（其中 85 个 shape-only） | Gate 0 census |
| dtype | q4f16（`MatMulNBits`） | 模型文件 |
| decode | 27.9–29.9 ms/token | `gemma-translate` decode 实验 |
| prefill | 2.5–2.6 s（KV 前缀复用后） | 同上 |
| 端到端 | 10.9–11.3 s / 305 tokens | 同上 |

## Gate 0 结果（摘要）

| 项 | 数值 |
|---|---:|
| 已融合 GQA 层 | 12 / 35（全部是 sliding layers） |
| 手工 full-attention 层 | 3（layers 4 / 9 / 14） |
| shared-KV 层 | 20（layers 15–34） |
| 保守可删节点 | **230（17.8%）** |
| 激进可删节点（折叠 q/k norm + rotary） | 404（31.3%） |
| 隐含单节点成本 | 21.6 µs |
| 保守投影 decode | 8.50s → **6.98s（−17.8%）** |
| 保守投影端到端 | 11.30s → **9.78s（−13.4%）** |

保守估计只保留「GQA 替换分解 attention 核心」，q/k norm 和 rotary 仍留在外面。
激进估计把 norm/rotary 也折进 GQA，但它依赖 runtime 支持：GQA 文档说
`q_norm_weight` / `k_norm_weight` 只有 CUDA 和 native WebGPU EP 支持，
**JSEP WebGPU/JS 必须拒绝**。

## Gate 1 结果（摘要）

产物：`models/derived/gemma4-e2b-q4f16-gate1/`（manifest + decoder，sha256 `2120f3d9…`）。

| 项 | 数值 |
|---|---:|
| 节点数 | 1289 → **1233（−56，−4.34%）** |
| 层分类 | 15 GQA / 0 manual / 20 shared-KV |
| decode 等价（S=1, past=4） | cosine **0.99999774**，rmse/std 0.00677，top5 5/5，argmax 一致 |
| prefill 等价（S=4, past=0） | cosine **0.99997830**，rmse/std 0.01097，top5 5/5，argmax 一致 |
| 单融该层 present 误差 | layers 4 / 9 / 14 分别为 **0 / 0 / 0** |
| 投影收益 | decode 8.50 → 8.13s（−4.3%），端到端 11.30 → 10.93s（−3.3%） |

两个导出细节：layer 4 的 `manual/mask/pos_unsq` 被 `shared_donor.13` 复用，保留；
layer 14 的 `k_scores_transpose` 被 4 个 shared full 层复用，用一个
`Transpose(present.14.key)` 重新产出。净省因此是 56 而不是 58。

**`onnx.checker` 不是本项目的判定条件**：原始模型本身就过不了 checker，
因为 `SimplifiedLayerNormalization` 的 domain 为空。Gate 1 的判定改为
`validate_graph` + ORT session 创建 + 数值等价。

## Gate 2 结果（摘要）

产物：`models/derived/gemma4-e2b-q4f16-gate2/`（sha256 `10bbfa63…`）。

| 项 | 数值 |
|---|---:|
| 节点数 | 1233 → **1078（−155）**；相对原始 1289 是 **−211（−16.4%）** |
| 层分类 | **35 GQA / 0 manual / 0 shared-KV** |
| Gate1→Gate2 decode 等价 | cosine **0.99999893**，rmse/std 0.00378，top5 5/5 |
| Gate1→Gate2 prefill 等价 | cosine **0.99999261**，rmse/std 0.00678，top5 5/5 |
| 原始→Gate2 累计等价 | cosine **0.99999672**，rmse/std 0.00981，top5 5/5 |
| 投影收益（相对原始） | decode 8.50 → **7.11s（−16.4%）**，端到端 11.30 → **9.91s（−12.3%）** |

关键实现细节：shared 层没有自己的 KV cache，K/V 来自 layer 13（16 个 sliding 层）
或 layer 14（4 个 full 层）。融合时把共享 cache 切成 `past` 和 `new` 两段，
**每个 donor 只切一次**（9 个节点），16/4 个层共用这份切片，每层只加 1 个 GQA。

另一个必须记住的 ORT 行为：**只有声明了可选的 `present_key` / `present_value`
输出，CPU 端的 GQA kernel 才会读取 `past_key` / `past_value`**。不声明时它会静默
忽略 past，只算当前 token。shared 层的 present 输出不暴露给图外，但必须在节点上声明。

## Gate 3 结果（负结果）

在 Chrome 155 + transformers.js 4.3.1 + WebGPU 下，同一页面加载两个 pipeline，
交替 A/B，greedy 64 tokens：

| 对比 | decode Δ | 首 token Δ |
|---|---:|---:|
| base → gate1 | **−4.3%**（30.46 → 29.14 ms/token） | −5.2% |
| gate1 → gate2 | **−2.5%**（29.10 → 28.37 ms/token） | −3.5% |
| base → gate2 | **−1.9 ~ −3.4%** | −4.0% |

- **等价性通过**：译文语义等价，字符 bigram Jaccard 约 0.73；没有崩坏。
  这也证明 WebGPU 端的 shared-KV GQA 正确读取了 past cache。
- **计时失败**：1.9–3.4% 远低于 10% 的 kill criterion，项目按规则停止。
- **节点数不是性能指标**：Gate 1 删 4.3% 节点换来 −4.3%；Gate 2 删 12.6%
  节点只换来 −2.5%。剩下的 `MatMulNBits` / `SimplifiedLayerNorm` 才是成本大头，
  shared attention 的节点本来就便宜，而且 runtime 可能已经做了部分融合。

后续可选（超出本轮 kill criterion）：去掉未使用的 present 输出、把 q/k norm +
rotary 折进 GQA、或换 WebLLM/MLC 运行时。详见 Gate 3 报告。

## Gate 4 结果：为什么节点数没有换成时间

不猜，直接数。两个新 harness 都不做 A/B，单模型单独加载：

- `bench/browser/count.mjs` — 在模型加载前 patch WebGPU 原型，数每个 token
  真实的 dispatch / pass / submit；
- `bench/browser/gpu-microbench.mjs` — 用最小 compute shader 测一次
  dispatch / pass / submit 各多少 µs。

**每个 decode token 的真实调用数（中位数）：**

| model | 图节点 | dispatch | compute pass | submit |
|---|---:|---:|---:|---:|
| base | 1289 | 1112 | 266 | 86 |
| gate1 | 1233 | 1089 | 238 | 84 |
| gate2 | 1078 | 1019 | 214 | 80 |

**单价（Apple Silicon + Chrome 155 / Metal 3）：**

| 操作 | 成本 |
|---|---:|
| 同 pass 内一次 dispatch | 1.32 µs |
| 一次 pass 边界 | 0.88 µs |
| 一次 submit | 18.2 µs |

**结论：**

1. **删节点 ≠ 删 dispatch。** 删 211 个节点只删 93 次 dispatch，比例
   2.27 : 1。节点数把收益高估了 2.3 倍。
2. **dispatch 本身很便宜。** 1112 次全加起来 1.47 ms，占 27 ms 的 5.4%。
3. **最贵的启动开销和节点数无关。** 86 次 submit 共 1.57 ms，而删掉 16.4%
   节点只让 submit 从 86 掉到 80 —— submit 次数由图的切分结构决定。
4. **Gate 1 和 Gate 2 赢的不是同一样东西。** 用管线单价预测，base→gate1
   只该省 0.09 ms，实测省 1.31 ms（删掉的是 Softmax/Where 这种真在计算的
   kernel）；gate1→gate2 预测省 0.19 ms，实测 0.73 ms（dispatch + 内存搬运）。

即使 211 个被删节点每一个都对应一次 dispatch，也只能省约 0.28 ms ≈ 1.0%。
**10% 的 kill criterion 在这条路线上不可达**，不是实现问题。

详见 [`docs/experiments/2026-10-19-gate4-why-node-count-is-not-time.md`](docs/experiments/2026-10-19-gate4-why-node-count-is-not-time.md)。

## Gate 5 结果：GPU 侧逐 pass 时间

给每个 compute pass 加 `timestampWrites`（标准 `timestamp-query`），把一个
decode step 拆成 208–266 个 GPU 时间段。

- **最贵的 1 个 pass 占 13%**：位置固定在倒数第三个，只有 **1 次 dispatch**，
  中位数约 **2.6 ms**。图尾确认是 `/lm_head/MatMul`，一个
  `262144 × 1536` 的 q4 `MatMulNBits`（约 201 MB 权重，≈77 GB/s）。
- **16–27 个超过 300 µs 的 pass 占 48–67%**，其余 200 多个 pass 合计只占
  三分之一。
- 这些重 pass 是每层的 `MatMulNBits` / `SimplifiedLayerNorm`，不是调度。

**LM head 这 2.6 ms 是一次 dispatch，任何减少节点数的方案都碰不到它**，
而它单独就占 decode 的约 10%。

详见 [`docs/experiments/2026-10-19-gate5-gpu-pass-profile.md`](docs/experiments/2026-10-19-gate5-gpu-pass-profile.md)。

## Gate 6 结果（负结果）：没有 ≥10% 的剩余路线

判据先于结果：一条路线只有在 token 级实测空间 ≥10% 时才进入下一轮。

| 路线 | 决定数字 | 状态 |
|---|---|---|
| 减少 dispatch / 节点融合 | 实测 base→gate2 **−1.9%**；GPU 侧上界 **7.0%** | 关闭 |
| LM head 带宽优化 | 已占读带宽 **79–82%**；整 pass 只占 token **9.92%** | 关闭 |
| dtype（q4f16→fp16） | fp16 字节是 **3.46×**，且已在带宽墙附近 | 关闭（未运行） |

两条主路线是在**上限**上失败的：即使 1112 次 dispatch 全删也只有 7.0%，
即使 LM head 的 kernel 完全免费也只有 9.92%。在同一批文件上继续做图手术不会
改变结论。

**这笔账的现状：**

| 分项 | 时间 | 性质 |
|---|---:|---|
| 每 token 必须读的 1.52 GB 权重 @ 109 GB/s | **13.93 ms（51.2%）** | 物理下界 |
| 实测总时间 | 27.2 ms | `runs/2026-10-19-exp2-dispatch-count.json` |
| GPU pass 时间之和 | 20.3–26.5 ms | 与 wall 同量级，GPU 全程在跑 |
| CPU 调度管线 | 3.0–3.3 ms（12%） | 与 GPU 执行重叠 |

地板上方那部分当时没有归因；Gate 7 把它拆开了（见下一节），并把地板修正为
12.08 ms（1.52 GB 里约 202 MB 是 rotary cache，不按 token 全读）。详见
[`docs/experiments/2026-10-20-gate6-decision.md`](docs/experiments/2026-10-20-gate6-decision.md)。

## Gate 7 结果：按 op class 拆开

`GPUComputePassEncoder.writeTimestamp()` 在 pass 内仍然不可用（缺
`chromium-experimental-timestamp-query-inside-passes`）。改用
**一 dispatch 一 pass**：拦截 `beginComputePass`，每次 dispatch 后结束并重开
pass，标准 `timestampWrites` 于是就是逐 dispatch 计时。op 名来自
shader module 的描述符 `label`（pipeline label 为空）。

base，4 个 decode step，每步 1112 个 dispatch，identity 覆盖率 100%：

| op class | ms/step（run1） | share |
|---|---:|---:|
| **MatMulNBits**（q4 权重矩阵乘） | **18.10** | **71.3%** |
| LayerNorm | 1.83 | 7.2% |
| MatMul | 1.10 | 4.4% |
| Mul | 0.66 | 2.6% |
| Add | 0.65 | 2.6% |
| 其余 20 类 | 3.04 | 12.0% |

非 `MatMulNBits` 的全部 kernel 合计 **7.28 ms（28.7%）**。第二次运行
（`...-run2.json`）给出 17.76 ms / 75.3%，matmul 时间差 1.9%，排序相同。
MatMulNBits 的 726 个权重张量共 1.3178 GB，读取用了 18.10 / 17.76 ms，
有效带宽 **72.8 / 74.2 GB/s = 109.06 GB/s 的 66.8% / 68.0%**；
同一个模型里 LM head 单独测到 86.18 GB/s。

所以 Gate 6 的「地板之上还有 13 ms」不是一块 13 ms 的别的计算：
**是 matmul 自己没跑满带宽**。详见
[`docs/experiments/2026-10-20-gate7-op-attribution.md`](docs/experiments/2026-10-20-gate7-op-attribution.md)。

## Gate 8 结果：小 matmul 的效率

每个 dispatch 记录它绑定的 buffer 大小；对每个 MatMulNBits 节点，
把消费的 initializer 字节排成指纹（242 个节点只有 9 种）。一个 dispatch 的
buffer 多重集包含哪个指纹，就归到哪个权重档位。

| 权重 MiB | 节点 | dispatch/step | GB/s（run1 / run2） |
|---:|---|---:|---:|
| 20.81 | gate_up（20 层） | 20 | 72.1 / 66.3 |
| 10.41 | gate_up（35 层） | 35 | 72.7 / 66.9 |
| 222.00 | LM head | 1 | 66.4 / 71.1 |
| 1.73 | q_proj | 56 | 47.5 / 47.1 |
| **0.22** | **k_proj** | **94** | **10.7 / 15.2** |
| 5.20 | down_proj | 15 | 63.3 / 64.0 |
| 3.47 | q_proj | 14 | 64.6 / 58.8 |
| 7.59 | per-layer projection | 1 | 69.3 / 55.9 |
| 0.43 | k_proj | 6 | 24.8 / 31.3 |

**带宽随权重字节数下降。** 94 次 0.22 MiB 的 k_proj 占 matmul 时间
6.5–9.4%，只有 10.7–15.2 GB/s；和最大档位（55.9–72.7）差 3.7–6.8 倍。
两个小档位合计 150/242 次 dispatch；如果都达到 66 GB/s，可省约
**1.70–2.27 ms（token 的 6.3–8.4%）**——推论，不是已选路线。
详见 [`docs/experiments/2026-10-20-gate8-matmul-efficiency.md`](docs/experiments/2026-10-20-gate8-matmul-efficiency.md)。

## Gate 9 结果（负结果）：两个候选都不到 10%

Gate 9 先修正了 Gate 8 的一个读法：**权重字节数不是 kernel 身份**。
同样 227,328 B 的档位里有 59 个 `[256,48,16]`（k/v/input_gate）和
35 个 `[1536,8,16]`（per-layer projection），运行时 workgroup 数是 32 vs 192。
按 `(字节, workgroup)` 重新分组后逐档重算。

一 dispatch 一 pass 的代价直接测到了：重放同一个 dispatch，split 比 batched
每次多 **1.0–1.3 µs**（两次运行一致）。对最小的 kernel（冷时间 12–23 µs）
是 4–11%，低于 15% 的修正阈值。

workgroup=192 这一族覆盖 0.22 / 1.73 / 3.47 / 5.20 / 10.41 MiB 五个真实形状，
拟合出下面这个式子（**推论，不是实测**；两轮范围）：

```
每次 dispatch 时间 ≈ 12.4–13.1 µs 固定成本 + 权重字节 / (79–85 GB/s)
```

| 候选 | 上界（两次运行） | 判定 |
|---|---:|---|
| A. 小档位并行度 / 固定成本 | **5.4% / 8.8% token** | 低于 10% |
| A（敏感性：同轮最好档位） | 6.7% / 9.6% | 低于 10% |
| B. 大档位内存效率（到同轮最好档位） | **6.6% / 6.0% token** | 低于 10% |
| B（跨负载用 Gate 5 的 86.18 GB/s） | 6.1% / 12.0% | 高负载轮不可比 |
| B（只在纯读上限 109 GB/s 下） | 16.8–22.7% | 没有实测 kernel 达到过 |

两轮运行的 loadavg 是 **2.96 / 4.70**，MatMulNBits 合计
**19.02 / 21.64 ms/step**、整体 **69.3 / 60.9 GB/s**。Gate 7 的
17.76–18.10 ms（loadavg 2.72–3.48）和 Gate 8 的 21.3–21.5 ms
（loadavg 5.0–5.9）都落在这条负载线上，跨负载不可比。

**结论：不进入优化 gate。** 小档位固定成本最多 **1.9–2.0 ms**，
大档位距同轮最好档位约 **1.6–1.8 ms**，都低于 10%，且都需要换 kernel
或换运行时。
详见 [`docs/experiments/2026-10-20-gate9-discrimination.md`](docs/experiments/2026-10-20-gate9-discrimination.md)。

## 目录

```text
browser-decode-budget/
├─ tools/
│  ├─ inspect_graph.py      # Census：逐层算子统计、融合候选、可删节点估算
│  ├─ dispatch_cost.py      # 把 census + 实测基线变成时间投影
│  ├─ fuse_manual_attention.py  # Gate 1：manual attention → GQA
│  ├─ fuse_shared_kv.py     # Gate 2：shared-KV → 切分 cache + GQA
│  ├─ compare_models.py     # CPU ORT 数值等价性检查
│  └─ tests/                # 合成 ONNX fixture 的单测
├─ bench/
│  ├─ browser/              # 驱动层：driver.mjs 是唯一入口（服务器/页面/ledger/退出码）
│  └─ analysis/             # ledger → 数字的纯函数 seam（不碰浏览器/GPU）
├─ models/                  # gitignored；见 models/README.md
├─ runs/                    # 每次实验的 JSON ledger（进 git）
└─ docs/experiments/        # 实验记录，负结果同样保留
```

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

make test                  # Python 单测
make test-js               # ledger 分析 seam + 驱动 seam 的单测（Node test runner）
make census                # 生成 runs/2026-10-19-gate0-census.json
make cost                  # 打印时间投影 + kill criterion
make gate0                 # 以上三步
make gate1                 # 生成 Gate 1 artifact
make gate1-check           # decode + prefill 两组等价性检查
make gate2                 # 生成 Gate 2 artifact（全部 35 层 GQA）
make gate2-check           # Gate1→Gate2 + 原始→Gate2 等价性检查
make bench-prepare         # 准备浏览器 runtime 资产
make bench                 # Chrome + WebGPU：默认 base vs gate2
make bench-op-attribution  # Gate 7：一 dispatch 一 pass，按 op class 归因
make bench-matmul-efficiency  # Gate 8：按权重字节看 matmul 的有效带宽
make bench-gate9            # Gate 9：workgroup 数 + split/batched 重放
```

默认读取 `~/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX`。
用 `MODEL=/path/to/model make census` 覆盖。

## Artifact 契约

Python 生产、浏览器消费。两侧只共享产物，不共享代码。

```text
models/derived/<artifact-id>/
├─ manifest.json
└─ onnx/…                    # transformers.js 期望的文件名
```

`manifest.json` 最少包含：

```json
{
  "id": "gemma4-e2b-q4f16-attn-fused-v1",
  "base": "onnx-community/gemma-4-E2B-it-ONNX@<revision>",
  "producer": {"kind": "surgery|export|download", "version": "..."},
  "dtype": "q4f16",
  "opset": [["", 21], ["com.microsoft", 1]],
  "files": {"decoder_model_merged_q4f16.onnx": "<sha256>"},
  "runtime": {"transformers.js": "...", "ort-web": "...", "chrome": "..."}
}
```

浏览器侧（bench / equivalence）只认这个契约，不关心模型是下载的、重导出的还是图手术来的。

## 路线

- **Gate 1 — easy fusion（已完成）**：只动 layers 4 / 9 / 14。它们有自己的
  `past_key_values.{4,9,14}`，把 `attn/manual/*` 换成 GQA，风险最低。
  判定条件是 ORT session 可创建 + CPU 数值等价；浏览器等价性留到 Gate 3。
  产物在 `models/derived/gemma4-e2b-q4f16-gate1/`。
- **Gate 2 — shared-KV fusion（已完成）**：20 层 shared-KV。16 个 sliding 层复用
  `present.13.key/value`，4 个 full 层复用 `present.14.key/value`。把共享 cache 切成
  `past` + `new`，每个 donor 只切一次，16/4 层共用；每层加 1 个 GQA。
  判定条件是 CPU 数值等价；产物在 `models/derived/gemma4-e2b-q4f16-gate2/`。
- **Gate 3 — measure（已完成，负结果）**：同会话交替 A/B，4–6 轮中位数，
  记录 runtime 版本和 loadavg。全量融合只快 1.9–3.4%，低于 10% 门槛。
  浏览器等价性通过；计时失败。融合这条线到此为止。
- **Gate 4 — 量纲（已完成）**：数真实 dispatch、测单价。结论是节点数不能当成本尺子。
- **Gate 5 — 逐 pass 归因（已完成）**：给每个 compute pass 加 timestamp，找到 LM head
  这个单 dispatch、2.6 ms 的下限。
- **Gate 6 — 路线筛选（已完成，负结果）**：dispatch、LM head 带宽、dtype 三条全关。
  本仓库在 `gate6-closed` 处冻结了这段历史。
- **Gate 7 — op 归因（已完成）**：一 dispatch 一 pass + shader module label，
  得到 MatMulNBits 71–75%、其余 25–29% 的账。下一个问题变成「为什么 matmul
  只跑到带宽上限的 67–68%」，而不是「另外 13 ms 花在哪」。
- **Gate 8 — matmul 效率（已完成）**：按权重指纹归因，带宽随权重字节数下降；
  94 次 0.22 MiB k_proj 只有 10.7–15.2 GB/s。下一步的问题是「小 matmul 是不是
  并行度受限」，而不是「总体带宽不够」。
- **Gate 9 — 判别（已完成，负结果）**：同一字节数混合了不同形状，按
  `(字节, workgroup)` 重新分组；split-pass 直接测得约 **1 µs/dispatch**；
  每次 dispatch 有 **12.4–13.1 µs 固定成本**。小档位和大档位两个候选的上界
  都低于 10%，优化线关闭。

## 非目标

- 不 fork transformers.js，不写自定义 WebGPU kernel；
- 不做通用 ONNX graph DSL / 插件框架；
- 不做 UI；
- 不把「下载 + 重导出 + 图手术 + 量化」串成一个 pipeline 框架——它们是四个 producer，
  共用 Artifact 接口即可。

## 来源

- [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v2)：35 层、4:1 local:global、KV 共享 20/35、PLE
- [Transformers.js v4](https://huggingface.co/blog/transformersjs-v4)：导出策略与 contrib ops
- [ORT GroupQueryAttention spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention)
- [Characterizing WebGPU Dispatch Overhead](https://arxiv.org/abs/2604.02344)：单次 dispatch 成本、Vulkan 上融合 +53%
- [ORT transformer optimizer supported models](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/README.md)：不含 Gemma
