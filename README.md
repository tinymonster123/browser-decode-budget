# fused-onnx

独立的 fused ONNX 实验台。目标只有一个：**验证「减少 WebGPU dispatch 次数」能不能把浏览器里的
decode 真正压下来**。产品接入是附属产物，不是本项目的主线。

从 `gemma-translate` 的 decode 实验中抽出。那边的结论是 decode 29.9 ms/token，
瓶颈在每算子 dispatch / 同步，而不是算力；本项目从那里开始。

## 状态

| Gate | 问题 | 状态 | 结论 |
|---|---|---|---|
| **0** | 还有多少 dispatch 可删？ | ✅ 2026-10-19 | **PROCEED**：保守可删 230/1289 节点（17.8%） |
| **1** | 3 层 full-attention 能否换成 GQA？ | ✅ 2026-10-19 | **PASS**：1289→1233，logits cosine 0.999998 |
| **2** | 20 层 shared-KV 能否映射到 GQA？ | ✅ 2026-10-19 | **PASS**：1233→1078，35/35 层全 GQA |
| **3** | 浏览器里真的更快吗？ | ❌ 2026-10-19 | **FAIL**：只快 1.9–3.4%，低于 10% 门槛；按 kill criterion 停止 |

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

## 目录

```text
fused-onnx/
├─ tools/
│  ├─ inspect_graph.py      # Census：逐层算子统计、融合候选、可删节点估算
│  ├─ dispatch_cost.py      # 把 census + 实测基线变成时间投影
│  ├─ fuse_manual_attention.py  # Gate 1：manual attention → GQA
│  ├─ fuse_shared_kv.py     # Gate 2：shared-KV → 切分 cache + GQA
│  ├─ compare_models.py     # CPU ORT 数值等价性检查
│  └─ tests/                # 合成 ONNX fixture 的单测
├─ bench/
│  ├─ browser/              # Gate 3：交替 A/B；Gate 4：计数 + 单价；Gate 5：逐 pass 计时
│  └─ analysis/             # Gate 6：ledger → 数字的纯函数 seam（不碰浏览器/GPU）
├─ models/                  # gitignored；见 models/README.md
├─ runs/                    # 每次实验的 JSON ledger（进 git）
└─ docs/experiments/        # 实验记录，负结果同样保留
```

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

make test                  # Python 单测
make test-js               # Gate 6 ledger 分析 seam 的单测（Node test runner）
make census                # 生成 runs/2026-10-19-gate0-census.json
make cost                  # 打印时间投影 + kill criterion
make gate0                 # 以上三步
make gate1                 # 生成 Gate 1 artifact
make gate1-check           # decode + prefill 两组等价性检查
make gate2                 # 生成 Gate 2 artifact（全部 35 层 GQA）
make gate2-check           # Gate1→Gate2 + 原始→Gate2 等价性检查
make bench-prepare         # 准备浏览器 runtime 资产
make bench                 # Chrome + WebGPU：默认 base vs gate2
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
