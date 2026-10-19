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
| 2 | 20 层 shared-KV 能否映射到 GQA？ | 未开始 | — |
| 3 | 浏览器里真的更快吗？ | 未开始 | — |

Gate 0 的完整报告见 [`docs/experiments/2026-10-19-gate0-census.md`](docs/experiments/2026-10-19-gate0-census.md)，
Gate 1 见 [`docs/experiments/2026-10-19-gate1-manual-fusion.md`](docs/experiments/2026-10-19-gate1-manual-fusion.md)。

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

## 目录

```text
fused-onnx/
├─ tools/
│  ├─ inspect_graph.py      # Census：逐层算子统计、融合候选、可删节点估算
│  ├─ dispatch_cost.py      # 把 census + 实测基线变成时间投影
│  ├─ fuse_manual_attention.py  # Gate 1：manual attention → GQA
│  ├─ compare_models.py     # CPU ORT 数值等价性检查
│  └─ tests/                # 合成 ONNX fixture 的单测
├─ bench/
│  └─ browser/              # Gate 3：复用 gemma-translate 的交替 A/B harness
├─ models/                  # gitignored；见 models/README.md
├─ runs/                    # 每次实验的 JSON ledger（进 git）
└─ docs/experiments/        # 实验记录，负结果同样保留
```

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

make test                  # 单测
make census                # 生成 runs/2026-10-19-gate0-census.json
make cost                  # 打印时间投影 + kill criterion
make gate0                 # 以上三步
make gate1                 # 生成 Gate 1 artifact
make gate1-check           # decode + prefill 两组等价性检查
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
- **Gate 2 — hard fusion**：20 层 shared-KV。它们没有自己的 KV cache：
  16 个 sliding 层复用 `present.13.key/value`，4 个 full 层复用 layer 14 的
  `k_scores_transpose` / `present.14.value`。先用 `/prototype` 回答
  「GQA 的外部 K/V 输入能不能表达这种语义」，再写融合器。
- **Gate 3 — measure**：同会话交替 A/B，4 轮中位数；同时记录 runtime 版本和 loadavg。

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
