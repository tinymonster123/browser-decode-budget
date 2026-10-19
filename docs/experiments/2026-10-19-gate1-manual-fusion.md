# 2026-10-19 — Gate 1：3 层 manual attention → GQA

## 问题

Gate 0 显示 layers 4 / 9 / 14 是手工拆开的 full-attention，每层约 20 个可删节点。
但 Gate 0 只数了节点，没有证明**替换后的图还是同一个模型**。

Gate 1 只回答两件事：

1. 能不能把这三层改成 `com.microsoft.GroupQueryAttention`，并且图仍然可加载？
2. 同一组输入下，输出是否等价？

## 方法

`tools/fuse_manual_attention.py` 对每一层做定点手术：

- 保留 `q_norm` / `k_norm` / `q_rotary` / `k_rotary` / `v_norm`，只替换 attention 核心；
- GQA 输入：`q_rotary` 输出、`k_rotary` 输出、`v_norm` 输出、该层自己的
  `past_key_values.<n>.key/value`、共享的 `seqlens_k` / `total_sequence_length` /
  `attention_bias`；
- GQA 属性：`num_heads=8`、`kv_num_heads=1`、`scale=1.0`、`local_window_size=-1`
  （full attention）、`do_rotary=0`、`causal=1`；
- GQA 输出直接接管 `present.<n>.key/value`；
- `o_proj` 的输入从旧的 `manual/output/Reshape` 改成 GQA 输出；
- 最后做一次稳定拓扑排序，并检查悬空输入 / 重复生产者。

### 两个导出细节

1. **layer 4 的 `manual/mask/pos_unsq` 不能删**。它被 `/model/shared_donor.13/mask/*`
   复用（sliding 共享层的 mask 需要这个 `position_ids` unsqueeze）。保留它，代价是
   该层少省 1 个节点。
2. **layer 14 的 `k_scores_transpose` 被 4 个 shared full 层复用**。
   融合后用一个 `Transpose(present.14.key, perm=[0,1,3,2])` 重新产出同名张量，
   让 layers 19/24/29/34 继续工作。多 1 个节点。

因此实际净省 **56** 个节点（Gate 0 估的是 58）。

## 结果

### 产物

```text
models/derived/gemma4-e2b-q4f16-gate1/
├─ manifest.json
└─ onnx/decoder_model_merged_q4f16.onnx   # sha256 2120f3d9efe616e4…
```

- 节点数：**1289 → 1233（−56，−4.34%）**
- 层分类：15 个 GQA、0 个 manual、20 个 shared-KV
- 图结构检查：无悬空输入、无重复生产者、ORT session 可正常创建

### 数值等价

`tools/compare_models.py`，同一组随机输入，CPU ORT：

| 场景 | logits max_abs | mean_abs | cosine | rmse/std | top5 | argmax |
|---|---:|---:|---:|---:|---:|---|
| decode：S=1, past=4, seed=1 | 0.0762 | 0.0197 | **0.99999774** | 0.00677 | 5/5 | 一致 |
| prefill：S=4, past=0, seed=7 | 0.3516 | 0.0596 | **0.99997830** | 0.01097 | 5/5 | 一致 |

逐层单融（只融一层，隔离上游传播）：

| 融合层 | cosine | rmse/std | 该层 present 最大误差 |
|---|---:|---:|---:|
| 4 | 0.999997 | 0.0070 | `present.4.*` = **0** |
| 9 | 0.999998 | 0.0094 | `present.9.*` = **0** |
| 14 | 1.000000 | 0.0014 | `present.14.*` = **0** |

三条同时融合时，`present.4.*` 仍是 0；`present.9.*` / `present.14.*` 出现
1e-4 – 6e-3 的差异，来自前面融合层改变了 hidden state，再传播到后续层，
不是 GQA 写 cache 的方式有问题。

### 关于 `onnx.checker`

**原始模型本身就过不了 `onnx.checker`**：

```text
No Op registered for SimplifiedLayerNormalization with domain_version of 21
```

这些节点的 `domain` 是空字符串，而 `SimplifiedLayerNormalization` 是 ORT 的
contrib 算子；ORT 运行时会注册它，`onnx.checker` 不会。所以本项目不把 checker
当 Gate 1 的判定条件，改用：

- `validate_graph`（结构完整性）；
- ORT session 创建（算子可解析）；
- 数值等价（真正关心的东西）。

## 这意味着什么

- **融合器是对的**：单层融合时该层 present 输出逐位相同，logits 的差异只来自
  fp16 attention kernel 的累加顺序，argmax 和 top5 都不变。
- **Gate 1 的收益很小**：56/1289 = 4.3%，投影 decode 8.50 → 8.13s、端到端
  11.30 → 10.93s（−3.3%），单独达不到 Gate 3 的 10% 阈值。
- **它的价值是把 Gate 2 的风险降下来**：同样的手术路径、同样的等价性检查，
  现在已经在真实模型上跑通了。

## 决策

**Gate 1 PASS（数值等价）**，进入 Gate 2。

- 浏览器侧等价性和计时留到 Gate 3；artifact manifest 里记为
  `"browser_equivalence": "pending"`。
- Gate 2 的目标是剩下的 **172 个 shared-KV 节点**（占融合后图的 13.95%）。
  如果全部成功，节点数 1233 → 1061，相对原始 1289 是 82.3%，即 Gate 0 保守
  投影的 decode −17.8% / 端到端 −13.4%。
- Gate 2 先用 `/prototype` 回答「GQA 的外部 K/V 输入能不能表达 shared-KV 语义」，
  再写生产融合器。

## 复现

```bash
make gate1          # 生成 artifact
make gate1-check    # decode + prefill 两组等价性检查
```

原始 ledger：

- [`runs/2026-10-19-gate1-census.json`](../../runs/2026-10-19-gate1-census.json)
- [`runs/2026-10-19-gate1-equivalence-decode.json`](../../runs/2026-10-19-gate1-equivalence-decode.json)
- [`runs/2026-10-19-gate1-equivalence-prefill.json`](../../runs/2026-10-19-gate1-equivalence-prefill.json)
- [`runs/2026-10-19-gate1-manifest.json`](../../runs/2026-10-19-gate1-manifest.json)
  （artifact 目录里的同名 manifest 是同一份，模型权重本身 gitignored）
