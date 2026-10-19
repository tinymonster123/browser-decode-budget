# 2026-10-19 — Gate 3：浏览器等价性 + 交替 A/B

## 问题

Gate 2 在 CPU 上把 35/35 层都换成了 GQA，图节点从 1289 降到 1078。
这能不能在真实的 Chrome + WebGPU 里换成 decode 时间？

kill criterion：decode 提升 < 10% 就写负结果，停止。

## 方法

`bench/browser/` 是一个独立页面，直接加载两个 transformers.js pipeline：

- **runtime**：Chrome 155 + `@huggingface/transformers` 4.3.1 + WebGPU；
- **dtype**：`q4f16`；
- **模型来源**：本地 HTTP 服务，`/base/`、`/gate1/`、`/gate2/` 分别指向三个 artifact；
- **prompt**：与 `gemma-translate` bench 相同的 Dennard scaling 段落；
- **生成**：greedy（`do_sample: false`），`max_new_tokens=64`；
- **计时**：`(lastTokenMs - firstTokenMs) / (tokens - 1)`，排除 prefill；
- **配对设计**：同一页面里两个 pipeline 同时加载，每轮交替 A/B 顺序，
  4 或 6 轮取中位数；每次运行都记录 loadavg。

同时做了三组对比：

| 对比 | 目的 |
|---|---|
| base → gate1 | 只融 3 层 manual attention，节点 −4.3% |
| gate1 → gate2 | 只融 20 层 shared-KV，节点 −12.6% |
| base → gate2 | 全量融合，节点 −16.4% |

## 结果

### 等价性

三个版本生成的译文**语义等价**，但不是逐字相同。例如：

```text
base  : ...核心从四个增加到十六个，再到六十四个...
gate1 : ...核心从四个增加到十六个，再到六十四个...
gate2 : ...四核变成了十六核，再变成六十四核...
```

字符 bigram Jaccard：

| 对比 | Jaccard | token 数 |
|---|---:|---:|
| gate1 vs gate2 | 0.7327 | 64 / 64 |
| base vs gate2 | 0.7255 | 64 / 64 |

这不是模型崩坏，而是 fp16 数值差异让采样在等价措辞之间切换。两个版本都完整翻译了
整段文字，没有复读、串词或提前截断。

**这也证明 WebGPU 端的 shared-KV GQA 正确读取了 past cache**：如果没读到，
20 层共享 attention 会退化成只看当前 token，译文会明显崩坏。

### 计时

| 对比 | loadavg | A ms/token | B ms/token | decode Δ | 首 token Δ |
|---|---|---:|---:|---:|---:|
| base → gate1 | 1.86 | 30.46 | 29.14 | **−4.3%** | 306.5 → 290.5 ms（−5.2%） |
| gate1 → gate2 | 2.05 | 29.10 | 28.37 | **−2.5%** | 313.5 → 302.5 ms（−3.5%） |
| base → gate2（6 轮） | 2.42 | 41.25 | 40.47 | **−1.9%** | 465 → 446.5 ms（−4.0%） |

每一组的每一轮都是 gate2 更快（符号一致），但幅度只有 **2–4%**，远低于 10% 的
kill criterion。绝对 ms/token 在不同 run 之间能从 29 漂到 41ms（系统负载），
所以只有同轮配对差有意义。

### 节点数不再线性映射到时间

| 改动 | 删除节点 | 占图比例 | 实测 decode Δ | 线性预期 |
|---|---:|---:|---:|---:|
| Gate 1（3 层 manual） | 56 | 4.3% | **−4.3%** | −4.3% ✅ |
| Gate 2（20 层 shared） | 155 | 12.6% | **−2.5%** | −12.6% ❌ |

Gate 1 几乎完美符合「时间 ∝ 节点数」；Gate 2 只有预期的 1/5。这是本轮最重要的发现：

**节点数是结构指标，不是性能指标。** 不同算子的单节点成本差很多，而且 WebGPU EP
可能已经在 runtime 侧融合/优化了部分子图。

## 结论

**Gate 3 FAIL**：全量融合在浏览器里只快 1.9–3.4%，没有达到 10% 的门槛。
项目按 kill criterion 停止在融合这条线上。

## 为什么 shared-KV 的收益被吃掉了

按可能性排序：

1. **剩余节点更贵**：1078 个节点里 `MatMulNBits`（242）和
   `SimplifiedLayerNormalization`（242）占了大头；融合只删了 attention 的
   Reshape/Transpose/Where/Softmax，这些本来就不是最贵的算子。Amdahl 上限很低。
2. **WebGPU EP 可能已经融合了 manual/shared attention**：如果 runtime 把
   `MatMul + Where + Softmax + MatMul` 合成一个 kernel，图上的节点减少不会变成
   dispatch 减少。Gate 1 的线性结果说明 manual 层没有被完全融合，但 shared 层
   的模式可能走了不同的优化路径。
3. **shared GQA 自己带来开销**：每个 donor 多 9 个 Slice/Reshape 节点（共 18 个），
   而且 20 个 GQA 节点都声明了未使用的 `present_key` / `present_value` 输出，
   kernel 仍会写这些 cache。
4. **测量噪声**：绝对时间在 29–41ms 之间漂移，2–4% 的效应接近检测下限。虽然符号
   一致，但不要把它当成精确值。

## 如果还要继续（超出本轮 kill criterion）

- 去掉 shared GQA 的未使用 present 输出，看 WebGPU 是否仍能正确读 past；
  如果能，省掉 20 层的 cache 写入。
- 把 q/k norm + rotary 折进 GQA（Gate 0 的 aggressive 方案），再减约 96 个节点；
  但 JSEP/native WebGPU 对 `q_norm_weight` 的支持需要先验证。
- 换运行时：WebLLM/MLC 在 Chrome 下的公开数据是 46–51 tok/s，对 ORT 路线有
  1.4–1.5× 的潜在空间，但那是另一条项目线。

## 复现

```bash
make bench-prepare

# 默认 base vs gate2
MAX_NEW=64 REPEATS=6 node bench/browser/run.mjs

# 指定任意两版
A_MODEL=gate1 A_NAME=gate1 B_MODEL=gate2 B_NAME=gate2 \
  MAX_NEW=64 REPEATS=4 node bench/browser/run.mjs
```

原始 ledger：

- [`runs/2026-10-19-gate3-gate1.json`](../../runs/2026-10-19-gate3-gate1.json)
- [`runs/2026-10-19-gate3-gate1-vs-gate2.json`](../../runs/2026-10-19-gate3-gate1-vs-gate2.json)
- [`runs/2026-10-19-gate3-base-vs-gate2.json`](../../runs/2026-10-19-gate3-base-vs-gate2.json)
