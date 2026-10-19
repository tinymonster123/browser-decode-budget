# 2026-10-19 — Gate 5：GPU 侧逐 pass 时间归因

## 问题

Gate 4 排除了调度开销：每 token 1112 次 dispatch、266 个 compute pass、86 次
submit，整条启动管线只有约 3.3 ms，占 27 ms 的 12%。

那剩下的 24 ms 在哪？

## 方法

`bench/browser/gpu-profile.mjs`。

在 ORT 拿到 device 时注入 `timestamp-query` feature，然后 patch
`beginComputePass`，给每个 pass 的描述符加上：

```js
timestampWrites: { querySet, beginningOfPassWriteIndex, endOfPassWriteIndex }
```

`finish()` 之前 `resolveQuerySet` 并复制到 `MAP_READ` buffer，token 结束后
统一 map 回来。这样每个 compute pass 都有一对 GPU 时间戳。

几个实现细节：

- `writeTimestamp()` 写在 pass 内部需要 Chrome 的实验 feature
  （`chromium-experimental-timestamp-query-inside-passes`），本机不支持；
  pass descriptor 上的 `timestampWrites` 走标准 `timestamp-query`，可用。
- 每个 encoder 配一个 64 槽的 query set（实际平均 3 个 pass），共 86 个/ token。
- 只在第一个 token 之后打开 profiling，prefill 不进时间线。
- 用 `token_callback` 的 pass 序号切分 decode step。

## 先验证测量本身

profiling 会把 wall time 从 27 ms 抬到 46–57 ms —— 每 token 多出 86 次
query set 创建 + resolve + copy，CPU 喂不上 GPU。所以**不能用 profiling 的
wall time 下结论**。

但可以用它自检时间戳单位：

| | base | gate1 | gate2 |
|---|---:|---:|---:|
| 每步 pass GPU 时间之和 | 20.3–26.5 ms | 20.8–24.8 ms | 21.1–29.3 ms |
| 未 profiling 的 token | 27.2 ms | 26.4 ms | 27.3 ms |

**pass 时间之和和未 profiling 的 token 时间同量级**，说明时间戳是纳秒、
pass 时长没有被 timestamp 写入显著放大。下面的逐 pass 归因可用。

## 结果

每步 208–266 个 pass，分布极不均匀：

| model | step | passes | sum | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 266 | 20.3 ms | 16 µs | 233 µs | 2.70 ms |
| base | 2 | 266 | 26.5 ms | 24 µs | 305 µs | 3.65 ms |
| gate1 | 1 | 238 | 24.8 ms | 29 µs | 300 µs | 5.29 ms |
| gate1 | 2 | 238 | 20.8 ms | 26 µs | 296 µs | 2.60 ms |
| gate2 | 1 | 214 | 21.1 ms | 22 µs | 322 µs | 2.60 ms |
| gate2 | 2 | 214 | 28.4 ms | 30 µs | 366 µs | 2.98 ms |

集中度：

- 最贵的 **1 个 pass 占 13%**；
- **16–27 个超过 300 µs 的 pass 占 48–67%**；
- 剩下 200 多个 pass 一共只占三分之一左右。

## 最关键的一个 pass：LM head

三个模型、每个 step 里都有同一个特征：

| model | pass 位置 | dispatch 数 | GPU 时间 |
|---|---|---:|---:|
| base | 263 / 266 | 1 | 2701 / 3647 µs |
| gate1 | 235 / 238 | 1 | 5287 / 2595 µs |
| gate2 | 211 / 214 | 1 | 2596 / 2592 µs |

位置固定在**倒数第三个** pass，永远只有 1 次 dispatch，中位数约 **2.6 ms**。

图尾核对：

```text
MatMulNBits   /lm_head/MatMul          <- vocab 262144 × hidden 1536, q4
Div           /lm_head/softcap/Div
Tanh          /lm_head/softcap/Tanh
Mul           /lm_head/softcap/Mul     -> logits
```

权重 `262144 × 1536 × 0.5 B ≈ 201 MB`。2.6 ms 对应约 77 GB/s 的有效带宽，
符合一个 memory-bound 的 q4 反量化 matmul。

**这 2.6 ms 是一次 dispatch。** 任何「减少节点数 / 减少 dispatch 数」的方案
都碰不到它，而它单独就占了 decode 的约 10%。

## 和 Gate 4 拼起来

一个 27 ms 的 decode token 大致是：

| 部分 | 量级 | 说明 |
|---|---:|---|
| LM head 单 pass | 2.6 ms | 1 次 dispatch，node fusion 碰不到 |
| 其余 15–26 个 >300 µs 的 pass | 7–16 ms | 每层的 matmul / layernorm 一类真实计算 |
| 剩下 200 多个 pass | 8.5–10.6 ms | 大量几十 µs 的小 kernel |
| 启动管线（Gate 4） | 3.3 ms | **和上面的 GPU 时间重叠**，不是额外相加 |

启动管线的 3.3 ms 是 CPU/队列侧的预算，和 GPU 执行并行发生，所以它不能简单
加到 27 ms 上；它的意义是给出上限：即使把调度开销全部消掉，也只能省 3.3 ms
里的可删部分（dispatch 1.5 ms），而 node fusion 实际只碰到其中 0.1 ms 左右。

这解释了 Gate 3 的负结果为什么是结构性的：**成本和节点数不在同一个维度上**。
真正的大头是少数几个 memory-bound 的 `MatMulNBits`，其中最大的一次只发
一个 dispatch。

## 尚未回答

- pass 和图节点没有一一对应的名字，只能靠图尾 + 时长聚类推断。本轮确认了
  LM head，其余重 pass 的归属（gate/up/down projection）没有逐个核对。
- 同一模型不同 step 的总和波动 20–29 ms，比未 profiling 的 token 波动大，
  说明 profiling 本身引入了时钟/调度噪声。结论只取跨模型重复出现的结构
  （LM head 单 pass、少数重 pass 占大头），不取单次绝对值。

## 复现

```bash
make bench-prepare
MODELS=base,gate1,gate2 STEPS=3 node bench/browser/gpu-profile.mjs
```

产物：`runs/2026-10-19-exp4-pass-profile.json`，含每步每个 pass 的
`seq / us / dispatches`。
