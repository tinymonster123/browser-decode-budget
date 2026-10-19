# bench/browser

Gate 3：在真实的 Chrome + WebGPU 里比较 base 和 Gate 2 artifact。

## 组成

| 文件 | 作用 |
|---|---|
| `prepare.mjs` | 从 `gemma-translate` 复制 `transformers.web.js` 和 ORT wasm 到 `vendor/` |
| `serve.mjs` | 本地静态服务：`/` 页面、`/base/` 原模型、`/gate2/` artifact |
| `index.html` | 在同一页面里加载两个 pipeline，做 warmup、输出等价性、交替 A/B |
| `run.mjs` | Playwright 驱动 Chrome，汇总中位数并写 `runs/2026-10-19-gate3-browser.json` |

`vendor/` 是构建输入，gitignored。

## 使用

```bash
node bench/browser/prepare.mjs
node bench/browser/run.mjs
```

可调环境变量：

```bash
REPEATS=4 MAX_NEW=64 HEADLESS=1 node bench/browser/run.mjs
MODEL_BASE=/path/to/base MODEL_GATE2=/path/to/gate2 node bench/browser/run.mjs

# 任意两版对比（A/B 名字只影响 ledger 和输出表）
A_MODEL=gate1 A_NAME=gate1 B_MODEL=gate2 B_NAME=gate2 \
  REPEATS=4 node bench/browser/run.mjs
```

## Gate 3 结果

| 对比 | decode Δ | 首 token Δ |
|---|---:|---:|
| base → gate1 | −4.3% | −5.2% |
| gate1 → gate2 | −2.5% | −3.5% |
| base → gate2 | −1.9 ~ −3.4% | −4.0% |

等价性通过（语义等价，bigram Jaccard ≈ 0.73），但计时低于 10% kill criterion。
完整分析见 `docs/experiments/2026-10-19-gate3-browser.md`。

## 判定

1. **先看等价性**：`equivalence.exact` 必须是 true，或输出是等价改写。
   如果 WebGPU 的 GQA 不读 `past_key`（CPU 端必须声明 present 输出才读），
   Gate 2 的译文会明显跑偏。
2. **再看计时**：只比较同一轮里的 `base` / `gate2`，4 轮中位数。
3. **kill criterion**：decode 提升 < 10% → 写负结果，停止。

## 已知约束

- 两个 q4f16 模型同时在内存里约 6.4GB；本机 16GB，先试同时加载，
  不行再退回“每次只加载一个，交替顺序”。
- WebGPU 在 headless Chrome 下可能不可用，默认 headed。
- 页面是 `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`，
  给 threaded wasm 留 SharedArrayBuffer。
