# bench/browser

Gate 3 的浏览器基准，尚未实现。

要求（从 `gemma-translate/probes/decode-bench.mjs` 带过来）：

- 模型只加载一次；
- 同一 offscreen 会话里交替 A/B（A B / B A），只比较同轮 ms/token；
- 4 轮中位数；
- 记录 `transformers.js` / `ort-web` 版本、Chrome 版本、loadavg；
- 先做输出等价性，再做速度。

Python 侧的 ORT 没有 WebGPU EP（本机只有 CoreML / CPU / Azure），所以计时必须在这里跑。
