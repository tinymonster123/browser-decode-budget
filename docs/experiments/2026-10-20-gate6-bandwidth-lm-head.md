# 2026-10-20 — Gate 6：带宽上限与 LM head 利用率

## 问题

Gate 5 给出了逐 pass 的 GPU 时间：LM head 单 pass 约 2.6 ms，是 decode 里最贵的一个 pass。
但当时只有一个反推的数字：262144 × 1536 的 q4 权重约 201 MB 除以 2.6 ms，得到约 77 GB/s。
没有这台 GPU 的实测带宽上限，就无法判断这个 kernel 是已经贴着极限，还是还留着一毫秒。

本轮回答两个数：

1. 这台 GPU 的实际读带宽是多少，在哪个 buffer 大小上进入平台。
2. LM head 的有效带宽是多少，占实测上限的百分之几。

## 先写判据

在看到任何数字之前定下：

1. 带宽平台必须落在硬件可能的范围内（本轮用 10–1000 GB/s 的自检窗口）。
2. 两次完整扫描的平台差异 **> 10%** 的数字不用于决策。
3. `timestamp-query` 不可用时，harness **拒绝**计算利用率，不用 wall time 替代。
4. 平台必须由至少四个 buffer 大小扫出来，并明确指出从哪个大小开始。
5. 本轮只产出数字；“选哪条路线”留给 #6。
6. 路线选择门槛来自 #1：一条路线必须有 **≥10% 的 token 级实测空间**。
   Gate 5 已记录 LM head 的中位数是 2.701 ms，占 27.2 ms 的 **9.93%**；
   即使这个 kernel 免费，它的上界也低于 10%。本轮只记录这个上界，是否关闭 #5 由 #6 决定。

## 方法

### 带宽探针

`bench/browser/bandwidth.mjs` + `gpu-microbench.html` 的 `__bandwidth`。

- 一个只读 kernel 扫过一个 buffer。每个线程读一个 `u32`，用 grid-stride loop
  覆盖整个 buffer；读到的值参与一个哨兵比较（几乎不可能成立），
  防止编译器把读优化掉。写流量因此可以忽略。
- 用 **pass descriptor 的 `timestampWrites`** 给 kernel 计时，不是 wall time：
  每次提交一个 compute pass，`resolveQuerySet` 后 map 回来读两个 u64 纳秒值。
- buffer 大小扫 `4, 8, 16, 32, 64, 128, 256` MB 七档。
- 30 次 repeat × 2 个配对 sweep，repeat-major，sweep 顺序每轮翻转；
  每个点取 **5 分位数**（快尾），和 #3 的估计量一致。
- 采样前连续压 500 ms，让 GPU 进入稳态时钟。
- 平台的**主定义**沿用 #2 的 seam：从某个大小开始，后面所有样本都落在
  **该起点值的 10% 以内**；返回尾部中位数和起始大小。两次 sweep 的平台还必须
  满足 10% 一致性。另给一个 5% 敏感性口径，见“不确定性与边界”。

**一个必须记录下来的坑**：WebGPU 默认的 `maxStorageBufferBindingSize` 是 128 MiB。
第一次跑 256 MB 时，bind group 是 invalid 的，kernel 没有真正读那个 buffer，
时间停在 128 MB 的水平，算出来 219 GB/s —— 一个看起来“更快”的假点。
现在 `setup()` 显式请求 adapter 的 `maxStorageBufferBindingSize` / `maxBufferSize`，
并且对 shader 做 `getCompilationInfo()` 检查；编译错误会让整轮失败，而不是产出零时间戳。

### LM head 利用率

不需要新的模型运行。三个输入都是已经记录的数据：

1. **per-pass GPU 时间**：`runs/2026-10-19-exp4-pass-profile.json`（Gate 5）。
2. **带宽上限**：本轮 `runs/2026-10-20-exp6-bandwidth.json` 的平台。
3. **每个 artifact 的权重大小**：`runs/2026-10-20-lm-head-weights*.json`，
   由 `tools/lm_head_weights.py` 从三个 ONNX 文件分别读出（base、gate1、gate2）。

LM head 在 Gate 5 时间线里是 **倒数第三个 pass**：最终 norm 和 softcap 尾巴之间的
一次 `MatMulNBits` dispatch。选择逻辑固定在 `lmHeadPasses`：

- 位置必须是倒数第三；
- 该 pass 必须只有 1 次 dispatch；
- 它必须是该 step 里最大的单 dispatch pass。

三个模型、每个 step 都通过了这三条。每模型取三个 decode step 的中位数，
三个 step 的原始值也写进利用率 ledger。

权重字节按 `MatMulNBits` 节点的**全部 initializer 输入**计。三个 artifact 都记录到
同一组数（`selection: named`，节点都是 `/lm_head/MatMul_Quant`）：

| tensor | shape | dtype | bytes |
|---|---|---:|---:|
| `lm_head_MatMul_weight_quant` | 262144 × 48 × 16 | UINT8（packed q4） | 201,326,592 |
| `lm_head_MatMul_weight_scales` | 262144 × 48 | FLOAT16 | 25,165,824 |
| `lm_head_MatMul_weight_zp` | 262144 × 24 | UINT8 | 6,291,456 |
| **total（base / gate1 / gate2 相同）** | | | **232,783,872** |

Gate 5 报告里的约 201 MB 只算了 packed weight；kernel 实际还要读 scales 和零值点。
本报告把两种口径都给出。

### timestamp 门

`requireTimestampQuery` 在任何计算之前检查：

- 每个模型的 `patchInfo.featureInjected === true`；
- 每个 step 的 timestamp coverage > 0。

任一不满足就抛错，拒绝计算。带宽探针自身也要求 `timestamp-query` 可用；
不可用时整轮报“拒绝测量”，不落任何 GB/s 数字。

## 结果

机器：Apple M4（10 核 GPU），Chrome 155，headed，Metal 3。
canonical 运行 loadavg 2.26。

### 带宽平台

canonical 的 p05 读数（GB/s）：

| size MB | sweep 0 | sweep 1 |
|---:|---:|---:|
| 4 | 100.57 | 99.77 |
| 8 | 103.46 | 102.72 |
| 16 | 105.71 | 106.72 |
| 32 | 108.15 | 109.33 |
| 64 | 111.15 | 111.07 |
| 128 | 110.28 | 108.85 |
| 256 | 109.90 | 109.44 |

- 平台值：**109.03 / 109.09 GB/s**，两次 sweep 差 **0.06%**，均值 **109.06 GB/s**。
- 10% 带的起点：**8 MB**（sweep 0 与 sweep 1 都是）。
- 复现运行：**108.52 GB/s**（10% 带起点 16 MB），与 canonical 差 **0.5%**。
- 曲线从 4 MB 的约 100 GB/s 上升到 64 MB 的约 111 GB/s；
  10% 带从 8 MB 起算，但 8 MB 仍比平台低约 5%。
- **5% 敏感性口径**：如果要求从起点起所有点都在平台值的 5% 内，
  canonical 与复现运行都从 **16 MB** 开始（105.7 vs 109.1 / 108.5）。
  平台值本身只变化不到 1%。因此报告同时给出“10% 带起点”和“5% 平坦起点”，
  不把一个随噪声移动的边界当成唯一答案。

### LM head 利用率

| model | LM head GPU 时间（中位数） | 有效带宽 | 占 109.06 GB/s |
|---|---:|---:|---:|
| base | 2701.3 µs | 86.18 GB/s | **79.0%** |
| gate1 | 2594.5 µs | 89.72 GB/s | **82.3%** |
| gate2 | 2595.8 µs | 89.68 GB/s | **82.2%** |

如果只算 packed weight（201,326,592 bytes，Gate 5 的口径）：

| model | 有效带宽 | 占上限 |
|---|---:|---:|
| base | 74.53 GB/s | 68.3% |
| gate1 | 77.59 GB/s | 71.1% |
| gate2 | 77.56 GB/s | 71.1% |

两种口径都说明同一件事：LM head 已经用掉了实测读带宽的 **七成到八成**。
它不是一个离带宽极限很远的 kernel。

每模型的三个 step 原始值（µs）：

| model | step 0 | step 1 | step 2 | 中位数 |
|---|---:|---:|---:|---:|
| base | 2634.5 | 2701.3 | 3647.4 | 2701.3 |
| gate1 | 2592.5 | 5286.6 | 2594.5 | 2594.5 |
| gate2 | 5956.3 | 2595.8 | 2592.1 | 2595.8 |

### 对 #5（dtype 比较）的输入

**这不是本轮的决定；#6 才做路线选择。这里只给出 #5 需要的数。**

LM head 的 pass 时间中位数 2.701 ms，占未 profiling 的 27.2 ms token 的 **9.93%**。
即使把这个 kernel 的耗时降到零，token 时间也只能减少 9.93%，低于 #1 的 10% 门槛。
按当前 79.0% 的利用率，剩余的理论空间是 21.0% × 2.701 ms ≈ **0.57 ms**，
即 token 的 **2.1%**；gate1/gate2 约 1.7%。

换句话说：dtype 路线在算术上就过不了 10% 的 token 级门槛，
而且 fp16 会把权重的字节数翻倍，在已经接近带宽上限的 kernel 上只会更慢。

## 不确定性与边界

- 探针是**只读** kernel；LM head 读权重、写一个很小的输出。利用率比较的是读主导的流量，
  写流量没有单独建模。
- 权重的口径会影响百分比：全部 initializer 输入是 79.0–82.3%，
  只算 packed weight 是 68.3–71.1%。两个数都记录在案。
- LM head 时间取三个 decode step 的中位数。原始 step 里各有一个离群值
  （base 3647 µs、gate1 5287 µs、gate2 5956 µs）；ledger 保留了全部 step，可重算。
  本轮没有重新跑 profile：ticket #4 明确说这一半复用既有 GPU 时间线，不需要新的模型运行。
- 带宽平台是 p05 估计量，不是 GPU 在空载时的理论峰值。
  平台值 109 GB/s 落在本轮 10–1000 GB/s 的自检窗口内。
- **平台起点的定义会移动**：10% 带起点在 canonical 是 8 MB、复现运行是 16 MB；
  5% 平坦口径两次都是 16 MB。平台值在 108.5–109.1 GB/s 之间（0.5%），
  所以利用率结论不受起点口径影响。
- 本轮没有改变任何模型或 kernel；它只给 #6 提供两个数。

## 复现

```bash
# 1) 带宽上限（headed，端口 8127）
node bench/browser/bandwidth.mjs

# 2) 从三个 artifact 记录 LM head 权重字节（只读 ONNX，不跑模型）
.venv/bin/python tools/lm_head_weights.py --name base \
  --model "$HOME/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_q4f16.onnx" \
  --json runs/2026-10-20-lm-head-weights.json
.venv/bin/python tools/lm_head_weights.py --name gate1 \
  --model models/derived/gemma4-e2b-q4f16-gate1/onnx/decoder_model_merged_q4f16.onnx \
  --json runs/2026-10-20-lm-head-weights-gate1.json
.venv/bin/python tools/lm_head_weights.py --name gate2 \
  --model models/derived/gemma4-e2b-q4f16-gate2/onnx/decoder_model_merged_q4f16.onnx \
  --json runs/2026-10-20-lm-head-weights-gate2.json

# 3) 从已记录的 Gate 5 ledger + 上述文件算利用率（不需要浏览器）
node bench/browser/lm-head-utilization.mjs
```

产物：

- `runs/2026-10-20-exp6-bandwidth.json`：canonical，14 行 × 30 个原始样本
  （每个样本含 `ns` 和 `gbps`）。
- `runs/2026-10-20-exp6-bandwidth-run2.json`：复现。
- `runs/2026-10-20-lm-head-weights.json`、`-gate1.json`、`-gate2.json`：
  三个 artifact 的 LM head initializer 字节数。
- `runs/2026-10-20-exp6-lm-head-utilization.json`：利用率 ledger，
  含输入、每模型的三个 step 原始值和逐模型结果。
- 分析面：`bench/analysis/ledger.mjs` 的 `bandwidthPlateau` / `combinePlateaus` /
  `utilization` / `requireTimestampQuery` / `lmHeadPasses` / `lmHeadUtilization`。
- 测试：`make test-js`、`make test`。
