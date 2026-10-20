PY ?= .venv/bin/python
MODEL ?= $(HOME)/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX
DECODER ?= $(MODEL)/onnx/decoder_model_merged_q4f16.onnx
CONFIG ?= $(MODEL)/config.json
CENSUS ?= runs/2026-10-19-gate0-census.json
ARTIFACT ?= models/derived/gemma4-e2b-q4f16-gate1/onnx/decoder_model_merged_q4f16.onnx
GATE1 ?= models/derived/gemma4-e2b-q4f16-gate1/onnx/decoder_model_merged_q4f16.onnx
GATE2 ?= models/derived/gemma4-e2b-q4f16-gate2/onnx/decoder_model_merged_q4f16.onnx

.PHONY: test test-js census cost gate0 gate1 gate1-check gate2 gate2-check bench-prepare bench \
	bench-op-attribution bench-bandwidth lm-head-weights lm-head-utilization weight-bytes

test:
	$(PY) -m unittest discover -s tools/tests -v

# Gate 6: ledger analysis seam. Plain JSON in, plain numbers out — no browser, no GPU.
test-js:
	node --test bench/analysis/*.test.mjs bench/browser/*.test.mjs

census:
	$(PY) tools/inspect_graph.py --model "$(DECODER)" --config "$(CONFIG)" --json "$(CENSUS)"

cost:
	$(PY) tools/dispatch_cost.py --census "$(CENSUS)"

gate0: test census cost

gate1:
	$(PY) tools/fuse_manual_attention.py --model "$(DECODER)" --layers 4,9,14 \
		--out-dir models/derived/gemma4-e2b-q4f16-gate1 --force
	cp models/derived/gemma4-e2b-q4f16-gate1/manifest.json \
		runs/2026-10-19-gate1-manifest.json

gate1-check:
	$(PY) tools/compare_models.py --a "$(DECODER)" --b "$(ARTIFACT)" \
		--tokens 1 --past 4 --seed 1 \
		--json runs/2026-10-19-gate1-equivalence-decode.json
	$(PY) tools/compare_models.py --a "$(DECODER)" --b "$(ARTIFACT)" \
		--tokens 4 --past 0 --seed 7 \
		--json runs/2026-10-19-gate1-equivalence-prefill.json
	$(PY) tools/inspect_graph.py --model "$(ARTIFACT)" --config "$(CONFIG)" \
		--json runs/2026-10-19-gate1-census.json --quiet

gate2:
	$(PY) tools/fuse_shared_kv.py --model "$(GATE1)" \
		--out-dir models/derived/gemma4-e2b-q4f16-gate2 --force
	cp models/derived/gemma4-e2b-q4f16-gate2/manifest.json \
		runs/2026-10-19-gate2-manifest.json
	$(PY) tools/inspect_graph.py --model "$(GATE2)" --config "$(CONFIG)" \
		--json runs/2026-10-19-gate2-census.json --quiet

gate2-check:
	$(PY) tools/compare_models.py --a "$(GATE1)" --b "$(GATE2)" \
		--tokens 1 --past 4 --seed 1 \
		--json runs/2026-10-19-gate2-equivalence-decode.json
	$(PY) tools/compare_models.py --a "$(GATE1)" --b "$(GATE2)" \
		--tokens 4 --past 0 --seed 7 \
		--json runs/2026-10-19-gate2-equivalence-prefill.json
	$(PY) tools/compare_models.py --a "$(DECODER)" --b "$(GATE2)" \
		--tokens 1 --past 4 --seed 1 \
		--json runs/2026-10-19-gate2-equivalence-cumulative.json

bench-prepare:
	node bench/browser/prepare.mjs

bench: bench-prepare
	node bench/browser/run.mjs

# Gate 4: count real WebGPU calls per token (single model, no A/B).
bench-count: bench-prepare
	node bench/browser/count.mjs

# Gate 6: measure the serialized dispatch cost (and the independent baseline).
bench-microbench: bench-prepare
	node bench/browser/gpu-microbench.mjs

# Gate 5: per-compute-pass GPU timestamps.
bench-profile: bench-prepare
	node bench/browser/gpu-profile.mjs

# Gate 6: achievable read bandwidth of this GPU, timed with timestamp-query.
bench-bandwidth: bench-prepare
	node bench/browser/bandwidth.mjs

# Gate 7: one pass per dispatch, attributed by the shader module label.
bench-op-attribution: bench-prepare
	node bench/browser/op-attribution.mjs

# Gate 6: LM head weight bytes (recorded once), then utilization from ledgers.
lm-head-weights:
	$(PY) tools/lm_head_weights.py --name base --model "$(DECODER)" \
		--json runs/2026-10-20-lm-head-weights.json
	$(PY) tools/lm_head_weights.py --name gate1 --model "$(GATE1)" \
		--json runs/2026-10-20-lm-head-weights-gate1.json
	$(PY) tools/lm_head_weights.py --name gate2 --model "$(GATE2)" \
		--json runs/2026-10-20-lm-head-weights-gate2.json

lm-head-utilization:
	node bench/browser/lm-head-utilization.mjs

# Gate 7: initializer bytes grouped by the op that consumes them.
weight-bytes:
	$(PY) tools/weight_bytes_by_op.py --model "$(DECODER)" \
		--json runs/2026-10-20-exp7-weight-bytes-by-op.json

# Gate 4 follow-up: what the cpu-timed pass costs, reconciled against exp 2.
bench-perf-now:
	node bench/browser/perf-now-cost.mjs
