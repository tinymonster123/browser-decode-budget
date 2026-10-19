PY ?= .venv/bin/python
MODEL ?= $(HOME)/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX
DECODER ?= $(MODEL)/onnx/decoder_model_merged_q4f16.onnx
CONFIG ?= $(MODEL)/config.json
CENSUS ?= runs/2026-10-19-gate0-census.json
ARTIFACT ?= models/derived/gemma4-e2b-q4f16-gate1/onnx/decoder_model_merged_q4f16.onnx

.PHONY: test census cost gate0 gate1 gate1-check

test:
	$(PY) -m unittest discover -s tools/tests -v

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
