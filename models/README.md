# models/

Model weights are **not** committed. This directory is where artifacts live.

Expected layout (transformers.js naming):

```text
models/
└─ onnx-community/gemma-4-E2B-it-ONNX/
   ├─ config.json
   ├─ generation_config.json
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   ├─ chat_template.jinja
   └─ onnx/
      ├─ decoder_model_merged_q4f16.onnx
      ├─ decoder_model_merged_q4f16.onnx_data
      ├─ embed_tokens_q4f16.onnx
      └─ embed_tokens_q4f16.onnx_data
```

For Gate 0 the tools read the existing copy at:

```text
~/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX
```

Later artifacts (fp16, surgically fused, re-quantized) should be written under
`models/derived/<artifact-id>/` and described by a `manifest.json` — see the
Artifact interface in the README.
