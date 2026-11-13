# Changelog

Finished work on the native inference engine and its surroundings, newest first. Each entry
keeps its number from the backlog ([docs/backlog.md](docs/backlog.md), which lists only open
items): `Bn` for a backlog item, `Dn` for a slice of phase D (more model families). The full
reports - design, measurements, tests, counterproofs and reviews - are in
[docs/native-inference.md](docs/native-inference.md) (German). Speeds are measured on an
RTX 5090 unless stated otherwise.

## 2026-11

- **2026-11-13 - B19: Q4_K on CUDA as fast as Q8_0.** The CUDA linear kernels read a Q4_K
  block as 32-bit words with one 64-bit address per block instead of single bytes; Qwen3.8-27B
  Q4_K_M decode on CUDA from 31.7 to 36.1 tokens/s (Q8_0 36.5, Vulkan 39.4).
- **2026-11-13 - D6e2: Gemma 4 on Vulkan.** Heads of 512 in a variant of `attention.comp`
  (4 queries per work group, 12 KB shared memory), K=V with the value norm, the second RoPE
  set, sandwich norms, GeGLU, the dense block beside the experts, layer scalar and logit
  softcap; experts routed on the device when the gate/up slices fit the binding alignment.
  26B-A4B Q8_0 at about 35 tokens/s.
- **2026-11-13 - D6e1: Gemma 4 on CUDA.** The same building blocks as kernels; gate and up of
  the GGUF experts kept as one buffer, so a decode step picks the experts on the device.
  26B-A4B Q8_0 at about 40 tokens/s.
- **2026-11-12 - D6d: Gemma 4 from GGUF.** The ggml-org files of 26B-A4B, 31B and 12B (Q8_0
  against the D6b references, the 12B Q4_0 QAT file behind `ILlmClient`); the tokenizer from
  GGUF.
- **2026-11-12 - D6b: the published Gemma 4 checkpoints** (26B-A4B, 31B, 12B with its
  `gemma4_unified` layout) from safetensors in Q8_0 on the CPU against transformers in
  float32; 26B-A4B behind `ILlmClient`.
- **2026-11-12 - D7h: Qwen3.8 as a chat model.** Template, parser following the model's
  `response_template`, family chosen by the template of the files; the client on CUDA and
  Vulkan. Phase D7 (Gated DeltaNet) is complete.
- **2026-11-11 - D7g: Qwen3.8-27B** from GGUF (Q8_0, Q4_K_M) on CPU, CUDA and Vulkan against
  transformers; Q4_K_M decode 31.5 (CUDA) to 39.4 tokens/s (Vulkan).
- **2026-11-11 - D7f: Qwen3.5 from GGUF** with tokenizer and client; the tiled order of the
  value heads written by the converter is computed directly on all backends.

## 2026-10

- **2026-10-25 - D7e2: Qwen3.5 on Vulkan.** DeltaNet layers as shaders, state and restore
  points on the device; prefill of 512 tokens in 42 ms, decode 3.1 ms/token (Q8_0); also
  checked on an M2 Pro through MoltenVK.
- **2026-10-25 - D7e1: Qwen3.5 on CUDA.** Prefill of 512 tokens in 42 instead of 1801 ms,
  decode 2.5 instead of 37 ms/token (Q8_0).
- **2026-10-25 - D7d: state checkpoints** before each turn, so truncation and the prompt
  cache work for recurrent models; a branch is bit-identical to a recomputation.
- **2026-10-25 - D7c: Qwen3.5 as a chat model** (pre-tokenizer, template, parser for the
  XML-style tool calls).
- **2026-10-21 - D7b: DeltaNet prefill on all cores** (512 tokens of Qwen3.5-0.8B in 1.1
  instead of 9.4 s with 22 threads).
- **2026-10-21 - D7a: Gated DeltaNet on the CPU** from safetensors (Qwen3.5-0.8B logits within
  4e-5 of transformers, greedy identical).
- **2026-10-21 - D6c: Gemma 4 as a chat model** (tokenizer, template, parser, family behind
  `ILlmClient`).
- **2026-10-20 - D6a: Gemma 4 on the CPU** from safetensors, against two tiny models from
  transformers.
- **2026-10-20 - B17 (issue #1): device features for TTS stacks.** The normed last row from
  the device (`ForwardEmbeds(..., aNormed)`), `CreateStack` without a head, parallel
  conversion while loading (0.37 instead of 5.3-6.4 s).
- **2026-10-20 - D5: gpt-oss-20b complete.** D5d2 on Vulkan (127 tokens/s) and D5d1 on CUDA
  (124 tokens/s); D5c3 reasoning of earlier turns back to the template; D5c2 Harmony template
  and parser behind `ILlmClient`; D5c1 the o200k tokenizer; D5b2 gpt-oss from GGUF; D5b MXFP4
  and sharded safetensors.
- **2026-10-19 - D5a: gpt-oss on the CPU** (sinks, YaRN, biases, clamped SwiGLU).
- **2026-10-19 - D4b: sliding window and NoPE on Vulkan and CUDA.**
- **2026-10-19 - D4: sliding window and restore points on the CPU** (narrow cache, NoPE as a
  layer property, SmolLM3 configuration).
- **2026-10-19 - D1b: Qwen2.5 as a family** (bias on Q/K/V on all backends, template, parser).
- **2026-10-19 - D3c: experts chosen on the device for decode.** Qwen3-30B-A3B from 38.9 to
  97.3 tokens/s (CUDA, one graph again) and from 42.9 to 90.4 (Vulkan, one batch per token).
- **2026-10-19 - D3b: MoE on Vulkan** (Qwen3-30B-A3B decode 43-44, prefill 489-642 tokens/s).
- **2026-10-19 - B16: Q5_0** on CPU, Vulkan and CUDA (Qwen's Q4_K_M file of Qwen2.5-0.5B).
- **2026-10-19 - B15: quick start.** Q8_0 is the default of `TNativeLlmClient.Create`,
  `MaxContext` caps a conversation, `demos/llm/llm-quickstart.lpr`.
- **2026-10-19 - B14: feedback from flydev's TTS project.** `ForwardEmbeds`, `CreateStack`,
  `ForgetWeights`, BF16 and F16 in `InferRowToF32`, `TBpeTokenizer.CreateFromVocabMerges`,
  `TInferThreadPool.RunWorkers` with a worker index; F16 KV cache stays the default.
- **2026-10-19 - B13: identical CPU results under Delphi and FPC.** All inference units set
  `{$EXCESSPRECISION OFF}`; Delphi decodes 3-14 % faster.
- **2026-10-17 - D3a: MoE on CUDA** (routing still on the host).
- **2026-10-17 - D2: MoE on the CPU** (Qwen3-30B-A3B from safetensors and GGUF).
- **2026-10-17 - D1: shared decoder types** (`TDecoderModel`, per-layer `Spec`,
  tokenizer/template/parser interfaces, family registration).
- **2026-10-17 - B11: GPUs with a small buffer limit.** The head is computed in parts
  (Adreno X1-85, 128 MB per binding).
- **2026-10-17 - B10: `matvec8.comp` Q8_0 on the Adreno X1-85.** The word index is computed
  without the carry the Adreno compiler lost.
- **2026-10-17 - B7: embedding context stays at 512 tokens** (decision). RAG splits documents
  into chunks anyway (`ChunkText`), short chunks match more precisely, attention stays small,
  and vectors stored by lembed (contract v1) stay compatible. A longer text is rejected, not
  truncated; a larger limit only when a use case needs whole long texts as one vector.
- **2026-10-17 - B6: `scripts/build-delphi-demos.ps1`** for Delphi 13 and Delphi 7; Delphi
  attention no longer allocates per call.
- **2026-10-17 - B5: platform runs refreshed** in the README table, Windows ARM64 inference
  for the first time.
- **2026-10-17 - B3: CUDA backend** (C1-C5c: driver API loaded at runtime, embedded PTX, the
  whole forward pass on the card, decode as a CUDA graph - faster than Vulkan).
- **2026-10-17 - B2: Q4_0 unpacking** (Pascal word-wise, NEON); Q4_0 decode at or above Q8_0
  on ARM and x64.
- **2026-10-17 - B1: K-quant two-row kernels** completed with Q6_K and AVX2 (Q4_K_M prefill
  462 to 579 tokens/s on an i9-14900K).
- **2026-10-17 - B8: decode attention for long context.** Half-float KV cache (default on)
  and `attndec.comp` on Apple GPUs (M2 Pro at 4000 tokens 74 instead of 43 tokens/s).
- **2026-10-16 - B8: Vulkan on macOS (MoltenVK).** `matvec8.comp`, `gemm8.comp` and
  `flash.comp`; the GPU beats the CPU of the M2 Pro at 200 tokens in prefill and decode.
- **2026-10-16 - B9: test for Vulkan attention heads above 128.**
- **2026-10-16 - B1: Q4_K two-row kernel** (NEON; Q4_K_M prefill 275 to 299 tokens/s on a
  Snapdragon X).
