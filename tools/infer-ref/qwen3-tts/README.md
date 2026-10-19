# qwen3-tts — generator of the dataset `qwen3-tts-lm`

Taken over unchanged from [flydev-fr/mormot2-qwen-tts](https://github.com/flydev-fr/mormot2-qwen-tts)
(commit `8f5a84692bb00c2829b7e910cad91e93c6175104`, there under `tools/tts/`; license as for mORMot and this library:
MPL 1.1 / GPL 2.0 / LGPL 2.1). Provenance: `UPSTREAM_BASE` and `NOTICE`.

- `qwen_ref.py` lets the official package `qwen-tts` 0.1.1 speak a sentence greedily
  (CPU, float32, eager) and stores inputs, states and logits.
- `make_lm_reference.py` builds the dataset from it in the format of `tests/data/infer-ref`
  (standard library only).

Because the scripts are unchanged, their paths point to the layout of the original repo.
Here, pass `--out` explicitly:

    pip install qwen-tts==0.1.1 soundfile
    python tools/infer-ref/qwen3-tts/qwen_ref.py --root <ref>
    python tools/infer-ref/qwen3-tts/make_lm_reference.py --ref <ref>/fr-vivian \
        --model <snapshot of Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice> \
        --out tests/data/infer-ref/qwen3-tts-lm

Model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, revision
`85e237c12c027371202489a0ec509ded67b5e4b5` (Apache-2.0); hashes of `config.json` and
`model.safetensors` are in the manifest of the dataset.

The tests look for the model in `vendor/models/qwen3-tts-12hz-0.6b-customvoice/`
(gitignored): `config.json` and `model.safetensors` for the stacks, and the tokenizer files
`vocab.json`, `merges.txt` and `tokenizer_config.json` of the same revision for
`VocabMergesMatchTokenizerJson` (the model ships no `tokenizer.json`; their SHA-256 hashes
are in `tests/test.infer.bpe.pas`). Without them the tests say SKIP, with
`LAI_REQUIRE_MODEL=1` they fail.
