# qwen3-tts — Erzeuger des Datensatzes `qwen3-tts-lm`

Unverändert übernommen aus [flydev-fr/mormot2-qwen-tts](https://github.com/flydev-fr/mormot2-qwen-tts)
(Commit `8f5a84692bb00c2829b7e910cad91e93c6175104`, dort unter `tools/tts/`; Lizenz wie mORMot und diese Bibliothek:
MPL 1.1 / GPL 2.0 / LGPL 2.1). Herkunft: `UPSTREAM_BASE` und `NOTICE`.

- `qwen_ref.py` lässt das offizielle Paket `qwen-tts` 0.1.1 einen Satz greedy sprechen
  (CPU, float32, eager) und speichert Eingaben, Zustände und Logits.
- `make_lm_reference.py` baut daraus den Datensatz im Format von `tests/data/infer-ref`
  (nur Standardbibliothek).

Weil die Skripte unverändert sind, zeigen ihre Pfade auf das Layout des Ursprungs-Repos.
Hier `--out` ausdrücklich angeben:

    pip install qwen-tts==0.1.1 soundfile
    python tools/infer-ref/qwen3-tts/qwen_ref.py --root <ref>
    python tools/infer-ref/qwen3-tts/make_lm_reference.py --ref <ref>/fr-vivian \
        --model <snapshot von Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice> \
        --out tests/data/infer-ref/qwen3-tts-lm

Modell: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, Revision
`85e237c12c027371202489a0ec509ded67b5e4b5` (Apache-2.0); Hashes von `config.json` und
`model.safetensors` stehen im Manifest des Datensatzes.
