# Ideen — ohne Termin

Gedanken, die noch kein Backlog-Eintrag sind: kein Plan, keine Zusage. Wird eine Idee
konkret, wandert sie nach [backlog.md](backlog.md) und wird hier gestrichen.

## Sprachausgabe auf der CPU: Kokoro-82M (notiert 2026-10-19)

Anlass: neural-api spricht Text mit Kokoro-82M ganz in FPC und ohne GPU
(`neuralkokorosession`, Beispiel `examples/KokoroTTS`; nur als Hinweis, dass es geht - LGPL,
keine Vorlage für Code).

- **Modell:** hexgrad/Kokoro-82M (Apache-2.0), StyleTTS2 mit iSTFTNet-Vocoder, 82 Mio.
  Parameter; Stimmen als Stil-Vektoren. Kein Decoder im Sinne von Phase D: eine neue
  Modellklasse (Text-Encoder, Dauer-/Prosodie-Vorhersage, Vocoder mit inverser STFT).
- **Ausspracheumwandlung (G2P):** Kokoro erwartet Phoneme. hexgrad/misaki liefert Wörterbücher
  und Regeln (Apache-2.0); der übliche Rückfall für unbekannte Wörter ist espeak-ng - **GPL**,
  käme so nicht in Frage. Für Deutsch wäre eigene Arbeit nötig (misaki deckt vor allem
  Englisch ab).
- **Abgrenzung:** flydev baut Qwen3-TTS auf mormot2-ai (`flydev-fr/mormot2-qwen-tts`, GPU,
  mehrsprachig); was er braucht, steht in [backlog.md](backlog.md) B14. Kokoro wäre der
  kleine Gegenpol: CPU, schnell, wenige Stimmen.
- **Was wiederverwendbar wäre:** safetensors-Loader, Linear-/Conv-Kernel der CPU, Threads.
  Neu: 1-D-Convolution in Breite, LSTM, AdaIN, inverse STFT, die G2P-Stufe.
- **Offen:** Bedarf (wer braucht Sprachausgabe in einer Handwerks-App?), Lizenzweg für G2P
  ohne GPL, Sprache Deutsch.
