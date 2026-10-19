"""Qwen3-TTS reference for the native Pascal engine (mormot.ai.tts.qwen).

Says one sentence in greedy mode (do_sample=False) with the official qwen-tts
package, on the CPU in float32, and saves everything needed to compare the
port step by step: prompt ids, talker inputs and logits, code predictor
logits (frame 0), the 16xN codes, the intermediate outputs of the 12 Hz
decoder, and the wav.

    pip install qwen-tts==0.1.1 soundfile
    python tools/tts/qwen_ref.py
    python tools/tts/qwen_ref.py --text "Hostile in local." --language Auto --out auto-vivian

Output: <root>/<out>/ (root: --root, or the QWEN_TTS_REF environment
variable, or ./ref): manifest.json (settings, ids, codes, shape of each
tensor) and one raw file per tensor, float32 (.f32) or int32 (.i32),
little-endian, C order (the last dimension varies fastest).
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel


class Dump:
    """The saved tensors and their description for the manifest."""

    def __init__(self, folder):
        self.folder = folder
        self.tensors = {}
        os.makedirs(folder, exist_ok=True)

    def add(self, name, value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            value = value.float() if value.is_floating_point() else value.long()
            value = value.numpy()
        a = np.ascontiguousarray(value)
        if a.dtype.kind == "f":
            a, kind = a.astype("<f4"), "f32"
        else:
            a, kind = a.astype("<i4"), "i32"
        file = f"{name}.{kind}"
        a.tofile(os.path.join(self.folder, file))
        self.tensors[name] = {"file": file, "dtype": kind, "shape": list(a.shape)}


def hook_talker(tts, dump):
    """Capture the inputs and logits of the talker and code predictor during generate."""
    talker = tts.model.talker
    predictor = talker.code_predictor
    seen = {"talker": 0, "predictor": 0}
    step_embeds, step_logits, pred_logits = [], [], []

    # inputs_embeds of each talker call: the prefill sequence, then the sum of
    # the 16 embeddings of the previous frame + the following text
    def talker_model_pre(module, args, kwargs):
        step_embeds.append(kwargs["inputs_embeds"][0].detach().clone())

    def talker_post(module, args, kwargs, output):
        n = seen["talker"]
        seen["talker"] += 1
        step_logits.append(output.logits[0, -1].detach().clone())
        if n == 0:
            dump.add("talker.trailing_text", kwargs["trailing_text_hidden"][0])
            dump.add("talker.tts_pad", kwargs["tts_pad_embed"].reshape(-1))
            # input of each layer then normed output of the last one (29 x L x 1024)
            dump.add("talker.prefill.layers", torch.stack([h[0] for h in output.hidden_states[0]]))
            dump.add("talker.prefill.past_hidden", output.past_hidden.reshape(-1))

    # frame 0: the prefill of the code predictor (talker hidden + embedding of
    # code 0) then the 15 logits, one per group
    def predictor_post(module, args, kwargs, output):
        n = seen["predictor"]
        seen["predictor"] += 1
        if n >= 15:
            return
        if n == 0:
            dump.add("predictor.f0.embeds", kwargs["inputs_embeds"][0])
        pred_logits.append(output.logits[0, -1].detach().clone())

    hooks = [
        talker.model.register_forward_pre_hook(talker_model_pre, with_kwargs=True),
        talker.register_forward_hook(talker_post, with_kwargs=True),
        predictor.register_forward_hook(predictor_post, with_kwargs=True),
    ]

    def finish():
        for h in hooks:
            h.remove()
        dump.add("talker.prefill.embeds", step_embeds[0])
        if len(step_embeds) > 1:
            dump.add("talker.step.embeds", torch.cat(step_embeds[1:], dim=0))
        dump.add("talker.prefill.logits", step_logits[0])
        if len(step_logits) > 1:
            dump.add("talker.step.logits", torch.stack(step_logits[1:]))
        dump.add("predictor.f0.logits", torch.stack(pred_logits))

    return finish


def dump_decoder(dump, decoder, codes, blocks):
    """Replay Qwen3TTSTokenizerV2Decoder.forward step by step; return the wav."""
    tf = decoder.pre_transformer
    outs = {}
    hooks = [tf.input_proj.register_forward_hook(lambda m, a, o: outs.__setitem__("in", o))]
    for i, layer in enumerate(tf.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, a, o, i=i: outs.__setitem__(i, o[0] if isinstance(o, tuple) else o)))
    try:
        h = decoder.quantizer.decode(codes.t().unsqueeze(0))          # 1 x 512 x N
        dump.add("dec.quant", h[0])
        h = decoder.pre_conv(h)                                        # 1 x 1024 x N
        dump.add("dec.preconv", h[0])
        t = tf(inputs_embeds=h.transpose(1, 2)).last_hidden_state      # 1 x N x 1024
        dump.add("dec.tf.in", outs["in"][0])                           # N x 512
        for i in range(len(tf.layers)):
            dump.add(f"dec.tf.layer{i}", outs[i][0])                   # N x 512
        dump.add("dec.tf.out", t[0])
        h = t.permute(0, 2, 1)
        for i, stage in enumerate(decoder.upsample):                   # 1024 x 2N, 1024 x 4N
            for b in stage:
                h = b(h)
            dump.add(f"dec.up{i}", h[0])
        last = len(decoder.decoder) - 1
        for i, b in enumerate(decoder.decoder):
            h = b(h)
            # 0: conv 1024->1536; 1..4: blocks 768/384/192/96; 5: snake; 6: conv -> 1
            if i == 0 or i == last or (blocks and i < last - 1):
                dump.add(f"dec.dec{i}", h[0])
    finally:
        for hk in hooks:
            hk.remove()
    return h.clamp(min=-1, max=1).reshape(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    p.add_argument("--text", default="Cible en vue, deux hostiles en local.")
    p.add_argument("--speaker", default="vivian")
    p.add_argument("--language", default="French", help="French, English... or Auto")
    p.add_argument("--streaming-text", action="store_true",
                   help="text given frame by frame (non_streaming_mode=False)")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--no-blocks", action="store_true",
                   help="do not save the outputs of the 4 decoder blocks (tens of MB)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="fr-vivian")
    p.add_argument("--root", default=os.environ.get("QWEN_TTS_REF", "ref"))
    args = p.parse_args()

    torch.manual_seed(0)
    folder = os.path.join(args.root, args.out)
    dump = Dump(folder)
    path = snapshot_download(args.model)
    print(f"loading {path} on {args.device} (float32, eager) ...", flush=True)
    tts = Qwen3TTSModel.from_pretrained(path, device_map=args.device, dtype=torch.float32,
                                        attn_implementation="eager")
    tts.model.eval()
    try:  # the 12 Hz decoder does not receive attn_implementation
        tts.model.speech_tokenizer.model.set_attn_implementation("eager")
    except Exception as e:
        print(f"decoder attention left as is: {e}", flush=True)

    text = tts._build_assistant_text(args.text)
    input_ids = tts._tokenize_texts([text])[0]
    ids = input_ids[0].tolist()
    dump.add("prompt.ids", input_ids[0])
    gen = tts._merge_generate_kwargs(do_sample=False, subtalker_dosample=False,
                                     max_new_tokens=args.max_frames)
    tc = tts.model.config.talker_config
    suppress = [i for i in range(tc.vocab_size - 1024, tc.vocab_size) if i != tc.codec_eos_token_id]

    finish = hook_talker(tts, dump)
    t0 = time.perf_counter()
    with torch.inference_mode():
        codes_list, hidden_list = tts.model.generate(
            input_ids=[input_ids], instruct_ids=[None], languages=[args.language],
            speakers=[args.speaker], non_streaming_mode=not args.streaming_text, **gen)
    t_gen = time.perf_counter() - t0
    finish()
    codes = codes_list[0]                                              # N x 16
    dump.add("codes", codes)
    dump.add("talker.hidden", hidden_list[0])                          # N x 1024

    t0 = time.perf_counter()
    with torch.inference_mode():
        wavs, sr = tts.model.speech_tokenizer.decode([{"audio_codes": codes}])
    t_dec = time.perf_counter() - t0
    wav = np.asarray(wavs[0], dtype=np.float32)
    dump.add("wav", wav)
    sf.write(os.path.join(folder, "out.wav"), wav, sr, subtype="PCM_16")

    with torch.inference_mode():
        mine = dump_decoder(dump, tts.model.speech_tokenizer.model.decoder, codes,
                            not args.no_blocks).numpy()
    n = min(len(mine), len(wav))
    gap = float(np.abs(mine[:n] - wav[:n]).max()) if n else 0.0

    dur = len(wav) / sr
    manifest = {
        "model": args.model,
        "snapshot": path,
        "text": args.text,
        "prompt": text,
        "speaker": args.speaker,
        "language": args.language,
        "non_streaming_mode": not args.streaming_text,
        "device": args.device,
        "dtype": "float32",
        "attn_implementation": "eager",
        "torch": torch.__version__,
        "generate": {
            **{k: v for k, v in gen.items()},
            "min_new_tokens": 2,
            "eos_token_id": tc.codec_eos_token_id,
            "suppress_tokens": [suppress[0], suppress[-1]],
            "note": "greedy: the talker repetition_penalty applies to the group 0 codes "
                    "already produced, eos forbidden for the first 2 frames, "
                    "2048..3071 but eos suppressed",
        },
        "sample_rate": sr,
        "frames": int(codes.shape[0]),
        "samples": len(wav),
        "audio_seconds": dur,
        "generate_seconds": t_gen,
        "decode_seconds": t_dec,
        "decoder_replay_max_abs": gap,
        "prompt_ids": ids,
        "prompt_tokens": tts.processor.tokenizer.convert_ids_to_tokens(ids),
        "codes": codes.tolist(),
        "tensors": dump.tensors,
    }
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"{dur:.2f}s of audio ({codes.shape[0]} frames) in {t_gen:.2f}s + decode {t_dec:.2f}s "
          f"(x{dur / max(t_gen + t_dec, 1e-6):.2f} real time); decoder replay max |diff| {gap:.2e}",
          flush=True)
    print(f"written to {folder}", flush=True)


if __name__ == "__main__":
    main()
