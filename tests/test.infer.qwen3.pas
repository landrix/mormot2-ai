/// tests for the Qwen3 decoder of the native inference engine (phase 2b, 2c)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - 2b: every state and the logits against tests/data/infer-ref/tiny-qwen3
//   (committed, BF16 weights) and tests/data/infer-ref/qwen3 (Qwen3-0.6B, FP32)
// - 2c, the gate of phase 2: greedy continuations token-identical to
//   transformers; the KV cache gives the same logits as a full forward pass
// - the real model needs vendor/models/qwen3-0.6b (or $LAI_QWEN3_DIR) and a
//   64-bit process; it says SKIP otherwise, LAI_REQUIRE_MODEL=1 makes that a
//   failure
unit test.infer.qwen3;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.perf,
  mormot.core.test,
  mormot.crypt.secure,
  mormot.ai.infer.tensor,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.decoder,
  mormot.ai.infer.model.qwen3,
  mormot.ai.infer.model.gptoss,
  mormot.ai.infer.model.gemma4,
  mormot.ai.infer.sampler,
  test.infer.ref,
  test.infer.fast,
  test.infer.bpe;

type
  /// what the decoder tests share: both CPU backends, the states of a Forward
  // kept by name, the comparisons against a reference
  TTestInferDecoder = class(TSynTestCase)
  protected
    fCpu: TInferCpuBackend;
    fFast: TInferCpuFastBackend;
    fNames: TRawUtf8DynArray;
    fStates: array of TSingleDynArray;
    procedure Setup; override;
    procedure CleanUp; override;
    /// the cases of a real model's reference on aModel (case0 .. case<aCases
    // - 1>, last rows): the last row of every state, the logits, greedy (a
    // step may differ only where the reference itself was less than 1 sure)
    procedure CheckLastRows(aModel: TDecoderModel; aRef: TInferRefSet;
      const aLabel: RawUtf8; aStateCos, aCos, aMaxAbs: double; aCases: PtrInt = 4);
    /// TOnQwen3State: copy every state, the model reuses its buffers
    procedure KeepState(const aName: RawUtf8; aValues: PSingleArray;
      aRows, aCols: PtrInt);
    function State(const aName: RawUtf8): PSingleArray;
    /// forward <aPrefix>.ids from an empty cache and compare every kept state
    procedure CheckStates(aModel: TDecoderModel; aRef: TInferRefSet;
      const aPrefix: RawUtf8; aAtol, aRtol: double; aAllLogits: boolean);
  end;

  /// phase 2b and 2c: the Qwen3 decoder against the references
  TTestInferQwen3 = class(TTestInferDecoder)
  protected
    fItemThread: array of TThreadID; // the thread of each item of ThreadItem
    procedure CheckArgMaxRandom;
    /// one item of RunItems in LoadOnAllCores: 1 ms, its thread noted
    procedure ThreadItem(aItem: integer);
    /// the real model folder, '' (and SKIP or a failure) when absent
    function RealModel: TFileName;
    /// the tiny Qwen3 quantized to aType: Exact = the scalar reference bit for
    // bit, the float path within aFloatTol, both against FP32 within aRefTol
    procedure TinyQuantized(aType: TInferDType; aFloatTol, aRefTol, aRefCos: double);
    /// Qwen3-0.6B quantized to aType against the FP32 reference: last logits
    // within aCos / aMaxAbs, greedy equal up to a step the reference was less
    // than aMargin sure of
    // - aGguf: the model from that file instead of quantized at load
    // - aRefName / aFolder: another model's reference and folder (Qwen2.5)
    procedure Qwen3Quantized(aType: TInferDType; aCos, aMaxAbs, aMargin: double;
      const aGguf: TFileName = ''; const aRefName: RawUtf8 = 'qwen3';
      const aFolder: TFileName = '');
    /// the folder of Qwen2.5-0.5B-Instruct, '' (SKIP, or a failure with
    // LAI_REQUIRE_MODEL=1) when absent or in a 32-bit process
    function Qwen25Folder: TFileName;
  published
    /// the tiny Qwen3 (BF16 weights): every state and all logits
    procedure TinyForwardMatchesReference;
    /// token by token through the KV cache = one forward pass, bit for bit
    procedure TinyCacheEqualsFullPass;
    /// the tiny greedy continuations are token-identical
    procedure TinyGreedyMatchesReference;
    /// wrong input and foreign configurations are refused
    procedure RefusesBadInput;
    /// 130 tokens: the cache grows twice and stays bit-identical to one pass
    procedure TinyLongContext;
    /// the context limit, Greedy edge cases, a context bound to its model
    procedure TinyLimitsAndContexts;
    /// Reuse: a cached prompt prefix gives the logits of a fresh pass, bit for
    // bit, on both backends and with Q8_0; Truncate, Id, another model
    procedure TinyPromptReuse;
    /// a model with its own lm_head (tie_word_embeddings false)
    procedure TinyUntiedHead;
    /// InferArgMax: ties, NaN, signs, -0, denormals, random bit patterns
    procedure ArgMaxTiesAndNan;
    /// Qwen3-0.6B: states of two prompts, last logits of all
    procedure Qwen3ForwardMatchesReference;
    /// Qwen3-0.6B: greedy continuations token-identical (the phase 2 gate)
    procedure Qwen3GreedyMatchesReference;
    /// the tiny Qwen3 with Q8_0 matrices: Exact = the scalar reference bit for
    // bit, the float path and the reference within a tolerance
    procedure TinyQ8;
    /// Qwen3-0.6B with Q8_0 matrices against the FP32 reference (phase 3):
    // logits within a tolerance, greedy equal up to the first near tie
    procedure Qwen3Q8AgainstReference;
    /// the tiny Qwen3 with Q4_0 layers (and a Q8_0 head)
    procedure TinyQ4;
    /// Qwen3-0.6B with Q4_0 layers against the FP32 reference
    procedure Qwen3Q4AgainstReference;
    /// the tiny Qwen3 from GGUF files: settings, matrices byte for byte like
    // ours, logits, F16 and an untied head, damaged files refused
    procedure TinyGguf;
    /// D5e: the weights converted at load on the threads of the fast backend
    // (Backend.RunItems) - byte for byte those of one thread, in F32, Q8_0 and
    // Q4_0 and from an F16 GGUF; a NaN still refused by name
    procedure LoadOnAllCores;
    /// Qwen's own Q8_0 GGUF of Qwen3-0.6B: every matrix our bytes, the Q8_0 gate
    procedure Qwen3Gguf;
    /// a Q4_K_M GGUF of Qwen3-0.6B (Q4_K and Q6_K matrices) against FP32
    procedure Qwen3GgufQ4KM;
    /// the decoder families: registration, NewDecoderModel by model_type and
    // architecture, the kind of every layer, foreign families refused
    procedure DecoderFamilies;
    /// tiny Qwen3-MoE: every state, the cache, greedy, unnormed top-k weights
    procedure TinyMoeMatchesReference;
    /// tiny Qwen3-MoE in Q8_0 and from GGUF with 3-D expert tensors
    procedure TinyMoeQuantized;
    /// Qwen3-MoE settings refused
    procedure TinyMoeRefused;
    /// TDecoderRouteTape: a run replaying the recorded choice of the same
    // tokens gives the same bits and no differing choice; the choice of other
    // tokens differs (with a gap) and changes the logits; nothing left to
    // replay raises
    procedure RouteTapeReplays;
    /// D1b, the tiny Qwen2 (bias on Q, K, V, no head norm, head_dim = hidden /
    // heads): every state, greedy; from GGUF (no key_length); a file without
    // a bias refused
    procedure TinyQwen2MatchesReference;
    /// Qwen2.5-0.5B-Instruct in FP32: states, last logits, greedy identical
    procedure Qwen25MatchesReference;
    /// Qwen's own GGUF of Qwen2.5-0.5B-Instruct: Q8_0 against FP32; its
    // Q4_K_M (mostly Q5_0: hidden 896 is no multiple of 256) against Qwen's
    // Q8_0 file and FP32
    procedure Qwen25Gguf;
    /// D4, sliding window and NoPE: tiny SmolLM3 (every combination of window
    // and rotary positions) and a tiny Qwen3 with use_sliding_window - every
    // state and greedy continuations that wrap the window
    procedure SlidingWindowMatchesReference;
    /// D4 gate: the sliding cache gives the bits of a full cache in every
    // chunk size, wrapping many times, keeping twice the window; a branch
    // after every length gives the bits of computing it again (Truncate where
    // Restorable, Reuse from scratch where not)
    procedure SlidingWindowCache;
    /// D4: the layer kinds derived from config.json as transformers does, and
    // the settings refused
    procedure SlidingWindowConfig;
    /// B14: embedding rows in place of ids (ForwardEmbeds) - the rows of Embed
    // give the bits of the ids, mixed with ids over chunks; no id in the cache
    // there, so Reuse stops at the first row
    procedure EmbeddingRows;
    /// B14: a decoder as one stack of a larger checkpoint (CreateStack) - the
    // settings from an object of config.json, the tensors under a prefix, its
    // own head, no embedding; the bits of the plain model; bad stacks refused
    // - D5e: no head either - the normed last row (aNormed) only, logits
    // refused before the cache changes
    procedure StackFromCheckpoint;
    /// B14: the talker and the code predictor of Qwen3-TTS-12Hz-0.6B-CustomVoice
    // fed with embedding rows, against PyTorch (flydev's qwen3-tts-lm
    // reference): every talker state and its logits, the 15 logit rows and
    // greedy codes of the predictor - skipped without the model
    procedure Qwen3TtsStacks;
    /// D5: tiny gpt-oss against transformers - YaRN factors, sinks, biases,
    // clamped SwiGLU: every state, the cache, greedy continuations
    procedure GptOssMatchesReference;
    /// D5: gpt-oss quantized to Q8_0 at load, and on the fast backend
    procedure GptOssQuantized;
    /// D5: the settings of a gpt-oss config.json - defaults, the transformers
    // 4 form, the YaRN variants, refusals
    procedure GptOssConfig;
    /// D5b: tiny gpt-oss with MXFP4 experts in two files and an index -
    // the repacked experts equal transformers' values, the states within the
    // error of Q8_0 activations, refusals of the quantization settings and
    // of exponents beyond the exact range
    procedure GptOssMxfp4;
    /// D5b: gpt-oss-20b (MXFP4 experts, three files) against transformers in
    // float32 - the last row of every state, the logits, greedy - skipped
    // without vendor/models/gpt-oss-20b
    procedure GptOss20b;
    /// D5b2: tiny gpt-oss as GGUF (llama.cpp's converter layout) - the same
    // settings, the same expert bytes and bit for bit the same logits as the
    // safetensors folder, Q8_0 matrices within tolerance, refusals
    procedure GptOssGguf;
    /// D6a: tiny Gemma 4 against transformers - heads of two shapes, K = V,
    // both rotary sets, the norms around both blocks, GeGLU, the dense block
    // beside the experts, layer scalars, the soft cap: every state, the
    // cache, greedy; a dense model from the default layer types and rope
    procedure Gemma4MatchesReference;
    /// D6a: tiny Gemma 4 quantized to Q8_0 and Q4_0 at load, and on the fast
    // backend
    procedure Gemma4Quantized;
    /// D6a: the settings of a Gemma 4 config.json refused
    procedure Gemma4Refused;
  end;

const
  /// the folder of Qwen3-TTS-12Hz-0.6B-CustomVoice under vendor/models
  TTS_MODEL = 'qwen3-tts-12hz-0.6b-customvoice';

/// the folder of Qwen3-TTS-12Hz-0.6B-CustomVoice (with a trailing delimiter),
// '' when absent
function TtsFolder: TFileName;

/// the talker of Qwen3-TTS (28 layers, codec embedding, codec head)
function NewTtsTalker(const aFolder: TFileName; aBackend: TInferBackend): TQwen3Model;

/// the code predictor of Qwen3-TTS (5 layers, no embedding, head of group 0)
function NewTtsPredictor(const aFolder: TFileName; aBackend: TInferBackend): TQwen3Model;

/// true when config.json and model.safetensors in aFolder have the sha256 of
// the manifest of aRef (model.files)
function TtsFilesMatch(const aFolder: TFileName; aRef: TInferRefSet;
  out aWhy: RawUtf8): boolean;

/// the greedy tokens of aPrefix against the reference: the first differing
// step, -1 when all agree; aMargin receives the reference top-2 margin there
function GreedyDiff(const aMine: TIntegerDynArray; aRef: TInferRefSet;
  const aPrefix: RawUtf8; out aMargin: double; out aDetail: RawUtf8): PtrInt;

/// the rotary factors of positions 0..63 of aModel against <aName>.cos and
// <aName>.sin of aRef: within one float32 step of the reference's cosf/sinf
function RotaryMatches(aModel: TDecoderModel; aRef: TInferRefSet;
  const aName: RawUtf8; out aWhy: RawUtf8): boolean;

/// aDoc.D[aName] := aValue - which FPC 3.2.4 on aarch64-darwin compiles to
// store 0 (TDocVariantData.SetDoubleByName): through a variant instead
procedure SetFloat(var aDoc: TDocVariantData; const aName: RawUtf8; aValue: double);


implementation

{ TTestInferDecoder }

procedure TTestInferDecoder.Setup;
begin
  fCpu := TInferCpuBackend.Create;
  fFast := TInferCpuFastBackend.Create;
end;

procedure TTestInferDecoder.CleanUp;
begin
  FreeAndNil(fFast); // re-entrant, as mORMot requires
  FreeAndNil(fCpu);
end;

procedure TTestInferDecoder.KeepState(const aName: RawUtf8; aValues: PSingleArray;
  aRows, aCols: PtrInt);
var
  n: PtrInt;
begin
  n := length(fNames);
  SetLength(fNames, n + 1);
  SetLength(fStates, n + 1);
  fNames[n] := aName;
  SetLength(fStates[n], aRows * aCols);
  MoveFast(aValues^, fStates[n][0], aRows * aCols * SizeOf(single));
end;

function TTestInferDecoder.State(const aName: RawUtf8): PSingleArray;
var
  i: PtrInt;
begin
  for i := 0 to high(fNames) do
    if fNames[i] = aName then
    begin
      result := pointer(fStates[i]);
      exit;
    end;
  result := nil;
end;

procedure TTestInferDecoder.CheckStates(aModel: TDecoderModel; aRef: TInferRefSet;
  const aPrefix: RawUtf8; aAtol, aRtol: double; aAllLogits: boolean);
var
  ctx: TQwen3Context;
  ids, n, i, count, vocab: PtrInt;
  idp: PIntegerArray;
  logits: TSingleDynArray;
  ref, mine: PSingleArray;
  cmp: TInferCompare;
  t: PInferRefTensor;
  name: RawUtf8;
begin
  fNames := nil;
  fStates := nil;
  idp := aRef.I32(aPrefix + '.ids', ids);
  vocab := aModel.Config.VocabSize;
  if aAllLogits then
    SetLength(logits, ids * vocab)
  else
    SetLength(logits, vocab);
  ctx := aModel.NewContext;
  try
    aModel.Forward(ctx, idp, ids, pointer(logits), aAllLogits, KeepState);
    CheckEqual(ctx.Length, ids, 'tokens cached');
  finally
    ctx.Free;
  end;
  for i := 0 to high(aRef.Tensors) do
  begin
    t := @aRef.Tensors[i];
    if copy(t^.Name, 1, length(aPrefix) + 1) <> aPrefix + '.' then
      continue;
    name := copy(t^.Name, length(aPrefix) + 2, maxInt);
    if (name = 'ids') or (name = 'greedy') or (name = 'greedy_margin') or
       (name = 'logits_steps') then
      continue;
    ref := aRef.F32(t^.Name, n);
    if name = 'logits' then
      mine := pointer(logits)
    else if name = 'logits_last' then
      mine := pointer(logits)
    else
      mine := State(name);
    if CheckFailed(mine <> nil, Utf8ToString(aPrefix + ' has no state ' + name)) then
      continue;
    count := n;
    cmp := InferCompare(mine, ref, count, aAtol, aRtol);
    CheckUtf8(InferCompareOk(cmp), '%.%: %', [aPrefix, name, InferCompareText(cmp)]);
    if (name = 'logits') or (name = 'logits_last') or (name = 'norm') then
      AddConsole(FormatString('%.%: %', [aPrefix, name, InferCompareText(cmp)]));
  end;
end;

procedure TTestInferQwen3.TinyForwardMatchesReference;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  k: integer;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fCpu);
    CheckEqual(model.Config.HeadDim, 32, 'head_dim from the configuration');
    CheckEqual(model.Config.KvHeads, 2, 'grouped-query attention');
    Check(model.Config.TiedEmbeddings, 'tied embeddings');
    for k := 0 to 2 do
      CheckStates(model, ref, FormatUtf8('case%', [k]), 2e-6, 2e-5, true);
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.TinyCacheEqualsFullPass;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  full, step: TQwen3Context;
  ids: PIntegerArray;
  n, i, v, vocab: PtrInt;
  all, one: TSingleDynArray;
  same: boolean;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  full := nil;
  step := nil;
  try
    model := TQwen3Model.Create(dir, fCpu);
    vocab := model.Config.VocabSize;
    ids := ref.I32('case2.ids', n);
    full := model.NewContext;
    step := model.NewContext;
    SetLength(all, n * vocab);
    SetLength(one, vocab);
    model.Forward(full, ids, n, pointer(all), true);
    same := true;
    for i := 0 to n - 1 do
    begin
      // one token at a time: the earlier ones come from the KV cache
      model.Forward(step, @ids^[i], 1, pointer(one));
      for v := 0 to vocab - 1 do
        if PCardinal(@one[v])^ <> PCardinal(@all[i * vocab + v])^ then
          same := false;
    end;
    Check(same, 'cached logits bit-identical to the full pass');
    CheckEqual(step.Length, n, 'cache length');
    // the same context, reset: the same result again
    step.Reset;
    model.Forward(step, ids, n, pointer(one));
    same := CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single));
    Check(same, 'context reused after Reset');
    // prefill in two chunks
    step.Reset;
    model.Forward(step, ids, 4, nil);
    model.Forward(step, @ids^[4], n - 4, pointer(one));
    same := CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single));
    Check(same, 'prefill in two chunks');
  finally
    step.Free;
    full.Free;
    model.Free;
    ref.Free;
  end;
end;

/// ids as text, for messages
function IdsOf(const aIds: TIntegerDynArray): RawUtf8;
var
  i: PtrInt;
begin
  result := '';
  for i := 0 to high(aIds) do
    result := result + Int32ToUtf8(aIds[i]) + ' ';
end;

function GreedyDiff(const aMine: TIntegerDynArray; aRef: TInferRefSet;
  const aPrefix: RawUtf8; out aMargin: double; out aDetail: RawUtf8): PtrInt;
var
  want: PIntegerArray;
  margins: PSingleArray;
  n, m, i: PtrInt;
begin
  want := aRef.I32(aPrefix + '.greedy', n);
  margins := aRef.F32(aPrefix + '.greedy_margin', m);
  aMargin := 0;
  result := -1;
  for i := 0 to n - 1 do
    if (i >= length(aMine)) or (aMine[i] <> want^[i]) then
    begin
      result := i;
      if i < m then
        aMargin := margins^[i];
      break;
    end;
  if (result < 0) and (length(aMine) <> n) then
    result := n;
  aDetail := FormatUtf8('% tokens, reference %', [length(aMine), n]);
end;

procedure TTestInferQwen3.TinyGreedyMatchesReference;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  ctx: TQwen3Context;
  ids: PIntegerArray;
  n, k, diff: PtrInt;
  prompt, mine: TIntegerDynArray;
  margin: double;
  detail: RawUtf8;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fCpu);
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        mine := model.Greedy(ctx, prompt, 16, [2]);
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, FormatUtf8('case%', [k]), margin, detail);
      CheckUtf8(diff < 0, 'case% differs at step % (margin %, %)',
        [k, diff, margin, detail]);
    end;
  finally
    model.Free;
    ref.Free;
  end;
end;

function Raises(aModel: TQwen3Model; aCtx: TQwen3Context; aId, aCount: integer): boolean;
var
  ids: array[0..1] of integer;
begin
  result := false;
  ids[0] := aId;
  ids[1] := aId;
  try
    aModel.Forward(aCtx, @ids, aCount, nil);
  except
    on EInferError do
      result := true;
  end;
end;

procedure TTestInferQwen3.RefusesBadInput;
var
  dir, tmp: TFileName;
  model: TQwen3Model;
  ctx: TQwen3Context;
  cfg: RawUtf8;
  refused: boolean;
  st, bad: RawByteString;
  hdr: variant;
  pos: PtrInt;

  function LoadRefused(const aFrom, aTo: RawUtf8): boolean;
  begin
    result := false;
    FileFromString(StringReplaceAll(cfg, aFrom, aTo), tmp + 'config.json');
    try
      TQwen3Model.Create(tmp, fCpu).Free;
    except
      on EInferError do
        result := true;
    end;
  end;

begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  model := TQwen3Model.Create(dir, fCpu);
  ctx := model.NewContext;
  try
    Check(Raises(model, ctx, -1, 1), 'negative id');
    Check(Raises(model, ctx, model.Config.VocabSize, 1), 'id beyond the vocabulary');
    Check(Raises(model, ctx, 3, 0), 'no tokens');
    CheckEqual(ctx.Length, 0, 'nothing cached by refused calls');
    refused := false;
    try
      model.Forward(nil, nil, 1, nil);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused, 'no context');
  finally
    ctx.Free;
    model.Free;
  end;
  // foreign configurations: a copy of the tiny model with one setting changed
  cfg := StringFromFile(dir + 'config.json');
  tmp := MakePath([GetSystemPath(spTemp), 'lai-qwen3-' + Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  try
    CopyFile(dir + 'model.safetensors', tmp + 'model.safetensors', false);
    Check(not LoadRefused('"x"', '"x"'), 'the copy loads');
    Check(LoadRefused('"silu"', '"gelu"'), 'other activation');
    // a sliding window is read since D4 (SlidingWindowConfig); a negative
    // size is refused
    Check(LoadRefused('"sliding_window": null', '"sliding_window": -3') = false,
      'a sliding_window without use_sliding_window: ignored');
    cfg := StringReplaceAll(cfg, '"use_sliding_window": false', '"use_sliding_window": true');
    Check(LoadRefused('"sliding_window": null', '"sliding_window": -3'),
      'a negative sliding window');
    cfg := StringReplaceAll(cfg, '"use_sliding_window": true', '"use_sliding_window": false');
    Check(LoadRefused('"head_dim": 32', '"head_dim": 16'), 'head_dim not the weights''');
    Check(LoadRefused('"num_key_value_heads": 2', '"num_key_value_heads": 3'),
      'heads not a multiple of the KV heads');
    Check(LoadRefused('"rope_type": "default"', '"rope_type": "yarn"'), 'scaled RoPE');
    Check(LoadRefused('"qwen3"', '"qwen2"'), 'other model type');
    Check(LoadRefused('"tie_word_embeddings": true', '"tie_word_embeddings": false'),
      'untied without lm_head.weight');
    Check(LoadRefused('"head_dim": 32,', ''), 'no head_dim');
    // a NaN weight: F32 carries it along, Q8_0 has no form for it
    FileFromString(cfg, tmp + 'config.json');
    st := StringFromFile(dir + 'model.safetensors');
    hdr := _JsonFast(copy(st, 9, PInteger(st)^));
    pos := 8 + PInteger(st)^ + _Safe(hdr)^.O['model.layers.0.self_attn.q_proj.weight']^.
      A['data_offsets']^.Values[0];
    FastSetRawByteString(bad, pointer(st), length(st)); // a copy of its own
    PWord(@PByteArray(bad)^[pos])^ := $7fc0; // BF16 quiet NaN
    FileFromString(bad, tmp + 'model.safetensors');
    TQwen3Model.Create(tmp, fCpu).Free;
    refused := false;
    try
      TQwen3Model.Create(tmp, fCpu, idtQ8_0).Free;
    except
      on E: EInferError do
        refused := PosEx('NaN or infinite weight', StringToUtf8(E.Message)) > 0;
    end;
    Check(refused, 'Q8_0 refuses a NaN weight, by name');
    // finite but beyond the Q8_0 range: 9.96e6 / 127 exceeds the largest FP16
    PWord(@PByteArray(bad)^[pos])^ := $4b18;
    FileFromString(bad, tmp + 'model.safetensors');
    TQwen3Model.Create(tmp, fCpu).Free;
    refused := false;
    try
      TQwen3Model.Create(tmp, fCpu, idtQ8_0).Free;
    except
      on E: EInferError do
        refused := PosEx('beyond the Q8_0 range', StringToUtf8(E.Message)) > 0;
    end;
    Check(refused, 'Q8_0 refuses a weight beyond its range, by name');
  finally
    DirectoryDeleteAll(tmp);
    RemoveDir(tmp);
  end;
end;

/// aCount deterministic pseudo-random ids in 3..aVocab-1
function RandomIds(aCount, aVocab: PtrInt): TIntegerDynArray;
var
  i: PtrInt;
  x: cardinal;
begin
  SetLength(result, aCount);
  x := 20261015;
  for i := 0 to aCount - 1 do
  begin
    x := x * 1103515245 + 12345;
    result[i] := 3 + integer((x shr 8) mod cardinal(aVocab - 3));
  end;
end;

procedure TTestInferQwen3.TinyLongContext;
var
  dir: TFileName;
  model: TQwen3Model;
  full, step: TQwen3Context;
  ids: TIntegerDynArray;
  all, one: TSingleDynArray;
  n, i, vocab: PtrInt;
  same: boolean;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  model := TQwen3Model.Create(dir, fCpu);
  full := model.NewContext;
  step := model.NewContext;
  try
    n := 130; // the cache starts at 64 tokens: it grows to 128, then 256
    vocab := model.Config.VocabSize;
    ids := RandomIds(n, vocab);
    SetLength(all, n * vocab);
    SetLength(one, vocab);
    model.Forward(full, pointer(ids), n, pointer(all), true);
    same := true;
    for i := 0 to n - 1 do
    begin
      model.Forward(step, @ids[i], 1, pointer(one));
      if not CompareMem(pointer(one), @all[i * vocab], vocab * SizeOf(single)) then
        same := false;
    end;
    Check(same, 'token by token over 130 positions = one pass, bit for bit');
    // a fresh context: the chunks themselves must grow the cache (64 -> 128 -> 256)
    FreeAndNil(step);
    step := model.NewContext;
    model.Forward(step, pointer(ids), 50, nil);
    model.Forward(step, @ids[50], 50, nil);
    model.Forward(step, @ids[100], n - 100, pointer(one));
    Check(CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single)),
      'three prefill chunks across the growth');
  finally
    step.Free;
    full.Free;
    model.Free;
  end;
end;

procedure TTestInferQwen3.TinyLimitsAndContexts;
var
  dir: TFileName;
  a, b: TQwen3Model;
  ctx, fresh: TQwen3Context;
  ids, g1, g2, g3, joined: TIntegerDynArray;
  limit, i: PtrInt;
  one: integer;
  refused: boolean;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  a := TQwen3Model.Create(dir, fCpu);
  b := TQwen3Model.Create(dir, fCpu);
  ctx := a.NewContext;
  fresh := a.NewContext;
  try
    limit := a.Config.MaxPositions;
    // Greedy: no new tokens wanted is an error, before anything is cached
    refused := false;
    try
      a.Greedy(ctx, [5, 6], 0, []);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused and (ctx.Length = 0), 'aMaxNew = 0 refused, nothing cached');
    // returned tokens enter the cache: a conversation can go on
    g1 := a.Greedy(ctx, [5, 6, 7], 4, []);
    CheckEqual(ctx.Length, 3 + 4, 'prompt and every returned token cached');
    g2 := a.Greedy(ctx, [9], 4, []);
    joined := nil;
    SetLength(joined, 3 + length(g1) + 1);
    joined[0] := 5;
    joined[1] := 6;
    joined[2] := 7;
    for i := 0 to high(g1) do
      joined[3 + i] := g1[i];
    joined[high(joined)] := 9;
    g3 := a.Greedy(fresh, joined, 4, []);
    CheckEqual(IdsOf(g2), IdsOf(g3), 'continued = fed again from scratch');
    // a context of model A is no context of model B, even of the same shape
    refused := false;
    try
      one := 5;
      b.Forward(ctx, @one, 1, nil);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused, 'another model''s cache refused');
    ctx.Reset;
    one := 5;
    b.Forward(ctx, @one, 1, nil);
    CheckEqual(ctx.Length, 1, 'after Reset the other model takes it over');
    // the context limit: Greedy stops there, Forward refuses beyond it
    ctx.Reset;
    ids := RandomIds(limit - 12, b.Config.VocabSize);
    g1 := b.Greedy(ctx, ids, 100, []);
    CheckEqual(length(g1), 13, '12 positions left: 13 tokens, the last uncached');
    CheckEqual(ctx.Length, limit, 'context full');
    refused := false;
    try
      one := 5;
      b.Forward(ctx, @one, 1, nil);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused and (ctx.Length = limit), 'no token beyond MaxPositions');
    // a model freed and another created, maybe at the same address: the cache
    // of the first is still not the second one's
    fresh.Reset;
    one := 5;
    a.Forward(fresh, @one, 1, nil);
    FreeAndNil(a);
    a := TQwen3Model.Create(dir, fCpu);
    refused := false;
    try
      a.Forward(fresh, @one, 1, nil);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused, 'a freed model''s cache refused by a new instance');
  finally
    fresh.Free;
    ctx.Free;
    b.Free;
    a.Free;
  end;
end;

procedure TTestInferQwen3.TinyPromptReuse;
var
  dir: TFileName;
  a, b: TQwen3Model;
  ctx, fresh: TQwen3Context;
  first, second: TIntegerDynArray;
  want, got: TSingleDynArray;
  k, kept, vocab: PtrInt;
  refused: boolean;

  procedure Truncated(aLength: PtrInt);
  begin
    refused := false;
    try
      ctx.Truncate(aLength);
    except
      on EInferError do
        refused := true;
    end;
  end;

begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  a := nil;
  b := nil;
  ctx := nil;
  fresh := nil;
  try
    for k := 0 to 2 do
    begin
      // the reference backend, the fast one, the fast one on Q8_0: the cache
      // of position i only depends on tokens 0..i, so a reused prefix gives
      // the same bits
      FreeAndNil(a);
      case k of
        0:
          a := TQwen3Model.Create(dir, fCpu);
        1:
          a := TQwen3Model.Create(dir, fFast);
      else
        a := TQwen3Model.Create(dir, fFast, idtQ8_0);
      end;
      vocab := a.Config.VocabSize;
      SetLength(want, vocab);
      SetLength(got, vocab);
      FreeAndNil(ctx);
      FreeAndNil(fresh);
      ctx := a.NewContext;
      fresh := a.NewContext;
      // a first request, then a second one sharing its first 40 tokens
      first := RandomIds(60, vocab);
      second := RandomIds(90, vocab);
      for kept := 40 to 59 do
        second[kept] := (first[kept] + 1) mod vocab;
      a.Forward(ctx, pointer(first), length(first), pointer(got));
      CheckEqual(ctx.Common(pointer(second), length(second)), 40, 'common prefix');
      CheckEqual(ctx.Id[39], first[39], 'Id of a cached position');
      kept := a.Reuse(ctx, pointer(second), length(second));
      CheckEqual(kept, 40, 'the shared tokens stay');
      CheckEqual(ctx.Length, 40, 'the others are forgotten');
      a.Forward(ctx, @second[kept], length(second) - kept, pointer(got));
      a.Forward(fresh, pointer(second), length(second), pointer(want));
      Check(CompareMem(pointer(got), pointer(want), vocab * SizeOf(single)),
        'logits after a reused prefix bit-identical to a fresh pass');
      // the same prompt again: its last token is computed for the logits
      kept := a.Reuse(ctx, pointer(second), length(second));
      CheckEqual(kept, length(second) - 1, 'all but the last token stay');
      a.Forward(ctx, @second[kept], 1, pointer(got));
      Check(CompareMem(pointer(got), pointer(want), vocab * SizeOf(single)),
        'the same prompt, the same logits');
      // a prompt that is a prefix of the cached sequence
      kept := a.Reuse(ctx, pointer(second), 30);
      CheckEqual(kept, 29, 'a shorter prompt keeps all but its last token');
      fresh.Reset;
      a.Forward(fresh, pointer(second), 30, pointer(want));
      a.Forward(ctx, @second[kept], 1, pointer(got));
      Check(CompareMem(pointer(got), pointer(want), vocab * SizeOf(single)),
        'a shorter prompt, the logits of a fresh pass');
      // nothing shared: everything is computed again
      first[0] := (second[0] + 1) mod vocab;
      CheckEqual(a.Reuse(ctx, pointer(first), length(first)), 0, 'no common token');
      CheckEqual(ctx.Length, 0, 'empty');
    end;
    // Truncate and Id outside the cached tokens
    a.Forward(ctx, pointer(first), 10, nil);
    Truncated(11);
    Check(refused and (ctx.Length = 10), 'Truncate beyond Length refused');
    Truncated(-1);
    Check(refused, 'Truncate(-1) refused');
    Truncated(10);
    Check(not refused and (ctx.Length = 10), 'Truncate(Length) keeps all');
    refused := false;
    try
      Check(ctx.Id[10] = -1, 'no Id beyond Length');
    except
      on EInferError do
        refused := true;
    end;
    Check(refused, 'Id beyond Length refused');
    CheckEqual(a.Reuse(ctx, pointer(first), 0), 0, 'an empty prompt reuses nothing');
    // another model of the same shape never takes the cached keys
    a.Forward(ctx, pointer(first), 10, nil);
    b := TQwen3Model.Create(dir, fCpu);
    CheckEqual(b.Reuse(ctx, pointer(first), length(first)), 0,
      'another model reuses nothing');
    CheckEqual(ctx.Length, 0, 'and empties the context');
    b.Forward(ctx, pointer(first), 5, nil);
    CheckEqual(ctx.Length, 5, 'which it then takes over');
  finally
    fresh.Free;
    ctx.Free;
    b.Free;
    a.Free;
  end;
end;

procedure TTestInferQwen3.TinyUntiedHead;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  dir := MakePath([dir, 'untied'], true);
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fCpu);
    Check(not model.Config.TiedEmbeddings, 'untied');
    CheckStates(model, ref, 'case0', 2e-6, 2e-5, true);
  finally
    model.Free;
    ref.Free;
  end;
end;

function SingleBits(aBits: cardinal): single;
begin
  PCardinal(@result)^ := aBits;
end;

procedure TTestInferQwen3.ArgMaxTiesAndNan;
var
  v: array[0..3] of single;
begin
  v[0] := 1;
  v[1] := 3;
  v[2] := 3;
  v[3] := -1;
  CheckEqual(InferArgMax(@v, 4), 1, 'a tie goes to the lowest index');
  v[0] := SingleBits($7fc00000); // NaN
  v[1] := 1;
  v[2] := 1;
  CheckEqual(InferArgMax(@v, 3), 1, 'a NaN is not chosen');
  v[1] := SingleBits($7fc00000);
  v[2] := SingleBits($ffc00000);
  CheckEqual(InferArgMax(@v, 3), 0, 'all NaN: the first');
  v[1] := SingleBits($ff800000); // -Inf
  v[2] := SingleBits($7fc00000);
  CheckEqual(InferArgMax(@v, 3), 1, '-Inf beats a NaN');
  v[0] := -3;
  v[1] := -0.5;
  v[2] := -2;
  v[3] := SingleBits($ff800000);
  CheckEqual(InferArgMax(@v, 4), 1, 'negative values');
  v[0] := SingleBits($80000000); // -0
  v[1] := 0;
  v[2] := SingleBits($80000001); // the negative value closest to 0
  CheckEqual(InferArgMax(@v, 3), 0, '-0 = +0: the first');
  v[0] := SingleBits($80000001);
  v[1] := SingleBits($00000001); // the positive value closest to 0
  v[2] := SingleBits($80000000);
  CheckEqual(InferArgMax(@v, 3), 1, 'denormals');
  CheckArgMaxRandom;
end;

procedure TTestInferQwen3.CheckArgMaxRandom;
var
  v: array[0..999] of single;
  b, seed: cardinal;
  i, n, round, expected: PtrInt;
  ok: boolean;
begin
  // random bit patterns (every exponent, both signs, infinities, denormals,
  // NaN now and then) against a plain float comparison over the non-NaN ones
  seed := 12345;
  ok := true;
  for round := 1 to 200 do
  begin
    n := round * 5; // 5 .. 1000
    for i := 0 to n - 1 do
    begin
      seed := seed * 1103515245 + 12345;
      b := seed;
      seed := seed * 1103515245 + 12345;
      b := (b shl 16) xor (seed shr 8);
      if round mod 3 = 0 then
        b := b and $c07fffff; // exponents near 0: many close values
      // the bits as they are: a signaling NaN returned as a single would
      // trap in the x87 of Delphi Win32
      PCardinal(@v[i])^ := b;
    end;
    expected := -1;
    for i := 0 to n - 1 do
      if PCardinal(@v[i])^ and $7fffffff <= $7f800000 then // no NaN
        if (expected < 0) or (v[i] > v[expected]) then
          expected := i;
    if expected < 0 then
      expected := 0;
    if InferArgMax(@v, n) <> expected then
      ok := false;
  end;
  Check(ok, 'random values as a float comparison');
end;

function TTestInferQwen3.RealModel: TFileName;
begin
  result := '';
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3-0.6B: 2.4 GB of FP32 weights need a 64-bit process');
  exit;
  {$endif CPU32}
  result := FindQwen3Folder('model.safetensors');
  if result <> '' then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b/model.safetensors');
  AddConsole('SKIP Qwen3-0.6B: no vendor/models/qwen3-0.6b (tools/infer-ref, ' +
    'make_reference_qwen3.py)');
end;

procedure TTestInferQwen3.Qwen3ForwardMatchesReference;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  prompts, p: PDocVariantData;
  i: PtrInt;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  refdir := FindInferRefDir('qwen3');
  if CheckFailed(refdir <> '', 'qwen3 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fFast);
    CheckEqual(model.Config.Layers, 28, 'Qwen3-0.6B depth');
    CheckEqual(model.Config.HeadDim, 128, 'head_dim 128, not 1024 / 16');
    prompts := ref.Manifest^.A['prompts'];
    for i := 0 to prompts^.Count - 1 do
    begin
      p := _Safe(prompts^.Values[i]);
      CheckStates(model, ref, p^.U['id'], 2e-3, 2e-3, false);
    end;
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.TinyQuantized(aType: TInferDType; aFloatTol, aRefTol,
  aRefCos: double);
var
  dir: TFileName;
  ref: TInferRefSet;
  exact: TInferCpuFastBackend;
  a, b, c: TQwen3Model;
  ca, cb, cc: TQwen3Context;
  la, lb, lc: TSingleDynArray;
  ids: PIntegerArray;
  n, m, k, v: PtrInt;
  cmp: TInferCompare;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  exact := TInferCpuFastBackend.Create;
  exact.Exact := true;
  a := nil;
  b := nil;
  c := nil;
  try
    a := TQwen3Model.Create(dir, fCpu, aType);
    b := TQwen3Model.Create(dir, exact, aType);
    c := TQwen3Model.Create(dir, fFast, aType);
    Check(a.Weights = aType, 'quantized matrices');
    v := a.Config.VocabSize;
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(la, n * v);
      SetLength(lb, n * v);
      SetLength(lc, n * v);
      ca := a.NewContext;
      cb := b.NewContext;
      cc := c.NewContext;
      try
        a.Forward(ca, ids, n, pointer(la), true);
        b.Forward(cb, ids, n, pointer(lb), true);
        c.Forward(cc, ids, n, pointer(lc), true);
      finally
        ca.Free;
        cb.Free;
        cc.Free;
      end;
      Check(CompareMem(pointer(la), pointer(lb), n * v * SizeOf(single)),
        'Exact on all cores = the scalar reference, bit for bit');
      cmp := InferCompare(pointer(lc), pointer(la), n * v, aFloatTol, aFloatTol / 2);
      CheckUtf8(InferCompareOk(cmp), 'case % % float path: %',
        [k, INFER_DTYPE_NAME[aType], InferCompareText(cmp)]);
      cmp := InferCompare(pointer(la), ref.F32(FormatUtf8('case%.logits', [k]), m),
        n * v, aRefTol, aRefTol);
      CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > aRefCos), 'case % % vs FP32: %',
        [k, INFER_DTYPE_NAME[aType], InferCompareText(cmp)]);
    end;
    // the untied head: its own matrix quantized, not the embedding
    FreeAndNil(c);
    FreeAndNil(b);
    FreeAndNil(a);
    // against the stored logits of the untied reference: an F32 model of the
    // same code would share a mistake in the head
    dir := MakePath([dir, 'untied'], true);
    ref.Free;
    ref := TInferRefSet.CreateFromFolder(dir);
    a := TQwen3Model.Create(dir, fCpu, aType);
    Check(not a.Config.TiedEmbeddings, 'untied');
    ids := ref.I32('case0.ids', n);
    v := a.Config.VocabSize;
    SetLength(la, n * v);
    ca := a.NewContext;
    try
      a.Forward(ca, ids, n, pointer(la), true);
    finally
      ca.Free;
    end;
    cmp := InferCompare(pointer(la), ref.F32('case0.logits', m), n * v, aRefTol,
      aRefTol);
    CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > aRefCos),
      'untied % vs the FP32 reference: %', [INFER_DTYPE_NAME[aType],
      InferCompareText(cmp)]);
  finally
    c.Free;
    b.Free;
    a.Free;
    exact.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.Qwen3Quantized(aType: TInferDType; aCos, aMaxAbs,
  aMargin: double; const aGguf: TFileName; const aRefName: RawUtf8;
  const aFolder: TFileName);
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  ctx: TQwen3Context;
  prompts, p: PDocVariantData;
  ids, want: PIntegerArray;
  margins: PSingleArray;
  i, n, m, diff: PtrInt;
  prompt, mine: TIntegerDynArray;
  logits: TSingleDynArray;
  cmp: TInferCompare;
  margin: double;
  timer: TPrecisionTimer;
begin
  dir := aFolder;
  if dir = '' then
    dir := RealModel;
  if dir = '' then
    exit;
  refdir := FindInferRefDir(aRefName);
  if CheckFailed(refdir <> '', Utf8ToString(aRefName + ' reference found')) then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  try
    timer.Start;
    if aGguf <> '' then
      model := NewDecoderModelFromGguf(aGguf, fFast)
    else
      model := NewDecoderModel(dir, fFast, aType);
    AddConsole(FormatString('% model loaded in %', [INFER_DTYPE_NAME[aType],
      timer.Stop]));
    prompts := ref.Manifest^.A['prompts'];
    for i := 0 to prompts^.Count - 1 do
    begin
      p := _Safe(prompts^.Values[i]);
      ids := ref.I32(p^.U['id'] + '.ids', n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      SetLength(logits, model.Config.VocabSize);
      ctx := model.NewContext;
      try
        model.Forward(ctx, pointer(prompt), n, pointer(logits));
      finally
        ctx.Free;
      end;
      cmp := InferCompare(pointer(logits), ref.F32(p^.U['id'] + '.logits_last', m),
        model.Config.VocabSize, 1, 0);
      want := ref.I32(p^.U['id'] + '.greedy', m);
      margins := ref.F32(p^.U['id'] + '.greedy_margin', m);
      ctx := model.NewContext;
      try
        mine := model.Greedy(ctx, prompt, m, [151645, 151643]);
      finally
        ctx.Free;
      end;
      diff := 0;
      while (diff < m) and (diff < length(mine)) and (mine[diff] = want^[diff]) do
        inc(diff);
      margin := 0;
      if diff < m then
        margin := margins^[diff];
      if (diff < m) and (diff < length(mine)) then
        AddConsole(FormatString('% %: logits cos % maxabs %, greedy equal for ' +
          '% of % tokens, reference margin there % (took %, reference %)',
          [p^.U['id'], INFER_DTYPE_NAME[aType], cmp.Cosine, cmp.MaxAbs, diff, m,
           margin, mine[diff], want^[diff]]))
      else if diff < m then
        AddConsole(FormatString('% %: logits cos % maxabs %, stopped after % ' +
          'of % tokens, reference margin there %', [p^.U['id'],
          INFER_DTYPE_NAME[aType], cmp.Cosine, cmp.MaxAbs, diff, m, margin]))
      else
        AddConsole(FormatString('% %: logits cos % maxabs %, greedy equal for ' +
          'all % tokens', [p^.U['id'], INFER_DTYPE_NAME[aType], cmp.Cosine,
          cmp.MaxAbs, m]));
      CheckUtf8(cmp.Cosine >= aCos, '% % logits cosine %',
        [p^.U['id'], INFER_DTYPE_NAME[aType], cmp.Cosine]);
      CheckUtf8(cmp.MaxAbs <= aMaxAbs, '% % logits max error %',
        [p^.U['id'], INFER_DTYPE_NAME[aType], cmp.MaxAbs]);
      CheckUtf8((diff >= m) or (margin < aMargin),
        '% % greedy leaves the reference at step % where it was % sure',
        [p^.U['id'], INFER_DTYPE_NAME[aType], diff, margin]);
    end;
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.TinyQ8;
begin
  TinyQuantized(idtQ8_0, 2e-2, 0.05, 0.999);
end;

procedure TTestInferQwen3.TinyQ4;
begin
  // 4 bits of random weights: measured 2026-10-15 cosine 0.959 to 0.995 and up
  // to 0.77 apart from FP32 - the gate is the real model, here the bits of
  // Exact and the float path count
  TinyQuantized(idtQ4_0, 2e-2, 1, 0.9);
end;

procedure TTestInferQwen3.Qwen3Q8AgainstReference;
begin
  // the gate of Q8_0, measured 2026-10-15 on the 193 tokens of chat_tool,
  // the most sensitive prompt: cosine 0.9935 and max error 1.72 in exact
  // (double) arithmetic - the same bits on aarch64 and x86_64 - and 0.9965 /
  // 1.36 with NEON, 0.9868 / 2.11 with AVX2; float against exact Q8_0
  // alone: cosine 0.998 to 0.999, up to 0.95 apart on both platforms (likely
  // because a changed rounding flips single int8 activations of the next
  // layer - not measured inside). Since the weights round ties away from
  // zero like a GGUF writer (3e-2): 0.9972 / 1.31 with NEON, 0.9961 / 1.40
  // with AVX2.
  // Greedy leaves the FP32 tokens only where the reference itself was less
  // than 0.28 sure. A broken quantizer fails all three by far (shifted rows:
  // cosine 0.23, max error 24)
  Qwen3Quantized(idtQ8_0, 0.98, 3, 0.5);
end;

procedure TTestInferQwen3.Qwen3Q4AgainstReference;
begin
  // Q4_0 layers, Q8_0 head, measured 2026-10-15: cosine 0.89 to 0.96 and max
  // error up to 9.8 on six prompts - 4 bits are too coarse for a 0.6B model:
  // greedy leaves the reference even where it was 1.94 sure (raw_fox, step 1),
  // and chat_tool answers instead of calling the tool. chat_tool is chaotic
  // under Q4_0: rounding the product and the sum of the quantizer once instead
  // of twice changes a few bytes in thousands and moves its cosine from 0.76 to
  // 0.66 (max error 11). No greedy condition then; bounds a broken quantizer
  // or kernel misses by far
  Qwen3Quantized(idtQ4_0, 0.5, 15, 1e9);
end;


/// the matrices of layer aLayer (or the head for -1): weight, rows, columns
procedure LayerMatrices(aModel: TQwen3Model; aLayer: PtrInt;
  var aW: array of TInferWeight; var aRows, aCols: array of PtrInt);
var
  h, f, qw, kvw: PtrInt;
  l: ^TQwen3Layer;
begin
  h := aModel.Config.Hidden;
  f := aModel.Config.Intermediate;
  qw := aModel.Config.Heads * aModel.Config.HeadDim;
  kvw := aModel.Config.KvHeads * aModel.Config.HeadDim;
  if aLayer < 0 then
  begin
    aW[0] := aModel.Head;
    aRows[0] := aModel.Config.VocabSize;
    aCols[0] := h;
    exit;
  end;
  l := @aModel.Layers[aLayer];
  aW[0] := l^.Q;
  aRows[0] := qw;
  aCols[0] := h;
  aW[1] := l^.K;
  aRows[1] := kvw;
  aCols[1] := h;
  aW[2] := l^.V;
  aRows[2] := kvw;
  aCols[2] := h;
  aW[3] := l^.O;
  aRows[3] := h;
  aCols[3] := qw;
  aW[4] := l^.Gate;
  aRows[4] := f;
  aCols[4] := h;
  aW[5] := l^.Up;
  aRows[5] := f;
  aCols[5] := h;
  aW[6] := l^.Down;
  aRows[6] := h;
  aCols[6] := f;
end;

const
  MATRIX_NAMES: array[0..6] of RawUtf8 = (
    'q', 'k', 'v', 'o', 'gate', 'up', 'down');

/// the same type and bytes in every matrix of two models of one shape
function SameMatrices(a, b: TQwen3Model; out aDetail: RawUtf8): boolean;
var
  i, k, n, bytes: PtrInt;
  wa, wb: array[0..6] of TInferWeight;
  rows, cols: array[0..6] of PtrInt;
begin
  aDetail := '';
  result := true;
  for i := -1 to a.Config.Layers - 1 do
  begin
    LayerMatrices(a, i, wa, rows, cols);
    LayerMatrices(b, i, wb, rows, cols);
    n := 7;
    if i < 0 then
      n := 1;
    for k := 0 to n - 1 do
    begin
      bytes := rows[k] * (cols[k] div INFER_DTYPE_BLOCK[wa[k].DType]) *
        INFER_DTYPE_SIZE[wa[k].DType];
      if (wa[k].DType <> wb[k].DType) or
         not CompareMem(wa[k].Data, wb[k].Data, bytes) then
      begin
        if i < 0 then
          aDetail := 'head'
        else
          aDetail := FormatUtf8('layer % %', [i, MATRIX_NAMES[k]]);
        aDetail := FormatUtf8('% (% / %)', [aDetail, INFER_DTYPE_NAME[wa[k].DType],
          INFER_DTYPE_NAME[wb[k].DType]]);
        result := false;
        exit;
      end;
    end;
  end;
end;

procedure TTestInferQwen3.TinyGguf;
var
  dir, gdir: TFileName;
  ref: TInferRefSet;
  g, s, e: TQwen3Model;
  exact: TInferCpuFastBackend;
  lg, le: TSingleDynArray;
  ids: PIntegerArray;
  n, m, k, v: PtrInt;
  cmp: TInferCompare;
  detail: RawUtf8;
  same: boolean;

  function Logits(aModel: TQwen3Model; aCase: PtrInt): TSingleDynArray;
  var
    c: TQwen3Context;
  begin
    ids := ref.I32(FormatUtf8('case%.ids', [aCase]), n);
    result := nil;
    SetLength(result, n * aModel.Config.VocabSize);
    c := aModel.NewContext;
    try
      aModel.Forward(c, ids, n, pointer(result), true);
    finally
      c.Free;
    end;
  end;

  procedure Refused(const aFile, aWhy: RawUtf8);
  var
    raised: boolean;
  begin
    raised := false;
    try
      TQwen3Model.CreateFromGguf(gdir + Utf8ToString(aFile), fCpu).Free;
    except
      on E: EInferError do
        raised := PosEx(aWhy, StringToUtf8(E.Message)) > 0;
    end;
    CheckUtf8(raised, '% refused (%)', [aFile, aWhy]);
  end;

begin
  dir := FindInferRefDir('tiny-qwen3');
  gdir := FindInferRefDir('tiny-qwen3-gguf');
  if CheckFailed((dir <> '') and (gdir <> ''), 'tiny-qwen3 and its GGUF found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  exact := TInferCpuFastBackend.Create;
  exact.Exact := true;
  g := nil;
  s := nil;
  e := nil;
  try
    // Q8_0 as llama.cpp writes it: the settings of config.json, the matrices
    // of our own quantizer byte for byte (both are ggml's reference)
    g := TQwen3Model.CreateFromGguf(gdir + 'q8_0.gguf', fCpu);
    s := TQwen3Model.Create(dir, fCpu, idtQ8_0);
    Check(g.Weights = idtQ8_0, 'Q8_0 file');
    Check(g.Config.TiedEmbeddings, 'no output.weight: tied');
    Check((g.Config.VocabSize = s.Config.VocabSize) and
      (g.Config.Hidden = s.Config.Hidden) and (g.Config.Layers = s.Config.Layers) and
      (g.Config.Heads = s.Config.Heads) and (g.Config.KvHeads = s.Config.KvHeads) and
      (g.Config.HeadDim = s.Config.HeadDim) and
      (g.Config.Intermediate = s.Config.Intermediate) and
      (g.Config.MaxPositions = s.Config.MaxPositions) and
      (g.Config.RopeTheta = s.Config.RopeTheta), 'the settings of config.json');
    same := SameMatrices(g, s, detail); // before the message reads detail
    CheckUtf8(same, 'Q8_0 matrices like ours: %', [detail]);
    // the embedding is Q8_0 here too (the safetensors model reads it in BF16):
    // up to 0.07 from FP32 measured, so 0.1; Exact on all cores = the reference
    e := TQwen3Model.CreateFromGguf(gdir + 'q8_0.gguf', exact);
    v := g.Config.VocabSize;
    for k := 0 to 2 do
    begin
      lg := Logits(g, k);
      le := Logits(e, k);
      Check(CompareMem(pointer(lg), pointer(le), n * v * SizeOf(single)),
        'GGUF: Exact = the reference backend');
      cmp := InferCompare(pointer(lg), ref.F32(FormatUtf8('case%.logits', [k]), m),
        n * v, 0.1, 0.1);
      CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.999),
        'case % GGUF Q8_0 vs FP32: %', [k, InferCompareText(cmp)]);
    end;
    FreeAndNil(e);
    FreeAndNil(g);
    FreeAndNil(s);
    // Q4_0 layers with a Q8_0 embedding and head: our Q4_0 model byte for byte
    g := TQwen3Model.CreateFromGguf(gdir + 'q4_0.gguf', fCpu);
    s := TQwen3Model.Create(dir, fCpu, idtQ4_0);
    Check(g.Weights = idtQ4_0, 'Q4_0 file');
    same := SameMatrices(g, s, detail);
    CheckUtf8(same, 'Q4_0 matrices like ours: %', [detail]);
    FreeAndNil(g);
    FreeAndNil(s);
    // F16 matrices widened, an untied head from output.weight
    ref.Free;
    ref := nil;
    ref := TInferRefSet.CreateFromFolder(MakePath([dir, 'untied'], true));
    g := TQwen3Model.CreateFromGguf(gdir + 'untied-f16.gguf', fCpu);
    Check(g.Weights = idtF32, 'F16 widened to F32');
    Check(not g.Config.TiedEmbeddings, 'output.weight: untied');
    lg := Logits(g, 0);
    cmp := InferCompare(pointer(lg), ref.F32('case0.logits', m), n * g.Config.VocabSize,
      0.01, 0.01);
    CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.9999),
      'untied F16 GGUF vs FP32: %', [InferCompareText(cmp)]);
    FreeAndNil(g);
    // damaged files
    Refused('bad-arch.gguf', 'architecture');
    Refused('bad-rope.gguf', 'rope scaling');
    Refused('bad-extra.gguf', 'understood');
    Refused('bad-missing.gguf', 'no tensor');
    Refused('bad-scale.gguf', 'no finite number');
    Refused('bad-shape.gguf', 'shaped');
    Refused('bad-ropedim.gguf', 'RoPE over');
    Refused('bad-embscale.gguf', 'no finite number');
  finally
    e.Free;
    s.Free;
    g.Free;
    exact.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.ThreadItem(aItem: integer);
begin
  fItemThread[aItem] := GetCurrentThreadId;
  SleepHiRes(1);
end;

procedure TTestInferQwen3.LoadOnAllCores;
var
  dir, gdir, tmp: TFileName;
  one, all: TQwen3Model;
  w: TInferDType;
  detail, why: RawUtf8;
  st, bad: RawByteString;
  hdr: variant;
  pos, i: PtrInt;
  others, missed: boolean;
begin
  dir := FindInferRefDir('tiny-qwen3');
  gdir := FindInferRefDir('tiny-qwen3-gguf');
  if CheckFailed((dir <> '') and (gdir <> ''), 'tiny-qwen3 and its GGUF found') then
    exit;
  if fFast.Threads < 2 then
  begin
    AddConsole('SKIP load on all cores: one thread only');
    exit;
  end;
  // RunItems of the fast backend runs on its pool: 64 items of 1 ms cannot
  // all stay on the calling thread; an entry 0 was not run
  SetLength(fItemThread, 64);
  FillCharFast(fItemThread[0], length(fItemThread) * SizeOf(fItemThread[0]), 0);
  fFast.RunItems(ThreadItem, length(fItemThread));
  others := false;
  missed := false;
  for i := 0 to high(fItemThread) do
    if fItemThread[i] = TThreadID(0) then
      missed := true
    else if fItemThread[i] <> GetCurrentThreadId then
      others := true;
  Check(not missed and others, 'every item run, some on the pool');
  // the base class: in order, on this thread
  FillCharFast(fItemThread[0], length(fItemThread) * SizeOf(fItemThread[0]), 0);
  fCpu.RunItems(ThreadItem, length(fItemThread));
  others := false;
  for i := 0 to high(fItemThread) do
    if fItemThread[i] <> GetCurrentThreadId then
      others := true;
  Check(not others, 'RunItems of the reference backend on this thread');
  one := nil;
  all := nil;
  tmp := '';
  try
    // BF16 widened (embedding rows of 4096 values per item and a rest),
    // Q8_0 and Q4_0 quantized row by row on the pool
    for w := low(w) to high(w) do
      if w in [idtF32, idtQ8_0, idtQ4_0] then
      begin
        one := TQwen3Model.Create(dir, fCpu, w);
        all := TQwen3Model.Create(dir, fFast, w);
        CheckUtf8(SameMatrices(one, all, detail), '% on all cores: %',
          [INFER_DTYPE_NAME[w], detail]);
        FreeAndNil(all);
        FreeAndNil(one);
      end;
    // F16 matrices of a GGUF file widened on the pool
    one := TQwen3Model.CreateFromGguf(gdir + 'untied-f16.gguf', fCpu);
    all := TQwen3Model.CreateFromGguf(gdir + 'untied-f16.gguf', fFast);
    CheckUtf8(SameMatrices(one, all, detail), 'F16 GGUF on all cores: %', [detail]);
    FreeAndNil(all);
    FreeAndNil(one);
    // a NaN in the last row of a matrix: refused by name from the pool too
    tmp := MakePath([GetSystemPath(spTemp), 'lai-cores-' +
      Int64ToUtf8(GetTickCount64)], true);
    ForceDirectories(tmp);
    CopyFile(dir + 'config.json', tmp + 'config.json', false);
    st := StringFromFile(dir + 'model.safetensors');
    hdr := _JsonFast(copy(st, 9, PInteger(st)^));
    pos := 8 + PInteger(st)^ + _Safe(hdr)^.O['model.layers.1.mlp.up_proj.weight']^.
      A['data_offsets']^.Values[1] - 2;
    FastSetRawByteString(bad, pointer(st), length(st));
    PWord(@PByteArray(bad)^[pos])^ := $7fc0; // BF16 quiet NaN
    FileFromString(bad, tmp + 'model.safetensors');
    why := '';
    try
      TQwen3Model.Create(tmp, fFast, idtQ8_0).Free;
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    CheckUtf8(PosEx('model.layers.1.mlp.up_proj.weight has a NaN or infinite weight ' +
      'at row 159 column 63', why) > 0, 'a NaN refused on the pool: %', [why]);
  finally
    all.Free;
    one.Free;
    if tmp <> '' then
    begin
      DirectoryDeleteAll(tmp);
      RemoveDir(tmp);
    end;
  end;
end;

const
  HF_MATRICES: array[0..6] of RawUtf8 = (
    'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
    'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight',
    'mlp.down_proj.weight');

/// the lowest cosine between a quantized matrix of aModel and the same weights
// in aFile (BF16), and the matrix it belongs to
function SourceCosine(aModel: TQwen3Model; aFile: TSafeTensorsFile;
  out aWorst: RawUtf8): double;
var
  i, k, n, r, c: PtrInt;
  w: array[0..6] of TInferWeight;
  rows, cols: array[0..6] of PtrInt;
  t: PInferTensor;
  mine: TSingleDynArray;
  name: RawUtf8;
  a, b, dot, na, nb, cs: double;
begin
  aWorst := '';
  result := 2;
  for i := -1 to aModel.Config.Layers - 1 do
  begin
    LayerMatrices(aModel, i, w, rows, cols);
    n := 7;
    if i < 0 then
      n := 1;
    for k := 0 to n - 1 do
    begin
      if i < 0 then
        name := 'model.embed_tokens.weight'
      else
        name := FormatUtf8('model.layers.%.%', [i, HF_MATRICES[k]]);
      if not (w[k].DType in INFER_BLOCK_DTYPES) then
      begin
        aWorst := FormatUtf8('% is %', [name, INFER_DTYPE_NAME[w[k].DType]]);
        result := -1;
        exit;
      end;
      t := aFile.Tensor(name);
      SetLength(mine, cols[k]);
      dot := 0;
      na := 0;
      nb := 0;
      for r := 0 to rows[k] - 1 do
      begin
        InferRowToF32(w[k].Data, w[k].DType, r, cols[k], pointer(mine));
        for c := 0 to cols[k] - 1 do
        begin
          a := mine[c];
          // the BF16 weight, widened exactly into the same buffer
          PCardinal(@mine[c])^ := cardinal(PWordArray(t^.Data)^[r * cols[k] + c]) shl 16;
          b := mine[c];
          dot := dot + a * b;
          na := na + a * a;
          nb := nb + b * b;
        end;
      end;
      cs := dot / Sqrt(na * nb);
      if cs < result then
      begin
        result := cs;
        aWorst := name;
      end;
    end;
  end;
end;

/// the lowest cosine between the matrices of two models of one architecture:
// every 61st embedding row (read per token), every layer matrix and the head
function ModelCosine(a, b: TQwen3Model; out aWorst: RawUtf8): double;
var
  i, k, n, r, h: PtrInt;
  wa, wb: array[0..6] of TInferWeight;
  rows, cols: array[0..6] of PtrInt;
  ra, rb: TSingleDynArray;
  id: integer;
  dot, na, nb, cs: double;

  procedure Add(aCount: PtrInt);
  var
    j: PtrInt;
  begin
    for j := 0 to aCount - 1 do
    begin
      dot := dot + ra[j] * rb[j];
      na := na + ra[j] * ra[j];
      nb := nb + rb[j] * rb[j];
    end;
  end;

  procedure Take(const aName: RawUtf8);
  begin
    cs := dot / Sqrt(na * nb);
    if cs < result then
    begin
      result := cs;
      aWorst := aName;
    end;
  end;

begin
  result := 2;
  aWorst := '';
  h := a.Config.Hidden;
  SetLength(ra, h);
  SetLength(rb, h);
  dot := 0;
  na := 0;
  nb := 0;
  id := 0;
  while id < a.Config.VocabSize do
  begin
    a.Embed(@id, 1, pointer(ra));
    b.Embed(@id, 1, pointer(rb));
    Add(h);
    inc(id, 61);
  end;
  Take('embedding');
  for i := -1 to a.Config.Layers - 1 do
  begin
    LayerMatrices(a, i, wa, rows, cols);
    LayerMatrices(b, i, wb, rows, cols);
    n := 7;
    if i < 0 then
      n := 1;
    for k := 0 to n - 1 do
    begin
      SetLength(ra, cols[k]);
      SetLength(rb, cols[k]);
      dot := 0;
      na := 0;
      nb := 0;
      for r := 0 to rows[k] - 1 do
      begin
        InferRowToF32(wa[k].Data, wa[k].DType, r, cols[k], pointer(ra));
        InferRowToF32(wb[k].Data, wb[k].DType, r, cols[k], pointer(rb));
        Add(cols[k]);
      end;
      if i < 0 then
        Take('head')
      else
        Take(FormatUtf8('layer % % (% against %)', [i, HF_MATRICES[k],
          INFER_DTYPE_NAME[wa[k].DType], INFER_DTYPE_NAME[wb[k].DType]]));
    end;
  end;
end;

/// vendor/models/qwen3-0.6b-q8_0.gguf next to the model folder, '' if absent
function RealGguf(const aModelDir: TFileName): TFileName;
begin
  result := ExpandFileName(IncludeTrailingPathDelimiter(aModelDir) + '..' +
    PathDelim + 'qwen3-0.6b-q8_0.gguf');
  if not FileExists(result) then
    result := '';
end;

procedure TTestInferQwen3.Qwen3Gguf;
var
  dir, gguf: TFileName;
  g: TQwen3Model;
  st: TSafeTensorsFile;
  detail: RawUtf8;
  cs: double;
  timer: TPrecisionTimer;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  gguf := RealGguf(dir);
  if gguf = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b-q8_0.gguf');
    AddConsole('SKIP Qwen3 GGUF: no vendor/models/qwen3-0.6b-q8_0.gguf ' +
      '(Qwen/Qwen3-0.6B-GGUF, see docs/native-inference.md)');
    exit;
  end;
  g := nil;
  st := nil;
  try
    timer.Start;
    g := TQwen3Model.CreateFromGguf(gguf, fFast);
    AddConsole(FormatString('Qwen3-0.6B GGUF loaded in %', [timer.Stop]));
    Check(g.Weights = idtQ8_0, 'Q8_0 file');
    Check(g.Config.TiedEmbeddings and (g.Config.Layers = 28) and
      (g.Config.HeadDim = 128) and (g.Config.MaxPositions = 40960), 'settings');
    // Qwen's file holds other weights than the BF16 release of the reference
    // revision (token_embd alone is the bytes of ggml's quantizer on it; in
    // the layers, scales differ in most blocks and values by a few percent):
    // every matrix close to its BF16 weights - one mapped to the wrong tensor
    // has a cosine near 0
    st := TSafeTensorsFile.Create(IncludeTrailingPathDelimiter(dir) +
      'model.safetensors');
    cs := SourceCosine(g, st, detail);
    AddConsole(FormatString('GGUF matrices against the BF16 weights: lowest ' +
      'cosine % (%)', [cs, detail]));
    // measured 2026-10-15: 0.9998 at worst (layer 2 down_proj)
    CheckUtf8(cs >= 0.999, 'GGUF matrices like the BF16 weights: % %', [cs, detail]);
  finally
    st.Free;
    g.Free;
  end;
  // the gate of Q8_0, but quantized from other weights: greedy leaves
  // raw_fox at a step the reference was 0.71 sure of (measured 2026-10-15),
  // logits cosine 0.997 and max error 1.24 at worst
  Qwen3Quantized(idtQ8_0, 0.98, 3, 1, gguf);
end;

procedure TTestInferQwen3.Qwen3GgufQ4KM;
var
  dir, gguf: TFileName;
  g: TQwen3Model;
  st: TSafeTensorsFile;
  detail: RawUtf8;
  cs: double;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  gguf := ExpandFileName(IncludeTrailingPathDelimiter(dir) + '..' + PathDelim +
    'qwen3-0.6b-q4_k_m.gguf');
  if not FileExists(gguf) then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b-q4_k_m.gguf');
    AddConsole('SKIP Qwen3 Q4_K_M: no vendor/models/qwen3-0.6b-q4_k_m.gguf ' +
      '(unsloth/Qwen3-0.6B-GGUF, see docs/native-inference.md)');
    exit;
  end;
  // unsloth's file mixes Q4_K and Q6_K (token_embd, half of attn_v and
  // ffn_down) and was quantized with an importance matrix: every matrix still
  // close to the BF16 weights - a wrong tensor or a broken K-quant unpacking
  // would not be
  g := TQwen3Model.CreateFromGguf(gguf, fFast);
  st := nil;
  try
    Check(g.Weights = idtQ4_K, 'Q4_K layers');
    Check(g.Head.DType = idtQ6_K, 'a Q6_K head (tied to token_embd)');
    st := TSafeTensorsFile.Create(IncludeTrailingPathDelimiter(dir) +
      'model.safetensors');
    cs := SourceCosine(g, st, detail);
    AddConsole(FormatString('Q4_K_M matrices against the BF16 weights: lowest ' +
      'cosine % (%)', [cs, detail]));
    CheckUtf8(cs >= 0.99, 'Q4_K_M matrices like the BF16 weights: % %', [cs, detail]);
  finally
    st.Free;
    g.Free;
  end;
  // measured 2026-10-15 (NEON, exact block sums): logits cosine 0.867
  // (chat_tool) to 0.990, max error up to 9.0; greedy leaves the reference
  // where it was 1.37 sure (chat_en answers "Paris." with a full stop) - far
  // better than Q4_0, still no greedy condition for a 0.6B model
  Qwen3Quantized(idtQ4_K, 0.8, 12, 1e9, gguf);
end;

procedure TTestInferQwen3.Qwen3GreedyMatchesReference;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  ctx: TQwen3Context;
  prompts, p: PDocVariantData;
  ids: PIntegerArray;
  i, n, diff: PtrInt;
  prompt, mine: TIntegerDynArray;
  margin: double;
  detail: RawUtf8;
  timer: TPrecisionTimer;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  refdir := FindInferRefDir('qwen3');
  if CheckFailed(refdir <> '', 'qwen3 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fFast);
    prompts := ref.Manifest^.A['prompts'];
    for i := 0 to prompts^.Count - 1 do
    begin
      p := _Safe(prompts^.Values[i]);
      ids := ref.I32(p^.U['id'] + '.ids', n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        timer.Start;
        mine := model.Greedy(ctx, prompt, 48, [151645, 151643]);
        timer.Stop;
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, p^.U['id'], margin, detail);
      CheckUtf8(diff < 0, '% differs at step % (reference margin %, %)',
        [p^.U['id'], diff, margin, detail]);
      AddConsole(FormatString('% greedy: % + % tokens in %, identical',
        [p^.U['id'], n, length(mine), timer.Time]));
    end;
  finally
    model.Free;
    ref.Free;
  end;
end;


procedure TTestInferQwen3.DecoderFamilies;
var
  dir, gdir, tmp: TFileName;
  m: TDecoderModel;
  i: PtrInt;
  attention: boolean;

  function Refused(const aFolder, aGguf: TFileName; const aWhy: RawUtf8): boolean;
  begin
    result := false;
    try
      if aGguf <> '' then
        NewDecoderModelFromGguf(aGguf, fCpu).Free
      else
        NewDecoderModel(aFolder, fCpu).Free;
    except
      on E: EInferError do
        result := PosEx(aWhy, StringToUtf8(E.Message)) > 0;
    end;
  end;

begin
  Check(DecoderModelClass('qwen3') = TQwen3Model, 'model_type qwen3');
  Check(DecoderModelClassOfArchitecture('qwen3') = TQwen3Model, 'architecture qwen3');
  Check(DecoderModelClass('qwen3_moe') = TQwen3MoeModel, 'model_type qwen3_moe');
  Check(DecoderModelClassOfArchitecture('qwen3moe') = TQwen3MoeModel,
    'architecture qwen3moe');
  Check(DecoderModelClass('qwen2') = TQwen2Model, 'model_type qwen2');
  Check(DecoderModelClass('llama') = nil, 'model_type llama: none');
  Check(DecoderModelClassOfArchitecture('llama') = nil, 'architecture llama: none');
  // registered once: a second registration changes nothing
  RegisterDecoderModel(TQwen3Model);
  Check(DecoderModelClass('qwen3') = TQwen3Model, 'still TQwen3Model');
  dir := FindInferRefDir('tiny-qwen3');
  gdir := FindInferRefDir('tiny-qwen3-gguf');
  if CheckFailed((dir <> '') and (gdir <> ''), 'tiny-qwen3 and its GGUF found') then
    exit;
  Check(DecoderModelClassOfFolder(dir) = TQwen3Model, 'the folder is qwen3');
  m := NewDecoderModel(dir, fCpu, idtQ8_0);
  try
    Check(m is TQwen3Model, 'NewDecoderModel: a TQwen3Model');
    Check(m.Weights = idtQ8_0, 'the weights asked for');
    CheckEqual(length(m.Config.Spec), m.Config.Layers, 'a kind per layer');
    attention := true;
    for i := 0 to high(m.Config.Spec) do
      if (m.Config.Spec[i].Mixer <> dmAttention) or (m.Config.Spec[i].Ffn <> dfDense) then
        attention := false;
    Check(attention, 'every layer attention and dense');
    Check(m.HasOnly(dmAttention, dfDense), 'HasOnly');
  finally
    m.Free;
  end;
  m := NewDecoderModelFromGguf(gdir + 'q8_0.gguf', fCpu);
  try
    Check(m is TQwen3Model, 'NewDecoderModelFromGguf: a TQwen3Model');
    CheckEqual(length(m.Config.Spec), m.Config.Layers, 'a kind per layer (GGUF)');
  finally
    m.Free;
  end;
  Check(Refused('', gdir + 'bad-arch.gguf', 'not supported'), 'GGUF of another architecture');
  // a folder of another model_type, and one without config.json
  tmp := MakePath([GetSystemPath(spTemp), 'lai-dec-' + Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  try
    Check(Refused(tmp, '', 'cannot read'), 'no config.json');
    FileFromString(StringReplaceAll(StringFromFile(dir + 'config.json'),
      '"qwen3"', '"llama"'), tmp + 'config.json');
    Check(Refused(tmp, '', 'not supported'), 'model_type llama');
  finally
    DirectoryDeleteAll(tmp);
  end;
end;


/// the bytes of a matrix of aRows x aCols as stored
function MatrixBytes(const w: TInferWeight; aRows, aCols: PtrInt): PtrInt;
begin
  result := (aRows * aCols div INFER_DTYPE_BLOCK[w.DType]) * INFER_DTYPE_SIZE[w.DType];
end;

procedure TTestInferQwen3.TinyMoeMatchesReference;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  full, step, ctx: TDecoderContext;
  ids: PIntegerArray;
  n, i, k, v, vocab, diff: PtrInt;
  all, one: TSingleDynArray;
  prompt, mine: TIntegerDynArray;
  same: boolean;
  margin: double;
  detail: RawUtf8;
begin
  dir := FindInferRefDir('tiny-qwen3moe');
  if CheckFailed(dir <> '', 'tiny-qwen3moe found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  full := nil;
  step := nil;
  try
    model := NewDecoderModel(dir, fCpu);
    Check(model is TQwen3MoeModel, 'qwen3_moe: a TQwen3MoeModel');
    CheckEqual(model.Config.Experts, 8, 'experts');
    CheckEqual(model.Config.ExpertsUsed, 2, 'experts per token');
    CheckEqual(model.Config.ExpertIntermediate, 64, 'expert width');
    Check(model.Config.NormTopK, 'norm_topk_prob');
    Check((model.Config.Spec[0].Ffn = dfMoe) and (model.Config.Spec[1].Ffn = dfDense) and
      (model.Config.Spec[2].Ffn = dfMoe), 'layer 1 dense (mlp_only_layers)');
    Check(not model.HasOnly(dmAttention, dfDense), 'not only dense layers');
    // every state: layer00_mlp is the mixture of experts of layer 0; three
    // layers, so up to 3e-6 apart where tiny-qwen3 stays within 2e-6
    for k := 0 to 2 do
      CheckStates(TQwen3Model(model), ref, FormatUtf8('case%', [k]), 5e-6, 2e-5, true);
    // one token at a time (one row per expert) = the prefill (rows gathered)
    vocab := model.Config.VocabSize;
    ids := ref.I32('case2.ids', n);
    full := model.NewContext;
    step := model.NewContext;
    SetLength(all, n * vocab);
    SetLength(one, vocab);
    model.Forward(full, ids, n, pointer(all), true);
    same := true;
    for i := 0 to n - 1 do
    begin
      model.Forward(step, @ids^[i], 1, pointer(one));
      for v := 0 to vocab - 1 do
        if PCardinal(@one[v])^ <> PCardinal(@all[i * vocab + v])^ then
          same := false;
    end;
    Check(same, 'cached logits bit-identical to the full pass');
    step.Reset;
    model.Forward(step, ids, 4, nil);
    model.Forward(step, @ids^[4], n - 4, pointer(one));
    Check(CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single)),
      'prefill in two chunks');
    // greedy continuations
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        mine := model.Greedy(ctx, prompt, 16, [2]);
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, FormatUtf8('case%', [k]), margin, detail);
      CheckUtf8(diff < 0, 'MoE case% differs at step % (margin %, %)',
        [k, diff, margin, detail]);
    end;
    // the top-k weights not renormed
    FreeAndNil(step);
    FreeAndNil(full);
    FreeAndNil(model);
    dir := MakePath([dir, 'unnormed'], true);
    ref.Free;
    ref := TInferRefSet.CreateFromFolder(dir);
    model := TQwen3MoeModel.Create(dir, fCpu);
    Check(not model.Config.NormTopK, 'norm_topk_prob false');
    CheckStates(TQwen3Model(model), ref, 'case0', 5e-6, 2e-5, true);
  finally
    step.Free;
    full.Free;
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.TinyMoeQuantized;
var
  dir, gdir: TFileName;
  ref: TInferRefSet;
  a, c, g: TDecoderModel;
  ca, cc: TDecoderContext;
  la, lc: TSingleDynArray;
  ids: PIntegerArray;
  n, m, k, v, l, e: PtrInt;
  cmp: TInferCompare;
  same: boolean;
  t: TInferDType;
  f: RawUtf8;

  function Refused(const aFile, aWhy: RawUtf8): boolean;
  begin
    result := false;
    try
      TQwen3MoeModel.CreateFromGguf(gdir + Utf8ToString(aFile), fCpu).Free;
    except
      on E: EInferError do
        result := PosEx(aWhy, StringToUtf8(E.Message)) > 0;
    end;
  end;

begin
  dir := FindInferRefDir('tiny-qwen3moe');
  if CheckFailed(dir <> '', 'tiny-qwen3moe found') then
    exit;
  gdir := MakePath([dir, 'gguf'], true);
  ref := TInferRefSet.CreateFromFolder(dir);
  a := nil;
  c := nil;
  g := nil;
  try
    // Q8_0 at load: the router stays F32, the experts in Q8_0
    a := TQwen3MoeModel.Create(dir, fCpu, idtQ8_0);
    c := TQwen3MoeModel.Create(dir, fFast, idtQ8_0);
    Check(a.Layers[0].Router.DType = idtF32, 'router in F32');
    Check(a.Layers[0].ExpertGate[3].DType = idtQ8_0, 'experts in Q8_0');
    v := a.Config.VocabSize;
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(la, n * v);
      SetLength(lc, n * v);
      ca := a.NewContext;
      cc := c.NewContext;
      try
        a.Forward(ca, ids, n, pointer(la), true);
        c.Forward(cc, ids, n, pointer(lc), true);
      finally
        ca.Free;
        cc.Free;
      end;
      cmp := InferCompare(pointer(lc), pointer(la), n * v, 2e-2, 1e-2);
      CheckUtf8(InferCompareOk(cmp), 'MoE case % Q8_0 float path: %',
        [k, InferCompareText(cmp)]);
      // the router of this tiny model is sharp (logits spread by 0.6 weights):
      // Q8_0 moves a state by up to 0.04 per layer, and token 5 of case 1 by
      // 0.2 in layer 2 - measured 2026-10-17 max 0.21, cosine 0.9991
      cmp := InferCompare(pointer(la), ref.F32(FormatUtf8('case%.logits', [k]), m),
        n * v, 0.3, 0.3);
      CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.998), 'MoE case % Q8_0 vs FP32: %',
        [k, InferCompareText(cmp)]);
      AddConsole(FormatString('MoE case % Q8_0 vs FP32: %', [k, InferCompareText(cmp)]));
    end;
    FreeAndNil(c);
    // GGUF with 3-D expert tensors: every expert byte for byte our own
    // quantization of it (ggml's reference quantizers), the settings alike
    for t := idtQ8_0 to idtQ4_0 do
    begin
      if t = idtQ8_0 then
        f := 'q8_0.gguf'
      else
        f := 'q4_0.gguf';
      FreeAndNil(a);
      FreeAndNil(g);
      a := TQwen3MoeModel.Create(dir, fCpu, t);
      g := NewDecoderModelFromGguf(gdir + Utf8ToString(f), fCpu);
      Check(g is TQwen3MoeModel, 'qwen3moe: a TQwen3MoeModel');
      Check((g.Config.Experts = a.Config.Experts) and
        (g.Config.ExpertsUsed = a.Config.ExpertsUsed) and
        (g.Config.ExpertIntermediate = a.Config.ExpertIntermediate) and
        g.Config.NormTopK, 'the expert settings of the file');
      Check((g.Config.Spec[0].Ffn = dfMoe) and (g.Config.Spec[1].Ffn = dfDense) and
        (g.Config.Spec[2].Ffn = dfMoe), 'dense where ffn_gate_inp is missing');
      same := true;
      for l := 0 to g.Config.Layers - 1 do
        if g.Config.Spec[l].Ffn = dfMoe then
        begin
          if not CompareMem(g.Layers[l].Router.Data, a.Layers[l].Router.Data,
               g.Config.Experts * g.Config.Hidden * SizeOf(single)) then
            same := false;
          for e := 0 to g.Config.Experts - 1 do
            if (g.Layers[l].ExpertGate[e].DType <> t) or
               (g.Layers[l].ExpertDown[e].DType <> t) or
               not CompareMem(g.Layers[l].ExpertGate[e].Data, a.Layers[l].ExpertGate[e].Data,
                 MatrixBytes(g.Layers[l].ExpertGate[e], 64, 64)) or
               not CompareMem(g.Layers[l].ExpertUp[e].Data, a.Layers[l].ExpertUp[e].Data,
                 MatrixBytes(g.Layers[l].ExpertUp[e], 64, 64)) or
               not CompareMem(g.Layers[l].ExpertDown[e].Data, a.Layers[l].ExpertDown[e].Data,
                 MatrixBytes(g.Layers[l].ExpertDown[e], 64, 64)) then
              same := false;
        end
        else if not CompareMem(g.Layers[l].Gate.Data, a.Layers[l].Gate.Data,
                  MatrixBytes(g.Layers[l].Gate, 160, 64)) then
          same := false;
      CheckUtf8(same, '% experts and router like ours', [f]);
    end;
    // the Q4_0 file decodes: its logits close to our Q4_0 model of the same
    // bytes (the embedding is Q8_0 in the file, BF16 in ours)
    ids := ref.I32('case0.ids', n);
    SetLength(la, n * v);
    SetLength(lc, n * v);
    ca := a.NewContext;
    cc := g.NewContext;
    try
      a.Forward(ca, ids, n, pointer(la), true);
      g.Forward(cc, ids, n, pointer(lc), true);
    finally
      ca.Free;
      cc.Free;
    end;
    cmp := InferCompare(pointer(lc), pointer(la), n * v, 0.1, 0.1);
    CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.999), 'Q4_0 GGUF vs ours: %',
      [InferCompareText(cmp)]);
    Check(Refused('bad-experts.gguf', 'shaped'), 'experts one short refused');
  finally
    g.Free;
    c.Free;
    a.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.RouteTapeReplays;
const
  A: array[0..5] of integer = (5, 17, 33, 9, 70, 41);
  B: array[0..5] of integer = (88, 3, 52, 61, 12, 29);
var
  dir: TFileName;
  model: TDecoderModel;
  c1, c2: TDecoderContext;
  tape: TDecoderRouteTape;
  l1, l2, l3: TSingleDynArray;
  v, routes, i: PtrInt;
  raised: boolean;
  x, y: array[0..7] of single;
  cx, cy: array[0..1] of integer;
  wx, wy: array[0..1] of single;
begin
  dir := FindInferRefDir('tiny-qwen3moe');
  if CheckFailed(dir <> '', 'tiny-qwen3moe found') then
    exit;
  model := nil;
  c1 := nil;
  c2 := nil;
  tape := TDecoderRouteTape.Create;
  try
    model := TQwen3MoeModel.Create(dir, fCpu);
    v := model.Config.VocabSize;
    SetLength(l1, v);
    SetLength(l2, v);
    SetLength(l3, v);
    c1 := model.NewContext;
    c2 := model.NewContext;
    // without a tape: A, and B for the forced choice below
    model.Forward(c1, @A[0], length(A), pointer(l2));
    c1.Reset;
    model.Forward(c1, @B[0], length(B), pointer(l3));
    c1.Reset;
    model.RouteTape := tape;
    tape.Mode := drmRecord;
    model.Forward(c1, @A[0], length(A), pointer(l1));
    Check(SameBits(pointer(l1), pointer(l2), v), 'recording leaves the logits as they were');
    // two layers with experts, one route per token in each
    routes := length(A) * 2;
    CheckUtf8(tape.Pending = routes, '% routes recorded', [tape.Pending]);
    tape.Mode := drmReplay;
    model.Forward(c2, @A[0], length(A), pointer(l2));
    Check(SameBits(pointer(l1), pointer(l2), v), 'the same tokens replayed: the same bits');
    CheckUtf8((tape.Routes = routes) and (tape.Differs = 0) and (tape.Pending = 0),
      'replayed %, differing %, pending %', [tape.Routes, tape.Differs, tape.Pending]);
    // nothing left to replay
    raised := false;
    try
      model.Forward(c2, @A[0], 1, pointer(l2));
    except
      on EInferError do
        raised := true;
    end;
    Check(raised, 'replay with nothing recorded');
    // Route itself: the recorded experts and weights handed out bit for bit,
    // whatever the logits of the replaying run - here experts 0 and 1 against
    // 7 and 6, the gap between the 2nd (6: 0.6) and the best left out (5: 0.5)
    for i := 0 to 7 do
    begin
      x[i] := (7 - i) * 0.25;
      y[i] := i * 0.1;
    end;
    c1.Reset;
    tape.Clear;
    tape.Mode := drmRecord;
    model.Route(@x, @cx, @wx);
    Check((cx[0] = 0) and (cx[1] = 1), 'the two highest logits of x');
    tape.Mode := drmReplay;
    model.Route(@y, @cy, @wy);
    Check((cy[0] = cx[0]) and (cy[1] = cx[1]), 'the recorded experts handed out');
    Check(SameBits(@wy, @wx, 2), 'the recorded weights handed out');
    CheckUtf8((tape.Differs = 1) and (Abs(tape.DiffersGap - 0.1) < 1e-6),
      'one differing choice at a gap of 0.1: %, %', [tape.Differs, tape.DiffersGap]);
    // the choice of A forced on the tokens of B: differing choices, other logits
    c1.Reset;
    c2.Reset;
    tape.Clear;
    tape.Mode := drmRecord;
    model.Forward(c1, @A[0], length(A), pointer(l1));
    tape.Mode := drmReplay;
    model.Forward(c2, @B[0], length(B), pointer(l2));
    CheckUtf8((tape.Differs > 0) and (tape.DiffersGap > 0),
      'other tokens: % differing choices, largest gap %', [tape.Differs, tape.DiffersGap]);
    Check(not SameBits(pointer(l2), pointer(l3), v), 'a forced choice changes the logits');
  finally
    if model <> nil then
      model.RouteTape := nil;
    tape.Free;
    c2.Free;
    c1.Free;
    model.Free;
  end;
end;

procedure TTestInferQwen3.TinyMoeRefused;
var
  dir, tmp: TFileName;
  cfg, why: RawUtf8;

  function LoadRefused(const aFrom, aTo: RawUtf8): boolean;
  begin
    result := false;
    FileFromString(StringReplaceAll(cfg, aFrom, aTo), tmp + 'config.json');
    try
      TQwen3MoeModel.Create(tmp, fCpu).Free;
    except
      on EInferError do
        result := true;
    end;
  end;

begin
  dir := FindInferRefDir('tiny-qwen3moe');
  if CheckFailed(dir <> '', 'tiny-qwen3moe found') then
    exit;
  // LF whatever a checkout made of the line ends: the patterns below rely on it
  cfg := StringReplaceAll(StringFromFile(dir + 'config.json'), #13#10, #10);
  tmp := MakePath([GetSystemPath(spTemp), 'lai-moe-' + Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  try
    CopyFile(dir + 'model.safetensors', tmp + 'model.safetensors', false);
    Check(not LoadRefused('"x"', '"x"'), 'the copy loads');
    Check(LoadRefused('"decoder_sparse_step": 1', '"decoder_sparse_step": 0'),
      'decoder_sparse_step 0');
    Check(LoadRefused('"num_experts": 8', '"num_expertz": 8'), 'no num_experts');
    Check(LoadRefused('"num_experts_per_tok": 2', '"num_experts_per_tok": 9'),
      'more experts per token than experts');
    Check(LoadRefused('"num_experts": 8', '"num_experts": 7'),
      'the experts of the file one more');
    Check(LoadRefused('"moe_intermediate_size": 64', '"moe_intermediate_size": 32'),
      'the expert width not that of the weights');
    // decoder_sparse_step 2 and no mlp_only_layers: layers 0 and 2 dense, 1
    // with experts - the file has it the other way round, the dense block of
    // layer 0 is missing (the first tensor looked for)
    FileFromString(StringReplaceAll(StringReplaceAll(cfg,
      '"decoder_sparse_step": 1', '"decoder_sparse_step": 2'),
      '"mlp_only_layers": [' + #10 + '  1' + #10 + ' ]', '"mlp_only_layers": []'),
      tmp + 'config.json');
    why := '';
    try
      TQwen3MoeModel.Create(tmp, fCpu).Free;
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    CheckUtf8(PosEx('no tensor model.layers.0.mlp.gate_proj.weight', why) > 0,
      'layer kinds other than the file: %', [why]);
    // a qwen3_moe folder is no qwen3 model
    FileFromString(cfg, tmp + 'config.json');
    Check(not LoadRefused('"x"', '"x"'), 'restored');
    try
      TQwen3Model.Create(tmp, fCpu).Free;
      Check(false, 'TQwen3Model loads qwen3_moe');
    except
      on EInferError do
        Check(true);
    end;
  finally
    DirectoryDeleteAll(tmp);
  end;
end;

function TTestInferQwen3.Qwen25Folder: TFileName;
begin
  result := '';
  {$ifdef CPU32}
  AddConsole('SKIP Qwen2.5-0.5B: 2 GB of FP32 weights need a 64-bit process');
  exit;
  {$endif CPU32}
  result := FindVendorModel('qwen2.5-0.5b-instruct' + PathDelim + 'model.safetensors');
  if result <> '' then
  begin
    result := ExtractFilePath(result);
    exit;
  end;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen2.5-0.5b-instruct/model.safetensors');
  AddConsole('SKIP Qwen2.5-0.5B: no vendor/models/qwen2.5-0.5b-instruct (tools/infer-ref, ' +
    'make_reference_qwen2.py)');
end;

procedure TTestInferQwen3.TinyQwen2MatchesReference;
var
  dir, gdir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  ctx: TDecoderContext;
  ids: PIntegerArray;
  prompt, mine: TIntegerDynArray;
  logits: TSingleDynArray;
  why: RawUtf8;
  k, n, diff, m: PtrInt;
  margin: double;
  detail: RawUtf8;
  cmp: TInferCompare;
begin
  dir := FindInferRefDir('tiny-qwen2');
  if CheckFailed(dir <> '', 'tiny-qwen2 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  try
    // the family from config.json, head_dim from hidden / heads
    model := NewDecoderModel(dir, fCpu);
    Check(model is TQwen2Model, 'model_type qwen2: a TQwen2Model');
    CheckEqual(model.Config.HeadDim, 16, 'head_dim = 64 / 4');
    Check((model.Layers[0].QBias <> nil) and (model.Layers[1].VBias <> nil) and
      (model.Layers[0].QNorm = nil), 'biases, no head norm');
    for k := 0 to 2 do
      CheckStates(TQwen2Model(model), ref, FormatUtf8('case%', [k]), 2e-6, 2e-5, true);
    // greedy, through the KV cache
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        mine := model.Greedy(ctx, prompt, 16, [2]);
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, FormatUtf8('case%', [k]), margin, detail);
      CheckUtf8(diff < 0, 'tiny qwen2 case% greedy differs at step % (margin %, %)',
        [k, diff, margin, detail]);
    end;
    FreeAndNil(model);
    // as llama.cpp writes a qwen2 file: no key_length, the biases F32
    gdir := MakePath([dir, 'gguf'], true);
    model := NewDecoderModelFromGguf(gdir + 'q8_0.gguf', fCpu);
    Check(model is TQwen2Model, 'architecture qwen2: a TQwen2Model');
    CheckEqual(model.Config.HeadDim, 16, 'GGUF head_dim = 64 / 4');
    ids := ref.I32('case2.ids', n);
    SetLength(logits, model.Config.VocabSize);
    ctx := model.NewContext;
    try
      model.Forward(ctx, ids, n, pointer(logits));
    finally
      ctx.Free;
    end;
    cmp := InferCompare(pointer(logits),
      @ref.F32('case2.logits', m)^[(n - 1) * model.Config.VocabSize],
      model.Config.VocabSize, 0.05, 0.05);
    CheckUtf8((cmp.NonFinite = 0) and (cmp.Cosine >= 0.999), 'tiny qwen2 Q8_0 GGUF: %',
      [InferCompareText(cmp)]);
    FreeAndNil(model);
    why := '';
    try
      NewDecoderModelFromGguf(gdir + 'bad-nobias.gguf', fCpu).Free;
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    CheckUtf8(PosEx('attn_k.bias', why) > 0, 'a missing bias refused: %', [why]);
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.Qwen25MatchesReference;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  ctx: TDecoderContext;
  prompts, p: PDocVariantData;
  ids: PIntegerArray;
  i, n, diff: PtrInt;
  prompt, mine: TIntegerDynArray;
  margin: double;
  detail: RawUtf8;
  timer: TPrecisionTimer;
begin
  dir := Qwen25Folder;
  if dir = '' then
    exit;
  refdir := FindInferRefDir('qwen25');
  if CheckFailed(refdir <> '', 'qwen25 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  try
    model := NewDecoderModel(dir, fFast);
    Check(model is TQwen2Model, 'Qwen2.5: a TQwen2Model');
    CheckEqual(model.Config.Layers, 24, 'Qwen2.5-0.5B depth');
    CheckEqual(model.Config.HeadDim, 64, 'head_dim 896 / 14');
    prompts := ref.Manifest^.A['prompts'];
    for i := 0 to prompts^.Count - 1 do
    begin
      p := _Safe(prompts^.Values[i]);
      CheckStates(TQwen2Model(model), ref, p^.U['id'], 2e-3, 2e-3, false);
      ids := ref.I32(p^.U['id'] + '.ids', n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        timer.Start;
        mine := model.Greedy(ctx, prompt, 48, [151645, 151643]);
        timer.Stop;
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, p^.U['id'], margin, detail);
      CheckUtf8(diff < 0, 'Qwen2.5 % differs at step % (reference margin %, %)',
        [p^.U['id'], diff, margin, detail]);
      AddConsole(FormatString('Qwen2.5 % greedy: % + % tokens in %', [p^.U['id'], n,
        length(mine), timer.Time]));
    end;
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.Qwen25Gguf;
var
  dir, gguf: TFileName;
  f: integer;
  q8: TFileName;
  g, g8: TQwen3Model;
  detail: RawUtf8;
  cs: double;
const
  FILES: array[0..1] of TFileName = (
    'qwen2.5-0.5b-instruct-q8_0.gguf', 'qwen2.5-0.5b-instruct-q4_k_m.gguf');
begin
  dir := Qwen25Folder;
  if dir = '' then
    exit;
  for f := 0 to high(FILES) do
  begin
    gguf := FindVendorModel(FILES[f]);
    if gguf = '' then
    begin
      CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
        'LAI_REQUIRE_MODEL=1 but no vendor/models/%', [FILES[f]]);
      AddConsole('SKIP Qwen2.5 GGUF: no vendor/models/%', [FILES[f]]);
      continue;
    end;
    if f = 0 then
      // close, greedy equal up to a step the reference was less sure of than
      // the logits differ: Qwen2.5-0.5B moves more under Q8_0 than Qwen3-0.6B
      // (measured 2026-10-19: cosine 0.990 to 0.995, max error 2.6; the
      // chat_system answer leaves the reference where it was 1.42 sure)
      Qwen3Quantized(idtQ8_0, 0.98, 3, 3, gguf, 'qwen25', dir)
    else
    begin
      // llama.cpp falls back to Q5_0 where a row is no multiple of 256 (hidden
      // 896): Q5_0 for the embedding and most matrices, Q4_K and Q6_K only for
      // ffn_down, Q8_0 for the head and half of attn_v. Against Qwen's Q8_0
      // file, not the BF16 weights: both GGUF files come from another
      // checkpoint than model.safetensors (Q8_0 itself has cosine 0.81 there
      // for layer 9 k_proj, measured 2026-10-19) - every matrix close to its
      // Q8_0 form, which a wrong fifth bit would not be
      q8 := FindVendorModel(FILES[0]);
      if q8 <> '' then
      begin
        g := NewDecoderModelFromGguf(gguf, fFast) as TQwen3Model;
        g8 := nil;
        try
          Check(g.Layers[0].Q.DType = idtQ5_0, 'Q5_0 layer matrices');
          Check(g.Head.DType = idtQ8_0, 'a Q8_0 head');
          g8 := NewDecoderModelFromGguf(q8, fFast) as TQwen3Model;
          cs := ModelCosine(g, g8, detail);
          AddConsole(FormatString('Qwen2.5 Q4_K_M against Q8_0: lowest cosine % (%)',
            [cs, detail]));
          CheckUtf8(cs >= 0.995, 'Qwen2.5 Q4_K_M like Q8_0: % %', [cs, detail]);
        finally
          g8.Free;
          g.Free;
        end;
      end;
      // against the FP32 reference (measured 2026-10-19, NEON): logits cosine
      // 0.968 (chat_tool) to 0.991, max error 3.9; greedy leaves the reference
      // where it was 0.87 sure at most
      Qwen3Quantized(idtQ5_0, 0.95, 6, 2, gguf, 'qwen25', dir);
    end;
  end;
end;



procedure TTestInferQwen3.SlidingWindowMatchesReference;
const
  NAMES: array[0..1] of RawUtf8 = ('tiny-smollm3', 'tiny-qwen3-sliding');
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  ctx: TDecoderContext;
  ids: PIntegerArray;
  prompt, mine: TIntegerDynArray;
  d, k, n, diff: PtrInt;
  margin: double;
  detail: RawUtf8;
  s: TDecoderLayerSpecDynArray;
begin
  for d := 0 to high(NAMES) do
  begin
    dir := FindInferRefDir(NAMES[d]);
    if CheckFailed(dir <> '', Utf8ToString(NAMES[d] + ' found')) then
      continue;
    ref := TInferRefSet.CreateFromFolder(dir);
    model := nil;
    try
      model := NewDecoderModel(dir, fCpu);
      s := model.Config.Spec;
      if d = 0 then
      begin
        Check(model is TSmolLM3Model, 'model_type smollm3: a TSmolLM3Model');
        CheckEqual(model.Config.HeadDim, 16, 'head_dim = 64 / 4');
        Check((s[0].Window = 0) and not s[0].NoRope and (s[1].Window = 6) and
          s[1].NoRope and (s[2].Window = 6) and not s[2].NoRope and
          (s[3].Window = 0) and s[3].NoRope, 'smollm3: the kind of every layer');
        Check((model.Layers[0].QNorm = nil) and (model.Layers[0].QBias = nil),
          'no head norm, no bias');
      end
      else
      begin
        Check(model.ClassType = TQwen3Model, 'model_type qwen3');
        Check((s[0].Window = 0) and (s[1].Window = 5) and not s[0].NoRope and
          not s[1].NoRope, 'qwen3: the window from max_window_layers');
      end;
      // every state of 5 (inside one window), 13 and 29 tokens - SmolLM3 has
      // four layers, twice tiny-qwen2: its final norm differs by 4.03e-6 (5e-6 allowed)
      for k := 0 to 2 do
        CheckStates(TQwen3Model(model), ref, FormatUtf8('case%', [k]),
          2.5e-6 * model.Config.Layers / 2, 2e-5, true);
      // greedy through the sliding cache: 24 tokens wrap the window again and
      // again
      for k := 0 to 2 do
      begin
        ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
        SetLength(prompt, n);
        MoveFast(ids^, prompt[0], n * SizeOf(integer));
        ctx := model.NewContext;
        try
          mine := model.Greedy(ctx, prompt, 24, []);
        finally
          ctx.Free;
        end;
        diff := GreedyDiff(mine, ref, FormatUtf8('case%', [k]), margin, detail);
        CheckUtf8(diff < 0, '% case% greedy differs at step % (margin %, %)',
          [NAMES[d], k, diff, margin, detail]);
      end;
    finally
      model.Free;
      ref.Free;
    end;
  end;
end;

/// feed aIds[0..aCount-1] in chunks of the sizes of aChunks (cycled), the
// logits of every position into aLogits ([aCount, vocab])
procedure FeedChunks(aModel: TDecoderModel; aCtx: TDecoderContext;
  aIds: PIntegerArray; aCount: PtrInt; const aChunks: array of integer;
  var aLogits: TSingleDynArray);
var
  p, c, k: PtrInt;
begin
  SetLength(aLogits, aCount * aModel.Config.VocabSize);
  p := 0;
  k := 0;
  while p < aCount do
  begin
    c := aChunks[k mod length(aChunks)];
    inc(k);
    if c > aCount - p then
      c := aCount - p;
    aModel.Forward(aCtx, @aIds^[p], c, @aLogits[p * aModel.Config.VocabSize], true);
    inc(p, c);
  end;
end;

procedure TTestInferQwen3.SlidingWindowCache;
const
  // chunk sizes: one by one (the window moves at every few tokens), around the
  // window of 6, larger than it, and a mix
  PATTERNS: array[0..6] of array[0..3] of integer = (
    (1, 1, 1, 1), (2, 2, 2, 2), (5, 5, 5, 5), (6, 6, 6, 6), (7, 7, 7, 7),
    (13, 13, 13, 13), (3, 11, 1, 20));
  BRANCHED = 8;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  ctx, full: TDecoderContext;
  backend: TInferBackend;
  b, p, n, m, t, vocab, used, restorable, lost: PtrInt;
  ids, more: PIntegerArray;
  seq, branch: TIntegerDynArray;
  want, got, one: TSingleDynArray;
  raised: boolean;
begin
  dir := FindInferRefDir('tiny-smollm3');
  if CheckFailed(dir <> '', 'tiny-smollm3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    // 29 tokens of case 2 and its 24 greedy ones: 53 positions, the window of
    // 6 moves on many times
    ids := ref.I32('case2.ids', n);
    more := ref.I32('case2.greedy', m);
    SetLength(seq, n + m);
    MoveFast(ids^, seq[0], n * SizeOf(integer));
    MoveFast(more^, seq[n], m * SizeOf(integer));
    n := length(seq);
    ids := ref.I32('case1.ids', m);
    SetLength(branch, BRANCHED);
    MoveFast(ids^, branch[0], BRANCHED * SizeOf(integer));
    for b := 0 to 1 do
    begin
      if b = 0 then
        backend := fCpu
      else
        backend := fFast;
      model := NewDecoderModel(dir, backend);
      ctx := nil;
      full := nil;
      try
        vocab := model.Config.VocabSize;
        // the reference: every key kept, the window only masks
        full := model.NewContext;
        full.FullCache := true;
        FeedChunks(model, full, pointer(seq), n, [n], want);
        Check(full.CachedRows(1) >= n, 'FullCache: every row of a window layer');
        // the sliding cache in every chunk pattern: the same bits
        for p := 0 to high(PATTERNS) do
        begin
          FreeAndNil(ctx);
          ctx := model.NewContext;
          FeedChunks(model, ctx, pointer(seq), n, PATTERNS[p], got);
          CheckUtf8(SameBits(pointer(got), pointer(want), n * vocab),
            'backend %, chunks of %: the logits of the full cache', [b, PATTERNS[p][0]]);
          CheckEqual(ctx.Length, n, 'every token cached');
          if p = 0 then
          begin
            // one by one: twice the window per sliding layer, all for a full one
            CheckUtf8((ctx.CachedRows(1) = 12) and (ctx.CachedRows(2) = 12) and
              (ctx.CachedRows(0) >= n) and (ctx.CachedRows(3) >= n),
              'rows kept: % % % %', [ctx.CachedRows(0), ctx.CachedRows(1),
              ctx.CachedRows(2), ctx.CachedRows(3)]);
            Check((ctx.CachedBase(1) >= n - 12) and (ctx.CachedBase(0) = 0),
              'the window has moved on, the full layer not');
          end;
        end;
        // the full cache fed in chunks: the same bits too
        FreeAndNil(ctx);
        ctx := model.NewContext;
        ctx.FullCache := true;
        FeedChunks(model, ctx, pointer(seq), n, PATTERNS[6], got);
        Check(SameBits(pointer(got), pointer(want), n * vocab), 'full cache in chunks');
        // going back: a branch after every length t of a sliding cache fed one
        // by one, against computing prefix and branch again
        restorable := 0;
        lost := 0;
        SetLength(one, BRANCHED * vocab);
        for t := 0 to 40 do
        begin
          FreeAndNil(ctx);
          ctx := model.NewContext;
          FeedChunks(model, ctx, pointer(seq), 40, [1], got);
          // the next token at t sees t - 5 .. t: the rows from there on must
          // still be in both sliding layers, and only then
          CheckUtf8(ctx.Restorable(t) = ((t = 0) or (t - 5 >= MaxPtrInt(ctx.CachedBase(1),
            ctx.CachedBase(2)))), 'Restorable(%) with the window from %',
            [t, ctx.CachedBase(1)]);
          if ctx.Restorable(t) then
          begin
            inc(restorable);
            ctx.Truncate(t);
          end
          else
          begin
            inc(lost);
            raised := false;
            try
              ctx.Truncate(t);
            except
              on EInferError do
                raised := true;
            end;
            Check(raised, 'Truncate to a length whose window is gone');
            CheckEqual(ctx.Length, 40, 'a refused Truncate keeps the cache');
            // Reuse starts again, the window cannot come back
            used := model.Reuse(ctx, pointer(seq), t + 1);
            CheckEqual(used, 0, 'Reuse of a prefix that cannot be restored');
            if t > 0 then
              model.Forward(ctx, pointer(seq), t, nil);
          end;
          model.Forward(ctx, pointer(branch), BRANCHED, pointer(one), true);
          full.Reset;
          if t > 0 then
            model.Forward(full, pointer(seq), t, nil);
          SetLength(got, BRANCHED * vocab);
          model.Forward(full, pointer(branch), BRANCHED, pointer(got), true);
          CheckUtf8(SameBits(pointer(one), pointer(got), BRANCHED * vocab),
            'backend %: a branch after % tokens', [b, t]);
        end;
        CheckUtf8((restorable > 5) and (lost > 5), 'restorable % and lost % lengths',
          [restorable, lost]);
        // a large prefill leaves many rows, the next decode step gives them back
        FreeAndNil(ctx);
        ctx := model.NewContext;
        FeedChunks(model, ctx, pointer(seq), 40, [40], got);
        Check(ctx.CachedRows(1) >= 40, 'rows of the prefill');
        model.Forward(ctx, @seq[40], 1, pointer(one));
        CheckEqual(ctx.CachedRows(1), 12, 'shrunk to twice the window');
        Check(SameBits(pointer(one), @want[40 * vocab], vocab), 'the logits after shrinking');
        // and after a step of a whole window: twice the window as well
        FreeAndNil(ctx);
        ctx := model.NewContext;
        FeedChunks(model, ctx, pointer(seq), 40, [40], got);
        SetLength(got, 6 * vocab);
        model.Forward(ctx, @seq[40], 6, pointer(got), true);
        CheckEqual(ctx.CachedRows(1), 12, 'shrunk by a step of one window');
        Check(SameBits(pointer(got), @want[40 * vocab], 6 * vocab),
          'the logits of that step');
        // Reuse keeps the longest prefix it can go back to
        FreeAndNil(ctx);
        ctx := model.NewContext;
        FeedChunks(model, ctx, pointer(seq), 40, [1], got);
        CheckEqual(model.Reuse(ctx, pointer(seq), n), 40, 'Reuse of the whole cache');
        CheckEqual(ctx.Usable(pointer(seq), 2), 0, 'a short prefix is gone');
        // a full cache can go back to every length
        FreeAndNil(ctx);
        ctx := model.NewContext;
        ctx.FullCache := true;
        FeedChunks(model, ctx, pointer(seq), 40, [1], got);
        for t := 0 to 40 do
          Check(ctx.Restorable(t), 'FullCache: every length restorable');
        raised := false;
        try
          ctx.FullCache := false;
        except
          on EInferError do
            raised := true;
        end;
        Check(raised, 'FullCache changed on a filled context');
        ctx.Reset;
        ctx.FullCache := false;
        FeedChunks(model, ctx, pointer(seq), n, [4], got);
        Check(SameBits(pointer(got), pointer(want), n * vocab),
          'a reset context switched to the sliding cache');
      finally
        ctx.Free;
        full.Free;
        model.Free;
      end;
    end;
  finally
    ref.Free;
  end;
end;

procedure TTestInferQwen3.SlidingWindowConfig;
var
  dir, qdir, tmp: TFileName;
  base, doc: TDocVariantData;
  why: RawUtf8;
  m: TDecoderModel;
  s: TDecoderLayerSpecDynArray;
  nullv: variant;

  procedure Start(const aFrom: TFileName);
  begin
    base.Clear;
    base.InitJsonFromFile(aFrom + 'config.json', JSON_FAST_FLOAT);
    CopyFile(aFrom + 'model.safetensors', tmp + 'model.safetensors', false);
    doc.Clear;
    doc.InitCopy(variant(base), JSON_FAST_FLOAT);
  end;

  function Load: TDecoderModel;
  begin
    FileFromString(doc.ToJson, tmp + 'config.json');
    doc.Clear;
    doc.InitCopy(variant(base), JSON_FAST_FLOAT); // the next change from the file
    result := NewDecoderModel(tmp, fCpu);
  end;

  function Refused: RawUtf8;
  begin
    result := '';
    try
      Load.Free;
    except
      on E: EInferError do
        result := StringToUtf8(E.Message);
    end;
  end;

begin
  dir := FindInferRefDir('tiny-smollm3');
  qdir := FindInferRefDir('tiny-qwen3-sliding');
  if CheckFailed((dir <> '') and (qdir <> ''), 'tiny-smollm3 and tiny-qwen3-sliding found') then
    exit;
  base.InitFast;
  doc.InitFast;
  SetVariantNull(nullv); // JSON null
  tmp := MakePath([GetSystemPath(spTemp), 'lai-d4-' + Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  try
    Start(dir);
    // no layer_types, no no_rope_layers: as SmolLM3Config derives them - every
    // second layer without RoPE (interval 2), the window on those
    doc.Delete('layer_types');
    doc.Delete('no_rope_layers');
    doc.I['no_rope_layer_interval'] := 2;
    m := Load;
    try
      s := m.Config.Spec;
      Check(not s[0].NoRope and s[1].NoRope and not s[2].NoRope and s[3].NoRope and
        (s[0].Window = 0) and (s[1].Window = 6) and (s[2].Window = 0) and
        (s[3].Window = 6), 'smollm3: kinds from no_rope_layer_interval');
    finally
      m.Free;
    end;
    // without use_sliding_window: no window anywhere
    doc.Delete('layer_types');
    doc.B['use_sliding_window'] := false;
    m := Load;
    try
      s := m.Config.Spec;
      Check((s[1].Window = 0) and (s[2].Window = 0) and s[1].NoRope and s[3].NoRope,
        'smollm3: no window without use_sliding_window');
    finally
      m.Free;
    end;
    // SmolLM3Config keeps sliding_window without use_sliding_window, and the
    // model masks by layer_types: the explicit sliding layers keep it
    doc.B['use_sliding_window'] := false;
    m := Load;
    try
      s := m.Config.Spec;
      Check((s[1].Window = 6) and (s[2].Window = 6) and (s[0].Window = 0),
        'smollm3: explicit sliding layers without use_sliding_window');
    finally
      m.Free;
    end;
    doc.AddOrUpdateValue('sliding_window', nullv);
    why := Refused;
    CheckUtf8(PosEx('without a sliding window', why) > 0,
      'a sliding layer type, sliding_window null: %', [why]);
    doc.I['sliding_window'] := 1 shl 30;
    why := Refused;
    CheckUtf8(PosEx('a sliding_window of', why) > 0, 'a window beyond the bound: %', [why]);
    doc.AddOrUpdateValue('layer_types', _JsonFast(
      '["full_attention","chunked_attention","sliding_attention","full_attention"]'));
    why := Refused;
    CheckUtf8(PosEx('chunked_attention', why) > 0, 'an unknown layer type: %', [why]);
    doc.AddOrUpdateValue('layer_types', _JsonFast('["full_attention"]'));
    why := Refused;
    CheckUtf8(PosEx('layer types for', why) > 0, 'layer types of another count: %', [why]);
    doc.AddOrUpdateValue('no_rope_layers', _JsonFast('[1,0,1]'));
    why := Refused;
    CheckUtf8(PosEx('no_rope_layers for', why) > 0, 'no_rope_layers of another count: %',
      [why]);
    doc.AddOrUpdateValue('no_rope_layers', _JsonFast('[1,2,1,0]'));
    why := Refused;
    CheckUtf8(PosEx('no_rope_layers[1] = 2', why) > 0, 'no_rope_layers with a 2: %', [why]);
    doc.Delete('no_rope_layers');
    doc.Delete('layer_types');
    doc.I['no_rope_layer_interval'] := 0;
    why := Refused;
    CheckUtf8(PosEx('no_rope_layer_interval = 0', why) > 0, 'no_rope_layer_interval 0: %',
      [why]);
    doc.B['mlp_bias'] := true;
    why := Refused;
    CheckUtf8(PosEx('mlp_bias', why) > 0, 'mlp_bias: %', [why]);
    // SmolLM3 from GGUF: llama.cpp fixes the NoPE layers - refused
    why := '';
    try
      TSmolLM3Model.CreateFromGguf(MakePath([FindInferRefDir('tiny-qwen2'), 'gguf'], true) +
        'q8_0.gguf', fCpu).Free;
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    CheckUtf8(PosEx('GGUF not supported', why) > 0, 'SmolLM3 from GGUF: %', [why]);
    // Qwen3: without max_window_layers its default 28 - no window in 2 layers
    Start(qdir);
    doc.Delete('max_window_layers');
    m := Load;
    try
      Check((m.Config.Spec[0].Window = 0) and (m.Config.Spec[1].Window = 0),
        'qwen3: max_window_layers 28 by default');
    finally
      m.Free;
    end;
    // without a sliding_window key its default 4096, null: none
    doc.Delete('sliding_window');
    m := Load;
    try
      CheckEqual(m.Config.Spec[1].Window, 4096, 'qwen3: sliding_window 4096 by default');
    finally
      m.Free;
    end;
    doc.AddOrUpdateValue('sliding_window', nullv);
    m := Load;
    try
      CheckEqual(m.Config.Spec[1].Window, 0, 'qwen3: sliding_window null');
    finally
      m.Free;
    end;
    // a sliding layer type without use_sliding_window: Qwen3Config drops the
    // window, transformers cannot mask it - refused
    doc.B['use_sliding_window'] := false;
    doc.AddOrUpdateValue('layer_types', _JsonFast('["sliding_attention","full_attention"]'));
    why := Refused;
    CheckUtf8(PosEx('without a sliding window', why) > 0,
      'qwen3: a sliding layer type without use_sliding_window: %', [why]);
    doc.AddOrUpdateValue('layer_types', _JsonFast('["sliding_attention","full_attention"]'));
    m := Load;
    try
      Check((m.Config.Spec[0].Window = 5) and (m.Config.Spec[1].Window = 0),
        'qwen3: the window from layer_types');
    finally
      m.Free;
    end;
    doc.B['use_sliding_window'] := false;
    m := Load;
    try
      Check((m.Config.Spec[0].Window = 0) and (m.Config.Spec[1].Window = 0),
        'qwen3: no window without use_sliding_window');
    finally
      m.Free;
    end;
    // Qwen3-MoE: no max_window_layers, every layer slides
    Start(FindInferRefDir('tiny-qwen3moe'));
    doc.B['use_sliding_window'] := true;
    doc.I['sliding_window'] := 5;
    m := Load;
    try
      Check((m.Config.Spec[0].Window = 5) and (m.Config.Spec[1].Window = 5) and
        (m.Config.Spec[2].Window = 5), 'qwen3_moe: a window on every layer');
    finally
      m.Free;
    end;
  finally
    DirectoryDeleteAll(tmp);
  end;
end;


procedure TTestInferQwen3.EmbeddingRows;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  a, b: TQwen3Context;
  ids: PIntegerArray;
  n, h, vocab, cut1, cut2: PtrInt;
  rows, all, mine: TSingleDynArray;
  raised: boolean;
  neg: integer;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  a := nil;
  b := nil;
  try
    model := TQwen3Model.Create(dir, fCpu);
    Check(model.HasEmbedding, 'a token embedding');
    ids := ref.I32('case2.ids', n);
    if CheckFailed(n >= 8, 'case2 long enough') then
      exit;
    h := model.Config.Hidden;
    vocab := model.Config.VocabSize;
    SetLength(all, n * vocab);
    SetLength(mine, n * vocab);
    SetLength(rows, n * h);
    a := model.NewContext;
    b := model.NewContext;
    model.Forward(a, ids, n, pointer(all), true);
    // the rows of Embed in place of the ids: the very same logits
    model.Embed(ids, n, pointer(rows));
    model.ForwardEmbeds(b, pointer(rows), n, pointer(mine), true);
    Check(CompareMem(pointer(all), pointer(mine), n * vocab * SizeOf(single)),
      'embedding rows give the bits of their ids');
    Check((b.Length = n) and (b.Id[0] = -1) and (b.Id[n - 1] = -1),
      'no id where a row was fed');
    // not even a caller's own -1 matches such a position
    neg := -1;
    CheckEqual(b.Common(@neg, 1), 0, 'a negative id matches no row');
    // ids, rows, ids in three chunks: the bits of one pass
    b.Reset;
    cut1 := 3;
    cut2 := n - 4;
    FillCharFast(mine[0], n * vocab * SizeOf(single), 0);
    model.Forward(b, ids, cut1, @mine[0], true);
    model.ForwardEmbeds(b, @rows[cut1 * h], cut2 - cut1, @mine[cut1 * vocab], true);
    model.Forward(b, @ids^[cut2], n - cut2, @mine[cut2 * vocab], true);
    Check(CompareMem(pointer(all), pointer(mine), n * vocab * SizeOf(single)),
      'ids and rows mixed over chunks');
    Check((b.Id[cut1 - 1] = ids^[cut1 - 1]) and (b.Id[cut1] = -1) and
      (b.Id[cut2 - 1] = -1) and (b.Id[cut2] = ids^[cut2]), 'ids around the rows');
    // the prompt cache keeps the ids before the first row only
    CheckEqual(model.Reuse(b, ids, n), cut1, 'Reuse stops at the first row');
    CheckEqual(b.Length, cut1, 'cache cut there');
    raised := false;
    try
      model.ForwardEmbeds(b, nil, 2, nil);
    except
      on EInferError do
        raised := true;
    end;
    Check(raised, 'no rows refused');
  finally
    b.Free;
    a.Free;
    model.Free;
    ref.Free;
  end;
end;

/// the message of the EInferError aModel's CreateStack raises, '' if none
function StackRefused(const aDir: TFileName; aBackend: TInferBackend;
  const aConfig, aPrefix, aEmbedding, aHead: RawUtf8): RawUtf8;
begin
  result := '';
  try
    TQwen3Model.CreateStack(aDir, aBackend, aConfig, aPrefix, aEmbedding, aHead).Free;
  except
    on E: EInferError do
      result := StringToUtf8(E.Message);
  end;
end;

procedure TTestInferQwen3.StackFromCheckpoint;
var
  dir, sdir: TFileName;
  ref: TInferRefSet;
  plain, stack, inner, rowsOnly, headless: TQwen3Model;
  ctx: TQwen3Context;
  ids: PIntegerArray;
  n, h, vocab: PtrInt;
  want, mine, rows, normed, normed2: TSingleDynArray;
  why: RawUtf8;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  sdir := dir + 'stack' + PathDelim;
  ref := TInferRefSet.CreateFromFolder(dir);
  plain := nil;
  stack := nil;
  inner := nil;
  rowsOnly := nil;
  headless := nil;
  ctx := nil;
  try
    plain := TQwen3Model.Create(dir, fCpu);
    // tiny-qwen3 under "outer.model.", its settings in an object of
    // config.json with the multimodal RoPE of a talker, its head on its own
    stack := TQwen3Model.CreateStack(sdir, fCpu, 'outer_config', 'outer.model.',
      'outer.model.embed_tokens.weight', 'outer.head.weight');
    inner := TQwen3Model.CreateStack(sdir, fCpu, 'outer_config.inner',
      'outer.model.', 'outer.model.embed_tokens.weight', 'outer.head.weight');
    rowsOnly := TQwen3Model.CreateStack(sdir, fCpu, 'outer_config', 'outer.model.',
      '', 'outer.head.weight');
    Check(not stack.Config.TiedEmbeddings and (stack.Config.Layers = 2),
      'the settings of the object');
    ids := ref.I32('case2.ids', n);
    h := plain.Config.Hidden;
    vocab := plain.Config.VocabSize;
    SetLength(want, n * vocab);
    SetLength(mine, n * vocab);
    ctx := plain.NewContext;
    plain.Forward(ctx, ids, n, pointer(want), true);
    FreeAndNil(ctx);
    ctx := stack.NewContext;
    stack.Forward(ctx, ids, n, pointer(mine), true);
    FreeAndNil(ctx);
    Check(CompareMem(pointer(want), pointer(mine), n * vocab * SizeOf(single)),
      'the stack gives the bits of the model');
    ctx := inner.NewContext;
    inner.Forward(ctx, ids, n, pointer(mine), true);
    FreeAndNil(ctx);
    Check(CompareMem(pointer(want), pointer(mine), n * vocab * SizeOf(single)),
      'a dotted path to the settings');
    // without a token embedding: rows only
    Check(not rowsOnly.HasEmbedding, 'no embedding');
    SetLength(rows, n * h);
    plain.Embed(ids, n, pointer(rows));
    ctx := rowsOnly.NewContext;
    rowsOnly.ForwardEmbeds(ctx, pointer(rows), n, pointer(mine), true);
    Check(CompareMem(pointer(want), pointer(mine), n * vocab * SizeOf(single)),
      'rows through a stack without embedding');
    why := '';
    try
      rowsOnly.Forward(ctx, ids, 1, nil);
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    CheckUtf8(PosEx('no token embedding', why) > 0, 'ids refused: %', [why]);
    FreeAndNil(ctx);
    // the normed last row: the 'norm' state of that row, with the logits
    // or without them
    SetLength(normed, h);
    SetLength(normed2, h);
    fNames := nil;
    fStates := nil;
    ctx := rowsOnly.NewContext;
    rowsOnly.ForwardEmbeds(ctx, pointer(rows), n, pointer(mine), false, KeepState,
      pointer(normed));
    FreeAndNil(ctx);
    Check(CompareMem(pointer(normed), @State('norm')^[(n - 1) * h], h * SizeOf(single)),
      'aNormed is the normed last row');
    Check(CompareMem(pointer(mine), @want[(n - 1) * vocab], vocab * SizeOf(single)),
      'the logits unchanged by aNormed');
    ctx := rowsOnly.NewContext;
    rowsOnly.ForwardEmbeds(ctx, pointer(rows), n, nil, false, nil, pointer(normed2));
    FreeAndNil(ctx);
    Check(CompareMem(pointer(normed), pointer(normed2), h * SizeOf(single)),
      'aNormed without logits');
    // no head either: the normed row only, logits refused before the cache
    // changes
    headless := TQwen3Model.CreateStack(sdir, fCpu, 'outer_config', 'outer.model.',
      '', '');
    Check(rowsOnly.HasHead and not headless.HasHead and not headless.HasEmbedding,
      'a stack without a head');
    ctx := headless.NewContext;
    FillCharFast(normed2[0], h * SizeOf(single), 0);
    headless.ForwardEmbeds(ctx, pointer(rows), n, nil, false, nil, pointer(normed2));
    Check(CompareMem(pointer(normed), pointer(normed2), h * SizeOf(single)),
      'the normed row without a head');
    why := '';
    try
      headless.ForwardEmbeds(ctx, pointer(rows), 1, pointer(mine));
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    CheckUtf8(PosEx('no head', why) > 0, 'logits refused: %', [why]);
    CheckEqual(ctx.Length, n, 'the cache unchanged by the refused call');
    FreeAndNil(ctx);
    // refused: a tensor the family has not under the layers, a multimodal
    // RoPE whose sections miss a frequency, a scaled RoPE, a missing object,
    // a missing head, no prefix
    why := StackRefused(sdir, fCpu, 'outer_config', 'broken.model.',
      'broken.model.embed_tokens.weight', 'outer.head.weight');
    CheckUtf8(PosEx('understood', why) > 0, 'unknown layer tensor: %', [why]);
    why := StackRefused(sdir, fCpu, 'bad_mrope', 'outer.model.', '', 'outer.head.weight');
    CheckUtf8(PosEx('mrope_section', why) > 0, 'mrope sections: %', [why]);
    why := StackRefused(sdir, fCpu, 'zero_mrope', 'outer.model.', '', 'outer.head.weight');
    CheckUtf8(PosEx('an mrope_section', why) > 0, 'a section not positive: %', [why]);
    why := StackRefused(sdir, fCpu, 'scaled', 'outer.model.', '', 'outer.head.weight');
    CheckUtf8(PosEx('rope_scaling', why) > 0, 'linear scaling: %', [why]);
    why := StackRefused(sdir, fCpu, 'outer_config.none', 'outer.model.', '',
      'outer.head.weight');
    CheckUtf8(PosEx('no object', why) > 0, 'missing object: %', [why]);
    why := StackRefused(sdir, fCpu, 'outer_config', 'outer.model.', '', 'outer.nohead.weight');
    CheckUtf8(why <> '', 'missing head: %', [why]);
    why := StackRefused(sdir, fCpu, 'outer_config', '', '', 'outer.head.weight');
    CheckUtf8(PosEx('needed', why) > 0, 'no prefix: %', [why]);
    // the folder itself is no qwen3 model
    why := '';
    try
      TQwen3Model.Create(sdir, fCpu).Free;
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    CheckUtf8(PosEx('model_type', why) > 0, 'the checkpoint is no qwen3: %', [why]);
  finally
    ctx.Free;
    headless.Free;
    rowsOnly.Free;
    inner.Free;
    stack.Free;
    plain.Free;
    ref.Free;
  end;
end;

function TtsFolder: TFileName;
begin
  result := FindVendorModel(TTS_MODEL + PathDelim + 'model.safetensors');
  if result <> '' then
    result := ExtractFilePath(result);
end;

function NewTtsTalker(const aFolder: TFileName; aBackend: TInferBackend): TQwen3Model;
begin
  result := TQwen3Model.CreateStack(aFolder, aBackend, 'talker_config',
    'talker.model.', 'talker.model.codec_embedding.weight', 'talker.codec_head.weight');
end;

function NewTtsPredictor(const aFolder: TFileName; aBackend: TInferBackend): TQwen3Model;
begin
  result := TQwen3Model.CreateStack(aFolder, aBackend,
    'talker_config.code_predictor_config', 'talker.code_predictor.model.', '',
    'talker.code_predictor.lm_head.0.weight');
end;

function TtsFilesMatch(const aFolder: TFileName; aRef: TInferRefSet;
  out aWhy: RawUtf8): boolean;
var
  files: PDocVariantData;
  f: PtrInt;
  name, want, got: RawUtf8;
begin
  result := false;
  aWhy := 'no model.files in the manifest';
  if not aRef.Manifest^.GetAsObject('model', files) or
     not files^.GetAsObject('files', files) then
    exit;
  for f := 0 to files^.Count - 1 do
  begin
    name := files^.Names[f];
    want := VariantToUtf8(files^.Values[f]);
    got := HashFileSha256(aFolder + Utf8ToString(name));
    if not IdemPropNameU(got, want) then
    begin
      aWhy := FormatUtf8('% has sha256 %, the manifest %', [name, got, want]);
      exit;
    end;
  end;
  result := files^.Count > 0;
end;

procedure TTestInferQwen3.Qwen3TtsStacks;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  st: TSafeTensorsFile;
  ctx: TQwen3Context;
  logits, head: TSingleDynArray;
  x, want: PSingleArray;
  codes: PIntegerArray;
  n, m, g, l, h, vocab: PtrInt;
  cmp: TInferCompare;
  worst: double;
  why, name: RawUtf8;
  t: PInferTensor;
begin
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3-TTS stacks: 1.8 GB mapped and as much widened need a ' +
    '64-bit process');
  exit;
  {$endif CPU32}
  dir := TtsFolder;
  if dir = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/%', [TTS_MODEL]);
    AddConsole('SKIP Qwen3-TTS stacks: no vendor/models/% (Hugging Face %, ' +
      'revision in tests/data/infer-ref/qwen3-tts-lm)', [TTS_MODEL, TTS_MODEL]);
    exit;
  end;
  refdir := FindInferRefDir('qwen3-tts-lm');
  if CheckFailed(refdir <> '', 'qwen3-tts-lm found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  st := nil;
  ctx := nil;
  try
    // the files of the reference: config.json and model.safetensors by hash
    CheckUtf8(TtsFilesMatch(dir, ref, why), 'the model of the reference: %', [why]);
    // the talker: 28 layers fed with the sums of text and codec embeddings
    model := NewTtsTalker(dir, fFast);
    h := model.Config.Hidden;
    x := ref.F32('talker.embed', m);
    n := m div h;
    fNames := nil;
    fStates := nil;
    SetLength(logits, model.Config.VocabSize);
    ctx := model.NewContext;
    model.ForwardEmbeds(ctx, x, n, pointer(logits), false, KeepState);
    FreeAndNil(ctx);
    worst := 0;
    for l := 0 to model.Config.Layers - 2 do
    begin
      name := FormatUtf8('layer%', [l]);
      if l < 10 then
        name := FormatUtf8('layer0%', [l]);
      want := ref.F32('talker.' + name, m);
      cmp := InferCompare(State(name), want, m, 1e-3, 1e-3);
      CheckUtf8(InferCompareOk(cmp), 'talker layer %: %', [l, InferCompareText(cmp)]);
      if cmp.MaxAbs > worst then
        worst := cmp.MaxAbs;
    end;
    cmp := InferCompare(State('norm'), ref.F32('talker.norm', m), m, 1e-3, 1e-3);
    CheckUtf8(InferCompareOk(cmp), 'talker norm: %', [InferCompareText(cmp)]);
    AddConsole(FormatString('Qwen3-TTS talker: largest state error % (27 layers), ' +
      'norm %', [worst, InferCompareText(cmp)]));
    cmp := InferCompare(pointer(logits), ref.F32('talker.logits_last', m), m, 1e-3, 1e-3);
    CheckUtf8(InferCompareOk(cmp), 'talker logits: %', [InferCompareText(cmp)]);
    AddConsole(FormatString('Qwen3-TTS talker logits: %', [InferCompareText(cmp)]));
    FreeAndNil(model);
    // the code predictor: 16 rows, the logits of code group g from head g
    // applied to the normed state of row g + 1
    model := NewTtsPredictor(dir, fFast);
    Check(not model.HasEmbedding, 'the predictor takes rows only');
    vocab := model.Config.VocabSize;
    x := ref.F32('predictor.embed', m);
    n := m div h;
    fNames := nil;
    fStates := nil;
    ctx := model.NewContext;
    model.ForwardEmbeds(ctx, x, n, nil, false, KeepState);
    FreeAndNil(ctx);
    st := TSafeTensorsFile.Create(dir + 'model.safetensors');
    want := ref.F32('predictor.logits', m);
    codes := ref.I32('predictor.codes', m);
    SetLength(logits, (n - 1) * vocab);
    SetLength(head, vocab * h);
    for g := 0 to n - 2 do
    begin
      t := st.Tensor(FormatUtf8('talker.code_predictor.lm_head.%.weight', [g]));
      for l := 0 to vocab * h - 1 do // BF16 widened exactly
        PCardinal(@head[l])^ := cardinal(PWordArray(t^.Data)^[l]) shl 16;
      fFast.Linear(@State('norm')^[(g + 1) * h], 1, h, InferWeight(pointer(head), idtF32),
        vocab, nil, @logits[g * vocab]);
    end;
    cmp := InferCompare(pointer(logits), want, (n - 1) * vocab, 1e-3, 1e-3);
    CheckUtf8(InferCompareOk(cmp), 'predictor logits: %', [InferCompareText(cmp)]);
    AddConsole(FormatString('Qwen3-TTS code predictor logits (15 groups): %',
      [InferCompareText(cmp)]));
    // its greedy codes: code g + 1 is the highest logit of group g
    m := 0;
    for g := 0 to n - 2 do
      if InferArgMax(@logits[g * vocab], vocab) = codes^[g + 1] then
        inc(m);
    CheckEqual(m, n - 1, 'the greedy codes of the reference');
  finally
    ctx.Free;
    st.Free;
    model.Free;
    ref.Free;
  end;
end;


{ gpt-oss (D5) }

procedure SetFloat(var aDoc: TDocVariantData; const aName: RawUtf8; aValue: double);
var
  v: variant;
begin
  v := aValue;
  aDoc.AddOrUpdateValue(aName, v);
end;

function RotaryMatches(aModel: TDecoderModel; aRef: TInferRefSet;
  const aName: RawUtf8; out aWhy: RawUtf8): boolean;
var
  cs, sn: TSingleDynArray;
  half, n: PtrInt;
  cmp, cmp2: TInferCompare;
begin
  half := aModel.Config.RopeDims shr 1;
  SetLength(cs, 64 * half);
  SetLength(sn, 64 * half);
  aModel.Rotary(0, 64, pointer(cs), pointer(sn));
  cmp := InferCompare(pointer(cs), aRef.F32(aName + '.cos', n), 64 * half, 4e-7, 0);
  cmp2 := InferCompare(pointer(sn), aRef.F32(aName + '.sin', n), 64 * half, 4e-7, 0);
  result := (n = 64 * half) and InferCompareOk(cmp) and InferCompareOk(cmp2);
  aWhy := FormatUtf8('cos %, sin %', [InferCompareText(cmp), InferCompareText(cmp2)]);
end;

procedure TTestInferQwen3.GptOssMatchesReference;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  full, step, ctx: TDecoderContext;
  ids: PIntegerArray;
  n, i, k, v, vocab, diff: PtrInt;
  all, one: TSingleDynArray;
  prompt, mine: TIntegerDynArray;
  same: boolean;
  margin: double;
  detail: RawUtf8;
  s: TDecoderLayerSpecDynArray;
  limit, alpha: single;
  tmp: TFileName;
  cfg: TDocVariantData;
  nullv: variant;
begin
  dir := FindInferRefDir('tiny-gpt-oss');
  if CheckFailed(dir <> '', 'tiny-gpt-oss found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  full := nil;
  step := nil;
  try
    model := NewDecoderModel(dir, fCpu);
    Check(model is TGptOssModel, 'gpt_oss: a TGptOssModel');
    CheckEqual(model.Config.HeadDim, 24, 'head_dim of config.json, not 64 / 4');
    CheckEqual(model.Config.Experts, 8, 'experts');
    CheckEqual(model.Config.ExpertsUsed, 3, 'experts per token');
    CheckEqual(model.Config.ExpertIntermediate, 64, 'expert width');
    Check(model.Config.NormTopK, 'softmax over the chosen logits');
    s := model.Config.Spec;
    Check((s[0].Window = 5) and (s[1].Window = 0) and (s[2].Window = 5) and
      (s[0].Ffn = dfMoe) and (s[1].Ffn = dfMoe) and (s[2].Ffn = dfMoe) and
      not s[0].NoRope and not s[1].NoRope, 'layer kinds');
    limit := 0.8;
    alpha := 1.702;
    Check(model.Config.SwiGluLimit = limit, 'swiglu_limit');
    Check(model.Config.SwiGluAlpha = alpha, 'swiglu_alpha by default');
    Check((model.Config.RopeFactor = 32) and (model.Config.RopeOriginal = 4096) and
      not model.Config.RopeTruncate and (model.Config.RopeTheta = 150000), 'YaRN');
    Check(model.Extras = [dxSinks, dxOutputBias, dxRouterBias, dxExpertBias,
      dxSwiGluLimit], 'every extra of gpt-oss');
    Check((model.Layers[1].Router.DType = idtF32) and
      (model.Layers[1].ExpertGate[5].DType = idtF32), 'F32 weights');
    // YaRN: the rotary factors, the attention factor 0.1 ln 32 + 1 included
    CheckUtf8(RotaryMatches(model, ref, 'rope', detail), 'YaRN factors: %', [detail]);
    // every state of 5, 13 and 29 tokens (the window of 5 slides)
    for k := 0 to 2 do
      CheckStates(model, ref, FormatUtf8('case%', [k]), 5e-6, 2e-5, true);
    // one token at a time = the prefill, bit for bit; two chunks alike
    vocab := model.Config.VocabSize;
    ids := ref.I32('case2.ids', n);
    full := model.NewContext;
    step := model.NewContext;
    SetLength(all, n * vocab);
    SetLength(one, vocab);
    model.Forward(full, ids, n, pointer(all), true);
    same := true;
    for i := 0 to n - 1 do
    begin
      model.Forward(step, @ids^[i], 1, pointer(one));
      for v := 0 to vocab - 1 do
        if PCardinal(@one[v])^ <> PCardinal(@all[i * vocab + v])^ then
          same := false;
    end;
    Check(same, 'cached logits bit-identical to the full pass');
    step.Reset;
    model.Forward(step, ids, 7, nil);
    model.Forward(step, @ids^[7], n - 7, pointer(one));
    Check(CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single)),
      'prefill in two chunks');
    // greedy continuations of 24 tokens, through the sliding cache
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        mine := model.Greedy(ctx, prompt, 24, []);
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, FormatUtf8('case%', [k]), margin, detail);
      CheckUtf8(diff < 0, 'gpt-oss case% greedy differs at step % (margin %, %)',
        [k, diff, margin, detail]);
    end;
    // experts of width 16 for a hidden width of 64: a row of down_proj is
    // longer than one of gate_up_proj (the buffer of LoadExperts)
    FreeAndNil(step);
    FreeAndNil(full);
    FreeAndNil(model);
    FreeAndNil(ref);
    dir := MakePath([dir, 'narrow'], true);
    ref := TInferRefSet.CreateFromFolder(dir);
    model := NewDecoderModel(dir, fCpu);
    CheckEqual(model.Config.ExpertIntermediate, 16, 'narrow experts');
    for k := 0 to 2 do
      CheckStates(model, ref, FormatUtf8('case%', [k]), 5e-6, 2e-5, true);
    // head_dim null: hidden / heads = 64 / 4, the 16 of this model
    FreeAndNil(model);
    tmp := MakePath([GetSystemPath(spTemp), 'lai-d5n-' + Int64ToUtf8(GetTickCount64)], true);
    ForceDirectories(tmp);
    try
      cfg.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
      SetVariantNull(nullv);
      cfg.AddOrUpdateValue('head_dim', nullv);
      FileFromString(cfg.ToJson, tmp + 'config.json');
      CopyFile(dir + 'model.safetensors', tmp + 'model.safetensors', false);
      model := NewDecoderModel(tmp, fCpu);
      CheckEqual(model.Config.HeadDim, 16, 'head_dim null: hidden / heads');
      CheckStates(model, ref, 'case2', 5e-6, 2e-5, true);
    finally
      FreeAndNil(model);
      DirectoryDeleteAll(tmp);
    end;
  finally
    step.Free;
    full.Free;
    model.Free;
    ref.Free;
  end;
end;

/// the factors of the second rotary set of aModel against aName.cos / .sin
function Rotary2Matches(aModel: TDecoderModel; aRef: TInferRefSet;
  const aName: RawUtf8; out aWhy: RawUtf8): boolean;
var
  cs, sn: TSingleDynArray;
  half, n: PtrInt;
  cmp, cmp2: TInferCompare;
begin
  half := aModel.Rope2HeadDim shr 1;
  SetLength(cs, 64 * half);
  SetLength(sn, 64 * half);
  aModel.Rotary2(0, 64, pointer(cs), pointer(sn));
  cmp := InferCompare(pointer(cs), aRef.F32(aName + '.cos', n), 64 * half, 4e-7, 0);
  cmp2 := InferCompare(pointer(sn), aRef.F32(aName + '.sin', n), 64 * half, 4e-7, 0);
  result := (n = 64 * half) and InferCompareOk(cmp) and InferCompareOk(cmp2);
  aWhy := FormatUtf8('cos %, sin %', [InferCompareText(cmp), InferCompareText(cmp2)]);
end;

procedure TTestInferQwen3.Gemma4MatchesReference;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  full, step, ctx: TDecoderContext;
  ids: PIntegerArray;
  n, i, k, v, vocab, diff: PtrInt;
  all, one, rows, viaRows: TSingleDynArray;
  prompt, mine: TIntegerDynArray;
  same: boolean;
  margin: double;
  detail: RawUtf8;
  s: TDecoderLayerSpecDynArray;
  keep: single;
  zero: boolean;
begin
  dir := FindInferRefDir('tiny-gemma4');
  if CheckFailed(dir <> '', 'tiny-gemma4 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  full := nil;
  step := nil;
  try
    model := NewDecoderModel(dir, fCpu);
    Check(model is TGemma4Model, 'gemma4: a TGemma4Model');
    s := model.Config.Spec;
    Check((s[0].Window = 5) and (s[1].Window = 0) and (s[2].Window = 5) and
      (s[3].Window = 0), 'sliding and full layers');
    Check((s[0].HeadDim = 16) and (s[0].KvHeads = 2) and not s[0].KIsV and
      not s[0].Rope2 and (s[1].HeadDim = 32) and (s[1].KvHeads = 1) and s[1].KIsV and
      s[1].Rope2 and (s[3].HeadDim = 32), 'the heads of both kinds');
    Check((s[0].Ffn = dfDenseMoe) and (s[3].Ffn = dfDenseMoe), 'dense and experts');
    CheckEqual(model.Rope2HeadDim, 32, 'the head of the second rotary set');
    Check((model.Config.Experts = 8) and (model.Config.ExpertsUsed = 3) and
      (model.Config.ExpertIntermediate = 32) and model.Config.NormTopK, 'experts');
    Check((model.Config.AttentionScale = 1) and (model.Config.EmbedScale = 8) and
      (model.Config.LogitSoftcap = 3) and (model.Config.Activation = daGeluTanh) and
      model.Config.SandwichNorm and model.Config.ValueNorm, 'the Gemma settings');
    Check((model.Config.RopeTheta = 10000) and (model.Config.Rope2Theta = 1000000) and
      (model.Config.Rope2Partial = 0.25), 'the rotary sets');
    Check(model.Extras = [dxLayerShapes, dxAttentionScale, dxGeluTanh, dxSandwichNorm,
      dxValueNorm, dxSharedKv, dxEmbedScale, dxLogitSoftcap, dxLayerScalar,
      dxSecondRope, dxDenseMoe], 'every extra of Gemma 4');
    Check((model.Layers[1].Scalar <> nil) and (model.Layers[1].Scalar^[0] <> 1),
      'a layer scalar');
    CheckUtf8(RotaryMatches(model, ref, 'rope', detail), 'sliding rope: %', [detail]);
    CheckUtf8(Rotary2Matches(model, ref, 'rope2', detail), 'proportional rope: %',
      [detail]);
    // every state of 5, 13 and 29 tokens (the window of 5 slides)
    for k := 0 to 2 do
      CheckStates(model, ref, FormatUtf8('case%', [k]), 1e-4, 1e-4, true);
    // one token at a time = the prefill, bit for bit; two chunks alike
    vocab := model.Config.VocabSize;
    ids := ref.I32('case2.ids', n);
    full := model.NewContext;
    step := model.NewContext;
    SetLength(all, n * vocab);
    SetLength(one, vocab);
    model.Forward(full, ids, n, pointer(all), true);
    same := true;
    for i := 0 to n - 1 do
    begin
      model.Forward(step, @ids^[i], 1, pointer(one));
      for v := 0 to vocab - 1 do
        if PCardinal(@one[v])^ <> PCardinal(@all[i * vocab + v])^ then
          same := false;
    end;
    Check(same, 'cached logits bit-identical to the full pass');
    step.Reset;
    model.Forward(step, ids, 7, nil);
    model.Forward(step, @ids^[7], n - 7, pointer(one));
    Check(CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single)),
      'prefill in two chunks');
    // the rows of Embed (scaled) through ForwardEmbeds: the logits of the ids
    SetLength(rows, n * model.Config.Hidden);
    SetLength(viaRows, vocab);
    model.Embed(ids, n, pointer(rows));
    step.Reset;
    model.ForwardEmbeds(step, pointer(rows), n, pointer(viaRows));
    Check(CompareMem(pointer(viaRows), @all[(n - 1) * vocab], vocab * SizeOf(single)),
      'embedding rows give the logits of their ids');
    // greedy continuations of 24 tokens, through the sliding cache
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        mine := model.Greedy(ctx, prompt, 24, []);
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, FormatUtf8('case%', [k]), margin, detail);
      CheckUtf8(diff < 0, 'Gemma 4 case% greedy differs at step % (margin %, %)',
        [k, diff, margin, detail]);
    end;
    // a layer scalar of 0 multiplies too: the state after that layer is 0
    // (BF16 weights: the scalar is widened, so it can be changed here)
    keep := model.Layers[1].Scalar^[0];
    model.Layers[1].Scalar^[0] := 0;
    try
      fNames := nil;
      fStates := nil;
      ids := ref.I32('case0.ids', n);
      ctx := model.NewContext;
      try
        model.Forward(ctx, ids, n, nil, false, KeepState);
      finally
        ctx.Free;
      end;
      zero := State('layer01') <> nil;
      if zero then
        for i := 0 to n * model.Config.Hidden - 1 do
          if State('layer01')^[i] <> 0 then
            zero := false;
      Check(zero, 'a layer scalar of 0');
    finally
      model.Layers[1].Scalar^[0] := keep;
    end;
    // the dense model: layer types and rope from the defaults, V of its own
    FreeAndNil(step);
    FreeAndNil(full);
    FreeAndNil(model);
    FreeAndNil(ref);
    dir := MakePath([dir, 'dense'], true);
    ref := TInferRefSet.CreateFromFolder(dir);
    model := NewDecoderModel(dir, fCpu);
    s := model.Config.Spec;
    Check((length(s) = 6) and (s[4].Window = 5) and (s[5].Window = 0) and
      (s[5].HeadDim = 32) and (s[5].KvHeads = 2) and not s[5].KIsV and s[5].Rope2 and
      (s[0].Ffn = dfDense), 'five sliding layers, then a full one');
    Check((model.Config.RopeTheta = 10000) and (model.Config.Rope2Theta = 1000000) and
      (model.Config.Rope2Partial = 0.25), 'the default rotary sets');
    CheckUtf8(RotaryMatches(model, ref, 'rope', detail), 'dense rope: %', [detail]);
    CheckUtf8(Rotary2Matches(model, ref, 'rope2', detail), 'dense rope2: %', [detail]);
    for k := 0 to 2 do
      CheckStates(model, ref, FormatUtf8('case%', [k]), 1e-4, 1e-4, true);
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        mine := model.Greedy(ctx, prompt, 24, []);
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, FormatUtf8('case%', [k]), margin, detail);
      CheckUtf8(diff < 0, 'dense Gemma 4 case% greedy differs at step % (margin %, %)',
        [k, diff, margin, detail]);
    end;
  finally
    step.Free;
    full.Free;
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.Gemma4Quantized;
var
  dir: TFileName;
  ref: TInferRefSet;
  q8, q4, f: TDecoderModel;
  tape: TDecoderRouteTape;
  lq, lf: TSingleDynArray;
  ids: PIntegerArray;
  n, m, k, v, w, e, r, nb, size: PtrInt;
  cmp: TInferCompare;
  q: TDecoderModel;
  sliced: boolean;
  blocks: array[0..63] of TInferBlockQ8_0; // a row of up to 2048 values
  blocks4: array[0..63] of TInferBlockQ4_0;

  /// every row of a quantized expert matrix is the quantization of the same
  // row of the F32 model: the slice of the 3-D tensor is the right one
  function SameRows(const aQ, aF: TInferWeight; aRows, aCols: PtrInt): boolean;
  var
    i: PtrInt;
  begin
    result := false;
    nb := aCols shr 5;
    size := nb * INFER_DTYPE_SIZE[aQ.DType];
    for i := 0 to aRows - 1 do
    begin
      if aQ.DType = idtQ8_0 then
      begin
        InferQuantizeRowQ8_0Ref(@PSingleArray(aF.Data)^[i * aCols], aCols, @blocks);
        if not CompareMem(@blocks, @PByteArray(aQ.Data)^[i * size], size) then
          exit;
      end
      else
      begin
        InferQuantizeRowQ4_0(@PSingleArray(aF.Data)^[i * aCols], aCols, @blocks4);
        if not CompareMem(@blocks4, @PByteArray(aQ.Data)^[i * size], size) then
          exit;
      end;
    end;
    result := true;
  end;

  procedure Run(aModel: TDecoderModel; aMode: TDecoderRouteMode; aLogits: PSingleArray);
  var
    c: TDecoderContext;
  begin
    // the same experts in both runs: the quantized one replays the choice of
    // the F32 one, so only the error of the quantization is measured (the
    // router of the tiny model is sharp: a near tie flips with Q8_0)
    tape.Mode := aMode;
    aModel.RouteTape := tape;
    c := aModel.NewContext;
    try
      aModel.Forward(c, ids, n, aLogits, true);
    finally
      c.Free;
      aModel.RouteTape := nil;
    end;
  end;

begin
  dir := FindInferRefDir('tiny-gemma4');
  if CheckFailed(dir <> '', 'tiny-gemma4 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  q8 := nil;
  q4 := nil;
  f := nil;
  tape := TDecoderRouteTape.Create;
  try
    // F32 on the fast backend; Q8_0 and Q4_0 at load, the experts sliced from
    // the 3-D tensors, the router F32, the tied head Q8_0
    f := TGemma4Model.Create(dir, fFast);
    q8 := TGemma4Model.Create(dir, fFast, idtQ8_0);
    q4 := TGemma4Model.Create(dir, fFast, idtQ4_0);
    Check((q8.Layers[1].Router.DType = idtF32) and
      (q8.Layers[1].ExpertUp[7].DType = idtQ8_0) and (q8.Head.DType = idtQ8_0) and
      (q4.Layers[1].ExpertDown[7].DType = idtQ4_0) and (q4.Head.DType = idtQ8_0),
      'the types of the quantized weights');
    // every expert of every layer: gate, up and down sliced at their rows
    sliced := true;
    for w := 0 to 1 do
    begin
      if w = 0 then
        q := q8
      else
        q := q4;
      for r := 0 to f.Config.Layers - 1 do
        for e := 0 to f.Config.Experts - 1 do
          if not SameRows(q.Layers[r].ExpertGate[e], f.Layers[r].ExpertGate[e], 32, 64) or
             not SameRows(q.Layers[r].ExpertUp[e], f.Layers[r].ExpertUp[e], 32, 64) or
             not SameRows(q.Layers[r].ExpertDown[e], f.Layers[r].ExpertDown[e], 64, 32) then
            sliced := false;
    end;
    Check(sliced, 'every quantized expert row is that of the F32 expert');
    v := f.Config.VocabSize;
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(lq, n * v);
      SetLength(lf, n * v);
      Run(f, drmOff, pointer(lf));
      cmp := InferCompare(pointer(lf), ref.F32(FormatUtf8('case%.logits', [k]), m),
        n * v, 1e-4, 1e-4);
      CheckUtf8(InferCompareOk(cmp), 'Gemma 4 case % F32 fast path: %',
        [k, InferCompareText(cmp)]);
      for w := 0 to 1 do
      begin
        if w = 0 then
          q := q8
        else
          q := q4;
        tape.Clear;
        Run(f, drmRecord, pointer(lf));
        Run(q, drmReplay, pointer(lq));
        CheckEqual(tape.Pending, 0, 'every choice replayed');
        // the tiny model is chaotic (scores of normed heads at a factor of 1:
        // a sharp softmax), its error grows with the tokens - measured
        // 2026-10-20 with the same experts: Q8_0 max 0.64, cosine 0.989
        // (case 2); Q4_0 max 2.1, cosine 0.705 (case 1)
        cmp := InferCompare(pointer(lq), pointer(lf), n * v, 0.8 + 1.7 * w, 0);
        CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.985 - w * 0.305),
          'Gemma 4 case % % vs FP32: %', [k, INFER_DTYPE_NAME[q.Weights],
          InferCompareText(cmp)]);
        AddConsole(FormatString('Gemma 4 case % % vs FP32 (the same experts): %',
          [k, INFER_DTYPE_NAME[q.Weights], InferCompareText(cmp)]));
      end;
    end;
  finally
    tape.Free;
    f.Free;
    q4.Free;
    q8.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.Gemma4Refused;
var
  dir, tmp: TFileName;
  cfg, c: TDocVariantData;
  ref: TInferRefSet;
  model: TDecoderModel;
  g1, g2, u1: TSingleDynArray;
  i: PtrInt;
  one, up: single;
  nullv: variant;

  function Loads(const aPath: RawUtf8; const aValue: variant): boolean;
  var
    l: TDocVariantData;
  begin
    l.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    l.SetValueByPath(aPath, aValue, true, '.');
    FileFromString(l.ToJson, tmp + 'config.json');
    result := true;
    try
      TGemma4Model.Create(tmp, fCpu).Free;
    except
      on E: EInferError do
      begin
        AddConsole(FormatString('% = %: "%"', [aPath, aValue, E.Message]));
        result := false;
      end;
    end;
  end;

  function Refused(const aPath: RawUtf8; const aValue: variant;
    const aWhy: RawUtf8): boolean;
  var
    c: TDocVariantData;
    why: RawUtf8;
  begin
    c.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT); // nested objects of its own
    c.SetValueByPath(aPath, aValue, true, '.');
    FileFromString(c.ToJson, tmp + 'config.json');
    why := '';
    try
      TGemma4Model.Create(tmp, fCpu).Free;
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    result := PosEx(aWhy, why) > 0;
    if not result then
      AddConsole(FormatString('% = %: "%"', [aPath, aValue, why]));
  end;

begin
  dir := FindInferRefDir('tiny-gemma4');
  if CheckFailed(dir <> '', 'tiny-gemma4 found') then
    exit;
  tmp := MakePath([GetSystemPath(spTemp), 'lai-d6-' + Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  try
    cfg.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    CopyFile(dir + 'model.safetensors', tmp + 'model.safetensors', false);
    // the copy itself loads
    FileFromString(cfg.ToJson, tmp + 'config.json');
    TGemma4Model.Create(tmp, fCpu).Free;
    Check(Refused('text_config.hidden_size_per_layer_input', 256,
      'per-layer embeddings'), 'per-layer embeddings');
    Check(Refused('text_config.num_kv_shared_layers', 2, 'shared KV'), 'shared KV');
    Check(Refused('text_config.attention_bias', true, 'attention_bias'), 'bias');
    Check(Refused('text_config.use_bidirectional_attention', 'all', 'bidirectional'),
      'bidirectional attention');
    Check(Refused('tie_word_embeddings', false, 'untied'), 'an untied head');
    Check(Refused('text_config.tie_word_embeddings', false, 'untied'),
      'an untied text head');
    Check(Refused('text_config.hidden_activation', 'silu', 'hidden_activation'),
      'another activation');
    Check(Refused('text_config.rope_parameters.full_attention.rope_type', 'yarn',
      'rope_type'), 'another full rope');
    Check(Refused('text_config.rope_parameters.sliding_attention.partial_rotary_factor',
      0.5, 'sliding rope_type'), 'a partial sliding rope');
    Check(Refused('text_config.rope_parameters.full_attention.factor', 2,
      'rope factor'), 'a scaled rope');
    Check(Refused('text_config.rope_parameters.full_attention.mscale', 1,
      'not supported'), 'an unknown rope key');
    Check(Refused('text_config.layer_types', _Arr(['sliding_attention',
      'chunked_attention', 'sliding_attention', 'full_attention']), 'layer type'),
      'an unknown layer type');
    Check(Refused('text_config.num_experts', 0, 'experts'), 'experts missing');
    Check(Refused('text_config.global_head_dim', 64, 'shaped'),
      'a global head of another width than the weights');
    Check(Refused('text_config.num_global_key_value_heads', 3, 'heads'),
      'global KV heads that do not divide the heads');
    // K of its own: the 2 KV heads of the config, the weights have 1
    Check(Refused('text_config.attention_k_eq_v', false, 'k_proj'),
      'K = V turned off');
    Check(Refused('model_type', 'gemma3', 'model_type'), 'another model_type');
    Check(Refused('text_config.per_layer_config', _ObjFast(['7', _ObjFast(['head_dim',
      32])]), 'names no layer'), 'per_layer_config of a layer that is not there');
    Check(Refused('text_config.per_layer_config', _ObjFast(['1', _ObjFast(['sliding_window',
      8])]), 'not supported'), 'per_layer_config of another setting');
    // a null rope entry: transformers builds no rotary set and fails later
    SetVariantNull(nullv);
    Check(Refused('text_config.rope_parameters.sliding_attention', nullv,
      'no object'), 'a null rope entry');
    // per_layer_config null: the global keys are off all the same - every
    // layer keeps head_dim, so the full layers no longer match the weights
    Check(Refused('text_config.per_layer_config', nullv, 'shaped'),
      'per_layer_config null');
    // accepted: per_layer_config by layer type in place of the global keys,
    // a last sliding layer made full
    Check(Loads('text_config.per_layer_config', _ObjFast(['full_attention',
      _ObjFast(['head_dim', 32, 'num_key_value_heads', 1])])),
      'per_layer_config by layer type');
    ref := TInferRefSet.CreateFromFolder(dir);
    try
      c.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
      c.SetValueByPath('text_config.layer_types', _Arr(['sliding_attention',
        'full_attention', 'sliding_attention', 'sliding_attention']), true, '.');
      FileFromString(c.ToJson, tmp + 'config.json');
      model := TGemma4Model.Create(tmp, fCpu);
      try
        Check((model.Config.Spec[3].Window = 0) and model.Config.Spec[3].Rope2,
          'the last layer full whatever layer_types says');
        CheckStates(model, ref, 'case0', 1e-4, 1e-4, true);
      finally
        model.Free;
      end;
    finally
      ref.Free;
    end;
    // GeluTanhGlu: the fast backend in parallel slices = the reference
    SetLength(g1, 5000);
    SetLength(u1, 5000);
    for i := 0 to high(g1) do
    begin
      g1[i] := (i mod 997) / 83 - 6;
      u1[i] := 1 + (i mod 13) / 7;
    end;
    g2 := copy(g1);
    fCpu.GeluTanhGlu(pointer(g1), pointer(u1), length(g1));
    fFast.GeluTanhGlu(pointer(g2), pointer(u1), length(g2));
    Check(CompareMem(pointer(g1), pointer(g2), length(g1) * SizeOf(single)),
      'GeluTanhGlu on all cores');
    one := 1;
    up := 1;
    fCpu.GeluTanhGlu(@one, @up, 1);
    CheckSame(one, 0.8411919906, 1e-6, 'gelu_tanh(1)');
  finally
    DirectoryDeleteAll(tmp);
    RemoveDir(tmp);
  end;
end;

procedure TTestInferQwen3.GptOssQuantized;
var
  dir: TFileName;
  ref: TInferRefSet;
  a, c, f: TDecoderModel;
  ca, cc, cf: TDecoderContext;
  la, lc, lf: TSingleDynArray;
  ids: PIntegerArray;
  n, m, k, v: PtrInt;
  cmp: TInferCompare;
begin
  dir := FindInferRefDir('tiny-gpt-oss');
  if CheckFailed(dir <> '', 'tiny-gpt-oss found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  a := nil;
  c := nil;
  f := nil;
  try
    // F32 on the fast backend: float lanes, sinks in its attention too
    f := TGptOssModel.Create(dir, fFast);
    // Q8_0 at load: the experts reordered, then quantized; the router F32
    a := TGptOssModel.Create(dir, fCpu, idtQ8_0);
    c := TGptOssModel.Create(dir, fFast, idtQ8_0);
    Check(a.Layers[0].Router.DType = idtF32, 'router in F32');
    Check((a.Layers[2].ExpertGate[7].DType = idtQ8_0) and
      (a.Layers[2].ExpertDown[7].DType = idtQ8_0), 'experts in Q8_0');
    v := a.Config.VocabSize;
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(la, n * v);
      SetLength(lc, n * v);
      SetLength(lf, n * v);
      ca := a.NewContext;
      cc := c.NewContext;
      cf := f.NewContext;
      try
        a.Forward(ca, ids, n, pointer(la), true);
        c.Forward(cc, ids, n, pointer(lc), true);
        f.Forward(cf, ids, n, pointer(lf), true);
      finally
        ca.Free;
        cc.Free;
        cf.Free;
      end;
      cmp := InferCompare(pointer(lf), ref.F32(FormatUtf8('case%.logits', [k]), m),
        n * v, 1e-4, 1e-4);
      CheckUtf8(InferCompareOk(cmp), 'gpt-oss case % F32 fast path: %',
        [k, InferCompareText(cmp)]);
      cmp := InferCompare(pointer(lc), pointer(la), n * v, 2e-2, 1e-2);
      CheckUtf8(InferCompareOk(cmp), 'gpt-oss case % Q8_0 float path: %',
        [k, InferCompareText(cmp)]);
      // the clamp (limit 0.8) and the top-3 router are sharp in this tiny
      // model: Q8_0 moves a logit by up to 0.49 - measured 2026-10-19 max
      // 0.49, cosine 0.9968 (case 1)
      cmp := InferCompare(pointer(la), ref.F32(FormatUtf8('case%.logits', [k]), m),
        n * v, 0.6, 0.3);
      CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.995),
        'gpt-oss case % Q8_0 vs FP32: %', [k, InferCompareText(cmp)]);
      AddConsole(FormatString('gpt-oss case % Q8_0 vs FP32: %',
        [k, InferCompareText(cmp)]));
    end;
  finally
    f.Free;
    c.Free;
    a.Free;
    ref.Free;
  end;
end;

/// true if creating a TGptOssModel over aDir raises an EInferError with aReason
function GptOssRefused(const aDir: TFileName; aBackend: TInferBackend;
  const aReason: RawUtf8; out aWhy: RawUtf8): boolean;
var
  m: TDecoderModel;
begin
  result := false;
  aWhy := 'accepted';
  try
    m := TGptOssModel.Create(aDir, aBackend);
    m.Free;
  except
    on E: Exception do
    begin
      aWhy := StringToUtf8(E.Message);
      result := (E is EInferError) and (PosEx(aReason, aWhy) > 0);
    end;
  end;
end;

/// the byte offset in a safetensors file of the data of tensor aName
function TensorOffset(const aContent: RawByteString; const aName: RawUtf8): PtrInt;
var
  len: Int64;
  doc: TDocVariantData;
  json: RawUtf8;
begin
  MoveFast(pointer(aContent)^, len, 8);
  FastSetString(json, @PByteArray(aContent)^[8], len);
  doc.InitJson(json, JSON_FAST);
  result := 8 + len + VariantToInt64Def(doc.O[aName]^.A['data_offsets']^.Values[0], -1);
end;

procedure TTestInferQwen3.GptOssMxfp4;
const
  SHARD1 = 'model-00001-of-00002.safetensors';
  SHARD2 = 'model-00002-of-00002.safetensors';
var
  dir, tmp: TFileName;
  ref: TInferRefSet;
  m, q, f: TDecoderModel;
  cm, cf: TDecoderContext;
  l, e, r, n, k, h, fi, v: PtrInt;
  row, lm, lf: TSingleDynArray;
  want: PSingleArray;
  ids: PIntegerArray;
  ok: boolean;
  cmp: TInferCompare;
  doc: TDocVariantData;
  content: RawByteString;
  why: RawUtf8;
  nullv: variant;

  procedure SameRows(const aWeight: TInferWeight; const aName: RawUtf8;
    aRows, aCols: PtrInt);
  var
    i, c: PtrInt;
  begin
    want := ref.F32(aName, c);
    CheckEqual(c, aRows * aCols, aName);
    SetLength(row, aCols);
    ok := true;
    for i := 0 to aRows - 1 do
    begin
      InferRowToF32(aWeight.Data, aWeight.DType, i, aCols, pointer(row));
      for c := 0 to aCols - 1 do
        if row[c] <> want^[i * aCols + c] then
          ok := false;
    end;
    CheckUtf8(ok, '% repacked = transformers', [aName]);
  end;

  procedure CopyFolder;
  begin
    tmp := MakePath([GetSystemPath(spTemp), 'lai-d5b-' +
      Int64ToUtf8(GetTickCount64)], true);
    ForceDirectories(tmp);
    CopyFile(dir + 'config.json', tmp + 'config.json', false);
    CopyFile(dir + 'model.safetensors.index.json',
      tmp + 'model.safetensors.index.json', false);
    CopyFile(dir + SHARD1, tmp + SHARD1, false);
    CopyFile(dir + SHARD2, tmp + SHARD2, false);
  end;

begin
  dir := FindInferRefDir('tiny-gpt-oss');
  if CheckFailed(dir <> '', 'tiny-gpt-oss found') then
    exit;
  dir := MakePath([dir, 'mxfp4'], true);
  ref := TInferRefSet.CreateFromFolder(dir);
  m := nil;
  q := nil;
  f := nil;
  tmp := '';
  try
    // no model.safetensors: the two files of the index
    m := NewDecoderModel(dir, fCpu);
    Check(m is TGptOssModel, 'gpt_oss: a TGptOssModel');
    h := m.Config.Hidden;
    fi := m.Config.ExpertIntermediate;
    ok := true;
    for l := 0 to m.Config.Layers - 1 do
      for e := 0 to m.Config.Experts - 1 do
        if (m.Layers[l].ExpertGate[e].DType <> idtMXFP4) or
           (m.Layers[l].ExpertUp[e].DType <> idtMXFP4) or
           (m.Layers[l].ExpertDown[e].DType <> idtMXFP4) then
          ok := false;
    Check(ok, 'every expert in MXFP4');
    Check((m.Layers[0].Q.DType = idtF32) and (m.Layers[0].Router.DType = idtF32),
      'the other matrices as asked (F32)');
    // the repacking: gate and up from the interleaved rows, the nibble pairs,
    // the scales - expert 3 of every layer as transformers dequantized it
    for l := 0 to m.Config.Layers - 1 do
    begin
      SameRows(m.Layers[l].ExpertGate[3], FormatUtf8('experts.l%.e3.gate', [l]), fi, h);
      SameRows(m.Layers[l].ExpertUp[3], FormatUtf8('experts.l%.e3.up', [l]), fi, h);
      SameRows(m.Layers[l].ExpertDown[3], FormatUtf8('experts.l%.e3.down', [l]), h, fi);
    end;
    // layer 0 expert 1: a zero block with the exponent 0 (gate row 0, block
    // 0) and one of -0 values with the exponent 255 (up row 0, block 1) -
    // accepted, both zero
    SetLength(row, h);
    InferRowToF32(m.Layers[0].ExpertGate[1].Data, idtMXFP4, 0, h, pointer(row));
    ok := true;
    for k := 0 to 31 do
      if row[k] <> 0 then
        ok := false;
    InferRowToF32(m.Layers[0].ExpertUp[1].Data, idtMXFP4, 0, h, pointer(row));
    for k := 32 to 63 do
      if row[k] <> 0 then
        ok := false;
    Check(ok, 'zero blocks outside the exact range accepted as zero');
    // and they really are outside it: the exponents as the generator wrote them
    Check((PInferBlockMxfp4Array(m.Layers[0].ExpertGate[1].Data)^[0].e = 0) and
      (PInferBlockMxfp4Array(m.Layers[0].ExpertUp[1].Data)^[1].e = 255),
      'the zero blocks keep the exponents 0 and 255');
    // every state: the experts see Q8_0 activations, the rest is exact -
    // measured 2026-10-20: logits within 0.0098, cosine 0.999992
    for k := 0 to 2 do
      CheckStates(m, ref, FormatUtf8('case%', [k]), 0.05, 0.02, true);
    // the fast backend: float lanes over the same MXFP4 tiles
    f := TGptOssModel.Create(dir, fFast);
    v := m.Config.VocabSize;
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(lm, n * v);
      SetLength(lf, n * v);
      cm := m.NewContext;
      cf := f.NewContext;
      try
        m.Forward(cm, ids, n, pointer(lm), true);
        f.Forward(cf, ids, n, pointer(lf), true);
      finally
        cm.Free;
        cf.Free;
      end;
      cmp := InferCompare(pointer(lf), pointer(lm), n * v, 1e-3, 1e-3);
      CheckUtf8(InferCompareOk(cmp), 'gpt-oss MXFP4 case % fast path: %',
        [k, InferCompareText(cmp)]);
    end;
    // Q8_0 asked for: the other matrices in Q8_0, the experts stay MXFP4
    q := TGptOssModel.Create(dir, fCpu, idtQ8_0);
    Check((q.Layers[1].Q.DType = idtQ8_0) and (q.Layers[1].Router.DType = idtF32) and
      (q.Layers[1].ExpertDown[5].DType = idtMXFP4), 'Q8_0 around MXFP4 experts');
    FreeAndNil(q);
    FreeAndNil(f);
    FreeAndNil(m);
    // refusals, on a copy
    CopyFolder;
    FileFromString(StringReplaceAll(StringFromFile(dir + 'config.json'),
      '"mxfp4"', '"fp8"'), tmp + 'config.json');
    CheckUtf8(GptOssRefused(tmp, fCpu, 'quantization "fp8"', why),
      'another quantization: %', [why]);
    doc.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    SetVariantNull(nullv);
    doc.AddOrUpdateValue('quantization_config', nullv);
    FileFromString(doc.ToJson, tmp + 'config.json');
    CheckUtf8(GptOssRefused(tmp, fCpu, 'no object', why),
      'quantization_config null: %', [why]);
    // MXFP4 blocks without the quantization_config: unknown tensors
    doc.Clear;
    doc.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    doc.Delete('quantization_config');
    FileFromString(doc.ToJson, tmp + 'config.json');
    CheckUtf8(GptOssRefused(tmp, fCpu, 'gate_up_proj', why),
      'blocks without quantization_config: %', [why]);
    CopyFile(dir + 'config.json', tmp + 'config.json', false);
    // a block with values and an exponent beyond the exact range (expert 0,
    // gate row 0, block 0 of layer 0)
    content := StringFromFile(tmp + SHARD1);
    n := TensorOffset(content, 'model.layers.0.mlp.experts.gate_up_proj_scales');
    PByteArray(content)^[n] := 150;
    FileFromString(content, tmp + SHARD1);
    CheckUtf8(GptOssRefused(tmp, fCpu, 'beyond the range', why),
      'exponent 150 with values: %', [why]);
    PByteArray(content)^[n] := 100;
    FileFromString(content, tmp + SHARD1);
    CheckUtf8(GptOssRefused(tmp, fCpu, 'beyond the range', why),
      'exponent 100 with values: %', [why]);
    // the index is needed: without it, no model.safetensors
    CopyFile(dir + SHARD1, tmp + SHARD1, false);
    DeleteFile(tmp + 'model.safetensors.index.json');
    CheckUtf8(GptOssRefused(tmp, fCpu, 'missing or empty file', why) and
      (PosEx('model.safetensors', why) > 0),
      'no index, no single file: %', [why]);
  finally
    f.Free;
    q.Free;
    m.Free;
    ref.Free;
    if tmp <> '' then
      DirectoryDeleteAll(tmp);
  end;
end;

/// two digits, as the state names of the decoder
function Pad0(i: PtrInt): RawUtf8;
begin
  result := UInt32ToUtf8(i);
  if i < 10 then
    result := '0' + result;
end;

procedure TTestInferDecoder.CheckLastRows(aModel: TDecoderModel; aRef: TInferRefSet;
  const aLabel: RawUtf8; aStateCos, aCos, aMaxAbs: double; aCases: PtrInt);
var
  ctx: TDecoderContext;
  ids, want: PIntegerArray;
  margins, last: PSingleArray;
  k, n, m, h, s, diff: PtrInt;
  prompt, mine: TIntegerDynArray;
  logits: TSingleDynArray;
  cmp: TInferCompare;
  margin, worst: double;
  name: RawUtf8;
  timer: TPrecisionTimer;
begin
  h := aModel.Config.Hidden;
  for k := 0 to aCases - 1 do
  begin
    ids := aRef.I32(FormatUtf8('case%.ids', [k]), n);
    SetLength(prompt, n);
    MoveFast(ids^, prompt[0], n * SizeOf(integer));
    SetLength(logits, aModel.Config.VocabSize);
    fNames := nil;
    fStates := nil;
    ctx := aModel.NewContext;
    try
      timer.Start;
      aModel.Forward(ctx, pointer(prompt), n, pointer(logits), false, KeepState);
      AddConsole(FormatString('% case % prefill of % tokens: %',
        [aLabel, k, n, timer.Stop]));
    finally
      ctx.Free;
    end;
    // the last row of every state
    worst := 1;
    for s := 0 to aModel.Config.Layers + 1 do
    begin
      if s = 0 then
        name := 'embed'
      else if s <= aModel.Config.Layers then
        name := FormatUtf8('layer%', [Pad0(s - 1)])
      else
        name := 'norm';
      last := aRef.F32(FormatUtf8('case%.last.%', [k, name]), m);
      if CheckFailed(State(name) <> nil, Utf8ToString('state ' + name)) then
        continue;
      cmp := InferCompare(@State(name)^[(n - 1) * h], last, h, 1, 1);
      if cmp.Cosine < worst then
        worst := cmp.Cosine;
      CheckUtf8(cmp.Cosine >= aStateCos, '% case % % cosine %',
        [aLabel, k, name, cmp.Cosine]);
    end;
    cmp := InferCompare(pointer(logits), aRef.F32(FormatUtf8('case%.logits_last',
      [k]), m), aModel.Config.VocabSize, 1, 0);
    want := aRef.I32(FormatUtf8('case%.greedy', [k]), m);
    margins := aRef.F32(FormatUtf8('case%.greedy_margin', [k]), m);
    ctx := aModel.NewContext;
    try
      timer.Start;
      mine := aModel.Greedy(ctx, prompt, m, []);
      AddConsole(FormatString('% case % greedy of % tokens: %',
        [aLabel, k, m, timer.Stop]));
    finally
      ctx.Free;
    end;
    diff := 0;
    while (diff < m) and (diff < length(mine)) and (mine[diff] = want^[diff]) do
      inc(diff);
    margin := 0;
    if diff < m then
      margin := margins^[diff];
    AddConsole(FormatString('% case %: states cosine >= %, logits cos % maxabs %, ' +
      'greedy equal for % of % tokens (reference margin there %)',
      [aLabel, k, worst, cmp.Cosine, cmp.MaxAbs, diff, m, margin]));
    CheckUtf8(cmp.Cosine >= aCos, '% case % logits cosine %', [aLabel, k, cmp.Cosine]);
    CheckUtf8(cmp.MaxAbs <= aMaxAbs, '% case % logits max error %',
      [aLabel, k, cmp.MaxAbs]);
    CheckUtf8((diff >= m) or (margin < 1),
      '% case % greedy leaves the reference at step % where it was % sure',
      [aLabel, k, diff, margin]);
  end;
end;

procedure TTestInferQwen3.GptOss20b;
const
  GPTOSS20B = 'gpt-oss-20b';
  GPTOSS20B_GGUF = 'gpt-oss-20b-mxfp4.gguf';
var
  dir, refdir, gguf: TFileName;
  ref: TInferRefSet;
  model, g: TDecoderModel;
  l, e, bytes: PtrInt;
  same: boolean;
  timer: TPrecisionTimer;
begin
  {$ifdef CPU32}
  AddConsole('SKIP gpt-oss-20b: 13 GB of weights need a 64-bit process');
  exit;
  {$endif CPU32}
  dir := FindVendorModel(GPTOSS20B + PathDelim + 'model.safetensors.index.json');
  if dir = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/gpt-oss-20b');
    AddConsole('SKIP gpt-oss-20b: no vendor/models/gpt-oss-20b (revision in ' +
      'tests/data/infer-ref/gpt-oss-20b/manifest.json)');
    exit;
  end;
  dir := ExtractFilePath(dir);
  refdir := FindInferRefDir(GPTOSS20B);
  if CheckFailed(refdir <> '', 'gpt-oss-20b reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  g := nil;
  try
    timer.Start;
    model := NewDecoderModel(dir, fFast);
    AddConsole(FormatString('gpt-oss-20b loaded in %', [timer.Stop]));
    Check(model is TGptOssModel, 'a TGptOssModel');
    Check(model.Layers[23].ExpertDown[31].DType = idtMXFP4, 'MXFP4 experts');
    // the experts see Q8_0 activations, as in ggml. The chat case (78 tokens,
    // Harmony tokens) loses most there: transformers itself with Q8_0 expert
    // inputs gives logits at cosine 0.99258 / max 3.32 from float32 (measured
    // 2026-10-20), the engine 0.99327 / 3.27; the other cases 0.99999
    CheckLastRows(model, ref, 'safetensors', 0.985, 0.99, 4);
    // the GGUF of the same revision (ggml-org/gpt-oss-20b-GGUF): its experts
    // byte for byte ours, its matrices and embedding in Q8_0
    gguf := FindVendorModel(GPTOSS20B_GGUF);
    if gguf = '' then
    begin
      CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
        'LAI_REQUIRE_MODEL=1 but no vendor/models/gpt-oss-20b-mxfp4.gguf');
      AddConsole('SKIP gpt-oss-20b GGUF: no vendor/models/gpt-oss-20b-mxfp4.gguf');
      exit;
    end;
    timer.Start;
    g := NewDecoderModelFromGguf(gguf, fFast);
    AddConsole(FormatString('gpt-oss-20b GGUF loaded in %', [timer.Stop]));
    Check(g is TGptOssModel, 'GGUF: a TGptOssModel');
    Check((g.Layers[0].Q.DType = idtQ8_0) and (g.Layers[5].ExpertUp[7].DType = idtMXFP4),
      'GGUF: Q8_0 matrices, MXFP4 experts');
    bytes := (model.Config.ExpertIntermediate * model.Config.Hidden div 32) *
      SizeOf(TInferBlockMxfp4);
    same := true;
    for l := 0 to model.Config.Layers - 1 do
      for e := 0 to model.Config.Experts - 1 do
        if not CompareMem(g.Layers[l].ExpertGate[e].Data,
             model.Layers[l].ExpertGate[e].Data, bytes) or
           not CompareMem(g.Layers[l].ExpertUp[e].Data,
             model.Layers[l].ExpertUp[e].Data, bytes) or
           not CompareMem(g.Layers[l].ExpertDown[e].Data,
             model.Layers[l].ExpertDown[e].Data, bytes) then
          same := false;
    Check(same, 'GGUF: every expert byte for byte as repacked from safetensors');
    FreeAndNil(model);
    CheckLastRows(g, ref, 'GGUF', 0.98, 0.99, 4);
  finally
    g.Free;
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.GptOssGguf;
const
  SHARD1 = 'model-00001-of-00002.safetensors';
  SHARD2 = 'model-00002-of-00002.safetensors';
var
  dir, src, tmp: TFileName;
  ref: TInferRefSet;
  s, g, q: TDecoderModel;
  cs, cg: TDecoderContext;
  ls, lg: TSingleDynArray;
  ids: PIntegerArray;
  l, e, k, n, v, bytes: PtrInt;
  ok: boolean;
  cmp: TInferCompare;
  why: RawUtf8;

  function Refused(const aFile, aWhy: RawUtf8): boolean;
  var
    m: TDecoderModel;
  begin
    result := false;
    why := 'accepted';
    try
      m := TGptOssModel.CreateFromGguf(dir + Utf8ToString(aFile), fCpu);
      m.Free;
    except
      on E: Exception do
      begin
        why := StringToUtf8(E.Message);
        result := (E is EInferError) and (PosEx(aWhy, why) > 0);
      end;
    end;
  end;

begin
  dir := FindInferRefDir('tiny-gpt-oss');
  if CheckFailed(dir <> '', 'tiny-gpt-oss found') then
    exit;
  src := MakePath([dir, 'mxfp4'], true);
  dir := MakePath([dir, 'gguf'], true);
  ref := TInferRefSet.CreateFromFolder(src);
  s := nil;
  g := nil;
  q := nil;
  // llama.cpp fixes the SwiGLU limit of gpt-oss at 7 (no metadata for it):
  // the safetensors folder with that limit is the counterpart of the file
  tmp := MakePath([GetSystemPath(spTemp), 'lai-d5b2-' +
    Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  try
    FileFromString(StringReplaceAll(StringFromFile(src + 'config.json'),
      '"swiglu_limit": 0.8', '"swiglu_limit": 7'), tmp + 'config.json');
    CopyFile(src + 'model.safetensors.index.json',
      tmp + 'model.safetensors.index.json', false);
    CopyFile(src + SHARD1, tmp + SHARD1, false);
    CopyFile(src + SHARD2, tmp + SHARD2, false);
    s := TGptOssModel.Create(tmp, fCpu);
    CheckUtf8(s.Config.SwiGluLimit = 7, 'the counterpart has the limit 7');
    g := NewDecoderModelFromGguf(dir + 'bf16.gguf', fCpu);
    Check(g is TGptOssModel, 'architecture gpt-oss: a TGptOssModel');
    // the settings of the metadata = those of config.json
    Check((g.Config.Hidden = s.Config.Hidden) and (g.Config.Layers = s.Config.Layers) and
      (g.Config.Heads = s.Config.Heads) and (g.Config.KvHeads = s.Config.KvHeads) and
      (g.Config.HeadDim = s.Config.HeadDim) and (g.Config.VocabSize = s.Config.VocabSize) and
      (g.Config.Experts = s.Config.Experts) and
      (g.Config.ExpertsUsed = s.Config.ExpertsUsed) and
      (g.Config.ExpertIntermediate = s.Config.ExpertIntermediate) and
      (g.Config.MaxPositions = s.Config.MaxPositions), 'sizes');
    Check((g.Config.RopeTheta = s.Config.RopeTheta) and
      (g.Config.RopeFactor = s.Config.RopeFactor) and
      (g.Config.RopeOriginal = s.Config.RopeOriginal) and
      (g.Config.RopeBetaFast = s.Config.RopeBetaFast) and
      (g.Config.RopeBetaSlow = s.Config.RopeBetaSlow) and
      (g.Config.RopeTruncate = s.Config.RopeTruncate) and
      (g.Config.RopeAttention = s.Config.RopeAttention), 'YaRN');
    Check((g.Config.SwiGluLimit = s.Config.SwiGluLimit) and
      (g.Config.SwiGluAlpha = s.Config.SwiGluAlpha), 'SwiGLU of llama.cpp');
    ok := length(g.Config.Spec) = length(s.Config.Spec);
    if ok then
      for l := 0 to high(g.Config.Spec) do
        if (g.Config.Spec[l].Window <> s.Config.Spec[l].Window) or
           (g.Config.Spec[l].Ffn <> dfMoe) then
          ok := false;
    Check(ok, 'the even layers slide, all with experts');
    Check(g.Extras = s.Extras, 'the same extras');
    // the experts in place: the converter's repacking = ours, byte for byte
    bytes := (s.Config.ExpertIntermediate * s.Config.Hidden div 32) *
      SizeOf(TInferBlockMxfp4);
    ok := true;
    for l := 0 to g.Config.Layers - 1 do
      for e := 0 to g.Config.Experts - 1 do
        if (g.Layers[l].ExpertGate[e].DType <> idtMXFP4) or
           not CompareMem(g.Layers[l].ExpertGate[e].Data,
             s.Layers[l].ExpertGate[e].Data, bytes) or
           not CompareMem(g.Layers[l].ExpertUp[e].Data,
             s.Layers[l].ExpertUp[e].Data, bytes) or
           not CompareMem(g.Layers[l].ExpertDown[e].Data,
             s.Layers[l].ExpertDown[e].Data, bytes) then
          ok := false;
    Check(ok, 'the MXFP4 experts byte for byte as repacked from safetensors');
    // the logits bit for bit, every position
    v := s.Config.VocabSize;
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(ls, n * v);
      SetLength(lg, n * v);
      cs := s.NewContext;
      cg := g.NewContext;
      try
        s.Forward(cs, ids, n, pointer(ls), true);
        g.Forward(cg, ids, n, pointer(lg), true);
      finally
        cs.Free;
        cg.Free;
      end;
      CheckUtf8(CompareMem(pointer(ls), pointer(lg), n * v * SizeOf(single)),
        'case % GGUF = safetensors bit for bit', [k]);
    end;
    // the matrices in Q8_0 as the published file: within the Q8_0 error
    q := NewDecoderModelFromGguf(dir + 'q8_0.gguf', fFast);
    Check((q.Layers[0].Q.DType = idtQ8_0) and (q.Layers[0].ExpertUp[2].DType = idtMXFP4),
      'Q8_0 matrices, MXFP4 experts');
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(lg, n * v);
      SetLength(ls, n * v);
      cs := s.NewContext;
      cg := q.NewContext;
      try
        s.Forward(cs, ids, n, pointer(ls), true);
        q.Forward(cg, ids, n, pointer(lg), true);
      finally
        cs.Free;
        cg.Free;
      end;
      // measured 2026-10-20: logits within 0.042, cosine 0.99985
      cmp := InferCompare(pointer(lg), pointer(ls), n * v, 0.1, 0.05);
      CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.999),
        'case % Q8_0 GGUF: %', [k, InferCompareText(cmp)]);
      AddConsole(FormatString('gpt-oss GGUF case % Q8_0 vs BF16: %',
        [k, InferCompareText(cmp)]));
    end;
    // YaRN betas of the file, not llama.cpp's defaults
    FreeAndNil(q);
    q := NewDecoderModelFromGguf(dir + 'betas.gguf', fCpu);
    Check((q.Config.RopeBetaFast = 16) and (q.Config.RopeBetaSlow = 2),
      'the YaRN betas of the metadata');
    // refusals
    CheckUtf8(Refused('bad-arch.gguf', 'architecture'), 'another architecture: %', [why]);
    CheckUtf8(Refused('bad-rope.gguf', 'rope scaling'), 'another rope scaling: %', [why]);
    CheckUtf8(Refused('bad-yarn.gguf', 'attention factor'), 'a YaRN attn_factor: %', [why]);
    CheckUtf8(Refused('bad-tied.gguf', 'tied embeddings'), 'no output.weight: %', [why]);
    CheckUtf8(Refused('bad-missing.gguf', 'attn_sinks'), 'a missing sink: %', [why]);
    CheckUtf8(Refused('bad-extra.gguf', 'understood'), 'an extra tensor: %', [why]);
    CheckUtf8(Refused('bad-scale.gguf', 'beyond the range'), 'an MXFP4 exponent of 150: %',
      [why]);
  finally
    q.Free;
    g.Free;
    s.Free;
    ref.Free;
    DirectoryDeleteAll(tmp);
  end;
end;

procedure TTestInferQwen3.GptOssConfig;
var
  dir, tmp: TFileName;
  ref: TInferRefSet;
  base, doc: TDocVariantData;
  rope: PDocVariantData;
  why, detail: RawUtf8;
  m, b: TDecoderModel;
  s: TDecoderLayerSpecDynArray;
  nullv: variant;
  la, lb: TSingleDynArray;
  ids: PIntegerArray;
  n: PtrInt;
  theta, r: variant;

  function Load: TDecoderModel;
  begin
    FileFromString(doc.ToJson, tmp + 'config.json');
    doc.Clear;
    doc.InitCopy(variant(base), JSON_FAST_FLOAT); // the next change from the file
    result := NewDecoderModel(tmp, fCpu);
  end;

  function Refused: RawUtf8;
  begin
    result := '';
    try
      Load.Free;
    except
      on E: EInferError do
        result := StringToUtf8(E.Message);
    end;
  end;

  /// the logits of case 2 of aModel into aLogits
  procedure Logits(aModel: TDecoderModel; var aLogits: TSingleDynArray);
  var
    ctx: TDecoderContext;
  begin
    SetLength(aLogits, aModel.Config.VocabSize);
    ctx := aModel.NewContext;
    try
      aModel.Forward(ctx, ids, n, pointer(aLogits));
    finally
      ctx.Free;
    end;
  end;

  /// the rope settings of doc (a copy of base)
  function RopeOf: PDocVariantData;
  begin
    if not doc.GetAsObject('rope_parameters', result) then
      result := nil;
  end;

begin
  dir := FindInferRefDir('tiny-gpt-oss');
  if CheckFailed(dir <> '', 'tiny-gpt-oss found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  base.InitFast;
  doc.InitFast;
  SetVariantNull(nullv); // JSON null
  ids := ref.I32('case2.ids', n);
  tmp := MakePath([GetSystemPath(spTemp), 'lai-d5-' + Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  b := nil;
  try
    base.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    CopyFile(dir + 'model.safetensors', tmp + 'model.safetensors', false);
    doc.InitCopy(variant(base), JSON_FAST_FLOAT);
    b := Load;
    Logits(b, la);
    // the transformers 4 form: rope_scaling, rope_theta at the top level -
    // the very same model
    // (copied first: adding to doc may move the object rope points into)
    rope := RopeOf;
    theta := rope^.Value['rope_theta'];
    rope^.Delete('rope_theta');
    r := variant(rope^);
    doc.Delete('rope_parameters');
    doc.AddValue('rope_theta', theta);
    doc.AddValue('rope_scaling', r);
    m := Load;
    try
      Logits(m, lb);
      Check(CompareMem(pointer(la), pointer(lb), length(la) * SizeOf(single)),
        'rope_scaling and rope_theta: the same logits');
    finally
      m.Free;
    end;
    // a theta in both places: the one of rope_parameters (transformers moves
    // the top-level one in by setdefault)
    SetFloat(doc, 'rope_theta', 10000);
    m := Load;
    try
      Check(m.Config.RopeTheta = 150000, 'rope_theta of rope_parameters first');
    finally
      m.Free;
    end;
    // rope_parameters without a theta: the one at the top level, else
    // GptOssConfig's 150000
    RopeOf^.Delete('rope_theta');
    SetFloat(doc, 'rope_theta', 10000);
    m := Load;
    try
      Check(m.Config.RopeTheta = 10000, 'rope_theta at the top level');
    finally
      m.Free;
    end;
    RopeOf^.Delete('rope_theta');
    m := Load;
    try
      Check(m.Config.RopeTheta = 150000, 'rope_theta 150000 by default');
    finally
      m.Free;
    end;
    // no rope settings at all: GptOssConfig's YaRN, which tiny-gpt-oss uses
    doc.Delete('rope_parameters');
    m := Load;
    try
      Check((m.Config.RopeFactor = 32) and (m.Config.RopeOriginal = 4096) and
        not m.Config.RopeTruncate, 'YaRN by default');
      Logits(m, lb);
      Check(CompareMem(pointer(la), pointer(lb), length(la) * SizeOf(single)),
        'no rope settings: the same logits');
    finally
      m.Free;
    end;
    // factor null: max_position_embeddings / original, as transformers
    RopeOf^.AddOrUpdateValue('factor', nullv);
    doc.I['max_position_embeddings'] := 131072;
    m := Load;
    try
      Check(m.Config.RopeFactor = 32, 'factor null: 131072 / 4096');
      Logits(m, lb);
      Check(CompareMem(pointer(la), pointer(lb), length(la) * SizeOf(single)),
        'factor null: the same logits');
    finally
      m.Free;
    end;
    // attention_factor null: none, derived from the factor
    RopeOf^.AddOrUpdateValue('attention_factor', nullv);
    m := Load;
    try
      Logits(m, lb);
      Check(CompareMem(pointer(la), pointer(lb), length(la) * SizeOf(single)),
        'attention_factor null: the same logits');
    finally
      m.Free;
    end;
    // the YaRN variants against their reference factors
    RopeOf^.B['truncate'] := true;
    m := Load;
    try
      Check(m.Config.RopeTruncate, 'truncate');
      CheckUtf8(RotaryMatches(m, ref, 'rope_t', detail), 'YaRN truncate: %', [detail]);
    finally
      m.Free;
    end;
    // truncate missing: true, as rope_parameters.get("truncate", True)
    RopeOf^.Delete('truncate');
    m := Load;
    try
      Check(m.Config.RopeTruncate, 'truncate true by default');
    finally
      m.Free;
    end;
    rope := RopeOf;
    SetFloat(rope^, 'attention_factor', 1.25);
    SetFloat(rope^, 'beta_fast', 16);
    SetFloat(rope^, 'beta_slow', 2);
    m := Load;
    try
      CheckUtf8(RotaryMatches(m, ref, 'rope_a', detail), 'YaRN attention_factor: %',
        [detail]);
    finally
      m.Free;
    end;
    // no layer_types: the even layers slide (GptOssConfig); no sliding_window
    // either: 128
    doc.Delete('layer_types');
    m := Load;
    try
      s := m.Config.Spec;
      Check((s[0].Window = 5) and (s[1].Window = 0) and (s[2].Window = 5),
        'gpt-oss: the even layers slide by default');
    finally
      m.Free;
    end;
    doc.Delete('sliding_window');
    m := Load;
    try
      s := m.Config.Spec;
      Check((s[0].Window = 128) and (s[1].Window = 0) and (s[2].Window = 128),
        'gpt-oss: sliding_window 128 by default');
    finally
      m.Free;
    end;
    // swiglu_alpha is read: another one, other logits
    SetFloat(doc, 'swiglu_alpha', 2);
    m := Load;
    try
      Check(m.Config.SwiGluAlpha = 2, 'swiglu_alpha 2');
      Logits(m, lb);
      Check(not CompareMem(pointer(la), pointer(lb), length(la) * SizeOf(single)),
        'swiglu_alpha changes the logits');
    finally
      m.Free;
    end;
    doc.Delete('swiglu_limit');
    m := Load;
    try
      Check(m.Config.SwiGluLimit = 7, 'swiglu_limit 7 by default');
    finally
      m.Free;
    end;
    // refused
    doc.AddOrUpdateValue('sliding_window', nullv);
    why := Refused;
    CheckUtf8(PosEx('without a sliding window', why) > 0, 'sliding_window null: %', [why]);
    doc.I['sliding_window'] := 0;
    why := Refused;
    CheckUtf8(PosEx('a sliding_window of 0', why) > 0, 'sliding_window 0: %', [why]);
    doc.AddOrUpdateValue('layer_types', _JsonFast('["full_attention"]'));
    why := Refused;
    CheckUtf8(PosEx('layer types for', why) > 0, 'layer types of another count: %', [why]);
    doc.AddOrUpdateValue('layer_types', _JsonFast(
      '["full_attention","chunked_attention","sliding_attention"]'));
    why := Refused;
    CheckUtf8(PosEx('chunked_attention', why) > 0, 'an unknown layer type: %', [why]);
    doc.AddOrUpdateValue('quantization_config', _JsonFast('{"quant_method":"mxfp4"}'));
    why := Refused;
    // MXFP4 settings over BF16 experts: the blocks are missing
    CheckUtf8(PosEx('gate_up_proj_blocks', why) > 0, 'MXFP4 without blocks: %', [why]);
    doc.B['attention_bias'] := false;
    why := Refused;
    CheckUtf8(PosEx('attention_bias', why) > 0, 'attention_bias false: %', [why]);
    doc.B['tie_word_embeddings'] := true;
    why := Refused;
    CheckUtf8(PosEx('tied', why) > 0, 'tied embeddings: %', [why]);
    SetFloat(doc, 'swiglu_limit', 0);
    why := Refused;
    CheckUtf8(PosEx('swiglu_limit', why) > 0, 'swiglu_limit 0: %', [why]);
    // num_key_value_heads null: every head its own KV head - 4 here, the file
    // has 2: refused by the shapes
    doc.AddOrUpdateValue('num_key_value_heads', nullv);
    why := Refused;
    CheckUtf8(PosEx('k_proj.weight is shaped', why) > 0, 'num_key_value_heads null: %',
      [why]);
    doc.Delete('head_dim');
    why := Refused;
    CheckUtf8(PosEx('"head_dim"', why) > 0, 'no head_dim: %', [why]);
    RopeOf^.U['rope_type'] := 'default';
    why := Refused;
    CheckUtf8(PosEx('rope_type "default"', why) > 0, 'plain RoPE: %', [why]);
    SetFloat(RopeOf^, 'mscale', 1);
    why := Refused;
    CheckUtf8(PosEx('rope setting mscale', why) > 0, 'mscale: %', [why]);
    RopeOf^.Delete('factor');
    why := Refused;
    CheckUtf8(PosEx('without a factor', why) > 0, 'no factor: %', [why]);
    SetFloat(RopeOf^, 'factor', 0);
    why := Refused;
    CheckUtf8(PosEx('a YaRN factor of', why) > 0, 'a factor of 0: %', [why]);
    SetFloat(RopeOf^, 'factor', 0.5);
    why := Refused;
    CheckUtf8(PosEx('invalid YaRN', why) > 0, 'a factor below 1: %', [why]);
    RopeOf^.Delete('original_max_position_embeddings');
    why := Refused;
    CheckUtf8(PosEx('original_max_position_embeddings', why) > 0, 'no original: %', [why]);
    // fewer layers than the file holds: a tensor not understood
    doc.I['num_hidden_layers'] := 2;
    doc.AddOrUpdateValue('layer_types', _JsonFast('["sliding_attention","full_attention"]'));
    why := Refused;
    CheckUtf8(PosEx('tensors in the file', why) > 0, 'a tensor left over: %', [why]);
  finally
    b.Free;
    DirectoryDeleteAll(tmp);
    ref.Free;
  end;
end;

end.
