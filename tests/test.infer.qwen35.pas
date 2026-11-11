/// tests for Qwen3.5 in the native inference engine (phase D, slices D7a, D7d, D7f, D7g)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the Gated DeltaNet and the gated full attention against transformers:
//   tests/data/infer-ref/tiny-qwen35 (the layout of the published checkpoints)
//   and tiny-qwen35/text (Qwen3_5ForCausalLM), both BF16, from
//   tools/infer-ref/make_reference_d7.py; the recurrent state of a context
//   (restore points 0 and Length only, a failed Forward), its restore points
//   before a CheckpointId token (D7d: a branch the bits of a fresh pass); the
//   settings refused; both tiny models from GGUF (D7f: tiny-qwen35/gguf, the
//   value heads of text tiled) and the damaged files refused
// - Qwen3.5-0.8B needs vendor/models/qwen3.5-0.8b (and for GGUF
//   qwen3.5-0.8b-q8_0.gguf, qwen3.5-0.8b-q4_0.gguf) and a 64-bit process; it
//   says SKIP otherwise, LAI_REQUIRE_MODEL=1 makes that a failure
// - Qwen3.8-27B (D7g) needs vendor/models/qwen3.8-27b-q8_0.gguf and/or
//   -q4_k_m.gguf (the RTX computer) and says SKIP without them, also under
//   LAI_REQUIRE_MODEL; its reference is tests/data/infer-ref/qwen38-27b
unit test.infer.qwen35;

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
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.loader.gguf,
  mormot.ai.infer.decoder,
  mormot.ai.infer.model.qwen35,
  test.infer.ref,
  test.infer.bpe,
  test.infer.qwen3;

type
  /// D7a: Qwen3.5 on the CPU against the references
  TTestInferQwen35 = class(TTestInferDecoder)
  protected
    /// the cases of a tiny reference folder: every state, the logits of one
    // pass and of one token at a time, the cache, greedy continuations
    procedure CheckTiny(aModel: TDecoderModel; aRef: TInferRefSet; const aLabel: RawUtf8);
    /// KeepState that raises at layer01: a Forward failing inside the layers
    procedure FailAtLayer1(const aName: RawUtf8; aValues: PSingleArray;
      aRows, aCols: PtrInt);
    /// the restore points of aModel's states (D7d)
    procedure CheckCheckpoints(aModel: TDecoderModel; aExact: boolean;
      const aLabel: RawUtf8);
  published
    /// tiny Qwen3.5 in the published layout: the layer kinds, the shapes, the
    // rotary part, every state, the recurrent and the chunked form, greedy
    procedure TinyMatchesReference;
    /// tiny Qwen3.5 saved as Qwen3_5ForCausalLM: an untied head, two value
    // heads per key head, a convolution of 3, half of each head rotating
    procedure TinyTextLayout;
    /// the recurrent state: restorable at 0 and Length only, Reuse computes
    // again what it cannot keep, a failed Forward leaves only 0
    procedure RecurrentState;
    /// restore points of the state (D7d): kept before every CheckpointId
    // token, a branch goes on from the last one before it - the bits of a
    // fresh pass, on the reference and the fast backend, in F32 and Q8_0
    procedure Checkpoints;
    /// tiny Qwen3.5 on the fast backend and quantized to Q8_0 and Q4_0 at load
    procedure TinyQuantized;
    /// the settings of a Qwen3.5 config.json refused
    procedure Refused;
    /// the rotary settings in their other forms, as transformers reads them:
    // partial_rotary_factor or rope_theta at the top, rope_scaling in place of
    // rope_parameters - the same model bit for bit
    procedure RopeForms;
    /// Qwen3.5-0.8B in F32 against transformers: the last row of every state,
    // the last logits, greedy - skipped without vendor/models/qwen3.5-0.8b
    procedure Qwen35Real;
    /// both tiny models from GGUF as llama.cpp's converter writes them (D7f):
    // the settings of the folder, the mtp layer left alone, the value heads
    // of text tiled; BF16 every case of the reference, Q8_0 near F32
    procedure TinyGguf;
    /// the damaged GGUF files refused, each by its own check
    procedure GgufRefused;
    /// ggml-org/Qwen3.5-0.8B-GGUF (Q8_0, Q4_0) against transformers - skipped
    // without vendor/models/qwen3.5-0.8b-q8_0.gguf / -q4_0.gguf
    procedure Qwen35Gguf;
    /// ggml-org/Qwen3.8-27B-GGUF (Q8_0, Q4_K_M) against transformers (D7g):
    // 64 layers, 16 key heads read by 48 tiled value heads - only where the
    // files were put (a large model of its own, not LAI_REQUIRE_MODEL)
    procedure Qwen38Gguf;
  end;


implementation

const
  /// the folder of Qwen3.5-0.8B under vendor/models
  QWEN35_MODEL = 'qwen3.5-0.8b';

procedure TTestInferQwen35.FailAtLayer1(const aName: RawUtf8; aValues: PSingleArray;
  aRows, aCols: PtrInt);
begin
  if aName = 'layer01' then
    raise EInferError.Create('failing at layer01 on purpose');
end;

procedure TTestInferQwen35.CheckTiny(aModel: TDecoderModel; aRef: TInferRefSet;
  const aLabel: RawUtf8);
var
  full, step, ctx: TDecoderContext;
  ids: PIntegerArray;
  n, m, i, k, vocab, diff: PtrInt;
  all, one, steps, rows, viaRows: TSingleDynArray;
  prompt, mine: TIntegerDynArray;
  same: boolean;
  margin: double;
  detail: RawUtf8;
  cmp: TInferCompare;
begin
  // every state of 5, 13 and 29 tokens, the logits of one pass (the reference
  // computes the chunked delta rule there)
  for k := 0 to 2 do
    CheckStates(aModel, aRef, FormatUtf8('case%', [k]), 1e-4, 1e-4, true);
  vocab := aModel.Config.VocabSize;
  full := nil;
  step := nil;
  try
    for k := 0 to 2 do
    begin
      ids := aRef.I32(FormatUtf8('case%.ids', [k]), n);
      full := aModel.NewContext;
      step := aModel.NewContext;
      SetLength(all, n * vocab);
      SetLength(one, vocab);
      SetLength(steps, n * vocab);
      aModel.Forward(full, ids, n, pointer(all), true);
      // one token at a time = the prefill, bit for bit: the state goes the
      // same way through every token
      same := true;
      for i := 0 to n - 1 do
      begin
        aModel.Forward(step, @ids^[i], 1, pointer(one));
        MoveFast(one[0], steps[i * vocab], vocab * SizeOf(single));
        if not CompareMem(pointer(one), @all[i * vocab], vocab * SizeOf(single)) then
          same := false;
      end;
      CheckUtf8(same, '% case %: one token at a time bit-identical to one pass',
        [aLabel, k]);
      // and within the error of the recurrent form of the reference
      cmp := InferCompare(pointer(steps), aRef.F32(FormatUtf8('case%.logits_steps',
        [k]), m), n * vocab, 1e-4, 1e-4);
      CheckUtf8(InferCompareOk(cmp), '% case % stepwise: %',
        [aLabel, k, InferCompareText(cmp)]);
      // a prefill in two chunks
      step.Reset;
      aModel.Forward(step, ids, n div 2, nil);
      aModel.Forward(step, @ids^[n div 2], n - n div 2, pointer(one));
      CheckUtf8(CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single)),
        '% case %: prefill in two chunks', [aLabel, k]);
      FreeAndNil(step);
      FreeAndNil(full);
    end;
    // the rows of Embed through ForwardEmbeds: the logits of the ids
    ids := aRef.I32('case2.ids', n);
    SetLength(all, vocab);
    SetLength(rows, n * aModel.Config.Hidden);
    SetLength(viaRows, vocab);
    aModel.Embed(ids, n, pointer(rows));
    full := aModel.NewContext;
    step := aModel.NewContext;
    aModel.Forward(full, ids, n, pointer(all));
    aModel.ForwardEmbeds(step, pointer(rows), n, pointer(viaRows));
    CheckUtf8(CompareMem(pointer(viaRows), pointer(all), vocab * SizeOf(single)),
      '%: embedding rows give the logits of their ids', [aLabel]);
  finally
    step.Free;
    full.Free;
  end;
  // greedy continuations of 24 tokens
  for k := 0 to 2 do
  begin
    ids := aRef.I32(FormatUtf8('case%.ids', [k]), n);
    SetLength(prompt, n);
    MoveFast(ids^, prompt[0], n * SizeOf(integer));
    ctx := aModel.NewContext;
    try
      mine := aModel.Greedy(ctx, prompt, 24, []);
    finally
      ctx.Free;
    end;
    diff := GreedyDiff(mine, aRef, FormatUtf8('case%', [k]), margin, detail);
    // a step may only leave the reference where it was nearly tied
    CheckUtf8((diff < 0) or (margin < 1e-3),
      '% case% greedy differs at step % (margin %, %)', [aLabel, k, diff, margin, detail]);
    if diff >= 0 then
      AddConsole(FormatString('% case% greedy leaves the reference at step % ' +
        '(margin %)', [aLabel, k, diff, margin]));
  end;
end;

procedure TTestInferQwen35.TinyMatchesReference;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  s: TDecoderLayerSpecDynArray;
  c: TDecoderConfig;
  detail: RawUtf8;
begin
  dir := FindInferRefDir('tiny-qwen35');
  if CheckFailed(dir <> '', 'tiny-qwen35 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  try
    model := NewDecoderModel(dir, fCpu);
    Check(model.ClassType = TQwen35Model, 'qwen3_5: a TQwen35Model');
    s := model.Config.Spec;
    Check((length(s) = 4) and (s[0].Mixer = dmDeltaNet) and (s[1].Mixer = dmDeltaNet) and
      (s[2].Mixer = dmDeltaNet) and (s[3].Mixer = dmAttention) and s[3].OutputGate and
      not s[0].OutputGate, 'three DeltaNet layers, then gated attention');
    c := model.Config;
    Check((c.LinKeyHeads = 2) and (c.LinValueHeads = 2) and (c.LinKeyDim = 16) and
      (c.LinValueDim = 16) and (c.LinConv = 4), 'the DeltaNet shapes');
    Check((c.HeadDim = 32) and (c.RopeDims = 8) and (c.RopeTheta = 10000000) and
      c.TiedEmbeddings, 'heads, the rotating quarter, a tied head');
    Check(model.Extras = [dxOutputGate, dxPartialRope], 'the extras of Qwen3.5');
    Check((model.Layers[0].LinB.DType = idtF32) and (model.Layers[0].LinNegA^[0] < 0) and
      (model.Layers[3].Q.Data <> nil) and (model.Layers[0].Q.Data = nil),
      'the weights of both kinds');
    CheckUtf8(RotaryMatches(model, ref, 'rope', detail), 'partial rope: %', [detail]);
    CheckTiny(model, ref, 'tiny');
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen35.TinyTextLayout;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  s: TDecoderLayerSpecDynArray;
  c: TDecoderConfig;
  detail: RawUtf8;
begin
  dir := FindInferRefDir('tiny-qwen35');
  if CheckFailed(dir <> '', 'tiny-qwen35 found') then
    exit;
  dir := MakePath([dir, 'text'], true);
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  try
    model := NewDecoderModel(dir, fCpu);
    Check(model.ClassType = TQwen35TextModel, 'qwen3_5_text: a TQwen35TextModel');
    s := model.Config.Spec;
    Check((length(s) = 5) and (s[0].Mixer = dmDeltaNet) and (s[1].Mixer = dmAttention) and
      (s[2].Mixer = dmDeltaNet) and (s[3].Mixer = dmDeltaNet) and
      (s[4].Mixer = dmAttention), 'the explicit layer types');
    c := model.Config;
    Check((c.LinKeyHeads = 2) and (c.LinValueHeads = 4) and (c.LinConv = 3) and
      (c.RopeDims = 16) and not c.TiedEmbeddings, 'shapes, an untied head');
    CheckUtf8(RotaryMatches(model, ref, 'rope', detail), 'half rope: %', [detail]);
    CheckTiny(model, ref, 'text');
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen35.RecurrentState;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  ctx, fresh: TDecoderContext;
  ids: PIntegerArray;
  n, vocab, kept: PtrInt;
  want, got: TSingleDynArray;
  other: TIntegerDynArray;
  raised: boolean;
begin
  dir := FindInferRefDir('tiny-qwen35');
  if CheckFailed(dir <> '', 'tiny-qwen35 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  ctx := nil;
  fresh := nil;
  try
    model := NewDecoderModel(dir, fCpu);
    vocab := model.Config.VocabSize;
    ids := ref.I32('case2.ids', n);
    SetLength(want, vocab);
    SetLength(got, vocab);
    ctx := model.NewContext;
    fresh := model.NewContext;
    model.Forward(ctx, ids, n, nil);
    // no way back into the sequence: the state holds all of it
    Check(ctx.Restorable(0) and ctx.Restorable(n) and not ctx.Restorable(n - 1) and
      not ctx.Restorable(1), 'restorable at 0 and Length only');
    raised := false;
    try
      ctx.Truncate(n - 1);
    except
      on EInferError do
        raised := true;
    end;
    Check(raised and (ctx.Length = n), 'Truncate inside the sequence refused');
    ctx.Truncate(n); // staying is fine
    // a prompt that branches off: computed again from the start, the logits
    // of a fresh pass bit for bit
    SetLength(other, n);
    MoveFast(ids^, other[0], n * SizeOf(integer));
    other[n - 3] := (other[n - 3] + 1) mod vocab;
    kept := model.Reuse(ctx, pointer(other), n);
    CheckEqual(kept, 0, 'a branch inside the sequence keeps nothing');
    model.Forward(ctx, pointer(other), n, pointer(got));
    model.Forward(fresh, pointer(other), n, pointer(want));
    Check(CompareMem(pointer(got), pointer(want), vocab * SizeOf(single)),
      'after the branch: the bits of a fresh pass');
    // a prompt that goes on from the whole sequence keeps it
    SetLength(other, n + 2);
    other[n] := 5;
    other[n + 1] := 6;
    kept := model.Reuse(ctx, pointer(other), n + 2);
    CheckEqual(kept, n, 'the whole sequence kept');
    model.Forward(ctx, @other[n], 2, pointer(got));
    fresh.Reset;
    model.Forward(fresh, pointer(other), n + 2, pointer(want));
    Check(CompareMem(pointer(got), pointer(want), vocab * SizeOf(single)),
      'going on from the kept state: the bits of a fresh pass');
    // a Forward failing inside the layers leaves the state undone
    raised := false;
    try
      model.Forward(ctx, ids, 3, nil, false, FailAtLayer1);
    except
      on EInferError do
        raised := true;
    end;
    Check(raised and (ctx.Length = n + 2), 'the failed Forward keeps the length');
    Check(not ctx.Restorable(n + 2) and ctx.Restorable(0), 'only 0 is restorable');
    raised := false;
    try
      model.Forward(ctx, ids, 1, nil);
    except
      on E: EInferError do
        raised := PosEx('undone', StringToUtf8(E.Message)) > 0;
    end;
    Check(raised, 'going on from an undone state refused');
    kept := model.Reuse(ctx, pointer(other), n + 2);
    CheckEqual(kept, 0, 'Reuse starts again');
    model.Forward(ctx, pointer(other), n + 2, pointer(got));
    Check(CompareMem(pointer(got), pointer(want), vocab * SizeOf(single)),
      'started again: the bits of a fresh pass');
  finally
    fresh.Free;
    ctx.Free;
    model.Free;
    ref.Free;
  end;
end;

/// the logits of aIds in a new context of aModel with aCheckpointId
function FreshLogits(aModel: TDecoderModel; const aIds: TIntegerDynArray;
  aCheckpointId: integer; aAll: boolean = false): TSingleDynArray;
var
  ctx: TDecoderContext;
begin
  if aAll then
    SetLength(result, length(aIds) * aModel.Config.VocabSize)
  else
    SetLength(result, aModel.Config.VocabSize);
  ctx := aModel.NewContext;
  try
    ctx.CheckpointId := aCheckpointId;
    aModel.Forward(ctx, pointer(aIds), length(aIds), pointer(result), aAll);
  finally
    ctx.Free;
  end;
end;

function SameSingles(const a, b: TSingleDynArray): boolean;
begin
  result := (length(a) = length(b)) and
    CompareMem(pointer(a), pointer(b), length(a) * SizeOf(single));
end;

function MaxGap(const a, b: TSingleDynArray): double;
var
  i: PtrInt;
begin
  result := 1e30;
  if length(a) <> length(b) then
    exit;
  result := 0;
  for i := 0 to high(a) do
    if abs(a[i] - b[i]) > result then
      result := abs(a[i] - b[i]);
end;

procedure TTestInferQwen35.CheckCheckpoints(aModel: TDecoderModel;
  aExact: boolean; const aLabel: RawUtf8);
const
  LEN = 20;
var
  vocab, i, kept, m: PtrInt;
  a, b: TIntegerDynArray;
  ctx: TDecoderContext;
  got, first, want: TSingleDynArray;
  raised: boolean;
begin
  vocab := aModel.Config.VocabSize;
  m := vocab - 1; // the turn start: at 5 and 12 only
  SetLength(a, LEN);
  for i := 0 to LEN - 1 do
    a[i] := (i * 7 + 3) mod m;
  a[5] := m;
  a[12] := m;
  SetLength(got, vocab);
  SetLength(first, vocab);
  ctx := aModel.NewContext;
  try
    CheckEqual(ctx.CheckpointId, -1, aLabel + ' no restore points by default');
    aModel.Forward(ctx, pointer(a), LEN, nil);
    CheckEqual(ctx.Checkpoints, 0, aLabel + ' none without CheckpointId');
    ctx.Reset;
    ctx.CheckpointId := m;
    aModel.Forward(ctx, pointer(a), LEN, pointer(first));
    CheckEqual(ctx.Checkpoints, 2, aLabel + ' two restore points');
    CheckUtf8(ctx.Restorable(5) and ctx.Restorable(12) and ctx.Restorable(LEN) and
      ctx.Restorable(0) and not ctx.Restorable(6) and not ctx.Restorable(11) and
      not ctx.Restorable(13), '% restorable at the restore points', [aLabel]);
    // the parts against one pass: equal where the backend computes every
    // row as one of a larger batch
    want := FreshLogits(aModel, a, -1);
    if aExact then
      CheckUtf8(SameSingles(first, want), '% the parts: the bits of one pass', [aLabel])
    else
      CheckUtf8(MaxGap(first, want) < 1e-3, '% the parts near one pass: %',
        [aLabel, MaxGap(first, want)]);
    // every row's logits at its place
    want := FreshLogits(aModel, a, -1, true);
    got := FreshLogits(aModel, a, m, true);
    if aExact then
      CheckUtf8(SameSingles(got, want), '% all logits: the bits of one pass', [aLabel])
    else
      CheckUtf8(MaxGap(got, want) < 1e-3, '% all logits near one pass: %',
        [aLabel, MaxGap(got, want)]);
    CheckUtf8(CompareMem(@got[(LEN - 1) * vocab], pointer(first), vocab * SizeOf(single)),
      '% the last row of all logits', [aLabel]);
    SetLength(got, vocab);
    // a branch after the second turn start: on from the restore point at 12,
    // the bits of a fresh pass of the branch
    b := copy(a, 0, LEN);
    b[14] := (b[14] + 1) mod m;
    kept := aModel.Reuse(ctx, pointer(b), LEN);
    CheckEqual(kept, 12, aLabel + ' the branch goes on from 12');
    CheckEqual(ctx.Checkpoints, 2, aLabel + ' both restore points stay');
    aModel.Forward(ctx, @b[kept], LEN - kept, pointer(got));
    CheckUtf8(SameSingles(got, FreshLogits(aModel, b, m)),
      '% the branch: the bits of a fresh pass', [aLabel]);
    // back to the first restore point, on with the first prompt: its logits
    ctx.Truncate(5);
    CheckEqual(ctx.Checkpoints, 1, aLabel + ' the later one dropped');
    aModel.Forward(ctx, @a[5], LEN - 5, pointer(got));
    CheckUtf8(SameSingles(got, first), '% back at 5: the first logits', [aLabel]);
    CheckEqual(ctx.Checkpoints, 2, aLabel + ' 12 kept again');
    // a branch that replaces the turn start at 12: a fresh pass of it does not
    // split there, so the rows before it go on from the restore point at 5
    b := copy(a, 0, LEN);
    b[12] := 1;
    CheckEqual(ctx.Usable(pointer(b), LEN), 5, aLabel + ' not the replaced one');
    b[12] := m;
    b[13] := (b[13] + 1) mod m;
    CheckEqual(ctx.Usable(pointer(b), LEN), 12, aLabel + ' a branch just after it');
    b := copy(a, 0, LEN);
    b[12] := 1;
    kept := aModel.Reuse(ctx, pointer(b), LEN);
    CheckEqual(kept, 5, aLabel + ' the replaced turn start: on from 5');
    aModel.Forward(ctx, @b[kept], LEN - kept, pointer(got));
    CheckUtf8(SameSingles(got, FreshLogits(aModel, b, m)),
      '% the replaced turn start: the bits of a fresh pass', [aLabel]);
    CheckEqual(ctx.Checkpoints, 1, aLabel + ' only 5 left');
    // the first prompt again, from 5
    kept := aModel.Reuse(ctx, pointer(a), LEN);
    CheckEqual(kept, 5, aLabel + ' the first prompt from 5');
    aModel.Forward(ctx, @a[kept], LEN - kept, pointer(got));
    CheckUtf8(SameSingles(got, first), '% the first logits again', [aLabel]);
    // a branch before the first turn start: from the start
    b := copy(a, 0, LEN);
    b[3] := (b[3] + 1) mod m;
    kept := aModel.Reuse(ctx, pointer(b), LEN);
    CheckEqual(kept, 0, aLabel + ' a branch before 5 starts again');
    CheckEqual(ctx.Checkpoints, 0, aLabel + ' and keeps none');
    aModel.Forward(ctx, pointer(b), LEN, pointer(got));
    CheckUtf8(SameSingles(got, FreshLogits(aModel, b, m)),
      '% started again: the bits of a fresh pass', [aLabel]);
    // a prompt going on from the whole sequence, a turn start first: a
    // restore point where the part starts
    SetLength(b, LEN + 3);
    b[LEN] := m;
    b[LEN + 1] := 1;
    b[LEN + 2] := 2;
    kept := aModel.Reuse(ctx, pointer(b), LEN + 3);
    CheckEqual(kept, LEN, aLabel + ' the whole sequence kept');
    aModel.Forward(ctx, @b[LEN], 3, pointer(got));
    CheckEqual(ctx.Checkpoints, 3, aLabel + ' a restore point at LEN');
    CheckUtf8(ctx.Restorable(LEN) and ctx.Restorable(LEN + 3) and
      not ctx.Restorable(LEN + 1), '% restorable at LEN', [aLabel]);
    CheckUtf8(SameSingles(got, FreshLogits(aModel, b, m)),
      '% gone on: the bits of a fresh pass', [aLabel]);
    // the oldest go first
    ctx.MaxCheckpoints := 1;
    CheckEqual(ctx.Checkpoints, 1, aLabel + ' one kept');
    CheckUtf8(ctx.Restorable(LEN) and not ctx.Restorable(12) and not ctx.Restorable(5),
      '% the last one kept', [aLabel]);
    ctx.MaxCheckpoints := 0;
    CheckUtf8(not ctx.Restorable(LEN) and ctx.Restorable(LEN + 3), '% none', [aLabel]);
    raised := false;
    try
      ctx.MaxCheckpoints := -1;
    except
      on EInferError do
        raised := true;
    end;
    CheckUtf8(raised, '% MaxCheckpoints -1 refused', [aLabel]);
    ctx.MaxCheckpoints := DECODER_CHECKPOINTS;
    // a token outside the vocabulary: refused before the first part runs
    ctx.Reset;
    b := copy(a, 0, LEN);
    b[LEN - 1] := vocab;
    raised := false;
    try
      aModel.Forward(ctx, pointer(b), LEN, nil);
    except
      on EInferError do
        raised := true;
    end;
    CheckUtf8(raised and (ctx.Length = 0) and (ctx.Checkpoints = 0),
      '% a bad id: nothing computed', [aLabel]);
    // more turn starts than kept: the oldest goes
    ctx.Reset;
    b := copy(a, 0, LEN);
    b[16] := m;
    ctx.MaxCheckpoints := 2;
    aModel.Forward(ctx, pointer(b), LEN, nil);
    CheckEqual(ctx.Checkpoints, 2, aLabel + ' two of three kept');
    CheckUtf8(ctx.Restorable(12) and ctx.Restorable(16) and not ctx.Restorable(5),
      '% the oldest one gone', [aLabel]);
    ctx.MaxCheckpoints := DECODER_CHECKPOINTS;
    // a prompt fed in two calls, the first ending inside a turn: the rows
    // after 8 are split where a fresh pass does not split them - no restore
    // point at 12, the one at 5 stays
    ctx.Reset;
    aModel.Forward(ctx, pointer(a), 8, nil);
    aModel.Forward(ctx, @a[8], LEN - 8, nil);
    CheckEqual(ctx.Checkpoints, 1, aLabel + ' two calls: none at 12');
    b := copy(a, 0, LEN);
    b[14] := (b[14] + 1) mod m;
    CheckEqual(ctx.Usable(pointer(b), LEN), 5, aLabel + ' two calls: on from 5');
    kept := aModel.Reuse(ctx, pointer(b), LEN);
    aModel.Forward(ctx, @b[kept], LEN - kept, pointer(got));
    CheckUtf8(SameSingles(got, FreshLogits(aModel, b, m)),
      '% two calls: the bits of a fresh pass', [aLabel]);
    // now split as a fresh pass: 12 is of use again
    b[14] := (b[14] + 1) mod m;
    CheckEqual(ctx.Usable(pointer(b), LEN), 12, aLabel + ' split again: on from 12');
    // back to 12 by hand, on with another token there: that restore point
    // holds no turn start any more
    ctx.Truncate(12);
    b := copy(a, 0, LEN);
    b[12] := 1;
    aModel.Forward(ctx, @b[12], LEN - 12, nil);
    b[14] := (b[14] + 1) mod m;
    CheckEqual(ctx.Usable(pointer(b), LEN), 5, aLabel + ' no turn start at 12: on from 5');
    // an agent loop: a prompt going on from the end of the cache, after a
    // generated token - split where a fresh pass would not split, so no
    // restore point is kept there, and none pushes out the one at 5
    ctx.Reset;
    ctx.MaxCheckpoints := 3;
    aModel.Forward(ctx, pointer(a), LEN, nil);
    b := copy(a, 0, LEN);
    SetLength(b, LEN + 4);
    b[LEN] := 1; // a generated token
    b[LEN + 1] := 2;
    b[LEN + 2] := m; // the next turn
    b[LEN + 3] := 3;
    aModel.Forward(ctx, @b[LEN], 1, nil);
    aModel.Forward(ctx, @b[LEN + 1], 3, nil);
    CheckEqual(ctx.Checkpoints, 2, aLabel + ' no restore point after a continuation');
    CheckUtf8(ctx.Restorable(5) and ctx.Restorable(12) and not ctx.Restorable(LEN + 2),
      '% the aligned ones kept', [aLabel]);
    b[14] := (b[14] + 1) mod m;
    CheckEqual(ctx.Usable(pointer(b), LEN + 4), 12, aLabel + ' a branch: on from 12');
    ctx.MaxCheckpoints := DECODER_CHECKPOINTS;
    // another CheckpointId: the rows were split by the old one
    ctx.Reset;
    ctx.CheckpointId := -1;
    aModel.Forward(ctx, pointer(a), 8, nil);
    ctx.CheckpointId := m;
    aModel.Forward(ctx, @a[8], LEN - 8, nil);
    CheckEqual(ctx.Checkpoints, 0, aLabel + ' set after the first rows: none kept');
    ctx.Reset;
    CheckEqual(ctx.CheckpointId, m, aLabel + ' Reset keeps the id');
    aModel.Forward(ctx, pointer(a), LEN, nil);
    CheckEqual(ctx.Checkpoints, 2, aLabel + ' from the start: kept');
    ctx.CheckpointId := 1;
    CheckEqual(ctx.Checkpoints, 0, aLabel + ' another id forgets them');
    ctx.CheckpointId := m;
    // Reset forgets them
    ctx.Reset;
    aModel.Forward(ctx, pointer(a), LEN, nil);
    CheckEqual(ctx.Checkpoints, 2, aLabel + ' computed again');
    ctx.Reset;
    CheckUtf8((ctx.Checkpoints = 0) and not ctx.Restorable(5), '% Reset', [aLabel]);
  finally
    ctx.Free;
  end;
end;

procedure TTestInferQwen35.Checkpoints;
var
  dir: TFileName;
  model: TDecoderModel;
begin
  dir := FindInferRefDir('tiny-qwen35');
  if CheckFailed(dir <> '', 'tiny-qwen35 found') then
    exit;
  model := TQwen35Model.Create(dir, fCpu);
  try
    CheckCheckpoints(model, true, 'reference');
  finally
    model.Free;
  end;
  model := TQwen35Model.Create(dir, fFast);
  try
    CheckCheckpoints(model, false, 'fast');
  finally
    model.Free;
  end;
  model := TQwen35Model.Create(dir, fFast, idtQ8_0);
  try
    CheckCheckpoints(model, false, 'fast Q8_0');
  finally
    model.Free;
  end;
end;

procedure TTestInferQwen35.TinyQuantized;
var
  dir: TFileName;
  ref: TInferRefSet;
  f, q: TDecoderModel;
  c: TDecoderContext;
  lf, lq: TSingleDynArray;
  ids: PIntegerArray;
  n, m, v, k, w: PtrInt;
  cmp: TInferCompare;
begin
  dir := FindInferRefDir('tiny-qwen35');
  if CheckFailed(dir <> '', 'tiny-qwen35 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  f := nil;
  q := nil;
  try
    f := TQwen35Model.Create(dir, fFast);
    v := f.Config.VocabSize;
    for w := 0 to 1 do
    begin
      FreeAndNil(q);
      if w = 0 then
        q := TQwen35Model.Create(dir, fFast, idtQ8_0)
      else
        q := TQwen35Model.Create(dir, fFast, idtQ4_0);
      Check((q.Layers[0].LinQkv.DType = q.Weights) and (q.Layers[0].LinB.DType = idtF32) and
        (q.Layers[0].LinA.DType = idtF32) and (q.Head.DType = idtQ8_0),
        'the types of the quantized weights');
      for k := 0 to 2 do
      begin
        ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
        SetLength(lf, n * v);
        SetLength(lq, n * v);
        c := f.NewContext;
        try
          f.Forward(c, ids, n, pointer(lf), true);
        finally
          c.Free;
        end;
        if w = 0 then
        begin
          cmp := InferCompare(pointer(lf), ref.F32(FormatUtf8('case%.logits', [k]), m),
            n * v, 1e-4, 1e-4);
          CheckUtf8(InferCompareOk(cmp), 'case % F32 fast path: %',
            [k, InferCompareText(cmp)]);
        end;
        c := q.NewContext;
        try
          q.Forward(c, ids, n, pointer(lq), true);
        finally
          c.Free;
        end;
        // the tiny model is chaotic (sharp attention, a state over every
        // token), its error grows with the tokens - measured 2026-10-21: Q8_0
        // max 0.28, cosine 0.9958 (case 2); Q4_0 max 2.63, cosine 0.753 (case 2)
        cmp := InferCompare(pointer(lq), pointer(lf), n * v, 0.5 + 2.7 * w, 0);
        AddConsole(FormatString('Qwen3.5 tiny case % % vs FP32: %',
          [k, INFER_DTYPE_NAME[q.Weights], InferCompareText(cmp)]));
        CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.99 - w * 0.26),
          'case % % vs FP32: %', [k, INFER_DTYPE_NAME[q.Weights], InferCompareText(cmp)]);
      end;
    end;
  finally
    q.Free;
    f.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen35.Refused;
var
  dir, tmp: TFileName;
  cfg: TDocVariantData;

  function Refused(const aPath: RawUtf8; const aValue: variant;
    const aWhy: RawUtf8): boolean;
  var
    c: TDocVariantData;
    why: RawUtf8;
  begin
    c.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    c.SetValueByPath(aPath, aValue, true, '.');
    FileFromString(c.ToJson, tmp + 'config.json');
    why := 'accepted';
    try
      TQwen35Model.Create(tmp, fCpu).Free;
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    result := PosEx(aWhy, why) > 0;
    if not result then
      AddConsole(FormatString('% = %: "%"', [aPath, aValue, why]));
  end;

  function Without(const aPath, aWhy: RawUtf8): boolean;
  var
    c: TDocVariantData;
    o: PDocVariantData;
    why: RawUtf8;
  begin
    c.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    o := c.O['text_config'];
    o^.Delete(aPath);
    FileFromString(c.ToJson, tmp + 'config.json');
    why := 'accepted';
    try
      TQwen35Model.Create(tmp, fCpu).Free;
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    result := PosEx(aWhy, why) > 0;
    if not result then
      AddConsole(FormatString('without %: "%"', [aPath, why]));
  end;

begin
  dir := FindInferRefDir('tiny-qwen35');
  if CheckFailed(dir <> '', 'tiny-qwen35 found') then
    exit;
  tmp := MakePath([GetSystemPath(spTemp), 'lai-d7-' + Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  try
    cfg.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    CopyFile(dir + 'model.safetensors', tmp + 'model.safetensors', false);
    // the copy itself loads
    FileFromString(cfg.ToJson, tmp + 'config.json');
    TQwen35Model.Create(tmp, fCpu).Free;
    Check(Refused('text_config.hidden_act', 'gelu', 'hidden_act'), 'another activation');
    Check(Refused('text_config.attention_bias', true, 'attention_bias'), 'a bias');
    Check(Refused('text_config.attn_output_gate', false, 'output gate'),
      'attention without its gate');
    Check(Refused('text_config.mlp_only_layers', _Arr([1]), 'mlp_only_layers'),
      'mlp_only_layers');
    Check(Refused('text_config.rope_parameters.rope_type', 'yarn', 'rope_type'),
      'another rope');
    Check(Refused('text_config.rope_parameters.factor', 2, 'rope setting'),
      'an unknown rope setting');
    Check(Refused('text_config.rope_parameters.partial_rotary_factor', 0.3,
      'rotating values'), 'an odd rotating part (32 * 0.3 = 9)');
    Check(Refused('text_config.rope_parameters.partial_rotary_factor', 1.5,
      'partial_rotary_factor'), 'more than the head');
    Check(Refused('text_config.layer_types', _Arr(['linear_attention',
      'sliding_attention', 'linear_attention', 'full_attention']), 'layer type'),
      'another layer type');
    Check(Refused('text_config.layer_types', _Arr(['full_attention']), 'layer types'),
      'layer types of another count');
    Check(Refused('text_config.full_attention_interval', 0, 'interval'),
      'an interval of 0');
    Check(Refused('text_config.linear_num_value_heads', 3, 'DeltaNet heads'),
      'value heads no multiple of the key heads');
    Check(Refused('text_config.linear_conv_kernel_dim', 0, 'DeltaNet heads'),
      'no convolution');
    // the shapes the file does not have
    Check(Refused('text_config.linear_value_head_dim', 32, 'shaped'),
      'a value head of another width');
    Check(Refused('text_config.head_dim', 64, 'shaped'), 'a head of another width');
    // the head as the top says: untied wants lm_head.weight
    Check(Refused('tie_word_embeddings', false, 'lm_head.weight'),
      'untied without a head (text_config says tied: ignored)');
    Check(Refused('text_config.tie_word_embeddings', false, 'accepted'),
      'text_config untied, the top tied: tied, it loads');
    Check(Without('rope_parameters', 'rope_theta'), 'no rope_parameters, no theta');
    // full attention only is a valid model: the DeltaNet shapes (defaults) are
    // not refused, the attention tensors of layers 0..2 are missing
    Check(Refused('text_config.full_attention_interval', 1, 'q_proj'),
      'full attention only: settings accepted');
  finally
    DirectoryDeleteAll(tmp);
  end;
end;

procedure TTestInferQwen35.RopeForms;
var
  dir, tmp: TFileName;
  base: TDecoderModel;
  want: TSingleDynArray;
  ids: PIntegerArray;
  ref: TInferRefSet;
  n: PtrInt;

  /// the model of the text folder with aJson as config.json: its RopeDims and
  // whether its logits are the bits of base
  function Same(const aJson: RawUtf8; out aDims: PtrInt): boolean;
  var
    m: TDecoderModel;
    c: TDecoderContext;
    got: TSingleDynArray;
  begin
    FileFromString(aJson, tmp + 'config.json');
    m := TQwen35TextModel.Create(tmp, fCpu);
    try
      aDims := m.Config.RopeDims;
      SetLength(got, length(want));
      c := m.NewContext;
      try
        m.Forward(c, ids, n, pointer(got));
      finally
        c.Free;
      end;
      result := CompareMem(pointer(got), pointer(want), length(want) * SizeOf(single));
    finally
      m.Free;
    end;
  end;

var
  cfg: TDocVariantData;
  rope: PDocVariantData;
  dims: PtrInt;
  c: TDecoderContext;
begin
  dir := FindInferRefDir('tiny-qwen35');
  if CheckFailed(dir <> '', 'tiny-qwen35 found') then
    exit;
  dir := MakePath([dir, 'text'], true);
  tmp := MakePath([GetSystemPath(spTemp), 'lai-d7r-' + Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  ref := TInferRefSet.CreateFromFolder(dir);
  base := nil;
  try
    CopyFile(dir + 'model.safetensors', tmp + 'model.safetensors', false);
    base := TQwen35TextModel.Create(dir, fCpu);
    ids := ref.I32('case1.ids', n);
    SetLength(want, base.Config.VocabSize);
    c := base.NewContext;
    try
      base.Forward(c, ids, n, pointer(want));
    finally
      c.Free;
    end;
    // partial_rotary_factor 0.5 at the top, none in rope_parameters
    cfg.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    // (rope fetched again after every change at the top: the values of cfg
    // may move)
    rope := cfg.O['rope_parameters'];
    rope^.Delete('partial_rotary_factor');
    SetFloat(cfg, 'partial_rotary_factor', 0.5);
    Check(Same(cfg.ToJson, dims) and (dims = 16), 'the factor at the top');
    // both: rope_parameters wins
    SetFloat(cfg, 'partial_rotary_factor', 0.75);
    rope := cfg.O['rope_parameters'];
    SetFloat(rope^, 'partial_rotary_factor', 0.5);
    Check(Same(cfg.ToJson, dims) and (dims = 16), 'rope_parameters before the top');
    // rope_theta at the top
    cfg.Delete('partial_rotary_factor');
    SetFloat(cfg, 'rope_theta', cfg.O['rope_parameters']^.D['rope_theta']);
    rope := cfg.O['rope_parameters'];
    rope^.Delete('rope_theta');
    Check(Same(cfg.ToJson, dims) and (dims = 16), 'rope_theta at the top');
    // the older name rope_scaling
    cfg.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    cfg.AddValue('rope_scaling', cfg.Value['rope_parameters']);
    cfg.Delete('rope_parameters');
    Check(Same(cfg.ToJson, dims) and (dims = 16), 'rope_scaling');
    // an empty rope_scaling beside rope_parameters: as absent (Python's "or")
    cfg.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    cfg.AddValue('rope_scaling', _ObjFast([]));
    Check(Same(cfg.ToJson, dims) and (dims = 16), 'an empty rope_scaling');
    // the older key type in place of rope_type
    cfg.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
    rope := cfg.O['rope_parameters'];
    rope^.Delete('rope_type');
    rope^.AddValue('type', 'default');
    Check(Same(cfg.ToJson, dims) and (dims = 16), 'type for rope_type');
  finally
    base.Free;
    ref.Free;
    DirectoryDeleteAll(tmp);
  end;
end;

procedure TTestInferQwen35.Qwen35Real;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  timer: TPrecisionTimer;
begin
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3.5-0.8B: 3.5 GB in F32 need a 64-bit process');
  exit;
  {$endif CPU32}
  dir := FindVendorModel(QWEN35_MODEL + PathDelim + 'model.safetensors.index.json');
  if dir = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/%', [QWEN35_MODEL]);
    AddConsole('SKIP Qwen3.5-0.8B: no vendor/models/' + QWEN35_MODEL + ' (revision ' +
      'in tests/data/infer-ref/qwen35-0.8b/manifest.json)');
    exit;
  end;
  dir := ExtractFilePath(dir);
  refdir := FindInferRefDir('qwen35-0.8b');
  if CheckFailed(refdir <> '', 'qwen35-0.8b reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  try
    timer.Start;
    model := NewDecoderModel(dir, fFast);
    AddConsole(FormatString('Qwen3.5-0.8B loaded in %', [timer.Stop]));
    Check(model.ClassType = TQwen35Model, 'a TQwen35Model');
    Check((model.Config.Layers = 24) and (model.Config.Spec[3].Mixer = dmAttention) and
      (model.Config.Spec[22].Mixer = dmDeltaNet) and (model.Config.RopeDims = 64),
      'the layers of Qwen3.5-0.8B');
    // measured 2026-10-21 (FPC, x86_64): every state cosine 0.99999999999,
    // logits max 4e-5, greedy identical
    CheckLastRows(model, ref, 'Qwen3.5-0.8B', 0.99999, 0.999999, 1e-3);
    FreeAndNil(model);
    // every matrix in Q8_0 at load (in_proj_b / in_proj_a stay F32)
    timer.Start;
    model := TQwen35Model.Create(dir, fFast, idtQ8_0);
    AddConsole(FormatString('Qwen3.5-0.8B quantized to Q8_0 at load in %',
      [timer.Stop]));
    // measured 2026-10-21 with the float delta rule (the chat case): states
    // cosine 0.9984 (aarch64) / 0.9987 (x86_64), logits cosine 0.9988 / 0.9984
    // and max 0.55 / 0.60, greedy identical - the error of Q8_0 activations,
    // grown over the turn
    CheckLastRows(model, ref, 'Qwen3.5-0.8B Q8_0', 0.997, 0.997, 1);
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen35.TinyGguf;
var
  dir: TFileName;
  ref: TInferRefSet;
  folder, g: TDecoderModel;
  a, b: TDecoderConfig;
  ctx: TDecoderContext;
  ids: PIntegerArray;
  lf, lg: TSingleDynArray;
  n, v, w, k: PtrInt;
  cmp: TInferCompare;
  sub, pre: RawUtf8;
  tiled: boolean;
begin
  dir := FindInferRefDir('tiny-qwen35');
  if CheckFailed(dir <> '', 'tiny-qwen35 found') then
    exit;
  // 0: the published layout (with an mtp layer after it), 1: text (2 key
  // heads read by 4 value heads, tiled in the file)
  for w := 0 to 1 do
  begin
    sub := '';
    pre := '';
    if w = 1 then
    begin
      sub := 'text';
      pre := 'text-';
    end;
    ref := TInferRefSet.CreateFromFolder(MakePath([dir, sub], true));
    folder := nil;
    g := nil;
    try
      folder := NewDecoderModel(MakePath([dir, sub]), fCpu);
      g := NewDecoderModelFromGguf(MakePath([dir, 'gguf', pre + 'bf16.gguf']), fCpu);
      Check(g.ClassType = TQwen35Model, 'qwen35: a TQwen35Model');
      a := folder.Config;
      b := g.Config;
      tiled := b.LinTiled;
      CheckUtf8((a.Layers = b.Layers) and (a.Hidden = b.Hidden) and
        (a.VocabSize = b.VocabSize) and (a.HeadDim = b.HeadDim) and
        (a.RopeDims = b.RopeDims) and (a.RopeTheta = b.RopeTheta) and
        (a.LinKeyHeads = b.LinKeyHeads) and (a.LinValueHeads = b.LinValueHeads) and
        (a.LinKeyDim = b.LinKeyDim) and (a.LinValueDim = b.LinValueDim) and
        (a.LinConv = b.LinConv) and (a.TiedEmbeddings = b.TiedEmbeddings) and
        (a.Intermediate = b.Intermediate) and (a.MaxPositions = b.MaxPositions),
        '% the settings of the folder', [sub]);
      for k := 0 to a.Layers - 1 do
        CheckUtf8((a.Spec[k].Mixer = b.Spec[k].Mixer) and
          (a.Spec[k].OutputGate = b.Spec[k].OutputGate), '% layer % kind', [sub, k]);
      CheckUtf8(tiled = (w = 1), '% tiled %', [sub, tiled]);
      Check((g.Layers[0].LinB.DType = idtF32) and (g.Layers[0].LinNegA^[0] < 0),
        'BF16 widened, ssm_a read as it is');
      // the norms (1 + weight) and -exp(A_log) as the folder computes them:
      // the logits of BF16 match those of the folder within the float sums
      ids := ref.I32('case2.ids', n);
      v := a.VocabSize;
      SetLength(lf, n * v);
      SetLength(lg, n * v);
      ctx := folder.NewContext;
      try
        folder.Forward(ctx, ids, n, pointer(lf), true);
      finally
        ctx.Free;
      end;
      ctx := g.NewContext;
      try
        g.Forward(ctx, ids, n, pointer(lg), true);
      finally
        ctx.Free;
      end;
      cmp := InferCompare(pointer(lg), pointer(lf), n * v, 1e-5, 1e-5);
      AddConsole(FormatString('Qwen3.5 % GGUF BF16 vs folder: %, same bits %',
        [sub, InferCompareText(cmp), CompareMem(pointer(lg), pointer(lf), n * v * 4)]));
      CheckUtf8(InferCompareOk(cmp), '% GGUF BF16 vs the folder: %',
        [sub, InferCompareText(cmp)]);
      CheckTiny(g, ref, 'gguf ' + sub);
      // Q8_0 as llama-quantize writes it (the head, ssm_alpha and ssm_beta too)
      FreeAndNil(g);
      g := TQwen35Model.CreateFromGguf(MakePath([dir, 'gguf', pre + 'q8_0.gguf']), fFast);
      Check((g.Weights = idtQ8_0) and (g.Layers[0].LinB.DType = idtQ8_0) and
        (g.Head.DType = idtQ8_0) and (g.Config.LinTiled = tiled),
        'the types of the Q8_0 file');
      for k := 0 to 2 do
      begin
        ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
        SetLength(lf, n * v);
        SetLength(lg, n * v);
        ctx := folder.NewContext;
        try
          folder.Forward(ctx, ids, n, pointer(lf), true);
        finally
          ctx.Free;
        end;
        ctx := g.NewContext;
        try
          g.Forward(ctx, ids, n, pointer(lg), true);
        finally
          ctx.Free;
        end;
        // as TinyQuantized (Q8_0 at load): the tiny model is chaotic
        cmp := InferCompare(pointer(lg), pointer(lf), n * v, 0.5, 0);
        AddConsole(FormatString('Qwen3.5 % GGUF Q8_0 case % vs F32: %',
          [sub, k, InferCompareText(cmp)]));
        CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.99),
          '% GGUF Q8_0 case % vs F32: %', [sub, k, InferCompareText(cmp)]);
      end;
    finally
      g.Free;
      folder.Free;
      ref.Free;
    end;
  end;
end;

procedure TTestInferQwen35.GgufRefused;
var
  dir: TFileName;

  function Refused(const aFile, aWhy: RawUtf8): boolean;
  var
    why: RawUtf8;
  begin
    why := 'accepted';
    try
      TQwen35Model.CreateFromGguf(dir + Utf8ToString(aFile), fCpu).Free;
    except
      on E: EInferError do
        why := StringToUtf8(E.Message);
    end;
    result := PosEx(aWhy, why) > 0;
    if not result then
      AddConsole(FormatString('%: "%"', [aFile, why]));
  end;

begin
  dir := FindInferRefDir('tiny-qwen35');
  if CheckFailed(dir <> '', 'tiny-qwen35 found') then
    exit;
  dir := MakePath([dir, 'gguf'], true);
  Check(Refused('bad-arch.gguf', 'architecture'), 'another architecture');
  Check(Refused('bad-prism.gguf', 'prism.hadamard.version'), 'a Hadamard basis');
  Check(Refused('bad-inner.gguf', 'inner size'), 'no whole value heads');
  Check(Refused('bad-mtp.gguf', 'prediction layers'), 'only prediction layers');
  Check(Refused('bad-recurrent.gguf', 'recurrent_layers'),
    'recurrent_layers of another count');
  Check(Refused('bad-rope.gguf', 'rotating values'), 'an odd rotary count');
  Check(Refused('bad-scaling.gguf', 'rope scaling'), 'a rope scaling');
  Check(Refused('bad-missing.gguf', 'blk.1.ssm_a'), 'a missing tensor');
  Check(Refused('bad-extra.gguf', 'understood'), 'an extra tensor');
  Check(Refused('bad-ssm-a.gguf', 'no -exp(A_log)'), 'A_log not negated');
end;

procedure TTestInferQwen35.Qwen35Gguf;
const
  FILES: array[0..1] of RawUtf8 = ('qwen3.5-0.8b-q8_0.gguf', 'qwen3.5-0.8b-q4_0.gguf');
var
  fn, refdir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  timer: TPrecisionTimer;
  w: PtrInt;
begin
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3.5-0.8B GGUF: a 64-bit process');
  exit;
  {$endif CPU32}
  refdir := FindInferRefDir('qwen35-0.8b');
  if CheckFailed(refdir <> '', 'qwen35-0.8b reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  try
    for w := 0 to high(FILES) do
    begin
      fn := FindVendorModel(Utf8ToString(FILES[w]));
      if fn = '' then
      begin
        CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
          'LAI_REQUIRE_MODEL=1 but no vendor/models/%', [FILES[w]]);
        AddConsole(FormatString('SKIP Qwen3.5-0.8B GGUF: no vendor/models/% ' +
          '(revision in UPSTREAM_BASE)', [FILES[w]]));
        continue;
      end;
      timer.Start;
      model := NewDecoderModelFromGguf(fn, fFast);
      try
        AddConsole(FormatString('% loaded in %', [FILES[w], timer.Stop]));
        // the Q8_0 file carries the mtp layer (25 blocks), the Q4_0 one not
        Check((model.Config.Layers = 24) and (model.Config.Spec[3].Mixer = dmAttention) and
          (model.Config.Spec[22].Mixer = dmDeltaNet) and (model.Config.RopeDims = 64) and
          not model.Config.LinTiled and model.Config.TiedEmbeddings,
          'the layers of Qwen3.5-0.8B');
        // measured 2026-11-11 (FPC, aarch64): Q8_0 states cosine >= 0.9983,
        // logits >= 0.9989 and max 0.64, greedy as the reference where it was
        // sure (as Q8_0 at load); Q4_0 cases 0..2 states >= 0.961, logits >=
        // 0.981 and max 3.03 (x86_64: 0.960, 0.981, 3.01) - the chat case 3 drifts off (logits cosine 0.64),
        // as Q4_0 at load from the folder does (0.66): the error of Q4_0 on
        // a 0.8B model, not of the file
        if w = 0 then
          CheckLastRows(model, ref, 'Qwen3.5-0.8B GGUF Q8_0', 0.997, 0.997, 1)
        else
          CheckLastRows(model, ref, 'Qwen3.5-0.8B GGUF Q4_0', 0.95, 0.97, 3.5, 3);
      finally
        model.Free;
      end;
    end;
  finally
    ref.Free;
  end;
end;

procedure TTestInferQwen35.Qwen38Gguf;
const
  FILES: array[0..1] of RawUtf8 = ('qwen3.8-27b-q8_0.gguf', 'qwen3.8-27b-q4_k_m.gguf');
var
  fn, refdir: TFileName;
  ref: TInferRefSet;
  model: TDecoderModel;
  timer: TPrecisionTimer;
  w: PtrInt;
begin
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3.8-27B GGUF: a 64-bit process');
  exit;
  {$endif CPU32}
  refdir := FindInferRefDir('qwen38-27b');
  if CheckFailed(refdir <> '', 'qwen38-27b reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  try
    for w := 0 to high(FILES) do
    begin
      fn := FindVendorModel(Utf8ToString(FILES[w]));
      if fn = '' then
      begin
        AddConsole(FormatString('SKIP Qwen3.8-27B GGUF: no vendor/models/% ' +
          '(revision in UPSTREAM_BASE)', [FILES[w]]));
        continue;
      end;
      timer.Start;
      model := NewDecoderModelFromGguf(fn, fFast);
      try
        AddConsole(FormatString('% loaded in %', [FILES[w], timer.Stop]));
        Check((model.Config.Layers = 64) and (model.Config.Spec[3].Mixer = dmAttention) and
          (model.Config.Spec[62].Mixer = dmDeltaNet) and (model.Config.RopeDims = 64) and
          (model.Config.HeadDim = 256) and (model.Config.Heads = 24) and
          (model.Config.KvHeads = 4) and (model.Config.LinKeyHeads = 16) and
          (model.Config.LinValueHeads = 48) and model.Config.LinTiled and
          (model.LinGroup = 1) and not model.Config.TiedEmbeddings,
          'the layers of Qwen3.8-27B');
        // measured 2026-11-11 (FPC, x86_64, the RTX computer): Q8_0 states
        // cosine >= 0.9983, logits >= 0.9987 and max 0.72 (the chat case 3;
        // plain text 0.9999), greedy as the reference where it was sure;
        // Q4_K_M plain text states >= 0.995, logits >= 0.997 and max 1.0, the
        // chat case states 0.951, logits 0.967 and max 4.6 with every greedy
        // token equal - CUDA and Vulkan with float activations land at 0.9695
        // and 4.43 there too (TTestInferCuda.Qwen38OnDevice): the error of
        // the Q4_K weights of the file, not of our arithmetic
        if w = 0 then
          CheckLastRows(model, ref, 'Qwen3.8-27B GGUF Q8_0', 0.997, 0.997, 1.5)
        else
          CheckLastRows(model, ref, 'Qwen3.8-27B GGUF Q4_K_M', 0.94, 0.96, 5.5);
      finally
        model.Free;
      end;
    end;
  finally
    ref.Free;
  end;
end;

end.
