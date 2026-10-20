/// tests for the reference data of the native inference engine (phase 0)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - phase 0 has no engine yet: these tests prove the TEST HARNESS - the
//   committed datasets are intact, corrupted data is rejected, InferCompare
//   catches what it must (scale, single outliers, NaN), and the references are
//   internally consistent (CLS = row 0 of the last state, embedding = CLS / norm,
//   position ids, batch rows = single runs). They confirm no Pascal forward.
// - data: tests/data/infer-ref/, written by tools/infer-ref/make_reference.py
unit test.infer;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.test,
  mormot.crypt.core,
  test.infer.ref;

type
  /// phase 0: reference data, loader and comparison
  TTestInferReference = class(TSynTestCase)
  protected
    function LoadSet(const aDataset: RawUtf8): TInferRefSet;
    /// ids, positions, state shapes, CLS and embedding of one run
    procedure CheckRun(aSet: TInferRefSet; const aPrefix: RawUtf8;
      aPad, aDim, aLayers: PtrInt);
    /// batch ids, mask and positions: [rows, width], mask = ids <> pad,
    // positions per row as in CheckRun
    procedure CheckBatchInputs(aSet: TInferRefSet; aPad: PtrInt);
    /// row aRow of a batch tensor against the single-run tensor
    procedure CheckBatchRow(aSet: TInferRefSet; const aBatch, aSingle: RawUtf8;
      aRow, aDim: PtrInt; aAtol, aRtol: double);
  published
    /// every dataset loads, and every tensor matches its sha256
    procedure DatasetsAreIntact;
    /// corrupted blobs and manifests are refused, not silently read
    procedure CorruptionIsRejected;
    /// InferCompare fails on scale, a single outlier and NaN
    procedure CompareCatchesErrors;
    /// the tiny XLM-R fixture is internally consistent
    procedure TinyFixtureIsConsistent;
    /// the BGE-M3 reference is internally consistent and matches the lembed oracle
    procedure BgeM3ReferenceIsConsistent;
    /// the operator fixtures satisfy the properties of their operator
    procedure OperatorFixturesAreSound;
  end;


implementation

const
  DATASETS: array[0..11] of RawUtf8 = (
    'ops', 'tiny-xlmr', 'bge-m3', 'bge-m3-embed-de', 'bge-m3-tokens',
    'q8', 'q4', 'q5', 'mxfp4', 'qk', 'tiny-gguf', 'lembed-tokens');

/// v / ||v|| in double, stored as single
procedure Normalized(aValues: PSingleArray; aCount: PtrInt;
  var aResult: TSingleDynArray);
var
  i: PtrInt;
  n: double;
begin
  n := InferL2Norm(aValues, aCount);
  SetLength(aResult, aCount);
  for i := 0 to aCount - 1 do
    aResult[i] := aValues^[i] / n;
end;

/// the manifest and blob of a dataset as raw content, for tampering
procedure ReadRaw(const aDataset: RawUtf8; out aJson: RawUtf8;
  out aBlob: RawByteString);
var
  dir: TFileName;
begin
  dir := FindInferRefDir(aDataset);
  aJson := RawUtf8FromFile(dir + 'manifest.json');
  aBlob := StringFromFile(dir + 'tensors.bin');
end;

/// true if TInferRefSet.Create refuses the content with EInferRef whose
// message contains aReason ('' = any reason)
function Rejects(const aJson: RawUtf8; const aBlob: RawByteString;
  const aReason: RawUtf8 = ''): boolean;
var
  s: TInferRefSet;
begin
  result := false;
  try
    s := TInferRefSet.Create(aJson, aBlob);
    s.Free;
  except
    on E: EInferRef do
      result := (aReason = '') or (PosEx(aReason, StringToUtf8(E.Message)) > 0);
  end;
end;


{ TTestInferReference }

function TTestInferReference.LoadSet(const aDataset: RawUtf8): TInferRefSet;
var
  dir: TFileName;
begin
  dir := FindInferRefDir(aDataset);
  if dir = '' then
    EInferRef.RaiseUtf8('dataset tests/data/infer-ref/% not found', [aDataset]);
  result := TInferRefSet.CreateFromFolder(dir);
end;

procedure TTestInferReference.CheckRun(aSet: TInferRefSet;
  const aPrefix: RawUtf8; aPad, aDim, aLayers: PtrInt);
var
  ids, pos: PIntegerArray;
  last, cls, emb: PSingleArray;
  n, np, cn, en, i, k, seen: PtrInt;
  t: PInferRefTensor;
  norm: TSingleDynArray;
  cmp: TInferCompare;
  posOk, shapeOk: boolean;
begin
  ids := aSet.I32(aPrefix + 'input_ids', n);
  pos := aSet.I32(aPrefix + 'position_ids', np);
  CheckEqual(np, n, aPrefix + 'one position per token');
  // XLM-R: real tokens count from pad + 1, padding sits at pad
  posOk := true;
  seen := 0;
  for i := 0 to n - 1 do
    if ids^[i] = aPad then
      posOk := posOk and (pos^[i] = aPad)
    else
    begin
      inc(seen);
      posOk := posOk and (pos^[i] = seen + aPad);
    end;
  CheckUtf8(posOk, aPrefix + 'position ids follow cumsum(ids <> pad) + pad');
  shapeOk := true;
  for k := 0 to aLayers do
  begin
    t := aSet.Tensor(FormatUtf8('%hidden.%', [aPrefix, k]));
    shapeOk := shapeOk and (length(t^.Shape) = 2) and (t^.Shape[0] = n) and
      (t^.Shape[1] = aDim);
  end;
  CheckUtf8(shapeOk, aPrefix + 'every hidden state is [tokens, dim]');
  last := aSet.F32(FormatUtf8('%hidden.%', [aPrefix, aLayers]), i);
  cls := aSet.F32(aPrefix + 'cls', cn);
  emb := aSet.F32(aPrefix + 'embedding', en);
  CheckEqual(cn, aDim, aPrefix + 'cls size');
  CheckEqual(en, aDim, aPrefix + 'embedding size');
  CheckUtf8(CompareMem(cls, last, aDim * SizeOf(single)),
    aPrefix + 'cls is row 0 of the last hidden state');
  Normalized(cls, aDim, norm);
  cmp := InferCompare(emb, pointer(norm), aDim, 1e-6, 1e-5);
  CheckUtf8(InferCompareOk(cmp), '%embedding = cls / norm: %',
    [aPrefix, InferCompareText(cmp)]);
  CheckUtf8(Abs(InferL2Norm(emb, aDim) - 1) < 1e-6, aPrefix + 'unit norm');
end;

procedure TTestInferReference.CheckBatchInputs(aSet: TInferRefSet; aPad: PtrInt);
var
  ids, mask, pos: PIntegerArray;
  n, rows, width, r, c, seen: PtrInt;
  t: PInferRefTensor;
  ok: boolean;
begin
  t := aSet.Tensor('batch/input_ids');
  if CheckFailed(length(t^.Shape) = 2, 'batch ids are [rows, width]') then
    exit;
  rows := t^.Shape[0];
  width := t^.Shape[1];
  ids := aSet.I32('batch/input_ids', n);
  mask := aSet.I32('batch/attention_mask', c);
  CheckEqual(c, n, 'one mask entry per id');
  pos := aSet.I32('batch/position_ids', c);
  CheckEqual(c, n, 'one position per id');
  ok := true;
  for r := 0 to rows - 1 do
  begin
    seen := 0;
    for c := 0 to width - 1 do
      if ids^[r * width + c] = aPad then
        ok := ok and (mask^[r * width + c] = 0) and (pos^[r * width + c] = aPad)
      else
      begin
        inc(seen);
        ok := ok and (mask^[r * width + c] = 1) and
          (pos^[r * width + c] = seen + aPad);
      end;
  end;
  CheckUtf8(ok, 'batch mask = ids <> pad, positions per row');
end;

procedure TTestInferReference.CheckBatchRow(aSet: TInferRefSet;
  const aBatch, aSingle: RawUtf8; aRow, aDim: PtrInt; aAtol, aRtol: double);
var
  b, s: PSingleArray;
  bn, sn: PtrInt;
  cmp: TInferCompare;
begin
  b := aSet.F32(aBatch, bn);
  s := aSet.F32(aSingle, sn);
  if CheckFailed(bn >= (aRow + 1) * sn, Utf8ToString(aBatch + ' holds the row')) then
    exit;
  cmp := InferCompare(@b^[aRow * sn], s, sn, aAtol, aRtol);
  CheckUtf8(InferCompareOk(cmp), '% row % = %: %',
    [aBatch, aRow, aSingle, InferCompareText(cmp)]);
end;

procedure TTestInferReference.DatasetsAreIntact;
var
  i: PtrInt;
  s: TInferRefSet;
  bad: RawUtf8;
  dir: TFileName;
  w: PDocVariantData;
begin
  for i := 0 to high(DATASETS) do
  begin
    dir := FindInferRefDir(DATASETS[i]);
    if CheckFailed(dir <> '', Utf8ToString(DATASETS[i] + ' found')) then
      continue;
    // a damaged dataset is one failed check - the others are still verified
    s := nil;
    try
      s := TInferRefSet.CreateFromFolder(dir);
    except
      on E: EInferRef do
      begin
        CheckUtf8(false, '%: %', [DATASETS[i], E.Message]);
        continue;
      end;
    end;
    try
      CheckUtf8(length(s.Tensors) > 0, DATASETS[i] + ' has tensors');
      CheckUtf8(s.VerifyTensorHashes(bad), '% tensor hashes (first bad: %)',
        [DATASETS[i], bad]);
      if DATASETS[i] = 'tiny-xlmr' then
      begin
        // the weights are the fixture contract - the seed only documents them
        w := s.Manifest^.O['weights'];
        CheckEqual(Sha256(StringFromFile(dir + 'model.safetensors')),
          w^.U['sha256'], 'tiny model.safetensors sha256');
        CheckEqual(Sha256(StringFromFile(dir + 'config.json')),
          s.Manifest^.U['config_sha256'], 'tiny config.json sha256');
      end
      else if DATASETS[i] = 'tiny-gguf' then
        CheckEqual(Sha256(StringFromFile(dir + 'model.gguf')),
          s.Manifest^.O['model']^.U['sha256'], 'tiny model.gguf sha256');
    finally
      s.Free;
    end;
  end;
end;

procedure TTestInferReference.CorruptionIsRejected;
var
  json, bad: RawUtf8;
  blob, tampered: RawByteString;
  doc: TDocVariantData;
  s: TInferRefSet;
  t: PDocVariantData;
  victim: PtrInt;
begin
  ReadRaw('ops', json, blob);
  if CheckFailed((json <> '') and (blob <> ''), 'ops dataset readable') then
    exit;
  CheckUtf8(not Rejects(json, blob), 'intact dataset accepted');
  // one flipped byte inside the tensor 'gelu/x'
  s := TInferRefSet.Create(json, blob);
  try
    victim := s.Tensor('gelu/x')^.Offset + 5;
  finally
    s.Free;
  end;
  SetString(tampered, PAnsiChar(pointer(blob)), length(blob)); // own copy
  PByteArray(tampered)^[victim] := PByteArray(tampered)^[victim] xor $40;
  CheckUtf8(Rejects(json, tampered), 'blob hash catches a flipped byte');
  // same flip with a blob hash that matches: the tensor hash must name the tensor
  doc.InitJson(json, JSON_FAST_FLOAT);
  doc.O['blob']^.AddOrUpdateValue('sha256', Sha256(tampered));
  s := TInferRefSet.Create(doc.ToJson, tampered);
  try
    CheckUtf8(not s.VerifyTensorHashes(bad), 'tensor hash catches the flip');
    CheckEqual(bad, 'gelu/x', 'the corrupted tensor is named');
  finally
    s.Free;
  end;
  doc.Clear;
  // a tensor outside the blob
  doc.InitJson(json, JSON_FAST_FLOAT);
  t := _Safe(doc.A['tensors']^.Values[0]);
  t^.AddOrUpdateValue('offset', length(blob));
  CheckUtf8(Rejects(doc.ToJson, blob), 'offset past the end refused');
  doc.Clear;
  // a shape whose product would overflow before any allocation
  doc.InitJson(json, JSON_FAST_FLOAT);
  t := _Safe(doc.A['tensors']^.Values[0]);
  t^.AddOrUpdateValue('shape', _Arr([Int64(1) shl 40, Int64(1) shl 40]));
  // the overflow guard itself must fire - the byte-length check would also refuse
  // a wrapped product, and hide a missing guard
  CheckUtf8(Rejects(doc.ToJson, blob, 'shape larger than the blob'),
    'huge shape refused');
  doc.Clear;
  // length that does not match the shape
  doc.InitJson(json, JSON_FAST_FLOAT);
  t := _Safe(doc.A['tensors']^.Values[0]);
  t^.AddOrUpdateValue('bytes', t^.I['bytes'] - 4);
  CheckUtf8(Rejects(doc.ToJson, blob), 'byte length must equal shape * 4');
  doc.Clear;
  // an element type the loader does not know
  doc.InitJson(json, JSON_FAST_FLOAT);
  t := _Safe(doc.A['tensors']^.Values[0]);
  t^.AddOrUpdateValue('dtype', 'f16');
  CheckUtf8(Rejects(doc.ToJson, blob), 'unknown dtype refused');
  doc.Clear;
  // a blob that is shorter than the manifest says
  CheckUtf8(Rejects(json, copy(blob, 1, length(blob) - 4)), 'truncated blob refused');
end;

procedure TTestInferReference.CompareCatchesErrors;
const
  N = 8;
var
  ref, act: array[0..N - 1] of single;
  i: PtrInt;
  c: TInferCompare;
begin
  for i := 0 to N - 1 do
    ref[i] := (i - 3.5) / 4;
  // identical
  MoveFast(ref, act, SizeOf(ref));
  c := InferCompare(@act, @ref, N, 1e-6, 1e-5);
  CheckUtf8(InferCompareOk(c), 'identical: ' + InferCompareText(c));
  CheckUtf8(c.MaxAbs = 0, 'identical: no difference');
  CheckUtf8(Abs(c.Cosine - 1) < 1e-12, 'identical: cosine 1');
  // a scale error: cosine stays 1, the elementwise check and the norms do not
  for i := 0 to N - 1 do
    act[i] := ref[i] * 1.001;
  c := InferCompare(@act, @ref, N, 1e-6, 1e-5);
  CheckUtf8(Abs(c.Cosine - 1) < 1e-9, 'scaled: cosine alone would pass');
  CheckUtf8(not InferCompareOk(c), 'scaled: the elementwise check fails');
  CheckUtf8(Abs(c.NormActual / c.NormReference - 1.001) < 1e-6, 'scaled: norm ratio');
  // one outlier is found and located
  MoveFast(ref, act, SizeOf(ref));
  act[5] := act[5] + 0.01;
  c := InferCompare(@act, @ref, N, 1e-6, 1e-5);
  CheckEqual(c.Failed, 1, 'outlier: one element fails');
  CheckEqual(c.Worst, 5, 'outlier: located');
  CheckUtf8(Abs(c.MaxAbs - 0.01) < 1e-6, 'outlier: max abs');
  // NaN fails and outranks the outlier, without an FPU exception
  PCardinal(@act[2])^ := $7fc00000;
  CheckUtf8(IsNonFiniteSingle(act[2]), 'NaN detected by its bits');
  c := InferCompare(@act, @ref, N, 1e-6, 1e-5);
  CheckEqual(c.NonFinite, 1, 'NaN counted');
  CheckEqual(c.Worst, 2, 'NaN is the worst element');
  CheckUtf8(not InferCompareOk(c), 'NaN fails');
  // +Inf in the reference is caught as well
  MoveFast(ref, act, SizeOf(ref));
  PCardinal(@ref[0])^ := $7f800000;
  c := InferCompare(@act, @ref, N, 1e-6, 1e-5);
  CheckEqual(c.NonFinite, 1, 'Inf in the reference counted');
  // zero vectors: equal, no division, cosine reported as 0
  FillCharFast(act, SizeOf(act), 0);
  FillCharFast(ref, SizeOf(ref), 0);
  c := InferCompare(@act, @ref, N, 1e-6, 1e-5);
  CheckUtf8(InferCompareOk(c), 'zero vectors equal');
  CheckUtf8(c.Cosine = 0, 'zero vectors: cosine 0, not NaN');
end;

procedure TTestInferReference.TinyFixtureIsConsistent;
var
  s: TInferRefSet;
  cfg: TDocVariantData;
  dim, layers, pad, i: PtrInt;
  prefix: RawUtf8;
begin
  s := LoadSet('tiny-xlmr');
  try
    cfg.InitJsonFromFile(FindInferRefDir('tiny-xlmr') + 'config.json', JSON_FAST);
    dim := cfg.I['hidden_size'];
    layers := cfg.I['num_hidden_layers'];
    pad := cfg.I['pad_token_id'];
    CheckEqual(pad, 1, 'XLM-R pad id');
    CheckEqual(cfg.U['hidden_act'], 'gelu', 'exact GELU');
    CheckBatchInputs(s, pad);
    for i := 0 to 2 do
    begin
      prefix := FormatUtf8('single/%/', [i]);
      CheckRun(s, prefix, pad, dim, layers);
      // padded batch: each row equals its single run on the real tokens
      CheckBatchRow(s, 'batch/embedding', prefix + 'embedding', i, dim, 1e-5, 1e-4);
    end;
  finally
    s.Free;
  end;
end;

procedure TTestInferReference.BgeM3ReferenceIsConsistent;
var
  s, de: TInferRefSet;
  cfg, cases: PDocVariantData;
  dim, layers, pad, i, n, rows: PtrInt;
  fp32: PSingleArray;
  q8: RawByteString;
  refdir: TFileName;
  cmp: TInferCompare;
  cosMin, cosSum: double;
  normOk: boolean;
  nonFinite: PtrInt;
begin
  s := LoadSet('bge-m3');
  try
    cfg := s.Manifest^.O['model']^.O['config'];
    dim := cfg^.I['hidden_size'];
    layers := cfg^.I['num_hidden_layers'];
    pad := cfg^.I['pad_token_id'];
    CheckEqual(dim, 1024, 'BGE-M3 width');
    CheckEqual(layers, 24, 'BGE-M3 depth');
    cases := s.Manifest^.A['cases'];
    CheckUtf8(cases^.Count >= 6, 'reference cases');
    for i := 0 to cases^.Count - 1 do
      CheckRun(s, FormatUtf8('case/%/', [_Safe(cases^.Values[i])^.U['id']]),
        pad, dim, layers);
    // row 0 is the 'trade' case (12 tokens, unpadded); row 1 is padded and has
    // its own single run - measured 2026-10-14: max abs diff 1.3e-7
    CheckBatchInputs(s, pad);
    CheckBatchRow(s, 'batch/embedding', 'case/trade/embedding', 0, dim, 1e-5, 1e-4);
    CheckBatchRow(s, 'batch/embedding', 'batch/single.1/embedding', 1, dim,
      1e-5, 1e-4);
  finally
    s.Free;
  end;
  // FP32 corpus vectors: unit length, and close to the lembed Q8 oracle
  de := LoadSet('bge-m3-embed-de');
  try
    fp32 := de.F32('docs', n);
    rows := n div 1024;
    refdir := FindInferRefDir('bge-m3-embed-de');
    refdir := ExtractFileDir(ExtractFileDir(ExcludeTrailingPathDelimiter(refdir)));
    q8 := StringFromFile(MakePath([refdir, 'embed-de', 'reference', 'docs.f32']));
    if CheckFailed(length(q8) = n * SizeOf(single),
       'lembed oracle docs.f32 has the same rows') then
      exit;
    normOk := true;
    nonFinite := 0;
    cosMin := 1;
    cosSum := 0;
    for i := 0 to rows - 1 do
    begin
      normOk := normOk and (Abs(InferL2Norm(@fp32^[i * 1024], 1024) - 1) < 1e-5);
      cmp := InferCompare(@fp32^[i * 1024], @PSingleArray(q8)^[i * 1024], 1024, 1, 0);
      inc(nonFinite, cmp.NonFinite); // a NaN would be skipped by the cosine
      if cmp.Cosine < cosMin then
        cosMin := cmp.Cosine;
      cosSum := cosSum + cmp.Cosine;
    end;
    CheckUtf8(normOk, 'FP32 corpus vectors have unit norm');
    CheckEqual(nonFinite, 0, 'no NaN/Inf in the FP32 vectors or the Q8 oracle');
    // Q8_0 weights cost a little precision, never the direction (measured
    // 2026-10-14: min 0.99868, mean 0.99929)
    CheckUtf8(cosMin > 0.998, 'FP32 vs lembed Q8: min cosine % (mean %) over % docs',
      [cosMin, cosSum / rows, rows]);
  finally
    de.Free;
  end;
end;

procedure TTestInferReference.OperatorFixturesAreSound;
var
  s: TInferRefSet;
  y, x, g, b: PSingleArray;
  keep: PIntegerArray;
  n, r, c, i: PtrInt;
  sum, mean, variance, z: double;
  ok: boolean;
begin
  s := LoadSet('ops');
  try
    // masked softmax: dropped entries exactly 0, rows sum to 1
    y := s.F32('softmax/y', n);
    keep := s.I32('softmax/keep', i);
    ok := true;
    for r := 0 to 2 do
    begin
      sum := 0;
      for c := 0 to 5 do
      begin
        if keep^[r * 6 + c] = 0 then
          ok := ok and (y^[r * 6 + c] = 0);
        sum := sum + y^[r * 6 + c];
      end;
      ok := ok and (Abs(sum - 1) < 1e-6);
    end;
    CheckUtf8(ok, 'softmax rows sum to 1, masked entries are 0');
    // L2: every row has unit norm, including the 1e-20-scaled one
    y := s.F32('l2/y', n);
    ok := true;
    for r := 0 to 2 do
      ok := ok and (Abs(InferL2Norm(@y^[r * 7], 7) - 1) < 1e-6);
    CheckUtf8(ok, 'l2 rows have unit norm');
    x := s.F32('l2/x', n);
    CheckUtf8(InferL2Norm(@x^[14], 7) > 0, 'the tiny row is not zero in double');
    // layernorm: (y - beta) / gamma has mean 0 and variance 1 on regular rows;
    // a constant row gives exactly beta
    y := s.F32('layernorm/y', n);
    g := s.F32('layernorm/gamma', i);
    b := s.F32('layernorm/beta', i);
    ok := true;
    for r := 0 to 1 do
    begin
      mean := 0;
      variance := 0;
      for c := 0 to 7 do
        mean := mean + (y^[r * 8 + c] - b^[c]) / g^[c];
      mean := mean / 8;
      for c := 0 to 7 do
      begin
        z := (y^[r * 8 + c] - b^[c]) / g^[c] - mean;
        variance := variance + z * z;
      end;
      variance := variance / 8;
      ok := ok and (Abs(mean) < 1e-5) and (Abs(variance - 1) < 1e-3);
    end;
    CheckUtf8(ok, 'layernorm normalises regular rows');
    ok := true;
    for c := 0 to 7 do
      ok := ok and (Abs(y^[3 * 8 + c] - b^[c]) < 1e-6);
    CheckUtf8(ok, 'layernorm of a constant row is beta');
    // exact GELU at its fixed points
    x := s.F32('gelu/x', n);
    y := s.F32('gelu/y', i);
    ok := true;
    for i := 0 to n - 1 do
      if x^[i] = 0 then
        ok := ok and (y^[i] = 0)
      else if x^[i] >= 10 then
        ok := ok and (Abs(y^[i] - x^[i]) < 1e-5)
      else if x^[i] <= -10 then
        ok := ok and (Abs(y^[i]) < 1e-6);
    CheckUtf8(ok, 'gelu: 0 at 0, identity for large x, 0 for very negative x');
    // linear: y = x W^T + b recomputed in double
    x := s.F32('linear/x', n);
    g := s.F32('linear/weight', i);
    b := s.F32('linear/bias', i);
    y := s.F32('linear/y', i);
    ok := true;
    for r := 0 to 2 do
      for c := 0 to 3 do
      begin
        sum := b^[c];
        for i := 0 to 4 do
          sum := sum + x^[r * 5 + i] * g^[c * 5 + i];
        ok := ok and (Abs(y^[r * 4 + c] - sum) < 1e-5);
      end;
    CheckUtf8(ok, 'linear y = x W^T + b (weight used transposed)');
    CheckEqual(length(s.Tensor('linear/y')^.Shape), 2, 'linear y is a matrix');
    CheckEqual(s.Tensor('linear/y')^.Shape[0], 3, 'linear rows');
    CheckEqual(s.Tensor('linear/y')^.Shape[1], s.Tensor('linear/weight')^.Shape[0],
      'linear y columns = weight rows (used transposed)');
  finally
    s.Free;
  end;
end;

end.
