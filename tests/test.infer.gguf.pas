/// tests for the GGUF / Q8_0 path of the native inference engine (phase 1b)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - Q8_0 arithmetic against fixtures made by ggml itself (the build of the lembed
//   oracle), the GGUF reader against a tiny file and damaged variants of it, the
//   tiny XLM-R as GGUF against a forward with ggml's kernels
// - the real gate needs vendor/models/bge-m3-q8_0.gguf (or $LAI_GGUF), the file of
//   reference contract v1: token ids against SentencePiece and lembed, vectors
//   against tests/data/embed-de/reference (lembed). Says SKIP without the file,
//   LAI_REQUIRE_MODEL=1 makes that a failure. The full corpus with nDCG@10 is
//   slow on the scalar reference: it runs with LAI_INFER_GATE=1 only, a few short
//   texts run always
unit test.infer.gguf;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.buffers,
  mormot.core.variants,
  mormot.core.perf,
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.infer.loader.gguf,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.tokenizer.ugm,
  mormot.ai.infer.model.xlmr,
  mormot.ai.infer.pipeline.embed,
  test.infer.ref;

type
  /// phase 1b: GGUF reader, Q8_0 kernels, BGE-M3 Q8 against lembed
  TTestInferGguf = class(TSynTestCase)
  protected
    fCpu: TInferCpuBackend;
    fFast: TInferCpuFastBackend;
    fHiddenSize: PtrInt;
    fStates: array of TSingleDynArray;
    procedure Setup; override;
    procedure CleanUp; override;
    procedure KeepState(aState: PtrInt; aValues: PSingleArray;
      aRows, aWidth: PtrInt);
    /// the real Q8 file, '' with SKIP (or a failure if LAI_REQUIRE_MODEL=1)
    function RealModel: TFileName;
    /// embed corpus texts and compare with the lembed vectors; aAll = whole
    // corpus plus nDCG@10, otherwise the aShort shortest docs and queries
    procedure CheckCorpus(aAll: boolean; aShort: PtrInt; aBackend: TInferBackend;
      const aName: RawUtf8);
  published
    /// FP16 rounding like ggml_fp32_to_fp16, and every half round-trips
    procedure HalfFloatLikeGgml;
    /// activation rows quantize to the very bytes ggml-cpu produces
    procedure QuantizeLikeGgml;
    /// the Q8_0 Linear against ggml_vec_dot_q8_0_q8_0, and dequantization
    procedure LinearQ8LikeGgml;
    /// Q4_0 weights quantize to the bytes of ggml's reference quantizer,
    // dequantize exactly and give the Q4_0 Linear of ggml-cpu; Q8_0 weights
    // quantize like ggml's reference quantizer (ties away from zero)
    procedure Q4LikeGgml;
    /// Q5_0 weights of ggml's reference quantizer dequantize exactly and give
    // the Q5_0 Linear of ggml-cpu
    procedure Q5LikeGgml;
    /// Q8_K activations, Q4_K and Q6_K dequantization and their Linear: the
    // bytes and sums of ggml itself
    procedure KQuantsLikeGgml;
    /// the tiny GGUF: metadata, tensors, typed access
    procedure ReaderReadsTinyFile;
    /// truncated, malformed and hostile files are refused, never read past
    procedure ReaderRefusesDamagedFiles;
    /// tiny XLM-R as GGUF (positions from 0, Q8_0): every state, single and batch
    procedure TinyGgufForwardMatchesReference;
    /// GGUF models this class does not understand are refused
    procedure GgufModelRefusesForeignFiles;
    /// a padding token inside a sequence (masked) neither takes a position nor
    // changes the real tokens - for both position rules
    procedure PaddingInsideSequence;
    /// tokenizer and embedder from the tiny GGUF: load, encode, embed exactly
    // like tokenizer + forward + CLS/L2, refuse too many tokens
    procedure TinyGgufTokenizerAndEmbedder;
    /// tokenizer metadata of another pipeline is refused
    procedure GgufTokenizerRefusesForeignMetadata;
    /// BGE-M3 tokenizer from the GGUF metadata: ids like SentencePiece on every
    // case and like lembed on every case without a special-token literal
    procedure BgeM3GgufTokenizer;
    /// BGE-M3 Q8: short corpus texts against the lembed vectors
    procedure BgeM3Q8ShortTexts;
    /// the gate of phase 1b: whole corpus, cosine >= 0.999, nDCG@10 >= 0.8863
    // (LAI_INFER_GATE=1)
    procedure BgeM3Q8CorpusGate;
  end;

/// the GGUF file of reference contract v1, '' when absent
function FindBgeM3Gguf: TFileName;


implementation

const
  /// nDCG@10 of the lembed oracle on embed-de, and the gate: at most one point
  // below (docs/native-inference.md, phase 1b)
  LEMBED_NDCG10 = 0.89629361138298;
  GATE_NDCG10 = 0.8863;
  GATE_COSINE = 0.999;

function FindBgeM3Gguf: TFileName;
var
  dir: TFileName;
  i: integer;
begin
  result := GetEnvironmentVariable('LAI_GGUF');
  if result <> '' then
  begin
    if not FileExists(result) then
      result := '';
    exit;
  end;
  dir := ExcludeTrailingPathDelimiter(Executable.ProgramFilePath);
  for i := 1 to 6 do
  begin
    result := MakePath([dir, 'vendor', 'models', 'bge-m3-q8_0.gguf']);
    if FileExists(result) then
      exit;
    dir := ExtractFileDir(dir);
  end;
  result := '';
end;

/// the embed-de corpus folder, '' when absent
function FindEmbedDe: TFileName;
var
  dir: TFileName;
  i: integer;
begin
  dir := ExcludeTrailingPathDelimiter(Executable.ProgramFilePath);
  for i := 1 to 6 do
  begin
    result := MakePath([dir, 'tests', 'data', 'embed-de'], true);
    if FileExists(result + 'corpus.jsonl') then
      exit;
    dir := ExtractFileDir(dir);
  end;
  result := '';
end;


{ a tiny GGUF writer for the damaged variants }

function GU32(v: cardinal): RawByteString;
begin
  SetLength(result, 4);
  MoveFast(v, pointer(result)^, 4);
end;

function GU64(v: Int64): RawByteString;
begin
  SetLength(result, 8);
  MoveFast(v, pointer(result)^, 8);
end;

function GStr(const s: RawByteString): RawByteString;
begin
  result := GU64(length(s)) + s;
end;

/// one key/value pair: key, type id, raw payload
function GKv(const aKey: RawByteString; aType: cardinal;
  const aPayload: RawByteString): RawByteString;
begin
  result := GStr(aKey) + GU32(aType) + aPayload;
end;

/// one tensor info: name, dims innermost first, ggml type, offset
function GTi(const aName: RawByteString; const aDims: array of Int64;
  aType: cardinal; aOffset: Int64): RawByteString;
var
  i: PtrInt;
begin
  result := GStr(aName) + GU32(length(aDims));
  for i := 0 to high(aDims) do
    result := result + GU64(aDims[i]);
  result := result + GU32(aType) + GU64(aOffset);
end;

/// a whole file: header, padding to aAlign, aData
function GFile(aVersion: cardinal; aTensors, aValues: Int64;
  const aKvs, aTis, aData: RawByteString; aAlign: PtrInt = 32): RawByteString;
begin
  result := 'GGUF' + GU32(aVersion) + GU64(aTensors) + GU64(aValues) + aKvs + aTis;
  while length(result) mod aAlign <> 0 do
    result := result + #0;
  result := result + aData;
end;

/// the reference mini file: one string key, an F32 [4] and a Q8_0 [32, 2]
function GMini: RawByteString;
var
  data: RawByteString;
begin
  data := StringOfChar(AnsiChar(#0), 32 + 2 * 34);
  result := GFile(3, 2, 1, GKv('general.architecture', 8, GStr('x')),
    GTi('a', [4], 0, 0) + GTi('b', [32, 2], 8, 32), data);
end;


/// aSource with aBytes written at aOffset after the end of aAnchor - a fresh
// copy, '' when the anchor is missing
function PatchAfter(const aSource, aAnchor: RawByteString; aOffset: PtrInt;
  const aBytes: RawByteString): RawByteString;
var
  p: PtrInt;
begin
  result := '';
  p := PosEx(aAnchor, aSource);
  if p = 0 then
    exit;
  FastSetRawByteString(result, pointer(aSource), length(aSource));
  MoveFast(pointer(aBytes)^,
    PByteArray(result)^[p - 1 + length(aAnchor) + aOffset], length(aBytes));
end;


{ TTestInferGguf }

procedure TTestInferGguf.Setup;
begin
  fCpu := TInferCpuBackend.Create;
  fFast := TInferCpuFastBackend.Create;
end;

procedure TTestInferGguf.CleanUp;
begin
  FreeAndNil(fFast); // re-entrant, as mORMot requires
  FreeAndNil(fCpu);
end;

procedure TTestInferGguf.KeepState(aState: PtrInt; aValues: PSingleArray;
  aRows, aWidth: PtrInt);
var
  n: PtrInt;
begin
  if aState >= length(fStates) then
    SetLength(fStates, aState + 1);
  n := aRows * aWidth * fHiddenSize;
  SetLength(fStates[aState], n);
  MoveFast(aValues^, pointer(fStates[aState])^, n * SizeOf(single));
end;

function TTestInferGguf.RealModel: TFileName;
begin
  result := FindBgeM3Gguf;
  if result <> '' then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/bge-m3-q8_0.gguf');
  AddConsole('SKIP BGE-M3 Q8: no vendor/models/bge-m3-q8_0.gguf (see ' +
    'docs/native-embeddings.md, lembed oracle)');
end;

procedure TTestInferGguf.HalfFloatLikeGgml;
var
  ref: TInferRefSet;
  dir: TFileName;
  x: PSingleArray;
  bits: PIntegerArray;
  n, m, i, bad: PtrInt;
  h, back: word;
  s: single;
begin
  dir := FindInferRefDir('q8');
  if CheckFailed(dir <> '', 'q8 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    x := ref.F32('half/x', n);
    bits := ref.I32('half/bits', m);
    CheckEqual(n, m, 'one result per input');
    bad := 0;
    for i := 0 to n - 1 do
      if InferSingleToHalf(x^[i]) <> bits^[i] then
      begin
        inc(bad);
        if bad <= 3 then
          CheckUtf8(false, 'half of % = % instead of %',
            [x^[i], InferSingleToHalf(x^[i]), bits^[i]]);
      end;
    CheckEqual(bad, 0, 'ggml_fp32_to_fp16 rounding');
  finally
    ref.Free;
  end;
  // every half (NaN aside) survives single and back unchanged
  bad := 0;
  for i := 0 to $ffff do
  begin
    h := i;
    if (h and $7c00 = $7c00) and (h and $3ff <> 0) then
      continue; // NaN: checked below
    s := InferHalfToSingle(h);
    back := InferSingleToHalf(s);
    if back <> h then
      inc(bad);
  end;
  CheckEqual(bad, 0, 'every finite half and both infinities round-trip');
  s := InferHalfToSingle($7e00);
  // bits only: comparing a NaN may raise, depending on the FPU mask
  CheckUtf8((PCardinal(@s)^ and $7f800000 = $7f800000) and
    (PCardinal(@s)^ and $7fffff <> 0), 'half NaN gives a single NaN');
  CheckUtf8(InferSingleToHalf(s) and $7c00 = $7c00, 'single NaN stays NaN');
  // bits again: Delphi 7 does not read every float literal exactly
  s := InferHalfToSingle(1);
  CheckEqual(PCardinal(@s)^, $33800000, 'smallest subnormal = 2^-24');
  s := InferHalfToSingle($7bff);
  CheckEqual(PCardinal(@s)^, $477fe000, 'largest half = 65504');
end;

procedure TTestInferGguf.QuantizeLikeGgml;
var
  ref: TInferRefSet;
  dir: TFileName;
  x: PSingleArray;
  blocks: PByteArray;
  t: PInferRefTensor;
  n, m, rows, cols, r: PtrInt;
  mine: RawByteString;
  bad: TSingleDynArray;
  refused: boolean;
begin
  dir := FindInferRefDir('q8');
  if CheckFailed(dir <> '', 'q8 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    t := ref.Tensor('quant/x');
    rows := t^.Shape[0];
    cols := t^.Shape[1];
    x := ref.F32('quant/x', n);
    blocks := ref.U8('quant/blocks', m);
    CheckEqual(m, rows * (cols div 32) * SizeOf(TInferBlockQ8_0), 'block bytes');
    SetLength(mine, (cols div 32) * SizeOf(TInferBlockQ8_0));
    for r := 0 to rows - 1 do
    begin
      InferQuantizeRowQ8_0(@x^[r * cols], cols, pointer(mine));
      CheckUtf8(CompareMem(pointer(mine), @blocks^[r * length(mine)], length(mine)),
        'row % quantized like ggml-cpu (ties to even, zero block, tiny values)', [r]);
    end;
    // NaN and infinity are refused wherever they sit in a block
    SetLength(bad, 32);
    for r := 0 to 3 do
    begin
      FillCharFast(pointer(bad)^, 32 * SizeOf(single), 0);
      if r < 2 then
        PCardinal(@bad[5 + r * 20])^ := $7fc00000  // NaN, then after a value
      else
        PCardinal(@bad[r])^ := $7f800000 or (cardinal(r and 1) shl 31); // +-Inf
      bad[0] := 1;
      refused := false;
      try
        InferQuantizeRowQ8_0(pointer(bad), 32, pointer(mine));
      except
        on EInferError do
          refused := true;
      end;
      CheckUtf8(refused, 'non-finite activation % refused', [r]);
    end;
  finally
    ref.Free;
  end;
end;

procedure TTestInferGguf.LinearQ8LikeGgml;
var
  ref: TInferRefSet;
  dir: TFileName;
  x, b, y, deq: PSingleArray;
  w: PByteArray;
  t: PInferRefTensor;
  n, rows, nin, nout, o: PtrInt;
  mine, row: TSingleDynArray;
  cmp: TInferCompare;
begin
  dir := FindInferRefDir('q8');
  if CheckFailed(dir <> '', 'q8 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    t := ref.Tensor('linear/x');
    rows := t^.Shape[0];
    nin := t^.Shape[1];
    nout := ref.Tensor('linear/bias')^.Shape[0];
    x := ref.F32('linear/x', n);
    b := ref.F32('linear/bias', n);
    y := ref.F32('linear/y', n);
    w := ref.U8('linear/weight', n);
    CheckUtf8(fCpu.Supports(ioLinear, idtQ8_0), 'the CPU backend takes Q8_0');
    CheckUtf8(not fCpu.Supports(ioGelu, idtQ8_0), 'only Linear takes Q8_0');
    SetLength(mine, rows * nout);
    fCpu.Linear(x, rows, nin, InferWeight(w, idtQ8_0), nout, b, pointer(mine));
    // ggml sums in single, we in double: the integer block sums are the same
    cmp := InferCompare(pointer(mine), y, rows * nout, 1e-5, 1e-5);
    CheckUtf8(InferCompareOk(cmp), 'Q8_0 linear like ggml: %', [InferCompareText(cmp)]);
    // dequantized rows: d * q is exact in single
    deq := ref.F32('dequant/y', n);
    SetLength(row, nin);
    for o := 0 to nout - 1 do
    begin
      InferRowToF32(w, idtQ8_0, o, nin, pointer(row));
      cmp := InferCompare(pointer(row), @deq^[o * nin], nin, 0, 0);
      CheckUtf8(InferCompareOk(cmp), 'dequantized row %: %', [o, InferCompareText(cmp)]);
    end;
  finally
    ref.Free;
  end;
end;

procedure TTestInferGguf.Q4LikeGgml;
var
  ref: TInferRefSet;
  dir: TFileName;
  x, b, y, deq: PSingleArray;
  blocks, w: PByteArray;
  t: PInferRefTensor;
  n, m, rows, cols, nin, nout, r: PtrInt;
  mine: RawByteString;
  out, row: TSingleDynArray;
  cmp: TInferCompare;
  refused: boolean;
begin
  dir := FindInferRefDir('q4');
  if CheckFailed(dir <> '', 'q4 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    // weights quantize to the bytes of quantize_row_q4_0_ref: the signed value
    // of the first largest magnitude, sums landing on .5 and whole numbers, a
    // zero block, tiny and large rows, the clip at 15
    t := ref.Tensor('quant/x');
    rows := t^.Shape[0];
    cols := t^.Shape[1];
    x := ref.F32('quant/x', n);
    blocks := ref.U8('quant/blocks', m);
    CheckEqual(m, rows * (cols div 32) * SizeOf(TInferBlockQ4_0), 'block bytes');
    SetLength(mine, (cols div 32) * SizeOf(TInferBlockQ4_0));
    deq := ref.F32('dequant/y', n);
    SetLength(row, cols);
    for r := 0 to rows - 1 do
    begin
      InferQuantizeRowQ4_0(@x^[r * cols], cols, pointer(mine));
      CheckUtf8(CompareMem(pointer(mine), @blocks^[r * length(mine)], length(mine)),
        'row % quantized like quantize_row_q4_0_ref', [r]);
      // dequantized: d * (q - 8) is exact in single
      InferRowToF32(@blocks^[r * length(mine)], idtQ4_0, 0, cols, pointer(row));
      cmp := InferCompare(pointer(row), @deq^[r * cols], cols, 0, 0);
      CheckUtf8(InferCompareOk(cmp), 'dequantized row %: %', [r, InferCompareText(cmp)]);
    end;
    // a NaN or an infinity has no Q4_0 form
    SetLength(out, 32);
    PCardinal(@out[7])^ := $7f800000;
    refused := false;
    try
      InferQuantizeRowQ4_0(pointer(out), 32, pointer(mine));
    except
      on EInferError do
        refused := true;
    end;
    Check(refused, 'infinity refused');
    // Q8_0 weights: the bytes of quantize_row_q8_0_ref - ties away from zero,
    // where the activation quantizer rounds them to even
    t := ref.Tensor('q8ref/x');
    rows := t^.Shape[0];
    cols := t^.Shape[1];
    x := ref.F32('q8ref/x', n);
    blocks := ref.U8('q8ref/blocks', m);
    SetLength(mine, (cols div 32) * SizeOf(TInferBlockQ8_0));
    for r := 0 to rows - 1 do
    begin
      InferQuantizeRowQ8_0Ref(@x^[r * cols], cols, pointer(mine));
      CheckUtf8(CompareMem(pointer(mine), @blocks^[r * length(mine)], length(mine)),
        'weight row % quantized like quantize_row_q8_0_ref', [r]);
    end;
    InferQuantizeRowQ8_0(x, cols, pointer(mine));
    Check(not CompareMem(pointer(mine), blocks, length(mine)),
      'the activation quantizer rounds the ties of row 0 otherwise');
    // the Linear against ggml_vec_dot_q4_0_q8_0
    t := ref.Tensor('linear/x');
    rows := t^.Shape[0];
    nin := t^.Shape[1];
    nout := ref.Tensor('linear/bias')^.Shape[0];
    x := ref.F32('linear/x', n);
    b := ref.F32('linear/bias', n);
    y := ref.F32('linear/y', n);
    w := ref.U8('linear/weight', n);
    CheckUtf8(fCpu.Supports(ioLinear, idtQ4_0), 'the CPU backend takes Q4_0');
    SetLength(out, rows * nout);
    fCpu.Linear(x, rows, nin, InferWeight(w, idtQ4_0), nout, b, pointer(out));
    // ggml sums in single, we in double: the integer block sums are the same
    cmp := InferCompare(pointer(out), y, rows * nout, 1e-5, 1e-5);
    CheckUtf8(InferCompareOk(cmp), 'Q4_0 linear like ggml: %', [InferCompareText(cmp)]);
  finally
    ref.Free;
  end;
end;

procedure TTestInferGguf.Q5LikeGgml;
var
  ref: TInferRefSet;
  dir: TFileName;
  x, b, y, deq: PSingleArray;
  blocks, w: PByteArray;
  t: PInferRefTensor;
  n, m, rows, cols, nin, nout, r: PtrInt;
  out, row: TSingleDynArray;
  cmp: TInferCompare;
begin
  dir := FindInferRefDir('q5');
  if CheckFailed(dir <> '', 'q5 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    // blocks of quantize_row_q5_0_ref - every q from 0 to 31, a zero block,
    // tiny and large rows, the clip, fifth bits all clear - dequantize to the
    // values of dequantize_row_q5_0: d * (q - 16) is exact in single
    t := ref.Tensor('quant/x');
    rows := t^.Shape[0];
    cols := t^.Shape[1];
    blocks := ref.U8('quant/blocks', m);
    CheckEqual(m, rows * (cols div 32) * SizeOf(TInferBlockQ5_0), 'block bytes');
    deq := ref.F32('dequant/y', n);
    SetLength(row, cols);
    for r := 0 to rows - 1 do
    begin
      InferRowToF32(blocks, idtQ5_0, r, cols, pointer(row));
      cmp := InferCompare(pointer(row), @deq^[r * cols], cols, 0, 0);
      CheckUtf8(InferCompareOk(cmp), 'dequantized row %: %', [r, InferCompareText(cmp)]);
    end;
    // the Linear against ggml_vec_dot_q5_0_q8_0
    t := ref.Tensor('linear/x');
    rows := t^.Shape[0];
    nin := t^.Shape[1];
    nout := ref.Tensor('linear/bias')^.Shape[0];
    x := ref.F32('linear/x', n);
    b := ref.F32('linear/bias', n);
    y := ref.F32('linear/y', n);
    w := ref.U8('linear/weight', n);
    CheckUtf8(fCpu.Supports(ioLinear, idtQ5_0), 'the CPU backend takes Q5_0');
    SetLength(out, rows * nout);
    fCpu.Linear(x, rows, nin, InferWeight(w, idtQ5_0), nout, b, pointer(out));
    // ggml sums in single, we in double: the integer block sums are the same
    cmp := InferCompare(pointer(out), y, rows * nout, 1e-5, 1e-5);
    CheckUtf8(InferCompareOk(cmp), 'Q5_0 linear like ggml: %', [InferCompareText(cmp)]);
  finally
    ref.Free;
  end;
end;

procedure TTestInferGguf.KQuantsLikeGgml;
var
  ref: TInferRefSet;
  dir: TFileName;
  x, b, y, deq: PSingleArray;
  blocks, w: PByteArray;
  t: PInferRefTensor;
  n, m, rows, cols, nin, nout, r, k, size: PtrInt;
  mine: RawByteString;
  out, row: TSingleDynArray;
  cmp: TInferCompare;
  kind: TInferDType;
  name: RawUtf8;
begin
  dir := FindInferRefDir('qk');
  if CheckFailed(dir <> '', 'qk found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    // Q8_K activations: the bytes of ggml-cpu's quantize_row_q8_K
    t := ref.Tensor('q8k/x');
    rows := t^.Shape[0];
    cols := t^.Shape[1];
    x := ref.F32('q8k/x', n);
    blocks := ref.U8('q8k/blocks', m);
    CheckEqual(m, rows * (cols div 256) * SizeOf(TInferBlockQ8_K), 'Q8_K bytes');
    SetLength(mine, (cols div 256) * SizeOf(TInferBlockQ8_K));
    for r := 0 to rows - 1 do
    begin
      InferQuantizeRowQ8_K(@x^[r * cols], cols, pointer(mine));
      CheckUtf8(CompareMem(pointer(mine), @blocks^[r * length(mine)], length(mine)),
        'Q8_K row % like quantize_row_q8_K', [r]);
    end;
    // Q4_K and Q6_K weights: dequantized bit for bit, the Linear like ggml
    t := ref.Tensor('linear/x');
    rows := t^.Shape[0];
    nin := t^.Shape[1];
    nout := ref.Tensor('linear/bias')^.Shape[0];
    x := ref.F32('linear/x', n);
    b := ref.F32('linear/bias', n);
    SetLength(row, nin);
    SetLength(out, rows * nout);
    for k := 0 to 1 do
    begin
      if k = 0 then
      begin
        kind := idtQ4_K;
        name := 'q4_K';
      end
      else
      begin
        kind := idtQ6_K;
        name := 'q6_K';
      end;
      size := (nin div 256) * INFER_DTYPE_SIZE[kind];
      w := ref.U8(name + '/weight', n);
      CheckEqual(n, nout * size, name + ' bytes');
      deq := ref.F32(name + '/dequant', n);
      for r := 0 to nout - 1 do
      begin
        InferRowToF32(w, kind, r, nin, pointer(row));
        cmp := InferCompare(pointer(row), @deq^[r * nin], nin, 0, 0);
        CheckUtf8(InferCompareOk(cmp), '% dequantized row %: %',
          [name, r, InferCompareText(cmp)]);
      end;
      CheckUtf8(fCpu.Supports(ioLinear, kind), 'the CPU backend takes %', [name]);
      fCpu.Linear(x, rows, nin, InferWeight(w, kind), nout, b, pointer(out));
      y := ref.F32(name + '/y', n);
      // ggml sums in single, we in double: the integer block sums are the same
      cmp := InferCompare(pointer(out), y, rows * nout, 1e-4, 1e-5);
      CheckUtf8(InferCompareOk(cmp), '% linear like ggml: %',
        [name, InferCompareText(cmp)]);
    end;
  finally
    ref.Free;
  end;
end;

procedure TTestInferGguf.ReaderReadsTinyFile;
var
  dir: TFileName;
  g: TGgufFile;
  t: PInferTensor;
  refused: boolean;
begin
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  g := TGgufFile.Create(dir + 'model.gguf');
  try
    CheckEqual(g.Str('general.architecture'), 'bert', 'architecture');
    CheckEqual(g.Int('bert.block_count'), 2, 'UINT32 as integer');
    CheckUtf8(Abs(g.Float('bert.attention.layer_norm_epsilon') - 1e-5) < 1e-12,
      'FLOAT32 as float');
    CheckUtf8(not g.Bool('bert.attention.causal'), 'BOOL');
    CheckUtf8(g.BoolDef('no.such.key', true), 'absent BOOL takes the default');
    CheckEqual(g.Alignment, 32, 'default alignment');
    CheckEqual(length(g.Tensors), 5 + 16 * 2, 'tensor count');
    t := g.Tensor('token_embd.weight');
    CheckUtf8((t^.DType = idtQ8_0) and InferShapeIs(t^, [64, 32]) and
      (t^.Bytes = 64 * 34), 'Q8_0 [64, 32], row-major');
    t := g.Tensor('position_embd.weight');
    CheckUtf8((t^.DType = idtF32) and InferShapeIs(t^, [38, 32]),
      'positions without the XLM-R offset rows');
    CheckUtf8(g.FindTensor('nope') = nil, 'missing tensor is nil');
    refused := false;
    try
      g.Str('bert.block_count');
    except
      on EInferError do
        refused := true;
    end;
    CheckUtf8(refused, 'a UINT32 is no string');
    refused := false;
    try
      g.Weight('token_embd.weight', idtF32, [64, 32]);
    except
      on EInferError do
        refused := true;
    end;
    CheckUtf8(refused, 'a Q8_0 tensor is no F32 weight');
  finally
    g.Free;
  end;
end;

procedure TTestInferGguf.ReaderRefusesDamagedFiles;
var
  dir: TFileName;
  tiny, mini, data: RawByteString;
  i, crashes, accepted: PtrInt;

  function Refused(const aContent: RawByteString; const aReason: RawUtf8): boolean;
  var
    g: TGgufFile;
  begin
    result := false;
    try
      g := TGgufFile.CreateFromContent(aContent);
      g.Free;
    except
      // only EInferError counts as refused: anything else (an access violation,
      // a range or memory error) escapes and is counted or fails the test
      on E: EInferError do
        result := (aReason = '') or (PosEx(aReason, StringToUtf8(E.Message)) > 0);
    end;
  end;

begin
  mini := GMini;
  CheckUtf8(not Refused(mini, ''), 'the mini file itself is valid');
  data := StringOfChar(AnsiChar(#0), 32 + 2 * 34);
  // envelope
  CheckUtf8(Refused('GGUX' + copy(mini, 5, MaxInt), 'not a GGUF'), 'bad magic');
  CheckUtf8(Refused(GFile(2, 2, 1, GKv('general.architecture', 8, GStr('x')),
    GTi('a', [4], 0, 0) + GTi('b', [32, 2], 8, 32), data), 'version'), 'version 2');
  CheckUtf8(Refused(GFile(3, Int64(1) shl 40, 1, '', '', ''), 'cannot fit'),
    'huge tensor count');
  CheckUtf8(Refused(GFile(3, 0, Int64(1) shl 62, '', '', ''), 'cannot fit'),
    'huge value count');
  // key/value pairs
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('k', 13, GU32(0)), '', ''), 'unknown type'),
    'unknown value type');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('k', 7, #2), '', ''), 'BOOL'), 'bool of 2');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('k', 8, GStr('a'#0'b')), '', ''), 'NUL'),
    'NUL in a string value');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('k'#0, 4, GU32(1)), '', ''), 'NUL'),
    'NUL in a key');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('', 4, GU32(1)), '', ''), 'empty key'),
    'empty key');
  CheckUtf8(Refused(GFile(3, 0, 2, GKv('k', 4, GU32(1)) + GKv('k', 4, GU32(2)),
    '', ''), 'twice'), 'duplicate key');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('k', 9, GU32(9) + GU64(1) + GU32(4) +
    GU64(1) + GU32(0)), '', ''), 'nested'), 'nested array');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('k', 9, GU32(4) + GU64(Int64(1) shl 60)),
    '', ''), 'more than the file'), 'huge array count');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('k', 9, GU32(8) + GU64(1000)), '', ''),
    'more than the file'), 'string array count beyond the file');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('k', 8, GU64(Int64(1) shl 50)), '', ''),
    'past the end'), 'huge string length');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('k', 8, GU64(-1)), '', ''),
    'out of range'), 'string length above High(Int64)');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('general.alignment', 4, GU32(24)), '', ''),
    'power of two'), 'alignment 24');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('general.alignment', 4, GU32(0)), '', ''),
    'power of two'), 'alignment 0');
  CheckUtf8(Refused(GFile(3, 0, 1, GKv('general.alignment', 10, GU64(32)), '', ''),
    'UINT32'), 'alignment of another type');
  // tensor infos
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [4], 3, 0), data), 'ggml type'),
    'Q4_1 not supported');
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [4], 2, 0), data), 'blocks'),
    'Q4_0 needs whole blocks');
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [0], 0, 0), data), 'dimension'),
    'zero dimension');
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [1, 1, 1, 1, 1], 0, 0), data),
    'dimensions'), 'five dimensions');
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [], 0, 0), data), 'dimensions'),
    'no dimension');
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [48], 8, 0), data), 'blocks'),
    'Q8_0 row of 48');
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [Int64(1) shl 40, 1 shl 30], 0, 0),
    data), 'exceeds'), 'shape product beyond the file');
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [4], 0, 4), data), 'aligned'),
    'misaligned offset');
  CheckUtf8(Refused(GFile(3, 2, 0, '', GTi('a', [4], 0, 0) + GTi('a', [4], 0, 32),
    data), 'twice'), 'duplicate tensor name');
  CheckUtf8(Refused(GFile(3, 2, 0, '', GTi('a', [16], 0, 0) + GTi('b', [4], 0, 32),
    data), 'expected'), 'overlapping tensors');
  CheckUtf8(Refused(GFile(3, 2, 0, '', GTi('a', [4], 0, 0) + GTi('b', [4], 0, 64),
    data), 'expected'), 'gap beyond the padding');
  CheckUtf8(Refused(GFile(3, 2, 0, '', GTi('b', [4], 0, 32) + GTi('a', [4], 0, 0),
    data), 'expected'), 'tensors out of file order');
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [64], 0, 0), data), 'outside'),
    'tensor past the data end');
  CheckUtf8(Refused(GFile(3, 1, 0, '', GTi('a', [4], 0, 0), data), 'unreferenced'),
    'unreferenced bytes at the end');
  CheckUtf8(not Refused(GFile(3, 1, 0, '', GTi('a', [4], 0, 0),
    copy(data, 1, 31)), ''), 'padding after the last tensor is fine');
  // every truncation of a real file is refused cleanly
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  tiny := StringFromFile(dir + 'model.gguf');
  CheckUtf8(not Refused(tiny, ''), 'the tiny file is valid');
  crashes := 0;
  accepted := 0;
  for i := 0 to length(tiny) - 1 do
    if (i < 4096) or (i mod 97 = 0) then
      try
        if not Refused(copy(tiny, 1, i), '') then
          inc(accepted);
      except
        inc(crashes); // anything but EInferError escaping Refused
      end;
  CheckEqual(crashes, 0, 'truncations raise EInferError only');
  CheckEqual(accepted, 0, 'no truncation is accepted');
  // flipping single header bytes never escapes as anything but EInferError
  crashes := 0;
  for i := 1 to 8192 do
  begin
    // a fresh copy: it is written through a pointer, never into tiny itself
    FastSetRawByteString(mini, pointer(tiny), length(tiny));
    // header and tensor infos of the tiny file (about 4.3 KB) and the start
    // of its data: within the first 8 KB
    PByteArray(mini)^[(i * 7919) mod 8192] :=
      PByteArray(mini)^[(i * 7919) mod 8192] xor (1 shl (i and 7));
    try
      Refused(mini, '');
    except
      inc(crashes);
    end;
  end;
  CheckEqual(crashes, 0, 'flipped header bytes give EInferError or a valid file');
end;

procedure TTestInferGguf.TinyGgufForwardMatchesReference;
var
  dir: TFileName;
  model: TXlmrModel;
  ctx: TXlmrContext;
  ref: TInferRefSet;
  ids: PIntegerArray;
  exp, emb: PSingleArray;
  t: PInferRefTensor;
  i, k, n, rn, h, rows, width, r, c, real: PtrInt;
  cmp: TInferCompare;
  mine: TSingleDynArray;
begin
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  ref := nil;
  model := nil;
  ctx := nil;
  try
    ref := TInferRefSet.CreateFromFolder(dir);
    model := TXlmrModel.CreateFromGguf(dir + 'model.gguf', fCpu);
    ctx := TXlmrContext.Create;
    h := model.Config.Hidden;
    fHiddenSize := h;
    CheckEqual(model.Config.FirstPosition, 0, 'GGUF positions count from 0');
    CheckEqual(model.MaxTokens, 38, 'every position row usable');
    SetLength(mine, h);
    for i := 0 to 2 do
    begin
      fStates := nil;
      ids := ref.I32(FormatUtf8('single/%/input_ids', [i]), n);
      model.Forward(ctx, ids, 1, n, nil, KeepState);
      for k := 0 to model.Config.Layers do
      begin
        exp := ref.F32(FormatUtf8('single/%/hidden.%', [i, k]), rn);
        cmp := InferCompare(pointer(fStates[k]), exp, rn, 1e-4, 1e-3);
        CheckUtf8(InferCompareOk(cmp), 'single/%/hidden.%: %',
          [i, k, InferCompareText(cmp)]);
      end;
      MoveFast(ctx.Hidden[0], pointer(mine)^, h * SizeOf(single));
      fCpu.L2Normalize(pointer(mine), h);
      emb := ref.F32(FormatUtf8('single/%/embedding', [i]), rn);
      cmp := InferCompare(pointer(mine), emb, h, 1e-4, 1e-3);
      CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine >= 0.9999),
        'single/%/embedding: %', [i, InferCompareText(cmp)]);
    end;
    // right-padded batch: the real tokens of each row as computed alone
    t := ref.Tensor('batch/input_ids');
    rows := t^.Shape[0];
    width := t^.Shape[1];
    ids := ref.I32('batch/input_ids', n);
    model.Forward(ctx, ids, rows, width);
    for r := 0 to rows - 1 do
    begin
      real := 0;
      for c := 0 to width - 1 do
        if ids^[r * width + c] <> model.Config.PadId then
          inc(real);
      exp := ref.F32(FormatUtf8('batch/%/hidden.%', [r, model.Config.Layers]), rn);
      cmp := InferCompare(@ctx.Hidden[r * width * h], exp, real * h, 1e-4, 1e-3);
      CheckUtf8(InferCompareOk(cmp), 'batch row %: %', [r, InferCompareText(cmp)]);
    end;
  finally
    ctx.Free;
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferGguf.GgufModelRefusesForeignFiles;
var
  dir: TFileName;
  tiny: RawByteString;

  function Patched(const aFind, aReplace: RawByteString): RawByteString;
  var
    p: PtrInt;
  begin
    // a fresh copy: it is written through a pointer, never into tiny itself
    FastSetRawByteString(result, pointer(tiny), length(tiny));
    p := PosEx(aFind, result);
    Check(p > 0, 'patch anchor found');
    if p > 0 then
      MoveFast(pointer(aReplace)^, PByteArray(result)^[p - 1], length(aReplace));
  end;

  function Refuses(const aContent: RawByteString; const aReason: RawUtf8): boolean;
  var
    fn: TFileName;
    m: TXlmrModel;
  begin
    result := false;
    fn := TemporaryFileName;
    FileFromString(aContent, fn);
    try
      try
        m := TXlmrModel.CreateFromGguf(fn, fCpu);
        m.Free;
      except
        on E: Exception do
          result := (E is EInferError) and
            (PosEx(aReason, StringToUtf8(E.Message)) > 0);
      end;
    finally
      DeleteFile(fn);
    end;
  end;

begin
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  tiny := StringFromFile(dir + 'model.gguf');
  CheckUtf8(not Refuses(tiny, 'x'), 'the tiny file itself loads');
  // same length replacements: the file stays well-formed, only the meaning changes
  CheckUtf8(Refuses(Patched(GStr('general.architecture') + GU32(8) + GStr('bert'),
    GStr('general.architecture') + GU32(8) + GStr('bart')), 'architecture'),
    'another architecture refused');
  CheckUtf8(Refuses(Patched(GStr('bert.attention.causal') + GU32(7) + #0,
    GStr('bert.attention.causal') + GU32(7) + #1), 'causal'), 'causal refused');
  CheckUtf8(Refuses(Patched(GStr('bert.embedding_length') + GU32(4) + GU32(32),
    GStr('bert.embedding_length') + GU32(4) + GU32(64)), 'shape'),
    'hidden size not matching the tensors refused');
  CheckUtf8(Refuses(Patched(GStr('tokenizer.ggml.padding_token_id') + GU32(4) + GU32(1),
    GStr('tokenizer.ggml.padding_token_id') + GU32(4) + GU32(64)), 'out of range'),
    'pad id outside the vocabulary refused');
  CheckUtf8(Refuses(Patched(GStr('bert.block_count') + GU32(4) + GU32(2),
    GStr('bert.block_count') + GU32(4) + GU32(1)), 'understood'),
    'tensors of a layer the header does not count refused');
end;

procedure TTestInferGguf.PaddingInsideSequence;
var
  dir: TFileName;
  model: TXlmrModel;
  ctx: TXlmrContext;
  withPad, alone: TSingleDynArray;
  ids: array[0..5] of integer;
  keep: array[0..5] of integer;
  i, h, src: PtrInt;
  cmp: TInferCompare;
begin
  for src := 0 to 1 do
  begin
    if src = 0 then
      dir := FindInferRefDir('tiny-xlmr')
    else
      dir := FindInferRefDir('tiny-gguf');
    if CheckFailed(dir <> '', 'tiny fixtures found') then
      exit;
    if src = 0 then
      model := TXlmrModel.Create(dir, fCpu)
    else
      model := TXlmrModel.CreateFromGguf(dir + 'model.gguf', fCpu);
    ctx := TXlmrContext.Create;
    try
      h := model.Config.Hidden;
      // <s> 7 9 11 </s> alone
      ids[0] := 0;
      ids[1] := 7;
      ids[2] := 9;
      ids[3] := 11;
      ids[4] := 2;
      model.Forward(ctx, @ids, 1, 5);
      alone := copy(ctx.Hidden, 0, 5 * h);
      // <s> 7 <pad> 9 11 </s>: the pad is masked and takes no position
      ids[0] := 0;
      ids[1] := 7;
      ids[2] := model.Config.PadId;
      ids[3] := 9;
      ids[4] := 11;
      ids[5] := 2;
      for i := 0 to 5 do
        keep[i] := ord(i <> 2);
      model.Forward(ctx, @ids, 1, 6, @keep);
      withPad := copy(ctx.Hidden, 0, 6 * h);
      cmp := InferCompare(pointer(withPad), pointer(alone), 2 * h, 1e-5, 1e-4);
      CheckUtf8(InferCompareOk(cmp), 'source %: tokens before the pad: %',
        [src, InferCompareText(cmp)]);
      cmp := InferCompare(@withPad[3 * h], @alone[2 * h], 3 * h, 1e-5, 1e-4);
      CheckUtf8(InferCompareOk(cmp), 'source %: tokens after the pad keep their ' +
        'positions: %', [src, InferCompareText(cmp)]);
    finally
      ctx.Free;
      model.Free;
    end;
  end;
end;

procedure TTestInferGguf.TinyGgufTokenizerAndEmbedder;
var
  dir: TFileName;
  emb: TInferEmbedder;
  ctx: TXlmrContext;
  ids: TIntegerDynArray;
  v, mine: TSingleDynArray;
  i, h: PtrInt;
  cmp: TInferCompare;
  refused: boolean;
  long: RawUtf8;
begin
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  emb := TInferEmbedder.CreateFromGguf(dir + 'model.gguf', fCpu);
  ctx := TXlmrContext.Create;
  try
    CheckEqual(emb.Tokenizer.VocabSize, 64, 'vocabulary = model rows');
    CheckEqual(emb.MaxTokens, 38, 'limited by the 38 positions, not 512');
    ids := emb.Tokenizer.Encode('the abc');
    CheckUtf8((length(ids) >= 4) and (ids[0] = 0) and (ids[high(ids)] = 2),
      '<s> ... </s>');
    for i := 1 to high(ids) - 1 do
      CheckUtf8((ids[i] > 3) and (ids[i] < 63), 'normal pieces only');
    // the embedder = tokenizer + forward + CLS + L2, nothing else
    h := emb.Dim;
    v := emb.Embed(['the abc']);
    emb.Model.Forward(ctx, pointer(ids), 1, length(ids));
    SetLength(mine, h);
    MoveFast(ctx.Hidden[0], pointer(mine)^, h * SizeOf(single));
    fCpu.L2Normalize(pointer(mine), h);
    cmp := InferCompare(pointer(v), pointer(mine), h, 0, 0);
    CheckUtf8(InferCompareOk(cmp), 'Embed = forward + CLS + L2: %',
      [InferCompareText(cmp)]);
    long := '';
    for i := 1 to 40 do
      long := long + 'the ';
    refused := false;
    try
      emb.Embed([long]);
    except
      on EInferError do
        refused := true;
    end;
    CheckUtf8(refused, 'more tokens than positions refused, never cut');
  finally
    ctx.Free;
    emb.Free;
  end;
end;

procedure TTestInferGguf.GgufTokenizerRefusesForeignMetadata;
var
  dir: TFileName;
  tiny, types: RawByteString;

  function Refuses(const aContent: RawByteString; const aReason: RawUtf8): boolean;
  var
    fn: TFileName;
    e: TInferEmbedder;
  begin
    result := false;
    if aContent = '' then
      exit; // anchor missing: fails the check
    fn := TemporaryFileName;
    FileFromString(aContent, fn);
    try
      try
        e := TInferEmbedder.CreateFromGguf(fn, fCpu);
        e.Free;
      except
        on E: EInferError do
          result := PosEx(aReason, StringToUtf8(E.Message)) > 0;
      end;
    finally
      DeleteFile(fn);
    end;
  end;

  function BoolOff(const aKey: RawUtf8): RawByteString;
  begin
    result := PatchAfter(tiny, GStr(aKey) + GU32(7), 0, #0);
  end;

  function TokenType(aIndex: PtrInt; aType: cardinal): RawByteString;
  begin
    result := PatchAfter(tiny, types, aIndex * 4, GU32(aType));
  end;

begin
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  tiny := StringFromFile(dir + 'model.gguf');
  types := GStr('tokenizer.ggml.token_type') + GU32(9) + GU32(5) + GU64(64);
  CheckUtf8(not Refuses(tiny, 'x'), 'the tiny file itself loads');
  CheckUtf8(Refuses(PatchAfter(tiny, GStr('tokenizer.ggml.model') + GU32(8) +
    GU64(2), 0, 'u5'), 'not t5'), 'another tokenizer model refused');
  CheckUtf8(Refuses(PatchAfter(tiny, GStr('tokenizer.ggml.pre') + GU32(8) +
    GU64(7), 0, 'qwen2xx'), 'pre-tokenizer'), 'another pre-tokenizer refused');
  CheckUtf8(Refuses(BoolOff('tokenizer.ggml.add_bos_token'), 'not both added'),
    'no <s> refused');
  CheckUtf8(Refuses(BoolOff('tokenizer.ggml.add_eos_token'), 'not both added'),
    'no </s> refused');
  CheckUtf8(Refuses(BoolOff('tokenizer.ggml.add_space_prefix'), 'whitespace'),
    'no space prefix refused');
  CheckUtf8(Refuses(BoolOff('tokenizer.ggml.remove_extra_whitespaces'),
    'whitespace'), 'unfolded whitespace refused');
  CheckUtf8(Refuses(TokenType(10, 4), 'not supported'), 'user-defined piece refused');
  CheckUtf8(Refuses(TokenType(10, 6), 'not supported'), 'byte piece refused');
  CheckUtf8(Refuses(TokenType(10, 7), 'unknown type'), 'unknown token type refused');
  CheckUtf8(Refuses(TokenType(10, 2), 'second unknown'), 'second <unk> refused');
  CheckUtf8(Refuses(TokenType(0, 1), 'control'), '<s> as a normal piece refused');
  CheckUtf8(Refuses(TokenType(3, 1), 'unknown'), '<unk> as a normal piece refused');
  CheckUtf8(Refuses(PatchAfter(tiny, GStr('tokenizer.ggml.bos_token_id') + GU32(4),
    0, GU32(64)), 'outside the vocabulary'), 'bos id outside the vocabulary refused');
  CheckUtf8(Refuses(PatchAfter(tiny, GStr('bert.pooling_type') + GU32(4), 0,
    GU32(1)), 'pooling'), 'mean pooling refused');
end;

procedure TTestInferGguf.BgeM3GgufTokenizer;
var
  fn: TFileName;
  dir, ldir: TFileName;
  g: TGgufFile;
  tok: TUgmTokenizer;
  ref, lem: TInferRefSet;
  cases: PDocVariantData;
  ids, offsets, lids, loffsets, literal: PIntegerArray;
  texts, norms: PByteArray;
  textOffsets, normOffsets: PIntegerArray;
  n, i, j, first, count, refBad, normBad, lembedSame, lembedOther: PtrInt;
  mine: TIntegerDynArray;
  text, expected: RawUtf8;

  function Same(aIds: PIntegerArray; aFirst, aCount: PtrInt): boolean;
  var
    k: PtrInt;
  begin
    result := length(mine) = aCount;
    if result then
      for k := 0 to aCount - 1 do
        if mine[k] <> aIds^[aFirst + k] then
        begin
          result := false;
          exit;
        end;
  end;

begin
  fn := RealModel;
  if fn = '' then
    exit;
  dir := FindInferRefDir('bge-m3-tokens');
  ldir := FindInferRefDir('lembed-tokens');
  if CheckFailed((dir <> '') and (ldir <> ''), 'token references found') then
    exit;
  g := nil;
  tok := nil;
  ref := nil;
  lem := nil;
  try
    g := TGgufFile.Create(fn);
    tok := TUgmTokenizer.CreateFromGguf(g);
    CheckEqual(tok.VocabSize, 250002, 'vocabulary');
    CheckUtf8((tok.BosId = 0) and (tok.PadId = 1) and (tok.EosId = 2) and
      (tok.UnkId = 3), 'special ids');
    ref := TInferRefSet.CreateFromFolder(dir);
    lem := TInferRefSet.CreateFromFolder(ldir);
    cases := ref.Manifest^.A['cases'];
    ids := ref.I32('ids', n);
    offsets := ref.I32('offsets', n);
    texts := ref.U8('texts', n);
    textOffsets := ref.I32('text_offsets', n);
    norms := ref.U8('normalized', n);
    normOffsets := ref.I32('normalized_offsets', n);
    lids := lem.I32('ids', n);
    loffsets := lem.I32('offsets', n);
    literal := lem.I32('special_literal', n);
    CheckEqual(n, cases^.Count, 'lembed ids for every case');
    refBad := 0;
    normBad := 0;
    lembedSame := 0;
    lembedOther := 0;
    for i := 0 to cases^.Count - 1 do
    begin
      FastSetString(text, @texts^[textOffsets^[i]], textOffsets^[i + 1] - textOffsets^[i]);
      mine := tok.Encode(text);
      first := offsets^[i];
      count := offsets^[i + 1] - first;
      if not Same(ids, first, count) then
        inc(refBad);
      FastSetString(expected, @norms^[normOffsets^[i]],
        normOffsets^[i + 1] - normOffsets^[i]);
      if tok.Normalize(text) <> expected then
        inc(normBad);
      // lembed: equal without a special literal; with one, lembed reads a
      // control token (parse_special=true) and we must not
      j := loffsets^[i];
      if literal^[i] = 0 then
      begin
        if Same(lids, j, loffsets^[i + 1] - j) then
          inc(lembedSame)
        else
          CheckUtf8(false, 'case %: other ids than lembed', [_Safe(cases^.Values[i])^.U['id']]);
      end
      else if not Same(lids, j, loffsets^[i + 1] - j) then
        inc(lembedOther);
    end;
    // pieces of other types (UNUSED [PAD250000]) as text: like lembed
    texts := lem.U8('extra/texts', n);
    textOffsets := lem.I32('extra/text_offsets', n);
    lids := lem.I32('extra/ids', n);
    loffsets := lem.I32('extra/offsets', n);
    for i := 0 to n - 2 do
    begin
      FastSetString(text, @texts^[textOffsets^[i]], textOffsets^[i + 1] - textOffsets^[i]);
      mine := tok.Encode(text);
      CheckUtf8(Same(lids, loffsets^[i], loffsets^[i + 1] - loffsets^[i]),
        'extra case "%" like lembed (unused piece never matched)', [text]);
    end;
    CheckEqual(refBad, 0, 'ids like SentencePiece on every case');
    CheckEqual(normBad, 0, 'normalized like SentencePiece on every case');
    CheckUtf8(lembedSame >= 1000, '% cases like lembed (gate: 1000)', [lembedSame]);
    CheckUtf8(lembedOther > 0, 'special literals: % cases where lembed reads control ' +
      'tokens and we read text', [lembedOther]);
    AddConsole(FormatString('GGUF tokenizer: % cases like lembed, % special-literal ' +
      'cases read as text', [lembedSame, lembedOther]));
  finally
    lem.Free;
    ref.Free;
    tok.Free;
    g.Free;
  end;
end;

type
  /// the parsed lines of a JSON Lines file
  TJsonRows = array of TDocVariantData;

/// the lines of a JSON Lines file, parsed
function ReadJsonl(const aFileName: TFileName; out aRows: TJsonRows): boolean;
var
  lines: TRawUtf8DynArray;
  all: RawUtf8;
  i, n: PtrInt;
begin
  result := false;
  all := StringFromFile(aFileName);
  if all = '' then
    exit;
  CsvToRawUtf8DynArray(pointer(all), lines, #10);
  SetLength(aRows, length(lines));
  n := 0;
  for i := 0 to high(lines) do
    if TrimU(lines[i]) <> '' then
    begin
      if not aRows[n].InitJson(lines[i], JSON_FAST) then
        exit;
      inc(n);
    end;
  SetLength(aRows, n);
  result := n > 0;
end;

/// raw float32 little-endian rows of aDim
function ReadF32(const aFileName: TFileName; aRows, aDim: PtrInt): TSingleDynArray;
var
  raw: RawByteString;
begin
  result := nil;
  raw := StringFromFile(aFileName);
  if length(raw) <> aRows * aDim * SizeOf(single) then
    exit;
  SetLength(result, aRows * aDim);
  MoveFast(pointer(raw)^, pointer(result)^, length(raw));
end;

procedure TTestInferGguf.CheckCorpus(aAll: boolean; aShort: PtrInt;
  aBackend: TInferBackend; const aName: RawUtf8);
var
  fn, data: TFileName;
  emb: TInferEmbedder;
  docs, queries: TJsonRows;
  docRef, queryRef, docVec, queryVec, v: TSingleDynArray;
  pick: TIntegerDynArray;
  i, j, k, d, h, best, below: PtrInt;
  cmp: TInferCompare;
  minCos, sc, dcg, idcg, ndcg, a: double;
  scores: array of double;
  used: array of boolean;
  grades: TIntegerDynArray;
  timer: TPrecisionTimer;

  procedure EmbedOne(const aText: RawUtf8; const aRef: TSingleDynArray;
    aRow: PtrInt; aWhat: PtrInt; var aInto: TSingleDynArray);
  begin
    v := emb.Embed([aText]);
    MoveFast(pointer(v)^, aInto[aRow * h], h * SizeOf(single));
    cmp := InferCompare(pointer(v), @aRef[aRow * h], h, 1, 1);
    if cmp.Cosine < minCos then
      minCos := cmp.Cosine;
    if (cmp.Cosine < GATE_COSINE) or (Abs(InferL2Norm(pointer(v), h) - 1) > 1e-5) then
    begin
      inc(below);
      CheckUtf8(false, '% row %: cosine % against lembed',
        [aWhat, aRow, cmp.Cosine]);
    end;
  end;

begin
  fn := RealModel;
  if fn = '' then
    exit;
  data := FindEmbedDe;
  if CheckFailed(data <> '', 'tests/data/embed-de found') then
    exit;
  if CheckFailed(ReadJsonl(data + 'corpus.jsonl', docs) and
     ReadJsonl(data + 'queries.jsonl', queries), 'corpus read') then
    exit;
  h := 1024;
  docRef := ReadF32(data + MakePath(['reference', 'docs.f32']), length(docs), h);
  queryRef := ReadF32(data + MakePath(['reference', 'queries.f32']), length(queries), h);
  if CheckFailed((docRef <> nil) and (queryRef <> nil), 'lembed vectors read') then
    exit;
  timer.Start;
  emb := TInferEmbedder.CreateFromGguf(fn, aBackend);
  try
    CheckEqual(emb.Dim, h, 'dimension');
    CheckEqual(emb.MaxTokens, 512, 'reference contract v1: 512 tokens');
    SetLength(docVec, length(docs) * h);
    SetLength(queryVec, length(queries) * h);
    minCos := 1;
    below := 0;
    if aAll then
    begin
      for i := 0 to high(docs) do
        EmbedOne(docs[i].U['text'], docRef, i, 0, docVec);
      for i := 0 to high(queries) do
        EmbedOne(queries[i].U['text'], queryRef, i, 1, queryVec);
    end
    else
    begin
      // the aShort shortest docs and queries: a quick look at every run
      for k := 0 to 1 do
      begin
        if k = 0 then
          d := length(docs)
        else
          d := length(queries);
        SetLength(pick, d);
        for i := 0 to d - 1 do
          if k = 0 then
            pick[i] := length(docs[i].U['text'])
          else
            pick[i] := length(queries[i].U['text']);
        for j := 1 to MinPtrInt(aShort, d) do
        begin
          best := 0;
          for i := 1 to d - 1 do
            if pick[i] < pick[best] then
              best := i;
          pick[best] := MaxInt;
          if k = 0 then
            EmbedOne(docs[best].U['text'], docRef, best, 0, docVec)
          else
            EmbedOne(queries[best].U['text'], queryRef, best, 1, queryVec);
        end;
      end;
    end;
  finally
    emb.Free;
  end;
  CheckEqual(below, 0, 'every vector with cosine >= 0.999 against lembed');
  AddConsole(FormatString('BGE-M3 Q8 (%): min cosine against lembed %, %',
    [aName, minCos, timer.Stop]));
  if not aAll then
    exit;
  // graded nDCG@10 exactly as the oracle's generator: gain 2^g - 1, log2(rank+1),
  // equal scores in corpus order
  SetLength(scores, length(docs));
  SetLength(used, length(docs));
  SetLength(grades, length(docs));
  ndcg := 0;
  for i := 0 to high(queries) do
  begin
    for j := 0 to high(docs) do
    begin
      sc := 0;
      for k := 0 to h - 1 do
      begin
        a := queryVec[i * h + k]; // widened before multiplying
        sc := sc + a * docVec[j * h + k];
      end;
      scores[j] := sc;
      used[j] := false;
      grades[j] := queries[i].O['rel']^.I[docs[j].U['id']];
    end;
    dcg := 0;
    for k := 1 to 10 do
    begin
      best := -1;
      for j := 0 to high(docs) do
        if not used[j] and ((best < 0) or (scores[j] > scores[best])) then
          best := j;
      used[best] := true;
      dcg := dcg + ((1 shl grades[best]) - 1) * Ln(2) / Ln(k + 1);
    end;
    idcg := 0;
    k := 0;
    for d := 2 downto 1 do
      for j := 0 to high(docs) do
        if (grades[j] = d) and (k < 10) then
        begin
          inc(k);
          idcg := idcg + ((1 shl d) - 1) * Ln(2) / Ln(k + 1);
        end;
    if idcg > 0 then
      ndcg := ndcg + dcg / idcg;
  end;
  ndcg := ndcg / length(queries);
  CheckUtf8(ndcg >= GATE_NDCG10, '% nDCG@10 % >= % (lembed %)',
    [aName, ndcg, GATE_NDCG10, LEMBED_NDCG10]);
  AddConsole(FormatString('BGE-M3 Q8 gate (%): nDCG@10 % (lembed %, gate %), ' +
    'min cosine %', [aName, ndcg, LEMBED_NDCG10, GATE_NDCG10, minCos]));
end;

procedure TTestInferGguf.BgeM3Q8ShortTexts;
begin
  CheckCorpus(false, 3, fCpu, 'reference');
  CheckCorpus(false, 3, fFast, 'fast');
end;

procedure TTestInferGguf.BgeM3Q8CorpusGate;
begin
  if GetEnvironmentVariable('LAI_INFER_GATE') <> '1' then
  begin
    AddConsole('SKIP BGE-M3 Q8 corpus gate: set LAI_INFER_GATE=1 (fast backend, ' +
      'a few minutes; LAI_INFER_GATE_REF=1 adds the scalar reference, an hour)');
    exit;
  end;
  // what ships: the fast backend; LAI_INFER_GATE_REF=1 adds the scalar reference
  // (about an hour)
  CheckCorpus(true, 0, fFast, 'fast');
  if GetEnvironmentVariable('LAI_INFER_GATE_REF') = '1' then
    CheckCorpus(true, 0, fCpu, 'reference');
end;

end.
