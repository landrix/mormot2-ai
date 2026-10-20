/// tests for the fast CPU backend of the native inference engine (phase 1c)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - two contracts of TInferCpuFastBackend against the scalar reference:
//   * Exact = true: the very bits of TInferCpuBackend - work split by output
//     feature, head or element range, never inside one sum; the SIMD kernels
//     return exact integer block sums and scale them in the reference's order
//   * Exact = false (default): float arithmetic where the CPU has a kernel (Q8_0
//     Linear in float lanes like ggml, float attention, a GELU table), checked
//     against the reference within stated tolerances; the activation quantizer
//     and the integer block sums stay exact in both modes
// - one arithmetic per backend whatever the load: a caller that finds the pool
//   busy computes on its own thread with the same kernels, bit for bit
// - the thread pool: every item runs exactly once, also under a burst of tiny
//   jobs, an exception reaches the caller, two callers at once both get the
//   right result
unit test.infer.fast;

interface

{$I mormot.defines.inc}

uses
  classes,
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.kernels,
  mormot.ai.infer.threads,
  mormot.ai.infer.model.xlmr,
  mormot.ai.infer.pipeline.embed,
  mormot.ai.llm.types,
  mormot.ai.embeddings,
  mormot.ai.embed.native,
  test.infer.ref,
  test.infer.gguf;

type
  /// phase 1c: the fast backend against the reference, bit for bit
  TTestInferFast = class(TSynTestCase)
  protected
    fRef: TInferCpuBackend;
    fFast, fExact: TInferCpuFastBackend;
    fHits: TIntegerDynArray;
    fFail: integer;
    fWorkerOf: TIntegerDynArray;
    fThreadOf: TPtrUIntDynArray;
    fPeers: integer;
    procedure Setup; override;
    procedure CleanUp; override;
    procedure CountHit(aItem: integer);
    procedure FailOne(aItem: integer);
    procedure MarkWorker(aItem, aWorker: integer);
    procedure WaitForPeer(aItem, aWorker: integer);
  published
    /// every kernel of this CPU against the plain Pascal loop
    procedure KernelsMatchPascal;
    /// exact mode: Linear F32 and Q8_0, odd shapes included: the same bytes
    procedure LinearIsBitIdentical;
    /// exact mode: Attention (with a mask), GELU and LayerNorm (both modes):
    // the same bytes as the reference
    procedure ElementwiseOpsAreBitIdentical;
    /// Q4_0 Linear: exact on all cores = the reference bit for bit (partial
    // tiles, rows beyond the stack buffer), the float path within tolerance
    procedure LinearQ4MatchesReference;
    /// Q5_0 Linear: the unpacking against its definition, exact = the
    // reference bit for bit, the float path within tolerance
    procedure LinearQ5MatchesReference;
    /// MXFP4 Linear: the unpacking against its definition (every byte, every
    // exponent), exact = the reference bit for bit, the float path within
    // tolerance
    procedure LinearMxfp4MatchesReference;
    /// Q4_K and Q6_K Linear: exact on all cores = the reference bit for bit,
    // the float path within tolerance, both near F32 of the dequantized rows
    procedure LinearKMatchesReference;
    /// default mode: Linear, Attention and GELU within their tolerances
    procedure FastModeWithinTolerance;
    /// the decoder operations (phase 3): CausalAttention exact = the reference
    // bit for bit, float within a tolerance; F32 Linear in float lanes
    procedure DecoderOpsMatchReference;
    /// RmsNorm, SwiGlu and Rope: every mode bit-identical to the reference
    // (rows or slices over the pool), the reference against the definition
    procedure DecoderNormAndSwiGlu;
    /// the thread pool: exactly once, bursts, exceptions
    procedure ThreadPoolRunsEveryItemOnce;
    /// RunWorkers: every item once, its index in 0..Workers, one index one
    // thread (the caller 0), and exceptions as with Run
    procedure ThreadPoolWorkerIndex;
    /// two threads embedding at once through one fast backend
    procedure ConcurrentCallersAreCorrect;
    /// tiny GGUF and, if present, BGE-M3 Q8: exact mode equal to the reference,
    // default mode within tolerance
    procedure ForwardIsBitIdentical;
    /// the IEmbedder adapter: shared per name, Embed = EmbedBatch, long text refused
    procedure NativeEmbedderAdapter;
  end;


/// deterministic pseudo-random values in [-aScale, aScale)
procedure Fill(var aValues: TSingleDynArray; aCount: PtrInt; aSeed: cardinal;
  aScale: single);
/// Q8_0 weights [aOut, aIn] with random scales and every int8 value
function RandomQ8(aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
/// random Q4_0 rows: every nibble, finite scales of both signs
function RandomQ4(aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
/// Q5_0 weights [aOut, aIn] with random scales, nibbles and fifth bits
function RandomQ5(aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
/// MXFP4 weights [aOut, aIn] with random nibbles and exponents in 110..140
function RandomMxfp4(aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
/// random K-quant rows (Q4_K, Q6_K): every bit pattern, finite scales
function RandomK(aType: TInferDType; aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
/// true when both float arrays hold the same bits
function SameBits(a, b: PSingleArray; aCount: PtrInt): boolean;
/// max |a - b| relative to max |b| - a scale-free error of a whole output
function RelError(a, b: PSingleArray; aCount: PtrInt): double;

implementation

/// deterministic pseudo-random values in [-aScale, aScale)
procedure Fill(var aValues: TSingleDynArray; aCount: PtrInt; aSeed: cardinal;
  aScale: single);
var
  i: PtrInt;
begin
  SetLength(aValues, aCount);
  for i := 0 to aCount - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    aValues[i] := ((aSeed shr 8) / 16777216 * 2 - 1) * aScale;
  end;
end;

/// Q8_0 weights [aOut, aIn] with random scales and every int8 value
function RandomQ8(aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
var
  b, j: PtrInt;
  blk: PInferBlockQ8_0;
begin
  SetLength(result, aOut * (aIn div 32) * SizeOf(TInferBlockQ8_0));
  blk := pointer(result);
  for b := 0 to aOut * (aIn div 32) - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    blk^.d := InferSingleToHalf(((aSeed shr 8) and 1023) / 65536 + 1e-4);
    for j := 0 to 31 do
    begin
      aSeed := aSeed * 1103515245 + 12345;
      blk^.qs[j] := shortint(aSeed shr 24); // -128 .. 127
    end;
    inc(blk);
  end;
end;

/// random Q4_0 rows: every nibble, scales of both signs
function RandomQ4(aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
var
  b, j: PtrInt;
  blk: PInferBlockQ4_0;
begin
  SetLength(result, aOut * (aIn div 32) * SizeOf(TInferBlockQ4_0));
  blk := pointer(result);
  for b := 0 to aOut * (aIn div 32) - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    blk^.d := InferSingleToHalf((integer((aSeed shr 8) and 1023) - 512) / 16384 + 1e-4);
    for j := 0 to 15 do
    begin
      aSeed := aSeed * 1103515245 + 12345;
      blk^.qs[j] := aSeed shr 24;
    end;
    inc(blk);
  end;
end;

/// 2^n in double, exact for -1022..1023
function Pow2(n: integer): double;
var
  i: integer;
begin
  result := 1;
  for i := 1 to abs(n) do
    result := result * 2;
  if n < 0 then
    result := 1 / result;
end;

/// MXFP4 weights [aOut, aIn] with random nibbles and exponents in 110..140
function RandomMxfp4(aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
var
  b, j: PtrInt;
  blk: PInferBlockMxfp4;
begin
  SetLength(result, aOut * (aIn div 32) * SizeOf(TInferBlockMxfp4));
  blk := pointer(result);
  for b := 0 to aOut * (aIn div 32) - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    blk^.e := 110 + (aSeed shr 16) mod 31;
    for j := 0 to 15 do
    begin
      aSeed := aSeed * 1103515245 + 12345;
      blk^.qs[j] := aSeed shr 24;
    end;
    inc(blk);
  end;
end;

function RandomQ5(aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
var
  b, j: PtrInt;
  blk: PInferBlockQ5_0;
begin
  SetLength(result, aOut * (aIn div 32) * SizeOf(TInferBlockQ5_0));
  blk := pointer(result);
  for b := 0 to aOut * (aIn div 32) - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    blk^.d := InferSingleToHalf((integer((aSeed shr 8) and 1023) - 512) / 16384 + 1e-4);
    for j := 0 to 3 do
    begin
      aSeed := aSeed * 1103515245 + 12345;
      blk^.qh[j] := aSeed shr 24;
    end;
    for j := 0 to 15 do
    begin
      aSeed := aSeed * 1103515245 + 12345;
      blk^.qs[j] := aSeed shr 24;
    end;
    inc(blk);
  end;
end;

function SameBits(a, b: PSingleArray; aCount: PtrInt): boolean;
begin
  result := CompareMem(a, b, aCount * SizeOf(single));
end;


{ TTestInferFast }

procedure TTestInferFast.Setup;
begin
  fRef := TInferCpuBackend.Create;
  fFast := TInferCpuFastBackend.Create(4); // a pool even on a 1-core runner
  fExact := TInferCpuFastBackend.Create(4);
  fExact.Exact := true;
end;

procedure TTestInferFast.CleanUp;
begin
  FreeAndNil(fExact); // re-entrant, as mORMot requires
  FreeAndNil(fFast);
  FreeAndNil(fRef);
end;

function DotD(a, b: PSingleArray; n: PtrInt): double;
var
  i: PtrInt;
  v: double;
begin
  result := 0;
  for i := 0 to n - 1 do
  begin
    v := a^[i];
    result := result + v * b^[i];
  end;
end;

function SumAbs(a, b: PSingleArray; n: PtrInt): double;
var
  i: PtrInt;
  v: double;
begin
  result := 0;
  for i := 0 to n - 1 do
  begin
    v := a^[i];
    result := result + Abs(v * b^[i]);
  end;
end;

/// sum over j of p[j] * v[j * n + d]
function AxpyD(p, v: PSingleArray; n, aKeys, d: PtrInt): double;
var
  j: PtrInt;
  a: double;
begin
  result := 0;
  for j := 0 to aKeys - 1 do
  begin
    a := p^[j];
    result := result + a * v^[j * n + d];
  end;
end;

/// max |a - b| relative to max |b| - a scale-free error of a whole output
function RelError(a, b: PSingleArray; aCount: PtrInt): double;
var
  i: PtrInt;
  m, e: double;
begin
  m := 0;
  e := 0;
  for i := 0 to aCount - 1 do
  begin
    // a NaN or an infinity on either side is an error, not a comparison that
    // happens to be false
    if (PCardinal(@a^[i])^ and $7f800000 = $7f800000) or
       (PCardinal(@b^[i])^ and $7f800000 = $7f800000) then
    begin
      result := 1e30;
      exit;
    end;
    if Abs(b^[i]) > m then
      m := Abs(b^[i]);
    if Abs(a^[i] - b^[i]) > e then
      e := Abs(a^[i] - b^[i]);
  end;
  if m = 0 then
    result := e
  else
    result := e / m;
end;

procedure TTestInferFast.KernelsMatchPascal;
var
  xs, d1, d2, g: TSingleDynArray;
  q1, q2: RawByteString;
  r, j, n: PtrInt;
  w, x: RawByteString;
  nb, b, k: PtrInt;
  s1, s2: TIntegerDynArray;
  dx, acc, accRef, magn: TDoubleDynArray;
  dxs, out8: TSingleDynArray;
  wb: PInferBlockQ8_0;
  a8: double;
begin
  AddConsole(Utf8ToString('Q8_0 kernel: ' + InferQ8KernelName));
  // the activation quantizer of this CPU against the Pascal loop: same bytes,
  // ties, a zero block, subnormal scales, large values included
  for r := 0 to 8 do
  begin
    Fill(xs, 32 * 9, 900 + r, 10);
    case r of
      1:
        for j := 0 to 31 do
          xs[j] := 0;
      2:
        for j := 32 to 63 do
          xs[j] := xs[j] * 1e-38;
      3:
        for j := 0 to high(xs) do
          xs[j] := xs[j] * 1e30;
      4:
        begin
          xs[0] := 127;
          for j := 1 to 31 do
            xs[j] := j - 15.5; // halves: ties to even
        end;
      5:
        begin
          for j := 64 to 95 do
            xs[j] := xs[j] * 1e-41;
          // a scale just above 2^-128: 1 / d close to the largest single
          for j := 96 to 127 do
            xs[j] := xs[j] * 5e-37;
        end;
      6:
        // a valid scale (787) from values beyond the largest half: no
        // intermediate of the quantizer may overflow on the way
        for j := 0 to 63 do
          xs[j] := 100000 - j * 1000;
      7, 8:
        begin
          // the zero threshold itself: amax = 127 * 2^-128 gives d = 2^-128
          // exactly (all q = 0); four ulps more give d = 2^-128 + 2^-149, the
          // first scale above it (one ulp still rounds back to 2^-128)
          for j := 32 to 63 do
            xs[j] := xs[j] * 1e-40;
          PCardinal(@xs[32])^ := $02FE0000 + 4 * cardinal(r - 7);
        end;
    end;
    SetLength(q1, 9 * SizeOf(TInferBlockQ8_0));
    SetLength(q2, 9 * SizeOf(TInferBlockQ8_0));
    InferQuantizeRowQ8_0(pointer(xs), length(xs), pointer(q1));
    InferQuantizeRowQ8_0Pascal(pointer(xs), length(xs), pointer(q2));
    CheckUtf8(q1 = q2, 'quantizer = Pascal, row %', [r]);
    if r >= 7 then
      // block 1 starts with its largest value: q = 0 at the threshold, 127 above
      CheckEqual(PInferBlockQ8_0Array(pointer(q1))^[1].qs[0], 127 * (r - 7),
        'quantizer at the zero threshold');
  end;
  if Assigned(InferQ8QuantizeRow) then
  begin
    // a NaN in block 3 of 9: status 1 + 3, blocks 0..2 written as by Pascal,
    // nothing written for block 3 or after it
    Fill(xs, 32 * 9, 950, 10);
    InferQuantizeRowQ8_0Pascal(pointer(xs), 3 * 32, pointer(q2));
    PCardinal(@xs[3 * 32 + 17])^ := $7fc00000;
    q1 := '';
    SetLength(q1, 9 * SizeOf(TInferBlockQ8_0));
    FillCharFast(pointer(q1)^, length(q1), $AA);
    j := InferQ8QuantizeRow(pointer(xs), 9, pointer(q1));
    CheckEqual(j, 4, 'quantizer status of a NaN in block 3');
    CheckUtf8(CompareMem(pointer(q1), pointer(q2), 3 * SizeOf(TInferBlockQ8_0)),
      'quantizer: the blocks before the bad one', []);
    for j := 3 * SizeOf(TInferBlockQ8_0) to length(q1) - 1 do
      if q1[j + 1] <> #$AA then
      begin
        CheckUtf8(false, 'quantizer wrote byte % after the bad block', [j]);
        break;
      end;
  end;
  // the float attention kernels against double sums
  if Assigned(InferDotsF32) then
  begin
    // rows of n values, n + 16 apart (as heads inside a wider row)
    n := 64;
    Fill(xs, (n + 16) * 7, 31, 2);
    Fill(d1, n, 32, 2);
    SetLength(d2, 8);
    d2[7] := 12345;
    InferDotsF32(pointer(d1), pointer(xs), (n + 16) * SizeOf(single), 7, n,
      pointer(d2));
    for j := 0 to 6 do
      CheckUtf8(Abs(d2[j] - DotD(pointer(d1), @xs[j * (n + 16)], n)) <=
        1e-5 * SumAbs(pointer(d1), @xs[j * (n + 16)], n), 'float dot %', [j]);
    CheckUtf8(d2[7] = 12345, 'float dot: nothing beyond the keys', []);
    d2[0] := 12345;
    InferDotsF32(pointer(d1), pointer(xs), (n + 16) * SizeOf(single), 0, n,
      pointer(d2));
    CheckUtf8(d2[0] = 12345, 'float dot: no key, nothing written', []);
    SetLength(d2, n + 1);
    FillCharFast(pointer(d2)^, n * SizeOf(single), 0);
    d2[n] := 12345;
    InferAxpyRowsF32(pointer(d1), pointer(xs), (n + 16) * SizeOf(single), 7, n,
      pointer(d2));
    for j := 0 to n - 1 do
      CheckUtf8(Abs(d2[j] - AxpyD(pointer(d1), pointer(xs), n + 16, 7, j)) <= 1e-4,
        'float axpy %', [j]);
    CheckUtf8(d2[n] = 12345, 'float axpy: nothing beyond the row', []);
    g := copy(d2);
    InferAxpyRowsF32(pointer(d1), pointer(xs), (n + 16) * SizeOf(single), 0, n,
      pointer(d2));
    CheckUtf8(SameBits(pointer(g), pointer(d2), n + 1),
      'float axpy: no key, nothing written', []);
  end;
  for nb := 0 to 9 do
  begin
    w := RandomQ8(4, nb * 32, 7 + nb);
    x := RandomQ8(1, nb * 32, 99 + nb);
    SetLength(s1, nb * 4 + 1);
    SetLength(s2, nb * 4 + 1);
    InferQ8BlockDots(pointer(w), pointer(x), nb, pointer(s1));
    InferQ8BlockDotsPascal(pointer(w), pointer(x), nb, pointer(s2));
    CheckUtf8(CompareMem(pointer(s1), pointer(s2), nb * 4), '1-row kernel, % blocks', [nb]);
    InferQ8BlockDots4(pointer(w), pointer(x), nb, pointer(s1),
      nb * SizeOf(TInferBlockQ8_0));
    InferQ8BlockDots4Pascal(pointer(w), pointer(x), nb, pointer(s2),
      nb * SizeOf(TInferBlockQ8_0));
    CheckUtf8(CompareMem(pointer(s1), pointer(s2), nb * 16), '4-row kernel, % blocks', [nb]);
    SetLength(dx, nb + 1);
    for b := 0 to nb - 1 do
      dx[b] := InferHalfToSingle(PInferBlockQ8_0Array(pointer(x))^[b].d);
    if Assigned(InferQ8Linear4) then
    begin
      // the fused kernel against the reference arithmetic, row by row
      SetLength(acc, 4);
      SetLength(accRef, 4);
      for k := 0 to 3 do
      begin
        acc[k] := k - 1.5; // a bias
        accRef[k] := acc[k];
        for b := 0 to nb - 1 do
        begin
          wb := @PInferBlockQ8_0Array(pointer(w))^[k * nb + b];
          accRef[k] := accRef[k] + s2[b * 4 + k] * (InferHalfToSingle(wb^.d) * dx[b]);
        end;
      end;
      InferQ8Linear4(pointer(w), pointer(x), nb, pointer(dx),
        nb * SizeOf(TInferBlockQ8_0), pointer(acc));
      CheckUtf8(CompareMem(pointer(acc), pointer(accRef), 4 * SizeOf(double)),
        'fused kernel = reference arithmetic, % blocks', [nb]);
    end;
    if not Assigned(InferQ8DotF32x4x2) then
      continue;
    // the float kernel: two different activation rows, within float rounding
    // of the exact double sums (no bias); activations as the quantizer writes
    // them, -127 .. 127 (the AVX2 kernel relies on it)
    x := RandomQ8(2, nb * 32, 199 + nb);
    for b := 0 to nb * 2 - 1 do
      for j := 0 to 31 do
        if PInferBlockQ8_0Array(pointer(x))^[b].qs[j] = -128 then
          PInferBlockQ8_0Array(pointer(x))^[b].qs[j] := -127;
    SetLength(dxs, nb * 2 + 1);
    for b := 0 to nb * 2 - 1 do
      dxs[b] := InferHalfToSingle(PInferBlockQ8_0Array(pointer(x))^[b].d);
    SetLength(out8, 8);
    InferQ8DotF32x4x2(pointer(w), pointer(x),
      @PInferBlockQ8_0Array(pointer(x))^[nb], nb, pointer(dxs), @dxs[nb],
      nb * SizeOf(TInferBlockQ8_0), pointer(out8));
    SetLength(accRef, 8);
    SetLength(magn, 8);
    for r := 0 to 1 do
    begin
      InferQ8BlockDots4Pascal(pointer(w), @PInferBlockQ8_0Array(pointer(x))^[r * nb],
        nb, pointer(s2), nb * SizeOf(TInferBlockQ8_0));
      for k := 0 to 3 do
      begin
        accRef[r * 4 + k] := 0;
        magn[r * 4 + k] := 0;
        for b := 0 to nb - 1 do
        begin
          wb := @PInferBlockQ8_0Array(pointer(w))^[k * nb + b];
          a8 := s2[b * 4 + k] * (InferHalfToSingle(wb^.d) * dxs[r * nb + b]);
          accRef[r * 4 + k] := accRef[r * 4 + k] + a8;
          magn[r * 4 + k] := magn[r * 4 + k] + Abs(a8);
        end;
      end;
    end;
    for k := 0 to 7 do
      CheckUtf8(Abs(out8[k] - accRef[k]) <= 1e-5 * (magn[k] + 1),
        'float kernel output %, % blocks: % <> %', [k, nb, out8[k], accRef[k]]);
  end;
end;

procedure TTestInferFast.LinearIsBitIdentical;
const
  SHAPES: array[0..5, 0..2] of integer = (
    (1, 32, 1), (3, 64, 5), (7, 96, 17), (37, 256, 33), (64, 1024, 100),
    (5, 4096, 64));
var
  i: PtrInt;
  x, wf, bias, y1, y2: TSingleDynArray;
  wq: RawByteString;
  rows, nin, nout: integer;
begin
  for i := 0 to high(SHAPES) do
  begin
    rows := SHAPES[i, 0];
    nin := SHAPES[i, 1];
    nout := SHAPES[i, 2];
    Fill(x, rows * nin, 11 + i, 3);
    Fill(wf, nout * nin, 23 + i, 0.5);
    Fill(bias, nout, 31 + i, 1);
    wq := RandomQ8(nout, nin, 41 + i);
    SetLength(y1, rows * nout);
    SetLength(y2, rows * nout);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wf), idtF32), nout,
      pointer(bias), pointer(y1));
    fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wf), idtF32), nout,
      pointer(bias), pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), 'F32 % x % x %',
      [rows, nin, nout]);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      pointer(bias), pointer(y1));
    fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      pointer(bias), pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), 'Q8_0 % x % x %',
      [rows, nin, nout]);
    // without bias, too
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      nil, pointer(y1));
    fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      nil, pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), 'Q8_0 no bias % x % x %',
      [rows, nin, nout]);
  end;
end;

/// random K-quant rows: every bit pattern of the quants and scales, finite
// half-float scales of both signs
function RandomK(aType: TInferDType; aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
var
  i, b, nb, size: PtrInt;
  p: PByteArray;
begin
  size := INFER_DTYPE_SIZE[aType];
  nb := aOut * (aIn div 256);
  SetLength(result, nb * size);
  p := pointer(result);
  for i := 0 to nb * size - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    p^[i] := aSeed shr 24;
  end;
  for b := 0 to nb - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    if aType = idtQ4_K then
    begin
      PInferBlockQ4_K(@p^[b * size])^.d :=
        InferSingleToHalf(((aSeed shr 8) and 1023) / 65536 + 1e-4);
      PInferBlockQ4_K(@p^[b * size])^.dmin :=
        InferSingleToHalf(((aSeed shr 18) and 1023) / 131072);
    end
    else
      PInferBlockQ6_K(@p^[b * size])^.d :=
        InferSingleToHalf((integer((aSeed shr 8) and 1023) - 512) / 65536 + 1e-4);
  end;
end;

procedure TTestInferFast.LinearKMatchesReference;
const
  // rows, in, out: partial tiles, several super-blocks
  // 513 blocks of 256: beyond the 512 the dot product takes at a time, in two
  // rows (the two-row kernel crosses that limit too)
  SHAPES: array[0..5, 0..2] of integer = (
    (1, 256, 1), (3, 512, 17), (37, 1024, 33), (5, 3072, 20), (64, 256, 40),
    (2, 513 * 256, 2));
var
  i, k, r, o, rows, nin, nout, nb: PtrInt;
  mine, pascal: TIntegerDynArray;
  xk, xk2: array of TInferBlockQ8_K;
  x, x2, bias, y1, y2, y3, wf: TSingleDynArray;
  wq: RawByteString;
  kind: TInferDType;
  e: double;
  cmp: TInferCompare;
  k2: TInferKSums2;
begin
  for k := 0 to 1 do
  begin
    if k = 0 then
      kind := idtQ4_K
    else
      kind := idtQ6_K;
    for i := 0 to high(SHAPES) do
    begin
      rows := SHAPES[i, 0];
      nin := SHAPES[i, 1];
      nout := SHAPES[i, 2];
      Fill(x, rows * nin, 311 + i, 3);
      Fill(bias, nout, 331 + i, 1);
      wq := RandomK(kind, nout, nin, 341 + i + 100 * k);
      SetLength(y1, rows * nout);
      SetLength(y2, rows * nout);
      SetLength(y3, rows * nout);
      fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), kind), nout,
        pointer(bias), pointer(y1));
      fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), kind), nout,
        pointer(bias), pointer(y2));
      CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), '% exact % x % x %',
        [INFER_DTYPE_NAME[kind], rows, nin, nout]);
      fFast.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), kind), nout,
        pointer(bias), pointer(y3));
      e := RelError(pointer(y3), pointer(y1), rows * nout);
      CheckUtf8(e <= 2e-6, '% float % x % x %: relative error %',
        [INFER_DTYPE_NAME[kind], rows, nin, nout, e]);
      // the kernel of this CPU (NEON, AVX2) gives the ints of the Pascal one
      nb := nin div 256;
      SetLength(xk, nb);
      InferQuantizeRowQ8_K(pointer(x), nin, pointer(xk));
      SetLength(mine, nb * 2);
      SetLength(pascal, nb * 2);
      if kind = idtQ4_K then
      begin
        InferQ4KSums(pointer(wq), pointer(xk), nb, pointer(mine));
        InferQ4KSumsPascal(pointer(wq), pointer(xk), nb, pointer(pascal));
      end
      else
      begin
        InferQ6KSums(pointer(wq), pointer(xk), nb, pointer(mine));
        InferQ6KSumsPascal(pointer(wq), pointer(xk), nb, pointer(pascal));
      end;
      CheckUtf8(CompareMem(pointer(mine), pointer(pascal), nb * 2 * SizeOf(integer)),
        '% kernel (%) = Pascal, % blocks', [INFER_DTYPE_NAME[kind],
        InferQ8KernelName, nb]);
      if kind = idtQ4_K then
        k2 := InferQ4KSums2
      else
        k2 := InferQ6KSums2;
      if Assigned(k2) then
      begin
        // the fused kernel for two activation rows = the Pascal one
        SetLength(xk2, nb * 2);
        InferQuantizeRowQ8_K(pointer(x), nin, pointer(xk2));
        Fill(x2, nin, 351 + i, 3);
        InferQuantizeRowQ8_K(pointer(x2), nin, @xk2[nb]);
        SetLength(mine, nb * 4);
        SetLength(pascal, nb * 4);
        k2(pointer(wq), pointer(xk2), nb, pointer(mine), nb * SizeOf(TInferBlockQ8_K));
        if kind = idtQ4_K then
          InferQ4KSums2Pascal(pointer(wq), pointer(xk2), nb, pointer(pascal),
            nb * SizeOf(TInferBlockQ8_K))
        else
          InferQ6KSums2Pascal(pointer(wq), pointer(xk2), nb, pointer(pascal),
            nb * SizeOf(TInferBlockQ8_K));
        CheckUtf8(CompareMem(pointer(mine), pointer(pascal), nb * 4 * SizeOf(integer)),
          '% two-row kernel (%) = Pascal, % blocks', [INFER_DTYPE_NAME[kind],
          InferQ8KernelName, nb]);
      end;
      // independent of the block sums: F32 on the dequantized weights, apart
      // only by the rounding of the activations to Q8_K
      SetLength(wf, nout * nin);
      for o := 0 to nout - 1 do
        InferRowToF32(pointer(wq), kind, o, nin, @wf[o * nin]);
      fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wf), idtF32), nout,
        pointer(bias), pointer(y3));
      // (random weights: outputs in the thousands, so a direction, not a bound)
      cmp := InferCompare(pointer(y1), pointer(y3), rows * nout, 0.05, 0.02);
      CheckUtf8(cmp.Cosine > 0.9999,
        '% against F32 of its dequantized weights, % x % x %: %',
        [INFER_DTYPE_NAME[kind], rows, nin, nout, InferCompareText(cmp)]);
    end;
  end;
  r := 0; // shapes with a partial block are refused
  try
    fFast.Linear(pointer(x), 1, 300, InferWeight(pointer(wq), idtQ6_K), 1, nil,
      pointer(y1));
  except
    on EInferError do
      r := 1;
  end;
  Check(r = 1, 'K-quants need whole blocks of 256');
end;

procedure TTestInferFast.LinearQ4MatchesReference;
const
  // rows, in, out: partial tiles, and 4352 inputs = 136 blocks per row, beyond
  // the tile buffer on the stack
  SHAPES: array[0..5, 0..2] of integer = (
    (1, 32, 1), (3, 96, 17), (37, 256, 33), (64, 1024, 100), (5, 3072, 48),
    (2, 4352, 20));
var
  i, j, rows, nin, nout: PtrInt;
  x, bias, y1, y2, y3: TSingleDynArray;
  wq: RawByteString;
  e: double;
  q4: array[0..15] of TInferBlockQ4_0;
  q8, q8b: array[0..15] of TInferBlockQ8_0;
  ok: boolean;
begin
  // the unpacking every Q4_0 path shares, against its definition: all 256 bytes
  for i := 0 to 15 do
  begin
    q4[i].d := i * 7 + 1;
    for j := 0 to 15 do
      q4[i].qs[j] := i * 16 + j;
  end;
  InferQ4ToQ8(@q4, 16, @q8);
  ok := true;
  for i := 0 to 15 do
  begin
    if q8[i].d <> q4[i].d then
      ok := false;
    for j := 0 to 15 do
      if (q8[i].qs[j] <> (q4[i].qs[j] and 15) - 8) or
         (q8[i].qs[j + 16] <> (q4[i].qs[j] shr 4) - 8) then
        ok := false;
  end;
  Check(ok, 'InferQ4ToQ8 = q - 8 for all 256 bytes');
  if Assigned(InferQ4Unpack) then
  begin
    FillCharFast(q8, SizeOf(q8), 0);
    InferQ4Unpack(@q4, 16, @q8);
    InferQ4ToQ8(@q4, 16, @q8b);
    Check(CompareMem(@q8, @q8b, SizeOf(q8)), 'Q4_0 unpack kernel = Pascal');
  end;
  for i := 0 to high(SHAPES) do
  begin
    rows := SHAPES[i, 0];
    nin := SHAPES[i, 1];
    nout := SHAPES[i, 2];
    Fill(x, rows * nin, 211 + i, 3);
    Fill(bias, nout, 231 + i, 1);
    wq := RandomQ4(nout, nin, 241 + i);
    SetLength(y1, rows * nout);
    SetLength(y2, rows * nout);
    SetLength(y3, rows * nout);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ4_0), nout,
      pointer(bias), pointer(y1));
    fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ4_0), nout,
      pointer(bias), pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), 'Q4_0 exact % x % x %',
      [rows, nin, nout]);
    fFast.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ4_0), nout,
      nil, pointer(y3));
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ4_0), nout,
      nil, pointer(y1));
    e := RelError(pointer(y3), pointer(y1), rows * nout);
    CheckUtf8(e <= 2e-6, 'Q4_0 float % x % x %: relative error %', [rows, nin, nout, e]);
  end;
end;

procedure TTestInferFast.LinearQ5MatchesReference;
const
  // rows, in, out: partial tiles, 896 inputs (Qwen2.5-0.5B: no multiple of
  // 256), and 4352 inputs = 136 blocks per row, beyond the tile buffer on the
  // stack
  SHAPES: array[0..5, 0..2] of integer = (
    (1, 32, 1), (3, 96, 17), (37, 896, 33), (64, 1024, 100), (1, 896, 70),
    (2, 4352, 20));
var
  i, j, k, rows, nin, nout: PtrInt;
  x, bias, y1, y2, y3, row: TSingleDynArray;
  wq: RawByteString;
  e: double;
  q5: array[0..15] of TInferBlockQ5_0;
  q8: array[0..15] of TInferBlockQ8_0;
  qh: cardinal;
  ok: boolean;
begin
  // the unpacking against the definition of a Q5_0 value: every byte value
  // in the nibbles, every fifth bit set and clear
  for i := 0 to 15 do
  begin
    q5[i].d := i * 7 + 1;
    for j := 0 to 15 do
      q5[i].qs[j] := i * 16 + j;
    qh := cardinal(i) * 2654435761 xor $A5A5A5A5;
    if i = 0 then
      qh := 0
    else if i = 1 then
      qh := $FFFFFFFF;
    for j := 0 to 3 do
      q5[i].qh[j] := (qh shr (j * 8)) and 255;
  end;
  InferQ5ToQ8(@q5, 16, @q8);
  ok := true;
  SetLength(row, 32);
  for i := 0 to 15 do
  begin
    if q8[i].d <> q5[i].d then
      ok := false;
    qh := q5[i].qh[0] or (cardinal(q5[i].qh[1]) shl 8) or
      (cardinal(q5[i].qh[2]) shl 16) or (cardinal(q5[i].qh[3]) shl 24);
    for k := 0 to 31 do
    begin
      if k < 16 then
        j := q5[i].qs[k] and 15
      else
        j := q5[i].qs[k - 16] shr 4;
      j := j + PtrInt((qh shr k) and 1) * 16 - 16;
      if q8[i].qs[k] <> j then
        ok := false;
    end;
    // and the dequantization: d * (q - 16) per value
    InferRowToF32(@q5, idtQ5_0, i, 32, pointer(row));
    for k := 0 to 31 do
      if row[k] <> InferHalfToSingle(q5[i].d) * q8[i].qs[k] then
        ok := false;
  end;
  Check(ok, 'InferQ5ToQ8 and InferRowToF32 = d * (q - 16) for all values');
  for i := 0 to high(SHAPES) do
  begin
    rows := SHAPES[i, 0];
    nin := SHAPES[i, 1];
    nout := SHAPES[i, 2];
    Fill(x, rows * nin, 311 + i, 3);
    Fill(bias, nout, 331 + i, 1);
    wq := RandomQ5(nout, nin, 341 + i);
    SetLength(y1, rows * nout);
    SetLength(y2, rows * nout);
    SetLength(y3, rows * nout);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ5_0), nout,
      pointer(bias), pointer(y1));
    fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ5_0), nout,
      pointer(bias), pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), 'Q5_0 exact % x % x %',
      [rows, nin, nout]);
    fFast.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ5_0), nout,
      nil, pointer(y3));
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ5_0), nout,
      nil, pointer(y1));
    e := RelError(pointer(y3), pointer(y1), rows * nout);
    CheckUtf8(e <= 2e-6, 'Q5_0 float % x % x %: relative error %', [rows, nin, nout, e]);
  end;
end;

procedure TTestInferFast.LinearMxfp4MatchesReference;
const
  // rows, in, out: partial tiles, 2880 inputs (gpt-oss-20b: 90 blocks, no
  // multiple of 256), 4352 inputs = 136 blocks, beyond the tile buffer on the
  // stack
  SHAPES: array[0..5, 0..2] of integer = (
    (1, 32, 1), (3, 96, 17), (37, 2880, 33), (64, 1024, 100), (1, 2880, 70),
    (2, 4352, 20));
  VALUES: array[0..15] of single = (
    0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6);
var
  i, j, k, e, rows, nin, nout: PtrInt;
  x, bias, y1, y2, y3, row: TSingleDynArray;
  wq: RawByteString;
  err, scale, want: double;
  mx: array[0..15] of TInferBlockMxfp4;
  q8, q8b: array[0..15] of TInferBlockQ8_0;
  half: array[byte] of word;
  ok, okRow, okKernel: boolean;
begin
  // the FP16 scale of each exponent as InferMxfp4ToQ8 writes it: the table a
  // kernel takes
  for e := 0 to 255 do
  begin
    mx[0].e := e;
    InferMxfp4ToQ8(@mx, 1, @q8);
    half[e] := q8[0].d;
  end;
  okKernel := true;
  // the unpacking against the definition of an MXFP4 value: every byte value
  // in the nibbles, every exponent - FP16 scale 2^(e - 128) in the exact
  // range, 0 outside it; the dequantization 2^(e - 127) times the E2M1 value
  ok := true;
  okRow := true;
  SetLength(row, 32);
  for e := 0 to 255 do
  begin
    for i := 0 to 15 do
    begin
      mx[i].e := e;
      for j := 0 to 15 do
        mx[i].qs[j] := i * 16 + j;
    end;
    InferMxfp4ToQ8(@mx, 16, @q8);
    if Assigned(InferMxfp4Unpack) then
    begin
      FillCharFast(q8b, SizeOf(q8b), 0);
      InferMxfp4Unpack(@mx, 16, @q8b, @half, @INFER_MXFP4_VALUES);
      if not CompareMem(@q8, @q8b, SizeOf(q8)) then
        okKernel := false;
    end;
    scale := Pow2(e - 128);
    for i := 0 to 15 do
    begin
      // FP16 holds 2^-24..2^15 exactly: e = 104..143 (not the constants
      // themselves, or a wrong constant would pass)
      if (e >= 104) and (e <= 143) then
      begin
        if InferHalfToSingle(q8[i].d) <> scale then
          ok := false;
      end
      else if q8[i].d <> 0 then
        ok := false;
      for k := 0 to 31 do
      begin
        if k < 16 then
          j := mx[i].qs[k] and 15
        else
          j := mx[i].qs[k - 16] shr 4;
        if q8[i].qs[k] <> 2 * VALUES[j] then
          ok := false;
      end;
      InferRowToF32(@mx, idtMXFP4, i, 32, pointer(row));
      for k := 0 to 31 do
      begin
        if k < 16 then
          j := mx[i].qs[k] and 15
        else
          j := mx[i].qs[k - 16] shr 4;
        // ggml: e 0 and 1 are 2^-128 and 2^-127, 255 is 2^127 (no NaN);
        // from 2^128 on an infinity of the sign of the value
        want := 2 * scale * VALUES[j];
        if Abs(want) >= Pow2(128) then
        begin
          if (PCardinal(@row[k])^ and $7fffffff <> $7f800000) or
             ((PCardinal(@row[k])^ shr 31 = 1) <> (want < 0)) then
            okRow := false;
        end
        else if row[k] <> want then
          okRow := false;
      end;
    end;
  end;
  Check(ok, 'InferMxfp4ToQ8: the doubled E2M1 values, 2^(e - 128) in the range');
  Check(okRow, 'InferRowToF32 = 2^(e - 127) * E2M1 for every exponent');
  Check(okKernel, 'MXFP4 unpack kernel = Pascal for every byte and exponent');
  for i := 0 to high(SHAPES) do
  begin
    rows := SHAPES[i, 0];
    nin := SHAPES[i, 1];
    nout := SHAPES[i, 2];
    Fill(x, rows * nin, 411 + i, 3);
    Fill(bias, nout, 431 + i, 1);
    wq := RandomMxfp4(nout, nin, 441 + i);
    SetLength(y1, rows * nout);
    SetLength(y2, rows * nout);
    SetLength(y3, rows * nout);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtMXFP4), nout,
      pointer(bias), pointer(y1));
    fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtMXFP4), nout,
      pointer(bias), pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), 'MXFP4 exact % x % x %',
      [rows, nin, nout]);
    fFast.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtMXFP4), nout,
      nil, pointer(y3));
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtMXFP4), nout,
      nil, pointer(y1));
    err := RelError(pointer(y3), pointer(y1), rows * nout);
    CheckUtf8(err <= 2e-6, 'MXFP4 float % x % x %: relative error %',
      [rows, nin, nout, err]);
  end;
end;

procedure TTestInferFast.ElementwiseOpsAreBitIdentical;
var
  q, k, v, y1, y2, s, g1, g2: TSingleDynArray;
  keep: TIntegerDynArray;
  seq, heads, hd, i: PtrInt;
begin
  for seq := 1 to 40 do
    if seq in [1, 2, 15, 16, 17, 40] then
    begin
      heads := 4;
      hd := 8;
      Fill(q, seq * heads * hd, seq, 2);
      Fill(k, seq * heads * hd, seq + 100, 2);
      Fill(v, seq * heads * hd, seq + 200, 2);
      SetLength(keep, seq);
      for i := 0 to seq - 1 do
        keep[i] := ord((i = 0) or (i mod 5 <> 3)); // a few masked keys
      SetLength(s, seq * seq);
      SetLength(y1, seq * heads * hd);
      SetLength(y2, seq * heads * hd);
      fRef.Attention(pointer(q), pointer(k), pointer(v), seq, heads, hd,
        pointer(keep), pointer(s), pointer(y1));
      fExact.Attention(pointer(q), pointer(k), pointer(v), seq, heads, hd,
        pointer(keep), pointer(s), pointer(y2));
      CheckUtf8(SameBits(pointer(y1), pointer(y2), seq * heads * hd),
        'attention, % tokens', [seq]);
    end;
  for i := 0 to 2 do
  begin
    Fill(g1, 5000 + i * 7000, 77 + i, 12); // beyond and below the split size
    g2 := copy(g1);
    fRef.Gelu(pointer(g1), length(g1));
    fExact.Gelu(pointer(g2), length(g2));
    CheckUtf8(SameBits(pointer(g1), pointer(g2), length(g1)), 'gelu of %',
      [length(g1)]);
  end;
  // LayerNorm: rows split over the pool, every row in the reference
  // arithmetic - in both modes; below, at and above the parallel threshold,
  // into another buffer and in place
  for i := 0 to 4 do
  begin
    seq := 1 + i * 16; // 1, 17, 33, 49, 65 rows
    Fill(g1, seq * 1024, 300 + i, 4);
    Fill(q, 1024, 400 + i, 1);
    Fill(k, 1024, 500 + i, 1);
    SetLength(y1, length(g1));
    SetLength(y2, length(g1));
    fRef.LayerNorm(pointer(g1), seq, 1024, pointer(q), pointer(k), 1e-5,
      pointer(y1));
    fFast.LayerNorm(pointer(g1), seq, 1024, pointer(q), pointer(k), 1e-5,
      pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), length(y1)),
      'layernorm, % rows', [seq]);
    g2 := copy(g1);
    fExact.LayerNorm(pointer(g2), seq, 1024, pointer(q), pointer(k), 1e-5,
      pointer(g2));
    CheckUtf8(SameBits(pointer(y1), pointer(g2), length(y1)),
      'layernorm in place, % rows', [seq]);
  end;
end;

procedure TTestInferFast.FastModeWithinTolerance;
const
  SHAPES: array[0..4, 0..2] of integer = (
    (1, 32, 4), (3, 64, 5), (37, 256, 33), (64, 1024, 100), (5, 4096, 64));
var
  i, j, rows, nin, nout, seq, heads, hd: PtrInt;
  x, bias, y1, y2, q, k, v, s, g1, g2: TSingleDynArray;
  keep: TIntegerDynArray;
  wq: RawByteString;
  e, worst: double;
  refused: boolean;
begin
  // Q8_0 Linear: float lanes against the double reference - a few float ulps
  // of the largest output
  worst := 0;
  for i := 0 to high(SHAPES) do
  begin
    rows := SHAPES[i, 0];
    nin := SHAPES[i, 1];
    nout := SHAPES[i, 2];
    Fill(x, rows * nin, 111 + i, 3);
    Fill(bias, nout, 131 + i, 1);
    wq := RandomQ8(nout, nin, 141 + i);
    SetLength(y1, rows * nout);
    SetLength(y2, rows * nout);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      pointer(bias), pointer(y1));
    fFast.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      pointer(bias), pointer(y2));
    e := RelError(pointer(y2), pointer(y1), rows * nout);
    if e > worst then
      worst := e;
    CheckUtf8(e <= 2e-6, 'Q8_0 % x % x %: relative error %', [rows, nin, nout, e]);
  end;
  AddConsole(FormatString('fast Q8_0 Linear: worst relative error %', [worst]));
  // attention: float scores and sums
  heads := 4;
  hd := 64;
  for seq := 1 to 40 do
    if seq in [1, 3, 17, 40] then
    begin
      Fill(q, seq * heads * hd, seq + 300, 1);
      Fill(k, seq * heads * hd, seq + 400, 1);
      Fill(v, seq * heads * hd, seq + 500, 2);
      SetLength(keep, seq);
      for j := 0 to seq - 1 do
        keep[j] := ord((j = 0) or (j mod 5 <> 3));
      SetLength(s, seq * seq);
      SetLength(y1, seq * heads * hd);
      SetLength(y2, seq * heads * hd);
      fRef.Attention(pointer(q), pointer(k), pointer(v), seq, heads, hd,
        pointer(keep), pointer(s), pointer(y1));
      fFast.Attention(pointer(q), pointer(k), pointer(v), seq, heads, hd,
        pointer(keep), pointer(s), pointer(y2));
      e := RelError(pointer(y2), pointer(y1), seq * heads * hd);
      CheckUtf8(e <= 1e-5, 'attention % tokens: relative error %', [seq, e]);
    end;
  // GELU table: absolute error far below the single precision of the result
  Fill(g1, 20000, 77, 12);
  g2 := copy(g1);
  fRef.Gelu(pointer(g1), length(g1));
  fFast.Gelu(pointer(g2), length(g2));
  // table error (< 1e-9) plus at most one single rounding step of the result
  worst := 0;
  for j := 0 to high(g1) do
  begin
    // a NaN or an infinity is a failure, not a comparison that happens to be false
    if (PCardinal(@g1[j])^ and $7f800000 = $7f800000) or
       (PCardinal(@g2[j])^ and $7f800000 = $7f800000) then
    begin
      worst := 1e30;
      break;
    end;
    e := Abs(g2[j] - g1[j]) - 1.2e-7 * Abs(g1[j]);
    if e > worst then
      worst := e;
  end;
  CheckUtf8(worst <= 1e-9, 'gelu table: error % beyond one single ulp', [worst]);
  // an activation block whose FP16 scale is +Inf (|x| >= 127 * 65520) is
  // refused before any kernel runs - on this thread and on the pool
  for i := 0 to 1 do
  begin
    rows := 1 + i * 63;
    nin := 1024;
    nout := 100;
    Fill(x, rows * nin, 171 + i, 3);
    x[(rows - 1) * nin + 40] := 1e7;
    wq := RandomQ8(nout, nin, 181 + i);
    SetLength(y2, rows * nout);
    refused := false;
    try
      fFast.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
        nil, pointer(y2));
    except
      on E: EInferError do
        refused := true;
    end;
    CheckUtf8(refused, 'Linear refuses an activation scale of +Inf, % rows', [rows]);
  end;
end;

procedure TTestInferFast.CountHit(aItem: integer);
begin
  InterlockedIncrement(fHits[aItem]);
end;

procedure TTestInferFast.FailOne(aItem: integer);
begin
  if aItem = 13 then
    EInferError.RaiseUtf8('item %', [aItem]);
  InterlockedIncrement(fHits[aItem]);
end;

procedure TTestInferFast.MarkWorker(aItem, aWorker: integer);
begin
  if aItem = fFail then
    EInferError.RaiseUtf8('item %', [aItem]);
  fWorkerOf[aItem] := aWorker;
  fThreadOf[aItem] := PtrUInt(GetCurrentThreadId);
  InterlockedIncrement(fHits[aItem]);
end;

/// causal grouped-query attention as defined, straight from the formula: an
// independent check of the reference branch the exact path shares
procedure NaiveCausal(const q, keys, values: TSingleDynArray; aRows, aHeads,
  aDim, aPast, aKvHeads: PtrInt; var y: TSingleDynArray; aWindow: PtrInt = 0;
  aSinks: PSingleArray = nil);
var
  r, h, g, j, t, total, first: PtrInt;
  w: TDoubleDynArray;
  m, sum, acc, a: double;
begin
  SetLength(y, aRows * aHeads * aDim);
  for r := 0 to aRows - 1 do
    for h := 0 to aHeads - 1 do
    begin
      g := h * aKvHeads div aHeads; // heads per KV head, in order
      total := aPast + r + 1;
      first := 0; // a sliding window: key j when total - aWindow <= j
      if (aWindow > 0) and (total > aWindow) then
        first := total - aWindow;
      SetLength(w, total);
      m := -1e300;
      for j := first to total - 1 do
      begin
        acc := 0;
        for t := 0 to aDim - 1 do
        begin
          a := q[(r * aHeads + h) * aDim + t]; // widened before the product
          acc := acc + a * keys[(j * aKvHeads + g) * aDim + t];
        end;
        w[j] := acc / Sqrt(aDim);
        if w[j] > m then
          m := w[j];
      end;
      // a sink: one more logit in the softmax, without a value
      if (aSinks <> nil) and (aSinks^[h] > m) then
        m := aSinks^[h];
      sum := 0;
      for j := first to total - 1 do
      begin
        w[j] := Exp(w[j] - m);
        sum := sum + w[j];
      end;
      if aSinks <> nil then
        sum := sum + Exp(aSinks^[h] - m);
      for t := 0 to aDim - 1 do
      begin
        acc := 0;
        for j := first to total - 1 do
          acc := acc + w[j] * values[(j * aKvHeads + g) * aDim + t];
        y[(r * aHeads + h) * aDim + t] := acc / sum;
      end;
    end;
end;

procedure TTestInferFast.DecoderOpsMatchReference;
const
  // past tokens, new rows, heads, KV heads, head size
  // - 4094 + 2 keys: the scores just fit the stack of CausalRows, 4099 + 2 do
  // not (its heap fallback), head size 272 has its sums on the heap
  CASES: array[0..10, 0..4] of integer = (
    (0, 1, 16, 8, 128), (0, 20, 16, 8, 128), (50, 1, 16, 8, 128),
    (50, 20, 16, 8, 128), (3, 7, 4, 1, 32), (0, 9, 2, 2, 8), (130, 3, 8, 8, 64),
    (5, 16, 4, 2, 32), // 16 rows: exactly two blocks of 8
    (4094, 2, 2, 1, 32), (4099, 2, 2, 1, 32), (2, 3, 2, 1, 272));
  // rows, in, out: in 1000 has no whole lanes of 16 - the exact loop
  SHAPES: array[0..4, 0..2] of integer = (
    (1, 1024, 300), (7, 1024, 37), (3, 1000, 20), (64, 2048, 48), (2, 64, 16));
var
  i, total, rows, qn, kvn, nin, nout, win, unseen: PtrInt;
  q, keys, values, y1, y2, y3, x, w, bias, sinks: TSingleDynArray;
  cmp: TInferCompare;
  refused: boolean;
begin
  for i := 0 to high(CASES) do
  begin
    rows := CASES[i, 1];
    total := CASES[i, 0] + rows;
    qn := rows * CASES[i, 2] * CASES[i, 4];
    kvn := total * CASES[i, 3] * CASES[i, 4];
    Fill(q, qn, 300 + i, 2);
    Fill(keys, kvn, 400 + i, 2);
    Fill(values, kvn, 500 + i, 2);
    SetLength(y1, qn);
    SetLength(y2, qn);
    SetLength(y3, qn);
    fRef.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y1));
    fExact.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y2));
    fFast.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y3));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), qn),
      'causal attention exact, case %', [i]);
    // the reference itself against the definition, written out here
    NaiveCausal(q, keys, values, rows, CASES[i, 2], CASES[i, 4], CASES[i, 0],
      CASES[i, 3], y2);
    cmp := InferCompare(pointer(y1), pointer(y2), qn, 1e-6, 1e-5);
    CheckUtf8(InferCompareOk(cmp), 'causal attention by definition, case %: %',
      [i, InferCompareText(cmp)]);
    cmp := InferCompare(pointer(y3), pointer(y1), qn, 1e-5, 1e-4);
    CheckUtf8(InferCompareOk(cmp), 'causal attention float, case %: %',
      [i, InferCompareText(cmp)]);
    // with one key the output is that value on every path
    if (total > 1) and (CASES[i, 4] and 15 = 0) and Assigned(InferDotsF32) then
      Check(not SameBits(pointer(y1), pointer(y3), qn),
        'the float path computes otherwise (a test that tests something)');
  end;
  // the first row of a fresh sequence sees one key: its output is that value
  Fill(q, 128, 1, 2);
  Fill(keys, 128, 2, 2);
  Fill(values, 128, 3, 2);
  SetLength(y1, 128);
  fFast.CausalAttention(pointer(q), 1, 1, 128, pointer(keys), pointer(values), 0, 1,
    pointer(y1));
  cmp := InferCompare(pointer(y1), pointer(values), 128, 1e-6, 1e-6);
  CheckUtf8(InferCompareOk(cmp), 'one key: the value itself: %', [InferCompareText(cmp)]);
  refused := false;
  try
    fFast.CausalAttention(pointer(q), 1, 3, 32, pointer(keys), pointer(values), 0, 2,
      pointer(y1));
  except
    on EInferError do
      refused := true;
  end;
  Check(refused, 'heads no multiple of the KV heads');
  // a sliding window: row r sees the last w keys up to itself; the keys no
  // row sees may be anything (a sliding cache has dropped them) - NaN here
  for i := 0 to high(CASES) do
  begin
    rows := CASES[i, 1];
    total := CASES[i, 0] + rows;
    win := 1 + (i * 7) mod 13;
    qn := rows * CASES[i, 2] * CASES[i, 4];
    kvn := total * CASES[i, 3] * CASES[i, 4];
    Fill(q, qn, 600 + i, 2);
    Fill(keys, kvn, 700 + i, 2);
    Fill(values, kvn, 800 + i, 2);
    unseen := (CASES[i, 0] + 1 - win) * CASES[i, 3] * CASES[i, 4];
    for nin := 0 to unseen - 1 do
    begin
      PCardinal(@keys[nin])^ := $7fc00000;
      PCardinal(@values[nin])^ := $7fc00000;
    end;
    SetLength(y1, qn);
    SetLength(y2, qn);
    SetLength(y3, qn);
    fRef.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y1), win);
    fExact.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y2), win);
    fFast.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y3), win);
    CheckUtf8(SameBits(pointer(y1), pointer(y2), qn),
      'windowed attention exact, case %, window %', [i, win]);
    NaiveCausal(q, keys, values, rows, CASES[i, 2], CASES[i, 4], CASES[i, 0],
      CASES[i, 3], y2, win);
    cmp := InferCompare(pointer(y1), pointer(y2), qn, 1e-6, 1e-5);
    CheckUtf8(InferCompareOk(cmp), 'windowed attention by definition, case %, ' +
      'window %: %', [i, win, InferCompareText(cmp)]);
    cmp := InferCompare(pointer(y3), pointer(y1), qn, 1e-5, 1e-4);
    CheckUtf8(InferCompareOk(cmp), 'windowed attention float, case %, window %: %',
      [i, win, InferCompareText(cmp)]);
  end;
  // attention sinks (gpt-oss), with and without a window: a logit per head in
  // every softmax - the same arithmetic on every path
  for i := 0 to high(CASES) do
  begin
    rows := CASES[i, 1];
    total := CASES[i, 0] + rows;
    win := (i and 1) * (2 + i);
    qn := rows * CASES[i, 2] * CASES[i, 4];
    kvn := total * CASES[i, 3] * CASES[i, 4];
    Fill(q, qn, 1600 + i, 2);
    Fill(keys, kvn, 1700 + i, 2);
    Fill(values, kvn, 1800 + i, 2);
    Fill(sinks, CASES[i, 2], 1900 + i, 3);
    SetLength(y1, qn);
    SetLength(y2, qn);
    SetLength(y3, qn);
    fRef.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y1), win, pointer(sinks));
    fExact.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y2), win, pointer(sinks));
    fFast.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y3), win, pointer(sinks));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), qn),
      'attention with sinks exact, case %', [i]);
    NaiveCausal(q, keys, values, rows, CASES[i, 2], CASES[i, 4], CASES[i, 0],
      CASES[i, 3], y2, win, pointer(sinks));
    cmp := InferCompare(pointer(y1), pointer(y2), qn, 1e-6, 1e-5);
    CheckUtf8(InferCompareOk(cmp), 'attention with sinks by definition, case %: %',
      [i, InferCompareText(cmp)]);
    cmp := InferCompare(pointer(y3), pointer(y1), qn, 1e-5, 1e-4);
    CheckUtf8(InferCompareOk(cmp), 'attention with sinks float, case %: %',
      [i, InferCompareText(cmp)]);
    // the sinks count: without them, other outputs
    fRef.CausalAttention(pointer(q), rows, CASES[i, 2], CASES[i, 4], pointer(keys),
      pointer(values), CASES[i, 0], CASES[i, 3], pointer(y2), win);
    Check(not SameBits(pointer(y1), pointer(y2), qn), 'the sinks change the output');
  end;
  // a sink far below every score takes nothing: the bits of no sink
  rows := CASES[3, 1];
  total := CASES[3, 0] + rows;
  qn := rows * CASES[3, 2] * CASES[3, 4];
  kvn := total * CASES[3, 3] * CASES[3, 4];
  Fill(q, qn, 1990, 2);
  Fill(keys, kvn, 1991, 2);
  Fill(values, kvn, 1992, 2);
  SetLength(sinks, CASES[3, 2]);
  for i := 0 to high(sinks) do
    sinks[i] := -1e30;
  SetLength(y1, qn);
  SetLength(y2, qn);
  fFast.CausalAttention(pointer(q), rows, CASES[3, 2], CASES[3, 4], pointer(keys),
    pointer(values), CASES[3, 0], CASES[3, 3], pointer(y1));
  fFast.CausalAttention(pointer(q), rows, CASES[3, 2], CASES[3, 4], pointer(keys),
    pointer(values), CASES[3, 0], CASES[3, 3], pointer(y2), 0, pointer(sinks));
  Check(SameBits(pointer(y1), pointer(y2), qn), 'a sink of -1e30 takes nothing');
  // a sink far above every score takes everything: the output vanishes
  for i := 0 to high(sinks) do
    sinks[i] := 80;
  fRef.CausalAttention(pointer(q), rows, CASES[3, 2], CASES[3, 4], pointer(keys),
    pointer(values), CASES[3, 0], CASES[3, 3], pointer(y2), 0, pointer(sinks));
  SetLength(y3, qn);
  FillCharFast(y3[0], qn * SizeOf(single), 0);
  cmp := InferCompare(pointer(y2), pointer(y3), qn, 1e-20, 0);
  CheckUtf8(InferCompareOk(cmp), 'a sink of 80 takes all: %', [InferCompareText(cmp)]);
  // a window of every key is no window: the same bits on both paths
  rows := CASES[3, 1];
  total := CASES[3, 0] + rows;
  qn := rows * CASES[3, 2] * CASES[3, 4];
  kvn := total * CASES[3, 3] * CASES[3, 4];
  Fill(q, qn, 900, 2);
  Fill(keys, kvn, 901, 2);
  Fill(values, kvn, 902, 2);
  SetLength(y1, qn);
  SetLength(y2, qn);
  fRef.CausalAttention(pointer(q), rows, CASES[3, 2], CASES[3, 4], pointer(keys),
    pointer(values), CASES[3, 0], CASES[3, 3], pointer(y1));
  fRef.CausalAttention(pointer(q), rows, CASES[3, 2], CASES[3, 4], pointer(keys),
    pointer(values), CASES[3, 0], CASES[3, 3], pointer(y2), total);
  Check(SameBits(pointer(y1), pointer(y2), qn), 'a window of every key, reference');
  fFast.CausalAttention(pointer(q), rows, CASES[3, 2], CASES[3, 4], pointer(keys),
    pointer(values), CASES[3, 0], CASES[3, 3], pointer(y1));
  fFast.CausalAttention(pointer(q), rows, CASES[3, 2], CASES[3, 4], pointer(keys),
    pointer(values), CASES[3, 0], CASES[3, 3], pointer(y2), total);
  Check(SameBits(pointer(y1), pointer(y2), qn), 'a window of every key, float');
  refused := false;
  try
    fFast.CausalAttention(pointer(q), 1, 2, 32, pointer(keys), pointer(values), 0, 1,
      pointer(y1), -1);
  except
    on EInferError do
      refused := true;
  end;
  Check(refused, 'a negative window');
  // F32 Linear: float lanes (whole lanes of 16) or the exact loop
  for i := 0 to high(SHAPES) do
  begin
    rows := SHAPES[i, 0];
    nin := SHAPES[i, 1];
    nout := SHAPES[i, 2];
    Fill(x, rows * nin, 600 + i, 3);
    Fill(w, nout * nin, 700 + i, 0.1);
    Fill(bias, nout, 800 + i, 1);
    if i = high(SHAPES) then
      bias := nil; // the float path without a bias
    SetLength(y1, rows * nout);
    SetLength(y3, rows * nout);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(w), idtF32), nout,
      pointer(bias), pointer(y1));
    fFast.Linear(pointer(x), rows, nin, InferWeight(pointer(w), idtF32), nout,
      pointer(bias), pointer(y3));
    if nin and 15 <> 0 then
      CheckUtf8(SameBits(pointer(y1), pointer(y3), rows * nout),
        'F32 % x % x %: no whole lanes, the exact loop', [rows, nin, nout])
    else
    begin
      cmp := InferCompare(pointer(y3), pointer(y1), rows * nout, 1e-4, 1e-4);
      CheckUtf8(InferCompareOk(cmp), 'F32 % x % x % float: %',
        [rows, nin, nout, InferCompareText(cmp)]);
    end;
  end;
  // no input at all: the bias, whatever the output held before
  Fill(bias, 3, 900, 1);
  Fill(y3, 3, 901, 5);
  fFast.Linear(nil, 1, 0, InferWeight(nil, idtF32), 3, pointer(bias), pointer(y3));
  Check(SameBits(pointer(y3), pointer(bias), 3), 'F32 Linear without inputs = bias');
end;

procedure TTestInferFast.DecoderNormAndSwiGlu;
const
  // rows, cols: below, at and above the parallel thresholds; 128 = a head
  NORMS: array[0..6, 0..1] of integer = (
    (1, 1024), (7, 1024), (8, 1024), (9, 1024), (200, 1024), (16, 128),
    (3200, 128));
  // 3000: in parallel, with a partial last slice
  GLUS: array[0..6] of integer = (1, 1000, 2047, 2048, 3000, 3072, 200 * 3072);
  ROPES: array[0..4] of integer = (1, 7, 8, 9, 200);
var
  i, r, c, k, j, rows, cols, n: PtrInt;
  x, w, y1, y2, y3, u, g1, g2, g3, want: TSingleDynArray;
  sum, a: double;
  silu, gc, uc, alpha: single;
  cmp: TInferCompare;
begin
  for i := 0 to high(NORMS) do
  begin
    rows := NORMS[i, 0];
    cols := NORMS[i, 1];
    n := rows * cols;
    Fill(x, n, 900 + i, 4);
    Fill(w, cols, 950 + i, 1);
    SetLength(y1, n);
    SetLength(y2, n);
    fRef.RmsNorm(pointer(x), rows, cols, pointer(w), 1e-6, pointer(y1));
    fFast.RmsNorm(pointer(x), rows, cols, pointer(w), 1e-6, pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), n), 'rmsnorm, % x %', [rows, cols]);
    y3 := copy(x);
    fExact.RmsNorm(pointer(y3), rows, cols, pointer(w), 1e-6, pointer(y3));
    CheckUtf8(SameBits(pointer(y1), pointer(y3), n), 'rmsnorm in place, % x %',
      [rows, cols]);
    // the reference against the definition
    SetLength(want, n);
    for r := 0 to rows - 1 do
    begin
      sum := 0;
      for c := 0 to cols - 1 do
        sum := sum + Sqr(x[r * cols + c]);
      for c := 0 to cols - 1 do
        want[r * cols + c] := x[r * cols + c] / Sqrt(sum / cols + 1e-6) * w[c];
    end;
    cmp := InferCompare(pointer(y1), pointer(want), n, 1e-6, 1e-6);
    CheckUtf8(InferCompareOk(cmp), 'rmsnorm by definition, % x %: %',
      [rows, cols, InferCompareText(cmp)]);
  end;
  for i := 0 to high(GLUS) do
  begin
    n := GLUS[i];
    Fill(g1, n, 1000 + i, 8);
    Fill(u, n, 1100 + i, 2);
    g2 := copy(g1);
    g3 := copy(g1);
    SetLength(want, n);
    SetLength(y3, n);
    for r := 0 to n - 1 do
    begin
      a := g1[r];
      want[r] := a / (1 + Exp(-a)) * u[r];
      silu := a / (1 + Exp(-a)); // float32 steps: silu rounded, then the product
      y3[r] := silu * u[r];
    end;
    fRef.SwiGlu(pointer(g1), pointer(u), n);
    fFast.SwiGlu(pointer(g2), pointer(u), n);
    fExact.SwiGlu(pointer(g3), pointer(u), n);
    CheckUtf8(SameBits(pointer(g1), pointer(g2), n) and
      SameBits(pointer(g1), pointer(g3), n), 'swiglu of %', [n]);
    CheckUtf8(SameBits(pointer(g1), pointer(y3), n), 'swiglu in float32 steps, %',
      [n]);
    cmp := InferCompare(pointer(g1), pointer(want), n, 1e-6, 1e-6);
    CheckUtf8(InferCompareOk(cmp), 'swiglu by definition, %: %',
      [n, InferCompareText(cmp)]);
  end;
  // the clamped SwiGLU of gpt-oss (limit 7, alpha 1.702): gate above and up
  // on both sides clamped - values of 8 reach the limit often
  alpha := 1.702;
  for i := 0 to high(GLUS) do
  begin
    n := GLUS[i];
    Fill(g1, n, 1300 + i, 8);
    Fill(u, n, 1400 + i, 8);
    g2 := copy(g1);
    g3 := copy(g1);
    SetLength(want, n);
    SetLength(y3, n);
    k := 0;
    for r := 0 to n - 1 do
    begin
      gc := g1[r];
      if gc > 7 then
        gc := 7;
      uc := u[r];
      if uc > 7 then
        uc := 7;
      if uc < -7 then
        uc := -7;
      if (gc <> g1[r]) or (uc <> u[r]) then
        inc(k);
      want[r] := (uc + 1) * gc / (1 + Exp(-1.702 * gc));
      // float32 steps: alpha * g, the sigmoid, g * it, u + 1, the product
      silu := gc * alpha;
      silu := 1 / (1 + Exp(-silu));
      silu := gc * silu;
      uc := uc + 1;
      y3[r] := uc * silu;
    end;
    fRef.SwiGlu(pointer(g1), pointer(u), n, 1.702, 7);
    fFast.SwiGlu(pointer(g2), pointer(u), n, 1.702, 7);
    fExact.SwiGlu(pointer(g3), pointer(u), n, 1.702, 7);
    CheckUtf8(SameBits(pointer(g1), pointer(g2), n) and
      SameBits(pointer(g1), pointer(g3), n), 'clamped swiglu of %', [n]);
    CheckUtf8(SameBits(pointer(g1), pointer(y3), n),
      'clamped swiglu in float32 steps, %', [n]);
    cmp := InferCompare(pointer(g1), pointer(want), n, 1e-5, 1e-6);
    CheckUtf8(InferCompareOk(cmp), 'clamped swiglu by definition, %: %',
      [n, InferCompareText(cmp)]);
    CheckUtf8((n < 100) or (k > n div 10), 'clamped swiglu of %: % clamped', [n, k]);
  end;
  // Rope: rows below, at and above the parallel threshold; 16 heads of 128
  for i := 0 to high(ROPES) do
  begin
    rows := ROPES[i];
    n := rows * 16 * 128;
    Fill(x, n, 1200 + i, 3);
    SetLength(w, rows * 64);
    SetLength(u, rows * 64);
    for r := 0 to rows - 1 do
      for c := 0 to 63 do
      begin
        a := (7 + r) / Exp(c / 64 * Ln(1000000)); // position / theta^(2c/d)
        w[r * 64 + c] := Cos(a);
        u[r * 64 + c] := Sin(a);
      end;
    y1 := copy(x);
    y2 := copy(x);
    y3 := copy(x);
    fRef.Rope(pointer(y1), rows, 16, 128, pointer(w), pointer(u));
    fFast.Rope(pointer(y2), rows, 16, 128, pointer(w), pointer(u));
    fExact.Rope(pointer(y3), rows, 16, 128, pointer(w), pointer(u));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), n) and
      SameBits(pointer(y1), pointer(y3), n), 'rope, % rows', [rows]);
    // rotate_half by definition: [x1, x2] -> [x1 c - x2 s, x2 c + x1 s]
    SetLength(want, n);
    for r := 0 to rows - 1 do
      for c := 0 to 15 do
        for k := 0 to 63 do
        begin
          j := (r * 16 + c) * 128 + k;
          want[j] := x[j] * w[r * 64 + k] - x[j + 64] * u[r * 64 + k];
          want[j + 64] := x[j + 64] * w[r * 64 + k] + x[j] * u[r * 64 + k];
        end;
    cmp := InferCompare(pointer(y1), pointer(want), n, 1e-6, 1e-6);
    CheckUtf8(InferCompareOk(cmp), 'rope by definition, % rows: %',
      [rows, InferCompareText(cmp)]);
  end;
end;

procedure TTestInferFast.ThreadPoolRunsEveryItemOnce;
var
  pool: TInferThreadPool;
  run, n, i, bad: integer;
  raised: boolean;
begin
  pool := TInferThreadPool.Create(5);
  try
    bad := 0;
    // a burst of jobs of every size: tickets of one job never leak into the next
    for run := 1 to 3000 do
    begin
      n := 1 + (run * 7919) mod 97;
      SetLength(fHits, n);
      FillCharFast(pointer(fHits)^, n * SizeOf(integer), 0);
      pool.Run(CountHit, n);
      for i := 0 to n - 1 do
        if fHits[i] <> 1 then
          inc(bad);
    end;
    CheckEqual(bad, 0, 'every item exactly once over 3000 jobs');
    // sizes 8, 2, 8, 1 in turn: a late worker holding an older snapshot must
    // never mix one job's Count with another's Base (seqlock, not Count twice)
    bad := 0;
    for run := 1 to 20000 do
    begin
      case run and 3 of
        0, 2:
          n := 8;
        1:
          n := 2;
      else
        n := 1;
      end;
      SetLength(fHits, n);
      FillCharFast(pointer(fHits)^, n * SizeOf(integer), 0);
      pool.Run(CountHit, n);
      for i := 0 to n - 1 do
        if fHits[i] <> 1 then
          inc(bad);
    end;
    CheckEqual(bad, 0, 'every item exactly once over 20000 jobs of 8, 2, 8, 1');
    // the ticket counter restarts before it wraps: jobs across the restart
    pool.Ticket := $40000000 - 100;
    bad := 0;
    for run := 1 to 200 do
    begin
      n := 1 + run mod 9;
      SetLength(fHits, n);
      FillCharFast(pointer(fHits)^, n * SizeOf(integer), 0);
      pool.Run(CountHit, n);
      for i := 0 to n - 1 do
        if fHits[i] <> 1 then
          inc(bad);
    end;
    CheckEqual(bad, 0, 'every item exactly once across the ticket restart');
    CheckUtf8(cardinal(pool.Ticket) < $40000000, 'ticket counter restarted');
    // an exception in one item reaches the caller; the others still ran
    SetLength(fHits, 50);
    FillCharFast(pointer(fHits)^, 50 * SizeOf(integer), 0);
    raised := false;
    try
      pool.Run(FailOne, 50);
    except
      on E: EInferError do
        raised := PosEx('item 13', StringToUtf8(E.Message)) > 0;
    end;
    CheckUtf8(raised, 'the item exception reaches Run');
    bad := 0;
    for i := 0 to 49 do
      if (i <> 13) and (fHits[i] <> 1) then
        inc(bad);
    CheckEqual(bad, 0, 'the other items ran once');
    // the pool is usable after an exception
    SetLength(fHits, 20);
    FillCharFast(pointer(fHits)^, 20 * SizeOf(integer), 0);
    pool.Run(CountHit, 20);
    bad := 0;
    for i := 0 to 19 do
      if fHits[i] <> 1 then
        inc(bad);
    CheckEqual(bad, 0, 'next job after an exception');
  finally
    pool.Free;
  end;
end;

procedure TTestInferFast.WaitForPeer(aItem, aWorker: integer);
var
  stop: Int64;
begin
  // every item waits (up to 5 s) until a second item has started - on
  // another thread, since this one waits: the pool's threads must take part,
  // whatever the scheduler
  MarkWorker(aItem, aWorker);
  InterlockedIncrement(fPeers);
  stop := GetTickCount64 + 5000;
  while (fPeers < 2) and (GetTickCount64 < stop) do
    SwitchToThread;
end;

procedure TTestInferFast.ThreadPoolWorkerIndex;
var
  pool: TInferThreadPool;
  owner: array[0..5] of PtrUInt; // the thread of each index, over all runs
  run, n, i, j, bad, outside: integer;
  raised, other: boolean;
begin
  fFail := -1;
  pool := TInferThreadPool.Create(5);
  try
    FillCharFast(owner, SizeOf(owner), 0);
    owner[0] := PtrUInt(GetCurrentThreadId);
    bad := 0;
    outside := 0;
    for run := 1 to 3000 do
    begin
      n := 1 + (run * 7919) mod 97;
      SetLength(fHits, n);
      SetLength(fWorkerOf, n);
      SetLength(fThreadOf, n);
      FillCharFast(pointer(fHits)^, n * SizeOf(integer), 0);
      pool.RunWorkers(MarkWorker, n);
      for i := 0 to n - 1 do
        if fHits[i] <> 1 then
          inc(bad)
        else if (fWorkerOf[i] < 0) or (fWorkerOf[i] > pool.Workers) then
          inc(outside)
        else if owner[fWorkerOf[i]] = 0 then
          owner[fWorkerOf[i]] := fThreadOf[i]
        else if owner[fWorkerOf[i]] <> fThreadOf[i] then
          inc(outside);
    end;
    CheckEqual(bad, 0, 'every item exactly once');
    CheckEqual(outside, 0, 'index in 0..Workers, the same thread for an index');
    for i := 0 to high(owner) do
      if owner[i] <> 0 then
      begin
        for j := 0 to i - 1 do
          if owner[j] = owner[i] then
            inc(bad);
      end;
    CheckEqual(bad, 0, 'one thread, one index');
    // the workers take part: an item waits until another thread took one
    SetLength(fHits, 8);
    SetLength(fWorkerOf, 8);
    SetLength(fThreadOf, 8);
    FillCharFast(pointer(fHits)^, 8 * SizeOf(integer), 0);
    fPeers := 0;
    pool.RunWorkers(WaitForPeer, 8);
    other := false;
    for i := 1 to 7 do
      if fThreadOf[i] <> fThreadOf[0] then
        other := true;
    Check(other, 'a pool thread ran an item');
    // an exception in one item reaches the caller; the others still ran
    SetLength(fHits, 50);
    SetLength(fWorkerOf, 50);
    SetLength(fThreadOf, 50);
    FillCharFast(pointer(fHits)^, 50 * SizeOf(integer), 0);
    fFail := 13;
    raised := false;
    try
      pool.RunWorkers(MarkWorker, 50);
    except
      on E: EInferError do
        raised := PosEx('item 13', StringToUtf8(E.Message)) > 0;
    end;
    fFail := -1;
    CheckUtf8(raised, 'the item exception reaches RunWorkers');
    bad := 0;
    for i := 0 to 49 do
      if (i <> 13) and (fHits[i] <> 1) then
        inc(bad);
    CheckEqual(bad, 0, 'the other items ran once');
    // Run after RunWorkers: the plain job again, not the one with an index
    SetLength(fHits, 20);
    FillCharFast(pointer(fHits)^, 20 * SizeOf(integer), 0);
    FillCharFast(pointer(fWorkerOf)^, 20 * SizeOf(integer), $ff);
    pool.Run(CountHit, 20);
    bad := 0;
    for i := 0 to 19 do
      if (fHits[i] <> 1) or (fWorkerOf[i] <> -1) then
        inc(bad);
    CheckEqual(bad, 0, 'Run after RunWorkers');
  finally
    pool.Free;
  end;
  // no worker thread: every item on the caller, index 0
  pool := TInferThreadPool.Create(0);
  try
    SetLength(fHits, 10);
    SetLength(fWorkerOf, 10);
    SetLength(fThreadOf, 10);
    FillCharFast(pointer(fHits)^, 10 * SizeOf(integer), 0);
    FillCharFast(pointer(fWorkerOf)^, 10 * SizeOf(integer), $ff);
    pool.RunWorkers(MarkWorker, 10);
    bad := 0;
    for i := 0 to 9 do
      if (fHits[i] <> 1) or (fWorkerOf[i] <> 0) then
        inc(bad);
    CheckEqual(bad, 0, 'without workers: index 0, every item once');
  finally
    pool.Free;
  end;
end;

type
  /// computes one Q8_0 Linear in a loop through a shared backend
  TLinearCaller = class(TThread)
  protected
    fBackend: TInferBackend;
    fX, fBias: TSingleDynArray;
    fW: RawByteString;
    fRows, fIn, fOut: integer;
    procedure Execute; override;
  public
    Result: TSingleDynArray;
    Mismatch: boolean;
    Done: integer;
    Expected: TSingleDynArray;
  end;

procedure TLinearCaller.Execute;
var
  i: integer;
begin
  SetLength(Result, fRows * fOut);
  for i := 1 to 30 do
  begin
    fBackend.Linear(pointer(fX), fRows, fIn, InferWeight(pointer(fW), idtQ8_0),
      fOut, pointer(fBias), pointer(Result));
    if not SameBits(pointer(Result), pointer(Expected), fRows * fOut) then
      Mismatch := true;
    inc(Done);
  end;
end;

type
  /// computes one Attention in a loop through a shared backend
  TAttentionCaller = class(TThread)
  protected
    fBackend: TInferBackend;
    fQ, fK, fV, fS: TSingleDynArray;
    fKeep: TIntegerDynArray;
    fSeq, fHeads, fDim: integer;
    procedure Execute; override;
  public
    Result: TSingleDynArray;
    Mismatch: boolean;
    Done: integer;
    Expected: TSingleDynArray;
  end;

procedure TAttentionCaller.Execute;
var
  i: integer;
begin
  SetLength(Result, fSeq * fHeads * fDim);
  SetLength(fS, fSeq * fSeq);
  for i := 1 to 40 do
  begin
    fBackend.Attention(pointer(fQ), pointer(fK), pointer(fV), fSeq, fHeads, fDim,
      pointer(fKeep), pointer(fS), pointer(Result));
    if not SameBits(pointer(Result), pointer(Expected), length(Result)) then
      Mismatch := true;
    inc(Done);
  end;
end;

procedure TTestInferFast.ConcurrentCallersAreCorrect;
var
  c: array[0..1] of TLinearCaller;
  a: array[0..1] of TAttentionCaller;
  i, j: PtrInt;
  sc: TSingleDynArray;
  alone: TInferCpuFastBackend;
begin
  // the expected bits: the fast arithmetic on one thread - a caller finding the
  // pool busy must compute exactly that, not fall back to another arithmetic
  alone := TInferCpuFastBackend.Create(1);
  try
  for i := 0 to 1 do
  begin
    c[i] := TLinearCaller.Create(true);
    c[i].fBackend := fFast;
    c[i].fRows := 9 + i;
    c[i].fIn := 256;
    c[i].fOut := 96 + i * 32;
    Fill(c[i].fX, c[i].fRows * c[i].fIn, 500 + i, 2);
    Fill(c[i].fBias, c[i].fOut, 600 + i, 1);
    c[i].fW := RandomQ8(c[i].fOut, c[i].fIn, 700 + i);
    SetLength(c[i].Expected, c[i].fRows * c[i].fOut);
    alone.Linear(pointer(c[i].fX), c[i].fRows, c[i].fIn,
      InferWeight(pointer(c[i].fW), idtQ8_0), c[i].fOut, pointer(c[i].fBias),
      pointer(c[i].Expected));
  end;
  finally
    alone.Free;
  end;
  try
    for i := 0 to 1 do
      {$ifdef HASTTHREADSTART} c[i].Start; {$else} c[i].Resume; {$endif}
    for i := 0 to 1 do
      c[i].WaitFor;
    for i := 0 to 1 do
      CheckUtf8(not c[i].Mismatch and (c[i].Done = 30) and (c[i].FatalException = nil),
        'caller % always right (pool or own thread), every round done', [i]);
  finally
    for i := 0 to 1 do
      c[i].Free;
  end;
  // exact mode under concurrency: whoever gets the pool, both callers compute
  // the reference bits (a serial caller keeps the backend's arithmetic)
  for i := 0 to 1 do
  begin
    a[i] := TAttentionCaller.Create(true);
    a[i].fBackend := fExact;
    a[i].fSeq := 20 + i * 5;
    a[i].fHeads := 4;
    a[i].fDim := 64;
    Fill(a[i].fQ, a[i].fSeq * 256, 800 + i, 1);
    Fill(a[i].fK, a[i].fSeq * 256, 810 + i, 1);
    Fill(a[i].fV, a[i].fSeq * 256, 820 + i, 2);
    SetLength(a[i].fKeep, a[i].fSeq);
    for j := 0 to a[i].fSeq - 1 do
      a[i].fKeep[j] := 1;
    SetLength(a[i].Expected, a[i].fSeq * 256);
    SetLength(sc, a[i].fSeq * a[i].fSeq);
    fRef.Attention(pointer(a[i].fQ), pointer(a[i].fK), pointer(a[i].fV), a[i].fSeq,
      4, 64, pointer(a[i].fKeep), pointer(sc), pointer(a[i].Expected));
  end;
  try
    for i := 0 to 1 do
      {$ifdef HASTTHREADSTART} a[i].Start; {$else} a[i].Resume; {$endif}
    for i := 0 to 1 do
      a[i].WaitFor;
    for i := 0 to 1 do
      CheckUtf8(not a[i].Mismatch and (a[i].Done = 40) and (a[i].FatalException = nil),
        'exact attention caller % = reference, every round done', [i]);
  finally
    for i := 0 to 1 do
      a[i].Free;
  end;
end;

procedure TTestInferFast.ForwardIsBitIdentical;
var
  dir, gguf: TFileName;
  m1, m2, m3: TXlmrModel;
  c1, c2, c3: TXlmrContext;
  ids: array[0..63] of integer;
  i: PtrInt;
  e1, e2, e3: TInferEmbedder;
  v1, v2, v3: TSingleDynArray;
  cmp: TInferCompare;
  e: double;
const
  TEXT = 'Kupferrohr 15 mm in Wand verlegen, inkl. Befestigung';
begin
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  m1 := TXlmrModel.CreateFromGguf(dir + 'model.gguf', fRef);
  m2 := TXlmrModel.CreateFromGguf(dir + 'model.gguf', fExact);
  m3 := TXlmrModel.CreateFromGguf(dir + 'model.gguf', fFast);
  c1 := TXlmrContext.Create;
  c2 := TXlmrContext.Create;
  c3 := TXlmrContext.Create;
  try
    for i := 0 to 35 do
      ids[i] := 4 + (i * 37) mod 59;
    ids[0] := 0;
    ids[35] := 2;
    m1.Forward(c1, @ids, 1, 36);
    m2.Forward(c2, @ids, 1, 36);
    m3.Forward(c3, @ids, 1, 36);
    CheckUtf8(SameBits(pointer(c1.Hidden), pointer(c2.Hidden), c1.HiddenCount),
      'tiny GGUF, 36 tokens, exact mode');
    e := RelError(pointer(c3.Hidden), pointer(c1.Hidden), c1.HiddenCount);
    CheckUtf8(e <= 1e-3, 'tiny GGUF, fast mode: relative error %', [e]);
  finally
    c3.Free;
    c2.Free;
    c1.Free;
    m3.Free;
    m2.Free;
    m1.Free;
  end;
  gguf := FindBgeM3Gguf;
  if gguf = '' then
  begin
    AddConsole('SKIP BGE-M3 Q8 against the reference: no vendor/models/bge-m3-q8_0.gguf');
    exit;
  end;
  // one embedder at a time: three mappings of the 634 MB file would not fit
  // the address space of a 32-bit process
  e1 := TInferEmbedder.CreateFromGguf(gguf, fRef);
  try
    v1 := e1.Embed([TEXT]);
  finally
    e1.Free;
  end;
  e2 := TInferEmbedder.CreateFromGguf(gguf, fExact);
  try
    v2 := e2.Embed([TEXT]);
  finally
    e2.Free;
  end;
  e3 := TInferEmbedder.CreateFromGguf(gguf, fFast);
  try
    v3 := e3.Embed([TEXT]);
    CheckUtf8(SameBits(pointer(v1), pointer(v2), length(v1)),
      'BGE-M3 Q8 vector, exact mode = reference');
    cmp := InferCompare(pointer(v3), pointer(v1), length(v1), 1, 1);
    // float lanes against double: the activation quantization amplifies the
    // rounding differences (measured 0.99981); the gate against lembed is the
    // corpus test in test.infer.gguf
    CheckUtf8(cmp.Cosine >= 0.9995, 'BGE-M3 Q8 vector, fast mode: cosine %',
      [cmp.Cosine]);
  finally
    e3.Free;
  end;
end;

procedure TTestInferFast.NativeEmbedderAdapter;
var
  dir: TFileName;
  a, b: IEmbedder;
  v: TLlmEmbedding;
  vs: TLlmEmbeddingDynArray;
  texts: TRawUtf8DynArray;
  long: RawUtf8;
  i: PtrInt;
  refused: boolean;
begin
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  a := SharedNativeEmbedder(StringToUtf8(dir + 'model.gguf'), 'tiny-test', 2);
  b := SharedNativeEmbedder('ignored once cached', 'tiny-test');
  CheckUtf8(a = b, 'one instance per model name');
  CheckEqual(a.Model, 'tiny-test', 'model name');
  v := a.Embed('the abc');
  CheckEqual(length(v), 32, 'tiny dimension');
  SetLength(texts, 3);
  texts[0] := 'abc';
  texts[1] := 'the abc';
  texts[2] := 'in the end';
  vs := a.EmbedBatch(texts);
  CheckEqual(length(vs), 3, 'one vector per text');
  CheckUtf8(CompareMem(pointer(vs[1]), pointer(v), 32 * SizeOf(single)),
    'EmbedBatch = Embed');
  long := '';
  for i := 1 to 40 do
    long := long + 'the ';
  refused := false;
  try
    a.Embed(long);
  except
    on EInferError do
      refused := true;
  end;
  CheckUtf8(refused, 'too many tokens refused, never cut');
end;

end.
