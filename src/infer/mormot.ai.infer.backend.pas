/// native inference engine - the backend contract
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - one non-generic abstract class with typed operations on F32 buffers; a
//   backend reports per operation whether it supports it, and the model falls
//   back to the CPU reference only for "not supported", never after a failure
// - every operation works on caller-owned memory: a backend allocates nothing
//   per call, so the hot loops stay allocation-free
unit mormot.ai.infer.backend;

{
  *****************************************************************************

    - TInferOp: the operations a model may ask for
    - TInferBackend: abstract backend with the typed operations

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

{$ifndef FPC}{$ifdef CPUX64}
  {$EXCESSPRECISION OFF} // single expressions in single, as FPC computes them
{$endif}{$endif}

uses
  sysutils,
  mormot.core.base,
  mormot.ai.infer.tensor,
  mormot.ai.infer.threads;

type
  /// the operations of the backend contract
  TInferOp = (
    ioLinear,
    ioLayerNorm,
    ioGelu,
    ioAdd,
    ioSoftmaxMasked,
    ioAttention,
    ioL2Normalize,
    ioCausalAttention,
    ioRmsNorm,
    ioSwiGlu,
    ioRope,
    ioGeluTanhGlu);

  /// a weight matrix [out, in] as stored in the file: its first byte and type
  // - F32 holds out * in singles, Q8_0 out rows of in div 32 blocks
  TInferWeight = record
    Data: pointer;
    DType: TInferDType;
  end;

  /// abstract backend: typed operations on caller-owned F32 buffers
  // - layout is always row-major; "rows" are tokens, "cols" are features
  TInferBackend = class
  public
    /// true when this backend implements aOp for weights of type aWeights
    function Supports(aOp: TInferOp; aWeights: TInferDType): boolean; virtual; abstract;
    /// y[rows, out] = x[rows, in] * w[out, in]^T + bias[out]
    // - w is stored as nn.Linear stores it: one row per output feature, of the
    // type w.DType (see Supports); a Q8_0 row needs aIn to be a multiple of 32
    // - bias may be nil; y must not overlap x
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); virtual; abstract;
    /// y[rows, cols] = (x - mean) / sqrt(var + eps) * gamma + beta, per row
    // - population variance; y may be x
    procedure LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
      gamma, beta: PSingleArray; aEps: double; y: PSingleArray); virtual; abstract;
    /// x := 0.5 x (1 + erf(x / sqrt 2)) in place - the exact GELU
    procedure Gelu(x: PSingleArray; aCount: PtrInt); virtual; abstract;
    /// x := x + y elementwise
    procedure Add(x, y: PSingleArray; aCount: PtrInt); virtual; abstract;
    /// softmax over each row of x[rows, cols], in place; aKeep[col] = 0 drops a
    // column: its result is exactly 0 and it does not enter the sum
    // - aKeep may be nil (keep all); a row with no kept column becomes all 0
    procedure SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
      aKeep: PIntegerArray); virtual; abstract;
    /// multi-head self-attention of one sequence
    // - q, k, v and y are [aSeq, aHeads * aHeadDim]; head h uses the columns
    // h*aHeadDim .. h*aHeadDim + aHeadDim - 1
    // - aKeep[key] = 0 masks a key position (padding); aScratch holds at least
    // aSeq * aSeq singles
    procedure Attention(q, k, v: PSingleArray; aSeq, aHeads, aHeadDim: PtrInt;
      aKeep: PIntegerArray; aScratch, y: PSingleArray); virtual; abstract;
    /// x := x / ||x|| in place; a zero vector stays zero
    procedure L2Normalize(x: PSingleArray; aCount: PtrInt); virtual; abstract;
    /// causal multi-head attention of aRows new tokens over a KV cache, with
    // grouped-query heads (a decoder)
    // - q and y are [aRows, aHeads * aHeadDim]; keys and values hold the rows
    // 0 .. aPast + aRows - 1 of [*, aKvHeads * aHeadDim], the new tokens
    // included; new row r sees the cache rows 0 .. aPast + r
    // - head h reads KV head h div (aHeads div aKvHeads); scores scaled by
    // 1 / sqrt(aHeadDim)
    // - aWindow > 0: a sliding window - row r sees only the last aWindow cache
    // rows up to itself, aPast + r - aWindow + 1 .. aPast + r (the keys before
    // need not be valid)
    // - aSinks <> nil ([aHeads]): attention sinks (gpt-oss) - a learned logit
    // per head joins the softmax of every row and takes its share of the
    // weight without a value: the weights of the keys sum to less than 1
    // - aScale > 0: the factor of the scores in place of 1 / sqrt(aHeadDim)
    // (Gemma 4: 1, its Q and K heads are normed)
    procedure CausalAttention(q: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      keys, values: PSingleArray; aPast, aKvHeads: PtrInt; y: PSingleArray;
      aWindow: PtrInt = 0; aSinks: PSingleArray = nil;
      aScale: double = 0); virtual; abstract;
    /// y[rows, cols] = x / sqrt(mean(x^2) + eps) * w, per row (RMSNorm)
    // - the normalized value is rounded to single before the weight, as the
    // float32 reference does; y may be x
    procedure RmsNorm(x: PSingleArray; aRows, aCols: PtrInt; w: PSingleArray;
      aEps: double; y: PSingleArray); virtual; abstract;
    /// gate := silu(gate) * up elementwise (the SwiGLU of a feed-forward
    // block), silu(g) = g / (1 + exp(-g)) rounded to single before the product
    // - aLimit > 0: the clamped variant of gpt-oss - g := min(gate, aLimit),
    // u := up clamped to -aLimit .. aLimit, gate := (u + 1) * g * sigmoid(aAlpha
    // * g), every step rounded to single; aAlpha is only used then
    procedure SwiGlu(gate, up: PSingleArray; aCount: PtrInt;
      aAlpha: single = 1; aLimit: single = 0); virtual; abstract;
    /// gate := gelu_tanh(gate) * up elementwise (the GeGLU of Gemma), with
    // gelu_tanh(g) = 0.5 g (1 + tanh(sqrt(2 / pi) (g + 0.044715 g^3))) in
    // double, rounded to single before the product
    procedure GeluTanhGlu(gate, up: PSingleArray; aCount: PtrInt); virtual; abstract;
    /// rotary positions in place, the halves of every head rotated together
    // (rotate_half): x[i], x[i + d/2] := x[i] c - x[i + d/2] s, x[i + d/2] c +
    // x[i] s, in double
    // - x is [aRows, aHeads * aHeadDim]; aCos and aSin are [aRows, d / 2],
    // the factors of each row's position
    // - d is aHeadDim, or aRotDim when > 0: only the first aRotDim values of
    // every head rotate (Qwen3.5, a partial rotary factor), the others stay
    procedure Rope(x: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      aCos, aSin: PSingleArray; aRotDim: PtrInt = 0); virtual; abstract;
    /// the gated delta rule of a Gated DeltaNet layer (Qwen3.5) over aRows
    // tokens, every value head with its own state
    // - qkv: aRows rows of aStride values, each [q | k | v]: aKeyHeads heads
    // of aKeyDim queries (L2-normed and scaled), as many keys (L2-normed),
    // aValueHeads heads of aValueDim values; value head h reads key head
    // h div (aValueHeads div aKeyHeads), as repeat_interleave does
    // - beta, decay: [aRows, aValueHeads], the factors of every token and head
    // - state: [aValueHeads, aValueDim, aKeyDim] - row j of a head is column j
    // of its S - updated in place
    // - per token and head: S := decay S; kv := S^T k; S += k (beta (v -
    // kv))^T; y := S^T q, in double sums, y [aRows, aValueHeads * aValueDim]
    procedure DeltaRule(qkv: PSingleArray; aRows, aStride, aKeyHeads,
      aValueHeads, aKeyDim, aValueDim: PtrInt; beta, decay, state,
      y: PSingleArray); virtual; abstract;
    /// forget what the backend keeps of every weight whose host memory
    // overlaps aFrom .. aTo - 1: a GPU backend keeps a device copy, found
    // again by address, type and size, so memory freed and allocated anew at
    // the same address would be served from the stale copy
    // - for the owner of weights, before it frees or overwrites them (a
    // model in its destructor does it for its own); nothing to do here: the
    // CPU reads the host memory itself
    procedure ForgetWeights(aFrom, aTo: pointer); virtual;
    /// aJob(0) .. aJob(aCount - 1), independent items: on the threads of the
    // backend when it has a pool that is free, else in order on this thread
    // - for work outside the operations, like converting weights at load; an
    // item may call operations of the backend (they then run on its thread)
    // - an exception in an item is raised once every item is done, as an
    // EInferError with its message when the items ran on the pool
    procedure RunItems(const aJob: TOnInferItem; aCount: integer); virtual;
  end;

/// a TInferWeight from its parts
function InferWeight(aData: pointer; aType: TInferDType): TInferWeight;

/// aBackend.ForgetWeights for the memory of aTensors (the tensors of one
// file: from the lowest to the highest byte they cover) - nothing for none
procedure InferForgetTensors(aBackend: TInferBackend;
  const aTensors: TInferTensorDynArray);

/// aBackend.ForgetWeights for aBytes bytes at aData - nothing for nil or 0
procedure InferForgetBytes(aBackend: TInferBackend; aData: pointer; aBytes: PtrInt);


implementation

function InferWeight(aData: pointer; aType: TInferDType): TInferWeight;
begin
  result.Data := aData;
  result.DType := aType;
end;

procedure InferForgetTensors(aBackend: TInferBackend;
  const aTensors: TInferTensorDynArray);
var
  i: PtrInt;
  lo, hi, a: PtrUInt;
begin
  if aBackend = nil then
    exit;
  lo := High(PtrUInt);
  hi := 0;
  for i := 0 to high(aTensors) do
    if (aTensors[i].Data <> nil) and (aTensors[i].Bytes > 0) then
    begin
      a := PtrUInt(aTensors[i].Data);
      if a < lo then
        lo := a;
      if a + PtrUInt(aTensors[i].Bytes) > hi then
        hi := a + PtrUInt(aTensors[i].Bytes);
    end;
  if hi > lo then
    aBackend.ForgetWeights(pointer(lo), pointer(hi));
end;

procedure InferForgetBytes(aBackend: TInferBackend; aData: pointer; aBytes: PtrInt);
begin
  if (aBackend <> nil) and (aData <> nil) and (aBytes > 0) then
    aBackend.ForgetWeights(aData, @PAnsiChar(aData)[aBytes]);
end;


{ TInferBackend }

procedure TInferBackend.ForgetWeights(aFrom, aTo: pointer);
begin
end;

procedure TInferBackend.RunItems(const aJob: TOnInferItem; aCount: integer);
var
  i: integer;
begin
  for i := 0 to aCount - 1 do
    aJob(i);
end;

end.
