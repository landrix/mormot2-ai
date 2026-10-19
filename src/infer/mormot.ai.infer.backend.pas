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

uses
  sysutils,
  mormot.core.base,
  mormot.ai.infer.tensor;

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
    ioRope);

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
    procedure CausalAttention(q: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      keys, values: PSingleArray; aPast, aKvHeads: PtrInt; y: PSingleArray;
      aWindow: PtrInt = 0); virtual; abstract;
    /// y[rows, cols] = x / sqrt(mean(x^2) + eps) * w, per row (RMSNorm)
    // - the normalized value is rounded to single before the weight, as the
    // float32 reference does; y may be x
    procedure RmsNorm(x: PSingleArray; aRows, aCols: PtrInt; w: PSingleArray;
      aEps: double; y: PSingleArray); virtual; abstract;
    /// gate := silu(gate) * up elementwise (the SwiGLU of a feed-forward
    // block), silu(g) = g / (1 + exp(-g)) rounded to single before the product
    procedure SwiGlu(gate, up: PSingleArray; aCount: PtrInt); virtual; abstract;
    /// rotary positions in place, the halves of every head rotated together
    // (rotate_half): x[i], x[i + d/2] := x[i] c - x[i + d/2] s, x[i + d/2] c +
    // x[i] s, in double
    // - x is [aRows, aHeads * aHeadDim]; aCos and aSin are [aRows, aHeadDim / 2],
    // the factors of each row's position
    procedure Rope(x: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      aCos, aSin: PSingleArray); virtual; abstract;
  end;

/// a TInferWeight from its parts
function InferWeight(aData: pointer; aType: TInferDType): TInferWeight;


implementation

function InferWeight(aData: pointer; aType: TInferDType): TInferWeight;
begin
  result.Data := aData;
  result.DType := aType;
end;

end.
