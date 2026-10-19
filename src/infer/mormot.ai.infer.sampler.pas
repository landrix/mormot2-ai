/// native inference engine - choosing the next token from the logits
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the warpers of Hugging Face's generate(), in its order: temperature, then
//   top-k, then top-p (nucleus) on what top-k kept, then min-p; then one draw
// - temperature <= 0 is greedy: the highest logit, the lowest id on a tie
//   (InferArgMax), exactly the decoding of phase 2
// - reproducible: an own splitmix64 generator, seeded per sampler - the same
//   seed gives the same random stream on every platform and compiler, so the
//   same tokens wherever the logits are the same (SIMD and Exp may move a draw
//   that falls right on a boundary); it does not draw what torch would (another
//   generator), only from the same distribution
// - top-k keeps every token tied with the k-th, as transformers does
// - exact without sorting the vocabulary: the probabilities are summed over
//   every token, then a threshold is lowered until the tokens top-k and top-p
//   can keep are all above it, and only those are sorted
unit mormot.ai.infer.sampler;

{
  *****************************************************************************

    - TInferSampling: temperature, top-k, top-p, min-p and seed
    - TInferSampler: draws the next token from the logits
    - InferSamplingFromConfig: the defaults of a generation_config.json

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
  mormot.core.os,
  mormot.core.text,
  mormot.core.variants,
  mormot.ai.infer.tensor;

type
  /// how the next token is chosen
  TInferSampling = record
    /// <= 0: greedy; else the logits are divided by it
    Temperature: double;
    /// > 0: only the TopK highest logits stay candidates, and those tied with
    // the k-th
    TopK: integer;
    /// < 1: only the most probable tokens whose probabilities sum to TopP stay
    TopP: double;
    /// > 0: only tokens at least MinP times as probable as the best one stay
    MinP: double;
    /// the seed of the generator; the same seed draws the same tokens
    Seed: Int64;
  end;

  /// draws tokens for one sequence (not thread-safe: one per sequence)
  TInferSampler = class
  protected
    fSampling: TInferSampling;
    fState: QWord;
    fProbs: TDoubleDynArray;
    fCand: TIntegerDynArray;
    fLogits: PSingleArray; // of the running Next: the order of the candidates
    function NextDouble: double;
    procedure SortCandidates(aCount: PtrInt);
  public
    /// raises EInferError for a NaN or negative setting, a TopP above 1 or a
    // MinP of 1 or more
    constructor Create(const aSampling: TInferSampling);
    /// the next token from aCount logits; a NaN or -Inf logit is never chosen
    // (unless every logit is one: then id 0, as InferArgMax does)
    function Next(aLogits: PSingleArray; aCount: PtrInt): PtrInt;
    /// the settings
    property Sampling: TInferSampling
      read fSampling;
  end;

/// false for a NaN or an infinity - by the bits: a comparison with a NaN
// raises EInvalidOp where the FPU traps invalid operations
function InferIsFinite(const aValue: double): boolean;

/// greedy decoding: temperature 0, no filters, seed 0
function InferSamplingGreedy: TInferSampling;

/// the sampling defaults of a Hugging Face generation_config.json: do_sample,
// temperature, top_k, top_p, min_p - greedy when the file is missing or says
// do_sample false; the seed is left at 0 for the caller to set
// - raises EInferError for a file that is no JSON object or a value of the
// wrong kind
function InferSamplingFromConfig(const aFileName: TFileName): TInferSampling;

/// the index of the highest value of aValues[0..aCount-1], the lowest index on
// ties (like torch.argmax); a NaN is never chosen unless all are NaN
function InferArgMax(aValues: PSingleArray; aCount: PtrInt): PtrInt;


implementation

function IsNanBits(aBits: cardinal): boolean;
begin
  // by its bits: the x87 compare of Delphi Win32 ignores the parity flag and
  // finds NaN = NaN
  result := (aBits and $7f800000 = $7f800000) and (aBits and $007fffff <> 0);
end;

/// a NaN or an infinity, by its bits: a comparison with a NaN raises
// EInvalidOp where the FPU traps invalid operations
function IsNonFiniteDouble(const aValue: double): boolean;
begin
  result := PInt64(@aValue)^ and $7ff0000000000000 = $7ff0000000000000;
end;

function InferIsFinite(const aValue: double): boolean;
begin
  result := not IsNonFiniteDouble(aValue);
end;

function InferArgMax(aValues: PSingleArray; aCount: PtrInt): PtrInt;
var
  i: PtrInt;
  b: cardinal;
  k, best: integer;
  p: PCardinalArray;
begin
  // the first value that is no NaN, then strictly greater ones only; all
  // NaN gives 0
  // - compared as integers, no float compare (a vocabulary of 150000 logits
  // per decoded token): the bits of a float, the lower 31 flipped when it is
  // negative, order as the float does; -0 (then -1) is taken as +0, a NaN
  // (exponent all ones, mantissa not 0) skipped; -Inf maps above low(integer)
  // - the flip without a branch: about half the logits are negative, a
  // branch on the sign is mispredicted every other value (5x slower); the
  // mask is a product, 0 or $7fffffff - a subtraction (0 - sign) is done in
  // Int64 by FPC on 64-bit targets, and its shr 1 would keep the sign bit
  result := -1;
  best := low(integer);
  p := pointer(aValues);
  for i := 0 to aCount - 1 do
  begin
    b := p^[i];
    if b and $7fffffff > $7f800000 then
      continue; // NaN
    k := integer(b xor ((b shr 31) * $7fffffff));
    if k = -1 then
      k := 0; // -0
    if k > best then
    begin
      best := k;
      result := i;
    end;
  end;
  if result < 0 then
    result := 0;
end;

function InferSamplingGreedy: TInferSampling;
begin
  FillCharFast(result, SizeOf(result), 0);
  result.TopP := 1;
end;

function InferSamplingFromConfig(const aFileName: TFileName): TInferSampling;
var
  doc: TDocVariantData;

  function Num(const aName: RawUtf8; aDefault: double): double;
  var
    i: integer;
  begin
    i := doc.GetValueIndex(aName);
    if (i < 0) or VarIsEmptyOrNull(doc.Values[i]) then
      result := aDefault
    else if (TVarData(doc.Values[i]).VType = varBoolean) or
            not VariantToDouble(doc.Values[i], result) or
            IsNonFiniteDouble(result) then
      EInferError.RaiseUtf8('InferSamplingFromConfig: % is no number in %',
        [aName, aFileName]);
  end;

var
  k: double;

begin
  result := InferSamplingGreedy;
  if not FileExists(aFileName) then
    exit;
  if not doc.InitJsonFromFile(aFileName, JSON_FAST_FLOAT) or not doc.IsObject then
    EInferError.RaiseUtf8('InferSamplingFromConfig: % is no JSON object', [aFileName]);
  if not doc.B['do_sample'] then
    exit; // transformers decodes greedy then, whatever else the file says
  result.Temperature := Num('temperature', 1);
  k := Num('top_k', 50); // the defaults of transformers
  if (k < 0) or (k > MaxInt) or (Frac(k) <> 0) then
    EInferError.RaiseUtf8('InferSamplingFromConfig: top_k % in %', [k, aFileName]);
  result.TopK := Trunc(k);
  result.TopP := Num('top_p', 1);
  result.MinP := Num('min_p', 0);
end;


{ TInferSampler }

constructor TInferSampler.Create(const aSampling: TInferSampling);
begin
  inherited Create;
  // NaN and infinities first, by their bits: no comparison may see them
  if IsNonFiniteDouble(aSampling.Temperature) or IsNonFiniteDouble(aSampling.TopP) or
     IsNonFiniteDouble(aSampling.MinP) or (aSampling.TopK < 0) or
     (aSampling.TopP <= 0) or (aSampling.TopP > 1) or
     (aSampling.MinP < 0) or (aSampling.MinP >= 1) then
    EInferError.RaiseUtf8('%.Create: temperature %, top_k %, top_p %, min_p %',
      [self, aSampling.Temperature, aSampling.TopK, aSampling.TopP, aSampling.MinP]);
  fSampling := aSampling;
  fState := QWord(aSampling.Seed);
end;

function TInferSampler.NextDouble: double;
var
  z: QWord;
begin
  // splitmix64 (Vigna, public domain): 53 random bits as a double in [0, 1)
  inc(fState, QWord($9E3779B97F4A7C15));
  z := fState;
  z := (z xor (z shr 30)) * QWord($BF58476D1CE4E5B9);
  z := (z xor (z shr 27)) * QWord($94D049BB133111EB);
  z := z xor (z shr 31);
  result := (z shr 11) * (1 / 9007199254740992.0);
end;

procedure TInferSampler.SortCandidates(aCount: PtrInt);
var
  i, j, child, root, tmp: PtrInt;

  // a before b: the higher logit first (as transformers ranks for top-k - two
  // logits may round to the same probability), the lower id on a tie
  function Before(a, b: integer): boolean;
  begin
    result := (fLogits^[a] > fLogits^[b]) or ((fLogits^[a] = fLogits^[b]) and (a < b));
  end;

  procedure Sift(aRoot, aEnd: PtrInt);
  begin
    root := aRoot;
    while root * 2 + 1 <= aEnd do
    begin
      child := root * 2 + 1;
      // a min-heap on Before: the root is the one that comes last
      if (child + 1 <= aEnd) and Before(fCand[child], fCand[child + 1]) then
        inc(child);
      if Before(fCand[root], fCand[child]) then
      begin
        tmp := fCand[root];
        fCand[root] := fCand[child];
        fCand[child] := tmp;
        root := child;
      end
      else
        exit;
    end;
  end;

begin
  // heap sort: O(n log n) for any input, no recursion
  for i := (aCount - 2) div 2 downto 0 do
    Sift(i, aCount - 1);
  for j := aCount - 1 downto 1 do
  begin
    tmp := fCand[0];
    fCand[0] := fCand[j];
    fCand[j] := tmp;
    Sift(0, j - 1);
  end;
end;

function TInferSampler.Next(aLogits: PSingleArray; aCount: PtrInt): PtrInt;
var
  i, n, keep, best: PtrInt;
  m, sum, inv, t, above, cum, mass, u: double;
  bits: cardinal;
  enough: boolean;
begin
  if fSampling.Temperature <= 0 then
  begin
    result := InferArgMax(aLogits, aCount);
    exit;
  end;
  if length(fProbs) < aCount then
  begin
    SetLength(fProbs, aCount);
    SetLength(fCand, aCount);
  end;
  // softmax of logits / temperature over every token: NaN and -Inf weigh 0
  best := InferArgMax(aLogits, aCount);
  m := aLogits^[best];
  bits := PCardinalArray(aLogits)^[best];
  if IsNanBits(bits) or (bits and $7fffffff = $7f800000) then
  begin
    result := best; // no finite best: nothing to draw from
    exit;
  end;
  // a temperature this close to 0 is greedy - tested before dividing: the
  // division itself would overflow and may trap
  if fSampling.Temperature < 1e-30 then
  begin
    result := best;
    exit;
  end;
  inv := 1 / fSampling.Temperature;
  sum := 0;
  for i := 0 to aCount - 1 do
  begin
    bits := PCardinalArray(aLogits)^[i];
    if bits and $7f800000 = $7f800000 then
      fProbs[i] := 0 // NaN or an infinity (+Inf would be the best: excluded above)
    else
      fProbs[i] := Exp((aLogits^[i] - m) * inv);
    sum := sum + fProbs[i];
  end;
  for i := 0 to aCount - 1 do
    fProbs[i] := fProbs[i] / sum;
  if (fSampling.TopK = 0) and (fSampling.TopP >= 1) and (fSampling.MinP = 0) then
  begin
    // no filter: one draw over every token, in id order - nothing to sort
    u := NextDouble;
    cum := 0;
    result := best; // rounding at the very end of the sum: a token that counts
    for i := 0 to aCount - 1 do
    begin
      cum := cum + fProbs[i];
      if (fProbs[i] > 0) and (u < cum) then
      begin
        result := i;
        break;
      end;
    end;
    exit;
  end;
  // the candidates: every token at or above a threshold, lowered until it is
  // sure to hold all that the filters keep - top-k: the k most probable;
  // top-p alone: tokens summing to TopP; min-p alone: its own cut exactly;
  // no filter at all: every token
  if fSampling.TopK > 0 then
    t := fProbs[best] * 1e-3
  else if fSampling.TopP < 1 then
    t := fProbs[best] * 1e-3
  else
    t := fProbs[best] * fSampling.MinP; // 0 without min-p: every token
  repeat
    n := 0;
    above := 0;
    for i := 0 to aCount - 1 do
      // a NaN or an infinity weighs 0 and is never compared (no float trap)
      if (fProbs[i] >= t) and
         (PCardinalArray(aLogits)^[i] and $7f800000 <> $7f800000) then
      begin
        fCand[n] := i;
        inc(n);
        above := above + fProbs[i];
      end;
    if fSampling.TopK > 0 then
      enough := n >= fSampling.TopK // top-p then works inside the top k
    else if fSampling.TopP < 1 then
      enough := above >= fSampling.TopP
    else
      enough := true;
    if enough or (t = 0) then
      break;
    t := t * 1e-3;
    if t < 1e-300 then
      t := 0; // the next pass takes every token
  until false;
  fLogits := aLogits;
  SortCandidates(n);
  // top-k: the k highest logits, and every token whose logit ties the k-th
  if (fSampling.TopK > 0) and (n > fSampling.TopK) then
  begin
    keep := fSampling.TopK;
    while (keep < n) and
          (aLogits^[fCand[keep]] = aLogits^[fCand[fSampling.TopK - 1]]) do
      inc(keep);
    n := keep;
  end;
  // top-p over what top-k kept (renormalized there, as the warpers do), or
  // over every token: the shortest prefix reaching TopP of that mass
  if fSampling.TopP < 1 then
  begin
    mass := 1;
    if fSampling.TopK > 0 then
    begin
      mass := 0;
      for i := 0 to n - 1 do
        mass := mass + fProbs[fCand[i]];
    end;
    cum := 0;
    keep := n;
    for i := 0 to n - 1 do
    begin
      cum := cum + fProbs[fCand[i]];
      if cum >= fSampling.TopP * mass then
      begin
        keep := i + 1;
        break;
      end;
    end;
    n := keep;
  end;
  // min-p: relative to the best token
  if fSampling.MinP > 0 then
  begin
    keep := 1;
    while (keep < n) and (fProbs[fCand[keep]] >= fSampling.MinP * fProbs[fCand[0]]) do
      inc(keep);
    n := keep;
  end;
  // one draw from what is left
  sum := 0;
  for i := 0 to n - 1 do
    sum := sum + fProbs[fCand[i]];
  u := NextDouble * sum;
  result := fCand[0]; // rounding at the very end of the sum: the best one
  cum := 0;
  for i := 0 to n - 1 do
  begin
    cum := cum + fProbs[fCand[i]];
    if u < cum then
    begin
      result := fCand[i];
      break;
    end;
  end;
end;

end.
