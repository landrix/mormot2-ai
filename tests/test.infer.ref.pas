/// reference data and numeric comparison for the native inference tests
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - reads the datasets of tests/data/infer-ref/ (written by
//   tools/infer-ref/make_reference.py): manifest.json + tensors.bin, raw
//   little-endian f32/i32 blobs described by name, shape, offset, length, sha256
// - InferCompare is the numeric judge of every engine test: elementwise
//   |a - r| <= atol + rtol * |r|, plus max abs/rel error, cosine and both norms -
//   cosine alone would miss a wrong scale
unit test.infer.ref;

{
  *****************************************************************************

    - TInferRefSet: one reference dataset, validated on load
    - TInferCompare / InferCompare: elementwise tolerance check with diagnostics
    - FindInferRefDir: locate tests/data/infer-ref/<dataset> from the executable

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.crypt.core;

type
  /// raised when a reference dataset is malformed or inconsistent
  EInferRef = class(ESynException);

  /// element type of a reference tensor
  TInferRefType = (irtF32, irtI32, irtU8);

  /// one tensor of a reference dataset, as described by its manifest
  TInferRefTensor = record
    /// unique name, e.g. 'case/trade/hidden.3'
    Name: RawUtf8;
    /// element type
    DType: TInferRefType;
    /// dimensions, row-major (C order)
    Shape: TInt64DynArray;
    /// number of elements = product of Shape
    Count: PtrInt;
    /// byte offset of the first element in tensors.bin
    Offset: PtrInt;
    /// byte length = Count * 4
    Bytes: PtrInt;
    /// lowercase hexadecimal sha256 of the Bytes at Offset
    Sha256: RawUtf8;
  end;
  PInferRefTensor = ^TInferRefTensor;
  TInferRefTensorDynArray = array of TInferRefTensor;

  /// one reference dataset: manifest.json + tensors.bin
  // - the constructor checks the blob hash, every tensor's bounds, shape and
  // length, so that accessors can hand out raw pointers into the blob
  TInferRefSet = class
  protected
    fManifest: TDocVariantData;
    fBlob: RawByteString;
    fTensors: TInferRefTensorDynArray;
    procedure Parse;
    function Data(const aName: RawUtf8; aType: TInferRefType;
      out aCount: PtrInt): pointer;
  public
    /// validate a dataset given as manifest JSON and blob content
    // - raises EInferRef on a malformed manifest, a blob hash mismatch, or a
    // tensor whose shape, length or position does not fit the blob
    constructor Create(const aManifestJson: RawUtf8; const aBlob: RawByteString);
    /// load and validate <aFolder>manifest.json + the blob it names
    constructor CreateFromFolder(const aFolder: TFileName);
    /// check the sha256 of every tensor; aFirstBad names the first mismatch
    function VerifyTensorHashes(out aFirstBad: RawUtf8): boolean;
    /// index in Tensors[], or -1
    function IndexOf(const aName: RawUtf8): PtrInt;
    /// the tensor description - raises EInferRef if missing
    function Tensor(const aName: RawUtf8): PInferRefTensor;
    /// pointer to the f32 elements of aName - raises EInferRef if missing or i32
    function F32(const aName: RawUtf8; out aCount: PtrInt): PSingleArray;
    /// pointer to the i32 elements of aName - raises EInferRef if missing or f32
    function I32(const aName: RawUtf8; out aCount: PtrInt): PIntegerArray;
    /// pointer to the u8 bytes of aName - raises EInferRef if missing or not u8
    function U8(const aName: RawUtf8; out aCount: PtrInt): PByteArray;
    /// the parsed manifest
    function Manifest: PDocVariantData;
    /// all tensors, in blob order
    property Tensors: TInferRefTensorDynArray
      read fTensors;
  end;

  /// the outcome of InferCompare
  TInferCompare = record
    /// number of compared elements
    Count: PtrInt;
    /// elements outside atol + rtol * |reference|
    Failed: PtrInt;
    /// NaN or infinite elements, in either array (they always fail)
    NonFinite: PtrInt;
    /// index of the element furthest outside its tolerance, -1 if none failed
    Worst: PtrInt;
    /// largest |actual - reference|
    MaxAbs: double;
    /// largest |actual - reference| / |reference| over non-zero references
    MaxRel: double;
    /// cosine similarity; 0 when either norm is 0
    Cosine: double;
    /// Euclidean norm of the actual values
    NormActual: double;
    /// Euclidean norm of the reference values
    NormReference: double;
  end;

/// compare aCount singles elementwise: |a - r| <= aAtol + aRtol * |r|
// - accumulates in double; non-finite values are counted, never computed with
function InferCompare(aActual, aReference: PSingleArray; aCount: PtrInt;
  aAtol, aRtol: double): TInferCompare;

/// true when no element failed and none was NaN/Inf
function InferCompareOk(const aCompare: TInferCompare): boolean;

/// a one-line diagnostic for a check message
function InferCompareText(const aCompare: TInferCompare): RawUtf8;

/// true for NaN and +-Inf (exponent bits all set), without an FPU operation
function IsNonFiniteSingle(const aValue: single): boolean;

/// Euclidean norm of aCount singles, accumulated in double
function InferL2Norm(aValues: PSingleArray; aCount: PtrInt): double;

/// tests/data/infer-ref/<aDataset>/ found from the executable upwards, or ''
function FindInferRefDir(const aDataset: RawUtf8): TFileName;


/// TDecoderModel.Route written out, the choice then sorted by id as the route
// kernels of the devices write it (slice D3c) - checked against the model's
// Route in TTestInferCuda.RouteMatchesHost
procedure RouteRef(aLogits: PSingleArray; aExperts, aK: PtrInt; aNorm: boolean;
  aIds: PIntegerArray; aW: PSingleArray);

implementation

const
  /// element size of each reference type
  REF_SIZE: array[TInferRefType] of integer = (4, 4, 1);

{ TInferRefSet }

constructor TInferRefSet.Create(const aManifestJson: RawUtf8;
  const aBlob: RawByteString);
begin
  inherited Create;
  if not fManifest.InitJson(aManifestJson, JSON_FAST_FLOAT) or
     (fManifest.Kind <> dvObject) then
    EInferRef.RaiseU('manifest is not a JSON object');
  fBlob := aBlob;
  Parse;
end;

constructor TInferRefSet.CreateFromFolder(const aFolder: TFileName);
var
  json: RawUtf8;
  doc: TDocVariantData;
  blob: PDocVariantData;
  folder, blobFile: TFileName;
begin
  folder := IncludeTrailingPathDelimiter(aFolder);
  json := RawUtf8FromFile(folder + 'manifest.json');
  if json = '' then
    EInferRef.RaiseUtf8('missing %manifest.json', [folder]);
  doc.InitJson(json, JSON_FAST);
  if not doc.GetAsObject('blob', blob) then
    EInferRef.RaiseUtf8('%manifest.json: no blob section', [folder]);
  blobFile := folder + Utf8ToString(blob^.U['file']);
  Create(json, StringFromFile(blobFile));
end;

procedure TInferRefSet.Parse;
var
  blob, list, item, shape: PDocVariantData;
  i, j: PtrInt;
  dim, count, bytes, offset: Int64;
  t: PInferRefTensor;
  dtype: RawUtf8;
begin
  if not fManifest.GetAsObject('blob', blob) then
    EInferRef.RaiseU('manifest: no blob section');
  if blob^.I['bytes'] <> length(fBlob) then
    EInferRef.RaiseUtf8('blob has % bytes, manifest says %',
      [length(fBlob), blob^.I['bytes']]);
  if Sha256(fBlob) <> blob^.U['sha256'] then
    EInferRef.RaiseU('blob sha256 mismatch');
  if not fManifest.GetAsArray('tensors', list) then
    EInferRef.RaiseU('manifest: no tensors array');
  SetLength(fTensors, list^.Count);
  for i := 0 to list^.Count - 1 do
  begin
    item := _Safe(list^.Values[i]);
    t := @fTensors[i];
    t^.Name := item^.U['name'];
    if t^.Name = '' then
      EInferRef.RaiseUtf8('tensor #% has no name', [i]);
    dtype := item^.U['dtype'];
    if dtype = 'f32' then
      t^.DType := irtF32
    else if dtype = 'i32' then
      t^.DType := irtI32
    else if dtype = 'u8' then
      t^.DType := irtU8
    else
      EInferRef.RaiseUtf8('%: unsupported dtype %', [t^.Name, dtype]);
    if not item^.GetAsArray('shape', shape) then
      EInferRef.RaiseUtf8('%: no shape', [t^.Name]);
    // product of the dimensions, refused before it can overflow or exceed the blob
    count := 1;
    SetLength(t^.Shape, shape^.Count);
    for j := 0 to shape^.Count - 1 do
    begin
      if not VariantToInt64(shape^.Values[j], dim) or (dim < 0) then
        EInferRef.RaiseUtf8('%: invalid dimension', [t^.Name]);
      t^.Shape[j] := dim;
      if (dim > length(fBlob)) or
         ((dim <> 0) and (count > length(fBlob) div dim)) then
        EInferRef.RaiseUtf8('%: shape larger than the blob', [t^.Name]);
      count := count * dim; // <= blob length: no overflow
    end;
    // compare in Int64 before narrowing to PtrInt: on Win32 count * 4 could wrap
    bytes := item^.I['bytes'];
    offset := item^.I['offset'];
    if bytes <> count * REF_SIZE[t^.DType] then
      EInferRef.RaiseUtf8('%: % bytes for % elements', [t^.Name, bytes, count]);
    if (offset < 0) or (offset mod REF_SIZE[t^.DType] <> 0) or
       (offset > length(fBlob) - bytes) then
      EInferRef.RaiseUtf8('%: offset % outside the blob', [t^.Name, offset]);
    t^.Count := count;
    t^.Offset := offset;
    t^.Bytes := bytes;
    t^.Sha256 := item^.U['sha256'];
    if IndexOf(t^.Name) <> i then
      EInferRef.RaiseUtf8('duplicated tensor name %', [t^.Name]);
  end;
end;

function TInferRefSet.VerifyTensorHashes(out aFirstBad: RawUtf8): boolean;
var
  i: PtrInt;
begin
  aFirstBad := '';
  for i := 0 to high(fTensors) do
    if Sha256(PAnsiChar(pointer(fBlob)) + fTensors[i].Offset,
       fTensors[i].Bytes) <> fTensors[i].Sha256 then
    begin
      aFirstBad := fTensors[i].Name;
      result := false;
      exit;
    end;
  result := true;
end;

function TInferRefSet.IndexOf(const aName: RawUtf8): PtrInt;
var
  i: PtrInt;
begin
  for i := 0 to high(fTensors) do
    if fTensors[i].Name = aName then
    begin
      result := i;
      exit;
    end;
  result := -1;
end;

function TInferRefSet.Tensor(const aName: RawUtf8): PInferRefTensor;
var
  i: PtrInt;
begin
  i := IndexOf(aName);
  if i < 0 then
    EInferRef.RaiseUtf8('no tensor %', [aName]);
  result := @fTensors[i];
end;

function TInferRefSet.Data(const aName: RawUtf8; aType: TInferRefType;
  out aCount: PtrInt): pointer;
var
  t: PInferRefTensor;
begin
  t := Tensor(aName);
  if t^.DType <> aType then
    EInferRef.RaiseUtf8('%: unexpected dtype', [aName]);
  aCount := t^.Count;
  result := PAnsiChar(pointer(fBlob)) + t^.Offset;
end;

function TInferRefSet.F32(const aName: RawUtf8; out aCount: PtrInt): PSingleArray;
begin
  result := Data(aName, irtF32, aCount);
end;

function TInferRefSet.U8(const aName: RawUtf8; out aCount: PtrInt): PByteArray;
begin
  result := Data(aName, irtU8, aCount);
end;

function TInferRefSet.I32(const aName: RawUtf8; out aCount: PtrInt): PIntegerArray;
begin
  result := Data(aName, irtI32, aCount);
end;

function TInferRefSet.Manifest: PDocVariantData;
begin
  result := @fManifest;
end;


{ comparison }

function IsNonFiniteSingle(const aValue: single): boolean;
begin
  result := PCardinal(@aValue)^ and $7f800000 = $7f800000;
end;

function InferL2Norm(aValues: PSingleArray; aCount: PtrInt): double;
var
  i: PtrInt;
  v: double;
begin
  result := 0;
  for i := 0 to aCount - 1 do
  begin
    v := aValues^[i];
    result := result + v * v;
  end;
  result := Sqrt(result);
end;

function InferCompare(aActual, aReference: PSingleArray; aCount: PtrInt;
  aAtol, aRtol: double): TInferCompare;
var
  i: PtrInt;
  a, r, diff, limit, excess, worstExcess, dot, na, nr: double;
begin
  FillCharFast(result, SizeOf(result), 0);
  result.Count := aCount;
  result.Worst := -1;
  worstExcess := 0;
  dot := 0;
  na := 0;
  nr := 0;
  for i := 0 to aCount - 1 do
  begin
    if IsNonFiniteSingle(aActual^[i]) or IsNonFiniteSingle(aReference^[i]) then
    begin
      inc(result.NonFinite);
      inc(result.Failed);
      if worstExcess < 1e300 then
        result.Worst := i; // the first NaN/Inf outranks every finite error
      worstExcess := 1e300;
      continue;
    end;
    a := aActual^[i];
    r := aReference^[i];
    diff := Abs(a - r);
    if diff > result.MaxAbs then
      result.MaxAbs := diff;
    if (r <> 0) and (diff / Abs(r) > result.MaxRel) then
      result.MaxRel := diff / Abs(r);
    limit := aAtol + aRtol * Abs(r);
    if diff > limit then
    begin
      inc(result.Failed);
      excess := diff - limit;
      if excess > worstExcess then
      begin
        worstExcess := excess;
        result.Worst := i;
      end;
    end;
    dot := dot + a * r;
    na := na + a * a;
    nr := nr + r * r;
  end;
  result.NormActual := Sqrt(na);
  result.NormReference := Sqrt(nr);
  if (na > 0) and (nr > 0) then
    result.Cosine := dot / (result.NormActual * result.NormReference);
end;

function InferCompareOk(const aCompare: TInferCompare): boolean;
begin
  result := (aCompare.Failed = 0) and (aCompare.NonFinite = 0);
end;

function InferCompareText(const aCompare: TInferCompare): RawUtf8;
begin
  FormatUtf8('n=% failed=% nonfinite=% worst=% maxabs=% maxrel=% cos=% ' +
    'norm=%/%', [aCompare.Count, aCompare.Failed, aCompare.NonFinite,
    aCompare.Worst, aCompare.MaxAbs, aCompare.MaxRel, aCompare.Cosine,
    aCompare.NormActual, aCompare.NormReference], result);
end;

function FindInferRefDir(const aDataset: RawUtf8): TFileName;
var
  dir, cand: TFileName;
  i: integer;
begin
  result := '';
  dir := ExcludeTrailingPathDelimiter(Executable.ProgramFilePath);
  for i := 1 to 6 do
  begin
    cand := MakePath([dir, 'tests', 'data', 'infer-ref', aDataset], true);
    if FileExists(cand + 'manifest.json') then
    begin
      result := cand;
      exit;
    end;
    dir := ExtractFileDir(dir);
  end;
end;

procedure RouteRef(aLogits: PSingleArray; aExperts, aK: PtrInt; aNorm: boolean;
  aIds: PIntegerArray; aW: PSingleArray);
var
  i, j, t, best, id: PtrInt;
  m, sum, picked: double;
  taken: boolean;
  v: single;
begin
  m := aLogits^[0];
  for i := 1 to aExperts - 1 do
    if aLogits^[i] > m then
      m := aLogits^[i];
  sum := 0;
  for i := 0 to aExperts - 1 do
    sum := sum + Exp(aLogits^[i] - m);
  picked := 0;
  for j := 0 to aK - 1 do
  begin
    best := -1;
    for i := 0 to aExperts - 1 do
    begin
      taken := false;
      for t := 0 to j - 1 do
        if aIds^[t] = i then
          taken := true;
      if not taken and ((best < 0) or (aLogits^[i] > aLogits^[best])) then
        best := i;
    end;
    aIds^[j] := best;
    aW^[j] := Exp(aLogits^[best] - m) / sum;
    picked := picked + aW^[j];
  end;
  if aNorm then
    for j := 0 to aK - 1 do
      aW^[j] := aW^[j] / picked;
  for j := 1 to aK - 1 do
  begin
    id := aIds^[j];
    v := aW^[j];
    t := j;
    while (t > 0) and (aIds^[t - 1] > id) do
    begin
      aIds^[t] := aIds^[t - 1];
      aW^[t] := aW^[t - 1];
      dec(t);
    end;
    aIds^[t] := id;
    aW^[t] := v;
  end;
end;


end.
