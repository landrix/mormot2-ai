/// native inference engine - GGUF reader
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room from the published GGUF v3 format description (ggml docs/gguf.md):
//   magic, version, tensor and key/value counts, the key/value pairs, the tensor
//   infos (name, dimensions innermost first, ggml type, offset), padding to the
//   alignment, then the data section
// - everything is validated before a tensor or a value is handed out: every read
//   is bounded by the file size, counts cannot overflow, keys and tensor names
//   are unique, booleans are 0 or 1, strings hold no NUL, arrays are not nested,
//   the alignment is a power of two, tensors have 1 to 4 positive dimensions and
//   a supported type, start on the alignment, stay inside the data section and
//   follow each other in file order without overlap - the only gaps allowed are
//   the alignment padding a writer adds. A damaged or hostile file is refused
//   with EInferError, never read past its end
// - shapes are reported row-major (C order) like TSafeTensorsFile: GGUF lists
//   the innermost dimension first, so [ne0, ne1] becomes [ne1, ne0]
unit mormot.ai.infer.loader.gguf;

{
  *****************************************************************************

    - TGgufFile: a validated, memory-mapped GGUF v3 file
    - Typed access to the key/value metadata

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
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.infer.tensor,
  mormot.ai.infer.memory;

const
  /// the only GGUF version read
  GGUF_VERSION = 3;
  /// the alignment of the data section when general.alignment is absent
  GGUF_DEFAULT_ALIGNMENT = 32;
  /// the most dimensions of a tensor (GGML_MAX_DIMS)
  GGUF_MAX_DIMS = 4;
  /// the most keys, tensors or array elements read into memory: a count the
  // file size allows may still be an allocation a 32-bit process cannot make
  // (real files hold a few dozen keys, some hundred tensors, 250k tokens)
  GGUF_MAX_ITEMS = 1 shl 24;

type
  /// the value types of the GGUF metadata
  TGgufType = (
    gtUInt8,
    gtInt8,
    gtUInt16,
    gtInt16,
    gtUInt32,
    gtInt32,
    gtFloat32,
    gtBool,
    gtString,
    gtArray,
    gtUInt64,
    gtInt64,
    gtFloat64);

  /// one key/value pair - a view into the mapped file
  TGgufValue = record
    /// the key, e.g. 'bert.block_count'
    Key: RawUtf8;
    /// the type of the value; gtArray for an array
    VType: TGgufType;
    /// the element type of an array
    ElemType: TGgufType;
    /// the number of elements of an array, 1 otherwise
    Count: Int64;
    /// the value, or the first element of an array (a string starts with its
    // 8-byte length)
    Data: PAnsiChar;
  end;
  PGgufValue = ^TGgufValue;
  TGgufValueDynArray = array of TGgufValue;

  /// a validated GGUF file, mapped read-only
  TGgufFile = class
  protected
    fFile: TInferMappedFile;
    fValues: TGgufValueDynArray;
    fKeys: TRawUtf8DynArray;
    fKeyIndex: TIntegerDynArray;
    fTensors: TInferTensorDynArray;
    fNames: TRawUtf8DynArray;
    fNameIndex: TIntegerDynArray;
    fAlignment: Int64;
    fDataStart: Int64;
    procedure Parse;
    function Value(const aKey: RawUtf8; aType: TGgufType): PGgufValue;
    function ArrayOf(const aKey: RawUtf8; aElem: TGgufType;
      aCount: Int64): PGgufValue;
  public
    /// map and validate aFileName - raises EInferError on any defect
    constructor Create(const aFileName: TFileName);
    /// validate an in-memory content - for tests and small fixtures
    constructor CreateFromContent(const aContent: RawByteString);
    /// unmap the file
    destructor Destroy; override;
    /// the key/value pair aKey, or nil
    function Find(const aKey: RawUtf8): PGgufValue;
    /// true when aKey exists
    function Has(const aKey: RawUtf8): boolean;
    /// a string value - raises EInferError if missing or of another type
    function Str(const aKey: RawUtf8): RawUtf8;
    /// an integer value of any integer type - raises EInferError if missing,
    // of another type, or (for UINT64) above High(Int64)
    function Int(const aKey: RawUtf8): Int64;
    /// a FLOAT32 or FLOAT64 value - raises EInferError otherwise
    function Float(const aKey: RawUtf8): double;
    /// a BOOL value - raises EInferError if missing or of another type
    function Bool(const aKey: RawUtf8): boolean;
    /// a BOOL value, aDefault if the key is absent - raises EInferError if
    // present with another type
    function BoolDef(const aKey: RawUtf8; aDefault: boolean): boolean;
    /// an array of strings, expected with aCount elements (-1 = any)
    function Strings(const aKey: RawUtf8; aCount: Int64 = -1): TRawUtf8DynArray;
    /// an array of FLOAT32, expected with aCount elements (-1 = any)
    // - returns a view into the file; aCount receives the length
    function Floats32(const aKey: RawUtf8; var aCount: Int64): PSingleArray;
    /// an array of INT32, expected with aCount elements (-1 = any)
    function Ints32(const aKey: RawUtf8; var aCount: Int64): PIntegerArray;
    /// an array of UINT8 or INT8 as raw bytes
    function Bytes(const aKey: RawUtf8): RawByteString;
    /// the tensor named aName, or nil
    function FindTensor(const aName: RawUtf8): PInferTensor;
    /// the tensor named aName - raises EInferError if missing
    function Tensor(const aName: RawUtf8): PInferTensor;
    /// the tensor aName, checked against a type and row-major dimensions
    // - raises EInferError if missing, of another type, or shaped otherwise
    function Weight(const aName: RawUtf8; aType: TInferDType;
      const aDims: array of Int64): pointer;
    /// all key/value pairs, in file order
    property Values: TGgufValueDynArray
      read fValues;
    /// all tensors, in file order (= by offset)
    property Tensors: TInferTensorDynArray
      read fTensors;
    /// the alignment of the data section
    property Alignment: Int64
      read fAlignment;
    /// the mapped file name, '' for an in-memory content
    function FileName: TFileName;
  end;

const
  /// the GGUF name of each value type, for messages
  GGUF_TYPE_NAME: array[TGgufType] of RawUtf8 = (
    'UINT8', 'INT8', 'UINT16', 'INT16', 'UINT32', 'INT32', 'FLOAT32', 'BOOL',
    'STRING', 'ARRAY', 'UINT64', 'INT64', 'FLOAT64');

  /// the size in bytes of each fixed-size value type (0 = variable)
  GGUF_TYPE_SIZE: array[TGgufType] of byte = (
    1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8);

/// the engine type of a ggml type id, idtUnknown if not supported
// - 0 F32, 1 F16, 2 Q4_0, 6 Q5_0, 8 Q8_0, 12 Q4_K, 14 Q6_K, 30 BF16, 39 MXFP4
function GgufTensorType(aGgmlType: cardinal): TInferDType;


implementation

/// the ggml name of a type id, for messages (ggml.h)
function GgufTypeName(aGgmlType: cardinal): RawUtf8;
const
  NAMES: array[0..30] of RawUtf8 = (
    'F32', 'F16', 'Q4_0', 'Q4_1', '', '', 'Q5_0', 'Q5_1', 'Q8_0', 'Q8_1',
    'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K', 'IQ2_XXS', 'IQ2_XS',
    'IQ3_XXS', 'IQ1_S', 'IQ4_NL', 'IQ3_S', 'IQ2_S', 'IQ4_XS', 'I8', 'I16',
    'I32', 'I64', 'F64', 'IQ1_M', 'BF16');
begin
  result := '';
  if aGgmlType <= high(NAMES) then
    result := NAMES[aGgmlType]
  else if aGgmlType = 34 then
    result := 'TQ1_0'
  else if aGgmlType = 35 then
    result := 'TQ2_0'
  else if aGgmlType = 39 then
    result := 'MXFP4';
  if result = '' then
    result := '?';
end;

function GgufTensorType(aGgmlType: cardinal): TInferDType;
begin
  case aGgmlType of
    0:
      result := idtF32;
    1:
      result := idtF16;
    2:
      result := idtQ4_0;
    6:
      result := idtQ5_0;
    8:
      result := idtQ8_0;
    12:
      result := idtQ4_K;
    14:
      result := idtQ6_K;
    30:
      result := idtBF16;
    39:
      result := idtMXFP4;
  else
    result := idtUnknown;
  end;
end;


{ TGgufReader - a bounded cursor over the header }

type
  TGgufReader = object
    Owner: TObject;
    Start, Cur: PAnsiChar;
    Left: Int64;
    procedure Need(aBytes: Int64; const aWhat: RawUtf8);
    function U32(const aWhat: RawUtf8): cardinal;
    function U64(const aWhat: RawUtf8): Int64;
    procedure Skip(aBytes: Int64; const aWhat: RawUtf8);
    function Str(const aWhat: RawUtf8): RawUtf8;
    function Pos: Int64;
  end;

procedure TGgufReader.Need(aBytes: Int64; const aWhat: RawUtf8);
begin
  if (aBytes < 0) or (aBytes > Left) then
    EInferError.RaiseUtf8('%: % runs past the end of the file', [Owner, aWhat]);
end;

function TGgufReader.U32(const aWhat: RawUtf8): cardinal;
begin
  Need(4, aWhat);
  MoveFast(Cur^, result, 4); // little-endian on every target we build
  inc(Cur, 4);
  dec(Left, 4);
end;

function TGgufReader.U64(const aWhat: RawUtf8): Int64;
begin
  Need(8, aWhat);
  MoveFast(Cur^, result, 8);
  inc(Cur, 8);
  dec(Left, 8);
  // a count or length above High(Int64) can never fit the file
  if result < 0 then
    EInferError.RaiseUtf8('%: % is out of range', [Owner, aWhat]);
end;

procedure TGgufReader.Skip(aBytes: Int64; const aWhat: RawUtf8);
begin
  Need(aBytes, aWhat);
  inc(Cur, PtrInt(aBytes)); // aBytes <= Left <= file size: fits PtrInt
  dec(Left, aBytes);
end;

function TGgufReader.Str(const aWhat: RawUtf8): RawUtf8;
var
  n: Int64;
begin
  n := U64(aWhat);
  Need(n, aWhat);
  FastSetString(result, Cur, n);
  if StrLen(pointer(result)) <> n then
    EInferError.RaiseUtf8('%: % contains NUL', [Owner, aWhat]);
  inc(Cur, PtrInt(n));
  dec(Left, n);
end;

function TGgufReader.Pos: Int64;
begin
  result := Cur - Start;
end;


{ TGgufFile }

constructor TGgufFile.Create(const aFileName: TFileName);
begin
  inherited Create;
  fFile := TInferMappedFile.Create(aFileName);
  Parse;
end;

constructor TGgufFile.CreateFromContent(const aContent: RawByteString);
begin
  inherited Create;
  fFile := TInferMappedFile.CreateFromContent(aContent);
  Parse;
end;

destructor TGgufFile.Destroy;
begin
  fFile.Free;
  inherited Destroy;
end;

/// true when aLen bytes at p hold no NUL - a string value of the file
function NoNul(p: PAnsiChar; aLen: Int64): boolean;
var
  i: PtrInt;
begin
  result := false;
  for i := 0 to PtrInt(aLen) - 1 do
    if p[i] = #0 then
      exit;
  result := true;
end;

procedure SortedUnique(const aOwner: TObject; const aWhat: RawUtf8;
  var aNames: TRawUtf8DynArray; var aIndex: TIntegerDynArray; aCount: PtrInt);
var
  i: PtrInt;
begin
  SetLength(aIndex, aCount);
  for i := 0 to aCount - 1 do
    aIndex[i] := i;
  QuickSortRawUtf8(aNames, aCount, @aIndex);
  for i := 1 to aCount - 1 do
    if aNames[i] = aNames[i - 1] then
      EInferError.RaiseUtf8('%: % "%" defined twice', [aOwner, aWhat, aNames[i]]);
end;

function SortedFind(const aNames: TRawUtf8DynArray; const aIndex: TIntegerDynArray;
  const aName: RawUtf8): PtrInt;
var
  lo, hi, mid, c: PtrInt;
begin
  lo := 0;
  hi := high(aNames);
  while lo <= hi do
  begin
    mid := (lo + hi) shr 1;
    c := StrComp(pointer(aNames[mid]), pointer(aName));
    if c = 0 then
    begin
      result := aIndex[mid];
      exit;
    end
    else if c < 0 then
      lo := mid + 1
    else
      hi := mid - 1;
  end;
  result := -1;
end;

procedure TGgufFile.Parse;
var
  r: TGgufReader;
  magic, version, t, nd: cardinal;
  nTensors, nValues, count, bytes, off, dim, prevEnd, dataLen, n: Int64;
  i, j: PtrInt;
  v: PGgufValue;
  ti: PInferTensor;
  shape: TInt64DynArray;
  align: PGgufValue;

  procedure ReadScalar(aType: TGgufType; const aWhat: RawUtf8);
  var
    len: Int64;
  begin
    if aType = gtString then
    begin
      len := r.U64(aWhat);
      r.Need(len, aWhat);
      if not NoNul(r.Cur, len) then
        EInferError.RaiseUtf8('%: % contains NUL', [self, aWhat]);
      r.Skip(len, aWhat);
    end
    else if aType = gtBool then
    begin
      r.Need(1, aWhat);
      if ord(r.Cur^) > 1 then
        EInferError.RaiseUtf8('%: % is a BOOL of value %', [self, aWhat, ord(r.Cur^)]);
      r.Skip(1, aWhat);
    end
    else
      r.Skip(GGUF_TYPE_SIZE[aType], aWhat);
  end;

begin
  r.Owner := self;
  r.Start := fFile.Buffer;
  r.Cur := r.Start;
  r.Left := fFile.Size;
  magic := r.U32('magic');
  if magic <> $46554747 then // 'GGUF'
    EInferError.RaiseUtf8('%: not a GGUF file', [self]);
  version := r.U32('version');
  if version <> GGUF_VERSION then
    EInferError.RaiseUtf8('%: GGUF version % not supported (only %)',
      [self, version, GGUF_VERSION]);
  nTensors := r.U64('tensor count');
  nValues := r.U64('key/value count');
  // a key/value pair takes at least 12 bytes (key length + type), a tensor
  // info at least 32 (name length, dims, one dim, type, offset): a hostile
  // count is refused before any allocation
  if (nTensors > r.Left div 32) or (nValues > r.Left div 12) or
     (nTensors > GGUF_MAX_ITEMS) or (nValues > GGUF_MAX_ITEMS) then
    EInferError.RaiseUtf8('%: % tensors and % values cannot fit % bytes',
      [self, nTensors, nValues, fFile.Size]);
  SetLength(fValues, nValues);
  for i := 0 to PtrInt(nValues) - 1 do
  begin
    v := @fValues[i];
    v^.Key := r.Str('a key');
    if v^.Key = '' then
      EInferError.RaiseUtf8('%: empty key', [self]);
    t := r.U32(v^.Key);
    if t > cardinal(ord(high(TGgufType))) then
      EInferError.RaiseUtf8('%: % has unknown type %', [self, v^.Key, t]);
    v^.VType := TGgufType(t);
    v^.ElemType := v^.VType;
    v^.Count := 1;
    if v^.VType = gtArray then
    begin
      t := r.U32(v^.Key);
      if t > cardinal(ord(high(TGgufType))) then
        EInferError.RaiseUtf8('%: % has unknown element type %', [self, v^.Key, t]);
      v^.ElemType := TGgufType(t);
      if v^.ElemType = gtArray then
        EInferError.RaiseUtf8('%: % is a nested array', [self, v^.Key]);
      v^.Count := r.U64(v^.Key);
      v^.Data := r.Cur;
      // every element takes at least one byte, a string at least its 8-byte
      // length: a hostile count is refused before the loop
      if (v^.Count > r.Left) or
         ((v^.ElemType = gtString) and (v^.Count > r.Left div 8)) then
        EInferError.RaiseUtf8('%: % has % elements - more than the file holds',
          [self, v^.Key, v^.Count]);
      if v^.ElemType in [gtString, gtBool] then
        for j := 0 to PtrInt(v^.Count) - 1 do
          ReadScalar(v^.ElemType, v^.Key)
      else
      begin
        if v^.Count > r.Left div GGUF_TYPE_SIZE[v^.ElemType] then
          EInferError.RaiseUtf8('%: % has % elements - more than the file holds',
            [self, v^.Key, v^.Count]);
        r.Skip(v^.Count * GGUF_TYPE_SIZE[v^.ElemType], v^.Key);
      end;
    end
    else
    begin
      v^.Data := r.Cur;
      ReadScalar(v^.VType, v^.Key);
    end;
  end;
  SetLength(fKeys, nValues);
  for i := 0 to PtrInt(nValues) - 1 do
    fKeys[i] := fValues[i].Key;
  SortedUnique(self, 'key', fKeys, fKeyIndex, PtrInt(nValues));
  // the alignment of the data section
  fAlignment := GGUF_DEFAULT_ALIGNMENT;
  align := Find('general.alignment');
  if align <> nil then
  begin
    if align^.VType <> gtUInt32 then
      EInferError.RaiseUtf8('%: general.alignment is % instead of UINT32',
        [self, GGUF_TYPE_NAME[align^.VType]]);
    fAlignment := PCardinal(align^.Data)^;
    if (fAlignment = 0) or (fAlignment and (fAlignment - 1) <> 0) then
      EInferError.RaiseUtf8('%: alignment % is no power of two', [self, fAlignment]);
  end;
  // tensor infos
  SetLength(fTensors, nTensors);
  for i := 0 to PtrInt(nTensors) - 1 do
  begin
    ti := @fTensors[i];
    ti^.Name := r.Str('a tensor name');
    if ti^.Name = '' then
      EInferError.RaiseUtf8('%: tensor without a name', [self]);
    nd := r.U32(ti^.Name);
    if (nd = 0) or (nd > GGUF_MAX_DIMS) then
      EInferError.RaiseUtf8('%: % has % dimensions', [self, ti^.Name, nd]);
    SetLength(shape, nd);
    for j := 0 to PtrInt(nd) - 1 do
      shape[j] := r.U64(ti^.Name);
    t := r.U32(ti^.Name);
    ti^.DType := GgufTensorType(t);
    if ti^.DType = idtUnknown then
      EInferError.RaiseUtf8('%: % has unsupported ggml type % (%)',
        [self, ti^.Name, t, GgufTypeName(t)]);
    off := r.U64(ti^.Name);
    // row-major like every other view of the engine: innermost dimension last
    SetLength(ti^.Shape, nd);
    for j := 0 to PtrInt(nd) - 1 do
    begin
      dim := shape[PtrInt(nd) - 1 - j];
      if dim <= 0 then
        EInferError.RaiseUtf8('%: % has a dimension of %', [self, ti^.Name, dim]);
      ti^.Shape[j] := dim;
    end;
    // a block type packs whole blocks per row
    if shape[0] mod INFER_DTYPE_BLOCK[ti^.DType] <> 0 then
      EInferError.RaiseUtf8('%: % rows of % are no whole % blocks',
        [self, ti^.Name, shape[0], INFER_DTYPE_NAME[ti^.DType]]);
    if not InferShapeCount(ti^.Shape, fFile.Size, count) or
       not InferDTypeBytes(ti^.DType, count, bytes) then
      EInferError.RaiseUtf8('%: % shape % exceeds the file',
        [self, ti^.Name, InferShapeText(ti^.Shape)]);
    ti^.Count := count;
    ti^.Bytes := bytes;
    ti^.Data := pointer(PtrInt(off)); // relative for now, checked below
    if off mod fAlignment <> 0 then
      EInferError.RaiseUtf8('%: % offset % is not aligned to %',
        [self, ti^.Name, off, fAlignment]);
    if off > fFile.Size then
      EInferError.RaiseUtf8('%: % offset % outside the file', [self, ti^.Name, off]);
  end;
  SetLength(fNames, nTensors);
  for i := 0 to PtrInt(nTensors) - 1 do
    fNames[i] := fTensors[i].Name;
  SortedUnique(self, 'tensor', fNames, fNameIndex, PtrInt(nTensors));
  // the data section starts at the next multiple of the alignment
  fDataStart := r.Pos;
  if fDataStart mod fAlignment <> 0 then
    inc(fDataStart, fAlignment - fDataStart mod fAlignment);
  if fDataStart > fFile.Size then
    EInferError.RaiseUtf8('%: no data section', [self]);
  dataLen := fFile.Size - fDataStart;
  // tensors follow each other in file order: each starts at the aligned end of
  // the one before, the last one ends at most one padding before the file end
  prevEnd := 0;
  for i := 0 to PtrInt(nTensors) - 1 do
  begin
    ti := @fTensors[i];
    off := PtrInt(ti^.Data);
    n := prevEnd;
    if n mod fAlignment <> 0 then
      inc(n, fAlignment - n mod fAlignment);
    if off <> n then
      EInferError.RaiseUtf8('%: % starts at %, expected % (overlap, gap or order)',
        [self, ti^.Name, off, n]);
    if ti^.Bytes > dataLen - off then
      EInferError.RaiseUtf8('%: % range [%,%) outside the % data bytes',
        [self, ti^.Name, off, off + ti^.Bytes, dataLen]);
    ti^.Data := fFile.Buffer + PtrInt(fDataStart + off);
    prevEnd := off + ti^.Bytes;
  end;
  if dataLen - prevEnd >= fAlignment then
    EInferError.RaiseUtf8('%: % unreferenced bytes after the last tensor',
      [self, dataLen - prevEnd]);
end;

function TGgufFile.FileName: TFileName;
begin
  result := fFile.FileName;
end;

function TGgufFile.Find(const aKey: RawUtf8): PGgufValue;
var
  i: PtrInt;
begin
  i := SortedFind(fKeys, fKeyIndex, aKey);
  if i < 0 then
    result := nil
  else
    result := @fValues[i];
end;

function TGgufFile.Has(const aKey: RawUtf8): boolean;
begin
  result := Find(aKey) <> nil;
end;

function TGgufFile.Value(const aKey: RawUtf8; aType: TGgufType): PGgufValue;
begin
  result := Find(aKey);
  if result = nil then
    EInferError.RaiseUtf8('%: no metadata %', [self, aKey]);
  if result^.VType <> aType then
    EInferError.RaiseUtf8('%: % is %, expected %',
      [self, aKey, GGUF_TYPE_NAME[result^.VType], GGUF_TYPE_NAME[aType]]);
end;

function TGgufFile.ArrayOf(const aKey: RawUtf8; aElem: TGgufType;
  aCount: Int64): PGgufValue;
begin
  result := Value(aKey, gtArray);
  if result^.ElemType <> aElem then
    EInferError.RaiseUtf8('%: % is an array of %, expected %',
      [self, aKey, GGUF_TYPE_NAME[result^.ElemType], GGUF_TYPE_NAME[aElem]]);
  if (aCount >= 0) and (result^.Count <> aCount) then
    EInferError.RaiseUtf8('%: % has % elements, expected %',
      [self, aKey, result^.Count, aCount]);
end;

function TGgufFile.Str(const aKey: RawUtf8): RawUtf8;
var
  v: PGgufValue;
  n: Int64;
begin
  v := Value(aKey, gtString);
  MoveFast(v^.Data^, n, 8);
  FastSetString(result, v^.Data + 8, n);
end;

function TGgufFile.Int(const aKey: RawUtf8): Int64;
var
  v: PGgufValue;
begin
  v := Find(aKey);
  if v = nil then
    EInferError.RaiseUtf8('%: no metadata %', [self, aKey]);
  case v^.VType of
    gtUInt8:
      result := PByte(v^.Data)^;
    gtInt8:
      result := PShortInt(v^.Data)^;
    gtUInt16:
      result := PWord(v^.Data)^;
    gtInt16:
      result := PSmallInt(v^.Data)^;
    gtUInt32:
      result := PCardinal(v^.Data)^;
    gtInt32:
      result := PInteger(v^.Data)^;
    gtUInt64, gtInt64:
      begin
        result := PInt64(v^.Data)^;
        if (v^.VType = gtUInt64) and (result < 0) then
          EInferError.RaiseUtf8('%: % is out of range', [self, aKey]);
      end;
  else
    EInferError.RaiseUtf8('%: % is %, expected an integer',
      [self, aKey, GGUF_TYPE_NAME[v^.VType]]);
    result := 0; // makes the compiler happy
  end;
end;

function TGgufFile.Float(const aKey: RawUtf8): double;
var
  v: PGgufValue;
begin
  v := Find(aKey);
  if v = nil then
    EInferError.RaiseUtf8('%: no metadata %', [self, aKey]);
  if v^.VType = gtFloat32 then
    result := PSingle(v^.Data)^
  else if v^.VType = gtFloat64 then
    result := PDouble(v^.Data)^
  else
  begin
    EInferError.RaiseUtf8('%: % is %, expected a float',
      [self, aKey, GGUF_TYPE_NAME[v^.VType]]);
    result := 0;
  end;
end;

function TGgufFile.Bool(const aKey: RawUtf8): boolean;
begin
  result := Value(aKey, gtBool)^.Data^ <> #0;
end;

function TGgufFile.BoolDef(const aKey: RawUtf8; aDefault: boolean): boolean;
begin
  if Has(aKey) then
    result := Bool(aKey)
  else
    result := aDefault;
end;

function TGgufFile.Strings(const aKey: RawUtf8; aCount: Int64): TRawUtf8DynArray;
var
  v: PGgufValue;
  p: PAnsiChar;
  n: Int64;
  i: PtrInt;
begin
  v := ArrayOf(aKey, gtString, aCount);
  if v^.Count > GGUF_MAX_ITEMS then
    EInferError.RaiseUtf8('%: % has % strings - too many', [self, aKey, v^.Count]);
  SetLength(result, v^.Count);
  p := v^.Data;
  for i := 0 to PtrInt(v^.Count) - 1 do
  begin
    MoveFast(p^, n, 8); // bounds checked by Parse
    FastSetString(result[i], p + 8, n);
    inc(p, 8 + PtrInt(n));
  end;
end;

function TGgufFile.Floats32(const aKey: RawUtf8; var aCount: Int64): PSingleArray;
var
  v: PGgufValue;
begin
  v := ArrayOf(aKey, gtFloat32, aCount);
  aCount := v^.Count;
  result := pointer(v^.Data);
end;

function TGgufFile.Ints32(const aKey: RawUtf8; var aCount: Int64): PIntegerArray;
var
  v: PGgufValue;
begin
  v := ArrayOf(aKey, gtInt32, aCount);
  aCount := v^.Count;
  result := pointer(v^.Data);
end;

function TGgufFile.Bytes(const aKey: RawUtf8): RawByteString;
var
  v: PGgufValue;
begin
  v := Value(aKey, gtArray);
  if not (v^.ElemType in [gtUInt8, gtInt8]) then
    EInferError.RaiseUtf8('%: % is an array of %, expected bytes',
      [self, aKey, GGUF_TYPE_NAME[v^.ElemType]]);
  FastSetRawByteString(result, v^.Data, v^.Count);
end;

function TGgufFile.FindTensor(const aName: RawUtf8): PInferTensor;
var
  i: PtrInt;
begin
  i := SortedFind(fNames, fNameIndex, aName);
  if i < 0 then
    result := nil
  else
    result := @fTensors[i];
end;

function TGgufFile.Tensor(const aName: RawUtf8): PInferTensor;
begin
  result := FindTensor(aName);
  if result = nil then
    EInferError.RaiseUtf8('%: no tensor %', [self, aName]);
end;

function TGgufFile.Weight(const aName: RawUtf8; aType: TInferDType;
  const aDims: array of Int64): pointer;
var
  t: PInferTensor;
  dims: TInt64DynArray;
  i: PtrInt;
begin
  t := Tensor(aName);
  if t^.DType <> aType then
    EInferError.RaiseUtf8('%: % is %, expected %',
      [self, aName, INFER_DTYPE_NAME[t^.DType], INFER_DTYPE_NAME[aType]]);
  if not InferShapeIs(t^, aDims) then
  begin
    SetLength(dims, length(aDims));
    for i := 0 to high(aDims) do
      dims[i] := aDims[i];
    EInferError.RaiseUtf8('%: % has shape %, expected %',
      [self, aName, InferShapeText(t^.Shape), InferShapeText(dims)]);
  end;
  result := t^.Data;
end;

end.
