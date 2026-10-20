/// native inference engine - safetensors reader
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room from the public safetensors format description: an 8-byte
//   little-endian header length, a JSON header naming every tensor with its
//   dtype, shape and [begin, end) byte range relative to the data section, then
//   the data section itself
// - the whole header is validated before any tensor is handed out: dtype, shape
//   product (overflow-safe), byte length, ranges inside the data section and not
//   overlapping each other and covering it without gaps or trailing bytes,
//   unique names, strict JSON integers, __metadata__ as a string map - a damaged
//   or hostile file is refused with EInferError, never read past its end, and
//   validation stays O(n log n) in the number of tensors
// - a checkpoint split into several files (model-00001-of-00003.safetensors...)
//   is opened through its model.safetensors.index.json: every file is
//   validated as above, and the index must name exactly the tensors each file
//   holds - one view over all of them (OpenFolder, CreateFromIndex)
unit mormot.ai.infer.loader.safetensors;

{
  *****************************************************************************

    - TSafeTensorsFile: a validated, memory-mapped safetensors file, or the
      files of a checkpoint split by model.safetensors.index.json

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
  mormot.core.data,     // TDocVariantOption
  mormot.core.variants,
  mormot.core.json,
  mormot.ai.infer.tensor,
  mormot.ai.infer.memory;

const
  /// the largest JSON header accepted (the format itself caps it at 100 MB)
  SAFETENSORS_MAX_HEADER = 100 shl 20;

  /// the largest model.safetensors.index.json accepted
  SAFETENSORS_MAX_INDEX = 64 shl 20;

  /// the most files an index may name
  SAFETENSORS_MAX_PARTS = 4096;

type
  /// a validated safetensors file, mapped read-only - or the files of a split
  // checkpoint as one (CreateFromIndex)
  TSafeTensorsFile = class
  protected
    fFile: TInferMappedFile;
    fParts: array of TSafeTensorsFile; // CreateFromIndex: one per file
    fTensors: TInferTensorDynArray;
    fNames: TRawUtf8DynArray;
    fNameIndex: TIntegerDynArray;
    fMetadata: TDocVariantData;
    procedure Parse;
    /// fNames and fNameIndex from fTensors
    procedure IndexNames;
  public
    /// map and validate aFileName - raises EInferError on any defect
    constructor Create(const aFileName: TFileName);
    /// validate an in-memory content - for tests and small fixtures
    constructor CreateFromContent(const aContent: RawByteString);
    /// map and validate every file of a model.safetensors.index.json (its
    // "weight_map" from tensor name to file name, the files beside it)
    // - raises EInferError when a file is not a plain name in the same folder,
    // a file fails Create, a tensor is mapped twice, or the map and the files
    // differ by a single tensor (missing, extra, or in another file)
    // - Metadata stays empty (the index metadata is no string map)
    constructor CreateFromIndex(const aIndexFile: TFileName);
    /// <aFolder>model.safetensors if it exists, else the files of
    // <aFolder>model.safetensors.index.json - the order transformers uses
    class function OpenFolder(const aFolder: TFileName): TSafeTensorsFile;
    /// unmap the file(s)
    destructor Destroy; override;
    /// the tensor named aName, or nil
    function Find(const aName: RawUtf8): PInferTensor;
    /// the tensor named aName - raises EInferError if missing
    function Tensor(const aName: RawUtf8): PInferTensor;
    /// the F32 elements of aName, checked against the expected dimensions
    // - raises EInferError if missing, not F32, or shaped otherwise
    function F32(const aName: RawUtf8; const aDims: array of Int64): PSingleArray;
    /// the optional "__metadata__" object (string values)
    function Metadata: PDocVariantData;
    /// all tensors, sorted by their position in the data section
    property Tensors: TInferTensorDynArray
      read fTensors;
  end;


implementation

{ TSafeTensorsFile }

constructor TSafeTensorsFile.Create(const aFileName: TFileName);
begin
  inherited Create;
  fFile := TInferMappedFile.Create(aFileName);
  Parse;
end;

constructor TSafeTensorsFile.CreateFromContent(const aContent: RawByteString);
begin
  inherited Create;
  fFile := TInferMappedFile.CreateFromContent(aContent);
  Parse;
end;

destructor TSafeTensorsFile.Destroy;
var
  i: PtrInt;
begin
  for i := 0 to high(fParts) do
    fParts[i].Free;
  fFile.Free;
  inherited Destroy;
end;

class function TSafeTensorsFile.OpenFolder(const aFolder: TFileName): TSafeTensorsFile;
var
  dir: TFileName;
begin
  dir := IncludeTrailingPathDelimiter(aFolder);
  if FileExists(dir + 'model.safetensors') or
     not FileExists(dir + 'model.safetensors.index.json') then
    result := TSafeTensorsFile.Create(dir + 'model.safetensors') // or raises
  else
    result := TSafeTensorsFile.CreateFromIndex(dir + 'model.safetensors.index.json');
end;

/// a name of a file beside the index: no folder part, nothing hidden or relative
function PlainFileName(const aName: RawUtf8): boolean;
var
  i: PtrInt;
begin
  result := false;
  if (aName = '') or (aName[1] = '.') or
     (StrLen(pointer(aName)) <> length(aName)) then
    exit;
  for i := 1 to length(aName) do
    if aName[i] in ['/', '\', ':', #0..#31] then
      exit;
  result := true;
end;

/// the index of aValue in the sorted aList, -1 if absent
function FindSorted(const aList: TRawUtf8DynArray; const aValue: RawUtf8): PtrInt;
var
  lo, hi, c: PtrInt;
begin
  lo := 0;
  hi := high(aList);
  while lo <= hi do
  begin
    result := (lo + hi) shr 1;
    c := StrComp(pointer(aList[result]), pointer(aValue));
    if c = 0 then
      exit
    else if c < 0 then
      lo := result + 1
    else
      hi := result - 1;
  end;
  result := -1;
end;

constructor TSafeTensorsFile.CreateFromIndex(const aIndexFile: TFileName);
var
  json: RawUtf8;
  doc: TDocVariantData;
  map: PDocVariantData;
  names, files, parts: TRawUtf8DynArray;
  idx: TIntegerDynArray;
  i, j, n, k, c, total: PtrInt;
  dir: TFileName;
  t: PInferTensor;
begin
  inherited Create;
  // the size first: a huge file is refused before it is read
  if FileSize(aIndexFile) > SAFETENSORS_MAX_INDEX then
    EInferError.RaiseUtf8('%: % is larger than % bytes',
      [self, aIndexFile, SAFETENSORS_MAX_INDEX]);
  json := StringFromFile(aIndexFile);
  if json = '' then
    EInferError.RaiseUtf8('%: cannot read %', [self, aIndexFile]);
  // an empty JSON object or array has no kind in mORMot: an array refused,
  // an empty map counted below
  if not doc.InitJson(json, JSON_FAST + [dvoNameCaseSensitive]) or
     (doc.Kind <> dvObject) then
    EInferError.RaiseUtf8('%: % is no JSON object', [self, aIndexFile]);
  // a key twice at the top: mORMot would take the first weight_map, Python
  // the last - refused, as in the tensor names below
  names := copy(doc.Names, 0, doc.Count);
  QuickSortRawUtf8(names, doc.Count);
  for i := 1 to doc.Count - 1 do
    if names[i] = names[i - 1] then
      EInferError.RaiseUtf8('%: % has "%" twice', [self, aIndexFile, names[i]]);
  if not _Safe(doc.Value['weight_map'], map) or (map^.Kind = dvArray) then
    EInferError.RaiseUtf8('%: % has no "weight_map" object', [self, aIndexFile]);
  n := map^.Count;
  if n = 0 then
    EInferError.RaiseUtf8('%: % maps no tensor', [self, aIndexFile]);
  // the tensor names sorted (a duplicated JSON key stays twice in the
  // document: refused) with the file of each
  names := copy(map^.Names, 0, n);
  SetLength(idx, n);
  for i := 0 to n - 1 do
    idx[i] := i;
  QuickSortRawUtf8(names, n, @idx);
  SetLength(files, n);
  for i := 0 to n - 1 do
  begin
    if (i > 0) and (names[i] = names[i - 1]) then
      EInferError.RaiseUtf8('%: % maps % twice', [self, aIndexFile, names[i]]);
    if TVarData(map^.Values[idx[i]]).VType <> varString then // mORMot JSON text
      EInferError.RaiseUtf8('%: the file of % is no string', [self, names[i]]);
    files[i] := VariantToUtf8(map^.Values[idx[i]]);
    if not PlainFileName(files[i]) then
      EInferError.RaiseUtf8('%: "%" is no file name beside the index',
        [self, files[i]]);
  end;
  // the distinct files, sorted
  parts := copy(files, 0, n);
  QuickSortRawUtf8(parts, n);
  j := 0;
  for i := 0 to n - 1 do
    if (j = 0) or (parts[i] <> parts[j - 1]) then
    begin
      parts[j] := parts[i];
      inc(j);
    end;
  SetLength(parts, j);
  if j > SAFETENSORS_MAX_PARTS then
    EInferError.RaiseUtf8('%: % names % files', [self, aIndexFile, j]);
  // every file mapped and validated, each of its tensors where the map says
  dir := ExtractFilePath(aIndexFile);
  SetLength(fParts, length(parts));
  total := 0;
  for i := 0 to high(parts) do
  begin
    fParts[i] := TSafeTensorsFile.Create(dir + Utf8ToString(parts[i]));
    t := pointer(fParts[i].fTensors);
    for j := 1 to length(fParts[i].fTensors) do
    begin
      k := FindSorted(names, t^.Name);
      if k < 0 then
        EInferError.RaiseUtf8('%: % of % is not in the index', [self, t^.Name, parts[i]]);
      if files[k] <> parts[i] then
        EInferError.RaiseUtf8('%: % is in %, the index says %',
          [self, t^.Name, parts[i], files[k]]);
      inc(total);
      inc(t);
    end;
  end;
  // and every tensor of the map in its file: the two sets are the same (a
  // name in two files cannot pass, the map names one file per tensor)
  for i := 0 to n - 1 do
    if fParts[FindSorted(parts, files[i])].Find(names[i]) = nil then
      EInferError.RaiseUtf8('%: % is not in %, where the index places it',
        [self, names[i], files[i]]);
  SetLength(fTensors, total);
  c := 0;
  for i := 0 to high(fParts) do
    for j := 0 to high(fParts[i].fTensors) do
    begin
      fTensors[c] := fParts[i].fTensors[j];
      inc(c);
    end;
  IndexNames;
end;

procedure TSafeTensorsFile.IndexNames;
var
  i, n: PtrInt;
begin
  // name lookup by binary search over a sorted copy
  n := length(fTensors);
  SetLength(fNames, n);
  SetLength(fNameIndex, n);
  for i := 0 to n - 1 do
  begin
    fNames[i] := fTensors[i].Name;
    fNameIndex[i] := i;
  end;
  QuickSortRawUtf8(fNames, n, @fNameIndex);
end;

/// true when every number outside the strings of aJson is a plain integer
// (digits, optional leading minus) and the text starts with an object
// - the variant types after parsing cannot tell: mORMot stores 1e0 and 0.0 as
// integers and 1e5 as text, so the header text itself is scanned
// - true and false are refused too (they contain an e) - no valid header holds
// them; a nested null passes here and is refused as a non-integer later
function HeaderLexOk(p: PUtf8Char; aLen: PtrInt): boolean;
var
  i: PtrInt;
  c: AnsiChar;
  inString: boolean;
begin
  result := false;
  i := 0;
  while (i < aLen) and (p[i] in [' ', #9, #10, #13]) do
    inc(i);
  if (i >= aLen) or (p[i] <> '{') then
    exit; // null, arrays, scalars: not an object
  inString := false;
  while i < aLen do
  begin
    c := p[i];
    if inString then
    begin
      if c = '\' then
        inc(i) // skip the escaped character
      else if c = '"' then
        inString := false;
    end
    else if c = '"' then
      inString := true
    else if c in ['.', 'e', 'E', '+'] then
      exit // a fraction or an exponent outside a string
    else if (c = '0') and (i + 1 < aLen) and (p[i + 1] in ['0'..'9']) and
            ((i = 0) or not (p[i - 1] in ['0'..'9'])) then
      exit; // a leading zero: not a JSON number, mORMot would keep it as text
    inc(i);
  end;
  result := true;
end;

/// a JSON integer, refusing null, booleans, floats and strings - VariantToInt64
// accepts some of those
function JsonInt(const aValue: variant; out aInt: Int64): boolean;
begin
  case TVarData(aValue).VType of
    varSmallint, varInteger, varShortInt, varByte, varWord, varLongWord, varInt64:
      result := VariantToInt64(aValue, aInt);
  else
    result := false;
  end;
end;

procedure TSafeTensorsFile.Parse;
var
  headerLen, dataStart, dataLen, count, prevEnd, b, e, v: Int64;
  json: RawUtf8;
  header: TDocVariantData;
  meta, item, shape, offsets: PDocVariantData;
  i, j, n: PtrInt;
  t: PInferTensor;
  dtypeName: RawUtf8;
  keys, index: TInt64DynArray;
  sorted: TInferTensorDynArray;
  names: TRawUtf8DynArray;
begin
  if fFile.Size < 8 then
    EInferError.RaiseUtf8('%: % bytes - too short for a header', [self, fFile.Size]);
  MoveFast(fFile.Buffer^, headerLen, 8); // little-endian on every target we build
  if (headerLen < 2) or (headerLen > SAFETENSORS_MAX_HEADER) or
     (headerLen > fFile.Size - 8) then
    EInferError.RaiseUtf8('%: invalid header length %', [self, headerLen]);
  FastSetString(json, fFile.Buffer + 8, headerLen);
  // the whole header must be one JSON object: no NUL, nothing after the closing
  // brace but the space padding writers add
  // names are matched case-sensitively, as the format's own parser does
  if (StrLen(pointer(json)) <> length(json)) or
     not HeaderLexOk(pointer(json), length(json)) or not IsValidJson(json, true) or
     not header.InitJson(json, JSON_FAST + [dvoNameCaseSensitive]) or
     (header.Kind <> dvObject) then
    EInferError.RaiseUtf8('%: header is not one JSON object', [self]);
  // unique keys, __metadata__ included: sorted copy, neighbours compared
  names := copy(header.Names, 0, header.Count);
  QuickSortRawUtf8(names, header.Count);
  for i := 1 to header.Count - 1 do
    if names[i] = names[i - 1] then
      EInferError.RaiseUtf8('%: key "%" defined twice', [self, names[i]]);
  dataStart := 8 + headerLen;
  dataLen := fFile.Size - dataStart;
  SetLength(fTensors, header.Count);
  n := 0;
  for i := 0 to header.Count - 1 do
  begin
    if header.Names[i] = '__metadata__' then
    begin
      // a string-to-string map, as the format defines it
      meta := _Safe(header.Values[i]);
      if meta^.Kind <> dvObject then
        EInferError.RaiseUtf8('%: __metadata__ is not an object', [self]);
      for j := 0 to meta^.Count - 1 do
        if TVarData(meta^.Values[j]).VType <> varString then // mORMot JSON text
          EInferError.RaiseUtf8('%: __metadata__.% is not a string',
            [self, meta^.Names[j]]);
      fMetadata.InitCopy(header.Values[i], JSON_FAST);
      continue;
    end;
    item := _Safe(header.Values[i]);
    t := @fTensors[n];
    t^.Name := header.Names[i];
    if t^.Name = '' then
      EInferError.RaiseUtf8('%: tensor without a name', [self]);
    // exactly dtype, shape and data_offsets: a duplicated or unknown key inside
    // a tensor entry is refused (the JSON check above is a sanity check only)
    if (item^.Kind <> dvObject) or (item^.Count <> 3) or
       (item^.GetValueIndex('dtype') < 0) or (item^.GetValueIndex('shape') < 0) or
       (item^.GetValueIndex('data_offsets') < 0) then
      EInferError.RaiseUtf8('%: % is not {dtype, shape, data_offsets}',
        [self, t^.Name]);
    if StrLen(pointer(t^.Name)) <> length(t^.Name) then
      EInferError.RaiseUtf8('%: a tensor name contains NUL', [self]);
    dtypeName := item^.U['dtype'];
    t^.DType := InferDTypeFromName(dtypeName);
    if t^.DType = idtUnknown then
      EInferError.RaiseUtf8('%: % has unsupported dtype "%"', [self, t^.Name, dtypeName]);
    // a scalar has shape []: mORMot parses an empty JSON array as an empty
    // document of undefined kind, which GetAsArray does not return
    j := item^.GetValueIndex('shape');
    if j < 0 then
      EInferError.RaiseUtf8('%: % has no shape', [self, t^.Name]);
    if not _Safe(item^.Values[j], shape) or (shape^.Kind = dvObject) then
      EInferError.RaiseUtf8('%: % shape is not an array', [self, t^.Name]);
    SetLength(t^.Shape, shape^.Count);
    for j := 0 to shape^.Count - 1 do
      if not JsonInt(shape^.Values[j], t^.Shape[j]) then
        EInferError.RaiseUtf8('%: % has a non-integer dimension', [self, t^.Name]);
    if not InferShapeCount(t^.Shape, dataLen, count) then
      EInferError.RaiseUtf8('%: % shape % exceeds the file',
        [self, t^.Name, InferShapeText(t^.Shape)]);
    t^.Count := count;
    t^.Bytes := count * INFER_DTYPE_SIZE[t^.DType]; // count <= dataLen: no overflow
    if not item^.GetAsArray('data_offsets', offsets) or (offsets^.Count <> 2) or
       not JsonInt(offsets^.Values[0], b) or not JsonInt(offsets^.Values[1], e) then
      EInferError.RaiseUtf8('%: % has no [begin, end] data_offsets', [self, t^.Name]);
    if (b < 0) or (e < b) or (e > dataLen) then
      EInferError.RaiseUtf8('%: % range [%,%) outside the % data bytes',
        [self, t^.Name, b, e, dataLen]);
    if e - b <> t^.Bytes then
      EInferError.RaiseUtf8('%: % has % bytes for % % elements',
        [self, t^.Name, e - b, count, dtypeName]);
    // elements aligned to their size, counted from the file start (writers pad
    // the header for this) - stricter than the format, by design: the kernels
    // read whole elements; wider SIMD alignment is not assumed
    if (dataStart + b) mod INFER_DTYPE_SIZE[t^.DType] <> 0 then
      EInferError.RaiseUtf8('%: % is not aligned to its % bytes elements',
        [self, t^.Name, INFER_DTYPE_SIZE[t^.DType]]);
    t^.Data := fFile.Buffer + PtrInt(dataStart + b); // < file size: fits PtrInt
    inc(n);
  end;
  // order by (start, size > 0): an empty tensor sorts before a filled one at the
  // same start; O(n log n), a hostile header with millions of entries stays fast
  SetLength(keys, n);
  SetLength(index, n);
  for i := 0 to n - 1 do
  begin
    keys[i] := (PAnsiChar(fTensors[i].Data) - (fFile.Buffer + PtrInt(dataStart))) * 2 +
      ord(fTensors[i].Bytes > 0);
    index[i] := i;
  end;
  if n > 1 then
    QuickSortInt64(pointer(keys), pointer(index), 0, n - 1);
  SetLength(sorted, n);
  for i := 0 to n - 1 do
    sorted[i] := fTensors[index[i]];
  fTensors := sorted;
  // the format requires the data section to be covered exactly: no overlap,
  // no gap, no unreferenced bytes at its end
  prevEnd := 0;
  for i := 0 to n - 1 do
  begin
    v := PAnsiChar(fTensors[i].Data) - (fFile.Buffer + PtrInt(dataStart));
    if v < prevEnd then
      EInferError.RaiseUtf8('%: % overlaps the tensor before it',
        [self, fTensors[i].Name]);
    if v > prevEnd then
      EInferError.RaiseUtf8('%: gap of % bytes before %',
        [self, v - prevEnd, fTensors[i].Name]);
    prevEnd := v + fTensors[i].Bytes;
  end;
  if prevEnd <> dataLen then
    EInferError.RaiseUtf8('%: % unreferenced bytes after the last tensor',
      [self, dataLen - prevEnd]);
  IndexNames;
end;

function TSafeTensorsFile.Find(const aName: RawUtf8): PInferTensor;
var
  lo, hi, mid, c: PtrInt;
begin
  lo := 0;
  hi := high(fNames);
  while lo <= hi do
  begin
    mid := (lo + hi) shr 1;
    c := StrComp(pointer(fNames[mid]), pointer(aName));
    if c = 0 then
    begin
      result := @fTensors[fNameIndex[mid]];
      exit;
    end
    else if c < 0 then
      lo := mid + 1
    else
      hi := mid - 1;
  end;
  result := nil;
end;

function TSafeTensorsFile.Tensor(const aName: RawUtf8): PInferTensor;
begin
  result := Find(aName);
  if result = nil then
    EInferError.RaiseUtf8('%: no tensor %', [self, aName]);
end;

function TSafeTensorsFile.F32(const aName: RawUtf8;
  const aDims: array of Int64): PSingleArray;
var
  t: PInferTensor;
  dims: TInt64DynArray;
  i: PtrInt;
begin
  t := Tensor(aName);
  if t^.DType <> idtF32 then
    EInferError.RaiseUtf8('%: % is %, expected F32',
      [self, aName, INFER_DTYPE_NAME[t^.DType]]);
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

function TSafeTensorsFile.Metadata: PDocVariantData;
begin
  result := @fMetadata;
end;

end.
