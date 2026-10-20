/// tests for the native inference engine itself (src/infer)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - phase 1a: the safetensors reader (real fixture + crafted damaged files),
//   erf, and every CPU backend operation against tests/data/infer-ref/ops
unit test.infer.engine;

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
  mormot.ai.infer.tensor,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  test.infer.ref;

type
  /// phase 1a: loader and CPU operations
  TTestInferEngine = class(TSynTestCase)
  protected
    fCpu: TInferCpuBackend;
    fOps: TInferRefSet;
    procedure Setup; override;
    procedure CleanUp; override;
    /// compare aCount singles against the ops tensor aName
    procedure CheckOp(const aName: RawUtf8; aActual: PSingleArray;
      aAtol, aRtol: double);
  published
    /// the tiny fixture's weights load, with the shapes the manifest lists
    procedure SafeTensorsReadsFixture;
    /// damaged headers and ranges are refused before any read
    procedure SafeTensorsRefusesDamage;
    /// a checkpoint in several files with model.safetensors.index.json: one
    // view over all of them, every mismatch between map and files refused
    procedure SafeTensorsIndex;
    /// erf/erfc at known values, odd symmetry, no cancellation in the tail
    procedure ErfIsAccurate;
    /// Linear, LayerNorm, Gelu, SoftmaxMasked, L2Normalize vs the ops fixtures
    procedure CpuOperationsMatchFixtures;
    /// Attention and Add against hand-computed values, and the edge contracts:
    // softmax without mask, a fully masked row, a zero vector
    procedure CpuOperationsKeepTheirContract;
  end;


implementation

{ helpers }

/// a safetensors content from a JSON header and a data section
// - the header is padded with spaces to a multiple of 8, as writers do, unless
// aUnpadded is set
function SafeTensors(const aHeader: RawUtf8; const aData: RawByteString;
  aUnpadded: boolean = false): RawByteString;
var
  len: Int64;
  header: RawUtf8;
begin
  header := aHeader;
  if not aUnpadded then
    while length(header) and 7 <> 0 do
      header := header + ' ';
  len := length(header);
  SetLength(result, 8);
  MoveFast(len, pointer(result)^, 8);
  result := result + header + aData;
end;

/// n singles 1, 2, 3 ... as raw bytes
function Floats(n: integer): RawByteString;
var
  i: integer;
  f: single;
begin
  SetLength(result, n * 4);
  for i := 0 to n - 1 do
  begin
    f := i + 1;
    MoveFast(f, PByteArray(result)^[i * 4], 4);
  end;
end;

/// true if TSafeTensorsFile refuses aContent with an EInferError naming aReason
// - any other exception class is a failure too: the loader promises EInferError
function Refused(const aContent: RawByteString; const aReason: RawUtf8): boolean;
var
  f: TSafeTensorsFile;
begin
  result := false;
  try
    f := TSafeTensorsFile.CreateFromContent(aContent);
    f.Free;
  except
    on E: Exception do
      result := (E is EInferError) and
        (PosEx(aReason, StringToUtf8(E.Message)) > 0);
  end;
end;

/// true if TSafeTensorsFile accepts aContent
function Accepted(const aContent: RawByteString): boolean;
var
  f: TSafeTensorsFile;
begin
  result := false;
  try
    f := TSafeTensorsFile.CreateFromContent(aContent);
    f.Free;
    result := true;
  except
    on Exception do
      result := false;
  end;
end;


{ TTestInferEngine }

procedure TTestInferEngine.Setup;
var
  dir: TFileName;
begin
  fCpu := TInferCpuBackend.Create;
  dir := FindInferRefDir('ops');
  if dir <> '' then
    fOps := TInferRefSet.CreateFromFolder(dir);
end;

procedure TTestInferEngine.CleanUp;
begin
  // re-entrant, as mORMot requires
  FreeAndNil(fOps);
  FreeAndNil(fCpu);
end;

procedure TTestInferEngine.CheckOp(const aName: RawUtf8; aActual: PSingleArray;
  aAtol, aRtol: double);
var
  ref: PSingleArray;
  n: PtrInt;
  c: TInferCompare;
begin
  ref := fOps.F32(aName, n);
  c := InferCompare(aActual, ref, n, aAtol, aRtol);
  CheckUtf8(InferCompareOk(c), '%: %', [aName, InferCompareText(c)]);
end;

procedure TTestInferEngine.SafeTensorsReadsFixture;
var
  dir: TFileName;
  f: TSafeTensorsFile;
  cfg: TDocVariantData;
  list: PDocVariantData;
  i: PtrInt;
  hidden, vocab: Int64;
  w: PSingleArray;
  t: PInferTensor;
  ref: TInferRefSet;
  name: RawUtf8;
  weights, pair: PDocVariantData;
  j: PtrInt;
  shapeOk: boolean;
begin
  dir := FindInferRefDir('tiny-xlmr');
  if CheckFailed(dir <> '', 'tiny-xlmr found') then
    exit;
  cfg.InitJsonFromFile(dir + 'config.json', JSON_FAST);
  hidden := cfg.I['hidden_size'];
  vocab := cfg.I['vocab_size'];
  f := TSafeTensorsFile.Create(dir + 'model.safetensors');
  try
    CheckEqual(f.Metadata^.U['format'], 'pt', 'metadata read');
    // every tensor the generator listed: [name, shape] pairs
    ref := TInferRefSet.CreateFromFolder(dir);
    try
      if CheckFailed(ref.Manifest^.GetAsObject('weights', weights) and
         weights^.GetAsArray('tensors', list), 'manifest lists the weights') then
        exit;
      CheckEqual(length(f.Tensors), list^.Count, 'tensor count');
      for i := 0 to list^.Count - 1 do
      begin
        pair := _Safe(list^.Values[i]);
        VariantToUtf8(pair^.Values[0], name);
        t := f.Find(name);
        if CheckFailed(t <> nil, Utf8ToString('tensor ' + name + ' present')) then
          continue;
        shapeOk := (t^.DType = idtF32) and
          (length(t^.Shape) = _Safe(pair^.Values[1])^.Count);
        for j := 0 to length(t^.Shape) - 1 do
          shapeOk := shapeOk and
            (t^.Shape[j] = _Safe(pair^.Values[1])^.I[UInt32ToUtf8(j)]);
        CheckUtf8(shapeOk, '% is F32 % as listed', [name, InferShapeText(t^.Shape)]);
      end;
    finally
      ref.Free;
    end;
    w := f.F32('embeddings.word_embeddings.weight', [vocab, hidden]);
    Check(w <> nil);
    // a wrong expectation is an error, not a silently reinterpreted buffer
    try
      f.F32('embeddings.word_embeddings.weight', [hidden, vocab]);
      Check(false, 'transposed shape accepted');
    except
      on EInferError do
        Check(true);
    end;
    Check(f.Find('no.such.tensor') = nil, 'missing tensor is nil');
  finally
    f.Free;
  end;
end;

procedure TTestInferEngine.SafeTensorsRefusesDamage;
const
  ONE: RawUtf8 = '{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}';
var
  ok: TSafeTensorsFile;
begin
  ok := TSafeTensorsFile.CreateFromContent(SafeTensors(ONE, Floats(6)));
  try
    CheckEqual(length(ok.Tensors), 1, 'well-formed content accepted');
    CheckUtf8(ok.F32('a', [2, 3])^[5] = 6, 'data read at its offset');
  finally
    ok.Free;
  end;
  Check(Refused('short', 'too short'), 'no header length');
  // a missing data section is caught by the shape bound before the range check
  Check(Refused(SafeTensors(ONE, ''), 'exceeds the file'), 'data section missing');
  Check(Refused(SafeTensors(ONE + ' ', Floats(6), true), 'not aligned'),
    'misaligned F32 data refused');
  Check(Refused(SafeTensors('[1,2]', ''), 'JSON object'), 'header not an object');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F99","shape":[1],"data_offsets":[0,4]}}', Floats(1)),
    'unsupported dtype'), 'unknown dtype');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,12]}}', Floats(4)),
    'bytes for'), 'length does not match the shape');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[1099511627776,1099511627776],' +
    '"data_offsets":[0,4]}}', Floats(1)), 'exceeds the file'), 'overflowing shape');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}', Floats(2)),
    'outside'), 'range past the end');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},' +
    '"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}', Floats(3)),
    'overlaps'), 'overlapping tensors');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},' +
    '"a":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}', Floats(4)),
    'twice'), 'duplicated name');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}', Floats(2)),
    'gap'), 'gap before the first tensor');
  Check(Refused(SafeTensors(ONE, Floats(7)), 'unreferenced'), 'trailing bytes');
  // valid edge cases of the format
  Check(Accepted(SafeTensors(
    '{"e":{"dtype":"F32","shape":[0,3],"data_offsets":[0,0]}}', '')),
    'empty tensor with a non-zero dimension');
  Check(Accepted(SafeTensors(
    '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},' +
    '"e":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}', Floats(1))),
    'empty tensor at the start of a filled one, listed after it');
  Check(Accepted(SafeTensors(
    '{"s":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}', Floats(1))),
    'scalar: shape []');
  Check(Accepted(SafeTensors(
    '{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[1],' +
    '"data_offsets":[0,4]}}', Floats(1))), 'string metadata');
  // more damage
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","data_offsets":[0,4]}}', Floats(1)), 'is not {dtype'),
    'missing shape');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[1],"shape":[1],"data_offsets":[0,4]}}',
    Floats(1)), 'is not {dtype'), 'key twice inside a tensor entry');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","Shape":[1],"data_offsets":[0,4]}}', Floats(1)),
    'is not {dtype'), 'key with other case');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"x":1}}', Floats(1)),
    'is not {dtype'), 'unknown key inside a tensor entry');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[1]}}', Floats(1)), 'data_offsets'),
    'missing data_offsets');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[null],"data_offsets":[0,4]}}', Floats(1)),
    'non-integer'), 'null dimension');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[true],"data_offsets":[0,4]}}', Floats(1)),
    'one JSON object'), 'boolean dimension (refused by the header scan)');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[1],"data_offsets":[null,4]}}', Floats(1)),
    'data_offsets'), 'null offset');
  Check(Refused(SafeTensors(ONE + 'x', Floats(6)), 'one JSON object'),
    'text after the closing brace');
  Check(Refused(SafeTensors(ONE + #0, Floats(6)), 'one JSON object'),
    'NUL inside the header');
  Check(Refused(SafeTensors('{"__metadata__":"x",' + copy(ONE, 2, 999), Floats(6)),
    'not an object'), 'metadata not an object');
  Check(Refused(SafeTensors('{"__metadata__":{"n":1},' + copy(ONE, 2, 999),
    Floats(6)), 'not a string'), 'metadata value not a string');
  Check(Refused(SafeTensors('{"__metadata__":{},"__metadata__":{},' +
    copy(ONE, 2, 999), Floats(6)), 'twice'), 'metadata twice');
  Check(Refused(#1#0#0#0#0#0#0#0'{', 'header length'), 'header length below 2');
  // the header text itself: integers only, an object only
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[1e0],"data_offsets":[0,4]}}', Floats(1)),
    'one JSON object'), 'exponent as a dimension');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0.0,4]}}', Floats(1)),
    'one JSON object'), 'fraction as an offset');
  Check(Refused(SafeTensors('{"__metadata__":{"n":1e5}}', ''), 'one JSON object'),
    'numeric metadata written as an exponent');
  Check(Refused(SafeTensors('null', ''), 'one JSON object'), 'null header');
  Check(Refused(SafeTensors('{"__metadata__":{"x":01},' + copy(ONE, 2, 999),
    Floats(6)), 'one JSON object'), 'leading zero in a metadata number');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[01],"data_offsets":[0,4]}}', Floats(1)),
    'one JSON object'), 'leading zero in a dimension');
  Check(Accepted(SafeTensors(
    '{"a":{"dtype":"F32","shape":[10],"data_offsets":[0,40]}}', Floats(10))),
    'a 0 inside a number is no leading zero');
  Check(Refused(SafeTensors(
    '{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}', Floats(1)),
    'exceeds the file'), 'negative dimension');
end;

type
  TErfPoint = record
    x, erf, erfc: double;
  end;

const
  /// erf and erfc by mpmath at 40 digits (tools: mpmath 1.3.0), across both
  // algorithm ranges and around the switch at 2.5
  ERF_POINTS: array[0..22] of TErfPoint = (
    (x: -10; erf: -1.0; erfc: 2.0),
    (x: -6; erf: -0.9999999999999999784803; erfc: 1.99999999999999997848),
    (x: -4; erf: -0.99999998458274209972; erfc: 1.99999998458274209972),
    (x: -2.6; erf: -0.9997639655834706509122; erfc: 1.999763965583470650912),
    (x: -2.5; erf: -0.9995930479825550410604; erfc: 1.99959304798255504106),
    (x: -2.4999; erf: -0.9995928300996665486189; erfc: 1.999592830099666548619),
    (x: -1; erf: -0.8427007929497148693412; erfc: 1.842700792949714869341),
    (x: -0.3; erf: -0.3286267594591274161896; erfc: 1.32862675945912741619),
    (x: -1e-8; erf: -1.128379167095512559892e-8; erfc: 1.000000011283791670955),
    (x: 0.001; erf: 0.001128378790969236403438; erfc: 0.9988716212090307635966),
    (x: 0.25; erf: 0.2763263901682369329851; erfc: 0.7236736098317630670149),
    (x: 0.75; erf: 0.7111556336535151315989; erfc: 0.2888443663464848684011),
    (x: 1.5; erf: 0.966105146475310727067; erfc: 0.03389485352468927293302),
    (x: 2.0; erf: 0.9953222650189527341621; erfc: 0.004677734981047265837931),
    (x: 2.4; erf: 0.9993114861033549211145; erfc: 0.00068851389664507888555),
    (x: 2.4999; erf: 0.9995928300996665486189; erfc: 0.0004071699003334513811182),
    (x: 2.5; erf: 0.9995930479825550410604; erfc: 0.0004069520174449589395642),
    (x: 2.5001; erf: 0.9995932657565293202594; erfc: 0.0004067342434706797405804),
    (x: 2.6; erf: 0.9997639655834706509122; erfc: 0.0002360344165293490878128),
    (x: 3.5; erf: 0.9999992569016276585873; erfc: 7.430983723414127455237e-7),
    (x: 4.5; erf: 0.9999999998033839558457; erfc: 1.966160441542887476279e-10),
    (x: 6; erf: 0.9999999999999999784803; erfc: 2.151973671249891311659e-17),
    (x: 10; erf: 1.0; erfc: 2.088487583762544757001e-45));

/// true if CreateFromIndex refuses aDir + 'model.safetensors.index.json' with
// an EInferError naming aReason, after writing aIndex there
function IndexRefused(const aDir: TFileName; const aIndex, aReason: RawUtf8): boolean;
var
  f: TSafeTensorsFile;
begin
  result := false;
  FileFromString(aIndex, aDir + 'model.safetensors.index.json');
  try
    f := TSafeTensorsFile.CreateFromIndex(aDir + 'model.safetensors.index.json');
    f.Free;
  except
    on E: Exception do
      result := (E is EInferError) and
        (PosEx(aReason, StringToUtf8(E.Message)) > 0);
  end;
end;

procedure TTestInferEngine.SafeTensorsIndex;
const
  X: RawUtf8 = '{"x":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}';
  Y: RawUtf8 = '{"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}';
  Z: RawUtf8 = '{"z":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}';
  GOOD: RawUtf8 = '{"metadata":{"total_size":28},"weight_map":' +
    '{"x":"a.safetensors","y":"b.safetensors"}}';
var
  dir: TFileName;
  f: TSafeTensorsFile;
begin
  dir := MakePath([GetSystemPath(spTemp), 'lai-index-' +
    Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(dir);
  try
    FileFromString(SafeTensors(X, Floats(6)), dir + 'a.safetensors');
    FileFromString(SafeTensors(Y, Floats(1)), dir + 'b.safetensors');
    FileFromString(GOOD, dir + 'model.safetensors.index.json');
    // the index alone: both files as one, each tensor at its own data
    f := TSafeTensorsFile.OpenFolder(dir);
    try
      CheckEqual(length(f.Tensors), 2, 'tensors of both files');
      CheckUtf8(f.F32('x', [2, 3])^[5] = 6, 'x from a.safetensors');
      CheckUtf8(f.F32('y', [1])^[0] = 1, 'y from b.safetensors');
      Check(f.Find('z') = nil, 'no other tensor');
      CheckEqual(f.Metadata^.Count, 0, 'no metadata of the index');
    finally
      f.Free;
    end;
    // a model.safetensors beside the index wins, as in transformers
    FileFromString(SafeTensors(Z, Floats(2)), dir + 'model.safetensors');
    f := TSafeTensorsFile.OpenFolder(dir);
    try
      Check((length(f.Tensors) = 1) and (f.Find('z') <> nil),
        'model.safetensors before the index');
    finally
      f.Free;
    end;
    DeleteFile(dir + 'model.safetensors');
    // the map
    Check(IndexRefused(dir, '{"metadata":{}}', 'weight_map'), 'no weight_map');
    Check(IndexRefused(dir, '{"weight_map":[1]}', 'weight_map'), 'weight_map an array');
    Check(IndexRefused(dir, '[1]', 'no JSON object'), 'no object');
    Check(IndexRefused(dir, '{"weight_map":{"x":"a.safetensors","y":"b.safetensors"},' +
      '"weight_map":{"x":"b.safetensors","y":"b.safetensors"}}', 'twice'),
      'weight_map twice: mORMot and Python would take different ones');
    Check(IndexRefused(dir, '{"weight_map":{}}', 'no tensor'), 'empty map');
    Check(IndexRefused(dir,
      '{"weight_map":{"x":"a.safetensors","y":"b.safetensors","x":"a.safetensors"}}',
      'twice'), 'a tensor mapped twice');
    Check(IndexRefused(dir, '{"weight_map":{"x":"a.safetensors","y":1}}',
      'no string'), 'a file that is no string');
    // files beside the index only
    Check(IndexRefused(dir, '{"weight_map":{"x":"../a.safetensors",' +
      '"y":"b.safetensors"}}', 'no file name'), 'a relative path');
    Check(IndexRefused(dir, '{"weight_map":{"x":"sub/a.safetensors",' +
      '"y":"b.safetensors"}}', 'no file name'), 'a folder');
    Check(IndexRefused(dir, '{"weight_map":{"x":"sub\\a.safetensors",' +
      '"y":"b.safetensors"}}', 'no file name'), 'a Windows folder');
    Check(IndexRefused(dir, '{"weight_map":{"x":"c:a.safetensors",' +
      '"y":"b.safetensors"}}', 'no file name'), 'a drive');
    Check(IndexRefused(dir, '{"weight_map":{"x":".a.safetensors",' +
      '"y":"b.safetensors"}}', 'no file name'), 'a hidden name');
    Check(IndexRefused(dir, '{"weight_map":{"x":"","y":"b.safetensors"}}',
      'no file name'), 'an empty name');
    // the map against the files
    // a file the map does not name is no part of the checkpoint (as in
    // transformers): never opened
    FileFromString('{"weight_map":{"x":"a.safetensors"}}',
      dir + 'model.safetensors.index.json');
    f := TSafeTensorsFile.OpenFolder(dir);
    try
      Check((length(f.Tensors) = 1) and (f.Find('y') = nil),
        'b.safetensors left out of the map');
    finally
      f.Free;
    end;
    Check(IndexRefused(dir, '{"weight_map":{"x":"a.safetensors",' +
      '"y":"b.safetensors","w":"b.safetensors"}}', 'where the index places it'),
      'a tensor the files lack');
    Check(IndexRefused(dir, '{"weight_map":{"x":"b.safetensors",' +
      '"y":"b.safetensors"}}', 'where the index places it'),
      'x mapped to the other file');
    Check(IndexRefused(dir, '{"weight_map":{"x":"a.safetensors",' +
      '"y":"a.safetensors"}}', 'where the index places it'),
      'y mapped to the other file');
    Check(IndexRefused(dir, '{"weight_map":{"x":"a.safetensors",' +
      '"y":"b.safetensors","v":"c.safetensors"}}', 'missing'), 'a missing file');
    // a tensor in two files: the map names one of them
    FileFromString(SafeTensors(X, Floats(6)), dir + 'c.safetensors');
    Check(IndexRefused(dir, '{"weight_map":{"x":"a.safetensors",' +
      '"y":"b.safetensors","v":"c.safetensors"}}', 'the index says'),
      'x in two files');
    // a tensor of a named file that the map does not name
    FileFromString(SafeTensors(Z, Floats(2)), dir + 'c.safetensors');
    Check(IndexRefused(dir, '{"weight_map":{"x":"a.safetensors",' +
      '"y":"b.safetensors","v":"c.safetensors"}}', 'not in the index'),
      'a tensor the map does not name');
    // a damaged file: the validation of TSafeTensorsFile.Create
    FileFromString(SafeTensors(Y, ''), dir + 'b.safetensors');
    Check(IndexRefused(dir, GOOD, 'exceeds the file'), 'a damaged file');
  finally
    DirectoryDeleteAll(dir);
  end;
end;

procedure TTestInferEngine.ErfIsAccurate;
var
  i: PtrInt;
  worstAbs, worstRel, e, b, lo, hi: double;
begin
  // erf everywhere and erfc below 1: absolute error (erfc = 1 - erf there);
  // erfc from 1 on comes from its own pieces: relative error
  worstAbs := 0;
  worstRel := 0;
  for i := 0 to high(ERF_POINTS) do
  begin
    e := Abs(InferErf(ERF_POINTS[i].x) - ERF_POINTS[i].erf);
    if e > worstAbs then
      worstAbs := e;
    if ERF_POINTS[i].x >= 1 then
    begin
      e := Abs(InferErfc(ERF_POINTS[i].x) / ERF_POINTS[i].erfc - 1);
      if e > worstRel then
        worstRel := e;
    end
    else
    begin
      e := Abs(InferErfc(ERF_POINTS[i].x) - ERF_POINTS[i].erfc);
      if e > worstAbs then
        worstAbs := e;
    end;
  end;
  CheckUtf8(worstAbs <= 2e-15, 'erf/erfc max abs error % <= 2e-15', [worstAbs]);
  CheckUtf8(worstRel <= 1e-14, 'erfc tail max rel error % <= 1e-14', [worstRel]);
  // reference values: erf(0.5), erf(1), erf(2), erf(3), erfc(5)
  CheckUtf8(Abs(InferErf(0.5) - 0.52049987781304653768) < 1e-15, 'erf(0.5) %',
    [InferErf(0.5)]);
  CheckUtf8(Abs(InferErf(1) - 0.84270079294971486934) < 1e-15, 'erf(1) %',
    [InferErf(1)]);
  CheckUtf8(Abs(InferErf(2) - 0.99532226501895273416) < 1e-15, 'erf(2) %',
    [InferErf(2)]);
  CheckUtf8(Abs(InferErf(3) - 0.99997790950300141456) < 1e-15, 'erf(3) %',
    [InferErf(3)]);
  CheckUtf8(Abs(InferErfc(5) / 1.5374597944280348502e-12 - 1) < 1e-12,
    'erfc(5) without cancellation %', [InferErfc(5)]);
  CheckUtf8(InferErf(-1.5) = -InferErf(1.5), 'odd symmetry');
  CheckUtf8(InferErf(0) = 0, 'erf(0) = 0');
  // both sides of every piece boundary agree (1, 1.5 .. 6), relative to erfc
  b := 1;
  while b <= 6 do
  begin
    lo := InferErfc(b - 1e-12);
    hi := InferErfc(b + 1e-12);
    CheckUtf8(Abs(lo / hi - 1) < 1e-10, 'erfc continuous at %: % / %', [b, lo, hi]);
    // erf from its pieces and erfc from its own: they must add up to 1
    CheckUtf8(Abs(InferErf(b - 0.25) + InferErfc(b - 0.25) - 1) < 1e-15,
      'erf + erfc = 1 at %', [b - 0.25]);
    b := b + 0.5;
  end;
end;

procedure TTestInferEngine.CpuOperationsMatchFixtures;
var
  x, w, b, g: PSingleArray;
  keep: PIntegerArray;
  n, i: PtrInt;
  y: TSingleDynArray;
begin
  if CheckFailed(fOps <> nil, 'ops fixtures found') then
    exit;
  // linear: [3,5] x [4,5]^T + [4]
  x := fOps.F32('linear/x', n);
  w := fOps.F32('linear/weight', n);
  b := fOps.F32('linear/bias', n);
  SetLength(y, 12);
  fCpu.Linear(x, 3, 5, InferWeight(w, idtF32), 4, b, pointer(y));
  CheckOp('linear/y', pointer(y), 1e-6, 1e-6);
  // layernorm, including a nearly constant and a constant row
  x := fOps.F32('layernorm/x', n);
  g := fOps.F32('layernorm/gamma', i);
  b := fOps.F32('layernorm/beta', i);
  SetLength(y, n);
  fCpu.LayerNorm(x, 4, 8, g, b, 1e-5, pointer(y));
  CheckOp('layernorm/y', pointer(y), 1e-6, 1e-5);
  // exact GELU, in place on a copy
  x := fOps.F32('gelu/x', n);
  SetLength(y, n);
  MoveFast(x^, pointer(y)^, n * SizeOf(single));
  fCpu.Gelu(pointer(y), n);
  CheckOp('gelu/y', pointer(y), 1e-30, 1e-6);
  // masked softmax: row by row with each row's own mask
  x := fOps.F32('softmax/scores', n);
  keep := fOps.I32('softmax/keep', i);
  SetLength(y, n);
  MoveFast(x^, pointer(y)^, n * SizeOf(single));
  for i := 0 to 2 do
    fCpu.SoftmaxMasked(@y[i * 6], 1, 6, @keep^[i * 6]);
  CheckOp('softmax/y', pointer(y), 1e-7, 1e-6);
  for i := 0 to 17 do
    if keep^[i] = 0 then
      CheckUtf8(y[i] = 0, 'masked softmax entry % is exactly 0', [i]);
  // L2, including the 1e-20-scaled row
  x := fOps.F32('l2/x', n);
  SetLength(y, n);
  MoveFast(x^, pointer(y)^, n * SizeOf(single));
  for i := 0 to 2 do
    fCpu.L2Normalize(@y[i * 7], 7);
  CheckOp('l2/y', pointer(y), 1e-7, 1e-6);
end;

procedure TTestInferEngine.CpuOperationsKeepTheirContract;
var
  q, k, v, y, scratch, x, z: array[0..7] of single;
  keep: array[0..3] of integer;
  s00, s01, p0, p1: double;
  i: integer;
  ok: boolean;
begin
  // attention, 2 positions x 2 heads x head dim 2: row-major [seq, heads * dim]
  for i := 0 to 7 do
  begin
    q[i] := (i + 1) * 0.25;
    k[i] := (7 - i) * 0.125;
    v[i] := i - 3;
  end;
  keep[0] := 1;
  keep[1] := 1;
  fCpu.Attention(@q, @k, @v, 2, 2, 2, @keep, @scratch, @y);
  // head 0, query 0: scores q0.k0 and q0.k1 over the first two columns
  s00 := (q[0] * k[0] + q[1] * k[1]) / Sqrt(2);
  s01 := (q[0] * k[4] + q[1] * k[5]) / Sqrt(2);
  p0 := Exp(s00) / (Exp(s00) + Exp(s01));
  p1 := 1 - p0;
  CheckUtf8(Abs(y[0] - (p0 * v[0] + p1 * v[4])) < 1e-6, 'attention h0 q0 d0 %', [y[0]]);
  CheckUtf8(Abs(y[1] - (p0 * v[1] + p1 * v[5])) < 1e-6, 'attention h0 q0 d1 %', [y[1]]);
  // head 1 uses columns 2..3 only
  s00 := (q[2] * k[2] + q[3] * k[3]) / Sqrt(2);
  s01 := (q[2] * k[6] + q[3] * k[7]) / Sqrt(2);
  p0 := Exp(s00) / (Exp(s00) + Exp(s01));
  CheckUtf8(Abs(y[2] - (p0 * v[2] + (1 - p0) * v[6])) < 1e-6, 'attention h1 q0 %', [y[2]]);
  // a masked key does not contribute: the query sees v of key 0 only
  keep[1] := 0;
  fCpu.Attention(@q, @k, @v, 2, 2, 2, @keep, @scratch, @y);
  CheckUtf8((Abs(y[0] - v[0]) < 1e-6) and (Abs(y[4] - v[0]) < 1e-6),
    'masked key ignored');
  // Add
  for i := 0 to 7 do
  begin
    x[i] := i;
    z[i] := 10 * i;
  end;
  fCpu.Add(@x, @z, 8);
  ok := true;
  for i := 0 to 7 do
    ok := ok and (x[i] = 11 * i);
  CheckUtf8(ok, 'add');
  // softmax without a mask, and a row with every column masked
  for i := 0 to 3 do
    x[i] := i;
  fCpu.SoftmaxMasked(@x, 1, 4, nil);
  CheckUtf8(Abs(x[0] + x[1] + x[2] + x[3] - 1) < 1e-6, 'softmax without mask sums to 1');
  for i := 0 to 3 do
  begin
    x[i] := i;
    keep[i] := 0;
  end;
  fCpu.SoftmaxMasked(@x, 1, 4, @keep);
  CheckUtf8((x[0] = 0) and (x[1] = 0) and (x[2] = 0) and (x[3] = 0),
    'a fully masked row becomes 0');
  // a zero vector stays zero, no NaN
  FillCharFast(x, SizeOf(x), 0);
  fCpu.L2Normalize(@x, 8);
  ok := true;
  for i := 0 to 7 do
    ok := ok and (x[i] = 0);
  CheckUtf8(ok, 'zero vector stays zero');
end;

end.
