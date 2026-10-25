/// tests for the Unigram tokenizer of the native inference engine (phase 1a)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the gate: the ids and the normalized text of every case of
//   tests/data/infer-ref/bge-m3-tokens (SentencePiece, parse_special=false)
// - needs vendor/models/bge-m3/tokenizer.json (or $LAI_MODEL_DIR), written by
//   tools/infer-ref; says SKIP without it, LAI_REQUIRE_MODEL=1 makes that a failure
unit test.infer.tokenizer;

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
  mormot.ai.infer.tokenizer.ugm,
  test.infer.ref;

type
  /// phase 1a: the tokenizer against the reference ids
  TTestInferTokenizer = class(TSynTestCase)
  protected
    fTokenizer: TUgmTokenizer;
    fSkipped: boolean;
    procedure Setup; override;
    procedure CleanUp; override;
    /// false (and SKIP or a failure) when tokenizer.json is not there
    function Ready: boolean;
  published
    /// every reference case: same ids, same normalized text
    procedure IdsMatchReference;
    /// invalid UTF-8 becomes U+FFFD instead of an error or a crash
    procedure InvalidUtf8IsReplaced;
    /// a tokenizer.json of another pipeline is refused
    procedure ForeignPipelineIsRefused;
    /// the BGE-M3 file with a truncation, a padding or a piece twice refused
    procedure ChangedFileIsRefused;
  end;

/// the folder of the converted BGE-M3 files, '' when absent
function FindBgeM3Folder(const aFile: TFileName): TFileName;


implementation

function FindBgeM3Folder(const aFile: TFileName): TFileName;
var
  dir: TFileName;
  i: integer;
begin
  result := GetEnvironmentVariable('LAI_MODEL_DIR');
  if result <> '' then
  begin
    result := IncludeTrailingPathDelimiter(result);
    if not FileExists(result + aFile) then
      result := '';
    exit;
  end;
  dir := ExcludeTrailingPathDelimiter(Executable.ProgramFilePath);
  for i := 1 to 6 do
  begin
    result := MakePath([dir, 'vendor', 'models', 'bge-m3'], true);
    if FileExists(result + aFile) then
      exit;
    dir := ExtractFileDir(dir);
  end;
  result := '';
end;


{ TTestInferTokenizer }

procedure TTestInferTokenizer.Setup;
var
  dir: TFileName;
begin
  dir := FindBgeM3Folder('tokenizer.json');
  if dir <> '' then
    fTokenizer := TUgmTokenizer.Create(dir + 'tokenizer.json')
  else
    fSkipped := true;
end;

procedure TTestInferTokenizer.CleanUp;
begin
  FreeAndNil(fTokenizer); // re-entrant, as mORMot requires
end;

function TTestInferTokenizer.Ready: boolean;
begin
  result := fTokenizer <> nil;
  if result then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/bge-m3/tokenizer.json');
  AddConsole('SKIP tokenizer: no vendor/models/bge-m3/tokenizer.json ' +
    '(tools/infer-ref, make_reference.py bge-m3)');
end;

procedure TTestInferTokenizer.IdsMatchReference;
var
  ref: TInferRefSet;
  dir: TFileName;
  cases, c: PDocVariantData;
  ids, offsets: PIntegerArray;
  n, i, j, first, count, idsBad, normBad: PtrInt;
  mine: TIntegerDynArray;
  same: boolean;
  text, norm: RawUtf8;
  failures: RawUtf8;
  texts, norms: PByteArray;
  textOffsets, normOffsets: PIntegerArray;
  expected: RawUtf8;
begin
  if not Ready then
    exit;
  dir := FindInferRefDir('bge-m3-tokens');
  if CheckFailed(dir <> '', 'bge-m3-tokens found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    cases := ref.Manifest^.A['cases'];
    ids := ref.I32('ids', n);
    offsets := ref.I32('offsets', n);
    // the texts as exact bytes: the JSON copy in the manifest may lose a NUL
    texts := ref.U8('texts', i);
    textOffsets := ref.I32('text_offsets', i);
    norms := ref.U8('normalized', i);
    normOffsets := ref.I32('normalized_offsets', i);
    CheckEqual(n, cases^.Count + 1, 'one offset per case plus the end');
    idsBad := 0;
    normBad := 0;
    failures := '';
    for i := 0 to cases^.Count - 1 do
    begin
      c := _Safe(cases^.Values[i]);
      FastSetString(text, @texts^[textOffsets^[i]],
        textOffsets^[i + 1] - textOffsets^[i]);
      mine := fTokenizer.Encode(text);
      first := offsets^[i];
      count := offsets^[i + 1] - first;
      same := length(mine) = count;
      if same then
        for j := 0 to count - 1 do
          if mine[j] <> ids^[first + j] then
          begin
            same := false;
            break;
          end;
      if not same then
      begin
        inc(idsBad);
        if idsBad <= 5 then
        begin
          // first failures in full: case, ours, expected
          failures := failures + ' ' + c^.U['id'] + ' [';
          for j := 0 to high(mine) do
            failures := failures + Int32ToUtf8(mine[j]) + ' ';
          failures := failures + '] expected [';
          for j := 0 to count - 1 do
            failures := failures + Int32ToUtf8(ids^[first + j]) + ' ';
          failures := failures + ']';
        end;
      end;
      norm := fTokenizer.Normalize(text);
      FastSetString(expected, @norms^[normOffsets^[i]],
        normOffsets^[i + 1] - normOffsets^[i]);
      if norm <> expected then
      begin
        inc(normBad);
        if normBad <= 5 then
          failures := failures + ' norm:' + c^.U['id'];
      end;
    end;
    CheckUtf8(idsBad = 0, '% of % cases with other ids:%',
      [idsBad, cases^.Count, failures]);
    CheckUtf8(normBad = 0, '% of % cases normalized otherwise',
      [normBad, cases^.Count]);
  finally
    ref.Free;
  end;
end;

const
  /// a lone continuation byte, a truncated 3-byte sequence and an invalid lead
  // (bytes, not a literal: ASCII-only literals)
  BAD_UTF8: array[0..7] of byte = (ord('a'), $80, ord('b'), $E2, $82, ord('c'), $FF, 0);
  /// overlong NUL, a surrogate, a code point above U+10FFFF - well-formed byte
  // patterns that are no valid UTF-8 (SentencePiece replaces each byte)
  BAD_FORMS: array[0..8] of byte = ($C0, $80, $ED, $A0, $80, $F4, $90, $80, $80);
  REPLACEMENT: array[0..2] of byte = ($EF, $BF, $BD);

procedure TTestInferTokenizer.InvalidUtf8IsReplaced;
var
  bad, fffd: RawUtf8;
  norm: RawUtf8;
  ids: TIntegerDynArray;
begin
  if not Ready then
    exit;
  FastSetString(bad, @BAD_UTF8, 7);
  FastSetString(fffd, @REPLACEMENT, 3);
  norm := fTokenizer.Normalize(bad);
  CheckUtf8(PosEx(fffd, norm) > 0, 'U+FFFD in the normalized text');
  CheckUtf8(IsValidUtf8(norm), 'normalized text is valid UTF-8');
  ids := fTokenizer.Encode(bad);
  CheckUtf8((length(ids) >= 3) and (ids[0] = fTokenizer.BosId) and
    (ids[high(ids)] = fTokenizer.EosId), 'encoded between <s> and </s>');
  FastSetString(bad, @BAD_FORMS, length(BAD_FORMS));
  norm := fTokenizer.Normalize(bad);
  CheckUtf8(IsValidUtf8(norm), 'overlong, surrogate and > U+10FFFF replaced');
  CheckUtf8(PosEx(fffd, norm) > 0, 'U+FFFD for the invalid forms');
end;

procedure TTestInferTokenizer.ForeignPipelineIsRefused;
var
  fn: TFileName;
  refused: boolean;
  t: TUgmTokenizer;
begin
  // a byte-level BPE tokenizer.json (the Qwen kind) is not this pipeline
  fn := TemporaryFileName;
  FileFromString('{"normalizer":null,"pre_tokenizer":{"type":"ByteLevel"},' +
    '"model":{"type":"BPE","vocab":{},"merges":[]}}', fn);
  refused := false;
  try
    try
      t := TUgmTokenizer.Create(fn);
      t.Free;
    except
      on EInferError do
        refused := true;
    end;
  finally
    DeleteFile(fn);
  end;
  CheckUtf8(refused, 'BPE tokenizer.json refused');
end;

/// true when TUgmTokenizer refuses aDoc with a message holding aWhy
function UgmRefused(var aDoc: TDocVariantData; const aWhy: RawUtf8): boolean;
var
  fn: TFileName;
  t: TUgmTokenizer;
begin
  result := false;
  fn := TemporaryFileName;
  FileFromString(aDoc.ToJson, fn);
  try
    try
      t := TUgmTokenizer.Create(fn);
      t.Free;
    except
      on E: EInferError do
        result := PosEx(aWhy, StringToUtf8(E.Message)) > 0;
    end;
  finally
    DeleteFile(fn);
  end;
end;

procedure TTestInferTokenizer.ChangedFileIsRefused;
var
  fn: TFileName;
  doc: TDocVariantData;
  vocab: PDocVariantData;
begin
  if not Ready then
    exit;
  fn := FindBgeM3Folder('tokenizer.json') + 'tokenizer.json';
  // a saved truncation or padding would change what Encode returns
  doc.InitJsonFromFile(fn, JSON_FAST_FLOAT);
  doc.AddOrUpdateValue('truncation', _ObjFast(['max_length', 8]));
  Check(UgmRefused(doc, 'truncation'), 'truncation');
  doc.Clear;
  doc.InitJsonFromFile(fn, JSON_FAST_FLOAT);
  doc.AddOrUpdateValue('padding', _ObjFast(['length', 8]));
  Check(UgmRefused(doc, 'padding'), 'padding');
  // a piece twice: the tokenizers library keeps the last id
  doc.Clear;
  doc.InitJsonFromFile(fn, JSON_FAST_FLOAT);
  vocab := doc.O['model']^.A['vocab'];
  vocab^.Values[1000] := vocab^.Values[1001];
  Check(UgmRefused(doc, 'two ids'), 'a piece twice');
end;

end.
