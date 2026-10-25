/// tests for the byte-level BPE tokenizer and the Qwen3 chat template (phase 2a)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the gates: tests/data/infer-ref/qwen3-tokens (ids, NFC text, decoding) and
//   tests/data/infer-ref/qwen3-chat (template ids and text), both written by
//   tools/infer-ref/make_reference_qwen3.py
// - the NFC, merge order, JSON and refusal tests need no model; the gates need
//   vendor/models/qwen3-0.6b/tokenizer.json (or $LAI_QWEN3_DIR) and say SKIP
//   without it, LAI_REQUIRE_MODEL=1 makes that a failure
unit test.infer.bpe;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.json,
  mormot.core.perf,
  mormot.core.test,
  mormot.crypt.secure,
  mormot.ai.llm.types,
  mormot.ai.infer.tensor,
  mormot.ai.infer.unicode,
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.chat,
  mormot.ai.infer.loader.gguf,
  test.infer.ref;

type
  /// phase 2a: Unicode, BPE tokenizer and chat template against the reference
  TTestInferBpe = class(TSynTestCase)
  protected
    fTokenizer: TBpeTokenizer;
    fTemplate: TQwen3ChatTemplate;
    procedure Setup; override;
    procedure CleanUp; override;
    /// false (and SKIP or a failure) when tokenizer.json is not there
    function Ready: boolean;
    /// every reference case through aTokenizer: same ids, decoding
    procedure CheckIds(aTokenizer: TBpeTokenizer; const aLabel: RawUtf8);
  published
    /// NFC of every reference text equals the HF normalizer (no model needed)
    procedure NfcMatchesReference;
    /// code point classes and NFC spot checks
    procedure UnicodeBasics;
    /// merges by rank, leftmost first; foreign pipelines refused
    procedure MergeOrderAndRefusals;
    /// Python json.dumps formatting of tool schemas
    procedure PythonJsonFormatting;
    /// every reference case: same ids; decoding gives the normalized text
    procedure IdsMatchReference;
    /// control-token literals in text stay text; invalid UTF-8 becomes U+FFFD
    procedure LiteralsAndInvalidUtf8;
    /// the tokenizer of Qwen's GGUF file = the one of tokenizer.json: every
    // token, every added token, the reference ids; foreign tokenizers refused
    procedure GgufTokenizer;
    /// vocab.json + merges.txt: header, CRLF and empty lines, added tokens from
    // tokenizer_config.json or (older folders) added_tokens.json, named special
    // tokens checked; foreign files refused (no model needed)
    procedure VocabMergesFiles;
    /// the pair of Qwen3-0.6B = its tokenizer.json; the pair of Qwen3-TTS (no
    // tokenizer.json) gives the reference ids and its 33 added tokens
    procedure VocabMergesMatchTokenizerJson;
    /// every chat case: same ids and same rendered text as transformers
    procedure ChatTemplateMatchesReference;
    /// the same cases through the Qwen2.5 template (tokenizer.json of
    // Qwen2.5-0.5B-Instruct): same ids, same text; the Qwen3 template refuses
    // a tokenizer without <think>
    procedure Qwen25ChatTemplateMatchesReference;
  end;

/// vendor/models/<aName> (a file) found from the executable upwards, '' when
// absent
function FindVendorModel(const aName: TFileName): TFileName;

/// the folder of the Qwen3-0.6B files, '' when absent
function FindQwen3Folder(const aFile: TFileName): TFileName;

/// a byte-level tokenizer.json: the 256 byte tokens (id = byte) plus aExtra
// and aMerges, and <|im_end|> as added token 300
function TinyBpeJson(const aNormalizer, aExtraVocab, aMerges: RawUtf8): RawUtf8;

/// a tokenizer from aJson, nil and aError when it is refused
function LoadBpe(const aJson: RawUtf8; out aError: RawUtf8): TBpeTokenizer;


implementation

function FindQwen3Folder(const aFile: TFileName): TFileName;
var
  dir: TFileName;
  i: integer;
begin
  result := GetEnvironmentVariable('LAI_QWEN3_DIR');
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
    result := MakePath([dir, 'vendor', 'models', 'qwen3-0.6b'], true);
    if FileExists(result + aFile) then
      exit;
    dir := ExtractFileDir(dir);
  end;
  result := '';
end;

function IdsText(aIds: PIntegerArray; aCount: PtrInt): RawUtf8;
var
  i: PtrInt;
begin
  result := '[';
  for i := 0 to aCount - 1 do
    result := result + Int32ToUtf8(aIds^[i]) + ' ';
  result := result + ']';
end;

function SameIds(const aMine: TIntegerDynArray; aRef: PIntegerArray;
  aCount: PtrInt): boolean;
var
  i: PtrInt;
begin
  result := length(aMine) = aCount;
  if result then
    for i := 0 to aCount - 1 do
      if aMine[i] <> aRef^[i] then
      begin
        result := false;
        exit;
      end;
end;


function FindVendorModel(const aName: TFileName): TFileName;
var
  dir: TFileName;
  i: integer;
begin
  dir := ExcludeTrailingPathDelimiter(Executable.ProgramFilePath);
  for i := 1 to 6 do
  begin
    result := MakePath([dir, 'vendor', 'models']) + PathDelim + aName;
    if FileExists(result) then
      exit;
    dir := ExtractFileDir(dir);
  end;
  result := '';
end;


{ TTestInferBpe }

procedure TTestInferBpe.Setup;
var
  dir: TFileName;
begin
  dir := FindQwen3Folder('tokenizer.json');
  if dir = '' then
    exit;
  fTokenizer := TBpeTokenizer.Create(dir + 'tokenizer.json');
  fTemplate := TQwen3ChatTemplate.Create(fTokenizer);
end;

procedure TTestInferBpe.CleanUp;
begin
  FreeAndNil(fTemplate); // re-entrant, as mORMot requires
  FreeAndNil(fTokenizer);
end;

function TTestInferBpe.Ready: boolean;
begin
  result := fTokenizer <> nil;
  if result then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b/tokenizer.json');
  AddConsole('SKIP qwen3 tokenizer: no vendor/models/qwen3-0.6b/tokenizer.json ' +
    '(tools/infer-ref, make_reference_qwen3.py)');
end;

procedure TTestInferBpe.NfcMatchesReference;
var
  ref: TInferRefSet;
  dir: TFileName;
  cases: PDocVariantData;
  texts, norms: PByteArray;
  textOffsets, normOffsets: PIntegerArray;
  n, i, bad: PtrInt;
  text, expected: RawUtf8;
  failures: RawUtf8;
begin
  dir := FindInferRefDir('qwen3-tokens');
  if CheckFailed(dir <> '', 'qwen3-tokens found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    cases := ref.Manifest^.A['cases'];
    texts := ref.U8('texts', n);
    textOffsets := ref.I32('text_offsets', n);
    norms := ref.U8('normalized', n);
    normOffsets := ref.I32('normalized_offsets', n);
    bad := 0;
    failures := '';
    for i := 0 to cases^.Count - 1 do
    begin
      FastSetString(text, @texts^[textOffsets^[i]], textOffsets^[i + 1] - textOffsets^[i]);
      FastSetString(expected, @norms^[normOffsets^[i]],
        normOffsets^[i + 1] - normOffsets^[i]);
      if NfcNormalize(text) <> expected then
      begin
        inc(bad);
        if bad <= 5 then
          failures := failures + ' ' + _Safe(cases^.Values[i])^.U['id'];
      end;
    end;
    CheckUtf8(bad = 0, '% of % texts normalized otherwise:%',
      [bad, cases^.Count, failures]);
  finally
    ref.Free;
  end;
end;

function Cps(const aCp: array of cardinal): RawUtf8;
begin
  result := CodePointsToUtf8(@aCp[0], length(aCp));
end;

/// aCount times aCp as UTF-8
function Repeated(aCp: cardinal; aCount: PtrInt): RawUtf8;
var
  cp: TCardinalDynArray;
  i: PtrInt;
begin
  SetLength(cp, aCount);
  for i := 0 to aCount - 1 do
    cp[i] := aCp;
  result := CodePointsToUtf8(pointer(cp), aCount);
end;

/// aBase, then aCount acute accents (class 230), then aCount dots below (220)
function Marks(aBase: cardinal; aCount: PtrInt): RawUtf8;
begin
  result := Cps([aBase]) + Repeated($301, aCount) + Repeated($323, aCount);
end;

procedure TTestInferBpe.UnicodeBasics;
var
  timer: TPrecisionTimer;
begin
  Check(UnicodeClassOf(ord('a')) = ucLetter, 'a');
  Check(UnicodeClassOf(ord('7')) = ucNumber, '7');
  Check(UnicodeClassOf(10) = ucSpace, 'LF');
  Check(UnicodeClassOf(ord('!')) = ucOther, '!');
  Check(UnicodeClassOf($e4) = ucLetter, 'a umlaut');
  Check(UnicodeClassOf($3000) = ucSpace, 'ideographic space');
  Check(UnicodeClassOf($200b) = ucOther, 'zero width space is no space');
  Check(UnicodeClassOf($2163) = ucNumber, 'roman numeral four');
  Check(UnicodeClassOf($301) = ucOther, 'combining mark');
  Check(UnicodeClassOf($4e00) = ucLetter, 'CJK');
  Check(CombiningClassOf($301) = 230, 'ccc acute');
  Check(CombiningClassOf($323) = 220, 'ccc dot below');
  Check(CombiningClassOf(ord('a')) = 0, 'ccc a');
  CheckEqual(NfcNormalize('abc'), 'abc', 'ASCII unchanged');
  CheckEqual(NfcNormalize(Cps([ord('e'), $301])), Cps([$e9]), 'e + acute');
  CheckEqual(NfcNormalize(Cps([$212b])), Cps([$c5]), 'angstrom sign');
  CheckEqual(NfcNormalize(Cps([$1100, $1161, $11a8])), Cps([$ac01]), 'Hangul LVT');
  CheckEqual(NfcNormalize(Cps([$ac00, $11a8])), Cps([$ac01]), 'Hangul LV + T');
  CheckEqual(NfcNormalize(Cps([ord('q'), $307, $323])), Cps([ord('q'), $323, $307]),
    'reordered by combining class');
  CheckEqual(NfcNormalize(Cps([$958])), Cps([$915, $93c]), 'composition exclusion');
  CheckEqual(NfcNormalize(Cps([ord('a'), $316, $301])), Cps([$e1, $316]),
    'not blocked by a lower class');
  CheckEqual(NfcNormalize(Cps([$301, ord('a')])), Cps([$301, ord('a')]),
    'leading mark kept');
  // a run longer than the insertion sort handles: counting sort, same order
  CheckEqual(NfcNormalize(Marks(ord('a'), 20)), Cps([$1ea1]) + Repeated($323, 19) +
    Repeated($301, 20), 'long run of marks reordered and composed');
  // hostile: 100000 marks in reverse class order - linear, not quadratic
  timer.Start;
  // U+1EA1 (3 bytes) + 49999 dots below + 50000 acutes (2 bytes each)
  CheckEqual(length(NfcNormalize(Marks(ord('a'), 50000))), 3 + 2 * 99999,
    '100000 marks');
  CheckUtf8(timer.TimeInMicroSec < 5000000, 'marks sorted in %', [timer.Stop]);
end;

function TinyBpeJson(const aNormalizer, aExtraVocab, aMerges: RawUtf8): RawUtf8;
const
  HEX: array[0..15] of AnsiChar = '0123456789abcdef';
var
  b, n, cp: integer;
begin
  result := '{"normalizer":' + aNormalizer + ',"pre_tokenizer":{"type":"Sequence",' +
    '"pretokenizers":[{"type":"Split","pattern":{"Regex":' +
    QuotedStrJson(QWEN2_SPLIT_REGEX) + '},"behavior":"Isolated","invert":false},' +
    '{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,' +
    '"use_regex":false}]},"decoder":{"type":"ByteLevel"},"model":{"type":"BPE",' +
    '"dropout":null,"unk_token":null,"continuing_subword_prefix":"",' +
    '"end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,' +
    '"ignore_merges":false,"vocab":{';
  n := 0;
  for b := 0 to 255 do
  begin
    // the GPT-2 byte alphabet, written as \u escapes
    if ((b >= ord('!')) and (b <= ord('~'))) or ((b >= $a1) and (b <= $ac)) or
       ((b >= $ae) and (b <= $ff)) then
      cp := b
    else
    begin
      cp := 256 + n;
      inc(n);
    end;
    result := result + FormatUtf8('"\u%%%%":%,', [HEX[(cp shr 12) and 15],
      HEX[(cp shr 8) and 15], HEX[(cp shr 4) and 15], HEX[cp and 15], b]);
  end;
  result := result + aExtraVocab + '},"merges":' + aMerges + '},"added_tokens":[' +
    '{"id":300,"content":"<|im_end|>","special":true}]}';
end;

function LoadBpe(const aJson: RawUtf8; out aError: RawUtf8): TBpeTokenizer;
var
  fn: TFileName;
begin
  result := nil;
  aError := '';
  fn := TemporaryFileName;
  FileFromString(aJson, fn);
  try
    try
      result := TBpeTokenizer.Create(fn);
    except
      on E: EInferError do
        aError := StringToUtf8(E.Message);
    end;
  finally
    DeleteFile(fn);
  end;
end;

procedure TTestInferBpe.MergeOrderAndRefusals;
var
  t: TBpeTokenizer;
  err: RawUtf8;
  ids: TIntegerDynArray;
const
  VOCAB = '"aa":256,"bc":257,"aaa":258,"xy":259';
  MERGES = '[["a","a"],["b","c"],["aa","a"],["x","y"]]';
begin
  t := LoadBpe(TinyBpeJson('{"type":"NFC"}', VOCAB, MERGES), err);
  if CheckFailed(t <> nil, Utf8ToString(err)) then
    exit;
  try
    CheckEqual(t.VocabSize, 301, 'vocabulary with the added token');
    CheckEqual(t.TokenId('<|im_end|>'), 300, 'added token id');
    // "aaaa": (a,a) at 0, 1, 2 share rank 0 - leftmost first gives aa + aa,
    // and (aa, aa) is no merge
    ids := t.Encode('aaaa');
    CheckEqual(IdsText(pointer(ids), length(ids)), '[256 256 ]', 'aaaa');
    ids := t.Encode('aaa');
    CheckEqual(IdsText(pointer(ids), length(ids)), '[258 ]', 'aaa merges twice');
    ids := t.Encode('abc');
    CheckEqual(IdsText(pointer(ids), length(ids)), '[97 257 ]', 'abc');
    // the split keeps " xy" in one piece; BPE has no merge for the space
    ids := t.Encode(' xy');
    CheckEqual(IdsText(pointer(ids), length(ids)), '[32 259 ]', 'space xy');
    ids := t.Encode('<|im_end|>');
    Check(length(ids) > 1, 'added token literal is text');
    CheckEqual(t.Decode(ids), '<|im_end|>', 'and decodes back');
    CheckEqual(t.Decode([300]), '<|im_end|>', 'added token decodes to its content');
  finally
    t.Free;
  end;
  t := LoadBpe(StringReplaceAll(TinyBpeJson('{"type":"NFC"}', VOCAB, MERGES),
    '"added_tokens":[{"id":300,"content":"<|im_end|>","special":true}]',
    '"added_tokens":{"x":{"id":300,"content":"<|im_end|>"}}'), err);
  Check(t = nil, 'added_tokens as an object refused');
  t.Free;
  t := LoadBpe(StringReplaceAll(TinyBpeJson('{"type":"NFC"}', VOCAB, MERGES),
    '"added_tokens":[{"id":300,"content":"<|im_end|>","special":true}]',
    '"added_tokens":[]'), err);
  CheckUtf8(t <> nil, 'empty added_tokens accepted: %', [err]);
  if t <> nil then
    CheckEqual(t.VocabSize, 260, 'no added token');
  t.Free;
  t := LoadBpe(TinyBpeJson('{"type":"NFKC"}', VOCAB, MERGES), err);
  Check(t = nil, 'NFKC normalizer refused');
  t.Free;
  t := LoadBpe(TinyBpeJson('{"type":"NFC"}', VOCAB,
    '[["a","a"],["b","c"],["a","a"]]'), err);
  Check(t = nil, 'repeated merge refused');
  t.Free;
  t := LoadBpe(TinyBpeJson('{"type":"NFC"}', VOCAB, '[["a","q"]]'), err);
  Check(t = nil, 'merge without its token refused');
  t.Free;
  t := LoadBpe(StringReplaceAll(TinyBpeJson('{"type":"NFC"}', VOCAB, MERGES),
    '"byte_fallback":false', '"byte_fallback":true'), err);
  Check(t = nil, 'byte fallback refused');
  t.Free;
  t := LoadBpe(StringReplaceAll(TinyBpeJson('{"type":"NFC"}', VOCAB, MERGES),
    '"Isolated"', '"Removed"'), err);
  Check(t = nil, 'other split behaviour refused');
  t.Free;
end;

function JsonRaises(const aJson: RawUtf8): boolean;
begin
  result := false;
  try
    PythonJson(aJson);
  except
    on EInferError do
      result := true;
  end;
end;

const
  /// U+D800 written as UTF-8 bytes: well-formed shape, invalid value (a byte
  // array, not a literal: Unicode Delphi would encode #$ED#$A0#$80 as text)
  SURROGATE_BYTES: array[0..2] of byte = ($ED, $A0, $80);

procedure TTestInferBpe.PythonJsonFormatting;
var
  surrogate: RawUtf8;
begin
  CheckEqual(PythonJson('{"a":1,"b":[true,false,null],"c":{}}'),
    '{"a": 1, "b": [true, false, null], "c": {}}', 'separators');
  CheckEqual(PythonJson(' [ ] '), '[]', 'empty array');
  CheckEqual(PythonJson('{"z":1,"a":2}'), '{"z": 1, "a": 2}', 'key order kept');
  CheckEqual(PythonJson('[0.5,1E5,1e-5,-0.0,100.0,1.50,0.0001,2.5E+20,1e16,1e15]'),
    '[0.5, 100000.0, 1e-05, -0.0, 100.0, 1.5, 0.0001, 2.5e+20, 1e+16, ' +
    '1000000000000000.0]', 'floats as Python repr');
  CheckEqual(PythonJson('[-0,12345678901234567890,3,0.001,123.456,1e-4,1e-3]'),
    '[0, 12345678901234567890, 3, 0.001, 123.456, 0.0001, 0.001]',
    'integers kept, small floats');
  CheckEqual(PythonJson('[1.25e2, 0.00012, 9.99e-5, 1.5e300, 1e-300]'),
    '[125.0, 0.00012, 9.99e-05, 1.5e+300, 1e-300]', 'exponent boundaries');
  CheckEqual(PythonJson('"a\nb\t\"q\" \u0001 \u00fc \/ \\"'),
    '"a\nb\t\"q\" \u0001 ' + Cps([$fc]) + ' / \\"', 'string escapes');
  CheckEqual(PythonJson('"\ud83d\ude00"'), '"' + Cps([$1f600]) + '"', 'surrogate pair');
  CheckEqual(PythonJson('"' + Cps([$7f]) + '"'), '"' + Cps([$7f]) + '"', 'DEL raw');
  Check(JsonRaises('[1.2345678901234567]'), 'more than 15 digits refused');
  Check(JsonRaises('[1e309]'), 'overflow refused');
  Check(JsonRaises('[NaN]'), 'NaN refused');
  Check(JsonRaises('[0.' + RawUtf8(StringOfChar('0', 20)) + '1e1000000]'),
    'huge exponent refused, not saturated');
  CheckEqual(PythonJson('"a\u0000b"'), '"a\u0000b"', 'NUL in a string');
  CheckEqual(PythonJson(RawUtf8(StringOfChar('[', 100)) + '0' +
    RawUtf8(StringOfChar(']', 100))), RawUtf8(StringOfChar('[', 100)) + '0' +
    RawUtf8(StringOfChar(']', 100)), '100 levels');
  Check(JsonRaises(RawUtf8(StringOfChar('[', 100000)) + '0' +
    RawUtf8(StringOfChar(']', 100000))), 'deep nesting refused, no stack overflow');
  FastSetString(surrogate, @SURROGATE_BYTES, length(SURROGATE_BYTES));
  Check(JsonRaises('"A' + surrogate + '"'), 'surrogate bytes refused');
  Check(JsonRaises('{"a":1,"a":2}'), 'duplicate key refused');
  Check(JsonRaises('"\ud83d"'), 'lone surrogate refused');
  Check(JsonRaises('"a' + #10 + 'b"'), 'raw control character refused');
  Check(JsonRaises('[1,]'), 'trailing comma refused');
  Check(JsonRaises('{} x'), 'trailing data refused');
  Check(JsonRaises('[01]'), 'leading zero refused');
  Check(JsonRaises(''), 'empty refused');
end;

procedure TTestInferBpe.IdsMatchReference;
begin
  if Ready then
    CheckIds(fTokenizer, 'tokenizer.json');
end;

procedure TTestInferBpe.CheckIds(aTokenizer: TBpeTokenizer; const aLabel: RawUtf8);
var
  ref: TInferRefSet;
  dir: TFileName;
  cases: PDocVariantData;
  ids, offsets, textOffsets, normOffsets: PIntegerArray;
  texts, norms: PByteArray;
  n, i, first, count, idsBad, decBad: PtrInt;
  mine: TIntegerDynArray;
  text, expected, failures: RawUtf8;
begin
  dir := FindInferRefDir('qwen3-tokens');
  if CheckFailed(dir <> '', 'qwen3-tokens found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    cases := ref.Manifest^.A['cases'];
    ids := ref.I32('ids', n);
    offsets := ref.I32('offsets', n);
    texts := ref.U8('texts', n);
    textOffsets := ref.I32('text_offsets', n);
    norms := ref.U8('normalized', n);
    normOffsets := ref.I32('normalized_offsets', n);
    idsBad := 0;
    decBad := 0;
    failures := '';
    for i := 0 to cases^.Count - 1 do
    begin
      FastSetString(text, @texts^[textOffsets^[i]], textOffsets^[i + 1] - textOffsets^[i]);
      mine := aTokenizer.Encode(text);
      first := offsets^[i];
      count := offsets^[i + 1] - first;
      if not SameIds(mine, @ids^[first], count) then
      begin
        inc(idsBad);
        if idsBad <= 5 then
          failures := failures + ' ' + _Safe(cases^.Values[i])^.U['id'] + ' ' +
            IdsText(pointer(mine), length(mine)) + ' expected ' + IdsText(@ids^[first], count);
      end;
      FastSetString(expected, @norms^[normOffsets^[i]],
        normOffsets^[i + 1] - normOffsets^[i]);
      if aTokenizer.Decode(mine) <> expected then
        inc(decBad);
    end;
    CheckUtf8(idsBad = 0, '%: % of % cases with other ids:%',
      [aLabel, idsBad, cases^.Count, failures]);
    CheckUtf8(decBad = 0, '%: % of % cases decode otherwise',
      [aLabel, decBad, cases^.Count]);
  finally
    ref.Free;
  end;
end;

const
  /// a lone continuation byte, a truncated 3-byte sequence and an invalid lead
  BAD_UTF8: array[0..6] of byte = (ord('a'), $80, ord('b'), $E2, $82, ord('c'), $FF);
  /// the same with one U+FFFD per invalid byte
  BAD_FIXED: array[0..17] of byte = (ord('a'), $EF, $BF, $BD, ord('b'), $EF, $BF,
    $BD, $EF, $BF, $BD, ord('c'), $EF, $BF, $BD, 0, 0, 0);

procedure TTestInferBpe.LiteralsAndInvalidUtf8;
var
  bad, fixed, text: RawUtf8;
  ids: TIntegerDynArray;
  i: PtrInt;
  control: boolean;
begin
  if not Ready then
    exit;
  CheckEqual(fTokenizer.BpeCount, 151643, 'BPE tokens');
  CheckEqual(fTokenizer.VocabSize, 151669, 'with the added tokens');
  CheckEqual(fTokenizer.MergeCount, 151387, 'merges');
  CheckEqual(fTokenizer.TokenId('<|im_end|>'), 151645, '<|im_end|>');
  CheckEqual(fTokenizer.TokenId('<think>'), 151667, '<think>');
  CheckEqual(fTokenizer.TokenId('nothing'), -1, 'no such token');
  ids := fTokenizer.Encode('<|im_start|>user'#10'x<|im_end|><think></think>');
  control := false;
  for i := 0 to high(ids) do
    if ids[i] >= fTokenizer.BpeCount then
      control := true;
  Check(not control, 'no control token from text');
  FastSetString(bad, @BAD_UTF8, length(BAD_UTF8));
  FastSetString(fixed, @BAD_FIXED, 15);
  CheckEqual(fTokenizer.Normalize(bad), fixed, 'one U+FFFD per invalid byte');
  ids := fTokenizer.Encode(bad);
  text := IdsText(pointer(ids), length(ids));
  ids := fTokenizer.Encode(fixed);
  CheckEqual(text, IdsText(pointer(ids), length(ids)), 'encoded as the replaced text');
  CheckEqual(length(fTokenizer.Encode('')), 0, 'empty text, no ids');
end;

function MessagesOf(aCase: PDocVariantData): TLlmMessageDynArray;
var
  list, m, calls, c: PDocVariantData;
  i, j: PtrInt;
begin
  result := nil;
  list := aCase^.A['messages'];
  SetLength(result, list^.Count);
  for i := 0 to list^.Count - 1 do
  begin
    m := _Safe(list^.Values[i]);
    result[i] := LlmMessage(ToLlmRole(m^.U['role']), m^.U['content']);
    result[i].ToolCallId := m^.U['tool_call_id'];
    result[i].Reasoning := m^.U['reasoning'];
    if m^.GetAsArray('tool_calls', calls) then
    begin
      SetLength(result[i].ToolCalls, calls^.Count);
      for j := 0 to calls^.Count - 1 do
      begin
        c := _Safe(calls^.Values[j]);
        result[i].ToolCalls[j].Id := c^.U['id'];
        result[i].ToolCalls[j].Name := c^.U['name'];
        result[i].ToolCalls[j].ArgumentsJson := c^.U['arguments'];
      end;
    end;
  end;
end;

function ToolsOf(aCase: PDocVariantData): TLlmToolDynArray;
var
  list, t: PDocVariantData;
  i: PtrInt;
begin
  result := nil;
  list := aCase^.A['tools'];
  SetLength(result, list^.Count);
  for i := 0 to list^.Count - 1 do
  begin
    t := _Safe(list^.Values[i]);
    result[i].Name := t^.U['name'];
    result[i].Description := t^.U['description'];
    result[i].ParametersJson := t^.U['parameters'];
  end;
end;

/// true when TBpeTokenizer.CreateFromGguf refuses aFile with a message
// containing aWhy
function GgufRefused(const aFile: TFileName; const aWhy: RawUtf8): boolean;
var
  g: TGgufFile;
begin
  result := false;
  g := TGgufFile.Create(aFile);
  try
    try
      TBpeTokenizer.CreateFromGguf(g).Free;
    except
      on E: EInferError do
        result := PosEx(aWhy, StringToUtf8(E.Message)) > 0;
    end;
  finally
    g.Free;
  end;
end;

procedure TTestInferBpe.GgufTokenizer;
var
  fn: TFileName;
  g: TGgufFile;
  tok: TBpeTokenizer;
  i, bad: PtrInt;
  raised: boolean;
  added: RawUtf8;
begin
  // a GGUF tokenizer of another kind: the Unigram one of the tiny XLM-R file
  fn := FindInferRefDir('tiny-gguf');
  if not CheckFailed(fn <> '', 'tiny-gguf found') then
  begin
    g := TGgufFile.Create(fn + 'model.gguf');
    try
      raised := false;
      try
        TBpeTokenizer.CreateFromGguf(g).Free;
      except
        on E: EInferError do
          raised := PosEx('no byte-level BPE', StringToUtf8(E.Message)) > 0;
      end;
      Check(raised, 'a Unigram (t5) tokenizer refused');
    finally
      g.Free;
    end;
  end;
  fn := FindInferRefDir('tiny-qwen3-gguf');
  if not CheckFailed(fn <> '', 'tiny-qwen3-gguf found') then
  begin
    Check(GgufRefused(fn + 'tok-bad-pre.gguf', 'is neither qwen2 nor gpt-4o'),
      'another pre-tokenizer refused');
    Check(GgufRefused(fn + 'tok-bad-type.gguf', 'of type 6'),
      'a byte token after the vocabulary refused');
  end;
  if not Ready then
    exit;
  fn := ExpandFileName(FindQwen3Folder('tokenizer.json') + '..' + PathDelim +
    'qwen3-0.6b-q8_0.gguf');
  if not FileExists(fn) then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b-q8_0.gguf');
    AddConsole('SKIP GGUF tokenizer: no vendor/models/qwen3-0.6b-q8_0.gguf');
    exit;
  end;
  g := TGgufFile.Create(fn);
  tok := nil;
  try
    tok := TBpeTokenizer.CreateFromGguf(g);
    CheckEqual(tok.VocabSize, fTokenizer.VocabSize, 'vocabulary size');
    CheckEqual(tok.BpeCount, fTokenizer.BpeCount, 'BPE tokens');
    CheckEqual(tok.MergeCount, fTokenizer.MergeCount, 'merges');
    bad := 0;
    for i := 0 to fTokenizer.VocabSize - 1 do
      if tok.TokenBytes(i) <> fTokenizer.TokenBytes(i) then
        inc(bad);
    CheckEqual(bad, 0, 'every token the same bytes');
    bad := 0;
    for i := fTokenizer.BpeCount to fTokenizer.VocabSize - 1 do
    begin
      added := fTokenizer.TokenBytes(i);
      if (added <> '') and (tok.TokenId(added) <> fTokenizer.TokenId(added)) then
        inc(bad);
    end;
    CheckEqual(bad, 0, 'every added token found by its content');
    CheckIds(tok, 'GGUF');
  finally
    tok.Free;
    g.Free;
  end;
end;

/// aFolder\aName with the content aText
procedure WriteText(const aFolder, aName: TFileName; const aText: RawUtf8);
begin
  FileFromString(aText, aFolder + aName);
end;

/// TBpeTokenizer.CreateFromVocabMerges on the files of aFolder, nil and aError
// when refused
function LoadPair(const aFolder: TFileName; out aError: RawUtf8): TBpeTokenizer;
begin
  result := nil;
  aError := '';
  try
    result := TBpeTokenizer.CreateFromVocabMerges(aFolder + 'vocab.json',
      aFolder + 'merges.txt');
  except
    on E: EInferError do
      aError := StringToUtf8(E.Message);
  end;
end;

procedure TTestInferBpe.VocabMergesFiles;
var
  json, vocab, err: RawUtf8;
  dir: TFileName;
  one, two: TBpeTokenizer;
  ids: TIntegerDynArray;
  i: PtrInt;
  same: boolean;
const
  TEXTS: array[0..3] of RawUtf8 = ('aaaa', 'abc aaa', ' xy  x', '<|im_end|>');
  // the vocabulary and merges of MergeOrderAndRefusals, as the older pair
  EXTRA = '"aa":256,"bc":257,"aaa":258,"xy":259';
  MERGES = '#version: 0.2'#13#10'a a'#13#10'b c'#13#10'#version 0.2'#10'aa a'#10'x y'#13#10;
  // the added tokens right after the 260 of the vocabulary, as transformers
  // numbers them; <|endoftext|> is Qwen2Tokenizer's default unk/eos/pad token
  DEC = '"added_tokens_decoder":{"260":{"content":"<|im_end|>","special":true},' +
    '"261":{"content":"<|endoftext|>","special":true}}';
  CONFIG = '{"tokenizer_class":"Qwen2Tokenizer","add_prefix_space":false,' + DEC + '}';
  LEGACY = '{"tokenizer_class":"Qwen2Tokenizer","add_bos_token":false}';
  NO_DEFAULTS = '{"tokenizer_class":"Qwen2Tokenizer","unk_token":null,' +
    '"eos_token":null,"pad_token":null}';

  function Refused(const aWhy: RawUtf8): boolean;
  begin
    two := LoadPair(dir, err);
    result := (two = nil) and (PosEx(aWhy, err) > 0);
    FreeAndNil(two);
  end;

  function Loads: boolean;
  begin
    two := LoadPair(dir, err);
    result := two <> nil;
    FreeAndNil(two);
  end;

  procedure WithConfig(const aInsert: RawUtf8);
  begin
    // CONFIG with aInsert after "add_prefix_space":false,
    WriteText(dir, 'tokenizer_config.json', StringReplaceAll(CONFIG,
      '"add_prefix_space":false,', '"add_prefix_space":false,' + aInsert));
  end;

begin
  // the reference: the same tokenizer from a tokenizer.json (its added token
  // at 300 - a tokenizer.json keeps the ids it gives)
  json := TinyBpeJson('{"type":"NFC"}', EXTRA, '[["a","a"],["b","c"],["aa","a"],["x","y"]]');
  one := LoadBpe(json, err);
  if CheckFailed(one <> nil, Utf8ToString(err)) then
    exit;
  dir := IncludeTrailingPathDelimiter(TemporaryFileName + '.bpe');
  two := nil;
  try
    ForceDirectories(dir);
    // vocab.json: the "vocab" object of the tokenizer.json
    i := PosEx('"vocab":{', json);
    vocab := copy(json, i + 8, PosEx('},"merges"', json) - i - 7);
    WriteText(dir, 'vocab.json', vocab);
    WriteText(dir, 'merges.txt', MERGES);
    WriteText(dir, 'tokenizer_config.json', CONFIG);
    two := LoadPair(dir, err);
    if CheckFailed(two <> nil, Utf8ToString(err)) then
      exit;
    CheckEqual(two.MergeCount, 4, 'every #version line skipped, CRLF and LF');
    CheckEqual(two.VocabSize, 262, 'the vocabulary and two added tokens');
    same := true;
    for i := 0 to 259 do
      if two.TokenBytes(i) <> one.TokenBytes(i) then
        same := false;
    Check(same, 'every token the same bytes');
    CheckEqual(two.TokenId('<|im_end|>'), 260, 'added token from the config');
    CheckEqual(two.TokenId('<|endoftext|>'), 261, 'second added token');
    for i := 0 to high(TEXTS) do
    begin
      ids := two.Encode(TEXTS[i]);
      json := IdsText(pointer(ids), length(ids));
      ids := one.Encode(TEXTS[i]);
      CheckEqual(json, IdsText(pointer(ids), length(ids)), TEXTS[i]);
    end;
    FreeAndNil(two);
    // with added_tokens_decoder, added_tokens.json and special_tokens_map.json
    // are not read (transformers 5.19)
    WriteText(dir, 'added_tokens.json', '{"<x>":262}');
    WriteText(dir, 'special_tokens_map.json', '{"pad_token":"<|pad|>"}');
    two := LoadPair(dir, err);
    if not CheckFailed(two <> nil, Utf8ToString(err)) then
      CheckEqual(two.TokenId('<x>'), -1, 'added_tokens.json beside a decoder ignored');
    FreeAndNil(two);
    // without it (an older folder): added_tokens.json gives the added tokens,
    // and the special tokens of special_tokens_map.json must exist
    WriteText(dir, 'tokenizer_config.json', LEGACY);
    WriteText(dir, 'added_tokens.json',
      '{"<x>":262,"<|im_end|>":260,"<|endoftext|>":261}');
    DeleteFile(dir + 'special_tokens_map.json');
    two := LoadPair(dir, err);
    if not CheckFailed(two <> nil, Utf8ToString(err)) then
    begin
      CheckEqual(two.TokenId('<x>'), 262, 'added_tokens.json read');
      CheckEqual(two.TokenId('<|im_end|>'), 260, 'in the order of the ids');
    end;
    FreeAndNil(two);
    WriteText(dir, 'special_tokens_map.json', '{"pad_token":{"content":"<|pad|>"}}');
    Check(Refused('special_tokens_map.json'),
      'special_tokens_map.json checked without a decoder');
    WriteText(dir, 'special_tokens_map.json', '{"pad_token":"<x>","unk_token":null}');
    CheckUtf8(Loads, 'special_tokens_map.json with added tokens: %', [err]);
    // every name is checked, also one the map would replace (stricter than
    // transformers on folders whose files disagree)
    WriteText(dir, 'tokenizer_config.json', StringReplaceAll(LEGACY,
      '"add_bos_token":false', '"add_bos_token":false,"pad_token":"<|nope|>"'));
    Check(Refused('pad_token of tokenizer_config.json'),
      'a config key the map replaces is checked too');
    DeleteFile(dir + 'special_tokens_map.json');
    // the default <|endoftext|> of unk/eos/pad must exist when none is named
    WriteText(dir, 'tokenizer_config.json', LEGACY);
    WriteText(dir, 'added_tokens.json', '{"<|im_end|>":260}');
    Check(Refused('Qwen2Tokenizer (default)'), 'default <|endoftext|> missing refused');
    DeleteFile(dir + 'added_tokens.json');
    DeleteFile(dir + 'tokenizer_config.json');
    Check(Refused('Qwen2Tokenizer (default)'), 'no config: the defaults apply');
    WriteText(dir, 'tokenizer_config.json', NO_DEFAULTS);
    two := LoadPair(dir, err);
    if not CheckFailed(two <> nil, Utf8ToString(err)) then
      CheckEqual(two.VocabSize, 260, 'defaults set to null: no added token');
    FreeAndNil(two);
    // the added ids must follow the vocabulary without a gap, each new
    WriteText(dir, 'tokenizer_config.json', StringReplaceAll(CONFIG, '"261"', '"262"'));
    Check(Refused('without a gap'), 'a gap in the added ids refused');
    WriteText(dir, 'tokenizer_config.json', StringReplaceAll(CONFIG, '"260"', '"259"'));
    Check(Refused('without a gap'), 'an added id inside the vocabulary refused');
    WriteText(dir, 'tokenizer_config.json', StringReplaceAll(CONFIG,
      '"content":"<|im_end|>"', '"content":"aa"'));
    Check(Refused('is in the vocabulary'), 'an added token of the vocabulary refused');
    // a tokenizer.json beside wins in transformers: refused here
    WriteText(dir, 'tokenizer_config.json', CONFIG);
    WriteText(dir, 'tokenizer.json', '{}');
    Check(Refused('use Create'), 'a tokenizer.json beside refused');
    DeleteFile(dir + 'tokenizer.json');
    WithConfig('"fast_tokenizer_files":["tokenizer.json"],');
    Check(Refused('fast_tokenizer_files'), 'fast_tokenizer_files refused');
    WithConfig('"vocab":{"z":0},');
    Check(Refused('vocab is set'), 'an inline vocabulary refused');
    WithConfig('"merges":[],');
    Check(Refused('merges is set'), 'inline merges refused');
    WithConfig('"add_eos_token":true,');
    Check(Refused('add_eos_token'), 'add_eos_token refused');
    WithConfig('"add_bos_token":true,');
    Check(Refused('add_bos_token'), 'add_bos_token refused');
    WithConfig('"_token":"<|nope|>",');
    Check(Refused('_token of'), 'a key named _token counts');
    WriteText(dir, 'tokenizer_config.json', CONFIG);
    WriteText(dir, 'tiktoken.model', 'x');
    Check(Refused('tiktoken.model'), 'a tiktoken.model beside refused');
    DeleteFile(dir + 'tiktoken.model');
    WriteText(dir, 'tokenizer.model.v3', 'x');
    Check(Refused('tokenizer.model.v3'), 'a tokenizer.model.v3 beside refused');
    DeleteFile(dir + 'tokenizer.model.v3');
    WriteText(dir, 'tokenizer_config.json', StringReplaceAll(CONFIG,
      '"content":"<|endoftext|>"', '"content":"<|im_end|>"'));
    Check(Refused('listed twice'), 'two added tokens with one content refused');
    // what the folder says must be this pipeline
    WriteText(dir, 'tokenizer_config.json', StringReplaceAll(CONFIG,
      'Qwen2Tokenizer', 'GPT2Tokenizer'));
    Check(Refused('GPT2Tokenizer'), 'GPT-2 tokenizer refused');
    WriteText(dir, 'tokenizer_config.json', StringReplaceAll(CONFIG,
      '"add_prefix_space":false', '"add_prefix_space":true'));
    Check(Refused('add_prefix_space'), 'prefix space refused');
    WriteText(dir, 'tokenizer_config.json', StringReplaceAll(CONFIG, DEC,
      '"added_tokens_decoder":[{"content":"<|im_end|>"}]'));
    Check(Refused('not an object'), 'added_tokens_decoder as an array refused');
    // named special tokens must be added ones
    WithConfig('"bos_token":null,"eos_token":"<|im_end|>","additional_special_tokens":' +
      '[{"content":"<|im_end|>"}],"extra_special_tokens":{"x":"<|im_end|>"},');
    CheckUtf8(Loads, 'added special tokens accepted: %', [err]);
    WithConfig('"unk_token":"aa",');
    Check(Refused('a vocabulary entry'), 'a special token of the vocabulary refused');
    WithConfig('"audio_token":{"content":"xy"},');
    Check(Refused('a vocabulary entry'), 'also as an AddedToken object');
    WithConfig('"additional_special_tokens":["<|im_end|>","<|nope|>"],');
    Check(Refused('"<|nope|>", which is no token'), 'a special token that is no token');
    WithConfig('"audio_token":"<|nope|>",');
    Check(Refused('audio_token'), 'a model-specific "..._token" key counts');
    WithConfig('"extra_special_tokens":["<|im_end|>"],"additional_special_tokens":["<|nope|>"],');
    Check(Refused('additional_special_tokens'),
      'additional_special_tokens checked beside extra_special_tokens');
    WithConfig('"extra_special_tokens":{"audio":"<|nope|>"},');
    Check(Refused('extra_special_tokens'), 'extra_special_tokens as an object checked');
    WithConfig('"model_specific_special_tokens":{"audio":"<|nope|>"},');
    Check(Refused('model_specific_special_tokens'), 'model_specific_special_tokens');
    WriteText(dir, 'tokenizer_config.json', CONFIG);
    // broken merges
    WriteText(dir, 'merges.txt', 'a a'#10'b  c'#10);
    Check(Refused('is not "a b"'), 'two spaces refused');
    WriteText(dir, 'merges.txt', 'a a'#10#10'b c'#10);
    Check(Refused('is not "a b"'), 'an empty line refused');
    WriteText(dir, 'merges.txt', 'a a'#10'b c '#10);
    Check(Refused('is not "a b"'), 'a trailing space refused');
    WriteText(dir, 'merges.txt', '#Version: 0.2'#10'a a'#10);
    Check(Refused('outside the vocabulary'), 'the header is case-sensitive');
    WriteText(dir, 'merges.txt', '');
    Check(Refused('cannot read'), 'empty merges.txt refused');
    WriteText(dir, 'merges.txt', MERGES);
    WriteText(dir, 'vocab.json', '[1,2]');
    Check(Refused('no vocabulary'), 'vocab.json no object');
  finally
    two.Free;
    one.Free;
    DirectoryDelete(dir);
  end;
end;

const
  /// Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice at 85e237c12c027371202489a0ec509ded67b5e4b5
  TTS_TOKENIZER_FILES: array[0..2] of TFileName = (
    'vocab.json', 'merges.txt', 'tokenizer_config.json');
  TTS_TOKENIZER_SHA256: array[0..2] of RawUtf8 = (
    'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670');

procedure TTestInferBpe.VocabMergesMatchTokenizerJson;
const
  PAIR: array[0..2] of TFileName = ('vocab.json', 'merges.txt', 'tokenizer_config.json');
var
  dir, tmp: TFileName;
  tok: TBpeTokenizer;
  i, bad: PtrInt;
  added: RawUtf8;
begin
  if not Ready then
    exit;
  dir := FindQwen3Folder('merges.txt');
  if CheckFailed(dir <> '', 'vocab.json and merges.txt beside tokenizer.json') then
    exit;
  // without the tokenizer.json beside them, which CreateFromVocabMerges refuses
  tmp := IncludeTrailingPathDelimiter(TemporaryFileName + '.pair');
  ForceDirectories(tmp);
  tok := nil;
  try
    for i := 0 to high(PAIR) do
      CopyFile(dir + PAIR[i], tmp + PAIR[i], false);
    tok := TBpeTokenizer.CreateFromVocabMerges(tmp + 'vocab.json', tmp + 'merges.txt');
    CheckEqual(tok.VocabSize, fTokenizer.VocabSize, 'vocabulary size');
    CheckEqual(tok.BpeCount, fTokenizer.BpeCount, 'BPE tokens');
    CheckEqual(tok.MergeCount, fTokenizer.MergeCount, 'merges ("#version" skipped)');
    bad := 0;
    for i := 0 to fTokenizer.VocabSize - 1 do
      if tok.TokenBytes(i) <> fTokenizer.TokenBytes(i) then
        inc(bad);
    CheckEqual(bad, 0, 'every token the same bytes');
    bad := 0;
    for i := fTokenizer.BpeCount to fTokenizer.VocabSize - 1 do
    begin
      added := fTokenizer.TokenBytes(i);
      if (added <> '') and (tok.TokenId(added) <> fTokenizer.TokenId(added)) then
        inc(bad);
    end;
    CheckEqual(bad, 0, 'every added token found by its content');
    CheckIds(tok, 'vocab.json + merges.txt');
  finally
    tok.Free;
    DirectoryDelete(tmp);
  end;
  // Qwen3-TTS ships no tokenizer.json: its pair, hash-checked
  dir := FindVendorModel('qwen3-tts-12hz-0.6b-customvoice' + PathDelim + 'vocab.json');
  if dir = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-tts-12hz-0.6b-customvoice/vocab.json');
    AddConsole('SKIP Qwen3-TTS tokenizer: no vendor/models/' +
      'qwen3-tts-12hz-0.6b-customvoice/vocab.json (tools/infer-ref/qwen3-tts)');
    exit;
  end;
  dir := ExtractFilePath(dir);
  for i := 0 to high(TTS_TOKENIZER_FILES) do
    if CheckFailed(IdemPropNameU(HashFileSha256(dir + TTS_TOKENIZER_FILES[i]),
         TTS_TOKENIZER_SHA256[i]), TTS_TOKENIZER_FILES[i] + ' sha256') then
      exit;
  tok := TBpeTokenizer.CreateFromVocabMerges(dir + 'vocab.json', dir + 'merges.txt');
  try
    // the vocabulary and merges of Qwen3, 7 added tokens more
    CheckEqual(tok.BpeCount, fTokenizer.BpeCount, 'TTS: BPE tokens');
    CheckEqual(tok.MergeCount, fTokenizer.MergeCount, 'TTS: merges (no "#version")');
    CheckEqual(tok.VocabSize, 151676, 'TTS: with the 33 added tokens');
    CheckEqual(tok.TokenId('<tts_pad>'), 151671, 'TTS: <tts_pad>');
    CheckEqual(tok.TokenId('<|audio_pad|>'), 151675, 'TTS: <|audio_pad|>');
    CheckEqual(tok.TokenId('<|im_end|>'), 151645, 'TTS: <|im_end|>');
    bad := 0;
    for i := 0 to fTokenizer.VocabSize - 1 do
      if tok.TokenBytes(i) <> fTokenizer.TokenBytes(i) then
        inc(bad);
    CheckEqual(bad, 0, 'TTS: the Qwen3 tokens unchanged');
    CheckIds(tok, 'Qwen3-TTS vocab.json + merges.txt');
  finally
    tok.Free;
  end;
end;

procedure TTestInferBpe.ChatTemplateMatchesReference;
var
  ref: TInferRefSet;
  dir: TFileName;
  cases, c: PDocVariantData;
  ids, offsets: PIntegerArray;
  n, i, first, count: PtrInt;
  mine: TIntegerDynArray;
  msgs: TLlmMessageDynArray;
  tools: TLlmToolDynArray;
  gen, think: boolean;
  text: RawUtf8;
begin
  if not Ready then
    exit;
  dir := FindInferRefDir('qwen3-chat');
  if CheckFailed(dir <> '', 'qwen3-chat found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    cases := ref.Manifest^.A['cases'];
    ids := ref.I32('ids', n);
    offsets := ref.I32('offsets', n);
    for i := 0 to cases^.Count - 1 do
    begin
      c := _Safe(cases^.Values[i]);
      msgs := MessagesOf(c);
      tools := ToolsOf(c);
      gen := c^.B['add_generation_prompt'];
      think := true;
      if c^.GetValueIndex('enable_thinking') >= 0 then
        think := c^.B['enable_thinking'];
      text := fTemplate.Render(msgs, tools, gen, think);
      CheckUtf8(text = c^.U['rendered'], '% rendered%%%', [c^.U['id'], #10, text, #10]);
      mine := fTemplate.Encode(msgs, tools, gen, think);
      first := offsets^[i];
      count := offsets^[i + 1] - first;
      CheckUtf8(SameIds(mine, @ids^[first], count), '% ids % expected %',
        [c^.U['id'], IdsText(pointer(mine), length(mine)), IdsText(@ids^[first], count)]);
    end;
    // a Reasoning is the <think> block, and the content is then not searched
    // for one (the template: reasoning_content is a string) - the reference
    // cannot hold this case, a <think> in the content is an added token there
    SetLength(msgs, 2);
    msgs[0] := LlmMessage(lrUser, 'x');
    msgs[1] := LlmMessage(lrAssistant, '<think>a</think>b');
    msgs[1].Reasoning := 'r';
    CheckEqual(fTemplate.Render(msgs, nil, false, true),
      '<|im_start|>user'#10'x<|im_end|>'#10'<|im_start|>assistant'#10'<think>'#10 +
      'r'#10'</think>'#10#10'<think>a</think>b<|im_end|>'#10, 'reasoning, content kept');
    // a float json.dumps would round, as the parser keeps it: passed on as written
    msgs[1] := LlmMessage(lrAssistant, '');
    SetLength(msgs[1].ToolCalls, 1);
    msgs[1].ToolCalls[0].Name := 'f';
    msgs[1].ToolCalls[0].ArgumentsJson := '{"n": 0.1000000000000000055511}';
    text := fTemplate.Render(msgs, nil, false, true);
    CheckUtf8(PosEx('"arguments": {"n": 0.1000000000000000055511}}', text) > 0,
      'an inexact float: %', [text]);
  finally
    ref.Free;
  end;
end;

procedure TTestInferBpe.Qwen25ChatTemplateMatchesReference;
var
  ref: TInferRefSet;
  dir, fn: TFileName;
  tok: TBpeTokenizer;
  tpl: TQwen2ChatTemplate;
  cases, c: PDocVariantData;
  ids, offsets: PIntegerArray;
  n, i, j, first, count: PtrInt;
  mine: TIntegerDynArray;
  msgs: TLlmMessageDynArray;
  tools: TLlmToolDynArray;
  gen, raised: boolean;
  text: RawUtf8;
begin
  fn := FindVendorModel('qwen2.5-0.5b-instruct' + PathDelim + 'tokenizer.json');
  if fn = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen2.5-0.5b-instruct');
    AddConsole('SKIP Qwen2.5 chat template: no vendor/models/qwen2.5-0.5b-instruct ' +
      '(tools/infer-ref, make_reference_qwen2.py)');
    exit;
  end;
  dir := FindInferRefDir('qwen25-chat');
  if CheckFailed(dir <> '', 'qwen25-chat found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  tok := nil;
  tpl := nil;
  try
    tok := TBpeTokenizer.Create(fn);
    tpl := TQwen2ChatTemplate.Create(tok);
    raised := false;
    try
      TQwen3ChatTemplate.Create(tok).Free;
    except
      on EInferError do
        raised := true;
    end;
    Check(raised, 'the Qwen3 template needs <think>');
    cases := ref.Manifest^.A['cases'];
    ids := ref.I32('ids', n);
    offsets := ref.I32('offsets', n);
    for i := 0 to cases^.Count - 1 do
    begin
      c := _Safe(cases^.Values[i]);
      msgs := MessagesOf(c);
      tools := ToolsOf(c);
      gen := c^.B['add_generation_prompt'];
      // enable_thinking means nothing to Qwen2.5: both give the same
      text := tpl.Render(msgs, tools, gen, i and 1 = 0);
      CheckUtf8(text = c^.U['rendered'], '% rendered%%%', [c^.U['id'], #10, text, #10]);
      // Qwen2.5 knows no reasoning: a Reasoning changes nothing
      for j := 0 to high(msgs) do
        if msgs[j].Role = lrAssistant then
          msgs[j].Reasoning := 'r';
      mine := tpl.Encode(msgs, tools, gen, i and 1 = 1);
      first := offsets^[i];
      count := offsets^[i + 1] - first;
      CheckUtf8(SameIds(mine, @ids^[first], count), '% ids % expected %',
        [c^.U['id'], IdsText(pointer(mine), length(mine)), IdsText(@ids^[first], count)]);
    end;
    // a call without arguments: {} (what tojson writes for an empty object)
    SetLength(msgs, 2);
    msgs[0] := LlmMessage(lrUser, 'x');
    msgs[1] := LlmMessage(lrAssistant, '');
    SetLength(msgs[1].ToolCalls, 1);
    msgs[1].ToolCalls[0].Name := 'now';
    msgs[1].ToolCalls[0].ArgumentsJson := '';
    text := tpl.Render(msgs, nil, false);
    CheckUtf8(PosEx('{"name": "now", "arguments": {}}', text) > 0, 'no arguments: %',
      [text]);
    // a float json.dumps would round, as a parser keeps it: passed on as written
    msgs[1].ToolCalls[0].ArgumentsJson := '{"n": 0.1000000000000000055511}';
    text := tpl.Render(msgs, nil, false);
    CheckUtf8(PosEx('"arguments": {"n": 0.1000000000000000055511}}', text) > 0,
      'an inexact float: %', [text]);
  finally
    tpl.Free;
    tok.Free;
    ref.Free;
  end;
end;


end.
