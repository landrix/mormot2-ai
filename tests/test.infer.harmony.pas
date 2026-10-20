/// tests for the tokenizer of gpt-oss (o200k) - phase D, slice D5c
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the gate: tests/data/infer-ref/gpt-oss-tokens (ids, pre-tokenizer pieces,
//   decoding), written by tools/infer-ref/make_reference_d5c.py
// - the classes, pieces and pipeline tests need no model; the ids need
//   vendor/models/gpt-oss-20b/tokenizer.json and say SKIP without it,
//   LAI_REQUIRE_MODEL=1 makes that a failure; the GGUF tokenizer needs
//   vendor/models/gpt-oss-20b-mxfp4.gguf
unit test.infer.harmony;

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
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.infer.unicode,
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.loader.gguf,
  test.infer.ref,
  test.infer.bpe;

type
  /// D5c: the o200k tokenizer of gpt-oss against the reference
  TTestInferHarmony = class(TSynTestCase)
  protected
    fTokenizer: TBpeTokenizer;
    procedure Setup; override;
    procedure CleanUp; override;
    /// false (and SKIP or a failure) when tokenizer.json is not there
    function Ready: boolean;
    /// every reference case through aTokenizer: same ids, decoding
    procedure CheckIds(aTokenizer: TBpeTokenizer; const aLabel: RawUtf8);
  published
    /// the o200k classes: spot checks, and the letters, numbers and white
    // space of the Qwen2 table for every code point
    procedure O200kClasses;
    /// every reference case: the same pre-tokenizer pieces (no model needed)
    procedure O200kSplitMatchesReference;
    /// no normalizer, ignore_merges, the o200k regex; foreign pipelines
    // refused (no model needed)
    procedure O200kPipeline;
    /// every reference case: same ids; decoding gives the text back
    procedure O200kIdsMatchReference;
    /// sizes, the Harmony control tokens, literals stay text
    procedure O200kTokens;
    /// the tokenizer of the GGUF file of gpt-oss-20b = the one of
    // tokenizer.json: every token, the reference ids
    procedure O200kGguf;
  end;

/// the folder of the gpt-oss-20b files, '' when aFile is not in it
function FindGptOssFolder(const aFile: TFileName): TFileName;

/// a byte-level tokenizer.json as TinyBpeJson writes it, with the o200k regex
// and ignore_merges as given
function TinyO200kJson(const aNormalizer, aExtraVocab, aMerges: RawUtf8;
  aIgnoreMerges: boolean): RawUtf8;


implementation

function FindGptOssFolder(const aFile: TFileName): TFileName;
begin
  result := FindVendorModel('gpt-oss-20b' + PathDelim + aFile);
  if result <> '' then
    result := ExtractFilePath(result);
end;

function TinyO200kJson(const aNormalizer, aExtraVocab, aMerges: RawUtf8;
  aIgnoreMerges: boolean): RawUtf8;
begin
  result := StringReplaceAll(TinyBpeJson(aNormalizer, aExtraVocab, aMerges),
    QuotedStrJson(QWEN2_SPLIT_REGEX), QuotedStrJson(O200K_SPLIT_REGEX));
  if aIgnoreMerges then
    result := StringReplaceAll(result, '"ignore_merges":false', '"ignore_merges":true');
end;

function IdsText(const aIds: array of integer): RawUtf8;
var
  i: PtrInt;
begin
  result := '[';
  for i := 0 to high(aIds) do
    result := result + Int32ToUtf8(aIds[i]) + ' ';
  result := result + ']';
end;

function Cps(const aCp: array of cardinal): RawUtf8;
begin
  result := CodePointsToUtf8(@aCp[0], length(aCp));
end;


{ TTestInferHarmony }

procedure TTestInferHarmony.Setup;
var
  dir: TFileName;
begin
  dir := FindGptOssFolder('tokenizer.json');
  if dir <> '' then
    fTokenizer := TBpeTokenizer.Create(dir + 'tokenizer.json');
end;

procedure TTestInferHarmony.CleanUp;
begin
  FreeAndNil(fTokenizer); // re-entrant, as mORMot requires
end;

function TTestInferHarmony.Ready: boolean;
begin
  result := fTokenizer <> nil;
  if result then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/gpt-oss-20b/tokenizer.json');
  AddConsole('SKIP gpt-oss tokenizer: no vendor/models/gpt-oss-20b/tokenizer.json ' +
    '(revision in tests/data/infer-ref/gpt-oss-tokens/manifest.json)');
end;

procedure TTestInferHarmony.O200kClasses;
var
  cp: cardinal;
  o: TO200kClass;
  u: TUnicodeClass;
  bad: integer;
  counts: array[TO200kClass] of integer;
begin
  Check(O200kClassOf(ord('A')) = ocUpper, 'A');
  Check(O200kClassOf(ord('a')) = ocLower, 'a');
  Check(O200kClassOf($c4) = ocUpper, 'A umlaut');
  Check(O200kClassOf($df) = ocLower, 'sharp s');
  Check(O200kClassOf($1c5) = ocUpper, 'title case Dz');
  Check(O200kClassOf($2b0) = ocLetter, 'modifier letter h');
  Check(O200kClassOf($4e00) = ocLetter, 'CJK');
  Check(O200kClassOf($5d0) = ocLetter, 'Hebrew alef');
  Check(O200kClassOf($301) = ocMark, 'combining acute');
  Check(O200kClassOf($93f) = ocMark, 'Devanagari vowel sign');
  Check(O200kClassOf(ord('7')) = ocNumber, '7');
  Check(O200kClassOf($2163) = ocNumber, 'roman numeral four');
  Check(O200kClassOf(10) = ocSpace, 'LF');
  Check(O200kClassOf($3000) = ocSpace, 'ideographic space');
  Check(O200kClassOf(ord('!')) = ocOther, '!');
  Check(O200kClassOf(ord('''')) = ocOther, 'apostrophe');
  Check(O200kClassOf($1f600) = ocOther, 'emoji');
  Check(O200kClassOf($10ffff) = ocOther, 'last code point');
  // the same regex engine: letters, numbers and white space as for Qwen2
  FillCharFast(counts, SizeOf(counts), 0);
  bad := 0;
  for cp := 0 to $10ffff do
  begin
    o := O200kClassOf(cp);
    inc(counts[o]);
    u := UnicodeClassOf(cp);
    case o of
      ocUpper, ocLower, ocLetter:
        if u <> ucLetter then
          inc(bad);
      ocNumber:
        if u <> ucNumber then
          inc(bad);
      ocSpace:
        if u <> ucSpace then
          inc(bad);
    else
      if u <> ucOther then
        inc(bad);
    end;
  end;
  CheckEqual(bad, 0, 'the classes of the Qwen2 table');
  // the counts of the generator (make_reference_d5c.py, surrogates included
  // here as other)
  CheckEqual(counts[ocUpper], 1889, 'upper');
  CheckEqual(counts[ocLower], 2258, 'lower');
  CheckEqual(counts[ocLetter], 136881, 'both');
  CheckEqual(counts[ocMark], 2501, 'marks');
  CheckEqual(counts[ocNumber], 1911, 'numbers');
  CheckEqual(counts[ocSpace], 25, 'white space');
end;

procedure TTestInferHarmony.O200kSplitMatchesReference;
var
  ref: TInferRefSet;
  dir: TFileName;
  tok: TBpeTokenizer;
  err, text, failures, mine, want: RawUtf8;
  cases: PDocVariantData;
  texts: PByteArray;
  textOffsets, pieces, pieceOffsets: PIntegerArray;
  n, i, j, pos, bad: PtrInt;
  parts: TRawUtf8DynArray;
begin
  dir := FindInferRefDir('gpt-oss-tokens');
  if CheckFailed(dir <> '', 'gpt-oss-tokens found') then
    exit;
  // Split needs no vocabulary: a tiny tokenizer of the same pipeline
  tok := LoadBpe(TinyO200kJson('null', '"xy":256', '[["x","y"]]', true), err);
  if CheckFailed(tok <> nil, Utf8ToString(err)) then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    cases := ref.Manifest^.A['cases'];
    texts := ref.U8('texts', n);
    textOffsets := ref.I32('text_offsets', n);
    pieces := ref.I32('pieces', n);
    pieceOffsets := ref.I32('piece_offsets', n);
    bad := 0;
    failures := '';
    for i := 0 to cases^.Count - 1 do
    begin
      FastSetString(text, @texts^[textOffsets^[i]], textOffsets^[i + 1] - textOffsets^[i]);
      parts := tok.Split(text);
      // the pieces as their end offsets, as the reference stores them
      mine := '';
      pos := 0;
      for j := 0 to high(parts) do
      begin
        inc(pos, length(parts[j]));
        mine := mine + Int32ToUtf8(pos) + ' ';
      end;
      want := '';
      for j := pieceOffsets^[i] to pieceOffsets^[i + 1] - 1 do
        want := want + Int32ToUtf8(pieces^[j]) + ' ';
      if mine <> want then
      begin
        inc(bad);
        if bad <= 5 then
          failures := failures + ' ' + _Safe(cases^.Values[i])^.U['id'] + ' [' +
            mine + '] expected [' + want + ']';
      end;
    end;
    CheckUtf8(bad = 0, '% of % cases split otherwise:%', [bad, cases^.Count, failures]);
  finally
    ref.Free;
    tok.Free;
  end;
end;

procedure TTestInferHarmony.O200kPipeline;
var
  t: TBpeTokenizer;
  err: RawUtf8;
  ids: TIntegerDynArray;
  parts: TRawUtf8DynArray;
const
  VOCAB = '"ab":256,"abc":257,"xy":258';
  MERGES = '[["a","b"],["x","y"]]';
begin
  // ignore_merges: "abc" is a token no merge reaches
  t := LoadBpe(TinyO200kJson('null', VOCAB, MERGES, true), err);
  if CheckFailed(t <> nil, Utf8ToString(err)) then
    exit;
  try
    Check(t.PreTokenizer = bptO200k, 'o200k');
    Check(not t.Nfc, 'no normalizer');
    Check(t.IgnoreMerges, 'ignore_merges');
    ids := t.Encode('abc');
    CheckEqual(IdsText(ids), '[257 ]', 'abc taken from the vocabulary');
    ids := t.Encode('abcd');
    CheckEqual(IdsText(ids), '[256 99 100 ]', 'abcd merged');
    ids := t.Encode('xy abc');
    CheckEqual(IdsText(ids), '[258 32 256 99 ]',
      'only a whole piece: " abc" is no token');
    // no NFC: e + combining acute stays two code points
    CheckEqual(t.Normalize(Cps([ord('e'), $301])), Cps([ord('e'), $301]),
      'not composed');
    ids := t.Encode(Cps([$e9]));
    CheckEqual(IdsText(ids), '[195 169 ]', 'precomposed e acute as its bytes');
    ids := t.Encode(Cps([ord('e'), $301]));
    CheckEqual(IdsText(ids), '[101 204 129 ]', 'decomposed: three bytes');
    parts := t.Split('HTTPServer iPhone');
    CheckEqual(RawUtf8ArrayToCsv(parts, '|'), 'HTTPServer| i|Phone', 'case split');
    parts := t.Split('ABC''s 12345');
    CheckEqual(RawUtf8ArrayToCsv(parts, '|'), 'ABC''s| |123|45', 'contraction, numbers');
  finally
    t.Free;
  end;
  t := LoadBpe(TinyO200kJson('null', VOCAB, MERGES, false), err);
  if not CheckFailed(t <> nil, Utf8ToString(err)) then
  try
    ids := t.Encode('abc');
    CheckEqual(IdsText(ids), '[256 99 ]', 'without ignore_merges: BPE alone');
  finally
    t.Free;
  end;
  // the regexes and normalizers in any combination: NFC with o200k
  t := LoadBpe(TinyO200kJson('{"type":"NFC"}', VOCAB, MERGES, false), err);
  if not CheckFailed(t <> nil, Utf8ToString(err)) then
  try
    Check(t.Nfc and (t.PreTokenizer = bptO200k), 'NFC and o200k');
    CheckEqual(t.Normalize(Cps([ord('e'), $301])), Cps([$e9]), 'composed');
  finally
    t.Free;
  end;
  t := LoadBpe(TinyBpeJson('{"type":"NFC"}', VOCAB, MERGES), err);
  if not CheckFailed(t <> nil, Utf8ToString(err)) then
  try
    Check(t.Nfc and (t.PreTokenizer = bptQwen2) and not t.IgnoreMerges, 'Qwen2');
  finally
    t.Free;
  end;
  t := LoadBpe(TinyO200kJson('{"type":"NFKC"}', VOCAB, MERGES, true), err);
  Check((t = nil) and (PosEx('neither NFC nor none', err) > 0), 'NFKC refused');
  t.Free;
  t := LoadBpe(TinyO200kJson('"NFC"', VOCAB, MERGES, true), err);
  Check((t = nil) and (PosEx('neither NFC nor none', err) > 0),
    'a normalizer that is no object refused');
  t.Free;
  t := LoadBpe(StringReplaceAll(TinyO200kJson('null', VOCAB, MERGES, true),
    '\\p{N}{1,3}', '\\p{N}{1,4}'), err);
  Check((t = nil) and (PosEx('neither that of Qwen2 nor o200k', err) > 0),
    'another regex refused');
  t.Free;
end;

procedure TTestInferHarmony.O200kIdsMatchReference;
begin
  if Ready then
    CheckIds(fTokenizer, 'tokenizer.json');
end;

procedure TTestInferHarmony.CheckIds(aTokenizer: TBpeTokenizer;
  const aLabel: RawUtf8);
var
  ref: TInferRefSet;
  dir: TFileName;
  cases: PDocVariantData;
  ids, offsets, textOffsets: PIntegerArray;
  texts: PByteArray;
  n, i, j, first, count, idsBad, decBad: PtrInt;
  mine: TIntegerDynArray;
  text, failures: RawUtf8;
  same: boolean;
begin
  dir := FindInferRefDir('gpt-oss-tokens');
  if CheckFailed(dir <> '', 'gpt-oss-tokens found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  try
    cases := ref.Manifest^.A['cases'];
    ids := ref.I32('ids', n);
    offsets := ref.I32('offsets', n);
    texts := ref.U8('texts', n);
    textOffsets := ref.I32('text_offsets', n);
    idsBad := 0;
    decBad := 0;
    failures := '';
    for i := 0 to cases^.Count - 1 do
    begin
      FastSetString(text, @texts^[textOffsets^[i]], textOffsets^[i + 1] - textOffsets^[i]);
      mine := aTokenizer.Encode(text);
      first := offsets^[i];
      count := offsets^[i + 1] - first;
      same := length(mine) = count;
      if same then
        for j := 0 to count - 1 do
          if mine[j] <> ids^[first + j] then
            same := false;
      if not same then
      begin
        inc(idsBad);
        if idsBad <= 5 then
          failures := failures + ' ' + _Safe(cases^.Values[i])^.U['id'] + ' ' +
            IdsText(mine);
      end;
      // no normalizer: the ids give back the text itself
      if aTokenizer.Decode(mine) <> text then
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

procedure TTestInferHarmony.O200kTokens;
var
  ids: TIntegerDynArray;
  i: PtrInt;
  control: boolean;
begin
  if not Ready then
    exit;
  Check(fTokenizer.PreTokenizer = bptO200k, 'o200k');
  Check(not fTokenizer.Nfc, 'no normalizer');
  Check(fTokenizer.IgnoreMerges, 'ignore_merges');
  CheckEqual(fTokenizer.BpeCount, 199998, 'BPE tokens');
  CheckEqual(fTokenizer.VocabSize, 200019, 'with the added tokens');
  CheckEqual(fTokenizer.MergeCount, 446189, 'merges');
  CheckEqual(fTokenizer.TokenId('<|start|>'), 200006, '<|start|>');
  CheckEqual(fTokenizer.TokenId('<|channel|>'), 200005, '<|channel|>');
  CheckEqual(fTokenizer.TokenId('<|message|>'), 200008, '<|message|>');
  CheckEqual(fTokenizer.TokenId('<|end|>'), 200007, '<|end|>');
  CheckEqual(fTokenizer.TokenId('<|return|>'), 200002, '<|return|>');
  CheckEqual(fTokenizer.TokenId('<|call|>'), 200012, '<|call|>');
  CheckEqual(fTokenizer.TokenId('<|constrain|>'), 200003, '<|constrain|>');
  CheckEqual(fTokenizer.TokenId('<|endoftext|>'), 199999, '<|endoftext|>');
  ids := fTokenizer.Encode('<|start|>user<|message|>x<|end|><|call|>');
  control := false;
  for i := 0 to high(ids) do
    if ids[i] >= fTokenizer.BpeCount then
      control := true;
  Check(not control, 'no control token from text');
  CheckEqual(fTokenizer.Decode(ids), '<|start|>user<|message|>x<|end|><|call|>',
    'and decoded back');
  CheckEqual(fTokenizer.Decode([200006, 200007]), '<|start|><|end|>',
    'control tokens decode to their content');
end;

procedure TTestInferHarmony.O200kGguf;
var
  fn: TFileName;
  g: TGgufFile;
  tok: TBpeTokenizer;
  i, bad: PtrInt;
begin
  if not Ready then
    exit;
  {$ifdef CPU32}
  AddConsole('SKIP gpt-oss GGUF tokenizer: a 13 GB file needs a 64-bit process');
  exit;
  {$endif CPU32}
  fn := FindVendorModel('gpt-oss-20b-mxfp4.gguf');
  if fn = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/gpt-oss-20b-mxfp4.gguf');
    AddConsole('SKIP gpt-oss GGUF tokenizer: no vendor/models/gpt-oss-20b-mxfp4.gguf');
    exit;
  end;
  g := TGgufFile.Create(fn);
  tok := nil;
  try
    tok := TBpeTokenizer.CreateFromGguf(g);
    Check(tok.PreTokenizer = bptO200k, 'gpt-4o: o200k');
    Check(not tok.Nfc and tok.IgnoreMerges, 'as tokenizer.json');
    CheckEqual(tok.BpeCount, fTokenizer.BpeCount, 'BPE tokens');
    CheckEqual(tok.MergeCount, fTokenizer.MergeCount, 'merges');
    // the file pads the vocabulary to the 201088 embedding rows: unused
    // tokens, none of them kept
    CheckEqual(tok.VocabSize, fTokenizer.VocabSize, 'vocabulary size');
    bad := 0;
    for i := 0 to fTokenizer.VocabSize - 1 do
      if tok.TokenBytes(i) <> fTokenizer.TokenBytes(i) then
        inc(bad);
    CheckEqual(bad, 0, 'every token the same bytes');
    CheckEqual(tok.TokenId('<|call|>'), 200012, 'an added token by its content');
    CheckIds(tok, 'GGUF');
  finally
    tok.Free;
    g.Free;
  end;
end;

end.
