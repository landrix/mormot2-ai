/// tests for the tokenizer (o200k) and the chat format of gpt-oss - phase D,
// slice D5c
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the gate: tests/data/infer-ref/gpt-oss-tokens (ids, pre-tokenizer pieces,
//   decoding), written by tools/infer-ref/make_reference_d5c.py
// - and of its Harmony chat format: tests/data/infer-ref/gpt-oss-chat (the
//   template of the revision rendered by transformers, and the ids), the
//   output parser, the options of a request, gpt-oss-20b behind ILlmClient
// - the classes, pieces, pipeline, template text, parser and option tests
//   need no model; the ids need vendor/models/gpt-oss-20b/tokenizer.json and
//   say SKIP without it, LAI_REQUIRE_MODEL=1 makes that a failure; the GGUF
//   tokenizer needs vendor/models/gpt-oss-20b-mxfp4.gguf, the chat test the
//   weights in vendor/models/gpt-oss-20b
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
  mormot.core.perf,
  mormot.core.test,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.native,
  mormot.ai.infer.tensor,
  mormot.ai.infer.tokenizer,
  mormot.ai.infer.chat,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.decoder,
  mormot.ai.infer.model.gptoss,
  mormot.ai.infer.sampler,
  mormot.ai.infer.unicode,
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.loader.gguf,
  test.infer.ref,
  test.infer.bpe,
  test.infer.llm;

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
    /// every chat case through THarmonyChatTemplate: the same text (aIds
    // false) or the same ids; a case transformers refused raises
    procedure CheckChat(aTokenizer: TBpeTokenizer; aIds: boolean);
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
    /// every chat case of gpt-oss-chat: the text of transformers, refusals
    // where it refuses (a tiny tokenizer: no model needed)
    procedure HarmonyTemplateText;
    /// every chat case: the ids of the HF tokenizer
    procedure HarmonyTemplateIds;
    /// what the template does beyond the reference: several calls in a turn,
    // today's date, the refusals, identity and effort
    procedure HarmonyTemplateRules;
    /// THarmonyOutputParser: channels, calls, recipients, cuts, the stream
    procedure HarmonyParser;
    /// NativeTemplateOptions and the family of gpt-oss
    procedure HarmonyRequestOptions;
    /// NativeDecode draws only ids of the tokenizer (a padded embedding)
    procedure DecodeDrawsTokenizerIds;
    /// gpt-oss-20b behind ILlmClient: an answer with reasoning, a tool call,
    // the answer after the tool result (skipped without the weights)
    procedure GptOss20bChat;
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
  Check((t = nil) and (PosEx('none of Qwen2, Qwen3.5, o200k', err) > 0),
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

{ Harmony (D5c2) }

const
  /// the control tokens of the tiny Harmony tokenizer, ids 300..
  HARMONY_TOKENS: array[0..7] of RawUtf8 = ('<|start|>', '<|channel|>',
    '<|message|>', '<|end|>', '<|call|>', '<|return|>', '<|constrain|>',
    '<|endoftext|>');

function TinyHarmonyTokenizer: TBpeTokenizer;
var
  added, err: RawUtf8;
  i: PtrInt;
begin
  added := '';
  for i := 0 to high(HARMONY_TOKENS) do
  begin
    if i > 0 then
      added := added + ',';
    added := added + FormatUtf8('{"id":%,"content":"%","special":true}',
      [300 + i, HARMONY_TOKENS[i]]);
  end;
  result := LoadBpe(StringReplaceAll(TinyO200kJson('null', '"xy":256',
    '[["x","y"]]', true), '{"id":300,"content":"<|im_end|>","special":true}',
    added), err);
  if result = nil then
    EInferError.RaiseUtf8('TinyHarmonyTokenizer: %', [err]);
end;

/// the ids of aText: every <|...|> the tokenizer knows as its control token,
// the rest encoded as text
function HarmonyIds(aTokenizer: TBpeTokenizer; const aText: RawUtf8): TIntegerDynArray;
var
  n, p, e, id: PtrInt;
  part: RawUtf8;
begin
  result := nil;
  n := 0;
  p := 1;
  while p <= length(aText) do
  begin
    e := PosEx('<|', aText, p);
    if e = 0 then
      e := length(aText) + 1;
    if e > p then
      aTokenizer.EncodeAppend(copy(aText, p, e - p), result, n);
    if e > length(aText) then
      break;
    p := PosEx('|>', aText, e);
    part := copy(aText, e, p + 2 - e);
    id := aTokenizer.TokenId(part);
    if (p = 0) or (id < 0) then
      EInferError.RaiseUtf8('HarmonyIds: % is no control token', [part]);
    if n = length(result) then
      SetLength(result, n * 2 + 16);
    result[n] := id;
    inc(n);
    p := p + 2;
  end;
  SetLength(result, n);
end;

function HarmonyMessages(aCase: PDocVariantData): TLlmMessageDynArray;
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

function HarmonyTools(aCase: PDocVariantData): TLlmToolDynArray;
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

function HarmonyOptions(aCase: PDocVariantData; const aDate: RawUtf8): TChatTemplateOptions;
var
  o: PDocVariantData;
begin
  result := ChatTemplateOptions(true);
  o := aCase^.O['options'];
  result.ReasoningEffort := o^.U['reasoning_effort'];
  result.ModelIdentity := o^.U['model_identity'];
  result.CurrentDate := aDate;
end;

procedure TTestInferHarmony.CheckChat(aTokenizer: TBpeTokenizer; aIds: boolean);
var
  ref: TInferRefSet;
  dir: TFileName;
  tpl: THarmonyChatTemplate;
  cases, c: PDocVariantData;
  ids, offsets: PIntegerArray;
  n, i, j, first, count, bad, refused: PtrInt;
  mine: TIntegerDynArray;
  text, date, failures: RawUtf8;
  raised, same: boolean;
begin
  dir := FindInferRefDir('gpt-oss-chat');
  if CheckFailed(dir <> '', 'gpt-oss-chat found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  tpl := THarmonyChatTemplate.Create(aTokenizer);
  try
    cases := ref.Manifest^.A['cases'];
    date := ref.Manifest^.U['date'];
    ids := ref.I32('ids', n);
    offsets := ref.I32('offsets', n);
    bad := 0;
    refused := 0;
    failures := '';
    for i := 0 to cases^.Count - 1 do
    begin
      c := _Safe(cases^.Values[i]);
      raised := false;
      text := '';
      mine := nil;
      try
        if aIds then
          mine := tpl.EncodeWith(HarmonyMessages(c), HarmonyTools(c),
            c^.B['add_generation_prompt'], HarmonyOptions(c, date))
        else
          text := tpl.RenderWith(HarmonyMessages(c), HarmonyTools(c),
            c^.B['add_generation_prompt'], HarmonyOptions(c, date));
      except
        on EInferError do
          raised := true;
      end;
      if c^.Exists('error') then
      begin
        same := raised;
        inc(refused);
      end
      else if raised then
        same := false
      else if aIds then
      begin
        first := offsets^[i];
        count := offsets^[i + 1] - first;
        same := length(mine) = count;
        if same then
          for j := 0 to count - 1 do
            if mine[j] <> ids^[first + j] then
              same := false;
      end
      else
        same := text = c^.U['rendered'];
      if not same then
      begin
        inc(bad);
        if bad <= 3 then
          failures := failures + ' ' + c^.U['id'] + ' [' + text + ']';
      end;
    end;
    CheckUtf8(bad = 0, '% of % cases differ:%', [bad, cases^.Count, failures]);
    CheckEqual(refused, ref.Manifest^.I['errors'], 'the refused cases');
  finally
    tpl.Free;
    ref.Free;
  end;
end;

procedure TTestInferHarmony.HarmonyTemplateText;
var
  tok: TBpeTokenizer;
begin
  // the text needs only the control tokens: a tiny tokenizer, no model
  tok := TinyHarmonyTokenizer;
  try
    CheckChat(tok, false);
  finally
    tok.Free;
  end;
end;

procedure TTestInferHarmony.HarmonyTemplateIds;
begin
  if Ready then
    CheckChat(fTokenizer, true);
end;

function Weather: TLlmTool;
begin
  result.Name := 'get_weather';
  result.Description := 'Wetter';
  result.ParametersJson := '{"type":"object","properties":{"city":{"type":"string"}},' +
    '"required":["city"]}';
end;

function Call(const aId, aName, aArgs: RawUtf8): TLlmToolCall;
begin
  result.Id := aId;
  result.Name := aName;
  result.ArgumentsJson := aArgs;
end;

function TemplateRaises(aTemplate: TChatTemplate; const aMessages: TLlmMessageDynArray;
  const aTools: TLlmToolDynArray; const aOptions: TChatTemplateOptions;
  const aWhy: RawUtf8): boolean;
begin
  result := false;
  try
    aTemplate.RenderWith(aMessages, aTools, true, aOptions);
  except
    on E: EInferError do
      result := PosEx(aWhy, StringToUtf8(E.Message)) > 0;
  end;
end;

procedure TTestInferHarmony.HarmonyTemplateRules;
var
  tok: TBpeTokenizer;
  tpl: THarmonyChatTemplate;
  m: TLlmMessageDynArray;
  tools: TLlmToolDynArray;
  o: TChatTemplateOptions;
  text, today: RawUtf8;
begin
  tok := TinyHarmonyTokenizer;
  tpl := THarmonyChatTemplate.Create(tok);
  try
    o := ChatTemplateOptions(true);
    o.CurrentDate := '2026-10-20';
    // two calls in one turn: a message each, results by their id (the
    // template itself writes the first call only)
    SetLength(m, 4);
    m[0] := LlmMessage(lrUser, 'Ulm und Bonn?');
    m[1] := LlmMessage(lrAssistant, '');
    SetLength(m[1].ToolCalls, 2);
    m[1].ToolCalls[0] := Call('c1', 'get_weather', '{"city":"Ulm"}');
    m[1].ToolCalls[1] := Call('c2', 'get_time', '{"city":"Bonn"}');
    m[2] := LlmMessage(lrTool, 'B');
    m[2].ToolCallId := 'c2';
    m[3] := LlmMessage(lrTool, 'A');
    m[3].ToolCallId := 'c1';
    text := tpl.RenderWith(m, nil, true, o);
    CheckEqual(copy(text, PosEx('<|start|>user', text), maxInt),
      '<|start|>user<|message|>Ulm und Bonn?<|end|>' +
      '<|start|>assistant to=functions.get_weather<|channel|>commentary json' +
      '<|message|>{"city": "Ulm"}<|call|>' +
      '<|start|>assistant to=functions.get_time<|channel|>commentary json' +
      '<|message|>{"city": "Bonn"}<|call|>' +
      '<|start|>functions.get_time to=assistant<|channel|>commentary<|message|>"B"<|end|>' +
      '<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>"A"<|end|>' +
      '<|start|>assistant', 'two calls, results by id');
    // reasoning and content before a call (gpt-oss wrote an analysis and a
    // preamble on commentary): the analysis, then the preamble - the template
    // refuses the pair, an agent loop must not break on it
    SetLength(m, 3);
    m[0] := LlmMessage(lrUser, 'Ulm?');
    m[1] := LlmMessage(lrAssistant, 'Ich schaue nach.');
    m[1].Reasoning := 'Wetter abfragen.';
    SetLength(m[1].ToolCalls, 1);
    m[1].ToolCalls[0] := Call('c1', 'get_weather', '{"city":"Ulm"}');
    m[2] := LlmMessage(lrTool, '18');
    m[2].ToolCallId := 'c1';
    text := tpl.RenderWith(m, nil, true, o);
    CheckEqual(copy(text, PosEx('<|start|>user', text), maxInt),
      '<|start|>user<|message|>Ulm?<|end|>' +
      '<|start|>assistant<|channel|>analysis<|message|>Wetter abfragen.<|end|>' +
      '<|start|>assistant<|channel|>commentary<|message|>Ich schaue nach.<|end|>' +
      '<|start|>assistant to=functions.get_weather<|channel|>commentary json' +
      '<|message|>{"city": "Ulm"}<|call|>' +
      '<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>"18"<|end|>' +
      '<|start|>assistant', 'reasoning and preamble');
    // the date of today when none is given
    today := StringToUtf8(FormatDateTime('yyyy"-"mm"-"dd', Now));
    text := tpl.Render(m, nil, true);
    // (a second before midnight, the test may see the next day)
    Check((PosEx('Current date: ' + today, text) > 0) or
      (PosEx('Current date: ' + StringToUtf8(FormatDateTime('yyyy"-"mm"-"dd', Now)),
        text) > 0), 'today');
    // refusals
    SetLength(m, 1);
    m[0] := LlmMessage(lrTool, 'x');
    Check(TemplateRaises(tpl, m, nil, o, 'no assistant turn with calls'),
      'a tool result without a call');
    // after an answer without calls, a tool result has no call (as the template)
    SetLength(m, 5);
    m[0] := LlmMessage(lrUser, 'a');
    m[1] := LlmMessage(lrAssistant, '');
    SetLength(m[1].ToolCalls, 1);
    m[1].ToolCalls[0] := Call('c1', 'f', '{}');
    m[2] := LlmMessage(lrTool, 'r');
    m[3] := LlmMessage(lrAssistant, 'fertig');
    m[4] := LlmMessage(lrTool, 'r2');
    Check(TemplateRaises(tpl, m, nil, o, 'no assistant turn with calls'),
      'a tool result after the answer');
    SetLength(m, 3);
    m[0] := LlmMessage(lrSystem, 'a');
    m[1] := LlmMessage(lrUser, 'b');
    m[2] := LlmMessage(lrSystem, 'c');
    Check(TemplateRaises(tpl, m, nil, o, 'only the first one'), 'a second system message');
    SetLength(m, 1);
    m[0] := LlmMessage(lrUser, 'x');
    o.ReasoningEffort := 'max';
    Check(TemplateRaises(tpl, m, nil, o, 'not low, medium or high'), 'effort max');
    o.ReasoningEffort := 'high';
    o.CurrentDate := '20.10.2026';
    Check(TemplateRaises(tpl, m, nil, o, 'no yyyy-mm-dd'), 'another date format');
    o.CurrentDate := '2026-10-20';
    SetLength(tools, 1);
    tools[0] := Weather;
    tools[0].ParametersJson := '[1]';
    Check(TemplateRaises(tpl, m, tools, o, 'are no object'), 'parameters no object');
    tools[0].ParametersJson := 'null';
    text := tpl.RenderWith(m, tools, true, o);
    Check(PosEx('type get_weather = () => any;', text) > 0,
      'falsy parameters: none, as the template');
    tools[0].ParametersJson := '{"properties":{"a":{}},"required":"a"}';
    Check(TemplateRaises(tpl, m, tools, o, '"required" is no array'),
      'required as a string (Python would test a substring)');
    tools[0].ParametersJson := '{"properties":{"a":{"type":"string","enum":[[1]]}}}';
    Check(TemplateRaises(tpl, m, tools, o, 'Python repr'), 'an array in an enum');
    tools[0].ParametersJson := '{"properties":{"a":{"oneOf":{"x":1}}}}';
    Check(TemplateRaises(tpl, m, tools, o, '"oneOf" is no array'), 'oneOf as an object');
    tools[0].ParametersJson := '{"properties":{"a":{"type":"number","default":1.2345678901234567}}}';
    Check(TemplateRaises(tpl, m, tools, o, '15 significant digits'),
      'a default json.dumps would round');
    // an identity and an effort of the request
    tools := nil;
    o.ModelIdentity := 'Du bist ein Assistent.';
    text := tpl.RenderWith(m, nil, true, o);
    Check(PosEx('<|message|>Du bist ein Assistent.'#10'Knowledge', text) > 0, 'identity');
    Check(PosEx('Reasoning: high'#10, text) > 0, 'effort');
  finally
    tpl.Free;
    tok.Free;
  end;
end;

/// push aText (control tokens marked as <|...|>) into a new parser
function Parsed(aTokenizer: TBpeTokenizer; const aText: RawUtf8; aCut: boolean;
  aLog: TDeltaLog = nil): THarmonyOutputParser;
var
  ids: TIntegerDynArray;
  i: PtrInt;
begin
  ids := HarmonyIds(aTokenizer, aText);
  if aLog = nil then
    result := THarmonyOutputParser.Create(aTokenizer)
  else
    result := THarmonyOutputParser.Create(aTokenizer, aLog.OnDelta);
  for i := 0 to high(ids) do
    if result.Push(ids[i]) then
      break;
  result.Finish(aCut, 10);
end;

procedure TTestInferHarmony.HarmonyParser;
var
  tok: TBpeTokenizer;
  p: THarmonyOutputParser;
  log: TDeltaLog;
  channel, recipient: RawUtf8;
begin
  tok := TinyHarmonyTokenizer;
  try
    // reasoning, then the answer
    p := Parsed(tok, '<|channel|>analysis<|message|>Denke nach.<|end|><|start|>assistant' +
      '<|channel|>final<|message|>Die Antwort.<|return|>', false);
    try
      CheckEqual(p.Reasoning, 'Denke nach.', 'analysis: reasoning');
      CheckEqual(p.Response.Reasoning, p.Reasoning, 'Response.Reasoning');
      CheckEqual(p.Response.Content, 'Die Antwort.', 'final: content');
      Check(p.Response.FinishReason = lfrStop, 'stop');
      Check(p.Response.ToolCalls = nil, 'no call');
    finally
      p.Free;
    end;
    // a call: recipient after the channel, <|constrain|>, json.dumps form
    p := Parsed(tok, '<|channel|>analysis<|message|>Wetter.<|end|><|start|>assistant' +
      '<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>' +
      '{"city":"Berlin","n":[1,2.50]}<|call|>', false);
    try
      if CheckFailed(length(p.Response.ToolCalls) = 1, 'one call') then
        exit;
      CheckEqual(p.Response.ToolCalls[0].Name, 'get_weather', 'name');
      CheckEqual(p.Response.ToolCalls[0].ArgumentsJson, '{"city": "Berlin", "n": [1, 2.5]}',
        'arguments as json.dumps');
      Check(p.Response.ToolCalls[0].Id <> '', 'an id');
      Check(p.Response.FinishReason = lfrToolCalls, 'finish tool_calls');
      CheckEqual(p.Response.Content, '', 'no content');
    finally
      p.Free;
    end;
    // the recipient before the channel, as the template writes it
    p := Parsed(tok, ' to=functions.f<|channel|>commentary json<|message|>{}<|call|>', false);
    try
      Check((length(p.Response.ToolCalls) = 1) and
        (p.Response.ToolCalls[0].Name = 'f') and
        (p.Response.ToolCalls[0].ArgumentsJson = '{}'), 'recipient first');
    finally
      p.Free;
    end;
    // a preamble on commentary, then the call
    p := Parsed(tok, '<|channel|>commentary<|message|>Ich schaue nach.<|end|>' +
      '<|start|>assistant to=functions.f<|channel|>commentary json<|message|>{"a":1}<|call|>',
      false);
    try
      CheckEqual(p.Response.Content, 'Ich schaue nach.', 'preamble: content');
      Check(length(p.Response.ToolCalls) = 1, 'and the call');
    finally
      p.Free;
    end;
    // no JSON object: the body stays visible as content
    p := Parsed(tok, '<|channel|>commentary to=functions.f json<|message|>[1, 2]<|call|>',
      false);
    try
      Check(p.Response.ToolCalls = nil, 'an array is no call');
      CheckEqual(p.Response.Content, '[1, 2]', 'kept as text');
    finally
      p.Free;
    end;
    // a built-in tool of OpenAI: none offered - reasoning, also on commentary
    p := Parsed(tok, '<|channel|>commentary to=browser.search<|message|>{"q":"x"}<|call|>',
      false);
    try
      Check(p.Response.ToolCalls = nil, 'no call to browser');
      CheckEqual(p.Reasoning, '{"q":"x"}', 'to the reasoning');
    finally
      p.Free;
    end;
    // a header cut by <|end|> is forgotten: its recipient makes no call
    p := Parsed(tok, '<|channel|>commentary to=functions.f<|end|><|start|>assistant' +
      '<|channel|>final<|message|>{}<|return|>', false);
    try
      Check(p.Response.ToolCalls = nil, 'no call from a broken header');
      CheckEqual(p.Response.Content, '{}', 'the answer');
    finally
      p.Free;
    end;
    // control tokens inside a body are text; <|endoftext|> stops
    p := Parsed(tok, '<|channel|>final<|message|>a<|channel|>b<|endoftext|>c', false);
    try
      CheckEqual(p.Response.Content, 'a<|channel|>b', 'control token as text');
      CheckEqual(p.Count, 11, 'stopped at <|endoftext|>'); // text: one id per byte
    finally
      p.Free;
    end;
    // cut: the reasoning is given out, an open call stays text
    p := Parsed(tok, '<|channel|>analysis<|message|>halb', true);
    try
      CheckEqual(p.Reasoning, 'halb', 'cut reasoning');
      Check(p.Response.FinishReason = lfrLength, 'length');
    finally
      p.Free;
    end;
    p := Parsed(tok, '<|channel|>commentary to=functions.f<|message|>{"a":', true);
    try
      Check(p.Response.ToolCalls = nil, 'cut call: none');
      CheckEqual(p.Response.Content, '{"a":', 'cut call: text');
    finally
      p.Free;
    end;
    // streamed: the deltas give the same answer
    log := TDeltaLog.Create;
    try
      p := Parsed(tok, '<|channel|>analysis<|message|>Erst denken.<|end|><|start|>' +
        'assistant<|channel|>commentary to=functions.get_weather json<|message|>' +
        '{"city":"Ulm"}<|call|>', false, log);
      try
        CheckEqual(log.Reasoning, p.Reasoning, 'streamed reasoning');
        CheckEqual(log.Content, p.Response.Content, 'streamed content');
        Check((length(log.CallNames) = 1) and (log.CallNames[0] = 'get_weather') and
          (log.CallArgs[0] = '{"city": "Ulm"}'), 'streamed call');
        CheckEqual(log.Finish, 'tool_calls', 'streamed finish');
        CheckEqual(log.Dones, 1, 'one done');
      finally
        p.Free;
      end;
    finally
      log.Free;
    end;
    THarmonyOutputParser.ParseHeader('assistant to=functions.x <|channel|>commentary ' +
      '<|constrain|>json', channel, recipient);
    Check((channel = 'commentary') and (recipient = 'functions.x'), 'header');
    THarmonyOutputParser.ParseHeader('<|channel|><|constrain|>json', channel, recipient);
    Check((channel = '') and (recipient = ''), 'no channel name');
  finally
    tok.Free;
  end;
end;

function Request(const aExtra: RawUtf8): TLlmChatRequest;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  SetLength(result.Messages, 1);
  result.Messages[0] := LlmMessage(lrUser, 'x');
  result.Temperature := -1;
  if aExtra <> '' then
    result.Extra := _JsonFast(aExtra);
end;

function OptionsRaise(const aExtra: RawUtf8): boolean;
begin
  result := false;
  try
    NativeTemplateOptions(Request(aExtra), true, '');
  except
    on ELlmClient do
      result := true;
  end;
end;

procedure TTestInferHarmony.HarmonyRequestOptions;
var
  o: TChatTemplateOptions;
  tok: TBpeTokenizer;
  tpl: THarmonyChatTemplate;
  f: TNativeFamily;
  ids: TIntegerDynArray;
  raised: boolean;
begin
  o := NativeTemplateOptions(Request(''), true, 'low');
  Check(o.EnableThinking and (o.ReasoningEffort = 'low') and (o.ModelIdentity = ''),
    'the defaults');
  o := NativeTemplateOptions(Request('{"reasoning_effort":"high"}'), true, 'low');
  CheckEqual(o.ReasoningEffort, 'high', 'OpenAI: a field of the request');
  o := NativeTemplateOptions(Request('{"chat_template_kwargs":{"reasoning_effort":' +
    '"medium","model_identity":"Ich","enable_thinking":false}}'), true, '');
  Check((o.ReasoningEffort = 'medium') and (o.ModelIdentity = 'Ich') and
    not o.EnableThinking, 'vLLM: chat_template_kwargs');
  o := NativeTemplateOptions(Request('{"reasoning_effort":"low","chat_template_kwargs":' +
    '{"reasoning_effort":"low"}}'), true, '');
  CheckEqual(o.ReasoningEffort, 'low', 'both, the same');
  o := NativeTemplateOptions(Request('{"reasoning_effort":""}'), true, 'high');
  CheckEqual(o.ReasoningEffort, 'high', 'an empty effort: the default');
  Check(OptionsRaise('{"reasoning_effort":"low","chat_template_kwargs":' +
    '{"reasoning_effort":"high"}}'), 'two efforts that differ');
  Check(OptionsRaise('{"reasoning_effort":3}'), 'an effort that is no string');
  Check(OptionsRaise('{"chat_template_kwargs":{"model_identity":true}}'),
    'an identity that is no string');
  // the template refuses an unknown effort: ELlmClient for the request
  tok := TinyHarmonyTokenizer;
  tpl := THarmonyChatTemplate.Create(tok);
  try
    ids := NativeChatPrompt(tpl, Request('{"reasoning_effort":"low"}'), true);
    Check(ids <> nil, 'a prompt');
    raised := false;
    try
      NativeChatPrompt(tpl, Request('{"reasoning_effort":"extreme"}'), true);
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised, 'an unknown effort refused');
  finally
    tpl.Free;
    tok.Free;
  end;
  Check(NativeFamily(TGptOssModel, f) and (f.Tokenizer = TBpeTokenizer) and
    (f.Template = THarmonyChatTemplate) and (f.Parser = THarmonyOutputParser),
    'gpt-oss registered');
end;

type
  /// a tokenizer of Size ids, each one byte of text
  TSizedTokenizer = class(TInferTokenizer)
  public
    Size: integer;
    function Encode(const aText: RawUtf8): TIntegerDynArray; override;
    procedure EncodeAppend(const aText: RawUtf8; var aIds: TIntegerDynArray;
      var aCount: PtrInt); override;
    function TokenId(const aContent: RawUtf8): integer; override;
    function TokenBytes(aId: integer): RawByteString; override;
    function Decode(const aIds: array of integer): RawUtf8; override;
    function VocabSize: integer; override;
  end;

  /// keeps the ids, never stops
  TIdsParser = class(TNativeOutputParser)
  protected
    function Token(aId: integer): boolean; override;
  public
    Ids: TIntegerDynArray;
  end;

function TSizedTokenizer.Encode(const aText: RawUtf8): TIntegerDynArray;
begin
  result := nil;
end;

procedure TSizedTokenizer.EncodeAppend(const aText: RawUtf8;
  var aIds: TIntegerDynArray; var aCount: PtrInt);
begin
end;

function TSizedTokenizer.TokenId(const aContent: RawUtf8): integer;
begin
  result := -1;
end;

function TSizedTokenizer.TokenBytes(aId: integer): RawByteString;
begin
  result := 'x';
end;

function TSizedTokenizer.Decode(const aIds: array of integer): RawUtf8;
begin
  result := '';
end;

function TSizedTokenizer.VocabSize: integer;
begin
  result := Size;
end;

function TIdsParser.Token(aId: integer): boolean;
var
  n: PtrInt;
begin
  n := length(Ids);
  SetLength(Ids, n + 1);
  Ids[n] := aId;
  result := false;
end;

procedure TTestInferHarmony.DecodeDrawsTokenizerIds;
const
  CASE0_IDS: array[0..4] of integer = (82, 83, 13, 15, 45);
var
  dir: TFileName;
  backend: TInferCpuBackend;
  model: TDecoderModel;
  tok: TSizedTokenizer;
  all, cut: TIdsParser;
  prompt: TIntegerDynArray;
  i, above, below: PtrInt;
  agree: boolean;
begin
  dir := FindInferRefDir('tiny-gpt-oss');
  if CheckFailed(dir <> '', 'tiny-gpt-oss found') then
    exit;
  backend := TInferCpuBackend.Create;
  model := nil;
  tok := TSizedTokenizer.Create('');
  all := nil;
  cut := nil;
  try
    model := NewDecoderModel(dir, backend);
    SetLength(prompt, length(CASE0_IDS));
    for i := 0 to high(CASE0_IDS) do
      prompt[i] := CASE0_IDS[i];
    // the 96 rows of the embedding, then only the first 80 as tokens: greedy
    // goes to rows 86, 87 and 91 here (tiny-gpt-oss case 0)
    tok.Size := model.Config.VocabSize;
    all := TIdsParser.Create(tok);
    NativeDecode(model, prompt, 24, all, InferSamplingGreedy);
    tok.Size := 80;
    cut := TIdsParser.Create(tok);
    NativeDecode(model, prompt, 24, cut, InferSamplingGreedy);
    above := 0;
    for i := 0 to high(all.Ids) do
      if all.Ids[i] >= 80 then
        inc(above);
    Check(above > 0, 'the whole embedding: rows beyond 80 drawn');
    below := 0;
    for i := 0 to high(cut.Ids) do
      if cut.Ids[i] < 80 then
        inc(below);
    CheckEqual(below, length(cut.Ids), 'a tokenizer of 80: only its ids');
    // the same draws until the first row beyond the tokenizer
    agree := true;
    i := 0;
    while (i < length(all.Ids)) and (all.Ids[i] < 80) do
    begin
      if cut.Ids[i] <> all.Ids[i] then
        agree := false;
      inc(i);
    end;
    Check(agree and (i > 0), 'the same draws before');
  finally
    cut.Free;
    all.Free;
    tok.Free;
    model.Free;
    backend.Free;
  end;
end;

procedure TTestInferHarmony.GptOss20bChat;
var
  dir: TFileName;
  llm: TNativeLlmClient;
  r: TLlmChatRequest;
  a: TLlmChatResponse;
  timer: TPrecisionTimer;
begin
  {$ifdef CPU32}
  AddConsole('SKIP gpt-oss-20b chat: 13 GB of weights need a 64-bit process');
  exit;
  {$endif CPU32}
  dir := FindGptOssFolder('model.safetensors.index.json');
  if dir = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/gpt-oss-20b');
    AddConsole('SKIP gpt-oss-20b chat: no vendor/models/gpt-oss-20b');
    exit;
  end;
  timer.Start;
  llm := TNativeLlmClient.Create(dir);
  try
    AddConsole(FormatString('gpt-oss-20b client loaded in %', [timer.Stop]));
    Check(llm.Template is THarmonyChatTemplate, 'the Harmony template');
    llm.ReasoningEffort := 'low';
    // a plain question: reasoning, then the answer
    Finalize(r);
    FillCharFast(r, SizeOf(r), 0);
    SetLength(r.Messages, 1);
    r.Messages[0] := LlmMessage(lrUser, 'What is 17 times 23? Answer with the number.');
    r.Temperature := 0;
    r.MaxTokens := 400;
    timer.Start;
    a := llm.ChatComplete(r);
    AddConsole(FormatString('17 x 23 in %: reasoning "%" answer "%" (% tokens)',
      [timer.Stop, copy(_Safe(a.Raw)^.A['choices']^._[0]^.O['message']^.U[
        'reasoning_content'], 1, 120), a.Content, a.Usage.CompletionTokens]));
    Check(a.FinishReason = lfrStop, 'stopped by itself');
    Check(PosEx('391', a.Content) > 0, 'the answer 391');
    Check(_Safe(a.Raw)^.A['choices']^._[0]^.O['message']^.U['reasoning_content'] <> '',
      'with reasoning');
    // a tool: the call to get_weather
    SetLength(r.Tools, 1);
    r.Tools[0] := Weather;
    r.Messages[0] := LlmMessage(lrUser, 'What is the weather in Berlin right now?');
    timer.Start;
    a := llm.ChatComplete(r);
    AddConsole(FormatString('weather in %: % calls, content "%" (% tokens)',
      [timer.Stop, length(a.ToolCalls), a.Content, a.Usage.CompletionTokens]));
    if not CheckFailed(length(a.ToolCalls) = 1, 'one tool call') then
    begin
      CheckEqual(a.ToolCalls[0].Name, 'get_weather', 'the tool');
      CheckUtf8(PosEx('"city": "Berlin"', a.ToolCalls[0].ArgumentsJson) > 0,
        'the city: %', [a.ToolCalls[0].ArgumentsJson]);
      Check(a.FinishReason = lfrToolCalls, 'finish tool_calls');
      // the next turn: the result back, the answer uses it
      SetLength(r.Messages, 3);
      r.Messages[1] := LlmMessage(lrAssistant, a.Content);
      r.Messages[1].ToolCalls := a.ToolCalls;
      r.Messages[2] := LlmMessage(lrTool, '{"temperature_c": 17, "sky": "rain"}');
      r.Messages[2].ToolCallId := a.ToolCalls[0].Id;
      // with the reasoning before the call, as TLlmAgent gives it back: the
      // analysis message of the template
      r.Messages[1].Reasoning := a.Reasoning;
      if not CheckFailed(a.Reasoning <> '', 'reasoning before the call') then
        Check(PosEx('<|channel|>analysis<|message|>' + a.Reasoning + '<|end|>',
          llm.Template.Render(r.Messages, r.Tools, true, true)) > 0,
          'the analysis goes back');
      timer.Start;
      a := llm.ChatComplete(r);
      AddConsole(FormatString('answer in %: "%"', [timer.Stop, a.Content]));
      Check(a.FinishReason = lfrStop, 'answered');
      Check(PosEx('17', a.Content) > 0, 'with the temperature of the tool');
    end;
  finally
    llm.Free;
  end;
end;

end.
