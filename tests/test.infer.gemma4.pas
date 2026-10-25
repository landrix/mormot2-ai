/// tests for the tokenizer and the chat format of Gemma 4 - phase D, slice D6c
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the gates: tests/data/infer-ref/gemma4-tokens (ids, decoding) and
//   gemma4-chat (the template of the revision rendered by transformers, and
//   the ids), written by tools/infer-ref/make_reference_d6c.py
// - the pipeline, template text, rules, parser and option tests need no
//   model; the ids need vendor/models/gemma-4-26B-A4B-it/tokenizer.json and
//   say SKIP without it, LAI_REQUIRE_MODEL=1 makes that a failure
unit test.infer.gemma4;

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
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.native,
  mormot.ai.infer.tensor,
  mormot.ai.infer.tokenizer,
  mormot.ai.infer.tokenizer.spbpe,
  mormot.ai.infer.chat,
  mormot.ai.infer.model.gemma4,
  test.infer.ref,
  test.infer.bpe,
  test.infer.llm;

type
  /// D6c: the tokenizer of Gemma 4 against the reference
  TTestInferGemma4 = class(TSynTestCase)
  protected
    fTokenizer: TSpBpeTokenizer;
    procedure Setup; override;
    procedure CleanUp; override;
    /// false (and SKIP or a failure) when tokenizer.json is not there
    function Ready: boolean;
    /// every chat case through TGemma4ChatTemplate: the same text (aIds
    // false) or the same ids; a case transformers refused raises
    procedure CheckChat(aTokenizer: TInferTokenizer; aIds: boolean);
  published
    /// U+2581 for a space, the whole text one word, byte fallback, decoding;
    // foreign pipelines refused (no model needed)
    procedure Gemma4TokenizerPipeline;
    /// every reference case: same ids; decoding gives the text back (a U+2581
    // of the input as a space)
    procedure Gemma4TokenizerIds;
    /// sizes, the control tokens, literals stay text
    procedure Gemma4TokenizerTokens;
    /// every chat case of gemma4-chat: the text of transformers (a tiny
    // tokenizer: no model needed)
    procedure Gemma4TemplateText;
    /// every chat case: the ids of the HF tokenizer
    procedure Gemma4TemplateIds;
    /// the refusals and the dictsort table beyond the reference
    procedure Gemma4TemplateRules;
    /// ParseGemma4ToolCall: the dialect of transformers
    procedure Gemma4ToolCallArguments;
    /// TGemma4OutputParser: thought channel, calls, stops, cuts, the prompt
    procedure Gemma4Parser;
    /// preserve_thinking and the registered family
    procedure Gemma4RequestOptions;
  end;

/// the folder of the gemma-4-26B-A4B-it files, '' when aFile is not in it
function FindGemma4Folder(const aFile: TFileName): TFileName;

/// a tokenizer.json of the Gemma 4 pipeline: the 256 byte tokens <0x00> ..
// <0xFF> (id = byte), then aExtraVocab and aMerges, and <|turn> as added token
// 300
function TinySpBpeJson(const aExtraVocab, aMerges: RawUtf8): RawUtf8;

/// a TSpBpeTokenizer from aJson, nil and aError when it is refused
function LoadSpBpe(const aJson: RawUtf8; out aError: RawUtf8): TSpBpeTokenizer;

/// the ids of aText: every one of aControls the tokenizer knows as its id,
// the rest encoded as text
function ControlIds(aTokenizer: TInferTokenizer; const aText: RawUtf8;
  const aControls: array of RawUtf8): TIntegerDynArray;

/// how many of the ids are one of aControls
function ControlCount(aTokenizer: TInferTokenizer; aIds: PIntegerArray;
  aCount: PtrInt; const aControls: array of RawUtf8): PtrInt;

/// the messages of a chat case of a reference (TLlmMessage as the generators
// write it: role, content, tool_call_id, reasoning, name, tool_calls)
function ChatCaseMessages(aCase: PDocVariantData): TLlmMessageDynArray;

/// the tools of a chat case of a reference (name, description, parameters)
function ChatCaseTools(aCase: PDocVariantData): TLlmToolDynArray;


implementation

function FindGemma4Folder(const aFile: TFileName): TFileName;
begin
  result := FindVendorModel('gemma-4-26B-A4B-it' + PathDelim + aFile);
  if result <> '' then
    result := ExtractFilePath(result);
end;

function TinySpBpeJson(const aExtraVocab, aMerges: RawUtf8): RawUtf8;
const
  HEX: array[0..15] of AnsiChar = '0123456789ABCDEF';
var
  b: integer;
begin
  result := '{"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"' +
    SP_SPACE + '"},"pre_tokenizer":{"type":"Split","pattern":{"String":" "},' +
    '"behavior":"MergedWithPrevious","invert":false},"decoder":{"type":"Sequence",' +
    '"decoders":[{"type":"Replace","pattern":{"String":"' + SP_SPACE + '"},' +
    '"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},"model":{"type":"BPE",' +
    '"dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,' +
    '"end_of_word_suffix":null,"fuse_unk":true,"byte_fallback":true,' +
    '"ignore_merges":false,"vocab":{';
  for b := 0 to 255 do
    result := result + FormatUtf8('"<0x%%>":%,', [HEX[b shr 4], HEX[b and 15], b]);
  result := result + aExtraVocab + '},"merges":' + aMerges + '},' +
    '"added_tokens":[{"id":300,"content":"<|turn>","special":true}]}';
end;

function LoadSpBpe(const aJson: RawUtf8; out aError: RawUtf8): TSpBpeTokenizer;
var
  fn: TFileName;
begin
  result := nil;
  aError := '';
  fn := TemporaryFileName;
  FileFromString(aJson, fn);
  try
    try
      result := TSpBpeTokenizer.Create(fn);
    except
      on E: EInferError do
        aError := StringToUtf8(E.Message);
    end;
  finally
    DeleteFile(fn);
  end;
end;

/// raw bytes as RawUtf8 - a #$c3 literal is a UTF-16 character on Unicode
// Delphi
function Bytes(const aBytes: array of byte): RawUtf8;
begin
  FastSetString(result, @aBytes[0], length(aBytes));
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


{ TTestInferGemma4 }

procedure TTestInferGemma4.Setup;
var
  dir: TFileName;
begin
  dir := FindGemma4Folder('tokenizer.json');
  if dir <> '' then
    fTokenizer := TSpBpeTokenizer.Create(dir + 'tokenizer.json');
end;

procedure TTestInferGemma4.CleanUp;
begin
  FreeAndNil(fTokenizer); // re-entrant, as mORMot requires
end;

function TTestInferGemma4.Ready: boolean;
begin
  result := fTokenizer <> nil;
  if result then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/gemma-4-26B-A4B-it/tokenizer.json');
  AddConsole('SKIP Gemma 4 tokenizer: no vendor/models/gemma-4-26B-A4B-it/' +
    'tokenizer.json (revision in tests/data/infer-ref/gemma4-tokens/manifest.json)');
end;

procedure TTestInferGemma4.Gemma4TokenizerPipeline;
var
  t: TSpBpeTokenizer;
  err, vocab, merges, json: RawUtf8;
  ids: TIntegerDynArray;
  refused: boolean;
begin
  // 256 a, 257 b, 258 c, 259 U+2581, 260 ab, 261 U+2581b, 262 bU+2581, 263 aa
  vocab := '"a":256,"b":257,"c":258,"' + SP_SPACE + '":259,"ab":260,"' + SP_SPACE +
    'b":261,"b' + SP_SPACE + '":262,"aa":263';
  merges := '[["a","b"],["' + SP_SPACE + '","b"],["b","' + SP_SPACE + '"],["a","a"]]';
  t := LoadSpBpe(TinySpBpeJson(vocab, merges), err);
  if CheckFailed(t <> nil, Utf8ToString(err)) then
    exit;
  try
    CheckEqual(t.BpeCount, 264, 'BPE tokens');
    CheckEqual(t.VocabSize, 301, 'up to the added token');
    CheckEqual(t.MergeCount, 4, 'merges');
    CheckEqual(t.Normalize('a b  c'), 'a' + SP_SPACE + 'b' + SP_SPACE + SP_SPACE + 'c',
      'every space U+2581');
    ids := t.Encode('ab');
    CheckEqual(IdsText(ids), '[260 ]', 'ab merged');
    ids := t.Encode('ab c');
    CheckEqual(IdsText(ids), '[260 259 258 ]', 'U+2581 alone');
    // one word: a merge crosses the space, where a pre-tokenizer would cut
    ids := t.Encode('b c');
    CheckEqual(IdsText(ids), '[262 258 ]', 'merged across the space');
    ids := t.Encode('a b');
    CheckEqual(IdsText(ids), '[256 261 ]', 'lower rank first');
    ids := t.Encode('aaa');
    CheckEqual(IdsText(ids), '[263 256 ]', 'leftmost on equal rank');
    // byte fallback: e acute and x are no tokens
    ids := t.Encode(Bytes([$c3, $a9, ord('x')]));
    CheckEqual(IdsText(ids), '[195 169 120 ]', 'byte fallback');
    CheckEqual(t.Decode(ids), Bytes([$c3, $a9, ord('x')]), 'bytes decoded');
    ids := t.Encode(Bytes([$f0, $9f, $98, $80]));
    CheckEqual(IdsText(ids), '[240 159 152 128 ]', 'four bytes');
    // invalid UTF-8: one U+FFFD per invalid byte (as SentencePiece), then its
    // bytes - a lone byte, a cut sequence, an encoded surrogate
    ids := t.Encode(Bytes([ord('a'), $ff]));
    CheckEqual(IdsText(ids), '[256 239 191 189 ]', 'invalid byte as U+FFFD');
    ids := t.Encode(Bytes([$e2, $96, ord('a')]));
    CheckEqual(IdsText(ids), '[239 191 189 239 191 189 256 ]', 'cut sequence');
    ids := t.Encode(Bytes([$ed, $a0, $80]));
    CheckEqual(IdsText(ids), '[239 191 189 239 191 189 239 191 189 ]',
      'encoded surrogate');
    // a U+2581 of the text is a space
    ids := t.Encode('a' + SP_SPACE + 'b');
    CheckEqual(IdsText(ids), '[256 261 ]', 'literal U+2581');
    CheckEqual(t.Decode(ids), 'a b', 'decoded as a space');
    CheckEqual(t.TokenBytes(259), ' ', 'U+2581 token');
    CheckEqual(t.TokenBytes(65), 'A', '<0x41>');
    CheckEqual(t.TokenBytes(299), '', 'a gap has no bytes');
    // added tokens are never matched in text
    CheckEqual(t.TokenId('<|turn>'), 300, 'TokenId');
    CheckEqual(t.TokenId('<turn|>'), -1, 'no such token');
    ids := t.Encode('<|turn>');
    CheckEqual(IdsText(ids), '[60 124 116 117 114 110 62 ]', 'literal as text');
    CheckEqual(t.Decode([300]), '<|turn>', 'control token decodes to its content');
    CheckEqual(IdsText(t.Encode('')), '[]', 'empty');
  finally
    t.Free;
  end;
  // a pre_tokenizer of none is the same pipeline
  json := StringReplaceAll(TinySpBpeJson(vocab, merges),
    '{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious",' +
    '"invert":false}', 'null');
  t := LoadSpBpe(json, err);
  Check(t <> nil, 'no pre_tokenizer');
  t.Free;
  // foreign pipelines
  t := LoadSpBpe(StringReplaceAll(TinySpBpeJson(vocab, merges),
    '"content":"' + SP_SPACE + '"}', '"content":"_"}'), err);
  Check((t = nil) and (PosEx('normalizer is not Replace', err) > 0), 'other normalizer');
  t.Free;
  t := LoadSpBpe(StringReplaceAll(TinySpBpeJson(vocab, merges),
    '"invert":false}', '"invert":true}'), err);
  Check((t = nil) and (PosEx('pre_tokenizer', err) > 0), 'inverted Split');
  t.Free;
  t := LoadSpBpe(StringReplaceAll(TinySpBpeJson(vocab, merges),
    '{"type":"Fuse"}', '{"type":"Strip","content":" ","start":1,"stop":0}'), err);
  Check((t = nil) and (PosEx('decoder', err) > 0), 'Strip decoder');
  t.Free;
  t := LoadSpBpe(StringReplaceAll(TinySpBpeJson(vocab, merges),
    '"byte_fallback":true', '"byte_fallback":false'), err);
  Check((t = nil) and (PosEx('byte fallback', err) > 0), 'no byte fallback');
  t.Free;
  t := LoadSpBpe(StringReplaceAll(TinySpBpeJson(vocab, merges),
    '"ignore_merges":false', '"ignore_merges":true'), err);
  Check((t = nil) and (PosEx('byte fallback', err) > 0), 'ignore_merges');
  t.Free;
  t := LoadSpBpe(StringReplaceAll(TinySpBpeJson(vocab, merges),
    '"<0x41>":65,', '"<0x4G>":65,'), err);
  Check((t = nil) and (PosEx('<0x41>', err) > 0), 'a byte token missing');
  t.Free;
  t := LoadSpBpe(TinySpBpeJson(vocab, '[["a","c"]]'), err);
  Check((t = nil) and (PosEx('outside the vocabulary', err) > 0), 'merge without token');
  t.Free;
  t := LoadSpBpe(TinySpBpeJson(vocab, '[["a","b"],["a","b"]]'), err);
  Check((t = nil) and (PosEx('repeats', err) > 0), 'merge twice');
  t.Free;
  t := LoadSpBpe(StringReplaceAll(TinySpBpeJson(vocab, merges), '"added_tokens"',
    '"truncation":{"direction":"Right","max_length":1,"strategy":"LongestFirst",' +
    '"stride":0},"added_tokens"'), err);
  Check((t = nil) and (PosEx('truncation', err) > 0), 'truncation');
  t.Free;
  t := LoadSpBpe(StringReplaceAll(TinySpBpeJson(vocab, merges), '"added_tokens"',
    '"padding":{"strategy":{"Fixed":8},"direction":"Right","pad_id":0},' +
    '"added_tokens"'), err);
  Check((t = nil) and (PosEx('padding', err) > 0), 'padding');
  t.Free;
  t := LoadSpBpe(TinySpBpeJson(vocab + ',"a":264', merges), err);
  Check((t = nil) and (PosEx('two ids', err) > 0), 'a text twice');
  t.Free;
  // a merge of two byte tokens: its token decodes as its literal text
  t := LoadSpBpe(TinySpBpeJson(vocab + ',"<0xC3><0xA9>":264', '[["<0xC3>","<0xA9>"]]'),
    err);
  if not CheckFailed(t <> nil, Utf8ToString(err)) then
  try
    ids := t.Encode(Bytes([$c3, $a9, ord('a')]));
    CheckEqual(IdsText(ids), '[264 256 ]', 'byte tokens merged');
    CheckEqual(t.Decode([264]), '<0xC3><0xA9>', 'as its text');
  finally
    t.Free;
  end;
  // the merges in the older "a b" form
  t := LoadSpBpe(TinySpBpeJson(vocab, '["a b","' + SP_SPACE + ' b"]'), err);
  if not CheckFailed(t <> nil, Utf8ToString(err)) then
  try
    CheckEqual(IdsText(t.Encode('ab b')), '[260 261 ]', '"a b" merges');
  finally
    t.Free;
  end;
  refused := false;
  try
    TSpBpeTokenizer.CreateFromGguf(nil).Free;
  except
    on EInferError do
      refused := true;
  end;
  Check(refused, 'GGUF not supported yet');
end;

procedure TTestInferGemma4.Gemma4TokenizerIds;
var
  ref: TInferRefSet;
  dir: TFileName;
  cases, c: PDocVariantData;
  ids, offsets, textOffsets: PIntegerArray;
  texts: PByteArray;
  n, i, j, first, count, idsBad, decBad: PtrInt;
  mine: TIntegerDynArray;
  text, want, failures: RawUtf8;
  same: boolean;
begin
  if not Ready then
    exit;
  dir := FindInferRefDir('gemma4-tokens');
  if CheckFailed(dir <> '', 'gemma4-tokens found') then
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
      c := _Safe(cases^.Values[i]);
      FastSetString(text, @texts^[textOffsets^[i]], textOffsets^[i + 1] - textOffsets^[i]);
      mine := fTokenizer.Encode(text);
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
          failures := failures + ' ' + c^.U['id'] + ' ' + IdsText(mine);
      end;
      if not c^.GetAsRawUtf8('decoded', want) then
        want := text;
      if fTokenizer.Decode(mine) <> want then
        inc(decBad);
    end;
    CheckUtf8(idsBad = 0, '% of % cases with other ids:%',
      [idsBad, cases^.Count, failures]);
    CheckUtf8(decBad = 0, '% of % cases decode otherwise', [decBad, cases^.Count]);
    CheckUtf8(cases^.Count > 3000, '% cases', [cases^.Count]);
  finally
    ref.Free;
  end;
end;

procedure TTestInferGemma4.Gemma4TokenizerTokens;
const
  CONTROL_TEXT: array[0..13] of RawUtf8 = (
    '<eos>', '<bos>', '<|tool>', '<tool|>', '<|tool_call>', '<tool_call|>',
    '<|tool_response>', '<tool_response|>', '<|"|>', '<|think|>', '<|channel>',
    '<channel|>', '<|turn>', '<turn|>');
  CONTROL_ID: array[0..13] of integer = (
    1, 2, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 106);
var
  ids: TIntegerDynArray;
  i, j: PtrInt;
  fromText: boolean;
  text: RawUtf8;
begin
  if not Ready then
    exit;
  CheckEqual(fTokenizer.BpeCount, 262144, 'BPE tokens');
  CheckEqual(fTokenizer.VocabSize, 262144, 'the added tokens are inside');
  CheckEqual(fTokenizer.MergeCount, 514906, 'merges');
  text := '';
  for i := 0 to high(CONTROL_TEXT) do
  begin
    CheckEqual(fTokenizer.TokenId(CONTROL_TEXT[i]), CONTROL_ID[i], CONTROL_TEXT[i]);
    text := text + CONTROL_TEXT[i] + 'x';
  end;
  ids := fTokenizer.Encode(text);
  fromText := false;
  for i := 0 to high(ids) do
    for j := 0 to high(CONTROL_ID) do
      if ids[i] = CONTROL_ID[j] then
        fromText := true;
  Check(not fromText, 'no control token from text');
  CheckEqual(fTokenizer.Decode(ids), text, 'and decoded back');
  CheckEqual(fTokenizer.Decode([105, 106]), '<|turn><turn|>',
    'control tokens decode to their content');
end;

const
  /// the control tokens of Gemma 4, as TinyGemma4Tokenizer numbers them from 300
  G4_TOKENS: array[0..13] of RawUtf8 = (
    '<bos>', '<eos>', '<|turn>', '<turn|>', '<|think|>', '<|channel>',
    '<channel|>', '<|tool>', '<tool|>', '<|tool_call>', '<tool_call|>',
    '<|tool_response>', '<tool_response|>', '<|"|>');

function TinyGemma4Tokenizer: TSpBpeTokenizer;
var
  added, err: RawUtf8;
  i: PtrInt;
begin
  added := '';
  for i := 0 to high(G4_TOKENS) do
  begin
    if i > 0 then
      added := added + ',';
    added := added + FormatUtf8('{"id":%,"content":%,"special":true}',
      [300 + i, QuotedStrJson(G4_TOKENS[i])]);
  end;
  // a and U+2581: a space decodes as a space, the rest falls back to bytes
  result := LoadSpBpe(StringReplaceAll(TinySpBpeJson('"a":256,"' + SP_SPACE +
    '":257', '[]'),
    '{"id":300,"content":"<|turn>","special":true}', added), err);
  if result = nil then
    EInferError.RaiseUtf8('TinyGemma4Tokenizer: %', [err]);
end;

function ControlIds(aTokenizer: TInferTokenizer; const aText: RawUtf8;
  const aControls: array of RawUtf8): TIntegerDynArray;
var
  n, p, start, i, id, len: PtrInt;
  part: RawUtf8;
begin
  result := nil;
  n := 0;
  len := 0;
  p := 1;
  start := 1;
  while p <= length(aText) do
  begin
    id := -1;
    if aText[p] = '<' then
      for i := 0 to high(aControls) do
        if copy(aText, p, length(aControls[i])) = aControls[i] then
        begin
          id := aTokenizer.TokenId(aControls[i]);
          len := length(aControls[i]);
          break;
        end;
    if id < 0 then
    begin
      inc(p);
      continue;
    end;
    if p > start then
    begin
      part := copy(aText, start, p - start);
      aTokenizer.EncodeAppend(part, result, n);
    end;
    if n = length(result) then
      SetLength(result, n * 2 + 16);
    result[n] := id;
    inc(n);
    inc(p, len);
    start := p;
  end;
  if p > start then
  begin
    part := copy(aText, start, p - start);
    aTokenizer.EncodeAppend(part, result, n);
  end;
  SetLength(result, n);
end;

function ControlCount(aTokenizer: TInferTokenizer; aIds: PIntegerArray;
  aCount: PtrInt; const aControls: array of RawUtf8): PtrInt;
var
  i, j: PtrInt;
begin
  result := 0;
  for i := 0 to aCount - 1 do
    for j := 0 to high(aControls) do
      if aIds^[i] = aTokenizer.TokenId(aControls[j]) then
        inc(result);
end;

function ChatCaseMessages(aCase: PDocVariantData): TLlmMessageDynArray;
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
    result[i].Name := m^.U['name'];
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

function ChatCaseTools(aCase: PDocVariantData): TLlmToolDynArray;
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

function Gemma4Options(aCase: PDocVariantData): TChatTemplateOptions;
var
  o: PDocVariantData;
begin
  o := aCase^.O['options'];
  result := ChatTemplateOptions(o^.B['enable_thinking']);
  result.PreserveThinking := o^.B['preserve_thinking'];
end;

procedure TTestInferGemma4.CheckChat(aTokenizer: TInferTokenizer; aIds: boolean);
var
  ref: TInferRefSet;
  dir: TFileName;
  tpl: TGemma4ChatTemplate;
  cases, c: PDocVariantData;
  ids, offsets: PIntegerArray;
  n, i, j, first, count, bad, refused, literals, literalControls: PtrInt;
  mineControls, hfControls: PtrInt;
  mine: TIntegerDynArray;
  text, failures: RawUtf8;
  raised, same: boolean;
begin
  dir := FindInferRefDir('gemma4-chat');
  if CheckFailed(dir <> '', 'gemma4-chat found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  tpl := TGemma4ChatTemplate.Create(aTokenizer);
  try
    cases := ref.Manifest^.A['cases'];
    ids := ref.I32('ids', n);
    offsets := ref.I32('offsets', n);
    bad := 0;
    refused := 0;
    literals := 0;
    literalControls := 0;
    failures := '';
    for i := 0 to cases^.Count - 1 do
    begin
      c := _Safe(cases^.Values[i]);
      raised := false;
      text := '';
      mine := nil;
      try
        if aIds then
          mine := tpl.EncodeWith(ChatCaseMessages(c), ChatCaseTools(c),
            c^.B['add_generation_prompt'], Gemma4Options(c))
        else
          text := tpl.RenderWith(ChatCaseMessages(c), ChatCaseTools(c),
            c^.B['add_generation_prompt'], Gemma4Options(c));
      except
        on E: EInferError do
        begin
          raised := true;
          text := StringToUtf8(E.Message);
        end;
      end;
      if c^.Exists('error') then
      begin
        same := raised;
        inc(refused);
      end
      else if raised then
        same := false
      else if aIds and c^.B['literal'] then
      begin
        // an added-token literal in a text: the HF ids read it as a control
        // token, ours keep it text - they must give the rendered text back
        // and hold no more control tokens than the HF ids (fewer in all)
        first := offsets^[i];
        count := offsets^[i + 1] - first;
        mineControls := ControlCount(aTokenizer, PIntegerArray(@mine[0]), length(mine),
          G4_TOKENS);
        hfControls := ControlCount(aTokenizer, PIntegerArray(@ids^[first]), count,
          G4_TOKENS);
        same := (aTokenizer.Decode(mine) = c^.U['rendered']) and
          (mineControls <= hfControls);
        inc(literals);
        inc(literalControls, hfControls - mineControls);
      end
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
    if aIds then
    begin
      CheckEqual(literals, ref.Manifest^.I['literals'], 'the literal cases');
      CheckUtf8(literalControls > 0, 'literals kept as text: % control tokens fewer',
        [literalControls]);
    end;
    CheckUtf8(cases^.Count > 400, '% cases', [cases^.Count]);
  finally
    tpl.Free;
    ref.Free;
  end;
end;

procedure TTestInferGemma4.Gemma4TemplateText;
var
  tok: TSpBpeTokenizer;
begin
  // the text needs only the control tokens: a tiny tokenizer, no model
  tok := TinyGemma4Tokenizer;
  try
    CheckChat(tok, false);
  finally
    tok.Free;
  end;
end;

procedure TTestInferGemma4.Gemma4TemplateIds;
begin
  if Ready then
    CheckChat(fTokenizer, true);
end;

function TemplateRaises(aTemplate: TChatTemplate; const aMessages: TLlmMessageDynArray;
  const aTools: TLlmToolDynArray; const aWhat: RawUtf8): boolean;
begin
  result := false;
  try
    aTemplate.RenderWith(aMessages, aTools, true, ChatTemplateOptions(false));
  except
    on E: EInferError do
      result := PosEx(aWhat, StringToUtf8(E.Message)) > 0;
  end;
end;

function OneTool(const aParameters: RawUtf8): TLlmToolDynArray;
begin
  SetLength(result, 1);
  result[0].Name := 'f';
  result[0].Description := 'd';
  result[0].ParametersJson := aParameters;
end;

procedure TTestInferGemma4.Gemma4TemplateRules;
var
  tok: TSpBpeTokenizer;
  tpl: TGemma4ChatTemplate;
  m: TLlmMessageDynArray;
  text: RawUtf8;
begin
  Check(not TGemma4ChatTemplate.DefaultEnableThinking, 'thinks on request only');
  Check(TChatTemplate.DefaultEnableThinking, 'the others think by default');
  tok := TinyGemma4Tokenizer;
  tpl := TGemma4ChatTemplate.Create(tok);
  try
    // a tool message answering nothing: refused, not dropped
    SetLength(m, 2);
    m[0] := LlmMessage(lrUser, 'x');
    m[1] := LlmMessage(lrTool, 'r');
    Check(TemplateRaises(tpl, m, nil, 'answers a tool call'), 'an orphan result');
    // arguments that are no object
    SetLength(m, 3);
    m[1] := LlmMessage(lrAssistant, '');
    SetLength(m[1].ToolCalls, 1);
    m[1].ToolCalls[0].Name := 'f';
    m[1].ToolCalls[0].ArgumentsJson := '[1]';
    m[2] := LlmMessage(lrTool, 'r');
    Check(TemplateRaises(tpl, m, nil, 'no JSON object'), 'array arguments');
    m[1].ToolCalls[0].ArgumentsJson := '{"a":';
    Check(TemplateRaises(tpl, m, nil, 'PythonJson'), 'invalid JSON');
    SetLength(m, 1);
    // schemas the template would write as a Python repr, or could not take
    Check(TemplateRaises(tpl, m, OneTool('[1]'), 'no object'), 'parameters an array');
    Check(TemplateRaises(tpl, m, OneTool('{"properties":[1]}'), 'no object'),
      'properties an array');
    Check(TemplateRaises(tpl, m, OneTool('{"properties":{"a":{"type":"string",' +
      '"description":[1]}}}'), 'Python repr'), 'a list as description');
    Check(TemplateRaises(tpl, m, OneTool('{"properties":{"a":{"type":["string"]}}}'),
      'Python repr'), 'a list as type');
    Check(TemplateRaises(tpl, m, OneTool('{"required":"a"}'), 'no array'),
      'required a string');
    Check(TemplateRaises(tpl, m, OneTool('{"properties":{"a":{"type":"array",' +
      '"items":{"type":5}}}}'), 'neither a string nor an array'), 'an item type number');
    // U+0531 ARMENIAN CAPITAL AYB: no lower() here, its order is not guessed
    Check(TemplateRaises(tpl, m, OneTool('{"properties":{"' + Bytes([$d4, $b1]) +
      '":{}}}'), 'upper case letter'), 'an upper case key outside the table');
    Check(TemplateRaises(tpl, m, OneTool('{"properties":{"' + Bytes([$ce, $bf, $ce, $a3]) +
      '":{}}}'), 'sigma'), 'a capital sigma: final or not by its place');
    Check(TemplateRaises(tpl, m, OneTool('{"properties":{"a":{"type":"' +
      Bytes([$c3, $9f]) + '"}}}'), 'not ASCII'), 'a type that is not ASCII');
    // the lower case of the table: each pair written lower case first - equal
    // in lower case, the stable sort keeps that order; raw code points would
    // put the capital first (Latin-1, Latin Extended-A, Greek, Cyrillic)
    text := tpl.RenderWith(m, OneTool('{"properties":{"' + Bytes([$d0, $b6]) +
      '":{},"' + Bytes([$d0, $96]) + '":{},"' + Bytes([$cf, $89]) + '":{},"' +
      Bytes([$ce, $a9]) + '":{},"' + Bytes([$c5, $82]) + '":{},"' +
      Bytes([$c5, $81]) + '":{},"' + Bytes([$c3, $a4]) + '":{},"' +
      Bytes([$c3, $84]) + '":{}}}'), true, ChatTemplateOptions(false));
    Check(PosEx(Bytes([$c3, $a4]) + ':{type:<|"|><|"|>},' + Bytes([$c3, $84]) +
      ':{type:<|"|><|"|>},' + Bytes([$c5, $82]) + ':{type:<|"|><|"|>},' +
      Bytes([$c5, $81]) + ':{type:<|"|><|"|>},' + Bytes([$cf, $89]) +
      ':{type:<|"|><|"|>},' + Bytes([$ce, $a9]) + ':{type:<|"|><|"|>},' +
      Bytes([$d0, $b6]) + ':{type:<|"|><|"|>},' + Bytes([$d0, $96]), text) > 0,
      'dictsort over the table');
    // U+24B6 CIRCLED LATIN CAPITAL LETTER A, U+2160 ROMAN NUMERAL ONE:
    // lower() maps them, the table does not
    Check(TemplateRaises(tpl, m, OneTool('{"properties":{"' + Bytes([$e2, $92, $b6]) +
      '":{}}}'), 'upper case symbol'), 'a circled capital key');
    Check(TemplateRaises(tpl, m, OneTool('{"properties":{"' + Bytes([$e2, $85, $a0]) +
      '":{}}}'), 'upper case symbol'), 'a Roman numeral key');
  finally
    tpl.Free;
    tok.Free;
  end;
end;

procedure TTestInferGemma4.Gemma4ToolCallArguments;
var
  name, args: RawUtf8;
begin
  Check(ParseGemma4ToolCall('call:get_weather{city:'#1'Ulm'#1',days:2}', name, args),
    'a call');
  CheckEqual(name, 'get_weather', 'name');
  CheckEqual(args, '{"city": "Ulm", "days": 2}', 'json.dumps form');
  Check(ParseGemma4ToolCall('call:f{o:{k:'#1'v'#1',L:[1,'#1'a,b:c'#1',true,null]},' +
    #1'q'#1':1.50}', name, args), 'nested, a quoted key');
  CheckEqual(args, '{"o": {"k": "v", "L": [1, "a,b:c", true, null]}, "q": 1.5}',
    'nested arguments');
  Check(ParseGemma4ToolCall('call:f{}', name, args) and (args = '{}'), 'no arguments');
  Check(ParseGemma4ToolCall('call:f{s:'#1'"\'#10#1'}', name, args), 'escapes');
  CheckEqual(args, '{"s": "\"\\\n"}', 'escaped');
  Check(ParseGemma4ToolCall('call:' + Bytes([$c3, $a4]) + 'x{' + Bytes([$c3, $a4]) +
    ':1}', name, args) and (name = Bytes([$c3, $a4]) + 'x'), 'unicode \w');
  Check(ParseGemma4ToolCall('call:f{a' + Bytes([$e2, $80, $bf]) + 'b:1}', name, args),
    'connector punctuation is \w');
  Check(ParseGemma4ToolCall('call:f{e' + Bytes([$cc, $81]) + Bytes([$e2, $80, $8d]) +
    ':1}', name, args), 'a mark and a join control are \w');
  Check(not ParseGemma4ToolCall('call:f{' + Bytes([$c2, $b2]) + ':1}', name, args),
    'a superscript two is no \w (transformers refuses it)');
  Check(not ParseGemma4ToolCall('call:f{a' + Bytes([$d9, $a3]) + ':1}', name, args),
    'a non-ASCII digit: not taken here, the call stays text');
  Check(not ParseGemma4ToolCall('call:f[1]', name, args), 'no object');
  Check(not ParseGemma4ToolCall('call:f{a:'#1'x}', name, args), 'an open string');
  Check(not ParseGemma4ToolCall('call:f{a b:1}', name, args), 'a key with a space');
  Check(not ParseGemma4ToolCall('call:f{ a:1}', name, args),
    'a key after a space stays unquoted');
  Check(not ParseGemma4ToolCall('call:{a:1}', name, args), 'no name');
  Check(not ParseGemma4ToolCall('cal:f{}', name, args), 'no call:');
  Check(not ParseGemma4ToolCall('call:f{a:1}x', name, args), 'trailing text');
end;

function G4Parsed(aTokenizer: TInferTokenizer; const aPrompt, aText: RawUtf8;
  aCut: boolean; aLog: TDeltaLog = nil): TGemma4OutputParser;
var
  ids: TIntegerDynArray;
  i: PtrInt;
begin
  if aLog = nil then
    result := TGemma4OutputParser.Create(aTokenizer)
  else
    result := TGemma4OutputParser.Create(aTokenizer, aLog.OnDelta);
  result.Prompt(ControlIds(aTokenizer, aPrompt, G4_TOKENS));
  ids := ControlIds(aTokenizer, aText, G4_TOKENS);
  for i := 0 to high(ids) do
    if result.Push(ids[i]) then
      break;
  result.Finish(aCut, 10);
end;

procedure TTestInferGemma4.Gemma4Parser;
const
  OFF = '<|turn>model'#10'<|channel>thought'#10'<channel|>';
  ON = '<|turn>model'#10;
var
  tok: TSpBpeTokenizer;
  p: TGemma4OutputParser;
  log: TDeltaLog;
begin
  tok := TinyGemma4Tokenizer;
  try
    p := G4Parsed(tok, OFF, #10'Die Antwort.'#10'<turn|>ignored', false);
    try
      CheckEqual(p.Response.Content, 'Die Antwort.'#10, 'content, <turn|> stops');
      CheckEqual(p.Reasoning, '', 'no reasoning');
      Check(p.Response.FinishReason = lfrStop, 'stop');
    finally
      p.Free;
    end;
    p := G4Parsed(tok, ON, '<|channel>thought'#10'Denke nach.'#10'<channel|>Ja.<eos>',
      false);
    try
      CheckEqual(p.Reasoning, 'Denke nach.', 'the thought channel');
      CheckEqual(p.Response.Content, 'Ja.', 'then the content');
    finally
      p.Free;
    end;
    // the prompt after tool results with thinking on ends in an open channel
    p := G4Parsed(tok, ON + '<|tool_response>response:f{value:<|"|>x<|"|>}' +
      '<tool_response|><|channel>thought'#10, 'weiter'#10'<channel|>Fertig.<turn|>',
      false);
    try
      CheckEqual(p.Reasoning, 'weiter', 'reasoning from the first token');
      CheckEqual(p.Response.Content, 'Fertig.', 'content after it');
    finally
      p.Free;
    end;
    // a closed channel in the prompt: content
    p := G4Parsed(tok, OFF, 'x<turn|>', false);
    try
      CheckEqual(p.Response.Content, 'x', 'closed channel');
    finally
      p.Free;
    end;
    // calls: text before them is content, <|tool_response> stops
    log := TDeltaLog.Create;
    try
      p := G4Parsed(tok, OFF, 'Moment.'#10'<|tool_call>call:get_weather{city:<|"|>Ulm' +
        '<|"|>}<tool_call|><|tool_call>call:now{}<tool_call|><|tool_response>', false, log);
      try
        CheckEqual(p.Response.Content, 'Moment.', 'content before the calls');
        if not CheckFailed(length(p.Response.ToolCalls) = 2, 'two calls') then
        begin
          CheckEqual(p.Response.ToolCalls[0].Name, 'get_weather', 'name');
          CheckEqual(p.Response.ToolCalls[0].ArgumentsJson, '{"city": "Ulm"}', 'args');
          CheckEqual(p.Response.ToolCalls[1].ArgumentsJson, '{}', 'no args');
        end;
        Check(p.Response.FinishReason = lfrToolCalls, 'finish tool_calls');
      finally
        p.Free;
      end;
      CheckEqual(log.Content, 'Moment.', 'streamed content');
      CheckEqual(log.Reasoning, '', 'no streamed reasoning');
      CheckEqual(length(log.CallNames), 2, 'streamed calls');
      CheckEqual(log.Invalid, 0, 'valid deltas');
    finally
      log.Free;
    end;
    // not a call: the block stays as text, quote tokens as their content
    p := G4Parsed(tok, OFF, '<|tool_call>call:f[<|"|>a<|"|>]<tool_call|><turn|>', false);
    try
      Check(p.Response.ToolCalls = nil, 'an array is no call');
      CheckEqual(p.Response.Content, '<|tool_call>call:f[<|"|>a<|"|>]<tool_call|>',
        'kept as text');
    finally
      p.Free;
    end;
    // a raw #1 in the text of a call could pose as a quote: no call
    p := G4Parsed(tok, OFF, '<|tool_call>call:f{a:'#1'x'#1'}<tool_call|>', false);
    try
      Check(p.Response.ToolCalls = nil, 'a #1 byte refused');
    finally
      p.Free;
    end;
    // cut inside a call and inside the reasoning
    p := G4Parsed(tok, OFF, 'a<|tool_call>call:f{', true);
    try
      CheckEqual(p.Response.Content, 'a<|tool_call>call:f{', 'cut call as text');
      Check(p.Response.FinishReason = lfrLength, 'length');
    finally
      p.Free;
    end;
    p := G4Parsed(tok, ON, '<|channel>thought'#10'halb', true);
    try
      CheckEqual(p.Reasoning, 'halb', 'cut reasoning');
      CheckEqual(p.Response.Content, '', 'no content');
    finally
      p.Free;
    end;
    // the reasoning streamed as it comes
    log := TDeltaLog.Create;
    try
      p := G4Parsed(tok, ON, '<|channel>thought'#10'eins zwei'#10'<channel|>Ja<turn|>',
        false, log);
      p.Free;
      CheckEqual(log.Reasoning, 'eins zwei', 'streamed reasoning');
      CheckEqual(log.Content, 'Ja', 'streamed content after it');
      CheckEqual(log.Invalid, 0, 'valid reasoning deltas');
    finally
      log.Free;
    end;
    // <|tool_response> without a call stops as well
    p := G4Parsed(tok, OFF, 'x<|tool_response>y', false);
    try
      CheckEqual(p.Response.Content, 'x', 'stopped');
      Check(p.Response.FinishReason = lfrStop, 'no call: stop');
    finally
      p.Free;
    end;
    // another channel than thought stays content, as does a cut channel line
    p := G4Parsed(tok, ON, '<|channel>final'#10'Sichtbar<channel|><turn|>', false);
    try
      CheckEqual(p.Reasoning, '', 'no reasoning from another channel');
      CheckEqual(p.Response.Content, '<|channel>final'#10'Sichtbar<channel|>',
        'kept as content');
    finally
      p.Free;
    end;
    p := G4Parsed(tok, ON, '<|channel>Sichtbar', true);
    try
      CheckEqual(p.Response.Content, '<|channel>Sichtbar', 'a cut channel line');
    finally
      p.Free;
    end;
    p := G4Parsed(tok, ON, '<|channel>x<channel|>y<turn|>', false);
    try
      CheckEqual(p.Response.Content, '<|channel>x<channel|>y', 'no newline: text');
    finally
      p.Free;
    end;
    p := G4Parsed(tok, ON, '<|channel><channel|>y<turn|>', false);
    try
      CheckEqual(p.Response.Content, '<|channel><channel|>y', 'an empty channel: text');
    finally
      p.Free;
    end;
    // a broken channel line does not swallow the call after it
    p := G4Parsed(tok, ON, '<|channel>x<|tool_call>call:f{}<tool_call|><turn|>', false);
    try
      CheckEqual(p.Response.Content, '<|channel>x', 'the line as content');
      Check(length(p.Response.ToolCalls) = 1, 'and the call');
    finally
      p.Free;
    end;
    // the thought line arriving in pieces, also when the prompt began it
    p := G4Parsed(tok, ON, '<|channel>tho', false);
    try
      CheckEqual(p.Response.Content, '<|channel>tho', 'a held line ends as content');
    finally
      p.Free;
    end;
    p := G4Parsed(tok, ON, '<|channel>tho<turn|>', false);
    try
      CheckEqual(p.Response.Content, '<|channel>tho', 'a held line before a stop');
    finally
      p.Free;
    end;
    // every control token ends the line: it goes on as in the content
    p := G4Parsed(tok, ON, '<|channel><|channel>thought'#10'r<channel|>c<turn|>', false);
    try
      CheckEqual(p.Reasoning, 'r', '<|channel> twice: the second opens');
      CheckEqual(p.Response.Content, '<|channel>c', 'the first as text');
    finally
      p.Free;
    end;
    p := G4Parsed(tok, ON, '<|channel>a<tool_call|>b<|"|>c<turn|>', false);
    try
      CheckEqual(p.Response.Content, '<|channel>a<tool_call|>b<|"|>c', 'as text');
    finally
      p.Free;
    end;
    p := G4Parsed(tok, ON, '<|channel><|"|>x<turn|>', false);
    try
      CheckEqual(p.Response.Content, '<|channel><|"|>x', 'a quote ends the line');
    finally
      p.Free;
    end;
    p := G4Parsed(tok, ON + '<|channel>tho', 'ught'#10'r<channel|>c<turn|>', false);
    try
      CheckEqual(p.Reasoning, 'r', 'a thought line begun in the prompt');
      CheckEqual(p.Response.Content, 'c', 'content after it');
    finally
      p.Free;
    end;
    // a prompt that ends after another complete channel line: content
    p := G4Parsed(tok, ON + '<|channel>final'#10, 'Sichtbar<turn|>', false);
    try
      CheckEqual(p.Reasoning, '', 'no reasoning after another channel');
      CheckEqual(p.Response.Content, 'Sichtbar', 'content');
    finally
      p.Free;
    end;
    // control tokens elsewhere are text
    p := G4Parsed(tok, OFF, 'a<|turn>b<bos><turn|>', false);
    try
      CheckEqual(p.Response.Content, 'a<|turn>b<bos>', 'control tokens as text');
    finally
      p.Free;
    end;
  finally
    tok.Free;
  end;
end;

function G4Request(const aExtra: RawUtf8): TLlmChatRequest;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  SetLength(result.Messages, 1);
  result.Messages[0] := LlmMessage(lrUser, 'x');
  result.Temperature := -1;
  if aExtra <> '' then
    result.Extra := _JsonFast(aExtra);
end;

procedure TTestInferGemma4.Gemma4RequestOptions;
var
  o: TChatTemplateOptions;
  f: TNativeFamily;
  raised: boolean;
begin
  o := NativeTemplateOptions(G4Request(''), false);
  Check(not o.EnableThinking and not o.PreserveThinking, 'the defaults');
  o := NativeTemplateOptions(G4Request('{"chat_template_kwargs":{"enable_thinking":' +
    'true,"preserve_thinking":true}}'), false);
  Check(o.EnableThinking and o.PreserveThinking, 'chat_template_kwargs');
  o := NativeTemplateOptions(G4Request('{"chat_template_kwargs":{"preserve_thinking":' +
    'null}}'), false);
  Check(not o.PreserveThinking, 'null is no value');
  raised := false;
  try
    NativeTemplateOptions(G4Request('{"chat_template_kwargs":{"preserve_thinking":1}}'),
      false);
  except
    on ELlmClient do
      raised := true;
  end;
  Check(raised, 'preserve_thinking no boolean');
  Check(NativeFamily(TGemma4Model, f) and (f.Tokenizer = TSpBpeTokenizer) and
    (f.Template = TGemma4ChatTemplate) and (f.Parser = TGemma4OutputParser),
    'gemma4 registered');
  Check(NativeFamily(TGemma4TextModel, f) and (f.Template = TGemma4ChatTemplate),
    'gemma4_text registered');
end;


end.
