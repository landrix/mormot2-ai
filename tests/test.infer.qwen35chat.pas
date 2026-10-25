/// tests for the tokenizer and the chat format of Qwen3.5 - phase D, slice D7c
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the gates: tests/data/infer-ref/qwen35-tokens (ids, NFC text, pre-tokenizer
//   pieces) and qwen35-chat (the template of the revision rendered by
//   transformers, and the ids), written by tools/infer-ref/make_reference_d7c.py
// - the pieces, pipeline, template text, parser and option tests need no
//   model; the ids need vendor/models/qwen3.5-0.8b/tokenizer.json and say SKIP
//   without it, LAI_REQUIRE_MODEL=1 makes that a failure; the chat test needs
//   its weights too (with a conversation going on from a restore point, D7d)
unit test.infer.qwen35chat;

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
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.chat,
  mormot.ai.infer.model.qwen35,
  test.infer.ref,
  test.infer.bpe,
  test.infer.gemma4,
  test.infer.llm;

type
  /// D7c: the tokenizer and the chat format of Qwen3.5 against the reference
  TTestInferQwen35Chat = class(TSynTestCase)
  protected
    fTokenizer: TBpeTokenizer;
    procedure Setup; override;
    procedure CleanUp; override;
    /// false (and SKIP or a failure) when tokenizer.json is not there
    function Ready: boolean;
    /// every chat case through TQwen35ChatTemplate: the same text (aIds
    // false) or the same ids; a case transformers refused raises
    procedure CheckChat(aTokenizer: TInferTokenizer; aIds: boolean);
  published
    /// the Qwen3.5 regex recognized, marks with the letters, NFC; truncation,
    // padding and a vocabulary text twice refused (no model needed)
    procedure Qwen35TokenizerPipeline;
    /// every reference case: the same pre-tokenizer pieces (no model needed)
    procedure Qwen35SplitMatchesReference;
    /// every reference case: same ids; decoding gives the NFC text back
    procedure Qwen35IdsMatchReference;
    /// every chat case of qwen35-chat: the text of transformers (a tiny
    // tokenizer: no model needed)
    procedure Qwen35TemplateText;
    /// every chat case: the ids of the HF tokenizer
    procedure Qwen35TemplateIds;
    /// the refusals, the default without thinking
    procedure Qwen35TemplateRules;
    /// ParseQwen35ToolCall: the layout, the values typed by the schemas
    procedure Qwen35ToolCallArguments;
    /// what the template writes of a call, the parser reads back
    procedure Qwen35CallRoundTrip;
    /// TQwen35OutputParser: reasoning after the prompt, calls, stops, cuts
    procedure Qwen35Parser;
    /// the registered family, the thinking of a request
    procedure Qwen35RequestOptions;
    /// Qwen3.5-0.8B behind ILlmClient: an answer, a tool call and its result
    procedure Qwen35Chat;
  end;


implementation

const
  QWEN35_MODEL = 'qwen3.5-0.8b';
  /// the control tokens of the Qwen3.5 template and parser
  Q35_TOKENS: array[0..8] of RawUtf8 = (
    '<|endoftext|>', '<|im_start|>', '<|im_end|>', '<think>', '</think>',
    '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>');

/// raw bytes as RawUtf8 - a #$cc literal is a UTF-16 character on Unicode
// Delphi
function Q35Raw(const aBytes: array of byte): RawUtf8;
begin
  FastSetString(result, @aBytes[0], length(aBytes));
end;

function FindQwen35Folder(const aFile: TFileName): TFileName;
begin
  result := FindVendorModel(QWEN35_MODEL + PathDelim + aFile);
  if result <> '' then
    result := ExtractFilePath(result);
end;

/// a tokenizer.json of the Qwen3.5 pipeline: NFC, its regex, the 256 byte
// tokens plus aExtraVocab and aMerges, aAdded as the added tokens
function TinyQwen35Json(const aExtraVocab, aMerges, aAdded: RawUtf8): RawUtf8;
begin
  result := StringReplaceAll(StringReplaceAll(
    TinyBpeJson('{"type":"NFC"}', aExtraVocab, aMerges),
    QuotedStrJson(QWEN2_SPLIT_REGEX), QuotedStrJson(QWEN35_SPLIT_REGEX)),
    '{"id":300,"content":"<|im_end|>","special":true}', aAdded);
end;

/// the added tokens of Q35_TOKENS from id 300
function Q35Added: RawUtf8;
var
  i: PtrInt;
begin
  result := '';
  for i := 0 to high(Q35_TOKENS) do
  begin
    if i > 0 then
      result := result + ',';
    result := result + FormatUtf8('{"id":%,"content":%,"special":true}',
      [300 + i, QuotedStrJson(Q35_TOKENS[i])]);
  end;
end;

function TinyQwen35Tokenizer: TBpeTokenizer;
var
  err: RawUtf8;
begin
  result := LoadBpe(TinyQwen35Json('"xy":256', '[["x","y"]]', Q35Added), err);
  if result = nil then
    EInferError.RaiseUtf8('TinyQwen35Tokenizer: %', [err]);
end;


{ TTestInferQwen35Chat }

procedure TTestInferQwen35Chat.Setup;
var
  dir: TFileName;
begin
  dir := FindQwen35Folder('tokenizer.json');
  if dir <> '' then
    fTokenizer := TBpeTokenizer.Create(dir + 'tokenizer.json');
end;

procedure TTestInferQwen35Chat.CleanUp;
begin
  FreeAndNil(fTokenizer);
end;

function TTestInferQwen35Chat.Ready: boolean;
begin
  result := fTokenizer <> nil;
  if result then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/%/tokenizer.json', [QWEN35_MODEL]);
  AddConsole('SKIP Qwen3.5 tokenizer: no vendor/models/' + QWEN35_MODEL +
    '/tokenizer.json (revision in tests/data/infer-ref/qwen35-tokens/manifest.json)');
end;

/// true when LoadBpe refuses aJson with a message holding aWhy
function BpeRefused(const aJson, aWhy: RawUtf8): boolean;
var
  t: TBpeTokenizer;
  err: RawUtf8;
begin
  t := LoadBpe(aJson, err);
  result := (t = nil) and (PosEx(aWhy, err) > 0);
  t.Free;
end;

procedure TTestInferQwen35Chat.Qwen35TokenizerPipeline;
var
  t: TBpeTokenizer;
  parts: TRawUtf8DynArray;
  json, acute, err: RawUtf8;
begin
  acute := Q35Raw([$cc, $81]); // U+0301 COMBINING ACUTE ACCENT
  t := TinyQwen35Tokenizer;
  try
    Check(t.PreTokenizer = bptQwen35, 'the Qwen3.5 regex');
    Check(t.Nfc, 'NFC');
    // a mark joins the letters before it, and starts a word of its own
    parts := t.Split('ab' + acute + 'c d');
    CheckEqual(length(parts), 2, 'marks inside a word');
    if length(parts) = 2 then
      CheckEqual(parts[0], 'ab' + acute + 'c', 'the word with its mark');
    parts := t.Split('1' + acute);
    CheckEqual(length(parts), 2, 'a mark after a digit is a word');
    // one punctuation is the prefix of a word, a run of them is a piece and
    // the mark then leads the word
    CheckEqual(length(t.Split('!' + acute + 'x')), 1, 'punctuation as the prefix');
    parts := t.Split('!!' + acute + 'x');
    CheckEqual(length(parts), 2, 'punctuation, then the mark');
    if length(parts) = 2 then
      CheckEqual(parts[1], acute + 'x', 'the mark leads the word');
    // NFC first: e + combining acute is one code point, one letter
    parts := t.Split('e' + acute);
    if CheckFailed(length(parts) = 1, 'one piece') then
      exit;
    CheckEqual(parts[0], Q35Raw([$c3, $a9]), 'composed');
    CheckEqual(t.TokenId('<think>'), 303, 'the control tokens');
  finally
    t.Free;
  end;
  // a saved truncation or padding changes what encode returns: refused
  json := TinyQwen35Json('"xy":256', '[["x","y"]]', Q35Added);
  Check(BpeRefused(StringReplaceAll(json, '"normalizer":',
    '"truncation":{"max_length":8},"normalizer":'), 'truncation'), 'truncation');
  Check(BpeRefused(StringReplaceAll(json, '"normalizer":',
    '"padding":{"length":8},"normalizer":'), 'padding'), 'padding');
  t := LoadBpe(StringReplaceAll(json, '"normalizer":',
    '"truncation":null,"padding":null,"normalizer":'), err);
  Check(t <> nil, 'null is none');
  t.Free;
  // a vocabulary text twice: the tokenizers library keeps the last id
  Check(BpeRefused(TinyQwen35Json('"xy":256,"xy":257', '[["x","y"]]', Q35Added),
    'two ids'), 'a text twice');
  // another regex is still refused
  Check(BpeRefused(StringReplaceAll(json, 'p{M}]+', 'p{M}]*'), 'Split regex'),
    'a foreign regex');
end;

procedure TTestInferQwen35Chat.Qwen35SplitMatchesReference;
var
  ref: TInferRefSet;
  dir: TFileName;
  tok: TBpeTokenizer;
  text, failures, mine, want: RawUtf8;
  cases: PDocVariantData;
  texts: PByteArray;
  textOffsets, pieces, pieceOffsets: PIntegerArray;
  n, i, j, pos, bad: PtrInt;
  parts: TRawUtf8DynArray;
begin
  dir := FindInferRefDir('qwen35-tokens');
  if CheckFailed(dir <> '', 'qwen35-tokens found') then
    exit;
  // Split needs no vocabulary: a tiny tokenizer of the same pipeline (NFC
  // included - the pieces are those of the normalized text)
  tok := TinyQwen35Tokenizer;
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
    CheckUtf8(ref.Manifest^.I['marks'] > 1000, '% cases with marks',
      [ref.Manifest^.I['marks']]);
  finally
    ref.Free;
    tok.Free;
  end;
end;

procedure TTestInferQwen35Chat.Qwen35IdsMatchReference;
var
  ref: TInferRefSet;
  dir: TFileName;
  cases: PDocVariantData;
  ids, offsets, textOffsets, normOffsets: PIntegerArray;
  texts, norms: PByteArray;
  n, i, j, first, count, idsBad, decBad: PtrInt;
  mine: TIntegerDynArray;
  text, norm, failures: RawUtf8;
  same: boolean;
begin
  if not Ready then
    exit;
  dir := FindInferRefDir('qwen35-tokens');
  if CheckFailed(dir <> '', 'qwen35-tokens found') then
    exit;
  Check(fTokenizer.PreTokenizer = bptQwen35, 'the Qwen3.5 regex');
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
      FastSetString(norm, @norms^[normOffsets^[i]], normOffsets^[i + 1] - normOffsets^[i]);
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
          failures := failures + ' ' + _Safe(cases^.Values[i])^.U['id'];
      end;
      // the ids give back the NFC text
      if fTokenizer.Decode(mine) <> norm then
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

procedure TTestInferQwen35Chat.CheckChat(aTokenizer: TInferTokenizer; aIds: boolean);
var
  ref: TInferRefSet;
  dir: TFileName;
  tpl: TQwen35ChatTemplate;
  cases, c: PDocVariantData;
  ids, offsets: PIntegerArray;
  n, i, j, first, count, bad, refused, literals, literalControls: PtrInt;
  mineControls, hfControls: PtrInt;
  mine: TIntegerDynArray;
  text, failures: RawUtf8;
  raised, same: boolean;
begin
  dir := FindInferRefDir('qwen35-chat');
  if CheckFailed(dir <> '', 'qwen35-chat found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  tpl := TQwen35ChatTemplate.Create(aTokenizer);
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
            c^.B['add_generation_prompt'],
            ChatTemplateOptions(c^.O['options']^.B['enable_thinking']))
        else
          text := tpl.RenderWith(ChatCaseMessages(c), ChatCaseTools(c),
            c^.B['add_generation_prompt'],
            ChatTemplateOptions(c^.O['options']^.B['enable_thinking']));
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
        // and hold no more control tokens than the HF ids
        first := offsets^[i];
        count := offsets^[i + 1] - first;
        mineControls := ControlCount(aTokenizer, PIntegerArray(@mine[0]), length(mine),
          Q35_TOKENS);
        hfControls := ControlCount(aTokenizer, PIntegerArray(@ids^[first]), count,
          Q35_TOKENS);
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
    CheckUtf8(cases^.Count > 200, '% cases', [cases^.Count]);
  finally
    tpl.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen35Chat.Qwen35TemplateText;
var
  tok: TBpeTokenizer;
begin
  // the text needs only the control tokens: a tiny tokenizer, no model
  tok := TinyQwen35Tokenizer;
  try
    CheckChat(tok, false);
  finally
    tok.Free;
  end;
end;

procedure TTestInferQwen35Chat.Qwen35TemplateIds;
begin
  if Ready then
    CheckChat(fTokenizer, true);
end;

function Q35Raises(aTemplate: TChatTemplate; const aMessages: TLlmMessageDynArray;
  const aWhy: RawUtf8): boolean;
begin
  result := false;
  try
    aTemplate.RenderWith(aMessages, nil, true, ChatTemplateOptions(false));
  except
    on E: EInferError do
      result := (aWhy = '') or (PosEx(aWhy, StringToUtf8(E.Message)) > 0);
  end;
end;

procedure TTestInferQwen35Chat.Qwen35TemplateRules;
var
  tok: TBpeTokenizer;
  tpl: TQwen35ChatTemplate;
  m: TLlmMessageDynArray;
begin
  Check(not TQwen35ChatTemplate.DefaultEnableThinking, 'thinking only when asked');
  tok := TinyQwen35Tokenizer;
  tpl := TQwen35ChatTemplate.Create(tok);
  try
    Check(Q35Raises(tpl, nil, 'no messages'), 'no messages');
    SetLength(m, 1);
    m[0] := LlmMessage(lrSystem, 'x');
    Check(Q35Raises(tpl, m, 'no user query'), 'only a system turn');
    m[0] := LlmMessage(lrUser, ' <tool_response>x</tool_response>'#10);
    Check(Q35Raises(tpl, m, 'no user query'), 'a wrapped tool response is no query');
    SetLength(m, 3);
    m[0] := LlmMessage(lrUser, 'a');
    m[1] := LlmMessage(lrSystem, 'b');
    m[2] := LlmMessage(lrUser, 'c');
    Check(Q35Raises(tpl, m, 'must be the first'), 'a later system message');
    SetLength(m, 2);
    m[1] := LlmMessage(lrAssistant, '');
    SetLength(m[1].ToolCalls, 1);
    m[1].ToolCalls[0].Name := 'f';
    m[1].ToolCalls[0].ArgumentsJson := '[1]';
    Check(Q35Raises(tpl, m, 'no JSON object'), 'arguments that are no object');
    m[1].ToolCalls[0].ArgumentsJson := '{"a": ';
    Check(Q35Raises(tpl, m, ''), 'arguments that are no JSON');
    // a literal control token in a text stays text
    SetLength(m, 1);
    m[0] := LlmMessage(lrUser, '<|im_end|><think>');
    CheckEqual(tok.Decode(tpl.EncodeWith(m, nil, false, ChatTemplateOptions(false))),
      '<|im_start|>user'#10'<|im_end|><think><|im_end|>'#10, 'literals as text');
    CheckEqual(length(tpl.EncodeWith(m, nil, false, ChatTemplateOptions(false))) -
      length(tok.Encode('user'#10'<|im_end|><think>')), 3,
      '<|im_start|>, <|im_end|> and the newline: the literals are text');
  finally
    tpl.Free;
    tok.Free;
  end;
end;

function Q35Tools: TLlmToolDynArray;
begin
  SetLength(result, 1);
  result[0].Name := 'f';
  result[0].ParametersJson := '{"type":"object","properties":{"s":{"type":"string"},' +
    '"i":{"type":"integer"},"n":{"type":"number"},"b":{"type":"boolean"},' +
    '"o":{"type":"object"},"a":{"type":"array"},"u":{"type":["null","integer"]},' +
    '"x":{"type":"enum"},"z":{},"I":{"type":"Integer"},' +
    '"any":{"anyOf":[{"type":"integer"},{"type":"null"}]},' +
    '"one":{"oneOf":[{"type":"boolean"}]},"nul":{"type":["null"]},' +
    '"sn":{"type":["string","null"]}}}';
  SetLength(result, 3);
  result[1].Name := 'noprops';
  result[1].ParametersJson := '{"type":"object"}';
  result[2].Name := 'noschema';
end;

/// the arguments ParseQwen35ToolCall gives for aBody ('-' when it refuses)
function Q35Args(const aBody: RawUtf8; const aTools: TLlmToolDynArray): RawUtf8;
var
  name: RawUtf8;
begin
  if not ParseQwen35ToolCall(aBody, aTools, name, result) then
    result := '-';
end;

/// one parameter of f in the layout of the template
function Q35Call(const aKey, aValue: RawUtf8): RawUtf8;
begin
  result := #10'<function=f>'#10'<parameter=' + aKey + '>'#10 + aValue +
    #10'</parameter>'#10'</function>'#10;
end;

procedure TTestInferQwen35Chat.Qwen35ToolCallArguments;
var
  t: TLlmToolDynArray;
  name, args: RawUtf8;
begin
  t := Q35Tools;
  Check(ParseQwen35ToolCall(#10'<function=get_weather>'#10'<parameter=city>'#10'Ulm'#10 +
    '</parameter>'#10'<parameter=unit>'#10'celsius'#10'</parameter>'#10'</function>'#10,
    nil, name, args), 'the layout of the template');
  CheckEqual(name, 'get_weather', 'the name');
  CheckEqual(args, '{"city": "Ulm", "unit": "celsius"}', 'without schema: strings');
  CheckEqual(Q35Args('<function=now></function>', nil), '{}', 'no arguments');
  CheckEqual(Q35Args(' <function=now>'#10' </function> ', nil), '{}', 'white space');
  // the values typed by the schema
  CheckEqual(Q35Args(Q35Call('s', '42'), t), '{"s": "42"}', 'string stays');
  CheckEqual(Q35Args(Q35Call('s', 'null'), t), '{"s": "null"}', 'a string "null"');
  CheckEqual(Q35Args(Q35Call('i', '42'), t), '{"i": 42}', 'integer');
  CheckEqual(Q35Args(Q35Call('i', ' -7 '), t), '{"i": -7}', 'integer, spaced');
  CheckEqual(Q35Args(Q35Call('i', '4.5'), t), '{"i": "4.5"}', 'no integer: string');
  CheckEqual(Q35Args(Q35Call('i', 'NULL'), t), '{"i": null}', 'null of an integer');
  CheckEqual(Q35Args(Q35Call('n', '1.50'), t), '{"n": 1.5}', 'number');
  CheckEqual(Q35Args(Q35Call('n', '1E5'), t), '{"n": 100000.0}', 'number, exponent');
  CheckEqual(Q35Args(Q35Call('n', 'viel'), t), '{"n": "viel"}', 'no number: string');
  CheckEqual(Q35Args(Q35Call('n', '0.1000000000000000055511'), t),
    '{"n": 0.1000000000000000055511}', 'an inexact float kept as written');
  CheckEqual(Q35Args(Q35Call('b', 'True'), t), '{"b": true}', 'boolean, any case');
  CheckEqual(Q35Args(Q35Call('b', 'false'), t), '{"b": false}', 'boolean false');
  CheckEqual(Q35Args(Q35Call('b', 'ja'), t), '{"b": "ja"}', 'no boolean: string');
  CheckEqual(Q35Args(Q35Call('o', '{"k": [1, 2]}'), t), '{"o": {"k": [1, 2]}}', 'object');
  CheckEqual(Q35Args(Q35Call('o', '[1]'), t), '{"o": "[1]"}', 'an array for an object');
  CheckEqual(Q35Args(Q35Call('o', '{bad'), t), '{"o": "{bad"}', 'no JSON: string');
  CheckEqual(Q35Args(Q35Call('a', '[1, "x"]'), t), '{"a": [1, "x"]}', 'array');
  CheckEqual(Q35Args(Q35Call('u', '3'), t), '{"u": 3}', 'a type list: not null');
  CheckEqual(Q35Args(Q35Call('x', '3'), t), '{"x": "3"}', 'an unknown type');
  CheckEqual(Q35Args(Q35Call('z', 'true'), t), '{"z": "true"}', 'no type: string');
  CheckEqual(Q35Args(Q35Call('q', '1'), t), '{"q": "1"}', 'a key without schema');
  CheckEqual(Q35Args(Q35Call('s', 'a'#10'b'#10), t), '{"s": "a\nb\n"}',
    'one newline at each end is the template''s');
  CheckEqual(Q35Args(Q35Call('s', '"zitat" \ </x> ' + Q35Raw([$c3, $bc])), t),
    '{"s": "\"zitat\" \\ </x> ' + Q35Raw([$c3, $bc]) + '"}', 'escapes');
  CheckEqual(Q35Args(#10'<function=f>'#10'<parameter=i>'#10'1'#10'</parameter>'#10 +
    '<parameter=s>'#10'x'#10'</parameter>'#10'</function>', t), '{"i": 1, "s": "x"}',
    'two parameters in their order');
  // other layouts stay text
  CheckEqual(Q35Args('{"name": "f", "arguments": {}}', t), '-', 'the JSON of Qwen3');
  CheckEqual(Q35Args('<function=f>', t), '-', 'no </function>');
  CheckEqual(Q35Args('<function=>'#10'</function>', t), '-', 'no name');
  CheckEqual(Q35Args('<function=a b>'#10'</function>', t), '-', 'a space in the name');
  CheckEqual(Q35Args('<function=f>x</function>', t), '-', 'text between the blocks');
  CheckEqual(Q35Args('<function=f><parameter=i>1</function>', t), '-',
    'no </parameter>');
  CheckEqual(Q35Args('<function=f><parameter=>1</parameter></function>', t), '-',
    'no key');
  CheckEqual(Q35Args('<function=f><parameter=i>1</parameter><parameter=i>2' +
    '</parameter></function>', t), '-', 'a key twice');
  CheckEqual(Q35Args('<function=f></function>x', t), '-', 'text after');
  CheckEqual(Q35Args('<function=f><parameter=a b>1</parameter></function>', t), '-',
    'a space in a key');
  Check(not ParseQwen35ToolCall('<function=f><parameter=i>1</function>', t, name,
    args) and (name = '') and (args = ''), 'nothing out on a refusal');
  // the type names, compound schemas, white space
  CheckEqual(Q35Args(Q35Call('I', '5'), t), '{"I": 5}', 'a type name in any case');
  CheckEqual(Q35Args(Q35Call('any', '42'), t), '{"any": 42}', 'anyOf');
  CheckEqual(Q35Args(Q35Call('any', 'null'), t), '{"any": null}', 'anyOf with null');
  CheckEqual(Q35Args(Q35Call('one', 'TRUE'), t), '{"one": true}', 'oneOf');
  CheckEqual(Q35Args(Q35Call('nul', 'Null'), t), '{"nul": null}', 'only null');
  CheckEqual(Q35Args(Q35Call('nul', 'x'), t), '{"nul": "x"}', 'only null, a text');
  CheckEqual(Q35Args(Q35Call('sn', 'null'), t), '{"sn": null}', 'a string or null');
  CheckEqual(Q35Args(Q35Call('sn', ' a '), t), '{"sn": " a "}', 'a string as written');
  CheckEqual(Q35Args(Q35Call('b', ' true '), t), '{"b": true}', 'a spaced boolean');
  CheckEqual(Q35Args(Q35Call('i', 'None'), t), '{"i": null}', 'None of the template');
  CheckEqual(Q35Args(Q35Call('i', '007'), t), '{"i": "007"}', 'no JSON integer');
  CheckEqual(Q35Args(Q35Call('i', '+5'), t), '{"i": "+5"}', 'no JSON integer, signed');
  CheckEqual(Q35Args(#10'<function=noprops>'#10'<parameter=i>'#10'1'#10'</parameter>' +
    #10'</function>', t), '{"i": "1"}', 'a schema without properties');
  CheckEqual(Q35Args(#10'<function=noschema>'#10'<parameter=i>'#10'1'#10'</parameter>' +
    #10'</function>', t), '{"i": "1"}', 'a tool without schema');
  // a value may hold </parameter>: it ends where a parameter or the end follows
  CheckEqual(Q35Args(Q35Call('s', 'a</parameter>b'), t), '{"s": "a</parameter>b"}',
    '</parameter> inside a value');
  CheckEqual(Q35Args(#10'<function=f>'#10'<parameter=s>'#10'x</parameter> y'#10 +
    '</parameter>'#10'<parameter=i>'#10'2'#10'</parameter>'#10'</function>', t),
    '{"s": "x</parameter> y", "i": 2}', 'and another parameter after it');
end;

/// the arguments of the first call of aArgs, written by the template and read
// back by the parser
function Q35RoundTrip(aTemplate: TChatTemplate; const aArgs: RawUtf8;
  const aTools: TLlmToolDynArray): RawUtf8;
var
  m: TLlmMessageDynArray;
  text, name: RawUtf8;
  i, j: PtrInt;
begin
  SetLength(m, 2);
  m[0] := LlmMessage(lrUser, 'x');
  m[1] := LlmMessage(lrAssistant, '');
  SetLength(m[1].ToolCalls, 1);
  m[1].ToolCalls[0].Name := 'f';
  m[1].ToolCalls[0].ArgumentsJson := aArgs;
  text := aTemplate.RenderWith(m, nil, false, ChatTemplateOptions(false));
  i := PosEx('<tool_call>', text);
  j := PosEx('</tool_call>', text, i);
  if (i = 0) or (j = 0) or not ParseQwen35ToolCall(copy(text, i + 11, j - i - 11),
       aTools, name, result) then
    result := '-';
end;

procedure TTestInferQwen35Chat.Qwen35CallRoundTrip;
var
  tok: TBpeTokenizer;
  tpl: TQwen35ChatTemplate;
begin
  tok := TinyQwen35Tokenizer;
  tpl := TQwen35ChatTemplate.Create(tok);
  try
    // Python str() of the values, typed back by the schema; None is null
    CheckEqual(Q35RoundTrip(tpl, '{"s": "a</parameter>b", "i": 3, "b": true, ' +
      '"o": {"k": [1, 2]}, "a": [1, "x"], "n": 1.5, "u": null}', Q35Tools),
      '{"s": "a</parameter>b", "i": 3, "b": true, "o": {"k": [1, 2]}, ' +
      '"a": [1, "x"], "n": 1.5, "u": null}', 'typed values');
    CheckEqual(Q35RoundTrip(tpl, '{"s": "x\ny\n", "n": 1E5}', Q35Tools),
      '{"s": "x\ny\n", "n": 100000.0}', 'newlines and an exponent');
    // a float json.dumps would round: kept as written both ways
    CheckEqual(Q35RoundTrip(tpl, '{"n": 0.1000000000000000055511, ' +
      '"o": {"k": [1.2345678901234567]}}', Q35Tools),
      '{"n": 0.1000000000000000055511, "o": {"k": [1.2345678901234567]}}',
      'inexact floats');
    CheckEqual(Q35RoundTrip(tpl, '{}', Q35Tools), '{}', 'no arguments');
    // the tags inside an object: it ends where it is JSON
    CheckEqual(Q35RoundTrip(tpl, '{"o": {"t": "a</parameter><parameter=x>b"}, ' +
      '"i": 1}', Q35Tools), '{"o": {"t": "a</parameter><parameter=x>b"}, "i": 1}',
      'an object holding the tags');
  finally
    tpl.Free;
    tok.Free;
  end;
end;

/// the parser after aPrompt, fed aText (control tokens as their ids)
function Q35Parsed(aTokenizer: TInferTokenizer; const aPrompt, aText: RawUtf8;
  aCut: boolean; aLog: TDeltaLog = nil): TQwen35OutputParser;
var
  ids: TIntegerDynArray;
  i: PtrInt;
begin
  if aLog = nil then
    result := TQwen35OutputParser.Create(aTokenizer)
  else
    result := TQwen35OutputParser.Create(aTokenizer, aLog.OnDelta);
  result.Tools := Q35Tools;
  result.Prompt(ControlIds(aTokenizer, aPrompt, Q35_TOKENS));
  ids := ControlIds(aTokenizer, aText, Q35_TOKENS);
  for i := 0 to high(ids) do
    if result.Push(ids[i]) then
      break;
  result.Finish(aCut, 10);
end;

procedure TTestInferQwen35Chat.Qwen35Parser;
const
  OFF = '<|im_start|>assistant'#10'<think>'#10#10'</think>'#10#10;
  ON = '<|im_start|>assistant'#10'<think>'#10;
var
  tok: TBpeTokenizer;
  p: TQwen35OutputParser;
  log: TDeltaLog;
begin
  tok := TinyQwen35Tokenizer;
  try
    p := Q35Parsed(tok, OFF, 'Die Antwort.'#10'<|im_end|>ignored', false);
    try
      CheckEqual(p.Response.Content, 'Die Antwort.'#10, 'content, <|im_end|> stops');
      CheckEqual(p.Reasoning, '', 'no reasoning without thinking');
      Check(p.Response.FinishReason = lfrStop, 'stop');
    finally
      p.Free;
    end;
    p := Q35Parsed(tok, ON, 'Denke nach.'#10'</think>'#10#10'Ja.<|im_end|>', false);
    try
      CheckEqual(p.Reasoning, 'Denke nach.', 'the prompt opened the reasoning');
      CheckEqual(p.Response.Content, 'Ja.', 'then the content');
    finally
      p.Free;
    end;
    // a prompt without the block: the model may open it itself
    p := Q35Parsed(tok, '<|im_start|>assistant'#10, '<think>r</think>a<|im_end|>', false);
    try
      CheckEqual(p.Reasoning, 'r', 'a <think> the model writes');
      CheckEqual(p.Response.Content, 'a', 'content');
    finally
      p.Free;
    end;
    // a cut reasoning stays reasoning
    p := Q35Parsed(tok, ON, 'lange', true);
    try
      CheckEqual(p.Reasoning, 'lange', 'cut reasoning');
      Check(p.Response.FinishReason = lfrLength, 'cut');
    finally
      p.Free;
    end;
    // calls: text before them is content, typed by the schema
    log := TDeltaLog.Create;
    try
      p := Q35Parsed(tok, OFF, 'Moment.'#10#10'<tool_call>'#10'<function=f>'#10 +
        '<parameter=i>'#10'3'#10'</parameter>'#10'</function>'#10'</tool_call>'#10 +
        '<tool_call>'#10'<function=now>'#10'</function>'#10'</tool_call><|im_end|>',
        false, log);
      try
        CheckEqual(p.Response.Content, 'Moment.', 'content before the calls');
        if not CheckFailed(length(p.Response.ToolCalls) = 2, 'two calls') then
        begin
          CheckEqual(p.Response.ToolCalls[0].Name, 'f', 'name');
          CheckEqual(p.Response.ToolCalls[0].ArgumentsJson, '{"i": 3}', 'typed');
          CheckEqual(p.Response.ToolCalls[1].Name, 'now', 'second');
          CheckEqual(p.Response.ToolCalls[1].ArgumentsJson, '{}', 'no args');
        end;
        Check(p.Response.FinishReason = lfrToolCalls, 'finish tool_calls');
      finally
        p.Free;
      end;
      CheckEqual(log.Finish, 'tool_calls', 'streamed finish');
      CheckEqual(log.Content, 'Moment.', 'streamed content');
      CheckEqual(length(log.CallArgs), 2, 'streamed calls');
      CheckEqual(log.Invalid, 0, 'valid deltas');
    finally
      log.Free;
    end;
    // a block that is no call stays text; a cut call too
    p := Q35Parsed(tok, OFF, '<tool_call>{"name": "f"}</tool_call><|im_end|>', false);
    try
      Check(p.Response.ToolCalls = nil, 'no call');
      CheckEqual(p.Response.Content, '<tool_call>{"name": "f"}</tool_call>',
        'the block as text');
    finally
      p.Free;
    end;
    p := Q35Parsed(tok, OFF, '<tool_call>'#10'<function=f>', true);
    try
      Check(p.Response.ToolCalls = nil, 'a cut call');
      CheckEqual(p.Response.Content, '<tool_call>'#10'<function=f>', 'as text');
    finally
      p.Free;
    end;
    p := Q35Parsed(tok, OFF, 'x<|endoftext|>y', false);
    try
      CheckEqual(p.Response.Content, 'x', '<|endoftext|> stops');
    finally
      p.Free;
    end;
  finally
    tok.Free;
  end;
end;

procedure TTestInferQwen35Chat.Qwen35RequestOptions;
var
  f: TNativeFamily;
  tok: TBpeTokenizer;
  tpl: TQwen35ChatTemplate;
  r: TLlmChatRequest;
  ids: TIntegerDynArray;
begin
  Check(NativeFamily(TQwen35Model, f), 'qwen3_5 registered');
  Check((f.Template = TQwen35ChatTemplate) and (f.Parser = TQwen35OutputParser) and
    (f.Tokenizer = TBpeTokenizer), 'its classes');
  Check(NativeFamily(TQwen35TextModel, f) and (f.Template = TQwen35ChatTemplate),
    'qwen3_5_text registered');
  tok := TinyQwen35Tokenizer;
  tpl := TQwen35ChatTemplate.Create(tok);
  try
    Finalize(r);
    FillCharFast(r, SizeOf(r), 0);
    SetLength(r.Messages, 1);
    r.Messages[0] := LlmMessage(lrUser, 'x');
    ids := NativeChatPrompt(tpl, r, tpl.DefaultEnableThinking);
    CheckEqual(tok.Decode(ids), '<|im_start|>user'#10'x<|im_end|>'#10 +
      '<|im_start|>assistant'#10'<think>'#10#10'</think>'#10#10, 'thinking off');
    r.Extra := _ObjFast(['chat_template_kwargs', _ObjFast(['enable_thinking', true])]);
    ids := NativeChatPrompt(tpl, r, tpl.DefaultEnableThinking);
    CheckEqual(tok.Decode(ids), '<|im_start|>user'#10'x<|im_end|>'#10 +
      '<|im_start|>assistant'#10'<think>'#10, 'thinking on by the request');
  finally
    tpl.Free;
    tok.Free;
  end;
end;

procedure TTestInferQwen35Chat.Qwen35Chat;
var
  dir: TFileName;
  llm: TNativeLlmClient;
  r: TLlmChatRequest;
  a: TLlmChatResponse;
  timer: TPrecisionTimer;
  b: TLlmChatResponse;
  reused: Int64;
  slots: integer;
begin
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3.5-0.8B chat: the weights need a 64-bit process');
  exit;
  {$endif CPU32}
  dir := FindQwen35Folder('model.safetensors.index.json');
  if dir = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/%', [QWEN35_MODEL]);
    AddConsole('SKIP Qwen3.5-0.8B chat: no vendor/models/' + QWEN35_MODEL);
    exit;
  end;
  timer.Start;
  llm := TNativeLlmClient.Create(dir);
  try
    AddConsole(FormatString('Qwen3.5-0.8B client loaded in %', [timer.Stop]));
    Check(llm.Template is TQwen35ChatTemplate, 'the Qwen3.5 template');
    // a plain question, thinking off (the default)
    Finalize(r);
    FillCharFast(r, SizeOf(r), 0);
    SetLength(r.Messages, 1);
    // (17 x 23 is too hard for it: transformers answers 331 too)
    r.Messages[0] := LlmMessage(lrUser,
      'What is the capital of France? Answer with one word.');
    r.Temperature := 0;
    r.MaxTokens := 200;
    timer.Start;
    a := llm.ChatComplete(r);
    AddConsole(FormatString('capital in %: "%" (% tokens)',
      [timer.Stop, a.Content, a.Usage.CompletionTokens]));
    Check(a.FinishReason = lfrStop, 'stopped by itself');
    Check(PosEx('Paris', a.Content) > 0, 'the answer Paris');
    CheckEqual(a.Reasoning, '', 'no reasoning');
    // the conversation goes on (D7d): the earlier turn loses its empty <think>
    // block, so the state at the end of the first request is no use - the
    // restore point before its last turn is; the same answer as without it
    SetLength(r.Messages, 3);
    r.Messages[1] := LlmMessage(lrAssistant, a.Content);
    r.Messages[2] := LlmMessage(lrUser,
      'And the capital of Germany? Answer with one word.');
    reused := llm.Cache.ReusedTokens;
    timer.Start;
    a := llm.ChatComplete(r);
    AddConsole(FormatString('next turn in %: "%" (% tokens reused)',
      [timer.Stop, a.Content, llm.Cache.ReusedTokens - reused]));
    Check(PosEx('Berlin', a.Content) > 0, 'the answer Berlin');
    Check(llm.Cache.ReusedTokens - reused > 10, 'the first turns from a restore point');
    slots := llm.Cache.Slots;
    llm.Cache.Slots := 0;
    b := llm.ChatComplete(r);
    llm.Cache.Slots := slots;
    CheckEqual(b.Content, a.Content, 'the same answer computed from the start');
    SetLength(r.Messages, 1);
    // a tool: the call to get_weather
    SetLength(r.Tools, 1);
    r.Tools[0].Name := 'get_weather';
    r.Tools[0].Description := 'The current weather of a city';
    r.Tools[0].ParametersJson := '{"type":"object","properties":{"city":{"type":' +
      '"string","description":"The city"}},"required":["city"]}';
    r.Messages[0] := LlmMessage(lrUser, 'What is the weather in Berlin right now?');
    timer.Start;
    a := llm.ChatComplete(r);
    AddConsole(FormatString('weather in %: % calls, content "%" (% tokens)',
      [timer.Stop, length(a.ToolCalls), a.Content, a.Usage.CompletionTokens]));
    if not CheckFailed(length(a.ToolCalls) = 1, 'one tool call') then
    begin
      CheckEqual(a.ToolCalls[0].Name, 'get_weather', 'the tool');
      CheckEqual(a.ToolCalls[0].ArgumentsJson, '{"city": "Berlin"}', 'the city');
      Check(a.FinishReason = lfrToolCalls, 'finish tool_calls');
      // the next turn: the result back, the answer uses it
      SetLength(r.Messages, 3);
      r.Messages[1] := LlmMessage(lrAssistant, a.Content);
      r.Messages[1].ToolCalls := a.ToolCalls;
      r.Messages[2] := LlmMessage(lrTool, '{"temperature_c": 17, "sky": "rain"}');
      r.Messages[2].ToolCallId := a.ToolCalls[0].Id;
      timer.Start;
      a := llm.ChatComplete(r);
      AddConsole(FormatString('answer in %: "%"', [timer.Stop, a.Content]));
      Check(a.FinishReason = lfrStop, 'answered');
      Check(PosEx('17', a.Content) > 0, 'with the temperature of the tool');
      // a new question after the tool loop (D7d): the loop went on from the
      // end of the cache and kept no restore points there; the one before
      // its turn is still there, the answer as without the cache
      SetLength(r.Messages, 5);
      r.Messages[3] := LlmMessage(lrAssistant, a.Content);
      r.Messages[4] := LlmMessage(lrUser, 'Thanks. And in Paris?');
      reused := llm.Cache.ReusedTokens;
      timer.Start;
      a := llm.ChatComplete(r);
      AddConsole(FormatString('after the tool loop in %: % calls, "%" (% tokens reused)',
        [timer.Stop, length(a.ToolCalls), a.Content, llm.Cache.ReusedTokens - reused]));
      Check(llm.Cache.ReusedTokens - reused > 100, 'the tools and the question reused');
      slots := llm.Cache.Slots;
      llm.Cache.Slots := 0;
      b := llm.ChatComplete(r);
      llm.Cache.Slots := slots;
      CheckEqual(b.Content, a.Content, 'after the tool loop: as without the cache');
      Check(length(b.ToolCalls) = length(a.ToolCalls),
        'after the tool loop: the same calls');
      if (a.ToolCalls <> nil) and (b.ToolCalls <> nil) then
        CheckEqual(b.ToolCalls[0].ArgumentsJson, a.ToolCalls[0].ArgumentsJson,
          'after the tool loop: the same arguments');
    end;
    // an integer parameter: typed by the schema the client gives the parser
    SetLength(r.Messages, 1);
    r.Tools[0].Name := 'set_heating';
    r.Tools[0].Description := 'Set the heating to a temperature';
    r.Tools[0].ParametersJson := '{"type":"object","properties":{"celsius":{"type":' +
      '"integer","description":"The temperature in degrees Celsius"}},' +
      '"required":["celsius"]}';
    r.Messages[0] := LlmMessage(lrUser, 'Please set the heating to 21 degrees.');
    timer.Start;
    a := llm.ChatComplete(r);
    AddConsole(FormatString('heating in %: % calls, content "%"',
      [timer.Stop, length(a.ToolCalls), a.Content]));
    if not CheckFailed(length(a.ToolCalls) = 1, 'one heating call') then
      CheckEqual(a.ToolCalls[0].ArgumentsJson, '{"celsius": 21}', 'an integer');
    // thinking on by the request: a reasoning first
    SetLength(r.Messages, 1);
    r.Tools := nil;
    r.Messages[0] := LlmMessage(lrUser, 'Is 91 a prime number?');
    r.Extra := _ObjFast(['chat_template_kwargs', _ObjFast(['enable_thinking', true])]);
    r.MaxTokens := 300;
    timer.Start;
    a := llm.ChatComplete(r);
    AddConsole(FormatString('thinking in %: reasoning "%" answer "%" (% tokens)',
      [timer.Stop, copy(a.Reasoning, 1, 80), copy(a.Content, 1, 80),
       a.Usage.CompletionTokens]));
    Check(a.Reasoning <> '', 'with reasoning');
  finally
    llm.Free;
  end;
end;

end.
