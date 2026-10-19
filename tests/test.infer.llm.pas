/// tests for the native ILlmClient (phase 2d, mormot.ai.llm.native)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the output parser and the prompt run on a tiny byte-level tokenizer and
//   need no model: every byte is one token, so a multi-byte character always
//   arrives cut in pieces
// - with vendor/models/qwen3-0.6b the client runs the reference prompts of
//   tests/data/infer-ref/qwen3 (complete and streamed, a tool call, a think
//   block cut by the token cap) and one agent loop with a tool
unit test.infer.llm;

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
  mormot.ai.agent,
  mormot.ai.llm.native,
  mormot.ai.infer.tensor,
  mormot.ai.infer.unicode,
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.chat,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.decoder,
  mormot.ai.infer.model.qwen3,
  mormot.ai.infer.sampler,
  test.infer.ref,
  test.infer.bpe;

type
  /// collects what a parser or client gives out as stream deltas
  TDeltaLog = class
  public
    Deltas, Dones, Usages, Invalid: integer;
    AfterDone, Fail: boolean;
    FirstRole, Content, Reasoning, Finish: RawUtf8;
    Usage: TLlmUsage;
    CallIndex: TIntegerDynArray;
    CallIds, CallNames, CallArgs: TRawUtf8DynArray;
    procedure OnDelta(const aDelta: TLlmStreamDelta);
  end;

  /// phase 2d: the native ILlmClient
  TTestInferLlmNative = class(TSynTestCase)
  protected
    fTokenizer: TBpeTokenizer;
    fTemplate: TQwen3ChatTemplate;
    fToolArgs: RawUtf8;
    fToolCalls: integer;
    procedure Setup; override;
    procedure CleanUp; override;
    /// feed the bytes of aText, one token each
    procedure Feed(aParser: TQwen3OutputParser; const aText: RawUtf8);
    /// the real model folder, '' (and SKIP or a failure) when absent
    function RealModel: TFileName;
    /// the tool of the agent test
    function Weather(const aArgumentsJson: RawUtf8): RawUtf8;
  published
    /// text in cut characters; newlines at the ends; role, finish, usage, done
    procedure ParserPlainText;
    /// a leading <think> block is reasoning, not content
    procedure ParserThinking;
    /// two tool calls with Python-formatted arguments
    procedure ParserToolCalls;
    /// malformed and cut tool calls stay text; a cut think block
    procedure ParserMalformedAndCut;
    /// invalid UTF-8, control tokens inside the text, wrong ids
    procedure ParserInvalidUtf8AndControls;
    /// ParseQwen3ToolCall accepts exactly {"name": string, "arguments": object}
    procedure ToolCallJson;
    /// NativeChatPrompt: the template, enable_thinking, refusals
    procedure ChatPrompt;
    /// NativeDecode on the tiny Qwen3: the reference tokens, the context limit
    procedure DecodeOnTinyModel;
    /// the prompt cache: NativeDecode in a kept context, TNativePromptCache
    procedure PromptCache;
    /// Qwen3-0.6B behind ILlmClient against the reference continuations
    procedure Qwen3ChatClient;
    /// TLlmAgent over the native client: a tool call and its answer
    procedure Qwen3AgentLoop;
    /// the native client on Q8_0 matrices
    procedure Qwen3ChatClientQ8;
    /// TInferSampler: the distributions of temperature, top-k, top-p, min-p
    procedure SamplerDraws;
    /// generation_config.json defaults and the request overrides
    procedure SamplingSettings;
    /// the native client samples, reproducibly per seed
    procedure Qwen3Sampling;
    /// the native client from one GGUF file: model, tokenizer, a chat answer
    procedure Qwen3ChatClientGguf;
    /// RegisterNativeFamily / NativeFamily: the classes of qwen3
    procedure NativeFamilies;
    /// Qwen3-30B-A3B (Q4_K_M GGUF) behind the client, where it was put
    procedure Qwen3MoeChatClientGguf;
    /// D1b: Qwen2.5-0.5B-Instruct behind the client - its own template (the
    // prompt tokens of the reference), a plain answer, a tool call parsed
    procedure Qwen25ChatClient;
  end;


implementation

const
  IM_END = 300;
  IM_START = 301;
  THINK = 302;
  THINK_END = 303;
  TOOL_CALL = 304;
  TOOL_CALL_END = 305;
  END_OF_TEXT = 308;

/// bytes from hexadecimal: non-ASCII test data without non-ASCII source
function Bin(const aHex: RawUtf8): RawUtf8;
begin
  result := '';
  FastSetString(result, length(aHex) shr 1);
  HexToBin(pointer(aHex), pointer(result), length(result));
end;


{ TDeltaLog }

procedure TDeltaLog.OnDelta(const aDelta: TLlmStreamDelta);
var
  r: RawUtf8;
  n: PtrInt;
begin
  if Fail then
    ELlmClient.RaiseUtf8('TDeltaLog: stream cancelled', []);
  inc(Deltas);
  if Deltas = 1 then
    FirstRole := aDelta.Role;
  if Dones > 0 then
    AfterDone := true;
  if not IsStrictUtf8(pointer(aDelta.ContentDelta), length(aDelta.ContentDelta)) then
    inc(Invalid);
  Content := Content + aDelta.ContentDelta;
  r := _Safe(aDelta.Raw)^.A['choices']^._[0]^.O['delta']^.U['reasoning_content'];
  if not IsStrictUtf8(pointer(r), length(r)) then
    inc(Invalid);
  Reasoning := Reasoning + r;
  if aDelta.HasToolCall then
  begin
    n := length(CallIds);
    SetLength(CallIndex, n + 1);
    SetLength(CallIds, n + 1);
    SetLength(CallNames, n + 1);
    SetLength(CallArgs, n + 1);
    CallIndex[n] := aDelta.ToolCallIndex;
    CallIds[n] := aDelta.ToolCallId;
    CallNames[n] := aDelta.ToolCallName;
    CallArgs[n] := aDelta.ToolCallArgsDelta;
  end;
  if aDelta.FinishReason <> '' then
    Finish := aDelta.FinishReason;
  if aDelta.HasUsage then
  begin
    inc(Usages);
    Usage := aDelta.Usage;
  end;
  if aDelta.Done then
    inc(Dones);
end;


{ TTestInferLlmNative }

procedure TTestInferLlmNative.Setup;
var
  json, err: RawUtf8;
begin
  json := StringReplaceAll(TinyBpeJson('{"type":"NFC"}', '"aa":256', '[["a","a"]]'),
    '"added_tokens":[{"id":300,"content":"<|im_end|>","special":true}]',
    '"added_tokens":[{"id":300,"content":"<|im_end|>","special":true},' +
    '{"id":301,"content":"<|im_start|>","special":true},' +
    '{"id":302,"content":"<think>","special":false},' +
    '{"id":303,"content":"</think>","special":false},' +
    '{"id":304,"content":"<tool_call>","special":false},' +
    '{"id":305,"content":"</tool_call>","special":false},' +
    '{"id":306,"content":"<tool_response>","special":false},' +
    '{"id":307,"content":"</tool_response>","special":false},' +
    '{"id":308,"content":"<|endoftext|>","special":true}]');
  fTokenizer := LoadBpe(json, err);
  if CheckFailed(fTokenizer <> nil, Utf8ToString(err)) then
    exit;
  fTemplate := TQwen3ChatTemplate.Create(fTokenizer);
end;

procedure TTestInferLlmNative.CleanUp;
begin
  FreeAndNil(fTemplate);
  FreeAndNil(fTokenizer);
end;

procedure TTestInferLlmNative.Feed(aParser: TQwen3OutputParser;
  const aText: RawUtf8);
var
  i: PtrInt;
begin
  for i := 1 to length(aText) do
    if aParser.Push(ord(aText[i])) then
      Check(false, 'a text byte is no stop token');
end;

procedure TTestInferLlmNative.ParserPlainText;
var
  log: TDeltaLog;
  p: TQwen3OutputParser;
  text: RawUtf8;
begin
  // "Gr" u-umlaut "sse " CJK, an emoji, "!" - the two-, three- and four-byte
  // characters arrive one byte per token
  text := 'Gr' + Bin('c3bc') + 'sse ' + Bin('e4b896') + Bin('f09f9880') + '!';
  log := TDeltaLog.Create;
  p := TQwen3OutputParser.Create(fTokenizer, log.OnDelta);
  try
    Feed(p, #10#10 + text + #10 + 'second line' + #10#10);
    Check(p.Push(IM_END), 'im_end stops');
    p.Finish(false, 7);
    CheckEqual(p.Response.Content, text + #10 + 'second line'#10#10,
      'leading newlines dropped, the others kept (rendered verbatim)');
    Check(p.Response.FinishReason = lfrStop, 'stop');
    Check(p.Response.ToolCalls = nil, 'no tool call');
    CheckEqual(p.Response.Usage.PromptTokens, 7, 'prompt tokens');
    CheckEqual(p.Response.Usage.CompletionTokens, length(text) + 17, 'every push counted');
    CheckEqual(p.Response.Usage.TotalTokens, length(text) + 24, 'total');
    Check(VarIsEmptyOrNull(p.Response.Raw), 'no reasoning, no Raw');
    CheckEqual(log.Content, p.Response.Content, 'stream = complete');
    CheckEqual(log.Invalid, 0, 'every delta valid UTF-8');
    CheckEqual(log.FirstRole, 'assistant', 'role on the first delta');
    CheckEqual(log.Finish, 'stop', 'finish delta');
    CheckEqual(log.Usages, 1, 'one usage');
    CheckEqual(log.Usage.CompletionTokens, p.Response.Usage.CompletionTokens,
      'stream usage');
    CheckEqual(log.Dones, 1, 'one Done');
    Check(not log.AfterDone, 'Done is the last delta');
    Check(log.Deltas > 4, 'given out piece by piece');
  finally
    p.Free;
    log.Free;
  end;
  // no callback: the same response
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, #10#10 + text + #10 + 'second line' + #10#10);
    Check(p.Push(END_OF_TEXT), 'endoftext stops too');
    p.Finish(false, 7);
    CheckEqual(p.Response.Content, text + #10 + 'second line'#10#10,
      'without stream');
  finally
    p.Free;
  end;
end;

procedure TTestInferLlmNative.ParserThinking;
var
  log: TDeltaLog;
  p: TQwen3OutputParser;
begin
  log := TDeltaLog.Create;
  p := TQwen3OutputParser.Create(fTokenizer, log.OnDelta);
  try
    p.Push(THINK);
    Feed(p, #10'Let me think.'#10'Twice.'#10);
    p.Push(THINK_END);
    Feed(p, #10#10'The answer.');
    p.Push(IM_END);
    p.Finish(false, 3);
    CheckEqual(p.Reasoning, 'Let me think.'#10'Twice.', 'reasoning');
    CheckEqual(p.Response.Content, 'The answer.', 'content');
    CheckEqual(_Safe(p.Response.Raw)^.A['choices']^._[0]^.O['message']^.
      U['reasoning_content'], p.Reasoning, 'reasoning_content in Raw');
    CheckEqual(log.Reasoning, p.Reasoning, 'streamed reasoning');
    CheckEqual(log.Content, 'The answer.', 'streamed content');
    CheckEqual(log.Invalid, 0, 'valid UTF-8');
  finally
    p.Free;
    log.Free;
  end;
  // the empty block of enable_thinking = false comes from the prompt, so the
  // answer starts with text; a <think> later on is text
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'a');
    p.Push(THINK);
    Feed(p, 'b');
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, 'a<think>b', 'think token inside the text');
    CheckEqual(p.Reasoning, '', 'no reasoning');
  finally
    p.Free;
  end;
end;

procedure TTestInferLlmNative.ParserToolCalls;
var
  log: TDeltaLog;
  p: TQwen3OutputParser;
  r: TLlmChatResponse;
begin
  log := TDeltaLog.Create;
  p := TQwen3OutputParser.Create(fTokenizer, log.OnDelta);
  try
    Feed(p, 'Sure.'#10);
    p.Push(TOOL_CALL);
    Feed(p, #10'{"name": "get_weather", "arguments": {"city": "Berlin", ' +
      '"days": 2.50}}'#10);
    p.Push(TOOL_CALL_END);
    Feed(p, #10);
    p.Push(TOOL_CALL);
    Feed(p, '{"arguments":{},"name":"now"}');
    p.Push(TOOL_CALL_END);
    p.Push(IM_END);
    p.Finish(false, 10);
    r := p.Response;
    CheckEqual(r.Content, 'Sure.', 'content before the calls');
    Check(r.FinishReason = lfrToolCalls, 'finish tool_calls');
    if CheckFailed(length(r.ToolCalls) = 2, 'two calls') then
      exit;
    CheckEqual(r.ToolCalls[0].Name, 'get_weather', 'name 0');
    CheckEqual(r.ToolCalls[0].ArgumentsJson, '{"city": "Berlin", "days": 2.5}',
      'arguments as json.dumps writes them');
    CheckEqual(r.ToolCalls[1].Name, 'now', 'name 1');
    CheckEqual(r.ToolCalls[1].ArgumentsJson, '{}', 'empty arguments');
    Check(IdemPChar(pointer(r.ToolCalls[0].Id), 'CALL_'), 'id 0');
    Check(length(r.ToolCalls[0].Id) > 8, 'id 0 length');
    Check(r.ToolCalls[0].Id <> r.ToolCalls[1].Id, 'ids differ');
    if CheckFailed(length(log.CallIds) = 2, 'two streamed calls') then
      exit;
    CheckEqual(log.CallIndex[0], 0, 'index 0');
    CheckEqual(log.CallIndex[1], 1, 'index 1');
    CheckEqual(log.CallIds[1], r.ToolCalls[1].Id, 'streamed id');
    CheckEqual(log.CallNames[0], 'get_weather', 'streamed name');
    CheckEqual(log.CallArgs[0], r.ToolCalls[0].ArgumentsJson, 'streamed arguments');
    CheckEqual(log.Content, 'Sure.', 'streamed content');
    CheckEqual(log.Finish, 'tool_calls', 'streamed finish');
  finally
    p.Free;
    log.Free;
  end;
end;

procedure TTestInferLlmNative.ParserMalformedAndCut;
var
  p: TQwen3OutputParser;
begin
  // reasoning, then a call: no content in between
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    p.Push(THINK);
    Feed(p, #10'I need the weather.'#10);
    p.Push(THINK_END);
    Feed(p, #10#10);
    p.Push(TOOL_CALL);
    Feed(p, #10'{"name": "w", "arguments": {}}'#10);
    p.Push(TOOL_CALL_END);
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Reasoning, 'I need the weather.', 'reasoning before the call');
    CheckEqual(p.Response.Content, '', 'no content');
    CheckEqual(length(p.Response.ToolCalls), 1, 'the call');
    Check(p.Response.FinishReason = lfrToolCalls, 'tool_calls');
  finally
    p.Free;
  end;
  // text after a call: the newline before the call is the template's
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'A'#10);
    p.Push(TOOL_CALL);
    Feed(p, '{"name": "w", "arguments": {}}');
    p.Push(TOOL_CALL_END);
    Feed(p, #10'B'#10);
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, 'A'#10'B', 'separators dropped');
  finally
    p.Free;
  end;
  // no arguments: kept as text, no call - with the newline before it
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'Hi'#10);
    p.Push(TOOL_CALL);
    Feed(p, '{"name": "f"}');
    p.Push(TOOL_CALL_END);
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, 'Hi'#10'<tool_call>{"name": "f"}</tool_call>',
      'malformed call is text');
    Check(p.Response.ToolCalls = nil, 'no call');
    Check(p.Response.FinishReason = lfrStop, 'stop');
  finally
    p.Free;
  end;
  // cut by the token cap inside a call
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'x'#10);
    p.Push(TOOL_CALL);
    Feed(p, '{"name": "f", "argu');
    p.Finish(true, 0);
    CheckEqual(p.Response.Content, 'x'#10'<tool_call>{"name": "f", "argu',
      'open call is text');
    Check(p.Response.ToolCalls = nil, 'no call');
    Check(p.Response.FinishReason = lfrLength, 'length');
  finally
    p.Free;
  end;
  // cut inside the think block: all reasoning, no content
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    p.Push(THINK);
    Feed(p, #10'abc'#10);
    p.Finish(true, 0);
    CheckEqual(p.Reasoning, 'abc', 'cut reasoning');
    CheckEqual(p.Response.Content, '', 'no content');
    Check(p.Response.FinishReason = lfrLength, 'length');
  finally
    p.Free;
  end;
  // a cut call does not turn finish into tool_calls when an earlier one exists
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    p.Push(TOOL_CALL);
    Feed(p, '{"name": "f", "arguments": {}}');
    p.Push(TOOL_CALL_END);
    p.Finish(true, 0);
    CheckEqual(length(p.Response.ToolCalls), 1, 'complete call kept');
    Check(p.Response.FinishReason = lfrLength, 'length wins');
  finally
    p.Free;
  end;
end;

procedure TTestInferLlmNative.ParserInvalidUtf8AndControls;
var
  log: TDeltaLog;
  p: TQwen3OutputParser;
  raised: boolean;
begin
  log := TDeltaLog.Create;
  p := TQwen3OutputParser.Create(fTokenizer, log.OnDelta);
  try
    // a stray continuation byte, then a lead byte the answer never completes
    Feed(p, Bin('ff') + 'a' + Bin('c3'));
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, Bin('efbfbd') + 'a' + Bin('efbfbd'),
      'U+FFFD per invalid byte');
    CheckEqual(log.Content, p.Response.Content, 'streamed the same');
    CheckEqual(log.Invalid, 0, 'valid deltas');
  finally
    p.Free;
    log.Free;
  end;
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'x');
    p.Push(IM_START);
    p.Push(THINK_END);
    p.Push(TOOL_CALL_END);
    Feed(p, 'y');
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, 'x<|im_start|></think></tool_call>y',
      'control tokens inside the text are text');
    raised := false;
    try
      p.Push(65);
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised, 'Push after Finish');
  finally
    p.Free;
  end;
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    raised := false;
    try
      p.Push(fTokenizer.VocabSize);
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised, 'id outside the tokenizer');
    raised := false;
    try
      p.Push(-1);
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised, 'negative id');
  finally
    p.Free;
  end;
end;

procedure TTestInferLlmNative.ToolCallJson;
var
  name, args: RawUtf8;
begin
  Check(ParseQwen3ToolCall(' {"name": "a", "arguments": {"x": [1, 2.0, "s"]}} ',
    name, args), 'plain call');
  CheckEqual(name, 'a', 'name');
  CheckEqual(args, '{"x": [1, 2.0, "s"]}', 'arguments');
  Check(ParseQwen3ToolCall('{"name":"b","arguments":{"n":{"deep":true}}}',
    name, args), 'compact call');
  CheckEqual(args, '{"n": {"deep": true}}', 'nested arguments re-formatted');
  Check(not ParseQwen3ToolCall('{"name": "a"}', name, args), 'no arguments');
  Check(name = '', 'name cleared on failure');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": {}, "x": 1}',
    name, args), 'extra key');
  Check(not ParseQwen3ToolCall('{"Name": "a", "arguments": {}}', name, args),
    'name in another case');
  Check(not ParseQwen3ToolCall('{"name": "a", "Arguments": {}}', name, args),
    'arguments in another case');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": {}, "": 1}', name, args),
    'an extra key with an empty name');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": {}} x', name, args),
    'text after the object');
  // more digits than json.dumps keeps: that float stays as the model wrote it
  Check(ParseQwen3ToolCall('{"name": "t", "arguments": {"v":0.1234567890123456, ' +
    '"w":1.50} }', name, args), 'a float of 16 digits');
  CheckEqual(args, '{"v": 0.1234567890123456, "w": 1.5}',
    'the long float verbatim, the rest formatted, no blank after');
  Check(ParseQwen3ToolCall('{"name":"t","\u0061rguments":{"v":1e999}}', name, args),
    'an escaped key, a float beyond the range');
  CheckEqual(args, '{"v": 1e999}', 'beyond the range verbatim');
  // the long float switches no other rule off
  Check(not ParseQwen3ToolCall('{"name":"f","arguments":{"a":1,"a":2,' +
    '"v":0.1234567890123456}}', name, args), 'duplicate argument key');
  Check(not ParseQwen3ToolCall('{"name":"f","arguments":{"s":"\ud800",' +
    '"v":0.1234567890123456}}', name, args), 'lone surrogate');
  Check(not ParseQwen3ToolCall('{"name":"f","arguments":{"s":"a'#10'b",' +
    '"v":0.1234567890123456}}', name, args), 'raw newline in a string');
  Check(not ParseQwen3ToolCall('{"name":"f","arguments":{"s":"' + Bin('ff') + '"}}',
    name, args), 'invalid UTF-8');
  Check(not ParseQwen3ToolCall('{"name": "", "arguments": {}}', name, args),
    'empty name');
  Check(not ParseQwen3ToolCall('{"name": 5, "arguments": {}}', name, args),
    'name no string');
  Check(not ParseQwen3ToolCall('{"name": 0.00001, "arguments": {}}', name, args),
    'name a float the document would keep as text');
  Check(not ParseQwen3ToolCall('{"name": null, "arguments": {}}', name, args),
    'name null');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": "{}"}', name, args),
    'arguments as a string');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": [1]}', name, args),
    'arguments as an array');
  Check(not ParseQwen3ToolCall('{"name": "a", "name": "b"}', name, args),
    'duplicate key');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": {}', name, args),
    'invalid JSON');
  Check(not ParseQwen3ToolCall('', name, args), 'empty');
  Check(not ParseQwen3ToolCall('[1]', name, args), 'array');
end;

procedure TTestInferLlmNative.ChatPrompt;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  ids, expected: TIntegerDynArray;
  images: TLlmImageDynArray;
  raised: boolean;

  function Refused: boolean;
  begin
    result := false;
    try
      NativeChatPrompt(fTemplate, req, true);
    except
      on ELlmClient do
        result := true;
    end;
  end;

begin
  if fTemplate = nil then
    exit;
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'hi');
  req := LlmChatRequest('', msgs);
  ids := NativeChatPrompt(fTemplate, req, true);
  expected := fTemplate.Encode(msgs, nil, true, true);
  CheckEqual(length(ids), length(expected), 'thinking by default');
  Check(CompareMem(pointer(ids), pointer(expected), length(ids) * SizeOf(integer)),
    'the template ids');
  CheckEqual(ids[high(ids)], 10, 'ends with the opened assistant turn');
  ids := NativeChatPrompt(fTemplate, req, false);
  CheckEqual(ids[high(ids) - 2], THINK_END, 'default false: empty think block');
  req.Extra := _JsonFast('{"chat_template_kwargs":{"enable_thinking":true}}');
  Check(NativeEnableThinking(req, false), 'request overrides the default');
  req.Extra := _JsonFast('{"chat_template_kwargs":{"enable_thinking":false}}');
  Check(not NativeEnableThinking(req, true), 'request switches thinking off');
  ids := NativeChatPrompt(fTemplate, req, true);
  expected := fTemplate.Encode(msgs, nil, true, false);
  CheckEqual(length(ids), length(expected), 'enable_thinking false');
  Check(CompareMem(pointer(ids), pointer(expected), length(ids) * SizeOf(integer)),
    'the template ids without thinking');
  req.Extra := _JsonFast('{"chat_template_kwargs":{"enable_thinking":"no"}}');
  Check(Refused, 'enable_thinking no boolean');
  req.Extra := _JsonFast('{"other":1}');
  Check(NativeEnableThinking(req, true), 'other Extra keys ignored');
  req.ResponseFormat := '{"type":"json_object"}';
  Check(Refused, 'ResponseFormat refused');
  req.ResponseFormat := '';
  SetLength(images, 1); // no [] literal: Delphi 7 has no dynamic array constants
  images[0] := LlmImageUrl('http://x/y.png');
  msgs[0] := LlmImageMessage(lrUser, 'look', images);
  req.Messages := msgs;
  Check(Refused, 'images refused');
  req.Messages := nil;
  Check(Refused, 'no messages');
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'hi');
  req.Messages := msgs;
  SetLength(req.Tools, 1);
  req.Tools[0].Name := 'f';
  req.Tools[0].ParametersJson := '{"type":';
  raised := Refused;
  Check(raised, 'invalid tool JSON as ELlmClient');
end;

procedure TTestInferLlmNative.DecodeOnTinyModel;
var
  dir: TFileName;
  cpu: TInferCpuBackend;
  model: TQwen3Model;
  p: TQwen3OutputParser;
  prompt: TIntegerDynArray;
  raised: boolean;

  function Decode(aPromptLen, aMaxNew: PtrInt): boolean;
  var
    i: PtrInt;
  begin
    SetLength(prompt, aPromptLen);
    for i := 0 to aPromptLen - 1 do
      prompt[i] := 33 + i mod 60; // printable bytes, all inside the tiny vocab
    FreeAndNil(p);
    p := TQwen3OutputParser.Create(fTokenizer);
    try
      NativeDecode(model, prompt, aMaxNew, p, InferSamplingGreedy);
      result := true;
    except
      on ELlmClient do
        result := false;
    end;
  end;

begin
  if fTokenizer = nil then
    exit;
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  cpu := TInferCpuBackend.Create;
  model := nil;
  p := nil;
  try
    model := TQwen3Model.Create(dir, cpu);
    CheckEqual(model.Config.MaxPositions, 512, 'tiny context');
    // case 0 of the reference: [77 29 40 57 65] continues with 3 3 3 3 3 3 49
    SetLength(prompt, 5);
    prompt[0] := 77;
    prompt[1] := 29;
    prompt[2] := 40;
    prompt[3] := 57;
    prompt[4] := 65;
    p := TQwen3OutputParser.Create(fTokenizer);
    NativeDecode(model, prompt, 7, p, InferSamplingGreedy);
    CheckEqual(p.Response.Content, #3#3#3#3#3#3'1', 'greedy ids of the reference');
    Check(p.Response.FinishReason = lfrLength, 'cut by aMaxNew');
    CheckEqual(p.Response.Usage.CompletionTokens, 7, 'seven tokens');
    CheckEqual(p.Response.Usage.PromptTokens, 5, 'prompt tokens');
    // the context limit: P + n - 1 positions are fed
    Check(Decode(511, 100), 'one free position');
    CheckEqual(p.Count, 2, 'gives two tokens');
    Check(p.Response.FinishReason = lfrLength, 'cut by the context');
    Check(Decode(512, 100), 'a full context');
    CheckEqual(p.Count, 1, 'still gives one token');
    Check(not Decode(513, 1), 'a prompt beyond the context');
    Check(not Decode(5, 0), 'no new tokens');
    raised := false;
    try
      FreeAndNil(p);
      p := TQwen3OutputParser.Create(fTokenizer);
      NativeDecode(model, nil, 1, p, InferSamplingGreedy);
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised, 'empty prompt');
  finally
    p.Free;
    model.Free;
    cpu.Free;
  end;
end;

procedure TTestInferLlmNative.PromptCache;
var
  dir: TFileName;
  cpu: TInferCpuBackend;
  model: TQwen3Model;
  cache: TNativePromptCache;
  c, c1, c2, c3, c4: TQwen3Context;
  first, second, other, third: TIntegerDynArray;
  alone, cached: RawUtf8;
  reused, i: PtrInt;
  raised: boolean;

  function Decode(const aPrompt: TIntegerDynArray; aCtx: TQwen3Context;
    out aReused: PtrInt): RawUtf8;
  var
    p: TQwen3OutputParser;
  begin
    p := TQwen3OutputParser.Create(fTokenizer);
    try
      aReused := NativeDecode(model, aPrompt, 9, p, InferSamplingGreedy, aCtx);
      result := p.Response.Content;
      CheckEqual(p.Response.Usage.PromptTokens, length(aPrompt),
        'the whole prompt in the usage');
    finally
      p.Free;
    end;
  end;

  function Ids(aCount, aSeed: PtrInt): TIntegerDynArray;
  var
    j: PtrInt;
  begin
    result := nil;
    SetLength(result, aCount);
    for j := 0 to aCount - 1 do
      result[j] := 33 + (j * 7 + aSeed) mod 60; // printable, in the tiny vocab
  end;

  procedure Fill(aCtx: TQwen3Context; const aIds: TIntegerDynArray);
  begin
    aCtx.Reset;
    model.Forward(aCtx, pointer(aIds), length(aIds), nil);
  end;

begin
  if fTokenizer = nil then
    exit;
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  cpu := TInferCpuBackend.Create;
  model := nil;
  cache := nil;
  c := nil;
  try
    model := TQwen3Model.Create(dir, cpu);
    // NativeDecode in a given context: the second turn starts with the first
    // prompt and its answer, only the rest is computed - the same answer
    first := Ids(20, 0);
    c := TQwen3Context.Create;
    alone := Decode(first, nil, reused);
    cached := Decode(first, c, reused);
    CheckEqual(reused, 0, 'an empty context reuses nothing');
    CheckEqual(cached, alone, 'decoded in a given context');
    CheckEqual(c.Length, 20 + 9 - 1, 'the prompt and the fed tokens stay');
    second := Ids(40, 3);
    for i := 0 to c.Length - 1 do
      second[i] := c.Id[i];
    alone := Decode(second, nil, reused);
    cached := Decode(second, c, reused);
    CheckEqual(reused, 28, 'the first turn and its answer came from the cache');
    CheckEqual(cached, alone, 'the same answer as a fresh decode');
    // the same prompt once more: all but its last token
    cached := Decode(second, c, reused);
    CheckEqual(reused, 39, 'a repeated prompt');
    CheckEqual(cached, alone, 'the same answer again');
    FreeAndNil(c);
    // the pool: the longest shared prefix wins, the least recently used goes
    cache := TNativePromptCache.Create(2);
    CheckEqual(cache.Slots, 2, 'slots');
    other := Ids(30, 5);
    third := Ids(10, 11);
    c1 := cache.Acquire(first);
    c2 := cache.Acquire(first);
    Check((c1 <> nil) and (c2 <> nil) and (c1 <> c2), 'an empty cache gives new ones');
    Fill(c1, first);
    Fill(c2, other);
    cache.Release(c1, 0);
    cache.Release(c2, 5);
    Check((cache.Requests = 2) and (cache.Hits = 1) and (cache.ReusedTokens = 5),
      'requests, hits, reused tokens');
    Check(cache.Acquire(second) = c1, 'second starts with first: c1');
    Check(cache.Acquire(other) = c2, 'other: c2');
    cache.Release(c1, 0);
    cache.Release(c2, 0);
    // nothing shared and the cache full: the least recently used, c1
    c3 := cache.Acquire(third);
    Check(c3 = c1, 'no shared token: the least recently used');
    // nothing shared, a free slot: a new one
    c4 := cache.Acquire(third);
    Check((c4 <> c2) and (c4 <> c1) and (c4.Length = 0), 'a free slot: a new one');
    Fill(c4, third);
    cache.Release(c4, 0); // idle: c2 c4
    Fill(c3, first);
    cache.Release(c3, 0); // full: c2 goes, idle: c4 c3
    c := cache.Acquire(other);
    Check(c = c4, 'c2 was evicted: other finds nothing, gets the oldest');
    c := nil;
    Check(cache.Acquire(first) = c3, 'c3 kept');
    cache.Release(c3, 0);
    cache.Release(c4, 0); // idle: c3 c4
    // fewer slots: the least recently used go
    cache.Slots := 1;
    c := cache.Acquire(first);
    Check((c = c4) and (c.Length = length(third)), 'Slots 1 kept the newest');
    c := nil;
    cache.Release(c4, 0);
    // Clear forgets every sequence
    cache.Clear;
    c := cache.Acquire(third);
    CheckEqual(c.Length, 0, 'after Clear: an empty context');
    // two contexts share a prefix: the longer one wins, not the first found
    cache.Slots := 2;
    Fill(c, Ids(10, 0));
    cache.Release(c, 0);
    c := cache.Acquire(third);
    Fill(c, first);
    cache.Release(c, 0);
    c := cache.Acquire(second);
    CheckEqual(c.Length, length(first), 'the longest shared prefix wins');
    Fill(c, third);
    // no slot: Release frees, every request starts empty
    cache.Slots := 0;
    cache.Release(c, 0);
    c := nil;
    c := cache.Acquire(third);
    CheckEqual(c.Length, 0, 'Slots 0 keeps nothing');
    FreeAndNil(c);
    raised := false;
    try
      cache.Slots := -1;
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised and (cache.Slots = 0), 'negative slots refused');
  finally
    c.Free;
    cache.Free;
    model.Free;
    cpu.Free;
  end;
end;

function TTestInferLlmNative.RealModel: TFileName;
begin
  result := '';
  {$ifdef CPU32}
  AddConsole('SKIP native client: 2.4 GB of FP32 weights need a 64-bit process');
  exit;
  {$endif CPU32}
  result := FindQwen3Folder('model.safetensors');
  if result <> '' then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b/model.safetensors');
  AddConsole('SKIP native client: no vendor/models/qwen3-0.6b (tools/infer-ref, ' +
    'make_reference_qwen3.py)');
end;

/// the messages and tools of a reference prompt
procedure RefRequest(p: PDocVariantData; var aRequest: TLlmChatRequest);
var
  m, t: PDocVariantData;
  i: PtrInt;
begin
  m := p^.A['messages'];
  SetLength(aRequest.Messages, m^.Count);
  for i := 0 to m^.Count - 1 do
    aRequest.Messages[i] := LlmMessage(ToLlmRole(_Safe(m^.Values[i])^.U['role']),
      _Safe(m^.Values[i])^.U['content']);
  t := p^.A['tools'];
  SetLength(aRequest.Tools, t^.Count);
  for i := 0 to t^.Count - 1 do
  begin
    aRequest.Tools[i].Name := _Safe(t^.Values[i])^.U['name'];
    aRequest.Tools[i].Description := _Safe(t^.Values[i])^.U['description'];
    aRequest.Tools[i].ParametersJson := _Safe(t^.Values[i])^.U['parameters'];
  end;
  aRequest.Temperature := 0; // the reference decodes greedy
  if p^.GetValueIndex('enable_thinking') >= 0 then
    aRequest.Extra := _ObjFast(['chat_template_kwargs',
      _ObjFast(['enable_thinking', p^.B['enable_thinking']])]);
end;

function FindPrompt(aRef: TInferRefSet; const aId: RawUtf8): PDocVariantData;
var
  prompts: PDocVariantData;
  i: PtrInt;
begin
  prompts := aRef.Manifest^.A['prompts'];
  for i := 0 to prompts^.Count - 1 do
  begin
    result := _Safe(prompts^.Values[i]);
    if result^.U['id'] = aId then
      exit;
  end;
  result := nil;
end;

procedure TTestInferLlmNative.Qwen3ChatClient;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  client: TNativeLlmClient;
  llm: ILlmClient;
  p: PDocVariantData;
  req: TLlmChatRequest;
  r: TLlmChatResponse;
  log: TDeltaLog;
  expected: RawUtf8;
  timer: TPrecisionTimer;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  refdir := FindInferRefDir('qwen3');
  if CheckFailed(refdir <> '', 'qwen3 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  try
    timer.Start;
    client := TNativeLlmClient.Create(dir);
    llm := client; // reference counted from here
    AddConsole(FormatString('native client loaded in %', [timer.Stop]));
    CheckEqual(llm.Config.DefaultModel, 'qwen3-0.6b', 'model name from the folder');
    // a short answer: the stop token ends it
    p := FindPrompt(ref, 'chat_en');
    if CheckFailed(p <> nil, 'chat_en') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    r := llm.ChatComplete(req);
    CheckEqual(r.Content, 'Paris', 'chat_en content');
    Check(r.FinishReason = lfrStop, 'chat_en stop');
    CheckEqual(r.Usage.PromptTokens, p^.I['tokens'], 'chat_en prompt tokens');
    CheckEqual(r.Usage.CompletionTokens, 2, 'Paris + im_end');
    CheckEqual(r.Model, 'qwen3-0.6b', 'response model');
    // the tool call of the reference continuation
    p := FindPrompt(ref, 'chat_tool');
    if CheckFailed(p <> nil, 'chat_tool') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    timer.Start;
    r := llm.ChatComplete(req);
    AddConsole(FormatString('chat_tool: % + % tokens in %',
      [r.Usage.PromptTokens, r.Usage.CompletionTokens, timer.Stop]));
    CheckEqual(r.Usage.PromptTokens, p^.I['tokens'], 'chat_tool prompt tokens');
    CheckEqual(r.Content, '', 'only a call');
    Check(r.FinishReason = lfrToolCalls, 'chat_tool finish');
    if not CheckFailed(length(r.ToolCalls) = 1, 'one call') then
    begin
      CheckEqual(r.ToolCalls[0].Name, 'get_weather', 'tool name');
      CheckEqual(r.ToolCalls[0].ArgumentsJson, '{"city": "Berlin"}', 'tool arguments');
    end;
    // thinking by default; the cap cuts the think block
    p := FindPrompt(ref, 'chat_think');
    if CheckFailed(p <> nil, 'chat_think') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    req.MaxTokens := p^.A['greedy']^.Count;
    r := llm.ChatComplete(req);
    expected := p^.U['greedy_text'];
    Check(IdemPChar(pointer(expected), '<THINK>'#10), 'reference starts thinking');
    delete(expected, 1, 8);
    CheckEqual(_Safe(r.Raw)^.A['choices']^._[0]^.O['message']^.U['reasoning_content'],
      expected, 'reasoning of the reference');
    CheckEqual(r.Content, '', 'no answer yet');
    Check(r.FinishReason = lfrLength, 'chat_think cut');
    // streamed: the deltas give the reference text
    p := FindPrompt(ref, 'chat_de');
    if CheckFailed(p <> nil, 'chat_de') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    req.MaxTokens := p^.A['greedy']^.Count;
    log := TDeltaLog.Create;
    try
      llm.ChatStream(req, log.OnDelta);
      CheckEqual(log.Content, p^.U['greedy_text'], 'chat_de streamed');
      CheckEqual(log.Invalid, 0, 'valid UTF-8 deltas');
      CheckEqual(log.FirstRole, 'assistant', 'role');
      CheckEqual(log.Finish, 'length', 'cut by MaxTokens');
      CheckEqual(log.Usage.CompletionTokens, req.MaxTokens, 'completion tokens');
      CheckEqual(log.Dones, 1, 'one Done');
      Check(not log.AfterDone, 'Done last');
    finally
      log.Free;
    end;
  finally
    ref.Free;
  end;
end;

procedure TTestInferLlmNative.Qwen3ChatClientQ8;
var
  dir: TFileName;
  client: TNativeLlmClient;
  llm: ILlmClient;
  req: TLlmChatRequest;
  r: TLlmChatResponse;
  msgs: TLlmMessageDynArray;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  client := TNativeLlmClient.Create(dir, '', nil, idtQ8_0);
  llm := client;
  Check(client.Model.Weights = idtQ8_0, 'Q8_0 matrices');
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'What is the capital of France? Answer in one word.');
  req := LlmChatRequest('', msgs);
  req.Extra := _ObjFast(['chat_template_kwargs', _ObjFast(['enable_thinking', false])]);
  req.Temperature := 0;
  r := llm.ChatComplete(req);
  CheckEqual(r.Content, 'Paris', 'Q8_0 answers chat_en like FP32');
  Check(r.FinishReason = lfrStop, 'stop');
end;

/// draw aDraws tokens from aLogits and count them per index
function Draws(const aSampling: TInferSampling; const aLogits: array of single;
  aDraws: integer): TIntegerDynArray;
var
  smp: TInferSampler;
  i: integer;
begin
  result := nil; // the caller's array may come in: counts from zero
  SetLength(result, length(aLogits));
  smp := TInferSampler.Create(aSampling);
  try
    for i := 1 to aDraws do
      inc(result[smp.Next(@aLogits[0], length(aLogits))]);
  finally
    smp.Free;
  end;
end;

function Sampling(aTemperature: double; aTopK: integer; aTopP, aMinP: double;
  aSeed: Int64): TInferSampling;
begin
  result.Temperature := aTemperature;
  result.TopK := aTopK;
  result.TopP := aTopP;
  result.MinP := aMinP;
  result.Seed := aSeed;
end;

function SingleOfBits(aBits: cardinal): single;
begin
  PCardinal(@result)^ := aBits;
end;

procedure TTestInferLlmNative.SamplerDraws;
const
  N = 70000;
  SEED42: array[0..11] of integer = (2, 1, 1, 1, 0, 2, 1, 2, 1, 2, 1, 2);
var
  c: TIntegerDynArray;
  l: array[0..2] of single;
  odd4: array[0..3] of single;
  big: TSingleDynArray;
  prob: TDoubleDynArray;
  a, b: TInferSampler;
  i, t, k: PtrInt;
  same, allRefused: boolean;
  m, sum, lo, hi, mid, top20: double;
  inNucleus, seen: array of boolean;
  distinct: integer;
  odd100: TSingleDynArray;

  function Near(aCount: integer; aShare: double): boolean;
  begin
    result := Abs(aCount / N - aShare) < 0.01;
  end;

  function Refused(const aSampling: TInferSampling): boolean;
  begin
    result := false;
    try
      TInferSampler.Create(aSampling).Free;
    except
      on EInferError do
        result := true;
    end;
  end;

begin
  // softmax of 0, ln 2, ln 4: 1/7, 2/7, 4/7
  l[0] := 0;
  l[1] := Ln(2);
  l[2] := Ln(4);
  c := Draws(Sampling(1, 0, 1, 0, 1), l, N);
  Check(Near(c[0], 1 / 7) and Near(c[1], 2 / 7) and Near(c[2], 4 / 7),
    'temperature 1: the softmax');
  c := Draws(Sampling(0.5, 0, 1, 0, 2), l, N);
  Check(Near(c[0], 1 / 21) and Near(c[1], 4 / 21) and Near(c[2], 16 / 21),
    'temperature 0.5: the logits doubled');
  c := Draws(Sampling(1, 2, 1, 0, 3), l, N);
  Check((c[0] = 0) and Near(c[1], 1 / 3) and Near(c[2], 2 / 3), 'top_k 2');
  c := Draws(Sampling(1, 0, 0.5, 0, 4), l, N);
  CheckEqual(c[2], N, 'top_p 0.5: the best token reaches it alone');
  c := Draws(Sampling(1, 0, 0.6, 0, 5), l, N);
  Check((c[0] = 0) and Near(c[1], 1 / 3) and Near(c[2], 2 / 3), 'top_p 0.6');
  c := Draws(Sampling(1, 0, 1, 0.4, 6), l, N);
  Check((c[0] = 0) and Near(c[1], 1 / 3) and Near(c[2], 2 / 3), 'min_p 0.4');
  c := Draws(Sampling(1, 2, 0.6, 0, 7), l, N);
  CheckEqual(c[2], N, 'top_p within the top k: 4/6 of their mass reaches 0.6');
  // top_p against all the mass, not just the candidates: A 0.625, B 0.3125
  // and 1000 tokens of 0.0000625 each - too small to be candidates at first;
  // top_p 0.65 needs B as well (0.625 alone is short of it)
  SetLength(big, 1002);
  big[0] := 0;
  big[1] := Ln(0.5);
  for i := 2 to high(big) do
    big[i] := Ln(1e-4);
  c := nil;
  SetLength(c, length(big));
  a := TInferSampler.Create(Sampling(1, 0, 0.65, 0, 10));
  try
    for i := 1 to N do
      inc(c[a.Next(pointer(big), length(big))]);
  finally
    a.Free;
  end;
  Check(Near(c[0], 2 / 3) and Near(c[1], 1 / 3), 'top_p over the whole mass');
  // top-k keeps the tokens tied with the k-th, as transformers does
  l[0] := 0;
  l[1] := 0;
  l[2] := 0;
  c := Draws(Sampling(1, 2, 1, 0, 11), l, N);
  Check(Near(c[0], 1 / 3) and Near(c[1], 1 / 3) and Near(c[2], 1 / 3),
    'top_k 2 of three equal tokens keeps all three');
  // top-k by logit: 0 and 1e-20 round to the same probability, top_k 1
  // still keeps only the higher logit
  l[0] := 0;
  l[1] := 1e-20;
  l[2] := -50;
  c := Draws(Sampling(1, 1, 1, 0, 15), l, 1000);
  CheckEqual(c[1], 1000, 'top_k by logit, not by rounded probability');
  // NaN and -Inf last in a draw without filters
  odd4[0] := 1;
  odd4[1] := 1;
  odd4[2] := SingleOfBits($ff800000);
  odd4[3] := SingleOfBits($7fc00000);
  c := Draws(Sampling(1, 0, 1, 0, 16), odd4, N);
  Check((c[2] = 0) and (c[3] = 0) and Near(c[0], 0.5), 'NaN and -Inf last, unfiltered');
  // top-p, then min-p (the order of transformers): 0.5 0.3 0.2 with top_p 0.6
  // keeps the first two, min_p 0.5 then keeps both (0.3 >= 0.25); min-p first
  // would leave 0.625 for the best, enough for top_p 0.6 alone
  l[0] := Ln(0.5);
  l[1] := Ln(0.3);
  l[2] := Ln(0.2);
  c := Draws(Sampling(1, 0, 0.6, 0.5, 12), l, N);
  Check((c[2] = 0) and Near(c[0], 0.625) and Near(c[1], 0.375),
    'top_p before min_p');
  // top-k beyond the finite tokens: the threshold falls to 0, -Inf never drawn
  SetLength(odd100, 100);
  for i := 0 to high(odd100) do
    odd100[i] := SingleOfBits($ff800000);
  odd100[0] := 0;
  odd100[7] := Ln(2);
  odd100[50] := Ln(4);
  c := nil;
  SetLength(c, 100);
  a := TInferSampler.Create(Sampling(1, 50, 1, 0, 13));
  try
    for i := 1 to N do
      inc(c[a.Next(pointer(odd100), 100)]);
  finally
    a.Free;
  end;
  Check((c[0] + c[7] + c[50] = N) and Near(c[0], 1 / 7) and Near(c[50], 4 / 7),
    'top_k 50 of 3 finite tokens');
  // a temperature too close to 0 to divide by: greedy
  c := Draws(Sampling(1e-310, 0, 1, 0, 14), l, 100);
  CheckEqual(c[0], 100, 'temperature 1e-310: greedy');
  l[0] := 0;
  l[1] := Ln(2);
  l[2] := Ln(4);
  c := Draws(Sampling(0, 0, 1, 0, 8), l, 100);
  CheckEqual(c[2], 100, 'temperature 0: greedy');
  // NaN and -Inf are never drawn
  odd4[0] := SingleOfBits($7fc00000);
  odd4[1] := 1;
  odd4[2] := SingleOfBits($ff800000);
  odd4[3] := 1;
  c := Draws(Sampling(1, 0, 1, 0, 9), odd4, N);
  Check((c[0] = 0) and (c[2] = 0) and Near(c[1], 0.5) and Near(c[3], 0.5),
    'NaN and -Inf never drawn');
  // the same seed draws the same tokens, another seed others
  a := TInferSampler.Create(Sampling(1, 0, 1, 0, 42));
  b := TInferSampler.Create(Sampling(1, 0, 1, 0, 42));
  try
    same := true;
    for i := 1 to 200 do
      if a.Next(@l, 3) <> b.Next(@l, 3) then
        same := false;
    Check(same, 'the same seed, the same draws');
  finally
    a.Free;
    b.Free;
  end;
  // the same draws on every compiler and CPU: splitmix64 from seed 42,
  // computed independently (Python) - the nearest boundary is 0.017 away
  a := TInferSampler.Create(Sampling(1, 0, 1, 0, 42));
  try
    same := true;
    for i := 0 to high(SEED42) do
      if a.Next(@l, 3) <> SEED42[i] then
        same := false;
    Check(same, 'seed 42 draws 2 1 1 1 0 2 1 2 1 2 1 2 everywhere');
  finally
    a.Free;
  end;
  a := TInferSampler.Create(Sampling(1, 0, 1, 0, 42));
  b := TInferSampler.Create(Sampling(1, 0, 1, 0, 43));
  try
    same := true;
    for i := 1 to 200 do
      if a.Next(@l, 3) <> b.Next(@l, 3) then
        same := false;
    Check(not same, 'another seed, other draws');
  finally
    a.Free;
    b.Free;
  end;
  allRefused := Refused(Sampling(1, 0, 0, 0, 0)) and Refused(Sampling(1, 0, 1.5, 0, 0)) and
    Refused(Sampling(1, 0, 1, 1, 0)) and Refused(Sampling(1, -1, 1, 0, 0)) and
    Refused(Sampling(1, 0, 1, -0.1, 0));
  Check(allRefused, 'settings out of range refused');
  Check(Refused(Sampling(SingleOfBits($7fc00000), 0, 1, 0, 0)), 'NaN temperature');
  // a vocabulary of real size: every draw inside the nucleus, found here by
  // bisection on a probability threshold (no sorting, unlike the sampler)
  SetLength(big, 151936);
  t := 12345;
  for i := 0 to high(big) do
  begin
    t := (t * 1103515245 + 12345) and $7fffffff;
    big[i] := (t / $7fffffff) * 12 - 6;
  end;
  SetLength(prob, length(big));
  m := big[InferArgMax(pointer(big), length(big))];
  sum := 0;
  for i := 0 to high(big) do
  begin
    prob[i] := Exp((big[i] - m) / 0.7);
    sum := sum + prob[i];
  end;
  for i := 0 to high(big) do
    prob[i] := prob[i] / sum;
  // the nucleus: every token at or above the largest threshold whose tokens
  // still sum to 0.9
  lo := 0;
  hi := 1;
  for k := 1 to 80 do
  begin
    mid := (lo + hi) / 2;
    sum := 0;
    for i := 0 to high(prob) do
      if prob[i] >= mid then
        sum := sum + prob[i];
    if sum >= 0.9 then
      lo := mid
    else
      hi := mid;
  end;
  SetLength(inNucleus, length(big));
  for i := 0 to high(prob) do
    inNucleus[i] := prob[i] >= lo;
  // the top 20: the largest threshold with 20 tokens at or above it
  lo := 0;
  hi := 1;
  for k := 1 to 80 do
  begin
    mid := (lo + hi) / 2;
    t := 0;
    for i := 0 to high(prob) do
      if prob[i] >= mid then
        inc(t);
    if t >= 20 then
      lo := mid
    else
      hi := mid;
  end;
  top20 := lo;
  SetLength(seen, length(big));
  a := TInferSampler.Create(Sampling(0.7, 0, 0.9, 0, 77));
  try
    same := true;
    distinct := 0;
    for i := 1 to 100 do
    begin
      k := a.Next(pointer(big), length(big));
      if not inNucleus[k] then
        same := false;
      if not seen[k] then
        inc(distinct);
      seen[k] := true;
    end;
    Check(same, 'every draw inside the top-p nucleus of 151936 tokens');
    CheckUtf8(distinct > 50, 'and spread over it: % tokens in 100 draws', [distinct]);
  finally
    a.Free;
  end;
  a := TInferSampler.Create(Sampling(0.7, 20, 1, 0, 78));
  try
    same := true;
    for i := 0 to high(seen) do
      seen[i] := false;
    distinct := 0;
    for i := 1 to 100 do
    begin
      k := a.Next(pointer(big), length(big));
      if prob[k] < top20 then
        same := false;
      if not seen[k] then
        inc(distinct);
      seen[k] := true;
    end;
    Check(same, 'every draw inside the top 20');
    CheckUtf8(distinct >= 8, 'and spread over them: % tokens', [distinct]);
  finally
    a.Free;
  end;
end;

procedure TTestInferLlmNative.SamplingSettings;
var
  fn: TFileName;
  s1, s2: TInferSampling;
  req: TLlmChatRequest;
  raised: boolean;

  function Bad(const aExtra: RawUtf8): boolean;
  begin
    result := false;
    req.Extra := _JsonFast(aExtra);
    try
      NativeSampling(req, InferSamplingGreedy);
    except
      on ELlmClient do
        result := true;
    end;
  end;

begin
  fn := TemporaryFileName;
  try
    FileFromString('{"do_sample": true, "temperature": 0.6, "top_k": 20, ' +
      '"top_p": 0.95}', fn);
    s1 := InferSamplingFromConfig(fn);
    Check((Abs(s1.Temperature - 0.6) < 1e-9) and (s1.TopK = 20) and
      (Abs(s1.TopP - 0.95) < 1e-9) and (s1.MinP = 0), 'the Qwen3 defaults');
    FileFromString('{"do_sample": false, "temperature": 0.6}', fn);
    s1 := InferSamplingFromConfig(fn);
    Check((s1.Temperature = 0) and (s1.TopP = 1), 'do_sample false: greedy');
    FileFromString('{"do_sample": true, "top_k": "x"}', fn);
    raised := false;
    try
      InferSamplingFromConfig(fn);
    except
      on EInferError do
        raised := true;
    end;
    Check(raised, 'a value of the wrong kind');
  finally
    DeleteFile(fn);
  end;
  s1 := InferSamplingFromConfig(fn);
  Check(s1.Temperature = 0, 'no file: greedy');
  req := LlmChatRequest('', nil);
  req.Temperature := 0.3;
  req.Extra := _JsonFast('{"top_k": 5, "top_p": 0.5, "min_p": 0.1, "seed": 7}');
  s1 := NativeSampling(req, InferSamplingFromConfig(fn));
  Check((Abs(s1.Temperature - 0.3) < 1e-9) and (s1.TopK = 5) and
    (Abs(s1.TopP - 0.5) < 1e-9) and (Abs(s1.MinP - 0.1) < 1e-9) and (s1.Seed = 7),
    'the request overrides every default');
  req.Temperature := -1;
  req.Extra := _JsonFast('{"top_p": null}');
  s1.Temperature := 0.8;
  s1.TopP := 0.9;
  s2 := NativeSampling(req, s1);
  Check((Abs(s2.Temperature - 0.8) < 1e-9) and (Abs(s2.TopP - 0.9) < 1e-9),
    'omitted and null keep the defaults');
  s1 := NativeSampling(req, s2);
  Check(s1.Seed <> s2.Seed, 'no seed: a random one per request');
  Check(Bad('{"top_k": "5"}'), 'top_k as a string');
  Check(Bad('{"top_k": 2.5}'), 'top_k a fraction');
  Check(Bad('{"top_p": 2}'), 'top_p above 1');
  Check(Bad('{"top_p": "x"}'), 'top_p no number');
  Check(Bad('{"min_p": 1}'), 'min_p 1');
  Check(Bad('{"seed": "s"}'), 'seed no integer');
  Check(Bad('{"top_k": true}'), 'top_k a boolean');
  Check(Bad('{"seed": false}'), 'seed a boolean');
  Check(Bad('{"top_p": true}'), 'top_p a boolean');
  req.Extra := Null;
  req.Temperature := SingleOfBits($7fc00000);
  raised := false;
  try
    NativeSampling(req, InferSamplingGreedy);
  except
    on ELlmClient do
      raised := true;
  end;
  Check(raised, 'a NaN temperature');
  fn := TemporaryFileName;
  try
    FileFromString('{"do_sample": true, "top_k": 0.5}', fn);
    raised := false;
    try
      InferSamplingFromConfig(fn);
    except
      on EInferError do
        raised := true;
    end;
    Check(raised, 'a fractional top_k in the config');
    FileFromString('{"do_sample": true, "temperature": true}', fn);
    raised := false;
    try
      InferSamplingFromConfig(fn);
    except
      on EInferError do
        raised := true;
    end;
    Check(raised, 'a boolean temperature in the config');
  finally
    DeleteFile(fn);
  end;
end;

procedure TTestInferLlmNative.Qwen3Sampling;
var
  dir: TFileName;
  client: TNativeLlmClient;
  llm: ILlmClient;
  req: TLlmChatRequest;
  r1, r2, r3: TLlmChatResponse;
  msgs: TLlmMessageDynArray;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  client := TNativeLlmClient.Create(dir);
  llm := client;
  Check((Abs(client.Sampling.Temperature - 0.6) < 1e-6) and
    (client.Sampling.TopK = 20) and (Abs(client.Sampling.TopP - 0.95) < 1e-6),
    'the defaults of generation_config.json');
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'Name a color.');
  req := LlmChatRequest('', msgs);
  req.MaxTokens := 12;
  req.Temperature := 1.2;
  req.Extra := _ObjFast(['chat_template_kwargs', _ObjFast(['enable_thinking', false]),
    'seed', 11]);
  r1 := llm.ChatComplete(req);
  r2 := llm.ChatComplete(req);
  Check(r1.Content <> '', 'an answer');
  CheckEqual(r1.Content, r2.Content, 'the same seed, the same answer');
  req.Extra := _ObjFast(['chat_template_kwargs', _ObjFast(['enable_thinking', false]),
    'seed', 12]);
  r3 := llm.ChatComplete(req);
  AddConsole(FormatString('sampled: "%" / "%"', [r1.Content, r3.Content]));
end;

procedure TTestInferLlmNative.Qwen3ChatClientGguf;
var
  dir, refdir, fn: TFileName;
  ref: TInferRefSet;
  client: TNativeLlmClient;
  llm: ILlmClient;
  p: PDocVariantData;
  req: TLlmChatRequest;
  r: TLlmChatResponse;
  log: TDeltaLog;
  timer: TPrecisionTimer;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  fn := ExpandFileName(IncludeTrailingPathDelimiter(dir) + '..' + PathDelim +
    'qwen3-0.6b-q8_0.gguf');
  if not FileExists(fn) then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b-q8_0.gguf');
    AddConsole('SKIP native client from GGUF: no vendor/models/qwen3-0.6b-q8_0.gguf');
    exit;
  end;
  refdir := FindInferRefDir('qwen3');
  if CheckFailed(refdir <> '', 'qwen3 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  try
    timer.Start;
    client := TNativeLlmClient.CreateFromGguf(fn);
    llm := client;
    AddConsole(FormatString('native client from GGUF loaded in %', [timer.Stop]));
    CheckEqual(llm.Config.DefaultModel, 'qwen3-0.6b-q8_0', 'model name from the file');
    Check(client.Sampling.Temperature = 0, 'greedy without generation_config.json');
    Check(client.Model.Weights = idtQ8_0, 'Q8_0 file');
    // the prompt is tokenized by the file's own tokenizer: the reference count
    p := FindPrompt(ref, 'chat_en');
    if CheckFailed(p <> nil, 'chat_en') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    r := llm.ChatComplete(req);
    CheckEqual(r.Usage.PromptTokens, p^.I['tokens'], 'chat_en prompt tokens');
    CheckEqual(r.Content, 'Paris', 'chat_en content');
    Check(r.FinishReason = lfrStop, 'chat_en stop');
    // streamed: valid UTF-8 deltas, one Done
    p := FindPrompt(ref, 'chat_de');
    if CheckFailed(p <> nil, 'chat_de') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    req.MaxTokens := 24;
    log := TDeltaLog.Create;
    try
      llm.ChatStream(req, log.OnDelta);
      AddConsole(FormatString('chat_de from GGUF: %', [log.Content]));
      Check(log.Content <> '', 'an answer');
      CheckEqual(log.Invalid, 0, 'valid UTF-8 deltas');
      CheckEqual(log.Dones, 1, 'one Done');
    finally
      log.Free;
    end;
  finally
    ref.Free;
  end;
end;

function TTestInferLlmNative.Weather(const aArgumentsJson: RawUtf8): RawUtf8;
begin
  inc(fToolCalls);
  fToolArgs := aArgumentsJson;
  result := '{"temperature": 21, "unit": "celsius", "sky": "sunny"}';
end;

procedure TTestInferLlmNative.Qwen3AgentLoop;
var
  dir: TFileName;
  client: TNativeLlmClient;
  llm: ILlmClient;
  toolbox: TLlmCallbackToolbox;
  tools: ILlmToolbox;
  agent: TLlmAgent;
  tool: TLlmTool;
  msgs: TLlmMessageDynArray;
  r, r2: TLlmChatResponse;
  timer: TPrecisionTimer;
  log: TDeltaLog;
  n: Int64;
  raised: boolean;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  client := TNativeLlmClient.Create(dir);
  llm := client;
  client.EnableThinking := false;
  client.MaxTokens := 40;
  client.Sampling := InferSamplingGreedy; // the agent sets no temperature
  toolbox := TLlmCallbackToolbox.Create;
  tools := toolbox;
  tool.Name := 'get_weather';
  tool.Description := 'Current weather for a city';
  tool.ParametersJson := '{"type":"object","properties":{"city":{"type":"string"}},' +
    '"required":["city"]}';
  toolbox.Add(tool, Weather);
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'What is the weather in Berlin?');
  fToolCalls := 0;
  agent := TLlmAgent.Create(llm, tools, '');
  try
    agent.MaxIterations := 3;
    timer.Start;
    r := agent.Run(msgs);
    AddConsole(FormatString('agent loop in %: %', [timer.Stop, r.Content]));
  finally
    agent.Free;
  end;
  CheckEqual(fToolCalls, 1, 'the tool ran once');
  CheckEqual(fToolArgs, '{"city": "Berlin"}', 'with the city');
  Check(r.ToolCalls = nil, 'the final answer calls no tool');
  Check(r.Content <> '', 'and says something');
  Check(PosEx('21', r.Content) > 0, 'with the tool result in it');
  // the second turn started with the first one: the prompt cache took it
  AddConsole(FormatString('prompt cache: % of % requests, % tokens reused',
    [client.Cache.Hits, client.Cache.Requests, client.Cache.ReusedTokens]));
  CheckEqual(client.Cache.Requests, 2, 'two requests');
  CheckEqual(client.Cache.Hits, 1, 'the second one from the cache');
  Check(client.Cache.ReusedTokens > 100, 'with the first prompt');
  // without the cache: the same answer
  client.Cache.Slots := 0;
  fToolCalls := 0;
  agent := TLlmAgent.Create(llm, tools, '');
  try
    agent.MaxIterations := 3;
    timer.Start;
    r2 := agent.Run(msgs);
    AddConsole(FormatString('agent loop without the cache in %', [timer.Stop]));
  finally
    agent.Free;
  end;
  CheckEqual(r2.Content, r.Content, 'the same answer without the cache');
  CheckEqual(client.Cache.Hits, 1, 'no hit without slots');
  // a cancelled stream: its context is freed, not given back
  client.Cache.Slots := 1;
  n := client.Cache.Requests;
  log := TDeltaLog.Create;
  try
    log.Fail := true;
    raised := false;
    try
      llm.ChatStream(LlmChatRequest('', msgs), log.OnDelta);
    except
      on ELlmClient do
        raised := true;
    end;
  finally
    log.Free;
  end;
  Check(raised, 'the stream was cancelled');
  CheckEqual(client.Cache.Requests, n, 'a failed request is not given back');
end;


procedure TTestInferLlmNative.NativeFamilies;
var
  f: TNativeFamily;
  p: TNativeOutputParser;
  refused: boolean;
begin
  Check(NativeFamily(TQwen3Model, f), 'qwen3 registered');
  Check(f.Model = TQwen3Model, 'model class');
  Check(f.Tokenizer = TBpeTokenizer, 'tokenizer class');
  Check(f.Template = TQwen3ChatTemplate, 'template class');
  Check(f.Parser = TQwen3OutputParser, 'parser class');
  Check(NativeFamily(TQwen3MoeModel, f) and (f.Template = TQwen3ChatTemplate) and
    (f.Parser = TQwen3OutputParser), 'qwen3_moe: the classes of Qwen3');
  Check(NativeFamily(TQwen2Model, f) and (f.Tokenizer = TBpeTokenizer) and
    (f.Template = TQwen2ChatTemplate) and (f.Parser = TQwen2OutputParser),
    'qwen2: its template and parser');
  Check(NativeFamily(TQwen3Model, f), 'qwen3 again');
  // the classes create what the client uses
  p := f.Parser.Create(fTokenizer);
  try
    Check(p is TQwen3OutputParser, 'a TQwen3OutputParser');
    Check(p.Push(IM_END), 'its stop token');
  finally
    p.Free;
  end;
  // a class without a family, and registrations that change nothing
  Check(not NativeFamily(TDecoderModel, f), 'TDecoderModel: none');
  Check(f.Model = nil, 'cleared');
  RegisterNativeFamily(TDecoderModel, nil, TQwen3ChatTemplate, TQwen3OutputParser);
  RegisterNativeFamily(TDecoderModel, TBpeTokenizer, nil, TQwen3OutputParser);
  RegisterNativeFamily(TDecoderModel, TBpeTokenizer, TQwen3ChatTemplate, nil);
  Check(not NativeFamily(TDecoderModel, f), 'a class missing: not registered');
  RegisterNativeFamily(nil, TBpeTokenizer, TQwen3ChatTemplate, TQwen3OutputParser);
  Check(not NativeFamily(nil, f), 'no model class: not registered');
  RegisterNativeFamily(TQwen3Model, TBpeTokenizer, TQwen3ChatTemplate, TNativeOutputParser);
  Check(NativeFamily(TQwen3Model, f) and (f.Parser = TQwen3OutputParser),
    'the first classes stay');
  refused := false;
  try
    TQwen3OutputParser.Create(nil).Free;
  except
    on EInferError do
      refused := true;
  end;
  Check(refused, 'a parser without a tokenizer');
end;


procedure TTestInferLlmNative.Qwen3MoeChatClientGguf;
var
  dir, refdir, fn: TFileName;
  ref: TInferRefSet;
  client: TNativeLlmClient;
  llm: ILlmClient;
  p: PDocVariantData;
  req: TLlmChatRequest;
  r: TLlmChatResponse;
  timer: TPrecisionTimer;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  // a large model of its own: only where it was put (not LAI_REQUIRE_MODEL)
  fn := ExpandFileName(IncludeTrailingPathDelimiter(dir) + '..' + PathDelim +
    'qwen3-30b-a3b-q4_k_m.gguf');
  if not FileExists(fn) then
  begin
    AddConsole('SKIP Qwen3-MoE client: no vendor/models/qwen3-30b-a3b-q4_k_m.gguf');
    exit;
  end;
  refdir := FindInferRefDir('qwen3');
  if CheckFailed(refdir <> '', 'qwen3 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  try
    timer.Start;
    client := TNativeLlmClient.CreateFromGguf(fn);
    llm := client;
    AddConsole(FormatString('Qwen3-30B-A3B client loaded in %', [timer.Stop]));
    Check(client.Model is TQwen3MoeModel, 'a TQwen3MoeModel');
    CheckEqual(client.Model.Config.Experts, 128, 'experts');
    CheckEqual(client.Model.Config.ExpertsUsed, 8, 'experts per token');
    // the same tokenizer and template as Qwen3-0.6B: the same prompt tokens
    p := FindPrompt(ref, 'chat_en');
    if CheckFailed(p <> nil, 'chat_en') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    // a broken model would talk until the default cap of 4096 tokens
    req.MaxTokens := 16;
    timer.Start;
    r := llm.ChatComplete(req);
    AddConsole(FormatString('Qwen3-30B-A3B chat_en: "%" (% tokens in %)',
      [r.Content, r.Usage.CompletionTokens, timer.Stop]));
    CheckEqual(r.Usage.PromptTokens, p^.I['tokens'], 'chat_en prompt tokens');
    Check(PosEx('Paris', r.Content) > 0, 'chat_en: Paris');
    Check(r.FinishReason = lfrStop, 'chat_en stop');
  finally
    ref.Free;
  end;
end;

procedure TTestInferLlmNative.Qwen25ChatClient;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  client: TNativeLlmClient;
  llm: ILlmClient;
  p: PDocVariantData;
  req: TLlmChatRequest;
  r: TLlmChatResponse;
  timer: TPrecisionTimer;
begin
  {$ifdef CPU32}
  AddConsole('SKIP Qwen2.5 client: 2 GB of FP32 weights need a 64-bit process');
  exit;
  {$endif CPU32}
  dir := FindVendorModel('qwen2.5-0.5b-instruct' + PathDelim + 'model.safetensors');
  if dir = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen2.5-0.5b-instruct');
    AddConsole('SKIP Qwen2.5 client: no vendor/models/qwen2.5-0.5b-instruct');
    exit;
  end;
  dir := ExtractFilePath(dir);
  refdir := FindInferRefDir('qwen25');
  if CheckFailed(refdir <> '', 'qwen25 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  try
    timer.Start;
    client := TNativeLlmClient.Create(dir);
    llm := client;
    AddConsole(FormatString('Qwen2.5-0.5B client loaded in %', [timer.Stop]));
    Check(client.Model is TQwen2Model, 'a TQwen2Model');
    Check(client.Template is TQwen2ChatTemplate, 'the Qwen2.5 template');
    // a plain answer
    p := FindPrompt(ref, 'chat_en');
    if CheckFailed(p <> nil, 'chat_en') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    req.MaxTokens := 16;
    r := llm.ChatComplete(req);
    CheckEqual(r.Usage.PromptTokens, p^.I['tokens'], 'chat_en prompt tokens');
    CheckUtf8(r.Content = 'Paris', 'chat_en: "%"', [r.Content]);
    Check(r.FinishReason = lfrStop, 'chat_en stop');
    Check(VarIsEmptyOrNull(r.Raw), 'no reasoning');
    // a tool call, parsed as for Qwen3
    p := FindPrompt(ref, 'chat_tool');
    if CheckFailed(p <> nil, 'chat_tool') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    req.MaxTokens := 48;
    timer.Start;
    r := llm.ChatComplete(req);
    AddConsole(FormatString('Qwen2.5 chat_tool: % call(s), "%" (% tokens in %)',
      [length(r.ToolCalls), r.Content, r.Usage.CompletionTokens, timer.Stop]));
    CheckEqual(r.Usage.PromptTokens, p^.I['tokens'], 'chat_tool prompt tokens');
    if not CheckFailed(length(r.ToolCalls) = 1, 'one tool call') then
    begin
      CheckUtf8(r.ToolCalls[0].Name = 'get_weather', 'name %', [r.ToolCalls[0].Name]);
      CheckUtf8(r.ToolCalls[0].ArgumentsJson = '{"city": "Berlin"}', 'arguments %',
        [r.ToolCalls[0].ArgumentsJson]);
    end;
    Check(r.Content = '', 'no content beside the call');
    Check(r.FinishReason = lfrToolCalls, 'finish reason tool_calls');
  finally
    ref.Free;
  end;
end;


end.
