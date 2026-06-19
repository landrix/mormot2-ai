// - regression tests for mormot.ai.llm.anthropic (native Anthropic Messages wire)
// - hermetic: exercises request building, response parsing and SSE decoding with
//   canned payloads, no network
unit test.llm.anthropic;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.json,
  mormot.core.test,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.sse,
  mormot.ai.llm.anthropic,
  test.llm.sse; // reuse TSseCollector

type
  TTestLlmAnthropic = class(TSynTestCase)
  protected
    // feed an SSE body through a fresh Anthropic parser (whole payload at once)
    function ParseSse(const aSse: RawUtf8; out aStream: TAnthropicSseStream): TSseCollector;
  published
    procedure RequestHoistsSystem;
    procedure RequestToolsInputSchema;
    procedure RequestToolRoundTrip;
    procedure RequestParallelToolResults;
    procedure RequestVision;
    procedure RequestVisionDefaultMedia;
    procedure ResponseTextAndUsage;
    procedure ResponseToolUse;
    procedure ResponseToolUseNoInput;
    procedure StopReasonMapping;
    procedure SseTextStream;
    procedure SseToolStream;
    procedure SseErrorEvent;
  end;


implementation

{ helpers }

function TTestLlmAnthropic.ParseSse(const aSse: RawUtf8;
  out aStream: TAnthropicSseStream): TSseCollector;
var
  coll: TSseCollector;
  data: RawUtf8;
begin
  coll := TSseCollector.Create;
  aStream := TAnthropicSseStream.Create(coll.Handle);
  data := aSse; // a local var: FPC forbids pointer() on a const string
  aStream.WriteBuffer(pointer(data)^, length(data));
  aStream.Flush;
  result := coll;
end;

// build a [system, user] request
function SystemUserRequest: TLlmChatRequest;
var
  msgs: TLlmMessageDynArray;
begin
  SetLength(msgs, 2);
  msgs[0] := LlmMessage(lrSystem, 'You are a helpful assistant.');
  msgs[1] := LlmMessage(lrUser, 'Hello there');
  result := LlmChatRequest('claude-opus-4-8', msgs);
end;


{ TTestLlmAnthropic }

procedure TTestLlmAnthropic.RequestHoistsSystem;
var
  req: TLlmChatRequest;
  json: RawUtf8;
  d, messages, m0: PDocVariantData;
  v: variant;
begin
  req := SystemUserRequest;
  json := AnthropicChatRequestJson(req, {stream=}false);
  v := _Json(json);
  d := _Safe(v);
  // system is a top-level field, not a message
  CheckEqual(d^.U['system'], 'You are a helpful assistant.', 'system hoisted');
  CheckEqual(d^.I['max_tokens'], ANTHROPIC_DEFAULT_MAX_TOKENS,
    'max_tokens defaulted (required by the API)');
  messages := d^.A['messages'];
  CheckEqual(messages^.Count, 1, 'only the user message remains in messages[]');
  m0 := messages^._[0];
  CheckEqual(m0^.U['role'], 'user', 'no system role in messages');
  CheckEqual(m0^.U['content'], 'Hello there', 'user content preserved');
  Check(not d^.Exists('stream'), 'no stream flag on a non-streamed request');
end;

procedure TTestLlmAnthropic.RequestToolsInputSchema;
var
  req: TLlmChatRequest;
  json: RawUtf8;
  d, tools, t0: PDocVariantData;
  v: variant;
begin
  req := SystemUserRequest;
  SetLength(req.Tools, 1);
  req.Tools[0].Name := 'get_weather';
  req.Tools[0].Description := 'Get the weather';
  req.Tools[0].ParametersJson :=
    '{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}';
  json := AnthropicChatRequestJson(req, false);
  v := _Json(json);
  d := _Safe(v);
  tools := d^.A['tools'];
  CheckEqual(tools^.Count, 1, 'one tool');
  t0 := tools^._[0];
  CheckEqual(t0^.U['name'], 'get_weather', 'tool name');
  // Anthropic uses input_schema, not function/parameters
  Check(t0^.Exists('input_schema'), 'tool carries input_schema');
  Check(not t0^.Exists('parameters'), 'no OpenAI-style parameters key');
  CheckEqual(t0^.O['input_schema']^.U['type'], 'object', 'schema type preserved');
end;

procedure TTestLlmAnthropic.RequestToolRoundTrip;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  json: RawUtf8;
  d, messages, asst, content, block, userMsg, tr: PDocVariantData;
  v: variant;
begin
  // user -> assistant(tool_use) -> tool(result): mirrors the agent loop history
  SetLength(msgs, 3);
  msgs[0] := LlmMessage(lrUser, 'weather?');
  msgs[1] := LlmMessage(lrAssistant, '');
  SetLength(msgs[1].ToolCalls, 1);
  msgs[1].ToolCalls[0].Id := 'toolu_1';
  msgs[1].ToolCalls[0].Name := 'get_weather';
  msgs[1].ToolCalls[0].ArgumentsJson := '{"location":"NYC"}';
  msgs[2] := LlmMessage(lrTool, '{"temp":"22C"}');
  msgs[2].ToolCallId := 'toolu_1';
  req := LlmChatRequest('claude-opus-4-8', msgs);

  json := AnthropicChatRequestJson(req, false);
  v := _Json(json);
  d := _Safe(v);
  messages := d^.A['messages'];
  CheckEqual(messages^.Count, 3, 'user + assistant + tool-result user');

  // assistant turn carries a tool_use content block
  asst := messages^._[1];
  CheckEqual(asst^.U['role'], 'assistant', 'assistant role');
  content := asst^.A['content'];
  CheckEqual(content^.Count, 1, 'assistant has one tool_use block (no text)');
  block := content^._[0];
  CheckEqual(block^.U['type'], 'tool_use', 'tool_use block');
  CheckEqual(block^.U['id'], 'toolu_1', 'tool_use id');
  CheckEqual(block^.U['name'], 'get_weather', 'tool_use name');
  CheckEqual(block^.O['input']^.U['location'], 'NYC', 'arguments parsed to object');

  // tool result is a tool_result block inside a user message
  userMsg := messages^._[2];
  CheckEqual(userMsg^.U['role'], 'user', 'tool result rides a user message');
  tr := userMsg^.A['content']^._[0];
  CheckEqual(tr^.U['type'], 'tool_result', 'tool_result block');
  CheckEqual(tr^.U['tool_use_id'], 'toolu_1', 'tool_use_id links the call');
  CheckEqual(tr^.U['content'], '{"temp":"22C"}', 'tool result content');
end;

procedure TTestLlmAnthropic.RequestParallelToolResults;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  json: RawUtf8;
  messages, userMsg, content: PDocVariantData;
begin
  // the agent appends ONE lrTool message per parallel tool call; Anthropic
  // requires all tool_result blocks of a turn in a SINGLE user message
  SetLength(msgs, 4);
  msgs[0] := LlmMessage(lrUser, 'weather in two cities?');
  msgs[1] := LlmMessage(lrAssistant, '');
  SetLength(msgs[1].ToolCalls, 2);
  msgs[1].ToolCalls[0].Id := 'toolu_a';
  msgs[1].ToolCalls[0].Name := 'get_weather';
  msgs[1].ToolCalls[0].ArgumentsJson := '{"location":"NYC"}';
  msgs[1].ToolCalls[1].Id := 'toolu_b';
  msgs[1].ToolCalls[1].Name := 'get_weather';
  msgs[1].ToolCalls[1].ArgumentsJson := '{"location":"LA"}';
  msgs[2] := LlmMessage(lrTool, '{"temp":"22C"}');
  msgs[2].ToolCallId := 'toolu_a';
  msgs[3] := LlmMessage(lrTool, '{"temp":"28C"}');
  msgs[3].ToolCallId := 'toolu_b';
  req := LlmChatRequest('claude-opus-4-8', msgs);

  json := AnthropicChatRequestJson(req, {stream=}false);
  messages := _Safe(_Json(json))^.A['messages'];
  // user, assistant, and ONE user message bundling both tool results
  CheckEqual(messages^.Count, 3, 'two tool results coalesced into one user turn');
  userMsg := messages^._[2];
  CheckEqual(userMsg^.U['role'], 'user', 'tool results ride a single user message');
  content := userMsg^.A['content'];
  CheckEqual(content^.Count, 2, 'both tool_result blocks in one message');
  CheckEqual(content^._[0]^.U['tool_use_id'], 'toolu_a', 'first result links call a');
  CheckEqual(content^._[1]^.U['tool_use_id'], 'toolu_b', 'second result links call b');
end;

procedure TTestLlmAnthropic.RequestVision;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  imgs: TLlmImageDynArray;
  json: RawUtf8;
  content, b0, b1, b2, src: PDocVariantData;
begin
  // a multimodal user turn: text + a base64 image + a URL image
  SetLength(imgs, 2);
  imgs[0] := LlmImageBase64('image/jpeg', 'BBBB');
  imgs[1] := LlmImageUrl('https://example.com/roof.png');
  SetLength(msgs, 1);
  msgs[0] := LlmImageMessage(lrUser, 'Describe these', imgs);
  req := LlmChatRequest('claude-opus-4-8', msgs);

  json := AnthropicChatRequestJson(req, {stream=}false);
  content := _Safe(_Json(json))^.A['messages']^._[0]^.A['content'];
  CheckEqual(content^.Count, 3, 'text block + two image blocks');
  b0 := content^._[0];
  CheckEqual(b0^.U['type'], 'text', 'first block is text');
  // base64 image: typed source with media_type + data (no data: URI prefix)
  b1 := content^._[1];
  CheckEqual(b1^.U['type'], 'image', 'second block is image');
  src := b1^.O['source'];
  CheckEqual(src^.U['type'], 'base64', 'base64 source');
  CheckEqual(src^.U['media_type'], 'image/jpeg', 'media type');
  CheckEqual(src^.U['data'], 'BBBB', 'raw base64 data (no data: prefix)');
  // URL image: url source
  b2 := content^._[2];
  src := b2^.O['source'];
  CheckEqual(src^.U['type'], 'url', 'url source');
  CheckEqual(src^.U['url'], 'https://example.com/roof.png', 'image url');
end;

procedure TTestLlmAnthropic.RequestVisionDefaultMedia;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  img: TLlmImage;
  json: RawUtf8;
  src: PDocVariantData;
begin
  // a base64 image without MediaType (manual construction): Anthropic requires
  // media_type, so the serializer must fall back to the default rather than emit
  // an empty/absent media_type that the API rejects
  Finalize(img);
  FillCharFast(img, SizeOf(img), 0);
  img.Source := lisBase64;
  img.Data := 'CCCC';
  SetLength(msgs, 1);
  msgs[0] := LlmImageMessage(lrUser, '', nil);
  SetLength(msgs[0].Images, 1);
  msgs[0].Images[0] := img;
  req := LlmChatRequest('claude-opus-4-8', msgs);

  json := AnthropicChatRequestJson(req, {stream=}false);
  src := _Safe(_Json(json))^.A['messages']^._[0]^.A['content']^._[0]^.O['source'];
  CheckEqual(src^.U['media_type'], 'image/png', 'empty media_type defaults to png');
  CheckEqual(src^.U['data'], 'CCCC', 'raw data preserved');
end;

procedure TTestLlmAnthropic.ResponseTextAndUsage;
const
  RESP_JSON =
    '{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8",' +
    '"content":[{"type":"text","text":"Hello"},{"type":"text","text":", world"}],' +
    '"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}';
var
  resp: TLlmChatResponse;
begin
  resp := ParseAnthropicChatResponse(RESP_JSON);
  CheckEqual(resp.Content, 'Hello, world', 'text blocks concatenated');
  CheckEqual(length(resp.ToolCalls), 0, 'no tool calls');
  Check(resp.FinishReason = lfrStop, 'end_turn -> lfrStop');
  CheckEqual(resp.Model, 'claude-opus-4-8', 'model echoed');
  CheckEqual(resp.Usage.PromptTokens, 10, 'input tokens');
  CheckEqual(resp.Usage.CompletionTokens, 5, 'output tokens');
  CheckEqual(resp.Usage.TotalTokens, 15, 'total tokens summed');
end;

procedure TTestLlmAnthropic.ResponseToolUse;
const
  RESP_JSON =
    '{"id":"msg_2","type":"message","role":"assistant","model":"claude-opus-4-8",' +
    '"content":[{"type":"text","text":"Let me check."},' +
    '{"type":"tool_use","id":"toolu_9","name":"get_weather","input":{"location":"NYC"}}],' +
    '"stop_reason":"tool_use","usage":{"input_tokens":12,"output_tokens":7}}';
var
  resp: TLlmChatResponse;
begin
  resp := ParseAnthropicChatResponse(RESP_JSON);
  CheckEqual(resp.Content, 'Let me check.', 'text projected alongside the tool call');
  CheckEqual(length(resp.ToolCalls), 1, 'one tool call');
  CheckEqual(resp.ToolCalls[0].Id, 'toolu_9', 'tool id');
  CheckEqual(resp.ToolCalls[0].Name, 'get_weather', 'tool name');
  Check(Pos(RawUtf8('"location":"NYC"'), resp.ToolCalls[0].ArgumentsJson) > 0,
    'input re-serialized to arguments JSON');
  Check(resp.FinishReason = lfrToolCalls, 'tool_use -> lfrToolCalls');
end;

procedure TTestLlmAnthropic.ResponseToolUseNoInput;
const
  // a (malformed/partial) tool_use block with no `input` field
  RESP_JSON =
    '{"id":"msg_3","type":"message","role":"assistant","model":"claude-opus-4-8",' +
    '"content":[{"type":"tool_use","id":"toolu_x","name":"ping"}],' +
    '"stop_reason":"tool_use","usage":{"input_tokens":3,"output_tokens":2}}';
var
  resp: TLlmChatResponse;
begin
  resp := ParseAnthropicChatResponse(RESP_JSON);
  CheckEqual(length(resp.ToolCalls), 1, 'one tool call');
  // a missing input must not become the literal 'null' (mORMot void doc ToJson)
  CheckEqual(resp.ToolCalls[0].ArgumentsJson, '{}',
    'missing input falls back to an empty object, not null');
end;

procedure TTestLlmAnthropic.StopReasonMapping;
begin
  CheckEqual(AnthropicStopToOpenAI('end_turn'), 'stop', 'end_turn');
  CheckEqual(AnthropicStopToOpenAI('stop_sequence'), 'stop', 'stop_sequence');
  CheckEqual(AnthropicStopToOpenAI('max_tokens'), 'length', 'max_tokens');
  CheckEqual(AnthropicStopToOpenAI('tool_use'), 'tool_calls', 'tool_use');
  CheckEqual(AnthropicStopToOpenAI('refusal'), 'content_filter', 'refusal');
  CheckEqual(AnthropicStopToOpenAI('pause_turn'), '', 'unknown -> empty');
end;

const
  // a representative Anthropic text stream; event: lines must be ignored, the
  // type discriminator lives in the data payload
  SSE_TEXT =
    'event: message_start'#10 +
    'data: {"type":"message_start","message":{"role":"assistant","usage":{"input_tokens":10,"output_tokens":1}}}'#10 +
    #10 +
    'event: content_block_start'#10 +
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}'#10 +
    #10 +
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}'#10 +
    #10 +
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}}'#10 +
    #10 +
    'data: {"type":"content_block_stop","index":0}'#10 +
    #10 +
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}'#10 +
    #10 +
    'data: {"type":"message_stop"}'#10#10;

  // a streamed tool call: id+name on content_block_start, args via input_json_delta
  SSE_TOOL =
    'data: {"type":"message_start","message":{"role":"assistant","usage":{"input_tokens":20,"output_tokens":1}}}'#10 +
    #10 +
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}'#10 +
    #10 +
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}'#10 +
    #10 +
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ation\":\"NYC\"}"}}'#10 +
    #10 +
    'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}'#10 +
    #10 +
    'data: {"type":"message_stop"}'#10#10;

procedure TTestLlmAnthropic.SseTextStream;
var
  s: TAnthropicSseStream;
  coll: TSseCollector;
begin
  coll := ParseSse(SSE_TEXT, s);
  try
    CheckEqual(coll.Text, 'Hello, world', 'reconstructed content');
    CheckEqual(s.FullText, 'Hello, world', 'stream accumulated content');
    CheckEqual(coll.FinishReason, 'stop', 'end_turn mapped to stop');
    CheckEqual(coll.DoneCount, 1, 'message_stop yields one Done');
    // TotalTokens must combine input_tokens (message_start) + output_tokens
    // (message_delta) = 10 + 5, not leave the prompt side at zero
    CheckEqual(coll.UsageTotal, 15, 'stream usage total = input + output tokens');
    Check(s.Done, 'stream marked done');
  finally
    s.Free;
    coll.Free;
  end;
end;

procedure TTestLlmAnthropic.SseToolStream;
var
  s: TAnthropicSseStream;
  coll: TSseCollector;
begin
  coll := ParseSse(SSE_TOOL, s);
  try
    CheckEqual(coll.ToolName, 'get_weather', 'tool name from content_block_start');
    CheckEqual(coll.ToolArgs, '{"location":"NYC"}', 'args assembled from input_json_delta');
    CheckEqual(coll.FinishReason, 'tool_calls', 'tool_use mapped to tool_calls');
    Check(s.Done, 'tool-call stream done');
  finally
    s.Free;
    coll.Free;
  end;
end;

const
  // an inband server error: HTTP 200, a partial answer, then an `error` event
  SSE_ERROR =
    'data: {"type":"message_start","message":{"role":"assistant","usage":{"input_tokens":5,"output_tokens":1}}}'#10 +
    #10 +
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Partial"}}'#10 +
    #10 +
    'event: error'#10 +
    'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'#10#10;

procedure TTestLlmAnthropic.SseErrorEvent;
var
  s: TAnthropicSseStream;
  coll: TSseCollector;
begin
  coll := ParseSse(SSE_ERROR, s);
  try
    // the error must be captured (so ChatStream can raise) rather than dropped;
    // the stream also ends so the read loop stops
    Check(Pos(RawUtf8('Overloaded'), s.StreamError) > 0, 'error message captured');
    Check(s.Done, 'error ends the stream');
  finally
    s.Free;
    coll.Free;
  end;
end;

end.
