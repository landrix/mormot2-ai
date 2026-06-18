// - regression tests for mormot.ai.llm (request building + response parsing)
unit test.llm.client;

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
  mormot.ai.llm.openai;

type
  TTestLlmClient = class(TSynTestCase)
  published
    procedure RequestJsonMinimal;
    procedure RequestJsonWithToolsAndParams;
    procedure ResponseParsing;
    procedure ProviderConfigs;
    procedure ExtraOverridesWithoutDuplicateKey;
    procedure AssistantToolOnlyOmitsContent;
  end;


implementation

// count non-overlapping occurrences of aSub in aText
function CountSubstr(const aSub, aText: RawUtf8): integer;
var
  n: PtrInt;
begin
  result := 0;
  n := 1;
  repeat
    n := PosEx(aSub, aText, n);
    if n = 0 then
      break;
    inc(result);
    inc(n, length(aSub));
  until false;
end;

{ TTestLlmClient }

procedure TTestLlmClient.RequestJsonMinimal;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  json: RawUtf8;
  d, m0: PDocVariantData;
begin
  SetLength(msgs, 2);
  msgs[0] := LlmMessage(lrSystem, 'You are terse.');
  msgs[1] := LlmMessage(lrUser, 'Hi');
  req := LlmChatRequest('gpt-4o-mini', msgs);

  json := OpenAIChatRequestJson(req, {stream=}false);
  d := _Safe(_Json(json));
  CheckEqual(d^.U['model'], 'gpt-4o-mini', 'model');
  Check(Pos(RawUtf8('"stream":false'), json) > 0, 'stream false present');
  // omitted fields must be absent (sentinel Temperature=-1, MaxTokens=0)
  Check(Pos(RawUtf8('"temperature"'), json) = 0, 'temperature omitted');
  Check(Pos(RawUtf8('"max_tokens"'), json) = 0, 'max_tokens omitted');
  Check(Pos(RawUtf8('"tools"'), json) = 0, 'tools omitted');
  CheckEqual(d^.A['messages']^.Count, 2, 'two messages');
  m0 := d^.A['messages']^._[0];
  CheckEqual(m0^.U['role'], 'system', 'first role');
  CheckEqual(m0^.U['content'], 'You are terse.', 'first content');
  CheckEqual(d^.A['messages']^._[1]^.U['role'], 'user', 'second role');
end;

procedure TTestLlmClient.RequestJsonWithToolsAndParams;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  json: RawUtf8;
  d, tool0, fn, params: PDocVariantData;
begin
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'Weather in NYC?');
  req := LlmChatRequest('gpt-4o', msgs);
  req.Temperature := 0.7;
  req.MaxTokens := 256;
  SetLength(req.Tools, 1);
  req.Tools[0].Name := 'get_weather';
  req.Tools[0].Description := 'Get the weather for a location';
  req.Tools[0].ParametersJson :=
    '{"type":"object","properties":{"loc":{"type":"string"}},"required":["loc"]}';

  json := OpenAIChatRequestJson(req, {stream=}true);
  d := _Safe(_Json(json));
  Check(Pos(RawUtf8('"stream":true'), json) > 0, 'stream true');
  Check(Pos(RawUtf8('"temperature":0.7'), json) > 0, 'temperature present');
  Check(Pos(RawUtf8('"max_tokens":256'), json) > 0, 'max_tokens present');
  CheckEqual(d^.A['tools']^.Count, 1, 'one tool');
  tool0 := d^.A['tools']^._[0];
  CheckEqual(tool0^.U['type'], 'function', 'tool type');
  fn := tool0^.O['function'];
  CheckEqual(fn^.U['name'], 'get_weather', 'tool name');
  // the raw JSON-Schema must be embedded as a real nested object, not a string
  params := fn^.O['parameters'];
  CheckEqual(params^.U['type'], 'object', 'parameters is an object');
  CheckEqual(params^.O['properties']^.O['loc']^.U['type'], 'string', 'nested schema');
end;

procedure TTestLlmClient.ResponseParsing;
const
  RESP_JSON =
    '{"id":"chatcmpl-1","model":"llama3.2","choices":[{"index":0,' +
    '"message":{"role":"assistant","content":"Hi there",' +
    '"tool_calls":[{"id":"call_1","type":"function","function":' +
    '{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}]},' +
    '"finish_reason":"tool_calls"}],' +
    '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}';
var
  r: TLlmChatResponse;
begin
  r := ParseOpenAIChatResponse(RESP_JSON);
  CheckEqual(r.Model, 'llama3.2', 'model');
  CheckEqual(r.Content, 'Hi there', 'content');
  Check(r.FinishReason = lfrToolCalls, 'finish reason');
  CheckEqual(length(r.ToolCalls), 1, 'one tool call');
  CheckEqual(r.ToolCalls[0].Id, 'call_1', 'tool call id');
  CheckEqual(r.ToolCalls[0].Name, 'get_weather', 'tool call name');
  CheckEqual(r.ToolCalls[0].ArgumentsJson, '{"loc":"NYC"}', 'tool call args');
  CheckEqual(r.Usage.PromptTokens, 5, 'prompt tokens');
  CheckEqual(r.Usage.TotalTokens, 7, 'total tokens');
end;

procedure TTestLlmClient.ProviderConfigs;
var
  c: TLlmProviderConfig;
begin
  c := OpenAIConfig('sk-test');
  CheckEqual(c.BaseUrl, 'https://api.openai.com/v1', 'openai base url');
  Check(c.AuthScheme = lasBearer, 'openai bearer');
  CheckEqual(c.DefaultModel, 'gpt-4o-mini', 'openai default model');

  c := OllamaConfig('http://172.16.122.3:11434/v1', 'llama3.2');
  CheckEqual(c.BaseUrl, 'http://172.16.122.3:11434/v1', 'ollama base url');
  Check(c.AuthScheme = lasNone, 'ollama no auth');

  c := LiteLLMConfig('http://proxy:4000/v1', 'sk-litellm', 'claude-3-5-sonnet');
  Check(c.AuthScheme = lasBearer, 'litellm bearer');
  CheckEqual(c.DefaultModel, 'claude-3-5-sonnet', 'litellm model');
end;

procedure TTestLlmClient.ExtraOverridesWithoutDuplicateKey;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  json: RawUtf8;
  d: PDocVariantData;
begin
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'hi');
  req := LlmChatRequest('base-model', msgs);
  // Extra repeats 'model' (override) and adds 'seed' (passthrough)
  req.Extra := _ObjFast(['model', 'override-model', 'seed', 7]);

  json := OpenAIChatRequestJson(req, {stream=}false);
  // the merge must overwrite, not append a second "model" key (invalid JSON)
  CheckEqual(CountSubstr('"model"', json), 1, 'model key appears once');
  d := _Safe(_Json(json));
  CheckEqual(d^.U['model'], 'override-model', 'Extra overrides the base model');
  CheckEqual(d^.I['seed'], 7, 'Extra passthrough field merged');
end;

procedure TTestLlmClient.AssistantToolOnlyOmitsContent;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  am: TLlmMessage;
  json: RawUtf8;
  m0: PDocVariantData;
begin
  // an assistant turn that only calls a tool: content must be omitted, not ""
  am := LlmMessage(lrAssistant, '');
  SetLength(am.ToolCalls, 1);
  am.ToolCalls[0].Id := 'c1';
  am.ToolCalls[0].Name := 'f';
  am.ToolCalls[0].ArgumentsJson := '{}';
  SetLength(msgs, 1);
  msgs[0] := am;
  req := LlmChatRequest('m', msgs);

  json := OpenAIChatRequestJson(req, {stream=}false);
  Check(Pos(RawUtf8('"content":""'), json) = 0, 'no empty-string content');
  m0 := _Safe(_Json(json))^.A['messages']^._[0];
  Check(m0^.GetValueIndex('content') < 0, 'content omitted for tool-only assistant');
  CheckEqual(m0^.A['tool_calls']^.Count, 1, 'tool_calls present');
end;

end.
