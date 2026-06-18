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
    procedure RequestJsonVision;
    procedure VisionDefaultMediaType;
    procedure ResponseParsing;
    procedure ProviderConfigs;
    procedure ExtraOverridesWithoutDuplicateKey;
    procedure AssistantToolOnlyOmitsContent;
    procedure EmbeddingsRequestJson;
    procedure EmbeddingsParsing;
    procedure EmbeddingsParsingReordered;
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

procedure TTestLlmClient.RequestJsonVision;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  imgs: TLlmImageDynArray;
  json: RawUtf8;
  content, p0, p1, p2: PDocVariantData;
begin
  // a multimodal user turn: text + a base64 image + a URL image
  SetLength(imgs, 2);
  imgs[0] := LlmImageBase64('image/png', 'AAAA');
  imgs[1] := LlmImageUrl('https://example.com/cat.png');
  SetLength(msgs, 1);
  msgs[0] := LlmImageMessage(lrUser, 'What is in these?', imgs);
  req := LlmChatRequest('gpt-4o-mini', msgs);

  json := OpenAIChatRequestJson(req, {stream=}false);
  // content is an array of typed parts, not a plain string
  content := _Safe(_Json(json))^.A['messages']^._[0]^.A['content'];
  CheckEqual(content^.Count, 3, 'text part + two image parts');
  p0 := content^._[0];
  CheckEqual(p0^.U['type'], 'text', 'first part is text');
  CheckEqual(p0^.U['text'], 'What is in these?', 'text content');
  p1 := content^._[1];
  CheckEqual(p1^.U['type'], 'image_url', 'second part is image_url');
  // base64 is inlined as a data: URI
  CheckEqual(p1^.O['image_url']^.U['url'], 'data:image/png;base64,AAAA',
    'base64 image inlined as data URI');
  p2 := content^._[2];
  CheckEqual(p2^.O['image_url']^.U['url'], 'https://example.com/cat.png',
    'URL image passed through');
end;

procedure TTestLlmClient.VisionDefaultMediaType;
var
  img: TLlmImage;
begin
  // a base64 image constructed without a MediaType (bypassing LlmImageBase64)
  // must still serialize to a valid data: URI, not 'data:;base64,...'
  Finalize(img);
  FillCharFast(img, SizeOf(img), 0);
  img.Source := lisBase64;
  img.Data := 'ZZZZ';
  CheckEqual(LlmImageMediaType(img), 'image/png', 'empty media defaults to png');
  CheckEqual(LlmImageDataUri(img), 'data:image/png;base64,ZZZZ',
    'data URI uses the default media type');
  // an explicit media type is preserved
  img.MediaType := 'image/webp';
  CheckEqual(LlmImageDataUri(img), 'data:image/webp;base64,ZZZZ',
    'explicit media type wins');
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

procedure TTestLlmClient.EmbeddingsRequestJson;
var
  input: TRawUtf8DynArray;
  json: RawUtf8;
  d: PDocVariantData;
begin
  SetLength(input, 2);
  input[0] := 'Hund';
  input[1] := 'Katze';
  json := OpenAIEmbeddingsRequestJson('text-embedding-3-small', input);
  d := _Safe(_Json(json));
  CheckEqual(d^.U['model'], 'text-embedding-3-small', 'model');
  // input must be a real JSON string array, in order, not a single string
  CheckEqual(d^.A['input']^.Count, 2, 'two inputs');
  CheckEqual(VariantToUtf8(d^.A['input']^.Values[0]), 'Hund', 'first input');
  CheckEqual(VariantToUtf8(d^.A['input']^.Values[1]), 'Katze', 'second input');
end;

procedure TTestLlmClient.EmbeddingsParsing;
const
  EMB_RESP =
    '{"object":"list","data":[' +
    '{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]},' +
    '{"object":"embedding","index":1,"embedding":[0.4,0.5,0.6]}],' +
    '"model":"text-embedding-3-small","usage":{"prompt_tokens":4,"total_tokens":4}}';
var
  vecs: TLlmEmbeddingDynArray;
begin
  vecs := ParseOpenAIEmbeddings(EMB_RESP);
  CheckEqual(length(vecs), 2, 'one vector per input');
  CheckEqual(length(vecs[0]), 3, 'vector dimension');
  CheckSame(vecs[0][0], 0.1, 1e-4, 'first component');
  CheckSame(vecs[0][2], 0.3, 1e-4, 'third component');
  CheckSame(vecs[1][0], 0.4, 1e-4, 'second vector first component');
end;

procedure TTestLlmClient.EmbeddingsParsingReordered;
const
  // data[] returned out of order: index 1 before index 0 - the parser must map
  // each vector by its "index", not by array position
  EMB_RESP =
    '{"object":"list","data":[' +
    '{"object":"embedding","index":1,"embedding":[0.4,0.5,0.6]},' +
    '{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],' +
    '"model":"text-embedding-3-small","usage":{"prompt_tokens":4,"total_tokens":4}}';
var
  vecs: TLlmEmbeddingDynArray;
begin
  vecs := ParseOpenAIEmbeddings(EMB_RESP);
  CheckEqual(length(vecs), 2, 'one vector per input');
  // vec for input 0 must be [0.1..] even though it arrived second
  CheckSame(vecs[0][0], 0.1, 1e-4, 'index 0 mapped to slot 0');
  CheckSame(vecs[1][0], 0.4, 1e-4, 'index 1 mapped to slot 1');
end;

end.
