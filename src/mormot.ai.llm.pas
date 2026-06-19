/// LandrixAI LLM Client - HTTP client and provider contract
// - part of the mormot.ai.* extension (LandrixAI)
// - clean-room from public provider API docs (OpenAI Chat Completions); no
//   third-party code, target license MPL/GPL/LGPL (mORMot contribution)
// - the OpenAI wire is canonical: the same TLlmClient drives OpenAI, LiteLLM and
//   Ollama (OpenAI-compatible endpoint) - only the provider config differs
unit mormot.ai.llm;

{
  *****************************************************************************

    - TLlmProviderConfig and the ILlmClient contract
    - OpenAI-wire request building and response parsing (free functions)
    - TLlmClient: blocking ChatComplete and streaming ChatStream over
      THttpClientSocket (the SSE body is parsed live by TLlmSseStream)

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  classes,
  mormot.core.base,
  mormot.core.text,
  mormot.core.buffers,
  mormot.core.variants,
  mormot.core.json,
  mormot.net.sock,
  mormot.net.client,
  mormot.ai.llm.types,
  mormot.ai.llm.sse;

type
  /// how the client authenticates to the provider
  TLlmAuthScheme = (
    lasNone,
    lasBearer);

  /// everything needed to reach one OpenAI-compatible endpoint
  // - BaseUrl points at the API root (e.g. 'https://api.openai.com/v1' or
  //   'http://localhost:11434/v1' for Ollama); '/chat/completions' is appended
  TLlmProviderConfig = record
    BaseUrl: RawUtf8;
    ApiKey: RawUtf8;
    AuthScheme: TLlmAuthScheme;
    DefaultModel: RawUtf8;
    /// socket connect/receive timeout in milliseconds (0 = library default)
    TimeoutMs: integer;
    /// optional cap on the total streamed response size in bytes (0 = unlimited)
    // - guards ChatStream against a runaway/oversized stream: the per-chunk
    //   global MaxHttpChunkSize bounds one chunk, this bounds the cumulative body
    MaxResponseBytes: Int64;
  end;

  /// raised on transport or provider errors
  ELlmClient = class(ESynException);

  /// the provider contract - one implementation per genuinely different wire
  ILlmClient = interface
    ['{6F1A9C24-7B3E-4D58-9A0C-1E2F3A4B5C6D}']
    /// a blocking, non-streamed chat completion
    function ChatComplete(const aRequest: TLlmChatRequest): TLlmChatResponse;
    /// a streamed chat completion: aOnDelta fires per SSE event as it arrives
    procedure ChatStream(const aRequest: TLlmChatRequest;
      const aOnDelta: TLlmStreamDeltaEvent);
    /// the active provider configuration
    function Config: TLlmProviderConfig;
  end;


/// build the OpenAI Chat Completions request JSON for a neutral request
// - aStream toggles the "stream" flag; omitted fields stay absent
function OpenAIChatRequestJson(const aRequest: TLlmChatRequest;
  aStream: boolean): RawUtf8;

/// parse a (non-streamed) OpenAI Chat Completions response into the neutral record
function ParseOpenAIChatResponse(const aJson: RawUtf8): TLlmChatResponse;

/// build the OpenAI Embeddings request JSON for a model + input texts
function OpenAIEmbeddingsRequestJson(const aModel: RawUtf8;
  const aInput: TRawUtf8DynArray): RawUtf8;

/// parse an OpenAI Embeddings response (data[].embedding) into vectors
// - honours each item's "index" field (the provider may reorder data[]), so a
//   batch response is mapped back to input order rather than array position
function ParseOpenAIEmbeddings(const aJson: RawUtf8): TLlmEmbeddingDynArray;


type
  /// OpenAI-wire LLM client driving OpenAI, LiteLLM and Ollama via config
  TLlmClient = class(TInterfacedObject, ILlmClient)
  protected
    fConfig: TLlmProviderConfig;
    // open a fresh connection to BaseUrl + aEndpoint; returns the request path in
    // aPath; sets the bearer header when configured
    function Connect(const aEndpoint: RawUtf8; out aPath: RawUtf8): THttpClientSocket;
  public
    /// create the client for a given provider configuration
    constructor Create(const aConfig: TLlmProviderConfig); reintroduce;
    function ChatComplete(const aRequest: TLlmChatRequest): TLlmChatResponse;
    procedure ChatStream(const aRequest: TLlmChatRequest;
      const aOnDelta: TLlmStreamDeltaEvent);
    /// embed one or more input texts via the OpenAI-wire /embeddings endpoint
    // - one vector per input, in order; works against OpenAI/LiteLLM (and Ollama
    //   which also exposes /v1/embeddings)
    function Embeddings(const aModel: RawUtf8;
      const aInput: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    function Config: TLlmProviderConfig;
  end;


implementation

{ ************ Request building }

// add a single chat message object to the messages array variant
procedure AddMessage(const aMessages: variant; const aMsg: TLlmMessage);
var
  m, calls, call, parts: variant;
  i: PtrInt;
begin
  if length(aMsg.Images) > 0 then
  begin
    // multimodal: content becomes an array of typed parts (text + image_url);
    // a base64 image is inlined as a data: URI, a URL is passed through
    parts := _Arr([]);
    if aMsg.Content <> '' then
      _Safe(parts)^.AddItem(_ObjFast(['type', 'text', 'text', aMsg.Content]));
    for i := 0 to high(aMsg.Images) do
      _Safe(parts)^.AddItem(_ObjFast([
        'type', 'image_url',
        'image_url', _ObjFast(['url', LlmImageDataUri(aMsg.Images[i])])]));
    m := _ObjFast(['role', LlmRoleText(aMsg.Role), 'content', parts]);
  end
  // an assistant turn that only calls tools omits content entirely: per the
  // OpenAI spec content is optional when tool_calls is present, and an empty
  // string "" is rejected by strict validators
  else if (aMsg.Content = '') and (aMsg.Role = lrAssistant) and
     (length(aMsg.ToolCalls) > 0) then
    m := _ObjFast(['role', LlmRoleText(aMsg.Role)])
  else
    m := _ObjFast(['role', LlmRoleText(aMsg.Role), 'content', aMsg.Content]);
  if aMsg.Name <> '' then
    _Safe(m)^.AddValue('name', aMsg.Name);
  if aMsg.ToolCallId <> '' then
    _Safe(m)^.AddValue('tool_call_id', aMsg.ToolCallId);
  if length(aMsg.ToolCalls) > 0 then
  begin
    calls := _Arr([]);
    for i := 0 to high(aMsg.ToolCalls) do
    begin
      call := _ObjFast([
        'id', aMsg.ToolCalls[i].Id,
        'type', 'function',
        'function', _ObjFast([
          'name', aMsg.ToolCalls[i].Name,
          'arguments', aMsg.ToolCalls[i].ArgumentsJson])]);
      _Safe(calls)^.AddItem(call);
    end;
    _Safe(m)^.AddValue('tool_calls', calls);
  end;
  _Safe(aMessages)^.AddItem(m);
end;

function OpenAIChatRequestJson(const aRequest: TLlmChatRequest;
  aStream: boolean): RawUtf8;
var
  body, messages, tools, tool, fn: variant;
  i: PtrInt;
begin
  body := _ObjFast(['model', aRequest.Model, 'stream', aStream]);
  messages := _Arr([]);
  for i := 0 to high(aRequest.Messages) do
    AddMessage(messages, aRequest.Messages[i]);
  _Safe(body)^.AddValue('messages', messages);
  if aRequest.Temperature >= 0 then
    _Safe(body)^.AddValue('temperature', aRequest.Temperature);
  if aRequest.MaxTokens > 0 then
    _Safe(body)^.AddValue('max_tokens', aRequest.MaxTokens);
  if length(aRequest.Tools) > 0 then
  begin
    tools := _Arr([]);
    for i := 0 to high(aRequest.Tools) do
    begin
      fn := _ObjFast([
        'name', aRequest.Tools[i].Name,
        'description', aRequest.Tools[i].Description]);
      if aRequest.Tools[i].ParametersJson <> '' then
        _Safe(fn)^.AddValue('parameters', _Json(aRequest.Tools[i].ParametersJson));
      tool := _ObjFast(['type', 'function', 'function', fn]);
      _Safe(tools)^.AddItem(tool);
    end;
    _Safe(body)^.AddValue('tools', tools);
  end;
  if aRequest.ResponseFormat <> '' then
    _Safe(body)^.AddValue('response_format', _Json(aRequest.ResponseFormat));
  // merge any provider-specific passthrough fields; AddOrUpdateFrom overwrites
  // rather than blindly appending, so Extra cannot create a duplicate JSON key
  if _Safe(aRequest.Extra)^.Count > 0 then
    _Safe(body)^.AddOrUpdateFrom(aRequest.Extra);
  result := _Safe(body)^.ToJson;
end;


{ ************ Response parsing }

function ParseOpenAIChatResponse(const aJson: RawUtf8): TLlmChatResponse;
var
  v: variant;
  d, choice, msg, calls, call, fn, usage: PDocVariantData;
  i: PtrInt;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  v := _Json(aJson);
  d := _Safe(v);
  result.Raw := v;
  result.Model := d^.U['model'];
  choice := d^.A['choices']^._[0];
  result.FinishReason := ToLlmFinishReason(choice^.U['finish_reason']);
  msg := choice^.O['message'];
  result.Content := msg^.U['content'];
  calls := msg^.A['tool_calls'];
  SetLength(result.ToolCalls, calls^.Count);
  for i := 0 to calls^.Count - 1 do
  begin
    call := calls^._[i];
    fn := call^.O['function'];
    result.ToolCalls[i].Id := call^.U['id'];
    result.ToolCalls[i].Name := fn^.U['name'];
    result.ToolCalls[i].ArgumentsJson := fn^.U['arguments'];
  end;
  usage := d^.O['usage'];
  if usage^.Count > 0 then
  begin
    result.Usage.PromptTokens := usage^.I['prompt_tokens'];
    result.Usage.CompletionTokens := usage^.I['completion_tokens'];
    result.Usage.TotalTokens := usage^.I['total_tokens'];
  end;
end;

function OpenAIEmbeddingsRequestJson(const aModel: RawUtf8;
  const aInput: TRawUtf8DynArray): RawUtf8;
var
  body, inputArr: variant;
  i: PtrInt;
begin
  inputArr := _Arr([]);
  for i := 0 to high(aInput) do
    _Safe(inputArr)^.AddItem(aInput[i]);
  body := _ObjFast(['model', aModel, 'input', inputArr]);
  result := _Safe(body)^.ToJson;
end;

function ParseOpenAIEmbeddings(const aJson: RawUtf8): TLlmEmbeddingDynArray;
var
  v: variant;
  d, data, item, emb: PDocVariantData;
  i, j, idx: PtrInt;
begin
  result := nil;
  v := _Json(aJson);
  d := _Safe(v);
  data := d^.A['data'];
  SetLength(result, data^.Count);
  for i := 0 to data^.Count - 1 do
  begin
    item := data^._[i];
    // the provider may return data[] out of order: the "index" field is the
    // authoritative slot, not the array position (else a batch maps the wrong
    // vector to an input). A MISSING index reads as 0 - identical to a real
    // index 0 - which would collapse every vector onto slot 0, so fall back to
    // the array position when the field is absent or out of range.
    if item^.GetValueIndex('index') >= 0 then
      idx := item^.I['index']
    else
      idx := i;
    if (idx < 0) or (idx >= data^.Count) then
      idx := i;
    emb := item^.A['embedding'];
    SetLength(result[idx], emb^.Count);
    for j := 0 to emb^.Count - 1 do
      result[idx][j] := emb^.Values[j]; // variant number -> single
  end;
end;


{ ************ TLlmClient }

constructor TLlmClient.Create(const aConfig: TLlmProviderConfig);
begin
  inherited Create;
  fConfig := aConfig;
end;

function TLlmClient.Config: TLlmProviderConfig;
begin
  result := fConfig;
end;

function TLlmClient.Connect(const aEndpoint: RawUtf8;
  out aPath: RawUtf8): THttpClientSocket;
var
  url: RawUtf8;
  timeout: cardinal;
begin
  url := fConfig.BaseUrl;
  // tolerate a trailing slash on the configured base URL
  if (url <> '') and (url[length(url)] = '/') then
    SetLength(url, length(url) - 1);
  url := url + aEndpoint;
  if fConfig.TimeoutMs > 0 then
    timeout := fConfig.TimeoutMs
  else
    timeout := 30000;
  result := THttpClientSocket.OpenUri(url, aPath, '', timeout, nil);
  try
    // trim defensively: a stray CR/space in the key (e.g. from a CRLF .env)
    // would inject a bad byte into the Authorization header
    if (fConfig.AuthScheme = lasBearer) and (fConfig.ApiKey <> '') then
      result.AuthBearer := TrimU(fConfig.ApiKey);
  except
    result.Free; // do not leak the open socket if header setup fails
    raise;
  end;
end;

function TLlmClient.ChatComplete(const aRequest: TLlmChatRequest): TLlmChatResponse;
var
  sock: THttpClientSocket;
  path, body: RawUtf8;
  status: integer;
begin
  body := OpenAIChatRequestJson(aRequest, {stream=}false);
  sock := Connect('/chat/completions', path);
  try
    status := sock.Request(path, 'POST', 0,
      'Accept: application/json'#13#10'Accept-Encoding: identity',
      body, 'application/json', false, nil, nil);
    if (status < 200) or (status >= 300) then
      ELlmClient.RaiseUtf8('ChatComplete: HTTP % - %', [status, sock.Content]);
    result := ParseOpenAIChatResponse(sock.Content);
  finally
    sock.Free;
  end;
end;

function TLlmClient.Embeddings(const aModel: RawUtf8;
  const aInput: TRawUtf8DynArray): TLlmEmbeddingDynArray;
var
  sock: THttpClientSocket;
  path, body: RawUtf8;
  status: integer;
begin
  body := OpenAIEmbeddingsRequestJson(aModel, aInput);
  sock := Connect('/embeddings', path);
  try
    status := sock.Request(path, 'POST', 0,
      'Accept: application/json'#13#10'Accept-Encoding: identity',
      body, 'application/json', false, nil, nil);
    if (status < 200) or (status >= 300) then
      ELlmClient.RaiseUtf8('Embeddings: HTTP % - %', [status, sock.Content]);
    result := ParseOpenAIEmbeddings(sock.Content);
  finally
    sock.Free;
  end;
end;

procedure TLlmClient.ChatStream(const aRequest: TLlmChatRequest;
  const aOnDelta: TLlmStreamDeltaEvent);
var
  sock: THttpClientSocket;
  path, body: RawUtf8;
  sse: TLlmSseStream;
  outStream: TStream;
  status: integer;
begin
  body := OpenAIChatRequestJson(aRequest, {stream=}true);
  sock := Connect('/chat/completions', path);
  try
    sse := TLlmSseStream.Create(aOnDelta);
    outStream := sse; // until/unless wrapped, freeing outStream frees sse
    try
      // an optional cap forwards every chunk to sse but raises past the limit;
      // the wrapper owns sse and frees it - so we only ever free outStream
      if fConfig.MaxResponseBytes > 0 then
        outStream := TLimitedStreamWriter.Create(sse, fConfig.MaxResponseBytes);
      // no compression: GetBody streams each chunk into sse.Write live, but it
      // refuses a Content-Encoding'd body
      status := sock.Request(path, 'POST', 0,
        'Accept: text/event-stream'#13#10'Accept-Encoding: identity',
        body, 'application/json', false, nil, outStream);
      sse.Flush; // emit a final event delivered without a closing newline
      if (status < 200) or (status >= 300) then
        // a non-2xx body is a JSON error (not SSE): RawBody keeps it readable
        ELlmClient.RaiseUtf8('ChatStream: HTTP % - %', [status, sse.RawBody]);
    finally
      outStream.Free; // frees sse (directly, or via the owning wrapper)
    end;
  finally
    sock.Free;
  end;
end;

end.
