/// LandrixAI LLM Client - native Anthropic Messages API driver
// - part of the mormot.ai.* extension (LandrixAI)
// - the OpenAI Chat Completions wire is the suite's lingua franca (mormot.ai.llm);
//   Anthropic speaks a genuinely different wire (Messages API), so it gets a
//   native adapter instead of an OpenAI-compatible shim
// - clean-room from the public Anthropic Messages API docs; no third-party code,
//   target license MPL/GPL/LGPL (mORMot contribution)
unit mormot.ai.llm.anthropic;

{
  *****************************************************************************

    Wire differences handled here vs. the OpenAI driver:
    - `system` is a top-level field, not a message with role=system
    - `max_tokens` is REQUIRED (a default is supplied when the request omits it)
    - tools carry `input_schema` (not function/parameters); the assistant emits
      `tool_use` content blocks and tool results come back as `tool_result`
      blocks inside a user message
    - auth is `x-api-key` + `anthropic-version` headers (not a Bearer token)
    - streaming SSE is event-typed (message_start / content_block_delta / ...)
      rather than self-contained OpenAI chunks

    TAnthropicClient implements the same ILlmClient contract as TLlmClient, so an
    agent, RAG pipeline or structured-output caller is provider-agnostic. The
    neutral TLlmChatRequest/TLlmChatResponse records are reused unchanged.

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
  mormot.ai.llm.sse,
  mormot.ai.llm;

const
  /// Anthropic API version pinned in the `anthropic-version` header
  ANTHROPIC_VERSION = '2023-06-01';

  /// `max_tokens` is required by the Messages API; used when the request omits it
  ANTHROPIC_DEFAULT_MAX_TOKENS = 4096;


/// configuration for the Anthropic Messages API (api.anthropic.com)
// - AuthScheme stays lasNone: TAnthropicClient sends x-api-key/anthropic-version
//   headers itself rather than a Bearer token
function AnthropicConfig(const aApiKey: RawUtf8;
  const aModel: RawUtf8 = 'claude-opus-4-8'): TLlmProviderConfig;

/// build an Anthropic-wire Messages request from a neutral request
// - system messages are hoisted into the top-level `system` field; tool calls /
//   results are translated into tool_use / tool_result content blocks
function AnthropicChatRequestJson(const aRequest: TLlmChatRequest;
  aStream: boolean): RawUtf8;

/// parse a (non-streamed) Anthropic Messages response into the neutral record
// - concatenates text blocks; maps tool_use blocks to ToolCalls; maps
//   stop_reason to the neutral finish reason
function ParseAnthropicChatResponse(const aJson: RawUtf8): TLlmChatResponse;

/// map an Anthropic stop_reason to the canonical OpenAI finish_reason string
// - so downstream ToLlmFinishReason and string consumers behave identically
function AnthropicStopToOpenAI(const aStop: RawUtf8): RawUtf8;


type
  /// write-only TStream decoding the Anthropic Messages SSE wire
  // - message_start (role/usage), content_block_start (tool_use id/name),
  //   content_block_delta (text_delta / input_json_delta), message_delta
  //   (stop_reason / usage), message_stop (terminal) -> neutral TLlmStreamDelta
  TAnthropicSseStream = class(TLlmSseStreamBase)
  protected
    fStreamError: RawUtf8;
    fStreamPromptTokens: integer; // input_tokens from message_start, for TotalTokens
    procedure ProcessData(const aPayload: RawUtf8); override;
  public
    /// non-empty if the stream carried an Anthropic `error` event - an inband
    // server error (HTTP 200, e.g. overloaded_error) that ChatStream raises as
    // ELlmClient instead of returning a silently truncated answer
    property StreamError: RawUtf8 read fStreamError;
  end;

  /// native Anthropic Messages API client (ILlmClient)
  TAnthropicClient = class(TInterfacedObject, ILlmClient)
  protected
    fConfig: TLlmProviderConfig;
    // open a fresh connection to BaseUrl + '/messages'; returns the request path
    function Connect(out aPath: RawUtf8): THttpClientSocket;
    // the x-api-key + anthropic-version (+ Accept) header block for a request
    function Headers(aStream: boolean): RawUtf8;
  public
    /// create the client for a given provider configuration
    constructor Create(const aConfig: TLlmProviderConfig); reintroduce;
    function ChatComplete(const aRequest: TLlmChatRequest): TLlmChatResponse;
    procedure ChatStream(const aRequest: TLlmChatRequest;
      const aOnDelta: TLlmStreamDeltaEvent);
    function Config: TLlmProviderConfig;
  end;


implementation

{ ************ Configuration }

function AnthropicConfig(const aApiKey, aModel: RawUtf8): TLlmProviderConfig;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.BaseUrl := 'https://api.anthropic.com/v1';
  result.ApiKey := aApiKey;
  result.AuthScheme := lasNone; // headers are sent explicitly, not as Bearer
  result.DefaultModel := aModel;
end;


{ ************ Request building }

function AnthropicStopToOpenAI(const aStop: RawUtf8): RawUtf8;
begin
  // normalise to the OpenAI vocabulary the neutral types already speak
  if (aStop = 'end_turn') or (aStop = 'stop_sequence') then
    result := 'stop'
  else if aStop = 'max_tokens' then
    result := 'length'
  else if aStop = 'tool_use' then
    result := 'tool_calls'
  else if aStop = 'refusal' then
    result := 'content_filter'
  else
    result := ''; // pause_turn / unknown -> not finished
end;

// one Anthropic image content block from a neutral image attachment
function AnthropicImageBlock(const aImage: TLlmImage): variant;
begin
  // Anthropic wraps the image in a typed `source` (base64 vs url), unlike
  // OpenAI's single image_url field
  if aImage.Source = lisBase64 then
    result := _ObjFast([
      'type', 'image',
      'source', _ObjFast([
        'type', 'base64',
        'media_type', LlmImageMediaType(aImage),
        'data', aImage.Data])])
  else
    result := _ObjFast([
      'type', 'image',
      'source', _ObjFast(['type', 'url', 'url', aImage.Data])]);
end;

// one Anthropic tool_result content block from a neutral lrTool message
function AnthropicToolResultBlock(const aMsg: TLlmMessage): variant;
begin
  result := _ObjFast([
    'type', 'tool_result',
    'tool_use_id', aMsg.ToolCallId,
    'content', aMsg.Content]);
end;

// translate one neutral message; system text is hoisted into aSystem, every
// other message is appended to aMessages as an Anthropic message object
// - lrTool is NOT handled here: the caller bundles consecutive tool results
//   into a single user message (see AnthropicChatRequestJson)
procedure AddAnthropicMessage(const aMessages: variant; var aSystem: RawUtf8;
  const aMsg: TLlmMessage);
var
  content, input: variant;
  i: PtrInt;
begin
  case aMsg.Role of
    lrSystem:
      // multiple system turns are concatenated; Anthropic has one `system` field
      if aSystem = '' then
        aSystem := aMsg.Content
      else
        aSystem := aSystem + #10 + aMsg.Content;
    lrUser:
      if length(aMsg.Images) > 0 then
      begin
        // multimodal user turn: text part (if any) + one image block per image
        content := _Arr([]);
        if aMsg.Content <> '' then
          _Safe(content)^.AddItem(
            _ObjFast(['type', 'text', 'text', aMsg.Content]));
        for i := 0 to high(aMsg.Images) do
          _Safe(content)^.AddItem(AnthropicImageBlock(aMsg.Images[i]));
        _Safe(aMessages)^.AddItem(
          _ObjFast(['role', 'user', 'content', content]));
      end
      else
        _Safe(aMessages)^.AddItem(
          _ObjFast(['role', 'user', 'content', aMsg.Content]));
    lrAssistant:
      if length(aMsg.ToolCalls) > 0 then
      begin
        // assistant turn requesting tools: text (if any) + one tool_use per call
        content := _Arr([]);
        if aMsg.Content <> '' then
          _Safe(content)^.AddItem(
            _ObjFast(['type', 'text', 'text', aMsg.Content]));
        for i := 0 to high(aMsg.ToolCalls) do
        begin
          // the model's arguments are a raw JSON object; embed as a real object
          if (aMsg.ToolCalls[i].ArgumentsJson <> '') and
             IsValidJson(aMsg.ToolCalls[i].ArgumentsJson) then
            input := _Json(aMsg.ToolCalls[i].ArgumentsJson)
          else
            input := _Obj([]);
          _Safe(content)^.AddItem(_ObjFast([
            'type', 'tool_use',
            'id', aMsg.ToolCalls[i].Id,
            'name', aMsg.ToolCalls[i].Name,
            'input', input]));
        end;
        _Safe(aMessages)^.AddItem(
          _ObjFast(['role', 'assistant', 'content', content]));
      end
      else
        _Safe(aMessages)^.AddItem(
          _ObjFast(['role', 'assistant', 'content', aMsg.Content]));
  end;
end;

function AnthropicChatRequestJson(const aRequest: TLlmChatRequest;
  aStream: boolean): RawUtf8;
var
  body, messages, tools, tool, content: variant;
  system: RawUtf8;
  maxTokens, i: integer;
begin
  system := '';
  messages := _Arr([]);
  i := 0;
  while i <= high(aRequest.Messages) do
    if aRequest.Messages[i].Role = lrTool then
    begin
      // the agent appends one lrTool message per parallel tool call; Anthropic
      // requires ALL tool_result blocks of one turn in a SINGLE user message
      // (separate user messages are rejected) - so coalesce the consecutive run
      content := _Arr([]);
      repeat
        _Safe(content)^.AddItem(AnthropicToolResultBlock(aRequest.Messages[i]));
        inc(i);
      until (i > high(aRequest.Messages)) or
            (aRequest.Messages[i].Role <> lrTool);
      _Safe(messages)^.AddItem(_ObjFast(['role', 'user', 'content', content]));
    end
    else
    begin
      AddAnthropicMessage(messages, system, aRequest.Messages[i]);
      inc(i);
    end;
  // max_tokens is mandatory on the Messages API
  if aRequest.MaxTokens > 0 then
    maxTokens := aRequest.MaxTokens
  else
    maxTokens := ANTHROPIC_DEFAULT_MAX_TOKENS;
  body := _ObjFast(['model', aRequest.Model, 'max_tokens', maxTokens]);
  if aStream then
    _Safe(body)^.AddValue('stream', true);
  if system <> '' then
    _Safe(body)^.AddValue('system', system);
  _Safe(body)^.AddValue('messages', messages);
  if aRequest.Temperature >= 0 then
    _Safe(body)^.AddValue('temperature', aRequest.Temperature);
  if length(aRequest.Tools) > 0 then
  begin
    tools := _Arr([]);
    for i := 0 to high(aRequest.Tools) do
    begin
      tool := _ObjFast([
        'name', aRequest.Tools[i].Name,
        'description', aRequest.Tools[i].Description]);
      // Anthropic names the parameter schema `input_schema`; default to an empty
      // object schema so a tool without parameters still validates
      if aRequest.Tools[i].ParametersJson <> '' then
        _Safe(tool)^.AddValue('input_schema', _Json(aRequest.Tools[i].ParametersJson))
      else
        _Safe(tool)^.AddValue('input_schema',
          _ObjFast(['type', 'object', 'properties', _Obj([])]));
      _Safe(tools)^.AddItem(tool);
    end;
    _Safe(body)^.AddValue('tools', tools);
  end;
  // merge any provider-specific passthrough (e.g. thinking, tool_choice);
  // AddOrUpdateFrom overwrites rather than duplicating a key
  if _Safe(aRequest.Extra)^.Count > 0 then
    _Safe(body)^.AddOrUpdateFrom(aRequest.Extra);
  result := _Safe(body)^.ToJson;
end;


{ ************ Response parsing }

function ParseAnthropicChatResponse(const aJson: RawUtf8): TLlmChatResponse;
var
  v: variant;
  d, content, block, usage: PDocVariantData;
  text, bt: RawUtf8;
  i, n: PtrInt;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  v := _Json(aJson);
  d := _Safe(v);
  result.Raw := v;
  result.Model := d^.U['model'];
  content := d^.A['content'];
  // upper-bound the tool-call array to the block count, then trim to the actual
  SetLength(result.ToolCalls, content^.Count);
  text := '';
  n := 0;
  for i := 0 to content^.Count - 1 do
  begin
    block := content^._[i];
    bt := block^.U['type'];
    if bt = 'text' then
      text := text + block^.U['text']
    else if bt = 'tool_use' then
    begin
      result.ToolCalls[n].Id := block^.U['id'];
      result.ToolCalls[n].Name := block^.U['name'];
      // re-serialize the input object into the raw JSON the neutral record holds;
      // a missing `input` yields mORMot's void doc whose ToJson is 'null', so
      // fall back to an empty object rather than the literal null
      if block^.O['input']^.IsObject then
        result.ToolCalls[n].ArgumentsJson := block^.O['input']^.ToJson
      else
        result.ToolCalls[n].ArgumentsJson := '{}';
      inc(n);
    end;
    // thinking / other block types are ignored for the projected text
  end;
  SetLength(result.ToolCalls, n);
  result.Content := text;
  result.FinishReason := ToLlmFinishReason(AnthropicStopToOpenAI(d^.U['stop_reason']));
  usage := d^.O['usage'];
  if usage^.Count > 0 then
  begin
    result.Usage.PromptTokens := usage^.I['input_tokens'];
    result.Usage.CompletionTokens := usage^.I['output_tokens'];
    result.Usage.TotalTokens :=
      result.Usage.PromptTokens + result.Usage.CompletionTokens;
  end;
end;


{ ************ TAnthropicSseStream }

procedure TAnthropicSseStream.ProcessData(const aPayload: RawUtf8);
var
  delta: TLlmStreamDelta;
  v: variant;
  d, msg, cb, dd, usage: PDocVariantData;
  etype, dtype: RawUtf8;
  emit: boolean;
begin
  Finalize(delta);
  FillCharFast(delta, SizeOf(delta), 0);
  v := _JsonFast(aPayload);
  d := _Safe(v);
  if d^.Count = 0 then
    exit;
  delta.Raw := v;
  etype := d^.U['type'];
  emit := true;
  if etype = 'message_start' then
  begin
    msg := d^.O['message'];
    delta.Role := msg^.U['role'];
    usage := msg^.O['usage'];
    if usage^.Count > 0 then
    begin
      // remember the prompt tokens: message_delta later carries only output_tokens
      fStreamPromptTokens := usage^.I['input_tokens'];
      delta.HasUsage := true;
      delta.Usage.PromptTokens := fStreamPromptTokens;
      delta.Usage.CompletionTokens := usage^.I['output_tokens'];
      delta.Usage.TotalTokens :=
        delta.Usage.PromptTokens + delta.Usage.CompletionTokens;
    end;
  end
  else if etype = 'content_block_start' then
  begin
    cb := d^.O['content_block'];
    if cb^.U['type'] = 'tool_use' then
    begin
      // start of a tool-call slot: id + name arrive here, args stream as deltas
      delta.HasToolCall := true;
      delta.ToolCallIndex := d^.I['index'];
      delta.ToolCallId := cb^.U['id'];
      delta.ToolCallName := cb^.U['name'];
    end
    else
      emit := false; // text block start carries nothing incremental
  end
  else if etype = 'content_block_delta' then
  begin
    dd := d^.O['delta'];
    dtype := dd^.U['type'];
    if dtype = 'text_delta' then
      delta.ContentDelta := dd^.U['text']
    else if dtype = 'input_json_delta' then
    begin
      delta.HasToolCall := true;
      delta.ToolCallIndex := d^.I['index'];
      delta.ToolCallArgsDelta := dd^.U['partial_json'];
    end
    else
      emit := false; // thinking_delta etc. not projected
  end
  else if etype = 'message_delta' then
  begin
    dd := d^.O['delta'];
    delta.FinishReason := AnthropicStopToOpenAI(dd^.U['stop_reason']);
    usage := d^.O['usage'];
    if usage^.Count > 0 then
    begin
      // message_delta carries only output_tokens; pair it with the input_tokens
      // from message_start so TotalTokens is complete for the consumer
      delta.HasUsage := true;
      delta.Usage.PromptTokens := fStreamPromptTokens;
      delta.Usage.CompletionTokens := usage^.I['output_tokens'];
      delta.Usage.TotalTokens :=
        fStreamPromptTokens + delta.Usage.CompletionTokens;
    end;
  end
  else if etype = 'message_stop' then
  begin
    fDone := true;
    delta.Done := true;
  end
  else if etype = 'error' then
  begin
    // Anthropic streams a server-side error inband (HTTP 200, e.g.
    // overloaded_error) - capture it and end the stream so ChatStream can raise
    // rather than hand back a silently truncated answer
    fStreamError := d^.O['error']^.U['message'];
    if fStreamError = '' then
      fStreamError := aPayload;
    fDone := true;
    delta.Done := true;
  end
  else
    emit := false; // ping / content_block_stop: nothing to project

  if not emit then
    exit;
  fText := fText + delta.ContentDelta;
  if delta.FinishReason <> '' then
    fFinishReason := delta.FinishReason;
  if Assigned(fOnDelta) then
    fOnDelta(delta);
end;


{ ************ TAnthropicClient }

constructor TAnthropicClient.Create(const aConfig: TLlmProviderConfig);
begin
  inherited Create;
  fConfig := aConfig;
end;

function TAnthropicClient.Config: TLlmProviderConfig;
begin
  result := fConfig;
end;

function TAnthropicClient.Connect(out aPath: RawUtf8): THttpClientSocket;
var
  url: RawUtf8;
  timeout: cardinal;
begin
  url := fConfig.BaseUrl;
  if (url <> '') and (url[length(url)] = '/') then
    SetLength(url, length(url) - 1);
  url := url + '/messages';
  if fConfig.TimeoutMs > 0 then
    timeout := fConfig.TimeoutMs
  else
    timeout := 30000;
  result := THttpClientSocket.OpenUri(url, aPath, '', timeout, nil);
end;

function TAnthropicClient.Headers(aStream: boolean): RawUtf8;
var
  accept: RawUtf8;
begin
  if aStream then
    accept := 'Accept: text/event-stream'
  else
    accept := 'Accept: application/json';
  // trim the key defensively: a stray CR (e.g. from a CRLF .env) would inject a
  // bad byte into the x-api-key header
  result := accept + #13#10 +
    'Accept-Encoding: identity'#13#10 +
    'anthropic-version: ' + ANTHROPIC_VERSION + #13#10 +
    'x-api-key: ' + TrimU(fConfig.ApiKey);
end;

function TAnthropicClient.ChatComplete(
  const aRequest: TLlmChatRequest): TLlmChatResponse;
var
  sock: THttpClientSocket;
  path, body: RawUtf8;
  status: integer;
begin
  body := AnthropicChatRequestJson(aRequest, {stream=}false);
  sock := Connect(path);
  try
    status := sock.Request(path, 'POST', 0, Headers({stream=}false),
      body, 'application/json', false, nil, nil);
    if (status < 200) or (status >= 300) then
      ELlmClient.RaiseUtf8('ChatComplete: HTTP % - %', [status, sock.Content]);
    result := ParseAnthropicChatResponse(sock.Content);
  finally
    sock.Free;
  end;
end;

procedure TAnthropicClient.ChatStream(const aRequest: TLlmChatRequest;
  const aOnDelta: TLlmStreamDeltaEvent);
var
  sock: THttpClientSocket;
  path, body: RawUtf8;
  sse: TAnthropicSseStream;
  outStream: TStream;
  status: integer;
begin
  body := AnthropicChatRequestJson(aRequest, {stream=}true);
  sock := Connect(path);
  try
    sse := TAnthropicSseStream.Create(aOnDelta);
    outStream := sse; // until/unless wrapped, freeing outStream frees sse
    try
      // optional cap: forwards every chunk to sse but raises past the limit; the
      // wrapper owns sse and frees it - so we only ever free outStream
      if fConfig.MaxResponseBytes > 0 then
        outStream := TLimitedStreamWriter.Create(sse, fConfig.MaxResponseBytes);
      status := sock.Request(path, 'POST', 0, Headers({stream=}true),
        body, 'application/json', false, nil, outStream);
      sse.Flush; // emit a final event delivered without a closing newline
      if (status < 200) or (status >= 300) then
        // a non-2xx body is a JSON error (not SSE): RawBody keeps it readable
        ELlmClient.RaiseUtf8('ChatStream: HTTP % - %', [status, sse.RawBody]);
      // an inband `error` event arrives with HTTP 200, so it is invisible to the
      // status check above - surface it rather than return a truncated answer
      if sse.StreamError <> '' then
        ELlmClient.RaiseUtf8('ChatStream: stream error - %', [sse.StreamError]);
    finally
      outStream.Free; // frees sse (directly, or via the owning wrapper)
    end;
  finally
    sock.Free;
  end;
end;

end.
