/// LandrixAI LLM Client - provider-neutral domain types
// - part of the mormot.ai.* extension (LandrixAI)
// - clean-room implementation from public provider API docs (OpenAI/Ollama);
//   no third-party code, target license MPL/GPL/LGPL (mORMot contribution)
// - the OpenAI Chat Completions schema is the canonical wire format: LiteLLM and
//   Ollama expose an OpenAI-compatible endpoint, so one mapping covers all three
unit mormot.ai.llm.types;

{
  *****************************************************************************

    - Roles, finish reasons, usage accounting
    - Messages, tool definitions and tool calls
    - Chat request / response and the streaming delta record

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.core.variants;


{ ************ Core Enumerations }

type
  /// the author of a chat message (OpenAI roles)
  TLlmRole = (
    lrSystem,
    lrUser,
    lrAssistant,
    lrTool);

  /// why the model stopped generating (OpenAI finish_reason)
  // - lfrNone = not finished yet (e.g. mid-stream) or unknown
  TLlmFinishReason = (
    lfrNone,
    lfrStop,
    lfrLength,
    lfrToolCalls,
    lfrContentFilter);


{ ************ Token Usage }

  /// token accounting reported by the provider
  TLlmUsage = record
    PromptTokens: integer;
    CompletionTokens: integer;
    TotalTokens: integer;
  end;

  /// a single embedding vector (float32, as returned by the provider/model)
  TLlmEmbedding = TSingleDynArray;
  /// one embedding vector per input text
  TLlmEmbeddingDynArray = array of TLlmEmbedding;


{ ************ Tools and Messages }

  /// a single function/tool call requested by the assistant
  TLlmToolCall = record
    /// provider-assigned id, echoed back when answering with role=tool
    Id: RawUtf8;
    /// the function name to invoke
    Name: RawUtf8;
    /// the call arguments, as a raw JSON object string
    ArgumentsJson: RawUtf8;
  end;
  TLlmToolCallDynArray = array of TLlmToolCall;

  /// how an image attachment is supplied to the model
  TLlmImageSource = (
    lisBase64,
    lisUrl);

  /// one image attachment for a multimodal (vision) message
  TLlmImage = record
    /// base64 inline data, or a URL the provider fetches
    Source: TLlmImageSource;
    /// MIME type for lisBase64 (e.g. 'image/png'); ignored for lisUrl
    MediaType: RawUtf8;
    /// the base64 payload (no 'data:' prefix) for lisBase64, or the URL for lisUrl
    Data: RawUtf8;
  end;
  TLlmImageDynArray = array of TLlmImage;

  /// one entry of the chat conversation
  TLlmMessage = record
    Role: TLlmRole;
    /// the textual content (may be '' for an assistant message that only calls tools)
    Content: RawUtf8;
    /// optional participant/tool name
    Name: RawUtf8;
    /// for Role=lrTool: which assistant tool call this message answers
    ToolCallId: RawUtf8;
    /// for Role=lrAssistant: the tool calls the model requested
    ToolCalls: TLlmToolCallDynArray;
    /// optional image attachments (vision/multimodal)
    // - when present the wire serializes Content + these as a content-parts array
    //   (OpenAI image_url / Anthropic image source); usually on a user message
    Images: TLlmImageDynArray;
  end;
  TLlmMessageDynArray = array of TLlmMessage;

  /// a tool/function the model is allowed to call
  TLlmTool = record
    Name: RawUtf8;
    Description: RawUtf8;
    /// the parameter JSON-Schema object, as a raw JSON string
    // - matches what mormot.ai.mcp generates per RTTI from a tool record, so an
    //   agent can expose MCP tools to the model without re-describing them
    ParametersJson: RawUtf8;
  end;
  TLlmToolDynArray = array of TLlmTool;


{ ************ Chat Request and Response }

  /// a chat completion request (OpenAI-wire neutral)
  TLlmChatRequest = record
    Model: RawUtf8;
    Messages: TLlmMessageDynArray;
    Tools: TLlmToolDynArray;
    /// sampling temperature; a negative value means "omit / provider default"
    Temperature: double;
    /// response token cap; <= 0 means "omit / provider default"
    MaxTokens: integer;
    /// request a streamed (SSE) response
    Stream: boolean;
    /// optional response_format value as raw JSON (e.g. '{"type":"json_object"}'
    // or a json_schema object); '' omits it - see mormot.ai.llm.structured
    ResponseFormat: RawUtf8;
    /// optional provider-specific fields merged verbatim into the request body
    Extra: variant;
  end;

  /// a complete (non-streamed) chat completion response
  TLlmChatResponse = record
    Content: RawUtf8;
    ToolCalls: TLlmToolCallDynArray;
    FinishReason: TLlmFinishReason;
    Usage: TLlmUsage;
    Model: RawUtf8;
    /// the full provider payload, for fields not projected above
    Raw: variant;
  end;

  /// one incremental chunk of a streamed chat completion
  // - emitted once per SSE "data:" event by TLlmSseStream
  TLlmStreamDelta = record
    /// the text fragment added by this chunk (may be '')
    ContentDelta: RawUtf8;
    /// the role, usually only present on the first chunk ('assistant')
    Role: RawUtf8;
    /// non-empty on the final content chunk ('stop'/'length'/'tool_calls'/...)
    FinishReason: RawUtf8;
    /// set when this chunk carries a streamed tool-call fragment
    HasToolCall: boolean;
    /// the tool-call slot this fragment belongs to (multiple calls interleave)
    ToolCallIndex: integer;
    /// the tool-call id (present on the first fragment of a slot)
    ToolCallId: RawUtf8;
    /// the function name (present on the first fragment of a slot)
    ToolCallName: RawUtf8;
    /// the incremental fragment of the JSON arguments string
    ToolCallArgsDelta: RawUtf8;
    /// set when this chunk carries usage accounting (final chunk)
    HasUsage: boolean;
    Usage: TLlmUsage;
    /// set for the terminal SSE "[DONE]" sentinel (no payload)
    Done: boolean;
    /// the full chunk payload, for fields not projected above
    Raw: variant;
  end;


{ ************ Helpers }

/// map an OpenAI role string to TLlmRole (defaults to lrUser)
function ToLlmRole(const aText: RawUtf8): TLlmRole;

/// the OpenAI role string for a TLlmRole
function LlmRoleText(aRole: TLlmRole): RawUtf8;

/// map an OpenAI finish_reason string to TLlmFinishReason
function ToLlmFinishReason(const aText: RawUtf8): TLlmFinishReason;

/// initialize a chat request with sensible "omit" defaults
// - Temperature = -1 (omit), MaxTokens = 0 (omit), Stream = false
function LlmChatRequest(const aModel: RawUtf8;
  const aMessages: TLlmMessageDynArray): TLlmChatRequest;

/// build a single chat message
function LlmMessage(aRole: TLlmRole; const aContent: RawUtf8): TLlmMessage;

/// build a base64 image attachment (aMediaType e.g. 'image/png')
function LlmImageBase64(const aMediaType, aBase64: RawUtf8): TLlmImage;

/// build a URL image attachment (the provider fetches it)
function LlmImageUrl(const aUrl: RawUtf8): TLlmImage;

/// build a multimodal message: text content + image attachments
function LlmImageMessage(aRole: TLlmRole; const aContent: RawUtf8;
  const aImages: TLlmImageDynArray): TLlmMessage;

/// the OpenAI image_url value: a 'data:' URI for base64, or the URL as-is
function LlmImageDataUri(const aImage: TLlmImage): RawUtf8;

/// the effective MIME type of a base64 image: its MediaType, or a sensible
// default when left empty - so a wire that requires it (Anthropic) stays valid
function LlmImageMediaType(const aImage: TLlmImage): RawUtf8;


implementation

uses
  mormot.core.text;

const
  /// fallback MIME for a base64 image whose MediaType was left empty
  LLM_DEFAULT_IMAGE_MEDIA = 'image/png';

function ToLlmRole(const aText: RawUtf8): TLlmRole;
begin
  // the OpenAI wire role is always lower-case, so an exact match suffices
  if aText = 'system' then
    result := lrSystem
  else if aText = 'assistant' then
    result := lrAssistant
  else if aText = 'tool' then
    result := lrTool
  else
    result := lrUser;
end;

function LlmRoleText(aRole: TLlmRole): RawUtf8;
begin
  case aRole of
    lrSystem:    result := 'system';
    lrAssistant: result := 'assistant';
    lrTool:      result := 'tool';
  else
    result := 'user';
  end;
end;

function ToLlmFinishReason(const aText: RawUtf8): TLlmFinishReason;
begin
  // the OpenAI wire finish_reason is always lower-case
  if aText = 'stop' then
    result := lfrStop
  else if aText = 'length' then
    result := lfrLength
  else if aText = 'tool_calls' then
    result := lfrToolCalls
  else if aText = 'content_filter' then
    result := lfrContentFilter
  else
    result := lfrNone;
end;

function LlmChatRequest(const aModel: RawUtf8;
  const aMessages: TLlmMessageDynArray): TLlmChatRequest;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.Model := aModel;
  result.Messages := aMessages;
  result.Temperature := -1; // omit
  result.MaxTokens := 0;    // omit
  result.Stream := false;
end;

function LlmMessage(aRole: TLlmRole; const aContent: RawUtf8): TLlmMessage;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.Role := aRole;
  result.Content := aContent;
end;

function LlmImageBase64(const aMediaType, aBase64: RawUtf8): TLlmImage;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.Source := lisBase64;
  result.MediaType := aMediaType;
  result.Data := aBase64;
end;

function LlmImageUrl(const aUrl: RawUtf8): TLlmImage;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.Source := lisUrl;
  result.Data := aUrl;
end;

function LlmImageMessage(aRole: TLlmRole; const aContent: RawUtf8;
  const aImages: TLlmImageDynArray): TLlmMessage;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.Role := aRole;
  result.Content := aContent;
  result.Images := aImages;
end;

function LlmImageMediaType(const aImage: TLlmImage): RawUtf8;
begin
  if aImage.MediaType <> '' then
    result := aImage.MediaType
  else
    result := LLM_DEFAULT_IMAGE_MEDIA;
end;

function LlmImageDataUri(const aImage: TLlmImage): RawUtf8;
begin
  // OpenAI image_url takes either a public URL or an inline data: URI
  if aImage.Source = lisBase64 then
    result := FormatUtf8('data:%;base64,%', [LlmImageMediaType(aImage), aImage.Data])
  else
    result := aImage.Data;
end;

end.
