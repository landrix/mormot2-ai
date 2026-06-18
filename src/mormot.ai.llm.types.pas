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


implementation

uses
  mormot.core.text;

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

end.
