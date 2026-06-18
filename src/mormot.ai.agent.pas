/// LandrixAI LLM Client - tool-calling agent loop
// - part of the mormot.ai.* extension (LandrixAI)
// - clean-room from the OpenAI tool-calling protocol; no third-party code,
//   target license MPL/GPL/LGPL (mORMot contribution)
unit mormot.ai.agent;

{
  *****************************************************************************

    - ILlmToolbox: the agent's tool source, decoupled from any backend
    - TLlmCallbackToolbox: a simple in-process toolbox (register + handler)
    - TLlmAgent: ask the model, run the tools it requests, feed results back,
      and loop until it answers

    The toolbox seam is deliberate: an adapter over the mormot.ai.mcp registry
    can implement ILlmToolbox so an agent drives the very tools an MCP server
    exposes - the same RTTI-generated schema describes both ends.

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.ai.llm.types,
  mormot.ai.llm;

type
  /// raised when the agent loop exceeds its iteration budget
  ELlmAgent = class(ESynException);

  /// the agent's tool source - implement to plug any toolset into the loop
  ILlmToolbox = interface
    ['{2A7C5E18-9D34-4B6F-8A0E-3F1C2D4B5A69}']
    /// the tool definitions to advertise in each chat request
    function List: TLlmToolDynArray;
    /// run one tool call; the returned text/JSON becomes the role=tool message
    function Execute(const aName, aArgumentsJson: RawUtf8): RawUtf8;
  end;

  /// handler signature for a callback-backed tool (a method, FPC 3.2.x safe)
  TLlmToolHandler = function(const aArgumentsJson: RawUtf8): RawUtf8 of object;

  /// a simple in-process toolbox: pair tool definitions with method handlers
  TLlmCallbackToolbox = class(TInterfacedObject, ILlmToolbox)
  protected
    fTools: TLlmToolDynArray;
    fHandlers: array of TLlmToolHandler;
  public
    /// register one tool definition together with the method that runs it
    procedure Add(const aTool: TLlmTool; const aHandler: TLlmToolHandler);
    function List: TLlmToolDynArray;
    function Execute(const aName, aArgumentsJson: RawUtf8): RawUtf8;
  end;

  /// drives a tool-calling conversation to completion
  TLlmAgent = class
  protected
    fClient: ILlmClient;
    fToolbox: ILlmToolbox;
    fModel: RawUtf8;
    fMaxIterations: integer;
  public
    /// create the agent over a client, a toolbox and the model to use
    constructor Create(const aClient: ILlmClient; const aToolbox: ILlmToolbox;
      const aModel: RawUtf8);
    /// run the loop from aMessages until the model answers without tool calls
    // - on each round the toolbox is advertised; any requested tool calls are
    //   executed and their results fed back as role=tool messages
    // - raises ELlmAgent if MaxIterations is exhausted while still calling tools
    function Run(const aMessages: TLlmMessageDynArray): TLlmChatResponse;
    /// safety bound on tool-call rounds (default 8)
    property MaxIterations: integer read fMaxIterations write fMaxIterations;
  end;


implementation

{ TLlmCallbackToolbox }

procedure TLlmCallbackToolbox.Add(const aTool: TLlmTool;
  const aHandler: TLlmToolHandler);
begin
  SetLength(fTools, length(fTools) + 1);
  fTools[high(fTools)] := aTool;
  SetLength(fHandlers, length(fHandlers) + 1);
  fHandlers[high(fHandlers)] := aHandler;
end;

function TLlmCallbackToolbox.List: TLlmToolDynArray;
begin
  result := fTools;
end;

function TLlmCallbackToolbox.Execute(const aName, aArgumentsJson: RawUtf8): RawUtf8;
var
  i: PtrInt;
begin
  for i := 0 to high(fTools) do
    if fTools[i].Name = aName then
    begin
      if Assigned(fHandlers[i]) then
        result := fHandlers[i](aArgumentsJson)
      else
        result := '';
      exit;
    end;
  // unknown tool: report it back so the model can recover instead of failing
  result := FormatUtf8('{"error":"unknown tool: %"}', [aName]);
end;


{ TLlmAgent }

constructor TLlmAgent.Create(const aClient: ILlmClient;
  const aToolbox: ILlmToolbox; const aModel: RawUtf8);
begin
  inherited Create;
  fClient := aClient;
  fToolbox := aToolbox;
  fModel := aModel;
  fMaxIterations := 8;
end;

function TLlmAgent.Run(const aMessages: TLlmMessageDynArray): TLlmChatResponse;
var
  msgs: TLlmMessageDynArray;
  req: TLlmChatRequest;
  resp: TLlmChatResponse;
  assistantMsg, toolMsg: TLlmMessage;
  iter, i: integer;
begin
  msgs := copy(aMessages); // work on our own growable copy of the history
  for iter := 1 to fMaxIterations do
  begin
    req := LlmChatRequest(fModel, msgs);
    req.Tools := fToolbox.List;
    resp := fClient.ChatComplete(req);
    if length(resp.ToolCalls) = 0 then
      exit(resp); // the model answered without requesting any tool

    // 1) replay the assistant turn that requested the calls (OpenAI requires the
    //    tool_calls message to precede its tool results)
    assistantMsg := LlmMessage(lrAssistant, resp.Content);
    assistantMsg.ToolCalls := resp.ToolCalls;
    SetLength(msgs, length(msgs) + 1);
    msgs[high(msgs)] := assistantMsg;

    // 2) execute each requested call and append its result as role=tool
    for i := 0 to high(resp.ToolCalls) do
    begin
      toolMsg := LlmMessage(lrTool,
        fToolbox.Execute(resp.ToolCalls[i].Name, resp.ToolCalls[i].ArgumentsJson));
      toolMsg.ToolCallId := resp.ToolCalls[i].Id;
      toolMsg.Name := resp.ToolCalls[i].Name;
      SetLength(msgs, length(msgs) + 1);
      msgs[high(msgs)] := toolMsg;
    end;
  end;
  ELlmAgent.RaiseUtf8('Run: exceeded % tool-call iterations', [fMaxIterations]);
end;

end.
