// - regression tests for mormot.ai.agent.mcp (MCP toolbox bridge)
unit test.llm.agent.mcp;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.core.rtti,
  mormot.core.variants,
  mormot.core.test,
  mormot.ai.mcp,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.agent,
  mormot.ai.agent.mcp,
  test.llm.agent; // reuse the scripted TStubLlmClient

type
  TEchoParams = packed record
    text: RawUtf8;
  end;

  TEchoTool = class(TMcpToolBase<TEchoParams>)
  protected
    function ExecuteTyped(const aParams: TEchoParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  // a tool that reports an MCP tool-level failure (result.isError per the spec)
  TFailTool = class(TMcpToolBase<TEchoParams>)
  protected
    function ExecuteTyped(const aParams: TEchoParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  TTestLlmAgentMcp = class(TSynTestCase)
  protected
    procedure EnsureRtti;
    function NewEchoServer: TMcpServer;
  published
    procedure BridgeListAndExecute;
    procedure AgentDrivesMcpTool;
    procedure ToolErrorsAndInvalidArgs;
  end;


implementation

{ TEchoTool }

function TEchoTool.ExecuteTyped(const aParams: TEchoParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
begin
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText('echo: ' + aParams.text);
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

{ TFailTool }

function TFailTool.ExecuteTyped(const aParams: TEchoParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  // mirror the MCP spec's tool-error shape: a content array plus isError=true
  result := _ObjFast([
    'content', _Arr([_ObjFast(['type', 'text', 'text', 'boom'])]),
    'isError', true]);
end;


{ helpers }

function McpToolCall(const aName, aArgs: RawUtf8): TLlmChatResponse;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.FinishReason := lfrToolCalls;
  SetLength(result.ToolCalls, 1);
  result.ToolCalls[0].Id := 'call_1';
  result.ToolCalls[0].Name := aName;
  result.ToolCalls[0].ArgumentsJson := aArgs;
end;

function McpFinal(const aContent: RawUtf8): TLlmChatResponse;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.FinishReason := lfrStop;
  result.Content := aContent;
end;


{ TTestLlmAgentMcp }

procedure TTestLlmAgentMcp.EnsureRtti;
begin
  if not RecordHasFields(TypeInfo(TEchoParams)) then
    Rtti.RegisterFromText(TypeInfo(TEchoParams), 'text:RawUtf8');
end;

function TTestLlmAgentMcp.NewEchoServer: TMcpServer;
begin
  EnsureRtti;
  result := TMcpServer.Create('test-mcp', '1.0.0');
  result.RegisterTool(TEchoTool.Create('echo', 'Echo the input text'));
  result.Start;
end;

procedure TTestLlmAgentMcp.BridgeListAndExecute;
var
  server: TMcpServer;
  box: ILlmToolbox;
  tools: TLlmToolDynArray;
  res: RawUtf8;
begin
  server := NewEchoServer;
  try
    box := TLlmMcpToolbox.Create(server);

    tools := box.List;
    CheckEqual(length(tools), 1, 'one tool listed');
    CheckEqual(tools[0].Name, 'echo', 'tool name');
    CheckEqual(tools[0].Description, 'Echo the input text', 'tool description');
    Check(Pos(RawUtf8('"text"'), tools[0].ParametersJson) > 0, 'schema has text param');
    Check(Pos(RawUtf8('object'), tools[0].ParametersJson) > 0, 'schema is an object');

    res := box.Execute('echo', '{"text":"hi"}');
    CheckEqual(res, 'echo: hi', 'tool executed via the bridge');

    // an unknown tool must surface as text, not raise
    res := box.Execute('nope', '{}');
    Check(Pos(RawUtf8('error'), res) > 0, 'unknown tool returns an error text');
  finally
    box := nil; // release the bridge before the (unowned) server
    server.Free;
  end;
end;

procedure TTestLlmAgentMcp.AgentDrivesMcpTool;
var
  server: TMcpServer;
  stub: TStubLlmClient;
  client: ILlmClient;
  box: ILlmToolbox;
  agent: TLlmAgent;
  msgs: TLlmMessageDynArray;
  resp: TLlmChatResponse;
  last: TLlmMessage;
begin
  server := NewEchoServer;
  stub := TStubLlmClient.Create;
  client := stub;
  // round 1: the model asks to call the MCP 'echo' tool; round 2: it answers
  stub.Push(McpToolCall('echo', '{"text":"world"}'));
  stub.Push(McpFinal('I said world.'));
  try
    box := TLlmMcpToolbox.Create(server);
    agent := TLlmAgent.Create(client, box, 'test-model');
    try
      SetLength(msgs, 1);
      msgs[0] := LlmMessage(lrUser, 'say world');
      resp := agent.Run(msgs);

      CheckEqual(resp.Content, 'I said world.', 'final answer');
      CheckEqual(stub.Calls, 2, 'two model round-trips');
      // the tool result fed back must be the real MCP tool output
      last := stub.LastRequest.Messages[high(stub.LastRequest.Messages)];
      Check(last.Role = lrTool, 'last history message is the tool result');
      CheckEqual(last.Content, 'echo: world', 'MCP tool output fed back to the model');
    finally
      agent.Free;
      box := nil;
    end;
  finally
    server.Free;
  end;
end;

procedure TTestLlmAgentMcp.ToolErrorsAndInvalidArgs;
var
  server: TMcpServer;
  box: ILlmToolbox;
  res: RawUtf8;
begin
  EnsureRtti;
  server := TMcpServer.Create('test-mcp', '1.0.0');
  try
    server.RegisterTool(TEchoTool.Create('echo', 'Echo the input text'));
    server.RegisterTool(TFailTool.Create('fail', 'Always fails'));
    server.Start;
    box := TLlmMcpToolbox.Create(server);

    // malformed argument JSON is rejected before the tool runs
    res := box.Execute('echo', '{bad json');
    Check(Pos(RawUtf8('invalid tool arguments'), res) > 0, 'malformed args rejected');

    // an MCP tool-level isError must be flagged, not handed back as a normal result
    res := box.Execute('fail', '{"text":"x"}');
    Check(Pos(RawUtf8('"isError":true'), res) > 0, 'tool error flagged');
    Check(Pos(RawUtf8('boom'), res) > 0, 'tool error content preserved');

    box := nil;
  finally
    server.Free;
  end;
end;

end.
