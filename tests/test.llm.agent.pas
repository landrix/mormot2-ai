// - regression tests for mormot.ai.agent (tool-calling loop)
unit test.llm.agent;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.core.test,
  mormot.ai.llm.types,
  mormot.ai.llm.sse,
  mormot.ai.llm,
  mormot.ai.agent;

type
  /// a scripted ILlmClient: ChatComplete returns the next pushed response and
  /// records the request it was given (so the loop's history can be asserted)
  TStubLlmClient = class(TInterfacedObject, ILlmClient)
  protected
    fResponses: array of TLlmChatResponse;
    fCalls: integer;
    fLastRequest: TLlmChatRequest;
  public
    procedure Push(const aResponse: TLlmChatResponse);
    function ChatComplete(const aRequest: TLlmChatRequest): TLlmChatResponse;
    procedure ChatStream(const aRequest: TLlmChatRequest;
      const aOnDelta: TLlmStreamDeltaEvent);
    function Config: TLlmProviderConfig;
    property Calls: integer read fCalls;
    property LastRequest: TLlmChatRequest read fLastRequest;
  end;

  /// a tool that records the arguments it was called with
  TWeatherTool = class
  public
    CalledWith: RawUtf8;
    function Handle(const aArgumentsJson: RawUtf8): RawUtf8;
  end;

  TTestLlmAgent = class(TSynTestCase)
  published
    procedure ToolCallingLoop;
    procedure MaxIterationsGuard;
  end;


implementation

{ TStubLlmClient }

procedure TStubLlmClient.Push(const aResponse: TLlmChatResponse);
begin
  SetLength(fResponses, length(fResponses) + 1);
  fResponses[high(fResponses)] := aResponse;
end;

function TStubLlmClient.ChatComplete(const aRequest: TLlmChatRequest): TLlmChatResponse;
begin
  fLastRequest := aRequest;
  if fCalls <= high(fResponses) then
    result := fResponses[fCalls]
  else
    result := fResponses[high(fResponses)]; // keep returning the last script
  inc(fCalls);
end;

procedure TStubLlmClient.ChatStream(const aRequest: TLlmChatRequest;
  const aOnDelta: TLlmStreamDeltaEvent);
begin
  // not exercised by these tests
end;

function TStubLlmClient.Config: TLlmProviderConfig;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
end;

{ TWeatherTool }

function TWeatherTool.Handle(const aArgumentsJson: RawUtf8): RawUtf8;
begin
  CalledWith := aArgumentsJson;
  result := '{"temp":"22C","sky":"sunny"}';
end;


{ helpers }

function ToolCallResponse(const aId, aName, aArgs: RawUtf8): TLlmChatResponse;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.FinishReason := lfrToolCalls;
  SetLength(result.ToolCalls, 1);
  result.ToolCalls[0].Id := aId;
  result.ToolCalls[0].Name := aName;
  result.ToolCalls[0].ArgumentsJson := aArgs;
end;

function FinalResponse(const aContent: RawUtf8): TLlmChatResponse;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.FinishReason := lfrStop;
  result.Content := aContent;
end;

function WeatherToolDef: TLlmTool;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.Name := 'get_weather';
  result.Description := 'Get the weather for a location';
  result.ParametersJson :=
    '{"type":"object","properties":{"loc":{"type":"string"}},"required":["loc"]}';
end;


{ TTestLlmAgent }

procedure TTestLlmAgent.ToolCallingLoop;
var
  stub: TStubLlmClient;
  client: ILlmClient;
  toolbox: TLlmCallbackToolbox;
  box: ILlmToolbox;
  weather: TWeatherTool;
  agent: TLlmAgent;
  msgs: TLlmMessageDynArray;
  resp: TLlmChatResponse;
  last: TLlmMessage;
begin
  stub := TStubLlmClient.Create;
  client := stub; // interface ref owns the stub from here
  stub.Push(ToolCallResponse('call_1', 'get_weather', '{"loc":"NYC"}'));
  stub.Push(FinalResponse('It is sunny in NYC.'));

  weather := TWeatherTool.Create;
  toolbox := TLlmCallbackToolbox.Create;
  box := toolbox;
  toolbox.Add(WeatherToolDef, weather.Handle);

  agent := TLlmAgent.Create(client, box, 'test-model');
  try
    SetLength(msgs, 1);
    msgs[0] := LlmMessage(lrUser, 'Weather in NYC?');
    resp := agent.Run(msgs);

    CheckEqual(resp.Content, 'It is sunny in NYC.', 'final answer returned');
    CheckEqual(stub.Calls, 2, 'two model round-trips');
    CheckEqual(weather.CalledWith, '{"loc":"NYC"}', 'tool got the model arguments');
    // history fed back: user + assistant(tool_calls) + tool result = 3
    CheckEqual(length(stub.LastRequest.Messages), 3, 'tool result fed back');
    last := stub.LastRequest.Messages[high(stub.LastRequest.Messages)];
    Check(last.Role = lrTool, 'last history message is the tool result');
    CheckEqual(last.ToolCallId, 'call_1', 'tool result references the call id');
    CheckEqual(last.Content, '{"temp":"22C","sky":"sunny"}', 'tool result content');
  finally
    agent.Free;
    weather.Free;
  end;
end;

procedure TTestLlmAgent.MaxIterationsGuard;
var
  stub: TStubLlmClient;
  client: ILlmClient;
  toolbox: TLlmCallbackToolbox;
  box: ILlmToolbox;
  weather: TWeatherTool;
  agent: TLlmAgent;
  msgs: TLlmMessageDynArray;
  raised: boolean;
  i: integer;
begin
  stub := TStubLlmClient.Create;
  client := stub;
  for i := 1 to 5 do // always asks for a tool -> never terminates on its own
    stub.Push(ToolCallResponse('c', 'get_weather', '{"loc":"X"}'));

  weather := TWeatherTool.Create;
  toolbox := TLlmCallbackToolbox.Create;
  box := toolbox;
  toolbox.Add(WeatherToolDef, weather.Handle);

  agent := TLlmAgent.Create(client, box, 'test-model');
  agent.MaxIterations := 3;
  try
    SetLength(msgs, 1);
    msgs[0] := LlmMessage(lrUser, 'loop forever');
    raised := false;
    try
      agent.Run(msgs);
    except
      on E: ELlmAgent do
        raised := true;
    end;
    Check(raised, 'ELlmAgent raised on iteration overflow');
    CheckEqual(stub.Calls, 3, 'stopped after exactly MaxIterations calls');
  finally
    agent.Free;
    weather.Free;
  end;
end;

end.
