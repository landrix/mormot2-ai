// LandrixAI agent demo - a tool-calling loop against an OpenAI-compatible
// endpoint: the model asks for get_weather, the agent runs it, feeds the result
// back, and the model produces a final natural-language answer.
//
//   llm-agent [baseUrl] [model]
//   defaults: http://172.16.122.3:11434/v1   Keyvan/german-text-3.1:latest
program llm.agent;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.openai,
  mormot.ai.agent;

type
  TDemoTools = class
    function GetWeather(const aArgumentsJson: RawUtf8): RawUtf8;
  end;

function TDemoTools.GetWeather(const aArgumentsJson: RawUtf8): RawUtf8;
begin
  ConsoleWrite(FormatUtf8('  [tool get_weather called with %]', [aArgumentsJson]),
    ccLightMagenta);
  // a canned result is enough to prove the loop feeds it back to the model
  result := '{"location":"Berlin","temp_c":18,"sky":"clear"}';
end;

function WeatherTool: TLlmTool;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.Name := 'get_weather';
  result.Description := 'Get the current weather for a city';
  result.ParametersJson :=
    '{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}';
end;

var
  cfg: TLlmProviderConfig;
  client: ILlmClient;
  tools: TDemoTools;
  toolbox: TLlmCallbackToolbox;
  box: ILlmToolbox;
  agent: TLlmAgent;
  msgs: TLlmMessageDynArray;
  resp: TLlmChatResponse;
  server, model: RawUtf8;
begin
  server := StringToUtf8(ParamStr(1));
  if server = '' then
    server := 'http://172.16.122.3:11434/v1';
  model := StringToUtf8(ParamStr(2));
  if model = '' then
    model := 'Keyvan/german-text-3.1:latest';

  cfg := OllamaConfig(server, model);
  client := TLlmClient.Create(cfg);
  tools := TDemoTools.Create;
  toolbox := TLlmCallbackToolbox.Create;
  box := toolbox;
  toolbox.Add(WeatherTool, tools.GetWeather);
  agent := TLlmAgent.Create(client, box, model);
  try
    SetLength(msgs, 2);
    msgs[0] := LlmMessage(lrSystem,
      'You are a helpful assistant. Use the get_weather tool when asked about weather.');
    msgs[1] := LlmMessage(lrUser,
      'What is the weather in Berlin? Answer in one short sentence.');
    ConsoleWrite(FormatUtf8('>>> %  model=%', [server, model]), ccLightBlue);
    ConsoleWrite('--- agent run ---', ccLightBlue);
    try
      resp := agent.Run(msgs);
      ConsoleWrite(FormatUtf8('ANSWER: %', [resp.Content]), ccLightGreen);
    except
      on E: Exception do
        ConsoleWrite(FormatUtf8('ERROR %: %', [E.ClassName, E.Message]), ccLightRed);
    end;
  finally
    agent.Free;
    tools.Free;
  end;
end.
