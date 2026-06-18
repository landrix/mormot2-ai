// LandrixAI Anthropic demo - a tool-calling loop against the native Anthropic
// Messages API. Same TLlmAgent + toolbox as the OpenAI demo (llm-agent.dpr);
// only the ILlmClient implementation differs (TAnthropicClient) - proving the
// neutral request/response records are provider-agnostic.
//
//   llm-anthropic
//   env: ANTHROPIC_API_KEY (required), ANTHROPIC_MODEL (default claude-opus-4-8)
//   the key is read from the environment ONLY - never hard-code or log it
program llm.anthropic;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  {$ifdef UNIX}
  mormot.lib.openssl11, // HTTPS/TLS on POSIX (Windows uses SChannel)
  {$endif}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.anthropic,
  mormot.ai.agent;

type
  TDemoTools = class
    function GetWeather(const aArgumentsJson: RawUtf8): RawUtf8;
  end;

function TDemoTools.GetWeather(const aArgumentsJson: RawUtf8): RawUtf8;
begin
  ConsoleWrite(FormatUtf8('  [tool get_weather called with %]', [aArgumentsJson]),
    ccLightMagenta);
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
  key, model: RawUtf8;
begin
  {$ifdef UNIX}
  OpenSslInitialize; // enable TLS so the HTTPS Anthropic endpoint works
  {$endif}
  key := TrimU(StringToUtf8(GetEnvironmentVariable('ANTHROPIC_API_KEY')));
  if key = '' then
  begin
    ConsoleWrite('set ANTHROPIC_API_KEY', ccLightRed);
    exit;
  end;
  model := TrimU(StringToUtf8(GetEnvironmentVariable('ANTHROPIC_MODEL')));
  if model = '' then
    model := 'claude-opus-4-8';

  cfg := AnthropicConfig(key, model);
  cfg.TimeoutMs := 120000;
  client := TAnthropicClient.Create(cfg);
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
    ConsoleWrite(FormatUtf8('>>> Anthropic Messages API  model=%', [model]), ccLightBlue);
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
