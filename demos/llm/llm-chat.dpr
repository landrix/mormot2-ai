// LandrixAI LLM client demo - streams a chat completion from an OpenAI-compatible
// endpoint (OpenAI / LiteLLM / Ollama) and prints tokens as they arrive.
//
//   llm-chat [baseUrl] [model] [prompt]
//   defaults: http://172.16.122.3:11434/v1   llama3.2   "Say hello ..."
program llm.chat;

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
  mormot.ai.llm.openai;

type
  // the streaming callback must be a method (of object) - see mormot.ai.llm.sse
  TPrinter = class
    procedure OnDelta(const aDelta: TLlmStreamDelta);
  end;

procedure TPrinter.OnDelta(const aDelta: TLlmStreamDelta);
begin
  if aDelta.ContentDelta <> '' then
    ConsoleWrite(aDelta.ContentDelta, ccWhite, {nolinefeed=}true);
  if aDelta.HasUsage then
    ConsoleWrite(FormatUtf8(#10'[usage: % prompt + % completion = % tokens]',
      [aDelta.Usage.PromptTokens, aDelta.Usage.CompletionTokens,
       aDelta.Usage.TotalTokens]), ccLightGray);
end;

var
  cfg: TLlmProviderConfig;
  client: TLlmClient;
  printer: TPrinter;
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  server, model, prompt: RawUtf8;
begin
  server := StringToUtf8(ParamStr(1));
  if server = '' then
    server := 'http://172.16.122.3:11434/v1';
  model := StringToUtf8(ParamStr(2));
  if model = '' then
    model := 'llama3.2';
  prompt := StringToUtf8(ParamStr(3));
  if prompt = '' then
    prompt := 'Say hello in one short sentence.';

  cfg := OllamaConfig(server, model);
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, prompt);
  req := LlmChatRequest(model, msgs);

  printer := TPrinter.Create;
  client := TLlmClient.Create(cfg);
  try
    ConsoleWrite(FormatUtf8('>>> %  model=%', [server, model]), ccLightBlue);
    ConsoleWrite(FormatUtf8('>>> prompt: %'#10'--- streaming ---', [prompt]), ccLightBlue);
    try
      client.ChatStream(req, printer.OnDelta);
      ConsoleWrite(#10'--- done ---', ccLightGreen);
    except
      on E: Exception do
        ConsoleWrite(FormatUtf8(#10'ERROR %: %', [E.ClassName, E.Message]), ccLightRed);
    end;
  finally
    client.Free;
    printer.Free;
  end;
end.
