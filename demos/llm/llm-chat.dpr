// LandrixAI LLM client demo - streams a chat completion from an OpenAI-compatible
// endpoint (OpenAI / LiteLLM / Ollama) and prints tokens as they arrive.
//
//   llm-chat [prompt]
//   provider via env: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY (key never logged)
program llm.chat;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  {$ifdef UNIX}
  mormot.lib.openssl11, // HTTPS/TLS provider on POSIX (Windows uses SChannel)
  {$endif}
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
  {$ifdef UNIX}
  OpenSslInitialize; // enable TLS so HTTPS endpoints (OpenAI/LiteLLM) work
  {$endif}
  // provider comes from the environment: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
  cfg := LlmConfigFromEnv;
  cfg.TimeoutMs := 120000;
  server := cfg.BaseUrl;
  model := cfg.DefaultModel;
  prompt := StringToUtf8(ParamStr(1));
  if prompt = '' then
    prompt := 'Say hello in one short sentence.';

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
