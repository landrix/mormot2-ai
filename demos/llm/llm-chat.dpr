// LandrixAI LLM client demo - streams a chat completion from an OpenAI-compatible
// endpoint (OpenAI / LiteLLM / Ollama) and prints tokens as they arrive.
//
//   llm-chat [prompt]
//   provider via env: LLM_PROVIDER=openai|anthropic (default openai)
//     openai:    LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
//     anthropic: ANTHROPIC_API_KEY (required), ANTHROPIC_MODEL (default claude-opus-4-8)
//   the key is read from the environment ONLY - never hard-code or log it
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
  mormot.ai.llm.openai,
  mormot.ai.llm.anthropic; // native Anthropic wire (LLM_PROVIDER=anthropic)

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
  client: ILlmClient; // interface-managed; works for OpenAI-wire or Anthropic
  printer: TPrinter;
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  server, model, prompt, provider, key: RawUtf8;
begin
  {$ifdef UNIX}
  OpenSslInitialize; // enable TLS so HTTPS endpoints work
  {$endif}
  // provider selected by LLM_PROVIDER; the same streaming path drives either wire
  provider := LowerCaseU(TrimU(StringToUtf8(GetEnvironmentVariable('LLM_PROVIDER'))));
  if provider = 'anthropic' then
  begin
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
    server := 'Anthropic Messages API';
    client := TAnthropicClient.Create(cfg);
  end
  else
  begin
    cfg := LlmConfigFromEnv; // LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
    cfg.TimeoutMs := 120000;
    server := cfg.BaseUrl;
    model := cfg.DefaultModel;
    client := TLlmClient.Create(cfg);
  end;
  prompt := StringToUtf8(ParamStr(1));
  if prompt = '' then
    prompt := 'Say hello in one short sentence.';

  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, prompt);
  req := LlmChatRequest(model, msgs);

  printer := TPrinter.Create;
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
    printer.Free; // client is interface-managed - no manual Free
  end;
end.
