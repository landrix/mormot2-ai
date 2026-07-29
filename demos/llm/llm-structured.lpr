// LandrixAI structured-output demo - extract a typed record from free text via
// an OpenAI-compatible endpoint: the record's RTTI drives the JSON schema, and
// the model's JSON answer is loaded straight back into the record.
//
//   llm-structured
//   provider via env: LLM_PROVIDER=openai|anthropic (default openai)
//     openai:    LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
//     anthropic: ANTHROPIC_API_KEY (required), ANTHROPIC_MODEL (default claude-opus-4-8)
//   on Anthropic the json_schema is sent as output_config.format (not response_format);
//   the key is read from the environment ONLY - never hard-code or log it
program llm.structured;

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
  mormot.core.rtti,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.openai,
  mormot.ai.llm.anthropic, // native Anthropic wire (LLM_PROVIDER=anthropic)
  mormot.ai.llm.structured;

type
  TInvoice = packed record
    vendor: RawUtf8;
    contact: RawUtf8;
    amount_eur: integer;
  end;

var
  cfg: TLlmProviderConfig;
  client: ILlmClient;
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  inv: TInvoice;
  server, model, provider, key: RawUtf8;
begin
  {$ifdef UNIX}
  OpenSslInitialize; // enable TLS so HTTPS endpoints work
  {$endif}
  // the same RTTI that an MCP tool would use to describe its input
  Rtti.RegisterFromText(TypeInfo(TInvoice),
    'vendor,contact:RawUtf8 amount_eur:integer');

  // provider selected by LLM_PROVIDER; ChatStructured + the schema are identical,
  // only the wire that carries the schema differs (response_format vs output_config)
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
  SetLength(msgs, 2);
  msgs[0] := LlmMessage(lrSystem,
    'Extract the invoice fields from the user text. Reply with JSON only.');
  msgs[1] := LlmMessage(lrUser,
    'Rechnung von Mueller GmbH, Ansprechpartner Hans Meier, Betrag 450 Euro.');
  req := LlmChatRequest(model, msgs);

  ConsoleWrite(FormatUtf8('>>> %  model=%', [server, model]), ccLightBlue);
  ConsoleWrite('--- structured extraction ---', ccLightBlue);
  try
    if ChatStructured(client, req, TypeInfo(TInvoice), inv) then
      ConsoleWrite(FormatUtf8('vendor=% | contact=% | amount_eur=%',
        [inv.vendor, inv.contact, inv.amount_eur]), ccLightGreen)
    else
      ConsoleWrite('could not parse the model answer into the record', ccLightRed);
  except
    on E: Exception do
      ConsoleWrite(FormatUtf8('ERROR %: %', [E.ClassName, E.Message]), ccLightRed);
  end;
end.
