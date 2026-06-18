// LandrixAI structured-output demo - extract a typed record from free text via
// an OpenAI-compatible endpoint: the record's RTTI drives the JSON schema, and
// the model's JSON answer is loaded straight back into the record.
//
//   llm-structured
//   provider via env: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY (key never logged)
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
  server, model: RawUtf8;
begin
  {$ifdef UNIX}
  OpenSslInitialize; // enable TLS so HTTPS endpoints (OpenAI/LiteLLM) work
  {$endif}
  // the same RTTI that an MCP tool would use to describe its input
  Rtti.RegisterFromText(TypeInfo(TInvoice),
    'vendor,contact:RawUtf8 amount_eur:integer');

  // provider comes from the environment: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
  cfg := LlmConfigFromEnv;
  cfg.TimeoutMs := 120000;
  server := cfg.BaseUrl;
  model := cfg.DefaultModel;
  client := TLlmClient.Create(cfg);
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
