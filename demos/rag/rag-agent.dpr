// LandrixAI agentic RAG demo - retrieve-on-demand: a tool-calling agent decides
// WHEN to retrieve. The search_docs MCP tool runs local lembed/vec0 retrieval
// (offline), the agent calls it as needed and answers grounded ONLY in the
// retrieved passages, citing the [n] sources.
//
// Difference to rag-chat.dpr: that one always retrieves then answers (one shot);
// here the model drives retrieval through the existing TLlmAgent loop via the
// MCP toolbox bridge - the same search_docs tool a remote MCP client would call.
//
//   env: SQLITE_EXT_DIR (dir with vec0/lembed0 + llama/ggml deps)
//        LEMBED_MODEL    (path to a .gguf embedding model)
//        LEMBED_DIM      (embedding dimensions; default 384 for all-MiniLM)
//        LLM_BASE_URL / LLM_API_KEY / LLM_MODEL  (generation, default OpenAI)
//   run with LD_LIBRARY_PATH=$SQLITE_EXT_DIR
program rag.agent;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  {$ifdef UNIX}
  mormot.lib.openssl11, // HTTPS/TLS for the generation endpoint
  {$endif}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.openai,
  mormot.ai.embeddings,
  mormot.ai.vectorstore,
  mormot.ai.rag,       // TLlmRag - reused only to ingest the document
  mormot.ai.mcp,       // TMcpServer, TMcpAuthContext
  mormot.ai.agent,     // TLlmAgent, ILlmToolbox
  mormot.ai.agent.mcp, // TLlmMcpToolbox bridge
  mormot.ai.rag.tool;  // TRagSearchTool, search_docs

const
  DOC =
    'Die Firma Mustermann GmbH bietet Dachsanierung, Heizungswartung und ' +
    'Fassadenarbeiten an. Eine Dachsanierung kostet ab 4500 Euro und dauert in ' +
    'der Regel drei Tage. Die Heizungswartung wird jaehrlich empfohlen und ' +
    'kostet 120 Euro. Fuer Fassadenarbeiten gibt es einen kostenlosen ' +
    'Vor-Ort-Termin. Der Notdienst ist rund um die Uhr unter 0800-123456 erreichbar.';

type
  /// a search_docs tool that logs each call, so the demo shows the agent
  /// retrieving on its own initiative
  TLoggingRagTool = class(TRagSearchTool)
  protected
    function ExecuteTyped(const aParams: TRagSearchParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

function TLoggingRagTool.ExecuteTyped(const aParams: TRagSearchParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  ConsoleWrite(FormatUtf8('  [search_docs called: query=%]', [aParams.query]),
    ccLightMagenta);
  result := inherited ExecuteTyped(aParams, aAuthCtx);
end;

var
  extdir, model, chatModel: RawUtf8;
  dim: integer;
  cfg: TLlmProviderConfig;
  storeObj: TVec0Store;
  store: IVectorStore;
  emb: IEmbedder;
  client: ILlmClient;
  rag: TLlmRag;
  server: TMcpServer;
  box: ILlmToolbox;
  agent: TLlmAgent;
  msgs: TLlmMessageDynArray;
  resp: TLlmChatResponse;
begin
  {$ifdef UNIX}
  OpenSslInitialize;
  {$endif}
  extdir := StringToUtf8(GetEnvironmentVariable('SQLITE_EXT_DIR'));
  model := StringToUtf8(GetEnvironmentVariable('LEMBED_MODEL'));
  dim := StrToIntDef(GetEnvironmentVariable('LEMBED_DIM'), 384);
  if (extdir = '') or (model = '') then
  begin
    ConsoleWrite('set SQLITE_EXT_DIR and LEMBED_MODEL', ccLightRed);
    exit;
  end;
  cfg := LlmConfigFromEnv;
  cfg.TimeoutMs := 120000;
  chatModel := cfg.DefaultModel;

  storeObj := TVec0Store.Create(':memory:', extdir, dim);
  store := storeObj;
  emb := TLembedEmbedder.Create(store, storeObj.Database, extdir, model, 'embedder');
  client := TLlmClient.Create(cfg);

  // ingest with the RAG pipeline (chunk -> local embed -> store)
  rag := TLlmRag.Create(client, emb, store, chatModel);
  try
    rag.ChunkChars := 140;
    rag.Overlap := 30;
    ConsoleWrite(FormatUtf8('>>> retrieval=local lembed (%d)  generation=%',
      [dim, chatModel]), ccLightBlue);
    ConsoleWrite(FormatUtf8('ingested % chunks', [rag.Ingest(DOC)]), ccLightGray);
  finally
    rag.Free; // store/emb/client stay alive via their interface references
  end;

  // expose retrieval as the search_docs MCP tool and let the agent drive it
  RegisterRagSearchRtti;
  server := TMcpServer.Create('rag-agent', '1.0.0');
  server.RegisterTool(TLoggingRagTool.Create(
    emb, store, 3, RAG_SEARCH_TOOL_NAME, RAG_SEARCH_TOOL_DESCRIPTION));
  server.Start;
  box := TLlmMcpToolbox.Create(server);
  agent := TLlmAgent.Create(client, box, chatModel);
  try
    SetLength(msgs, 2);
    msgs[0] := LlmMessage(lrSystem, RAG_AGENT_SYSTEM_PROMPT);
    msgs[1] := LlmMessage(lrUser,
      'Was kostet eine Dachsanierung und wie lange dauert sie? Antworte kurz.');
    ConsoleWrite('--- agent run (retrieve-on-demand) ---', ccLightBlue);
    try
      resp := agent.Run(msgs);
      ConsoleWrite(FormatUtf8('ANSWER: %', [resp.Content]), ccLightGreen);
    except
      on E: Exception do
        ConsoleWrite(FormatUtf8('ERROR %: %', [E.ClassName, E.Message]), ccLightRed);
    end;
  finally
    agent.Free;
    box := nil;   // release the bridge before the (unowned) server
    server.Free;
  end;
end.
