// LandrixAI RAG demo - local retrieval (sqlite-lembed embeddings + sqlite-vec
// store, offline) + provider generation (OpenAI): ingest a document, then answer
// a question grounded ONLY in the retrieved chunks.
//
//   env: SQLITE_EXT_DIR (dir with vec0/lembed0 + llama/ggml deps)
//        LEMBED_MODEL    (path to a .gguf embedding model)
//        LEMBED_DIM      (embedding dimensions; default 384 for all-MiniLM)
//        LLM_BASE_URL / LLM_API_KEY / LLM_MODEL  (generation, default OpenAI)
//   run with LD_LIBRARY_PATH=$SQLITE_EXT_DIR
program rag.chat;

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
  mormot.ai.rag;

const
  DOC =
    'Die Firma Mustermann GmbH bietet Dachsanierung, Heizungswartung und ' +
    'Fassadenarbeiten an. Eine Dachsanierung kostet ab 4500 Euro und dauert in ' +
    'der Regel drei Tage. Die Heizungswartung wird jaehrlich empfohlen und ' +
    'kostet 120 Euro. Fuer Fassadenarbeiten gibt es einen kostenlosen ' +
    'Vor-Ort-Termin. Der Notdienst ist rund um die Uhr unter 0800-123456 erreichbar.';

var
  extdir, model, chatModel: RawUtf8;
  dim: integer;
  cfg: TLlmProviderConfig;
  storeObj: TVec0Store;
  store: IVectorStore;
  emb: IEmbedder;
  client: ILlmClient;
  rag: TLlmRag;
  resp: TLlmChatResponse;
  i: PtrInt;
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
  rag := TLlmRag.Create(client, emb, store, chatModel);
  try
    rag.ChunkChars := 140;
    rag.Overlap := 30;
    rag.TopK := 3;

    ConsoleWrite(FormatUtf8('>>> retrieval=local lembed (%d)  generation=%',
      [dim, chatModel]), ccLightBlue);
    ConsoleWrite(FormatUtf8('ingested % chunks', [rag.Ingest(DOC)]), ccLightGray);

    try
      resp := rag.Query('Was kostet eine Dachsanierung und wie lange dauert sie?');
      ConsoleWrite('--- retrieved chunks ---', ccLightCyan);
      for i := 0 to high(rag.LastHits) do
        ConsoleWrite(FormatUtf8('  [%] dist=% | %',
          [i + 1, rag.LastHits[i].Distance, rag.LastHits[i].Text]), ccLightCyan);
      ConsoleWrite(FormatUtf8('ANSWER: %', [resp.Content]), ccLightGreen);
    except
      on E: Exception do
        ConsoleWrite(FormatUtf8('ERROR %: %', [E.ClassName, E.Message]), ccLightRed);
    end;
  finally
    rag.Free;
  end;
end.
