// - regression tests for mormot.ai.rag.tool (agentic RAG: search_docs MCP tool)
// - uses fake IEmbedder + IVectorStore so the wiring (embed -> search -> format
//   -> bridge -> agent) is exercised without the vec0/lembed shared libraries
unit test.llm.rag.tool;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.core.rtti,
  mormot.core.variants,
  mormot.core.test,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.embeddings,
  mormot.ai.vectorstore,
  mormot.ai.mcp,
  mormot.ai.agent,
  mormot.ai.agent.mcp,
  mormot.ai.rag,      // TLlmRag.Ingest (strict-count regression)
  mormot.ai.rag.tool,
  test.llm.agent; // reuse the scripted TStubLlmClient

type
  /// a fake embedder: returns a fixed 1-dim vector (the store ignores it)
  TFakeEmbedder = class(TInterfacedObject, IEmbedder)
  public
    function Embed(const aText: RawUtf8): TLlmEmbedding;
    function EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    function Model: RawUtf8;
  end;

  /// a fake vector store: returns canned hits, records the query and top-k it saw
  TFakeStore = class(TInterfacedObject, IVectorStore)
  protected
    fHits: TRagHitDynArray;
    fLastTopK: integer;
  public
    procedure SetHits(const aTexts: array of RawUtf8);
    function Add(const aText: RawUtf8; const aVector: TLlmEmbedding): Int64;
    function AddBatch(const aTexts: TRawUtf8DynArray;
      const aVectors: TLlmEmbeddingDynArray): integer;
    procedure Upsert(const aId, aText: RawUtf8; const aVector: TLlmEmbedding);
    procedure Delete(const aId: RawUtf8);
    function Search(const aQuery: TLlmEmbedding; aTopK: integer): TRagHitDynArray;
    function Count: Int64;
    property LastTopK: integer read fLastTopK;
  end;

  /// an embedder that returns FEWER vectors than requested, to drive the
  /// strict-count guard in TLlmRag.Ingest
  TShortEmbedder = class(TInterfacedObject, IEmbedder)
  public
    function Embed(const aText: RawUtf8): TLlmEmbedding;
    function EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    function Model: RawUtf8;
  end;

  TTestLlmRagTool = class(TSynTestCase)
  protected
    function NewServer(const aStore: IVectorStore; aTopK: integer): TMcpServer;
  published
    procedure SearchViaBridge;
    procedure EmptyQueryIsError;
    procedure NoHits;
    procedure AgentRetrievesOnDemand;
    procedure IngestRejectsShortEmbedding;
    procedure IngestStoresAllChunks;
  end;


implementation

{ TFakeEmbedder }

function TFakeEmbedder.Embed(const aText: RawUtf8): TLlmEmbedding;
begin
  SetLength(result, 1);
  result[0] := length(aText); // arbitrary but deterministic; the store ignores it
end;

function TFakeEmbedder.EmbedBatch(
  const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
var
  i: PtrInt;
begin
  SetLength(result, length(aTexts));
  for i := 0 to high(aTexts) do
    result[i] := Embed(aTexts[i]);
end;

function TFakeEmbedder.Model: RawUtf8;
begin
  result := 'fake-embed';
end;


{ TFakeStore }

procedure TFakeStore.SetHits(const aTexts: array of RawUtf8);
var
  i: PtrInt;
begin
  SetLength(fHits, length(aTexts));
  for i := 0 to high(aTexts) do
  begin
    fHits[i].DocId := i + 1;
    fHits[i].Text := aTexts[i];
    fHits[i].Distance := i; // nearest first
  end;
end;

function TFakeStore.Add(const aText: RawUtf8; const aVector: TLlmEmbedding): Int64;
begin
  result := 0; // not exercised
end;

function TFakeStore.AddBatch(const aTexts: TRawUtf8DynArray;
  const aVectors: TLlmEmbeddingDynArray): integer;
begin
  result := length(aTexts); // not exercised by these tests
end;

procedure TFakeStore.Upsert(const aId, aText: RawUtf8;
  const aVector: TLlmEmbedding);
begin
  // not exercised by the rag-tool tests (keyed ops are covered in test.llm.rag)
end;

procedure TFakeStore.Delete(const aId: RawUtf8);
begin
  // not exercised here
end;

function TFakeStore.Search(const aQuery: TLlmEmbedding;
  aTopK: integer): TRagHitDynArray;
var
  n: integer;
begin
  fLastTopK := aTopK;
  n := length(fHits);
  if (aTopK > 0) and (aTopK < n) then
    n := aTopK;
  result := copy(fHits, 0, n);
end;

function TFakeStore.Count: Int64;
begin
  result := length(fHits);
end;


{ TTestLlmRagTool }

function TTestLlmRagTool.NewServer(const aStore: IVectorStore;
  aTopK: integer): TMcpServer;
begin
  result := TMcpServer.Create('test-rag', '1.0.0');
  RegisterRagSearchTool(result, TFakeEmbedder.Create, aStore, aTopK);
  result.Start;
end;

procedure TTestLlmRagTool.SearchViaBridge;
var
  store: TFakeStore;
  istore: IVectorStore;
  server: TMcpServer;
  box: ILlmToolbox;
  tools: TLlmToolDynArray;
  res: RawUtf8;
begin
  store := TFakeStore.Create;
  istore := store;
  store.SetHits(['das dach ist undicht', 'heizung gewartet']);
  server := NewServer(istore, 4);
  try
    box := TLlmMcpToolbox.Create(server);

    tools := box.List;
    CheckEqual(length(tools), 1, 'one tool listed');
    CheckEqual(tools[0].Name, RAG_SEARCH_TOOL_NAME, 'tool name is search_docs');
    Check(Pos(RawUtf8('"query"'), tools[0].ParametersJson) > 0,
      'schema advertises the query parameter');

    res := box.Execute(RAG_SEARCH_TOOL_NAME, '{"query":"dach undicht"}');
    Check(Pos(RawUtf8('[1] das dach ist undicht'), res) > 0, 'first hit cited');
    Check(Pos(RawUtf8('[2] heizung gewartet'), res) > 0, 'second hit cited');
    CheckEqual(store.LastTopK, 4, 'tool passed its top-k to the store');
  finally
    box := nil; // release the bridge before the (unowned) server
    server.Free;
  end;
end;

procedure TTestLlmRagTool.EmptyQueryIsError;
var
  store: IVectorStore;
  server: TMcpServer;
  box: ILlmToolbox;
  res: RawUtf8;
begin
  store := TFakeStore.Create;
  server := NewServer(store, 4);
  try
    box := TLlmMcpToolbox.Create(server);
    // an empty query must surface as a tool-level error, not a search
    res := box.Execute(RAG_SEARCH_TOOL_NAME, '{"query":"   "}');
    Check(Pos(RawUtf8('"isError":true'), res) > 0, 'empty query flagged as error');
    Check(Pos(RawUtf8('must not be empty'), res) > 0, 'error message preserved');
  finally
    box := nil;
    server.Free;
  end;
end;

procedure TTestLlmRagTool.NoHits;
var
  store: IVectorStore;
  server: TMcpServer;
  box: ILlmToolbox;
  res: RawUtf8;
begin
  store := TFakeStore.Create; // no hits seeded
  server := NewServer(store, 4);
  try
    box := TLlmMcpToolbox.Create(server);
    res := box.Execute(RAG_SEARCH_TOOL_NAME, '{"query":"unrelated"}');
    Check(Pos(RawUtf8('No matching passages'), res) > 0, 'empty result reported');
    Check(Pos(RawUtf8('isError'), res) = 0, 'a miss is not an error');
  finally
    box := nil;
    server.Free;
  end;
end;

function RagToolCall(const aQuery: RawUtf8): TLlmChatResponse;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.FinishReason := lfrToolCalls;
  SetLength(result.ToolCalls, 1);
  result.ToolCalls[0].Id := 'call_1';
  result.ToolCalls[0].Name := RAG_SEARCH_TOOL_NAME;
  result.ToolCalls[0].ArgumentsJson := FormatUtf8('{"query":"%"}', [aQuery]);
end;

function RagFinal(const aContent: RawUtf8): TLlmChatResponse;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.FinishReason := lfrStop;
  result.Content := aContent;
end;

procedure TTestLlmRagTool.AgentRetrievesOnDemand;
var
  store: TFakeStore;
  istore: IVectorStore;
  server: TMcpServer;
  stub: TStubLlmClient;
  client: ILlmClient;
  box: ILlmToolbox;
  agent: TLlmAgent;
  msgs: TLlmMessageDynArray;
  resp: TLlmChatResponse;
  last: TLlmMessage;
begin
  store := TFakeStore.Create;
  istore := store;
  store.SetHits(['das dach ist undicht und muss saniert werden']);
  server := NewServer(istore, 4);
  stub := TStubLlmClient.Create;
  client := stub;
  // round 1: the model retrieves; round 2: it answers grounded in the passage
  stub.Push(RagToolCall('dach'));
  stub.Push(RagFinal('Das Dach ist undicht. [1]'));
  try
    box := TLlmMcpToolbox.Create(server);
    agent := TLlmAgent.Create(client, box, 'test-model');
    try
      SetLength(msgs, 2);
      msgs[0] := LlmMessage(lrSystem, RAG_AGENT_SYSTEM_PROMPT);
      msgs[1] := LlmMessage(lrUser, 'Was ist mit dem Dach?');
      resp := agent.Run(msgs);

      CheckEqual(resp.Content, 'Das Dach ist undicht. [1]', 'final grounded answer');
      CheckEqual(stub.Calls, 2, 'one retrieval round-trip then the answer');
      // the retrieved passage must have been fed back to the model
      last := stub.LastRequest.Messages[high(stub.LastRequest.Messages)];
      Check(last.Role = lrTool, 'last history message is the retrieval result');
      Check(Pos(RawUtf8('[1] das dach ist undicht'), last.Content) > 0,
        'retrieved passage fed back to the model');
    finally
      agent.Free;
      box := nil;
    end;
  finally
    server.Free;
  end;
end;

{ TShortEmbedder }

function TShortEmbedder.Embed(const aText: RawUtf8): TLlmEmbedding;
begin
  SetLength(result, 1);
  result[0] := 1;
end;

function TShortEmbedder.EmbedBatch(
  const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
var
  i: PtrInt;
begin
  // deliberately return one FEWER vector than requested (or none for a single
  // chunk) to trip the strict-count guard — never returns a full set
  SetLength(result, length(aTexts) - 1); // 0 for a single chunk
  for i := 0 to high(result) do
    result[i] := Embed(aTexts[i]);
end;

function TShortEmbedder.Model: RawUtf8;
begin
  result := 'short-embed';
end;

procedure TTestLlmRagTool.IngestRejectsShortEmbedding;
var
  store: IVectorStore;
  emb: IEmbedder;
  rag: TLlmRag;
  raised: boolean;
begin
  store := TFakeStore.Create;
  emb := TShortEmbedder.Create;
  // one chunk requested, zero vectors returned -> count mismatch must raise
  // (no silent partial index)
  rag := TLlmRag.Create(nil, emb, store, 'test-model');
  try
    rag.ChunkChars := 1000; // 'kurzer text' stays a single chunk
    rag.Overlap := 0;
    raised := false;
    try
      rag.Ingest('kurzer text');
    except
      on E: Exception do
        raised := true;
    end;
    Check(raised, 'a short embedding reply must fail the ingestion loudly');
  finally
    rag.Free;
  end;
end;

procedure TTestLlmRagTool.IngestStoresAllChunks;
var
  store: IVectorStore;
  emb: IEmbedder;
  rag: TLlmRag;
  n: integer;
begin
  store := TFakeStore.Create;
  emb := TFakeEmbedder.Create; // returns exactly one vector per chunk
  rag := TLlmRag.Create(nil, emb, store, 'test-model');
  try
    rag.ChunkChars := 1000;
    rag.Overlap := 0;
    n := rag.Ingest('ein dokument'); // single chunk, matching vector count
    CheckEqual(n, 1, 'all chunks stored when the embedder count matches');
  finally
    rag.Free;
  end;
end;

end.
