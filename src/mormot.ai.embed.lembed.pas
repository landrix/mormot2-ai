/// LandrixAI - local IEmbedder via sqlite-lembed (GGUF model in SQLite)
// - part of the mormot.ai.* extension (LandrixAI)
// - implements IEmbedder by running a GGUF embedding model INSIDE SQLite through
//   the sqlite-lembed (lembed0) extension. The embedder OWNS a dedicated in-memory
//   SQLite connection that hosts only lembed0 + the registered model - it is a
//   process-scope embedding SERVICE, fully decoupled from any vector store.
// - Why decoupled: embedding generation (text -> vector) is orthogonal to vector
//   STORAGE (IVectorStore, per-DB). A single embedder serves N stores/DBs, so the
//   GGUF model is loaded into RAM exactly ONCE - not once per DB connection. Use
//   SharedLembedEmbedder() to get the process-wide singleton per model name.
// - Thread-safe: the underlying llama/lembed context is not re-entrant, so a shared
//   singleton serialises Embed/EmbedBatch under a lightweight lock (embeddings are
//   fast + CPU-bound; a small critical section is fine - scale out via a second
//   IEmbedder backend, e.g. Ollama, if concurrency ever dominates).
// - SQL patterns mirror the public sqlite-lembed interface
unit mormot.ai.embed.lembed;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text, // ESynException
  mormot.core.os,   // TOSLightLock
  mormot.db.raw.sqlite3,
  mormot.ai.llm.types,
  mormot.ai.embeddings,
  mormot.ai.vectorstore,           // BlobToVector helper (SQLite-free)
  mormot.ai.vectorstore.sqlitevec; // Enable/LoadSqliteExtension + SqliteExtSuffix

type
  /// a local IEmbedder using sqlite-lembed (a GGUF model run inside SQLite)
  // - owns a dedicated in-memory connection that hosts only lembed0 + the model,
  //   so it is independent of any vector store and loads the model just once
  TLembedEmbedder = class(TInterfacedObject, IEmbedder)
  protected
    fDB: TSqlDataBase; // owned dedicated connection (:memory:), freed here
    fModel: RawUtf8;
    fLock: TOSLightLock; // serialises the non-re-entrant lembed context
  public
    /// create a dedicated in-memory connection, load lembed0 from aExtDir and
    /// register the GGUF model at aModelPath under the logical name aModelName
    // - prefer SharedLembedEmbedder() over calling this directly, so the same
    //   model is not loaded twice within one process
    constructor Create(const aExtDir, aModelPath, aModelName: RawUtf8); reintroduce;
    destructor Destroy; override;
    function Embed(const aText: RawUtf8): TLlmEmbedding;
    function EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    function Model: RawUtf8;
  end;


/// the process-wide shared lembed embedder for a logical model name
// - returns a cached singleton: the first call for aModelName loads the model,
//   every later call with the same name returns the SAME IEmbedder - so N vector
//   stores / DBs share one in-RAM model instead of loading one copy each
// - the logical NAME is the cache key (two paths under the same name collide by
//   design - the name is the model's identity); aExtDir/aModelPath are only used
//   on the first (constructing) call
function SharedLembedEmbedder(const aExtDir, aModelPath,
  aModelName: RawUtf8): IEmbedder;


implementation

{ TLembedEmbedder }

constructor TLembedEmbedder.Create(const aExtDir, aModelPath, aModelName: RawUtf8);
var
  r: TSqlRequest;
begin
  inherited Create;
  fLock.Init;
  fModel := aModelName;
  // own a private in-memory connection: it hosts only lembed0 + the model, never
  // a vector store, so this embedder can serve any number of stores/DBs
  fDB := TSqlDataBase.Create(Utf8ToString(SQLITE_MEMORY_DATABASE_NAME), '');
  EnableSqliteExtensions(fDB.DB);
  LoadSqliteExtension(fDB.DB, aExtDir + '/lembed0' + SqliteExtSuffix);
  r.Prepare(fDB.DB,
    'INSERT INTO temp.lembed_models(name, model) VALUES (?, lembed_model_from_file(?));');
  try
    r.Bind(1, aModelName);
    r.Bind(2, aModelPath);
    r.Step;
  finally
    r.Close;
  end;
end;

destructor TLembedEmbedder.Destroy;
begin
  fDB.Free; // owned here (unlike the old shared-connection design)
  fLock.Done;
  inherited Destroy;
end;

function TLembedEmbedder.Embed(const aText: RawUtf8): TLlmEmbedding;
var
  r: TSqlRequest;
  blob: RawByteString;
begin
  fLock.Lock;
  try
    r.Prepare(fDB.DB, 'SELECT lembed(?, ?);');
    try
      r.Bind(1, fModel);
      r.Bind(2, aText);
      if r.Step = SQLITE_ROW then
        blob := r.FieldBlob(0);
    finally
      r.Close;
    end;
  finally
    fLock.UnLock;
  end;
  // fail loudly instead of returning an empty vector that would later be stored
  // or matched as a zero-length blob
  if blob = '' then
    ESynException.RaiseUtf8('%.Embed: lembed returned no vector (model %)',
      [self, fModel]);
  result := BlobToVector(blob);
end;

function TLembedEmbedder.EmbedBatch(
  const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
var
  i: PtrInt;
  r: TSqlRequest;
  blob: RawByteString;
begin
  SetLength(result, length(aTexts));
  if aTexts = nil then
    exit;
  fLock.Lock;
  try
    // prepare the lembed statement ONCE and reuse it per text via Reset(); calling
    // Embed() per item re-prepares 'SELECT lembed(?,?)' on every chunk (and would
    // also re-acquire the lock per item)
    r.Prepare(fDB.DB, 'SELECT lembed(?, ?);');
    try
      for i := 0 to high(aTexts) do
      begin
        r.Bind(1, fModel);
        r.Bind(2, aTexts[i]);
        blob := '';
        if r.Step = SQLITE_ROW then
          blob := r.FieldBlob(0);
        r.Reset;
        // same loud-fail contract as Embed(): never return a zero-length vector
        if blob = '' then
          ESynException.RaiseUtf8('%.EmbedBatch: lembed returned no vector (model %)',
            [self, fModel]);
        result[i] := BlobToVector(blob);
      end;
    finally
      r.Close;
    end;
  finally
    fLock.UnLock;
  end;
end;

function TLembedEmbedder.Model: RawUtf8;
begin
  result := fModel;
end;


{ process-wide singleton registry (keyed by logical model name) }

type
  TLembedCacheEntry = record
    Name: RawUtf8;
    Emb: IEmbedder;
  end;

var
  gLembedLock: TOSLightLock;
  gLembed: array of TLembedCacheEntry;

function SharedLembedEmbedder(const aExtDir, aModelPath,
  aModelName: RawUtf8): IEmbedder;
var
  i: PtrInt;
begin
  result := nil;
  gLembedLock.Lock;
  try
    for i := 0 to high(gLembed) do
      if gLembed[i].Name = aModelName then
      begin
        result := gLembed[i].Emb; // cache hit: reuse the already-loaded model
        exit;
      end;
    // first request for this name: load the model once, then cache it
    result := TLembedEmbedder.Create(aExtDir, aModelPath, aModelName);
    SetLength(gLembed, length(gLembed) + 1);
    gLembed[high(gLembed)].Name := aModelName;
    gLembed[high(gLembed)].Emb := result;
  finally
    gLembedLock.UnLock;
  end;
end;

initialization
  gLembedLock.Init;

finalization
  gLembed := nil; // release the cached IEmbedder refs (frees each owned connection)
  gLembedLock.Done;

end.
