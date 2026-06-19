/// LandrixAI - local IEmbedder via sqlite-lembed (GGUF model in SQLite)
// - part of the mormot.ai.* extension (LandrixAI)
// - implements IEmbedder by running a GGUF embedding model INSIDE SQLite through
//   the sqlite-lembed (lembed0) extension. It shares an existing SQLite connection
//   (typically a TVec0Store's) and loads lembed0 into it via the loader from
//   mormot.ai.vectorstore.sqlitevec - so a full RAG pipeline runs offline, no API.
// - SQL patterns mirror the public sqlite-lembed interface
unit mormot.ai.embed.lembed;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text, // ESynException
  mormot.db.raw.sqlite3,
  mormot.ai.llm.types,
  mormot.ai.embeddings,
  mormot.ai.vectorstore,           // IVectorStore (pin) + BlobToVector helper
  mormot.ai.vectorstore.sqlitevec; // LoadSqliteExtension + SqliteExtSuffix

type
  /// a local IEmbedder using sqlite-lembed (a GGUF model run inside SQLite)
  // - shares the store's connection; loads lembed0 and registers the model
  TLembedEmbedder = class(TInterfacedObject, IEmbedder)
  protected
    fStore: IVectorStore; // pins the owning store so its DB outlives this embedder
    fDB: TSqlDataBase;    // shared, owned by fStore - never freed here
    fModel: RawUtf8;
  public
    /// load lembed0 into the store's connection and register the GGUF model
    // - aStore owns the SQLite connection (e.g. a TVec0Store); the embedder keeps
    //   a reference to it, so the connection cannot be freed while the embedder
    //   is alive (avoids a dangling shared handle)
    constructor Create(const aStore: IVectorStore; aDB: TSqlDataBase;
      const aExtDir, aModelPath, aModelName: RawUtf8);
    function Embed(const aText: RawUtf8): TLlmEmbedding;
    function EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    function Model: RawUtf8;
  end;


implementation

constructor TLembedEmbedder.Create(const aStore: IVectorStore; aDB: TSqlDataBase;
  const aExtDir, aModelPath, aModelName: RawUtf8);
var
  r: TSqlRequest;
begin
  inherited Create;
  fStore := aStore;
  fDB := aDB;
  fModel := aModelName;
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

function TLembedEmbedder.Embed(const aText: RawUtf8): TLlmEmbedding;
var
  r: TSqlRequest;
  blob: RawByteString;
begin
  r.Prepare(fDB.DB, 'SELECT lembed(?, ?);');
  try
    r.Bind(1, fModel);
    r.Bind(2, aText);
    if r.Step = SQLITE_ROW then
      blob := r.FieldBlob(0);
  finally
    r.Close;
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
  // prepare the lembed statement ONCE and reuse it per text via Reset(); calling
  // Embed() per item re-prepares 'SELECT lembed(?,?)' on every chunk
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
end;

function TLembedEmbedder.Model: RawUtf8;
begin
  result := fModel;
end;

end.
