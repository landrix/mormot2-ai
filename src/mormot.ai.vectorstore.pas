/// LandrixAI - local SQLite vector store + local embeddings
// - part of the mormot.ai.* extension (LandrixAI)
// - loads the sqlite-vec (vec0) and sqlite-lembed (lembed0) extensions into
//   mORMot's static SQLite and exposes a vector store + a local IEmbedder, so a
//   full RAG pipeline can run in-process, offline, with no API
// - mORMot does not bind sqlite3_enable_load_extension, so loading is enabled via
//   db_config(SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION) - proven cross-platform (FPC,
//   Linux+Windows); the extension shared libs live next to their llama/ggml deps
// - SQL patterns mirror the public sqlite-vec/sqlite-lembed interfaces
unit mormot.ai.vectorstore;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.db.raw.sqlite3,
  mormot.db.raw.sqlite3.static,
  mormot.ai.llm.types,
  mormot.ai.embeddings;

type
  /// one KNN hit: the stored document id, its text and the vec0 distance
  TRagHit = record
    DocId: Int64;
    Text: RawUtf8;
    Distance: double;
  end;
  TRagHitDynArray = array of TRagHit;

  /// a local document + vector store (closest-first KNN)
  IVectorStore = interface
    ['{8C1A4F92-5D63-4E7B-9A20-3F4C5D6E7A8B}']
    /// store a text and its embedding; returns the assigned document id
    function Add(const aText: RawUtf8; const aVector: TLlmEmbedding): Int64;
    /// store many text+vector pairs ATOMICALLY (single transaction)
    // - aTexts and aVectors must have the same length (else raises)
    // - either all pairs are committed or none (a failure rolls the batch back),
    //   so a document is never left partially indexed
    // - returns the number of pairs stored (= length(aTexts))
    function AddBatch(const aTexts: TRawUtf8DynArray;
      const aVectors: TLlmEmbeddingDynArray): integer;
    /// the aTopK documents nearest to a query vector, closest first
    function Search(const aQuery: TLlmEmbedding; aTopK: integer): TRagHitDynArray;
    /// number of stored documents
    function Count: Int64;
  end;


/// the platform shared-library suffix for a SQLite extension
function SqliteExtSuffix: RawUtf8;

/// enable run-time extension loading on a connection (mORMot omits the dedicated
/// enable API, so SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION via db_config is used)
procedure EnableSqliteExtensions(aDB: TSqlite3DB);

/// load a SQLite extension shared library by absolute path
procedure LoadSqliteExtension(aDB: TSqlite3DB; const aPath: RawUtf8);

/// a TLlmEmbedding as the float32 blob that vec0/lembed exchange
// - host-native byte order; vec0 and lembed both use little-endian float32, which
//   matches every supported target (x86_64/aarch64) - a big-endian port would need
//   an explicit swap here
function VectorToBlob(const aVec: TLlmEmbedding): RawByteString;
/// decode a float32 blob (e.g. from lembed) back into a TLlmEmbedding
function BlobToVector(const aBlob: RawByteString): TLlmEmbedding;


type
  /// sqlite-vec (vec0) backed local vector store, owning its SQLite connection
  TVec0Store = class(TInterfacedObject, IVectorStore)
  protected
    fDB: TSqlDataBase;
    fDim: integer;
    // guard against a wrong-length vector (turns an opaque vec0 error - or a
    // zero-length blob from a failed embedding - into a clear message)
    procedure CheckDim(const aVec: TLlmEmbedding);
  public
    /// open/create aDbPath (':memory:' allowed), load vec0 from aExtDir and
    /// create the documents + vec0 tables for aDim-dimensional vectors
    constructor Create(const aDbPath, aExtDir: RawUtf8; aDim: integer); reintroduce;
    destructor Destroy; override;
    function Add(const aText: RawUtf8; const aVector: TLlmEmbedding): Int64;
    function AddBatch(const aTexts: TRawUtf8DynArray;
      const aVectors: TLlmEmbeddingDynArray): integer;
    function Search(const aQuery: TLlmEmbedding; aTopK: integer): TRagHitDynArray;
    function Count: Int64;
    /// the underlying connection, shared e.g. with a TLembedEmbedder
    property Database: TSqlDataBase read fDB;
  end;

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

function SqliteExtSuffix: RawUtf8;
begin
  {$ifdef OSWINDOWS}
  result := '.dll';
  {$else}
  {$ifdef OSDARWIN}
  result := '.dylib';
  {$else}
  result := '.so';
  {$endif OSDARWIN}
  {$endif OSWINDOWS}
end;

procedure EnableSqliteExtensions(aDB: TSqlite3DB);
var
  enabled: integer;
begin
  enabled := 0;
  if not Assigned(sqlite3.db_config) then
    ESqlite3Exception.RaiseUtf8('EnableSqliteExtensions: db_config unavailable', []);
  if sqlite3.db_config(aDB, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1, @enabled) <> SQLITE_OK then
    ESqlite3Exception.RaiseUtf8('EnableSqliteExtensions failed: %',
      [Utf8ToString(sqlite3.errmsg(aDB))]);
end;

procedure LoadSqliteExtension(aDB: TSqlite3DB; const aPath: RawUtf8);
var
  msg: PUtf8Char;
begin
  msg := nil;
  if sqlite3.load_extension(aDB, pointer(aPath), nil, msg) <> SQLITE_OK then
    ESqlite3Exception.RaiseUtf8('LoadSqliteExtension % failed: %',
      [aPath, Utf8ToString(msg)]);
end;

function VectorToBlob(const aVec: TLlmEmbedding): RawByteString;
begin
  SetString(result, PAnsiChar(pointer(aVec)), length(aVec) * SizeOf(single));
end;

function BlobToVector(const aBlob: RawByteString): TLlmEmbedding;
begin
  result := nil;
  SetLength(result, length(aBlob) div SizeOf(single));
  if result <> nil then
    MoveFast(pointer(aBlob)^, pointer(result)^, length(result) * SizeOf(single));
end;


{ TVec0Store }

constructor TVec0Store.Create(const aDbPath, aExtDir: RawUtf8; aDim: integer);
begin
  inherited Create;
  fDim := aDim;
  fDB := TSqlDataBase.Create(Utf8ToString(aDbPath), '');
  EnableSqliteExtensions(fDB.DB);
  LoadSqliteExtension(fDB.DB, aExtDir + '/vec0' + SqliteExtSuffix);
  fDB.Execute('CREATE TABLE IF NOT EXISTS documents(' +
    'id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL);');
  fDB.Execute(FormatUtf8(
    'CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(embedding float[%]);',
    [fDim]));
end;

destructor TVec0Store.Destroy;
begin
  fDB.Free;
  inherited Destroy;
end;

procedure TVec0Store.CheckDim(const aVec: TLlmEmbedding);
begin
  if length(aVec) <> fDim then
    ESynException.RaiseUtf8('%: vector has % dims, expected % (empty = failed embedding?)',
      [self, length(aVec), fDim]);
end;

function TVec0Store.Add(const aText: RawUtf8; const aVector: TLlmEmbedding): Int64;
var
  r: TSqlRequest;
  blob: RawByteString;
begin
  CheckDim(aVector);
  blob := VectorToBlob(aVector);
  // both inserts in one transaction: a failed vector insert must not leave an
  // orphan documents row (which Count would over-report and Search never returns)
  fDB.TransactionBegin;
  try
    r.Prepare(fDB.DB, 'INSERT INTO documents(content) VALUES (?);');
    try
      r.Bind(1, aText);
      r.Step;
    finally
      r.Close;
    end;
    result := fDB.LastInsertRowID;
    r.Prepare(fDB.DB, 'INSERT INTO vec_documents(rowid, embedding) VALUES (?, ?);');
    try
      r.Bind(1, result);
      r.BindBlob(2, blob);
      r.Step;
    finally
      r.Close;
    end;
    fDB.Commit;
  except
    fDB.RollBack;
    raise;
  end;
end;

function TVec0Store.AddBatch(const aTexts: TRawUtf8DynArray;
  const aVectors: TLlmEmbeddingDynArray): integer;
var
  i: PtrInt;
  rDoc, rVec: TSqlRequest;
  rowid: Int64;
  blob: RawByteString;
begin
  result := 0;
  if length(aTexts) <> length(aVectors) then
    ESynException.RaiseUtf8('%.AddBatch: % texts but % vectors',
      [self, length(aTexts), length(aVectors)]);
  if aTexts = nil then
    exit;
  // validate every vector BEFORE opening the transaction: a wrong-length vector
  // must fail the whole batch without writing anything
  for i := 0 to high(aVectors) do
    CheckDim(aVectors[i]);
  // one transaction for the whole document: either all chunks land or none —
  // a mid-batch failure rolls back so the document is never partially indexed.
  // Both INSERTs are prepared ONCE and reused per row via Reset() — preparing
  // inside the loop would re-compile the statements on every chunk.
  fDB.TransactionBegin;
  try
    rDoc.Prepare(fDB.DB, 'INSERT INTO documents(content) VALUES (?);');
    try
      rVec.Prepare(fDB.DB,
        'INSERT INTO vec_documents(rowid, embedding) VALUES (?, ?);');
      try
        for i := 0 to high(aTexts) do
        begin
          blob := VectorToBlob(aVectors[i]);
          rDoc.Bind(1, aTexts[i]);
          rDoc.Step;
          rowid := fDB.LastInsertRowID;
          rDoc.Reset;
          rVec.Bind(1, rowid);
          rVec.BindBlob(2, blob);
          rVec.Step;
          rVec.Reset;
        end;
      finally
        rVec.Close;
      end;
    finally
      rDoc.Close;
    end;
    fDB.Commit;
    result := length(aTexts);
  except
    fDB.RollBack;
    raise;
  end;
end;

function TVec0Store.Search(const aQuery: TLlmEmbedding;
  aTopK: integer): TRagHitDynArray;
var
  r: TSqlRequest;
  blob: RawByteString;
  n: integer;
begin
  result := nil;
  CheckDim(aQuery);
  // a non-positive TopK means "no results requested" (vec0 also rejects k = 0);
  // return empty rather than silently substituting the nearest hit
  if aTopK <= 0 then
    exit;
  blob := VectorToBlob(aQuery);
  r.Prepare(fDB.DB,
    'WITH matches AS (SELECT rowid, distance FROM vec_documents ' +
    '  WHERE embedding MATCH ? AND k = ? ORDER BY distance) ' +
    'SELECT d.id, d.content, m.distance FROM matches m ' +
    'JOIN documents d ON d.id = m.rowid ORDER BY m.distance;');
  try
    r.BindBlob(1, blob);
    r.Bind(2, aTopK);
    // vec0 returns at most k rows: preallocate to k and trim to the actual count
    // afterwards, instead of reallocating on every row
    SetLength(result, aTopK);
    n := 0;
    while r.Step = SQLITE_ROW do
    begin
      // defensive: vec0 returns at most k rows, but never write past the
      // preallocated array should the extension/query ever return more
      if n >= length(result) then
        break;
      result[n].DocId := r.FieldInt(0);
      r.FieldUtf8(1, result[n].Text);
      result[n].Distance := r.FieldDouble(2);
      inc(n);
    end;
    SetLength(result, n);
  finally
    r.Close;
  end;
end;

function TVec0Store.Count: Int64;
var
  r: TSqlRequest;
begin
  result := 0;
  r.Prepare(fDB.DB, 'SELECT COUNT(*) FROM documents;');
  try
    if r.Step = SQLITE_ROW then
      result := r.FieldInt(0);
  finally
    r.Close;
  end;
end;


{ TLembedEmbedder }

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
