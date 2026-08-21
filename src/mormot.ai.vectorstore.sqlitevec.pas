/// LandrixAI - sqlite-vec (vec0) backed IVectorStore + extension loader
// - part of the mormot.ai.* extension (LandrixAI)
// - this is the SQLite edge of the vector abstraction: it loads the sqlite-vec
//   (vec0) extension into mORMot's static SQLite and implements IVectorStore on
//   top, so a full RAG pipeline can run in-process, offline, with no API. The
//   static-SQLite dependency is isolated HERE - the engine (mormot.ai.rag) only
//   sees mormot.ai.vectorstore (the interface), never this unit.
// - mORMot does not bind sqlite3_enable_load_extension, so loading is enabled via
//   db_config(SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION) - proven cross-platform (FPC,
//   Linux+Windows); the extension shared libs live next to their llama/ggml deps.
//   The loader helpers are public so the lembed embedder (mormot.ai.embed.lembed)
//   can load lembed0 into the same connection.
// - SQL patterns mirror the public sqlite-vec interface
unit mormot.ai.vectorstore.sqlitevec;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.db.raw.sqlite3,
  mormot.db.raw.sqlite3.static,
  mormot.ai.llm.types,
  mormot.ai.vectorstore;

/// the platform shared-library suffix for a SQLite extension
function SqliteExtSuffix: RawUtf8;

/// enable run-time extension loading on a connection (mORMot omits the dedicated
/// enable API, so SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION via db_config is used)
procedure EnableSqliteExtensions(aDB: TSqlite3DB);

/// load a SQLite extension shared library by absolute path
procedure LoadSqliteExtension(aDB: TSqlite3DB; const aPath: RawUtf8);


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
    procedure Upsert(const aId, aText: RawUtf8; const aVector: TLlmEmbedding);
    procedure Delete(const aId: RawUtf8);
    function Search(const aQuery: TLlmEmbedding; aTopK: integer): TRagHitDynArray;
    function Count: Int64;
  protected
    // resolve the internal rowid mapped to an external id (0 = not found)
    function RowIdOfKey(const aId: RawUtf8): Int64;
    /// the underlying connection, shared e.g. with a TLembedEmbedder
    property Database: TSqlDataBase read fDB;
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


{ TVec0Store }

constructor TVec0Store.Create(const aDbPath, aExtDir: RawUtf8; aDim: integer);
begin
  inherited Create;
  fDim := aDim;
  fDB := TSqlDataBase.Create(Utf8ToString(aDbPath), '');
  EnableSqliteExtensions(fDB.DB);
  LoadSqliteExtension(fDB.DB, aExtDir + '/vec0' + SqliteExtSuffix);
  // ext_id is the OPTIONAL stable external key (entity id) for Upsert/Delete rows;
  // NULL for anonymous Add rows. The partial UNIQUE index makes an id map to at most
  // one row (and lets Upsert find + replace it) without constraining the Add rows.
  fDB.Execute('CREATE TABLE IF NOT EXISTS documents(' +
    'id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL, ext_id TEXT);');
  fDB.Execute('CREATE UNIQUE INDEX IF NOT EXISTS documents_ext_id_idx ' +
    'ON documents(ext_id) WHERE ext_id IS NOT NULL;');
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
  // Serialize the WHOLE write, not just its statements. TSqlDataBase locks per
  // Execute but holds nothing across a transaction, and the direct
  // Prepare/Bind/Step calls below bypass even that. Two concurrent writers would
  // interleave: TransactionBegin rolls back whatever foreign transaction it
  // finds open, and `rowid` comes from LastInsertRowID, which is per CONNECTION
  // - so one writer could pair its document with the other's vector.
  // Reentrant by contract: TSqlDataBase descends from TObjectOSLock and
  // TOSLock.Lock is explicitly reentrant, so the inner per-statement locking
  // still works.
  fDB.Lock;
  try
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
  finally
    fDB.UnLock;
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
  // one transaction for the whole document: either all chunks land or none -
  // a mid-batch failure rolls back so the document is never partially indexed.
  // Both INSERTs are prepared ONCE and reused per row via Reset() - preparing
  // inside the loop would re-compile the statements on every chunk.
  // Serialize the WHOLE write, not just its statements. TSqlDataBase locks per
  // Execute but holds nothing across a transaction, and the direct
  // Prepare/Bind/Step calls below bypass even that. Two concurrent writers would
  // interleave: TransactionBegin rolls back whatever foreign transaction it
  // finds open, and `rowid` comes from LastInsertRowID, which is per CONNECTION
  // - so one writer could pair its document with the other's vector.
  // Reentrant by contract: TSqlDataBase descends from TObjectOSLock and
  // TOSLock.Lock is explicitly reentrant, so the inner per-statement locking
  // still works.
  fDB.Lock;
  try
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
  finally
    fDB.UnLock;
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
  // Readers take the lock too. mORMot initializes SQLite with
  // SQLITE_CONFIG_MULTITHREAD, and its own header spells out what that means:
  // "application is responsible for serializing access to database
  // connections and prepared statements - as is the case with our
  // TSqlDatabase and its explicit Lock/LockJson/UnLock"
  // (mormot.db.raw.sqlite3.pas, TSqlite3Library.BeforeInitialization). So a
  // read running concurrently with a write on the SAME connection is not
  // merely dirty, it is outside what the library was configured to allow.
  fDB.Lock;
  try
  r.Prepare(fDB.DB,
    'WITH matches AS (SELECT rowid, distance FROM vec_documents ' +
    '  WHERE embedding MATCH ? AND k = ? ORDER BY distance) ' +
    'SELECT d.id, d.content, m.distance, d.ext_id FROM matches m ' +
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
      // ext_id is NULL for anonymous Add rows -> FieldUtf8 yields '' (empty Key)
      r.FieldUtf8(3, result[n].Key);
      inc(n);
    end;
    SetLength(result, n);
  finally
    r.Close;
  end;
  finally
    fDB.UnLock;
  end;
end;

function TVec0Store.RowIdOfKey(const aId: RawUtf8): Int64;
var
  r: TSqlRequest;
begin
  result := 0; // 0 = not found (documents.id is AUTOINCREMENT, starts at 1)
  // reentrant: Upsert/Delete already hold this lock when they call here
  fDB.Lock;
  try
    r.Prepare(fDB.DB, 'SELECT id FROM documents WHERE ext_id = ?;');
    try
      r.Bind(1, aId);
      if r.Step = SQLITE_ROW then
        result := r.FieldInt(0);
    finally
      r.Close;
    end;
  finally
    fDB.UnLock;
  end;
end;

procedure TVec0Store.Upsert(const aId, aText: RawUtf8;
  const aVector: TLlmEmbedding);
var
  r: TSqlRequest;
  blob: RawByteString;
  rowid: Int64;
begin
  CheckDim(aVector);
  if aId = '' then
    ESynException.RaiseUtf8('%.Upsert: empty id', [self]);
  blob := VectorToBlob(aVector);
  // one transaction: the documents row and its vec_documents row must stay in sync
  // (a half-applied replace would leave a stale or orphan vector)
  // Serialize the WHOLE write, not just its statements. TSqlDataBase locks per
  // Execute but holds nothing across a transaction, and the direct
  // Prepare/Bind/Step calls below bypass even that. Two concurrent writers would
  // interleave: TransactionBegin rolls back whatever foreign transaction it
  // finds open, and `rowid` comes from LastInsertRowID, which is per CONNECTION
  // - so one writer could pair its document with the other's vector.
  // Reentrant by contract: TSqlDataBase descends from TObjectOSLock and
  // TOSLock.Lock is explicitly reentrant, so the inner per-statement locking
  // still works.
  fDB.Lock;
  try
  fDB.TransactionBegin;
  try
    rowid := RowIdOfKey(aId);
    if rowid <> 0 then
    begin
      // replace in place: keep the same rowid so the id->row mapping is stable
      r.Prepare(fDB.DB, 'UPDATE documents SET content = ? WHERE id = ?;');
      try
        r.Bind(1, aText);
        r.Bind(2, rowid);
        r.Step;
      finally
        r.Close;
      end;
      // vec0 has no in-place embedding UPDATE across versions: delete + re-insert
      // the same rowid (both inside this transaction)
      r.Prepare(fDB.DB, 'DELETE FROM vec_documents WHERE rowid = ?;');
      try
        r.Bind(1, rowid);
        r.Step;
      finally
        r.Close;
      end;
    end
    else
    begin
      r.Prepare(fDB.DB, 'INSERT INTO documents(content, ext_id) VALUES (?, ?);');
      try
        r.Bind(1, aText);
        r.Bind(2, aId);
        r.Step;
      finally
        r.Close;
      end;
      rowid := fDB.LastInsertRowID;
    end;
    r.Prepare(fDB.DB, 'INSERT INTO vec_documents(rowid, embedding) VALUES (?, ?);');
    try
      r.Bind(1, rowid);
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
  finally
    fDB.UnLock;
  end;
end;

procedure TVec0Store.Delete(const aId: RawUtf8);
var
  r: TSqlRequest;
  rowid: Int64;
begin
  // Serialize the WHOLE write, not just its statements. TSqlDataBase locks per
  // Execute but holds nothing across a transaction, and the direct
  // Prepare/Bind/Step calls below bypass even that. Two concurrent writers would
  // interleave: TransactionBegin rolls back whatever foreign transaction it
  // finds open, and `rowid` comes from LastInsertRowID, which is per CONNECTION
  // - so one writer could pair its document with the other's vector.
  // Reentrant by contract: TSqlDataBase descends from TObjectOSLock and
  // TOSLock.Lock is explicitly reentrant, so the inner per-statement locking
  // still works.
  fDB.Lock;
  try
  fDB.TransactionBegin;
  try
    rowid := RowIdOfKey(aId);
    if rowid <> 0 then // absent id is a no-op (idempotent delete)
    begin
      r.Prepare(fDB.DB, 'DELETE FROM vec_documents WHERE rowid = ?;');
      try
        r.Bind(1, rowid);
        r.Step;
      finally
        r.Close;
      end;
      r.Prepare(fDB.DB, 'DELETE FROM documents WHERE id = ?;');
      try
        r.Bind(1, rowid);
        r.Step;
      finally
        r.Close;
      end;
    end;
    fDB.Commit;
  except
    fDB.RollBack;
    raise;
  end;
  finally
    fDB.UnLock;
  end;
end;

function TVec0Store.Count: Int64;
var
  r: TSqlRequest;
begin
  result := 0;
  fDB.Lock; // see Search: every access to this connection is ours to serialize
  try
    r.Prepare(fDB.DB, 'SELECT COUNT(*) FROM documents;');
    try
      if r.Step = SQLITE_ROW then
        result := r.FieldInt(0);
    finally
      r.Close;
    end;
  finally
    fDB.UnLock;
  end;
end;

end.
