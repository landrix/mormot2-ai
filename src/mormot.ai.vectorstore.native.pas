/// LandrixAI - native IVectorStore: plain SQLite tables + search in Pascal
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - replaces sqlite-vec (vec0) without any C extension: vectors are stored as a
//   float32 BLOB column in an ordinary table (the single source of truth, one
//   atomic row with text and key), and the nearest-neighbour search runs in
//   Pascal over that column. Because the data is a plain table, the search
//   engine stays replaceable (quantised in-memory index, IVF, sqlite-vector)
//   without migrating data - see docs/native-embeddings.md
// - search (stage 2b): every row also stores a 1-bit SIGN code of its vector
//   (dim/8 bytes). Above IndexMinRows rows, a search scans these codes in RAM
//   (Hamming distance, popcount) to pick Oversample candidates, then rescores
//   them EXACTLY with their float32 vectors from SQLite - measured at 2M x 1024
//   (aarch64/WSL, WAL, 2000 candidates): p95 88 ms with one client, 105 ms
//   with four concurrent clients, recall@10 0.97 against the exact scan
//   (docs/native-embeddings.md, demos/rag/vector-bench.lpr). Stores below
//   IndexMinRows use the exact streamed scan (stage 2a)
// - the RAM index follows the table through a generation counter that triggers
//   bump on every change: this store's own writes patch the index in place, a
//   change made by any other connection makes the next search reload it (only
//   the sign codes: 128 bytes per 1024-dim row)
// - every store is bound to a TVectorContract (model, dimension, metric ...),
//   persisted next to its table; opening it with a different contract fails, so
//   vectors of two models can never be mixed silently. Several "spaces" (one per
//   contract or purpose) can live side by side in one database file - that is how
//   a re-index runs in the background while the old space keeps serving
// - metric: cosine. The store normalises every vector to unit length on write
//   and on query, so cosine similarity is the dot product and TRagHit.Distance is
//   the cosine distance 1 - dot
// - CONCURRENCY: the store opens its own connection. When the application writes
//   domain tables of the SAME file through another connection, the file must use
//   WAL (PRAGMA journal_mode=WAL, set by the owner - this unit never changes it):
//   in rollback-journal mode a running scan holds a SHARED lock that makes every
//   application COMMIT fail with SQLITE_BUSY (mORMot sets no busy timeout on the
//   application side), and an application read blocks the store's writes.
//   JournalMode reports the mode found at open
// - clean-room
unit mormot.ai.vectorstore.native;

{
  *****************************************************************************

    - EVectorStore and TVectorContract: what a stored vector means
    - TNativeVectorStore: filterable store over plain SQLite tables, with an
      exact scan and a 1-bit RAM index rescored exactly
    - VectorContract / VectorContractToJson / IsValidVectorSpace: contract helpers

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

{$ifndef FPC}{$ifdef CPUX64}
  {$EXCESSPRECISION OFF} // single expressions in single, as FPC computes them
{$endif}{$endif}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.lib.static,      // SetFpuFlags/ResetFpuFlags
  mormot.db.raw.sqlite3,
  mormot.db.raw.sqlite3.static,
  mormot.ai.llm.types,
  mormot.ai.sqlite,
  mormot.ai.vectorstore;

type
  /// raised on contract violations and invalid input of a native vector store
  EVectorStore = class(ESynException);

  /// what a vector in a store MEANS - two stores (or a store and an embedder)
  // are compatible only if their contracts are equal field by field
  // - metric (cosine) and normalisation (L2) are fixed by this implementation
  // and part of the persisted contract text, so a future change is detected too
  TVectorContract = record
    /// the embedding model identity, e.g. 'BAAI/bge-m3'
    Model: RawUtf8;
    /// SHA-256 of the exact model file, '' when not pinned (e.g. a hosted API)
    ModelSha256: RawUtf8;
    /// the vector dimension
    Dim: integer;
    /// token limit the texts were cut to before embedding (0 = not recorded)
    MaxTokens: integer;
    /// instruction prefixes applied before embedding ('' for bge-m3)
    QueryPrefix: RawUtf8;
    DocumentPrefix: RawUtf8;
  end;

  /// a native, filterable vector store over plain SQLite tables
  // - owns its connection; aDbPath may be the application database itself, so
  //   SearchWhere filters can join domain tables of that file
  // - does NOT change the journal mode of the file - that stays the owner's call
  // - thread-safe: database access holds the connection lock; the RAM index has
  //   its own read/write lock, so concurrent searches scan it in parallel
  TNativeVectorStore = class(TInterfacedObject, IVectorStore, IFilteredVectorStore)
  protected
    fDB: TSqlDataBase;
    fSpace: RawUtf8;
    fItems: RawUtf8;
    fMeta: RawUtf8;
    fContract: TVectorContract;
    fBlobSize: integer;
    fSignWords: integer;
    fJournalMode: RawUtf8;
    fIndex: TObject; // TSignIndex, private to the implementation
    fIndexMinRows: integer;
    fOversample: integer;
    fLastAdded: Int64;
    /// validate (dimension, finite, non-zero) and scale to unit length
    function UnitVector(const aVec: TLlmEmbedding; const aWhat: RawUtf8): TLlmEmbedding;
    function RowIdOfKey(const aId: RawUtf8): Int64;
    procedure OpenSpace;
    /// the key query of SearchWhere must be one read-only, one-column statement
    procedure CheckKeySql(const aKeySql: RawUtf8);
    /// the exact streamed scan (small stores, and the reference for the index)
    function Scan(const aQuery: TLlmEmbedding; aTopK: integer;
      const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
    /// the 1-bit code of a unit vector, as stored in the sign column
    function SignCode(const aQuery: TLlmEmbedding): RawByteString;
    /// current generation (bumped by triggers on every change of the items)
    function ReadGen: Int64;
    /// (re)load the RAM index when its generation is not the table's
    procedure EnsureIndex;
    /// patch the RAM index after a committed write of this store
    // - aGenBefore/aGenAfter: generation read inside the write transaction
    //   before/after the statements; if the index was not at aGenBefore (a
    //   foreign write slipped in), it is marked stale instead of patched
    procedure IndexApply(aGenBefore, aGenAfter: Int64;
      const aRowIds: TInt64DynArray; const aCodes: TRawByteStringDynArray;
      aDeleted: boolean);
    /// row ids whose external key is returned by a key query, ascending
    function KeyRowIds(const aKeySql: RawUtf8;
      const aParams: array of const): TInt64DynArray;
    /// the index-backed search: Hamming candidates, exact float32 rescoring
    function IndexSearch(const aQuery: TLlmEmbedding; aTopK: integer;
      const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
    /// text + key of the winners, inside the caller's read snapshot
    function FetchHits(const aRowIds: array of Int64;
      const aScores: array of double; aCount: PtrInt): TRagHitDynArray;
    /// shared entry of Search and SearchWhere
    function DoSearch(const aQuery: TLlmEmbedding; aTopK: integer;
      const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
  public
    /// open or create the vector space aSpace in aDbPath (':memory:' allowed)
    // - aSpace: 1..32 chars [a-z0-9_], starting with a letter; it prefixes the
    //   two tables <space>_items and <space>_meta
    // - raises EVectorStore if the space exists with a different contract
    constructor Create(const aDbPath, aSpace: RawUtf8;
      const aContract: TVectorContract); reintroduce;
    destructor Destroy; override;
    // IVectorStore
    function Add(const aText: RawUtf8; const aVector: TLlmEmbedding): Int64;
    function AddBatch(const aTexts: TRawUtf8DynArray;
      const aVectors: TLlmEmbeddingDynArray): integer;
    procedure Upsert(const aId, aText: RawUtf8; const aVector: TLlmEmbedding);
    procedure Delete(const aId: RawUtf8);
    function Search(const aQuery: TLlmEmbedding; aTopK: integer): TRagHitDynArray;
    function Count: Int64;
    // IFilteredVectorStore
    function SearchWhere(const aQuery: TLlmEmbedding; aTopK: integer;
      const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
    /// the contract this store was opened with (= the persisted one)
    property Contract: TVectorContract read fContract;
    /// the space name (table prefix)
    property Space: RawUtf8 read fSpace;
    /// the journal mode of the file at open, lower-case ('wal', 'delete',
    // 'memory' ...) - anything but 'wal' (or 'memory') is unsafe when another
    // connection writes the same file concurrently, see the unit header
    property JournalMode: RawUtf8 read fJournalMode;
    /// below this many rows the exact streamed scan is used (default 20000)
    // - 0 uses the index whenever it can help, i.e. unless aTopK >= rows (then
    //   the exact scan returns every row anyway) - used by the tests
    property IndexMinRows: integer read fIndexMinRows write fIndexMinRows;
    /// candidates taken from the 1-bit index and rescored exactly (default 2000)
    // - higher = better recall, slower; at least aTopK is always used
    // - capped at NATIVE_MAX_OVERSAMPLE (100000) per search: the candidates
    //   travel as one JSON array into the rescoring statement
    // - measured at 2M x 1024 (gate 5): 1000 -> recall@10 0.94, 2000 -> 0.97,
    //   4000 -> 0.99 (docs/native-embeddings.md)
    property Oversample: integer read fOversample write fOversample;
    /// rows currently held by the RAM index (0 before the first search)
    function IndexedRows: integer;
    /// load the RAM index now instead of on the first search
    // - call it at server start: loading takes seconds for millions of rows
    //   (2.8-11 s for 2M x 1024 measured, depending on the file cache) and
    //   blocks the store meanwhile
    procedure WarmUp;
  end;

/// build a contract value
function VectorContract(const aModel, aModelSha256: RawUtf8; aDim: integer;
  aMaxTokens: integer = 0; const aQueryPrefix: RawUtf8 = '';
  const aDocumentPrefix: RawUtf8 = ''): TVectorContract;

/// the canonical persisted form of a contract (stable field order)
function VectorContractToJson(const aContract: TVectorContract): RawUtf8;

/// true if aSpace is a valid space name (1..32 of [a-z0-9_], first a letter)
function IsValidVectorSpace(const aSpace: RawUtf8): boolean;


implementation

const
  /// version of the on-disk layout and of the contract text
  // - 2: sign column + generation triggers (stage 2b)
  NATIVE_FORMAT = 2;
  /// largest accepted dimension - far above any real embedding model, but keeps
  // the BLOB size check and the per-row arithmetic away from overflow
  NATIVE_MAX_DIM = 65536;
  /// upper bound of the Oversample setting actually used by one search
  NATIVE_MAX_OVERSAMPLE = 100000;
  /// IEEE-754 single: all exponent bits set = NaN or +-Inf
  SINGLE_EXP_MASK = $7F800000;
  /// IEEE-754 double: the same test for a computed score
  DOUBLE_EXP_MASK = Int64($7FF0000000000000);

function VectorContract(const aModel, aModelSha256: RawUtf8; aDim: integer;
  aMaxTokens: integer; const aQueryPrefix, aDocumentPrefix: RawUtf8): TVectorContract;
begin
  result.Model := aModel;
  result.ModelSha256 := LowerCase(aModelSha256);
  result.Dim := aDim;
  result.MaxTokens := aMaxTokens;
  result.QueryPrefix := aQueryPrefix;
  result.DocumentPrefix := aDocumentPrefix;
end;

function VectorContractToJson(const aContract: TVectorContract): RawUtf8;
var
  doc: TDocVariantData;
begin
  doc.InitFast;
  doc.AddValue('format', NATIVE_FORMAT);
  // AddValueText, never AddValueFromText: the latter parses '1' or 'true' into a
  // number/boolean, so two different prefixes could serialise identically
  doc.AddValueText('model', aContract.Model);
  doc.AddValueText('model_sha256', LowerCase(aContract.ModelSha256));
  doc.AddValue('dim', aContract.Dim);
  doc.AddValue('metric', 'cosine');
  doc.AddValue('normalize', 'l2');
  doc.AddValue('max_tokens', aContract.MaxTokens);
  doc.AddValueText('query_prefix', aContract.QueryPrefix);
  doc.AddValueText('document_prefix', aContract.DocumentPrefix);
  result := doc.ToJson;
end;

function IsValidVectorSpace(const aSpace: RawUtf8): boolean;
var
  i: PtrInt;
begin
  result := false;
  if (aSpace = '') or (length(aSpace) > 32) or
     not (aSpace[1] in ['a'..'z']) then
    exit;
  for i := 2 to length(aSpace) do
    if not (aSpace[i] in ['a'..'z', '0'..'9', '_']) then
      exit;
  result := true;
end;

/// cosine similarity of a unit query against a stored unit float32 row
function DotRow(q: PSingleArray; row: PSingleArray; n: PtrInt): double;
var
  i: PtrInt;
begin
  result := 0;
  for i := 0 to n - 1 do
    result := result + q^[i] * row^[i];
end;


{ top-k collector: kept sorted best-first, k is small (tens), so insertion
  into a fixed array beats any heap here }

type
  // object, not record, where the compiler has no methods on records (Delphi 7)
  TTopK = {$ifdef USERECORDWITHMETHODS}record{$else}object{$endif}
    Score: array of double;
    RowId: array of Int64;
    Count, Capacity: PtrInt;
    procedure Init(aCapacity: PtrInt);
    /// a ranks before b: higher score, then lower rowid (deterministic ties)
    function Before(aScore: double; aRowId: Int64; aIndex: PtrInt): boolean;
    procedure Offer(aScore: double; aRowId: Int64);
  end;

procedure TTopK.Init(aCapacity: PtrInt);
begin
  Capacity := aCapacity;
  Count := 0;
  // grow on demand: a caller asking for k = MaxInt must not allocate 16 GB
  // up front when the store holds a few hundred rows
  SetLength(Score, MinPtrInt(aCapacity, 256));
  SetLength(RowId, length(Score));
end;

function TTopK.Before(aScore: double; aRowId: Int64; aIndex: PtrInt): boolean;
begin
  result := (aScore > Score[aIndex]) or
            ((aScore = Score[aIndex]) and (aRowId < RowId[aIndex]));
end;

procedure TTopK.Offer(aScore: double; aRowId: Int64);
var
  i: PtrInt;
begin
  if Count = Capacity then
    if not Before(aScore, aRowId, Count - 1) then
      exit // not better than the current worst
    else
      dec(Count); // drop the worst, make room
  if Count = length(Score) then
  begin
    SetLength(Score, MinPtrInt(Capacity, length(Score) * 2));
    SetLength(RowId, length(Score));
  end;
  i := Count;
  while (i > 0) and Before(aScore, aRowId, i - 1) do
  begin
    Score[i] := Score[i - 1];
    RowId[i] := RowId[i - 1];
    dec(i);
  end;
  Score[i] := aScore;
  RowId[i] := aRowId;
  inc(Count);
end;


{ TSignIndex: the in-memory 1-bit index of one store }

type
  TQWordDynArray = array of QWord;

  /// sign codes of all rows, slot = position in rowid order
  // - rowids are AUTOINCREMENT, so appending keeps RowIds sorted and a slot is
  //   found by binary search; deleted rows become dead slots until Compact
  // - guarded by Lock: searches hold ReadLock while scanning, writes hold
  //   WriteLock while patching - never WriteLock inside a ReadLock
  // - TRWLightLock spins (with SwitchToThread) while it waits: a write waits
  //   for running scans (tens of ms at millions of rows). Acceptable for the
  //   expected write rate (embeddings arrive at a few per second); a lock-free
  //   scan would need copy-on-write arrays instead of in-place patching
  TSignIndex = class
  public
    Lock: TRWLightLock;
    Words: integer;
    RowIds: TInt64DynArray;
    Alive: TBooleanDynArray;
    Codes: TQWordDynArray;
    Count, AliveCount: integer;
    Gen: Int64;
    Loaded: boolean;
    constructor Create(aWords: integer);
    procedure Clear;
    function SlotOf(aRowId: Int64): integer;
    /// append a row; false if aRowId is not above the last one (= reload)
    function Append(aRowId: Int64; aCode: pointer): boolean;
    procedure SetCode(aSlot: integer; aCode: pointer);
    procedure Kill(aSlot: integer);
    /// drop dead slots once they are a quarter of the index
    procedure CompactIfNeeded;
    /// the aWant best slots by Hamming distance to aQuery (ties: lower slot),
    // restricted to aMask when given; returns their rowids in slot order
    function Candidates(aQuery: PQWordArray; aWant: integer;
      const aMask: TBooleanDynArray): TInt64DynArray;
  end;

constructor TSignIndex.Create(aWords: integer);
begin
  inherited Create;
  Lock.Init;
  Words := aWords;
end;

procedure TSignIndex.Clear;
begin
  RowIds := nil;
  Alive := nil;
  Codes := nil;
  Count := 0;
  AliveCount := 0;
  Loaded := false;
end;

function TSignIndex.SlotOf(aRowId: Int64): integer;
var
  lo, hi, mid: integer;
begin
  lo := 0;
  hi := Count - 1;
  while lo <= hi do
  begin
    mid := (lo + hi) shr 1;
    if RowIds[mid] < aRowId then
      lo := mid + 1
    else if RowIds[mid] > aRowId then
      hi := mid - 1
    else
    begin
      result := mid;
      exit;
    end;
  end;
  result := -1;
end;

function TSignIndex.Append(aRowId: Int64; aCode: pointer): boolean;
begin
  result := (Count = 0) or (aRowId > RowIds[Count - 1]);
  if not result then
    exit;
  if Count = length(RowIds) then
  begin
    SetLength(RowIds, NextGrow(Count));
    SetLength(Alive, length(RowIds));
    SetLength(Codes, length(RowIds) * Words);
  end;
  RowIds[Count] := aRowId;
  Alive[Count] := true;
  MoveFast(aCode^, Codes[Count * Words], Words * SizeOf(QWord));
  inc(Count);
  inc(AliveCount);
end;

procedure TSignIndex.SetCode(aSlot: integer; aCode: pointer);
begin
  MoveFast(aCode^, Codes[aSlot * Words], Words * SizeOf(QWord));
end;

procedure TSignIndex.Kill(aSlot: integer);
begin
  if Alive[aSlot] then
  begin
    Alive[aSlot] := false;
    dec(AliveCount);
  end;
end;

procedure TSignIndex.CompactIfNeeded;
var
  i, n: integer;
begin
  if (Count - AliveCount) * 4 < Count then
    exit;
  n := 0;
  for i := 0 to Count - 1 do
    if Alive[i] then
    begin
      if n <> i then
      begin
        RowIds[n] := RowIds[i];
        Alive[n] := true;
        MoveFast(Codes[i * Words], Codes[n * Words], Words * SizeOf(QWord));
      end;
      inc(n);
    end;
  Count := n;
end;

function PopCount64(v: QWord): integer;
  {$ifdef HASINLINE} inline; {$endif}
begin
  // mORMot's popcount: SSE4.2 POPCNT on x86_64, its fast pure-pascal version
  // elsewhere (measured 3x faster than FPC's PopCnt() on aarch64)
  {$ifdef CPU64}
  result := GetBitsCountPtrInt(PtrInt(v));
  {$else}
  result := GetBitsCountPtrInt(PtrInt(cardinal(v))) +
            GetBitsCountPtrInt(PtrInt(cardinal(v shr 32)));
  {$endif CPU64}
end;

function TSignIndex.Candidates(aQuery: PQWordArray; aWant: integer;
  const aMask: TBooleanDynArray): TInt64DynArray;
var
  dist, hist: TIntegerDynArray;
  i, w, d, bits, valid, threshold, atThreshold, n, sum: integer;
  code: PQWordArray;
begin
  result := nil;
  if (Count = 0) or (aWant <= 0) then
    exit;
  bits := Words * 64;
  // pass 1: distance of every live (and allowed) slot + its histogram; -1
  // marks a skipped slot - integers, so neither the marker nor a distance of
  // 65536 (possible with padding) can collide with a real value
  SetLength(dist, Count);
  SetLength(hist, bits + 1);
  valid := 0;
  for i := 0 to Count - 1 do
  begin
    if not Alive[i] or ((aMask <> nil) and not aMask[i]) then
    begin
      dist[i] := -1;
      continue;
    end;
    code := @Codes[i * Words];
    d := 0;
    for w := 0 to Words - 1 do
      inc(d, PopCount64(code^[w] xor aQuery^[w]));
    dist[i] := d;
    inc(hist[d]);
    inc(valid);
  end;
  if valid = 0 then
    exit;
  if aWant >= valid then
  begin
    // fewer valid slots than wanted: every valid slot is a candidate
    threshold := bits;
    atThreshold := MaxInt;
  end
  else
  begin
    // the threshold distance: everything below it is taken, at it only as
    // many as still fit (in slot = rowid order, so ties are deterministic) -
    // a counting sort instead of keeping a sorted candidate list
    sum := 0;
    threshold := bits;
    for d := 0 to bits do
    begin
      if sum + hist[d] >= aWant then
      begin
        threshold := d;
        break;
      end;
      inc(sum, hist[d]);
    end;
    atThreshold := aWant - sum;
  end;
  SetLength(result, MinPtrInt(aWant, valid));
  n := 0;
  for i := 0 to Count - 1 do
  begin
    d := dist[i];
    if (d < 0) or (d > threshold) then
      continue; // skipped slot, or beyond the threshold
    if d = threshold then
      if atThreshold > 0 then
        dec(atThreshold)
      else
        continue;
    // invariant: at most MinPtrInt(aWant, valid) slots pass the tests above -
    // checked anyway, a buffer overrun must never depend on that arithmetic
    if n = length(result) then
      break;
    result[n] := RowIds[i];
    inc(n);
  end;
  SetLength(result, n);
end;


{ TNativeVectorStore }

constructor TNativeVectorStore.Create(const aDbPath, aSpace: RawUtf8;
  const aContract: TVectorContract);
begin
  inherited Create;
  if not IsValidVectorSpace(aSpace) then
    EVectorStore.RaiseUtf8('%: invalid space name "%" (1..32 of [a-z0-9_], ' +
      'starting with a letter)', [self, aSpace]);
  if (aContract.Model = '') or (aContract.Dim <= 0) or
     (aContract.Dim > NATIVE_MAX_DIM) then
    EVectorStore.RaiseUtf8('%: contract needs a model and a dimension in 1..%',
      [self, NATIVE_MAX_DIM]);
  fSpace := aSpace;
  fItems := aSpace + '_items';
  fMeta := aSpace + '_meta';
  fContract := aContract;
  // a record assigned directly bypasses VectorContract(): normalise here too
  fContract.ModelSha256 := LowerCase(aContract.ModelSha256);
  fBlobSize := aContract.Dim * SizeOf(single);
  fSignWords := (aContract.Dim + 63) shr 6;
  fIndex := TSignIndex.Create(fSignWords);
  fIndexMinRows := 20000;
  fOversample := 2000;
  fDB := TSqlDataBase.Create(Utf8ToString(aDbPath), '');
  // mORMot sets no busy timeout: a lock held by another connection (the
  // application writing its domain tables) would fail at once with SQLITE_BUSY
  fDB.BusyTimeout := 5000;
  fJournalMode := LowerCase(fDB.ExecuteNoExceptionUtf8('PRAGMA journal_mode;'));
  OpenSpace;
end;

destructor TNativeVectorStore.Destroy;
begin
  fDB.Free;
  fIndex.Free;
  inherited Destroy;
end;

procedure TNativeVectorStore.WarmUp;
begin
  EnsureIndex;
end;

function TNativeVectorStore.IndexedRows: integer;
begin
  TSignIndex(fIndex).Lock.ReadLock;
  try
    result := TSignIndex(fIndex).AliveCount;
  finally
    TSignIndex(fIndex).Lock.ReadUnLock;
  end;
end;

procedure TNativeVectorStore.OpenSpace;
var
  r: TSqlRequest;
  stored, expected: RawUtf8;
  hasItems: boolean;
begin
  expected := VectorContractToJson(fContract);
  fDB.Lock;
  try
    // IMMEDIATE: take the write lock up front - a deferred transaction that
    // reads first and upgrades later can fail with SQLITE_BUSY at once (SQLite
    // skips the busy handler to avoid the upgrade deadlock), and two processes
    // creating the same space must not both see "no contract yet"
    fDB.TransactionBegin(tbImmediate);
    try
      fDB.Execute(FormatUtf8('CREATE TABLE IF NOT EXISTS %(' +
        'key TEXT PRIMARY KEY, value TEXT NOT NULL);', [fMeta]));
      stored := '';
      r.Prepare(fDB.DB, FormatUtf8('SELECT value FROM % WHERE key = ''contract'';',
        [fMeta]));
      try
        if r.Step = SQLITE_ROW then
          r.FieldUtf8(0, stored);
      finally
        r.Close;
      end;
      if stored = '' then
      begin
        // a contract-less items table means someone created the space by hand
        // (or a half-written older layout): never adopt vectors of unknown meaning
        r.Prepare(fDB.DB,
          'SELECT 1 FROM sqlite_master WHERE type = ''table'' AND name = ?;');
        try
          r.Bind(1, fItems);
          hasItems := r.Step = SQLITE_ROW;
        finally
          r.Close;
        end;
        if hasItems then
          EVectorStore.RaiseUtf8('%: table % exists without a contract - refusing ' +
            'to adopt vectors of unknown meaning', [self, fItems]);
        r.Prepare(fDB.DB, FormatUtf8(
          'INSERT INTO %(key, value) VALUES (''contract'', ?);', [fMeta]));
        try
          r.Bind(1, expected);
          r.Step;
        finally
          r.Close;
        end;
      end
      else if stored <> expected then
        EVectorStore.RaiseUtf8('%: space "%" was built under another contract ' +
          '(stored %, requested %) - re-index into a new space', [self, fSpace,
          stored, expected]);
      // ext_id: optional stable external key (Upsert/Delete rows), NULL for
      // anonymous Add rows; the partial UNIQUE index maps an id to at most one row
      // the CHECK makes a wrong-sized vector impossible at the storage level too
      // sign: the 1-bit code of the vector (bit i = component i > 0), stored so
      // the RAM index loads dim/8 bytes per row instead of the float32 vector
      fDB.Execute(FormatUtf8('CREATE TABLE IF NOT EXISTS %(' +
        'id INTEGER PRIMARY KEY AUTOINCREMENT, ext_id TEXT, ' +
        'content TEXT NOT NULL, embedding BLOB NOT NULL ' +
        'CHECK (typeof(embedding) = ''blob'' AND length(embedding) = %), ' +
        'sign BLOB NOT NULL ' +
        'CHECK (typeof(sign) = ''blob'' AND length(sign) = %));',
        [fItems, fBlobSize, fSignWords * 8]));
      fDB.Execute(FormatUtf8('CREATE UNIQUE INDEX IF NOT EXISTS %_ext_id ' +
        'ON %(ext_id) WHERE ext_id IS NOT NULL;', [fItems, fItems]));
      // generation: bumped by trigger on EVERY change of the items, whoever
      // writes - the RAM index compares it to know whether it is current
      fDB.Execute(FormatUtf8('INSERT OR IGNORE INTO %(key, value) ' +
        'VALUES (''gen'', ''0'');', [fMeta]));
      fDB.Execute(FormatUtf8('CREATE TRIGGER IF NOT EXISTS %_gen_ins AFTER INSERT ' +
        'ON % BEGIN UPDATE % SET value = CAST(value AS INTEGER) + 1 ' +
        'WHERE key = ''gen''; END;', [fItems, fItems, fMeta]));
      fDB.Execute(FormatUtf8('CREATE TRIGGER IF NOT EXISTS %_gen_upd AFTER UPDATE ' +
        'ON % BEGIN UPDATE % SET value = CAST(value AS INTEGER) + 1 ' +
        'WHERE key = ''gen''; END;', [fItems, fItems, fMeta]));
      fDB.Execute(FormatUtf8('CREATE TRIGGER IF NOT EXISTS %_gen_del AFTER DELETE ' +
        'ON % BEGIN UPDATE % SET value = CAST(value AS INTEGER) + 1 ' +
        'WHERE key = ''gen''; END;', [fItems, fItems, fMeta]));
      fDB.Commit;
    except
      RollbackIfOpen(fDB);
      raise;
    end;
  finally
    fDB.UnLock;
  end;
end;

function TNativeVectorStore.UnitVector(const aVec: TLlmEmbedding;
  const aWhat: RawUtf8): TLlmEmbedding;
var
  i: PtrInt;
  sum, scale, x: double;
begin
  if length(aVec) <> fContract.Dim then
    EVectorStore.RaiseUtf8('%: % has % dims, contract says % ' +
      '(empty = failed embedding?)', [self, aWhat, length(aVec), fContract.Dim]);
  sum := 0;
  for i := 0 to high(aVec) do
  begin
    // test the bits, not the value: with FPU exceptions unmasked (the default
    // here) even comparing a NaN raises EInvalidOp instead of our error
    if PCardinal(@aVec[i])^ and SINGLE_EXP_MASK = SINGLE_EXP_MASK then
      EVectorStore.RaiseUtf8('%: % has a non-finite component at %',
        [self, aWhat, i]);
    // widen BEFORE squaring: FPC multiplies two singles in single precision,
    // so 1e20 would overflow and 1e-30 underflow to a false "zero vector"
    x := aVec[i];
    sum := sum + x * x;
  end;
  if sum <= 0 then
    EVectorStore.RaiseUtf8('%: % is the zero vector (no direction)', [self, aWhat]);
  scale := 1 / sqrt(sum);
  SetLength(result, length(aVec));
  for i := 0 to high(aVec) do
    result[i] := aVec[i] * scale;
end;

function TNativeVectorStore.SignCode(const aQuery: TLlmEmbedding): RawByteString;
var
  i: PtrInt;
  p: PQWordArray;
begin
  result := '';
  SetLength(result, fSignWords * 8);
  p := pointer(result);
  FillCharFast(p^, fSignWords * 8, 0);
  // bit test, not "> 0": no float compare (NaN-safe), and -0.0 counts as 0
  for i := 0 to high(aQuery) do
    if (PCardinal(@aQuery[i])^ and $80000000 = 0) and
       (PCardinal(@aQuery[i])^ <> 0) then
      p^[i shr 6] := p^[i shr 6] or (QWord(1) shl (i and 63));
end;

function TNativeVectorStore.ReadGen: Int64;
var
  r: TSqlRequest;
  found: boolean;
begin
  result := 0;
  found := false;
  fDB.Lock; // reentrant: callers inside a transaction already hold it
  try
    r.Prepare(fDB.DB, FormatUtf8('SELECT CAST(value AS INTEGER) FROM % ' +
      'WHERE key = ''gen'';', [fMeta]));
    try
      if r.Step = SQLITE_ROW then
      begin
        result := r.FieldInt(0);
        found := true;
      end;
    finally
      r.Close;
    end;
  finally
    fDB.UnLock;
  end;
  // without the row the triggers bump nothing: the index could never notice a
  // foreign change again - refuse instead of trusting a constant generation
  if not found then
    EVectorStore.RaiseUtf8('%: generation row missing in % - the store ' +
      'metadata was damaged', [self, fMeta]);
end;

function TNativeVectorStore.RowIdOfKey(const aId: RawUtf8): Int64;
var
  r: TSqlRequest;
begin
  result := 0; // 0 = not found (AUTOINCREMENT ids start at 1)
  fDB.Lock; // reentrant: Upsert/Delete already hold it
  try
    r.Prepare(fDB.DB, FormatUtf8('SELECT id FROM % WHERE ext_id = ?;', [fItems]));
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

function TNativeVectorStore.Add(const aText: RawUtf8;
  const aVector: TLlmEmbedding): Int64;
var
  texts: TRawUtf8DynArray;
  vecs: TLlmEmbeddingDynArray;
begin
  // one code path for all inserts: the index delta and the generation
  // bookkeeping live in AddBatch
  SetLength(texts, 1);
  SetLength(vecs, 1);
  texts[0] := aText;
  vecs[0] := aVector;
  fDB.Lock; // keep fLastAdded ours until it is read
  try
    AddBatch(texts, vecs);
    result := fLastAdded;
  finally
    fDB.UnLock;
  end;
end;

function TNativeVectorStore.AddBatch(const aTexts: TRawUtf8DynArray;
  const aVectors: TLlmEmbeddingDynArray): integer;
var
  i: PtrInt;
  r: TSqlRequest;
  blobs, signs: TRawByteStringDynArray;
  rowids: TInt64DynArray;
  u: TLlmEmbedding;
  genBefore, genAfter: Int64;
begin
  result := 0;
  if length(aTexts) <> length(aVectors) then
    EVectorStore.RaiseUtf8('%.AddBatch: % texts but % vectors',
      [self, length(aTexts), length(aVectors)]);
  if aTexts = nil then
    exit;
  // validate + normalise EVERYTHING before the transaction opens: one bad vector
  // fails the whole batch without writing anything
  SetLength(blobs, length(aVectors));
  SetLength(signs, length(aVectors));
  SetLength(rowids, length(aVectors));
  for i := 0 to high(aVectors) do
  begin
    u := UnitVector(aVectors[i], FormatUtf8('vector #%', [i]));
    blobs[i] := VectorToBlob(u);
    signs[i] := SignCode(u);
  end;
  fDB.Lock;
  try
    fDB.TransactionBegin(tbImmediate); // see OpenSpace
    try
      genBefore := ReadGen;
      r.Prepare(fDB.DB, FormatUtf8(
        'INSERT INTO %(content, embedding, sign) VALUES (?, ?, ?);', [fItems]));
      try
        for i := 0 to high(aTexts) do
        begin
          r.Bind(1, aTexts[i]);
          r.BindBlob(2, blobs[i]);
          r.BindBlob(3, signs[i]);
          r.Step;
          r.Reset;
          rowids[i] := fDB.LastInsertRowID; // per connection: under our lock
        end;
      finally
        r.Close;
      end;
      genAfter := ReadGen;
      fDB.Commit;
      result := length(aTexts);
      fLastAdded := rowids[high(rowids)];
    except
      RollbackIfOpen(fDB);
      raise;
    end;
    IndexApply(genBefore, genAfter, rowids, signs, false);
  finally
    fDB.UnLock;
  end;
end;

procedure TNativeVectorStore.Upsert(const aId, aText: RawUtf8;
  const aVector: TLlmEmbedding);
var
  r: TSqlRequest;
  u: TLlmEmbedding;
  blob: RawByteString;
  rowids: TInt64DynArray;
  signs: TRawByteStringDynArray;
  genBefore, genAfter: Int64;
begin
  if aId = '' then
    EVectorStore.RaiseUtf8('%.Upsert: empty id', [self]);
  u := UnitVector(aVector, 'vector');
  blob := VectorToBlob(u);
  SetLength(signs, 1);
  signs[0] := SignCode(u);
  SetLength(rowids, 1);
  fDB.Lock;
  try
    // IMMEDIATE: RowIdOfKey reads before the write - see OpenSpace
    fDB.TransactionBegin(tbImmediate);
    try
      genBefore := ReadGen;
      rowids[0] := RowIdOfKey(aId);
      if rowids[0] <> 0 then
      begin
        // in place: one row holds text and vector, so the replace is atomic and
        // the id -> row mapping stays stable
        r.Prepare(fDB.DB, FormatUtf8(
          'UPDATE % SET content = ?, embedding = ?, sign = ? WHERE id = ?;',
          [fItems]));
        try
          r.Bind(1, aText);
          r.BindBlob(2, blob);
          r.BindBlob(3, signs[0]);
          r.Bind(4, rowids[0]);
          r.Step;
        finally
          r.Close;
        end;
      end
      else
      begin
        r.Prepare(fDB.DB, FormatUtf8(
          'INSERT INTO %(ext_id, content, embedding, sign) VALUES (?, ?, ?, ?);',
          [fItems]));
        try
          r.Bind(1, aId);
          r.Bind(2, aText);
          r.BindBlob(3, blob);
          r.BindBlob(4, signs[0]);
          r.Step;
        finally
          r.Close;
        end;
        rowids[0] := fDB.LastInsertRowID;
      end;
      genAfter := ReadGen;
      fDB.Commit;
    except
      RollbackIfOpen(fDB);
      raise;
    end;
    IndexApply(genBefore, genAfter, rowids, signs, false);
  finally
    fDB.UnLock;
  end;
end;

procedure TNativeVectorStore.Delete(const aId: RawUtf8);
var
  r: TSqlRequest;
  rowids: TInt64DynArray;
  genBefore, genAfter: Int64;
begin
  SetLength(rowids, 1);
  fDB.Lock;
  try
    fDB.TransactionBegin(tbImmediate); // see OpenSpace
    try
      genBefore := ReadGen;
      rowids[0] := RowIdOfKey(aId);
      if rowids[0] <> 0 then // absent id = no-op
      begin
        r.Prepare(fDB.DB, FormatUtf8('DELETE FROM % WHERE id = ?;', [fItems]));
        try
          r.Bind(1, rowids[0]);
          r.Step;
        finally
          r.Close;
        end;
      end;
      genAfter := ReadGen;
      fDB.Commit;
    except
      RollbackIfOpen(fDB);
      raise;
    end;
    if rowids[0] <> 0 then
      IndexApply(genBefore, genAfter, rowids, nil, true);
  finally
    fDB.UnLock;
  end;
end;

procedure TNativeVectorStore.IndexApply(aGenBefore, aGenAfter: Int64;
  const aRowIds: TInt64DynArray; const aCodes: TRawByteStringDynArray;
  aDeleted: boolean);
var
  idx: TSignIndex;
  i, slot: integer;
begin
  idx := TSignIndex(fIndex);
  // called under fDB.Lock (lock order: database, then index - never reversed)
  idx.Lock.WriteLock;
  try
    if not idx.Loaded then
      exit; // nothing loaded yet: the first search loads the current state
    if idx.Gen <> aGenBefore then
    begin
      // another connection changed the table in between: patching would mix
      // states - let the next search reload instead
      idx.Loaded := false;
      exit;
    end;
    for i := 0 to high(aRowIds) do
    begin
      slot := idx.SlotOf(aRowIds[i]);
      if aDeleted then
      begin
        if slot >= 0 then
          idx.Kill(slot);
      end
      else if slot >= 0 then
        idx.SetCode(slot, pointer(aCodes[i]))
      else if not idx.Append(aRowIds[i], pointer(aCodes[i])) then
      begin
        idx.Loaded := false; // out-of-order rowid: cannot happen with
        exit;                // AUTOINCREMENT, but never guess - reload
      end;
    end;
    idx.CompactIfNeeded;
    idx.Gen := aGenAfter;
  finally
    idx.Lock.WriteUnLock;
  end;
end;

function TNativeVectorStore.Count: Int64;
var
  r: TSqlRequest;
begin
  result := 0;
  fDB.Lock;
  try
    r.Prepare(fDB.DB, FormatUtf8('SELECT COUNT(*) FROM %;', [fItems]));
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

const
  /// authorizer action code for a recursive CTE (not declared by mORMot)
  SQLITE_RECURSIVE_ = 33;

/// authorizer active while a key query is COMPILED: only reading is allowed
// - some statements act at prepare time already (e.g. PRAGMA busy_timeout=0
//   changes the connection while being prepared), so they must be denied
//   during compilation - checks on the compiled statement would come too late
function KeySqlAuthorizer(pUserData: pointer; code: integer;
  const zTab, zCol, zDb, zAuthContext: PUtf8Char): integer; cdecl;
begin
  case code of
    SQLITE_SELECT, SQLITE_READ, SQLITE_FUNCTION, SQLITE_RECURSIVE_:
      result := SQLITE_OK;
    SQLITE_PRAGMA:
      // FTS5 runs "PRAGMA data_version" (no argument = read only) internally
      // on every MATCH; allowing exactly that keeps full-text key queries -
      // the hybrid FTS5 + vector search - possible. Any other pragma, or
      // data_version with an argument, stays denied
      if (zCol = nil) and (zTab <> nil) and
         (StrIComp(zTab, PAnsiChar('data_version')) = 0) then // no allocation
        result := SQLITE_OK
      else
        result := SQLITE_DENY;
  else
    result := SQLITE_DENY; // PRAGMA, ATTACH, writes, transactions, DDL ...
  end;
end;

procedure TNativeVectorStore.CheckKeySql(const aKeySql: RawUtf8);
var
  stmt: TSqlite3Statement;
  tail: PUtf8Char;
  rc, cols: integer;
  ro: boolean;
  wrapped: RawUtf8;
begin
  if TrimU(aKeySql) = '' then
    EVectorStore.RaiseUtf8('%.SearchWhere: empty key query - use Search', [self]);
  // a NUL ends the text for SQLite and hides what follows from the tail check;
  // a final ';' is consumed by the first prepare but breaks the IN (...) wrap
  if PosExChar(#0, aKeySql) > 0 then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query contains a NUL byte', [self]);
  if TrimU(aKeySql)[length(TrimU(aKeySql))] = ';' then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query must not end with ";"', [self]);
  // compile the key query ON ITS OWN first: it must be exactly one complete,
  // read-only statement with one result column. Only then is it embedded in
  // IN (...) - this rejects a text that closes the parenthesis itself
  // ("SELECT 'a') OR 1=1 --": a syntax error standalone) and a second
  // statement after a ';' (which TSqlRequest.Prepare would silently ignore)
  stmt := 0;
  tail := nil;
  sqlite3.set_authorizer(fDB.DB, KeySqlAuthorizer, nil);
  try
    rc := sqlite3.prepare_v2(fDB.DB, pointer(aKeySql), length(aKeySql), stmt, tail);
  finally
    sqlite3.set_authorizer(fDB.DB, nil, nil);
  end;
  try
    if (rc <> SQLITE_OK) or (stmt = 0) then
      EVectorStore.RaiseUtf8('%.SearchWhere: key query does not compile: %',
        [self, sqlite3.errmsg(fDB.DB)]);
    cols := sqlite3.column_count(stmt);
    ro := sqlite3.stmt_readonly(stmt) <> 0;
  finally
    if stmt <> 0 then
      sqlite3.finalize(stmt);
  end;
  if (tail <> nil) and (TrimU(RawUtf8(tail)) <> '') then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query must be ONE statement', [self]);
  if not ro then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query is not read-only', [self]);
  if cols <> 1 then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query returns % columns, ' +
      'expected exactly 1 (the external key)', [self, cols]);
  // finally compile it exactly as Scan will embed it: a text that is valid on
  // its own can still break the wrap (e.g. an unterminated /* comment) - report
  // that here as EVectorStore instead of a raw SQLite error in the scan
  stmt := 0;
  sqlite3.set_authorizer(fDB.DB, KeySqlAuthorizer, nil);
  try
    wrapped := FormatUtf8('SELECT 1 WHERE NULL IN (%' + #10 + ')', [aKeySql]);
    rc := sqlite3.prepare_v2(fDB.DB, pointer(wrapped), length(wrapped), stmt, tail);
    if stmt <> 0 then
      sqlite3.finalize(stmt);
  finally
    sqlite3.set_authorizer(fDB.DB, nil, nil);
  end;
  if rc <> SQLITE_OK then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query cannot be used as a ' +
      'sub-select: %', [self, sqlite3.errmsg(fDB.DB)]);
end;

function TNativeVectorStore.SearchWhere(const aQuery: TLlmEmbedding;
  aTopK: integer; const aKeySql: RawUtf8;
  const aParams: array of const): TRagHitDynArray;
begin
  // the connection lock only for the validation: the search itself takes it
  // where it needs the database, so filtered searches also scan in parallel
  fDB.Lock;
  try
    CheckKeySql(aKeySql);
  finally
    fDB.UnLock;
  end;
  result := DoSearch(aQuery, aTopK, aKeySql, aParams);
end;

function TNativeVectorStore.Search(const aQuery: TLlmEmbedding;
  aTopK: integer): TRagHitDynArray;
begin
  result := DoSearch(aQuery, aTopK, '', []);
end;

function TNativeVectorStore.DoSearch(const aQuery: TLlmEmbedding; aTopK: integer;
  const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
var
  q: TLlmEmbedding;
  rows: integer;
begin
  result := nil;
  q := UnitVector(aQuery, 'query');
  // non-positive TopK = no results requested (same contract as TVec0Store)
  if aTopK <= 0 then
    exit;
  EnsureIndex;
  rows := IndexedRows;
  // the exact streamed scan when the index cannot help: a small store, or a
  // request for every row (aTopK >= rows) - fetching all rows through the
  // candidate list would only add work to the plain stream
  if (rows < fIndexMinRows) or (aTopK >= rows) then
    result := Scan(q, aTopK, aKeySql, aParams)
  else
    result := IndexSearch(q, aTopK, aKeySql, aParams);
end;

procedure TNativeVectorStore.EnsureIndex;
var
  idx: TSignIndex;
  gen: Int64;
  current: boolean;
  r: TSqlRequest;
  stmt: TSqlite3Statement;
  fresh: TSignIndex;
begin
  idx := TSignIndex(fIndex);
  fDB.Lock; // lock order: database, then index
  try
    gen := ReadGen;
    idx.Lock.ReadLock;
    try
      current := idx.Loaded and (idx.Gen = gen);
    finally
      idx.Lock.ReadUnLock;
    end;
    if current then
      exit;
    // (re)load: only the sign codes, in rowid order, plus the generation they
    // belong to - both inside ONE read snapshot. Built aside, swapped in under
    // the write lock, so running searches keep their consistent old index
    fresh := TSignIndex.Create(fSignWords);
    try
      fDB.TransactionBegin(tbDeferred);
      try
        gen := ReadGen;
        r.Prepare(fDB.DB, FormatUtf8('SELECT id, sign FROM % ORDER BY id;',
          [fItems]));
        try
          stmt := r.Request;
          while r.Step = SQLITE_ROW do
          begin
            if sqlite3.column_bytes(stmt, 1) <> fSignWords * 8 then
              EVectorStore.RaiseUtf8('%: row % has a corrupt sign code',
                [self, sqlite3.column_int64(stmt, 0)]);
            fresh.Append(sqlite3.column_int64(stmt, 0), sqlite3.column_blob(stmt, 1));
          end;
        finally
          r.Close;
        end;
        fDB.Commit;
      except
        RollbackIfOpen(fDB);
        raise;
      end;
      idx.Lock.WriteLock;
      try
        idx.RowIds := fresh.RowIds;
        idx.Alive := fresh.Alive;
        idx.Codes := fresh.Codes;
        idx.Count := fresh.Count;
        idx.AliveCount := fresh.AliveCount;
        idx.Gen := gen;
        idx.Loaded := true;
      finally
        idx.Lock.WriteUnLock;
      end;
    finally
      fresh.Free;
    end;
  finally
    fDB.UnLock;
  end;
end;

/// re-raise a SQLITE_AUTH failure as EVectorStore (the read-only authorizer
// refused something only visible at Step time, e.g. a pragma function)
procedure RaiseIfAuth(aStore: TObject; aError: Exception);
begin
  if (aError is ESqlite3Exception) and
     (ESqlite3Exception(aError).ErrorCode = SQLITE_AUTH) then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query uses a forbidden ' +
      'operation (only reading is allowed): %', [aStore, aError.Message]);
end;

function TNativeVectorStore.KeyRowIds(const aKeySql: RawUtf8;
  const aParams: array of const): TInt64DynArray;
var
  r: TSqlRequest;
  n: integer;
begin
  result := nil;
  n := 0;
  fDB.Lock;
  try
    fDB.TransactionBegin(tbDeferred);
    try
      // same guards as the exact scan: authorizer for prepare AND step
      sqlite3.set_authorizer(fDB.DB, KeySqlAuthorizer, nil);
      try
        r.Prepare(fDB.DB, FormatUtf8('SELECT id FROM % WHERE ext_id IN (%' +
          #10 + ') ORDER BY id', [fItems, aKeySql]));
        try
          if sqlite3.stmt_readonly(r.Request) = 0 then
            EVectorStore.RaiseUtf8('%.SearchWhere: key query is not read-only',
              [self]);
          if r.ParamCount <> length(aParams) then
            EVectorStore.RaiseUtf8('%.SearchWhere: key query has % placeholders ' +
              'but % params were given', [self, r.ParamCount, length(aParams)]);
          if length(aParams) > 0 then
            r.Bind(aParams);
          while r.Step = SQLITE_ROW do
          begin
            if n = length(result) then
              SetLength(result, NextGrow(n));
            result[n] := r.FieldInt(0);
            inc(n);
          end;
        finally
          r.Close;
        end;
      finally
        sqlite3.set_authorizer(fDB.DB, nil, nil);
      end;
      fDB.Commit;
    except
      on E: Exception do
      begin
        RollbackIfOpen(fDB);
        RaiseIfAuth(self, E);
        raise;
      end;
    end;
  finally
    fDB.UnLock;
  end;
  SetLength(result, n);
end;

function TNativeVectorStore.IndexSearch(const aQuery: TLlmEmbedding; aTopK: integer;
  const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
var
  idx: TSignIndex;
  code: RawByteString;
  allowed, cands: TInt64DynArray;
  mask: TBooleanDynArray;
  a, slot: integer;
  top: TTopK;
  r: TSqlRequest;
  stmt: TSqlite3Statement;
  sql, json: RawUtf8;
  score: double;
  fpu: cardinal;
begin
  result := nil;
  idx := TSignIndex(fIndex);
  code := SignCode(aQuery);
  if aKeySql <> '' then
  begin
    // only a PRE-selection for the mask: the filter is applied again below,
    // inside the snapshot that also yields the vectors and texts
    allowed := KeyRowIds(aKeySql, aParams); // ascending rowids
    if allowed = nil then
      exit;
  end;
  // 1. candidates from the 1-bit index - under the READ lock only, without the
  //    database lock: concurrent searches scan in parallel
  idx.Lock.ReadLock;
  try
    mask := nil;
    if allowed <> nil then
    begin
      // allowed and RowIds are both ascending: one merge walk marks the slots
      SetLength(mask, idx.Count);
      a := 0;
      for slot := 0 to idx.Count - 1 do
      begin
        while (a < length(allowed)) and (allowed[a] < idx.RowIds[slot]) do
          inc(a);
        if a = length(allowed) then
          break;
        mask[slot] := allowed[a] = idx.RowIds[slot];
      end;
    end;
    cands := idx.Candidates(pointer(code), MaxPtrInt(MinPtrInt(fOversample,
      NATIVE_MAX_OVERSAMPLE), aTopK), mask);
  finally
    idx.Lock.ReadUnLock;
  end;
  if cands = nil then
    exit;
  // 2. exact rescoring + the winners' text in ONE read snapshot, through ONE
  //    statement: the candidates come as a JSON array, and the business filter
  //    is applied AGAIN here - the pre-selection above ran in an earlier
  //    snapshot, so a row whose filter attributes changed meanwhile must not
  //    slip through. The index only proposes; this snapshot decides.
  json := Int64DynArrayToCsv(pointer(cands), length(cands), '[', ']');
  sql := FormatUtf8('SELECT id, embedding FROM % WHERE ', [fItems]);
  if aKeySql <> '' then
    // key query first: its placeholders are 1..n, the JSON array is n+1
    sql := FormatUtf8('%ext_id IN (%' + #10 + ') AND ', [sql, aKeySql]);
  sql := sql + 'id IN (SELECT value FROM json_each(?))';
  top.Init(aTopK);
  fDB.Lock;
  try
    fDB.TransactionBegin(tbDeferred);
    try
      if aKeySql <> '' then
        sqlite3.set_authorizer(fDB.DB, KeySqlAuthorizer, nil); // see Scan
      try
        r.Prepare(fDB.DB, sql);
        try
          stmt := r.Request;
          if sqlite3.stmt_readonly(stmt) = 0 then
            EVectorStore.RaiseUtf8('%.SearchWhere: key query is not read-only',
              [self]);
          if r.ParamCount <> length(aParams) + 1 then
            EVectorStore.RaiseUtf8('%.SearchWhere: key query has % placeholders ' +
              'but % params were given', [self, r.ParamCount - 1, length(aParams)]);
          if length(aParams) > 0 then
            r.Bind(aParams);
          r.Bind(length(aParams) + 1, json);
          fpu := SetFpuFlags(ffLibrary); // see Scan: corrupt rows must not trap
          try
            while r.Step = SQLITE_ROW do
            begin
              if sqlite3.column_bytes(stmt, 1) <> fBlobSize then
                EVectorStore.RaiseUtf8('%: row % has a corrupt embedding',
                  [self, sqlite3.column_int64(stmt, 0)]);
              score := DotRow(pointer(aQuery), pointer(sqlite3.column_blob(stmt, 1)),
                fContract.Dim);
              if PInt64(@score)^ and DOUBLE_EXP_MASK = DOUBLE_EXP_MASK then
                EVectorStore.RaiseUtf8('%: row % has a corrupt embedding ' +
                  '(non-finite values)', [self, sqlite3.column_int64(stmt, 0)]);
              top.Offer(score, sqlite3.column_int64(stmt, 0));
            end;
          finally
            ResetFpuFlags(fpu);
          end;
        finally
          r.Close;
        end;
      finally
        if aKeySql <> '' then
          sqlite3.set_authorizer(fDB.DB, nil, nil);
      end;
      result := FetchHits(top.RowId, top.Score, top.Count);
      fDB.Commit;
    except
      on E: Exception do
      begin
        RollbackIfOpen(fDB);
        RaiseIfAuth(self, E);
        raise;
      end;
    end;
  finally
    fDB.UnLock;
  end;
end;

function TNativeVectorStore.FetchHits(const aRowIds: array of Int64;
  const aScores: array of double; aCount: PtrInt): TRagHitDynArray;
var
  r: TSqlRequest;
  i: PtrInt;
begin
  // text + key for the winners only - called inside the caller's read snapshot
  SetLength(result, aCount);
  if aCount = 0 then
    exit;
  r.Prepare(fDB.DB, FormatUtf8(
    'SELECT content, ext_id FROM % WHERE id = ?;', [fItems]));
  try
    for i := 0 to aCount - 1 do
    begin
      r.Bind(1, aRowIds[i]);
      if r.Step = SQLITE_ROW then
      begin
        r.FieldUtf8(0, result[i].Text);
        r.FieldUtf8(1, result[i].Key); // NULL ext_id -> '' (anonymous row)
      end;
      r.Reset;
      result[i].DocId := aRowIds[i];
      result[i].Distance := 1 - aScores[i];
    end;
  finally
    r.Close;
  end;
end;

function TNativeVectorStore.Scan(const aQuery: TLlmEmbedding; aTopK: integer;
  const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
var
  r: TSqlRequest;
  top: TTopK;
  sql: RawUtf8;
  stmt: TSqlite3Statement;
  row: pointer;
  score: double;
  fpu: cardinal;
begin
  result := nil;
  top.Init(aTopK);
  sql := FormatUtf8('SELECT id, embedding FROM %', [fItems]);
  if aKeySql <> '' then
    // validated by CheckKeySql; the line break keeps a trailing "-- comment"
    // of the key query from swallowing the closing parenthesis
    sql := FormatUtf8('% WHERE ext_id IN (%' + #10 + ')', [sql, aKeySql]);
  fDB.Lock;
  try
  // ONE read transaction around the scan AND the winner fetch: otherwise another
  // connection could update or delete a winner in between, and the hit would
  // pair new text (or none) with the old score
  fDB.TransactionBegin(tbDeferred);
  try
    // the read-only authorizer stays active for the WHOLE life of the filtered
    // statement, not only its compilation: table-valued pragma functions
    // (pragma_table_info ...) prepare their PRAGMA only at the first Step, and
    // SQLite may re-prepare a statement transparently while stepping
    if aKeySql <> '' then
      sqlite3.set_authorizer(fDB.DB, KeySqlAuthorizer, nil);
    try
    r.Prepare(fDB.DB, sql);
    try
      stmt := r.Request;
      if sqlite3.stmt_readonly(stmt) = 0 then
        EVectorStore.RaiseUtf8('%.SearchWhere: key query is not read-only', [self]);
      if r.ParamCount <> length(aParams) then
        EVectorStore.RaiseUtf8('%.SearchWhere: key query has % placeholders but ' +
          '% params were given', [self, r.ParamCount, length(aParams)]);
      if length(aParams) > 0 then
        r.Bind(aParams);
      // stream the rows: the BLOB is read in place (no copy, no allocation per
      // row); the CHECK constraint guarantees its size, verified anyway because
      // a foreign writer could have bypassed it with an older schema
      // mask the FPU exceptions for the arithmetic of the scan: a foreign row
      // with Inf or signalling-NaN bytes would otherwise trap INSIDE DotRow
      // (Inf * 0 = invalid on x86_64) before the score test below could
      // report it; masked, every such row ends in a non-finite score
      fpu := SetFpuFlags(ffLibrary);
      try
      while r.Step = SQLITE_ROW do
      begin
        if sqlite3.column_bytes(stmt, 1) <> fBlobSize then
          EVectorStore.RaiseUtf8('%: row % has a corrupt embedding (% bytes, ' +
            'expected %)', [self, sqlite3.column_int64(stmt, 0),
            sqlite3.column_bytes(stmt, 1), fBlobSize]);
        row := sqlite3.column_blob(stmt, 1);
        score := DotRow(pointer(aQuery), row, fContract.Dim);
        // the query is finite, so a non-finite score means NaN/Inf bytes in
        // the row (a foreign writer) - test the bits: comparing a NaN raises
        // EInvalidOp once the FPU flags are restored
        if PInt64(@score)^ and DOUBLE_EXP_MASK = DOUBLE_EXP_MASK then
          EVectorStore.RaiseUtf8('%: row % has a corrupt embedding (non-finite ' +
            'values)', [self, sqlite3.column_int64(stmt, 0)]);
        top.Offer(score, sqlite3.column_int64(stmt, 0));
      end;
      finally
        ResetFpuFlags(fpu);
      end;
    finally
      r.Close;
    end;
    finally
      if aKeySql <> '' then
        sqlite3.set_authorizer(fDB.DB, nil, nil);
    end;
    result := FetchHits(top.RowId, top.Score, top.Count);
    fDB.Commit; // ends the read snapshot
  except
    on E: Exception do
    begin
      RollbackIfOpen(fDB);
      RaiseIfAuth(self, E);
      raise;
    end;
  end;
  finally
    fDB.UnLock;
  end;
end;

end.
