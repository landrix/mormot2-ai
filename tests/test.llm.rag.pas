// - regression tests for mormot.ai.rag (chunking) + mormot.ai.vectorstore blobs
unit test.llm.rag;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.core.unicode,
  mormot.core.test,
  mormot.ai.llm.types,
  mormot.ai.vectorstore,
  mormot.ai.vectorstore.sqlitevec,
  mormot.ai.rag;

type
  TTestLlmRag = class(TSynTestCase)
  published
    procedure ChunkingEmpty;
    procedure ChunkingSingle;
    procedure ChunkingMultipleWithOverlap;
    procedure ChunkingUtf8Safe;
    procedure ChunkingUtf8LongToken;
    procedure VectorBlobRoundTrip;
    procedure VectorStoreKeyedOps;
    procedure EmptyQuestionIsADeterministicMiss;
  end;


implementation

// build a 4-dim embedding inline (FPC-safe: no inline var / nested function)
function V4(a, b, c, d: single): TLlmEmbedding;
begin
  SetLength(result, 4);
  result[0] := a;
  result[1] := b;
  result[2] := c;
  result[3] := d;
end;

procedure TTestLlmRag.ChunkingEmpty;
begin
  CheckEqual(length(ChunkText('', 100, 20)), 0, 'empty text yields no chunks');
end;

procedure TTestLlmRag.ChunkingSingle;
var
  chunks: TRawUtf8DynArray;
begin
  chunks := ChunkText('a short text', 100, 20);
  CheckEqual(length(chunks), 1, 'text below the window is one chunk');
  CheckEqual(chunks[0], 'a short text', 'chunk content preserved');
end;

procedure TTestLlmRag.ChunkingMultipleWithOverlap;
var
  text: RawUtf8;
  chunks: TRawUtf8DynArray;
  i: PtrInt;
begin
  // ~270 chars of words -> several chunks at window 100
  text := 'das dach ist undicht und muss saniert werden der kunde mueller ' +
          'wuenscht einen kostenvoranschlag fuer die dachsanierung sowie ' +
          'einen termin zur heizungswartung im naechsten monat bitte ' +
          'beachten sie den notdienst rund um die uhr unter der hotline';
  chunks := ChunkText(text, 100, 20);
  Check(length(chunks) > 2, 'long text splits into several chunks');
  for i := 0 to high(chunks) do
  begin
    Check(chunks[i] <> '', 'no empty chunk');
    // each chunk stays within the window (whitespace cut never exceeds it)
    Check(length(chunks[i]) <= 100, 'chunk within the window size');
  end;
  // progress is guaranteed: far fewer chunks than characters
  Check(length(chunks) < length(text), 'chunking made progress (no runaway)');
end;

procedure TTestLlmRag.ChunkingUtf8Safe;
var
  text: RawUtf8;
  chunks: TRawUtf8DynArray;
  i: PtrInt;
begin
  // German text full of multi-byte UTF-8: small window + overlap forces many
  // overlap boundaries; each chunk must stay valid UTF-8 (overlap start snaps to
  // a whitespace, never into the middle of an umlaut/euro byte sequence)
  text := 'Die Größe der Küche für die Tür ändert sich häufig, ' +
          'während die Möbel für 1200 € geliefert werden müssen und ' +
          'der Außenbereich später saniert wird übrigens schön';
  chunks := ChunkText(text, 24, 8);
  Check(length(chunks) > 3, 'small window yields many chunks');
  for i := 0 to high(chunks) do
    Check(IsValidUtf8(chunks[i]), 'every chunk is valid UTF-8');
end;

procedure TTestLlmRag.ChunkingUtf8LongToken;
var
  text: RawUtf8;
  chunks: TRawUtf8DynArray;
  i: PtrInt;
begin
  // a single long token with NO whitespace, all multi-byte: build the UTF-8 for
  // 'ü' (C3 BC) from explicit bytes so the input is byte-exact regardless of the
  // source-file codepage. The whitespace backup cannot help here, so the window
  // cut must still land on a UTF-8 boundary rather than splitting a codepoint.
  text := '';
  for i := 1 to 40 do
    text := text + #$C3 + #$BC;
  chunks := ChunkText(text, 15, 4);
  Check(length(chunks) > 1, 'long no-whitespace token still splits');
  for i := 0 to high(chunks) do
  begin
    Check(chunks[i] <> '', 'no empty chunk');
    Check(IsValidUtf8(chunks[i]), 'every chunk is valid UTF-8 (no split codepoint)');
  end;
end;

procedure TTestLlmRag.VectorBlobRoundTrip;
var
  v, w: TLlmEmbedding;
  blob: RawByteString;
begin
  SetLength(v, 3);
  v[0] := 1.5;
  v[1] := -2.25;
  v[2] := 3.75;
  blob := VectorToBlob(v);
  CheckEqual(length(blob), 3 * SizeOf(single), 'blob is float32-packed');
  w := BlobToVector(blob);
  CheckEqual(length(w), 3, 'round-trip dimension');
  CheckSame(w[0], 1.5, 1e-6, 'first');
  CheckSame(w[1], -2.25, 1e-6, 'second');
  CheckSame(w[2], 3.75, 1e-6, 'third');
end;

procedure TTestLlmRag.VectorStoreKeyedOps;
var
  extdir: RawUtf8;
  store: IVectorStore;
  hits: TRagHitDynArray;
  i: PtrInt;
  hadAddr2: boolean;
begin
  // real-vec0 test: needs only the vec0 extension (no GGUF model). Runs when
  // SQLITE_EXT_DIR points at a dir containing vec0; otherwise skipped so the
  // suite stays green where the (gitignored) vendored binary is absent (CI).
  extdir := StringToUtf8(GetEnvironmentVariable('SQLITE_EXT_DIR'));
  if (extdir = '') or
     not FileExists(Utf8ToString(extdir + '/vec0' + SqliteExtSuffix)) then
  begin
    Check(true, 'sqlite-vec (vec0) not available -> keyed-ops test skipped');
    exit;
  end;
  store := TVec0Store.Create(':memory:', extdir, 4);
  // three entity-keyed rows on an orthonormal basis (distinct nearest regions)
  store.Upsert('addr-1', 'mueller hamburg', V4(1, 0, 0, 0));
  store.Upsert('addr-2', 'schmidt berlin',  V4(0, 1, 0, 0));
  store.Upsert('addr-3', 'meyer koeln',      V4(0, 0, 1, 0));
  CheckEqual(store.Count, 3, 'three keyed rows stored');
  // nearest to addr-1's vector is addr-1, and its external Key round-trips
  hits := store.Search(V4(1, 0, 0, 0), 3);
  Check(length(hits) >= 1, 'search returns hits');
  CheckEqual(hits[0].Key, 'addr-1', 'top hit maps back to its external key');
  CheckEqual(hits[0].Text, 'mueller hamburg', 'top hit text round-trips');
  // Upsert on an existing id REPLACES text+vector in place (no new row)
  store.Upsert('addr-1', 'mueller hamburg altona', V4(0, 0, 0, 1));
  CheckEqual(store.Count, 3, 'upsert on existing id replaces, no new row');
  hits := store.Search(V4(0, 0, 0, 1), 1);
  CheckEqual(hits[0].Key, 'addr-1', 'replaced vector moved addr-1 to the new region');
  CheckEqual(hits[0].Text, 'mueller hamburg altona', 'replaced text is visible');
  // Delete removes only that entity; deleting an absent id is a no-op (idempotent)
  store.Delete('addr-2');
  CheckEqual(store.Count, 2, 'delete removed one row');
  store.Delete('does-not-exist');
  CheckEqual(store.Count, 2, 'deleting an absent id is a no-op');
  // addr-2 must no longer appear among the hits for its old region
  hits := store.Search(V4(0, 1, 0, 0), 3);
  hadAddr2 := false;
  for i := 0 to high(hits) do
    if hits[i].Key = 'addr-2' then
      hadAddr2 := true;
  Check(not hadAddr2, 'deleted entity no longer appears in search results');
end;


procedure TTestLlmRag.EmptyQuestionIsADeterministicMiss;
var
  rag: TLlmRag;
  resp: TLlmChatResponse;
begin
  // Deliberately wired with NO client, embedder or store: an empty question
  // must not reach any of them. It used to be passed straight to the embedder,
  // which raises on an empty vector - while Ingest handles empty input cleanly
  // and the RAG tool rejects an empty query outright. Only this path differed.
  rag := TLlmRag.Create(nil, nil, nil, 'model');
  try
    resp := rag.Query('   ');
    Check(resp.FinishReason = lfrStop, 'answered, not raised');
    CheckEqual(resp.Content, 'No matching information was found.',
      'the deterministic miss answer, in English');
    CheckEqual(length(rag.LastHits), 0, 'and nothing was retrieved');
    // ...and it is overridable: the sentence used to be hardcoded German in a
    // library whose own system prompt is English
    rag.NoAnswerText := 'Nichts gefunden.';
    resp := rag.Query('');
    CheckEqual(resp.Content, 'Nichts gefunden.', 'caller-defined miss answer');
  finally
    rag.Free;
  end;
end;

end.
