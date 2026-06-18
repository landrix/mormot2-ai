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
  mormot.ai.rag;

type
  TTestLlmRag = class(TSynTestCase)
  published
    procedure ChunkingEmpty;
    procedure ChunkingSingle;
    procedure ChunkingMultipleWithOverlap;
    procedure ChunkingUtf8Safe;
    procedure VectorBlobRoundTrip;
  end;


implementation

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

end.
