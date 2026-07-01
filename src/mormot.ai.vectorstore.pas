/// LandrixAI - vector store abstraction (backend-neutral)
// - part of the mormot.ai.* extension (LandrixAI)
// - defines ONLY the IVectorStore interface, the TRagHit result and the float32
//   blob helpers - NO storage backend. The RAG/agent engine depends on this unit
//   alone, so it never drags in a concrete store's dependencies (e.g. the static
//   SQLite linked by sqlite-vec). Concrete backends live in sibling units:
//     mormot.ai.vectorstore.sqlitevec  - TVec0Store (sqlite-vec, local/edge)
//     mormot.ai.vectorstore.pgvector   - TPgVectorStore (server/multi-tenant, future)
//   which backend is wired in is a composition decision (DI), never an {$ifdef}.
// - clean-room; target license MPL/GPL/LGPL
unit mormot.ai.vectorstore;

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.ai.llm.types;

type
  /// one KNN hit: the stored document id, its text and the backend distance
  TRagHit = record
    DocId: Int64;
    Text: RawUtf8;
    Distance: double;
    /// the stable external key set via Upsert (empty for anonymous Add rows)
    // - lets a hit map straight back to a domain entity (e.g. an address UUID)
    //   without a caller-side id table
    Key: RawUtf8;
  end;
  TRagHitDynArray = array of TRagHit;

  /// a local document + vector store (closest-first KNN)
  // - two ways to store: anonymous documents (Add/AddBatch, e.g. RAG chunks) OR
  //   entity-keyed rows (Upsert/Delete, keyed by a stable external id such as an
  //   address UUID). Both live in the same store; Search returns TRagHit.Key for
  //   the keyed rows (empty for anonymous ones).
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
    /// store or REPLACE the text+vector for a stable external key (entity id)
    // - first call for aId inserts, later calls replace text + vector in place
    //   (the internal mapping stays stable), so re-embedding an entity is idempotent
    // - aId must be non-empty; how the id maps to the backend row is the backend's
    //   private detail (the SQLite backend keeps a small id->rowid map)
    procedure Upsert(const aId, aText: RawUtf8; const aVector: TLlmEmbedding);
    /// remove the row previously stored under aId (no-op if absent)
    procedure Delete(const aId: RawUtf8);
    /// the aTopK documents nearest to a query vector, closest first
    // - each hit carries TRagHit.Key for entity-keyed rows (empty otherwise)
    function Search(const aQuery: TLlmEmbedding; aTopK: integer): TRagHitDynArray;
    /// number of stored documents
    function Count: Int64;
  end;


/// a TLlmEmbedding as the float32 blob that vector backends exchange
// - host-native byte order; sqlite-vec/lembed use little-endian float32, which
//   matches every supported target (x86_64/aarch64) - a big-endian port would need
//   an explicit swap here
function VectorToBlob(const aVec: TLlmEmbedding): RawByteString;
/// decode a float32 blob (e.g. from a backend column) back into a TLlmEmbedding
function BlobToVector(const aBlob: RawByteString): TLlmEmbedding;


implementation

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

end.
