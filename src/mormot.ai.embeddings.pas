/// LandrixAI LLM Client - embeddings abstraction (backend-neutral)
// - part of the mormot.ai.* extension (LandrixAI)
// - defines ONLY the IEmbedder interface, so the RAG engine depends on this unit
//   without pulling any concrete embedder (and its deps). Implementations live in
//   sibling units:
//     mormot.ai.embed.provider  - TProviderEmbedder (OpenAI-wire /embeddings)
//     mormot.ai.embed.lembed    - TLembedEmbedder (local GGUF model in SQLite)
//     mormot.ai.embed.ollama    - TOllamaEmbedder (future, from the merge)
// - clean-room from the OpenAI Embeddings spec; target license MPL/GPL/LGPL
unit mormot.ai.embeddings;

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.ai.llm.types;

type
  /// turns text into an embedding vector - provider-backed or local
  IEmbedder = interface
    ['{4D9E1F73-2A8C-4B16-9E50-7C3A1B2D4E6F}']
    /// embed a single text into one vector
    function Embed(const aText: RawUtf8): TLlmEmbedding;
    /// embed several texts at once (one vector per input, in order)
    function EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    /// the embedding model name in use
    function Model: RawUtf8;
  end;

implementation

end.
