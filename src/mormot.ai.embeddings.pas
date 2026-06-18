/// LandrixAI LLM Client - embeddings abstraction
// - part of the mormot.ai.* extension (LandrixAI)
// - IEmbedder is backend-agnostic: a provider endpoint (this unit) or, later, a
//   local model in SQLite via sqlite-lembed - so RAG can switch source freely
// - clean-room from the OpenAI Embeddings spec; target license MPL/GPL/LGPL
unit mormot.ai.embeddings;

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.ai.llm.types,
  mormot.ai.llm;

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

  /// an IEmbedder backed by a provider's OpenAI-wire /embeddings endpoint
  // - owns its TLlmClient, built from the given provider config
  TProviderEmbedder = class(TInterfacedObject, IEmbedder)
  protected
    fClient: TLlmClient;
    fModel: RawUtf8;
  public
    /// create from a provider config and an embedding model name
    // - e.g. OpenAIConfig(key) + 'text-embedding-3-small'
    constructor Create(const aConfig: TLlmProviderConfig; const aModel: RawUtf8);
    destructor Destroy; override;
    function Embed(const aText: RawUtf8): TLlmEmbedding;
    function EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    function Model: RawUtf8;
  end;


implementation

constructor TProviderEmbedder.Create(const aConfig: TLlmProviderConfig;
  const aModel: RawUtf8);
begin
  inherited Create;
  fClient := TLlmClient.Create(aConfig);
  fModel := aModel;
end;

destructor TProviderEmbedder.Destroy;
begin
  fClient.Free;
  inherited Destroy;
end;

function TProviderEmbedder.Embed(const aText: RawUtf8): TLlmEmbedding;
var
  input: TRawUtf8DynArray;
  batch: TLlmEmbeddingDynArray;
begin
  SetLength(input, 1);
  input[0] := aText;
  batch := fClient.Embeddings(fModel, input);
  if length(batch) > 0 then
    result := batch[0]
  else
    result := nil;
end;

function TProviderEmbedder.EmbedBatch(
  const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
begin
  result := fClient.Embeddings(fModel, aTexts);
end;

function TProviderEmbedder.Model: RawUtf8;
begin
  result := fModel;
end;

end.
