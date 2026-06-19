/// LandrixAI - provider-backed IEmbedder (OpenAI-wire /embeddings)
// - part of the mormot.ai.* extension (LandrixAI)
// - implements IEmbedder against a provider's OpenAI-wire /embeddings endpoint;
//   owns a TLlmClient built from the given provider config. This unit (not the
//   IEmbedder interface in mormot.ai.embeddings) is what pulls the LLM client.
// - clean-room from the OpenAI Embeddings spec; target license MPL/GPL/LGPL
unit mormot.ai.embed.provider;

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.core.text, // ESynException
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.embeddings;

type
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
  // fail loudly instead of returning an empty vector (same contract as the local
  // TLembedEmbedder): a zero-length vector would later be stored or matched as a
  // garbage blob by the vector store
  if (length(batch) = 0) or (length(batch[0]) = 0) then
    ESynException.RaiseUtf8('%.Embed: provider returned no vector (model %)',
      [self, fModel]);
  result := batch[0];
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
