/// LandrixAI LLM Client - provider configuration factories
// - part of the mormot.ai.* extension (LandrixAI)
// - the OpenAI Chat Completions wire is canonical, so OpenAI, LiteLLM and Ollama
//   are the same TLlmClient with a different TLlmProviderConfig
unit mormot.ai.llm.openai;

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.ai.llm;

/// configuration for the OpenAI API (api.openai.com)
function OpenAIConfig(const aApiKey: RawUtf8;
  const aModel: RawUtf8 = 'gpt-4o-mini'): TLlmProviderConfig;

/// configuration for a local/remote Ollama via its OpenAI-compatible endpoint
// - aBaseUrl points at the '/v1' root, e.g. 'http://localhost:11434/v1'
// - Ollama ignores the API key, so none is sent
function OllamaConfig(const aBaseUrl: RawUtf8 = 'http://localhost:11434/v1';
  const aModel: RawUtf8 = 'llama3.2'): TLlmProviderConfig;

/// configuration for a LiteLLM proxy (OpenAI-compatible, multi-provider)
function LiteLLMConfig(const aBaseUrl, aApiKey, aModel: RawUtf8): TLlmProviderConfig;


implementation

function OpenAIConfig(const aApiKey, aModel: RawUtf8): TLlmProviderConfig;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.BaseUrl := 'https://api.openai.com/v1';
  result.ApiKey := aApiKey;
  result.AuthScheme := lasBearer;
  result.DefaultModel := aModel;
end;

function OllamaConfig(const aBaseUrl, aModel: RawUtf8): TLlmProviderConfig;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.BaseUrl := aBaseUrl;
  result.AuthScheme := lasNone;
  result.DefaultModel := aModel;
end;

function LiteLLMConfig(const aBaseUrl, aApiKey, aModel: RawUtf8): TLlmProviderConfig;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.BaseUrl := aBaseUrl;
  result.ApiKey := aApiKey;
  result.AuthScheme := lasBearer;
  result.DefaultModel := aModel;
end;

end.
