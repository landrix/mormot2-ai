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

/// build a provider config from environment variables (for tools/demos)
// - LLM_BASE_URL (default 'https://api.openai.com/v1'), LLM_MODEL (default
//   'gpt-4o-mini'), LLM_API_KEY (falls back to OPENAI_API_KEY)
// - bearer auth is used when a key is present, otherwise none (e.g. local Ollama)
// - the key is read from the environment ONLY - never hard-code or log it
function LlmConfigFromEnv: TLlmProviderConfig;


implementation

uses
  sysutils,
  mormot.core.text,
  mormot.core.unicode;

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

function LlmConfigFromEnv: TLlmProviderConfig;
var
  key: RawUtf8;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  // trim every value: a .env saved with CRLF line endings leaves a trailing CR
  // that would corrupt e.g. the Authorization header (bare CR -> 400 at the edge)
  result.BaseUrl := TrimU(StringToUtf8(GetEnvironmentVariable('LLM_BASE_URL')));
  if result.BaseUrl = '' then
    result.BaseUrl := 'https://api.openai.com/v1';
  result.DefaultModel := TrimU(StringToUtf8(GetEnvironmentVariable('LLM_MODEL')));
  if result.DefaultModel = '' then
    result.DefaultModel := 'gpt-4o-mini';
  key := TrimU(StringToUtf8(GetEnvironmentVariable('LLM_API_KEY')));
  if key = '' then
    key := TrimU(StringToUtf8(GetEnvironmentVariable('OPENAI_API_KEY')));
  result.ApiKey := key;
  if key <> '' then
    result.AuthScheme := lasBearer
  else
    result.AuthScheme := lasNone; // e.g. a local Ollama needs no key
end;

end.
