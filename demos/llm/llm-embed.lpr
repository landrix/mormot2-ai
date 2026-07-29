// LandrixAI embeddings demo - embeds a few words via the provider's /embeddings
// endpoint and prints cosine similarities (related words score higher).
//
//   llm-embed
//   provider via env: LLM_BASE_URL / LLM_API_KEY; embedding model via
//   LLM_EMBED_MODEL (default text-embedding-3-small)
program llm.embed;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  {$ifdef UNIX}
  mormot.lib.openssl11, // HTTPS/TLS provider on POSIX (Windows uses SChannel)
  {$endif}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.openai,
  mormot.ai.embeddings,
  mormot.ai.embed.provider; // TProviderEmbedder (OpenAI-wire /embeddings)

function Cosine(const a, b: TLlmEmbedding): double;
var
  i: PtrInt;
  dot, na, nb: double;
begin
  dot := 0;
  na := 0;
  nb := 0;
  for i := 0 to high(a) do
  begin
    dot := dot + a[i] * b[i];
    na := na + a[i] * a[i];
    nb := nb + b[i] * b[i];
  end;
  if (na = 0) or (nb = 0) then
    result := 0
  else
    result := dot / (sqrt(na) * sqrt(nb));
end;

var
  cfg: TLlmProviderConfig;
  emb: IEmbedder;
  model: RawUtf8;
  texts: TRawUtf8DynArray;
  vecs: TLlmEmbeddingDynArray;
begin
  {$ifdef UNIX}
  OpenSslInitialize;
  {$endif}
  cfg := LlmConfigFromEnv;
  cfg.TimeoutMs := 120000;
  model := StringToUtf8(GetEnvironmentVariable('LLM_EMBED_MODEL'));
  if model = '' then
    model := 'text-embedding-3-small';
  emb := TProviderEmbedder.Create(cfg, model);

  SetLength(texts, 3);
  texts[0] := 'Hund';
  texts[1] := 'Katze';
  texts[2] := 'Auto';

  ConsoleWrite(FormatUtf8('>>> %  embed-model=%', [cfg.BaseUrl, model]), ccLightBlue);
  try
    vecs := emb.EmbedBatch(texts);
    ConsoleWrite(FormatUtf8('dim=%  (3 vectors)', [length(vecs[0])]), ccLightGray);
    ConsoleWrite(FormatUtf8('cos(Hund,Katze)=%  vs  cos(Hund,Auto)=%',
      [Cosine(vecs[0], vecs[1]), Cosine(vecs[0], vecs[2])]), ccLightGreen);
  except
    on E: Exception do
      ConsoleWrite(FormatUtf8('ERROR %: %', [E.ClassName, E.Message]), ccLightRed);
  end;
end.
