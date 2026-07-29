// LandrixAI vision demo - a multimodal (image + text) request. Same neutral
// TLlmChatRequest as the text demos; the message carries an image attachment
// (LlmImageMessage), and the client serializes it to the provider's wire
// (OpenAI image_url here). Switch the provider via the LLM_* env vars.
//
//   llm-vision
//   env: VISION_PROVIDER (openai [default] | anthropic) selects the client.
//        openai    -> LLM_BASE_URL / LLM_MODEL / LLM_API_KEY (e.g. gpt-4o-mini)
//        anthropic -> ANTHROPIC_API_KEY / ANTHROPIC_MODEL (default claude-opus-4-8)
//        image: VISION_IMAGE_B64 (+VISION_IMAGE_MEDIA) or VISION_IMAGE_URL
//   the key is read from the environment ONLY - never hard-code or log it
program llm.vision;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  {$ifdef UNIX}
  mormot.lib.openssl11, // HTTPS/TLS on POSIX (Windows uses SChannel)
  {$endif}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.openai,
  mormot.ai.llm.anthropic;

var
  cfg: TLlmProviderConfig;
  client: ILlmClient;
  msgs: TLlmMessageDynArray;
  imgs: TLlmImageDynArray;
  req: TLlmChatRequest;
  resp: TLlmChatResponse;
  imageUrl, imageB64, media, srcLabel, provider, akey: RawUtf8;
begin
  {$ifdef UNIX}
  OpenSslInitialize;
  {$endif}
  // same neutral request; only the ILlmClient differs per provider
  provider := LowerCaseU(TrimU(StringToUtf8(GetEnvironmentVariable('VISION_PROVIDER'))));
  if provider = 'anthropic' then
  begin
    akey := TrimU(StringToUtf8(GetEnvironmentVariable('ANTHROPIC_API_KEY')));
    if akey = '' then
    begin
      ConsoleWrite('set ANTHROPIC_API_KEY for VISION_PROVIDER=anthropic', ccLightRed);
      exit;
    end;
    cfg := AnthropicConfig(akey,
      TrimU(StringToUtf8(GetEnvironmentVariable('ANTHROPIC_MODEL'))));
    if cfg.DefaultModel = '' then
      cfg.DefaultModel := 'claude-opus-4-8';
    cfg.TimeoutMs := 120000;
    client := TAnthropicClient.Create(cfg);
  end
  else
  begin
    cfg := LlmConfigFromEnv;
    cfg.TimeoutMs := 120000;
    client := TLlmClient.Create(cfg);
  end;
  // VISION_IMAGE_B64 (+ optional VISION_IMAGE_MEDIA) inlines a base64 image, fully
  // self-contained; otherwise VISION_IMAGE_URL lets the provider fetch a URL
  imageB64 := TrimU(StringToUtf8(GetEnvironmentVariable('VISION_IMAGE_B64')));
  imageUrl := TrimU(StringToUtf8(GetEnvironmentVariable('VISION_IMAGE_URL')));
  if (imageB64 = '') and (imageUrl = '') then
    // a stable public test image (a gull portrait); override via the env vars
    imageUrl := 'https://upload.wikimedia.org/wikipedia/commons/thumb/' +
      'd/dd/Gull_portrait_ca_usa.jpg/320px-Gull_portrait_ca_usa.jpg';

  SetLength(imgs, 1);
  if imageB64 <> '' then
  begin
    media := TrimU(StringToUtf8(GetEnvironmentVariable('VISION_IMAGE_MEDIA')));
    if media = '' then
      media := 'image/png';
    imgs[0] := LlmImageBase64(media, imageB64);
    srcLabel := FormatUtf8('base64 (%, % chars)', [media, length(imageB64)]);
  end
  else
  begin
    imgs[0] := LlmImageUrl(imageUrl);
    srcLabel := imageUrl;
  end;
  SetLength(msgs, 1);
  msgs[0] := LlmImageMessage(lrUser,
    'What is in this image? Answer in one short sentence.', imgs);
  req := LlmChatRequest(cfg.DefaultModel, msgs);
  req.MaxTokens := 100;

  ConsoleWrite(FormatUtf8('>>> %  model=%', [cfg.BaseUrl, cfg.DefaultModel]), ccLightBlue);
  ConsoleWrite(FormatUtf8('image: %', [srcLabel]), ccLightGray);
  try
    resp := client.ChatComplete(req);
    ConsoleWrite(FormatUtf8('ANSWER: %', [resp.Content]), ccLightGreen);
  except
    on E: Exception do
      ConsoleWrite(FormatUtf8('ERROR %: %', [E.ClassName, E.Message]), ccLightRed);
  end;
end.
