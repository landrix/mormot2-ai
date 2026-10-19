// LandrixAI - the shortest way to a local model: load it, ask, stream the answer
// - native engine, no server and no library to ship: a Hugging Face folder (git
//   clone, safetensors; quantized to Q8_0 at load) or one llama.cpp GGUF file;
//   the family (Qwen3, Qwen3-MoE, Qwen2.5) comes from the files
//
//   usage: llm-quickstart <model folder | file.gguf> [question] [cpu | vulkan | cuda]
//   e.g.   llm-quickstart vendor/models/qwen3-0.6b "Why is the sky blue?"
program llm.quickstart;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.core.unicode,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.native,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.vulkan,
  mormot.ai.infer.backend.cuda,
  mormot.ai.infer.model.qwen3.vulkan,
  mormot.ai.infer.model.qwen3.cuda;

type
  // the streaming callback must be a method (of object)
  TPrinter = class
    procedure OnDelta(const aDelta: TLlmStreamDelta);
  end;

procedure TPrinter.OnDelta(const aDelta: TLlmStreamDelta);
begin
  if aDelta.ContentDelta <> '' then
    ConsoleWrite(aDelta.ContentDelta, ccWhite, {nolinefeed=}true);
end;

var
  source: TFileName;
  question, mode: RawUtf8;
  backend: TInferBackend;
  client: TNativeLlmClient;
  llm: ILlmClient;
  printer: TPrinter;
  msgs: TLlmMessageDynArray;
begin
  if ParamCount < 1 then
  begin
    ConsoleWrite('usage: llm-quickstart <model folder | file.gguf> [question] ' +
      '[cpu | vulkan | cuda]');
    exit;
  end;
  source := ParamStr(1);
  question := 'Why is the sky blue? Answer in two sentences.';
  if ParamCount >= 2 then
    question := StringToUtf8(ParamStr(2));
  mode := 'cpu';
  if ParamCount >= 3 then
    mode := LowerCaseU(StringToUtf8(ParamStr(3)));
  // a GPU backend computes what the device does not, and is freed last
  backend := nil;
  if mode = 'vulkan' then
    backend := TInferVulkanBackend.Create
  else if mode = 'cuda' then
    backend := TInferCudaBackend.Create;
  printer := TPrinter.Create;
  try
    // nil backend: the client computes on all cores with one of its own
    if SameText(ExtractFileExt(source), '.gguf') then
      client := TNativeLlmClient.CreateFromGguf(source, '', backend)
    else
      client := TNativeLlmClient.Create(source, '', backend);
    llm := client; // reference counted from here
    if mode = 'vulkan' then
      client.Device := TQwen3Vulkan.Create(client.Model,
        TInferVulkanBackend(backend))
    else if mode = 'cuda' then
      client.Device := TQwen3Cuda.Create(client.Model, TInferCudaBackend(backend));
    // the answer only: no <think> block before it
    client.EnableThinking := false;
    SetLength(msgs, 1);
    msgs[0] := LlmMessage(lrUser, question);
    llm.ChatStream(LlmChatRequest('', msgs), printer.OnDelta);
    ConsoleWrite('');
  finally
    llm := nil; // the model before its backend
    printer.Free;
    backend.Free;
  end;
end.
