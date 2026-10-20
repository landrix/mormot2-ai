/// LandrixAI LLM Client - local ILlmClient on the native inference engine
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - runs a decoder model (src/infer, docs/native-inference.md phase 2) behind
//   the same ILlmClient contract as the HTTP providers: the agent loop, the MCP
//   tool bridge and RAG work unchanged against a model on this machine
// - a decoder family registers its model, tokenizer, chat template and output
//   parser classes (RegisterNativeFamily); the client picks them by the
//   model_type of config.json or the architecture of a GGUF file - Qwen3,
//   Qwen2.5 and gpt-oss are registered here
// - the prompt is the model's chat template (TChatTemplate); the answer is
//   parsed token by token (TNativeOutputParser) - for Qwen3 a leading <think>
//   block becomes reasoning (returned like vLLM does, as reasoning_content in
//   Raw, never in Content), every <tool_call> block a TLlmToolCall, the rest
//   the content; for gpt-oss the Harmony messages (THarmonyOutputParser):
//   the analysis channel becomes reasoning, a message to functions.NAME a tool
//   call, the final channel the content
// - ChatComplete and ChatStream share one decoding path, so a streamed answer
//   is the complete answer cut into pieces - nothing else
// - only the ids the tokenizer knows are drawn: a padded embedding (gpt-oss
//   has 201088 rows for 200019 tokens, Qwen3 151936 for 151669) never yields
//   an id without text
// - sampling (mormot.ai.infer.sampler): the defaults of the model's
//   generation_config.json, as vLLM takes them; a request overrides them with
//   Temperature (0 = greedy) and Extra top_p, top_k, min_p and seed - the wire
//   names of OpenAI and vLLM
// - Images and ResponseFormat are refused (text model, no constrained
//   decoding yet) rather than silently dropped
unit mormot.ai.llm.native;

{
  *****************************************************************************

    - TNativeOutputParser: generated token ids to content, reasoning, tool calls
    - TQwen3OutputParser: the parser of Qwen3
    - TQwen2OutputParser: the parser of Qwen2.5 (no reasoning)
    - THarmonyOutputParser: the parser of gpt-oss (Harmony)
    - NativeChatPrompt: a TLlmChatRequest as prompt ids
    - RegisterNativeFamily: the classes of a decoder family
    - TNativeLlmClient: ILlmClient over a model folder or a GGUF file

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.json,
  mormot.core.search,
  mormot.ai.llm.types,
  mormot.ai.llm.sse,
  mormot.ai.llm,
  mormot.ai.infer.tensor,
  mormot.ai.infer.unicode,
  mormot.ai.infer.tokenizer,
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.chat,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.decoder,
  mormot.ai.infer.model.qwen3,
  mormot.ai.infer.model.gptoss,
  mormot.ai.infer.sampler;

const
  /// the answer cap when a request leaves MaxTokens at 0
  // - the reasoning of a thinking model counts: a cap that cuts the think
  // block leaves an empty Content with finish reason length
  NATIVE_LLM_DEFAULT_MAX_TOKENS = 4096;

type
  /// the bytes of one kind of output text on their way to UTF-8 deltas
  // - a token may end inside a multi-byte character: those bytes wait for the
  // next token; newlines at the very start are dropped, newlines at the end
  // wait until text follows or Flush decides - the chat template strips them
  // around <think> blocks and writes its own before a tool call, but renders
  // the content of a plain answer verbatim
  // - invalid UTF-8 becomes U+FFFD, one per invalid byte
  TNativeTextChannel = object
  public
    /// bytes not given out yet: a cut character, or trailing newlines
    Pending: RawUtf8;
    /// true once some text was given out
    Started: boolean;
    /// everything given out so far
    Text: RawUtf8;
    /// append aBytes; returns the text that can be given out now
    function Add(const aBytes: RawUtf8): RawUtf8;
    /// the end of the text (or of a piece of it): returns what is left, a cut
    // character as U+FFFD, trailing newlines dropped when aDropNewlines
    function Flush(aDropNewlines: boolean): RawUtf8;
  end;

  /// turns the token ids a model generates into a TLlmChatResponse
  // - Push one id after the other, then Finish; with aOnDelta assigned every
  // piece is also given out as a TLlmStreamDelta as soon as it is known
  // - a family overrides Token (and FlushOpen) for the grammar of its answers:
  // which ids stop, open reasoning or a tool call; the response is assembled
  // here - content, reasoning (Reasoning and ReasoningDelta, and as vLLM
  // returns it: reasoning_content in Raw), tool calls, finish reason and usage
  TNativeOutputParser = class
  protected
    fTokenizer: TInferTokenizer;
    fOnDelta: TLlmStreamDeltaEvent;
    fContent, fReasoning: TNativeTextChannel;
    fCalls: TLlmToolCallDynArray;
    fCount: integer;
    fRoleSent, fFinished: boolean;
    fResponse: TLlmChatResponse;
    procedure Send(var aDelta: TLlmStreamDelta);
    procedure SendContent(const aText: RawUtf8);
    procedure SendReasoning(const aText: RawUtf8);
    /// a tool call parsed from the answer: the content before it ends (its
    // trailing newlines dropped), then the call gets an id and goes out
    procedure AddToolCall(const aName, aArguments: RawUtf8);
    /// the bytes of a token as RawUtf8 - copied, so no code page is involved
    function Bytes(aId: integer): RawUtf8;
    /// one generated id (inside the tokenizer, counted): true for a stop
    // token, which adds no text
    function Token(aId: integer): boolean; virtual; abstract;
    /// the answer ends: give out what is still open (a cut reasoning or tool
    // call) - nothing by default
    procedure FlushOpen; virtual;
  public
    /// aTokenizer (not owned); a family looks up its control tokens here and
    // raises EInferError when one is missing
    constructor Create(aTokenizer: TInferTokenizer;
      const aOnDelta: TLlmStreamDeltaEvent = nil); virtual;
    /// one generated id; returns true for a stop token, which adds no text
    // - raises ELlmClient for an id outside the tokenizer, or after Finish
    function Push(aId: integer): boolean;
    /// the answer is complete: aLength = true when it was cut by the token cap
    // or the context, false when it stopped on a stop token
    // - gives out the final delta (finish reason and usage), then the Done one
    procedure Finish(aLength: boolean; aPromptTokens: integer);
    /// the answer, after Finish
    property Response: TLlmChatResponse
      read fResponse;
    /// the reasoning text (also Response.Reasoning, and in Response.Raw as
    // reasoning_content)
    property Reasoning: RawUtf8
      read fReasoning.Text;
    /// the ids pushed so far, stop token included
    property Count: integer
      read fCount;
    /// the tokenizer of the ids
    property Tokenizer: TInferTokenizer
      read fTokenizer;
  end;

  /// an output parser class, as a decoder family registers it
  TNativeOutputParserClass = class of TNativeOutputParser;

  /// where the parser is inside the generated answer
  TQwen3ParserState = (
    qpsStart,
    qpsReasoning,
    qpsContent,
    qpsToolCall);

  /// turns the token ids a Qwen3 model generates into a TLlmChatResponse
  // - Push one id after the other, then Finish; with aOnDelta assigned every
  // piece is also given out as a TLlmStreamDelta as soon as it is known
  // - <think> as the first token opens the reasoning, </think> closes it; a
  // <tool_call> ... </tool_call> block whose JSON is {"name": string,
  // "arguments": object} and nothing else becomes a tool call, its arguments
  // written like Python json.dumps (as the chat template renders them back),
  // only a float json.dumps would round kept as the model wrote it;
  // any other block stays in the content as text, like vLLM's hermes parser
  // - control tokens anywhere else are text: their content goes out verbatim
  // - newlines: dropped at the start of reasoning and content, at the end of
  // the reasoning and before a tool call (the template writes them itself);
  // kept at the end of a plain answer, which the template renders verbatim
  TQwen3OutputParser = class(TNativeOutputParser)
  protected
    fStop, fEndOfText, fThink, fThinkEnd, fToolCall, fToolCallEnd: integer;
    fState: TQwen3ParserState;
    fToolBody: RawUtf8;
    procedure EndToolCall;
    /// stop on <|im_end|> or <|endoftext|>
    function Token(aId: integer): boolean; override;
    procedure FlushOpen; override;
    /// true: the model may open its answer with <think> (Qwen3)
    class function Thinks: boolean; virtual;
  public
    /// aTokenizer (not owned) must know <|im_end|>, <tool_call> and
    // </tool_call>, and <think> and </think> when Thinks - raises EInferError
    // otherwise
    constructor Create(aTokenizer: TInferTokenizer;
      const aOnDelta: TLlmStreamDeltaEvent = nil); override;
  end;

  /// the output parser of Qwen2.5: that of Qwen3 without reasoning - the same
  // <tool_call> blocks, no <think> (the tokenizer has none)
  TQwen2OutputParser = class(TQwen3OutputParser)
  protected
    class function Thinks: boolean; override;
  end;

  /// where the Harmony parser is inside the generated answer
  THarmonyParserState = (
    hpsHeader,
    hpsBody,
    hpsBetween);

  /// what the body of a Harmony message becomes
  THarmonyTarget = (
    htContent,
    htReasoning,
    htToolCall);

  /// turns the token ids gpt-oss generates into a TLlmChatResponse
  // - the prompt ends with <|start|>assistant: the answer is a sequence of
  // messages, each a header up to <|message|> (<|channel|> and the channel,
  // optionally to= a recipient and a content type, <|constrain|> included,
  // before or after the channel), then its body up to <|end|> (another message
  // follows after <|start|>), <|call|> or <|return|> - both stop, as does
  // <|endoftext|>
  // - the analysis channel becomes reasoning (reasoning_content in Raw), the
  // final and the commentary channel without a recipient the content; a
  // message to functions.NAME a tool call when its body is a JSON object
  // (written like Python json.dumps, as the template renders the arguments
  // back; only a float json.dumps would round kept as the model wrote it),
  // else its body stays in the content as text; a message to another
  // recipient (a built-in tool of OpenAI, none is offered) goes to the
  // reasoning
  // - text is verbatim, but for the newlines at the very start of the content
  // and of the reasoning (TNativeTextChannel); control tokens inside a body are
  // text, their content verbatim; text between messages is content
  THarmonyOutputParser = class(TNativeOutputParser)
  protected
    fStart, fChannel, fMessage, fEnd, fCall, fReturn, fConstrain,
    fEndOfText: integer;
    fState: THarmonyParserState;
    fTarget: THarmonyTarget;
    fHeader, fBody, fRecipient: RawUtf8;
    /// the header is complete: where its body goes
    procedure EndHeader;
    /// the body is complete: a tool call is parsed
    procedure EndBody;
    function Token(aId: integer): boolean; override;
    procedure FlushOpen; override;
  public
    /// aTokenizer (not owned) must know the Harmony control tokens - raises
    // EInferError otherwise
    constructor Create(aTokenizer: TInferTokenizer;
      const aOnDelta: TLlmStreamDeltaEvent = nil); override;
    /// the channel and the recipient of a Harmony header (the text of its
    // tokens, control tokens as their content): the word after <|channel|>,
    // the word after "to=" - '' when absent
    class procedure ParseHeader(const aHeader: RawUtf8;
      out aChannel, aRecipient: RawUtf8);
  end;

/// the enable_thinking switch of a request: Extra.chat_template_kwargs.
// enable_thinking as vLLM and SGLang read it, else aDefault
// - raises ELlmClient when the value is there but no boolean
function NativeEnableThinking(const aRequest: TLlmChatRequest;
  aDefault: boolean): boolean;

/// the template options of a request: enable_thinking (NativeEnableThinking);
// reasoning_effort as OpenAI sends it (a field of the request: Extra) or as
// vLLM reads it (Extra.chat_template_kwargs), else aDefaultEffort;
// model_identity of chat_template_kwargs; an empty string is no value (the
// default stays - transformers would write it as it is)
// - raises ELlmClient for a value that is no string, or two efforts that differ,
// whatever the family (Qwen ignores both options, but refuses such a request
// too, as vLLM does)
function NativeTemplateOptions(const aRequest: TLlmChatRequest;
  aDefaultThinking: boolean; const aDefaultEffort: RawUtf8 = ''): TChatTemplateOptions;

/// the prompt ids of a request: its messages and tools through aTemplate, with
// the assistant turn opened and the options of NativeTemplateOptions
// - raises ELlmClient for what the engine cannot honour (no messages, images,
// ResponseFormat) and for a tool or an option the template refuses
function NativeChatPrompt(aTemplate: TChatTemplate;
  const aRequest: TLlmChatRequest; aDefaultThinking: boolean;
  const aDefaultEffort: RawUtf8 = ''): TIntegerDynArray;

/// parse the text between <tool_call> and </tool_call>
// - true when it is strict JSON (see PythonJson) and an object with exactly
// the keys "name" (a non-empty string) and "arguments" (an object);
// aArguments is then that object as Python json.dumps writes it, except a
// float of more than 15 digits or beyond the double range: kept verbatim, so
// a tool that reads its arguments as doubles rounds or refuses that value
function ParseQwen3ToolCall(const aBody: RawUtf8;
  out aName, aArguments: RawUtf8): boolean;

/// parse the body of a Harmony message to functions.NAME
// - true when it is strict JSON (see PythonJson) and an object; aArguments is
// then that object as Python json.dumps writes it, a float of more than 15
// digits or beyond the double range kept verbatim
function ParseHarmonyArguments(const aBody: RawUtf8; out aArguments: RawUtf8): boolean;

/// the sampling of a request: aDefaults, overridden by Temperature (when >= 0)
// and by Extra top_p, top_k, min_p and seed; without a seed a random one
// - the filters only shape a sampled draw: with greedy defaults (a model
// without generation_config.json) a request must set a Temperature above 0,
// top_p alone keeps it greedy - unlike vLLM, which samples at 1 then
// - raises ELlmClient for a value of the wrong kind or out of range
function NativeSampling(const aRequest: TLlmChatRequest;
  const aDefaults: TInferSampling): TInferSampling;

/// decoding of aPrompt into aParser, which is finished afterwards
// - every token drawn by a TInferSampler of aSampling (greedy at temperature 0)
// among the ids of aParser.Tokenizer: a row of a padded embedding beyond them
// is never drawn
// - at most aMaxNew tokens, and never more than the context holds: the last
// token is not fed back, so a prompt of P tokens leaves C - P + 1, C the
// model's MaxPositions or aMaxContext when that is smaller (and above 0)
// - raises ELlmClient for an empty prompt, one longer than the context, or
// aMaxNew <= 0
// - aContext: nil decodes in a context of its own; else the tokens it caches
// at the start of aPrompt are not computed again (a prompt cache), and it
// holds the prompt and the drawn tokens afterwards (but the last one); after
// an exception, free or Reset it
// - returns how many prompt tokens came from aContext
function NativeDecode(aModel: TDecoderModel; const aPrompt: TIntegerDynArray;
  aMaxNew: PtrInt; aParser: TNativeOutputParser; const aSampling: TInferSampling;
  aContext: TDecoderContext = nil; aMaxContext: PtrInt = 0): PtrInt;

const
  /// how many finished requests a TNativeLlmClient keeps the KV cache of
  NATIVE_LLM_DEFAULT_CACHE_SLOTS = 1;

type
  /// contexts of the prompt cache
  TDecoderContextDynArray = array of TDecoderContext;
  /// the former name of TDecoderContextDynArray
  TQwen3ContextDynArray = TDecoderContextDynArray;

  /// the KV caches of finished requests, kept for the next request whose
  // prompt starts with the same tokens - an agent loop or a chat sends the
  // whole conversation again, one turn longer: only the new turn is computed
  // - thread-safe; a context serves one request at a time (Acquire, Release)
  // - memory: a context keeps 2 x Layers x KvHeads x HeadDim singles per token
  // of its capacity (its longest sequence, grown by doubling) - 224 KB per
  // token for Qwen3-0.6B, 896 MB for 4096 - and scratch buffers for its
  // longest prompt (about 60 KB per token more)
  // tokens - so few slots, and none on a small device
  TNativePromptCache = class
  protected
    fSafe: TLightLock;
    fModel: TDecoderModel;
    fIdle: TDecoderContextDynArray; // the least recently used first
    fSlots: integer;
    fRequests, fHits, fReused: Int64;
    procedure SetSlots(aSlots: integer);
    function GetRequests: Int64;
    function GetHits: Int64;
    function GetReused: Int64;
  public
    /// keep at most aSlots idle contexts (0: none, every request starts empty)
    constructor Create(aSlots: integer = NATIVE_LLM_DEFAULT_CACHE_SLOTS);
    /// free the idle contexts
    destructor Destroy; override;
    /// a context for aPrompt, out of the cache: the idle one caching the most
    // of its first tokens; when none caches any, the least recently used one
    // if the cache is full, else a new one
    function Acquire(const aPrompt: TIntegerDynArray): TDecoderContext;
    /// give back the context of a finished request, which took aReused prompt
    // tokens from it: kept as the most recently used, the least recently used
    // one freed beyond Slots
    // - the context of a failed request is freed instead, not given back
    procedure Release(aContext: TDecoderContext; aReused: PtrInt);
    /// forget every cached sequence
    procedure Clear;
    /// remove the idle contexts and return them, to be freed by the caller
    // (e.g. outside a lock of its own: freeing a large cache takes a while)
    function Take: TDecoderContextDynArray;
    /// where Acquire makes a new context: Model.NewContext (on Model.Device
    // when it has one), a TDecoderContext when nil (not owned)
    // - set before the first Acquire, and Clear when Model.Device changes
    property Model: TDecoderModel
      read fModel write fModel;
    /// how many idle contexts are kept
    property Slots: integer
      read fSlots write SetSlots;
    /// the requests given back so far (a failed one is not given back)
    property Requests: Int64
      read GetRequests;
    /// how many of them took prompt tokens from the cache
    property Hits: Int64
      read GetHits;
    /// prompt tokens not computed again, over every request
    property ReusedTokens: Int64
      read GetReused;
  end;

  /// the classes of a decoder family, as TNativeLlmClient creates them
  TNativeFamily = record
    Model: TDecoderModelClass;
    Tokenizer: TInferTokenizerClass;
    Template: TChatTemplateClass;
    Parser: TNativeOutputParserClass;
  end;

  /// a local ILlmClient: a model folder or GGUF file run by the native engine
  // - thread-safe: the model is immutable, every request decodes in its own
  // context (KV cache) with its own parser
  // - prompt cache: the context of a finished request is kept (Cache), and a
  // request that starts with the same tokens - the next turn of the same
  // conversation - only computes the rest; the answer stays the same
  TNativeLlmClient = class(TInterfacedObject, ILlmClient)
  protected
    fBackend: TInferBackend;
    fOwnBackend: boolean;
    fFamily: TNativeFamily;
    fTokenizer: TInferTokenizer;
    fTemplate: TChatTemplate;
    fModel: TDecoderModel;
    fDevice: TDecoderDevice;
    fSafe: TLightLock; // fActive, and Device against Generate
    fActive: integer;  // requests running
    fCache: TNativePromptCache;
    fConfig: TLlmProviderConfig;
    fMaxTokens, fMaxContext: integer;
    fEnableThinking: boolean;
    fReasoningEffort: RawUtf8;
    fSampling: TInferSampling;
    procedure Generate(const aRequest: TLlmChatRequest; aParser: TNativeOutputParser);
    procedure SetDevice(aDevice: TDecoderDevice);
    procedure SetFamily(aModel: TDecoderModelClass; const aSource: TFileName);
  public
    /// load tokenizer.json, config.json and model.safetensors of aFolder, with
    // the family its config.json names (model_type)
    // - aModelName: the name in Config and in every response, the folder name
    // when ''; aBackend: nil computes on a TInferCpuFastBackend of its own,
    // else on aBackend (not owned)
    // - aWeights: idtQ8_0 (default) quantizes every matrix at load - a quarter
    // of the memory of idtF32 and about twice its decoding speed, but rounding
    // the weights can flip a close decision (with Q8_0, Qwen3-0.6B answers the
    // weather question of the reference itself instead of calling the tool);
    // idtF32 keeps the model's own decisions - Qwen3-0.6B greedy is
    // token-identical to transformers - for tests and comparisons
    constructor Create(const aFolder: TFileName; const aModelName: RawUtf8 = '';
      aBackend: TInferBackend = nil; aWeights: TInferDType = idtQ8_0); reintroduce;
    /// one llama.cpp GGUF file: weights, settings and the tokenizer from it,
    // with the family its general.architecture names
    // (NewDecoderModelFromGguf, then the family's tokenizer CreateFromGguf)
    // - aModelName: the name in Config and in every response, the file name
    // without extension when ''; aBackend as for Create
    // - a GGUF file has no generation_config.json: Sampling is greedy until set
    constructor CreateFromGguf(const aFileName: TFileName;
      const aModelName: RawUtf8 = ''; aBackend: TInferBackend = nil);
    destructor Destroy; override;
    function ChatComplete(const aRequest: TLlmChatRequest): TLlmChatResponse;
    procedure ChatStream(const aRequest: TLlmChatRequest;
      const aOnDelta: TLlmStreamDeltaEvent);
    function Config: TLlmProviderConfig;
    /// the answer cap of a request without MaxTokens (default 4096, the
    // reasoning included)
    property MaxTokens: integer
      read fMaxTokens write fMaxTokens;
    /// the most tokens of a conversation, prompt and answer: 0 (the default)
    // for the model's MaxPositions (40960 for Qwen3), else the smaller one
    // - the KV cache grows with the conversation, by doubling: the cap keeps
    // it below about twice its size (Qwen3-0.6B: 224 KB per token in floats,
    // keys and values; half that on a GPU device with KvF16);
    // a longer prompt raises ELlmClient, a longer answer stops with finish
    // reason length
    // - set it before the first request: lowering it later does not shrink the
    // contexts the prompt cache already holds (Cache.Slots := 0 frees them)
    property MaxContext: integer
      read fMaxContext write fMaxContext;
    /// whether the model reasons before it answers when a request does not
    // say (Qwen3: a <think> block)
    // (default true, the model's own default)
    property EnableThinking: boolean
      read fEnableThinking write fEnableThinking;
    /// the reasoning effort of a request that does not say (gpt-oss: 'low',
    // 'medium' or 'high'; '' = the template's medium)
    property ReasoningEffort: RawUtf8
      read fReasoningEffort write fReasoningEffort;
    /// the sampling of a request that does not say otherwise: from the
    // model's generation_config.json (greedy without one); its Seed is unused,
    // every request draws its own unless Extra.seed is given
    property Sampling: TInferSampling
      read fSampling write fSampling;
    /// the prompt cache (NATIVE_LLM_DEFAULT_CACHE_SLOTS slots); Cache.Slots :=
    // 0 frees the memory and turns it off
    property Cache: TNativePromptCache
      read fCache;
    /// the model
    property Model: TDecoderModel
      read fModel;
    /// the classes of the model's family
    property Family: TNativeFamily
      read fFamily;
    /// where the model computes: nil (the default) on the backend of Create,
    // else a TDecoderDevice of Model - e.g. TQwen3Vulkan.Create(Model, gpu)
    // decodes on a GPU - owned: freed with the client or when replaced
    // - raises ELlmClient while a request runs or for a device of another
    // model (aDevice is then not taken over); the prompt cache is cleared
    // - a context taken with Cache.Acquire outside of a request must be given
    // back before: the old device is freed here
    property Device: TDecoderDevice
      read fDevice write SetDevice;
    /// the chat template, and through it the tokenizer
    property Template: TChatTemplate
      read fTemplate;
  end;


/// make a decoder family known to TNativeLlmClient: the tokenizer, chat
// template and output parser classes of its model class
// - call it from an initialization section: the list is not locked; a model
// class registered twice is listed twice, NativeFamily finds the first entry
procedure RegisterNativeFamily(aModel: TDecoderModelClass;
  aTokenizer: TInferTokenizerClass; aTemplate: TChatTemplateClass;
  aParser: TNativeOutputParserClass);

/// the registered classes of a model class - false when it has none
function NativeFamily(aModel: TDecoderModelClass;
  out aFamily: TNativeFamily): boolean;


implementation

{ TNativeTextChannel }

/// append aLen bytes to aDest, no code page conversion on any compiler
procedure AppendBytes(var aDest: RawUtf8; p: pointer; aLen: PtrInt);
var
  n: PtrInt;
begin
  if aLen <= 0 then
    exit;
  n := length(aDest);
  SetLength(aDest, n + aLen);
  MoveFast(p^, PByteArray(aDest)^[n], aLen);
end;

/// how many bytes at the end of p[0..aLen-1] start a character that is not
// complete yet
function IncompleteTail(p: PByteArray; aLen: PtrInt): PtrInt;
var
  i, need: PtrInt;
  b: byte;
begin
  result := 0;
  i := aLen - 1;
  // back over at most three continuation bytes to the lead byte
  while (i >= 0) and (i >= aLen - 4) and (p^[i] and $c0 = $80) do
    dec(i);
  if (i < 0) or (i < aLen - 4) then
    exit;
  b := p^[i];
  if (b >= $c2) and (b <= $df) then
    need := 2
  else if (b >= $e0) and (b <= $ef) then
    need := 3
  else if (b >= $f0) and (b <= $f4) then
    need := 4
  else
    exit; // ASCII or an invalid lead: nothing to wait for
  if aLen - i < need then
    result := aLen - i;
end;

/// aText as valid UTF-8: every invalid byte becomes U+FFFD
function ValidUtf8(const aText: RawUtf8): RawUtf8;
var
  cp: TCardinalDynArray;
  n: PtrInt;
begin
  if IsStrictUtf8(pointer(aText), length(aText)) then
  begin
    result := aText;
    exit;
  end;
  n := Utf8ToCodePoints(pointer(aText), length(aText), cp);
  result := CodePointsToUtf8(pointer(cp), n);
end;

function TNativeTextChannel.Add(const aBytes: RawUtf8): RawUtf8;
var
  n, lead: PtrInt;
begin
  result := '';
  AppendBytes(Pending, pointer(aBytes), length(aBytes));
  if not Started then
  begin
    lead := 0;
    while (lead < length(Pending)) and (Pending[lead + 1] = #10) do
      inc(lead);
    if lead > 0 then
      delete(Pending, 1, lead);
  end;
  n := length(Pending) - IncompleteTail(pointer(Pending), length(Pending));
  while (n > 0) and (Pending[n] = #10) do
    dec(n); // held back until text follows or Flush decides
  if n = 0 then
    exit;
  result := ValidUtf8(copy(Pending, 1, n));
  delete(Pending, 1, n);
  Started := true;
  AppendBytes(Text, pointer(result), length(result));
end;

function TNativeTextChannel.Flush(aDropNewlines: boolean): RawUtf8;
var
  n: PtrInt;
begin
  result := ValidUtf8(Pending);
  Pending := '';
  n := length(result);
  if aDropNewlines then
    while (n > 0) and (result[n] = #10) do
      dec(n);
  SetLength(result, n);
  if not Started then
  begin
    n := 0;
    while (n < length(result)) and (result[n + 1] = #10) do
      inc(n);
    delete(result, 1, n);
  end;
  if result <> '' then
    Started := true;
  AppendBytes(Text, pointer(result), length(result));
end;


{ TNativeOutputParser }

function FinishReasonText(aReason: TLlmFinishReason): RawUtf8;
begin
  case aReason of
    lfrStop:
      result := 'stop';
    lfrLength:
      result := 'length';
    lfrToolCalls:
      result := 'tool_calls';
    lfrContentFilter:
      result := 'content_filter';
  else
    result := '';
  end;
end;

constructor TNativeOutputParser.Create(aTokenizer: TInferTokenizer;
  const aOnDelta: TLlmStreamDeltaEvent);
begin
  inherited Create;
  if aTokenizer = nil then
    EInferError.RaiseUtf8('%.Create: no tokenizer', [self]);
  fTokenizer := aTokenizer;
  fOnDelta := aOnDelta;
end;

procedure TNativeOutputParser.FlushOpen;
begin
end;

function TNativeOutputParser.Bytes(aId: integer): RawUtf8;
var
  b: RawByteString;
begin
  b := fTokenizer.TokenBytes(aId);
  FastSetString(result, pointer(b), length(b));
end;

procedure TNativeOutputParser.Send(var aDelta: TLlmStreamDelta);
begin
  if not Assigned(fOnDelta) then
    exit;
  if not fRoleSent then
  begin
    aDelta.Role := 'assistant';
    fRoleSent := true;
  end;
  fOnDelta(aDelta);
end;

procedure TNativeOutputParser.SendContent(const aText: RawUtf8);
var
  d: TLlmStreamDelta;
begin
  if (aText = '') or not Assigned(fOnDelta) then
    exit;
  Finalize(d);
  FillCharFast(d, SizeOf(d), 0);
  d.ContentDelta := aText;
  Send(d);
end;

procedure TNativeOutputParser.SendReasoning(const aText: RawUtf8);
var
  d: TLlmStreamDelta;
begin
  if (aText = '') or not Assigned(fOnDelta) then
    exit;
  Finalize(d);
  FillCharFast(d, SizeOf(d), 0);
  d.ReasoningDelta := aText;
  // the chunk shape of vLLM's reasoning parser
  d.Raw := _ObjFast(['choices', _Arr([_ObjFast(['index', 0,
    'delta', _ObjFast(['reasoning_content', aText])])])]);
  Send(d);
end;

procedure TNativeOutputParser.AddToolCall(const aName, aArguments: RawUtf8);
var
  call: TLlmToolCall;
  d: TLlmStreamDelta;
  n: PtrInt;
begin
  // the newlines before a call are the template's, which writes its own
  SendContent(fContent.Flush(true));
  call.Name := aName;
  call.ArgumentsJson := aArguments;
  call.Id := 'call_' + CardinalToHexLower(Random32) + CardinalToHexLower(Random32);
  n := length(fCalls);
  SetLength(fCalls, n + 1);
  fCalls[n] := call;
  if not Assigned(fOnDelta) then
    exit;
  Finalize(d);
  FillCharFast(d, SizeOf(d), 0);
  d.HasToolCall := true;
  d.ToolCallIndex := n;
  d.ToolCallId := call.Id;
  d.ToolCallName := call.Name;
  d.ToolCallArgsDelta := call.ArgumentsJson; // complete: parsed as a whole
  Send(d);
end;

function TNativeOutputParser.Push(aId: integer): boolean;
begin
  if fFinished then
    ELlmClient.RaiseUtf8('%.Push after Finish', [self]);
  if (aId < 0) or (aId >= fTokenizer.VocabSize) then
    ELlmClient.RaiseUtf8('%.Push: the model generated id %, outside the ' +
      'tokenizer (0..%)', [self, aId, fTokenizer.VocabSize - 1]);
  inc(fCount);
  result := Token(aId);
end;

procedure TNativeOutputParser.Finish(aLength: boolean; aPromptTokens: integer);
var
  d: TLlmStreamDelta;
begin
  if fFinished then
    ELlmClient.RaiseUtf8('%.Finish called twice', [self]);
  fFinished := true;
  FlushOpen;
  // after tool calls, trailing newlines only separated them
  SendContent(fContent.Flush(fCalls <> nil));
  Finalize(fResponse);
  FillCharFast(fResponse, SizeOf(fResponse), 0);
  fResponse.Content := fContent.Text;
  fResponse.ToolCalls := fCalls;
  if aLength then
    fResponse.FinishReason := lfrLength
  else if fCalls <> nil then
    fResponse.FinishReason := lfrToolCalls
  else
    fResponse.FinishReason := lfrStop;
  fResponse.Usage.PromptTokens := aPromptTokens;
  fResponse.Usage.CompletionTokens := fCount;
  fResponse.Usage.TotalTokens := aPromptTokens + fCount;
  fResponse.Reasoning := fReasoning.Text;
  if fReasoning.Text <> '' then
    // the message shape of vLLM's reasoning parser
    fResponse.Raw := _ObjFast(['choices', _Arr([_ObjFast(['index', 0,
      'message', _ObjFast(['role', 'assistant',
        'reasoning_content', fReasoning.Text])])])]);
  if not Assigned(fOnDelta) then
    exit;
  Finalize(d);
  FillCharFast(d, SizeOf(d), 0);
  d.FinishReason := FinishReasonText(fResponse.FinishReason);
  d.HasUsage := true;
  d.Usage := fResponse.Usage;
  Send(d);
  Finalize(d);
  FillCharFast(d, SizeOf(d), 0);
  d.Done := true;
  Send(d);
end;


{ TQwen3OutputParser }

function NeedToken(aTokenizer: TInferTokenizer; const aContent: RawUtf8): integer;
begin
  result := aTokenizer.TokenId(aContent);
  if result < 0 then
    EInferError.RaiseUtf8('TQwen3OutputParser: no % token', [aContent]);
end;

constructor TQwen3OutputParser.Create(aTokenizer: TInferTokenizer;
  const aOnDelta: TLlmStreamDeltaEvent);
begin
  inherited Create(aTokenizer, aOnDelta);
  fStop := NeedToken(aTokenizer, '<|im_end|>');
  fEndOfText := aTokenizer.TokenId('<|endoftext|>'); // -1 when absent
  fThink := -1; // no id: never matched
  fThinkEnd := -1;
  if Thinks then
  begin
    fThink := NeedToken(aTokenizer, '<think>');
    fThinkEnd := NeedToken(aTokenizer, '</think>');
  end;
  fToolCall := NeedToken(aTokenizer, '<tool_call>');
  fToolCallEnd := NeedToken(aTokenizer, '</tool_call>');
end;

class function TQwen3OutputParser.Thinks: boolean;
begin
  result := true;
end;

class function TQwen2OutputParser.Thinks: boolean;
begin
  result := false;
end;

procedure TQwen3OutputParser.EndToolCall;
var
  name, args: RawUtf8;
begin
  if ParseQwen3ToolCall(fToolBody, name, args) then
    AddToolCall(name, args)
  else
    // not a call the agent could run: keep it visible as text
    SendContent(fContent.Add(Bytes(fToolCall) + fToolBody + Bytes(fToolCallEnd)));
end;

function TQwen3OutputParser.Token(aId: integer): boolean;
begin
  result := (aId = fStop) or ((fEndOfText >= 0) and (aId = fEndOfText));
  if result then
    exit;
  if fState = qpsStart then
    if aId = fThink then
    begin
      fState := qpsReasoning;
      exit;
    end
    else
      fState := qpsContent;
  case fState of
    qpsReasoning:
      if aId = fThinkEnd then
      begin
        SendReasoning(fReasoning.Flush(true));
        fState := qpsContent;
      end
      else
        SendReasoning(fReasoning.Add(Bytes(aId)));
    qpsContent:
      if aId = fToolCall then
      begin
        fToolBody := '';
        fState := qpsToolCall;
      end
      else
        SendContent(fContent.Add(Bytes(aId)));
    qpsToolCall:
      if aId = fToolCallEnd then
      begin
        EndToolCall;
        fState := qpsContent;
      end
      else
        fToolBody := fToolBody + Bytes(aId);
  end;
end;

procedure TQwen3OutputParser.FlushOpen;
begin
  case fState of
    qpsReasoning:
      SendReasoning(fReasoning.Flush(true));
    qpsToolCall:
      // cut before </tool_call>: no call, the text stays
      SendContent(fContent.Add(Bytes(fToolCall) + fToolBody));
  end;
end;


{ THarmonyOutputParser }

function ParseHarmonyArguments(const aBody: RawUtf8; out aArguments: RawUtf8): boolean;
var
  body: RawUtf8;
begin
  result := false;
  aArguments := '';
  body := TrimU(aBody);
  if (body = '') or (body[1] <> '{') then
    exit;
  try
    // strict JSON, one object and nothing after it, in json.dumps formatting
    aArguments := PythonJson(body, {keepinexactfloats=}true);
  except
    on EInferError do
    begin
      aArguments := '';
      exit;
    end;
  end;
  result := true;
end;

function NeedHarmonyToken(aTokenizer: TInferTokenizer; const aContent: RawUtf8): integer;
begin
  result := aTokenizer.TokenId(aContent);
  if result < 0 then
    EInferError.RaiseUtf8('THarmonyOutputParser: no % token', [aContent]);
end;

constructor THarmonyOutputParser.Create(aTokenizer: TInferTokenizer;
  const aOnDelta: TLlmStreamDeltaEvent);
begin
  inherited Create(aTokenizer, aOnDelta);
  fStart := NeedHarmonyToken(aTokenizer, '<|start|>');
  fChannel := NeedHarmonyToken(aTokenizer, '<|channel|>');
  fMessage := NeedHarmonyToken(aTokenizer, '<|message|>');
  fEnd := NeedHarmonyToken(aTokenizer, '<|end|>');
  fCall := NeedHarmonyToken(aTokenizer, '<|call|>');
  fReturn := NeedHarmonyToken(aTokenizer, '<|return|>');
  fConstrain := NeedHarmonyToken(aTokenizer, '<|constrain|>');
  fEndOfText := aTokenizer.TokenId('<|endoftext|>'); // -1 when absent
  // the prompt ends with <|start|>assistant: the first header follows
  fState := hpsHeader;
end;

class procedure THarmonyOutputParser.ParseHeader(const aHeader: RawUtf8;
  out aChannel, aRecipient: RawUtf8);
var
  h, w: RawUtf8;
  p: PUtf8Char;
  nextIsChannel: boolean;
begin
  aChannel := '';
  aRecipient := '';
  // words, the control tokens as words of their own
  h := StringReplaceAll(aHeader, ['<|channel|>', ' <|channel|> ',
    '<|constrain|>', ' <|constrain|> ']);
  p := pointer(h);
  nextIsChannel := false;
  while p <> nil do
  begin
    w := GetNextItem(p, ' ');
    if w = '' then
      continue;
    if w = '<|channel|>' then
    begin
      nextIsChannel := true;
      continue;
    end;
    if nextIsChannel then
    begin
      nextIsChannel := false;
      if w <> '<|constrain|>' then
      begin
        aChannel := w;
        continue;
      end;
    end;
    if (aRecipient = '') and (length(w) > 3) and (copy(w, 1, 3) = 'to=') then
      aRecipient := copy(w, 4, maxInt);
  end;
end;

procedure THarmonyOutputParser.EndHeader;
var
  channel: RawUtf8;
begin
  ParseHeader(fHeader, channel, fRecipient);
  fHeader := '';
  fBody := '';
  if (length(fRecipient) > 10) and (copy(fRecipient, 1, 10) = 'functions.') then
  begin
    fTarget := htToolCall;
    fRecipient := copy(fRecipient, 11, maxInt);
  end
  else if (fRecipient <> '') or (channel = 'analysis') then
    fTarget := htReasoning
  else
    fTarget := htContent;
  fState := hpsBody;
end;

procedure THarmonyOutputParser.EndBody;
var
  args: RawUtf8;
begin
  case fTarget of
    htToolCall:
      if ParseHarmonyArguments(fBody, args) then
        AddToolCall(fRecipient, args)
      else
        // not a call the agent could run: keep it visible as text
        SendContent(fContent.Add(fBody));
    htReasoning:
      // a later message goes on from here: nothing waits
      SendReasoning(fReasoning.Flush(false));
  end;
  fBody := '';
end;

function THarmonyOutputParser.Token(aId: integer): boolean;
var
  stop: boolean;
begin
  result := false;
  stop := (aId = fCall) or (aId = fReturn) or
    ((fEndOfText >= 0) and (aId = fEndOfText));
  case fState of
    hpsHeader:
      if aId = fMessage then
        EndHeader
      else if stop then
        result := true // a header without a body: nothing to give out
      else if aId = fEnd then
      begin
        // a header without a body: forgotten, its recipient included
        fHeader := '';
        fState := hpsBetween;
      end
      else if aId = fStart then
        fHeader := ''
      else
        // text, <|channel|>, <|constrain|>: as their content
        fHeader := fHeader + Bytes(aId);
    hpsBody:
      if stop or (aId = fEnd) then
      begin
        EndBody;
        fState := hpsBetween;
        result := stop;
      end
      else if aId = fStart then
      begin
        // a new message without <|end|>
        EndBody;
        fState := hpsHeader;
      end
      else
        case fTarget of
          htToolCall:
            fBody := fBody + Bytes(aId);
          htReasoning:
            SendReasoning(fReasoning.Add(Bytes(aId)));
        else
          SendContent(fContent.Add(Bytes(aId)));
        end;
    hpsBetween:
      if aId = fStart then
      begin
        fHeader := '';
        fState := hpsHeader;
      end
      else if stop then
        result := true
      else
        // text outside of a message: the model's, keep it
        SendContent(fContent.Add(Bytes(aId)));
  end;
end;

procedure THarmonyOutputParser.FlushOpen;
begin
  if fState <> hpsBody then
    exit;
  case fTarget of
    htReasoning:
      SendReasoning(fReasoning.Flush(false));
    htToolCall:
      // cut before <|call|>: no call, the text stays
      SendContent(fContent.Add(fBody));
  end;
end;


{ request helpers }

function NativeEnableThinking(const aRequest: TLlmChatRequest;
  aDefault: boolean): boolean;
var
  kw: PDocVariantData;
  i: integer;
begin
  result := aDefault;
  kw := _Safe(aRequest.Extra)^.O['chat_template_kwargs'];
  i := kw^.GetValueIndex('enable_thinking');
  if i < 0 then
    exit;
  if TVarData(kw^.Values[i]).VType <> varBoolean then
    ELlmClient.RaiseUtf8('chat_template_kwargs.enable_thinking: % is no boolean',
      [VariantSaveJson(kw^.Values[i])]);
  result := kw^.Values[i];
end;

function NativeTemplateOptions(const aRequest: TLlmChatRequest;
  aDefaultThinking: boolean; const aDefaultEffort: RawUtf8): TChatTemplateOptions;
var
  x, kw: PDocVariantData;
  top, inner: RawUtf8;
  hasTop, hasInner: boolean;

  function Text(aDoc: PDocVariantData; const aName, aWhere: RawUtf8;
    out aValue: RawUtf8): boolean;
  var
    i: integer;
  begin
    aValue := '';
    i := aDoc^.GetValueIndex(aName);
    result := (i >= 0) and not VarIsEmptyOrNull(aDoc^.Values[i]);
    if not result then
      exit;
    if not VarIsString(aDoc^.Values[i]) then
      ELlmClient.RaiseUtf8('%%: % is no string',
        [aWhere, aName, VariantSaveJson(aDoc^.Values[i])]);
    VariantToUtf8(aDoc^.Values[i], aValue);
  end;

begin
  result := ChatTemplateOptions(NativeEnableThinking(aRequest, aDefaultThinking));
  x := _Safe(aRequest.Extra);
  kw := x^.O['chat_template_kwargs'];
  // an empty value is no value: the default stays
  hasTop := Text(x, 'reasoning_effort', '', top) and (top <> '');
  hasInner := Text(kw, 'reasoning_effort', 'chat_template_kwargs.', inner) and
    (inner <> '');
  if hasTop and hasInner and (top <> inner) then
    ELlmClient.RaiseUtf8('reasoning_effort "%" and chat_template_kwargs.' +
      'reasoning_effort "%" differ', [top, inner]);
  if hasTop then
    result.ReasoningEffort := top
  else if hasInner then
    result.ReasoningEffort := inner
  else
    result.ReasoningEffort := aDefaultEffort;
  Text(kw, 'model_identity', 'chat_template_kwargs.', result.ModelIdentity);
end;

function NativeChatPrompt(aTemplate: TChatTemplate;
  const aRequest: TLlmChatRequest; aDefaultThinking: boolean;
  const aDefaultEffort: RawUtf8): TIntegerDynArray;
var
  i: PtrInt;
  options: TChatTemplateOptions;
begin
  if aRequest.Messages = nil then
    ELlmClient.RaiseUtf8('NativeChatPrompt: no messages', []);
  for i := 0 to high(aRequest.Messages) do
    if aRequest.Messages[i].Images <> nil then
      ELlmClient.RaiseUtf8('NativeChatPrompt: message % carries images - ' +
        'the native engine runs text models only', [i]);
  if aRequest.ResponseFormat <> '' then
    ELlmClient.RaiseUtf8('NativeChatPrompt: ResponseFormat is not supported ' +
      'by the native engine yet (no constrained decoding)', []);
  options := NativeTemplateOptions(aRequest, aDefaultThinking, aDefaultEffort);
  try
    result := aTemplate.EncodeWith(aRequest.Messages, aRequest.Tools,
      {addgenerationprompt=}true, options);
  except
    on E: EInferError do
      ELlmClient.RaiseUtf8('NativeChatPrompt: %', [E.Message]);
  end;
end;

function ParseQwen3ToolCall(const aBody: RawUtf8;
  out aName, aArguments: RawUtf8): boolean;
var
  body, canon: RawUtf8;
  doc: TDocVariantData;
  iname, iargs: integer;
  p, e: PUtf8Char;
begin
  result := false;
  aName := '';
  aArguments := '';
  body := TrimU(aBody);
  if (body = '') or (body[1] <> '{') then
    exit;
  try
    // strict JSON (no duplicate key, valid UTF-8, no raw control character)
    // in json.dumps formatting, keys unescaped; a float json.dumps would
    // round stays as the model wrote it
    canon := PythonJson(body, {keepinexactfloats=}true);
  except
    on EInferError do
      exit;
  end;
  // two top-level keys, counted on the text: the document skips a key with
  // an empty name
  if JsonObjectPropCount(PUtf8Char(pointer(canon)) + 1, nil, true) <> 2 then
    exit;
  // only names and kinds are read: JSON_FAST keeps a float like 1e999 as text
  // where a double would refuse it
  if not doc.InitJson(canon, JSON_FAST) or
     not doc.IsObject or
     (doc.Count <> 2) then
    exit;
  // exact names: the document lookup would ignore the case
  iname := -1;
  iargs := -1;
  if doc.Names[0] = 'name' then
    iname := 0
  else if doc.Names[1] = 'name' then
    iname := 1;
  if doc.Names[0] = 'arguments' then
    iargs := 0
  else if doc.Names[1] = 'arguments' then
    iargs := 1;
  if (iname < 0) or (iargs < 0) or
     not _Safe(doc.Values[iargs])^.IsObject then
    exit;
  // the kind of the name on the text: JSON_FAST keeps a number it cannot hold
  // (0.00001) as a string too
  p := JsonObjectItem(pointer(canon), 'name');
  if (p = nil) or (p^ <> '"') then
    exit;
  VariantToUtf8(doc.Values[iname], aName);
  if aName = '' then
    exit;
  // the arguments as they stand in the canonical text, digits untouched
  p := JsonObjectItem(pointer(canon), 'arguments');
  if p = nil then
    exit;
  e := GotoEndJsonItem(p);
  if e = nil then
    exit;
  FastSetString(aArguments, p, e - p);
  result := true;
end;


{ NativeSampling }

function NativeSampling(const aRequest: TLlmChatRequest;
  const aDefaults: TInferSampling): TInferSampling;
var
  x: PDocVariantData;
  seen: boolean;

  function Num(const aName: RawUtf8; var aValue: double): boolean;
  var
    i: integer;
  begin
    result := false;
    i := x^.GetValueIndex(aName);
    if (i < 0) or VarIsEmptyOrNull(x^.Values[i]) then
      exit;
    // a boolean converts to 0 or 1 in mORMot: refused as no number
    if (TVarData(x^.Values[i]).VType = varBoolean) or
       not VariantToDouble(x^.Values[i], aValue) or not InferIsFinite(aValue) then
      ELlmClient.RaiseUtf8('NativeSampling: % is no number', [aName]);
    result := true;
  end;

  function Int(const aName: RawUtf8; var aValue: Int64): boolean;
  var
    i: integer;
  begin
    result := false;
    i := x^.GetValueIndex(aName);
    if (i < 0) or VarIsEmptyOrNull(x^.Values[i]) then
      exit;
    if (TVarData(x^.Values[i]).VType = varBoolean) or
       not VariantToInt64(x^.Values[i], aValue) then
      ELlmClient.RaiseUtf8('NativeSampling: % is no integer', [aName]);
    result := true;
  end;

var
  d: double;
  v: Int64;
begin
  result := aDefaults;
  if not InferIsFinite(aRequest.Temperature) then
    ELlmClient.RaiseUtf8('NativeSampling: Temperature is no number', []);
  if aRequest.Temperature >= 0 then
    result.Temperature := aRequest.Temperature;
  x := _Safe(aRequest.Extra);
  if Num('top_p', d) then
    result.TopP := d;
  if Num('min_p', d) then
    result.MinP := d;
  if Int('top_k', v) then
    if (v < 0) or (v > MaxInt) then
      ELlmClient.RaiseUtf8('NativeSampling: top_k %', [v])
    else
      result.TopK := v;
  seen := Int('seed', v);
  if seen then
    result.Seed := v
  else
    result.Seed := Int64(Random64);
  try
    TInferSampler.Create(result).Free; // the ranges, once
  except
    on E: EInferError do
      ELlmClient.RaiseUtf8('NativeSampling: %', [E.Message]);
  end;
end;


{ NativeDecode }

function NativeDecode(aModel: TDecoderModel; const aPrompt: TIntegerDynArray;
  aMaxNew: PtrInt; aParser: TNativeOutputParser; const aSampling: TInferSampling;
  aContext: TDecoderContext; aMaxContext: PtrInt): PtrInt;
var
  logits: TSingleDynArray;
  ctx: TDecoderContext;
  sampler: TInferSampler;
  n, limit, ids: PtrInt;
  id: integer;
  stopped: boolean;
begin
  limit := aModel.Config.MaxPositions;
  if (aMaxContext > 0) and (aMaxContext < limit) then
    limit := aMaxContext;
  if aPrompt = nil then
    ELlmClient.RaiseUtf8('NativeDecode: empty prompt', []);
  if length(aPrompt) > limit then
    ELlmClient.RaiseUtf8('NativeDecode: a prompt of % tokens exceeds the ' +
      'context of %', [length(aPrompt), limit]);
  if aMaxNew <= 0 then
    ELlmClient.RaiseUtf8('NativeDecode: % new tokens', [aMaxNew]);
  // token n is fed at position P + n - 1 only when another one follows
  if aMaxNew > limit - length(aPrompt) + 1 then
    aMaxNew := limit - length(aPrompt) + 1;
  SetLength(logits, aModel.Config.VocabSize);
  // the rows a tokenizer has no id for (padding) are never drawn
  ids := aModel.Config.VocabSize;
  if aParser.Tokenizer.VocabSize < ids then
    ids := aParser.Tokenizer.VocabSize;
  sampler := nil;
  ctx := aContext;
  if ctx = nil then
    ctx := aModel.NewContext;
  try
    sampler := TInferSampler.Create(aSampling);
    // the cached start of the prompt stays, the rest is computed
    result := aModel.Reuse(ctx, pointer(aPrompt), length(aPrompt));
    aModel.Forward(ctx, @aPrompt[result], length(aPrompt) - result, pointer(logits));
    n := 0;
    repeat
      id := sampler.Next(pointer(logits), ids);
      inc(n);
      stopped := aParser.Push(id);
      if stopped or (n = aMaxNew) then
        break;
      aModel.Forward(ctx, @id, 1, pointer(logits));
    until false;
  finally
    sampler.Free;
    if ctx <> aContext then
      ctx.Free;
  end;
  aParser.Finish(not stopped, length(aPrompt));
end;


{ TNativePromptCache }

constructor TNativePromptCache.Create(aSlots: integer);
begin
  inherited Create;
  SetSlots(aSlots);
end;

destructor TNativePromptCache.Destroy;
begin
  Clear;
  inherited Destroy;
end;

procedure TNativePromptCache.SetSlots(aSlots: integer);
var
  i, n: PtrInt;
  drop: TDecoderContextDynArray;
begin
  if aSlots < 0 then
    ELlmClient.RaiseUtf8('%.Slots = %', [self, aSlots]);
  drop := nil;
  fSafe.Lock;
  try
    fSlots := aSlots;
    n := length(fIdle) - aSlots;
    if n > 0 then
    begin
      // the least recently used go
      SetLength(drop, n);
      for i := 0 to n - 1 do
        drop[i] := fIdle[i];
      for i := n to high(fIdle) do
        fIdle[i - n] := fIdle[i];
      SetLength(fIdle, aSlots);
    end;
  finally
    fSafe.UnLock;
  end;
  for i := 0 to high(drop) do
    drop[i].Free; // outside the lock: freeing a large cache takes a while
end;

procedure TNativePromptCache.Clear;
var
  i: PtrInt;
  drop: TDecoderContextDynArray;
begin
  drop := Take;
  for i := 0 to high(drop) do
    drop[i].Free;
end;

function TNativePromptCache.Take: TDecoderContextDynArray;
begin
  fSafe.Lock;
  try
    result := fIdle;
    fIdle := nil;
  finally
    fSafe.UnLock;
  end;
end;

function TNativePromptCache.Acquire(const aPrompt: TIntegerDynArray): TDecoderContext;
var
  i, best, common, longest: PtrInt;
begin
  result := nil;
  fSafe.Lock;
  try
    best := -1;
    longest := 0;
    for i := 0 to high(fIdle) do
    begin
      // as Reuse keeps them: the last token is always computed again
      common := fIdle[i].Usable(pointer(aPrompt), length(aPrompt) - 1);
      if common > longest then
      begin
        longest := common;
        best := i;
      end;
    end;
    // no shared token: the buffers of the least recently used, once full
    if (best < 0) and (fIdle <> nil) and (length(fIdle) >= fSlots) then
      best := 0;
    if best >= 0 then
    begin
      result := fIdle[best];
      for i := best to high(fIdle) - 1 do
        fIdle[i] := fIdle[i + 1];
      SetLength(fIdle, length(fIdle) - 1);
    end;
  finally
    fSafe.UnLock;
  end;
  if result = nil then
    if fModel <> nil then
      result := fModel.NewContext
    else
      result := TDecoderContext.Create;
end;

procedure TNativePromptCache.Release(aContext: TDecoderContext; aReused: PtrInt);
var
  drop: TDecoderContext;
  i: PtrInt;
begin
  if aContext = nil then
    exit;
  drop := nil;
  fSafe.Lock;
  try
    inc(fRequests);
    if aReused > 0 then
    begin
      inc(fHits);
      inc(fReused, aReused);
    end;
    // no slot, or a context of another device than the model's now
    if (fSlots = 0) or ((fModel <> nil) and (aContext.Device <> fModel.Device)) then
      drop := aContext
    else
    begin
      if length(fIdle) >= fSlots then
      begin
        drop := fIdle[0];
        for i := 1 to high(fIdle) do
          fIdle[i - 1] := fIdle[i];
        fIdle[high(fIdle)] := aContext;
      end
      else
      try
        SetLength(fIdle, length(fIdle) + 1);
        fIdle[high(fIdle)] := aContext;
      except
        drop := aContext; // no memory to keep it: caching is best effort
      end;
    end;
  finally
    fSafe.UnLock;
  end;
  drop.Free;
end;

function TNativePromptCache.GetRequests: Int64;
begin
  fSafe.Lock; // an Int64 is no atomic read on a 32-bit CPU
  result := fRequests;
  fSafe.UnLock;
end;

function TNativePromptCache.GetHits: Int64;
begin
  fSafe.Lock;
  result := fHits;
  fSafe.UnLock;
end;

function TNativePromptCache.GetReused: Int64;
begin
  fSafe.Lock;
  result := fReused;
  fSafe.UnLock;
end;


{ TNativeLlmClient }

constructor TNativeLlmClient.Create(const aFolder: TFileName;
  const aModelName: RawUtf8; aBackend: TInferBackend; aWeights: TInferDType);
var
  dir: TFileName;
begin
  inherited Create;
  dir := IncludeTrailingPathDelimiter(aFolder);
  fMaxTokens := NATIVE_LLM_DEFAULT_MAX_TOKENS;
  fEnableThinking := true;
  fBackend := aBackend;
  if fBackend = nil then
  begin
    fBackend := TInferCpuFastBackend.Create;
    fOwnBackend := true;
  end;
  // the family of config.json: its tokenizer and template before the weights
  SetFamily(DecoderModelClassOfFolder(dir), dir + 'config.json');
  fTokenizer := fFamily.Tokenizer.Create(dir + 'tokenizer.json');
  fTemplate := fFamily.Template.Create(fTokenizer);
  fModel := fFamily.Model.Create(dir, fBackend, aWeights);
  fSampling := InferSamplingFromConfig(dir + 'generation_config.json');
  fCache := TNativePromptCache.Create;
  fCache.Model := fModel;
  // the stop and control tokens: fail at load, not at the first request
  fFamily.Parser.Create(fTokenizer).Free;
  fConfig.DefaultModel := aModelName;
  if fConfig.DefaultModel = '' then
    fConfig.DefaultModel := StringToUtf8(ExtractFileName(ExcludeTrailingPathDelimiter(aFolder)));
end;

constructor TNativeLlmClient.CreateFromGguf(const aFileName: TFileName;
  const aModelName: RawUtf8; aBackend: TInferBackend);
begin
  inherited Create;
  fMaxTokens := NATIVE_LLM_DEFAULT_MAX_TOKENS;
  fEnableThinking := true;
  fBackend := aBackend;
  if fBackend = nil then
  begin
    fBackend := TInferCpuFastBackend.Create;
    fOwnBackend := true;
  end;
  fModel := NewDecoderModelFromGguf(aFileName, fBackend);
  SetFamily(TDecoderModelClass(fModel.ClassType), aFileName);
  fTokenizer := fFamily.Tokenizer.CreateFromGguf(fModel.Gguf);
  if fTokenizer.VocabSize > fModel.Config.VocabSize then
    ELlmClient.RaiseUtf8('%: % tokens but % embedding rows in %',
      [self, fTokenizer.VocabSize, fModel.Config.VocabSize, aFileName]);
  fTemplate := fFamily.Template.Create(fTokenizer);
  fSampling := InferSamplingGreedy;
  fCache := TNativePromptCache.Create;
  fCache.Model := fModel;
  // the stop and control tokens: fail at load, not at the first request
  fFamily.Parser.Create(fTokenizer).Free;
  fConfig.DefaultModel := aModelName;
  if fConfig.DefaultModel = '' then
    fConfig.DefaultModel := StringToUtf8(ChangeFileExt(ExtractFileName(aFileName), ''));
end;

procedure TNativeLlmClient.SetFamily(aModel: TDecoderModelClass;
  const aSource: TFileName);
begin
  if not NativeFamily(aModel, fFamily) then
    ELlmClient.RaiseUtf8('%: no chat family registered for % (%)',
      [self, aModel.ModelType, aSource]);
end;

destructor TNativeLlmClient.Destroy;
begin
  fCache.Free; // device contexts first, then their device
  if fModel <> nil then
    fModel.Device := nil;
  fDevice.Free;
  fModel.Free;
  fTemplate.Free;
  fTokenizer.Free;
  if fOwnBackend then
    fBackend.Free;
  inherited Destroy;
end;

procedure TNativeLlmClient.Generate(const aRequest: TLlmChatRequest;
  aParser: TNativeOutputParser);
var
  maxNew, reused: PtrInt;
  prompt: TIntegerDynArray;
  sampling: TInferSampling;
  ctx: TDecoderContext;
begin
  maxNew := aRequest.MaxTokens;
  if maxNew <= 0 then
    maxNew := fMaxTokens;
  prompt := NativeChatPrompt(fTemplate, aRequest, fEnableThinking,
    fReasoningEffort);
  sampling := NativeSampling(aRequest, fSampling);
  // Device stays as it is while this request runs (SetDevice refuses)
  fSafe.Lock;
  inc(fActive);
  fSafe.UnLock;
  try
    ctx := fCache.Acquire(prompt);
    try
      reused := NativeDecode(fModel, prompt, maxNew, aParser, sampling, ctx,
        fMaxContext);
    except
      ctx.Free; // a failed request leaves nothing in the cache
      raise;
    end;
    fCache.Release(ctx, reused);
  finally
    fSafe.Lock;
    dec(fActive);
    fSafe.UnLock;
  end;
end;

function TNativeLlmClient.ChatComplete(
  const aRequest: TLlmChatRequest): TLlmChatResponse;
var
  parser: TNativeOutputParser;
begin
  parser := fFamily.Parser.Create(fTokenizer);
  try
    Generate(aRequest, parser);
    result := parser.Response;
    result.Model := fConfig.DefaultModel;
  finally
    parser.Free;
  end;
end;

procedure TNativeLlmClient.ChatStream(const aRequest: TLlmChatRequest;
  const aOnDelta: TLlmStreamDeltaEvent);
var
  parser: TNativeOutputParser;
begin
  parser := fFamily.Parser.Create(fTokenizer, aOnDelta);
  try
    Generate(aRequest, parser);
  finally
    parser.Free;
  end;
end;

procedure TNativeLlmClient.SetDevice(aDevice: TDecoderDevice);
var
  old: TDecoderDevice;
  drop: TDecoderContextDynArray;
  i: PtrInt;
begin
  if (aDevice <> nil) and (aDevice.Model <> fModel) then
    ELlmClient.RaiseUtf8('%.Device: % computes another model', [self, aDevice]);
  old := nil;
  drop := nil;
  fSafe.Lock;
  try
    // the same instance again (two callers, say): nothing to free
    if aDevice <> fDevice then
    begin
      // a running request holds a context of the old device (or the CPU)
      if fActive > 0 then
        ELlmClient.RaiseUtf8('%.Device: % requests are running', [self, fActive]);
      // first the device of new contexts: a context of the old one given
      // back from now on is dropped by Release (Model.Device differs)
      fModel.Device := aDevice;
      // the cached contexts belong to the old device (or the CPU): out of
      // the cache here, freed below - not under this spin lock
      drop := fCache.Take;
      old := fDevice;
      fDevice := aDevice;
    end;
  finally
    fSafe.UnLock;
  end;
  // the contexts first, then their device
  for i := 0 to high(drop) do
    drop[i].Free;
  old.Free;
end;

function TNativeLlmClient.Config: TLlmProviderConfig;
begin
  result := fConfig;
end;

{ families }

var
  NativeFamilies: array of TNativeFamily;

procedure RegisterNativeFamily(aModel: TDecoderModelClass;
  aTokenizer: TInferTokenizerClass; aTemplate: TChatTemplateClass;
  aParser: TNativeOutputParserClass);
var
  n: PtrInt;
begin
  if (aModel = nil) or (aTokenizer = nil) or (aTemplate = nil) or
     (aParser = nil) then
    exit;
  n := length(NativeFamilies);
  SetLength(NativeFamilies, n + 1);
  NativeFamilies[n].Model := aModel;
  NativeFamilies[n].Tokenizer := aTokenizer;
  NativeFamilies[n].Template := aTemplate;
  NativeFamilies[n].Parser := aParser;
end;

function NativeFamily(aModel: TDecoderModelClass;
  out aFamily: TNativeFamily): boolean;
var
  i: PtrInt;
begin
  for i := 0 to high(NativeFamilies) do
    if NativeFamilies[i].Model = aModel then
    begin
      aFamily := NativeFamilies[i];
      result := true;
      exit;
    end;
  FillCharFast(aFamily, SizeOf(aFamily), 0);
  result := false;
end;


initialization
  RegisterNativeFamily(TQwen3Model, TBpeTokenizer, TQwen3ChatTemplate,
    TQwen3OutputParser);
  RegisterNativeFamily(TQwen3MoeModel, TBpeTokenizer, TQwen3ChatTemplate,
    TQwen3OutputParser);
  RegisterNativeFamily(TQwen2Model, TBpeTokenizer, TQwen2ChatTemplate,
    TQwen2OutputParser);
  RegisterNativeFamily(TGptOssModel, TBpeTokenizer, THarmonyChatTemplate,
    THarmonyOutputParser);

end.
