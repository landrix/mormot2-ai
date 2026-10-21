/// native inference engine - chat templates of the supported decoder models
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - TChatTemplate is what the client of every decoder family calls; a family
//   brings its own descendant (Build) - the Jinja template of a model is not
//   run: TQwen3ChatTemplate writes the
//   same text in Pascal (tokenizer_config.json of Qwen/Qwen3-0.6B revision
//   c1899de289a04d12100db370d81485cdf75e47ca), gate tests/data/infer-ref/qwen3-chat;
//   TQwen2ChatTemplate that of Qwen2.5 (Qwen/Qwen2.5-0.5B-Instruct revision
//   7ae557604adf67be50417f59c2c2f167def9a775), gate tests/data/infer-ref/qwen25-chat;
//   THarmonyChatTemplate that of gpt-oss (openai/gpt-oss-20b revision
//   6cee5e81ee83917806bbde320786a8fb61efebee, chat_template.jinja), gate
//   tests/data/infer-ref/gpt-oss-chat; TGemma4ChatTemplate that of Gemma 4
//   (google/gemma-4-26B-A4B-it revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52,
//   chat_template.jinja - the same file in gemma-4-31B-it and gemma-4-12B-it),
//   gate tests/data/infer-ref/gemma4-chat
// - one difference by design: the template emits the control tokens
//   (<|im_start|> <|im_end|> <think> </think> <tool_call> </tool_call>
//   <tool_response> </tool_response>) as ids, and every message text is text.
//   Transformers renders a string and tokenizes it, so a literal "<|im_end|>"
//   in user content becomes a control token there - here it stays text, and a
//   user cannot close his own turn or fake a tool response
// - tools are serialized like Python json.dumps (", " and ": ", ensure_ascii
//   off, key order kept), as the template's tojson filter does - PythonJson
unit mormot.ai.infer.chat;

{
  *****************************************************************************

    - PythonJson: JSON re-serialized the way Python json.dumps writes it
    - TChatTemplate: messages and tools to token ids (and text), per family
    - TQwen3ChatTemplate: the template of Qwen3
    - TQwen2ChatTemplate: the template of Qwen2.5
    - THarmonyChatTemplate: the template of gpt-oss (Harmony)
    - TGemma4ChatTemplate: the template of Gemma 4

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

{$ifndef FPC}{$ifdef CPUX64}
  {$EXCESSPRECISION OFF} // single expressions in single, as FPC computes them
{$endif}{$endif}

uses
  sysutils,
  mormot.core.base,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.llm.types,
  mormot.ai.infer.tensor,
  mormot.ai.infer.unicode,
  mormot.ai.infer.tokenizer;

type
  /// what a request may set in a chat template beyond messages and tools
  // - a family reads what it knows and ignores the rest
  TChatTemplateOptions = record
    /// false asks the model to answer without reasoning first (Qwen3: an empty
    // <think> block); gpt-oss always reasons and ignores it
    EnableThinking: boolean;
    /// gpt-oss: 'low', 'medium' or 'high' - '' for the template's medium
    ReasoningEffort: RawUtf8;
    /// gpt-oss: the first line of the system message - '' for the template's
    // own ("You are ChatGPT, a large language model trained by OpenAI.")
    ModelIdentity: RawUtf8;
    /// gpt-oss: the date of the system message as yyyy-mm-dd - '' for today
    // (local time, as the template's strftime_now)
    CurrentDate: RawUtf8;
    /// Gemma 4: the reasoning of an assistant turn with tool calls is written
    // back also before the last user message (preserve_thinking)
    PreserveThinking: boolean;
  end;

  /// the output of one template run: text is buffered and encoded in one
  // piece up to the next control token, exactly as the tokenizer would cut
  // the rendered text
  TChatSegments = object
    Tokenizer: TInferTokenizer;
    WithIds: boolean;
    Ids: TIntegerDynArray;
    Count: PtrInt;
    Pending, Text: RawUtf8;
    procedure Add(const aText: RawUtf8);
    procedure Control(aId: integer; const aName: RawUtf8);
    procedure Flush;
  end;

  /// the chat template of a decoder family: a conversation as token ids
  // - a family overrides Build (and Create, for its control tokens)
  // - thread-safe: every call works on its own TChatSegments
  TChatTemplate = class
  protected
    fTokenizer: TInferTokenizer;
    /// write the conversation into aOut: text with Add, control tokens with
    // Control
    procedure Build(var aOut: TChatSegments; const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
      const aOptions: TChatTemplateOptions); virtual; abstract;
    /// the id of a control token - raises EInferError when the tokenizer has
    // none
    function Need(const aContent: RawUtf8): integer;
  public
    /// keep aTokenizer (not owned) - raises EInferError when nil
    constructor Create(aTokenizer: TInferTokenizer); virtual;
    /// whether the family thinks when a request does not say (enable_thinking)
    // - true by default; false for a template that thinks only on request
    class function DefaultEnableThinking: boolean; virtual;
    /// the token ids of a conversation
    // - aAddGenerationPrompt opens the assistant turn; aEnableThinking = false
    // asks the model to answer without reasoning first (Qwen3: an empty
    // <think> block)
    // - raises EInferError for a tool whose JSON is invalid or not supported
    function Encode(const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
      aEnableThinking: boolean = true): TIntegerDynArray;
    /// the same conversation as text, control tokens written as their content
    // - what transformers renders before tokenizing; for logs and tests
    function Render(const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
      aEnableThinking: boolean = true): RawUtf8;
    /// the token ids of a conversation, with every option of a request
    function EncodeWith(const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
      const aOptions: TChatTemplateOptions): TIntegerDynArray;
    /// the same conversation as text, with every option of a request
    function RenderWith(const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
      const aOptions: TChatTemplateOptions): RawUtf8;
    /// the tokenizer the ids belong to
    property Tokenizer: TInferTokenizer
      read fTokenizer;
  end;

  /// a chat template class, as a decoder family registers it
  TChatTemplateClass = class of TChatTemplate;

  /// the chat template of Qwen3: ChatML with <think> blocks and <tools> in the
  // system turn
  // - the Reasoning of an assistant message is its reasoning_content: written
  // as its <think> block in the turns after the last user query, and its
  // content is then not searched for one; '' is none (the content may still
  // hold a <think> block)
  TQwen3ChatTemplate = class(TChatTemplate)
  protected
    fImStart, fImEnd, fThink, fThinkEnd, fToolCall, fToolCallEnd,
    fToolResponse, fToolResponseEnd: integer;
    procedure Build(var aOut: TChatSegments; const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
      const aOptions: TChatTemplateOptions); override;
  public
    /// look up the control tokens in aTokenizer (not owned) - raises
    // EInferError when one is missing
    constructor Create(aTokenizer: TInferTokenizer); override;
  end;

  /// the chat template of Qwen2.5 (Instruct): ChatML, always a system turn
  // (the model's default text without a system message), <tools> in it, no
  // reasoning - aEnableThinking is ignored
  // - the arguments of a tool call in the history go through the template's
  // tojson: written like Python json.dumps (PythonJson, a float kept as it
  // stands), '' as {}
  // - <tool_response> and </tool_response> are text for Qwen2.5 (no added
  // tokens, as the HF tokenizer splits them); a tokenizer that has them gets
  // their ids
  TQwen2ChatTemplate = class(TChatTemplate)
  protected
    fImStart, fImEnd, fToolCall, fToolCallEnd, fToolResponse,
    fToolResponseEnd: integer;
    procedure Build(var aOut: TChatSegments; const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
      const aOptions: TChatTemplateOptions); override;
  public
    /// look up the control tokens in aTokenizer (not owned) - raises
    // EInferError when one is missing
    constructor Create(aTokenizer: TInferTokenizer); override;
  end;

  /// the chat template of gpt-oss: the Harmony format of its
  // chat_template.jinja - a system message (identity, knowledge cutoff, date,
  // reasoning effort, channels), a developer message (the instructions of a
  // first system message, the tools as a TypeScript namespace "functions"),
  // then the turns; an assistant turn is its final message, a turn with tool
  // calls an analysis message (its Reasoning, else its content - unless a
  // later assistant turn answers without calls) and the calls to
  // functions.NAME
  // - Reasoning is the template's "thinking": dropped in a turn without calls,
  // except in the last one when no generation prompt follows (the analysis
  // before the final message, as in training); '' is no thinking (the template
  // would write an empty analysis message for a present but empty one there)
  // - the tool schemas are written as the template writes them, its quirks
  // included: the Python values of the schema (str() for enum and type lists,
  // tojson for defaults), the indentation the template puts into nested
  // objects and oneOf; what Python would write as a repr (an array or object
  // in an enum) or the template could not take (a description that is no
  // string, a "required", oneOf or enum that is no array) raises EInferError
  // - differences by design: control tokens come as ids and every text is text
  // (the template refuses an assistant content or thinking holding
  // "<|channel|>..."; here it is text); an empty tool description is written as "// " (transformers
  // fails without one); several calls in one assistant turn each get their
  // message (the template writes the first only) and a tool result takes the
  // name of the call with its ToolCallId (the template: the last call); a
  // system message after the first raises (the template drops it); a
  // ReasoningEffort other than low, medium or high and a CurrentDate other
  // than yyyy-mm-dd raise, an empty one is the default (as an empty
  // ModelIdentity); parameters that are truthy but no object raise (the
  // template would write "() => any"); a turn with calls, Reasoning and
  // content writes the reasoning as analysis and the content as a commentary
  // preamble, as gpt-oss wrote them (the template refuses the pair)
  THarmonyChatTemplate = class(TChatTemplate)
  protected
    fStart, fChannel, fMessage, fEnd, fCall, fReturn: integer;
    procedure Build(var aOut: TChatSegments; const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
      const aOptions: TChatTemplateOptions); override;
  public
    /// look up the Harmony control tokens in aTokenizer (not owned) - raises
    // EInferError when one is missing
    constructor Create(aTokenizer: TInferTokenizer); override;
  end;

  /// the chat template of Gemma 4: its chat_template.jinja - <bos>, a system
  // turn when thinking is on, tools are given or the first message is a
  // system one (<|think|>, the trimmed instructions, each tool as
  // <|tool>declaration:NAME{...}<tool|>), then <|turn>user / <|turn>model
  // turns up to <turn|>; an assistant turn writes its Reasoning as
  // <|channel>thought ... <channel|> (only after the last user message, or
  // with PreserveThinking before calls), its calls as
  // <|tool_call>call:NAME{key:value}<tool_call|>, the results of the tool
  // messages after it as <|tool_response>response:NAME{value:...}
  // <tool_response|>, then its content without <|channel> blocks; the model
  // turn goes on after the results (no <turn|>), and the next assistant
  // message continues it
  // - EnableThinking is off unless a request sets it (the template's default);
  // off, the generation prompt closes an empty thought channel
  // - strings of the tool schemas and arguments sit between <|"|> tokens; the
  // schema keys in the order of Jinja's dictsort (Python lower(): ASCII,
  // Latin-1, Latin Extended-A, Greek and Cyrillic - another upper case
  // letter or a capital sigma in a key raises), the type names upper case
  // (ASCII only), numbers
  // as Python writes them; what the template would write as a Python repr (a
  // list or object as a description or type, outside an item type list) or
  // could not take (parameters or properties that are truthy but no object,
  // a "required" that is no array of scalars) raises EInferError
  // - differences by design: control tokens come as ids and every text is
  // text; a tool message that answers no assistant turn with calls raises
  // (the template drops it); a system message after the first is rendered as
  // a system turn, as the template does
  TGemma4ChatTemplate = class(TChatTemplate)
  protected
    fBos, fTurn, fTurnEnd, fThink, fChannel, fChannelEnd, fTool, fToolEnd,
    fToolCall, fToolCallEnd, fToolResponse, fToolResponseEnd, fQuote: integer;
    procedure Build(var aOut: TChatSegments; const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
      const aOptions: TChatTemplateOptions); override;
  public
    /// look up the control tokens of Gemma 4 in aTokenizer (not owned) -
    // raises EInferError when one is missing
    constructor Create(aTokenizer: TInferTokenizer); override;
    /// false: the template thinks only when a request asks for it
    class function DefaultEnableThinking: boolean; override;
  end;

/// the options of a request that only sets EnableThinking
function ChatTemplateOptions(aEnableThinking: boolean): TChatTemplateOptions;

/// re-serialize JSON the way Python json.dumps(json.loads(aJson)) writes it
// - separators ", " and ": ", non-ASCII kept (ensure_ascii=False), only " \
// and control characters escaped (\u00xx in lower case), key order kept
// - numbers: an integer literal is written as such ("-0" as "0"); a float as
// Python's repr - done on the decimal digits, so it is exact on every
// compiler, and a float with more than 15 significant digits or outside
// 1e-307..1e308 raises EInferError instead of guessing the binary rounding
// - also raises EInferError for invalid JSON, duplicate keys, NaN/Infinity,
// lone surrogates or invalid UTF-8
// - aKeepInexactFloats: such a float is written as it stands in aJson instead
// of raising - for text that must pass on as it is (a model's tool call);
// every other rule stays, an exponent of more than six digits still raises
function PythonJson(const aJson: RawUtf8;
  aKeepInexactFloats: boolean = false): RawUtf8;

/// a JSON string literal the way Python json.dumps writes it (see PythonJson)
// - raises EInferError for invalid UTF-8
function PythonJsonString(const aText: RawUtf8): RawUtf8;


implementation

{ PythonJson }

const
  HEX_LOWER: array[0..15] of AnsiChar = '0123456789abcdef';
  /// deepest object / array nesting PythonJson accepts
  JSON_MAX_DEPTH = 128;

procedure AddEscaped(var aOut: RawUtf8; p: PUtf8Char; aLen: PtrInt);
var
  i, start: PtrInt;
  c: AnsiChar;
  esc: string[7];
begin
  Append(aOut, '"');
  start := 0;
  for i := 0 to aLen - 1 do
  begin
    c := p[i];
    if (c >= ' ') and (c <> '"') and (c <> '\') then
      continue;
    if i > start then
      Append(aOut, @p[start], i - start);
    case c of
      '"':
        esc := '\"';
      '\':
        esc := '\\';
      #10:
        esc := '\n';
      #13:
        esc := '\r';
      #9:
        esc := '\t';
      #8:
        esc := '\b';
      #12:
        esc := '\f';
    else
      begin
        esc := '\u00';
        esc := esc + HEX_LOWER[ord(c) shr 4] + HEX_LOWER[ord(c) and 15];
      end;
    end;
    Append(aOut, @esc[1], length(esc));
    start := i + 1;
  end;
  if aLen > start then
    Append(aOut, @p[start], aLen - start);
  Append(aOut, '"');
end;

function PythonJsonString(const aText: RawUtf8): RawUtf8;
begin
  if not IsStrictUtf8(pointer(aText), length(aText)) then
    EInferError.RaiseUtf8('PythonJsonString: invalid UTF-8', []);
  result := '';
  AddEscaped(result, pointer(aText), length(aText));
end;

type
  /// the kind of a JSON value in a THarmonyJson tree
  THarmonyJsonKind = (
    hjNull,
    hjFalse,
    hjTrue,
    hjNumber,
    hjString,
    hjArray,
    hjObject);

  /// one value of a tool schema
  THarmonyJsonNode = record
    Kind: THarmonyJsonKind;
    /// hjString: the decoded text; hjNumber: its Python form (str() = repr)
    Text: RawUtf8;
    /// the value as Python json.dumps writes it: THarmonyJson.Out from
    // OutStart (0-based), OutLen bytes
    OutStart, OutLen: integer;
    /// hjArray, hjObject: the children are Child[First .. First + Count - 1]
    First, Count: integer;
  end;

  PHarmonyJson = ^THarmonyJson;

  /// a tool schema as the Jinja template of gpt-oss walks it: the Python
  // values (json.loads), their json.dumps form (the template's tojson), the
  // keys in their order
  THarmonyJson = object
    Nodes: array of THarmonyJsonNode;
    NodeCount: integer;
    Child: TIntegerDynArray;
    Keys: TRawUtf8DynArray; // the key of Child[i] when its parent is an object
    ChildCount: integer;
    Out: RawUtf8;
    /// a new node at aOutStart of the written text
    function NewNode(aOutStart: integer): integer;
    /// the children of aNode, in their order
    procedure AddChildren(aNode: integer; const aKids: TIntegerDynArray;
      const aKeys: TRawUtf8DynArray; aCount: integer);
    /// parse aJson by the rules of PythonJson (a float it would round raises)
    procedure Parse(const aJson: RawUtf8);
    /// the member aKey of aNode - -1 (the Jinja undefined) when aNode is no
    // object or has no such member
    function Member(aNode: integer; const aKey: RawUtf8): integer;
    /// the i-th child of an array or object node
    function Item(aNode, i: integer): integer;
    /// Python truthiness; undefined (-1) is false
    function Truthy(aNode: integer): boolean;
    /// true when aNode is the string aValue (Python ==)
    function IsText(aNode: integer; const aValue: RawUtf8): boolean;
    /// the template's tojson of aNode
    function Json(aNode: integer): RawUtf8;
    /// Python str() of a scalar - raises for an array or object, which the
    // template would write as a Python repr
    function Str(aNode: integer; const aWhat: RawUtf8): RawUtf8;
    /// aNode as the string operand of a "+" - raises for anything else, as
    // Python does
    function TextOf(aNode: integer; const aWhat: RawUtf8): RawUtf8;
    /// true when aName is in aList, a "required" array; undefined or a falsy
    // value is none - another truthy value raises (Python's "in" would test a
    // substring or a key)
    function Listed(aList: integer; const aName: RawUtf8): boolean;
  end;

  /// recursive descent over the JSON text, writing the Python form
  TPythonJsonWriter = object
    P, Start: PUtf8Char;
    Out: RawUtf8;
    Depth: integer;
    KeepInexactFloats: boolean;
    /// not nil: every value is recorded there
    Tree: PHarmonyJson;
    procedure Fail(const aWhat: RawUtf8);
    procedure SkipBlanks;
    procedure Value;
    procedure StringValue(out aDecoded: RawUtf8);
    procedure Number;
    procedure Literal(const aWord: RawUtf8);
  end;

procedure TPythonJsonWriter.Fail(const aWhat: RawUtf8);
begin
  EInferError.RaiseUtf8('PythonJson: % at offset %', [aWhat, P - Start]);
end;

procedure TPythonJsonWriter.SkipBlanks;
begin
  while P^ in [' ', #9, #10, #13] do
    inc(P);
end;

function HexValue(c: AnsiChar): integer;
begin
  case c of
    '0'..'9':
      result := ord(c) - ord('0');
    'a'..'f':
      result := ord(c) - ord('a') + 10;
    'A'..'F':
      result := ord(c) - ord('A') + 10;
  else
    result := -1;
  end;
end;

procedure TPythonJsonWriter.StringValue(out aDecoded: RawUtf8);
var
  cp, lo: cardinal;
  run: PUtf8Char;
  buf: array[0..7] of AnsiChar;

  function Hex4: cardinal;
  var
    k, h: integer;
  begin
    result := 0;
    for k := 1 to 4 do
    begin
      h := HexValue(P^);
      if h < 0 then
        Fail('bad \u escape');
      result := (result shl 4) or cardinal(h);
      inc(P);
    end;
  end;

begin
  // P is on the opening quote
  inc(P);
  aDecoded := '';
  repeat
    run := P;
    while not (P^ in [#0..#31, '"', '\']) do
      inc(P);
    if P > run then
      Append(aDecoded, run, P - run);
    case P^ of
      '"':
        begin
          inc(P);
          break;
        end;
      '\':
        begin
          inc(P);
          case P^ of
            '"', '\', '/':
              Append(aDecoded, P^);
            'b':
              Append(aDecoded, #8);
            'f':
              Append(aDecoded, #12);
            'n':
              Append(aDecoded, #10);
            'r':
              Append(aDecoded, #13);
            't':
              Append(aDecoded, #9);
            'u':
              begin
                inc(P);
                cp := Hex4;
                if (cp >= $dc00) and (cp <= $dfff) then
                  Fail('lone low surrogate');
                if (cp >= $d800) and (cp <= $dbff) then
                begin
                  if (P[0] <> '\') or (P[1] <> 'u') then
                    Fail('lone high surrogate');
                  inc(P, 2);
                  lo := Hex4;
                  if (lo < $dc00) or (lo > $dfff) then
                    Fail('lone high surrogate');
                  cp := $10000 + ((cp - $d800) shl 10) + (lo - $dc00);
                end;
                Append(aDecoded, @buf, Ucs4ToUtf8(cp, @buf));
                continue;
              end;
          else
            Fail('bad escape');
          end;
          inc(P);
        end;
    else
      Fail('unterminated string or raw control character');
    end;
  until false;
  if not IsStrictUtf8(pointer(aDecoded), length(aDecoded)) then
    Fail('invalid UTF-8 in a string');
end;

procedure TPythonJsonWriter.Literal(const aWord: RawUtf8);
var
  i: PtrInt;
begin
  for i := 1 to length(aWord) do
    if P[i - 1] <> aWord[i] then // stops at the #0 ending the text
      Fail('unexpected value');
  inc(P, length(aWord));
  Append(Out, aWord);
end;

procedure TPythonJsonWriter.Number;
var
  neg, isFloat: boolean;
  intStart, intEnd, fracStart, fracEnd: PUtf8Char;
  exp10, e: integer;
  expNeg: boolean;
  digits, m, s: RawUtf8;
  i, k, lead, decpt: integer;
  numStart: PUtf8Char;
begin
  numStart := P;
  neg := P^ = '-';
  if neg then
    inc(P);
  if not (P^ in ['0'..'9']) then
    Fail('bad number');
  intStart := P;
  if P^ = '0' then
    inc(P)
  else
    while P^ in ['0'..'9'] do
      inc(P);
  intEnd := P;
  fracStart := P;
  fracEnd := P;
  isFloat := false;
  if P^ = '.' then
  begin
    isFloat := true;
    inc(P);
    fracStart := P;
    if not (P^ in ['0'..'9']) then
      Fail('bad fraction');
    while P^ in ['0'..'9'] do
      inc(P);
    fracEnd := P;
  end;
  exp10 := 0;
  if P^ in ['e', 'E'] then
  begin
    isFloat := true;
    inc(P);
    expNeg := false;
    if P^ in ['+', '-'] then
    begin
      expNeg := P^ = '-';
      inc(P);
    end;
    if not (P^ in ['0'..'9']) then
      Fail('bad exponent');
    while P^ in ['0'..'9'] do
    begin
      // no saturation: a clipped exponent would silently give another value
      if exp10 >= 100000 then
        Fail('exponent too large');
      exp10 := exp10 * 10 + ord(P^) - ord('0');
      inc(P);
    end;
    if expNeg then
      exp10 := -exp10;
  end;
  FastSetString(digits, intStart, intEnd - intStart);
  if not isFloat then
  begin
    // a Python int: arbitrary precision, written back as is; -0 is 0
    if (digits = '0') or not neg then
      Append(Out, digits)
    else
      Append(Out, ['-', digits]);
    exit;
  end;
  // value = digits . frac x 10^exp10: significant digits m, decimal point decpt
  Append(digits, fracStart, fracEnd - fracStart);
  e := exp10 - (fracEnd - fracStart); // value = digits x 10^e
  lead := 1;
  while (lead <= length(digits)) and (digits[lead] = '0') do
    inc(lead);
  k := length(digits);
  while (k >= lead) and (digits[k] = '0') do
  begin
    dec(k);
    inc(e);
  end;
  if neg then
    s := '-'
  else
    s := '';
  if k < lead then
  begin
    Append(Out, [s, '0.0']); // a zero keeps its sign in Python
    exit;
  end;
  m := copy(digits, lead, k - lead + 1);
  decpt := length(m) + e; // value = 0.m x 10^decpt
  if (length(m) > 15) or (decpt > 308) or (decpt < -306) then
    if KeepInexactFloats then
    begin
      Append(Out, numStart, P - numStart); // the literal, already validated
      exit;
    end
    else if length(m) > 15 then
      Fail('float with more than 15 significant digits not supported')
    else
      Fail('float out of the supported range');
  // Python repr: fixed notation for -4 < decpt <= 16, else exponent
  if (decpt > -4) and (decpt <= 16) then
  begin
    if decpt <= 0 then
      Append(Out, [s, '0.', RawUtf8(StringOfChar('0', -decpt)), m])
    else if decpt < length(m) then
      Append(Out, [s, copy(m, 1, decpt), '.', copy(m, decpt + 1, maxInt)])
    else
      Append(Out, [s, m, RawUtf8(StringOfChar('0', decpt - length(m))), '.0']);
  end
  else
  begin
    Append(Out, [s, m[1]]);
    if length(m) > 1 then
      Append(Out, ['.', copy(m, 2, maxInt)]);
    i := decpt - 1;
    if i < 0 then
      Append(Out, 'e-')
    else
      Append(Out, 'e+');
    i := abs(i);
    if i < 10 then
      Append(Out, '0');
    Append(Out, [i]);
  end;
end;

procedure TPythonJsonWriter.Value;
var
  keys: TRawUtf8DynArray;
  kids: TIntegerDynArray;
  key: RawUtf8;
  n, i, node: integer;
  kind: THarmonyJsonKind;
begin
  SkipBlanks;
  // a tool schema nests a few levels: a deep one is hostile, refused before
  // the recursion exhausts the stack
  if (P^ in ['{', '[']) and (Depth >= JSON_MAX_DEPTH) then
    Fail('nested too deeply');
  node := -1;
  if Tree <> nil then
    node := Tree^.NewNode(length(Out));
  kind := hjNull;
  case P^ of
    '{':
      begin
        kind := hjObject;
        inc(Depth);
        inc(P);
        Append(Out, '{');
        SkipBlanks;
        n := 0;
        if P^ = '}' then
          inc(P)
        else
          repeat
            SkipBlanks;
            if P^ <> '"' then
              Fail('object key expected');
            StringValue(key);
            for i := 0 to n - 1 do
              if keys[i] = key then
                Fail('duplicate key');
            if n = length(keys) then
            begin
              SetLength(keys, n * 2 + 4);
              SetLength(kids, length(keys));
            end;
            keys[n] := key;
            inc(n);
            if n > 1 then
              Append(Out, ', ');
            AddEscaped(Out, pointer(key), length(key));
            SkipBlanks;
            if P^ <> ':' then
              Fail('":" expected');
            inc(P);
            Append(Out, ': ');
            if node >= 0 then
              kids[n - 1] := Tree^.NodeCount; // the node Value records next
            Value;
            SkipBlanks;
            if P^ = ',' then
              inc(P)
            else if P^ = '}' then
            begin
              inc(P);
              break;
            end
            else
              Fail('"," or "}" expected');
          until false;
        Append(Out, '}');
        dec(Depth);
        if node >= 0 then
          Tree^.AddChildren(node, kids, keys, n);
      end;
    '[':
      begin
        kind := hjArray;
        inc(Depth);
        inc(P);
        Append(Out, '[');
        SkipBlanks;
        n := 0;
        if P^ = ']' then
          inc(P)
        else
          repeat
            if n > 0 then
              Append(Out, ', ');
            if n = length(kids) then
            begin
              SetLength(kids, n * 2 + 4);
              SetLength(keys, length(kids)); // no keys: ''
            end;
            if node >= 0 then
              kids[n] := Tree^.NodeCount;
            inc(n);
            Value;
            SkipBlanks;
            if P^ = ',' then
              inc(P)
            else if P^ = ']' then
            begin
              inc(P);
              break;
            end
            else
              Fail('"," or "]" expected');
          until false;
        Append(Out, ']');
        dec(Depth);
        if node >= 0 then
          Tree^.AddChildren(node, kids, keys, n);
      end;
    '"':
      begin
        kind := hjString;
        StringValue(key);
        AddEscaped(Out, pointer(key), length(key));
      end;
    '-', '0'..'9':
      begin
        kind := hjNumber;
        Number;
      end;
    't':
      begin
        kind := hjTrue;
        Literal('true');
      end;
    'f':
      begin
        kind := hjFalse;
        Literal('false');
      end;
    'n':
      Literal('null');
  else
    Fail('value expected');
  end;
  if node < 0 then
    exit;
  Tree^.Nodes[node].Kind := kind;
  Tree^.Nodes[node].OutLen := length(Out) - Tree^.Nodes[node].OutStart;
  if kind = hjString then
    Tree^.Nodes[node].Text := key
  else if kind = hjNumber then
    Tree^.Nodes[node].Text := copy(Out, Tree^.Nodes[node].OutStart + 1,
      Tree^.Nodes[node].OutLen);
end;

function PythonJson(const aJson: RawUtf8; aKeepInexactFloats: boolean): RawUtf8;
var
  w: TPythonJsonWriter;
begin
  w.Start := pointer(aJson);
  w.P := w.Start;
  w.Out := '';
  w.Depth := 0;
  w.KeepInexactFloats := aKeepInexactFloats;
  w.Tree := nil;
  if w.P = nil then
    w.Fail('empty');
  w.Value;
  w.SkipBlanks;
  if (w.P - w.Start <> length(aJson)) then
    w.Fail('trailing data');
  result := w.Out;
end;


{ TChatTemplate }

constructor TChatTemplate.Create(aTokenizer: TInferTokenizer);
begin
  inherited Create;
  if aTokenizer = nil then
    EInferError.RaiseUtf8('%.Create: no tokenizer', [self]);
  fTokenizer := aTokenizer;
end;

class function TChatTemplate.DefaultEnableThinking: boolean;
begin
  result := true;
end;

function TChatTemplate.Need(const aContent: RawUtf8): integer;
begin
  result := fTokenizer.TokenId(aContent);
  if result < 0 then
    EInferError.RaiseUtf8('%: tokenizer has no % token', [self, aContent]);
end;


{ TQwen3ChatTemplate }

constructor TQwen3ChatTemplate.Create(aTokenizer: TInferTokenizer);
begin
  inherited Create(aTokenizer);
  fImStart := Need('<|im_start|>');
  fImEnd := Need('<|im_end|>');
  fThink := Need('<think>');
  fThinkEnd := Need('</think>');
  fToolCall := Need('<tool_call>');
  fToolCallEnd := Need('</tool_call>');
  fToolResponse := Need('<tool_response>');
  fToolResponseEnd := Need('</tool_response>');
end;


{ TChatSegments }

procedure TChatSegments.Add(const aText: RawUtf8);
begin
  Append(Pending, aText);
end;

procedure TChatSegments.Flush;
begin
  if Pending = '' then
    exit;
  if WithIds then
    Tokenizer.EncodeAppend(Pending, Ids, Count);
  Append(Text, Pending);
  Pending := '';
end;

procedure TChatSegments.Control(aId: integer; const aName: RawUtf8);
begin
  Flush;
  if WithIds then
  begin
    if Count = length(Ids) then
      SetLength(Ids, Count * 2 + 64);
    Ids[Count] := aId;
    inc(Count);
  end;
  Append(Text, aName);
end;

/// Python str.endswith, case-sensitive (mORMot's EndWith is not)
function EndsWithExact(const s, aSuffix: RawUtf8): boolean;
begin
  result := (length(s) >= length(aSuffix)) and
    CompareMem(@PByteArray(s)^[length(s) - length(aSuffix)], pointer(aSuffix),
      length(aSuffix));
end;

/// Python str.strip/lstrip/rstrip of '\n' only
function StripNewlines(const s: RawUtf8; aLeft, aRight: boolean): RawUtf8;
var
  a, b: PtrInt;
begin
  a := 1;
  b := length(s);
  if aLeft then
    while (a <= b) and (s[a] = #10) do
      inc(a);
  if aRight then
    while (b >= a) and (s[b] = #10) do
      dec(b);
  result := copy(s, a, b - a + 1);
end;

/// Python s.split(aSep)[0] and s.split(aSep)[-1]
function SplitFirst(const s, aSep: RawUtf8): RawUtf8;
var
  i: PtrInt;
begin
  i := PosEx(aSep, s);
  if i = 0 then
    result := s
  else
    result := copy(s, 1, i - 1);
end;

function SplitLast(const s, aSep: RawUtf8): RawUtf8;
var
  i, last: PtrInt;
begin
  last := 0;
  i := PosEx(aSep, s);
  while i > 0 do
  begin
    last := i;
    i := PosEx(aSep, s, i + 1);
  end;
  if last = 0 then
    result := s
  else
    result := copy(s, last + length(aSep), maxInt);
end;

function ToolJson(const aTool: TLlmTool): RawUtf8;
begin
  // {"type": "function", "function": {name, description?, parameters?}} -
  // the OpenAI tool object as the template's tojson writes it
  result := '{"type": "function", "function": {"name": ' +
    PythonJsonString(aTool.Name);
  if aTool.Description <> '' then
    result := result + ', "description": ' + PythonJsonString(aTool.Description);
  if aTool.ParametersJson <> '' then
    result := result + ', "parameters": ' + PythonJson(aTool.ParametersJson);
  result := result + '}}';
end;

const
  TOOLS_INTRO =
    '# Tools'#10#10'You may call one or more functions to assist with the user ' +
    'query.'#10#10'You are provided with function signatures within ' +
    '<tools></tools> XML tags:'#10'<tools>';
  TOOLS_OUTRO =
    #10'</tools>'#10#10'For each function call, return a json object with ' +
    'function name and arguments within ';
  TOOLS_FORMAT = #10'{"name": <function-name>, "arguments": <args-json-object>}'#10;

procedure TQwen3ChatTemplate.Build(var aOut: TChatSegments;
  const aMessages: TLlmMessageDynArray; const aTools: TLlmToolDynArray;
  aAddGenerationPrompt: boolean; const aOptions: TChatTemplateOptions);
var
  n, i, j, lastQuery: PtrInt;
  multiStep: boolean;
  content, reasoning: RawUtf8;
  m: ^TLlmMessage;
begin
  n := length(aMessages);
  // the system turn, with the tool list when there are tools
  if aTools <> nil then
  begin
    aOut.Control(fImStart, '<|im_start|>');
    aOut.Add('system'#10);
    if (n > 0) and (aMessages[0].Role = lrSystem) then
      aOut.Add(aMessages[0].Content + #10#10);
    aOut.Add(TOOLS_INTRO);
    for i := 0 to high(aTools) do
      aOut.Add(#10 + ToolJson(aTools[i]));
    aOut.Add(TOOLS_OUTRO);
    aOut.Control(fToolCall, '<tool_call>');
    aOut.Control(fToolCallEnd, '</tool_call>');
    aOut.Add(' XML tags:'#10);
    aOut.Control(fToolCall, '<tool_call>');
    aOut.Add(TOOLS_FORMAT);
    aOut.Control(fToolCallEnd, '</tool_call>');
    aOut.Control(fImEnd, '<|im_end|>');
    aOut.Add(#10);
  end
  else if (n > 0) and (aMessages[0].Role = lrSystem) then
  begin
    aOut.Control(fImStart, '<|im_start|>');
    aOut.Add('system'#10 + aMessages[0].Content);
    aOut.Control(fImEnd, '<|im_end|>');
    aOut.Add(#10);
  end;
  // the last real user query: a user turn that is no wrapped tool response
  multiStep := true;
  lastQuery := n - 1;
  for i := n - 1 downto 0 do
    if multiStep and (aMessages[i].Role = lrUser) and
       not ((PosEx('<tool_response>', aMessages[i].Content) = 1) and
            EndsWithExact(aMessages[i].Content, '</tool_response>')) then
    begin
      multiStep := false;
      lastQuery := i;
    end;
  for i := 0 to n - 1 do
  begin
    m := @aMessages[i];
    content := m^.Content;
    if (m^.Role = lrUser) or ((m^.Role = lrSystem) and (i > 0)) then
    begin
      aOut.Control(fImStart, '<|im_start|>');
      aOut.Add(LlmRoleText(m^.Role) + #10 + content);
      aOut.Control(fImEnd, '<|im_end|>');
      aOut.Add(#10);
    end
    else if m^.Role = lrAssistant then
    begin
      // the reasoning of the message (reasoning_content) - else a <think>
      // block its content still holds
      reasoning := m^.Reasoning;
      if (reasoning = '') and (PosEx('</think>', content) > 0) then
      begin
        reasoning := StripNewlines(SplitLast(StripNewlines(
          SplitFirst(content, '</think>'), false, true), '<think>'), true, false);
        content := StripNewlines(SplitLast(content, '</think>'), true, false);
      end;
      aOut.Control(fImStart, '<|im_start|>');
      if (i > lastQuery) and ((i = n - 1) or (reasoning <> '')) then
      begin
        aOut.Add('assistant'#10);
        aOut.Control(fThink, '<think>');
        aOut.Add(#10 + StripNewlines(reasoning, true, true) + #10);
        aOut.Control(fThinkEnd, '</think>');
        aOut.Add(#10#10 + StripNewlines(content, true, false));
      end
      else
        aOut.Add('assistant'#10 + content);
      for j := 0 to high(m^.ToolCalls) do
      begin
        if (j > 0) or (content <> '') then
          aOut.Add(#10);
        aOut.Control(fToolCall, '<tool_call>');
        // name and arguments go in verbatim, as the template does
        aOut.Add(#10'{"name": "' + m^.ToolCalls[j].Name + '", "arguments": ' +
          m^.ToolCalls[j].ArgumentsJson + '}'#10);
        aOut.Control(fToolCallEnd, '</tool_call>');
      end;
      aOut.Control(fImEnd, '<|im_end|>');
      aOut.Add(#10);
    end
    else if m^.Role = lrTool then
    begin
      if (i = 0) or (aMessages[i - 1].Role <> lrTool) then
      begin
        aOut.Control(fImStart, '<|im_start|>');
        aOut.Add('user');
      end;
      aOut.Add(#10);
      aOut.Control(fToolResponse, '<tool_response>');
      aOut.Add(#10 + content + #10);
      aOut.Control(fToolResponseEnd, '</tool_response>');
      if (i = n - 1) or (aMessages[i + 1].Role <> lrTool) then
      begin
        aOut.Control(fImEnd, '<|im_end|>');
        aOut.Add(#10);
      end;
    end;
    // a system message at index 0 was written above
  end;
  if aAddGenerationPrompt then
  begin
    aOut.Control(fImStart, '<|im_start|>');
    aOut.Add('assistant'#10);
    if not aOptions.EnableThinking then
    begin
      aOut.Control(fThink, '<think>');
      aOut.Add(#10#10);
      aOut.Control(fThinkEnd, '</think>');
      aOut.Add(#10#10);
    end;
  end;
  aOut.Flush;
end;

{ TQwen2ChatTemplate }

const
  /// the system text of the Qwen2.5 template when the conversation has none
  QWEN25_SYSTEM = 'You are Qwen, created by Alibaba Cloud. You are a helpful ' +
    'assistant.';

constructor TQwen2ChatTemplate.Create(aTokenizer: TInferTokenizer);
begin
  inherited Create(aTokenizer);
  fImStart := Need('<|im_start|>');
  fImEnd := Need('<|im_end|>');
  fToolCall := Need('<tool_call>');
  fToolCallEnd := Need('</tool_call>');
  fToolResponse := aTokenizer.TokenId('<tool_response>'); // -1: text
  fToolResponseEnd := aTokenizer.TokenId('</tool_response>');
end;

procedure TQwen2ChatTemplate.Build(var aOut: TChatSegments;
  const aMessages: TLlmMessageDynArray; const aTools: TLlmToolDynArray;
  aAddGenerationPrompt: boolean; const aOptions: TChatTemplateOptions);
var
  n, i, j: PtrInt;
  args: RawUtf8;
  m: ^TLlmMessage;
begin
  n := length(aMessages);
  // the system turn: the first message when it is a system one, else the
  // default; the tool list after it
  aOut.Control(fImStart, '<|im_start|>');
  if (n > 0) and (aMessages[0].Role = lrSystem) then
    aOut.Add('system'#10 + aMessages[0].Content)
  else
    aOut.Add('system'#10 + QWEN25_SYSTEM);
  if aTools <> nil then
  begin
    aOut.Add(#10#10 + TOOLS_INTRO);
    for i := 0 to high(aTools) do
      aOut.Add(#10 + ToolJson(aTools[i]));
    aOut.Add(TOOLS_OUTRO);
    aOut.Control(fToolCall, '<tool_call>');
    aOut.Control(fToolCallEnd, '</tool_call>');
    aOut.Add(' XML tags:'#10);
    aOut.Control(fToolCall, '<tool_call>');
    aOut.Add(TOOLS_FORMAT);
    aOut.Control(fToolCallEnd, '</tool_call>');
  end;
  aOut.Control(fImEnd, '<|im_end|>');
  aOut.Add(#10);
  for i := 0 to n - 1 do
  begin
    m := @aMessages[i];
    if (m^.Role = lrUser) or ((m^.Role = lrSystem) and (i > 0)) or
       ((m^.Role = lrAssistant) and (m^.ToolCalls = nil)) then
    begin
      aOut.Control(fImStart, '<|im_start|>');
      aOut.Add(LlmRoleText(m^.Role) + #10 + m^.Content);
      aOut.Control(fImEnd, '<|im_end|>');
      aOut.Add(#10);
    end
    else if m^.Role = lrAssistant then
    begin
      aOut.Control(fImStart, '<|im_start|>');
      aOut.Add('assistant');
      if m^.Content <> '' then
        aOut.Add(#10 + m^.Content);
      for j := 0 to high(m^.ToolCalls) do
      begin
        args := m^.ToolCalls[j].ArgumentsJson;
        if args = '' then
          args := '{}'
        else
          args := PythonJson(args, {keepinexactfloats=}true);
        aOut.Add(#10);
        aOut.Control(fToolCall, '<tool_call>');
        aOut.Add(#10'{"name": "' + m^.ToolCalls[j].Name + '", "arguments": ' + args +
          '}'#10);
        aOut.Control(fToolCallEnd, '</tool_call>');
      end;
      aOut.Control(fImEnd, '<|im_end|>');
      aOut.Add(#10);
    end
    else if m^.Role = lrTool then
    begin
      if (i = 0) or (aMessages[i - 1].Role <> lrTool) then
      begin
        aOut.Control(fImStart, '<|im_start|>');
        aOut.Add('user');
      end;
      aOut.Add(#10);
      if fToolResponse >= 0 then
        aOut.Control(fToolResponse, '<tool_response>')
      else
        aOut.Add('<tool_response>');
      aOut.Add(#10 + m^.Content + #10);
      if fToolResponseEnd >= 0 then
        aOut.Control(fToolResponseEnd, '</tool_response>')
      else
        aOut.Add('</tool_response>');
      if (i = n - 1) or (aMessages[i + 1].Role <> lrTool) then
      begin
        aOut.Control(fImEnd, '<|im_end|>');
        aOut.Add(#10);
      end;
    end;
    // a system message at index 0 was written above
  end;
  if aAddGenerationPrompt then
  begin
    aOut.Control(fImStart, '<|im_start|>');
    aOut.Add('assistant'#10);
  end;
  aOut.Flush;
end;

{ THarmonyJson }

function THarmonyJson.NewNode(aOutStart: integer): integer;
begin
  if NodeCount = length(Nodes) then
    SetLength(Nodes, NodeCount * 2 + 16);
  result := NodeCount;
  inc(NodeCount);
  Nodes[result].Kind := hjNull;
  Nodes[result].Text := '';
  Nodes[result].OutStart := aOutStart;
  Nodes[result].OutLen := 0;
  Nodes[result].First := 0;
  Nodes[result].Count := 0;
end;

procedure THarmonyJson.AddChildren(aNode: integer; const aKids: TIntegerDynArray;
  const aKeys: TRawUtf8DynArray; aCount: integer);
var
  i: integer;
begin
  if ChildCount + aCount > length(Child) then
  begin
    SetLength(Child, (ChildCount + aCount) * 2 + 16);
    SetLength(Keys, length(Child));
  end;
  Nodes[aNode].First := ChildCount;
  Nodes[aNode].Count := aCount;
  for i := 0 to aCount - 1 do
  begin
    Child[ChildCount] := aKids[i];
    Keys[ChildCount] := aKeys[i];
    inc(ChildCount);
  end;
end;

procedure THarmonyJson.Parse(const aJson: RawUtf8);
var
  w: TPythonJsonWriter;
begin
  NodeCount := 0;
  ChildCount := 0;
  w.Start := pointer(aJson);
  w.P := w.Start;
  w.Out := '';
  w.Depth := 0;
  w.KeepInexactFloats := false;
  w.Tree := @self;
  if w.P = nil then
    w.Fail('empty');
  w.Value;
  w.SkipBlanks;
  if w.P - w.Start <> length(aJson) then
    w.Fail('trailing data');
  Out := w.Out;
end;

function THarmonyJson.Member(aNode: integer; const aKey: RawUtf8): integer;
var
  i: integer;
begin
  result := -1;
  if (aNode < 0) or (Nodes[aNode].Kind <> hjObject) then
    exit;
  // the keys are unique: PythonJson refuses a duplicate
  for i := Nodes[aNode].First to Nodes[aNode].First + Nodes[aNode].Count - 1 do
    if Keys[i] = aKey then
    begin
      result := Child[i];
      exit;
    end;
end;

function THarmonyJson.Item(aNode, i: integer): integer;
begin
  result := Child[Nodes[aNode].First + i];
end;

function THarmonyJson.Truthy(aNode: integer): boolean;
begin
  result := false;
  if aNode < 0 then
    exit;
  case Nodes[aNode].Kind of
    hjTrue:
      result := true;
    hjNumber:
      // the Python form of a zero: 0, 0.0 or -0.0
      result := (Nodes[aNode].Text <> '0') and (Nodes[aNode].Text <> '0.0') and
        (Nodes[aNode].Text <> '-0.0');
    hjString:
      result := Nodes[aNode].Text <> '';
    hjArray, hjObject:
      result := Nodes[aNode].Count > 0;
  end;
end;

function THarmonyJson.IsText(aNode: integer; const aValue: RawUtf8): boolean;
begin
  result := (aNode >= 0) and (Nodes[aNode].Kind = hjString) and
    (Nodes[aNode].Text = aValue);
end;

function THarmonyJson.Json(aNode: integer): RawUtf8;
begin
  result := copy(Out, Nodes[aNode].OutStart + 1, Nodes[aNode].OutLen);
end;

function THarmonyJson.Str(aNode: integer; const aWhat: RawUtf8): RawUtf8;
begin
  case Nodes[aNode].Kind of
    hjNull:
      result := 'None';
    hjFalse:
      result := 'False';
    hjTrue:
      result := 'True';
    hjNumber, hjString:
      result := Nodes[aNode].Text;
  else
    EInferError.RaiseUtf8('chat template: % is an array or object, ' +
      'which the template writes as a Python repr', [aWhat]);
  end;
end;

function THarmonyJson.TextOf(aNode: integer; const aWhat: RawUtf8): RawUtf8;
begin
  if Nodes[aNode].Kind <> hjString then
    EInferError.RaiseUtf8('chat template: % is no string', [aWhat]);
  result := Nodes[aNode].Text;
end;

function THarmonyJson.Listed(aList: integer; const aName: RawUtf8): boolean;
var
  i, v: integer;
begin
  result := false;
  if not Truthy(aList) then
    exit;
  if Nodes[aList].Kind <> hjArray then
    EInferError.RaiseUtf8('chat template: "required" is no array', []);
  for i := 0 to Nodes[aList].Count - 1 do
  begin
    v := Item(aList, i);
    if (Nodes[v].Kind = hjString) and (Nodes[v].Text = aName) then
    begin
      result := true;
      exit;
    end;
  end;
end;


{ THarmonyChatTemplate }

const
  /// the indentation the template writes before the default of a oneOf
  // variant and before the type of a property of a nested object: those
  // expressions have no "-" in their tag, and lstrip_blocks strips only before
  // block tags
  HARMONY_INDENT_DEFAULT = '                    ';
  HARMONY_INDENT_TYPE = '                ';
  /// the identity line of the system message when none is given
  HARMONY_IDENTITY = 'You are ChatGPT, a large language model trained by OpenAI.';

/// Jinja's length of a string: its code points
function CodePointCount(const s: RawUtf8): PtrInt;
var
  i: PtrInt;
begin
  result := 0;
  for i := 1 to length(s) do
    if ord(s[i]) and $c0 <> $80 then
      inc(result);
end;

/// render_typescript_type of the template
function HarmonyType(var T: THarmonyJson; aSpec: integer): RawUtf8;
var
  ty, items, it, one, props, req, v, d, i, n: integer;
  inner: RawUtf8;
begin
  // a value that is no object has no members (all undefined): "any"
  result := 'any';
  if (aSpec < 0) or (T.Nodes[aSpec].Kind <> hjObject) then
    exit;
  ty := T.Member(aSpec, 'type');
  if T.IsText(ty, 'array') then
  begin
    items := T.Member(aSpec, 'items');
    // without "items", the subscript of Jinja finds the items() method of the
    // dict - truthy, but without a type: the same "any[]" as an empty one; a
    // value that is no object has no type either
    if T.Truthy(items) and (T.Nodes[items].Kind = hjObject) then
    begin
      it := T.Member(items, 'type');
      if T.IsText(it, 'string') then
        result := 'string[]'
      else if T.IsText(it, 'number') or T.IsText(it, 'integer') then
        result := 'number[]'
      else if T.IsText(it, 'boolean') then
        result := 'boolean[]'
      else
      begin
        inner := HarmonyType(T, items);
        if (inner = 'object | object') or (CodePointCount(inner) > 50) then
          result := 'any[]'
        else
          result := inner + '[]';
      end;
    end
    else
      result := 'any[]';
    if T.Truthy(T.Member(aSpec, 'nullable')) then
      result := result + ' | null';
  end
  else if (ty >= 0) and (T.Nodes[ty].Kind = hjArray) and (T.Nodes[ty].Count > 0) then
  begin
    // a list of types, joined (one alone written as it is): Python str()
    result := T.Str(T.Item(ty, 0), 'an item of "type"');
    for i := 1 to T.Nodes[ty].Count - 1 do
      result := result + ' | ' + T.Str(T.Item(ty, i), 'an item of "type"');
  end
  else if T.Truthy(T.Member(aSpec, 'oneOf')) then
  begin
    one := T.Member(aSpec, 'oneOf');
    if T.Nodes[one].Kind <> hjArray then
      EInferError.RaiseUtf8('THarmonyChatTemplate: "oneOf" is no array', []);
    // the template sets its flag of object variants inside the loop, where
    // Jinja scopes it: never seen after the loop - every variant is written
    result := '';
    n := T.Nodes[one].Count;
    for i := 0 to n - 1 do
    begin
      v := T.Item(one, i);
      result := result + HarmonyType(T, v);
      d := T.Member(v, 'description');
      if T.Truthy(d) then
        result := result + '// ' + T.TextOf(d, 'the description of a oneOf variant');
      d := T.Member(v, 'default');
      if d >= 0 then
        result := result + HARMONY_INDENT_DEFAULT + '// default: ' + T.Json(d);
      if i < n - 1 then
        result := result + ' | '#10;
    end;
  end
  else if T.IsText(ty, 'string') then
  begin
    v := T.Member(aSpec, 'enum');
    if T.Truthy(v) then
    begin
      if T.Nodes[v].Kind <> hjArray then
        EInferError.RaiseUtf8('THarmonyChatTemplate: "enum" is no array', []);
      // '"' + enum|join('" | "') + '"': join takes str() of each value
      result := '"';
      for i := 0 to T.Nodes[v].Count - 1 do
      begin
        if i > 0 then
          result := result + '" | "';
        result := result + T.Str(T.Item(v, i), 'an enum value');
      end;
      result := result + '"';
    end
    else
    begin
      result := 'string';
      if T.Truthy(T.Member(aSpec, 'nullable')) then
        result := result + ' | null';
    end;
  end
  else if T.IsText(ty, 'number') or T.IsText(ty, 'integer') then
    result := 'number'
  else if T.IsText(ty, 'boolean') then
    result := 'boolean'
  else if T.IsText(ty, 'object') then
  begin
    props := T.Member(aSpec, 'properties');
    if T.Truthy(props) then
    begin
      if T.Nodes[props].Kind <> hjObject then
        EInferError.RaiseUtf8('THarmonyChatTemplate: "properties" is no object', []);
      req := T.Member(aSpec, 'required');
      result := '{'#10;
      n := T.Nodes[props].Count;
      for i := 0 to n - 1 do
      begin
        result := result + T.Keys[T.Nodes[props].First + i];
        if not T.Listed(req, T.Keys[T.Nodes[props].First + i]) then
          result := result + '?';
        result := result + ': '#10 + HARMONY_INDENT_TYPE +
          HarmonyType(T, T.Item(props, i));
        if i < n - 1 then
          result := result + ', ';
      end;
      result := result + '}';
    end
    else
      result := 'object';
  end;
end;

/// render_tool_namespace("functions", tools) of the template
function HarmonyTools(const aTools: TLlmToolDynArray): RawUtf8;
var
  T: THarmonyJson;
  i, j, props, req, spec, d: integer;
  s, key: RawUtf8;
begin
  result := '## functions'#10#10'namespace functions {'#10#10;
  for i := 0 to high(aTools) do
  begin
    s := '// ' + aTools[i].Description + #10'type ' + aTools[i].Name + ' = ';
    props := -1;
    req := -1;
    if aTools[i].ParametersJson <> '' then
    begin
      T.Parse(aTools[i].ParametersJson);
      // falsy (null, {}, [], "", 0): no parameters, as the template sees them
      if T.Truthy(0) and (T.Nodes[0].Kind <> hjObject) then
        EInferError.RaiseUtf8('THarmonyChatTemplate: the parameters of % are ' +
          'no object', [aTools[i].Name]);
      props := T.Member(0, 'properties');
      req := T.Member(0, 'required');
    end;
    if T.Truthy(props) then
    begin
      if T.Nodes[props].Kind <> hjObject then
        EInferError.RaiseUtf8('THarmonyChatTemplate: "properties" of % is no ' +
          'object', [aTools[i].Name]);
      s := s + '(_: {'#10;
      for j := 0 to T.Nodes[props].Count - 1 do
      begin
        key := T.Keys[T.Nodes[props].First + j];
        spec := T.Item(props, j);
        d := T.Member(spec, 'description');
        if T.Truthy(d) then
          s := s + '// ' + T.TextOf(d, 'a parameter description') + #10;
        s := s + key;
        if not T.Listed(req, key) then
          s := s + '?';
        s := s + ': ' + HarmonyType(T, spec);
        d := T.Member(spec, 'default');
        if d >= 0 then
          if T.Truthy(T.Member(spec, 'enum')) then
            s := s + ', // default: ' + T.TextOf(d, 'the default of an enum')
          else if T.Truthy(T.Member(spec, 'oneOf')) then
            s := s + '// default: ' + T.TextOf(d, 'the default of a oneOf')
          else
            s := s + ', // default: ' + T.Json(d);
        s := s + ','#10;
      end;
      s := s + '}) => any;'#10#10;
    end
    else
      s := s + '() => any;'#10#10;
    result := result + s;
  end;
  result := result + '} // namespace functions';
end;

/// the date of the system message: aDate checked, or today
function HarmonyDate(const aDate: RawUtf8): RawUtf8;
var
  i: PtrInt;
begin
  if aDate = '' then
  begin
    // strftime_now("%Y-%m-%d"): the local date, as datetime.now()
    result := StringToUtf8(FormatDateTime('yyyy"-"mm"-"dd', Now));
    exit;
  end;
  result := aDate;
  if length(aDate) = 10 then
    for i := 1 to 10 do
      if (i = 5) or (i = 8) then
      begin
        if aDate[i] <> '-' then
          break;
      end
      else if not (aDate[i] in ['0'..'9']) then
        break
      else if i = 10 then
        exit;
  EInferError.RaiseUtf8('THarmonyChatTemplate: CurrentDate "%" is no ' +
    'yyyy-mm-dd', [aDate]);
end;

constructor THarmonyChatTemplate.Create(aTokenizer: TInferTokenizer);
begin
  inherited Create(aTokenizer);
  fStart := Need('<|start|>');
  fChannel := Need('<|channel|>');
  fMessage := Need('<|message|>');
  fEnd := Need('<|end|>');
  fCall := Need('<|call|>');
  fReturn := Need('<|return|>');
end;

procedure THarmonyChatTemplate.Build(var aOut: TChatSegments;
  const aMessages: TLlmMessageDynArray; const aTools: TLlmToolDynArray;
  aAddGenerationPrompt: boolean; const aOptions: TChatTemplateOptions);
var
  n, first, i, j, k: PtrInt;
  text, name, args: RawUtf8;
  calls: TLlmToolCallDynArray; // of the last turn with calls, nil after an answer
  answered: boolean;
  m: ^TLlmMessage;

  procedure Message(const aHeader, aChannel, aBody: RawUtf8; aEndId: integer;
    const aEndName: RawUtf8);
  begin
    // <|start|>header[<|channel|>channel]<|message|>body<end>
    aOut.Control(fStart, '<|start|>');
    aOut.Add(aHeader);
    if aChannel <> '' then
    begin
      aOut.Control(fChannel, '<|channel|>');
      aOut.Add(aChannel);
    end;
    aOut.Control(fMessage, '<|message|>');
    aOut.Add(aBody);
    aOut.Control(aEndId, aEndName);
  end;

begin
  // the system message
  text := aOptions.ModelIdentity;
  if text = '' then
    text := HARMONY_IDENTITY;
  text := text + #10'Knowledge cutoff: 2024-06'#10'Current date: ' +
    HarmonyDate(aOptions.CurrentDate) + #10#10'Reasoning: ';
  if aOptions.ReasoningEffort = '' then
    text := text + 'medium'
  else if (aOptions.ReasoningEffort = 'low') or (aOptions.ReasoningEffort = 'medium') or
          (aOptions.ReasoningEffort = 'high') then
    text := text + aOptions.ReasoningEffort
  else
    EInferError.RaiseUtf8('%: ReasoningEffort "%" is not low, medium or high',
      [self, aOptions.ReasoningEffort]);
  text := text + #10#10'# Valid channels: analysis, commentary, final. Channel ' +
    'must be included for every message.';
  if aTools <> nil then
    text := text + #10'Calls to these tools must go to the commentary channel: ' +
      '''functions''.';
  Message('system', '', text, fEnd, '<|end|>');
  // the developer message: the instructions of a first system message, the
  // tools
  n := length(aMessages);
  first := 0;
  text := '';
  if (n > 0) and (aMessages[0].Role = lrSystem) then
  begin
    first := 1;
    if aMessages[0].Content <> '' then
      text := '# Instructions'#10#10 + aMessages[0].Content + #10#10;
  end;
  if aTools <> nil then
    text := text + '# Tools'#10#10 + HarmonyTools(aTools);
  if text <> '' then
    Message('developer', '', text, fEnd, '<|end|>');
  // the turns
  calls := nil;
  for i := first to n - 1 do
  begin
    m := @aMessages[i];
    case m^.Role of
      lrUser:
        Message('user', '', m^.Content, fEnd, '<|end|>');
      lrAssistant:
        if m^.ToolCalls <> nil then
        begin
          // its reasoning or content is the analysis before the calls -
          // unless a later assistant turn answers without calls (then it is
          // dropped)
          answered := false;
          for k := i + 1 to n - 1 do
            if (aMessages[k].Role = lrAssistant) and (aMessages[k].ToolCalls = nil) then
              answered := true;
          if not answered then
            if m^.Reasoning = '' then
            begin
              if m^.Content <> '' then
                Message('assistant', 'analysis', m^.Content, fEnd, '<|end|>');
            end
            else
            begin
              Message('assistant', 'analysis', m^.Reasoning, fEnd, '<|end|>');
              // both: the content is the preamble gpt-oss wrote on commentary
              // (THarmonyOutputParser takes it from there) - the template
              // refuses this pair
              if m^.Content <> '' then
                Message('assistant', 'commentary', m^.Content, fEnd, '<|end|>');
            end;
          for j := 0 to high(m^.ToolCalls) do
          begin
            args := m^.ToolCalls[j].ArgumentsJson;
            if args = '' then
              args := '{}'
            else
              args := PythonJson(args, {keepinexactfloats=}true);
            Message('assistant to=functions.' + m^.ToolCalls[j].Name,
              'commentary json', args, fCall, '<|call|>');
          end;
          calls := m^.ToolCalls;
        end
        else if (i = n - 1) and not aAddGenerationPrompt then
        begin
          // the end of a conversation the model wrote: its reasoning, then
          // the final message with <|return|>
          if m^.Reasoning <> '' then
            Message('assistant', 'analysis', m^.Reasoning, fEnd, '<|end|>');
          Message('assistant', 'final', m^.Content, fReturn, '<|return|>');
        end
        else
        begin
          Message('assistant', 'final', m^.Content, fEnd, '<|end|>');
          calls := nil;
        end;
      lrTool:
        begin
          if calls = nil then
            EInferError.RaiseUtf8('%: message % answers a tool call, but no ' +
              'assistant turn with calls is before it', [self, i]);
          // the call it answers, by its id - else the last one, as the template
          name := calls[high(calls)].Name;
          if m^.ToolCallId <> '' then
            for k := 0 to high(calls) do
              if calls[k].Id = m^.ToolCallId then
              begin
                name := calls[k].Name;
                break;
              end;
          Message('functions.' + name + ' to=assistant', 'commentary',
            PythonJsonString(m^.Content), fEnd, '<|end|>');
        end;
    else
      EInferError.RaiseUtf8('%: message % is a system message - only the first ' +
        'one is taken (as the developer instructions)', [self, i]);
    end;
  end;
  if aAddGenerationPrompt then
  begin
    aOut.Control(fStart, '<|start|>');
    aOut.Add('assistant');
  end;
  aOut.Flush;
end;

{ TGemma4ChatTemplate }

const
  GEMMA4_QUOTE = '<|"|>';

/// true for the code points of Python's str.isspace()
function IsPythonSpace(c: cardinal): boolean;
begin
  case c of
    9..13, $1c..$20, $85, $a0, $1680, $2000..$200a, $2028, $2029, $202f,
    $205f, $3000:
      result := true;
  else
    result := false;
  end;
end;

/// Python str.strip() - the trim filter of Jinja
function PythonStrip(const s: RawUtf8): RawUtf8;
var
  cp: TCardinalDynArray;
  n, a, b: PtrInt;
begin
  n := Utf8ToCodePoints(pointer(s), length(s), cp);
  a := 0;
  while (a < n) and IsPythonSpace(cp[a]) do
    inc(a);
  b := n;
  while (b > a) and IsPythonSpace(cp[b - 1]) do
    dec(b);
  if a = b then
    result := ''
  else
    result := CodePointsToUtf8(@cp[a], b - a);
end;

/// the strip_thinking macro: split at "<channel|>", of a part holding
// "<|channel>" only what is before it, the rest joined and trimmed
function Gemma4StripThinking(const s: RawUtf8): RawUtf8;
var
  rest, part: RawUtf8;
  i: PtrInt;
begin
  result := '';
  rest := s;
  repeat
    i := PosEx('<channel|>', rest);
    if i = 0 then
      part := rest
    else
      part := copy(rest, 1, i - 1);
    result := result + SplitFirst(part, '<|channel>');
    if i = 0 then
      break;
    rest := copy(rest, i + length('<channel|>'), maxInt);
  until false;
  result := PythonStrip(result);
end;

/// Python str.lower() of a key, for the order of the dictsort filter -
// ASCII, Latin-1, Latin Extended-A, Greek and Cyrillic; another upper case
// letter raises (its order is not guessed), as do a Roman numeral or a
// circled capital (upper case symbols) and the capital sigma (final or not by
// its context)
function Gemma4KeyLower(const s: RawUtf8): RawUtf8;
var
  cp: TCardinalDynArray;
  n, i: PtrInt;
  c: cardinal;
begin
  n := Utf8ToCodePoints(pointer(s), length(s), cp);
  for i := 0 to n - 1 do
  begin
    c := cp[i];
    case c of
      ord('A')..ord('Z'), $c0..$d6, $d8..$de, $391..$3a1, $3a4..$3ab, $410..$42f:
        inc(c, 32);
      $400..$40f:
        inc(c, 80);
      $100..$12f, $132..$137, $14a..$177:
        c := c or 1;
      $139..$148, $179..$17e:
        if c and 1 <> 0 then
          inc(c);
      $178:
        c := $ff;
      $386:
        c := $3ac;
      $2160..$216f, $24b6..$24cf:
        // Roman numerals and circled capitals: no letters, but lower() maps them
        EInferError.RaiseUtf8('TGemma4ChatTemplate: the key "%" has an upper ' +
          'case symbol whose Python lower() is not implemented', [s]);
      $3a3:
        // Python's lower() writes a final sigma at the end of a word
        EInferError.RaiseUtf8('TGemma4ChatTemplate: the key "%" has a capital ' +
          'sigma, whose Python lower() depends on its place in the word', [s]);
      $388..$38a:
        inc(c, 37);
      $38c:
        c := $3cc;
      $38e, $38f:
        inc(c, 63);
    else
      if (c >= $80) and (O200kClassOf(c) = ocUpper) then
        EInferError.RaiseUtf8('TGemma4ChatTemplate: the key "%" has an upper ' +
          'case letter whose Python lower() is not implemented', [s]);
    end;
    cp[i] := c;
  end;
  result := CodePointsToUtf8(pointer(cp), n);
end;

/// Python str.upper() of a type name - ASCII only
function Gemma4Upper(const s, aWhat: RawUtf8): RawUtf8;
var
  i: PtrInt;
begin
  for i := 1 to length(s) do
    if ord(s[i]) >= $80 then
      EInferError.RaiseUtf8('TGemma4ChatTemplate: % "%" is not ASCII', [aWhat, s]);
  result := UpperCase(s);
end;

type
  /// the macros of the Gemma 4 template over a parsed JSON value
  TGemma4Writer = object
    Out: ^TChatSegments;
    T: THarmonyJson;
    Quote: integer;
    /// <|"|>aText<|"|>
    procedure Quoted(const aText: RawUtf8);
    /// the members of an object (indexes into T.Child) in dictsort order
    function Sorted(aNode: integer): TIntegerDynArray;
    /// value | upper - undefined is '', a list or object raises
    function Upper(aNode: integer; const aWhat: RawUtf8): RawUtf8;
    /// the format_argument macro
    procedure Argument(aNode: integer; aEscapeKeys: boolean);
    /// the elements of a "required" list as <|"|>a<|"|>,<|"|>b<|"|>
    procedure Required(aNode: integer);
    /// an item type that is no string: value | map('upper') | list, written
    // by format_argument
    procedure TypeList(aNode: integer);
    /// the format_parameters macro over the object aNode
    procedure Parameters(aNode: integer; aFilterKeys: boolean);
    /// the format_function_declaration macro
    procedure Declaration(const aTool: TLlmTool);
  end;

procedure TGemma4Writer.Quoted(const aText: RawUtf8);
begin
  Out^.Control(Quote, GEMMA4_QUOTE);
  Out^.Add(aText);
  Out^.Control(Quote, GEMMA4_QUOTE);
end;

function TGemma4Writer.Sorted(aNode: integer): TIntegerDynArray;
var
  n, i, j, k: integer;
  lower: TRawUtf8DynArray;
  l: RawUtf8;
begin
  // Jinja's dictsort: by the key in lower case, stable (Python's sorted)
  n := T.Nodes[aNode].Count;
  SetLength(result, n);
  SetLength(lower, n);
  for i := 0 to n - 1 do
  begin
    k := T.Nodes[aNode].First + i;
    l := Gemma4KeyLower(T.Keys[k]);
    j := i;
    // UTF-8 bytes compare as the code points Python compares
    while (j > 0) and (SortDynArrayRawByteString(lower[j - 1], l) > 0) do
    begin
      result[j] := result[j - 1];
      lower[j] := lower[j - 1];
      dec(j);
    end;
    result[j] := k;
    lower[j] := l;
  end;
end;

function TGemma4Writer.Upper(aNode: integer; const aWhat: RawUtf8): RawUtf8;
begin
  if aNode < 0 then
    result := '' // undefined renders as ''
  else
    result := Gemma4Upper(T.Str(aNode, aWhat), aWhat);
end;

procedure TGemma4Writer.Argument(aNode: integer; aEscapeKeys: boolean);
var
  kids: TIntegerDynArray;
  i: integer;
begin
  case T.Nodes[aNode].Kind of
    hjNull:
      Out^.Add('null');
    hjString:
      Quoted(T.Nodes[aNode].Text);
    hjFalse:
      Out^.Add('false');
    hjTrue:
      Out^.Add('true');
    hjNumber:
      Out^.Add(T.Nodes[aNode].Text);
    hjObject:
      begin
        Out^.Add('{');
        kids := Sorted(aNode);
        for i := 0 to high(kids) do
        begin
          if i > 0 then
            Out^.Add(',');
          if aEscapeKeys then
            Quoted(T.Keys[kids[i]])
          else
            Out^.Add(T.Keys[kids[i]]);
          Out^.Add(':');
          Argument(T.Child[kids[i]], aEscapeKeys);
        end;
        Out^.Add('}');
      end;
    hjArray:
      begin
        Out^.Add('[');
        for i := 0 to T.Nodes[aNode].Count - 1 do
        begin
          if i > 0 then
            Out^.Add(',');
          Argument(T.Item(aNode, i), aEscapeKeys);
        end;
        Out^.Add(']');
      end;
  end;
end;

procedure TGemma4Writer.Required(aNode: integer);
var
  i: integer;
begin
  // a for loop over the value: a string would give its characters, an object
  // its keys - only an array is taken
  if T.Nodes[aNode].Kind <> hjArray then
    EInferError.RaiseUtf8('TGemma4ChatTemplate: "required" is no array', []);
  for i := 0 to T.Nodes[aNode].Count - 1 do
  begin
    if i > 0 then
      Out^.Add(',');
    Quoted(T.Str(T.Item(aNode, i), 'an element of "required"'));
  end;
end;

procedure TGemma4Writer.TypeList(aNode: integer);
var
  i: integer;
begin
  if T.Nodes[aNode].Kind <> hjArray then
    EInferError.RaiseUtf8('TGemma4ChatTemplate: an item type is neither a ' +
      'string nor an array', []);
  Out^.Add('[');
  for i := 0 to T.Nodes[aNode].Count - 1 do
  begin
    if i > 0 then
      Out^.Add(',');
    Quoted(Upper(T.Item(aNode, i), 'an item type'));
  end;
  Out^.Add(']');
end;

procedure TGemma4Writer.Parameters(aNode: integer; aFilterKeys: boolean);
var
  kids, items: TIntegerDynArray;
  i, j, v, sub, iv: integer;
  first, firstItem, addComma: boolean;
  key, typ: RawUtf8;

  procedure Comma;
  begin
    // {% if add_comma %},{% else %}{% set add_comma = true %}{% endif %}
    if addComma then
      Out^.Add(',')
    else
      addComma := true;
  end;

begin
  first := true;
  kids := Sorted(aNode);
  for i := 0 to high(kids) do
  begin
    key := T.Keys[kids[i]];
    if aFilterKeys and ((key = 'description') or (key = 'type') or
       (key = 'properties') or (key = 'required') or (key = 'nullable')) then
      continue;
    if not first then
      Out^.Add(',');
    first := false;
    v := T.Child[kids[i]];
    Out^.Add(key);
    Out^.Add(':{');
    addComma := false;
    sub := T.Member(v, 'description');
    if T.Truthy(sub) then
    begin
      Out^.Add('description:');
      Quoted(T.Str(sub, 'a description'));
      addComma := true;
    end;
    typ := Upper(T.Member(v, 'type'), 'a type');
    if typ = 'STRING' then
    begin
      sub := T.Member(v, 'enum');
      if T.Truthy(sub) then
      begin
        Comma;
        Out^.Add('enum:');
        Argument(sub, true);
      end;
    end
    else if typ = 'ARRAY' then
    begin
      sub := T.Member(v, 'items');
      if (sub >= 0) and (T.Nodes[sub].Kind = hjObject) and T.Truthy(sub) then
      begin
        Comma;
        Out^.Add('items:{');
        firstItem := true;
        items := Sorted(sub);
        for j := 0 to high(items) do
        begin
          iv := T.Child[items[j]];
          if T.Nodes[iv].Kind = hjNull then
            continue;
          if not firstItem then
            Out^.Add(',');
          firstItem := false;
          if T.Keys[items[j]] = 'properties' then
          begin
            Out^.Add('properties:{');
            if T.Nodes[iv].Kind = hjObject then
              Parameters(iv, false);
            Out^.Add('}');
          end
          else if T.Keys[items[j]] = 'required' then
          begin
            Out^.Add('required:[');
            Required(iv);
            Out^.Add(']');
          end
          else if T.Keys[items[j]] = 'type' then
          begin
            Out^.Add('type:');
            if T.Nodes[iv].Kind = hjString then
              Quoted(Upper(iv, 'an item type'))
            else
              TypeList(iv);
          end
          else
          begin
            Out^.Add(T.Keys[items[j]]);
            Out^.Add(':');
            Argument(iv, true);
          end;
        end;
        Out^.Add('}');
      end;
    end;
    if T.Truthy(T.Member(v, 'nullable')) then
    begin
      Comma;
      Out^.Add('nullable:true');
    end;
    if typ = 'OBJECT' then
    begin
      Comma;
      Out^.Add('properties:{');
      sub := T.Member(v, 'properties');
      if (sub >= 0) and (T.Nodes[sub].Kind = hjObject) then
        Parameters(sub, false)
      else
        // the other keys of the property taken as its parameters
        Parameters(v, true);
      Out^.Add('}');
      sub := T.Member(v, 'required');
      if T.Truthy(sub) then
      begin
        Comma;
        Out^.Add('required:[');
        Required(sub);
        Out^.Add(']');
      end;
    end;
    Comma;
    Out^.Add('type:');
    Quoted(typ);
    Out^.Add('}');
  end;
end;

procedure TGemma4Writer.Declaration(const aTool: TLlmTool);
var
  sub: integer;
begin
  Out^.Add('declaration:' + aTool.Name + '{description:');
  Quoted(aTool.Description);
  if aTool.ParametersJson <> '' then
  begin
    T.Parse(aTool.ParametersJson);
    if T.Truthy(0) then
    begin
      if T.Nodes[0].Kind <> hjObject then
        EInferError.RaiseUtf8('TGemma4ChatTemplate: the parameters of % are ' +
          'no object', [aTool.Name]);
      Out^.Add(',parameters:{');
      sub := T.Member(0, 'properties');
      if T.Truthy(sub) then
      begin
        if T.Nodes[sub].Kind <> hjObject then
          EInferError.RaiseUtf8('TGemma4ChatTemplate: the properties of % are ' +
            'no object', [aTool.Name]);
        Out^.Add('properties:{');
        Parameters(sub, false);
        Out^.Add('},');
      end;
      sub := T.Member(0, 'required');
      if T.Truthy(sub) then
      begin
        Out^.Add('required:[');
        Required(sub);
        Out^.Add('],');
      end;
      sub := T.Member(0, 'type');
      if T.Truthy(sub) then
      begin
        // the template closes the parameters only here
        Out^.Add('type:');
        Quoted(Upper(sub, 'the type of the parameters'));
        Out^.Add('}');
      end;
    end;
  end;
  Out^.Add('}');
end;

constructor TGemma4ChatTemplate.Create(aTokenizer: TInferTokenizer);
begin
  inherited Create(aTokenizer);
  fBos := Need('<bos>');
  fTurn := Need('<|turn>');
  fTurnEnd := Need('<turn|>');
  fThink := Need('<|think|>');
  fChannel := Need('<|channel>');
  fChannelEnd := Need('<channel|>');
  fTool := Need('<|tool>');
  fToolEnd := Need('<tool|>');
  fToolCall := Need('<|tool_call>');
  fToolCallEnd := Need('<tool_call|>');
  fToolResponse := Need('<|tool_response>');
  fToolResponseEnd := Need('<tool_response|>');
  fQuote := Need(GEMMA4_QUOTE);
end;

class function TGemma4ChatTemplate.DefaultEnableThinking: boolean;
begin
  result := false;
end;

const
  // ns.prev_message_type of the template
  G4_NONE = 0;
  G4_THINK = 1;
  G4_TOOL = 2;
  G4_TOOL_CALL = 3;
  G4_TOOL_RESPONSE = 4;

procedure TGemma4ChatTemplate.Build(var aOut: TChatSegments;
  const aMessages: TLlmMessageDynArray; const aTools: TLlmToolDynArray;
  aAddGenerationPrompt: boolean; const aOptions: TChatTemplateOptions);
var
  n, first, i, j, k, lastUser: PtrInt;
  prev: integer;
  prevRole, nextRole: TLlmRole;
  hasPrevRole, nextFound, trOut, hasContent: boolean;
  w: TGemma4Writer;
  m: ^TLlmMessage;
  text, name: RawUtf8;
  answered: TBooleanDynArray;
  kids: TIntegerDynArray;
begin
  w.Out := @aOut;
  w.Quote := fQuote;
  n := length(aMessages);
  first := 0;
  prev := G4_NONE;
  aOut.Control(fBos, '<bos>');
  if aOptions.EnableThinking or (aTools <> nil) or
     ((n > 0) and (aMessages[0].Role = lrSystem)) then
  begin
    aOut.Control(fTurn, '<|turn>');
    aOut.Add('system'#10);
    if aOptions.EnableThinking then
    begin
      aOut.Control(fThink, '<|think|>');
      aOut.Add(#10);
      prev := G4_THINK;
    end;
    if (n > 0) and (aMessages[0].Role = lrSystem) then
    begin
      aOut.Add(PythonStrip(aMessages[0].Content));
      first := 1;
    end;
    for i := 0 to high(aTools) do
    begin
      aOut.Control(fTool, '<|tool>');
      w.Declaration(aTools[i]);
      aOut.Control(fToolEnd, '<tool|>');
      prev := G4_TOOL;
    end;
    aOut.Control(fTurnEnd, '<turn|>');
    aOut.Add(#10);
  end;
  // the last user message: reasoning before it is dropped
  lastUser := -1;
  for i := first to n - 1 do
    if aMessages[i].Role = lrUser then
      lastUser := i;
  SetLength(answered, n);
  hasPrevRole := false;
  prevRole := lrUser;
  for i := first to n - 1 do
  begin
    m := @aMessages[i];
    if m^.Role = lrTool then
      if answered[i] then
        continue
      else
        EInferError.RaiseUtf8('%: message % answers a tool call, but no ' +
          'assistant turn with calls is before it', [self, i]);
    prev := G4_NONE;
    // a model turn goes on in the next assistant message
    if not ((m^.Role = lrAssistant) and hasPrevRole and (prevRole = lrAssistant)) then
    begin
      aOut.Control(fTurn, '<|turn>');
      case m^.Role of
        lrAssistant:
          aOut.Add('model'#10);
        lrUser:
          aOut.Add('user'#10);
      else
        aOut.Add('system'#10);
      end;
    end;
    if (m^.Reasoning <> '') and ((i > lastUser) or
       (aOptions.PreserveThinking and (m^.ToolCalls <> nil))) then
    begin
      aOut.Control(fChannel, '<|channel>');
      aOut.Add('thought'#10 + m^.Reasoning + #10);
      aOut.Control(fChannelEnd, '<channel|>');
    end;
    trOut := false;
    if m^.ToolCalls <> nil then
    begin
      for j := 0 to high(m^.ToolCalls) do
      begin
        aOut.Control(fToolCall, '<|tool_call>');
        aOut.Add('call:' + m^.ToolCalls[j].Name + '{');
        if m^.ToolCalls[j].ArgumentsJson <> '' then
        begin
          w.T.Parse(m^.ToolCalls[j].ArgumentsJson);
          // null is the template's none: nothing between the braces
          if w.T.Nodes[0].Kind = hjNull then
            SetLength(kids, 0)
          else if w.T.Nodes[0].Kind <> hjObject then
            EInferError.RaiseUtf8('%: the arguments of call % are no JSON object',
              [self, m^.ToolCalls[j].Name])
          else
            kids := w.Sorted(0);
          for k := 0 to high(kids) do
          begin
            if k > 0 then
              aOut.Add(',');
            aOut.Add(w.T.Keys[kids[k]] + ':');
            w.Argument(w.T.Child[kids[k]], false);
          end;
        end;
        aOut.Add('}');
        aOut.Control(fToolCallEnd, '<tool_call|>');
      end;
      prev := G4_TOOL_CALL;
      // the tool messages that follow: their results, named by the call they
      // answer (the last call with that id, as the template loops)
      k := i + 1;
      while (k < n) and (aMessages[k].Role = lrTool) do
      begin
        name := aMessages[k].Name;
        if name = '' then
          name := 'unknown';
        for j := 0 to high(m^.ToolCalls) do
          if m^.ToolCalls[j].Id = aMessages[k].ToolCallId then
            name := m^.ToolCalls[j].Name;
        aOut.Control(fToolResponse, '<|tool_response>');
        aOut.Add('response:' + name + '{value:');
        w.Quoted(aMessages[k].Content);
        aOut.Add('}');
        aOut.Control(fToolResponseEnd, '<tool_response|>');
        answered[k] := true;
        trOut := true;
        prev := G4_TOOL_RESPONSE;
        inc(k);
      end;
    end;
    if m^.Role = lrAssistant then
      text := Gemma4StripThinking(m^.Content)
    else
      text := PythonStrip(m^.Content);
    aOut.Add(text);
    hasContent := text <> '';
    nextFound := false;
    nextRole := lrUser;
    for j := i + 1 to n - 1 do
      if aMessages[j].Role <> lrTool then
      begin
        nextRole := aMessages[j].Role;
        nextFound := true;
        break;
      end;
    if (prev = G4_TOOL_CALL) and not trOut then
      // calls without results: the model turn waits for them
      aOut.Control(fToolResponse, '<|tool_response>')
    else if (m^.Role = lrAssistant) and nextFound and (nextRole = lrAssistant) and
            ((m^.ToolCalls = nil) or trOut) then
      // continued by the next assistant message
    else if not (trOut and not hasContent and not nextFound) then
    begin
      aOut.Control(fTurnEnd, '<turn|>');
      aOut.Add(#10);
    end;
    prevRole := m^.Role;
    hasPrevRole := true;
  end;
  if aAddGenerationPrompt then
    if (prev <> G4_TOOL_RESPONSE) and (prev <> G4_TOOL_CALL) then
    begin
      aOut.Control(fTurn, '<|turn>');
      aOut.Add('model'#10);
      if not aOptions.EnableThinking then
      begin
        aOut.Control(fChannel, '<|channel>');
        aOut.Add('thought'#10);
        aOut.Control(fChannelEnd, '<channel|>');
      end;
    end
    else if (prev = G4_TOOL_RESPONSE) and aOptions.EnableThinking then
    begin
      aOut.Control(fChannel, '<|channel>');
      aOut.Add('thought'#10);
    end;
  aOut.Flush;
end;

function ChatTemplateOptions(aEnableThinking: boolean): TChatTemplateOptions;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.EnableThinking := aEnableThinking;
end;

function TChatTemplate.Encode(const aMessages: TLlmMessageDynArray;
  const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
  aEnableThinking: boolean): TIntegerDynArray;
begin
  result := EncodeWith(aMessages, aTools, aAddGenerationPrompt,
    ChatTemplateOptions(aEnableThinking));
end;

function TChatTemplate.Render(const aMessages: TLlmMessageDynArray;
  const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
  aEnableThinking: boolean): RawUtf8;
begin
  result := RenderWith(aMessages, aTools, aAddGenerationPrompt,
    ChatTemplateOptions(aEnableThinking));
end;

function TChatTemplate.EncodeWith(const aMessages: TLlmMessageDynArray;
  const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
  const aOptions: TChatTemplateOptions): TIntegerDynArray;
var
  o: TChatSegments;
begin
  FillCharFast(o, SizeOf(o), 0);
  o.Tokenizer := fTokenizer;
  o.WithIds := true;
  try
    Build(o, aMessages, aTools, aAddGenerationPrompt, aOptions);
    result := o.Ids;
    SetLength(result, o.Count);
  finally
    Finalize(o.Ids);
    Finalize(o.Pending);
    Finalize(o.Text);
  end;
end;

function TChatTemplate.RenderWith(const aMessages: TLlmMessageDynArray;
  const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
  const aOptions: TChatTemplateOptions): RawUtf8;
var
  o: TChatSegments;
begin
  FillCharFast(o, SizeOf(o), 0);
  o.Tokenizer := fTokenizer;
  try
    Build(o, aMessages, aTools, aAddGenerationPrompt, aOptions);
    result := o.Text;
  finally
    Finalize(o.Ids);
    Finalize(o.Pending);
    Finalize(o.Text);
  end;
end;

end.
