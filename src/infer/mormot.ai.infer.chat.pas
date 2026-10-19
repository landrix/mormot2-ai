/// native inference engine - chat templates of the supported decoder models
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - TChatTemplate is what the client of every decoder family calls; a family
//   brings its own descendant (Build) - the Jinja template of a model is not
//   run: TQwen3ChatTemplate writes the
//   same text in Pascal (tokenizer_config.json of Qwen/Qwen3-0.6B revision
//   c1899de289a04d12100db370d81485cdf75e47ca), gate tests/data/infer-ref/qwen3-chat;
//   TQwen2ChatTemplate that of Qwen2.5 (Qwen/Qwen2.5-0.5B-Instruct revision
//   7ae557604adf67be50417f59c2c2f167def9a775), gate tests/data/infer-ref/qwen25-chat
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

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

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
      const aTools: TLlmToolDynArray; aAddGenerationPrompt, aEnableThinking: boolean);
      virtual; abstract;
    /// the id of a control token - raises EInferError when the tokenizer has
    // none
    function Need(const aContent: RawUtf8): integer;
  public
    /// keep aTokenizer (not owned) - raises EInferError when nil
    constructor Create(aTokenizer: TInferTokenizer); virtual;
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
    /// the tokenizer the ids belong to
    property Tokenizer: TInferTokenizer
      read fTokenizer;
  end;

  /// a chat template class, as a decoder family registers it
  TChatTemplateClass = class of TChatTemplate;

  /// the chat template of Qwen3: ChatML with <think> blocks and <tools> in the
  // system turn
  TQwen3ChatTemplate = class(TChatTemplate)
  protected
    fImStart, fImEnd, fThink, fThinkEnd, fToolCall, fToolCallEnd,
    fToolResponse, fToolResponseEnd: integer;
    procedure Build(var aOut: TChatSegments; const aMessages: TLlmMessageDynArray;
      const aTools: TLlmToolDynArray; aAddGenerationPrompt, aEnableThinking: boolean);
      override;
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
      const aTools: TLlmToolDynArray; aAddGenerationPrompt, aEnableThinking: boolean);
      override;
  public
    /// look up the control tokens in aTokenizer (not owned) - raises
    // EInferError when one is missing
    constructor Create(aTokenizer: TInferTokenizer); override;
  end;

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
  /// recursive descent over the JSON text, writing the Python form
  TPythonJsonWriter = object
    P, Start: PUtf8Char;
    Out: RawUtf8;
    Depth: integer;
    KeepInexactFloats: boolean;
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
  key: RawUtf8;
  n, i: integer;
begin
  SkipBlanks;
  // a tool schema nests a few levels: a deep one is hostile, refused before
  // the recursion exhausts the stack
  if (P^ in ['{', '[']) and (Depth >= JSON_MAX_DEPTH) then
    Fail('nested too deeply');
  case P^ of
    '{':
      begin
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
              SetLength(keys, n * 2 + 4);
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
      end;
    '[':
      begin
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
      end;
    '"':
      begin
        StringValue(key);
        AddEscaped(Out, pointer(key), length(key));
      end;
    '-', '0'..'9':
      Number;
    't':
      Literal('true');
    'f':
      Literal('false');
    'n':
      Literal('null');
  else
    Fail('value expected');
  end;
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
  aAddGenerationPrompt, aEnableThinking: boolean);
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
      reasoning := '';
      if PosEx('</think>', content) > 0 then
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
    if not aEnableThinking then
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
  aAddGenerationPrompt, aEnableThinking: boolean);
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

function TChatTemplate.Encode(const aMessages: TLlmMessageDynArray;
  const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
  aEnableThinking: boolean): TIntegerDynArray;
var
  o: TChatSegments;
begin
  FillCharFast(o, SizeOf(o), 0);
  o.Tokenizer := fTokenizer;
  o.WithIds := true;
  try
    Build(o, aMessages, aTools, aAddGenerationPrompt, aEnableThinking);
    result := o.Ids;
    SetLength(result, o.Count);
  finally
    Finalize(o.Ids);
    Finalize(o.Pending);
    Finalize(o.Text);
  end;
end;

function TChatTemplate.Render(const aMessages: TLlmMessageDynArray;
  const aTools: TLlmToolDynArray; aAddGenerationPrompt: boolean;
  aEnableThinking: boolean): RawUtf8;
var
  o: TChatSegments;
begin
  FillCharFast(o, SizeOf(o), 0);
  o.Tokenizer := fTokenizer;
  try
    Build(o, aMessages, aTools, aAddGenerationPrompt, aEnableThinking);
    result := o.Text;
  finally
    Finalize(o.Ids);
    Finalize(o.Pending);
    Finalize(o.Text);
  end;
end;

end.
