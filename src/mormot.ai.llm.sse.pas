/// LandrixAI LLM Client - Server-Sent-Events stream parser
// - part of the mormot.ai.* extension (LandrixAI)
// - clean-room from the SSE (text/event-stream) and OpenAI streaming specs;
//   no third-party code, target license MPL/GPL/LGPL (mORMot contribution)
unit mormot.ai.llm.sse;

{
  *****************************************************************************

    TLlmSseStreamBase is a write-only TStream that an HTTP client fills as the
    response body arrives. mORMot's THttpSocket.GetBody writes each transfer
    chunk straight into the supplied stream, so overriding Write() lets us parse
    the "data:" Server-Sent-Events incrementally and fire a delta callback while
    the model is still generating - no extra socket handling required.

    The base handles the wire-agnostic mechanics (chunk buffering, line framing,
    raw-body retention). Each provider subclass overrides ProcessData to decode
    its own event payloads into the neutral TLlmStreamDelta:
    - TLlmSseStream parses the OpenAI streaming wire (here)
    - the Anthropic Messages SSE wire lives in mormot.ai.llm.anthropic

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  classes,
  mormot.core.base,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.json,
  mormot.ai.llm.types;

type
  /// callback invoked for each streamed delta (including the terminal Done one)
  // - a method pointer (of object) keeps this compatible with FPC 3.2.x and
  //   Delphi; closures/function-references are deliberately avoided
  TLlmStreamDeltaEvent = procedure(const aDelta: TLlmStreamDelta) of object;

  /// write-only TStream that frames an SSE body and dispatches "data:" payloads
  // - pass an instance as the OutStream of THttpClientSocket.Request: its Write
  //   is called per transfer chunk, complete lines are framed, and each non-empty
  //   "data:" payload is handed to the wire-specific ProcessData override
  // - FullText/FinishReason/Done expose the accumulated state for convenience
  TLlmSseStreamBase = class(TStream)
  protected
    fBuf: RawUtf8;        // accumulates bytes until a full line (#10) is seen
    fText: RawUtf8;       // accumulated assistant content
    fFinishReason: RawUtf8;
    fRaw: RawUtf8;        // raw bytes seen before the first "data:" line
    fOnDelta: TLlmStreamDeltaEvent;
    fPosition: Int64;
    fDone: boolean;
    fStreamError: RawUtf8; // an inband provider error, delivered with HTTP 200
    fSawData: boolean;    // stop accumulating fRaw once a real SSE event arrives
    procedure ProcessLine(const aLine: RawUtf8);
    /// decode one non-empty "data:" payload into a delta and fire OnDelta
    // - wire-specific; the subclass also accumulates fText/fFinishReason/fDone
    procedure ProcessData(const aPayload: RawUtf8); virtual; abstract;
    function GetSize: Int64; override;
  public
    /// create the parser with the per-delta callback (may be nil)
    constructor Create(const aOnDelta: TLlmStreamDeltaEvent);
    /// TStream contract: incoming body bytes - parses complete lines
    function Write(const Buffer; Count: Longint): Longint; override;
    /// TStream contract: write-only, always returns 0
    function Read(var Buffer; Count: Longint): Longint; override;
    /// TStream contract: forward-only, just reports the byte position
    function Seek(const Offset: Int64; Origin: TSeekOrigin): Int64; override;
    /// process any buffered trailing line - call once the body is fully received
    // - guards against a final event delivered without a closing newline
    procedure Flush;
    /// the full assistant text accumulated so far
    property FullText: RawUtf8 read fText;
    /// the finish reason once the stream reported one
    property FinishReason: RawUtf8 read fFinishReason;
    /// the raw body seen before any SSE event - non-empty only for a non-SSE
    // (e.g. JSON error) response, so callers can surface the provider's message
    property RawBody: RawUtf8 read fRaw;
    /// true once the terminal sentinel/event was seen
    // - a 2xx stream that ends WITHOUT this was truncated: the caller must
    //   refuse it rather than hand back a partial answer as if it were whole
    property Done: boolean read fDone;
    /// non-empty if the stream carried an inband provider error
    // - such an error arrives with HTTP 200, so the status check cannot see
    //   it. Lives on the base class because BOTH wires can carry one - the
    //   OpenAI side simply had no handler for it until now
    property StreamError: RawUtf8 read fStreamError;
  end;

  /// write-only TStream that decodes an OpenAI-style SSE chat stream
  // - each "data:" line is a self-contained OpenAI chunk (choices[0].delta);
  //   the terminal "[DONE]" sentinel ends the stream
  TLlmSseStream = class(TLlmSseStreamBase)
  protected
    procedure ProcessData(const aPayload: RawUtf8); override;
  end;


implementation

const
  /// how much of a non-SSE response body is kept for the error message
  // - it exists to make a provider's JSON/HTML error readable, nothing more
  SSE_RAW_DIAGNOSTIC_MAX = 8 shl 10;

{ TLlmSseStreamBase }

// index (1-based) of the first #10 in s, or 0 if none - avoids RawUtf8/Char
// ambiguities of the RTL Pos() on a single control byte
function IndexOfLF(const s: RawUtf8): PtrInt;
var
  i: PtrInt;
begin
  for i := 1 to length(s) do
    if s[i] = #10 then
    begin
      result := i;
      exit;
    end;
  result := 0;
end;

constructor TLlmSseStreamBase.Create(const aOnDelta: TLlmStreamDeltaEvent);
begin
  inherited Create;
  fOnDelta := aOnDelta;
end;

function TLlmSseStreamBase.Write(const Buffer; Count: Longint): Longint;
var
  chunk, line: RawUtf8;
  nl: PtrInt;
begin
  result := Count;
  if Count <= 0 then
    exit;
  FastSetString(chunk, @Buffer, Count);
  fBuf := fBuf + chunk;
  inc(fPosition, Count);
  // Keep the leading raw body until the first real SSE event, so a non-SSE
  // (e.g. JSON error) response stays available for diagnostics. Hard cap: the
  // comment here used to claim it was bounded while nothing bounded it -
  // MaxResponseBytes is opt-in and NO factory sets it, so a body that never
  // produces a data: line (a streaming proxy dump, an HTML error page) grew
  // without limit. A few KB is all a diagnostic excerpt needs.
  if not fSawData and
     (length(fRaw) < SSE_RAW_DIAGNOSTIC_MAX) then
    fRaw := fRaw + copy(chunk, 1, SSE_RAW_DIAGNOSTIC_MAX - length(fRaw));
  // process every complete line currently buffered
  repeat
    nl := IndexOfLF(fBuf);
    if nl = 0 then
      break;
    line := copy(fBuf, 1, nl - 1);
    delete(fBuf, 1, nl);
    // strip a trailing CR (SSE uses CRLF or LF)
    if (line <> '') and (line[length(line)] = #13) then
      SetLength(line, length(line) - 1);
    ProcessLine(line);
  until fDone;
end;

procedure TLlmSseStreamBase.ProcessLine(const aLine: RawUtf8);
var
  payload: RawUtf8;
begin
  if aLine = '' then
    exit;              // event boundary / keep-alive blank line
  if aLine[1] = ':' then
    exit;              // SSE comment line
  // only the "data:" field carries the JSON payload (event:/id:/retry: ignored);
  // SSE field names are case-sensitive and lower-case per the spec
  if copy(aLine, 1, 5) <> 'data:' then
    exit;
  fSawData := true; // a real SSE event: stop retaining the raw body
  payload := TrimU(copy(aLine, 6, maxInt));
  if payload = '' then
    exit;
  ProcessData(payload); // wire-specific decode
end;

function TLlmSseStreamBase.Read(var Buffer; Count: Longint): Longint;
begin
  result := 0; // write-only sink
end;

function TLlmSseStreamBase.Seek(const Offset: Int64; Origin: TSeekOrigin): Int64;
begin
  // forward-only stream: report the current byte position so the HTTP client's
  // position bookkeeping (OutStreamInitialPos) stays consistent
  result := fPosition;
end;

function TLlmSseStreamBase.GetSize: Int64;
begin
  // a write-only forward sink: the size is whatever has been written so far
  result := fPosition;
end;

procedure TLlmSseStreamBase.Flush;
begin
  // a well-formed stream ends each event with a newline; this catches a final
  // event delivered in a last chunk without its closing newline
  if fBuf = '' then
    exit;
  if fBuf[length(fBuf)] = #13 then
    SetLength(fBuf, length(fBuf) - 1);
  ProcessLine(fBuf);
  fBuf := '';
end;


{ TLlmSseStream }

procedure TLlmSseStream.ProcessData(const aPayload: RawUtf8);
var
  delta, extra: TLlmStreamDelta;
  v: variant;
  d, choice, deltaObj, tcArr, tc, fn, usage, moreCalls: PDocVariantData;
  i: PtrInt;
begin
  moreCalls := nil;
  // prepare a cleared delta (scalar fields are not auto-initialized in FPC)
  Finalize(delta);
  FillCharFast(delta, SizeOf(delta), 0);

  if aPayload = '[DONE]' then
  begin
    fDone := true;
    delta.Done := true;
    if Assigned(fOnDelta) then
      fOnDelta(delta);
    exit;
  end;

  v := _JsonFastFloat(aPayload); // foreign JSON: see ParseOpenAIEmbeddings
  d := _Safe(v);
  // an inband error event: `data: {"error": {...}}` with HTTP 200. The
  // Anthropic wire has handled this from the start ("rather than hand back a
  // silently truncated answer"); on this side it used to be swallowed as an
  // empty chunk and the caller kept whatever text had arrived so far.
  if d^.GetValueIndex('error') >= 0 then
  begin
    fStreamError := d^.O['error']^.U['message'];
    if fStreamError = '' then
      fStreamError := aPayload;
    exit;
  end;
  if d^.Count > 0 then
  begin
    delta.Raw := v;
    choice := d^.A['choices']^._[0];     // choices[0]; fake-empty if absent
    delta.FinishReason := choice^.U['finish_reason'];
    deltaObj := choice^.O['delta'];
    delta.ContentDelta := deltaObj^.U['content'];
    delta.Role := deltaObj^.U['role'];
    tcArr := deltaObj^.A['tool_calls'];
    if tcArr^.Count > 0 then
    begin
      tc := tcArr^._[0];
      delta.HasToolCall := true;
      delta.ToolCallIndex := tc^.I['index'];
      delta.ToolCallId := tc^.U['id'];
      fn := tc^.O['function'];
      delta.ToolCallName := fn^.U['name'];
      delta.ToolCallArgsDelta := fn^.U['arguments'];
      // one chunk can carry SEVERAL calls - the wire interleaves parallel
      // ones, and TLlmStreamDelta models exactly that ("the tool-call slot
      // this fragment belongs to"). Reading slot 0 only dropped the rest
      // without a trace. Emitted below, after this chunk's own delta.
      if tcArr^.Count > 1 then
        moreCalls := tcArr;
    end;
    usage := d^.O['usage'];
    if usage^.Count > 0 then
    begin
      delta.HasUsage := true;
      delta.Usage.PromptTokens := usage^.I['prompt_tokens'];
      delta.Usage.CompletionTokens := usage^.I['completion_tokens'];
      delta.Usage.TotalTokens := usage^.I['total_tokens'];
    end;
  end;

  // accumulate convenience state
  fText := fText + delta.ContentDelta;
  if delta.FinishReason <> '' then
    fFinishReason := delta.FinishReason;
  if Assigned(fOnDelta) then
    fOnDelta(delta);

  // The further calls of this chunk, one delta each and carrying ONLY the
  // tool fields. Content, finish_reason and usage belong to the CHUNK, not
  // to each call in it: repeating them here would append the text a second
  // time and count the tokens twice - which is why the obvious loop over
  // tool_calls would have been a regression, not a fix. Raw stays empty for
  // the same reason: it is the chunk's, and it is already out.
  if (moreCalls <> nil) and
     Assigned(fOnDelta) then
    for i := 1 to moreCalls^.Count - 1 do
    begin
      Finalize(extra);
      FillCharFast(extra, SizeOf(extra), 0);
      tc := moreCalls^._[i];
      extra.HasToolCall := true;
      extra.ToolCallIndex := tc^.I['index'];
      extra.ToolCallId := tc^.U['id'];
      fn := tc^.O['function'];
      extra.ToolCallName := fn^.U['name'];
      extra.ToolCallArgsDelta := fn^.U['arguments'];
      fOnDelta(extra);
    end;
end;

end.
