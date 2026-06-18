/// LandrixAI LLM Client - Server-Sent-Events stream parser
// - part of the mormot.ai.* extension (LandrixAI)
// - clean-room from the SSE (text/event-stream) and OpenAI streaming specs;
//   no third-party code, target license MPL/GPL/LGPL (mORMot contribution)
unit mormot.ai.llm.sse;

{
  *****************************************************************************

    TLlmSseStream is a write-only TStream that an HTTP client fills as the
    response body arrives. mORMot's THttpSocket.GetBody writes each transfer
    chunk straight into the supplied stream, so overriding Write() lets us parse
    the "data:" Server-Sent-Events incrementally and fire a delta callback while
    the model is still generating - no extra socket handling required.

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

  /// write-only TStream that decodes an OpenAI-style SSE chat stream
  // - pass an instance as the OutStream of THttpClientSocket.Request: its Write
  //   is called per transfer chunk, and complete "data:" lines are parsed at
  //   once, firing OnDelta with the projected TLlmStreamDelta
  // - FullText/FinishReason/Done expose the accumulated state for convenience
  TLlmSseStream = class(TStream)
  protected
    fBuf: RawUtf8;        // accumulates bytes until a full line (#10) is seen
    fText: RawUtf8;       // accumulated assistant content
    fFinishReason: RawUtf8;
    fOnDelta: TLlmStreamDeltaEvent;
    fPosition: Int64;
    fDone: boolean;
    procedure ProcessLine(const aLine: RawUtf8);
  public
    /// create the parser with the per-delta callback (may be nil)
    constructor Create(const aOnDelta: TLlmStreamDeltaEvent);
    /// TStream contract: incoming body bytes - parses complete lines
    function Write(const Buffer; Count: Longint): Longint; override;
    /// TStream contract: write-only, always returns 0
    function Read(var Buffer; Count: Longint): Longint; override;
    /// TStream contract: forward-only, just reports the byte position
    function Seek(const Offset: Int64; Origin: TSeekOrigin): Int64; override;
    /// the full assistant text accumulated so far
    property FullText: RawUtf8 read fText;
    /// the finish reason once the stream reported one
    property FinishReason: RawUtf8 read fFinishReason;
    /// true once the terminal "[DONE]" sentinel was seen
    property Done: boolean read fDone;
  end;


implementation

{ TLlmSseStream }

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

constructor TLlmSseStream.Create(const aOnDelta: TLlmStreamDeltaEvent);
begin
  inherited Create;
  fOnDelta := aOnDelta;
end;

function TLlmSseStream.Write(const Buffer; Count: Longint): Longint;
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

procedure TLlmSseStream.ProcessLine(const aLine: RawUtf8);
var
  payload: RawUtf8;
  delta: TLlmStreamDelta;
  v: variant;
  d, choice, deltaObj, tcArr, tc, fn, usage: PDocVariantData;
begin
  if aLine = '' then
    exit;              // event boundary / keep-alive blank line
  if aLine[1] = ':' then
    exit;              // SSE comment line
  // only the "data:" field carries the JSON payload (event:/id:/retry: ignored);
  // SSE field names are case-sensitive and lower-case per the spec
  if copy(aLine, 1, 5) <> 'data:' then
    exit;
  payload := TrimU(copy(aLine, 6, maxInt));
  if payload = '' then
    exit;

  // prepare a cleared delta (scalar fields are not auto-initialized in FPC)
  Finalize(delta);
  FillCharFast(delta, SizeOf(delta), 0);

  if payload = '[DONE]' then
  begin
    fDone := true;
    delta.Done := true;
    if Assigned(fOnDelta) then
      fOnDelta(delta);
    exit;
  end;

  v := _JsonFast(payload);
  d := _Safe(v);
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
end;

function TLlmSseStream.Read(var Buffer; Count: Longint): Longint;
begin
  result := 0; // write-only sink
end;

function TLlmSseStream.Seek(const Offset: Int64; Origin: TSeekOrigin): Int64;
begin
  // forward-only stream: report the current byte position so the HTTP client's
  // position bookkeeping (OutStreamInitialPos) stays consistent
  result := fPosition;
end;

end.
