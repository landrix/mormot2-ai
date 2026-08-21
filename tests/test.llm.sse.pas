// - regression tests for mormot.ai.llm.sse (SSE chat-stream parser)
unit test.llm.sse;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  classes,
  mormot.core.base,
  mormot.core.text,
  mormot.core.test,
  mormot.ai.llm.types,
  mormot.ai.llm.sse;

type
  /// accumulates streamed deltas so a test can assert the reconstructed result
  TSseCollector = class
  public
    ContentPieces: integer;
    Text: RawUtf8;
    FinishReason: RawUtf8;
    ToolName: RawUtf8;
    ToolArgs: RawUtf8;
    UsageTotal: integer;
    UsageCount: integer;     // how OFTEN usage arrived, not just the value
    ToolCallDeltas: integer; // deltas carrying a tool call
    ToolSlots: RawUtf8;      // 'index:name ' per call, in arrival order
    DoneCount: integer;
    procedure Handle(const aDelta: TLlmStreamDelta);
  end;

  TTestLlmSse = class(TSynTestCase)
  protected
    // run the SSE text through a fresh parser, writing it in pieceLen-byte
    // chunks (pieceLen=0 means "all at once"); returns the collector + stream
    function Parse(const aSse: RawUtf8; aPieceLen: integer;
      out aStream: TLlmSseStream): TSseCollector;
  published
    procedure StreamWholePayload;
    procedure StreamSingleByteChunks;
    procedure StreamToolCalls;
    procedure FlushTrailingEvent;
    procedure RawBodyOnNonSse;
    procedure TruncatedStreamIsNotDone;
    procedure InbandErrorIsCaptured;
    procedure ParallelToolCallsAllArrive;
    procedure RawBodyIsActuallyBounded;
    procedure EndlessLineIsRefused;
    procedure FalsyErrorKeyDoesNotAbortTheStream;
  end;


implementation

{ TSseCollector }

procedure TSseCollector.Handle(const aDelta: TLlmStreamDelta);
begin
  if aDelta.ContentDelta <> '' then
  begin
    inc(ContentPieces);
    Text := Text + aDelta.ContentDelta;
  end;
  if aDelta.FinishReason <> '' then
    FinishReason := aDelta.FinishReason;
  if aDelta.HasToolCall then
  begin
    if aDelta.ToolCallName <> '' then
      ToolName := aDelta.ToolCallName;
    ToolArgs := ToolArgs + aDelta.ToolCallArgsDelta;
    inc(ToolCallDeltas);
    ToolSlots := ToolSlots +
      FormatUtf8('%:% ', [aDelta.ToolCallIndex, aDelta.ToolCallName]);
  end;
  if aDelta.HasUsage then
  begin
    UsageTotal := aDelta.Usage.TotalTokens;
    inc(UsageCount);
  end;
  if aDelta.Done then
    inc(DoneCount);
end;


const
  // a representative OpenAI chat.completion.chunk stream (content + usage + DONE)
  SSE_CONTENT =
    'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}'#10 +
    #10 +
    'data: {"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}'#10 +
    #10 +
    'data: {"choices":[{"index":0,"delta":{"content":", world"},"finish_reason":null}]}'#10 +
    #10 +
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'#10 +
    #10 +
    'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}'#10 +
    #10 +
    'data: [DONE]'#10#10;

  // a streamed tool call assembled from fragments (note the escaped quotes that
  // live inside the JSON "arguments" string)
  SSE_TOOLCALL =
    'data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}'#10 +
    #10 +
    'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]},"finish_reason":null}]}'#10 +
    #10 +
    'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"NYC\"}"}}]},"finish_reason":null}]}'#10 +
    #10 +
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'#10 +
    #10 +
    'data: [DONE]'#10#10;


{ TTestLlmSse }

function TTestLlmSse.Parse(const aSse: RawUtf8; aPieceLen: integer;
  out aStream: TLlmSseStream): TSseCollector;
var
  coll: TSseCollector;
  off, n: PtrInt;
begin
  coll := TSseCollector.Create;
  aStream := TLlmSseStream.Create(coll.Handle);
  if aPieceLen <= 0 then
    aStream.WriteBuffer(pointer(aSse)^, length(aSse))
  else
  begin
    off := 1;
    while off <= length(aSse) do
    begin
      n := aPieceLen;
      if off + n - 1 > length(aSse) then
        n := length(aSse) - off + 1;
      aStream.WriteBuffer(aSse[off], n);
      inc(off, n);
    end;
  end;
  result := coll;
end;

procedure TTestLlmSse.StreamWholePayload;
var
  s: TLlmSseStream;
  coll: TSseCollector;
begin
  coll := Parse(SSE_CONTENT, 0, s);
  try
    CheckEqual(coll.Text, 'Hello, world', 'reconstructed content');
    CheckEqual(s.FullText, 'Hello, world', 'stream accumulated content');
    CheckEqual(coll.ContentPieces, 2, 'two non-empty content deltas');
    CheckEqual(coll.FinishReason, 'stop', 'finish reason');
    CheckEqual(s.FinishReason, 'stop', 'stream finish reason');
    CheckEqual(coll.UsageTotal, 12, 'usage total tokens');
    CheckEqual(coll.DoneCount, 1, 'exactly one DONE');
    Check(s.Done, 'stream marked done');
  finally
    s.Free;
    coll.Free;
  end;
end;

procedure TTestLlmSse.StreamSingleByteChunks;
var
  s: TLlmSseStream;
  coll: TSseCollector;
begin
  // identical payload, but delivered one byte per Write() - proves the line
  // buffer reassembles events across arbitrary transfer-chunk boundaries
  coll := Parse(SSE_CONTENT, 1, s);
  try
    CheckEqual(coll.Text, 'Hello, world', 'content across 1-byte chunks');
    CheckEqual(coll.ContentPieces, 2, 'two content deltas across chunks');
    CheckEqual(coll.FinishReason, 'stop', 'finish reason across chunks');
    CheckEqual(coll.UsageTotal, 12, 'usage across chunks');
    CheckEqual(coll.DoneCount, 1, 'one DONE across chunks');
    Check(s.Done, 'done across chunks');
  finally
    s.Free;
    coll.Free;
  end;
end;

procedure TTestLlmSse.StreamToolCalls;
var
  s: TLlmSseStream;
  coll: TSseCollector;
begin
  // split at 3 bytes to also exercise boundaries inside the escaped JSON args
  coll := Parse(SSE_TOOLCALL, 3, s);
  try
    CheckEqual(coll.ToolName, 'get_weather', 'assembled tool name');
    CheckEqual(coll.ToolArgs, '{"location":"NYC"}', 'assembled tool arguments');
    CheckEqual(coll.FinishReason, 'tool_calls', 'tool-call finish reason');
    Check(s.Done, 'tool-call stream done');
  finally
    s.Free;
    coll.Free;
  end;
end;

procedure TTestLlmSse.FlushTrailingEvent;
const
  // the final event arrives without a closing newline (truncated last chunk)
  SSE_NOTRAIL =
    'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}'#10 +
    #10 +
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}';
var
  s: TLlmSseStream;
  coll: TSseCollector;
  data: RawUtf8;
begin
  data := SSE_NOTRAIL; // a local var: FPC forbids pointer() on a const string
  coll := TSseCollector.Create;
  s := TLlmSseStream.Create(coll.Handle);
  try
    s.WriteBuffer(pointer(data)^, length(data));
    // without a trailing newline the last event stays buffered until Flush
    CheckEqual(s.FinishReason, '', 'final event buffered before flush');
    s.Flush;
    CheckEqual(coll.Text, 'Hi', 'content seen');
    CheckEqual(s.FinishReason, 'stop', 'final event recovered by flush');
  finally
    s.Free;
    coll.Free;
  end;
end;

procedure TTestLlmSse.RawBodyOnNonSse;
const
  // a non-SSE JSON error body (e.g. HTTP 401) - no "data:" lines at all
  ERR_BODY = '{"error":{"message":"invalid api key"}}';
var
  s: TLlmSseStream;
  coll: TSseCollector;
  data: RawUtf8;
begin
  data := ERR_BODY; // a local var: FPC forbids pointer() on a const string
  coll := TSseCollector.Create;
  s := TLlmSseStream.Create(coll.Handle);
  try
    s.WriteBuffer(pointer(data)^, length(data));
    CheckEqual(s.FullText, '', 'no SSE content parsed');
    // the raw body is retained so the client can surface the provider's message
    Check(Pos(RawUtf8('invalid api key'), s.RawBody) > 0, 'raw error body retained');
  finally
    s.Free;
    coll.Free;
  end;
end;


procedure TTestLlmSse.TruncatedStreamIsNotDone;
const
  // a body that simply stops: two content chunks, no [DONE]. A proxy that
  // terminated the chunked body cleanly, or `Connection: close` with no
  // Content-Length, looks exactly like this - no exception anywhere.
  TRUNCATED =
    'data: {"choices":[{"delta":{"content":"Hel"}}]}'#10#10 +
    'data: {"choices":[{"delta":{"content":"lo"}}]}'#10#10;
var
  s: TLlmSseStream;
  coll: TSseCollector;
begin
  coll := Parse(TRUNCATED, 0, s);
  try
    CheckEqual(s.FullText, 'Hello', 'the partial text did arrive');
    // and THIS is what tells the client the answer is incomplete. Nothing else
    // can: ChatStream returns void, so a caller cannot inspect the result.
    Check(not s.Done, 'a stream without its terminal event is not done');
    CheckEqual(coll.DoneCount, 0, 'and no terminal delta was emitted');
  finally
    s.Free;
    coll.Free;
  end;
end;

procedure TTestLlmSse.InbandErrorIsCaptured;
const
  // HTTP 200, and the failure rides inside the stream - invisible to any status
  // check. The Anthropic wire has always handled this shape; this one swallowed
  // it as an empty chunk and kept the partial text.
  INBAND_ERROR =
    'data: {"choices":[{"delta":{"content":"Hel"}}]}'#10#10 +
    'data: {"error":{"message":"model overloaded","type":"server_error"}}'#10#10;
var
  s: TLlmSseStream;
  coll: TSseCollector;
begin
  coll := Parse(INBAND_ERROR, 0, s);
  try
    CheckEqual(s.StreamError, 'model overloaded', 'the inband error is captured');
    Check(not s.Done, 'and such a stream never reaches its terminal event');
    CheckEqual(s.FullText, 'Hel', 'the partial text is kept for diagnosis');
  finally
    s.Free;
    coll.Free;
  end;
end;


procedure TTestLlmSse.ParallelToolCallsAllArrive;
const
  // one chunk, two calls - the wire interleaves parallel tool calls and our own
  // delta type models the slots. Reading tool_calls[0] only dropped every
  // further call without a trace.
  PARALLEL =
    'data: {"choices":[{"delta":{"content":"go","tool_calls":[' +
    '{"index":0,"id":"a","function":{"name":"get_weather","arguments":"{}"}},' +
    '{"index":1,"id":"b","function":{"name":"get_time","arguments":"{}"}}]}}],' +
    '"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}'#10#10 +
    'data: [DONE]'#10#10;
var
  s: TLlmSseStream;
  coll: TSseCollector;
begin
  coll := Parse(PARALLEL, 0, s);
  try
    CheckEqual(coll.ToolCallDeltas, 2, 'both calls reach the consumer');
    CheckEqual(coll.ToolSlots, '0:get_weather 1:get_time ',
      'each in its own slot, in order');
    // ...and this is why the obvious loop would have been a regression: the
    // extra deltas must NOT repeat what belongs to the chunk itself
    CheckEqual(coll.ContentPieces, 1, 'the text is emitted exactly once');
    CheckEqual(coll.Text, 'go', 'and not duplicated');
    CheckEqual(coll.UsageCount, 1, 'usage is counted exactly once');
    CheckEqual(coll.UsageTotal, 7, 'with the right total');
    CheckEqual(s.FullText, 'go', 'the accumulated text is not doubled either');
    Check(s.Done, 'terminal sentinel still seen');
  finally
    s.Free;
    coll.Free;
  end;
end;


procedure TTestLlmSse.RawBodyIsActuallyBounded;
var
  s: TLlmSseStream;
  coll: TSseCollector;
  huge: RawUtf8;
begin
  // A body that never produces a data: line - an HTML error page from a proxy,
  // or a streaming dump. The comment claimed it was 'bounded in size' while
  // nothing bounded it: MaxResponseBytes is opt-in and NO factory sets one, so
  // this grew for as long as the body did.
  SetLength(huge, 200 shl 10); // 200 KB
  FillCharFast(pointer(huge)^, length(huge), ord('x'));
  coll := Parse(huge, 0, s);
  try
    Check(length(s.RawBody) > 0, 'a diagnostic excerpt is still kept');
    Check(length(s.RawBody) <= 8 shl 10,
      'but it is capped - it exists to make an error readable, nothing more');
  finally
    s.Free;
    coll.Free;
  end;
end;


procedure TTestLlmSse.EndlessLineIsRefused;
var
  s: TLlmSseStream;
  huge: RawUtf8;
  raised: boolean;
begin
  // fRaw was capped, but fBuf - the line buffer - is the REAL memory path: it
  // keeps every byte until an LF arrives, and MaxResponseBytes (the only other
  // bound) is opt-in and set by no factory. One endless line grew it forever.
  SetLength(huge, 5 shl 20); // 5 MB, no LF anywhere
  FillCharFast(pointer(huge)^, length(huge), ord('x'));
  s := TLlmSseStream.Create(nil);
  try
    raised := false;
    try
      s.Write(pointer(huge)^, length(huge));
    except
      on E: ESynException do
        raised := true;
    end;
    Check(raised, 'an SSE line that never ends is refused, not buffered');
  finally
    s.Free;
  end;
end;

procedure TTestLlmSse.FalsyErrorKeyDoesNotAbortTheStream;
const
  // every chunk carries "error":null - what a server with a fixed response
  // struct sends on success. Aborting on the KEY would fail the whole stream on
  // its very first delta.
  NULL_ERROR =
    'data: {"error":null,"choices":[{"delta":{"content":"Hel"}}]}'#10#10 +
    'data: {"error":null,"choices":[{"delta":{"content":"lo"}}]}'#10#10 +
    'data: [DONE]'#10#10;
var
  s: TLlmSseStream;
  coll: TSseCollector;
begin
  coll := Parse(NULL_ERROR, 0, s);
  try
    CheckEqual(s.StreamError, '', 'a null error key is not an inband error');
    CheckEqual(s.FullText, 'Hello', 'and the content still arrives');
    Check(s.Done, 'and the stream completes normally');
  finally
    s.Free;
    coll.Free;
  end;
end;

end.
