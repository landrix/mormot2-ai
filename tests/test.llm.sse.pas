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
  end;
  if aDelta.HasUsage then
    UsageTotal := aDelta.Usage.TotalTokens;
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

end.
