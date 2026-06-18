/// 'ask_claude' MCP tool — drives the Claude Code CLI and returns its answer
// - showcases the Streamable HTTP transport carrying real, dynamically
//   generated content: an MCP client calls tools/call "ask_claude", the server
//   shells out to `claude -p --output-format json`, and streams the answer back
//   as a chunked text/event-stream SSE event.
// - prompt is sent on stdin (not the command line) so any quotes/newlines in
//   the user's prompt are passed verbatim with no shell-escaping hazard.
// - requires the `claude` CLI on PATH (or next to the demo .exe) and uses the
//   mormot-os extension (TExternalProcess / RunRedirect stdin overload).
unit demo.mcp.claude;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.rtti,
  mormot.core.text,
  mormot.core.unicode,
  mormot.core.variants,
  mormot.core.json,
  mormot.ai.mcp,
  mormot.ai.mcp.server, // IMcpStreamEmitter / TMcpStreamCall
  mormot.ext.os; // RunRedirect stdin overload + TExternalProcess

type
  /// input parameters for the 'ask_claude' tool
  TAskClaudeParams = record
    /// the question / instruction to send to the Claude CLI
    Prompt: RawUtf8;
  end;

  /// MCP tool that asks the Claude CLI a question and returns its answer
  TAskClaudeTool = class(TMcpToolBase<TAskClaudeParams>)
  protected
    function ExecuteTyped(const aParams: TAskClaudeParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

type
  /// streams a tools/call "ask_claude" token-by-token over the Streamable HTTP
  // transport: runs `claude -p --output-format stream-json` and emits each
  // text_delta as an SSE notifications/progress event, then returns the final
  // tools/call result. Wire its HandleStreamCall to transport.OnStreamCall.
  TClaudeStreamer = class
  protected
    fCli: TFileName;
    // parse newline-delimited stream-json, emit each text token, accumulate text
    procedure DrainBuffer(var aBuf: RawUtf8; const aEmitter: IMcpStreamEmitter;
      var aFullText, aResultText: RawUtf8; var aProgress: integer);
  public
    constructor Create;
    /// TMcpStreamCall handler — set as TMcpStreamableHttpTransport.OnStreamCall
    function HandleStreamCall(const aRequestJson, aSessionId: RawUtf8;
      const aEmitter: IMcpStreamEmitter; out aResponseJson: RawUtf8): boolean;
  end;

/// register the RTTI for TAskClaudeParams (needed for arg deserialization/schema)
procedure EnsureClaudeRtti;

/// locate the Claude CLI: next to the exe, then PATH, then bare 'claude'
function FindClaude: TFileName;

/// true if the located Claude CLI answers `--version` with exit code 0
function ClaudeAvailable(const aCli: TFileName): boolean;

/// create + register the 'ask_claude' tool on the given server
procedure RegisterClaudeTool(const aServer: TMcpServer);


implementation

procedure EnsureClaudeRtti;
begin
  if not RecordHasFields(TypeInfo(TAskClaudeParams)) then
    Rtti.RegisterFromText(TypeInfo(TAskClaudeParams), 'Prompt:RawUtf8');
end;

function QuoteCmd(const aPath: TFileName): TFileName;
begin
  if (aPath <> '') and
     (Pos(' ', aPath) > 0) and
     (aPath[1] <> '"') then
    result := '"' + aPath + '"'
  else
    result := aPath;
end;

function FindClaude: TFileName;
var
  output: RawByteString;
  exitcode: integer;
  nl: PtrInt;
  line: RawUtf8;
  {$ifdef OSWINDOWS}
  i: integer;
const
  EXTS: array[0..2] of TFileName = ('.exe', '.cmd', '.bat');
  {$endif OSWINDOWS}
begin
  {$ifdef OSWINDOWS}
  for i := 0 to high(EXTS) do
  begin
    result := Executable.ProgramFilePath + 'claude' + EXTS[i];
    if FileExists(result) then
      exit;
  end;
  // note: mormot.ext.os's RunRedirect hides the vanilla one, so use the stdin
  // overload with an empty input ('') for these no-input probe commands
  output := RunRedirect('where claude', '', @exitcode, nil, 5000);
  if exitcode = 0 then
  begin
    nl := PosEx(#10, RawUtf8(output));
    if nl = 0 then
      line := TrimU(RawUtf8(output))
    else
      line := TrimU(copy(RawUtf8(output), 1, nl - 1));
    if (line <> '') and
       FileExists(TFileName(line)) then
      exit(TFileName(line));
  end;
  result := 'claude';
  {$else}
  result := Executable.ProgramFilePath + 'claude';
  if FileExists(result) then
    exit;
  output := RunRedirect('/bin/sh -c "command -v claude"', '', @exitcode, nil, 5000);
  if exitcode = 0 then
  begin
    line := TrimU(RawUtf8(output));
    if (line <> '') and
       FileExists(TFileName(line)) then
      exit(TFileName(line));
  end;
  result := 'claude';
  {$endif OSWINDOWS}
end;

function ClaudeAvailable(const aCli: TFileName): boolean;
var
  exitcode: integer;
begin
  RunRedirect(QuoteCmd(aCli) + ' --version', '', @exitcode, nil, 10000);
  result := exitcode = 0;
end;

// pull the "result" field out of a `claude -p --output-format json` envelope,
// tolerating any non-JSON preamble (stderr warnings merged into the pipe)
function ExtractResult(const aJson: RawByteString): RawUtf8;
var
  doc: TDocVariantData;
  raw: RawUtf8;
  start: PtrInt;
begin
  raw := TrimU(RawUtf8(aJson));
  start := PosEx('{', raw);
  if start > 1 then
    raw := copy(raw, start, maxInt);
  if doc.InitJson(raw, JSON_FAST) and
     (doc.GetValueIndex('result') >= 0) then
    result := TrimU(doc.U['result'])
  else
    result := TrimU(RawUtf8(aJson)); // show something even if parsing fails
end;

{ TAskClaudeTool }

function TAskClaudeTool.ExecuteTyped(const aParams: TAskClaudeParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
  cli: TFileName;
  output: RawByteString;
  exitcode: integer;
  answer: RawUtf8;
begin
  builder := TMcpResponseBuilder.Create;
  try
    if TrimU(aParams.Prompt) = '' then
      builder.AddText('{"error":"prompt is required"}')
    else
    begin
      cli := FindClaude;
      // prompt on stdin (verbatim, no shell escaping); JSON envelope on stdout
      output := RunRedirect(
        QuoteCmd(cli) + ' -p --output-format json',
        {stdinput=} aParams.Prompt,
        @exitcode, nil,
        {waitfordelayms=} 120000);
      if exitcode = 0 then
      begin
        answer := ExtractResult(output);
        if answer = '' then
          answer := '(empty response from Claude)';
        builder.AddText(answer);
      end
      else
        builder.AddText(FormatUtf8('Claude CLI failed (exit %): %',
          [exitcode, TrimU(copy(RawUtf8(output), 1, 500))]));
    end;
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

procedure RegisterClaudeTool(const aServer: TMcpServer);
begin
  EnsureClaudeRtti;
  aServer.RegisterTool(TAskClaudeTool.Create('ask_claude',
    'Ask the Claude CLI a question and return its answer ' +
    '(requires the `claude` CLI on PATH)'));
end;

{ TClaudeStreamer }

constructor TClaudeStreamer.Create;
begin
  inherited Create;
  fCli := FindClaude;
end;

procedure TClaudeStreamer.DrainBuffer(var aBuf: RawUtf8;
  const aEmitter: IMcpStreamEmitter; var aFullText, aResultText: RawUtf8;
  var aProgress: integer);
var
  nl: PtrInt;
  line, kind, token: RawUtf8;
  doc: TDocVariantData;
  evt, delta: PDocVariantData;
  notif: variant;
begin
  // stream-json is newline-delimited JSON (NDJSON); process complete lines only
  repeat
    nl := PosEx(#10, aBuf);
    if nl = 0 then
      break;
    line := TrimU(copy(aBuf, 1, nl - 1));
    delete(aBuf, 1, nl);
    if line = '' then
      continue;
    if not doc.InitJson(line, JSON_FAST) then
      continue;
    kind := doc.U['type'];
    if kind = 'result' then
      // authoritative final text (used for the final tool result)
      aResultText := doc.U['result']
    else if kind = 'stream_event' then
    begin
      // {"type":"stream_event","event":{"type":"content_block_delta",
      //   "delta":{"type":"text_delta","text":"..."}}}
      evt := _Safe(doc.GetValueOrNull('event'));
      if evt^.U['type'] = 'content_block_delta' then
      begin
        delta := _Safe(evt^.GetValueOrNull('delta'));
        if delta^.U['type'] = 'text_delta' then
        begin
          token := delta^.U['text'];
          if token <> '' then
          begin
            aFullText := aFullText + token;
            inc(aProgress);
            // emit the token as an MCP progress notification (one SSE event)
            notif := _ObjFast([
              'jsonrpc', '2.0',
              'method', 'notifications/progress',
              'params', _ObjFast([
                'progressToken', 'ask_claude',
                'progress', aProgress,
                'message', token])]);
            aEmitter.Emit(ToUtf8(notif));
          end;
        end;
      end;
    end;
  until false;
end;

function TClaudeStreamer.HandleStreamCall(const aRequestJson, aSessionId: RawUtf8;
  const aEmitter: IMcpStreamEmitter; out aResponseJson: RawUtf8): boolean;
var
  doc: TDocVariantData;
  params, args: PDocVariantData;
  idVar: variant;
  prompt, fullText, resultText, buf: RawUtf8;
  proc: TExternalProcess;
  progress: integer;
  t0: Int64;
  resp: variant;
begin
  result := false;
  aResponseJson := '';
  if not doc.InitJson(aRequestJson, JSON_FAST) then
    exit;
  // only intercept tools/call for the 'ask_claude' tool
  if doc.U['method'] <> 'tools/call' then
    exit;
  params := _Safe(doc.GetValueOrNull('params'));
  if params^.U['name'] <> 'ask_claude' then
    exit;
  // from here we own the response (return true even on error)
  result := true;
  idVar := doc.GetValueOrNull('id');
  args := _Safe(params^.GetValueOrNull('arguments'));
  prompt := args^.U['prompt'];

  if TrimU(prompt) = '' then
    resultText := '(error: prompt is required)'
  else if fCli = '' then
    resultText := '(error: Claude CLI not found)'
  else
  begin
    fullText := '';
    resultText := '';
    buf := '';
    progress := 0;
    proc := TExternalProcess.Create;
    try
      if proc.Start(QuoteCmd(fCli) +
           ' -p --output-format stream-json --verbose --include-partial-messages') then
      begin
        proc.WriteAndCloseStdin(prompt); // prompt on stdin, then EOF
        t0 := GetTickCount64;
        // poll on THIS thread so every emit/WriteString happens on the I/O
        // thread (no cross-thread socket locking); each token flushes at once
        while proc.Running and
              (GetTickCount64 - t0 < 180000) do
        begin
          buf := buf + RawUtf8(proc.ReadAvailable);
          DrainBuffer(buf, aEmitter, fullText, resultText, progress);
          if not proc.HasDataAvailable then
            SleepHiRes(15);
        end;
        // drain anything left after exit
        buf := buf + RawUtf8(proc.ReadAvailable);
        DrainBuffer(buf, aEmitter, fullText, resultText, progress);
      end
      else
        resultText := '(error: failed to start Claude CLI)';
    finally
      proc.Free;
    end;
    if resultText = '' then
      resultText := fullText; // fall back to accumulated tokens
    if resultText = '' then
      resultText := '(empty response from Claude)';
  end;

  // final JSON-RPC tools/call result carrying the full text
  resp := _ObjFast([
    'jsonrpc', '2.0',
    'id', idVar,
    'result', _ObjFast([
      'content', _ArrFast([
        _ObjFast(['type', 'text', 'text', resultText])])])]);
  aResponseJson := ToUtf8(resp);
end;

end.
