/// MCP Stdio Transport - Standard Input/Output for CLI Integration
// - this unit is part of the mormot-mcp-server project
// - licensed under MPL/GPL/LGPL three license
// - adopted into the mormot.ai.* namespace for landrix (LandrixAI) from
//   flydev-fr/mormot2-extensions
unit mormot.ai.mcp.stdio;

{
  *****************************************************************************

   MCP Stdio Transport Implementation
    - Line-based JSON-RPC over stdin/stdout
    - Worker thread for non-blocking input
    - Suitable for CLI tool integration

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.threads,
  mormot.ai.mcp,
  mormot.ai.mcp.legacy; // LEGACY-ERA


{ ************ Stdio Transport }

{$I-} // avoid io check in writeln()

type
  /// Worker thread for reading from stdin
  // - runs in background to avoid blocking main thread
  TMcpStdioWorker = class(TSynThread)
  private
    fTransport: TObject;  // TMcpStdioTransport
    fInputClosed: boolean;
    procedure ProcessLine(const aLine: RawUtf8);
  protected
    procedure Execute; override;
  public
    constructor Create(aTransport: TObject); reintroduce;
  end;

  /// Stdio transport for MCP over standard input/output
  // - reads JSON-RPC requests line-by-line from stdin
  // - writes JSON-RPC responses to stdout
  // - suitable for CLI tools and process-based integrations
  // - the session ends when the client closes stdin: IsActive turns false
  TMcpStdioTransport = class(TObject)
  private
    fServer: TMcpServer;
    fWorker: TMcpStdioWorker;
    fActive: boolean;
    fOutputLock: TLightLock;
    fLegacy: TMcpLegacyStdioBridge;     // LEGACY-ERA
    fAcceptLegacyInitialize: boolean;   // LEGACY-ERA
    procedure WriteOutput(const aLine: RawUtf8);
  public
    /// initialize with MCP server instance
    constructor Create(aServer: TMcpServer);
    /// finalize and cleanup
    destructor Destroy; override;
    /// start stdio communication
    procedure Start;
    /// stop stdio communication
    // - the worker reads stdin blocking: while the client keeps stdin open and
    // silent, Stop waits for its next line (or for EOF)
    procedure Stop;
    /// check if transport is active
    // - false once stopped, and once the client closed stdin (EOF)
    function IsActive: boolean;
    /// process a single request (called by worker thread)
    procedure ProcessRequest(const aRequest: RawUtf8);
    /// LEGACY-ERA: also answer the pre-2026-07-28 "initialize" handshake
    // - true by default: clients that open stdio with "initialize" would
    //   otherwise never connect (see mormot.ai.mcp.legacy for why and when
    //   this goes away)
    // - false = modern only: "initialize" is rejected like any request
    //   without _meta
    property AcceptLegacyInitialize: boolean
      read fAcceptLegacyInitialize write fAcceptLegacyInitialize;
  end;


implementation

uses
  classes,
  sysutils,
  mormot.core.json;

{ ************ TMcpStdioWorker }

constructor TMcpStdioWorker.Create(aTransport: TObject);
begin
  inherited Create(false);  // Start immediately
  FreeOnTerminate := false;
  fTransport := aTransport;
end;

procedure TMcpStdioWorker.ProcessLine(const aLine: RawUtf8);
begin
  if (fTransport <> nil) and (aLine <> '') then
    TMcpStdioTransport(fTransport).ProcessRequest(aLine);
end;

procedure TMcpStdioWorker.Execute;
var
  line: string;
begin
  // Blocking read loop. ReadLn waits for the next request by itself - the loop
  // used to sleep 500 ms BEFORE every read, adding half a second to each
  // request. It also never noticed EOF: {$I-} suppresses the I/O exception it
  // waited for, so a client closing stdin left the server process running.
  // Text I/O decodes with the system code page by default (ANSI on Windows, for
  // Delphi and FPC alike); MCP stdio messages are UTF-8 on every platform
  SetTextCodePage(Input, CP_UTF8);
  try
    while not Terminated do
    begin
      if Eof(Input) then
        break;            // the client closed stdin: the session is over
      ReadLn(line);
      if IOResult <> 0 then
        break;
      ProcessLine(StringToUtf8(line));
    end;
  finally
    fInputClosed := true;
  end;
end;


{ ************ TMcpStdioTransport }

constructor TMcpStdioTransport.Create(aServer: TMcpServer);
begin
  inherited Create;
  fServer := aServer;
  fActive := false;
  fOutputLock.Init;
  fLegacy := TMcpLegacyStdioBridge.Create(aServer); // LEGACY-ERA
  fAcceptLegacyInitialize := true;                   // LEGACY-ERA
end;

destructor TMcpStdioTransport.Destroy;
begin
  Stop;
  fLegacy.Free; // LEGACY-ERA
  fOutputLock.Done;
  inherited;
end;

procedure TMcpStdioTransport.WriteOutput(const aLine: RawUtf8);
begin
  fOutputLock.Lock;
  try
    // MCP stdio messages are UTF-8 on every platform. Declaring the Output text
    // file as UTF-8 lets the RTL pass the RawUtf8 bytes through unchanged. The
    // Windows branch used to Utf8Decode into an AnsiString instead, which turned
    // every character outside the ANSI code page into '?'.
    // Still through Output, not the raw handle: a host (or a test) may redirect
    // Output with AssignFile. Set on every write, because that redirection
    // resets the code page of the text file.
    SetTextCodePage(Output, CP_UTF8);
    WriteLn(aLine);
    Flush(Output);  // Ensure immediate delivery
  finally
    fOutputLock.UnLock;
  end;
end;

procedure TMcpStdioTransport.ProcessRequest(const aRequest: RawUtf8);
var
  request, response: RawUtf8;
  route: TMcpLegacyRoute; // LEGACY-ERA
begin
  if not fActive then
    exit;

  try
    request := aRequest;
    // LEGACY-ERA: a client of the "initialize" era is answered or translated by
    // the bridge; everything else reaches the core exactly as it came in
    if fAcceptLegacyInitialize then
      route := fLegacy.Route(request, response)
    else
      route := lrPassThrough;
    if route <> lrAnswered then
    begin
      // Execute MCP request
      response := fServer.ExecuteRequest(request);
      if (route = lrRewritten) and
         (response <> '') then
        response := fLegacy.AdaptResponse(response); // LEGACY-ERA
    end;

    // Send response if not a notification
    if response <> '' then
      WriteOutput(response);

  except
    on E: Exception do
      // Any exception, not only ESynException: an uncaught one would end the
      // worker thread silently and leave a server that reads no more requests.
      // The message goes through QuotedStrJson - a quote or backslash in it
      // used to produce invalid JSON.
      WriteOutput('{"jsonrpc":"2.0","error":{"code":-32603,"message":' +
        QuotedStrJson(StringToUtf8(E.Message)) + '},"id":null}');
  end;
end;

procedure TMcpStdioTransport.Start;
begin
  if fActive then
    exit;

  fActive := true;
  if RunFromSynTests then
    exit;

  // Start worker thread
  fWorker := TMcpStdioWorker.Create(self);
end;

procedure TMcpStdioTransport.Stop;
begin
  if not fActive then
    exit;

  fActive := false;

  // Stop worker thread
  if fWorker <> nil then
  begin
    fWorker.Terminate;
    fWorker.WaitFor;
    FreeAndNilSafe(fWorker);
  end;
end;

function TMcpStdioTransport.IsActive: boolean;
begin
  // EOF on stdin ends the session, so a host loop "while IsActive" returns and
  // the process exits instead of lingering after the client went away
  result := fActive and
            ((fWorker = nil) or
             not fWorker.fInputClosed);
end;


end.
