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
  mormot.ai.mcp;


{ ************ Stdio Transport }

{$I-} // avoid io check in writeln()

type
  /// Worker thread for reading from stdin
  // - runs in background to avoid blocking main thread
  TMcpStdioWorker = class(TSynThread)
  private
    fTransport: TObject;  // TMcpStdioTransport
    procedure ProcessLine(const aLine: RawUtf8);
  protected
    procedure Execute; override;
  public
    constructor Create(aTransport: TObject);
  end;

  /// Stdio transport for MCP over standard input/output
  // - reads JSON-RPC requests line-by-line from stdin
  // - writes JSON-RPC responses to stdout
  // - suitable for CLI tools and process-based integrations
  TMcpStdioTransport = class(TObject)
  private
    fServer: TMcpServer;
    fWorker: TMcpStdioWorker;
    fActive: boolean;
    fOutputLock: TLightLock;
    procedure WriteOutput(const aLine: RawUtf8);
  public
    /// initialize with MCP server instance
    constructor Create(aServer: TMcpServer);
    /// finalize and cleanup
    destructor Destroy; override;
    /// start stdio communication
    procedure Start;
    /// stop stdio communication
    procedure Stop;
    /// check if transport is active
    function IsActive: boolean;
    /// process a single request (called by worker thread)
    procedure ProcessRequest(const aRequest: RawUtf8);
  end;


implementation

uses
   classes, sysutils, strutils;

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
  while not SleepOrTerminated(500) do
  begin
    try
      // Read line from stdin
      ReadLn(line);

      // Process the line
      ProcessLine(StringToUtf8(line));
    except
      on E: ESynException do
      begin
        // EOF or error - terminate gracefully
        if not Terminated then
          Terminate;
      end;
    end;
  end;
end;


{ ************ TMcpStdioTransport }

constructor TMcpStdioTransport.Create(aServer: TMcpServer);
begin
  inherited Create;
  fServer := aServer;
  fActive := false;
  fOutputLock.Init;
end;

destructor TMcpStdioTransport.Destroy;
begin
  Stop;
  fOutputLock.Done;
  inherited;
end;

procedure TMcpStdioTransport.WriteOutput(const aLine: RawUtf8);
var
  ansiLine: AnsiString;
begin
  fOutputLock.Lock;
  try
    {$ifdef OSWINDOWS}
    ansiLine := Utf8Decode(aLine);
    WriteLn(ansiLine);
    {$else}
    WriteLn(aLine);
    {$endif OSWINDOWS}
    Flush(Output);  // Ensure immediate delivery
  finally
    fOutputLock.UnLock;
  end;
end;

procedure TMcpStdioTransport.ProcessRequest(const aRequest: RawUtf8);
var
  response: RawUtf8;
begin
  if not fActive then
    exit;
    
  try
    // Execute MCP request
    response := fServer.ExecuteRequest(aRequest, 'stdio');
    
    // Send response if not a notification
    if response <> '' then
      WriteOutput(response);
      
  except
    on E: ESynException do
    begin
      // Log error but don't crash
      response := '{"jsonrpc":"2.0","error":{"code":-32603,"message":"' + 
        StringToUtf8(E.Message) + '"},"id":null}';
      WriteOutput(response);
    end;
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
  result := fActive;
end;


end.
