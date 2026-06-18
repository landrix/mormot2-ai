program demo.mcp.sse;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

{.$define WITH_LOGS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.core.rtti,
  mormot.core.log,
  mormot.ai.mcp,
  mormot.ai.mcp.server,
  demo.mcp.shared;

var
  server: TMcpServer;
  transport: TMcpSseTransport;
  port: integer;
begin
  // setup logs
  {$ifdef WITH_LOGS}
  with TSynLog.Family do
  begin
    Level := LOG_VERBOSE; // disable logs for benchmarking
    EchoToConsole := LOG_VERBOSE;
    EchoToConsoleBackground := True;
    PerThreadLog := ptIdentifiedInOneFile;
    NoFile := True;
  end;
  {$endif WITH_LOGS}

  port := 8081;
  if ParamCount > 0 then
    ToInteger(PChar(ParamStr(1)), port);

  server := CreateDemoServer('SseDemoServer', '1.0');
  try
    transport := TMcpSseTransport.Create(server);
    try
      transport.Port := port;
      transport.Start;
      ConsoleWrite('MCP SSE Demo', ccLightCyan);
      ConsoleWrite('About: Server-Sent Events transport for MCP.', ccLightGray);
      ConsoleWrite('SUCCESS - listening on http://localhost:%', [port], ccLightGreen);
      ConsoleWrite('Handshake: GET /sse', ccYellow);
      ConsoleWrite('Then POST: /messages?session_id=...', ccYellow);
      ConsoleWrite('Stop: press ENTER to stop.', ccLightGray);
      ConsoleWaitForEnterKey;
    finally
      transport.Free;
    end;
  finally
    server.Free;
  end;
end.
