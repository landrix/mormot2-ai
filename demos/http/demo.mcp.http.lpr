program demo.mcp.http;

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
  mormot.core.unicode,
  mormot.ai.mcp,
  mormot.ai.mcp.server,
  demo.mcp.shared;

var
  server: TMcpServer;
  transport: TMcpHttpTransport;
  port: integer;
begin
  port := 8080;
  if ParamCount > 0 then
    // StringToUtf8, not PChar(): PChar is PWideChar under Delphi
    ToInteger(StringToUtf8(ParamStr(1)), port);

  server := CreateDemoServer('HttpDemoServer', '1.0');
  try
    transport := TMcpHttpTransport.Create(server);
    try
      transport.Port := port;
      transport.Start;
      ConsoleWrite('MCP HTTP Demo', ccLightCyan);
      ConsoleWrite('About: JSON-RPC over HTTP at /mcp.', ccLightGray);
      ConsoleWrite('SUCCESS - listening on http://localhost:%/mcp', [port], ccLightGreen);
      ConsoleWrite('Quick test:', ccLightGray);
      ConsoleWrite('  {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}', ccYellow);
      ConsoleWrite('  {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"public_ip","arguments":{}}}', ccYellow);
      ConsoleWrite('Stop: press ENTER to stop.', ccLightGray);
      ConsoleWaitForEnterKey;
    finally
      transport.Free;
    end;
  finally
    server.Free;
  end;
end.
