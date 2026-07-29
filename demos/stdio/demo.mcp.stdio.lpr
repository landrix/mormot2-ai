program demo.mcp.stdio;

{$I mormot.defines.inc}

{$APPTYPE CONSOLE}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.ai.mcp,
  mormot.ai.mcp.stdio,
  demo.mcp.shared;

var
  server: TMcpServer;
  transport: TMcpStdioTransport;
  arg: RawUtf8;
begin
  if ParamCount > 0 then
  begin
    arg := LowerCaseU(StringToUtf8(ParamStr(1)));
    if (arg = 'help') or (arg = '--help') or (arg = '-h') then
    begin
      ConsoleWrite('MCP STDIO Demo', ccLightCyan);
      ConsoleWrite('About: JSON-RPC over stdin/stdout for piping.', ccLightGray);
      ConsoleWrite('Example request:', ccLightGray);
      ConsoleWrite('  {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}', ccYellow);
      ConsoleWrite('Tip: send one JSON-RPC request per line.', ccLightGray);
      exit;
    end;
  end;

  server := CreateDemoServer('StdioDemoServer', '1.0');
  try
    transport := TMcpStdioTransport.Create(server);
    try
      transport.Start;
      ConsoleWrite('MCP STDIO Demo', ccLightCyan);
      ConsoleWrite('Status: ready - stdin in, stdout out', ccLightGray);
      ConsoleWrite('Tip: send one JSON-RPC request per line.', ccLightGray);
      while transport.IsActive do
        Sleep(100);
    finally
      transport.Free;
    end;
  finally
    server.Free;
  end;
end.
