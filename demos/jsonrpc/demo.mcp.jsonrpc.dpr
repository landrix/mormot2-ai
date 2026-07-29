program demo.mcp.jsonrpc;

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
  mormot.core.variants,
  mormot.core.json,
  mormot.ai.mcp,
  demo.mcp.shared;

var
  server: TMcpServer;
  request, response: RawUtf8;
begin
  ConsoleWrite('MCP JSON-RPC Demo', ccLightCyan);
  ConsoleWrite('About: runs in-memory JSON-RPC calls to validate tools/resources.', ccLightGray);
  ConsoleWrite('Status: starting...', ccLightGray);
  server := CreateDemoServer('DemoServer', '1.0');
  try
    request := '{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}';
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 1/6 - Initialize:'#13#10+'%', [response], ccLightMagenta);

    request := '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}';
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 2/6 - Tools list:'#13#10+'%', [response], ccLightCyan);

    request := '{"jsonrpc":"2.0","id":3,"method":"tools/call",' +
      '"params":{"name":"add","arguments":{"a":5,"b":3}}}';
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 3/6 - Tool call add:'#13#10+'%', [response], ccLightBlue);

    request := '{"jsonrpc":"2.0","id":4,"method":"tools/call",' +
      '"params":{"name":"public_ip","arguments":{}}}';
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 4/6 - Tool call public_ip:'#13#10+'%', [response], ccLightBlue);

    request := '{"jsonrpc":"2.0","id":5,"method":"resources/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}';
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 5/6 - Resources list:'#13#10+'%', [response], ccLightGreen);

    request := '{"jsonrpc":"2.0","id":6,"method":"resources/read",' +
      '"params":{"uri":"version://info"}}';
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 6/6 - Resource read:'#13#10+'%', [response], ccGreen);
    ConsoleWrite('SUCCESS - JSON-RPC demo completed.', ccLightGreen);
  finally
    server.Free;
  end;
end.
