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
  failed: integer;

  // Build a request the way a conforming 2026-07-28 client does: every request
  // carries its own protocol version and client capabilities in params._meta,
  // because the stateless revision has no handshake that could establish them.
  function Rpc(aId: integer; const aMethod: RawUtf8;
    const aParams: variant): RawUtf8;
  begin
    result := _Safe(_ObjFast([
      'jsonrpc', '2.0',
      'id', aId,
      'method', aMethod,
      'params', McpRequestParams(aParams, 'demo.mcp.jsonrpc', '1.0')]))^.ToJson;
  end;

  // Report each step and remember failures, so the final line cannot claim
  // SUCCESS while individual calls were rejected (it used to do exactly that).
  procedure Step(const aLabel, aResponse: RawUtf8; aColor: TConsoleColor);
  begin
    if _Safe(_JsonFast(aResponse))^.GetValueIndex('error') >= 0 then
    begin
      inc(failed);
      ConsoleWrite('% [FAILED]:'#13#10 + '%', [aLabel, aResponse], ccLightRed);
    end
    else
      ConsoleWrite('%:'#13#10 + '%', [aLabel, aResponse], aColor);
  end;

begin
  failed := 0;
  ConsoleWrite('MCP JSON-RPC Demo (protocol ' + MCP_PROTOCOL_VERSION + ')', ccLightCyan);
  ConsoleWrite('About: runs in-memory JSON-RPC calls to validate tools/resources.', ccLightGray);
  ConsoleWrite('Status: starting...', ccLightGray);
  server := CreateDemoServer('DemoServer', '1.0');
  try
    // server/discover replaces the removed `initialize` handshake
    response := server.ExecuteRequest(Rpc(1, 'server/discover', Null));
    Step('Step 1/6 - Discover', response, ccLightMagenta);

    response := server.ExecuteRequest(Rpc(2, 'tools/list', Null));
    Step('Step 2/6 - Tools list', response, ccLightCyan);

    response := server.ExecuteRequest(Rpc(3, 'tools/call',
      _ObjFast(['name', 'add', 'arguments', _ObjFast(['a', 5, 'b', 3])])));
    Step('Step 3/6 - Tool call add', response, ccLightBlue);

    response := server.ExecuteRequest(Rpc(4, 'tools/call',
      _ObjFast(['name', 'public_ip', 'arguments', _ObjFast([])])));
    Step('Step 4/6 - Tool call public_ip', response, ccLightBlue);

    response := server.ExecuteRequest(Rpc(5, 'resources/list', Null));
    Step('Step 5/6 - Resources list', response, ccLightGreen);

    response := server.ExecuteRequest(Rpc(6, 'resources/read',
      _ObjFast(['uri', 'version://info'])));
    Step('Step 6/6 - Resource read', response, ccGreen);

    if failed = 0 then
      ConsoleWrite('SUCCESS - JSON-RPC demo completed.', ccLightGreen)
    else
      ConsoleWrite('FAILED - % of 6 steps returned an error.', [failed], ccLightRed);
  finally
    server.Free;
  end;
  if failed <> 0 then
    ExitCode := 1;
end.
