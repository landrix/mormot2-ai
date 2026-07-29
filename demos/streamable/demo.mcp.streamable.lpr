program demo.mcp.streamable;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.ai.mcp,
  mormot.ai.mcp.server,
  demo.mcp.shared,
  demo.mcp.claude;

var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  streamer: TClaudeStreamer;
  port: integer;
  cli: TFileName;
  allowClaude: boolean;
begin
  port := 8082;
  if ParamCount > 0 then
    ToInteger(PChar(ParamStr(1)), port);

  // SECURITY: the 'ask_claude' tool hands arbitrary remote prompts to the local
  // Claude CLI (cost abuse + workspace/data exfiltration if reachable). This
  // transport ships with CORS '*' and NO authentication, so the tool is DISABLED
  // by default and must be opted into explicitly. Even then we bind loopback
  // only, so it is never exposed beyond this machine.
  allowClaude := GetEnvironmentVariable('MCP_ENABLE_ASK_CLAUDE') = '1';

  streamer := nil;
  // build the server with the shared demo tools (add, public_ip, version);
  // 'ask_claude' is only added when explicitly enabled (see above)
  server := TMcpServer.Create('StreamableDemoServer', '1.0');
  RegisterDemoServices(server);
  if allowClaude then
    RegisterClaudeTool(server);
  server.Start;
  try
    transport := TMcpStreamableHttpTransport.Create(server);
    try
      transport.Port := port;
      transport.BindAddress := '127.0.0.1'; // loopback only — never all interfaces
      transport.Start;
      if allowClaude then
      begin
        // stream tools/call "ask_claude" token-by-token over the SSE transport
        streamer := TClaudeStreamer.Create;
        transport.OnStreamCall := streamer.HandleStreamCall;
      end;
      ConsoleWrite('MCP Streamable HTTP Demo (protocol ' +
        MCP_PROTOCOL_VERSION + ', stateless)', ccLightCyan);
      ConsoleWrite('Single endpoint: http://127.0.0.1:%/mcp (loopback only)', [port], ccLightGreen);
      if allowClaude then
      begin
        ConsoleWrite('Tools: add, public_ip, ask_claude (streams token-by-token)', ccLightGreen);
        ConsoleWrite('WARNING: ask_claude is ENABLED — it runs the local Claude CLI ' +
          'for any prompt sent to this port.', ccLightRed);
        // report whether the Claude CLI (used by the ask_claude tool) is usable
        cli := FindClaude;
        if ClaudeAvailable(cli) then
          ConsoleWrite('Claude CLI detected: % (ask_claude is live)', [cli], ccLightGreen)
        else
          ConsoleWrite('Claude CLI NOT found: ask_claude will return an error ' +
            '(install + `claude login`)', ccLightRed);
      end
      else
        ConsoleWrite('Tools: add, public_ip (ask_claude disabled — ' +
          'set MCP_ENABLE_ASK_CLAUDE=1 to enable)', ccLightGreen);
      ConsoleWrite('', ccLightGray);
      ConsoleWrite('Connect MCP Inspector (Streamable HTTP) to the URL above, ' +
        'or test with curl:', ccLightGray);
      // Every POST mirrors method (and, for tools/call, the tool name) into the
      // standard headers — the server REQUIRES them and answers 400 + -32020
      // otherwise, so an example without them would simply not run.
      ConsoleWrite('  1. Discover (replaces the removed `initialize` handshake):', ccYellow);
      ConsoleWrite('     curl -X POST http://localhost:%/mcp -i \', [port], ccWhite);
      ConsoleWrite('       -H "Content-Type: application/json" \', ccWhite);
      ConsoleWrite('       -H "Accept: text/event-stream, application/json" \', ccWhite);
      ConsoleWrite('       -H "MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + '" \', ccWhite);
      ConsoleWrite('       -H "Mcp-Method: server/discover" \', ccWhite);
      ConsoleWrite('       -d ''{"jsonrpc":"2.0","id":1,"method":"server/discover",' +
        '"params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"' +
        MCP_PROTOCOL_VERSION + '","io.modelcontextprotocol/clientCapabilities":{},' +
        '"io.modelcontextprotocol/clientInfo":{"name":"curl","version":"1.0"}}}}''', ccWhite);
      ConsoleWrite('', ccLightGray);
      ConsoleWrite('  2. List tools (no session needed - every request stands alone):', ccYellow);
      ConsoleWrite('     curl -X POST http://localhost:%/mcp \', [port], ccWhite);
      ConsoleWrite('       -H "Content-Type: application/json" \', ccWhite);
      ConsoleWrite('       -H "Accept: text/event-stream, application/json" \', ccWhite);
      ConsoleWrite('       -H "MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + '" \', ccWhite);
      ConsoleWrite('       -H "Mcp-Method: tools/list" \', ccWhite);
      ConsoleWrite('       -d ''{"jsonrpc":"2.0","id":2,"method":"tools/list","params":' +
        '{"_meta":{"io.modelcontextprotocol/protocolVersion":"' + MCP_PROTOCOL_VERSION +
        '","io.modelcontextprotocol/clientCapabilities":{}}}}''', ccWhite);
      ConsoleWrite('', ccLightGray);
      ConsoleWrite('  3. Ask Claude (streams the CLI answer back as SSE):', ccYellow);
      ConsoleWrite('     curl -N -X POST http://localhost:%/mcp \', [port], ccWhite);
      ConsoleWrite('       -H "Content-Type: application/json" \', ccWhite);
      ConsoleWrite('       -H "Accept: text/event-stream, application/json" \', ccWhite);
      ConsoleWrite('       -H "MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + '" \', ccWhite);
      ConsoleWrite('       -H "Mcp-Method: tools/call" -H "Mcp-Name: ask_claude" \', ccWhite);
      ConsoleWrite('       -d ''{"jsonrpc":"2.0","id":3,"method":"tools/call","params":' +
        '{"name":"ask_claude","arguments":{"prompt":"What is mORMot in one sentence?"},' +
        '"_meta":{"io.modelcontextprotocol/protocolVersion":"' + MCP_PROTOCOL_VERSION +
        '","io.modelcontextprotocol/clientCapabilities":{}}}}''', ccWhite);
      ConsoleWrite('', ccLightGray);
      ConsoleWrite('  4. There is nothing to terminate: sessions were removed in ' +
        MCP_PROTOCOL_VERSION + '.', ccYellow);
      ConsoleWrite('     GET and DELETE on the endpoint answer 405 Method Not Allowed.', ccLightGray);
      ConsoleWrite('', ccLightGray);
      ConsoleWrite('Press ENTER to stop.', ccLightGray);
      ConsoleWaitForEnterKey;
    finally
      transport.Free;   // stop the server first (no more OnStreamCall calls)
      streamer.Free;     // then the streamer it referenced
    end;
  finally
    server.Free;
  end;
end.
