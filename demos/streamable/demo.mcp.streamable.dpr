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
begin
  port := 8082;
  if ParamCount > 0 then
    ToInteger(PChar(ParamStr(1)), port);

  streamer := nil;
  // build the server with the shared demo tools (add, public_ip, version)
  // plus the 'ask_claude' tool that drives the Claude CLI
  server := TMcpServer.Create('StreamableDemoServer', '1.0');
  RegisterDemoServices(server);
  RegisterClaudeTool(server);
  server.Start;
  try
    transport := TMcpStreamableHttpTransport.Create(server);
    try
      transport.Port := port;
      transport.Start;
      // stream tools/call "ask_claude" token-by-token over the SSE transport
      streamer := TClaudeStreamer.Create;
      transport.OnStreamCall := streamer.HandleStreamCall;
      ConsoleWrite('MCP Streamable HTTP Demo (protocol 2025-03-26)', ccLightCyan);
      ConsoleWrite('Single endpoint: http://localhost:%/mcp', [port], ccLightGreen);
      ConsoleWrite('Tools: add, public_ip, ask_claude (streams token-by-token)', ccLightGreen);
      // report whether the Claude CLI (used by the ask_claude tool) is usable
      cli := FindClaude;
      if ClaudeAvailable(cli) then
        ConsoleWrite('Claude CLI detected: % (ask_claude is live)', [cli], ccLightGreen)
      else
        ConsoleWrite('Claude CLI NOT found: ask_claude will return an error ' +
          '(install + `claude login`)', ccLightRed);
      ConsoleWrite('', ccLightGray);
      ConsoleWrite('Connect MCP Inspector (Streamable HTTP) to the URL above, ' +
        'or test with curl:', ccLightGray);
      ConsoleWrite('  1. Initialize:', ccYellow);
      ConsoleWrite('     curl -X POST http://localhost:%/mcp \', [port], ccWhite);
      ConsoleWrite('       -H "Content-Type: application/json" \', ccWhite);
      ConsoleWrite('       -H "Accept: text/event-stream, application/json" \', ccWhite);
      ConsoleWrite('       -d ''{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}'' -i', ccWhite);
      ConsoleWrite('', ccLightGray);
      ConsoleWrite('  2. List tools (use Mcp-Session-Id from step 1):', ccYellow);
      ConsoleWrite('     curl -X POST http://localhost:%/mcp \', [port], ccWhite);
      ConsoleWrite('       -H "Content-Type: application/json" \', ccWhite);
      ConsoleWrite('       -H "Accept: text/event-stream, application/json" \', ccWhite);
      ConsoleWrite('       -H "Mcp-Session-Id: <SESSION_ID>" \', ccWhite);
      ConsoleWrite('       -d ''{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}''', ccWhite);
      ConsoleWrite('', ccLightGray);
      ConsoleWrite('  3. Ask Claude (streams the CLI answer back as SSE):', ccYellow);
      ConsoleWrite('     curl -N -X POST http://localhost:%/mcp \', [port], ccWhite);
      ConsoleWrite('       -H "Content-Type: application/json" \', ccWhite);
      ConsoleWrite('       -H "Accept: text/event-stream, application/json" \', ccWhite);
      ConsoleWrite('       -H "Mcp-Session-Id: <SESSION_ID>" \', ccWhite);
      ConsoleWrite('       -d ''{"jsonrpc":"2.0","id":3,"method":"tools/call","params":' +
        '{"name":"ask_claude","arguments":{"prompt":"What is mORMot in one sentence?"}}}''', ccWhite);
      ConsoleWrite('', ccLightGray);
      ConsoleWrite('  4. Terminate session:', ccYellow);
      ConsoleWrite('     curl -X DELETE http://localhost:%/mcp \', [port], ccWhite);
      ConsoleWrite('       -H "Mcp-Session-Id: <SESSION_ID>"', ccWhite);
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
