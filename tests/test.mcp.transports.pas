// - transport tests for mormot.ai.mcp
unit test.mcp.transports;

interface

{$I mormot.defines.inc}

uses
  {$I mormot.uses.inc}
  sysutils, classes,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.buffers,
  mormot.core.rtti,
  mormot.core.variants,
  mormot.core.json,
  mormot.core.test,
  mormot.net.sock,
  mormot.net.http,
  mormot.net.client,
  mormot.ai.mcp,
  mormot.ai.mcp.server,
  mormot.ai.mcp.stdio;

const
  HTTP_KEEPALIVE_MS = 10000;

type
  {$ifdef FPC}
  TMcpTextRec = TextRec;
  {$else}
  TMcpTextRec = TTextRec;
  {$endif}

  TCalcParams = packed record
    A: integer;
    B: integer;
    Enabled: boolean;
    Name: RawUtf8;
  end;

  TCalcTool = class(TMcpToolBase<TCalcParams>)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  TTestMcpTransports = class(TSynTestCase)
  protected
    procedure EnsureCalcParamsRtti;
    function VariantToInt64Loose(const V: variant; out aValue: Int64): boolean;
    function StartHttpTransport(const aServer: TMcpServer;
      out aTransport: TMcpHttpTransport): integer;
    procedure BackupStdIo(out aInput, aOutput: TMcpTextRec);
    procedure RestoreStdIo(const aInput, aOutput: TMcpTextRec);
    function WaitForFileNotEmpty(const aFileName: TFileName;
      aTimeoutMs: integer): boolean;
    function WaitForFileLineCount(const aFileName: TFileName;
      aLines, aTimeoutMs: integer): boolean;
  published
    procedure HttpTransportPost;
    procedure HttpTransportAcceptHeader;
    procedure HttpTransportOptions;
    procedure HttpTransportWarpSequence;
    procedure HttpTransportNotification;
    procedure HttpTransportBadJson;
    procedure HttpTransportInvalidMethod;
    // the legacy HTTP+SSE transport (TMcpSseTransport) and its session suite
    // were removed with protocol version 2026-07-28 — the spec classifies that
    // transport as Deprecated and this project serves only the modern revision
    procedure StdioTransportProcess;
    procedure StdioTransportMultiple;
    procedure StdioTransportBadJson;
  end;

  TTestMcpStreamableTransport = class(TSynTestCase)
  protected
    procedure EnsureCalcParamsRtti;
    function StartStreamableTransport(const aServer: TMcpServer;
      out aTransport: TMcpStreamableHttpTransport): integer;
    /// assert a buffered response body is a JSON-RPC error with that code
    procedure CheckErrorCode(const aBody: RawUtf8; aExpectedCode: integer);
    /// the payload of the last `data:` line of an SSE response
    function SseDataJson(const aBody: RawUtf8): RawUtf8;
  published
    /// a plain POST is answered without any session handshake
    procedure PostDiscover;
    procedure PostNotification;
    /// GET and DELETE were the session-era verbs and must now yield 405
    procedure GetAndDeleteReturn405;
    procedure OriginValidation;
    procedure MissingAcceptHeader;
    procedure WrongContentType;
    procedure PostStreamsChunked;
    procedure HeaderValidationFailures;
    procedure ParamHeadersMustMatchTheBody;
    procedure ProtocolErrorsUseHttpStatus;
    procedure ConcurrentPosts;
    procedure StreamHookIsValidatedAndContained;
    procedure SubscriptionStreamDelivers;
    procedure CapabilityErrorUsesHttpStatus;
    procedure BearerAuthGuardsTheEndpoint;
    /// the two paths that used to run without the caller they authenticated
    procedure HookAndNotificationSeeTheCaller;
    /// the deferred hand-off re-checks the token and must honour the answer
    procedure DeferredHandoffRefusesALapsedToken;
    /// a verifier that throws must not answer the caller with its own message
    procedure ThrowingVerifierRefusesWithoutLeaking;
    /// a listen stream cannot hold its worker thread forever
    procedure SubscriptionStreamHasAMaximumLifetime;
  end;

implementation

const
  /// the per-request protocol metadata every request must carry since 2026-07-28
  MCP_TEST_META = '"_meta":{"io.modelcontextprotocol/protocolVersion":"' +
    MCP_PROTOCOL_VERSION + '","io.modelcontextprotocol/clientCapabilities":{}}';

/// POST an MCP request the way a conforming 2026-07-28 client would
// - injects the mandatory _meta into params and mirrors method (and name/uri,
//   where the spec requires it) into the standard headers, so the tests state
//   what they are testing instead of repeating protocol boilerplate
function McpPost(aClient: THttpClientSocket; const aEndpoint, aJson: RawUtf8): integer;
var
  doc: TDocVariantData;
  params: PDocVariantData;
  wrapped: variant;
  method, name, headers, body: RawUtf8;
begin
  doc.InitJson(aJson, JSON_FAST);
  body := aJson;
  headers := 'Accept: text/event-stream, application/json'#13#10 +
    'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION;
  if doc.IsObject and doc.GetAsRawUtf8('method', method) then
  begin
    headers := headers + #13#10 + 'Mcp-Method: ' + method;
    // notifications carry no id and need no _meta
    if not VarIsVoid(doc.GetValueOrNull('id')) then
    begin
      // a non-object params is a deliberately malformed envelope: post it as is
      if doc.GetValueIndex('params') >= 0 then
      begin
        if not doc.GetAsDocVariant('params', params) or
           not params^.IsObject then
          exit(aClient.Post(aEndpoint, aJson, JSON_CONTENT_TYPE,
            HTTP_KEEPALIVE_MS, headers));
      end
      else
      begin
        doc.AddValue('params', _ObjFast([]));
        if not doc.GetAsDocVariant('params', params) then
          exit(aClient.Post(aEndpoint, aJson, JSON_CONTENT_TYPE,
            HTTP_KEEPALIVE_MS, headers));
      end;
      // build params through the SAME production helper a real client uses —
      // it merges into an existing _meta (e.g. a progressToken) instead of
      // appending a second key that would shadow the protocol fields
      // materialize first: AddOrUpdateValue may reallocate the values array and
      // invalidate the params pointer we would still be reading from
      wrapped := McpRequestParams(variant(params^), 'mcp.tests', '1.0');
      doc.AddOrUpdateValue('params', wrapped);
      doc.GetAsDocVariant('params', params);
      name := '';
      if method = 'resources/read' then
        name := params^.U['uri']
      else if (method = 'tools/call') or (method = 'prompts/get') then
        name := params^.U['name'];
      if name <> '' then
        headers := headers + #13#10 + 'Mcp-Name: ' + name;
      body := doc.ToJson;
    end;
  end;
  result := aClient.Post(aEndpoint, body, JSON_CONTENT_TYPE,
    HTTP_KEEPALIVE_MS, headers);
end;

{ TCalcTool }

function TCalcTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
begin
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText(FormatUtf8('% + % = %', [aParams.A, aParams.B, aParams.A + aParams.B]));
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

{ TTestMcpTransports }

procedure TTestMcpTransports.EnsureCalcParamsRtti;
begin
  //{$ifndef HASEXTRECORDRTTI}
  if not RecordHasFields(TypeInfo(TCalcParams)) then
    Rtti.RegisterFromText(TypeInfo(TCalcParams),
      'A,B:integer Enabled:boolean Name:RawUtf8');
  //{$endif HASEXTRECORDRTTI}
end;

function TTestMcpTransports.VariantToInt64Loose(const V: variant; out aValue: Int64): boolean;
var
  d: double;
  s: RawUtf8;
  wasString: boolean;
begin
  result := VariantToInt64(V, aValue);
  if result then
    exit;
  if VariantToDouble(V, d) then
  begin
    aValue := trunc(d);
    exit(true);
  end;
  VariantToUtf8(V, s, wasString);
  s := TrimU(s);
  if s = '' then
    exit(false);
  if ToInt64(s, aValue) then
    exit(true);
  if ToDouble(s, d) then
  begin
    aValue := trunc(d);
    exit(true);
  end;
  result := false;
end;

function TTestMcpTransports.StartHttpTransport(const aServer: TMcpServer;
  out aTransport: TMcpHttpTransport): integer;
var
  i: integer;
  port: integer;
  base: integer;
  transport: TMcpHttpTransport;
begin
  aTransport := nil;
  base := 18000 + integer(GetTickCount64 mod 1000);
  for i := 0 to 19 do
  begin
    port := base + i;
    // nil first: if Create itself raises, the handler below frees this variable
    transport := nil;
    try
      transport := TMcpHttpTransport.Create(aServer);
      transport.Port := port;
      transport.Start;
      aTransport := transport;
      result := port;
      exit;
    except
      on Exception do
      begin
        transport.Free;
        continue;
      end;
    end;
  end;
  result := 0;
  Check(false, 'Unable to start HTTP transport on an available port');
end;

procedure TTestMcpTransports.BackupStdIo(out aInput, aOutput: TMcpTextRec);
begin
  aInput := TMcpTextRec(Input);
  aOutput := TMcpTextRec(Output);
end;

procedure TTestMcpTransports.RestoreStdIo(const aInput, aOutput: TMcpTextRec);
begin
  TMcpTextRec(Input) := aInput;
  TMcpTextRec(Output) := aOutput;
end;

function TTestMcpTransports.WaitForFileNotEmpty(const aFileName: TFileName;
  aTimeoutMs: integer): boolean;
var
  endTick: Int64;
begin
  endTick := GetTickCount64 + aTimeoutMs;
  repeat
    if FileSize(aFileName) > 0 then
      exit(true);
    SleepHiRes(10);
  until GetTickCount64 > endTick;
  result := false;
end;

function TTestMcpTransports.WaitForFileLineCount(const aFileName: TFileName;
  aLines, aTimeoutMs: integer): boolean;
var
  endTick: Int64;
  content: RawUtf8;
  i, lines: integer;
begin
  endTick := GetTickCount64 + aTimeoutMs;
  repeat
    content := StringFromFile(aFileName);
    lines := 0;
    for i := 1 to length(content) do
      if content[i] = #10 then
        inc(lines);
    if (content <> '') and (content[length(content)] <> #10) then
      inc(lines);
    if lines >= aLines then
      exit(true);
    SleepHiRes(10);
  until GetTickCount64 > endTick;
  result := false;
end;
procedure TTestMcpTransports.HttpTransportPost;
var
  server: TMcpServer;
  transport: TMcpHttpTransport;
  port: integer;
  tool: IMcpTool;
  client: THttpClientSocket;
  status: integer;
  request: RawUtf8;
  doc, resultDoc, contentDoc, itemDoc: PDocVariantData;
  docVar, resultVar, contentVar, itemVar: variant;
  tmp: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('HttpTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    tool := TCalcTool.Create('calc', 'Add two numbers');
    server.RegisterTool(tool);
    server.Start;

    port := StartHttpTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    request := '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"a":5,"b":3,"enabled":true,"name":"x"}}}';
    status := McpPost(client, '/mcp', request);
    CheckEqual(status, HTTP_SUCCESS);
    docVar := _JsonFast(client.Content);
    doc := _Safe(docVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    contentVar := resultDoc^.GetValueOrNull('content');
    contentDoc := _Safe(contentVar);
    if CheckFailed(contentDoc^.IsArray, 'content not array') then
      exit;
    if CheckFailed(contentDoc^.Count > 0, 'content empty') then
      exit;
    itemVar := contentDoc^.Values[0];
    if CheckFailed(_Safe(itemVar, itemDoc), 'content[0] not object') then
      exit;
    if CheckFailed(itemDoc^.IsObject, 'content[0] not object') then
      exit;
    Check(itemDoc^.GetAsRawUtf8('text', tmp));
    CheckEqual(tmp, '5 + 3 = 8');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpTransports.HttpTransportAcceptHeader;
var
  server: TMcpServer;
  transport: TMcpHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
  request: RawUtf8;
  doc, resultDoc, capsDoc: PDocVariantData;
  docVar, resultVar: variant;
  tmp: RawUtf8;
begin
  server := TMcpServer.Create('HttpTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartHttpTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    // a real-world client now opens with server/discover instead of a handshake
    request := '{"jsonrpc":"2.0","id":0,"method":"server/discover","params":' +
      '{"_meta":{"io.modelcontextprotocol/protocolVersion":"' +
      MCP_PROTOCOL_VERSION + '","io.modelcontextprotocol/clientCapabilities":{},' +
      '"io.modelcontextprotocol/clientInfo":{"name":"dev.warp.Warp",' +
      '"version":"v0.2026.01.28.08.14.stable_04"}}}}';
    status := McpPost(client, '/mcp', request);
    CheckEqual(status, HTTP_SUCCESS);
    docVar := _JsonFast(client.Content);
    doc := _Safe(docVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    Check(resultDoc^.IsObject);
    // nothing is negotiated anymore: discover states what the server supports
    CheckEqual(VariantToUtf8(resultDoc^.A['supportedVersions']^.Values[0]),
      MCP_PROTOCOL_VERSION);
    Check(resultDoc^.GetAsRawUtf8('resultType', tmp));
    CheckEqual(tmp, MCP_RESULT_COMPLETE);
    capsDoc := _Safe(resultDoc^.GetValueOrNull('capabilities'));
    Check(capsDoc^.IsObject);
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpTransports.HttpTransportOptions;
var
  server: TMcpServer;
  transport: TMcpHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
begin
  server := TMcpServer.Create('HttpTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartHttpTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    status := client.Request('/mcp', 'OPTIONS', HTTP_KEEPALIVE_MS, JSON_CONTENT_TYPE_HEADER);
    CheckEqual(status, HTTP_NOCONTENT);
    Check(PosEx('Access-Control-Allow-Origin', client.Headers) > 0);
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpTransports.HttpTransportWarpSequence;
var
  server: TMcpServer;
  transport: TMcpHttpTransport;
  port: integer;
  tool: IMcpTool;
  client: THttpClientSocket;
  status: integer;
  request: RawUtf8;
  doc, resultDoc, listDoc, itemDoc: PDocVariantData;
  tmp: RawUtf8;
  docVar, resultVar, listVar, itemVar: variant;
begin
  server := TMcpServer.Create('HttpTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    tool := TCalcTool.Create('add', 'Add two numbers');
    server.RegisterTool(tool);
    server.Start;
    port := StartHttpTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // a real-world client now opens with server/discover instead of a handshake
    request := '{"jsonrpc":"2.0","id":0,"method":"server/discover","params":' +
      '{"_meta":{"io.modelcontextprotocol/protocolVersion":"' +
      MCP_PROTOCOL_VERSION + '","io.modelcontextprotocol/clientCapabilities":{},' +
      '"io.modelcontextprotocol/clientInfo":{"name":"dev.warp.Warp",' +
      '"version":"v0.2026.01.28.08.14.stable_04"}}}}';
    status := McpPost(client, '/mcp', request);
    CheckEqual(status, HTTP_SUCCESS);
    docVar := _JsonFast(client.Content);
    doc := _Safe(docVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    Check(resultDoc^.IsObject);

    request := '{"jsonrpc":"2.0","method":"notifications/cancelled"}';
    status := McpPost(client, '/mcp', request);
    CheckEqual(status, HTTP_NOCONTENT);

    request := '{"jsonrpc":"2.0","id":1,"method":"resources/list","params":' +
      '{"_meta":{"progressToken":0}}}';
    status := McpPost(client, '/mcp', request);
    CheckEqual(status, HTTP_SUCCESS);
    docVar := _JsonFast(client.Content);
    doc := _Safe(docVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    listVar := resultDoc^.GetValueOrNull('resources');
    listDoc := _Safe(listVar);
    Check(listDoc^.IsArray);

    request := '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":' +
      '{"_meta":{"progressToken":1}}}';
    status := McpPost(client, '/mcp', request);
    CheckEqual(status, HTTP_SUCCESS);
    docVar := _JsonFast(client.Content);
    doc := _Safe(docVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    listVar := resultDoc^.GetValueOrNull('tools');
    listDoc := _Safe(listVar);
    Check(listDoc^.IsArray);
    if listDoc^.Count > 0 then
    begin
      itemVar := listDoc^.Values[0];
      if CheckFailed(_Safe(itemVar, itemDoc), 'tools[0] not object') then
        exit;
      Check(itemDoc^.GetAsRawUtf8('name', tmp));
      CheckEqual(tmp, 'add');
    end;
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpTransports.HttpTransportNotification;
var
  server: TMcpServer;
  transport: TMcpHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
  request: RawUtf8;
begin
  server := TMcpServer.Create('HttpTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartHttpTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    request := '{"jsonrpc":"2.0","method":"notifications/cancelled"}';
    status := McpPost(client, '/mcp', request);
    CheckEqual(status, HTTP_NOCONTENT);
    Check(TrimU(client.Content) = '');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpTransports.HttpTransportBadJson;
var
  server: TMcpServer;
  transport: TMcpHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
  doc, errDoc: PDocVariantData;
  docVar, errVar: variant;
  code: integer;
  code64: Int64;
  okCode: boolean;
  codeVar: variant;
begin
  server := TMcpServer.Create('HttpTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartHttpTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    status := McpPost(client, '/mcp', '{');
    CheckEqual(status, HTTP_SUCCESS);
    docVar := _JsonFast(client.Content);
    doc := _Safe(docVar);
    errVar := doc^.GetValueOrNull('error');
    errDoc := _Safe(errVar);
    Check(errDoc^.IsObject);
    codeVar := errDoc^.GetValueOrDefault('code', Null);
    okCode := VariantToInt64Loose(codeVar, code64);
    if okCode then
      code := integer(code64)
    else
      code := 0;
    Check(code <> 0);
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpTransports.HttpTransportInvalidMethod;
var
  server: TMcpServer;
  transport: TMcpHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
begin
  server := TMcpServer.Create('HttpTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartHttpTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    status := client.Request('/mcp', 'DELETE', HTTP_KEEPALIVE_MS, JSON_CONTENT_TYPE_HEADER);
    CheckEqual(status, HTTP_BADREQUEST);
    Check(PosEx('Only POST method supported', client.Content) > 0);
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpTransports.StdioTransportProcess;
var
  server: TMcpServer;
  transport: TMcpStdioTransport;
  tool: IMcpTool;
  inputFile, outputFile: TFileName;
  request, requestLine, responseText: RawUtf8;
  doc, resultDoc: PDocVariantData;
  docVar, resultVar: variant;
  inputBackup, outputBackup: TMcpTextRec;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('StdioTestServer', '1.0');
  transport := nil;
  inputFile := TemporaryFileName;
  outputFile := TemporaryFileName;
  try
    tool := TCalcTool.Create('calc', 'Add two numbers');
    server.RegisterTool(tool);
    server.Start;

    request := '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"a":2,"b":4,"enabled":true,"name":"x"},' + MCP_TEST_META + '}}'#13#10;
    requestLine := TrimU(request);
    Check(FileFromString(request, inputFile));
    Check(FileFromString('', outputFile));

    BackupStdIo(inputBackup, outputBackup);
    AssignFile(Input, inputFile);
    Reset(Input);
    AssignFile(Output, outputFile);
    Rewrite(Output);
    try
      transport := TMcpStdioTransport.Create(server);
      transport.Start;
      if RunFromSynTests then
        transport.ProcessRequest(requestLine);
      Check(WaitForFileNotEmpty(outputFile, 5000));
      CloseFile(Input);   // unblock ReadLn before Stop/WaitFor
      CloseFile(Output);
      transport.Stop;
    finally
      RestoreStdIo(inputBackup, outputBackup);
    end;
    responseText := StringFromFile(outputFile);
    responseText := TrimU(responseText);
    // only the last line counts; SplitRight works on the RawUtf8 itself (no match
    // -> the whole text) - LastDelimiter takes a string and forced a conversion
    responseText := SplitRight(responseText, #10);
    responseText := TrimU(responseText);
    docVar := _JsonFast(responseText);
    doc := _Safe(docVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    Check(resultDoc^.IsObject);
  finally
    transport.Free;
    server.Free;
    DeleteFile(inputFile);
    DeleteFile(outputFile);
  end;
end;

procedure TTestMcpTransports.StdioTransportMultiple;
var
  server: TMcpServer;
  transport: TMcpStdioTransport;
  tool: IMcpTool;
  inputFile, outputFile: TFileName;
  request1, request2, responseText: RawUtf8;
  inputBackup, outputBackup: TMcpTextRec;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('StdioTestServer', '1.0');
  transport := nil;
  inputFile := TemporaryFileName;
  outputFile := TemporaryFileName;
  try
    tool := TCalcTool.Create('calc', 'Add two numbers');
    server.RegisterTool(tool);
    server.Start;

    request1 := '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"a":2,"b":4,"enabled":true,"name":"x"},' + MCP_TEST_META + '}}'#13#10;
    request2 := '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"a":10,"b":5,"enabled":true,"name":"x"},' + MCP_TEST_META + '}}'#13#10;
    Check(FileFromString(request1 + request2, inputFile));
    Check(FileFromString('', outputFile));

    BackupStdIo(inputBackup, outputBackup);
    AssignFile(Input, inputFile);
    Reset(Input);
    AssignFile(Output, outputFile);
    Rewrite(Output);
    try
      transport := TMcpStdioTransport.Create(server);
      transport.Start;
      if RunFromSynTests then
      begin
        transport.ProcessRequest(TrimU(request1));
        transport.ProcessRequest(TrimU(request2));
      end;
      Check(WaitForFileLineCount(outputFile, 2, 5000));
      CloseFile(Input);   // unblock ReadLn before Stop/WaitFor
      CloseFile(Output);
      transport.Stop;
    finally
      RestoreStdIo(inputBackup, outputBackup);
    end;

    responseText := StringFromFile(outputFile);
    Check(PosEx('2 + 4 = 6', responseText) > 0);
    Check(PosEx('10 + 5 = 15', responseText) > 0);
  finally
    transport.Free;
    server.Free;
    DeleteFile(inputFile);
    DeleteFile(outputFile);
  end;
end;

procedure TTestMcpTransports.StdioTransportBadJson;
var
  server: TMcpServer;
  transport: TMcpStdioTransport;
  inputFile, outputFile: TFileName;
  responseText: RawUtf8;
  doc, errDoc: PDocVariantData;
  docVar, errVar: variant;
  code: integer;
  code64: Int64;
  okCode: boolean;
  inputBackup, outputBackup: TMcpTextRec;
begin
  server := TMcpServer.Create('StdioTestServer', '1.0');
  transport := nil;
  inputFile := TemporaryFileName;
  outputFile := TemporaryFileName;
  try
    server.Start;
    Check(FileFromString('{'#13#10, inputFile));
    Check(FileFromString('', outputFile));

    BackupStdIo(inputBackup, outputBackup);
    AssignFile(Input, inputFile);
    Reset(Input);
    AssignFile(Output, outputFile);
    Rewrite(Output);
    try
      transport := TMcpStdioTransport.Create(server);
      transport.Start;
      if RunFromSynTests then
        transport.ProcessRequest('{');
      Check(WaitForFileNotEmpty(outputFile, 5000));
      CloseFile(Input);   // unblock ReadLn before Stop/WaitFor
      CloseFile(Output);
      transport.Stop;
    finally
      RestoreStdIo(inputBackup, outputBackup);
    end;

    responseText := StringFromFile(outputFile);
    responseText := TrimU(responseText);
    docVar := _JsonFast(responseText);
    doc := _Safe(docVar);
    errVar := doc^.GetValueOrNull('error');
    errDoc := _Safe(errVar);
    Check(errDoc^.IsObject);
    okCode := VariantToInt64Loose(errDoc^.GetValueOrDefault('code', Null), code64);
    if okCode then
      code := integer(code64)
    else
      code := 0;
    Check(code <> 0);
  finally
    transport.Free;
    server.Free;
    DeleteFile(inputFile);
    DeleteFile(outputFile);
  end;
end;

{ TTestMcpStreamableTransport }

procedure TTestMcpStreamableTransport.EnsureCalcParamsRtti;
begin
  if not RecordHasFields(TypeInfo(TCalcParams)) then
    Rtti.RegisterFromText(TypeInfo(TCalcParams),
      'A,B:integer Enabled:boolean Name:RawUtf8');
end;

function TTestMcpStreamableTransport.SseDataJson(const aBody: RawUtf8): RawUtf8;
var
  lines: TRawUtf8DynArray;
  i: PtrInt;
begin
  result := '';
  lines := CsvToRawUtf8DynArray(aBody, #10);
  for i := 0 to high(lines) do
    if IdemPChar(pointer(lines[i]), 'DATA: ') then
      result := TrimU(copy(lines[i], 7, MaxInt));
end;

procedure TTestMcpStreamableTransport.CheckErrorCode(const aBody: RawUtf8;
  aExpectedCode: integer);
var
  docVar, errVar: variant;
  err: PDocVariantData;
  code: Int64;
begin
  docVar := _JsonFast(aBody);
  errVar := _Safe(docVar)^.GetValueOrNull('error');
  err := _Safe(errVar);
  if CheckFailed(err^.IsObject, 'response carries a JSON-RPC error object') then
    exit;
  Check(VariantToInt64(err^.GetValueOrDefault('code', 0), code), 'error code');
  CheckEqual(integer(code), aExpectedCode, 'error code');
end;

function TTestMcpStreamableTransport.StartStreamableTransport(
  const aServer: TMcpServer;
  out aTransport: TMcpStreamableHttpTransport): integer;
var
  i, port, base: integer;
  transport: TMcpStreamableHttpTransport;
begin
  aTransport := nil;
  base := 20000 + integer(GetTickCount64 mod 1000);
  for i := 0 to 19 do
  begin
    port := base + i;
    // nil first: if Create itself raises, the handler below frees this variable
    transport := nil;
    try
      transport := TMcpStreamableHttpTransport.Create(aServer);
      transport.Port := port;
      transport.Start;
      aTransport := transport;
      result := port;
      exit;
    except
      on Exception do
      begin
        transport.Free;
        continue;
      end;
    end;
  end;
  result := 0;
  Check(false, 'Unable to start Streamable transport on an available port');
end;

procedure TTestMcpStreamableTransport.PostDiscover;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
  request, sessionId: RawUtf8;
  lines: TRawUtf8DynArray;
  dataJson: RawUtf8;
  doc, resultDoc: PDocVariantData;
  docVar, resultVar: variant;
  tmp: RawUtf8;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // A modern client opens with server/discover — there is no handshake, so
    // this is an ordinary request carrying its own protocol metadata.
    request := '{"jsonrpc":"2.0","id":1,"method":"server/discover","params":' +
      '{"_meta":{"' + MCP_META_PROTOCOL_VERSION + '":"' + MCP_PROTOCOL_VERSION +
      '","' + MCP_META_CLIENT_CAPABILITIES + '":{}}}}';
    status := client.Post(transport.Endpoint, request, JSON_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS,
      'Accept: text/event-stream, application/json'#13#10 +
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: server/discover');
    CheckEqual(status, HTTP_SUCCESS, 'discover status');

    // No session is minted anymore: the header must be absent entirely
    sessionId := '';
    FindNameValue(client.Headers, 'MCP-SESSION-ID:', sessionId);
    CheckEqual(TrimU(sessionId), '', 'no Mcp-Session-Id may be issued');

    // Verify Content-Type is text/event-stream
    Check(PosEx('text/event-stream', client.ContentType) > 0,
      'response should be SSE');

    // Parse SSE: verify event structure (event + data; ids are gone)
    lines := CsvToRawUtf8DynArray(client.Content, #10);
    Check(length(lines) > 0, 'SSE response must have lines');

    // Verify event: message line exists
    dataJson := '';
    for status := 0 to high(lines) do
    begin
      if IdemPChar(pointer(lines[status]), 'EVENT: ') then
        Check(TrimU(copy(lines[status], 8, MaxInt)) = 'message',
          'SSE event type should be message');
      // stream resumability was removed: an 'id:' line must NOT be emitted
      Check(not IdemPChar(pointer(lines[status]), 'ID: '),
        'SSE events must carry no id (no resumability)');
      if IdemPChar(pointer(lines[status]), 'DATA: ') then
      begin
        dataJson := TrimU(copy(lines[status], 7, MaxInt));
      end;
    end;
    Check(dataJson <> '', 'SSE data line must exist');

    // Parse JSON-RPC response
    docVar := _JsonFast(dataJson);
    doc := _Safe(docVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    Check(resultDoc^.IsObject, 'result must be object');

    // discover reports exactly the one revision we serve, and every result
    // carries the mandatory resultType
    Check(resultDoc^.GetAsRawUtf8('resultType', tmp));
    CheckEqual(tmp, MCP_RESULT_COMPLETE, 'resultType');
    Check(resultDoc^.GetValueIndex('supportedVersions') >= 0,
      'supportedVersions present');
    CheckEqual(VariantToUtf8(resultDoc^.A['supportedVersions']^.Values[0]),
      MCP_PROTOCOL_VERSION, 'reports 2026-07-28');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.PostStreamsChunked;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port, status, i, evtCount: integer;
  client: THttpClientSocket;
  request, sessionId: RawUtf8;
  lines: TRawUtf8DynArray;
begin
  // Regression guard for the streaming fix: the Streamable HTTP transport must
  // deliver a real chunked text/event-stream (held-open, no fixed
  // Content-Length), NOT a single buffered body. The previous implementation
  // assigned Ctxt.OutContent and the framework sent one Content-Length response;
  // that path is now forbidden by the hfTransferChunked assertions below.
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // initialize
    request := '{"jsonrpc":"2.0","id":1,"method":"server/discover"}';
    status := McpPost(client, transport.Endpoint, request);
    CheckEqual(status, HTTP_SUCCESS, 'initialize status');
    Check(hfTransferChunked in client.Http.HeaderFlags,
      'response must be a chunked SSE stream, not a buffered body');

    // a second request on the SAME socket: JSON-RPC batching is not part of
    // this revision (one request per POST), so what must be proven here is that
    // the connection survives the first stream and is reusable for keep-alive
    request := '{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}';
    status := McpPost(client, transport.Endpoint, request);
    CheckEqual(status, HTTP_SUCCESS, 'second request (keep-alive reuse after stream)');
    Check(hfTransferChunked in client.Http.HeaderFlags,
      'second response must be chunked');
    Check(PosEx('text/event-stream', client.ContentType) > 0, 'SSE content-type');

    // exactly one SSE event carrying the response
    lines := CsvToRawUtf8DynArray(client.Content, #10);
    evtCount := 0;
    for i := 0 to high(lines) do
      if IdemPChar(pointer(lines[i]), 'EVENT: MESSAGE') then
        inc(evtCount);
    CheckEqual(evtCount, 1, 'one discrete SSE event streamed');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.PostNotification;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
  request, sessionId: RawUtf8;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // Initialize first
    request := '{"jsonrpc":"2.0","id":1,"method":"server/discover"}';
    status := McpPost(client, transport.Endpoint, request);
    CheckEqual(status, HTTP_SUCCESS);

    // Send notification — should get 202 Accepted
    request := '{"jsonrpc":"2.0","method":"notifications/cancelled"}';
    status := McpPost(client, transport.Endpoint, request);
    CheckEqual(status, HTTP_ACCEPTED, 'notification should return 202');
    Check(TrimU(client.Content) = '', 'notification body should be empty');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.GetAndDeleteReturn405;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    // GET was the standalone notification stream, DELETE the session teardown.
    // Both verbs are gone with protocol sessions; the spec asks for 405 so an
    // older client can tell "removed" from "never existed". DELETE is routed
    // explicitly because RunMethods does not publish it at all — which is
    // exactly why it needs its own assertion here.
    status := client.Get(transport.Endpoint, HTTP_KEEPALIVE_MS,
      'Accept: text/event-stream');
    CheckEqual(status, HTTP_NOTALLOWED, 'GET should return 405');
    status := client.Request(transport.Endpoint, 'DELETE', HTTP_KEEPALIVE_MS,
      'Accept: text/event-stream');
    CheckEqual(status, HTTP_NOTALLOWED, 'DELETE should return 405');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.OriginValidation;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
  request: RawUtf8;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartStreamableTransport(server, transport);
    // Set specific CORS origin
    transport.CorsOrigins := 'https://allowed.example.com';
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // Request with wrong Origin should get 403
    request := '{"jsonrpc":"2.0","id":1,"method":"server/discover"}';
    status := client.Post(transport.Endpoint, request, JSON_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS,
      'Accept: text/event-stream, application/json'#13#10 +
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: server/discover'#13#10 +
      'Origin: https://evil.example.com');
    CheckEqual(status, HTTP_FORBIDDEN, 'wrong origin should return 403');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.MissingAcceptHeader;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
  request: RawUtf8;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // Accept header is filtered by mORMot's THttpAsyncServer (HeadersUnFiltered=false)
    // so the server cannot validate it. The transport is lenient: if Accept is not
    // found in InHeaders, the request is accepted (same as SSE transport behavior).
    request := '{"jsonrpc":"2.0","id":1,"method":"server/discover"}';
    status := McpPost(client, transport.Endpoint, request);
    CheckEqual(status, HTTP_SUCCESS, 'should succeed when Accept is filtered');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.WrongContentType;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port: integer;
  client: THttpClientSocket;
  status: integer;
  request: RawUtf8;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    request := '{"jsonrpc":"2.0","id":1,"method":"server/discover"}';
    status := client.Post(transport.Endpoint, request, TEXT_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS,
      'Accept: text/event-stream, application/json');
    CheckEqual(status, 415, 'wrong content-type should return 415');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.ParamHeadersMustMatchTheBody;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  tool: TCalcTool;
  port, status: integer;
  client: THttpClientSocket;
  stdHeaders: RawUtf8;

  // a tools/call for 'calc' with the given arguments and the given extra headers
  function CallWith(const aArguments, aExtraHeaders: RawUtf8): integer;
  begin
    result := client.Post(transport.Endpoint,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":' + aArguments + ',' + MCP_TEST_META + '}}',
      JSON_CONTENT_TYPE, HTTP_KEEPALIVE_MS,
      'Accept: text/event-stream, application/json'#13#10 +
      stdHeaders + aExtraHeaders);
  end;

begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    tool := TCalcTool.Create('calc', 'Add two numbers');
    // A is an integer, Name a string — both legal to mirror
    tool.MirrorToHeader('A', 'A-Value');
    tool.MirrorToHeader('Name', 'Who');
    server.RegisterTool(tool);
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    stdHeaders := 'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: tools/call'#13#10 + 'Mcp-Name: calc'#13#10;

    // matching headers pass straight through
    status := CallWith('{"A":42,"B":1,"Name":"bob"}',
      'Mcp-Param-A-Value: 42'#13#10 + 'Mcp-Param-Who: bob'#13#10);
    CheckEqual(status, HTTP_SUCCESS, 'matching Mcp-Param headers are accepted');

    // a mirrored argument WITHOUT its header: a conforming client always sends
    // it, and letting it through would leave the header unchecked
    status := CallWith('{"A":42,"B":1,"Name":"bob"}',
      'Mcp-Param-Who: bob'#13#10);
    CheckEqual(status, HTTP_BADREQUEST, 'a missing Mcp-Param header -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // header and body disagree — the split source of truth the rule prevents
    status := CallWith('{"A":42,"B":1,"Name":"bob"}',
      'Mcp-Param-A-Value: 7'#13#10 + 'Mcp-Param-Who: bob'#13#10);
    CheckEqual(status, HTTP_BADREQUEST, 'a disagreeing Mcp-Param header -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // a header for an argument that is not there at all
    status := CallWith('{"A":42,"B":1}',
      'Mcp-Param-A-Value: 42'#13#10 + 'Mcp-Param-Who: ghost'#13#10);
    CheckEqual(status, HTTP_BADREQUEST, 'a header without its argument -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // an absent argument with no header is fine: nothing to disagree about
    status := CallWith('{"A":42,"B":1}', 'Mcp-Param-A-Value: 42'#13#10);
    CheckEqual(status, HTTP_SUCCESS, 'an absent argument needs no header');

    // "servers SHOULD compare integer values NUMERICALLY": +42 is 42
    status := CallWith('{"A":42,"B":1}', 'Mcp-Param-A-Value: +42'#13#10);
    CheckEqual(status, HTTP_SUCCESS, 'integers compare numerically, not as text');

    // a value that is not header-safe travels base64-encoded and MUST be
    // decoded before the comparison, or every non-ASCII value would mismatch.
    // The source stays pure ASCII: the u-umlaut travels as the JSON escape
    // \u00fc in the body and as
    // its precomputed UTF-8 base64 in the header. A #$C3#$BC literal is two
    // WideChars (U+00C3 U+00BC) in Delphi and was re-encoded into the body but
    // not into BinToBase64 - the test then failed on the literal, not the server.
    status := CallWith('{"A":1,"B":1,"Name":"M\u00fcller"}',
      'Mcp-Param-A-Value: 1'#13#10 +
      'Mcp-Param-Who: =?base64?TcO8bGxlcg==?='#13#10);
    CheckEqual(status, HTTP_SUCCESS, 'a base64 sentinel value is decoded first');
  finally
    client.Free;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.HeaderValidationFailures;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port, status: integer;
  client: THttpClientSocket;
  request, base: RawUtf8;

  // POST with fully controlled headers — the point of these cases is what the
  // headers say, so McpPost (which derives them) must not be used here
  function RawPost(const aBody, aHeaders: RawUtf8): integer;
  begin
    result := client.Post(transport.Endpoint, aBody, JSON_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS, 'Accept: text/event-stream, application/json'#13#10 +
        aHeaders);
  end;

begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.RegisterTool(TCalcTool.Create('calc', 'Add two numbers'));
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    base := '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{' +
      MCP_TEST_META + '}}';

    // Every one of these MUST be 400 + a -32020 body. The headers exist so an
    // intermediary can route without parsing; if header and body disagree, the
    // proxy and this server would act on different data — which is the exact
    // split-source-of-truth the rule prevents.

    // 1. MCP-Protocol-Version missing entirely
    status := RawPost(base, 'Mcp-Method: tools/list');
    CheckEqual(status, HTTP_BADREQUEST, 'missing MCP-Protocol-Version -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // 2. Mcp-Method missing
    status := RawPost(base, 'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION);
    CheckEqual(status, HTTP_BADREQUEST, 'missing Mcp-Method -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // 3. Mcp-Method disagrees with the body
    status := RawPost(base, 'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION +
      #13#10 + 'Mcp-Method: resources/list');
    CheckEqual(status, HTTP_BADREQUEST, 'Mcp-Method mismatch -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // 4. protocol version header disagrees with _meta
    status := RawPost(base, 'MCP-Protocol-Version: 2025-11-25'#13#10 +
      'Mcp-Method: tools/list');
    CheckEqual(status, HTTP_BADREQUEST, 'version header mismatch -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // 5. header present, but the body carries no _meta at all: an absent body
    // value is a mismatch too, otherwise the header would go unchecked
    status := RawPost('{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: tools/list');
    CheckEqual(status, HTTP_BADREQUEST, 'absent _meta version -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // 6. tools/call without the required Mcp-Name
    request := '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":' +
      '{"name":"calc","arguments":{"a":1,"b":2},' + MCP_TEST_META + '}}';
    status := RawPost(request, 'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION +
      #13#10 + 'Mcp-Method: tools/call');
    CheckEqual(status, HTTP_BADREQUEST, 'missing Mcp-Name -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // 7. Mcp-Name disagrees with params.name
    status := RawPost(request, 'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION +
      #13#10 + 'Mcp-Method: tools/call'#13#10 + 'Mcp-Name: other');
    CheckEqual(status, HTTP_BADREQUEST, 'Mcp-Name mismatch -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);

    // 8. the base64 sentinel decodes and then matches
    status := RawPost(request, 'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION +
      #13#10 + 'Mcp-Method: tools/call'#13#10 +
      'Mcp-Name: =?base64?' + BinToBase64('calc') + '?=');
    CheckEqual(status, HTTP_SUCCESS, 'a base64 sentinel Mcp-Name is accepted');

    // 9. the sentinel markers are CASE-SENSITIVE per spec: '=?BASE64?..?=' is
    // a literal name a client is required to send base64-encoded, so decoding
    // it would silently rewrite a legitimate value
    status := RawPost(request, 'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION +
      #13#10 + 'Mcp-Method: tools/call'#13#10 +
      'Mcp-Name: =?BASE64?' + BinToBase64('calc') + '?=');
    CheckEqual(status, HTTP_BADREQUEST, 'uppercase sentinel must NOT decode');
    CheckErrorCode(client.Content, MCP_ERROR_HEADER_MISMATCH);
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.ProtocolErrorsUseHttpStatus;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port, status: integer;
  client: THttpClientSocket;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // A protocol rejection must NOT be streamed: the SSE head is written before
    // the outcome is known, so a streamed rejection is HTTP 200 by construction
    // and a client can no longer tell a refusal from a result. Each case below
    // therefore asserts a buffered JSON body with the status the spec requires.

    // unimplemented method -> 404 (spec: "MUST respond with 404 Not Found and a
    // JSON-RPC error with code -32601"), which also distinguishes this server
    // from a legacy one that simply does not host the endpoint
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":1,"method":"tools/nope"}');
    CheckEqual(status, HTTP_NOTFOUND, 'unknown method -> 404');
    CheckErrorCode(client.Content, JSONRPC_METHOD_NOT_FOUND);
    Check(PosEx('text/event-stream', client.ContentType) = 0,
      'a rejection is buffered JSON, never a stream');

    // unsupported protocol version -> 400 + -32022, listing what we speak
    status := client.Post(transport.Endpoint,
      '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{"_meta":{"' +
      MCP_META_PROTOCOL_VERSION + '":"2025-11-25","' +
      MCP_META_CLIENT_CAPABILITIES + '":{}}}}', JSON_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS, 'Accept: application/json'#13#10 +
      'MCP-Protocol-Version: 2025-11-25'#13#10 + 'Mcp-Method: tools/list');
    CheckEqual(status, HTTP_BADREQUEST, 'unsupported version -> 400');
    CheckErrorCode(client.Content, MCP_ERROR_UNSUPPORTED_PROTOCOL_VERSION);
    Check(PosEx(MCP_PROTOCOL_VERSION, client.Content) > 0,
      'the error names the supported version');

    // missing _meta -> 400 + -32602
    status := client.Post(transport.Endpoint,
      '{"jsonrpc":"2.0","id":3,"method":"tools/list"}', JSON_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS, 'Accept: application/json'#13#10 +
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: tools/list');
    CheckEqual(status, HTTP_BADREQUEST, 'missing _meta -> 400');

    // a JSON-RPC batch (array body) is not part of this revision — and the
    // rejection must still be a recognizable JSON-RPC error, because the spec's
    // compatibility probe reads an unrecognizable 400 body as "legacy server"
    // and would downgrade to the removed initialize handshake
    status := client.Post(transport.Endpoint,
      '[{"jsonrpc":"2.0","id":4,"method":"tools/list"}]', JSON_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS, 'Accept: application/json'#13#10 +
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: tools/list');
    CheckEqual(status, HTTP_BADREQUEST, 'array body -> 400');
    CheckErrorCode(client.Content, JSONRPC_INVALID_REQUEST);

    // a request that IS dispatchable still streams as before
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":5,"method":"tools/list"}');
    CheckEqual(status, HTTP_SUCCESS, 'a valid request -> 200');
    Check(PosEx('text/event-stream', client.ContentType) > 0,
      'a valid request is streamed');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

type
  /// OnStreamCall hooks used by StreamHookIsValidatedAndContained
  TStreamHookProbe = class
  public
    Called: boolean;
    /// what the hook saw of the caller - the point of the aAuthCtx parameter
    SeenUserId: RawUtf8;
    SeenAuthenticated: boolean;
    /// answers with a bare result — no resultType, no serverInfo
    function Bare(const aRequestJson: RawUtf8; const aEmitter: IMcpStreamEmitter;
      const aAuthCtx: TMcpAuthContext; out aResponseJson: RawUtf8): boolean;
    /// raises, the way a buggy or hostile hook would
    function Throws(const aRequestJson: RawUtf8;
      const aEmitter: IMcpStreamEmitter; const aAuthCtx: TMcpAuthContext;
      out aResponseJson: RawUtf8): boolean;
    /// answers a Multi Round-Trip Request: a hook may do that too
    function NeedsInput(const aRequestJson: RawUtf8;
      const aEmitter: IMcpStreamEmitter; const aAuthCtx: TMcpAuthContext;
      out aResponseJson: RawUtf8): boolean;
  end;

function TStreamHookProbe.Bare(const aRequestJson: RawUtf8;
  const aEmitter: IMcpStreamEmitter; const aAuthCtx: TMcpAuthContext;
  out aResponseJson: RawUtf8): boolean;
begin
  Called := true;
  SeenUserId := aAuthCtx.UserID;
  SeenAuthenticated := aAuthCtx.IsAuthenticated;
  aResponseJson := '{"jsonrpc":"2.0","id":1,"result":{"content":"from-hook"}}';
  result := true;
end;

function TStreamHookProbe.Throws(const aRequestJson: RawUtf8;
  const aEmitter: IMcpStreamEmitter; const aAuthCtx: TMcpAuthContext;
  out aResponseJson: RawUtf8): boolean;
begin
  aResponseJson := '';
  result := false; // never reached — keeps the compiler from warning
  raise EMcpException.CreateU('hook blew up');
end;

function TStreamHookProbe.NeedsInput(const aRequestJson: RawUtf8;
  const aEmitter: IMcpStreamEmitter; const aAuthCtx: TMcpAuthContext;
  out aResponseJson: RawUtf8): boolean;
begin
  Called := true;
  aResponseJson := '{"jsonrpc":"2.0","id":1,"result":{' +
    '"resultType":"' + MCP_RESULT_INPUT_REQUIRED + '",' +
    '"requestState":"hook-state"}}';
  result := true;
end;

procedure TTestMcpStreamableTransport.StreamHookIsValidatedAndContained;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  probe: TStreamHookProbe;
  port, status: integer;
  client: THttpClientSocket;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  probe := TStreamHookProbe.Create;
  try
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // 1. A hook must never see a request the protocol layer rejects. Before
    // the preflight moved ahead of the deferral, a hook answering `true` ran
    // on unvalidated input — for the Claude demo that means executing the
    // local CLI for a request the server was supposed to refuse.
    transport.OnStreamCall := probe.Bare;
    probe.Called := false;
    status := client.Post(transport.Endpoint,
      '{"jsonrpc":"2.0","id":1,"method":"tools/list"}', JSON_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS, 'Accept: application/json'#13#10 +
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: tools/list');
    CheckEqual(status, HTTP_BADREQUEST, 'invalid request -> 400');
    Check(not probe.Called, 'the hook must not run on a rejected request');

    // 2. A hook that IS reached still cannot ship a result without the
    // mandatory protocol fields: the transport finalizes what it returns.
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":1,"method":"tools/list"}');
    CheckEqual(status, HTTP_SUCCESS, 'valid request reaches the hook');
    Check(probe.Called, 'the hook ran');
    Check(PosEx('from-hook', client.Content) > 0, 'the hook answer is used');
    Check(PosEx('"resultType":"' + MCP_RESULT_COMPLETE + '"', client.Content) > 0,
      'a hook result is stamped with resultType');
    Check(PosEx(MCP_META_SERVER_INFO, client.Content) > 0,
      'a hook result carries serverInfo');
    // tools/list is a cacheable method: answering it through a hook does not
    // exempt the response from the hints the spec requires on that result
    Check(PosEx('"ttlMs"', client.Content) > 0, 'a hook result carries ttlMs');
    Check(PosEx('"cacheScope"', client.Content) > 0,
      'a hook result carries cacheScope');

    // 3. A hook that raises must not escape into the connection's OnRead: it
    // has no exception handler and would tear the worker down mid-stream.
    transport.OnStreamCall := probe.Throws;
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":2,"method":"tools/list"}');
    CheckEqual(status, HTTP_SUCCESS, 'a throwing hook still answers');
    CheckErrorCode(SseDataJson(client.Content), JSONRPC_INTERNAL_ERROR);

    // 4. A hook may answer a Multi Round-Trip Request, and finalization must
    // not flatten that: overwriting `input_required` with `complete` would hand
    // the client a "finished" result still carrying requestState, which it
    // would never look at — the round trip would stall with no error anywhere.
    transport.OnStreamCall := probe.NeedsInput;
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":' +
      '{"name":"whatever"}}');
    CheckEqual(status, HTTP_SUCCESS, 'an interim hook result still answers 200');
    Check(PosEx('"resultType":"' + MCP_RESULT_INPUT_REQUIRED + '"',
      client.Content) > 0, 'the hook keeps its own resultType');
    Check(PosEx('hook-state', client.Content) > 0, 'and its requestState');
    Check(PosEx('"ttlMs"', client.Content) = 0,
      'an interim result carries no caching hints, whoever produced it');

    // and the server is still alive afterwards
    transport.OnStreamCall := nil;
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":3,"method":"tools/list"}');
    CheckEqual(status, HTTP_SUCCESS, 'transport survives a throwing hook');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
    probe.Free;
  end;
end;

type
  /// drives a live subscription stream from outside the blocked POST
  // - the client's POST does not return until the stream ends, so the events
  //   under test have to be produced from another thread
  TSubscriptionDriver = class(TThread)
  public
    Server: TMcpServer;
    Opened: boolean;
    procedure Execute; override;
  end;

procedure TSubscriptionDriver.Execute;
var
  waited: integer;
begin
  // Wait for the stream to actually register, do NOT guess with a sleep: if
  // the POST were slower than a fixed delay, both notifications would be lost
  // and CancelAllSubscriptions would hit nothing — leaving the stream open and
  // the test HANGING instead of failing. Poll the real state with a deadline.
  waited := 0;
  while (Server.SubscriptionCount = 0) and
        (waited < 5000) do
  begin
    SleepHiRes(10);
    inc(waited, 10);
  end;
  Opened := Server.SubscriptionCount > 0;
  if Opened then
  begin
    Server.NotifyToolsListChanged;
    Server.NotifyResourceUpdated('version://info');
  end;
  // Tear down either way: a stream left open would block the test forever.
  // The loop polls Cancelled every 50ms, so this returns promptly.
  Server.CancelAllSubscriptions;
end;

procedure TTestMcpStreamableTransport.SubscriptionStreamDelivers;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  driver: TSubscriptionDriver;
  port, status: integer;
  client: THttpClientSocket;
  request, body: RawUtf8;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  driver := nil;
  try
    // no resource needs to exist: subscribing to a URI is an opt-in filter,
    // not a lookup — the notification is raised through the server API
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    driver := TSubscriptionDriver.Create(true);
    driver.Server := server;
    driver.FreeOnTerminate := false;
    driver.Start;

    // subscriptions/listen replaces the removed GET stream and
    // resources/subscribe: one long-lived POST response carrying every
    // notification the client opted into.
    request := '{"jsonrpc":"2.0","id":7,"method":"subscriptions/listen",' +
      '"params":{"notifications":{"toolsListChanged":true,' +
      '"resourceSubscriptions":["version://info"]}}}';
    status := McpPost(client, transport.Endpoint, request);
    CheckEqual(status, HTTP_SUCCESS, 'listen stream completes');
    Check(driver.Opened, 'the subscription registered before the driver fired');
    body := client.Content;

    // 1. the acknowledgement MUST come first, before any notification
    Check(PosEx('notifications/subscriptions/acknowledged', body) > 0,
      'stream is acknowledged');
    Check(PosEx('notifications/subscriptions/acknowledged', body) <
          PosEx('notifications/tools/list_changed', body),
      'the acknowledgement precedes every notification');

    // 2. both opted-in notification types arrive, tagged with the
    // subscription id (= the id of the listen request), which is how a stdio
    // client demultiplexes concurrent streams
    Check(PosEx('notifications/tools/list_changed', body) > 0,
      'tools/list_changed delivered');
    Check(PosEx('notifications/resources/updated', body) > 0,
      'resources/updated delivered for the watched URI');
    Check(PosEx(MCP_META_SUBSCRIPTION_ID, body) > 0,
      'messages carry the subscription id');

    // 3. a type the client did NOT request must never appear
    Check(PosEx('notifications/resources/list_changed', body) = 0,
      'the server must not send an unrequested notification type');

    // 4. server-side teardown: "A server MUST send notifications/cancelled
    // referencing a subscriptions/listen request ID when it tears down that
    // subscription stream" — followed by the empty response to the listen
    // request, so the client can tell this from a dropped connection.
    Check(PosEx('notifications/cancelled', body) > 0,
      'a server-side teardown announces itself as a cancellation');
    Check(PosEx('"requestId":7', body) > 0,
      'and references the subscriptions/listen request it tears down');
    Check(PosEx('shutting down', body) > 0,
      'the reason tells the client whether reconnecting makes sense');
    Check(PosEx('"id":7', body) > 0, 'graceful closure response, correlated');
    Check(PosEx('notifications/cancelled', body) < PosEx('"id":7', body),
      'the cancellation precedes the response that closes the request');
  finally
    if driver <> nil then
    begin
      driver.WaitFor;
      driver.Free;
    end;
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

procedure TTestMcpStreamableTransport.ConcurrentPosts;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port, i, status: integer;
  clients: array[0..3] of THttpClientSocket;
  request: RawUtf8;
begin
  // The session registry that used to serialize concurrent access is gone, and
  // THttpAsyncServer runs handlers on several worker threads. Nothing else in
  // this suite exercises more than one connection at a time, so a regression in
  // the shared tool/resource registry (or in the deferred streaming path) would
  // go unnoticed until production. Keep-alive reuse is covered too: each client
  // issues two requests on the same socket.
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  FillCharFast(clients, SizeOf(clients), 0);
  try
    server.RegisterTool(TCalcTool.Create('calc', 'Add two numbers'));
    server.Start;
    port := StartStreamableTransport(server, transport);
    for i := 0 to high(clients) do
      clients[i] := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));
    request := '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":' +
      '{"name":"calc","arguments":{"a":2,"b":3,"enabled":true,"name":"x"}}}';
    for i := 0 to high(clients) do
    begin
      status := McpPost(clients[i], transport.Endpoint, request);
      CheckEqual(status, HTTP_SUCCESS, 'concurrent call');
      Check(PosEx('2 + 3 = 5', clients[i].Content) > 0, 'concurrent result');
    end;
    // second round on the SAME sockets: the chunked stream must have terminated
    // cleanly enough for the connection to be reusable
    for i := 0 to high(clients) do
    begin
      status := McpPost(clients[i], transport.Endpoint,
        '{"jsonrpc":"2.0","id":2,"method":"tools/list"}');
      CheckEqual(status, HTTP_SUCCESS, 'keep-alive reuse after a stream');
      Check(PosEx('"calc"', clients[i].Content) > 0, 'reused connection result');
    end;
  finally
    for i := 0 to high(clients) do
      clients[i].Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

type
  /// a tool that needs an input type the test client will not declare
  // - implements IMcpTool directly rather than deriving from TMcpToolBase<T>:
  //   this test is about the transport's status code, not about RTTI schemas
  TCapabilityHungryTool = class(TInterfacedObject, IMcpTool, IMcpInteractiveTool)
  public
    function GetName: RawUtf8;
    function GetDescription: RawUtf8;
    function GetInputSchema: variant;
    function Execute(const Args: variant;
      const AuthCtx: TMcpAuthContext): variant;
    function ExecuteInteractive(const Args: variant;
      const Context: TMcpCallContext): variant;
  end;

function TCapabilityHungryTool.GetName: RawUtf8;
begin
  result := 'needs_elicitation';
end;

function TCapabilityHungryTool.GetDescription: RawUtf8;
begin
  result := 'Always asks the client for input';
end;

function TCapabilityHungryTool.GetInputSchema: variant;
begin
  result := _ObjFast(['type', 'object', 'additionalProperties', false]);
end;

function TCapabilityHungryTool.Execute(const Args: variant;
  const AuthCtx: TMcpAuthContext): variant;
begin
  result := Null; // never called: the server prefers ExecuteInteractive
end;

function TCapabilityHungryTool.ExecuteInteractive(const Args: variant;
  const Context: TMcpCallContext): variant;
begin
  result := Null;
  raise EMcpInputRequired.Create(
    _ObjFast(['who', McpInputRequest(MCP_INPUT_ELICITATION,
      _ObjFast(['mode', 'form', 'message', 'Who is asking?']))]),
    'state');
end;

procedure TTestMcpStreamableTransport.CapabilityErrorUsesHttpStatus;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port, status: integer;
  client: THttpClientSocket;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.RegisterTool(TCapabilityHungryTool.Create);
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // "If processing a request requires a capability the client did not include
    // in io.modelcontextprotocol/clientCapabilities, the server MUST return a
    // MissingRequiredClientCapabilityError (-32021) ... On HTTP, the response
    // status MUST be 400 Bad Request."
    // This is the one required status that CANNOT be decided by the preflight:
    // whether a capability is needed only emerges once the handler runs. So the
    // transport must run the request BEFORE committing to a status — writing
    // the SSE head first would pin every such answer at 200.
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":' +
      '{"name":"needs_elicitation","arguments":{}}}');
    CheckEqual(status, HTTP_BADREQUEST, '-32021 -> 400, decided after dispatch');
    CheckErrorCode(client.Content, MCP_ERROR_MISSING_CLIENT_CAPABILITY);
    Check(PosEx('text/event-stream', client.ContentType) = 0,
      'and it is buffered JSON, not a stream a client would read as success');
    Check(PosEx('requiredCapabilities', client.Content) > 0,
      'the body names the capability the client has to add');

    // an ordinary request still streams, and still answers 200
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":2,"method":"tools/list"}');
    CheckEqual(status, HTTP_SUCCESS, 'a normal request is unaffected');
    Check(PosEx('text/event-stream', client.ContentType) > 0,
      'and is still streamed');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;

type
  /// accepts exactly one token, and only for one audience
  TTransportVerifier = class(TInterfacedObject, IMcpTokenVerifier)
  public
    function VerifyToken(const aToken, aResource: RawUtf8;
      out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
  end;

function TTransportVerifier.VerifyToken(const aToken, aResource: RawUtf8;
  out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
begin
  if aToken = 'expired-token' then
    exit(mtrExpired);
  if aToken = 'other-audience' then
    exit(mtrWrongAudience);
  if (aToken <> 'good-token') and
     (aToken <> 'narrow-token') then
    exit(mtrInvalid);
  aAuthCtx.IsAuthenticated := true;
  aAuthCtx.UserID := 'user-1';
  aAuthCtx.Issuer := 'https://as.example.com';
  if aToken = 'good-token' then
    AddRawUtf8(aAuthCtx.Scopes, 'files');
  result := mtrValid;
end;

type
  /// echoes back who the server thinks is calling
  // - the whole point of the verifier naht: if the transport drops the context,
  //   this tool reports an unauthenticated caller for a perfectly valid token
  TIdentityEchoTool = class(TInterfacedObject, IMcpTool, IMcpInteractiveTool)
  public
    function GetName: RawUtf8;
    function GetDescription: RawUtf8;
    function GetInputSchema: variant;
    function Execute(const Args: variant;
      const AuthCtx: TMcpAuthContext): variant;
    function ExecuteInteractive(const Args: variant;
      const Context: TMcpCallContext): variant;
  end;

function TIdentityEchoTool.GetName: RawUtf8;
begin
  result := 'whoami';
end;

function TIdentityEchoTool.GetDescription: RawUtf8;
begin
  result := 'Reports the authenticated caller';
end;

function TIdentityEchoTool.GetInputSchema: variant;
begin
  result := _ObjFast(['type', 'object', 'additionalProperties', false]);
end;

function TIdentityEchoTool.Execute(const Args: variant;
  const AuthCtx: TMcpAuthContext): variant;
begin
  result := Null; // the server prefers ExecuteInteractive
end;

function TIdentityEchoTool.ExecuteInteractive(const Args: variant;
  const Context: TMcpCallContext): variant;
begin
  // a per-operation scope requirement — the only way such a decision can reach
  // the transport, since the token was checked before the method was known
  if not McpScopeSatisfied(Context.Auth.Scopes, 'files:write') then
    raise EMcpInsufficientScope.CreateScope('files:write');
  result := _ObjFast(['content', _ArrFast([_ObjFast([
    'type', 'text',
    'text', 'authenticated=' + BOOL_UTF8[Context.Auth.IsAuthenticated] +
      ' user=' + Context.Auth.UserID +
      ' issuer=' + Context.Auth.Issuer])])]);
end;

procedure TTestMcpStreamableTransport.BearerAuthGuardsTheEndpoint;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port, status: integer;
  client: THttpClientSocket;
  scopes, authServers: TRawUtf8DynArray;

  // POST a tools/list with an explicit Authorization header
  // - hand-built rather than routed through McpPost, because the token has to
  //   ride along and the point of the test is what the transport does with it
  function PostAs(const aToken: RawUtf8; aId: integer;
    const aTool: RawUtf8 = ''): integer;
  var
    hdr, body, method: RawUtf8;
    doc: TDocVariantData;
    params: variant;
  begin
    if aTool = '' then
    begin
      method := 'tools/list';
      params := Null;
    end
    else
    begin
      method := 'tools/call';
      params := _ObjFast(['name', aTool, 'arguments', _ObjFast([])]);
    end;
    doc.InitJson('{"jsonrpc":"2.0"}', JSON_FAST);
    doc.AddOrUpdateValue('id', aId);
    doc.AddOrUpdateValue('method', method);
    doc.AddOrUpdateValue('params', McpRequestParams(params, 'mcp.tests', '1.0'));
    body := doc.ToJson;
    hdr := 'Accept: application/json, text/event-stream'#13#10 +
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: ' + method;
    if aTool <> '' then
      hdr := hdr + #13#10 + 'Mcp-Name: ' + aTool;
    if aToken <> '' then
      hdr := hdr + #13#10 + 'Authorization: Bearer ' + aToken;
    result := client.Post(transport.Endpoint, body, JSON_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS, hdr);
  end;

begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.AuthResource := 'https://mcp.example.com/mcp';
    AddRawUtf8(scopes, 'mcp:use');
    server.ScopesSupported := scopes;
    AddRawUtf8(authServers, 'https://as.example.com');
    server.AuthorizationServers := authServers;
    server.TokenVerifier := TTransportVerifier.Create;
    server.RegisterTool(TIdentityEchoTool.Create);
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // --- no token: 401 with a challenge that says how to come back ---------
    status := PostAs('', 1);
    CheckEqual(status, HTTP_UNAUTHORIZED, 'an unauthenticated request -> 401');
    Check(PosEx('Bearer', client.Headers) > 0,
      'the refusal carries a WWW-Authenticate challenge');
    Check(PosEx(MCP_WELL_KNOWN_RESOURCE, client.Headers) > 0,
      'pointing at the metadata that names the authorization server');
    Check(PosEx('tools', client.Content) = 0,
      'and no data whatsoever reaches an unauthorized caller');

    // --- a token for a DIFFERENT resource is not merely too weak -----------
    // "MCP servers MUST only accept tokens specifically intended for
    // themselves": this is what stops a token stolen from another service.
    status := PostAs('other-audience', 2);
    CheckEqual(status, HTTP_UNAUTHORIZED, 'a token for another audience -> 401');
    status := PostAs('expired-token', 3);
    CheckEqual(status, HTTP_UNAUTHORIZED, 'an expired token -> 401');
    status := PostAs('nonsense', 4);
    CheckEqual(status, HTTP_UNAUTHORIZED, 'an unknown token -> 401');

    // --- the good token gets through, and still streams as before ---------
    // this also proves the transport reads the header at all: mORMot parses it
    // into Ctxt.AuthBearer, and if that were empty every request would 401
    status := PostAs('good-token', 5);
    CheckEqual(status, HTTP_SUCCESS, 'a valid token is served');
    Check(PosEx('text/event-stream', client.ContentType) > 0,
      'and the response streams exactly as on an open server');

    // --- the verified identity REACHES the handler -------------------------
    // Without this the whole verifier naht is decorative: the transport would
    // check the token and then hand the tool an unauthenticated context.
    status := PostAs('good-token', 6, 'whoami');
    CheckEqual(status, HTTP_SUCCESS, 'the scoped call is served');
    Check(PosEx('authenticated=true', client.Content) > 0,
      'the tool sees the caller as authenticated, not fail-closed');
    Check(PosEx('user=user-1', client.Content) > 0,
      'and receives the principal the verifier resolved');
    Check(PosEx('issuer=https://as.example.com', client.Content) > 0,
      'and the issuer that minted the token');

    // --- a token that lacks the scope gets 403 plus what to ask for --------
    // "the server SHOULD respond with HTTP 403 Forbidden ... scope=... the
    // minimum scopes needed for the operation"
    status := PostAs('narrow-token', 7, 'whoami');
    CheckEqual(status, HTTP_FORBIDDEN,
      'a valid token lacking the scope -> 403, not a 200 with an error body');
    Check(PosEx('insufficient_scope', client.Headers) > 0,
      'the challenge says why');
    Check(PosEx('files:write', client.Headers) > 0,
      'and names the scope to step up to — without it the client can only ' +
      'retry the same failing request');

    // --- the discovery document is public: it is read WITHOUT a token ------
    // both forms are routed, and the spec has clients probe the path form
    // first: for a resource at /mcp the document lives at the well-known path
    // with /mcp appended, NOT at the resource path with /.well-known appended
    status := client.Get(McpResourceMetadataPath(server.AuthResource));
    CheckEqual(status, HTTP_SUCCESS, 'the path form is served');
    Check(PosEx('authorization_servers', client.Content) > 0,
      'and names where to authenticate');
    status := client.Get(MCP_WELL_KNOWN_RESOURCE);
    CheckEqual(status, HTTP_SUCCESS, 'the metadata is served unauthenticated');
    Check(PosEx('https://mcp.example.com/mcp', client.Content) > 0,
      'and names this resource, so a client can request a token for it');
    Check(PosEx('bearer_methods_supported', client.Content) > 0,
      'and how to present it');
  finally
    client.Free;
    if transport <> nil then
      transport.Stop;
    transport.Free;
    server.Free;
  end;
end;


type
  /// records what reached a handler - the only way to observe a notification,
  /// which is answered with 202 and no body whatsoever
  TAuthRecordingTool = class(TInterfacedObject, IMcpTool)
  public
    function GetName: RawUtf8;
    function GetDescription: RawUtf8;
    function GetInputSchema: variant;
    function Execute(const Args: variant;
      const AuthCtx: TMcpAuthContext): variant;
  end;

var
  RecordedAuthenticated: boolean;
  RecordedUserId: RawUtf8;

function TAuthRecordingTool.GetName: RawUtf8;
begin
  result := 'record_identity';
end;

function TAuthRecordingTool.GetDescription: RawUtf8;
begin
  result := 'Records the caller it was handed';
end;

function TAuthRecordingTool.GetInputSchema: variant;
begin
  result := _ObjFast(['type', 'object', 'properties', _ObjFast([])]);
end;

function TAuthRecordingTool.Execute(const Args: variant;
  const AuthCtx: TMcpAuthContext): variant;
begin
  RecordedAuthenticated := AuthCtx.IsAuthenticated;
  RecordedUserId := AuthCtx.UserID;
  result := _ObjFast(['content', _ArrFast([])]);
end;

procedure TTestMcpStreamableTransport.HookAndNotificationSeeTheCaller;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  probe: TStreamHookProbe;
  port, status: integer;
  client: THttpClientSocket;
  scopes, authServers: TRawUtf8DynArray;

  // POST as an authenticated caller; aId <= 0 sends a notification (no id)
  function PostAs(const aMethod, aTool: RawUtf8; aId: integer): integer;
  var
    hdr, body: RawUtf8;
    doc: TDocVariantData;
    params: variant;
  begin
    if aTool = '' then
      params := Null
    else
      params := _ObjFast(['name', aTool, 'arguments', _ObjFast([])]);
    doc.InitJson('{"jsonrpc":"2.0"}', JSON_FAST);
    if aId > 0 then
      doc.AddOrUpdateValue('id', aId);
    doc.AddOrUpdateValue('method', aMethod);
    doc.AddOrUpdateValue('params', McpRequestParams(params, 'mcp.tests', '1.0'));
    body := doc.ToJson;
    hdr := 'Accept: application/json, text/event-stream'#13#10 +
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: ' + aMethod + #13#10 +
      'Authorization: Bearer good-token';
    if aTool <> '' then
      hdr := hdr + #13#10 + 'Mcp-Name: ' + aTool;
    result := client.Post(transport.Endpoint, body, JSON_CONTENT_TYPE,
      HTTP_KEEPALIVE_MS, hdr);
  end;

begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  probe := TStreamHookProbe.Create;
  try
    server.AuthResource := 'https://mcp.example.com/mcp';
    AddRawUtf8(scopes, 'files');
    server.ScopesSupported := scopes;
    AddRawUtf8(authServers, 'https://as.example.com');
    server.AuthorizationServers := authServers;
    server.TokenVerifier := TTransportVerifier.Create;
    server.RegisterTool(TAuthRecordingTool.Create);
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // --- the hook replaces the authorized dispatch, so it MUST be handed the
    // caller: without it a hook can only fail closed or wave everyone through,
    // and there is no other route to the identity (a 'current caller' kept on
    // the transport would be racy across the worker pool).
    transport.OnStreamCall := probe.Bare;
    probe.Called := false;
    probe.SeenAuthenticated := false;
    probe.SeenUserId := '';
    status := PostAs('tools/list', '', 1);
    CheckEqual(status, HTTP_SUCCESS, 'an authenticated request reaches the hook');
    Check(probe.Called, 'the hook ran');
    Check(probe.SeenAuthenticated, 'the hook sees an authenticated caller');
    CheckEqual(probe.SeenUserId, 'user-1', 'and which caller it is');

    // --- the notification branch: it resolved the token and then dispatched
    // through the context-less overload, so a handler saw an anonymous caller
    // on this one path while every sibling path passed the identity down.
    transport.OnStreamCall := nil;
    RecordedAuthenticated := false;
    RecordedUserId := '';
    status := PostAs('tools/call', 'record_identity', 0);
    CheckEqual(status, HTTP_ACCEPTED, 'a notification is accepted with 202');
    Check(RecordedAuthenticated,
      'a notification handler sees the authenticated caller');
    CheckEqual(RecordedUserId, 'user-1', 'and which caller it is');
  finally
    if transport <> nil then
      transport.Stop;
    transport.Free;
    client.Free;
    server.Free;
    probe.Free;
  end;
end;


type
  /// valid on the first call, expired on every one after it: the streamable
  /// transport asks twice per request (preflight, then the deferred hand-off)
  TLapsingVerifier = class(TInterfacedObject, IMcpTokenVerifier)
  public
    Calls: integer;
    function VerifyToken(const aToken, aResource: RawUtf8;
      out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
  end;

function TLapsingVerifier.VerifyToken(const aToken, aResource: RawUtf8;
  out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
begin
  inc(Calls);
  if Calls > 1 then
    // the token lapsed in between - expiry, revocation, or a verifier that
    // reaches a backend and got a different answer this time
    exit(mtrExpired);
  aAuthCtx.IsAuthenticated := true;
  aAuthCtx.UserID := 'user-1';
  aAuthCtx.Issuer := 'https://as.example.com';
  result := mtrValid;
end;

procedure TTestMcpStreamableTransport.DeferredHandoffRefusesALapsedToken;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  verifier: TLapsingVerifier;
  port, status: integer;
  client: THttpClientSocket;
  scopes, authServers: TRawUtf8DynArray;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  verifier := TLapsingVerifier.Create;
  try
    server.AuthResource := 'https://mcp.example.com/mcp';
    AddRawUtf8(scopes, 'files');
    server.ScopesSupported := scopes;
    AddRawUtf8(authServers, 'https://as.example.com');
    server.AuthorizationServers := authServers;
    server.TokenVerifier := verifier;
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // A request with an id defers and streams: mcp() checks the token, hands
    // off, and the async connection re-resolves it because the context does not
    // survive the hand-off. The SECOND answer is the one that counts, and its
    // result used to be discarded - the request then ran with the zeroed
    // context AuthorizeToken leaves behind, anonymous on a server that HAS
    // authorization on, and was answered 200 instead of 401.
    status := client.Post(transport.Endpoint,
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":' +
      VariantSaveJson(McpRequestParams(Null, 'mcp.tests', '1.0')) + '}',
      JSON_CONTENT_TYPE, HTTP_KEEPALIVE_MS,
      'Accept: application/json, text/event-stream'#13#10 +
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: tools/list'#13#10 +
      'Authorization: Bearer good-token');
    CheckEqual(verifier.Calls, 2, 'the transport asked twice');
    CheckEqual(status, HTTP_UNAUTHORIZED, 'the lapsed token is refused');
    Check(PosEx('Bearer', client.Headers) > 0,
      'and the refusal says how to come back (RFC 6750)');
    Check(PosEx('tools', client.Content) = 0,
      'no data reaches a caller whose token lapsed');
  finally
    if transport <> nil then
      transport.Stop;
    transport.Free;
    client.Free;
    server.Free;
  end;
end;


type
  /// a verifier whose backend is down - the failure mode an embedder does not
  /// write on purpose but eventually has (database gone, JWKS endpoint timing out)
  TThrowingVerifier = class(TInterfacedObject, IMcpTokenVerifier)
  public
    function VerifyToken(const aToken, aResource: RawUtf8;
      out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
  end;

function TThrowingVerifier.VerifyToken(const aToken, aResource: RawUtf8;
  out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
begin
  result := mtrInvalid; // never reached - keeps the compiler quiet
  // a principal written BEFORE the failure, the way a real verifier would
  aAuthCtx.UserID := 'half-written';
  raise EMcpException.CreateU('SQL logic error near "WHERE": secret-ish detail');
end;

procedure TTestMcpStreamableTransport.ThrowingVerifierRefusesWithoutLeaking;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port, status: integer;
  client: THttpClientSocket;
  scopes, authServers: TRawUtf8DynArray;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.AuthResource := 'https://mcp.example.com/mcp';
    AddRawUtf8(scopes, 'files');
    server.ScopesSupported := scopes;
    AddRawUtf8(authServers, 'https://as.example.com');
    server.AuthorizationServers := authServers;
    server.TokenVerifier := TThrowingVerifier.Create;
    server.Start;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // The verifier is foreign code. Its exception used to travel all the way
    // out: mORMot renders class name and message into the 500 body, so an
    // unauthenticated caller received the backend's error text.
    status := client.Post(transport.Endpoint,
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":' +
      VariantSaveJson(McpRequestParams(Null, 'mcp.tests', '1.0')) + '}',
      JSON_CONTENT_TYPE, HTTP_KEEPALIVE_MS,
      'Accept: application/json, text/event-stream'#13#10 +
      'MCP-Protocol-Version: ' + MCP_PROTOCOL_VERSION + #13#10 +
      'Mcp-Method: tools/list'#13#10 +
      'Authorization: Bearer any-token');
    CheckEqual(status, HTTP_UNAUTHORIZED,
      'a verifier that cannot answer has authorized nobody');
    Check(PosEx('SQL logic error', client.Content) = 0,
      'the backend error text does not reach the caller');
    Check(PosEx('EMcpException', client.Content) = 0,
      'nor the exception class name');
    Check(PosEx('half-written', client.Content) = 0,
      'nor anything the verifier had already written into the context');
    Check(PosEx('tools', client.Content) = 0, 'and no data either');
  finally
    if transport <> nil then
      transport.Stop;
    transport.Free;
    client.Free;
    server.Free;
  end;
end;


procedure TTestMcpStreamableTransport.SubscriptionStreamHasAMaximumLifetime;
var
  server: TMcpServer;
  transport: TMcpStreamableHttpTransport;
  port, status: integer;
  client: THttpClientSocket;
  body: RawUtf8;
begin
  server := TMcpServer.Create('StreamableTestServer', '1.0');
  transport := nil;
  client := nil;
  try
    server.Start;
    // The token-expiry check only bites when a verifier reports an expiry, and
    // an open server has no token at all - so without this bound a listen
    // stream holds one of the eight slots (and its HTTP worker) forever, and
    // the caller need not even authenticate to do it. One second here; the
    // default is an hour.
    server.MaxSubscriptionSeconds := 1;
    port := StartStreamableTransport(server, transport);
    client := THttpClientSocket.Open('127.0.0.1', UInt32ToUtf8(port));

    // nothing ever fires on this stream: it has to end itself
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":7,"method":"subscriptions/listen",' +
      '"params":{"notifications":{"toolsListChanged":true}}}');
    CheckEqual(status, HTTP_SUCCESS, 'the stream completed on its own');
    body := client.Content;
    Check(PosEx('notifications/cancelled', body) > 0,
      'the teardown is announced, as it is for every other end of a stream');
    Check(PosEx('maximum lifetime', body) > 0,
      'and says why, so a client knows to reconnect rather than give up');
    CheckEqual(server.SubscriptionCount, 0, 'the slot came back');
  finally
    if transport <> nil then
      transport.Stop;
    transport.Free;
    client.Free;
    server.Free;
  end;
end;

end.
