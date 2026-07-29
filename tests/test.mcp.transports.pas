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
    procedure ProtocolErrorsUseHttpStatus;
    procedure ConcurrentPosts;
    procedure StreamHookIsValidatedAndContained;
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
    if PosEx(#10, responseText) > 0 then
      responseText := Copy(responseText, LastDelimiter(#10, responseText) + 1, MaxInt);
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
    /// answers with a bare result — no resultType, no serverInfo
    function Bare(const aRequestJson: RawUtf8; const aEmitter: IMcpStreamEmitter;
      out aResponseJson: RawUtf8): boolean;
    /// raises, the way a buggy or hostile hook would
    function Throws(const aRequestJson: RawUtf8;
      const aEmitter: IMcpStreamEmitter; out aResponseJson: RawUtf8): boolean;
  end;

function TStreamHookProbe.Bare(const aRequestJson: RawUtf8;
  const aEmitter: IMcpStreamEmitter; out aResponseJson: RawUtf8): boolean;
begin
  Called := true;
  aResponseJson := '{"jsonrpc":"2.0","id":1,"result":{"content":"from-hook"}}';
  result := true;
end;

function TStreamHookProbe.Throws(const aRequestJson: RawUtf8;
  const aEmitter: IMcpStreamEmitter; out aResponseJson: RawUtf8): boolean;
begin
  aResponseJson := '';
  result := false; // never reached — keeps the compiler from warning
  raise EMcpException.CreateU('hook blew up');
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

    // 3. A hook that raises must not escape into the connection's OnRead: it
    // has no exception handler and would tear the worker down mid-stream.
    transport.OnStreamCall := probe.Throws;
    status := McpPost(client, transport.Endpoint,
      '{"jsonrpc":"2.0","id":2,"method":"tools/list"}');
    CheckEqual(status, HTTP_SUCCESS, 'a throwing hook still answers');
    CheckErrorCode(SseDataJson(client.Content), JSONRPC_INTERNAL_ERROR);
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

end.
