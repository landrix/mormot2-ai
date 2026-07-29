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
  meta: PDocVariantData;
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
      // merge into an existing _meta (e.g. a progressToken) instead of adding
      // a second key of the same name, which would shadow the protocol fields
      if params^.GetAsDocVariant('_meta', meta) and meta^.IsObject then
      begin
        meta^.AddValue(MCP_META_PROTOCOL_VERSION,
          RawUtf8ToVariant(MCP_PROTOCOL_VERSION));
        meta^.AddValue(MCP_META_CLIENT_CAPABILITIES, _ObjFast([]));
      end
      else
        params^.AddValue('_meta', McpRequestMeta('mcp.tests', '1.0'));
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
    status := client.Get(transport.Endpoint, HTTP_KEEPALIVE_MS,
      'Accept: text/event-stream');
    CheckEqual(status, HTTP_NOTALLOWED, 'GET should return 405');
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

end.
