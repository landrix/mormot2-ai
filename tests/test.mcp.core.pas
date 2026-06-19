// - regression tests for mormot.ai.mcp
unit test.mcp.core;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  classes,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.buffers,
  mormot.core.rtti,
  mormot.core.variants,
  mormot.core.json,
  mormot.core.test,
  mormot.ai.mcp;

type
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

  TVersionResource = class(TMcpResourceBase)
  protected
    function GetContent: RawUtf8; override;
  end;

  /// a tool that raises a PLAIN Exception (not ESynException) — used to prove a
  /// tool error is translated into a JSON-RPC error, never escapes the handler
  TThrowingTool = class(TMcpToolBase<TCalcParams>)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  TTestMcpCore = class(TSynTestCase)
  protected
    procedure EnsureCalcParamsRtti;
    function VariantToInt64Loose(const V: variant; out aValue: Int64): boolean;
    procedure CheckErrorResponse(const aResponse: RawUtf8;
      aExpectedCode: integer; const aMessageContains: RawUtf8);
    function DocPropType(const props: PDocVariantData;
      const propName: RawUtf8): RawUtf8;
  published
    procedure SchemaFromRecord;
    procedure JsonRpcProcessor;
    procedure ServerToolsResources;
    procedure ResponseBuilderTextAndFile;
    procedure ServerNotActive;
    procedure BadRequests;
    procedure ToolExceptionBecomesError;
    procedure InvalidJsonRpcEnvelope;
    procedure NotificationsNoResponse;
    procedure InitializeVersionNegotiation;
  end;

implementation

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

{ TVersionResource }

function TVersionResource.GetContent: RawUtf8;
begin
  result := '{"version":"1.0.0","protocol":"MCP"}';
end;

{ TThrowingTool }

function TThrowingTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  // a plain RTL Exception (NOT ESynException): the dispatcher must still catch it
  raise Exception.Create('tool blew up');
end;

{ TTestMcpCore }

procedure TTestMcpCore.EnsureCalcParamsRtti;
begin
  //{$ifndef HASEXTRECORDRTTI}
  if not RecordHasFields(TypeInfo(TCalcParams)) then
    Rtti.RegisterFromText(TypeInfo(TCalcParams),
      'A,B:integer Enabled:boolean Name:RawUtf8');
  //{$endif HASEXTRECORDRTTI}
end;

function TTestMcpCore.VariantToInt64Loose(const V: variant; out aValue: Int64): boolean;
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

procedure TTestMcpCore.CheckErrorResponse(const aResponse: RawUtf8;
  aExpectedCode: integer; const aMessageContains: RawUtf8);
var
  doc, errDoc: PDocVariantData;
  docVar, errVar: variant;
  code: integer;
  code64: Int64;
  okCode: boolean;
  msg: RawUtf8;
begin
  docVar := _JsonFast(aResponse);
  doc := _Safe(docVar);
  errVar := doc^.GetValueOrNull('error');
  errDoc := _Safe(errVar);
  Check(errDoc^.IsObject);
  okCode := VariantToInt64Loose(errDoc^.GetValueOrDefault('code', 0), code64);
  if okCode then
    code := integer(code64)
  else
    code := 0;
  if aExpectedCode <> 0 then
    CheckEqual(code, aExpectedCode)
  else
    Check(code <> 0);
  if aMessageContains <> '' then
  begin
    errDoc^.GetAsRawUtf8('message', msg);
    Check(PosEx(aMessageContains, msg) > 0);
  end;
end;

function TTestMcpCore.DocPropType(const props: PDocVariantData;
  const propName: RawUtf8): RawUtf8;
var
  prop: variant;
  propDoc: PDocVariantData;
begin
  result := '';
  if (props = nil) or not props^.IsObject then
    exit;
  prop := props^.GetValueOrNull(propName);
  propDoc := _Safe(prop);
  if propDoc^.IsObject then
    propDoc^.GetAsRawUtf8('type', result);
end;

procedure TTestMcpCore.SchemaFromRecord;
var
  schema: variant;
  doc, props, req: PDocVariantData;
  propsVar: variant;
  typ: RawUtf8;
begin
  EnsureCalcParamsRtti;
  schema := TMcpSchemaGenerator.GenerateSchema(TypeInfo(TCalcParams));
  doc := _Safe(schema);
  Check(doc^.IsObject);
  Check(doc^.GetAsRawUtf8('type', typ));
  CheckEqual(typ, 'object');

  propsVar := doc^.GetValueOrNull('properties');
  props := _Safe(propsVar);
  Check(props^.IsObject);
  CheckEqual(DocPropType(props, 'a'), 'integer');
  CheckEqual(DocPropType(props, 'b'), 'integer');
  CheckEqual(DocPropType(props, 'enabled'), 'boolean');
  CheckEqual(DocPropType(props, 'name'), 'string');
end;

procedure TTestMcpCore.JsonRpcProcessor;
var
  proc: TMcpJsonRpcProcessor;
  method: RawUtf8;
  params, requestId: variant;
  ok: boolean;
  response: RawUtf8;
  doc, resultDoc: PDocVariantData;
  responseVar, resultVar: variant;
  jsonrpc: RawUtf8;
  v: variant;
  okFlag: boolean;
begin
  proc := TMcpJsonRpcProcessor.Create('TestServer', '1.0');
  try
    ok := proc.ParseRequest('{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}',
      method, params, requestId);
    Check(ok);
    CheckEqual(method, 'ping');
    Check(VariantToIntegerDef(requestId, 0) = 1);

    v := _ObjFast(['ok', true]);
    response := proc.CreateSuccessResponse(requestId, v);
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    Check(doc^.GetAsRawUtf8('jsonrpc', jsonrpc));
    CheckEqual(jsonrpc, '2.0');
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    Check(resultDoc^.IsObject);
    Check(resultDoc^.GetAsBoolean('ok', okFlag));
    Check(okFlag);
  finally
    proc.Free;
  end;
end;

procedure TTestMcpCore.ServerToolsResources;
var
  server: TMcpServer;
  tool: IMcpTool;
  res: IMcpResource;
  response: RawUtf8;
  doc, resultDoc, listDoc, itemDoc, contentDoc: PDocVariantData;
  responseVar, resultVar, listVar, contentVar, itemVar: variant;
  toolName: RawUtf8;
  tmp: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TCalcTool.Create('calc', 'Add two numbers');
    res := TVersionResource.Create('version://info', 'Version',
      'Server version information', 'application/json');
    server.RegisterTool(tool);
    server.RegisterResource(res);
    server.Start;

    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}');
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    listVar := resultDoc^.GetValueOrNull('tools');
    listDoc := _Safe(listVar);
    Check(listDoc^.IsArray);
    Check(listDoc^.Count >= 1);
    itemVar := listDoc^.Values[0];
    itemDoc := _Safe(itemVar);
    Check(itemDoc^.GetAsRawUtf8('name', toolName));
    CheckEqual(toolName, 'calc');

    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"a":5,"b":3,"enabled":true,"name":"x"}}}');
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    contentVar := resultDoc^.GetValueOrNull('content');
    contentDoc := _Safe(contentVar);
    Check(contentDoc^.IsArray);
    Check(contentDoc^.Count = 1);
    itemVar := contentDoc^.Values[0];
    itemDoc := _Safe(itemVar);
    Check(itemDoc^.GetAsRawUtf8('type', tmp));
    CheckEqual(tmp, 'text');
    Check(itemDoc^.GetAsRawUtf8('text', tmp));
    CheckEqual(tmp, '5 + 3 = 8');

    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":3,"method":"resources/list","params":{}}');
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    listVar := resultDoc^.GetValueOrNull('resources');
    listDoc := _Safe(listVar);
    Check(listDoc^.IsArray);
    Check(listDoc^.Count = 1);
    itemVar := listDoc^.Values[0];
    itemDoc := _Safe(itemVar);
    Check(itemDoc^.GetAsRawUtf8('uri', tmp));
    CheckEqual(tmp, 'version://info');

    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"version://info"}}');
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    listVar := resultDoc^.GetValueOrNull('contents');
    listDoc := _Safe(listVar);
    Check(listDoc^.IsArray);
    itemVar := listDoc^.Values[0];
    itemDoc := _Safe(itemVar);
    Check(itemDoc^.GetAsRawUtf8('uri', tmp));
    CheckEqual(tmp, 'version://info');
    Check(itemDoc^.GetAsRawUtf8('text', tmp));
    Check(tmp <> '');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ResponseBuilderTextAndFile;
var
  builder: TMcpResponseBuilder;
  response: variant;
  doc, contentDoc, itemDoc: PDocVariantData;
  contentVar, itemVar: variant;
  tmpFile: TFileName;
  content: RawByteString;
  base64: RawUtf8;
  tmp: RawUtf8;
begin
  tmpFile := TemporaryFileName;
  content := 'mcp-test';
  Check(FileFromString(content, tmpFile));
  try
    builder := TMcpResponseBuilder.Create;
    try
      builder.AddText('hello');
      builder.AddFile(StringToUtf8(tmpFile));
      response := builder.Build;
    finally
      builder.Free;
    end;

    doc := _Safe(response);
    contentVar := doc^.GetValueOrNull('content');
    contentDoc := _Safe(contentVar);
    if CheckFailed(contentDoc^.IsArray, 'content not array') then
      exit;
    if CheckFailed(contentDoc^.Count >= 2, 'content count < 2') then
      exit;

    itemVar := contentDoc^.Values[0];
    if CheckFailed(_Safe(itemVar, itemDoc), 'content[0] not object') then
      exit;
    Check(itemDoc^.GetAsRawUtf8('type', tmp));
    CheckEqual(tmp, 'text');
    Check(itemDoc^.GetAsRawUtf8('text', tmp));
    CheckEqual(tmp, 'hello');

    itemVar := contentDoc^.Values[1];
    if CheckFailed(_Safe(itemVar, itemDoc), 'content[1] not object') then
      exit;
    Check(itemDoc^.GetAsRawUtf8('type', tmp));
    CheckEqual(tmp, 'resource');
    base64 := BinToBase64(content);
    Check(itemDoc^.GetAsRawUtf8('data', tmp));
    CheckEqual(tmp, base64);
    Check(itemDoc^.GetAsRawUtf8('mimeType', tmp));
    Check(tmp <> '');
    Check(itemDoc^.GetAsRawUtf8('fileName', tmp));
    Check(tmp <> '');
  finally
    DeleteFile(tmpFile);
  end;
end;

procedure TTestMcpCore.ServerNotActive;
var
  server: TMcpServer;
  response: RawUtf8;
  doc, errDoc: PDocVariantData;
  docVar, errVar: variant;
  code: integer;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}');
    docVar := _JsonFast(response);
    doc := _Safe(docVar);
    errVar := doc^.GetValueOrNull('error');
    errDoc := _Safe(errVar);
    Check(errDoc^.IsObject);
    code := errDoc^.GetValueOrDefault('code', 0);
    Check(code = JSONRPC_INTERNAL_ERROR);
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.BadRequests;
var
  server: TMcpServer;
  tool: IMcpTool;
  res: IMcpResource;
  response: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TCalcTool.Create('calc', 'Add two numbers');
    res := TVersionResource.Create('version://info', 'Version',
      'Server version information', 'application/json');
    server.RegisterTool(tool);
    server.RegisterResource(res);
    server.Start;

    // invalid JSON / malformed envelope -> Invalid Request (-32600)
    response := server.ExecuteRequest('{');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');

    response := server.ExecuteRequest('{"jsonrpc":"2.0","id":1}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');

    // unknown method -> Method not found (-32601)
    response := server.ExecuteRequest('{"jsonrpc":"2.0","id":2,"method":"nope"}');
    CheckErrorResponse(response, JSONRPC_METHOD_NOT_FOUND, 'Method not found');

    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{}}');
    CheckErrorResponse(response, JSONRPC_INTERNAL_ERROR, 'Missing tool name');

    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"missing","arguments":{}}}');
    CheckErrorResponse(response, JSONRPC_INTERNAL_ERROR, 'Tool not found');

    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":5,"method":"resources/read","params":{}}');
    CheckErrorResponse(response, JSONRPC_INTERNAL_ERROR, 'Missing uri');

    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":6,"method":"resources/read","params":{"uri":"missing://info"}}');
    CheckErrorResponse(response, JSONRPC_INTERNAL_ERROR, 'Resource not found');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ToolExceptionBecomesError;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TThrowingTool.Create('boom', 'Always throws'));
    server.Start;
    // a plain Exception from the tool must be turned into a JSON-RPC internal
    // error (not crash the worker, not escape ExecuteRequest)
    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"boom",' +
      '"arguments":{"a":1,"b":2,"enabled":true,"name":"x"}}}');
    CheckErrorResponse(response, JSONRPC_INTERNAL_ERROR, 'tool blew up');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.InvalidJsonRpcEnvelope;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // wrong jsonrpc version -> Invalid Request
    response := server.ExecuteRequest(
      '{"jsonrpc":"1.0","id":1,"method":"ping","params":{}}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
    // missing jsonrpc field -> Invalid Request
    response := server.ExecuteRequest('{"id":2,"method":"ping"}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
    // scalar params (must be object or array) -> Invalid Request
    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":3,"method":"ping","params":5}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.NotificationsNoResponse;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    response := server.ExecuteRequest('{"jsonrpc":"2.0","method":"ping","params":{}}');
    Check(TrimU(response) = '');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.InitializeVersionNegotiation;
var
  server: TMcpServer;
  response: RawUtf8;

  // sends initialize with the given client protocolVersion and returns the
  // protocolVersion the server negotiated back
  function Negotiated(const aClientVersion: RawUtf8): RawUtf8;
  var
    rv, resv: variant;
    rd: PDocVariantData;
  begin
    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":1,"method":"initialize","params":' +
      '{"protocolVersion":"' + aClientVersion + '","capabilities":{},' +
      '"clientInfo":{"name":"t","version":"1"}}}');
    rv := _JsonFast(response);
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    Check(rd^.GetAsRawUtf8('protocolVersion', result), 'protocolVersion present');
  end;

begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // a supported version is echoed back to the client
    CheckEqual(Negotiated('2025-03-26'), MCP_PROTOCOL_VERSION_20250326,
      'echo supported 2025-03-26');
    CheckEqual(Negotiated(MCP_PROTOCOL_VERSION_LATEST), MCP_PROTOCOL_VERSION_LATEST,
      'echo latest');
    // an unknown version falls back to the server's latest supported version
    CheckEqual(Negotiated('1999-01-01'), MCP_PROTOCOL_VERSION_LATEST,
      'fallback to latest');
  finally
    server.Free;
  end;
end;

end.
