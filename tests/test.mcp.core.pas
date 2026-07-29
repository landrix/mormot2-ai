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

  /// a tool returning a JSON ARRAY instead of the required result object
  // - the protocol has no non-object result; this must surface as an error
  //   rather than silently shipping an empty success (see ResultMustBeAnObject)
  TArrayResultTool = class(TMcpToolBase<TCalcParams>)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  /// a tool doing a Multi Round-Trip Request: it asks for a name on the first
  /// round and completes once the client hands one back
  TElicitingTool = class(TMcpToolBase<TCalcParams>, IMcpInteractiveTool)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  public
    /// what the tool asks for; the tests vary it to exercise the server gate
    InputMethod: RawUtf8;
    /// set when the second round really saw the client's answers
    SawResponses: RawUtf8;
    function ExecuteInteractive(const Args: variant;
      const Context: TMcpCallContext): variant;
  end;

  /// a resource that always needs input — resources/read is the second method
  /// the spec allows an InputRequiredResult on
  TGatedResource = class(TMcpResourceBase, IMcpInteractiveResource)
  protected
    function GetContent: RawUtf8; override;
  public
    function ReadInteractive(const Context: TMcpCallContext): RawUtf8;
  end;

  TTestMcpCore = class(TSynTestCase)
  protected
    procedure EnsureCalcParamsRtti;
    function VariantToInt64Loose(const V: variant; out aValue: Int64): boolean;
    procedure CheckErrorResponse(const aResponse: RawUtf8;
      aExpectedCode: integer; const aMessageContains: RawUtf8);
    function DocPropType(const props: PDocVariantData;
      const propName: RawUtf8): RawUtf8;
    /// run a request, adding the _meta fields every request must carry
    // - since 2026-07-28 each request declares its protocol version and client
    //   capabilities; adding that here keeps the test literals about the RPC
    //   under test instead of repeating protocol boilerplate everywhere
    // - a payload that is not a JSON object, or a notification (no id), is
    //   passed through untouched so the negative tests still exercise the
    //   parser and the notification path
    function Exec(aServer: TMcpServer; const aJson: RawUtf8): RawUtf8;
    /// like Exec, but with client capabilities the caller chooses
    // - Exec goes through McpRequestParams, which declares NO capabilities; the
    //   MRTR gate is precisely about what the client did or did not declare
    function ExecCaps(aServer: TMcpServer; const aJson: RawUtf8;
      const aCapabilities: variant): RawUtf8;
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
    procedure DiscoverReportsVersionAndIdentity;
    procedure RequestMetaIsMandatory;
    procedure ResultsCarryResultTypeAndServerInfo;
    procedure RequestParamsMergeExistingMeta;
    procedure ResultMustBeAnObject;
    procedure PreflightDecidesHttpStatus;
    procedure CacheableResultsCarryHints;
    procedure SubscriptionFilterAndFanout;
    procedure InputRequiredRoundTrip;
    procedure InputRequiredNeedsClientCapability;
    procedure InputRequiredRejectsMalformedResults;
    procedure MrtrRetryFieldsAreTypedAndCountAsPresent;
    procedure RequestStateCodecBindsAndExpires;
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

{ TArrayResultTool }

function TArrayResultTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  result := _ArrFast(['not', 'an', 'object']);
end;

{ TElicitingTool }

function TElicitingTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  // never reached: the server prefers ExecuteInteractive on this tool
  raise Exception.Create('ExecuteTyped must not be called on an interactive tool');
end;

function TElicitingTool.ExecuteInteractive(const Args: variant;
  const Context: TMcpCallContext): variant;
var
  answers, who, content: PDocVariantData;
  name: RawUtf8;
  builder: TMcpResponseBuilder;
begin
  // the value under our own key is an ElicitResult: {action, content{…}}
  answers := _Safe(Context.InputResponses);
  if not answers^.GetAsDocVariant('who', who) or
     not who^.GetAsDocVariant('content', content) or
     not content^.GetAsRawUtf8('name', name) then
  begin
    // first round: nothing to work with yet. The state carries what we already
    // know, so this handler keeps nothing server-side between the rounds.
    SawResponses := '';
    raise EMcpInputRequired.Create(
      _ObjFast(['who', McpInputRequest(InputMethod, _ObjFast([
        'mode', 'form',
        'message', 'Who is asking?']))]),
      'round-1-state');
  end;
  SawResponses := name;
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText('hello ' + name + ' (' + Context.RequestState + ')');
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

{ TGatedResource }

function TGatedResource.GetContent: RawUtf8;
begin
  result := '{"gated":true}';
end;

function TGatedResource.ReadInteractive(const Context: TMcpCallContext): RawUtf8;
begin
  if Context.RequestState = '' then
    // requestState only, no inputRequests: the spec allows that shape, and it
    // means "retry immediately, carrying this back"
    raise EMcpInputRequired.Create(Null, 'resource-state');
  result := '{"gated":false,"state":"' + Context.RequestState + '"}';
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

    response := Exec(server, 
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

    response := Exec(server, 
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

    response := Exec(server, 
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

    response := Exec(server, 
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
    response := Exec(server, 
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
    response := Exec(server, '{');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');

    response := Exec(server, '{"jsonrpc":"2.0","id":1}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');

    // unknown method -> Method not found (-32601)
    response := Exec(server, '{"jsonrpc":"2.0","id":2,"method":"nope"}');
    CheckErrorResponse(response, JSONRPC_METHOD_NOT_FOUND, 'Method not found');

    // naming something that does not exist is INVALID PARAMS, not an internal
    // error: 2026-07-28 removed the dedicated -32002 "resource not found", so
    // an unknown tool/resource and a missing name are all -32602
    response := Exec(server,
      '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{}}');
    CheckErrorResponse(response, JSONRPC_INVALID_PARAMS, 'Missing tool name');

    response := Exec(server,
      '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"missing","arguments":{}}}');
    CheckErrorResponse(response, JSONRPC_INVALID_PARAMS, 'Tool not found');

    response := Exec(server,
      '{"jsonrpc":"2.0","id":5,"method":"resources/read","params":{}}');
    CheckErrorResponse(response, JSONRPC_INVALID_PARAMS, 'Missing uri');

    response := Exec(server,
      '{"jsonrpc":"2.0","id":6,"method":"resources/read","params":{"uri":"missing://info"}}');
    CheckErrorResponse(response, JSONRPC_INVALID_PARAMS, 'Resource not found');
    // -32603 stays reserved for a genuine server-side failure: a tool that
    // throws still lands there (see ToolExceptionBecomesError)
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
    response := Exec(server, 
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
    response := Exec(server, 
      '{"jsonrpc":"1.0","id":1,"method":"ping","params":{}}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
    // missing jsonrpc field -> Invalid Request
    response := Exec(server, '{"id":2,"method":"ping"}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
    // scalar params (must be object or array) -> Invalid Request
    response := Exec(server, 
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
    response := Exec(server, '{"jsonrpc":"2.0","method":"ping","params":{}}');
    Check(TrimU(response) = '');
  finally
    server.Free;
  end;
end;

function TTestMcpCore.Exec(aServer: TMcpServer; const aJson: RawUtf8): RawUtf8;
var
  doc: TDocVariantData;
  params: PDocVariantData;
begin
  doc.InitJson(aJson, JSON_FAST);
  // malformed payloads and notifications go through untouched
  if not doc.IsObject or
     VarIsVoid(doc.GetValueOrNull('id')) then
    exit(aServer.ExecuteRequest(aJson));
  // params present but not an object = a deliberately malformed envelope:
  // hand it over untouched, that is exactly what such a test asserts
  if doc.GetValueIndex('params') >= 0 then
    if not doc.GetAsDocVariant('params', params) or
       not params^.IsObject then
      exit(aServer.ExecuteRequest(aJson));
  // build params through the SAME production helper a real client uses, so the
  // tests exercise McpRequestParams instead of a second, divergent copy of the
  // _meta rules (McpPost in test.mcp.transports does the same)
  doc.AddOrUpdateValue('params',
    McpRequestParams(doc.GetValueOrNull('params'), 'mcp.tests', '1.0'));
  result := aServer.ExecuteRequest(doc.ToJson);
end;

procedure TTestMcpCore.DiscoverReportsVersionAndIdentity;
var
  server: TMcpServer;
  response: RawUtf8;
  rv, resv: variant;
  rd, versions: PDocVariantData;
  tmp: RawUtf8;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // server/discover replaces `initialize`: it is the one RPC a client may
    // call to learn versions, capabilities and identity up front
    response := Exec(server, '{"jsonrpc":"2.0","id":1,"method":"server/discover"}');
    rv := _JsonFast(response);
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    Check(rd^.GetAsDocVariant('supportedVersions', versions), 'supportedVersions');
    Check(versions^.IsArray, 'supportedVersions is an array');
    CheckEqual(versions^.Count, 1, 'exactly one supported version');
    CheckEqual(VariantToUtf8(versions^.Values[0]), MCP_PROTOCOL_VERSION,
      'reports 2026-07-28');
    Check(rd^.GetValueIndex('capabilities') >= 0, 'capabilities present');
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'resultType present');
    CheckEqual(tmp, MCP_RESULT_COMPLETE, 'resultType complete');
    // the handshake methods of earlier revisions are gone, not merely ignored
    CheckErrorResponse(Exec(server,
      '{"jsonrpc":"2.0","id":2,"method":"initialize","params":{}}'),
      JSONRPC_METHOD_NOT_FOUND, 'initialize');
    CheckErrorResponse(Exec(server, '{"jsonrpc":"2.0","id":3,"method":"ping"}'),
      JSONRPC_METHOD_NOT_FOUND, 'ping');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.RequestMetaIsMandatory;
var
  server: TMcpServer;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // NOTE: deliberately NOT via Exec — these requests must stay unadorned.
    // Without a handshake the per-request fields are the only place the server
    // can learn the version, so their absence is a malformed request (-32602).
    CheckErrorResponse(server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'),
      JSONRPC_INVALID_PARAMS, MCP_META_PROTOCOL_VERSION);
    // protocolVersion present, but clientCapabilities missing
    CheckErrorResponse(server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"' + MCP_PROTOCOL_VERSION + '"}}}'),
      JSONRPC_INVALID_PARAMS, MCP_META_CLIENT_CAPABILITIES);
    // a version we do not speak is rejected with the MCP-specific code
    CheckErrorResponse(server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"2025-11-25",' +
      '"' + MCP_META_CLIENT_CAPABILITIES + '":{}}}}'),
      MCP_ERROR_UNSUPPORTED_PROTOCOL_VERSION, 'Unsupported protocol version');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ResultsCarryResultTypeAndServerInfo;
var
  server: TMcpServer;
  rv, resv, errv: variant;
  rd, meta, si: PDocVariantData;
  tmp: RawUtf8;
begin
  server := TMcpServer.Create('TestServer', '9.9');
  try
    server.Start;
    rv := _JsonFast(Exec(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'));
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    // resultType is REQUIRED on every result, not just on discover
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'resultType present');
    CheckEqual(tmp, MCP_RESULT_COMPLETE);
    // identity now travels per result instead of once in the handshake
    Check(rd^.GetAsDocVariant('_meta', meta), '_meta present');
    Check(meta^.GetAsDocVariant(MCP_META_SERVER_INFO, si), 'serverInfo present');
    Check(si^.GetAsRawUtf8('name', tmp));
    CheckEqual(tmp, 'TestServer');
    Check(si^.GetAsRawUtf8('version', tmp));
    CheckEqual(tmp, '9.9');
    // an error response carries no result, so it must not be stamped
    rv := _JsonFast(Exec(server, '{"jsonrpc":"2.0","id":2,"method":"nope"}'));
    errv := _Safe(rv)^.GetValueOrNull('error');
    Check(_Safe(errv)^.IsObject, 'error object');
    Check(VarIsVoid(_Safe(rv)^.GetValueOrNull('result')), 'no result on error');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.SubscriptionFilterAndFanout;
var
  server: TMcpServer;
  tools, res, none, extra: TMcpSubscription;
  queued: TRawUtf8DynArray;
  ack: RawUtf8;
  doc, params, meta, agreed: PDocVariantData;
  dv: variant;
  i: PtrInt;

  // JSON-RPC method of the single queued notification, '' when none
  function DrainedMethod(aSub: TMcpSubscription): RawUtf8;
  var
    d: variant;
  begin
    result := '';
    if not aSub.Drain(queued) then
      exit;
    CheckEqual(length(queued), 1, 'exactly one notification');
    d := _JsonFast(queued[0]);
    _Safe(d)^.GetAsRawUtf8('method', result);
  end;

begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // Three streams with different filters. The spec is strict: "The server
    // MUST NOT send notification types the client has not explicitly
    // requested" — so this is a whitelist, and `none` must stay empty.
    tools := server.OpenSubscription(1,
      _ObjFast(['notifications', _ObjFast(['toolsListChanged', true])]));
    res := server.OpenSubscription(2, _ObjFast(['notifications', _ObjFast([
      'resourcesListChanged', true,
      'resourceSubscriptions', _ArrFast(['version://info'])])]));
    none := server.OpenSubscription(3, _ObjFast([]));
    Check(tools <> nil, 'first stream opens');
    Check(none <> nil, 'a stream without any filter is legal');

    // acknowledgement: first message, carries the subscription id, and echoes
    // only the subset the server actually honors
    ack := server.SubscriptionAcknowledgement(res);
    dv := _JsonFast(ack);
    doc := _Safe(dv);
    Check(doc^.GetAsRawUtf8('method', ack), 'ack method');
    CheckEqual(ack, 'notifications/subscriptions/acknowledged');
    Check(doc^.GetAsDocVariant('params', params), 'ack params');
    Check(params^.GetAsDocVariant('_meta', meta), 'ack _meta');
    CheckEqual(VariantToUtf8(meta^.GetValueOrNull(MCP_META_SUBSCRIPTION_ID)),
      '2', 'ack carries the listen request id as subscription id');
    Check(params^.GetAsDocVariant('notifications', agreed), 'agreed filter');
    Check(agreed^.GetValueIndex('resourcesListChanged') >= 0, 'echoes what we honor');
    Check(agreed^.GetValueIndex('toolsListChanged') < 0,
      'does not echo a type this stream did not request');
    Check(agreed^.GetValueIndex('promptsListChanged') < 0,
      'never echoes a type the server cannot raise at all');

    // fan-out follows the filters, not the connection
    server.NotifyToolsListChanged;
    CheckEqual(DrainedMethod(tools), 'notifications/tools/list_changed');
    Check(not res.Drain(queued), 'unsubscribed stream gets nothing');
    Check(not none.Drain(queued), 'empty filter gets nothing');

    server.NotifyResourcesListChanged;
    CheckEqual(DrainedMethod(res), 'notifications/resources/list_changed');
    Check(not tools.Drain(queued), 'tools stream unaffected');

    // resource updates are per URI, not per stream
    server.NotifyResourceUpdated('version://info');
    CheckEqual(DrainedMethod(res), 'notifications/resources/updated');
    server.NotifyResourceUpdated('other://thing');
    Check(not res.Drain(queued), 'a URI nobody watches notifies nobody');

    // every message carries the subscription id so a stdio client can
    // demultiplex several streams on one channel
    server.NotifyResourceUpdated('version://info');
    Check(res.Drain(queued), 'queued');
    dv := _JsonFast(queued[0]);
    Check(_Safe(dv)^.GetAsDocVariant('params', params));
    Check(params^.GetAsDocVariant('_meta', meta));
    CheckEqual(VariantToUtf8(meta^.GetValueOrNull(MCP_META_SUBSCRIPTION_ID)), '2');
    CheckEqual(params^.U['uri'], 'version://info', 'the updated URI');

    // the graceful-closure response is the JSON-RPC answer to the long-lived
    // request, so a client can tell an orderly end from a dropped connection
    dv := _JsonFast(server.SubscriptionEndResponse(tools));
    doc := _Safe(dv);
    CheckEqual(VariantToUtf8(doc^.GetValueOrNull('id')), '1', 'correlated by id');
    Check(doc^.GetValueIndex('result') >= 0, 'carries an (empty) result');

    // the cap exists so one client cannot take every HTTP worker thread
    CheckEqual(server.MaxSubscriptions, 8, 'bounded by default');
    for i := 4 to 8 do
      Check(server.OpenSubscription(i, _ObjFast([])) <> nil, 'below the cap');
    extra := server.OpenSubscription(99, _ObjFast([]));
    Check(extra = nil, 'refused past the cap instead of starving the pool');

    // Backpressure: a stream nobody drains must not grow without bound —
    // MaxSubscriptions caps how MANY queues exist, not how large one gets.
    // Past the limit the subscription is dropped, which the owning stream
    // turns into a closed stream; the client reconnects and re-reads state.
    Check(not res.Cancelled, 'still live before the flood');
    for i := 0 to MCP_SUBSCRIPTION_MAX_PENDING + 10 do
      server.NotifyResourceUpdated('version://info');
    Check(res.Cancelled, 'a stream that cannot keep up is dropped');
    Check(not res.Drain(queued), 'and its queue is released, not retained');
    Check(not tools.Cancelled, 'the flood does not affect other streams');

    // a closed stream stops receiving, and closing is safe while others live
    server.CloseSubscription(tools);
    server.NotifyToolsListChanged; // must not touch the freed subscription
    Check(not res.Drain(queued), 'still nothing for the resource stream');
  finally
    server.Free; // frees whatever is still registered
  end;
end;

procedure TTestMcpCore.CacheableResultsCarryHints;
var
  server: TMcpServer;
  rv, resv: variant;
  rd: PDocVariantData;
  tmp: RawUtf8;
  ttl: Int64;

  // read the hints off one result; aTtl < 0 asserts they are ABSENT
  procedure CheckHints(const aRequest: RawUtf8; aTtl: integer;
    const aScope, aWhat: RawUtf8);
  begin
    rv := _JsonFast(Exec(server, aRequest));
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    if CheckFailed(rd^.IsObject, aWhat) then
      exit;
    if aTtl < 0 then
    begin
      Check(rd^.GetValueIndex('ttlMs') < 0, aWhat + ' must carry no ttlMs');
      Check(rd^.GetValueIndex('cacheScope') < 0,
        aWhat + ' must carry no cacheScope');
      exit;
    end;
    Check(VariantToInt64(rd^.GetValueOrDefault('ttlMs', -1), ttl),
      aWhat + ' has ttlMs');
    CheckEqual(integer(ttl), aTtl, aWhat + ' ttlMs');
    Check(rd^.GetAsRawUtf8('cacheScope', tmp), aWhat + ' has cacheScope');
    CheckEqual(tmp, aScope, aWhat + ' cacheScope');
  end;

begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TCalcTool.Create('calc', 'Add two numbers'));
    server.RegisterResource(TVersionResource.Create('version://info', 'Version',
      'Server version information', 'application/json'));
    server.Start;

    // The spec REQUIRES caching hints on exactly these four results (of the
    // methods we implement); a client that gets none must assume ttl 0 anyway,
    // but "MUST include" is not satisfied by omission.
    // Defaults are deliberately conservative: never reuse, never share.
    CheckEqual(server.ListCacheTtlMs, MCP_CACHE_TTL_DEFAULT, 'default list ttl');
    CheckEqual(MCP_CACHE_SCOPE[server.ListCacheScope], MCP_CACHE_SCOPE[mcsPrivate],
      'default scope is private: this server cannot know whether the host '  +
      'filters per caller, and public crosses authorization contexts');
    CheckHints('{"jsonrpc":"2.0","id":1,"method":"server/discover"}',
      0, MCP_CACHE_SCOPE[mcsPrivate], 'server/discover');
    CheckHints('{"jsonrpc":"2.0","id":2,"method":"tools/list"}',
      0, MCP_CACHE_SCOPE[mcsPrivate], 'tools/list');
    CheckHints('{"jsonrpc":"2.0","id":3,"method":"resources/list"}',
      0, MCP_CACHE_SCOPE[mcsPrivate], 'resources/list');
    CheckHints('{"jsonrpc":"2.0","id":4,"method":"resources/read",' +
      '"params":{"uri":"version://info"}}', 0, MCP_CACHE_SCOPE[mcsPrivate],
      'resources/read');

    // tools/call has side effects and is NOT in the cacheable list
    CheckHints('{"jsonrpc":"2.0","id":5,"method":"tools/call","params":' +
      '{"name":"calc","arguments":{"A":1,"B":2}}}', -1, '', 'tools/call');

    // configured values reach the wire, and list vs read are independent
    server.ListCacheTtlMs := 300000;
    server.ReadCacheTtlMs := 60000;
    server.ListCacheScope := mcsPublic;
    server.ReadCacheScope := mcsPublic;
    CheckHints('{"jsonrpc":"2.0","id":6,"method":"tools/list"}',
      300000, MCP_CACHE_SCOPE[mcsPublic], 'configured tools/list');
    CheckHints('{"jsonrpc":"2.0","id":7,"method":"resources/read",' +
      '"params":{"uri":"version://info"}}', 60000, MCP_CACHE_SCOPE[mcsPublic],
      'configured resources/read');

    // The two scopes are independent ON PURPOSE: publishing a static tool list
    // must not drag resource CONTENT into shared proxy caches, which is exactly
    // what the spec calls out as typically per-user. A single knob would force
    // that trade-off on the operator.
    server.ReadCacheScope := mcsPrivate;
    CheckHints('{"jsonrpc":"2.0","id":8,"method":"tools/list"}',
      300000, MCP_CACHE_SCOPE[mcsPublic], 'list stays public');
    CheckHints('{"jsonrpc":"2.0","id":9,"method":"resources/read",' +
      '"params":{"uri":"version://info"}}', 60000, MCP_CACHE_SCOPE[mcsPrivate],
      'read can stay private while the list is public');

    // a negative TTL would be a spec violation on the wire ("servers MUST
    // provide a ttlMs value that is >= 0"), so it is clamped, not forwarded
    server.ListCacheTtlMs := -1;
    CheckHints('{"jsonrpc":"2.0","id":10,"method":"tools/list"}',
      0, MCP_CACHE_SCOPE[mcsPublic], 'negative ttl is clamped');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.RequestParamsMergeExistingMeta;
var
  server: TMcpServer;
  params, wrapped: variant;
  doc, meta: PDocVariantData;
  tmp: RawUtf8;
  n: integer;
  i: PtrInt;
begin
  // A caller that already put its own key into _meta (a progressToken, say)
  // must end up with ONE merged _meta. Appending a second one would be
  // invisible: TDocVariantData stores duplicates but every lookup returns the
  // first, so the server would reject the request for missing protocol fields.
  params := _ObjFast([
    'name', 'calc',
    '_meta', _ObjFast(['progressToken', 'tok-1'])]);
  wrapped := McpRequestParams(params, 'mcp.tests', '1.0');
  doc := _Safe(wrapped);
  n := 0;
  for i := 0 to doc^.Count - 1 do
    if doc^.Names[i] = '_meta' then
      inc(n);
  CheckEqual(n, 1, 'exactly one _meta, never a second appended one');
  Check(doc^.GetAsRawUtf8('name', tmp), 'caller params preserved');
  CheckEqual(tmp, 'calc');
  Check(doc^.GetAsDocVariant('_meta', meta), '_meta is an object');
  Check(meta^.GetAsRawUtf8('progressToken', tmp), 'caller _meta key preserved');
  CheckEqual(tmp, 'tok-1');
  Check(meta^.GetAsRawUtf8(MCP_META_PROTOCOL_VERSION, tmp), 'version merged in');
  CheckEqual(tmp, MCP_PROTOCOL_VERSION);
  Check(meta^.GetValueIndex(MCP_META_CLIENT_CAPABILITIES) >= 0,
    'capabilities merged in');

  // and the server actually accepts what the helper produced
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TCalcTool.Create('calc', 'Add two numbers'));
    server.Start;
    EnsureCalcParamsRtti;
    wrapped := _ObjFast(['jsonrpc', '2.0', 'id', 1, 'method', 'tools/call',
      'params', McpRequestParams(_ObjFast([
        'name', 'calc',
        'arguments', _ObjFast(['A', 2, 'B', 3]),
        '_meta', _ObjFast(['progressToken', 'tok-1'])]))]);
    tmp := server.ExecuteRequest(_Safe(wrapped)^.ToJson);
    Check(PosEx('"error"', tmp) = 0, 'a merged _meta passes validation');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ResultMustBeAnObject;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TArrayResultTool.Create('arr', 'Returns an array'));
    server.Start;
    // Copying nothing (the old behaviour) turned a broken handler into an empty
    // success response; the caller then saw resultType:complete and no content.
    response := Exec(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"arr"}}');
    CheckErrorResponse(response, JSONRPC_INTERNAL_ERROR, 'must be a JSON object');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.PreflightDecidesHttpStatus;
var
  server: TMcpServer;
  errorJson: RawUtf8;
  status: integer;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // Preflight is what lets a Streamable HTTP transport pick the status BEFORE
    // it opens a stream — once the SSE head is written the answer is 200 by
    // construction, so every non-200 outcome has to be decided here.

    // valid request -> dispatchable, 200
    Check(server.PreflightRequest(
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"' + MCP_PROTOCOL_VERSION + '",' +
      '"' + MCP_META_CLIENT_CAPABILITIES + '":{}}}}', errorJson, status),
      'valid request passes');
    CheckEqual(status, HTTP_MCP_SUCCESS);
    CheckEqual(errorJson, '');

    // unimplemented method -> spec REQUIRES 404 (not 400), so a client can tell
    // "this method does not exist" from "this request was refused"
    Check(not server.PreflightRequest(
      '{"jsonrpc":"2.0","id":2,"method":"nope","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"' + MCP_PROTOCOL_VERSION + '",' +
      '"' + MCP_META_CLIENT_CAPABILITIES + '":{}}}}', errorJson, status),
      'unknown method rejected');
    CheckEqual(status, HTTP_MCP_NOT_FOUND);
    CheckErrorResponse(errorJson, JSONRPC_METHOD_NOT_FOUND, 'Method not found');

    // missing _meta -> 400
    Check(not server.PreflightRequest(
      '{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}',
      errorJson, status), 'missing _meta rejected');
    CheckEqual(status, HTTP_MCP_BAD_REQUEST);
    CheckErrorResponse(errorJson, JSONRPC_INVALID_PARAMS, '');
    // ...and it still tells a first-contact client which version to use: there
    // is no handshake left, and server/discover needs valid _meta itself
    Check(PosEx(MCP_PROTOCOL_VERSION, errorJson) > 0,
      'the rejection names the supported version');

    // unsupported version -> 400 with the MCP-specific code
    Check(not server.PreflightRequest(
      '{"jsonrpc":"2.0","id":4,"method":"tools/list","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"2025-11-25",' +
      '"' + MCP_META_CLIENT_CAPABILITIES + '":{}}}}', errorJson, status),
      'unsupported version rejected');
    CheckEqual(status, HTTP_MCP_BAD_REQUEST);
    CheckErrorResponse(errorJson, MCP_ERROR_UNSUPPORTED_PROTOCOL_VERSION, '');

    // malformed envelope -> 400
    Check(not server.PreflightRequest('{', errorJson, status),
      'malformed envelope rejected');
    CheckEqual(status, HTTP_MCP_BAD_REQUEST);

    // a notification carries no _meta and is never rejected here
    Check(server.PreflightRequest(
      '{"jsonrpc":"2.0","method":"notifications/cancelled"}', errorJson, status),
      'notification passes preflight');
    CheckEqual(status, HTTP_MCP_SUCCESS);
  finally
    server.Free;
  end;
end;

function TTestMcpCore.ExecCaps(aServer: TMcpServer; const aJson: RawUtf8;
  const aCapabilities: variant): RawUtf8;
var
  doc, params, meta: TDocVariantData;
  src: PDocVariantData;
  i: PtrInt;
begin
  doc.InitJson(aJson, JSON_FAST);
  // Build a FRESH params document instead of writing through the pointer
  // GetAsDocVariant hands back: that pointer aliases doc's own storage, so
  // storing it back under 'params' frees the value while the copy source still
  // points at it. Same rule as McpRequestParams, for the same reason.
  params.InitObject([], JSON_FAST);
  if doc.GetAsDocVariant('params', src) and src^.IsObject then
    for i := 0 to src^.Count - 1 do
      if src^.Names[i] <> '_meta' then
        params.AddValue(src^.Names[i], src^.Values[i]);
  meta.InitObject([
    MCP_META_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION,
    MCP_META_CLIENT_CAPABILITIES, aCapabilities], JSON_FAST);
  params.AddValue('_meta', variant(meta));
  doc.AddOrUpdateValue('params', variant(params));
  result := aServer.ExecuteRequest(doc.ToJson);
end;

procedure TTestMcpCore.InputRequiredRoundTrip;
var
  server: TMcpServer;
  tool: TElicitingTool;
  rv, resv: variant;
  rd, requests, entry: PDocVariantData;
  tmp: RawUtf8;
  caps: variant;
begin
  EnsureCalcParamsRtti;
  caps := _ObjFast(['elicitation', _ObjFast([])]);
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TElicitingTool.Create('greet', 'Greet the caller');
    tool.InputMethod := MCP_INPUT_ELICITATION;
    server.RegisterTool(tool);
    server.RegisterResource(TGatedResource.Create('gate://one', 'Gate',
      'Needs a round trip', 'application/json'));
    // a read that WOULD be cacheable, to prove the retry is not
    server.ReadCacheTtlMs := 60000;
    server.ReadCacheScope := mcsPublic;
    server.Start;

    // --- round 1: the tool has nothing to work with and asks -----------------
    rv := _JsonFast(ExecCaps(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{}}}', caps));
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    if CheckFailed(rd^.IsObject, 'round 1 is a result, not an error') then
      exit;
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'round 1 has resultType');
    CheckEqual(tmp, MCP_RESULT_INPUT_REQUIRED, 'round 1 resultType');
    Check(rd^.GetAsRawUtf8('requestState', tmp), 'round 1 carries requestState');
    CheckEqual(tmp, 'round-1-state', 'requestState reaches the client verbatim');
    if not CheckFailed(rd^.GetAsDocVariant('inputRequests', requests),
         'round 1 carries inputRequests') then
    begin
      CheckEqual(requests^.Count, 1, 'one input request');
      CheckEqual(requests^.Names[0], 'who', 'server-assigned identifier');
      if not CheckFailed(_Safe(requests^.Values[0], entry), 'request object') then
      begin
        Check(entry^.GetAsRawUtf8('method', tmp), 'input request has a method');
        CheckEqual(tmp, MCP_INPUT_ELICITATION, 'input request method');
      end;
    end;
    // An interim result is NOT a completed one, and the caching sentence of the
    // spec is scoped to resultType 'complete'. Hints here would tell a proxy to
    // replay "I need input" — the client would then loop on its own cache.
    Check(rd^.GetValueIndex('ttlMs') < 0, 'no ttlMs on an interim result');
    Check(rd^.GetValueIndex('cacheScope') < 0, 'no cacheScope on an interim result');
    CheckEqual(tool.SawResponses, '', 'round 1 saw no answers');

    // --- round 2: the client answers and echoes the state back ---------------
    // note the different id: "The JSON-RPC id MUST be different between the
    // initial request and the retry, as they are independent requests."
    rv := _JsonFast(ExecCaps(server,
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{},"requestState":"round-1-state",' +
      '"inputResponses":{"who":{"action":"accept","content":{"name":"octocat"}}}}}',
      caps));
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    if CheckFailed(rd^.IsObject, 'round 2 is a result') then
      exit;
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'round 2 has resultType');
    CheckEqual(tmp, MCP_RESULT_COMPLETE, 'round 2 completes');
    CheckEqual(tool.SawResponses, 'octocat',
      'the handler received the client answers, keyed by its own identifier');
    Check(PosEx('round-1-state', VariantSaveJson(resv)) > 0,
      'the handler also received the state it issued');

    // --- resources/read: the requestState-only shape, and its cacheability ---
    rv := _JsonFast(ExecCaps(server,
      '{"jsonrpc":"2.0","id":3,"method":"resources/read",' +
      '"params":{"uri":"gate://one"}}', caps));
    rd := _Safe(_Safe(rv)^.GetValueOrNull('result'));
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'gated read has resultType');
    CheckEqual(tmp, MCP_RESULT_INPUT_REQUIRED, 'gated read asks for a retry');
    Check(rd^.GetValueIndex('inputRequests') < 0,
      'a requestState-only result omits inputRequests rather than sending {}');
    Check(rd^.GetAsRawUtf8('requestState', tmp), 'gated read carries state');

    rv := _JsonFast(ExecCaps(server,
      '{"jsonrpc":"2.0","id":4,"method":"resources/read",' +
      '"params":{"uri":"gate://one","requestState":"resource-state"}}', caps));
    rd := _Safe(_Safe(rv)^.GetValueOrNull('result'));
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'retried read has resultType');
    CheckEqual(tmp, MCP_RESULT_COMPLETE, 'retried read completes');
    // The server is configured for a shareable read cache, yet this particular
    // answer was shaped by one caller's round trip: handing a proxy `public`
    // on it is the cross-authorization-context replay the spec warns about.
    Check(rd^.GetAsRawUtf8('cacheScope', tmp), 'retried read has cacheScope');
    CheckEqual(tmp, MCP_CACHE_SCOPE[mcsPrivate],
      'an MRTR retry is never shareable, whatever the server was configured for');
    CheckEqual(VariantToIntegerDef(rd^.GetValueOrDefault('ttlMs', -1), -1), 0,
      'an MRTR retry is never reusable either');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.InputRequiredNeedsClientCapability;
var
  server: TMcpServer;
  tool: TElicitingTool;
  response: RawUtf8;
  rv: variant;
  data, required: PDocVariantData;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TElicitingTool.Create('greet', 'Greet the caller');
    tool.InputMethod := MCP_INPUT_ELICITATION;
    server.RegisterTool(tool);
    server.Start;

    // "Servers MUST NOT send an inputRequests that the client has not declared
    // support for in its capabilities." The client here declared none at all,
    // so the elicitation request must never reach the wire — -32021 is the code
    // the spec reserved for exactly this.
    response := ExecCaps(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{}}}', _ObjFast([]));
    CheckErrorResponse(response, MCP_ERROR_MISSING_CLIENT_CAPABILITY,
      'elicitation');
    // "the server MUST return a MissingRequiredClientCapabilityError (-32021)
    // whose data.requiredCapabilities lists the missing capabilities" — the
    // free-text message is for humans; only this field lets a client fix itself.
    rv := _JsonFast(response);
    if not CheckFailed(_Safe(_Safe(rv)^.GetValueOrNull('error'))^.
         GetAsDocVariant('data', data), '-32021 carries data') then
      if not CheckFailed(data^.GetAsArray('requiredCapabilities', required),
           'data.requiredCapabilities is present') then
      begin
        CheckEqual(required^.Count, 1, 'one missing capability');
        CheckEqual(VariantToUtf8(required^.Values[0]), 'elicitation',
          'and it names the one the client must add');
      end;
    // the status a transport MUST send for it: "On HTTP, the response status
    // MUST be 400 Bad Request". It cannot be decided before the handler ran,
    // so it is derived from the finished response.
    CheckEqual(McpHttpStatus(response), 400, '-32021 is an HTTP 400');
    CheckEqual(McpHttpStatus('{"jsonrpc":"2.0","id":1,"result":{}}'), 200,
      'an ordinary result is 200');
    CheckEqual(McpHttpStatus(''), 200, 'a notification leaves the status alone');

    // a client declaring a DIFFERENT capability is still missing this one
    CheckErrorResponse(ExecCaps(server,
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{}}}',
      _ObjFast(['sampling', _ObjFast([])])),
      MCP_ERROR_MISSING_CLIENT_CAPABILITY, 'elicitation');

    // the same tool asking for sampling now goes through, since that IS declared
    tool.InputMethod := MCP_INPUT_SAMPLING;
    Check(PosEx('"resultType":"input_required"', ExecCaps(server,
      '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{}}}',
      _ObjFast(['sampling', _ObjFast([])]))) > 0,
      'a declared capability lets the input request through');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.InputRequiredRejectsMalformedResults;
var
  server: TMcpServer;
  tool: TElicitingTool;
  caps: variant;
  raised: boolean;
begin
  EnsureCalcParamsRtti;
  caps := _ObjFast(['elicitation', _ObjFast([]), 'roots', _ObjFast([])]);
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TElicitingTool.Create('greet', 'Greet the caller');
    tool.InputMethod := MCP_INPUT_ELICITATION;
    server.RegisterTool(tool);
    server.Start;

    // A typo in the method name must fail where it is written, not ship a
    // request object no client can dispatch.
    raised := false;
    try
      McpInputRequest('elicitation/created', _ObjFast([]));
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'McpInputRequest rejects a method outside the allowed three');

    // Neither field set: "Servers MUST include at least one of inputRequests or
    // requestState in every InputRequiredResult" — a client receiving neither
    // could only retry the identical request, forever. Our own handler is at
    // fault, so it surfaces as an internal error, not as a malformed result.
    raised := false;
    try
      server.ValidateInputRequests(Null, '', mcpToolsCall, caps);
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'an empty InputRequiredResult is rejected');

    // The spec lists exactly three methods that may answer with an
    // InputRequiredResult; tools/list is not one of them.
    raised := false;
    try
      server.ValidateInputRequests(
        _ObjFast(['who', McpInputRequest(MCP_INPUT_ROOTS, _ObjFast([]))]),
        '', mcpToolsList, caps);
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'tools/list may not answer with an InputRequiredResult');

    // A present-but-wrong-shaped inputRequests must NOT be silently treated as
    // "requestState-only": it would be dropped from the result and the client
    // would retry without the input the handler is waiting for — a round trip
    // that never terminates, instead of a defect anybody notices.
    raised := false;
    try
      server.ValidateInputRequests(_ArrFast(['not', 'a', 'map']),
        'some-state', mcpToolsCall, caps);
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'an inputRequests that is not an object is rejected, even '  +
      'when a requestState would have carried the result on its own');

    // params of a server-to-client request is an object; null would produce a
    // message this server's own parser rejects
    raised := false;
    try
      McpInputRequest(MCP_INPUT_ELICITATION, Null);
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'McpInputRequest rejects non-object params');

    // a well-formed one on an allowed method passes
    server.ValidateInputRequests(
      _ObjFast(['who', McpInputRequest(MCP_INPUT_ROOTS, _ObjFast([]))]),
      '', mcpResourcesRead, caps);
    Check(true, 'a valid InputRequiredResult passes the gate');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.MrtrRetryFieldsAreTypedAndCountAsPresent;
var
  server: TMcpServer;
  rv: variant;
  rd: PDocVariantData;
  tmp: RawUtf8;
  caps: variant;

  // read the cacheScope off a resources/read answer
  function ScopeOf(const aParams: RawUtf8): RawUtf8;
  begin
    rv := _JsonFast(ExecCaps(server, '{"jsonrpc":"2.0","id":1,' +
      '"method":"resources/read","params":' + aParams + '}', caps));
    rd := _Safe(_Safe(rv)^.GetValueOrNull('result'));
    result := '';
    rd^.GetAsRawUtf8('cacheScope', result);
  end;

begin
  caps := _ObjFast([]);
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterResource(TVersionResource.Create('version://info', 'Version',
      'Server version information', 'application/json'));
    server.ReadCacheTtlMs := 60000;
    server.ReadCacheScope := mcsPublic;
    server.Start;

    // baseline: a plain read really is shareable with this configuration
    CheckEqual(ScopeOf('{"uri":"version://info"}'), MCP_CACHE_SCOPE[mcsPublic],
      'a plain read follows the configured cache scope');

    // PRESENCE decides, not content. VarIsVoid() treats an EMPTY object as
    // void, so testing the value would let this exact request through as
    // shareable — and a proxy could then replay one caller's round-trip answer
    // to everyone else.
    CheckEqual(ScopeOf('{"uri":"version://info","inputResponses":{}}'),
      MCP_CACHE_SCOPE[mcsPrivate],
      'an empty inputResponses is still a retry, and still personal');
    CheckEqual(ScopeOf('{"uri":"version://info","requestState":""}'),
      MCP_CACHE_SCOPE[mcsPrivate],
      'an empty requestState is present, not absent');

    // Both fields are typed on the wire. Accepting a number as requestState
    // would stringify it and hand a handler a state this server never issued.
    CheckErrorResponse(ExecCaps(server, '{"jsonrpc":"2.0","id":2,' +
      '"method":"resources/read","params":{"uri":"version://info",' +
      '"requestState":42}}', caps), JSONRPC_INVALID_PARAMS, 'requestState');
    CheckErrorResponse(ExecCaps(server, '{"jsonrpc":"2.0","id":3,' +
      '"method":"resources/read","params":{"uri":"version://info",' +
      '"inputResponses":["nope"]}}', caps),
      JSONRPC_INVALID_PARAMS, 'inputResponses');

    // Retry fields on a method that cannot take part in a round trip are
    // meaningless — they must not quietly degrade that method's cacheability.
    rv := _JsonFast(ExecCaps(server, '{"jsonrpc":"2.0","id":4,' +
      '"method":"tools/list","params":{"requestState":"stray"}}', caps));
    rd := _Safe(_Safe(rv)^.GetValueOrNull('result'));
    Check(rd^.GetAsRawUtf8('cacheScope', tmp), 'tools/list still has a scope');
    CheckEqual(tmp, MCP_CACHE_SCOPE[server.ListCacheScope],
      'a stray requestState on tools/list changes nothing');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.RequestStateCodecBindsAndExpires;
var
  codec, other: TMcpRequestStateCodec;
  blob, tampered: RawUtf8;
  state: variant;
  dot: PtrInt;
  weak: boolean;
begin
  codec := TMcpRequestStateCodec.Create('0123456789abcdef0123456789abcdef');
  try
    blob := codec.Encode(_ObjFast(['step', 1]), 'user-42',
      'tools/call:greet');

    // the happy path: same principal, same request, inside the TTL
    Check(codec.Decode(blob, 'user-42', 'tools/call:greet', state),
      'a freshly issued state verifies');
    CheckEqual(VariantToIntegerDef(_Safe(state)^.GetValueOrDefault('step', 0), 0),
      1, 'the handler gets its own state back');

    // "servers MUST treat requestState as an attacker-controlled input" — the
    // three replay defences the spec asks for, each on its own:
    Check(not codec.Decode(blob, 'user-43', 'tools/call:greet', state),
      'state presented by a different principal is rejected');
    Check(not codec.Decode(blob, 'user-42', 'tools/call:other', state),
      'state presented on a different request is rejected');

    // flipping a single payload byte must break the signature. The payload is
    // the part before the dot; corrupting the last character of it is enough.
    dot := PosExChar('.', blob);
    tampered := copy(blob, 1, dot - 2) + 'X' + copy(blob, dot - 1, maxInt);
    Check(not codec.Decode(tampered, 'user-42', 'tools/call:greet', state),
      'a tampered payload fails verification');
    Check(not codec.Decode(copy(blob, 1, dot - 1), 'user-42',
      'tools/call:greet', state), 'a blob without a signature is rejected');
    Check(not codec.Decode('', 'user-42', 'tools/call:greet', state),
      'an empty blob is rejected');

    // a state signed with someone else's secret is not ours, however
    // well-formed it looks — this is the cross-server forgery case
    other := TMcpRequestStateCodec.Create('fedcba9876543210fedcba9876543210');
    try
      Check(not codec.Decode(other.Encode(_ObjFast(['step', 9]), 'user-42',
        'tools/call:greet'), 'user-42', 'tools/call:greet', state),
        'a state signed with a different secret is rejected');
    finally
      other.Free;
    end;
  finally
    codec.Free;
  end;

  // the deadline is inside the signed payload, so it cannot be pushed out
  codec := TMcpRequestStateCodec.Create('0123456789abcdef0123456789abcdef', 1);
  try
    blob := codec.Encode(_ObjFast(['step', 1]), 'user-42', 'tools/call:greet');
    SleepHiRes(1100);
    Check(not codec.Decode(blob, 'user-42', 'tools/call:greet', state),
      'state presented after its TTL lapsed is rejected');
  finally
    codec.Free;
  end;

  // a secret too short to be unguessable is refused outright: signing with it
  // would look like protection while providing none
  weak := false;
  try
    TMcpRequestStateCodec.Create('short').Free;
  except
    on EMcpException do
      weak := true;
  end;
  Check(weak, 'a weak secret is refused at construction');
end;

end.
