/// Model Context Protocol (MCP) Server Implementation for mORMot v2
// - this unit is part of the mormot-mcp-server project
// - licensed under MPL/GPL/LGPL three license
// - adopted into the mormot.ai.* namespace for landrix (LandrixAI) from
//   flydev-fr/mormot2-extensions
unit mormot.ai.mcp;

{
  *****************************************************************************

    - Core Types and Authentication Context
    - IInvokable Interfaces for Tools and Resources
    - RTTI-based Schema Generation
    - JSON-RPC 2.0 Protocol Processor
    - Generic Tool and Resource Base Classes
    - Main MCP Server with Tool/Resource Registry

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils, // RTL Exception base type (broad catch in ExecuteRequest)
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.rtti,
  mormot.core.buffers,
  mormot.core.data,
  mormot.core.variants,
  mormot.core.json,
  mormot.core.collections,
  mormot.core.threads,
  mormot.core.interfaces;


{ ************ Core Types and Authentication Context }

const
  /// MCP Protocol Version (legacy baseline)
  MCP_PROTOCOL_VERSION = '2024-11-05';

  /// MCP Protocol Version that introduced Streamable HTTP
  MCP_PROTOCOL_VERSION_20250326 = '2025-03-26';

  /// MCP Protocol Version (2025-06-18 revision)
  MCP_PROTOCOL_VERSION_20250618 = '2025-06-18';

  /// latest MCP Protocol Version implemented by this server
  // - returned as fallback when the client requests an unsupported version
  MCP_PROTOCOL_VERSION_LATEST = '2025-11-25';

  /// MCP protocol revisions this server can negotiate
  // - HandleInitialize echoes the client's requested version if it appears
  //   here (per MCP version negotiation), otherwise falls back to _LATEST
  // - assembled from the named version constants so there is a single source
  //   of truth and the list cannot drift from the constants above
  MCP_SUPPORTED_PROTOCOL_VERSIONS: array[0..3] of RawUtf8 = (
    MCP_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION_20250326,
    MCP_PROTOCOL_VERSION_20250618, MCP_PROTOCOL_VERSION_LATEST);

  /// JSON-RPC 2.0 Error Codes
  JSONRPC_PARSE_ERROR = -32700;
  JSONRPC_INVALID_REQUEST = -32600;
  JSONRPC_METHOD_NOT_FOUND = -32601;
  JSONRPC_INVALID_PARAMS = -32602;
  JSONRPC_INTERNAL_ERROR = -32603;

type
  /// Authentication context passed to tool/resource execution
  // - provides user identity and authorization information
  TMcpAuthContext = packed record
    /// whether the request carries a verified identity
    // - stays FALSE until a real auth resolver sets it; the core transport never
    //   sets it true merely because a session id is present (a session id is an
    //   opaque correlation handle, not proof of identity). A tool that gates on
    //   identity MUST treat false as "unauthenticated" and fail closed.
    IsAuthenticated: boolean;
    /// unique user identifier
    // - until a real auth resolver runs this carries only the transport session
    //   id for correlation — do NOT treat it as an authenticated principal
    UserID: RawUtf8;
    /// user display name
    UserName: RawUtf8;
    /// assigned roles for authorization
    Roles: TRawUtf8DynArray;
  end;

  /// MCP-specific error information
  TMcpError = packed record
    /// JSON-RPC error code
    Code: integer;
    /// human-readable error message
    Message: RawUtf8;
    /// optional additional error data
    Data: variant;
  end;

  /// MCP-specific exception class
  EMcpException = class(ESynException);

  /// raised when a JSON-RPC method is not implemented (mapped to -32601)
  EMcpMethodNotFound = class(EMcpException);


{ ************ IInvokable Interfaces for Tools and Resources }

type
  /// MCP Tool interface - represents an executable operation
  IMcpTool = interface(IInvokable)
    ['{8F3C5A1D-9E2B-4F7C-A6D8-3B9E4C5F6A7D}']
    /// return the unique tool name
    function GetName: RawUtf8;
    /// return the human-readable description
    function GetDescription: RawUtf8;
    /// return the JSON schema for input parameters as TDocVariant
    function GetInputSchema: variant;
    /// execute the tool with given arguments and auth context
    // - Args is a TDocVariantData containing the input parameters
    // - returns a TDocVariantData with 'content' array field
    function Execute(const Args: variant; const AuthCtx: TMcpAuthContext): variant;
  end;

  /// MCP Resource interface - represents readable data
  IMcpResource = interface(IInvokable)
    ['{7E4D3B2C-8A1F-4E9D-B5C6-2A8E3D4F5B6C}']
    /// return the unique resource URI
    function GetUri: RawUtf8;
    /// return the human-readable name
    function GetName: RawUtf8;
    /// return the resource description
    function GetDescription: RawUtf8;
    /// return the MIME type of the resource content
    function GetMimeType: RawUtf8;
    /// read and return the resource content
    function Read: RawUtf8;
  end;


{ ************ RTTI-based Schema Generation }

type
  /// Generate JSON schema from Object RTTI
  // - generates basic schema with type, properties, and required fields
  TMcpSchemaGenerator = class
  public
    /// Returns a TDocVariantData with schema structure
    class function GenerateSchema(aTypeInfo: PRttiInfo): variant;
  end;


{ ************ JSON-RPC 2.0 Protocol Processor }

type
  /// Stateless JSON-RPC 2.0 request/response processor for MCP
  // - handles all MCP protocol methods
  TMcpJsonRpcProcessor = class
  private
    fServerName: RawUtf8;
    fServerVersion: RawUtf8;
    fProtocolVersion: RawUtf8;
    function ExtractRequestId(const aRequest: variant): variant;
    function CreateResponse(const aRequestId: variant): variant;
    function CreateErrorResponse(const aRequestId: variant; 
      aErrorCode: integer; const aErrorMsg: RawUtf8): variant;
  public
    /// initialize with server information
    constructor Create(const aServerName, aServerVersion: RawUtf8);
    /// parse JSON-RPC request and return method name and params
    // - raises exception on parse error
    function ParseRequest(const aJson: RawUtf8; 
      out aMethod: RawUtf8; out aParams: variant; out aRequestId: variant): boolean;
    /// create a successful JSON-RPC response
    function CreateSuccessResponse(const aRequestId, aResult: variant): RawUtf8;
    /// create an error JSON-RPC response
    function CreateError(const aRequestId: variant; aErrorCode: integer; 
      const aErrorMsg: RawUtf8): RawUtf8;
    /// handle 'initialize' method
    function HandleInitialize(const aParams: variant): variant;
    /// handle 'ping' method
    function HandlePing: variant;
    /// the server name exposed in initialize response
    property ServerName: RawUtf8 read fServerName write fServerName;
    /// the server version exposed in initialize response
    property ServerVersion: RawUtf8 read fServerVersion write fServerVersion;
  end;


{ ************ Generic Tool Base Class }

type
  /// Generic base class for strongly-typed MCP tools
  // - T is the input parameters record type
  // - automatically generates JSON schema from T's RTTI
  TMcpToolBase<T: record> = class(TInterfacedObject, IMcpTool)
  protected
    fName: RawUtf8;
    fDescription: RawUtf8;
    /// override this to implement tool logic
    function ExecuteTyped(const aParams: T; const aAuthCtx: TMcpAuthContext): variant; virtual; abstract;
  public
    /// initialize with tool name and description
    constructor Create(const aName, aDescription: RawUtf8); virtual;
    /// IMcpTool implementation
    function GetName: RawUtf8;
    function GetDescription: RawUtf8;
    function GetInputSchema: variant;
    function Execute(const aArgs: variant; const aAuthCtx: TMcpAuthContext): variant;
  end;


{ ************ Resource Base Class }

type
  /// Abstract base class for MCP resources
  TMcpResourceBase = class(TInterfacedObject, IMcpResource)
  protected
    fUri: RawUtf8;
    fName: RawUtf8;
    fDescription: RawUtf8;
    fMimeType: RawUtf8;
    /// override this to provide resource content
    function GetContent: RawUtf8; virtual; abstract;
  public
    /// initialize with resource metadata
    constructor Create(const aUri, aName, aDescription, aMimeType: RawUtf8); virtual;
    /// IMcpResource implementation
    function GetUri: RawUtf8;
    function GetName: RawUtf8;
    function GetDescription: RawUtf8;
    function GetMimeType: RawUtf8;
    function Read: RawUtf8;
  end;


{ ************ Main MCP Server with Registry }

type
  /// Main MCP server with tool/resource registry
  // - thread-safe registration and execution
  // - processes JSON-RPC requests
  TMcpServer = class(TSynPersistent)
  private
    fTools: IKeyValue<RawUtf8, IMcpTool>; // name -> IMcpTool
    fResources: IKeyValue<RawUtf8, IMcpResource>; // uri -> IMcpResource
    fProcessor: TMcpJsonRpcProcessor;
    fActive: boolean;
    fSafe: TLightLock;
    function ExecuteToolCall(const aParams: variant; const aAuthCtx: TMcpAuthContext): variant;
    function ExecuteResourceRead(const aParams: variant): variant;
    function ListTools: variant;
    function ListResources: variant;
  public
    /// initialize the MCP server
    constructor Create(const aServerName: RawUtf8 = 'M-MCP-Server';
      const aServerVersion: RawUtf8 = '1.0.0'); reintroduce;
    /// finalize and release resources
    destructor Destroy; override;
    /// register a tool implementation
    // - thread-safe, can be called before or after Start
    procedure RegisterTool(const aTool: IMcpTool);
    /// register a resource implementation
    // - thread-safe
    procedure RegisterResource(const aResource: IMcpResource);
    /// unregister a tool by name
    function UnregisterTool(const aName: RawUtf8): boolean;
    /// unregister a resource by URI
    function UnregisterResource(const aUri: RawUtf8): boolean;
    /// start the server (activates tool/resource access)
    procedure Start;
    /// stop the server
    procedure Stop;
    /// process a JSON-RPC request and return JSON response
    // - aSessionId can be used for session-specific operations
    function ExecuteRequest(const aRequestJson: RawUtf8; 
      const aSessionId: RawUtf8 = ''): RawUtf8;
    /// check if server is active
    function IsActive: boolean;
  end;


{ ************ Response Builder Helper }

type
  /// Helper to build MCP tool responses with content array
  TMcpResponseBuilder = class
  private
    fContent: TDocVariantData;
  public
    /// initialize builder
    constructor Create;
    /// add text content block
    function AddText(const aText: RawUtf8): TMcpResponseBuilder;
    /// add base64-encoded file content
    function AddFile(const aFilePath: RawUtf8; const aFileName: RawUtf8 = ''): TMcpResponseBuilder;
    /// build final response as variant
    function Build: variant;
  end;


implementation


{ ************ TMcpSchemaGenerator Implementation }

class function TMcpSchemaGenerator.GenerateSchema(aTypeInfo: PRttiInfo): variant;
var
  rc: TRttiCustom;
  prop: PRttiCustomProp;
  props, schema: TDocVariantData;
  required: TDocVariantData;
  i: PtrInt;
  propSchema: TDocVariantData;
  jsonType: RawUtf8;

  function JsonTypeFromRtti(const aRtti: TRttiCustom): RawUtf8;
  begin
    if aRtti = nil then
    begin
      result := 'string';
      exit;
    end;
    case aRtti.Parser of
      ptBoolean:
        result := 'boolean';
      ptByte, ptCardinal, ptInt64, ptInteger, ptQWord, ptWord, ptOrm:
        result := 'integer';
      ptCurrency, ptDouble, ptExtended, ptSingle, ptDateTime, ptDateTimeMS,
      ptUnixTime, ptUnixMSTime:
        result := 'number';
      ptRawByteString, ptRawJson, ptRawUtf8, ptString, ptSynUnicode,
      ptUnicodeString, ptWideString, ptWinAnsi, ptGuid, ptHash128, ptHash256,
      ptHash512, ptTimeLog, ptPUtf8Char, ptEnumeration:
        result := 'string';
      ptSet, ptArray, ptDynArray:
        result := 'array';
      ptRecord, ptClass, ptInterface:
        result := 'object';
      ptVariant, ptCustom:
        result := 'object';
    else
      result := 'string';
    end;
  end;
begin
  // Initialize schema structure
  schema.InitObject(['type', 'object'], JSON_FAST);
  props.InitObject([], JSON_FAST);
  required.InitArray([], JSON_FAST);

  // Get RTTI context for the supplied type
  if aTypeInfo = nil then
  begin
    result := variant(schema);
    exit;
  end;
  rc := Rtti.RegisterType(aTypeInfo);
  if rc = nil then
  begin
    result := variant(schema);
    exit;
  end;

  // Iterate through properties
  for i := 0 to rc.Props.Count - 1 do
  begin
    prop := @rc.Props.List[i];
    if prop.Name = '' then
      continue;

    // Initialize property schema
    propSchema.InitObject([], JSON_FAST);

    // Determine JSON type from RTTI
    jsonType := JsonTypeFromRtti(prop.Value);

    propSchema.AddValue('type', jsonType);

    // Add to properties
    props.AddValue(LowerCaseU(prop.Name), variant(propSchema));

    // All properties are required (no optional metadata available)
    required.AddItem(LowerCaseU(prop.Name));
  end;

  schema.AddValue('properties', variant(props));
  if required.Count > 0 then
    schema.AddValue('required', variant(required));

  result := variant(schema);
end;


{ ************ TMcpJsonRpcProcessor Implementation }

constructor TMcpJsonRpcProcessor.Create(const aServerName, aServerVersion: RawUtf8);
begin
  inherited Create;
  fServerName := aServerName;
  fServerVersion := aServerVersion;
  fProtocolVersion := MCP_PROTOCOL_VERSION_LATEST;
end;

function TMcpJsonRpcProcessor.ExtractRequestId(const aRequest: variant): variant;
begin
  with _Safe(aRequest)^ do
    result := GetValueOrNull('id');
end;

function TMcpJsonRpcProcessor.CreateResponse(const aRequestId: variant): variant;
var
  doc: TDocVariantData;
begin
  doc.InitObject(['jsonrpc', '2.0'], JSON_FAST);
  doc.AddValue('id', aRequestId);
  result := variant(doc);
end;

function TMcpJsonRpcProcessor.CreateErrorResponse(const aRequestId: variant;
  aErrorCode: integer; const aErrorMsg: RawUtf8): variant;
var
  doc, errorObj: TDocVariantData;
begin
  doc.InitObject(['jsonrpc', '2.0'], JSON_FAST);
  doc.AddValue('id', aRequestId);
  
  errorObj.InitObject(['code', aErrorCode, 'message', aErrorMsg], JSON_FAST);
  doc.AddValue('error', variant(errorObj));
  
  result := variant(doc);
end;

function TMcpJsonRpcProcessor.ParseRequest(const aJson: RawUtf8;
  out aMethod: RawUtf8; out aParams: variant; out aRequestId: variant): boolean;
var
  doc: PDocVariantData;
  request: variant;
  jsonrpc: RawUtf8;
  paramsIdx: PtrInt;
begin
  result := false;
  aMethod := '';
  aParams := Null;
  aRequestId := Null;

  // Parse JSON (invalid JSON yields a non-object -> rejected below)
  request := _JsonFast(aJson);
  doc := _Safe(request);
  if not doc.IsObject then
    exit;

  // JSON-RPC 2.0 envelope: the spec REQUIRES "jsonrpc":"2.0". Reject anything else
  // (missing / wrong version) so a malformed/foreign payload cannot be dispatched.
  if not doc.GetAsRawUtf8('jsonrpc', jsonrpc) or (jsonrpc <> '2.0') then
    exit;

  // Extract method (required, must be a string)
  if not doc.GetAsRawUtf8('method', aMethod) then
    exit;

  // Extract params (optional): if present it MUST be a structured value
  // (object or array) per the spec — a scalar params is a malformed request.
  paramsIdx := doc.GetValueIndex('params');
  if paramsIdx >= 0 then
  begin
    aParams := doc.Values[paramsIdx];
    if not (_Safe(aParams)^.IsObject or _Safe(aParams)^.IsArray) then
      exit;
  end
  else
    aParams := Null;

  // Extract id (absent => notification; when present must be string/number/null)
  aRequestId := ExtractRequestId(request);

  result := true;
end;

function TMcpJsonRpcProcessor.CreateSuccessResponse(const aRequestId, aResult: variant): RawUtf8;
var
  response: variant;
  doc: PDocVariantData;
begin
  response := CreateResponse(aRequestId);
  _ObjAddProp('result', aResult, response);
  result := ToUtf8(response);
end;

function TMcpJsonRpcProcessor.CreateError(const aRequestId: variant;
  aErrorCode: integer; const aErrorMsg: RawUtf8): RawUtf8;
var
  response: variant;
begin
  response := CreateErrorResponse(aRequestId, aErrorCode, aErrorMsg);
  result := ToUtf8(response);
end;

function TMcpJsonRpcProcessor.HandleInitialize(const aParams: variant): variant;
var
  result_doc, capabilities, serverInfo: TDocVariantData;
  clientVersion, negotiated: RawUtf8;
  i: integer;
begin
  // MCP version negotiation: echo the client's requested protocolVersion if we
  // support it; otherwise return our latest supported version and let the
  // client decide whether to proceed.
  clientVersion := _Safe(aParams)^.U['protocolVersion'];
  negotiated := fProtocolVersion; // default = latest supported
  for i := 0 to high(MCP_SUPPORTED_PROTOCOL_VERSIONS) do
    if MCP_SUPPORTED_PROTOCOL_VERSIONS[i] = clientVersion then
    begin
      negotiated := clientVersion;
      break;
    end;

  result_doc.InitObject(['protocolVersion', negotiated], JSON_FAST);
  
  // Add capabilities
  capabilities.InitObject([], JSON_FAST);
  capabilities.AddValue('tools', _ObjFast([]));
  capabilities.AddValue('resources', _ObjFast([]));
  result_doc.AddValue('capabilities', variant(capabilities));
  
  // Add server info
  serverInfo.InitObject(['name', fServerName, 'version', fServerVersion], JSON_FAST);
  result_doc.AddValue('serverInfo', variant(serverInfo));
  
  result := variant(result_doc);
end;

function TMcpJsonRpcProcessor.HandlePing: variant;
begin
  result := _Obj([]);
end;


{ ************ TMcpToolBase<T> Implementation }

constructor TMcpToolBase<T>.Create(const aName, aDescription: RawUtf8);
begin
  inherited Create;
  fName := aName;
  fDescription := aDescription;
end;

function TMcpToolBase<T>.GetName: RawUtf8;
begin
  result := fName;
end;

function TMcpToolBase<T>.GetDescription: RawUtf8;
begin
  result := fDescription;
end;

function TMcpToolBase<T>.GetInputSchema: variant;
var
  typeInfo: PRttiInfo;
begin
  typeInfo := System.TypeInfo(T);
  result := TMcpSchemaGenerator.GenerateSchema(typeInfo);
end;

function TMcpToolBase<T>.Execute(const aArgs: variant; 
  const aAuthCtx: TMcpAuthContext): variant;
var
  params: T;
  doc: PDocVariantData;
  json: RawUtf8;
begin
  // Deserialize arguments into typed record
  if _Safe(aArgs, doc) then
    json := doc^.ToJson
  else
    json := '{}';
  RecordLoadJson(params, json, TypeInfo(T));
  
  // Execute typed implementation
  result := ExecuteTyped(params, aAuthCtx);
end;


{ ************ TMcpResourceBase Implementation }

constructor TMcpResourceBase.Create(const aUri, aName, aDescription, aMimeType: RawUtf8);
begin
  inherited Create;
  fUri := aUri;
  fName := aName;
  fDescription := aDescription;
  fMimeType := aMimeType;
end;

function TMcpResourceBase.GetUri: RawUtf8;
begin
  result := fUri;
end;

function TMcpResourceBase.GetName: RawUtf8;
begin
  result := fName;
end;

function TMcpResourceBase.GetDescription: RawUtf8;
begin
  result := fDescription;
end;

function TMcpResourceBase.GetMimeType: RawUtf8;
begin
  result := fMimeType;
end;

function TMcpResourceBase.Read: RawUtf8;
begin
  result := GetContent;
end;


{ ************ TMcpServer Implementation }

constructor TMcpServer.Create(const aServerName, aServerVersion: RawUtf8);
begin
  inherited Create;
  fSafe.Init;
  fProcessor := TMcpJsonRpcProcessor.Create(aServerName, aServerVersion);
  fTools := Collections.NewPlainKeyValue<RawUtf8, IMcpTool>;
  fResources := Collections.NewPlainKeyValue<RawUtf8, IMcpResource>;
  fActive := false;
end;

destructor TMcpServer.Destroy;
begin
  Stop;
  fTools := nil;
  fResources := nil;
  fProcessor.Free;
  fSafe.Done;
  inherited;
end;

procedure TMcpServer.RegisterTool(const aTool: IMcpTool);
var
  name: RawUtf8;
begin
  if aTool = nil then
    exit;
  name := aTool.GetName;
  fSafe.Lock;
  try
    fTools.Add(name, aTool);
  finally
    fSafe.UnLock;
  end;
end;

procedure TMcpServer.RegisterResource(const aResource: IMcpResource);
var
  uri: RawUtf8;
begin
  if aResource = nil then
    exit;
  uri := aResource.GetUri;
  fSafe.Lock;
  try
    fResources.Add(uri, aResource);
  finally
    fSafe.UnLock;
  end;
end;

function TMcpServer.UnregisterTool(const aName: RawUtf8): boolean;
begin
  fSafe.Lock;
  try
    result := fTools.Remove(aName);
  finally
    fSafe.UnLock;
  end;
end;

function TMcpServer.UnregisterResource(const aUri: RawUtf8): boolean;
begin
  fSafe.Lock;
  try
    result := fResources.Remove(aUri);
  finally
    fSafe.UnLock;
  end;
end;

procedure TMcpServer.Start;
begin
  fSafe.Lock;
  try
    fActive := true;
  finally
    fSafe.UnLock;
  end;
end;

procedure TMcpServer.Stop;
begin
  fSafe.Lock;
  try
    fActive := false;
  finally
    fSafe.UnLock;
  end;
end;

function TMcpServer.IsActive: boolean;
begin
  result := fActive;
end;

function TMcpServer.ListTools: variant;
var
  doc, toolsList: TDocVariantData;
  toolObj: TDocVariantData;
  pair: TPair<RawUtf8, IMcpTool>;
begin
  doc.InitObject([], JSON_FAST);
  toolsList.InitArray([], JSON_FAST);

  fSafe.Lock;
  try
    for pair in fTools do
    begin
      toolObj.InitObject([
        'name', pair.Key,
        'description', pair.Value.GetDescription,
        'inputSchema', pair.Value.GetInputSchema
      ], JSON_FAST);
      toolsList.AddItem(variant(toolObj));
    end;
  finally
    fSafe.UnLock;
  end;

  doc.AddValue('tools', variant(toolsList));
  result := variant(doc);
end;

function TMcpServer.ListResources: variant;
var
  doc, resourcesList: TDocVariantData;
  resourceObj: TDocVariantData;
  pair: TPair<RawUtf8, IMcpResource>;
begin
  doc.InitObject([], JSON_FAST);
  resourcesList.InitArray([], JSON_FAST);

  fSafe.Lock;
  try
    for pair in fResources do
    begin
      resourceObj.InitObject([
        'uri', pair.Value.GetUri,
        'name', pair.Value.GetName,
        'description', pair.Value.GetDescription,
        'mimeType', pair.Value.GetMimeType
      ], JSON_FAST);
      resourcesList.AddItem(variant(resourceObj));
    end;
  finally
    fSafe.UnLock;
  end;

  doc.AddValue('resources', variant(resourcesList));
  result := variant(doc);
end;

function TMcpServer.ExecuteToolCall(const aParams: variant;
  const aAuthCtx: TMcpAuthContext): variant;
var
  doc: PDocVariantData;
  toolName: RawUtf8;
  args: variant;
  tool: IMcpTool;
begin
  if _Safe(aParams, doc) then
    if not doc.GetAsRawUtf8('name', toolName) then
      raise EMcpException.CreateU('Missing tool name in tools/call');

  args := doc.GetValueOrDefault('arguments',  Null);

  fSafe.Lock;
  try
    if not fTools.TryGetValue(toolName, tool) then
      raise EMcpException.CreateUtf8('Tool not found: %', [toolName]);
  finally
    fSafe.UnLock;
  end;

  result := tool.Execute(args, aAuthCtx);
end;

function TMcpServer.ExecuteResourceRead(const aParams: variant): variant;
var
  doc: PDocVariantData;
  uri, content: RawUtf8;
  resource: IMcpResource;
  result_doc, contentsList, contentItem: TDocVariantData;
begin
  if _Safe(aParams, doc) then
    if not doc.GetAsRawUtf8('uri', uri) then
      raise EMcpException.CreateU('Missing uri in resources/read');

  fSafe.Lock;
  try
    if not fResources.TryGetValue(uri, resource) then
      raise EMcpException.CreateUtf8('Resource not found: %', [uri]);
  finally
    fSafe.UnLock;
  end;

  content := resource.Read;

  // Build response
  result_doc.InitObject([], JSON_FAST);
  contentsList.InitArray([], JSON_FAST);
  
  contentItem.InitObject([
    'uri', uri,
    'mimeType', resource.GetMimeType,
    'text', content
  ], JSON_FAST);
  
  contentsList.AddItem(variant(contentItem));
  result_doc.AddValue('contents', variant(contentsList));
  
  result := variant(result_doc);
end;

function TMcpServer.ExecuteRequest(const aRequestJson, aSessionId: RawUtf8): RawUtf8;
var
  method: RawUtf8;
  params, requestId, resultData: variant;
  authCtx: TMcpAuthContext;
  isNotification: boolean;
begin
  requestId := Null;
  isNotification := false;
  if not fActive then
  begin
    result := fProcessor.CreateError(Null, JSONRPC_INTERNAL_ERROR, 'Server not active');
    exit;
  end;

  // Parse request: a malformed envelope (bad JSON / missing or wrong jsonrpc /
  // scalar params) is an Invalid Request — answer with -32600 and no id (we
  // could not reliably extract one), never dispatch it.
  if not fProcessor.ParseRequest(aRequestJson, method, params, requestId) then
  begin
    result := fProcessor.CreateError(Null, JSONRPC_INVALID_REQUEST,
      'Invalid JSON-RPC request');
    exit;
  end;

  isNotification := VarIsVoid(requestId);

  // Auth context. NOTE: a session id is NOT authentication — it is an opaque
  // transport correlation handle. We therefore leave IsAuthenticated = false
  // (fail-closed) and only carry the session id for correlation. Real identity
  // must be injected by a backend auth resolver (Phase B) before any tool may
  // trust IsAuthenticated/Roles. See DESIGN.md / CONCEPT.md roadmap.
  FillCharFast(authCtx, SizeOf(authCtx), 0);
  authCtx.IsAuthenticated := false;
  authCtx.UserID := aSessionId; // correlation only, not an authenticated identity

  try
    // Dispatch to handler — any handler/tool exception is mapped to a JSON-RPC
    // error below, so it never escapes into the HTTP worker.
    if method = 'initialize' then
      resultData := fProcessor.HandleInitialize(params)
    else if method = 'ping' then
      resultData := fProcessor.HandlePing
    else if method = 'tools/list' then
      resultData := ListTools
    else if method = 'tools/call' then
      resultData := ExecuteToolCall(params, authCtx)
    else if method = 'resources/list' then
      resultData := ListResources
    else if method = 'resources/read' then
      resultData := ExecuteResourceRead(params)
    else if method = 'notifications/initialized' then
      resultData := Null
    else if isNotification then
      resultData := Null
    else
      raise EMcpMethodNotFound.CreateUtf8('Method not found: %', [method]);

    // Create success response (unless notification)
    if isNotification then
      result := ''
    else
      result := fProcessor.CreateSuccessResponse(requestId, resultData);

  except
    // Catch EVERY exception (not just ESynException): tools may raise plain
    // Exception, EConvertError, DB/OS errors. Translate to a JSON-RPC error so
    // the transport stays alive and the client gets a well-formed response.
    on E: EMcpMethodNotFound do
      if isNotification then
        result := ''
      else
        result := fProcessor.CreateError(requestId, JSONRPC_METHOD_NOT_FOUND,
          StringToUtf8(E.Message));
    on E: Exception do
      if isNotification then
        result := ''
      else
        result := fProcessor.CreateError(requestId, JSONRPC_INTERNAL_ERROR,
          StringToUtf8(E.Message));
  end;
end;


{ ************ TMcpResponseBuilder Implementation }

constructor TMcpResponseBuilder.Create;
begin
  inherited Create;
  fContent.InitArray([], JSON_FAST);
end;

function TMcpResponseBuilder.AddText(const aText: RawUtf8): TMcpResponseBuilder;
var
  textItem: TDocVariantData;
begin
  textItem.InitObject(['type', 'text', 'text', aText], JSON_FAST);
  fContent.AddItem(variant(textItem));
  result := self;
end;

function TMcpResponseBuilder.AddFile(const aFilePath, aFileName: RawUtf8): TMcpResponseBuilder;
var
  fileItem: variant;
  content: RawByteString;
  base64: RawUtf8;
  mimeType, fileName: RawUtf8;
begin
  if not FileExists(Utf8ToString(aFilePath)) then
    raise EMcpException.CreateUtf8('File not found: %', [aFilePath]);

  content := StringFromFile(Utf8ToString(aFilePath));
  base64 := BinToBase64(content);
  
  if aFileName = '' then
    fileName := ExtractNameU(aFilePath)
  else
    fileName := aFileName;
    
  mimeType := GetMimeContentType(content, Utf8ToString(aFilePath));

  fileItem := _ObjFast([
    'type', 'resource',
    'mimeType', mimeType,
    'data', base64,
    'fileName', fileName
  ]);
  
  fContent.AddItem(fileItem);
  result := self;
end;

function TMcpResponseBuilder.Build: variant;
var
  contentCopy: TDocVariantData;
  i: integer;
begin
  SetVariantNull(result);

  contentCopy.InitArray([], JSON_FAST);
  for i := 0 to fContent.Count - 1 do
    contentCopy.AddItem(fContent.Values[i]);

  _ObjAddProp('content', contentCopy, result);
end;


end.
