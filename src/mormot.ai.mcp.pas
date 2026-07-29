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
  /// the one MCP protocol revision this server speaks
  // - the 2026-07-28 revision made MCP stateless: there is no `initialize`
  //   handshake and no version negotiation anymore. Every request carries its
  //   own version in _meta, and the server accepts or rejects it per request.
  // - this implementation is deliberately single-version ("modern" only, in the
  //   spec's terminology): earlier, handshake-based revisions are NOT served.
  //   A legacy client is answered with an error naming this version, which the
  //   spec recommends as the only diagnostic such a client can surface.
  MCP_PROTOCOL_VERSION = '2026-07-28';

  /// protocol revisions reported by server/discover and in -32022 errors
  // - single entry by design; kept as an array because the wire format is a
  //   list and consumers iterate it
  MCP_SUPPORTED_PROTOCOL_VERSIONS: array[0..0] of RawUtf8 = (
    MCP_PROTOCOL_VERSION);

  /// JSON-RPC 2.0 Error Codes
  JSONRPC_PARSE_ERROR = -32700;
  JSONRPC_INVALID_REQUEST = -32600;
  JSONRPC_METHOD_NOT_FOUND = -32601;
  JSONRPC_INVALID_PARAMS = -32602;
  JSONRPC_INTERNAL_ERROR = -32603;

  /// MCP-defined error codes (spec-reserved sub-range -32020..-32099)
  // - the codes -32001/-32003/-32004 of the draft and -32002 (resource not
  //   found) of earlier revisions MUST NOT be emitted anymore; a missing
  //   resource is a plain -32602 (Invalid params) now
  /// HTTP headers disagree with the request body, or a required header is absent
  MCP_ERROR_HEADER_MISMATCH = -32020;
  /// the request needs a client capability the client did not declare
  MCP_ERROR_MISSING_CLIENT_CAPABILITY = -32021;
  /// the requested protocol version is not implemented by this server
  MCP_ERROR_UNSUPPORTED_PROTOCOL_VERSION = -32022;

  /// reserved _meta keys of the MCP specification
  // - the `io.modelcontextprotocol/` prefix is reserved for MCP itself
  /// protocol version of a single request (REQUIRED on every client request)
  MCP_META_PROTOCOL_VERSION = 'io.modelcontextprotocol/protocolVersion';
  /// client capabilities relevant to a single request (REQUIRED)
  MCP_META_CLIENT_CAPABILITIES = 'io.modelcontextprotocol/clientCapabilities';
  /// client name/version, for display and logging only (optional, unverified)
  MCP_META_CLIENT_INFO = 'io.modelcontextprotocol/clientInfo';
  /// minimum log level the server should emit for this request (optional)
  MCP_META_LOG_LEVEL = 'io.modelcontextprotocol/logLevel';
  /// server name/version, echoed in every result's _meta (unverified)
  MCP_META_SERVER_INFO = 'io.modelcontextprotocol/serverInfo';
  /// correlates a notification with the subscriptions/listen request it came from
  MCP_META_SUBSCRIPTION_ID = 'io.modelcontextprotocol/subscriptionId';

  /// values of the mandatory `resultType` field on every result
  /// the request completed and the result carries the final content
  MCP_RESULT_COMPLETE = 'complete';
  /// interim result: the server needs more input (Multi Round-Trip Request)
  MCP_RESULT_INPUT_REQUIRED = 'input_required';

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
    function ExtractRequestId(const aRequest: variant): variant;
    function CreateResponse(const aRequestId: variant): variant;
    function CreateErrorResponse(const aRequestId: variant;
      aErrorCode: integer; const aErrorMsg: RawUtf8; const aData: variant): variant;
  public
    /// initialize with server information
    constructor Create(const aServerName, aServerVersion: RawUtf8);
    /// parse JSON-RPC request and return method name and params
    // - raises exception on parse error
    function ParseRequest(const aJson: RawUtf8;
      out aMethod: RawUtf8; out aParams: variant; out aRequestId: variant): boolean;
    /// create a successful JSON-RPC response
    // - stamps the mandatory `resultType` and the server identity into the
    //   result, so every handler can return its payload unadorned
    function CreateSuccessResponse(const aRequestId, aResult: variant): RawUtf8;
    /// create an error JSON-RPC response
    function CreateError(const aRequestId: variant; aErrorCode: integer;
      const aErrorMsg: RawUtf8): RawUtf8; overload;
    /// create an error JSON-RPC response carrying a `data` member
    // - used for the MCP errors that MUST report details (-32021 lists the
    //   missing capabilities, -32022 the supported versions)
    function CreateError(const aRequestId: variant; aErrorCode: integer;
      const aErrorMsg: RawUtf8; const aData: variant): RawUtf8; overload;
    /// `io.modelcontextprotocol/serverInfo` value (name + version)
    function ServerInfo: variant;
    /// stamp `resultType` and `_meta.serverInfo` onto a handler payload
    // - resultType is REQUIRED on every result since 2026-07-28; serverInfo is
    //   a SHOULD that replaces the identity formerly sent once in `initialize`
    function FinalizeResult(const aResult: variant;
      const aResultType: RawUtf8): variant;
    /// validate the per-request protocol metadata in `params._meta`
    // - since the protocol is stateless, EVERY request must carry its version
    //   and the client capabilities; there is no connection state to fall back on
    // - returns false and fills aError with -32602 (a required field is absent)
    //   or -32022 (the version is not the one we speak, incl. its `data`)
    function ValidateRequestMeta(const aParams: variant;
      out aError: TMcpError): boolean;
    /// handle the mandatory 'server/discover' RPC
    // - reports supported versions, capabilities and identity in one round trip
    function HandleDiscover: variant;
    /// the server name reported in server/discover and in every result's _meta
    property ServerName: RawUtf8 read fServerName write fServerName;
    /// the server version reported alongside ServerName
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
    // - no session parameter: protocol-level sessions were removed in
    //   2026-07-28. State that must span requests is passed as explicit,
    //   server-minted handles in the tool arguments instead.
    function ExecuteRequest(const aRequestJson: RawUtf8): RawUtf8;
    /// check if server is active
    function IsActive: boolean;
    /// the JSON-RPC processor, for transports that must emit protocol-level
    // errors themselves (e.g. a -32020 header mismatch, which is detected
    // before the body is ever dispatched)
    property Processor: TMcpJsonRpcProcessor read fProcessor;
  end;


{ ************ Client-side Request Metadata Helper }

/// build the per-request `_meta` object every client request MUST carry
// - since 2026-07-28 there is no handshake: protocol version and client
//   capabilities travel with each individual request
// - aClientName/aClientVersion are optional (clientInfo is a SHOULD, and is
//   explicitly display/logging only — never a security input)
function McpRequestMeta(const aClientName: RawUtf8 = '';
  const aClientVersion: RawUtf8 = ''): variant;

/// wrap params with the mandatory `_meta`, ready to be put into a request
// - pass the handler params (may be a void variant for parameterless RPCs)
function McpRequestParams(const aParams: variant;
  const aClientName: RawUtf8 = ''; const aClientVersion: RawUtf8 = ''): variant;


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
  aErrorCode: integer; const aErrorMsg: RawUtf8; const aData: variant): variant;
var
  doc, errorObj: TDocVariantData;
begin
  doc.InitObject(['jsonrpc', '2.0'], JSON_FAST);
  doc.AddValue('id', aRequestId);

  errorObj.InitObject(['code', aErrorCode, 'message', aErrorMsg], JSON_FAST);
  // `data` is optional per JSON-RPC; only emit it when a caller supplied one,
  // so an ordinary error stays a two-field object as before
  if not VarIsVoid(aData) then
    errorObj.AddValue('data', aData);
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

function TMcpJsonRpcProcessor.ServerInfo: variant;
begin
  result := _ObjFast(['name', fServerName, 'version', fServerVersion]);
end;

function TMcpJsonRpcProcessor.FinalizeResult(const aResult: variant;
  const aResultType: RawUtf8): variant;
var
  doc: TDocVariantData;
  src: PDocVariantData;
  meta: PDocVariantData;
  i: PtrInt;
begin
  // Copy the handler payload, then stamp the protocol fields on top. Copying
  // (instead of mutating aResult) keeps handlers free to return shared or
  // cached documents without us writing into them.
  doc.InitObject([], JSON_FAST);
  src := _Safe(aResult);
  if src^.IsObject then
    for i := 0 to src^.Count - 1 do
      doc.AddValue(src^.Names[i], src^.Values[i]);

  doc.AddValue('resultType', RawUtf8ToVariant(aResultType));

  // merge into an existing _meta rather than replacing it: a handler may
  // already have attached its own keys (e.g. a subscriptionId)
  if doc.GetAsDocVariant('_meta', meta) and meta^.IsObject then
    meta^.AddValue(MCP_META_SERVER_INFO, ServerInfo)
  else
    doc.AddValue('_meta', _ObjFast([MCP_META_SERVER_INFO, ServerInfo]));

  result := variant(doc);
end;

function TMcpJsonRpcProcessor.ValidateRequestMeta(const aParams: variant;
  out aError: TMcpError): boolean;
var
  meta: PDocVariantData;
  version: RawUtf8;
  i: PtrInt;
begin
  result := false;
  aError.Data := Null;

  // _meta lives inside params; a request without params cannot carry the
  // required protocol fields and is malformed
  if not _Safe(aParams)^.GetAsDocVariant('_meta', meta) or
     not meta^.IsObject then
  begin
    aError.Code := JSONRPC_INVALID_PARAMS;
    aError.Message := 'Missing _meta: ' + MCP_META_PROTOCOL_VERSION +
      ' and ' + MCP_META_CLIENT_CAPABILITIES + ' are required on every request';
    exit;
  end;

  if not meta^.GetAsRawUtf8(MCP_META_PROTOCOL_VERSION, version) or
     (version = '') then
  begin
    aError.Code := JSONRPC_INVALID_PARAMS;
    aError.Message := 'Missing required _meta field ' + MCP_META_PROTOCOL_VERSION;
    exit;
  end;

  // clientCapabilities is REQUIRED even when empty: its presence is what tells
  // the server the client declared *nothing*, as opposed to having forgotten
  // the field. Without that distinction -32021 could never be decided.
  if meta^.GetValueIndex(MCP_META_CLIENT_CAPABILITIES) < 0 then
  begin
    aError.Code := JSONRPC_INVALID_PARAMS;
    aError.Message := 'Missing required _meta field ' + MCP_META_CLIENT_CAPABILITIES;
    exit;
  end;

  for i := 0 to high(MCP_SUPPORTED_PROTOCOL_VERSIONS) do
    if MCP_SUPPORTED_PROTOCOL_VERSIONS[i] = version then
    begin
      result := true;
      exit;
    end;

  // unsupported version: the client needs the list to pick a common one
  aError.Code := MCP_ERROR_UNSUPPORTED_PROTOCOL_VERSION;
  aError.Message := 'Unsupported protocol version';
  aError.Data := _ObjFast([
    'supported', _ArrFast([MCP_PROTOCOL_VERSION]),
    'requested', version]);
end;

function TMcpJsonRpcProcessor.HandleDiscover: variant;
begin
  // Servers MUST implement server/discover. It reports what a client would
  // otherwise have learned from `initialize`: versions, capabilities, identity.
  // (serverInfo lands in _meta via FinalizeResult, where the spec puts it.)
  result := _ObjFast([
    'supportedVersions', _ArrFast([MCP_PROTOCOL_VERSION]),
    'capabilities', _ObjFast([
      'tools', _ObjFast([]),
      'resources', _ObjFast([])])]);
end;

function TMcpJsonRpcProcessor.CreateSuccessResponse(const aRequestId, aResult: variant): RawUtf8;
var
  response: variant;
begin
  response := CreateResponse(aRequestId);
  _ObjAddProp('result', FinalizeResult(aResult, MCP_RESULT_COMPLETE), response);
  result := ToUtf8(response);
end;

function TMcpJsonRpcProcessor.CreateError(const aRequestId: variant;
  aErrorCode: integer; const aErrorMsg: RawUtf8): RawUtf8;
begin
  result := CreateError(aRequestId, aErrorCode, aErrorMsg, Null);
end;

function TMcpJsonRpcProcessor.CreateError(const aRequestId: variant;
  aErrorCode: integer; const aErrorMsg: RawUtf8; const aData: variant): RawUtf8;
begin
  result := ToUtf8(CreateErrorResponse(aRequestId, aErrorCode, aErrorMsg, aData));
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

function TMcpServer.ExecuteRequest(const aRequestJson: RawUtf8): RawUtf8;
var
  method: RawUtf8;
  params, requestId, resultData: variant;
  authCtx: TMcpAuthContext;
  metaError: TMcpError;
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

  // Every request must carry its protocol version and client capabilities:
  // stateless means there is no earlier handshake that could have supplied
  // them. Notifications are exempt — the per-request fields are specified for
  // requests, and a notification has no response to carry the error in.
  if not isNotification then
    if not fProcessor.ValidateRequestMeta(params, metaError) then
    begin
      result := fProcessor.CreateError(requestId, metaError.Code,
        metaError.Message, metaError.Data);
      exit;
    end;

  // Auth context. Identity must be injected by a backend auth resolver before
  // any tool may trust IsAuthenticated/Roles; until then we stay fail-closed.
  // (Previously the transport session id was carried here for correlation —
  // protocol sessions no longer exist, so there is nothing to carry.)
  FillCharFast(authCtx, SizeOf(authCtx), 0);
  authCtx.IsAuthenticated := false;

  try
    // Dispatch to handler — any handler/tool exception is mapped to a JSON-RPC
    // error below, so it never escapes into the HTTP worker.
    if method = 'server/discover' then
      resultData := fProcessor.HandleDiscover
    else if method = 'tools/list' then
      resultData := ListTools
    else if method = 'tools/call' then
      resultData := ExecuteToolCall(params, authCtx)
    else if method = 'resources/list' then
      resultData := ListResources
    else if method = 'resources/read' then
      resultData := ExecuteResourceRead(params)
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


{ ************ Client-side Request Metadata Helper }

function McpRequestMeta(const aClientName, aClientVersion: RawUtf8): variant;
var
  doc: TDocVariantData;
begin
  doc.InitObject([
    MCP_META_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION,
    // REQUIRED even when empty — see ValidateRequestMeta
    MCP_META_CLIENT_CAPABILITIES, _ObjFast([])], JSON_FAST);
  if aClientName <> '' then
    doc.AddValue(MCP_META_CLIENT_INFO,
      _ObjFast(['name', aClientName, 'version', aClientVersion]));
  result := variant(doc);
end;

function McpRequestParams(const aParams: variant;
  const aClientName, aClientVersion: RawUtf8): variant;
var
  doc: TDocVariantData;
  src: PDocVariantData;
  i: PtrInt;
begin
  doc.InitObject([], JSON_FAST);
  src := _Safe(aParams);
  if src^.IsObject then
    for i := 0 to src^.Count - 1 do
      doc.AddValue(src^.Names[i], src^.Values[i]);
  doc.AddValue('_meta', McpRequestMeta(aClientName, aClientVersion));
  result := variant(doc);
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
