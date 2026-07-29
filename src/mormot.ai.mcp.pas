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

  /// default freshness hint: 0 = immediately stale, i.e. never reuse
  // - deliberately conservative. The registry can change at any moment via
  //   RegisterTool, and without subscriptions/listen there is no invalidation
  //   signal to correct a stale client. An embedder whose registry is static
  //   raises this; a wrong-but-fast default would hand out stale tool lists.
  MCP_CACHE_TTL_DEFAULT = 0;

  /// HTTP status codes an MCP-over-HTTP transport MUST use, as decided by the
  // protocol layer in TMcpServer.PreflightRequest
  // - declared here (and not taken from mormot.net.http) so this unit stays
  //   transport-agnostic: it names the status, the transport sends it
  /// the request is dispatchable, or its error is a plain application error
  HTTP_MCP_SUCCESS = 200;
  /// malformed envelope, bad/absent _meta, unsupported version, header mismatch
  // - the spec REQUIRES 400 for -32020/-32021/-32022 and tells clients to look
  //   for a "recognized modern JSON-RPC error" in a 400 body before falling
  //   back to the legacy handshake era
  HTTP_MCP_BAD_REQUEST = 400;
  /// the server does not implement the requested RPC method (-32601)
  // - spec: "it MUST respond with 404 Not Found and a JSON-RPC error with code
  //   -32601", which distinguishes it from a legacy server's bare 404
  HTTP_MCP_NOT_FOUND = 404;
  /// the server cannot serve the request at all (not active)
  HTTP_MCP_SERVER_ERROR = 500;

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

  /// raised when the params of a request are unusable (mapped to -32602)
  // - this covers "the named thing does not exist": since 2026-07-28 a missing
  //   resource is NOT its own error code anymore (the former -32002 was
  //   removed), it is a plain Invalid params
  EMcpInvalidParams = class(EMcpException);

  /// who may cache a result, i.e. the `cacheScope` field of a cacheable result
  // - an enumeration, not a string: the wire accepts exactly two values, and a
  //   free-form string lets a typo ship a successful but spec-invalid response.
  //   It also keeps the field out of the refcounted-string world, so a server
  //   reconfigured while requests are in flight cannot corrupt one (a RawUtf8
  //   read/written concurrently is not safe under FPC; a byte-sized enum is).
  TMcpCacheScope = (
    /// the response may only be reused within the SAME authorization context
    // - the safe default, and what the spec asks for on results that depend on
    //   the authenticated caller
    mcsPrivate,
    /// the response holds no user-specific data: ANY shared gateway or proxy
    // may store it and serve it to ANY other user
    // - the spec warns explicitly that this crosses authorization contexts: a
    //   `public` tools/list from an authenticated endpoint may be replayed to
    //   a different access token. Only choose it where the answer is identical
    //   for every caller.
    mcsPublic);

const
  /// the wire values of TMcpCacheScope, in enum order
  MCP_CACHE_SCOPE: array[TMcpCacheScope] of RawUtf8 = (
    'private',
    'public');

type
  /// the JSON-RPC methods this server dispatches
  // - resolved once from the wire name, then used by BOTH the pre-dispatch
  //   validation and the dispatch itself, so the two can never disagree about
  //   what "an implemented method" is (a transport MUST answer 404 for an
  //   unimplemented one, and it decides that before dispatching)
  TMcpMethod = (
    mcpUnknown,
    mcpDiscover,
    mcpToolsList,
    mcpToolsCall,
    mcpResourcesList,
    mcpResourcesRead);


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
    // - raises EMcpException if aResult is neither void nor a JSON object: the
    //   spec has no non-object result, and silently dropping such a payload
    //   would ship an empty success response instead of surfacing the bug
    function FinalizeResult(const aResult: variant;
      const aResultType: RawUtf8): variant;
    // (a response produced outside the dispatch is finalized by
    // TMcpServer.FinalizeHookResponse, which also knows the caching config)
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
    fListCacheTtlMs: integer;
    fReadCacheTtlMs: integer;
    fListCacheScope: TMcpCacheScope;
    fReadCacheScope: TMcpCacheScope;
    /// the actual preflight — hands the parsed request back so the dispatch
    // does not have to parse the very same body a second time
    function Preflight(const aRequestJson: RawUtf8; out aMethod: RawUtf8;
      out aParams, aRequestId: variant; out aErrorJson: RawUtf8;
      out aHttpStatus: integer): boolean;
    /// stamp the mandatory caching hints onto a cacheable result
    // - does nothing for the methods the spec does not list as cacheable
    procedure AddCacheHints(var aResult: variant; aMethod: TMcpMethod);
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
    // - runs PreflightRequest first, so no dispatch can ever happen on a
    //   request the protocol layer would reject
    function ExecuteRequest(const aRequestJson: RawUtf8): RawUtf8;
    /// finalize a response produced OUTSIDE the dispatch (a streaming hook)
    // - a hook answering e.g. tools/call replaces the handler, not the
    //   protocol: its result still needs resultType, serverInfo and — when the
    //   method is a cacheable one — the mandatory caching hints
    // - aMethod is the JSON-RPC method of the request the hook answered
    function FinalizeHookResponse(const aResponseJson,
      aMethod: RawUtf8): RawUtf8;
    /// validate a request WITHOUT dispatching it
    // - checks the JSON-RPC envelope, the mandatory per-request _meta and
    //   whether the method exists at all — everything that decides the HTTP
    //   status of a Streamable HTTP response
    // - returns true when the request may be dispatched (aHttpStatus 200)
    // - returns false with the ready-to-send JSON-RPC error in aErrorJson and
    //   the status the transport MUST use: 400 for a malformed envelope, bad
    //   _meta or an unsupported version, 404 for an unimplemented method
    // - transports MUST call this before handing a request to any streaming
    //   hook: a hook that answers a request the protocol layer rejects would
    //   execute unvalidated input (and skip resultType/serverInfo entirely)
    function PreflightRequest(const aRequestJson: RawUtf8;
      out aErrorJson: RawUtf8; out aHttpStatus: integer): boolean;
    /// check if server is active
    function IsActive: boolean;
    /// the JSON-RPC processor, for transports that must emit protocol-level
    // errors themselves (e.g. a -32020 header mismatch, which is detected
    // before the body is ever dispatched)
    property Processor: TMcpJsonRpcProcessor read fProcessor;
    /// freshness hint (ms) for server/discover, tools/list and resources/list
    // - the spec REQUIRES a caching hint on those results; 0 (the default)
    //   means "immediately stale", which is always correct, just not cheap
    // - raise it only if the registry is effectively static for that long:
    //   there is no invalidation signal until subscriptions/listen exists
    property ListCacheTtlMs: integer
      read fListCacheTtlMs write fListCacheTtlMs;
    /// freshness hint (ms) for resources/read
    // - separate from the list TTL because resource CONTENT usually changes on
    //   a different timescale than the set of resources
    property ReadCacheTtlMs: integer
      read fReadCacheTtlMs write fReadCacheTtlMs;
    /// who may cache server/discover, tools/list and resources/list
    // - defaults to mcsPrivate because this server cannot know whether a host
    //   application filters tools or resources per caller. Announcing public
    //   lets a shared proxy replay one caller's list to another caller, across
    //   authorization contexts — choose it only when every caller sees the same
    //   answer. It is a caching hint either way, never an access control.
    property ListCacheScope: TMcpCacheScope
      read fListCacheScope write fListCacheScope;
    /// who may cache resources/read
    // - deliberately SEPARATE from ListCacheScope: a static tool list is a
    //   reasonable candidate for public, while resource CONTENT is exactly what
    //   the spec names as typically per-user. One shared knob would force an
    //   operator who wants a cacheable tool list to publish resource bodies too.
    property ReadCacheScope: TMcpCacheScope
      read fReadCacheScope write fReadCacheScope;
  end;


{ ************ Method Resolution }

/// resolve a JSON-RPC method name to the handler this server implements
// - returns mcpUnknown for anything not implemented, which a Streamable HTTP
//   transport MUST turn into 404 + -32601 (spec: Protocol Version Header)
function McpMethodFromName(const aMethod: RawUtf8): TMcpMethod;


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
      doc.AddValue(src^.Names[i], src^.Values[i])
  else if not VarIsVoid(aResult) then
    // an array or a scalar cannot be a result: every MCP result is an object.
    // Copying nothing (the previous behaviour) turned a broken handler into a
    // silently empty success — fail loudly instead, it becomes -32603.
    raise EMcpException.CreateUtf8(
      'MCP result must be a JSON object, got %', [VariantToUtf8(aResult)]);

  // AddOrUpdateValue, not AddValue: TDocVariantData happily stores a SECOND
  // entry under an existing name, and every later lookup would still see the
  // handler's stale value while the wire carries both.
  doc.AddOrUpdateValue('resultType', RawUtf8ToVariant(aResultType));

  // merge into an existing _meta rather than replacing it: a handler may
  // already have attached its own keys (e.g. a subscriptionId)
  if doc.GetAsDocVariant('_meta', meta) and meta^.IsObject then
    meta^.AddOrUpdateValue(MCP_META_SERVER_INFO, ServerInfo)
  else
    doc.AddOrUpdateValue('_meta', _ObjFast([MCP_META_SERVER_INFO, ServerInfo]));

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
  // Always report what we speak, on EVERY _meta rejection — not just on
  // -32022. A client that does not know this server's version has nowhere
  // else to learn it: server/discover is itself a request and needs valid
  // _meta, so an error without the version list would leave a first-contact
  // client with no way forward but guessing.
  aError.Data := _ObjFast([
    'supported', _ArrFast([MCP_PROTOCOL_VERSION])]);

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
  fListCacheTtlMs := MCP_CACHE_TTL_DEFAULT;
  fReadCacheTtlMs := MCP_CACHE_TTL_DEFAULT;
  // never assume a shared cache is safe: both default to private
  fListCacheScope := mcsPrivate;
  fReadCacheScope := mcsPrivate;
end;

procedure TMcpServer.AddCacheHints(var aResult: variant; aMethod: TMcpMethod);
var
  doc: PDocVariantData;
  ttl: integer;
  scope: TMcpCacheScope;
begin
  // "Servers MUST include caching hints on results with resultType 'complete'
  // returned by server/discover, tools/list, prompts/list, resources/list,
  // resources/templates/list and resources/read." Deciding that here — in one
  // place keyed on the method — keeps a new handler from silently omitting
  // them, and keeps the list next to the spec sentence it implements.
  // Read each setting exactly once into a local: a server reconfigured while
  // requests are in flight then yields the old or the new value, never a mix.
  case aMethod of
    mcpDiscover,
    mcpToolsList,
    mcpResourcesList:
      begin
        ttl := fListCacheTtlMs;
        scope := fListCacheScope;
      end;
    mcpResourcesRead:
      begin
        ttl := fReadCacheTtlMs;
        scope := fReadCacheScope;
      end;
  else
    exit; // tools/call is not cacheable: it has side effects
  end;
  if ttl < 0 then
    ttl := 0; // spec: servers MUST provide a ttlMs >= 0
  doc := _Safe(aResult);
  if not doc^.IsObject then
    exit; // FinalizeResult rejects that anyway, with a better message
  doc^.AddOrUpdateValue('ttlMs', ttl);
  doc^.AddOrUpdateValue('cacheScope', RawUtf8ToVariant(MCP_CACHE_SCOPE[scope]));
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
      raise EMcpInvalidParams.CreateU('Missing tool name in tools/call');

  args := doc.GetValueOrDefault('arguments',  Null);

  fSafe.Lock;
  try
    if not fTools.TryGetValue(toolName, tool) then
      // -32602, not -32603: naming something that does not exist is bad input,
      // not a server failure (and -32002 was removed in 2026-07-28)
      raise EMcpInvalidParams.CreateUtf8('Tool not found: %', [toolName]);
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
      raise EMcpInvalidParams.CreateU('Missing uri in resources/read');

  fSafe.Lock;
  try
    if not fResources.TryGetValue(uri, resource) then
      // -32602: the dedicated "resource not found" code (-32002) was removed
      // in 2026-07-28 — an unknown URI is Invalid params like any other
      raise EMcpInvalidParams.CreateUtf8('Resource not found: %', [uri]);
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

function TMcpServer.FinalizeHookResponse(const aResponseJson,
  aMethod: RawUtf8): RawUtf8;
var
  doc: TDocVariantData;
  idx: PtrInt;
  res: variant;
begin
  result := aResponseJson;
  if aResponseJson = '' then
    exit;
  doc.InitJson(aResponseJson, JSON_FAST);
  idx := doc.GetValueIndex('result');
  if idx < 0 then
    exit; // an error response — nothing to stamp
  res := doc.Values[idx];
  AddCacheHints(res, McpMethodFromName(aMethod));
  doc.Values[idx] := fProcessor.FinalizeResult(res, MCP_RESULT_COMPLETE);
  result := doc.ToJson;
end;

function TMcpServer.PreflightRequest(const aRequestJson: RawUtf8;
  out aErrorJson: RawUtf8; out aHttpStatus: integer): boolean;
var
  method: RawUtf8;
  params, requestId: variant;
begin
  result := Preflight(aRequestJson, method, params, requestId,
    aErrorJson, aHttpStatus);
end;

function TMcpServer.Preflight(const aRequestJson: RawUtf8;
  out aMethod: RawUtf8; out aParams, aRequestId: variant;
  out aErrorJson: RawUtf8; out aHttpStatus: integer): boolean;
var
  metaError: TMcpError;
begin
  result := false;
  aErrorJson := '';
  aMethod := '';
  aParams := Null;
  aRequestId := Null;
  aHttpStatus := HTTP_MCP_BAD_REQUEST;

  if not fActive then
  begin
    aErrorJson := fProcessor.CreateError(Null, JSONRPC_INTERNAL_ERROR,
      'Server not active');
    aHttpStatus := HTTP_MCP_SERVER_ERROR;
    exit;
  end;

  // Parse request: a malformed envelope (bad JSON / missing or wrong jsonrpc /
  // scalar params) is an Invalid Request — answer with -32600 and no id (we
  // could not reliably extract one), never dispatch it.
  if not fProcessor.ParseRequest(aRequestJson, aMethod, aParams, aRequestId) then
  begin
    aErrorJson := fProcessor.CreateError(Null, JSONRPC_INVALID_REQUEST,
      'Invalid JSON-RPC request');
    exit;
  end;

  // Notifications are exempt from everything below: the per-request _meta is
  // specified for requests, an unknown notification is silently ignored per
  // JSON-RPC, and there is no response to carry an error in anyway.
  if VarIsVoid(aRequestId) then
  begin
    aHttpStatus := HTTP_MCP_SUCCESS;
    exit(true);
  end;

  // Every request must carry its protocol version and client capabilities:
  // stateless means there is no earlier handshake that could have supplied them
  if not fProcessor.ValidateRequestMeta(aParams, metaError) then
  begin
    aErrorJson := fProcessor.CreateError(aRequestId, metaError.Code,
      metaError.Message, metaError.Data);
    exit; // 400 — both -32602 and -32022 are "modern JSON-RPC errors" the spec
  end;    // tells clients to recognize on a 400 before falling back to legacy

  // An unimplemented method MUST be 404 (not 400), so a client can tell it
  // apart from a request this server refused to accept.
  if McpMethodFromName(aMethod) = mcpUnknown then
  begin
    aErrorJson := fProcessor.CreateError(aRequestId, JSONRPC_METHOD_NOT_FOUND,
      'Method not found: ' + aMethod);
    aHttpStatus := HTTP_MCP_NOT_FOUND;
    exit;
  end;

  aHttpStatus := HTTP_MCP_SUCCESS;
  result := true;
end;

function TMcpServer.ExecuteRequest(const aRequestJson: RawUtf8): RawUtf8;
var
  method: RawUtf8;
  params, requestId, resultData: variant;
  authCtx: TMcpAuthContext;
  isNotification: boolean;
  status: integer;
begin
  // Single validation gate, shared with the transports: whatever Preflight
  // rejects never reaches a handler, here or anywhere else. It hands the parsed
  // request back, so the body is parsed exactly once per dispatch.
  if not Preflight(aRequestJson, method, params, requestId, result, status) then
    exit;
  isNotification := VarIsVoid(requestId);

  // Auth context. Identity must be injected by a backend auth resolver before
  // any tool may trust IsAuthenticated/Roles; until then we stay fail-closed.
  // (Previously the transport session id was carried here for correlation —
  // protocol sessions no longer exist, so there is nothing to carry.)
  FillCharFast(authCtx, SizeOf(authCtx), 0);
  authCtx.IsAuthenticated := false;

  try
    // Dispatch to handler — any handler/tool exception is mapped to a JSON-RPC
    // error below, so it never escapes into the HTTP worker.
    case McpMethodFromName(method) of
      mcpDiscover:
        resultData := fProcessor.HandleDiscover;
      mcpToolsList:
        resultData := ListTools;
      mcpToolsCall:
        resultData := ExecuteToolCall(params, authCtx);
      mcpResourcesList:
        resultData := ListResources;
      mcpResourcesRead:
        resultData := ExecuteResourceRead(params);
    else
      // only reachable for a notification: Preflight turned every unknown
      // *request* method into -32601 before we got here
      resultData := Null;
    end;

    // Create success response (unless notification)
    if isNotification then
      result := ''
    else
    begin
      // caching hints belong on the complete result, before it is finalized
      AddCacheHints(resultData, McpMethodFromName(method));
      result := fProcessor.CreateSuccessResponse(requestId, resultData);
    end;

  except
    // Catch EVERY exception (not just ESynException): tools may raise plain
    // Exception, EConvertError, DB/OS errors. Translate to a JSON-RPC error so
    // the transport stays alive and the client gets a well-formed response.
    on E: EMcpInvalidParams do
      if isNotification then
        result := ''
      else
        result := fProcessor.CreateError(requestId, JSONRPC_INVALID_PARAMS,
          StringToUtf8(E.Message));
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


{ ************ Method Resolution }

function McpMethodFromName(const aMethod: RawUtf8): TMcpMethod;
begin
  // one place decides what this server implements; both PreflightRequest (which
  // must answer 404 for anything else) and the dispatch read it from here
  if aMethod = 'server/discover' then
    result := mcpDiscover
  else if aMethod = 'tools/list' then
    result := mcpToolsList
  else if aMethod = 'tools/call' then
    result := mcpToolsCall
  else if aMethod = 'resources/list' then
    result := mcpResourcesList
  else if aMethod = 'resources/read' then
    result := mcpResourcesRead
  else
    result := mcpUnknown;
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
  doc, meta: TDocVariantData;
  src, srcMeta: PDocVariantData;
  required: variant;
  i: PtrInt;
begin
  doc.InitObject([], JSON_FAST);
  src := _Safe(aParams);
  if src^.IsObject then
    for i := 0 to src^.Count - 1 do
      if src^.Names[i] <> '_meta' then
        doc.AddValue(src^.Names[i], src^.Values[i]);

  // Merge into the caller's own _meta instead of appending a second one:
  // TDocVariantData stores duplicate names happily, but every lookup returns
  // the FIRST — so an appended _meta would be invisible to the server, which
  // would then reject the request for missing protocol fields. Caller keys
  // (e.g. a progressToken) survive; the protocol fields are authoritative.
  meta.InitObject([], JSON_FAST);
  if src^.GetAsDocVariant('_meta', srcMeta) and srcMeta^.IsObject then
    for i := 0 to srcMeta^.Count - 1 do
      meta.AddValue(srcMeta^.Names[i], srcMeta^.Values[i]);
  // keep the mandatory fields in a named local: _Safe() on a function result
  // would point into a temporary the compiler may release before the loop ends
  required := McpRequestMeta(aClientName, aClientVersion);
  srcMeta := _Safe(required);
  for i := 0 to srcMeta^.Count - 1 do
    meta.AddOrUpdateValue(srcMeta^.Names[i], srcMeta^.Values[i]);

  doc.AddValue('_meta', variant(meta));
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
