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
  mormot.core.interfaces,
  mormot.crypt.core; // HmacSha256 for the MRTR requestState envelope


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

  /// the retry fields a client echoes back on a Multi Round-Trip Request
  // - both live directly in `params`, NOT in params._meta (see the tools/call
  //   retry example of the spec)
  /// map of client answers, keyed by the identifiers the server assigned
  MCP_PARAM_INPUT_RESPONSES = 'inputResponses';
  /// the opaque blob the server handed out, echoed back verbatim
  MCP_PARAM_REQUEST_STATE = 'requestState';

  /// the ONLY three server-to-client request methods an InputRequiredResult may
  // ask for — "inputRequests values are request objects that MUST be one of
  // ElicitRequest, CreateMessageRequest, or ListRootsRequest"
  MCP_INPUT_ELICITATION = 'elicitation/create';
  MCP_INPUT_SAMPLING = 'sampling/createMessage';
  MCP_INPUT_ROOTS = 'roots/list';

  /// the clientCapabilities key each input request method requires, in the same
  // order as MCP_INPUT_METHODS
  // - "Servers MUST NOT send an inputRequests that the client has not declared
  //   support for in its capabilities."
  MCP_INPUT_METHODS: array[0..2] of RawUtf8 = (
    MCP_INPUT_ELICITATION,
    MCP_INPUT_SAMPLING,
    MCP_INPUT_ROOTS);
  MCP_INPUT_CAPABILITIES: array[0..2] of RawUtf8 = (
    'elicitation',
    'sampling',
    'roots');

  /// how long an encoded requestState stays acceptable, in seconds
  // - the spec asks for "a short expiry (TTL)" inside the integrity-protected
  //   payload: it bounds the replay window of a state blob that leaked
  MCP_REQUEST_STATE_TTL_SEC = 300;

  /// how many notifications may queue up on one subscription before it is
  // dropped as unable to keep up
  // - a bound is required, not a nicety: without one a stalled client turns
  //   every resource update into permanent memory growth
  MCP_SUBSCRIPTION_MAX_PENDING = 256;

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

  /// raised when the client cannot serve an input request the server needs
  // (mapped to -32021 MissingRequiredClientCapability)
  // - not a server fault and not bad params: the request is well-formed, the
  //   client simply lacks a capability this call turned out to need
  EMcpInputCapabilityMissing = class(EMcpException)
  protected
    fCapabilities: TRawUtf8DynArray;
  public
    /// name the capability the client is missing, and what needed it
    // - the spec REQUIRES the error to carry `data.requiredCapabilities`: a
    //   client cannot act on a free-text message, and telling it which
    //   capability to add is the only way the retry can ever succeed
    constructor CreateCapability(const aCapability, aForMethod: RawUtf8);
    /// the missing capability names, for `error.data.requiredCapabilities`
    property Capabilities: TRawUtf8DynArray
      read fCapabilities;
  end;

  /// what a handler knows about the request it is serving, beyond its arguments
  // - carries the two Multi Round-Trip Request (MRTR) retry fields, so a
  //   handler that asked for input on a previous round can pick the answers up
  //   on this one
  TMcpCallContext = record
    /// the JSON-RPC method being served ('tools/call', 'resources/read', …)
    Method: RawUtf8;
    /// identity of the caller, as resolved by the backend's auth resolver
    Auth: TMcpAuthContext;
    /// what the client declared it can do, from _meta.clientCapabilities
    // - a handler MUST consult this before asking for an input type: requesting
    //   something the client cannot serve is a protocol violation. The server
    //   enforces it as a backstop (see TMcpServer.ValidateInputRequests), but
    //   failing there costs the caller a round trip for nothing.
    ClientCapabilities: variant;
    /// the client's answers to a previous InputRequiredResult, or void
    // - keys are the identifiers this server assigned in `inputRequests`
    // - "If additional, unexpected parameters are provided in the
    //   InputResponses object, the server SHOULD ignore any information it does
    //   not recognize or need."
    InputResponses: variant;
    /// the opaque blob this server handed out earlier, echoed back verbatim
    // - UNVERIFIED and attacker-controlled: the client is free to forge it.
    //   "If requestState influences authorization, resource access, or business
    //   logic, servers MUST protect its integrity" — decode it through
    //   TMcpRequestStateCodec (or an equivalent) instead of trusting it. The
    //   server cannot do that for you: only the handler knows what it encoded.
    RequestState: RawUtf8;
    /// whether the request carried an `inputResponses` field at all
    // - presence, not content: an empty `{}` is a client that answered with
    //   nothing, which is still a retry and still shapes a caller-specific
    //   answer. Deciding on the value would let `{}` slip past as "absent"
    //   and let a personalized result be cached as shareable.
    HasInputResponses: boolean;
    /// whether the request carried a `requestState` field at all
    // - same reasoning: an empty string is a present-but-empty state
    HasRequestState: boolean;
  end;

  /// raised by a handler to answer with an InputRequiredResult instead of a
  // result: the server needs more input before it can complete the request
  // - an exception, not a return value, so it works for every handler shape
  //   (a tool returning a variant and a resource returning RawUtf8 alike) and
  //   cannot be confused with a completed result
  // - the server turns this into `resultType: "input_required"`, and rejects it
  //   on any method where the spec forbids it
  EMcpInputRequired = class(EMcpException)
  protected
    fInputRequests: variant;
    fRequestState: RawUtf8;
  public
    /// ask the client for input, and/or carry state into the retry
    // - at least one of the two MUST be given: "Servers MUST include at least
    //   one of inputRequests or requestState in every InputRequiredResult"
    // - aInputRequests is an object whose keys are server-assigned identifiers
    //   and whose values are {method, params} request objects; build it with
    //   McpInputRequest for the method-name and shape checks
    constructor Create(const aInputRequests: variant;
      const aRequestState: RawUtf8 = ''); reintroduce;
    /// the server-initiated requests the client must fulfil (may be void)
    property InputRequests: variant
      read fInputRequests;
    /// opaque state the client MUST echo back on the retry (may be '')
    property RequestState: RawUtf8
      read fRequestState;
  end;

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
  /// which notification types a `subscriptions/listen` request opted into
  // - the server MUST NOT send a type the client did not ask for, so this is a
  //   whitelist, never a default-on set
  TMcpNotificationFilter = record
    /// notifications/tools/list_changed
    ToolsListChanged: boolean;
    /// notifications/prompts/list_changed (accepted, never raised: no prompts)
    PromptsListChanged: boolean;
    /// notifications/resources/list_changed
    ResourcesListChanged: boolean;
    /// resource URIs to watch, delivering notifications/resources/updated
    ResourceSubscriptions: TRawUtf8DynArray;
  end;

  /// one live `subscriptions/listen` stream, identified by its request id
  // - deliberately holds NO connection pointer: notifications are queued here
  //   by whichever thread produces them, and the thread that owns the stream
  //   drains the queue. That is what keeps a server-side push from ever
  //   touching a connection object it does not own — the use-after-free shape
  //   that the deleted session registry had.
  TMcpSubscription = class
  protected
    fId: variant;
    fFilter: TMcpNotificationFilter;
    fPending: TRawUtf8DynArray;
    fPendingCount: integer;
    fCancelled: boolean;
    fCancelReason: RawUtf8;
    fSafe: TLightLock;
    function GetCancelled: boolean;
  public
    /// initialize for the given subscriptions/listen request id and filter
    constructor Create(const aId: variant;
      const aFilter: TMcpNotificationFilter); reintroduce;
    /// release the queue and its lock
    destructor Destroy; override;
    /// queue one ready-made JSON-RPC notification for delivery
    // - called from arbitrary threads (whoever changed the tool list)
    // - a client that does not keep up must not be able to exhaust memory:
    //   past MCP_SUBSCRIPTION_MAX_PENDING the subscription is cancelled and
    //   the queue dropped. Ending the stream is the honest outcome — the
    //   client reconnects and re-reads the current state, which is exactly
    //   what it would have to do after any dropped stream.
    procedure Push(const aJson: RawUtf8);
    /// take everything queued so far; false when nothing was pending
    function Drain(out aJson: TRawUtf8DynArray): boolean;
    /// whether this filter asked for notifications about that resource URI
    function WatchesResource(const aUri: RawUtf8): boolean;
    /// mark as cancelled so the owning stream stops at its next turn
    // - aReason is echoed in the notifications/cancelled the stream sends on
    //   its way out; it is optional on the wire but the only thing that tells
    //   a client apart "the server is shutting down" from "you were too slow"
    procedure Cancel(const aReason: RawUtf8 = '');
    /// why this subscription was cancelled, for notifications/cancelled
    function CancelReason: RawUtf8;
    /// the JSON-RPC id of the originating request — also the subscription id
    property Id: variant read fId;
    /// the notification types this stream opted into
    property Filter: TMcpNotificationFilter read fFilter;
    /// set once the stream should end
    // - read through the lock: the flag is written by another thread, and on
    //   a weakly ordered architecture (our aarch64 target) an unsynchronized
    //   read has no visibility guarantee — a cancelled stream could run on
    property Cancelled: boolean read GetCancelled;
  end;

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
    mcpResourcesRead,
    mcpSubscriptionsListen);


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

  /// a tool that takes part in Multi Round-Trip Requests
  // - optional: the server calls ExecuteInteractive when the tool implements
  //   this, and plain IMcpTool.Execute otherwise, so existing tools keep
  //   working untouched
  // - raise EMcpInputRequired from ExecuteInteractive to ask for input
  IMcpInteractiveTool = interface(IMcpTool)
    ['{1C4A6F2D-9B37-4E58-8A0D-6F3B2E9C4D71}']
    /// execute with the full request context, including any input responses
    function ExecuteInteractive(const Args: variant;
      const Context: TMcpCallContext): variant;
  end;

  /// a resource that takes part in Multi Round-Trip Requests
  // - resources/read is one of the three methods that may answer with an
  //   InputRequiredResult, so a resource may need the retry fields too
  IMcpInteractiveResource = interface(IMcpResource)
    ['{5B8E1A3C-4D62-4F79-B1E5-8C2A7D0F3B94}']
    /// read with the full request context, including any input responses
    function ReadInteractive(const Context: TMcpCallContext): RawUtf8;
  end;


{ ************ Multi Round-Trip Request State }

type
  /// integrity-protected encoder for the opaque MRTR `requestState` blob
  // - requestState travels through the client, which "could attempt to modify
  //   it to alter server behavior, bypass authorization checks, or corrupt
  //   server logic". This wraps the handler's own state in an HMAC-SHA256
  //   envelope that also binds it to the caller, to the request, and to a
  //   deadline — the three replay defences the spec asks for.
  // - it is a tool, not a policy: the server never decodes requestState itself
  //   (it cannot know what a handler encoded). A handler that keeps no
  //   security-relevant state may skip it, which the spec permits only when
  //   "tampering can cause nothing worse than request failure".
  TMcpRequestStateCodec = class
  protected
    fSecret: RawByteString;
    fTtlSec: integer;
    function Mac(const aPayload: RawByteString): TSha256Digest;
  public
    /// initialize with the signing secret shared by every server instance
    // - MUST be the same on all instances behind a load balancer: MCP is
    //   stateless, so the retry may well land on a different one than the round
    //   that issued the state
    // - raises EMcpException on an empty or too short secret: a codec that
    //   silently signs with nothing at all is worse than none
    constructor Create(const aSecret: RawByteString;
      aTtlSec: integer = MCP_REQUEST_STATE_TTL_SEC); reintroduce;
    /// wrap the handler's state into a signed, bound, expiring blob
    // - aPrincipal is the authenticated caller the state belongs to; a blob
    //   issued for one principal is rejected when presented by another
    // - aBinding identifies the originating request: pass the method plus a
    //   digest of the salient parameters, so state cannot be moved to a
    //   different call
    function Encode(const aState: variant; const aPrincipal, aBinding: RawUtf8): RawUtf8;
    /// verify a blob and recover the handler's state
    // - returns false — without telling the caller why — when the signature,
    //   the principal, the binding or the deadline does not check out
    function Decode(const aRequestState, aPrincipal, aBinding: RawUtf8;
      out aState: variant): boolean;
    /// how long an encoded blob stays valid, in seconds
    property TtlSec: integer
      read fTtlSec;
  end;

/// the HTTP status an MCP-over-HTTP transport MUST send for a finished response
// - the spec pins three of its own error codes to 400 and -32601 to 404. Two of
//   those can only be decided AFTER a handler ran (-32021 depends on what the
//   handler turned out to need), so a transport cannot rely on the preflight
//   alone: it has to look at the response it is about to send.
// - every other outcome, including an application-level -32602 or -32603, is a
//   perfectly ordinary 200 carrying a JSON-RPC error
function McpHttpStatus(const aResponseJson: RawUtf8): integer;

/// build one entry of an InputRequiredResult's `inputRequests` map
// - rejects any method other than the three the spec allows, so a typo becomes
//   a loud server-side failure instead of a response no client understands
function McpInputRequest(const aMethod: RawUtf8; const aParams: variant): variant;


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
    // - aResultType is the mandatory `resultType` field; it stays 'complete'
    //   except for the interim result of a Multi Round-Trip Request
    function CreateSuccessResponse(const aRequestId, aResult: variant;
      const aResultType: RawUtf8 = MCP_RESULT_COMPLETE): RawUtf8;
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
    fSubscriptions: array of TMcpSubscription;
    fSubscriptionSafe: TLightLock;
    fMaxSubscriptions: integer;
    /// queue one notification on every subscription that opted into it
    // - aUri selects the watchers for notifications/resources/updated and is
    //   ignored (and empty) for the list-changed notifications
    procedure Broadcast(const aNotification, aUri: RawUtf8);
    /// the actual preflight — hands the parsed request back so the dispatch
    // does not have to parse the very same body a second time
    function Preflight(const aRequestJson: RawUtf8; out aMethod: RawUtf8;
      out aParams, aRequestId: variant; out aErrorJson: RawUtf8;
      out aHttpStatus: integer): boolean;
    /// stamp the mandatory caching hints onto a cacheable result
    // - does nothing for the methods the spec does not list as cacheable
    // - aPersonalized forces the conservative hint on a result that was shaped
    //   by MRTR input responses (see the implementation for why)
    procedure AddCacheHints(var aResult: variant; aMethod: TMcpMethod;
      aPersonalized: boolean = false);
    /// gather everything a handler may need beyond its own arguments
    function CallContext(const aParams: variant; const aMethod: RawUtf8;
      const aAuthCtx: TMcpAuthContext): TMcpCallContext;
    /// turn a handler's EMcpInputRequired into the interim result document
    function InputRequiredResult(const aInputRequests: variant;
      const aRequestState: RawUtf8): variant;
    /// build the -32021 response, carrying the capabilities the spec requires
    function CapabilityError(const aRequestId: variant;
      aError: EMcpInputCapabilityMissing): RawUtf8;
    function ExecuteToolCall(const aParams: variant;
      const aContext: TMcpCallContext): variant;
    function ExecuteResourceRead(const aParams: variant;
      const aContext: TMcpCallContext): variant;
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
    /// open a subscriptions/listen stream for an already-validated request
    // - the caller (a transport) owns the returned object and MUST pass it to
    //   CloseSubscription when its stream ends, whatever ends it
    // - returns nil when MaxSubscriptions is already reached: an unbounded
    //   number of long-lived streams would exhaust the HTTP worker pool, so
    //   refusing is a availability guard, not a protocol decision
    function OpenSubscription(const aRequestId, aParams: variant): TMcpSubscription;
    /// end a subscription and release it
    procedure CloseSubscription(aSubscription: TMcpSubscription);
    /// how many subscriptions/listen streams are currently open
    function SubscriptionCount: integer;
    /// ask every open stream to end, without waiting for them
    // - for an orderly shutdown: each owning thread notices at its next turn,
    //   sends the graceful-closure response and releases its subscription
    procedure CancelAllSubscriptions;
    /// the first message on a stream: what the server agreed to deliver
    // - MUST precede every notification of that subscription
    function SubscriptionAcknowledgement(
      aSubscription: TMcpSubscription): RawUtf8;
    /// the notifications/cancelled a server-side teardown MUST send
    // - sent before SubscriptionEndResponse: this says why the stream ends,
    //   the empty result then closes the long-lived request itself
    function SubscriptionCancelledNotification(
      aSubscription: TMcpSubscription): RawUtf8;
    /// the empty result that ends a subscription gracefully
    // - lets a client tell an orderly shutdown from a dropped connection
    function SubscriptionEndResponse(aSubscription: TMcpSubscription): RawUtf8;
    /// tell subscribers the tool list changed
    // - call after RegisterTool/UnregisterTool; only streams that opted into
    //   toolsListChanged receive it
    procedure NotifyToolsListChanged;
    /// tell subscribers the resource list changed
    procedure NotifyResourcesListChanged;
    /// tell subscribers watching that URI that the resource changed
    procedure NotifyResourceUpdated(const aUri: RawUtf8);
    /// finalize a response produced OUTSIDE the dispatch (a streaming hook)
    // - a hook answering e.g. tools/call replaces the handler, not the
    //   protocol: its result still needs resultType, serverInfo and — when the
    //   method is a cacheable one — the mandatory caching hints
    // - takes the whole REQUEST body, not just its method name: the MRTR rules
    //   (no caching hints on an interim result, no shareable cache on a retry)
    //   depend on the request's params, and a hook that answers a round trip
    //   must obey them exactly as the dispatcher does
    function FinalizeHookResponse(const aResponseJson,
      aRequestJson: RawUtf8): RawUtf8;
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
    /// reject an InputRequiredResult the spec would not allow on the wire
    // - the server calls this on every EMcpInputRequired before answering, so a
    //   handler cannot put a malformed or forbidden interim result on the wire
    // - public so a handler can check its own construction up front: failing
    //   here costs a round trip, failing at raise-time costs the whole call
    // - raises EMcpInputCapabilityMissing when the client did not declare the
    //   needed capability (-32021), and EMcpException on anything else, which
    //   is a defect in the calling handler and becomes -32603
    procedure ValidateInputRequests(const aInputRequests: variant;
      const aRequestState: RawUtf8; aMethod: TMcpMethod;
      const aClientCapabilities: variant);
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
    /// how many subscriptions/listen streams may be open at once (default 8)
    // - each open stream occupies one HTTP worker thread for its whole life
    //   (the transport drains its queue there), so an unbounded number would
    //   let a single client starve the pool and take the API down. Raise it
    //   only together with the transport's ServerThreadPoolCount.
    property MaxSubscriptions: integer
      read fMaxSubscriptions write fMaxSubscriptions;
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

/// read the `notifications` filter of a subscriptions/listen request
// - an absent or malformed filter yields an all-false filter: the server MUST
//   NOT send a type the client did not explicitly request
function McpParseNotificationFilter(const aParams: variant): TMcpNotificationFilter;


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


{ ************ Multi Round-Trip Request State }

constructor EMcpInputRequired.Create(const aInputRequests: variant;
  const aRequestState: RawUtf8);
begin
  // The message is diagnostic only: this exception never reaches a client as an
  // error, the server converts it into an InputRequiredResult.
  inherited CreateU('MCP handler requires additional input');
  fInputRequests := aInputRequests;
  fRequestState := aRequestState;
end;

constructor EMcpInputCapabilityMissing.CreateCapability(
  const aCapability, aForMethod: RawUtf8);
begin
  CreateUtf8('The client did not declare the "%" capability that % requires',
    [aCapability, aForMethod]);
  AddRawUtf8(fCapabilities, aCapability);
end;

function McpHttpStatus(const aResponseJson: RawUtf8): integer;
var
  doc: TDocVariantData;
  err: PDocVariantData;
  code: Int64;
begin
  result := HTTP_MCP_SUCCESS;
  if aResponseJson = '' then
    exit; // a notification: the transport decides (202), not the payload
  doc.InitJson(aResponseJson, JSON_FAST);
  if not doc.GetAsDocVariant('error', err) or
     not VariantToInt64(err^.GetValueOrDefault('code', 0), code) then
    exit;
  case code of
    MCP_ERROR_HEADER_MISMATCH,
    MCP_ERROR_MISSING_CLIENT_CAPABILITY,
    MCP_ERROR_UNSUPPORTED_PROTOCOL_VERSION:
      result := HTTP_MCP_BAD_REQUEST;
    JSONRPC_METHOD_NOT_FOUND:
      result := HTTP_MCP_NOT_FOUND;
  end;
  // deliberately NOT -32602/-32603: those are ordinary application outcomes
  // (unknown tool, handler failure) and the spec pins no status to them. Only
  // the codes it names get a status of their own.
end;

function McpInputRequest(const aMethod: RawUtf8; const aParams: variant): variant;
var
  i: PtrInt;
begin
  // params of a JSON-RPC request is an object (none of the three allowed
  // requests takes positional params). Letting a null or a scalar through
  // would have this server emit a message its own parser rejects.
  if not _Safe(aParams)^.IsObject then
    raise EMcpException.CreateUtf8(
      'inputRequests params for % must be a JSON object', [aMethod]);
  for i := 0 to high(MCP_INPUT_METHODS) do
    if MCP_INPUT_METHODS[i] = aMethod then
    begin
      result := _ObjFast(['method', aMethod, 'params', aParams]);
      exit;
    end;
  // fail here, at the point of the typo, rather than shipping a request object
  // no client can dispatch
  raise EMcpException.CreateUtf8(
    '% is not a valid inputRequests method: MCP allows only %, % and %',
    [aMethod, MCP_INPUT_ELICITATION, MCP_INPUT_SAMPLING, MCP_INPUT_ROOTS]);
end;


{ TMcpRequestStateCodec }

const
  /// shortest secret we accept, in bytes
  // - HMAC-SHA256 with a key shorter than this is not meaningfully unguessable,
  //   and the whole point of the envelope is that a client cannot forge one
  MCP_REQUEST_STATE_MIN_SECRET = 32;

constructor TMcpRequestStateCodec.Create(const aSecret: RawByteString;
  aTtlSec: integer);
begin
  inherited Create;
  if length(aSecret) < MCP_REQUEST_STATE_MIN_SECRET then
    raise EMcpException.CreateUtf8(
      'TMcpRequestStateCodec needs a secret of at least % bytes, got %',
      [MCP_REQUEST_STATE_MIN_SECRET, length(aSecret)]);
  if aTtlSec <= 0 then
    raise EMcpException.CreateUtf8(
      'TMcpRequestStateCodec needs a positive TTL, got %', [aTtlSec]);
  fSecret := aSecret;
  fTtlSec := aTtlSec;
end;

function TMcpRequestStateCodec.Mac(const aPayload: RawByteString): TSha256Digest;
begin
  HmacSha256(fSecret, aPayload, result);
end;

function TMcpRequestStateCodec.Encode(const aState: variant;
  const aPrincipal, aBinding: RawUtf8): RawUtf8;
var
  payload: RawByteString;
begin
  // Everything the spec asks to verify on receipt travels INSIDE the signed
  // payload: the principal it was issued to, the request it belongs to, and
  // the moment it stops being acceptable. Signing the envelope rather than
  // just the state is what makes those three unforgeable.
  payload := ToUtf8(_ObjFast([
    'p', aPrincipal,
    'b', aBinding,
    'e', UnixTimeUtc + fTtlSec,
    's', aState]));
  result := BinToBase64uri(payload) + '.' + BinToBase64uri(Mac(payload));
end;

function TMcpRequestStateCodec.Decode(const aRequestState, aPrincipal,
  aBinding: RawUtf8; out aState: variant): boolean;
var
  dot: PtrInt;
  payload, sig: RawByteString;
  doc: TDocVariantData;
  expected: TSha256Digest;
begin
  result := false;
  VarClear(aState);
  dot := PosExChar('.', aRequestState);
  if dot <= 1 then
    exit;
  payload := Base64uriToBin(copy(aRequestState, 1, dot - 1));
  sig := Base64uriToBin(copy(aRequestState, dot + 1, maxInt));
  if (payload = '') or
     (length(sig) <> SizeOf(expected)) then
    exit;

  // Verify BEFORE parsing: a forged payload must never reach the JSON parser,
  // let alone the handler. IsEqual is the constant-time compare — a byte-wise
  // one would leak how much of a guessed signature was right.
  expected := Mac(payload);
  if not IsEqual(PSha256Digest(pointer(sig))^, expected) then
    exit;

  doc.InitJson(RawUtf8(payload), JSON_FAST);
  if not doc.IsObject then
    exit;
  // Bind checks are separate from the signature: a blob can be perfectly
  // authentic and still be replayed by another user, onto another call, or
  // after it should have lapsed.
  if (doc.U['p'] <> aPrincipal) or
     (doc.U['b'] <> aBinding) or
     (doc.I['e'] <= UnixTimeUtc) then
    exit;
  aState := doc.GetValueOrDefault('s', Null);
  result := true;
end;


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
  // listChanged/subscribe are advertised because the registry raises those
  // notifications itself (see RegisterTool/RegisterResource) — a client that
  // opens subscriptions/listen for them will really be told about changes.
  result := _ObjFast([
    'supportedVersions', _ArrFast([MCP_PROTOCOL_VERSION]),
    'capabilities', _ObjFast([
      'tools', _ObjFast(['listChanged', true]),
      'resources', _ObjFast([
        'listChanged', true,
        'subscribe', true])])]);
end;

function TMcpJsonRpcProcessor.CreateSuccessResponse(const aRequestId, aResult: variant;
  const aResultType: RawUtf8): RawUtf8;
var
  response: variant;
begin
  response := CreateResponse(aRequestId);
  _ObjAddProp('result', FinalizeResult(aResult, aResultType), response);
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
  fSubscriptionSafe.Init;
  fMaxSubscriptions := 8; // see the property: each one holds a worker thread
end;

function TMcpServer.OpenSubscription(const aRequestId,
  aParams: variant): TMcpSubscription;
var
  n: PtrInt;
begin
  result := nil;
  fSubscriptionSafe.Lock;
  try
    n := length(fSubscriptions);
    if n >= fMaxSubscriptions then
      exit; // caller turns this into an error response
    result := TMcpSubscription.Create(aRequestId,
      McpParseNotificationFilter(aParams));
    SetLength(fSubscriptions, n + 1);
    fSubscriptions[n] := result;
  finally
    fSubscriptionSafe.UnLock;
  end;
end;

procedure TMcpServer.CloseSubscription(aSubscription: TMcpSubscription);
var
  i, n: PtrInt;
begin
  if aSubscription = nil then
    exit;
  fSubscriptionSafe.Lock;
  try
    n := length(fSubscriptions);
    for i := 0 to n - 1 do
      if fSubscriptions[i] = aSubscription then
      begin
        // remove BEFORE freeing, and while holding the lock: a concurrent
        // Broadcast must never reach an object that is about to be released
        if i < n - 1 then
          MoveFast(fSubscriptions[i + 1], fSubscriptions[i],
            (n - 1 - i) * SizeOf(pointer));
        SetLength(fSubscriptions, n - 1);
        break;
      end;
  finally
    fSubscriptionSafe.UnLock;
  end;
  aSubscription.Cancel;
  aSubscription.Free;
end;

function TMcpServer.SubscriptionCount: integer;
begin
  fSubscriptionSafe.Lock;
  try
    result := length(fSubscriptions);
  finally
    fSubscriptionSafe.UnLock;
  end;
end;

procedure TMcpServer.CancelAllSubscriptions;
var
  i: PtrInt;
begin
  fSubscriptionSafe.Lock;
  try
    for i := 0 to high(fSubscriptions) do
      fSubscriptions[i].Cancel('the server is shutting down');
  finally
    fSubscriptionSafe.UnLock;
  end;
end;

procedure TMcpServer.Broadcast(const aNotification, aUri: RawUtf8);
var
  i: PtrInt;
  sub: TMcpSubscription;
  wants: boolean;
  json: RawUtf8;
begin
  fSubscriptionSafe.Lock;
  try
    // Push() under the registry lock on purpose: it is a short append into the
    // subscription's own queue, and holding the lock is what guarantees the
    // object is still alive (CloseSubscription unlinks under the same lock).
    for i := 0 to high(fSubscriptions) do
    begin
      sub := fSubscriptions[i];
      if aNotification = 'notifications/tools/list_changed' then
        wants := sub.Filter.ToolsListChanged
      else if aNotification = 'notifications/resources/list_changed' then
        wants := sub.Filter.ResourcesListChanged
      else if aNotification = 'notifications/resources/updated' then
        wants := sub.WatchesResource(aUri)
      else
        wants := false;
      if not wants then
        continue;
      // every message on the stream carries the subscription id, which is the
      // id of the listen request — that is how a client demultiplexes stdio
      json := _Safe(_ObjFast([
        'jsonrpc', '2.0',
        'method', aNotification,
        'params', _ObjFast([
          '_meta', _ObjFast([MCP_META_SUBSCRIPTION_ID, sub.Id])])]))^.ToJson;
      if aUri <> '' then
        json := _Safe(_ObjFast([
          'jsonrpc', '2.0',
          'method', aNotification,
          'params', _ObjFast([
            '_meta', _ObjFast([MCP_META_SUBSCRIPTION_ID, sub.Id]),
            'uri', aUri])]))^.ToJson;
      sub.Push(json);
    end;
  finally
    fSubscriptionSafe.UnLock;
  end;
end;

procedure TMcpServer.NotifyToolsListChanged;
begin
  Broadcast('notifications/tools/list_changed', '');
end;

procedure TMcpServer.NotifyResourcesListChanged;
begin
  Broadcast('notifications/resources/list_changed', '');
end;

procedure TMcpServer.NotifyResourceUpdated(const aUri: RawUtf8);
begin
  if aUri <> '' then
    Broadcast('notifications/resources/updated', aUri);
end;

function TMcpServer.SubscriptionAcknowledgement(
  aSubscription: TMcpSubscription): RawUtf8;
var
  agreed: TDocVariantData;
  uris: TDocVariantData;
  i: PtrInt;
begin
  // "The notifications field in the acknowledgment reflects the subset the
  // server agreed to honor. Notification types the server does not support are
  // omitted." promptsListChanged is therefore never echoed: there are no
  // prompts in this server, so it could never fire.
  agreed.InitObject([], JSON_FAST);
  if aSubscription.Filter.ToolsListChanged then
    agreed.AddValue('toolsListChanged', true);
  if aSubscription.Filter.ResourcesListChanged then
    agreed.AddValue('resourcesListChanged', true);
  if aSubscription.Filter.ResourceSubscriptions <> nil then
  begin
    uris.InitArray([], JSON_FAST);
    for i := 0 to high(aSubscription.Filter.ResourceSubscriptions) do
      uris.AddItem(aSubscription.Filter.ResourceSubscriptions[i]);
    agreed.AddValue('resourceSubscriptions', variant(uris));
  end;
  result := _Safe(_ObjFast([
    'jsonrpc', '2.0',
    'method', 'notifications/subscriptions/acknowledged',
    'params', _ObjFast([
      '_meta', _ObjFast([MCP_META_SUBSCRIPTION_ID, aSubscription.Id]),
      'notifications', variant(agreed)])]))^.ToJson;
end;

function TMcpServer.SubscriptionCancelledNotification(
  aSubscription: TMcpSubscription): RawUtf8;
var
  params: TDocVariantData;
  reason: RawUtf8;
begin
  // "A server MUST send notifications/cancelled referencing a
  // subscriptions/listen request ID when it tears down that subscription
  // stream." That is the only purpose a server may send it for — it is NOT a
  // general-purpose "I gave up on your request" message.
  // The empty subscriptions/listen response (SubscriptionEndResponse) is a
  // separate SHOULD and follows this one: this says why the stream ends, that
  // one closes the long-lived request it belongs to.
  params.InitObject([
    'requestId', aSubscription.Id,
    '_meta', _ObjFast([MCP_META_SUBSCRIPTION_ID, aSubscription.Id])], JSON_FAST);
  reason := aSubscription.CancelReason;
  if reason <> '' then
    params.AddValue('reason', RawUtf8ToVariant(reason));
  result := _Safe(_ObjFast([
    'jsonrpc', '2.0',
    'method', 'notifications/cancelled',
    'params', variant(params)]))^.ToJson;
end;

function TMcpServer.SubscriptionEndResponse(
  aSubscription: TMcpSubscription): RawUtf8;
var
  res: variant;
begin
  // the JSON-RPC response to the long-lived request: an empty result that says
  // "this ended on purpose", as opposed to a stream that just stops
  res := _ObjFast(['_meta',
    _ObjFast([MCP_META_SUBSCRIPTION_ID, aSubscription.Id])]);
  result := fProcessor.CreateSuccessResponse(aSubscription.Id, res);
end;

procedure TMcpServer.AddCacheHints(var aResult: variant; aMethod: TMcpMethod;
  aPersonalized: boolean);
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
  if aPersonalized then
  begin
    // A resources/read whose answer was shaped by this caller's input responses
    // is by definition not the same answer for the next caller. Handing a proxy
    // `public` on it is exactly the cross-authorization-context replay the spec
    // warns about, so a retry never carries a shareable or reusable hint.
    ttl := 0;
    scope := mcsPrivate;
  end;
  doc := _Safe(aResult);
  if not doc^.IsObject then
    exit; // FinalizeResult rejects that anyway, with a better message
  doc^.AddOrUpdateValue('ttlMs', ttl);
  doc^.AddOrUpdateValue('cacheScope', RawUtf8ToVariant(MCP_CACHE_SCOPE[scope]));
end;

destructor TMcpServer.Destroy;
var
  i: PtrInt;
begin
  Stop;
  // Release any stream still registered. Reaching this with a non-empty list
  // means a transport was not stopped first — its worker would then drain a
  // queue belonging to a freed server, so callers MUST free the transport
  // before the server (as the demos and tests do). Cancel first, so a thread
  // that is between two turns leaves its loop instead of touching the object.
  fSubscriptionSafe.Lock;
  try
    for i := 0 to high(fSubscriptions) do
    begin
      fSubscriptions[i].Cancel;
      fSubscriptions[i].Free;
    end;
    fSubscriptions := nil;
  finally
    fSubscriptionSafe.UnLock;
  end;
  fTools := nil;
  fResources := nil;
  fProcessor.Free;
  fSubscriptionSafe.Done;
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
  // outside the registry lock: the fan-out takes a different lock, and telling
  // subscribers is what makes the advertised listChanged capability true
  NotifyToolsListChanged;
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
  NotifyResourcesListChanged;
end;

function TMcpServer.UnregisterTool(const aName: RawUtf8): boolean;
begin
  fSafe.Lock;
  try
    result := fTools.Remove(aName);
  finally
    fSafe.UnLock;
  end;
  if result then
    NotifyToolsListChanged;
end;

function TMcpServer.UnregisterResource(const aUri: RawUtf8): boolean;
begin
  fSafe.Lock;
  try
    result := fResources.Remove(aUri);
  finally
    fSafe.UnLock;
  end;
  if result then
    NotifyResourcesListChanged;
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

function TMcpServer.CallContext(const aParams: variant; const aMethod: RawUtf8;
  const aAuthCtx: TMcpAuthContext): TMcpCallContext;
var
  doc, meta: PDocVariantData;
  i: PtrInt;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.Method := aMethod;
  result.Auth := aAuthCtx;
  doc := _Safe(aParams);
  // inputResponses and requestState sit directly in params, NOT in _meta:
  // _meta is the protocol's own envelope, these two are request payload.
  // Both are typed on the wire, and a handler must not have to defend against
  // a client that sends something else: an InputResponses is an object, and a
  // requestState is the string this server handed out — accepting a number
  // here would silently stringify it and hand the handler a state it never
  // issued. Anything else is Invalid params, before any handler runs.
  i := doc^.GetValueIndex(MCP_PARAM_INPUT_RESPONSES);
  if i >= 0 then
  begin
    if not _Safe(doc^.Values[i])^.IsObject then
      raise EMcpInvalidParams.CreateUtf8('% must be a JSON object',
        [MCP_PARAM_INPUT_RESPONSES]);
    result.InputResponses := doc^.Values[i];
    result.HasInputResponses := true;
  end;
  i := doc^.GetValueIndex(MCP_PARAM_REQUEST_STATE);
  if i >= 0 then
  begin
    if not VarIsString(doc^.Values[i]) then
      raise EMcpInvalidParams.CreateUtf8('% must be a string',
        [MCP_PARAM_REQUEST_STATE]);
    VariantToUtf8(doc^.Values[i], result.RequestState);
    result.HasRequestState := true;
  end;
  if doc^.GetAsDocVariant('_meta', meta) then
    result.ClientCapabilities :=
      meta^.GetValueOrDefault(MCP_META_CLIENT_CAPABILITIES, Null);
end;

procedure TMcpServer.ValidateInputRequests(const aInputRequests: variant;
  const aRequestState: RawUtf8; aMethod: TMcpMethod;
  const aClientCapabilities: variant);
var
  requests, entry, caps: PDocVariantData;
  i, k: PtrInt;
  m: RawUtf8;
begin
  // "Servers MUST NOT send InputRequiredResult responses on any other client
  // requests" than prompts/get, resources/read and tools/call. Only two of
  // those exist here; asking for input from tools/list would produce a result
  // no conforming client would act on.
  if not (aMethod in [mcpToolsCall, mcpResourcesRead]) then
    raise EMcpException.CreateU('An InputRequiredResult is only allowed on ' +
      'tools/call and resources/read');

  requests := _Safe(aInputRequests);
  // "Servers MUST include at least one of inputRequests or requestState in
  // every InputRequiredResult" — otherwise the client learns nothing and can
  // only retry the identical request, forever.
  if (aRequestState = '') and
     not (requests^.IsObject and (requests^.Count > 0)) then
    raise EMcpException.CreateU('An InputRequiredResult needs at least one of ' +
      'inputRequests or requestState');
  if not requests^.IsObject then
  begin
    // Only a genuinely absent value means "requestState-only". A present but
    // wrong-shaped one (an array, a string) would be dropped by
    // InputRequiredResult, and the client would retry without the input the
    // handler is waiting for — an endless round trip instead of a loud defect.
    if not VarIsVoid(aInputRequests) then
      raise EMcpException.CreateU('inputRequests must be a JSON object ' +
        'mapping server-assigned identifiers to request objects');
    exit;
  end;

  caps := _Safe(aClientCapabilities);
  for i := 0 to requests^.Count - 1 do
  begin
    entry := _Safe(requests^.Values[i]);
    if not (entry^.IsObject and entry^.GetAsRawUtf8('method', m)) then
      raise EMcpException.CreateUtf8(
        'inputRequests["%"] must be an object with a method', [requests^.Names[i]]);
    k := 0;
    while (k <= high(MCP_INPUT_METHODS)) and
          (MCP_INPUT_METHODS[k] <> m) do
      inc(k);
    if k > high(MCP_INPUT_METHODS) then
      raise EMcpException.CreateUtf8(
        'inputRequests["%"] asks for %, which is not one of the three allowed ' +
        'server-to-client requests', [requests^.Names[i], m]);
    // "Servers MUST NOT send an inputRequests that the client has not declared
    // support for in its capabilities." Enforced here rather than trusting the
    // handler: the capabilities are protocol state, and this is the last point
    // where the violation can still be turned into the error the spec reserved
    // for it instead of a response the client cannot answer.
    if caps^.GetValueIndex(MCP_INPUT_CAPABILITIES[k]) < 0 then
      raise EMcpInputCapabilityMissing.CreateCapability(MCP_INPUT_CAPABILITIES[k], m);
  end;
end;

function TMcpServer.CapabilityError(const aRequestId: variant;
  aError: EMcpInputCapabilityMissing): RawUtf8;
var
  caps: TDocVariantData;
begin
  caps.InitArrayFrom(aError.Capabilities, JSON_FAST);
  // "the server MUST return a MissingRequiredClientCapabilityError (-32021)
  // whose data.requiredCapabilities lists the missing capabilities". The
  // message alone is not machine-readable, and a client that cannot tell WHICH
  // capability to add can only fail the same way on every retry.
  result := fProcessor.CreateError(aRequestId,
    MCP_ERROR_MISSING_CLIENT_CAPABILITY, StringToUtf8(aError.Message),
    _ObjFast(['requiredCapabilities', variant(caps)]));
end;

function TMcpServer.InputRequiredResult(const aInputRequests: variant;
  const aRequestState: RawUtf8): variant;
var
  doc: TDocVariantData;
begin
  // Both fields are optional individually (ValidateInputRequests has already
  // established that at least one is there), so each is emitted only when set:
  // an empty `inputRequests: {}` would tell a client to gather nothing and
  // retry, which is not what a handler that only passes state along means.
  doc.InitObject([], JSON_FAST);
  if _Safe(aInputRequests)^.IsObject then
    doc.AddValue('inputRequests', aInputRequests);
  if aRequestState <> '' then
    doc.AddValue(MCP_PARAM_REQUEST_STATE, RawUtf8ToVariant(aRequestState));
  result := variant(doc);
end;

function TMcpServer.ExecuteToolCall(const aParams: variant;
  const aContext: TMcpCallContext): variant;
var
  doc: PDocVariantData;
  toolName: RawUtf8;
  args: variant;
  tool: IMcpTool;
  interactive: IMcpInteractiveTool;
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

  // A tool that opted into Multi Round-Trip Requests gets the full context;
  // every other tool keeps the two-argument call it was written against.
  if Supports(tool, IMcpInteractiveTool, interactive) then
    result := interactive.ExecuteInteractive(args, aContext)
  else
    result := tool.Execute(args, aContext.Auth);
end;

function TMcpServer.ExecuteResourceRead(const aParams: variant;
  const aContext: TMcpCallContext): variant;
var
  doc: PDocVariantData;
  uri, content: RawUtf8;
  resource: IMcpResource;
  interactive: IMcpInteractiveResource;
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

  if Supports(resource, IMcpInteractiveResource, interactive) then
    content := interactive.ReadInteractive(aContext)
  else
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
  aRequestJson: RawUtf8): RawUtf8;
var
  doc, req: TDocVariantData;
  params: PDocVariantData;
  idx: PtrInt;
  res: variant;
  method, resultType: RawUtf8;
  m: TMcpMethod;
  personalized: boolean;
begin
  result := aResponseJson;
  if aResponseJson = '' then
    exit;
  req.InitJson(aRequestJson, JSON_FAST);
  method := req.U['method'];
  m := McpMethodFromName(method);
  // A hook answers the SAME request the dispatcher would have, so it inherits
  // the same rules — including the MRTR ones. Deriving both from the request
  // body (rather than being handed a method name) is what keeps a hook from
  // quietly bypassing them; that has now happened twice in this transport.
  personalized := (m in [mcpToolsCall, mcpResourcesRead]) and
                  req.GetAsDocVariant('params', params) and
                  ((params^.GetValueIndex(MCP_PARAM_INPUT_RESPONSES) >= 0) or
                   (params^.GetValueIndex(MCP_PARAM_REQUEST_STATE) >= 0));

  doc.InitJson(aResponseJson, JSON_FAST);
  idx := doc.GetValueIndex('result');
  if idx < 0 then
    exit; // an error response — nothing to stamp
  res := doc.Values[idx];
  // Respect a resultType the hook set itself: a hook may legitimately answer a
  // Multi Round-Trip Request, and overwriting `input_required` with `complete`
  // would hand the client a "finished" result still carrying inputRequests —
  // which it would never look at, so the round trip would silently stall.
  if not _Safe(res)^.GetAsRawUtf8('resultType', resultType) or
     (resultType <> MCP_RESULT_INPUT_REQUIRED) then
  begin
    resultType := MCP_RESULT_COMPLETE;
    AddCacheHints(res, m, personalized);
  end;
  doc.Values[idx] := fProcessor.FinalizeResult(res, resultType);
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
  callCtx: TMcpCallContext;
  isNotification, personalized: boolean;
  status: integer;
  m: TMcpMethod;
begin
  // Single validation gate, shared with the transports: whatever Preflight
  // rejects never reaches a handler, here or anywhere else. It hands the parsed
  // request back, so the body is parsed exactly once per dispatch.
  if not Preflight(aRequestJson, method, params, requestId, result, status) then
    exit;
  isNotification := VarIsVoid(requestId);
  m := McpMethodFromName(method);

  // Auth context. Identity must be injected by a backend auth resolver before
  // any tool may trust IsAuthenticated/Roles; until then we stay fail-closed.
  // (Previously the transport session id was carried here for correlation —
  // protocol sessions no longer exist, so there is nothing to carry.)
  FillCharFast(authCtx, SizeOf(authCtx), 0);
  authCtx.IsAuthenticated := false;
  personalized := false;

  try
    // INSIDE the try: CallContext type-checks the MRTR retry fields and raises
    // EMcpInvalidParams on a malformed one. Building it before the try would
    // let that escape into the HTTP worker, which has no handler for it.
    callCtx := CallContext(params, method, authCtx);
    // A request carrying MRTR retry fields produced a caller-specific answer.
    // Presence decides, not content — VarIsVoid() considers an EMPTY object
    // void, so testing the value would let `inputResponses: {}` be cached as
    // shareable. Only the two methods that may take part in a round trip
    // count: retry fields elsewhere are meaningless and must not degrade
    // their cacheability.
    personalized := (m in [mcpToolsCall, mcpResourcesRead]) and
                    (callCtx.HasRequestState or callCtx.HasInputResponses);

    // Dispatch to handler — any handler/tool exception is mapped to a JSON-RPC
    // error below, so it never escapes into the HTTP worker.
    case m of
      mcpDiscover:
        resultData := fProcessor.HandleDiscover;
      mcpToolsList:
        resultData := ListTools;
      mcpToolsCall:
        resultData := ExecuteToolCall(params, callCtx);
      mcpResourcesList:
        resultData := ListResources;
      mcpResourcesRead:
        resultData := ExecuteResourceRead(params, callCtx);
      mcpSubscriptionsListen:
        // Only a streaming transport can serve this: it is a long-lived
        // response stream, not a request/response. The Streamable HTTP
        // transport intercepts it before we get here; reaching this point
        // means the caller is stdio or the plain HTTP transport, where the
        // honest answer is "not available", NOT the empty success response
        // this would otherwise fall through to.
        raise EMcpMethodNotFound.CreateU('subscriptions/listen requires a ' +
          'streaming transport and is not available on this one');
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
      AddCacheHints(resultData, m, personalized);
      result := fProcessor.CreateSuccessResponse(requestId, resultData);
    end;

  except
    // A handler asking for more input is not a failure: it is the interim half
    // of a Multi Round-Trip Request. It carries NO caching hints — the spec
    // mandates those only on results with resultType 'complete', and caching an
    // "I need input" answer would make the client re-ask itself forever.
    on E: EMcpInputRequired do
      if isNotification then
        result := ''
      else
        try
          ValidateInputRequests(E.InputRequests, E.RequestState, m,
            callCtx.ClientCapabilities);
          result := fProcessor.CreateSuccessResponse(requestId,
            InputRequiredResult(E.InputRequests, E.RequestState),
            MCP_RESULT_INPUT_REQUIRED);
        except
          on C: EMcpInputCapabilityMissing do
            result := CapabilityError(requestId, C);
          on V: Exception do
            // our own handler built something the spec forbids: that is a
            // server defect, and -32603 is what says so
            result := fProcessor.CreateError(requestId, JSONRPC_INTERNAL_ERROR,
              StringToUtf8(V.Message));
        end;
    // Catch EVERY exception (not just ESynException): tools may raise plain
    // Exception, EConvertError, DB/OS errors. Translate to a JSON-RPC error so
    // the transport stays alive and the client gets a well-formed response.
    on E: EMcpInputCapabilityMissing do
      if isNotification then
        result := ''
      else
        result := CapabilityError(requestId, E);
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
  else if aMethod = 'subscriptions/listen' then
    result := mcpSubscriptionsListen
  else
    result := mcpUnknown;
end;

function McpParseNotificationFilter(const aParams: variant): TMcpNotificationFilter;
var
  filter, uris: PDocVariantData;
  i: PtrInt;
  uri: RawUtf8;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  if not _Safe(aParams)^.GetAsDocVariant('notifications', filter) or
     not filter^.IsObject then
    exit; // no filter at all: a stream that receives nothing, which is legal
  result.ToolsListChanged := filter^.B['toolsListChanged'];
  result.PromptsListChanged := filter^.B['promptsListChanged'];
  result.ResourcesListChanged := filter^.B['resourcesListChanged'];
  if filter^.GetAsDocVariant('resourceSubscriptions', uris) and
     uris^.IsArray then
    for i := 0 to uris^.Count - 1 do
    begin
      uri := VariantToUtf8(uris^.Values[i]);
      if uri <> '' then
        AddRawUtf8(result.ResourceSubscriptions, uri);
    end;
end;


{ ************ TMcpSubscription }

constructor TMcpSubscription.Create(const aId: variant;
  const aFilter: TMcpNotificationFilter);
begin
  inherited Create;
  fSafe.Init;
  fId := aId;
  fFilter := aFilter;
end;

destructor TMcpSubscription.Destroy;
begin
  fPending := nil;
  fSafe.Done;
  inherited;
end;

function TMcpSubscription.GetCancelled: boolean;
begin
  fSafe.Lock;
  try
    result := fCancelled;
  finally
    fSafe.UnLock;
  end;
end;

procedure TMcpSubscription.Push(const aJson: RawUtf8);
begin
  if aJson = '' then
    exit;
  fSafe.Lock;
  try
    if fCancelled then
      exit; // do not grow a queue nobody will drain
    if fPendingCount >= MCP_SUBSCRIPTION_MAX_PENDING then
    begin
      // Backpressure: the reader is not keeping up (a stalled client, or a
      // burst of resource updates). Without a bound this queue would grow
      // until the process dies, and MaxSubscriptions only caps how MANY
      // queues exist, not how large one gets. Drop the stream instead.
      fCancelled := true;
      if fCancelReason = '' then
        fCancelReason := 'the client was not draining this subscription';
      fPending := nil;
      fPendingCount := 0;
      exit;
    end;
    if fPendingCount = length(fPending) then
      SetLength(fPending, NextGrow(fPendingCount));
    fPending[fPendingCount] := aJson;
    inc(fPendingCount);
  finally
    fSafe.UnLock;
  end;
end;

function TMcpSubscription.Drain(out aJson: TRawUtf8DynArray): boolean;
begin
  fSafe.Lock;
  try
    result := fPendingCount > 0;
    if not result then
      exit;
    SetLength(fPending, fPendingCount); // hand over exactly what is queued
    aJson := fPending;
    fPending := nil;
    fPendingCount := 0;
  finally
    fSafe.UnLock;
  end;
end;

function TMcpSubscription.WatchesResource(const aUri: RawUtf8): boolean;
begin
  // the filter is immutable after Create, so this needs no lock
  result := FindRawUtf8(fFilter.ResourceSubscriptions, aUri) >= 0;
end;

procedure TMcpSubscription.Cancel(const aReason: RawUtf8);
begin
  fSafe.Lock;
  try
    // keep the FIRST reason: it is the one that actually ended the stream, and
    // a later blanket Cancel (shutdown sweeping up everything) would otherwise
    // overwrite the specific cause with a generic one
    if not fCancelled then
      fCancelReason := aReason;
    fCancelled := true;
  finally
    fSafe.UnLock;
  end;
end;

function TMcpSubscription.CancelReason: RawUtf8;
begin
  fSafe.Lock;
  try
    result := fCancelReason;
  finally
    fSafe.UnLock;
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
