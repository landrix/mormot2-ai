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

  /// where a client asks to continue a paginated list
  // - "The cursor is an opaque string token, representing a position in the
  //   result set"; an ABSENT cursor starts at the beginning
  MCP_PARAM_CURSOR = 'cursor';
  /// where the server says a further page exists
  // - omitted entirely when the list is exhausted: "Clients SHOULD treat a
  //   missing nextCursor as the end of results"
  MCP_RESULT_NEXT_CURSOR = 'nextCursor';
  /// hard ceiling on completion/complete suggestions
  // - "Maximum 100 items per response"; the server truncates to this and sets
  //   `hasMore`, so an over-eager implementation cannot break the contract
  MCP_COMPLETION_MAX_VALUES = 100;

  /// how many entries one page of a list carries by default
  // - "Page size is determined by the server, and clients MUST NOT assume a
  //   fixed page size", so this is ours to pick and ours to change
  MCP_DEFAULT_PAGE_SIZE = 100;

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
  /// no token, or one this server cannot accept
  // - spec table: "401 Unauthorized — Authorization required or token invalid"
  HTTP_MCP_UNAUTHORIZED = 401;
  /// a valid token that does not carry the scopes this operation needs
  // - spec table: "403 Forbidden — Invalid scopes or insufficient permissions"
  HTTP_MCP_FORBIDDEN = 403;

  /// the well-known location of the Protected Resource Metadata document
  // - "MCP servers MUST implement OAuth 2.0 Protected Resource Metadata
  //   (RFC9728)"; it is how a client finds the authorization server at all
  MCP_WELL_KNOWN_RESOURCE = '/.well-known/oauth-protected-resource';

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
    /// OAuth scopes the presented access token actually carries
    // - what the token was GRANTED, not what the caller is; a scope is an
    //   upper bound on the delegation, never a role. Empty on an unauthorized
    //   request, and empty is what a scope check must fail closed on.
    Scopes: TRawUtf8DynArray;
    /// when the presented token stops being valid, as Unix seconds (0 = never)
    // - a subscriptions/listen stream can stay open for far longer than a token
    //   lives, and the one check at connect time would then keep feeding a
    //   caller whose authorization has since lapsed. The stream watches this and
    //   ends itself; a verifier that leaves it 0 opts out of that.
    ExpiresUnix: Int64;
    /// the issuer that minted the token, as established by the verifier
    // - recorded for auditing: two deployments may share a principal name
    //   while trusting different issuers, and a log line without this cannot
    //   tell them apart
    Issuer: RawUtf8;
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

  /// why a presented access token was refused
  // - the distinction drives the HTTP status: everything here except
  //   mtrInsufficientScope is a 401, that one is a 403
  TMcpTokenResult = (
    /// the token verified, and this server is among its intended audiences
    mtrValid,
    /// no Authorization header, or not a Bearer one
    mtrMissing,
    /// signature, format or issuer did not check out
    mtrInvalid,
    /// well-formed and authentic, but past its expiry
    mtrExpired,
    /// authentic, but minted for a DIFFERENT resource
    // - "MCP servers MUST only accept tokens specifically intended for
    //   themselves and MUST reject tokens that do not include them in the
    //   audience claim". A 401, not a 403: the token is not merely too weak
    //   here, it does not belong to this server at all.
    mtrWrongAudience,
    /// authentic and ours, but lacking a scope the operation needs
    mtrInsufficientScope);

  /// verifies the bearer tokens presented to this MCP server
  // - the ONE place a deployment plugs its identity into the protocol layer:
  //   this unit deliberately knows no keys, no JWKS and no user store, so it
  //   can stay generic while the backend decides what a valid token is
  // - "MCP servers MUST validate access tokens before processing the request"
  //   — the transport calls this ahead of any dispatch, never after
  IMcpTokenVerifier = interface(IInvokable)
    ['{2F8B5C1A-7D34-4E96-A0B2-9C5E1D7A3F60}']
    /// check one bearer token and, on success, fill in who is calling
    // - aResource is this server's canonical URI: the verifier MUST confirm the
    //   token was issued for it and return mtrWrongAudience otherwise, which is
    //   what stops a token stolen from another service from working here
    // - aAuthCtx is only meaningful when the result is mtrValid
    function VerifyToken(const aToken, aResource: RawUtf8;
      out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
  end;

const
  /// the OAuth error code that goes with each refusal, in enum order
  // - RFC 6750 defines exactly three: invalid_request, invalid_token and
  //   insufficient_scope. An expired or wrong-audience token is `invalid_token`
  //   — the client cannot fix either by asking for more scope, it needs a new
  //   token, which is what that code tells it to do.
  MCP_TOKEN_ERROR: array[TMcpTokenResult] of RawUtf8 = (
    '',
    'invalid_request',
    'invalid_token',
    'invalid_token',
    'invalid_token',
    'insufficient_scope');

type

  /// MCP-specific exception class
  EMcpException = class(ESynException);

  /// raised when a JSON-RPC method is not implemented (mapped to -32601)
  EMcpMethodNotFound = class(EMcpException);

  /// raised when the params of a request are unusable (mapped to -32602)
  // - this covers "the named thing does not exist": since 2026-07-28 a missing
  //   resource is NOT its own error code anymore (the former -32002 was
  //   removed), it is a plain Invalid params
  EMcpInvalidParams = class(EMcpException);

  /// raised by a handler whose caller lacks the OAuth scope it requires
  // - the only way a per-operation scope decision can reach the transport: the
  //   token was checked before the body was read, so which scope this call
  //   needs is not yet knowable there. The transport turns this into the 403
  //   plus WWW-Authenticate challenge the spec prescribes, and the named scopes
  //   are what the client asks for in its step-up authorization.
  // - name EVERY scope the operation needs at once: "Challenging incrementally
  //   (returning one missing scope, then another on the subsequent retry)
  //   forces multiple authorization round-trips for a single operation"
  EMcpInsufficientScope = class(EMcpException)
  protected
    fScope: RawUtf8;
  public
    /// aScope is the space-separated set the operation requires
    constructor CreateScope(const aScope: RawUtf8); reintroduce;
    /// the scopes to put in the challenge, space-separated per RFC 6750
    property Scope: RawUtf8
      read fScope;
  end;

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
    mcpResourcesTemplatesList,
    mcpPromptsList,
    mcpPromptsGet,
    mcpCompletionComplete,
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

  /// MCP Prompt interface — a server-defined message template a USER picks
  // - "Prompts are designed to be user-controlled … This refers to who decides
  //   when the prompt is used, not who authors its content". Unlike a tool, a
  //   prompt is not called by the model on its own initiative: it is offered to
  //   the person, typically as a slash command.
  IMcpPrompt = interface(IInvokable)
    ['{3E7B9D14-5C82-4A6F-B0D3-7A1E4F8C2B95}']
    /// the unique identifier clients call it by
    function GetName: RawUtf8;
    /// optional human-readable name for display ('' to omit)
    function GetTitle: RawUtf8;
    /// optional human-readable description ('' to omit)
    function GetDescription: RawUtf8;
    /// the `arguments` array published in prompts/list, or void for none
    // - each entry is {name, description?, required?}; the completion API can
    //   auto-complete these, so the names are part of the public contract
    function GetArguments: variant;
    /// render the prompt into `messages`
    // - aArgs is the client's `arguments` object (void when it sent none)
    // - return either a full result ({description?, messages:[…]}) or just the
    //   messages array — the server wraps a bare array for convenience
    // - raise EMcpInvalidParams for a missing required argument: "Missing
    //   required arguments: -32602"
    function Render(const Args: variant;
      const AuthCtx: TMcpAuthContext): variant;
  end;

  /// a parameterized resource, published as an RFC 6570 URI template
  // - the template itself is never read: it tells a client which URIs it may
  //   construct, and the resulting concrete URI goes to resources/read
  // - LIMITATION, know this before publishing one: resources/read resolves a
  //   URI by EXACT lookup in the resource registry. Nothing expands a template
  //   or matches a concrete URI back against it, so a client that dutifully
  //   builds `file:///src/main.pas` from `file:///{path}` gets -32602 unless
  //   that exact URI is also registered. Publish a template only when the URIs
  //   it describes really exist as registered resources; a matcher/resolver
  //   hook is the missing piece and is not built yet.
  IMcpResourceTemplate = interface(IInvokable)
    ['{6D3F8A21-4E95-4C7B-9F16-8B2D5A0E3C74}']
    /// the RFC 6570 template, e.g. 'file:///{path}' — also the registry key
    function GetUriTemplate: RawUtf8;
    /// the name clients show for the family of resources
    function GetName: RawUtf8;
    /// optional display name ('' to omit)
    function GetTitle: RawUtf8;
    /// optional description ('' to omit)
    function GetDescription: RawUtf8;
    /// optional MIME type shared by the resources it produces ('' to omit)
    function GetMimeType: RawUtf8;
  end;

  /// something whose arguments completion/complete can suggest values for
  // - optional add-on to IMcpPrompt or IMcpResourceTemplate: the server offers
  //   completion for exactly those that implement it, so a prompt with free-text
  //   arguments needs no extra code
  // - aContext carries the arguments the user already filled in, so a suggestion
  //   can depend on an earlier choice ("framework" after "language")
  IMcpCompletable = interface(IInvokable)
    ['{8C5A2E76-1B49-4D03-A7F8-3E6C9B14D5A2}']
    /// suggest values for one argument, ranked by relevance
    // - aValue is what the user typed so far (possibly empty)
    // - return at most 100 entries; the server truncates beyond that and sets
    //   `hasMore`, so an over-eager implementation cannot break the response
    // - AuthCtx identifies the caller: suggestions ARE data, and a shared
    //   server must be able to offer a user only what that user may see. A
    //   completion that ignores it leaks the shape of everything it knows —
    //   file names, record ids — to anyone who can type a prefix.
    function Complete(const ArgumentName, ArgumentValue: RawUtf8;
      const Context: variant; const AuthCtx: TMcpAuthContext): TRawUtf8DynArray;
  end;

  /// a prompt that takes part in Multi Round-Trip Requests
  // - prompts/get is the third method allowed to answer with an
  //   InputRequiredResult ("Servers MAY also respond to prompts/get with an
  //   InputRequiredResult"), e.g. to elicit an argument it cannot guess
  IMcpInteractivePrompt = interface(IMcpPrompt)
    ['{9A2D6E38-7F41-4B5C-8E0A-2D6B9F3C7A18}']
    /// render with the full request context, including any input responses
    function RenderInteractive(const Args: variant;
      const Context: TMcpCallContext): variant;
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

/// where the Protected Resource Metadata of that resource identifier lives
// - RFC 9728 INSERTS the well-known segment between host and path, it does not
//   append it: `https://example.com/public/mcp` publishes its metadata at
//   `https://example.com/.well-known/oauth-protected-resource/public/mcp`.
//   Appending would send every client to a 404 — and a client that cannot read
//   the metadata cannot find the authorization server at all.
// - a resource without a path keeps the plain root form
function McpResourceMetadataUrl(const aResource: RawUtf8): RawUtf8;

/// the path part of the well-known URL, i.e. what a transport must route
function McpResourceMetadataPath(const aResource: RawUtf8): RawUtf8;

/// does a granted scope set satisfy the scope an operation requires?
// - "Servers MUST account for scope hierarchies, where a broader scope implies
//   narrower ones": `files` covers `files:read`, and `a:b` covers `a:b:c`. The
//   separator is ':', the convention OAuth deployments use for hierarchy.
// - the implication runs one way only: holding `files:read` does NOT grant
//   `files`. Reading that backwards would turn every narrow grant into a broad
//   one, which is the whole point of scoping something narrowly.
// - an empty granted set satisfies nothing: a request without a token must fail
//   every check, not pass the ones nobody thought to guard
// - the ':' hierarchy is a CONVENTION, not something OAuth defines: scope
//   values are implementation-defined, and a deployment where `admin` and
//   `admin:delete` are unrelated permissions would see the first silently grant
//   the second. Pass aHierarchical=false there and require exact matches.
// - this is a helper for handlers to call, never a policy the server applies
//   behind your back: nothing in this unit checks scopes on its own
function McpScopeSatisfied(const aGranted: TRawUtf8DynArray;
  const aRequired: RawUtf8; aHierarchical: boolean = true): boolean;

/// extract the token of an `Authorization: Bearer …` header value
// - returns '' when the header is absent, empty, or not the Bearer scheme;
//   the scheme name is case-insensitive per RFC 7235, the token is not
function McpBearerToken(const aAuthorizationHeader: RawUtf8): RawUtf8;

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

/// wrap a list position into the opaque cursor a client gets handed
// - the position is the NAME of the last entry delivered, not its index: a
//   keyset cursor still points at the right place after entries are registered
//   or removed, which is what "Servers SHOULD provide stable cursors" asks for.
//   An index would silently skip or repeat entries across such a change.
// - base64uri is not obfuscation, it is a fence: a cursor that looks like a
//   plain name invites clients to build one themselves, and the spec forbids
//   exactly that ("Don't attempt to parse or modify cursors")
function McpEncodeCursor(const aAfterName: RawUtf8): RawUtf8;

/// unwrap a cursor back into the list position it names
// - returns false when the token is not one we minted, which the caller turns
//   into -32602 ("Invalid cursors SHOULD result in an error with code -32602")
// - an EMPTY cursor string decodes to an empty position and is VALID: the spec
//   is explicit that "an empty string is a valid cursor and thus MUST NOT be
//   treated as the end of results". It simply starts from the beginning.
function McpDecodeCursor(const aCursor: RawUtf8; out aAfterName: RawUtf8): boolean;

/// read the pagination cursor out of a request's params
// - returns whether a cursor was SUPPLIED, which is not the same as whether it
//   is non-empty: `cursor: ""` is a valid cursor per the spec, so presence has
//   to be tested on the key, never on the value
function McpRequestCursor(const aParams: variant; out aCursor: RawUtf8): boolean;

/// cut one page out of a SORTED name list, starting after the cursor position
// - aNames MUST be sorted: a cursor over an unordered set (a dictionary
//   enumeration, say) would repeat some entries and skip others as soon as the
//   registry changes between two pages
// - returns the index range [aFirst, aLast] to emit, and aNextCursor as the
//   token for the following page ('' when this page is the last one)
// - raises EMcpInvalidParams on a malformed cursor
procedure McpPageRange(const aNames: TRawUtf8DynArray; const aCursor: RawUtf8;
  aHasCursor: boolean; aPageSize: integer;
  out aFirst, aLast: PtrInt; out aNextCursor: RawUtf8);


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
    fPrompts: IKeyValue<RawUtf8, IMcpPrompt>; // name -> IMcpPrompt
    fTemplates: IKeyValue<RawUtf8, IMcpResourceTemplate>; // uriTemplate -> impl
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
    fListPageSize: integer;
    fTokenVerifier: IMcpTokenVerifier;
    fAuthResource: RawUtf8;
    fAuthorizationServers: TRawUtf8DynArray;
    fScopesSupported: TRawUtf8DynArray;
    procedure SetScopesSupported(const aScopes: TRawUtf8DynArray);
    procedure SetTokenVerifier(const aVerifier: IMcpTokenVerifier);
    procedure SetAuthResource(const aResource: RawUtf8);
    /// refuse an authorization setting once requests can already be running
    procedure CheckNotStarted(const aWhat: RawUtf8);
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
    function ListTools(const aParams: variant): variant;
    function ListResources(const aParams: variant): variant;
    function ListPrompts(const aParams: variant): variant;
    function ListResourceTemplates(const aParams: variant): variant;
    function GetPrompt(const aParams: variant;
      const aContext: TMcpCallContext): variant;
    function CompleteArgument(const aParams: variant;
      const aAuthCtx: TMcpAuthContext): variant;
    /// sorted key snapshot of a registry, taken under the lock
    function SortedToolNames: TRawUtf8DynArray;
    function SortedResourceUris: TRawUtf8DynArray;
    function SortedPromptNames: TRawUtf8DynArray;
    function SortedTemplateUris: TRawUtf8DynArray;
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
    /// register a prompt template
    // - thread-safe
    procedure RegisterPrompt(const aPrompt: IMcpPrompt);
    /// unregister a prompt by name
    function UnregisterPrompt(const aName: RawUtf8): boolean;
    /// register a parameterized resource (RFC 6570 URI template)
    // - thread-safe; broadcasts resources/list_changed, since a template is
    //   part of what the resource surface offers
    procedure RegisterResourceTemplate(const aTemplate: IMcpResourceTemplate);
    /// unregister a resource template by its URI template
    function UnregisterResourceTemplate(const aUriTemplate: RawUtf8): boolean;
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
    function ExecuteRequest(const aRequestJson: RawUtf8): RawUtf8; overload;
    /// dispatch on behalf of an already-authenticated caller
    // - the transport verified the token; without handing the resulting context
    //   down, every tool would see an unauthenticated caller no matter what was
    //   presented, and Scopes/Roles/Issuer would exist but never arrive
    function ExecuteRequest(const aRequestJson: RawUtf8;
      const aAuthCtx: TMcpAuthContext): RawUtf8; overload;
    /// dispatch, reporting a per-operation scope refusal back to the transport
    // - a handler raising EMcpInsufficientScope cannot be answered with a
    //   JSON-RPC error: the spec wants HTTP 403 plus a WWW-Authenticate naming
    //   the missing scopes, and only the transport can send those. Returning it
    //   explicitly beats letting the exception escape into the HTTP worker.
    // - aScopeChallenge is non-empty exactly when the caller must answer 403
    function ExecuteRequest(const aRequestJson: RawUtf8;
      const aAuthCtx: TMcpAuthContext; out aScopeChallenge: RawUtf8): RawUtf8;
      overload;
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
    /// tell subscribers the prompt list changed
    // - only streams that opted into promptsListChanged receive it
    procedure NotifyPromptsListChanged;
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
    /// whether this server refuses unauthenticated requests
    // - simply "a verifier is plugged in": authorization is OPTIONAL in MCP,
    //   and every auth path below is a no-op while this is false
    function IsProtected: boolean;
    /// the RFC 9728 Protected Resource Metadata document, as JSON
    // - "MCP servers MUST implement OAuth 2.0 Protected Resource Metadata";
    //   it is served unauthenticated at MCP_WELL_KNOWN_RESOURCE, since a client
    //   reads it precisely because it does not have a token yet
    function ProtectedResourceMetadata: RawUtf8;
    /// the WWW-Authenticate value for a refusal, per RFC 6750
    // - aScope names what the operation needed, and the spec asks for all of it
    //   in ONE challenge: dribbling out one missing scope at a time costs a
    //   full authorization round trip per scope
    function AuthChallenge(aResult: TMcpTokenResult;
      const aScope: RawUtf8 = ''): RawUtf8;
    /// verify one Authorization header before anything is dispatched
    // - "MCP servers MUST validate access tokens before processing the
    //   request": a transport calls this alongside the preflight, never after
    // - returns mtrValid (and a filled aAuthCtx) on an open server too, so a
    //   caller does not need to special-case protection being off
    function Authorize(const aAuthorizationHeader: RawUtf8;
      out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
    /// same, for a transport that already parsed the Bearer scheme itself
    // - mORMot's HTTP server extracts the token into Ctxt.AuthBearer, so
    //   re-splitting the header there would only add a second parser to keep
    //   in step with the first
    function AuthorizeToken(const aToken: RawUtf8;
      out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
    /// the HTTP status that refusal reason must be answered with
    function AuthHttpStatus(aResult: TMcpTokenResult): integer;
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
    /// how many entries one page of tools/list, resources/list or prompts/list
    /// carries (default MCP_DEFAULT_PAGE_SIZE; 0 disables paging entirely)
    // - the spec leaves page size to the server and forbids clients to assume
    //   one, so this can be tuned per deployment without breaking anyone
    // - 0 means "one page, however long": honest for a small static registry,
    //   and it keeps `nextCursor` out of the response altogether
    property ListPageSize: integer
      read fListPageSize write fListPageSize;
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
    /// plug in the deployment's token verification to protect this server
    // - authorization is OPTIONAL in MCP: leaving this nil keeps the server
    //   open, which is the right default for stdio (the spec says stdio SHOULD
    //   NOT use this at all and take credentials from the environment) and for
    //   a loopback demo. Setting it is what turns protection on — one switch,
    //   so "is this server protected?" has a single, readable answer.
    // - settable only BEFORE Start: swapping a verifier under live requests is
    //   a data race no lock can make meaningful (half the requests would run
    //   against each), and an interface read concurrent with a write can even
    //   see a refcount already dropped to zero. Auth is startup configuration.
    property TokenVerifier: IMcpTokenVerifier
      read fTokenVerifier write SetTokenVerifier;
    /// this server's canonical URI, i.e. the audience tokens must be minted for
    // - RFC 8707/9728 resource identifier, e.g. 'https://mcp.example.com/mcp':
    //   absolute, no fragment, and conventionally no trailing slash
    // - REQUIRED once TokenVerifier is set: without it there is nothing to
    //   check an audience against, and a server that cannot tell its own tokens
    //   from another service's is exactly what the audience rule exists for
    // - settable only BEFORE Start, for the same reason as TokenVerifier: a
    //   resource changed mid-flight would validate audiences against one value
    //   while the published metadata still names another
    property AuthResource: RawUtf8
      read fAuthResource write SetAuthResource;
    /// issuer URLs of the authorization servers a client may obtain a token from
    // - published in the Protected Resource Metadata; this is how a client
    //   discovers where to authenticate at all
    property AuthorizationServers: TRawUtf8DynArray
      read fAuthorizationServers write fAuthorizationServers;
    /// the scopes advertised as needed for basic functionality
    // - "intended to represent the minimal set of scopes necessary for basic
    //   functionality", with anything further requested through a step-up
    // - `offline_access` is rejected here: refresh tokens are a client concern
    //   and the spec says a protected resource SHOULD NOT ask for it
    property ScopesSupported: TRawUtf8DynArray
      read fScopesSupported write SetScopesSupported;
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

constructor EMcpInsufficientScope.CreateScope(const aScope: RawUtf8);
begin
  CreateUtf8('This operation requires the scope(s): %', [aScope]);
  fScope := aScope;
end;

constructor EMcpInputCapabilityMissing.CreateCapability(
  const aCapability, aForMethod: RawUtf8);
begin
  CreateUtf8('The client did not declare the "%" capability that % requires',
    [aCapability, aForMethod]);
  AddRawUtf8(fCapabilities, aCapability);
end;

function McpResourceMetadataPath(const aResource: RawUtf8): RawUtf8;
var
  p: PUtf8Char;
  slash: PtrInt;
begin
  result := MCP_WELL_KNOWN_RESOURCE;
  // find the '/' that ends the authority: skip 'scheme://' first, or a host
  // containing no slash at all would make us treat the scheme's own '//' as
  // the path
  p := pointer(aResource);
  if p = nil then
    exit;
  slash := PosEx('://', aResource);
  if slash = 0 then
    exit; // not an absolute URI: nothing to split, keep the root form
  slash := PosEx('/', aResource, slash + 3);
  if slash = 0 then
    exit; // no path component — the metadata lives at the root
  // insert, do NOT append: the path of the resource becomes a SUFFIX of the
  // well-known path (RFC 9728 §3.1, same construction as RFC 8414)
  result := MCP_WELL_KNOWN_RESOURCE + copy(aResource, slash, maxInt);
  // a trailing slash on the resource would produce a doubled one here
  while (length(result) > 1) and
        (result[length(result)] = '/') do
    SetLength(result, length(result) - 1);
end;

function McpResourceMetadataUrl(const aResource: RawUtf8): RawUtf8;
var
  slash: PtrInt;
begin
  result := aResource;
  if result = '' then
    exit;
  slash := PosEx('://', aResource);
  if slash = 0 then
    exit(aResource + MCP_WELL_KNOWN_RESOURCE);
  slash := PosEx('/', aResource, slash + 3);
  if slash = 0 then
    result := aResource // scheme://host, no path
  else
    result := copy(aResource, 1, slash - 1); // strip the path, keep the origin
  result := result + McpResourceMetadataPath(aResource);
end;

function McpScopeSatisfied(const aGranted: TRawUtf8DynArray;
  const aRequired: RawUtf8; aHierarchical: boolean): boolean;
var
  i, n: PtrInt;
begin
  result := true;
  if aRequired = '' then
    exit; // the operation asks for nothing
  result := false;
  for i := 0 to high(aGranted) do
  begin
    n := length(aGranted[i]);
    if n = 0 then
      continue;
    if aGranted[i] = aRequired then
      exit(true);
    // a broader scope implies the narrower ones BELOW it: the granted string
    // must be a prefix of the required one AND end exactly on a ':' boundary,
    // or `file` would silently cover `files:write`
    if aHierarchical and
       (length(aRequired) > n) and
       (aRequired[n + 1] = ':') and
       CompareMem(pointer(aGranted[i]), pointer(aRequired), n) then
      exit(true);
  end;
end;

function McpBearerToken(const aAuthorizationHeader: RawUtf8): RawUtf8;
var
  p: PUtf8Char;
begin
  result := '';
  p := pointer(aAuthorizationHeader);
  if p = nil then
    exit;
  while p^ = ' ' do
    inc(p);
  // RFC 7235: the scheme is case-insensitive ("bearer" is as valid as "Bearer")
  if not IdemPChar(p, 'BEARER ') then
    exit;
  inc(p, 7);
  while p^ = ' ' do
    inc(p);
  FastSetString(result, p, StrLen(p));
  // a lone "Bearer" with no token is a missing token, not an empty one
  TrimSelf(result);
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

function McpRequestCursor(const aParams: variant; out aCursor: RawUtf8): boolean;
var
  doc: PDocVariantData;
  i: PtrInt;
begin
  aCursor := '';
  result := false;
  doc := _Safe(aParams);
  if not doc^.IsObject then
    exit;
  // GetValueIndex, not a value test: `cursor: ""` IS a cursor ("an empty string
  // is a valid cursor and thus MUST NOT be treated as the end of results"), and
  // asking VarIsVoid or comparing to '' would collapse it into "absent".
  i := doc^.GetValueIndex(MCP_PARAM_CURSOR);
  if i < 0 then
    exit;
  result := true;
  VariantToUtf8(doc^.Values[i], aCursor);
end;

const
  /// prefix inside the encoded cursor, so a token from somewhere else (or a
  /// hand-built one) fails to decode instead of landing on a plausible position
  MCP_CURSOR_MARK = 'n:';

function McpEncodeCursor(const aAfterName: RawUtf8): RawUtf8;
begin
  result := BinToBase64uri(MCP_CURSOR_MARK + aAfterName);
end;

function McpDecodeCursor(const aCursor: RawUtf8; out aAfterName: RawUtf8): boolean;
var
  plain: RawByteString;
begin
  aAfterName := '';
  // "an empty string is a valid cursor": it names no position, so the page
  // starts at the beginning — the same place an absent cursor starts.
  if aCursor = '' then
    exit(true);
  plain := Base64uriToBin(aCursor);
  // Base64uriToBin returns '' on anything it cannot decode; combined with the
  // marker check below, a forged or truncated token is rejected rather than
  // quietly restarting the list from the top (which would loop a client
  // forever over page one).
  // Exact comparison, NOT IdemPChar: that one is case-insensitive and needs its
  // pattern in uppercase, so a lowercase marker would never match — and a
  // marker that tolerates case is a weaker fence than one that does not.
  result := (length(plain) >= length(MCP_CURSOR_MARK)) and
            CompareMemFixed(pointer(plain), PAnsiChar(MCP_CURSOR_MARK),
              length(MCP_CURSOR_MARK));
  if not result then
    exit;
  aAfterName := copy(plain, length(MCP_CURSOR_MARK) + 1, maxInt);
  // The keyset comparison is StrComp, which stops at the first #0. A cursor
  // carrying an embedded null would therefore be accepted here but compared
  // only up to that byte — a "valid" token silently producing the wrong window.
  // Names never contain #0, so such a cursor is not one of ours.
  if PosExChar(#0, aAfterName) <> 0 then
  begin
    aAfterName := '';
    result := false;
  end;
end;

procedure McpPageRange(const aNames: TRawUtf8DynArray; const aCursor: RawUtf8;
  aHasCursor: boolean; aPageSize: integer;
  out aFirst, aLast: PtrInt; out aNextCursor: RawUtf8);
var
  after: RawUtf8;
  n: PtrInt;
begin
  aNextCursor := '';
  n := length(aNames);
  aFirst := 0;
  aLast := n - 1;
  if aHasCursor then
  begin
    if not McpDecodeCursor(aCursor, after) then
      raise EMcpInvalidParams.CreateU('invalid cursor');
    // An empty position names nothing, so it skips nothing — this is what makes
    // `cursor: ""` genuinely equivalent to sending no cursor at all. Without the
    // guard, StrComp(nil, nil) returns 0 and the `<= 0` test would swallow an
    // entry whose own name is empty.
    if after <> '' then
      // Keyset: skip everything up to and including the named entry. Comparing
      // by NAME (not by a remembered index) is what survives a registry change
      // between two pages — an entry inserted before the cursor cannot shift the
      // window and make us skip an unseen one.
      while (aFirst < n) and
            (StrComp(pointer(aNames[aFirst]), pointer(after)) <= 0) do
        inc(aFirst);
  end;
  // <= 0 covers both "paging off" (0) and a nonsensical negative: neither can
  // mean a page size, and refusing to page is the harmless reading of both.
  if aPageSize > 0 then
    if aLast - aFirst + 1 > aPageSize then
    begin
      aLast := aFirst + aPageSize - 1;
      // Only NOW is there a further page — emitting a cursor on the last page
      // would keep a client asking for an empty one forever.
      aNextCursor := McpEncodeCursor(aNames[aLast]);
    end;
  // an exhausted list yields an empty range (aFirst > aLast), which the callers
  // render as an empty array — not as an error: a cursor pointing past the end
  // is a race with a shrinking registry, not a client mistake
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
      'prompts', _ObjFast(['listChanged', true]),
      // "Servers that support completions MUST declare the completions
      // capability" — an empty object, it has no sub-features
      'completions', _ObjFast([]),
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
  fPrompts := Collections.NewPlainKeyValue<RawUtf8, IMcpPrompt>;
  fTemplates := Collections.NewPlainKeyValue<RawUtf8, IMcpResourceTemplate>;
  fActive := false;
  fListCacheTtlMs := MCP_CACHE_TTL_DEFAULT;
  fReadCacheTtlMs := MCP_CACHE_TTL_DEFAULT;
  // never assume a shared cache is safe: both default to private
  fListCacheScope := mcsPrivate;
  fReadCacheScope := mcsPrivate;
  fSubscriptionSafe.Init;
  fMaxSubscriptions := 8; // see the property: each one holds a worker thread
  fListPageSize := MCP_DEFAULT_PAGE_SIZE;
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
      else if aNotification = 'notifications/prompts/list_changed' then
        wants := sub.Filter.PromptsListChanged
      else if aNotification = 'notifications/resources/list_changed' then
        wants := sub.Filter.ResourcesListChanged
      else if aNotification = 'notifications/resources/updated' then
        wants := sub.WatchesResource(aUri)
      else
        // an unrouted notification is a WIRING bug, not a filter decision: the
        // capability was announced, the notify method exists, and the message
        // would vanish here without a trace. Fail loudly instead.
        raise EMcpException.CreateUtf8(
          'Broadcast: no subscription filter routes %', [aNotification]);
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

procedure TMcpServer.NotifyPromptsListChanged;
begin
  Broadcast('notifications/prompts/list_changed', '');
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
  // omitted." All three fire here, so all three are echoed back when asked for.
  agreed.InitObject([], JSON_FAST);
  if aSubscription.Filter.ToolsListChanged then
    agreed.AddValue('toolsListChanged', true);
  if aSubscription.Filter.PromptsListChanged then
    agreed.AddValue('promptsListChanged', true);
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
    mcpResourcesList,
    mcpResourcesTemplatesList,
    mcpPromptsList:
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

procedure TMcpServer.RegisterPrompt(const aPrompt: IMcpPrompt);
var
  name: RawUtf8;
begin
  if aPrompt = nil then
    exit;
  name := aPrompt.GetName;
  fSafe.Lock;
  try
    fPrompts.Add(name, aPrompt);
  finally
    fSafe.UnLock;
  end;
  NotifyPromptsListChanged;
end;

function TMcpServer.UnregisterPrompt(const aName: RawUtf8): boolean;
begin
  fSafe.Lock;
  try
    result := fPrompts.Remove(aName);
  finally
    fSafe.UnLock;
  end;
  if result then
    NotifyPromptsListChanged;
end;

procedure TMcpServer.Start;
begin
  // Validate the authorization configuration HERE, where it is still a startup
  // error a developer sees — not on the first request, where it would be a
  // runtime failure in front of a user, or worse, silently serve a metadata
  // document no client can act on.
  if fTokenVerifier <> nil then
  begin
    if fAuthResource = '' then
      raise EMcpException.CreateU('A protected server needs AuthResource: ' +
        'without a canonical URI there is no audience to validate against');
    if fAuthorizationServers = nil then
      // "The Protected Resource Metadata document returned by the MCP server
      // MUST include the authorization_servers field containing at least one
      // authorization server." A document without it tells a client nothing
      // about where to obtain a token, so it could never come back with one.
      raise EMcpException.CreateU('A protected server MUST name at least one ' +
        'entry in AuthorizationServers: a client has no other way to learn ' +
        'where to authenticate');
  end;
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

function TMcpServer.SortedToolNames: TRawUtf8DynArray;
var
  pair: TPair<RawUtf8, IMcpTool>;
  n: PtrInt;
begin
  result := nil;
  fSafe.Lock;
  try
    SetLength(result, fTools.Count);
    n := 0;
    for pair in fTools do
    begin
      result[n] := pair.Key;
      inc(n);
    end;
  finally
    fSafe.UnLock;
  end;
  // A dictionary enumerates in hash order, which is neither stable across
  // insertions nor the same on two machines. Pagination over that would repeat
  // some tools and skip others; even unpaginated it would make tools/list a
  // moving target for caches and diffs. Sorting is the cheapest way to make the
  // list a well-defined sequence.
  QuickSortRawUtf8(result, length(result));
end;

function TMcpServer.SortedResourceUris: TRawUtf8DynArray;
var
  pair: TPair<RawUtf8, IMcpResource>;
  n: PtrInt;
begin
  result := nil;
  fSafe.Lock;
  try
    SetLength(result, fResources.Count);
    n := 0;
    for pair in fResources do
    begin
      result[n] := pair.Key;
      inc(n);
    end;
  finally
    fSafe.UnLock;
  end;
  QuickSortRawUtf8(result, length(result));
end;

// The four Sorted*() below are deliberately four near-identical functions rather
// than one generic helper: IKeyValue<RawUtf8,T> would need a generic method, and
// the FPC/Delphi generic surface is exactly where this codebase has to stay
// boring. What matters is that NONE of them may drop the QuickSortRawUtf8 call —
// a registry enumerated in hash order breaks pagination (repeats and gaps), and
// the compiler cannot tell you about it. A fifth registry copied from here must
// keep both the lock and the sort.

function TMcpServer.SortedPromptNames: TRawUtf8DynArray;
var
  pair: TPair<RawUtf8, IMcpPrompt>;
  n: PtrInt;
begin
  result := nil;
  fSafe.Lock;
  try
    SetLength(result, fPrompts.Count);
    n := 0;
    for pair in fPrompts do
    begin
      result[n] := pair.Key;
      inc(n);
    end;
  finally
    fSafe.UnLock;
  end;
  QuickSortRawUtf8(result, length(result));
end;

function TMcpServer.ListPrompts(const aParams: variant): variant;
var
  doc, promptsList: TDocVariantData;
  names: TRawUtf8DynArray;
  prompt: IMcpPrompt;
  cursor, nextCursor, txt: RawUtf8;
  promptObj, args: variant;
  first, last, i: PtrInt;
  hasCursor: boolean;
begin
  names := SortedPromptNames;
  hasCursor := McpRequestCursor(aParams, cursor);
  McpPageRange(names, cursor, hasCursor, fListPageSize, first, last, nextCursor);

  doc.InitObject([], JSON_FAST);
  promptsList.InitArray([], JSON_FAST);
  for i := first to last do
  begin
    fSafe.Lock;
    try
      if not fPrompts.TryGetValue(names[i], prompt) then
        prompt := nil;
    finally
      fSafe.UnLock;
    end;
    if prompt = nil then
      continue;
    // A FRESH variant per entry. Reusing one TDocVariantData and calling
    // InitObject again would leak: Init() nils VName/VValue without releasing
    // them (it is written for an uninitialized record), so every entry after
    // the first orphans its predecessor's arrays.
    promptObj := _ObjFast(['name', names[i]]);
    // title/description/arguments are all OPTIONAL: emit them only when the
    // prompt actually supplies one, rather than shipping empty strings a client
    // would then have to treat as "present but blank"
    txt := prompt.GetTitle;
    if txt <> '' then
      _ObjAddProp('title', txt, promptObj);
    txt := prompt.GetDescription;
    if txt <> '' then
      _ObjAddProp('description', txt, promptObj);
    args := prompt.GetArguments;
    if _Safe(args)^.IsArray and
       (_Safe(args)^.Count > 0) then
      _ObjAddProp('arguments', args, promptObj);
    promptsList.AddItem(promptObj);
  end;

  doc.AddValue('prompts', variant(promptsList));
  if nextCursor <> '' then
    doc.AddValue(MCP_RESULT_NEXT_CURSOR, RawUtf8ToVariant(nextCursor));
  result := variant(doc);
end;

function TMcpServer.GetPrompt(const aParams: variant;
  const aContext: TMcpCallContext): variant;
var
  doc, res: PDocVariantData;
  promptName: RawUtf8;
  args, rendered: variant;
  prompt: IMcpPrompt;
  interactive: IMcpInteractivePrompt;
  wrap: TDocVariantData;
begin
  // Single-argument _Safe: it always yields a usable (possibly empty) doc, so a
  // non-object `params` produces a clean -32602 below instead of dereferencing
  // an unset pointer. The two-argument overload leaves its out-param untouched
  // when it returns false — safe only as long as the caller checks, and this
  // code must not depend on a guard that lives in another function.
  doc := _Safe(aParams);
  if not doc^.GetAsRawUtf8('name', promptName) then
    raise EMcpInvalidParams.CreateU('Missing prompt name in prompts/get');
  args := doc^.GetValueOrDefault('arguments', Null);

  fSafe.Lock;
  try
    if not fPrompts.TryGetValue(promptName, prompt) then
      // "Invalid prompt name: -32602" — naming something that does not exist is
      // bad input, not a server failure
      raise EMcpInvalidParams.CreateUtf8('Prompt not found: %', [promptName]);
  finally
    fSafe.UnLock;
  end;

  // A prompt that opted into Multi Round-Trip Requests gets the full context;
  // every other prompt keeps the two-argument call it was written against.
  if Supports(prompt, IMcpInteractivePrompt, interactive) then
    rendered := interactive.RenderInteractive(args, aContext)
  else
    rendered := prompt.Render(args, aContext.Auth);

  // Convenience: a prompt may return just the messages array. Wrapping it here
  // means every prompt does not have to build the envelope, and the result
  // still leaves this method as the object the spec requires.
  res := _Safe(rendered);
  if res^.IsArray then
  begin
    wrap.InitObject(['messages', rendered], JSON_FAST);
    result := variant(wrap);
  end
  else if res^.IsObject then
    result := rendered
  else
    // neither shape: the prompt is broken, and shipping it would produce a
    // result no client can read
    raise EMcpException.CreateUtf8(
      'prompt % returned neither a messages array nor a result object',
      [promptName]);
end;

procedure TMcpServer.RegisterResourceTemplate(const aTemplate: IMcpResourceTemplate);
var
  uri: RawUtf8;
begin
  if aTemplate = nil then
    exit;
  uri := aTemplate.GetUriTemplate;
  fSafe.Lock;
  try
    fTemplates.Add(uri, aTemplate);
  finally
    fSafe.UnLock;
  end;
  // A template widens the resource surface, so the resources list changed as
  // far as a client is concerned — there is no separate templates notification.
  NotifyResourcesListChanged;
end;

function TMcpServer.UnregisterResourceTemplate(const aUriTemplate: RawUtf8): boolean;
begin
  fSafe.Lock;
  try
    result := fTemplates.Remove(aUriTemplate);
  finally
    fSafe.UnLock;
  end;
  if result then
    NotifyResourcesListChanged;
end;

function TMcpServer.SortedTemplateUris: TRawUtf8DynArray;
var
  pair: TPair<RawUtf8, IMcpResourceTemplate>;
  n: PtrInt;
begin
  result := nil;
  fSafe.Lock;
  try
    SetLength(result, fTemplates.Count);
    n := 0;
    for pair in fTemplates do
    begin
      result[n] := pair.Key;
      inc(n);
    end;
  finally
    fSafe.UnLock;
  end;
  QuickSortRawUtf8(result, length(result));
end;

function TMcpServer.ListResourceTemplates(const aParams: variant): variant;
var
  doc, list: TDocVariantData;
  obj: variant;
  uris: TRawUtf8DynArray;
  tpl: IMcpResourceTemplate;
  cursor, nextCursor, txt: RawUtf8;
  first, last, i: PtrInt;
  hasCursor: boolean;
begin
  uris := SortedTemplateUris;
  hasCursor := McpRequestCursor(aParams, cursor);
  McpPageRange(uris, cursor, hasCursor, fListPageSize, first, last, nextCursor);

  doc.InitObject([], JSON_FAST);
  list.InitArray([], JSON_FAST);
  for i := first to last do
  begin
    fSafe.Lock;
    try
      if not fTemplates.TryGetValue(uris[i], tpl) then
        tpl := nil;
    finally
      fSafe.UnLock;
    end;
    if tpl = nil then
      continue;
    // fresh variant per entry — see ListPrompts for why reuse leaks
    obj := _ObjFast([
      'uriTemplate', uris[i],
      'name', tpl.GetName]);
    txt := tpl.GetTitle;
    if txt <> '' then
      _ObjAddProp('title', txt, obj);
    txt := tpl.GetDescription;
    if txt <> '' then
      _ObjAddProp('description', txt, obj);
    txt := tpl.GetMimeType;
    if txt <> '' then
      _ObjAddProp('mimeType', txt, obj);
    list.AddItem(obj);
  end;

  doc.AddValue('resourceTemplates', variant(list));
  if nextCursor <> '' then
    doc.AddValue(MCP_RESULT_NEXT_CURSOR, RawUtf8ToVariant(nextCursor));
  result := variant(doc);
end;

function TMcpServer.CompleteArgument(const aParams: variant;
  const aAuthCtx: TMcpAuthContext): variant;
var
  doc, refDoc, argDoc, ctxDoc: PDocVariantData;
  refType, refName, argName, argValue: RawUtf8;
  target: IMcpCompletable;
  prompt: IMcpPrompt;
  tpl: IMcpResourceTemplate;
  values: TRawUtf8DynArray;
  arr: TDocVariantData;
  completion: TDocVariantData;
  ctx: variant;
  total, i: PtrInt;
  truncated: boolean;
begin
  doc := _Safe(aParams);
  if not doc^.GetAsDocVariant('ref', refDoc) or
     not refDoc^.GetAsRawUtf8('type', refType) then
    raise EMcpInvalidParams.CreateU('completion/complete needs a ref with a type');
  if not doc^.GetAsDocVariant('argument', argDoc) or
     not argDoc^.IsObject then
    raise EMcpInvalidParams.CreateU('completion/complete needs an argument object');
  argName := argDoc^.U['name'];
  argValue := argDoc^.U['value'];
  if argName = '' then
    raise EMcpInvalidParams.CreateU('completion/complete needs argument.name');
  // Already-resolved arguments, so a suggestion can depend on an earlier choice.
  // A `context` of the wrong shape is refused rather than silently read as "no
  // context": the caller would otherwise get suggestions computed without the
  // constraint it believed it had sent.
  SetVariantNull(ctx);
  if doc^.GetValueIndex('context') >= 0 then
    if doc^.GetAsDocVariant('context', ctxDoc) and
       ctxDoc^.IsObject then
      ctx := ctxDoc^.GetValueOrNull('arguments')
    else
      raise EMcpInvalidParams.CreateU('completion/complete context must be an object');

  target := nil;
  if refType = 'ref/prompt' then
  begin
    refName := refDoc^.U['name'];
    fSafe.Lock;
    try
      if not fPrompts.TryGetValue(refName, prompt) then
        prompt := nil;
    finally
      fSafe.UnLock;
    end;
    if prompt = nil then
      // "Invalid prompt name: -32602"
      raise EMcpInvalidParams.CreateUtf8('Prompt not found: %', [refName]);
    Supports(prompt, IMcpCompletable, target);
  end
  else if refType = 'ref/resource' then
  begin
    refName := refDoc^.U['uri'];
    fSafe.Lock;
    try
      if not fTemplates.TryGetValue(refName, tpl) then
        tpl := nil;
    finally
      fSafe.UnLock;
    end;
    if tpl = nil then
      raise EMcpInvalidParams.CreateUtf8('Resource template not found: %',
        [refName]);
    Supports(tpl, IMcpCompletable, target);
  end
  else
    raise EMcpInvalidParams.CreateUtf8(
      'unknown completion ref type %: MCP defines ref/prompt and ref/resource',
      [refType]);

  values := nil;
  // A prompt or template that offers no completion is not an error: it simply
  // has nothing to suggest, and an empty values array says exactly that.
  if target <> nil then
    values := target.Complete(argName, argValue, ctx, aAuthCtx);

  total := length(values);
  // "Maximum 100 items per response" — enforced HERE, not trusted to the
  // implementation: a handler returning more would otherwise put an
  // over-long response on the wire and violate the spec on its behalf.
  truncated := total > MCP_COMPLETION_MAX_VALUES;
  arr.InitArray([], JSON_FAST);
  for i := 0 to total - 1 do
  begin
    if i >= MCP_COMPLETION_MAX_VALUES then
      break;
    arr.AddItem(RawUtf8ToVariant(values[i]));
  end;

  completion.InitObject(['values', variant(arr)], JSON_FAST);
  // total/hasMore are optional; report them because we know both exactly
  completion.AddValue('total', total);
  completion.AddValue('hasMore', truncated);
  result := _ObjFast(['completion', variant(completion)]);
end;

function TMcpServer.ListTools(const aParams: variant): variant;
var
  doc, toolsList: TDocVariantData;
  names: TRawUtf8DynArray;
  tool: IMcpTool;
  cursor, nextCursor: RawUtf8;
  first, last, i: PtrInt;
  hasCursor: boolean;
begin
  names := SortedToolNames;
  hasCursor := McpRequestCursor(aParams, cursor);
  McpPageRange(names, cursor, hasCursor, fListPageSize, first, last, nextCursor);

  doc.InitObject([], JSON_FAST);
  toolsList.InitArray([], JSON_FAST);
  for i := first to last do
  begin
    // Re-resolve under the lock per entry: a tool unregistered between the
    // snapshot and here simply drops out of this page rather than raising.
    fSafe.Lock;
    try
      if not fTools.TryGetValue(names[i], tool) then
        tool := nil;
    finally
      fSafe.UnLock;
    end;
    if tool = nil then
      continue;
    // fresh variant per entry — see ListPrompts for why reuse leaks
    toolsList.AddItem(_ObjFast([
      'name', names[i],
      'description', tool.GetDescription,
      'inputSchema', tool.GetInputSchema]));
  end;

  doc.AddValue('tools', variant(toolsList));
  if nextCursor <> '' then
    doc.AddValue(MCP_RESULT_NEXT_CURSOR, RawUtf8ToVariant(nextCursor));
  result := variant(doc);
end;

function TMcpServer.ListResources(const aParams: variant): variant;
var
  doc, resourcesList: TDocVariantData;
  uris: TRawUtf8DynArray;
  res: IMcpResource;
  cursor, nextCursor: RawUtf8;
  first, last, i: PtrInt;
  hasCursor: boolean;
begin
  uris := SortedResourceUris;
  hasCursor := McpRequestCursor(aParams, cursor);
  McpPageRange(uris, cursor, hasCursor, fListPageSize, first, last, nextCursor);

  doc.InitObject([], JSON_FAST);
  resourcesList.InitArray([], JSON_FAST);
  for i := first to last do
  begin
    fSafe.Lock;
    try
      if not fResources.TryGetValue(uris[i], res) then
        res := nil;
    finally
      fSafe.UnLock;
    end;
    if res = nil then
      continue;
    // fresh variant per entry — see ListPrompts for why reuse leaks
    resourcesList.AddItem(_ObjFast([
      'uri', res.GetUri,
      'name', res.GetName,
      'description', res.GetDescription,
      'mimeType', res.GetMimeType]));
  end;

  doc.AddValue('resources', variant(resourcesList));
  if nextCursor <> '' then
    doc.AddValue(MCP_RESULT_NEXT_CURSOR, RawUtf8ToVariant(nextCursor));
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
  // requests" than prompts/get, resources/read and tools/call. Asking for input
  // from tools/list would produce a result no conforming client acts on.
  if not (aMethod in [mcpToolsCall, mcpResourcesRead, mcpPromptsGet]) then
    raise EMcpException.CreateU('An InputRequiredResult is only allowed on ' +
      'tools/call, resources/read and prompts/get');

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

procedure TMcpServer.CheckNotStarted(const aWhat: RawUtf8);
begin
  if fActive then
    raise EMcpException.CreateUtf8('% must be set before Start: authorization ' +
      'is startup configuration, and changing it under live requests would ' +
      'have some of them run against the old value and some against the new',
      [aWhat]);
end;

procedure TMcpServer.SetTokenVerifier(const aVerifier: IMcpTokenVerifier);
begin
  CheckNotStarted('TokenVerifier');
  fTokenVerifier := aVerifier;
end;

procedure TMcpServer.SetAuthResource(const aResource: RawUtf8);
begin
  CheckNotStarted('AuthResource');
  fAuthResource := aResource;
end;

procedure TMcpServer.SetScopesSupported(const aScopes: TRawUtf8DynArray);
var
  i: PtrInt;
begin
  CheckNotStarted('ScopesSupported');
  for i := 0 to high(aScopes) do
    if aScopes[i] = 'offline_access' then
      // "MCP Servers (Protected Resources) SHOULD NOT include offline_access in
      // WWW-Authenticate scope or Protected Resource Metadata scopes_supported,
      // as refresh tokens are not a resource requirement." Refusing it here
      // beats documenting it: this list is copied into config files.
      raise EMcpException.CreateU('offline_access is a client concern and must ' +
        'not be advertised in scopes_supported');
  fScopesSupported := aScopes;
end;

function TMcpServer.IsProtected: boolean;
begin
  result := fTokenVerifier <> nil;
end;

function TMcpServer.ProtectedResourceMetadata: RawUtf8;
var
  doc, arr: TDocVariantData;
begin
  doc.InitObject([], JSON_FAST);
  // `resource` is the only REQUIRED member: it is the canonical URI clients put
  // in the RFC 8707 `resource` parameter, so it MUST equal what we validate
  // audiences against — publishing anything else would have clients obtain
  // tokens this server then rejects.
  doc.AddValue('resource', fAuthResource);
  if fAuthorizationServers <> nil then
  begin
    arr.InitArrayFrom(fAuthorizationServers, JSON_FAST);
    doc.AddValue('authorization_servers', variant(arr));
  end;
  if fScopesSupported <> nil then
  begin
    arr.InitArrayFrom(fScopesSupported, JSON_FAST);
    doc.AddValue('scopes_supported', variant(arr));
  end;
  // we read the token from the Authorization header and nowhere else: "Access
  // tokens MUST NOT be included in the URI query string"
  arr.InitArray(['header'], JSON_FAST);
  doc.AddValue('bearer_methods_supported', variant(arr));
  result := doc.ToJson;
end;

function TMcpServer.AuthChallenge(aResult: TMcpTokenResult;
  const aScope: RawUtf8): RawUtf8;
begin
  result := 'Bearer';
  // RFC 6750 §3: the `error` parameter belongs in the challenge itself, not only
  // in a response body. A client acts on WWW-Authenticate — told nothing, it
  // cannot tell "fetch a fresh token" (invalid_token) from "ask for more scope"
  // (insufficient_scope) and has no reason to retry at all.
  // - mtrMissing is the ONE case that stays silent: "If the request lacks any
  //   authentication information … the resource server SHOULD NOT include an
  //   error code" (§3.1). Nothing was presented, so nothing was rejected — the
  //   bare challenge IS the message. MCP_TOKEN_ERROR still names invalid_request
  //   for the JSON body, which is our own diagnostic, not the RFC's challenge.
  if (aResult <> mtrMissing) and
     (MCP_TOKEN_ERROR[aResult] <> '') then
    result := result + ' error="' + MCP_TOKEN_ERROR[aResult] + '",';
  // resource_metadata points at the document that names the authorization
  // server — without it a client that has never seen this server has no way to
  // find out where to authenticate, which is the whole point of the challenge
  if fAuthResource <> '' then
    result := result + ' resource_metadata="' +
      McpResourceMetadataUrl(fAuthResource) + '",';
  if aScope <> '' then
    result := result + ' scope="' + aScope + '",'
  else if fScopesSupported <> nil then
    result := result + ' scope="' + RawUtf8ArrayToCsv(fScopesSupported, ' ') + '",';
  if result[length(result)] = ',' then
    SetLength(result, length(result) - 1);
end;

function TMcpServer.AuthHttpStatus(aResult: TMcpTokenResult): integer;
begin
  case aResult of
    mtrValid:
      result := HTTP_MCP_SUCCESS;
    mtrInsufficientScope:
      // "403 Forbidden — Invalid scopes or insufficient permissions"
      result := HTTP_MCP_FORBIDDEN;
  else
    // everything else is "Authorization required or token invalid" — including
    // a wrong audience: such a token is not weak here, it is not ours at all
    result := HTTP_MCP_UNAUTHORIZED;
  end;
end;

function TMcpServer.Authorize(const aAuthorizationHeader: RawUtf8;
  out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
begin
  result := AuthorizeToken(McpBearerToken(aAuthorizationHeader), aAuthCtx);
end;

function TMcpServer.AuthorizeToken(const aToken: RawUtf8;
  out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
var
  verifier: IMcpTokenVerifier;
begin
  Finalize(aAuthCtx);
  FillCharFast(aAuthCtx, SizeOf(aAuthCtx), 0);
  // read the interface into a local: an embedder may swap the verifier at any
  // moment, and half of this function running against each one is worse than
  // either. The local also keeps the instance alive for the call.
  verifier := fTokenVerifier;
  if verifier = nil then
    exit(mtrValid); // authorization is OPTIONAL and not switched on here
  if fAuthResource = '' then
    // Fail CLOSED. Without a canonical URI the audience check is undefined, and
    // "MUST only accept tokens specifically intended for themselves" cannot be
    // satisfied — accepting everything would be the one unacceptable reading.
    raise EMcpException.CreateU('AuthResource must be set before a ' +
      'TokenVerifier: there is no audience to validate against otherwise');
  if aToken = '' then
    exit(mtrMissing);
  result := verifier.VerifyToken(aToken, fAuthResource, aAuthCtx);
  if result <> mtrValid then
  begin
    // Never hand a partially filled context to a caller that only checks the
    // context: a verifier may have written a principal before rejecting.
    // Finalize first — the record holds managed fields, and zeroing them
    // without releasing would leak every rejected request.
    Finalize(aAuthCtx);
    FillCharFast(aAuthCtx, SizeOf(aAuthCtx), 0);
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
  personalized := (m in [mcpToolsCall, mcpResourcesRead, mcpPromptsGet]) and
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
  anonymous: TMcpAuthContext;
begin
  // No caller identity: stays fail-closed, which is what stdio and in-process
  // dispatch get. An HTTP transport MUST use the overload below instead — its
  // token check would otherwise be decorative.
  FillCharFast(anonymous, SizeOf(anonymous), 0);
  result := ExecuteRequest(aRequestJson, anonymous);
end;

function TMcpServer.ExecuteRequest(const aRequestJson: RawUtf8;
  const aAuthCtx: TMcpAuthContext): RawUtf8;
var
  ignored: RawUtf8;
begin
  // stdio and in-process callers have no HTTP status to set, so a scope refusal
  // can only reach them as the JSON-RPC error the third overload also produces
  result := ExecuteRequest(aRequestJson, aAuthCtx, ignored);
end;

function TMcpServer.ExecuteRequest(const aRequestJson: RawUtf8;
  const aAuthCtx: TMcpAuthContext; out aScopeChallenge: RawUtf8): RawUtf8;
var
  method: RawUtf8;
  params, requestId, resultData: variant;
  callCtx: TMcpCallContext;
  isNotification, personalized: boolean;
  status: integer;
  m: TMcpMethod;
begin
  // Single validation gate, shared with the transports: whatever Preflight
  // rejects never reaches a handler, here or anywhere else. It hands the parsed
  // request back, so the body is parsed exactly once per dispatch.
  aScopeChallenge := '';
  if not Preflight(aRequestJson, method, params, requestId, result, status) then
    exit;
  isNotification := VarIsVoid(requestId);
  m := McpMethodFromName(method);

  personalized := false;

  try
    // INSIDE the try: CallContext type-checks the MRTR retry fields and raises
    // EMcpInvalidParams on a malformed one. Building it before the try would
    // let that escape into the HTTP worker, which has no handler for it.
    callCtx := CallContext(params, method, aAuthCtx);
    // A request carrying MRTR retry fields produced a caller-specific answer.
    // Presence decides, not content — VarIsVoid() considers an EMPTY object
    // void, so testing the value would let `inputResponses: {}` be cached as
    // shareable. Only the two methods that may take part in a round trip
    // count: retry fields elsewhere are meaningless and must not degrade
    // their cacheability.
    personalized := (m in [mcpToolsCall, mcpResourcesRead, mcpPromptsGet]) and
                    (callCtx.HasRequestState or callCtx.HasInputResponses);

    // Dispatch to handler — any handler/tool exception is mapped to a JSON-RPC
    // error below, so it never escapes into the HTTP worker.
    case m of
      mcpDiscover:
        resultData := fProcessor.HandleDiscover;
      mcpToolsList:
        resultData := ListTools(params);
      mcpToolsCall:
        resultData := ExecuteToolCall(params, callCtx);
      mcpResourcesList:
        resultData := ListResources(params);
      mcpResourcesRead:
        resultData := ExecuteResourceRead(params, callCtx);
      mcpResourcesTemplatesList:
        resultData := ListResourceTemplates(params);
      mcpPromptsList:
        resultData := ListPrompts(params);
      mcpPromptsGet:
        resultData := GetPrompt(params, callCtx);
      mcpCompletionComplete:
        resultData := CompleteArgument(params, callCtx.Auth);
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
    // A per-operation scope refusal. The JSON-RPC error is built either way, so
    // stdio still gets a usable answer; an HTTP transport reads the challenge
    // and answers 403 with it instead, which is what lets a client step up.
    on E: EMcpInsufficientScope do
      if isNotification then
        result := ''
      else
      begin
        aScopeChallenge := E.Scope;
        result := fProcessor.CreateError(requestId, JSONRPC_INVALID_REQUEST,
          StringToUtf8(E.Message));
      end;
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
  else if aMethod = 'resources/templates/list' then
    result := mcpResourcesTemplatesList
  else if aMethod = 'completion/complete' then
    result := mcpCompletionComplete
  else if aMethod = 'prompts/list' then
    result := mcpPromptsList
  else if aMethod = 'prompts/get' then
    result := mcpPromptsGet
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
