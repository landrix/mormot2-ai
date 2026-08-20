/// MCP Transport Layer - HTTP and Streamable HTTP Implementations
// - this unit is part of the mormot-mcp-server project
// - licensed under MPL/GPL/LGPL three license
// - adopted into the mormot.ai.* namespace for landrix (LandrixAI) from
//   flydev-fr/mormot2-extensions
unit mormot.ai.mcp.server;

{
  *****************************************************************************

   MCP Server Transport Implementations
    - Abstract Transport Base Class
    - HTTP Transport using THttpAsyncServer
    - Streamable HTTP Transport (MCP 2026-07-28, stateless)

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

{$define WITH_LOGS}

uses
  classes,
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.rtti,
  mormot.core.log,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.buffers,
  mormot.core.json,
  mormot.core.data,
  mormot.core.variants,
  mormot.core.perf,
  mormot.core.threads, // TOnNotifyThread, in the THttpAsyncServer constructor
  mormot.net.sock,
  mormot.net.http,
  mormot.net.server,
  mormot.net.async,
  mormot.ai.mcp;


{ ************ Abstract Transport Base }

type
  /// Abstract base class for MCP transports
  TMcpTransportBase = class(TInterfacedObject)
  protected
    fServer: TMcpServer;
    fActive: boolean;
    fPort: integer;
    fHost: RawUtf8;
  public
    /// initialize with MCP server instance
    constructor Create(aServer: TMcpServer); reintroduce; virtual;
    /// the MCP server this transport fronts
    property Server: TMcpServer
      read fServer;
    /// start the transport
    procedure Start; virtual; abstract;
    /// stop the transport
    procedure Stop; virtual; abstract;
    /// check if transport is active
    function IsActive: boolean;
    /// the port number
    property Port: integer read fPort write fPort;
    /// the host address
    property Host: RawUtf8 read fHost write fHost;
  end;


{ ************ HTTP Transport }

type
  /// HTTP transport using mORMot's async HTTP server
  // - handles POST requests with JSON-RPC payload
  // - supports CORS for browser clients
  // - NOT the MCP-specified transport: the spec defines stdio and Streamable
  //   HTTP only. This is plain JSON-RPC-over-HTTP for simple integrations, and
  //   it deliberately answers 200 with the JSON-RPC error in the body (the
  //   usual JSON-RPC convention) instead of the 400/404 the Streamable HTTP
  //   transport MUST use. A spec-conforming client belongs on
  //   TMcpStreamableHttpTransport, which does map the status codes.
  // - the protocol itself is still enforced: it calls ExecuteRequest, which
  //   runs the same PreflightRequest gate as every other entry point
  {$M+}
  TMcpHttpTransport = class(TMcpTransportBase)
  private
    fHttpServer: THttpAsyncServer;
    fEndpoint: RawUtf8;
    fCorsEnabled: boolean;
    fCorsOrigins: RawUtf8;
    procedure SetCorsHeaders(var Ctxt: THttpServerRequest);
    // serve the RFC 9728 Protected Resource Metadata, unauthenticated
    function OnResourceMetadata(Ctxt: THttpServerRequestAbstract): cardinal;
  public
    /// initialize HTTP transport
    constructor Create(aServer: TMcpServer); override;
    /// finalize and cleanup
    destructor Destroy; override;
    /// start HTTP server
    procedure Start; override;
    /// stop HTTP server
    procedure Stop; override;
    /// the endpoint path for MCP requests (default: '/mcp')
    property Endpoint: RawUtf8 read fEndpoint write fEndpoint;
    /// enable/disable CORS support
    property CorsEnabled: boolean read fCorsEnabled write fCorsEnabled;
    /// allowed CORS origins ('*' for all)
    property CorsOrigins: RawUtf8 read fCorsOrigins write fCorsOrigins;

  published
    // all service URI are implemented by these published methods using RTTI
    function mcp(ctxt: THttpServerRequest): cardinal;
  end;


{ ************ Streamable HTTP Transport (MCP 2026-07-28) }

type
  /// low-level raw socket writer used while streaming a chunked SSE response
  // - returns false if the underlying connection write failed
  TMcpRawWrite = function(const aData: RawByteString): boolean of object;

  /// lets a streaming tool emit intermediate SSE 'message' events mid-request
  // - each Emit() flushes one more SSE event to the client before the final
  //   tool response — this is what enables a token-by-token flow
  IMcpStreamEmitter = interface
    ['{2B7E6A41-3C5D-4E8F-9A1B-7D2C4E6F8A0B}']
    /// wrap aJsonMessage (a complete JSON-RPC message, e.g. a
    // notifications/progress) as one SSE 'message' event and send it now
    procedure Emit(const aJsonMessage: RawUtf8);
  end;

  /// optional per-request streaming handler (see OnStreamCall)
  // - aRequestJson is a single JSON-RPC request
  // - aAuthCtx is the caller the transport authenticated for THIS request. A
  //   hook that returns true replaces the whole authorized dispatch, scope
  //   check included, so it MUST gate on this context itself - there is no
  //   other way to reach it, and under THttpAsyncServer any 'current caller'
  //   kept on the transport would be racy across the worker pool
  // - return true if handled: push intermediate events via aEmitter and set
  //   aResponseJson to the final JSON-RPC response (or '' to send none)
  // - return false to let the transport process the request normally
  TMcpStreamCall = function(const aRequestJson: RawUtf8;
    const aEmitter: IMcpStreamEmitter; const aAuthCtx: TMcpAuthContext;
    out aResponseJson: RawUtf8): boolean of object;

  /// Streamable HTTP transport implementing MCP 2026-07-28
  // - the endpoint accepts POST and OPTIONS only: GET (the standalone
  //   notification stream) and DELETE (session teardown) were removed with
  //   protocol sessions and are answered with 405
  // - POST responses always use SSE (text/event-stream) for requests; that
  //   stream is scoped to its request and is not resumable (no Last-Event-ID,
  //   no event ids) — a broken stream means the client re-issues the request
  // - stateless: no Mcp-Session-Id is minted, echoed or required
  // - NO batching: the body must be a single JSON-RPC request or notification;
  //   an array is rejected with 400 (batch input left the protocol with this
  //   revision)
  {$M+}
  TMcpStreamableHttpTransport = class(TMcpTransportBase)
  private
    fHttpServer: THttpAsyncServer;
    fEndpoint: RawUtf8;
    fBindAddress: RawUtf8;
    fCorsEnabled: boolean;
    fCorsOrigins: RawUtf8;
    fOnStreamCall: TMcpStreamCall;
    // -- helper methods --
    procedure SetCorsHeaders(var Ctxt: THttpServerRequest);
    function ValidateOrigin(var Ctxt: THttpServerRequest): boolean;
    // -- SSE formatting --
    // no event id: streams are not resumable in this revision
    function FormatSseEvent(const aEvent, aData: RawUtf8): RawUtf8;
    // wrap a payload as one HTTP/1.1 chunked-transfer frame (hex-len CRLF .. CRLF)
    function SseChunk(const aPayload: RawUtf8): RawUtf8;
    // Stream the chunked text/event-stream response for a deferred POST: writes
    // the HTTP head, then one SSE event per JSON-RPC response (calling
    // OnStreamCall first so a tool can push intermediate token events), then the
    // terminating 0-chunk. All writes go through aWrite (the connection's raw
    // socket writer) so they flush incrementally. Called from
    // TMcpStreamableAsyncConnection.OnRead after the handler returned
    // HTTP_ASYNCRESPONSE. aBody is the request body; aOutHeaders carries the
    // CORS lines the handler prepared.
    // - only ever reached for a request that already passed PreflightRequest in
    //   mcp(), which is why this path can hardcode 200: every status other than
    //   200 was answered before the stream was opened
    procedure StreamDeferredResponse(const aWrite: TMcpRawWrite;
      const aBody, aOutHeaders: RawUtf8; const aAuthCtx: TMcpAuthContext);
    // send a JSON-RPC error as a plain buffered JSON response with an explicit
    // HTTP status — used for everything the protocol layer rejects up front
    function SendProtocolError(var Ctxt: THttpServerRequest;
      const aErrorJson: RawUtf8; aStatus: integer): cardinal;
    // refuse a request at the HTTP layer with the RFC 6750 challenge that tells
    // the client how to come back — scheme, scopes, and where the metadata is
    function SendAuthChallenge(var Ctxt: THttpServerRequest;
      aResult: TMcpTokenResult; const aScope: RawUtf8 = ''): cardinal;
    // Hold a subscriptions/listen stream open: acknowledge, then deliver
    // queued notifications until the client disconnects or the server ends it.
    // Runs on the connection's own thread for the lifetime of the stream (see
    // TMcpServer.MaxSubscriptions for why that is bounded).
    procedure StreamSubscription(const aWrite: TMcpRawWrite;
      const aBody: RawUtf8; const aAuthCtx: TMcpAuthContext);
    // -- GET/DELETE are gone with protocol sessions: answer 405 --
    function OnMethodNotAllowed(Ctxt: THttpServerRequestAbstract): cardinal;
    // serve the RFC 9728 Protected Resource Metadata, unauthenticated
    function OnResourceMetadata(Ctxt: THttpServerRequestAbstract): cardinal;
  public
    /// initialize with MCP server instance
    constructor Create(aServer: TMcpServer); override;
    /// finalize and cleanup
    destructor Destroy; override;
    /// start HTTP server on configured port
    procedure Start; override;
    /// stop HTTP server and clear sessions
    procedure Stop; override;
    /// the endpoint path (default: '/mcp')
    property Endpoint: RawUtf8 read fEndpoint write fEndpoint;
    /// optional bind address to restrict the listening socket
    // - empty (default) binds all interfaces (e.g. '0.0.0.0'), preserving the
    //   original behavior
    // - set to '127.0.0.1' to expose the server on loopback only — strongly
    //   recommended for demos/tools that wrap local resources, since this
    //   transport ships with CORS '*' and no authentication
    property BindAddress: RawUtf8 read fBindAddress write fBindAddress;
    /// enable/disable CORS support (default: true)
    property CorsEnabled: boolean read fCorsEnabled write fCorsEnabled;
    /// allowed CORS origins (default: '*')
    property CorsOrigins: RawUtf8 read fCorsOrigins write fCorsOrigins;
    /// optional hook to stream a request token-by-token (see TMcpStreamCall)
    // - when assigned and it returns true for a given request, the transport
    //   emits the intermediate events it pushed plus its final response;
    //   otherwise the request is processed normally via the MCP server
    // - a hook that handles a request bypasses ExecuteRequest entirely, so it
    //   receives the authenticated context and owns the authorization
    property OnStreamCall: TMcpStreamCall read fOnStreamCall write fOnStreamCall;
  published
    /// single endpoint handler — routes by HTTP method
    // - uses RTTI-based route publishing (same pattern as TMcpHttpTransport)
    function mcp(Ctxt: THttpServerRequest): cardinal;
  end;


implementation

const
  /// how often a subscription stream looks for queued notifications
  // - small enough that a change reaches the client promptly, large enough
  //   that an idle stream costs nothing measurable
  MCP_SUBSCRIPTION_POLL_MS = 50;
  /// how long a subscription stream may stay silent before a keep-alive
  // - an SSE comment line; also the probe that detects a client which
  //   disappeared without closing the socket
  MCP_SUBSCRIPTION_KEEPALIVE_MS = 15000;

type
  TMcpStreamableAsyncServer = class(THttpAsyncServer)
  public
    Transport: TMcpStreamableHttpTransport;
    constructor Create(const aPort: RawUtf8; const OnStart, OnStop: TOnNotifyThread;
      const ProcessName: RawUtf8; ServerThreadPoolCount: integer = 32;
      KeepAliveTimeOut: integer = 30000; ProcessOptions: THttpServerOptions = [];
      aLog: TSynLogClass = nil); override;
  end;

  TMcpStreamableAsyncConnection = class(THttpAsyncServerConnection)
  protected
    fStreaming: boolean;        // true while pushing chunked SSE: keep conn open
    fStreamWriteFailed: boolean; // a WriteRaw failed mid-stream -> close at end
    function OnRead: TPollAsyncSocketOnReadWrite; override;
    // while streaming, every WriteString triggers AfterWrite; keep the
    // connection open (soContinue) instead of the base "unexpected -> soClose"
    function AfterWrite: TPollAsyncSocketOnReadWrite; override;
    // raw socket writer handed to TMcpStreamableHttpTransport.StreamDeferredResponse
    function WriteRaw(const aData: RawByteString): boolean;
  end;


{ ************ TMcpTransportBase }

constructor TMcpTransportBase.Create(aServer: TMcpServer);
begin
  inherited Create;
  fServer := aServer;
  fActive := false;
  fPort := 3000;
  fHost := 'localhost';
end;

function TMcpTransportBase.IsActive: boolean;
begin
  result := fActive;
end;
{ ************ TMcpHttpTransport }

constructor TMcpHttpTransport.Create(aServer: TMcpServer);
begin
  inherited Create(aServer);
  fEndpoint := '/mcp';
  fCorsEnabled := true;
  fCorsOrigins := '*';
end;

destructor TMcpHttpTransport.Destroy;
begin
  Stop;
  inherited;
end;

procedure TMcpHttpTransport.SetCorsHeaders(var Ctxt: THttpServerRequest);
begin
  if not fCorsEnabled then
    exit;
    
  Ctxt.OutCustomHeaders := Ctxt.OutCustomHeaders +
    'Access-Control-Allow-Origin: ' + fCorsOrigins + #13#10 +
    'Access-Control-Allow-Methods: POST, GET, OPTIONS' + #13#10 +
    // Authorization must be allowed or a browser client's preflight fails and
    // the authenticated POST never leaves the page; WWW-Authenticate must be
    // exposed or its JavaScript cannot read the challenge that tells it where
    // to authenticate — a protected server would be unusable from a browser
    // without either.
    'Access-Control-Allow-Headers: Content-Type, Authorization' + #13#10 +
    'Access-Control-Expose-Headers: WWW-Authenticate' + #13#10 +
    'Access-Control-Max-Age: 86400' + #13#10;
end;

function TMcpHttpTransport.OnResourceMetadata(
  Ctxt: THttpServerRequestAbstract): cardinal;
begin
  // public by construction: a client fetches this precisely because it has no
  // token yet, so requiring one would make discovery impossible
  Ctxt.SetOutCustomHeader(['Access-Control-Allow-Origin', '*']);
  if not fServer.IsProtected then
    exit(HTTP_NOTFOUND); // nothing to discover on an open server
  result := Ctxt.SetOutJson(fServer.ProtectedResourceMetadata);
end;

function TMcpHttpTransport.mcp(ctxt: THttpServerRequest): cardinal;
var
  requestBody, responseBody, scopeChallenge: RawUtf8;
  tokenResult: TMcpTokenResult;
  authCtx: TMcpAuthContext;
begin
  // Set CORS headers
  SetCorsHeaders(Ctxt);

  // Handle OPTIONS preflight
  if Ctxt.Method = 'OPTIONS' then
    exit(HTTP_NOCONTENT);

  // Handle GET for server info
  if Ctxt.Method = 'GET' then
  begin
    result := Ctxt.SetOutJson('{"status":"active","protocol":"MCP"}');
    exit;
  end;

  // Handle POST for JSON-RPC
  if Ctxt.Method <> 'POST' then
  begin
    Ctxt.SetOutJson('{"error":"Only POST method supported"}');
    exit(HTTP_BADREQUEST);
  end;

  // Authorization before the body is even read: "MCP servers MUST validate
  // access tokens before processing the request". A no-op unless a verifier
  // is plugged in.
  tokenResult := fServer.AuthorizeToken(Ctxt.AuthBearer, authCtx);
  if tokenResult <> mtrValid then
  begin
    Ctxt.SetOutCustomHeader(['WWW-Authenticate', fServer.AuthChallenge(tokenResult)]);
    Ctxt.SetOutJson('{"error":"' + MCP_TOKEN_ERROR[tokenResult] + '"}');
    exit(fServer.AuthHttpStatus(tokenResult));
  end;

  // Read request body
  requestBody := Ctxt.InContent;

  // Execute MCP request ON BEHALF OF the caller we just verified: without
  // handing the context down, every tool would see an unauthenticated caller
  // no matter what token was presented.
  responseBody := fServer.ExecuteRequest(requestBody, authCtx, scopeChallenge);

  // a handler that refused for lack of scope wants 403 + the challenge naming
  // what to ask for, not a 200 carrying a JSON-RPC error
  if scopeChallenge <> '' then
  begin
    Ctxt.SetOutCustomHeader(['WWW-Authenticate',
      fServer.AuthChallenge(mtrInsufficientScope, scopeChallenge)]);
    Ctxt.SetOutJson(responseBody);
    exit(HTTP_MCP_FORBIDDEN);
  end;

  // Send response
  if responseBody = '' then
    exit(HTTP_NOCONTENT);

  Ctxt.SetOutJson(responseBody);
  // The status comes from the response, not from the fact that we produced one:
  // the spec pins -32021 (and -32020/-32022/-32601) to a status of their own,
  // and -32021 can only be decided after the handler ran.
  result := McpHttpStatus(responseBody);
end;

procedure TMcpHttpTransport.Start;
var
  wellKnown: RawUtf8;
begin
  if fActive then
    exit;

  // Create and start HTTP server
  fHttpServer := THttpAsyncServer.Create(
    ToUtf8(fPort), nil, nil, 'mcp', 32,
    5 * 60 * 1000,         // 5 minutes keep alive connections
    [hsoNoXPoweredHeader,  // not needed for a benchmark
     //hsoHeadersInterning,  // reduce memory contention for /plaintext and /json
     hsoNoStats,           // disable low-level statistic counters
     //hsoThreadCpuAffinity, // worse scaling on multi-servers
     hsoThreadSmooting,    // seems a good option, even if not magical
     hsoEnablePipelining,  // as expected by /plaintext
     {$ifdef WITH_LOGS}
     hsoLogVerbose,
     {$endif WITH_LOGS}
     hsoIncludeDateHeader  // required by TFB General Test Requirements #5
    ]);
  //  if pin2Core <> -1 then
  //    fHttpServer.Async.SetCpuAffinity(pin2Core);
  fHttpServer.HttpQueueLength := 10000; // needed e.g. from wrk/ab benchmarks
  fHttpServer.ServerName := 'MMCP-HTTP';
  // use default routing using RTTI on the TRawAsyncServer published methods
  fHttpServer.Route.RunMethods(
    [urmGet, urmPost, urmOptions, urmPut, urmDelete, urmPatch], self);
  // the RFC 9728 discovery document, on both forms a client may probe
  fHttpServer.Route.Get(MCP_WELL_KNOWN_RESOURCE, OnResourceMetadata);
  wellKnown := McpResourceMetadataPath(fServer.AuthResource);
  if wellKnown <> MCP_WELL_KNOWN_RESOURCE then
    fHttpServer.Route.Get(wellKnown, OnResourceMetadata);
  // wait for the server to be ready and raise exception e.g. on binding issue
  fHttpServer.WaitStarted;
  
  fActive := true;
end;

procedure TMcpHttpTransport.Stop;
begin
  if not fActive then
    exit;
    
  if fHttpServer <> nil then
  begin
    fHttpServer.Shutdown;
    FreeAndNil(fHttpServer);
  end;
  
  fActive := false;
end;

{ ************ TMcpStreamableAsyncServer / TMcpStreamableAsyncConnection }

constructor TMcpStreamableAsyncServer.Create(const aPort: RawUtf8;
  const OnStart, OnStop: TOnNotifyThread; const ProcessName: RawUtf8;
  ServerThreadPoolCount: integer; KeepAliveTimeOut: integer;
  ProcessOptions: THttpServerOptions; aLog: TSynLogClass);
begin
  fConnectionClass := TMcpStreamableAsyncConnection; // must be set before inherited
  inherited Create(aPort, OnStart, OnStop, ProcessName, ServerThreadPoolCount,
    KeepAliveTimeOut, ProcessOptions, aLog);
end;

function TMcpStreamableAsyncConnection.WriteRaw(const aData: RawByteString): boolean;
begin
  result := fOwner.WriteString(self, aData, 5000);
  if not result then
    fStreamWriteFailed := true;
end;

function TMcpStreamableAsyncConnection.AfterWrite: TPollAsyncSocketOnReadWrite;
begin
  if fStreaming then
    // each chunk's WriteString calls AfterWrite; stay open until the stream ends
    result := soContinue
  else
    result := inherited AfterWrite;
end;

function TMcpStreamableAsyncConnection.OnRead: TPollAsyncSocketOnReadWrite;
var
  transport: TMcpStreamableHttpTransport;
  authCtx: TMcpAuthContext;
  tokenResult: TMcpTokenResult;
  refusal: RawUtf8;
  status: integer;
begin
  result := inherited OnRead;
  // The published mcp() handler defers request batches by returning
  // HTTP_ASYNCRESPONSE, which leaves the connection in hrsWaitAsyncProcessing
  // WITHOUT the framework sending any response (see DoRequest in
  // mormot.net.async). We now stream the chunked SSE response ourselves and
  // hand back to AfterWrite for the standard cleanup (fCurrentProcess decrement)
  // and connection close.
  if (fHttp.State = hrsWaitAsyncProcessing) and
     (rfAsynchronous in fHttp.ResponseFlags) and
     (fRequest <> nil) and
     (fRequest.OutContentType = 'text/event-stream') then
  begin
    transport := (fServer as TMcpStreamableAsyncServer).Transport;
    // stream the chunked SSE response incrementally through WriteRaw (each
    // WriteString flushes to the socket immediately, enabling token-by-token
    // delivery when a tool emits intermediate events). fStreaming keeps the
    // connection open across the many writes (see AfterWrite override).
    fStreaming := true;
    fStreamWriteFailed := false;
    // Re-resolve the caller here. mcp() already refused an unauthorized request
    // before deferring, but its context did not survive the hand-off — and a
    // handler reached through this path must see the same identity as one
    // reached through the buffered path, or authorization would depend on which
    // transport happened to answer.
    // The SECOND answer counts, and it can differ from the first: a token that
    // was good at preflight may have expired, been revoked, or hit a verifier
    // that changed its mind in between. Dispatching anyway would run the
    // request with the zeroed context AuthorizeToken leaves behind — anonymous
    // on a server that HAS authorization switched on, and answered 200 where
    // the caller must see 401. Nothing has gone out on the socket yet, so the
    // refusal is still ours to send.
    tokenResult := transport.Server.AuthorizeToken(fRequest.AuthBearer, authCtx);
    if tokenResult <> mtrValid then
    begin
      // RFC 6750: carry the challenge that says how to come back. Written raw
      // because THttpAsyncServer was told not to generate a response for this
      // request (rfAsynchronous) - SendAuthChallenge needs a THttpServerRequest
      // that is no longer the one answering here. Same body shape it uses: NOT
      // a JSON-RPC error, because this refusal is an HTTP-layer one and there
      // is no request id to correlate it with that we did not invent.
      refusal := '{"error":"' + MCP_TOKEN_ERROR[tokenResult] + '"}';
      status := transport.Server.AuthHttpStatus(tokenResult);
      WriteRaw(FormatUtf8('HTTP/1.1 % %'#13#10 +
        'Content-Type: application/json'#13#10 +
        'WWW-Authenticate: %'#13#10 +
        'Content-Length: %'#13#10#13#10 + '%',
        [status, StatusCodeToText(status)^,
         transport.Server.AuthChallenge(tokenResult, ''),
         length(refusal), refusal]));
    end
    else
      transport.StreamDeferredResponse(WriteRaw, fHttp.Content,
        fRequest.OutCustomHeaders, authCtx);
    fStreaming := false;
    // finalize once: hrsResponseDone lets the inherited AfterWrite run the
    // standard cleanup (fCurrentProcess) and either keep-alive (parser reset,
    // soContinue) or close on a failed write.
    if fStreamWriteFailed then
      include(fHttp.HeaderFlags, hfConnectionClose);
    fHttp.State := hrsResponseDone;
    result := AfterWrite;
  end;
end;


{ ************ TMcpStreamableHttpTransport }

constructor TMcpStreamableHttpTransport.Create(aServer: TMcpServer);
begin
  inherited Create(aServer);
  fEndpoint := '/mcp';
  fCorsEnabled := true;
  fCorsOrigins := '*';
  // no session map anymore: 2026-07-28 removed protocol-level sessions, so the
  // transport keeps no per-client state at all
end;

destructor TMcpStreamableHttpTransport.Destroy;
begin
  Stop;
  inherited;
end;

procedure TMcpStreamableHttpTransport.Start;
var
  bind, wellKnown: RawUtf8;
begin
  if fActive then
    exit;
  // bind spec: 'host:port' when BindAddress is set (e.g. '127.0.0.1' for
  // loopback-only), else just the port (all interfaces) as before
  if fBindAddress <> '' then
    bind := fBindAddress + ':' + ToUtf8(fPort)
  else
    bind := ToUtf8(fPort);
  fHttpServer := TMcpStreamableAsyncServer.Create(
    bind, nil, nil, 'mcp-streamable', 32,
    5 * 60 * 1000,
    [hsoNoXPoweredHeader,
     hsoNoStats,
     hsoThreadSmooting,
     {$ifdef WITH_LOGS}
     hsoLogVerbose,
     {$endif WITH_LOGS}
     hsoIncludeDateHeader]);
  // let the streaming connection reach this transport for deferred responses
  TMcpStreamableAsyncServer(fHttpServer).Transport := self;
  fHttpServer.HttpQueueLength := 10000;
  fHttpServer.ServerName := 'MMCP-Streamable';
  // RTTI-based route publishing: the published 'mcp' method handles /mcp
  // for GET, POST, OPTIONS, PUT, PATCH (not DELETE — RTTI doesn't route it)
  fHttpServer.Route.RunMethods(
    [urmGet, urmPost, urmOptions, urmPut, urmPatch], self);
  // DELETE used to terminate a session; sessions are gone, so it is routed
  // explicitly only to answer 405 (RunMethods does not route DELETE at all)
  fHttpServer.Route.Delete(fEndpoint, OnMethodNotAllowed);
  // The discovery document. RFC 9728 INSERTS the resource's path after the
  // well-known segment, so a server at https://host/mcp publishes at
  // /.well-known/oauth-protected-resource/mcp — and the spec has clients probe
  // that form first, the root form second. Both are routed, since either may
  // be what a given client tries.
  fHttpServer.Route.Get(MCP_WELL_KNOWN_RESOURCE, OnResourceMetadata);
  fHttpServer.Route.Options(MCP_WELL_KNOWN_RESOURCE, OnResourceMetadata);
  wellKnown := McpResourceMetadataPath(fServer.AuthResource);
  if wellKnown <> MCP_WELL_KNOWN_RESOURCE then
  begin
    fHttpServer.Route.Get(wellKnown, OnResourceMetadata);
    fHttpServer.Route.Options(wellKnown, OnResourceMetadata);
  end;
  fHttpServer.WaitStarted;
  fActive := true;
end;

procedure TMcpStreamableHttpTransport.Stop;
begin
  if not fActive then
    exit;

  // Order matters, and getting it wrong is a use-after-free:
  // THttpAsyncServer.Shutdown waits only a bounded time (10s in
  // mormot.net.async) for each worker to leave, then force-frees threads and
  // sockets. A subscription stream sits in its loop on such a worker, so it
  // MUST be told to leave BEFORE the shutdown starts — otherwise the wait can
  // expire while the stream is still writing to a connection being freed.
  // Clearing fActive and cancelling makes every stream exit within one poll
  // interval (50ms), far inside that budget.
  fActive := false;
  if fServer <> nil then
    fServer.CancelAllSubscriptions;

  if fHttpServer <> nil then
  begin
    fHttpServer.Shutdown;
    FreeAndNil(fHttpServer);
  end;
end;

function TMcpStreamableHttpTransport.SendAuthChallenge(
  var Ctxt: THttpServerRequest; aResult: TMcpTokenResult;
  const aScope: RawUtf8): cardinal;
begin
  // RFC 6750: a refusal carries the challenge that tells the client HOW to come
  // back — which scheme, which scopes, and where to find the metadata naming
  // the authorization server. A bare 401 leaves a first-contact client stuck.
  Ctxt.SetOutCustomHeader(['WWW-Authenticate', fServer.AuthChallenge(aResult, aScope)]);
  Ctxt.OutContentType := JSON_CONTENT_TYPE_VAR;
  // The body is deliberately NOT a JSON-RPC error: this refusal happens at the
  // HTTP layer, before the body was even read, so there is no request id to
  // correlate it with — and inventing one would be a lie.
  Ctxt.OutContent := '{"error":"' + MCP_TOKEN_ERROR[aResult] + '"}';
  result := fServer.AuthHttpStatus(aResult);
end;

function TMcpStreamableHttpTransport.OnResourceMetadata(
  Ctxt: THttpServerRequestAbstract): cardinal;
begin
  // RFC 9728, served WITHOUT authorization: a client reads this document
  // precisely because it does not have a token yet. Requiring one would make
  // discovery impossible — the metadata is public by construction.
  // Registered explicitly rather than by RTTI: the well-known path contains
  // dots and dashes, which no Pascal method name can carry.
  Ctxt.SetOutCustomHeader(['Access-Control-Allow-Origin', '*']);
  if Ctxt.Method = 'OPTIONS' then
    exit(HTTP_NOCONTENT);
  if Ctxt.Method <> 'GET' then
    exit(HTTP_NOTALLOWED);
  if not fServer.IsProtected then
    // nothing to discover: an open server has no authorization server to name,
    // and publishing an empty document would suggest otherwise
    exit(HTTP_NOTFOUND);
  result := Ctxt.SetOutJson(fServer.ProtectedResourceMetadata);
end;

function TMcpStreamableHttpTransport.SendProtocolError(
  var Ctxt: THttpServerRequest; const aErrorJson: RawUtf8;
  aStatus: integer): cardinal;
begin
  // A rejected request never becomes an SSE stream: it is a buffered JSON body
  // with the status the spec prescribes. Streaming it would force HTTP 200 (the
  // stream head is written before the outcome is known), and the spec REQUIRES
  // 400/404 here — a client that only sees 200 cannot tell a rejection from a
  // result, and the backward-compatibility probe would misclassify the server.
  Ctxt.OutContentType := JSON_CONTENT_TYPE_VAR;
  Ctxt.OutContent := aErrorJson;
  result := aStatus;
end;

function TMcpStreamableHttpTransport.mcp(Ctxt: THttpServerRequest): cardinal;
var
  body, contentType, headerError, errorJson: RawUtf8;
  doc: TDocVariantData;
  method: RawUtf8;
  status: integer;
  tokenResult: TMcpTokenResult;
  authCtx: TMcpAuthContext;
begin
  // --- CORS headers on every response ---
  SetCorsHeaders(Ctxt);

  // --- OPTIONS: CORS preflight ---
  if Ctxt.Method = 'OPTIONS' then
    exit(HTTP_NOCONTENT);

  // --- Only POST is an MCP endpoint method now ---
  // GET (standalone notification stream) and DELETE (session teardown) were
  // removed together with protocol sessions; the spec prescribes 405 for both
  // so an older client can tell "gone" from "never existed".
  if Ctxt.Method <> 'POST' then
    exit(HTTP_NOTALLOWED);

  // --- Origin validation (DNS rebinding protection) ---
  if not ValidateOrigin(Ctxt) then
    exit(HTTP_FORBIDDEN);

  // --- Authorization, BEFORE anything is parsed or dispatched ---
  // "MCP servers MUST validate access tokens before processing the request,
  // ensuring the access token is issued specifically for the MCP server, and
  // take all necessary steps to ensure no data is returned to unauthorized
  // parties." Ahead of the body on purpose: a refused caller must not be able
  // to reach the JSON parser, let alone a handler.
  // On an unprotected server this is a no-op returning mtrValid.
  tokenResult := fServer.AuthorizeToken(Ctxt.AuthBearer, authCtx);
  if tokenResult <> mtrValid then
    exit(SendAuthChallenge(Ctxt, tokenResult));

  // --- Validate Content-Type ---
  // mORMot parses Content-Type out of headers into Ctxt.InContentType
  contentType := LowerCaseU(Ctxt.InContentType);
  if (contentType = '') or
     (PosEx('application/json', contentType) = 0) then
    exit(415); // Unsupported Media Type

  // --- Parse body ---
  // Use InitJson (not InitJsonInPlace) because InContent may be a shared
  // reference-counted string — modifying it in-place causes EInvalidPointer.
  // The body MUST be a single JSON-RPC request or notification: JSON-RPC
  // batching is not part of this revision, so an array is simply malformed.
  // A rejected body still answers with a JSON-RPC error, never an empty 400:
  // the spec's backward-compatibility probe treats a 400 whose body is "not a
  // recognized modern JSON-RPC error" as evidence of a legacy server, and would
  // downgrade to the removed initialize handshake against this very server.
  body := Ctxt.InContent;
  doc.InitJson(body, JSON_FAST);
  if not doc.IsObject then
    exit(SendProtocolError(Ctxt, fServer.Processor.CreateError(
      Null, JSONRPC_INVALID_REQUEST,
      'Request body must be a single JSON-RPC object ' +
      '(batching is not part of this protocol revision)'),
      HTTP_MCP_BAD_REQUEST));
  if not doc.GetAsRawUtf8('method', method) then
    exit(SendProtocolError(Ctxt, fServer.Processor.CreateError(
      doc.GetValueOrNull('id'), JSONRPC_INVALID_REQUEST,
      'Missing required member "method"'), HTTP_MCP_BAD_REQUEST));

  // --- Standard header validation (MUST, -32020 on mismatch) ---
  // Headers mirror body fields so intermediaries can route without parsing;
  // if the two disagree, a proxy and this server would act on different data.
  // The check itself lives in the core unit so a server mounted as a route on
  // a foreign HTTP host enforces exactly the same contract as this transport.
  if not McpValidateRequestHeaders(Ctxt.InHeaders, body, fServer, headerError) then
    exit(SendProtocolError(Ctxt, fServer.Processor.CreateError(
      doc.GetValueOrNull('id'), MCP_ERROR_HEADER_MISMATCH, headerError),
      HTTP_MCP_BAD_REQUEST));

  // --- Accept header validation skipped ---
  // mORMot's THttpAsyncServer filters standard headers (Accept, Content-Type,
  // Content-Length, etc.) out of InHeaders by default (HeadersUnFiltered=false),
  // so the Accept value is not reliably available here: FindNameValuePointer
  // ('ACCEPT: ') may match ACCEPT-ENCODING instead. The spec requires clients to
  // send Accept: application/json, text/event-stream — we cannot enforce it.

  // --- Protocol validation BEFORE anything executes ---
  // This is the only place that can still choose an HTTP status: once we defer
  // and the SSE head goes out, the response is 200 by construction. So the
  // envelope, the mandatory _meta and the method's existence are all decided
  // here — and a rejected request is answered as JSON with 400/404, never
  // streamed. It also means no request reaches OnStreamCall unvalidated.
  if not fServer.PreflightRequest(body, errorJson, status) then
    exit(SendProtocolError(Ctxt, errorJson, status));

  // --- Notification (no id): process and return 202 with no body ---
  if VarIsVoid(doc.GetValueOrNull('id')) then
  begin
    // WITH the context: this was the one path that dropped it, which made a
    // notification run as an anonymous caller while every sibling path (:379,
    // the deferred SSE resolve, StreamDeferredResponse) passes it down. The
    // one-argument overload nulls the context by contract - see its comment
    // in the core unit: a token check that a transport then discards is
    // decorative. There is no scope challenge to report: a notification is
    // answered with 202 and no body, whatever the handler decided.
    fServer.ExecuteRequest(body, authCtx);
    exit(HTTP_ACCEPTED);
  end;

  // --- Requests present: defer and stream a chunked SSE response ---
  // We must NOT assemble a buffered OutContent here: THttpAsyncServer would
  // send it in one shot with a fixed Content-Length, which is not streaming
  // (the original bug). Instead we return HTTP_ASYNCRESPONSE so the framework
  // leaves the connection parked (hrsWaitAsyncProcessing) WITHOUT generating a
  // response. TMcpStreamableAsyncConnection.OnRead then writes a real chunked
  // text/event-stream via BuildDeferredResponse, which executes each request
  // as it emits the matching SSE event, and lets AfterWrite close cleanly.
  Ctxt.OutContentType := 'text/event-stream'; // marker consumed by OnRead
  Ctxt.RespStatus := HTTP_ASYNCRESPONSE;
  result := HTTP_ASYNCRESPONSE;
end;

function TMcpStreamableHttpTransport.OnMethodNotAllowed(
  Ctxt: THttpServerRequestAbstract): cardinal;
begin
  // GET and DELETE were the session-era verbs (standalone SSE stream / session
  // teardown). Both are gone; the spec asks for a plain 405 in this revision.
  result := HTTP_NOTALLOWED;
end;

procedure TMcpStreamableHttpTransport.SetCorsHeaders(var Ctxt: THttpServerRequest);
begin
  if not fCorsEnabled then
    exit;
  // Mcp-Session-Id is gone; the standard request headers of this revision must
  // be allowed instead, or a browser client cannot send them at all
  Ctxt.OutCustomHeaders := Ctxt.OutCustomHeaders +
    'Access-Control-Allow-Origin: ' + fCorsOrigins + #13#10 +
    'Access-Control-Allow-Methods: POST, OPTIONS' + #13#10 +
    'Access-Control-Allow-Headers: Content-Type, Authorization, ' +
      'MCP-Protocol-Version, Mcp-Method, Mcp-Name' + #13#10 +
    // without this a browser client cannot read the 401 challenge at all, and
    // so can never discover where to obtain a token
    'Access-Control-Expose-Headers: WWW-Authenticate' + #13#10 +
    'Access-Control-Max-Age: 86400' + #13#10;
end;

function TMcpStreamableHttpTransport.ValidateOrigin(
  var Ctxt: THttpServerRequest): boolean;
var
  origin: RawUtf8;
  p: PUtf8Char;
  len: PtrInt;
begin
  // If CORS allows all origins, accept everything
  if fCorsOrigins = '*' then
    exit(true);
  // Extract Origin header — Origin is a custom header that stays in InHeaders
  // (mORMot only filters standard headers like Content-Type, Accept, etc.)
  origin := '';
  p := FindNameValuePointer(pointer(Ctxt.InHeaders), 'ORIGIN: ', len);
  if p = nil then
    p := FindNameValuePointer(pointer(Ctxt.InHeaders), 'ORIGIN:', len);
  if p <> nil then
    FastSetString(origin, p, len);
  // Missing Origin is accepted (non-browser clients like CLI tools don't send it)
  if origin = '' then
    exit(true);
  // Check against configured origins
  result := origin = fCorsOrigins;
end;

function TMcpStreamableHttpTransport.FormatSseEvent(
  const aEvent, aData: RawUtf8): RawUtf8;
var
  i, start: integer;
begin
  result := '';
  if aEvent <> '' then
    result := 'event: ' + aEvent + #13#10;
  // no 'id:' line — SSE resumability (Last-Event-ID) was removed
  if aData = '' then
    result := result + 'data:' + #13#10
  else
  begin
    // Split on newlines: each line gets its own 'data: ' prefix
    start := 1;
    for i := 1 to length(aData) do
      if aData[i] = #10 then
      begin
        result := result + 'data: ' + copy(aData, start, i - start) + #13#10;
        start := i + 1;
      end;
    if start <= length(aData) then
      result := result + 'data: ' + copy(aData, start, MaxInt) + #13#10;
  end;
  result := result + #13#10; // blank line terminates the event
end;

function TMcpStreamableHttpTransport.SseChunk(const aPayload: RawUtf8): RawUtf8;
begin
  // HTTP/1.1 chunked transfer-encoding frame: hex-length CRLF data CRLF
  result := StringToUtf8(IntToHex(length(aPayload), 1)) + #13#10 +
    aPayload + #13#10;
end;

type
  // pushes intermediate SSE 'message' events for a streaming tool call, by
  // wrapping each JSON message as one chunked SSE frame and writing it now
  // also an IMcpNotificationSink, which is the same act seen from the core:
  // put one JSON-RPC message on the stream that carries this request's
  // response. The two names exist because the core must not know about SSE.
  TMcpStreamEmitter = class(TInterfacedObject, IMcpStreamEmitter,
    IMcpNotificationSink)
  protected
    fTransport: TMcpStreamableHttpTransport;
    fWrite: TMcpRawWrite;
  public
    constructor Create(aTransport: TMcpStreamableHttpTransport;
      const aWrite: TMcpRawWrite);
    procedure Emit(const aJsonMessage: RawUtf8);
    /// IMcpNotificationSink — same wire act as Emit
    procedure Send(const aJsonMessage: RawUtf8);
  end;

  /// the progress sink of a plain (hookless) request, which writes the SSE head
  /// itself on the first message it is asked to send
  // - this is what lets a handler report progress WITHOUT the transport having
  //   to commit to HTTP 200 before the handler ran; see Send for why that
  //   matters and HeadWritten for how the caller finds out what happened
  TMcpLazyHeadSink = class(TInterfacedObject, IMcpNotificationSink)
  protected
    fTransport: TMcpStreamableHttpTransport;
    fWrite: TMcpRawWrite;
    fOutHeaders: RawUtf8;
    fHeadWritten: boolean;
  public
    constructor Create(aTransport: TMcpStreamableHttpTransport;
      const aWrite: TMcpRawWrite; const aOutHeaders: RawUtf8);
    procedure Send(const aJsonMessage: RawUtf8);
    /// true once a notification went out — the response is then already a 200
    ///  SSE stream and the caller must finish it as one
    property HeadWritten: boolean
      read fHeadWritten;
  end;

constructor TMcpStreamEmitter.Create(aTransport: TMcpStreamableHttpTransport;
  const aWrite: TMcpRawWrite);
begin
  inherited Create;
  fTransport := aTransport;
  fWrite := aWrite;
end;

procedure TMcpStreamEmitter.Emit(const aJsonMessage: RawUtf8);
begin
  fWrite(fTransport.SseChunk(
    fTransport.FormatSseEvent('message', aJsonMessage)));
end;

procedure TMcpStreamEmitter.Send(const aJsonMessage: RawUtf8);
begin
  Emit(aJsonMessage);
end;

{ TMcpLazyHeadSink }

constructor TMcpLazyHeadSink.Create(aTransport: TMcpStreamableHttpTransport;
  const aWrite: TMcpRawWrite; const aOutHeaders: RawUtf8);
begin
  inherited Create;
  fTransport := aTransport;
  fWrite := aWrite;
  fOutHeaders := aOutHeaders;
end;

procedure TMcpLazyHeadSink.Send(const aJsonMessage: RawUtf8);
begin
  // The head is written HERE, on the first message, not up front. Writing it
  // before the handler runs would fix the response at 200 and cost the caller
  // the 400 that -32021 MUST carry and the 403+WWW-Authenticate a scope refusal
  // MUST carry. A handler that never reports progress therefore keeps the full
  // status choice; one that does has already put bytes on the wire, so 200 is
  // then the only honest answer anyway.
  if not fHeadWritten then
  begin
    fHeadWritten := true;
    fWrite('HTTP/1.1 200 OK'#13#10 +
      'Content-Type: text/event-stream'#13#10 +
      'Cache-Control: no-cache'#13#10 +
      'X-Accel-Buffering: no'#13#10 +
      'Transfer-Encoding: chunked'#13#10 +
      fOutHeaders +
      #13#10);
  end;
  fWrite(fTransport.SseChunk(
    fTransport.FormatSseEvent('message', aJsonMessage)));
end;

procedure TMcpStreamableHttpTransport.StreamSubscription(
  const aWrite: TMcpRawWrite; const aBody: RawUtf8;
  const aAuthCtx: TMcpAuthContext);
var
  doc: TDocVariantData;
  requestId: variant;
  sub: TMcpSubscription;
  pending: TRawUtf8DynArray;
  i: PtrInt;
  idle: integer;
  alive: boolean;
begin
  doc.InitJson(aBody, JSON_FAST);
  requestId := doc.GetValueOrNull('id');
  sub := fServer.OpenSubscription(requestId, doc.GetValueOrNull('params'));
  if sub = nil then
  begin
    // At the cap: refuse rather than take the last worker thread. The stream
    // head is already out, so this has to travel as an SSE event.
    // It carries the subscription id like every other message on a listen
    // stream — in `data`, the only place a JSON-RPC error can hold it — so a
    // client demultiplexing several streams can still tell which one failed.
    aWrite(SseChunk(FormatSseEvent('message', fServer.Processor.CreateError(
      requestId, JSONRPC_INTERNAL_ERROR,
      'Too many concurrent subscriptions',
      _ObjFast(['_meta', _ObjFast([MCP_META_SUBSCRIPTION_ID, requestId])])))));
    exit;
  end;
  try
    // MUST be the first message on the stream, before any notification
    if not aWrite(SseChunk(FormatSseEvent('message',
        fServer.SubscriptionAcknowledgement(sub)))) then
      exit;
    idle := 0;
    alive := true;
    while alive and fActive do
    begin
      // Drain BEFORE testing for cancellation: an orderly shutdown must still
      // deliver what is already queued. Testing first would silently drop
      // notifications that were produced microseconds before the cancel.
      if sub.Drain(pending) then
      begin
        idle := 0;
        for i := 0 to high(pending) do
        begin
          alive := aWrite(SseChunk(FormatSseEvent('message', pending[i])));
          if not alive then
            break; // client hung up: closing the stream IS the cancellation
        end;
        continue; // more may have arrived while we were writing
      end;
      if sub.Cancelled then
        break; // nothing left to deliver, and we were asked to stop
      // A stream outlives a token easily: the one check at connect time was
      // minutes ago, and "take all necessary steps to ensure no data is
      // returned to unauthorized parties" does not stop applying because the
      // connection stayed open. A verifier that reports no expiry opts out.
      if (aAuthCtx.ExpiresUnix > 0) and
         (UnixTimeUtc >= aAuthCtx.ExpiresUnix) then
      begin
        sub.Cancel('the access token presented for this stream has expired');
        break;
      end;
      SleepHiRes(MCP_SUBSCRIPTION_POLL_MS);
      inc(idle, MCP_SUBSCRIPTION_POLL_MS);
      if idle < MCP_SUBSCRIPTION_KEEPALIVE_MS then
        continue;
      idle := 0;
      // an SSE comment line: keeps intermediaries and idle timeouts from
      // dropping a quiet stream, and is the only way we notice a client that
      // vanished without a FIN (the write then fails)
      alive := aWrite(SseChunk(':'#13#10));
    end;
    // Server-side teardown, in the order the spec asks for:
    // 1. "A server MUST send notifications/cancelled referencing a
    //    subscriptions/listen request ID when it tears down that subscription
    //    stream" — it names the reason, which is all the client has to decide
    //    whether reconnecting makes sense.
    // 2. The empty response to the long-lived request (a SHOULD) then closes
    //    the request itself: this ended on purpose rather than the connection
    //    dropping.
    // Reaching here with alive=true means WE ended it (cancelled, or the
    // transport went inactive) — a client that closed its own stream is gone
    // and gets neither, which is also what the spec expects: closing the
    // stream IS the client's cancellation, and needs no answer.
    if alive then
    begin
      aWrite(SseChunk(FormatSseEvent('message',
        fServer.SubscriptionCancelledNotification(sub))));
      aWrite(SseChunk(FormatSseEvent('message',
        fServer.SubscriptionEndResponse(sub))));
    end;
  finally
    fServer.CloseSubscription(sub);
  end;
end;

procedure TMcpStreamableHttpTransport.StreamDeferredResponse(
  const aWrite: TMcpRawWrite; const aBody, aOutHeaders: RawUtf8;
  const aAuthCtx: TMcpAuthContext);
var
  responseJson, scopeChallenge: RawUtf8;
  emitter: IMcpStreamEmitter;
  // kept as the concrete class, not the interface: the HeadWritten answer is
  // what decides whether a status may still be chosen below
  lazySink: TMcpLazyHeadSink;
  lazySinkRef: IMcpNotificationSink;
  handled, headWritten: boolean;
  status: integer;

  // the SSE response head: chunked, NO Content-Length. We deliberately keep the
  // connection alive (no 'Connection: close') so a client can reuse the socket
  // for subsequent requests — the terminating 0-chunk delimits this response.
  // aOutHeaders carries the CORS lines (each CRLF-terminated); the trailing
  // CRLF below ends the header block. X-Accel-Buffering keeps reverse proxies
  // from holding events back, which would defeat streaming.
  procedure WriteStreamHead;
  begin
    if headWritten then
      exit;
    headWritten := true;
    aWrite('HTTP/1.1 200 OK'#13#10 +
      'Content-Type: text/event-stream'#13#10 +
      'Cache-Control: no-cache'#13#10 +
      'X-Accel-Buffering: no'#13#10 +
      'Transfer-Encoding: chunked'#13#10 +
      aOutHeaders +
      #13#10);
  end;

begin
  headWritten := false;
  // subscriptions/listen is not a request/response: it keeps this stream open
  // and pushes notifications onto it until one side ends it.
  if McpMethodFromName(_Safe(_JsonFast(aBody))^.U['method']) =
       mcpSubscriptionsListen then
  begin
    WriteStreamHead;
    StreamSubscription(aWrite, aBody, aAuthCtx);
    aWrite('0'#13#10#13#10);
    exit;
  end;

  // The body is a single JSON-RPC request that mcp() already ran through
  // PreflightRequest — a malformed envelope, bad _meta, an unsupported version
  // or an unknown method never reaches this point, so the hook below always
  // sees a request the protocol layer accepted.
  handled := false;
  responseJson := '';
  // The hook is FOREIGN code and FinalizeHookResponse rejects a malformed
  // result — neither may escape into the connection's OnRead, which has no
  // handler and would tear down the worker mid-stream. ExecuteRequest already
  // catches everything itself; this guard covers the hook path.
  try
    if Assigned(fOnStreamCall) then
    begin
      // A hook may push intermediate token events, so its stream head has to go
      // out BEFORE it runs — which fixes the response at 200. That is the price
      // of streaming, and only a hook pays it.
      WriteStreamHead;
      emitter := TMcpStreamEmitter.Create(self, aWrite);
      handled := fOnStreamCall(aBody, emitter, aAuthCtx, responseJson);
      if handled then
        // a hook builds its response by hand and would otherwise ship a result
        // without the mandatory resultType, serverInfo and caching hints
        responseJson := fServer.FinalizeHookResponse(responseJson, aBody)
      else
        responseJson := fServer.ExecuteRequest(aBody, aAuthCtx, scopeChallenge);
    end
    else
    begin
      // No hook: we run the request FIRST and only then commit to a status.
      // That is what lets -32021 carry the 400 the spec requires — it depends
      // on what the handler turned out to need and cannot be known at
      // preflight time.
      // The handler still gets a progress sink: it writes the SSE head itself
      // on the first notification, so a handler that stays silent leaves the
      // status choice below untouched, while one that reports progress gets a
      // live stream instead of notifications buffered until the very end.
      lazySink := TMcpLazyHeadSink.Create(self, aWrite, aOutHeaders);
      // hold an interface reference for as long as we read HeadWritten below:
      // ExecuteRequest's own reference dies with the call, and on a refcounted
      // TInterfacedObject that would free the object under our feet
      lazySinkRef := lazySink;
      responseJson := fServer.ExecuteRequest(aBody, aAuthCtx, scopeChallenge,
        lazySinkRef);
      if lazySink.HeadWritten then
      begin
        // progress already went out: the head is on the wire, this IS a 200 SSE
        // stream now, and the rest of it is the response event
        headWritten := true;
        status := HTTP_MCP_SUCCESS;
      end
      else
        status := McpHttpStatus(responseJson);
      if scopeChallenge <> '' then
        // a handler refused for lack of scope: 403 with the challenge naming
        // what to ask for, which is what a client steps up with
        status := HTTP_MCP_FORBIDDEN;
      if status <> HTTP_MCP_SUCCESS then
      begin
        // A rejection is a buffered JSON body with its own status, never a
        // stream — exactly as SendProtocolError does for preflight failures.
        if scopeChallenge <> '' then
          // RFC 6750: the 403 MUST carry what the client has to ask for, or it
          // cannot step up and will simply retry the same failing request
          aWrite(FormatUtf8('HTTP/1.1 % %'#13#10 +
            'Content-Type: application/json'#13#10 +
            'WWW-Authenticate: %'#13#10 +
            'Content-Length: %'#13#10 +
            aOutHeaders + #13#10 + '%',
            [status, StatusCodeToText(status)^,
             fServer.AuthChallenge(mtrInsufficientScope, scopeChallenge),
             length(responseJson), responseJson]))
        else
          aWrite(FormatUtf8('HTTP/1.1 % %'#13#10 +
            'Content-Type: application/json'#13#10 +
            'Content-Length: %'#13#10 +
            aOutHeaders + #13#10 + '%',
            [status, StatusCodeToText(status)^, length(responseJson),
             responseJson]));
        exit;
      end;
      WriteStreamHead;
    end;
  except
    on E: Exception do
    begin
      // Once anything has been written we are committed to the stream, so an
      // escaped error can only be delivered as its final event. WriteStreamHead
      // is idempotent, and covers the no-hook path where nothing went out yet.
      WriteStreamHead;
      responseJson := fServer.Processor.CreateError(
        _Safe(_JsonFast(aBody))^.GetValueOrNull('id'), JSONRPC_INTERNAL_ERROR,
        StringToUtf8(E.Message));
    end;
  end;

  // final SSE event carrying the JSON-RPC response, which ends the stream
  if responseJson <> '' then
    aWrite(SseChunk(FormatSseEvent('message', responseJson)));

  // terminating zero-length chunk closes the chunked body
  aWrite('0'#13#10#13#10);
end;


end.
