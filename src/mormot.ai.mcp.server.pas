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
  // - return true if handled: push intermediate events via aEmitter and set
  //   aResponseJson to the final JSON-RPC response (or '' to send none)
  // - return false to let the transport process the request normally
  TMcpStreamCall = function(const aRequestJson: RawUtf8;
    const aEmitter: IMcpStreamEmitter; out aResponseJson: RawUtf8): boolean of object;

  /// Streamable HTTP transport implementing MCP 2026-07-28
  // - the endpoint accepts POST and OPTIONS only: GET (the standalone
  //   notification stream) and DELETE (session teardown) were removed with
  //   protocol sessions and are answered with 405
  // - POST responses always use SSE (text/event-stream) for requests; that
  //   stream is scoped to its request and is not resumable (no Last-Event-ID,
  //   no event ids) — a broken stream means the client re-issues the request
  // - stateless: no Mcp-Session-Id is minted, echoed or required
  // - supports JSON-RPC batch input (array of messages)
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
    /// verify the standard request headers against the JSON-RPC body
    // - MCP-Protocol-Version, Mcp-Method and (for tools/call, resources/read,
    //   prompts/get) Mcp-Name are REQUIRED and MUST match the body, so that an
    //   intermediary routing on headers cannot disagree with what we execute
    // - returns false and fills aErrorMsg on mismatch -> -32020 + HTTP 400
    function ValidateStandardHeaders(var Ctxt: THttpServerRequest;
      const aBody: RawUtf8; out aErrorMsg: RawUtf8): boolean;
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
      const aBody, aOutHeaders: RawUtf8);
    // send a JSON-RPC error as a plain buffered JSON response with an explicit
    // HTTP status — used for everything the protocol layer rejects up front
    function SendProtocolError(var Ctxt: THttpServerRequest;
      const aErrorJson: RawUtf8; aStatus: integer): cardinal;
    // -- GET/DELETE are gone with protocol sessions: answer 405 --
    function OnMethodNotAllowed(Ctxt: THttpServerRequestAbstract): cardinal;
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
    property OnStreamCall: TMcpStreamCall read fOnStreamCall write fOnStreamCall;
  published
    /// single endpoint handler — routes by HTTP method
    // - uses RTTI-based route publishing (same pattern as TMcpHttpTransport)
    function mcp(Ctxt: THttpServerRequest): cardinal;
  end;


implementation

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
    'Access-Control-Allow-Headers: Content-Type' + #13#10 +
    'Access-Control-Max-Age: 86400' + #13#10;
end;

function TMcpHttpTransport.mcp(ctxt: THttpServerRequest): cardinal;
var
  requestBody, responseBody: RawUtf8;
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

  // Read request body
  requestBody := Ctxt.InContent;
  
  // Execute MCP request
  responseBody := fServer.ExecuteRequest(requestBody);
  
  // Send response
  if responseBody = '' then
    exit(HTTP_NOCONTENT);

  result := Ctxt.SetOutJson(responseBody);
end;

procedure TMcpHttpTransport.Start;
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
    transport.StreamDeferredResponse(WriteRaw, fHttp.Content,
      fRequest.OutCustomHeaders);
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
  bind: RawUtf8;
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
  fHttpServer.WaitStarted;
  fActive := true;
end;

procedure TMcpStreamableHttpTransport.Stop;
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
  if not ValidateStandardHeaders(Ctxt, body, headerError) then
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
    fServer.ExecuteRequest(body);
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
    'Access-Control-Allow-Headers: Content-Type, MCP-Protocol-Version, ' +
      'Mcp-Method, Mcp-Name' + #13#10 +
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

function TMcpStreamableHttpTransport.ValidateStandardHeaders(
  var Ctxt: THttpServerRequest; const aBody: RawUtf8;
  out aErrorMsg: RawUtf8): boolean;
var
  doc: TDocVariantData;
  params: PDocVariantData;
  method, name, bodyName: RawUtf8;

  // read a custom request header (mORMot keeps non-standard headers in InHeaders)
  function Header(const aName: RawUtf8): RawUtf8;
  var
    p: PUtf8Char;
    len: PtrInt;
  begin
    result := '';
    p := FindNameValuePointer(pointer(Ctxt.InHeaders), pointer(aName + ': '), len);
    if p = nil then
      p := FindNameValuePointer(pointer(Ctxt.InHeaders), pointer(aName + ':'), len);
    if p <> nil then
      FastSetString(result, p, len);
  end;

  // a value that is not header-safe travels as '=?base64?<b64>?=' — the server
  // MUST decode it before comparing, otherwise every non-ASCII tool name would
  // look like a mismatch
  // - the markers are CASE-SENSITIVE per spec ("MUST appear exactly as shown
  //   (lowercase)"): a case-insensitive match would decode '=?BASE64?x?=',
  //   which is a literal name a client is required to send base64-encoded —
  //   so accepting it would silently rewrite a legitimate value
  function DecodeSentinel(const aValue: RawUtf8): RawUtf8;
  begin
    result := aValue;
    if (length(aValue) > 11) and
       (copy(aValue, 1, 9) = '=?base64?') and
       (copy(aValue, length(aValue) - 1, 2) = '?=') then
      result := Base64ToBin(copy(aValue, 10, length(aValue) - 11));
  end;

begin
  result := false;
  doc.InitJson(aBody, JSON_FAST);

  // MCP-Protocol-Version: REQUIRED, and must equal the _meta value. We only
  // check presence/equality here — whether we *speak* that version is decided
  // centrally in ValidateRequestMeta (-32022), not per transport.
  name := Header('MCP-PROTOCOL-VERSION');
  if name = '' then
  begin
    aErrorMsg := 'Missing required header MCP-Protocol-Version';
    exit;
  end;
  // Compare against the body — but only for requests. The spec leaves header
  // requirements for notification POSTs undefined, and a notification carries
  // no _meta to compare against.
  if not VarIsVoid(doc.GetValueOrNull('id')) then
  begin
    bodyName := '';
    if doc.GetAsDocVariant('params', params) and params^.IsObject then
      if params^.GetAsDocVariant('_meta', params) and params^.IsObject then
        bodyName := params^.U[MCP_META_PROTOCOL_VERSION];
    // An absent body value is a mismatch too, not a free pass: leaving the
    // header unchecked is exactly the split-source-of-truth the -32020 rule
    // exists to prevent (a proxy routes on the header, we execute the body).
    if bodyName <> name then
    begin
      aErrorMsg := 'Header mismatch: MCP-Protocol-Version header value ''' +
        name + ''' does not match request body value ''' + bodyName + '''';
      exit;
    end;
  end;

  // Mcp-Method: REQUIRED on all requests, mirrors "method"
  method := Header('MCP-METHOD');
  if method = '' then
  begin
    aErrorMsg := 'Missing required header Mcp-Method';
    exit;
  end;
  if method <> doc.U['method'] then
  begin
    aErrorMsg := 'Header mismatch: Mcp-Method header value ''' + method +
      ''' does not match body value ''' + doc.U['method'] + '''';
    exit;
  end;

  // Mcp-Name: REQUIRED for the three name-carrying methods, mirroring
  // params.name (tools/call, prompts/get) or params.uri (resources/read)
  if (method = 'tools/call') or
     (method = 'resources/read') or
     (method = 'prompts/get') then
  begin
    bodyName := '';
    if doc.GetAsDocVariant('params', params) and params^.IsObject then
      if method = 'resources/read' then
        bodyName := params^.U['uri']
      else
        bodyName := params^.U['name'];
    name := DecodeSentinel(Header('MCP-NAME'));
    if name = '' then
    begin
      aErrorMsg := 'Missing required header Mcp-Name for ' + method;
      exit;
    end;
    if name <> bodyName then
    begin
      aErrorMsg := 'Header mismatch: Mcp-Name header value ''' + name +
        ''' does not match body value ''' + bodyName + '''';
      exit;
    end;
  end;

  result := true;
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
  TMcpStreamEmitter = class(TInterfacedObject, IMcpStreamEmitter)
  protected
    fTransport: TMcpStreamableHttpTransport;
    fWrite: TMcpRawWrite;
  public
    constructor Create(aTransport: TMcpStreamableHttpTransport;
      const aWrite: TMcpRawWrite);
    procedure Emit(const aJsonMessage: RawUtf8);
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

procedure TMcpStreamableHttpTransport.StreamDeferredResponse(
  const aWrite: TMcpRawWrite; const aBody, aOutHeaders: RawUtf8);
var
  responseJson: RawUtf8;
  emitter: IMcpStreamEmitter;
  handled: boolean;
begin
  // HTTP response head: chunked, NO Content-Length. We deliberately keep the
  // connection alive (no 'Connection: close') so a client can reuse the socket
  // for subsequent requests — the terminating 0-chunk delimits this response.
  // aOutHeaders carries the CORS lines (each CRLF-terminated); the trailing
  // CRLF below ends the header block. X-Accel-Buffering keeps reverse proxies
  // from holding events back, which would defeat streaming.
  aWrite('HTTP/1.1 200 OK'#13#10 +
    'Content-Type: text/event-stream'#13#10 +
    'Cache-Control: no-cache'#13#10 +
    'X-Accel-Buffering: no'#13#10 +
    'Transfer-Encoding: chunked'#13#10 +
    aOutHeaders +
    #13#10);

  // shared emitter so a streaming tool can push intermediate token events
  emitter := TMcpStreamEmitter.Create(self, aWrite);

  // The body is a single JSON-RPC request that mcp() already ran through
  // PreflightRequest — a malformed envelope, bad _meta, an unsupported version
  // or an unknown method never reaches this point, so the hook below always
  // sees a request the protocol layer accepted.
  // Let a streaming hook handle it first — it pushes token events through the
  // emitter and supplies the final response; otherwise process normally.
  handled := false;
  responseJson := '';
  // The hook is FOREIGN code and FinalizeResponseJson rejects a malformed
  // result — neither may escape into the connection's OnRead, which has no
  // handler and would tear down the worker mid-stream. ExecuteRequest already
  // catches everything itself; this guard covers the hook path.
  try
    if Assigned(fOnStreamCall) then
      handled := fOnStreamCall(aBody, emitter, responseJson);
    if handled then
      // a hook builds its response by hand and would otherwise ship a result
      // without the mandatory resultType and without serverInfo
      responseJson := fServer.Processor.FinalizeResponseJson(responseJson)
    else
      responseJson := fServer.ExecuteRequest(aBody);
  except
    on E: Exception do
      responseJson := fServer.Processor.CreateError(
        _Safe(_JsonFast(aBody))^.GetValueOrNull('id'), JSONRPC_INTERNAL_ERROR,
        StringToUtf8(E.Message));
  end;

  // final SSE event carrying the JSON-RPC response, which ends the stream
  if responseJson <> '' then
    aWrite(SseChunk(FormatSseEvent('message', responseJson)));

  // terminating zero-length chunk closes the chunked body
  aWrite('0'#13#10#13#10);
end;


end.
