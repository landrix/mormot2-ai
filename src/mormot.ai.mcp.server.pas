/// MCP Transport Layer - HTTP and SSE Implementations
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
    - SSE Transport with Session Management

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
  mormot.core.collections,
  mormot.core.threads,
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


{ ************ SSE Transport with Session Management }

type
  /// SSE session information
  TMcpSseSession = class
  public
    /// unique session identifier
    SessionId: RawUtf8;
    /// async connection handle for the active SSE stream (0 if none)
    ConnectionHandle: TConnectionAsyncHandle;
    /// message queue for this session (tofo: migrate to IKeyValue)
    MessageQueue: TRawUtf8DynArray;
    /// lock for thread-safe queue access
    QueueLock: TLightLock;
    /// last activity timestamp
    LastActivity: TDateTime;
    /// initialize session
    constructor Create(const aSessionId: RawUtf8);
    /// cleanup
    destructor Destroy; override;
    /// enqueue a message
    procedure EnqueueMessage(const aMessage: RawUtf8);
    /// dequeue all messages
    function DequeueMessages: TRawUtf8DynArray;
  end;

  /// SSE transport with session management
  // - uses Server-Sent Events for streaming responses
  // - maintains sessions for async communication
  TMcpSseTransport = class(TMcpTransportBase)
  private
    fHttpServer: THttpAsyncServer;
    fSessions: IKeyValue<RawUtf8, TMcpSseSession>;  // sessionId -> TMcpSseSession
    fSseEndpoint: RawUtf8;
    fMessagesEndpoint: RawUtf8;
    fCorsEnabled: boolean;
    fCorsOrigins: RawUtf8;
    function MatchesSseEndpoint(const aUrl: RawUtf8): boolean;
    function AcceptsEventStream(const aHeaders: RawUtf8): boolean;
    function BuildSseHeaders: RawUtf8;
    function BuildSseEvent(const aEvent, aData: RawUtf8): RawUtf8;
    function BuildSseChunk(const aData: RawUtf8): RawUtf8;
    function BuildSseResponse(const aSessionId: RawUtf8;
      const aQueued: TRawUtf8DynArray): RawUtf8;
    function TrySendSessionMessage(aSession: TMcpSseSession;
      const aMessage: RawUtf8): boolean;
    procedure AttachSessionConnection(const aSessionId: RawUtf8;
      aHandle: TConnectionAsyncHandle);
    procedure DetachSessionConnection(const aSessionId: RawUtf8);
    procedure RemoveSession(const aSessionId: RawUtf8);
    procedure ClearSessions;
    function OnSseConnect(Ctxt: THttpServerRequestAbstract): cardinal;
    function OnPostMessage(Ctxt: THttpServerRequestAbstract): cardinal;
    function OnSseOptions(Ctxt: THttpServerRequestAbstract): cardinal;
    function OnMessagesOptions(Ctxt: THttpServerRequestAbstract): cardinal;

    procedure SetCorsHeaders(var Ctxt: THttpServerRequestAbstract);
    function GetOrCreateSession(const aSessionId: RawUtf8): TMcpSseSession;
    procedure CleanupExpiredSessions;
  public
    /// initialize SSE transport
    constructor Create(aServer: TMcpServer); override;
    /// finalize and cleanup
    destructor Destroy; override;
    /// start SSE server
    procedure Start; override;
    /// stop SSE server
    procedure Stop; override;
    /// push notification to a specific session
    procedure PushNotification(const aSessionId, aNotification: RawUtf8);
    /// SSE endpoint path (default: '/sse')
    property SseEndpoint: RawUtf8 read fSseEndpoint write fSseEndpoint;
    /// messages endpoint path (default: '/messages')
    property MessagesEndpoint: RawUtf8 read fMessagesEndpoint write fMessagesEndpoint;
    /// enable/disable CORS support
    property CorsEnabled: boolean read fCorsEnabled write fCorsEnabled;
    /// allowed CORS origins
    property CorsOrigins: RawUtf8 read fCorsOrigins write fCorsOrigins;
  end;


{ ************ Streamable HTTP Transport (MCP 2025-03-26) }

type
  /// Session state for the Streamable HTTP transport
  // - tracks session lifecycle, activity, and event counter for resumability stub
  TMcpStreamableSession = class
  public
    /// unique session identifier (UUID without braces)
    SessionId: RawUtf8;
    /// whether the initialize handshake has completed
    Initialized: boolean;
    /// last activity timestamp for expiration via DeleteDeprecated
    LastActivity: TDateTime;
    /// per-session event counter for SSE event IDs (resumability stub)
    LastEventId: Int64;
    /// initialize session with a given ID
    constructor Create(const aSessionId: RawUtf8);
    destructor Destroy; override;
    /// update LastActivity to now
    procedure Touch;
    /// increment and return the next event ID
    function NextEventId: Int64;
  end;

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
  // - aRequestJson is a single JSON-RPC request; aSessionId the MCP session
  // - return true if handled: push intermediate events via aEmitter and set
  //   aResponseJson to the final JSON-RPC response (or '' to send none)
  // - return false to let the transport process the request normally
  TMcpStreamCall = function(const aRequestJson, aSessionId: RawUtf8;
    const aEmitter: IMcpStreamEmitter; out aResponseJson: RawUtf8): boolean of object;

  /// Streamable HTTP transport implementing MCP 2025-03-26
  // - single endpoint handles POST, GET, DELETE, and OPTIONS
  // - POST responses always use SSE (text/event-stream) for requests
  // - session management via Mcp-Session-Id header
  // - supports JSON-RPC batch input (array of messages)
  // - resumability stubbed: SSE event IDs assigned but no replay
  {$M+}
  TMcpStreamableHttpTransport = class(TMcpTransportBase)
  private
    fHttpServer: THttpAsyncServer;
    fSessions: IKeyValue<RawUtf8, TMcpStreamableSession>;
    FSafe: IAutoLocker;
    fEndpoint: RawUtf8;
    fCorsEnabled: boolean;
    fCorsOrigins: RawUtf8;
    fOnStreamCall: TMcpStreamCall;
    // -- helper methods --
    procedure SetCorsHeaders(var Ctxt: THttpServerRequest);
    function ValidateOrigin(var Ctxt: THttpServerRequest): boolean;
    function ExtractSessionId(var Ctxt: THttpServerRequest): RawUtf8;
    function GetOrCreateSession(const aSessionId: RawUtf8): TMcpStreamableSession;
    procedure RemoveSession(const aSessionId: RawUtf8);
    procedure ClearSessions;
    // -- SSE formatting --
    function FormatSseEvent(const aEvent, aData: RawUtf8; aId: Int64): RawUtf8;
    // wrap a payload as one HTTP/1.1 chunked-transfer frame (hex-len CRLF .. CRLF)
    function SseChunk(const aPayload: RawUtf8): RawUtf8;
    // Stream the chunked text/event-stream response for a deferred POST: writes
    // the HTTP head, then one SSE event per JSON-RPC response (calling
    // OnStreamCall first so a tool can push intermediate token events), then the
    // terminating 0-chunk. All writes go through aWrite (the connection's raw
    // socket writer) so they flush incrementally. Called from
    // TMcpStreamableAsyncConnection.OnRead after the handler returned
    // HTTP_ASYNCRESPONSE. aBody is the request body; aOutHeaders carries the
    // CORS + Mcp-Session-Id lines the handler prepared.
    procedure StreamDeferredResponse(const aWrite: TMcpRawWrite;
      const aBody, aOutHeaders: RawUtf8);
    // -- protocol version patching --
    function PatchProtocolVersion(const aJson: RawUtf8): RawUtf8;
    // -- explicit route callback for DELETE (TOnHttpServerRequest signature) --
    function OnDelete(Ctxt: THttpServerRequestAbstract): cardinal;
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
  TMcpSseAsyncServer = class(THttpAsyncServer)
  public
    Transport: TMcpSseTransport;
    constructor Create(const aPort: RawUtf8; const OnStart, OnStop: TOnNotifyThread;
      const ProcessName: RawUtf8; ServerThreadPoolCount: integer = 32;
      KeepAliveTimeOut: integer = 30000; ProcessOptions: THttpServerOptions = [];
      aLog: TSynLogClass = nil); override;
  end;

  TMcpSseAsyncConnection = class(THttpAsyncServerConnection)
  protected
    fSessionId: RawUtf8;
    function DecodeHeaders: integer; override;
    function OnRead: TPollAsyncSocketOnReadWrite; override;
    function AfterWrite: TPollAsyncSocketOnReadWrite; override;
    procedure OnClose; override;
  end;

  // Custom async server/connection for the Streamable HTTP transport.
  // The published mcp() handler defers request responses via HTTP_ASYNCRESPONSE
  // (so THttpAsyncServer does NOT generate a buffered response); this connection
  // then writes a real chunked text/event-stream and lets AfterWrite finalize.
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

{ ************ TMcpSseAsyncServer / TMcpSseAsyncConnection }

constructor TMcpSseAsyncServer.Create(const aPort: RawUtf8;
  const OnStart, OnStop: TOnNotifyThread; const ProcessName: RawUtf8;
  ServerThreadPoolCount: integer; KeepAliveTimeOut: integer;
  ProcessOptions: THttpServerOptions; aLog: TSynLogClass);
begin
  fConnectionClass := TMcpSseAsyncConnection; // must be set before inherited
  inherited Create(aPort, OnStart, OnStop, ProcessName, ServerThreadPoolCount,
    KeepAliveTimeOut, ProcessOptions, aLog);
end;

function TMcpSseAsyncConnection.DecodeHeaders: integer;
var
  serv: TMcpSseAsyncServer;
  transport: TMcpSseTransport;
  session: TMcpSseSession;
  sessionId: RawUtf8;
  queued: TRawUtf8DynArray;
  response: RawUtf8;
begin
  result := inherited DecodeHeaders;
  if result <> HTTP_SUCCESS then
    exit;

  serv := fServer as TMcpSseAsyncServer;
  transport := serv.Transport;
  if (transport = nil) or
     (fHttp.CommandMethod <> 'GET') or
     not transport.MatchesSseEndpoint(fHttp.CommandUri) or
     not transport.AcceptsEventStream(fHttp.Headers) then
    exit;

  sessionId := ToUtf8(RandomGuid);
  session := transport.GetOrCreateSession(sessionId);
  transport.AttachSessionConnection(sessionId, Handle);
  fSessionId := sessionId;

  queued := session.DequeueMessages;
  response := transport.BuildSseResponse(sessionId, queued);

  fHttp.State := hrsUpgraded;
  include(fInternalFlags, ifSeparateWLock); // allow async writes while idle
  if not fOwner.WriteString(self, response, 1000) then
    result := HTTP_BADREQUEST;
end;

function TMcpSseAsyncConnection.OnRead: TPollAsyncSocketOnReadWrite;
begin
  if fHttp.State <> hrsUpgraded then
    result := inherited OnRead
  else
  begin
    fRd.Reset; // ignore any input on SSE stream
    result := soContinue;
  end;
end;

function TMcpSseAsyncConnection.AfterWrite: TPollAsyncSocketOnReadWrite;
begin
  if fHttp.State <> hrsUpgraded then
    result := inherited AfterWrite
  else
    result := soContinue;
end;

procedure TMcpSseAsyncConnection.OnClose;
var
  serv: TMcpSseAsyncServer;
begin
  inherited OnClose;
  if fSessionId = '' then
    exit;
  serv := fServer as TMcpSseAsyncServer;
  if serv.Transport <> nil then
    serv.Transport.DetachSessionConnection(fSessionId);
  fSessionId := '';
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
  responseBody := fServer.ExecuteRequest(requestBody, '');
  
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


{ ************ TMcpSseSession }

constructor TMcpSseSession.Create(const aSessionId: RawUtf8);
begin
  inherited Create;
  SessionId := aSessionId;
  ConnectionHandle := 0;
  QueueLock.Init;
  LastActivity := NowUtc;
end;

destructor TMcpSseSession.Destroy;
begin
  QueueLock.Done;
  inherited;
end;

procedure TMcpSseSession.EnqueueMessage(const aMessage: RawUtf8);
begin
  QueueLock.Lock;
  try
    AddRawUtf8(MessageQueue, aMessage);
    LastActivity := NowUtc;
  finally
    QueueLock.UnLock;
  end;
end;

function TMcpSseSession.DequeueMessages: TRawUtf8DynArray;
begin
  QueueLock.Lock;
  try
    result := MessageQueue;
    MessageQueue := nil;
    LastActivity := NowUtc;
  finally
    QueueLock.UnLock;
  end;
end;


{ ************ TMcpSseTransport }

constructor TMcpSseTransport.Create(aServer: TMcpServer);
begin
  inherited Create(aServer);
  fSseEndpoint := '/sse';
  fMessagesEndpoint := '/messages';
  fCorsEnabled := true;
  fCorsOrigins := '*';
  fSessions := Collections.NewPlainKeyValue<RawUtf8, TMcpSseSession>;{(
    [kvoThreadSafe], 60);}
end;

destructor TMcpSseTransport.Destroy;
begin
  Stop;
  ClearSessions;
  fSessions := nil;
  inherited;
end;

procedure TMcpSseTransport.SetCorsHeaders(var Ctxt: THttpServerRequestAbstract);
begin
  if not fCorsEnabled then
    exit;
    
  Ctxt.OutCustomHeaders := Ctxt.OutCustomHeaders +
    'Access-Control-Allow-Origin: ' + fCorsOrigins + #13#10 +
    'Access-Control-Allow-Methods: POST, GET, OPTIONS' + #13#10 +
    'Access-Control-Allow-Headers: Content-Type' + #13#10;
end;

function TMcpSseTransport.MatchesSseEndpoint(const aUrl: RawUtf8): boolean;
var
  len: integer;
begin
  result := false;
  len := length(fSseEndpoint);
  if (len = 0) or (aUrl = '') then
    exit;
  if length(aUrl) = len then
    result := (aUrl = fSseEndpoint)
  else if (length(aUrl) > len) and
          (CompareMem(pointer(aUrl), pointer(fSseEndpoint), len)) and
          (aUrl[len + 1] = '?') then
    result := true;
end;

function TMcpSseTransport.AcceptsEventStream(const aHeaders: RawUtf8): boolean;
var
  p: PUtf8Char;
  len: PtrInt;
  value: RawUtf8;
begin
  result := true; // be lenient if Accept header is missing
  if aHeaders = '' then
    exit;
  p := FindNameValuePointer(pointer(aHeaders), 'ACCEPT: ', len);
  if p = nil then
    p := FindNameValuePointer(pointer(aHeaders), 'ACCEPT:', len);
  if p = nil then
    exit;
  FastSetString(value, p, len);
  value := LowerCaseU(value);
  result := PosEx('text/event-stream', value) > 0;
end;

function TMcpSseTransport.BuildSseHeaders: RawUtf8;
begin
  result :=
    'HTTP/1.1 200 OK'#13#10 +
    'Content-Type: text/event-stream'#13#10 +
    'Cache-Control: no-cache'#13#10 +
    'Connection: keep-alive'#13#10 +
    'X-Accel-Buffering: no'#13#10 +
    'Transfer-Encoding: chunked'#13#10;
  if fCorsEnabled then
    result := result +
      'Access-Control-Allow-Origin: ' + fCorsOrigins + #13#10 +
      'Access-Control-Allow-Methods: POST, GET, OPTIONS' + #13#10 +
      'Access-Control-Allow-Headers: Content-Type' + #13#10;
  result := result + #13#10;
end;

function TMcpSseTransport.BuildSseEvent(const aEvent, aData: RawUtf8): RawUtf8;
var
  i, start: integer;
begin
  result := '';
  if aEvent <> '' then
    result := 'event: ' + aEvent + #13#10;
  if aData = '' then
    result := result + 'data:' + #13#10
  else
  begin
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
  result := result + #13#10;
end;

function TMcpSseTransport.BuildSseChunk(const aData: RawUtf8): RawUtf8;
begin
  result := StringToUtf8(IntToHex(length(aData), 1)) + #13#10 +
    aData + #13#10;
end;

function TMcpSseTransport.BuildSseResponse(const aSessionId: RawUtf8;
  const aQueued: TRawUtf8DynArray): RawUtf8;
var
  i: integer;
  endpointUrl: RawUtf8;
  payload: RawUtf8;
begin
  result := BuildSseHeaders;
  endpointUrl := FormatUtf8('%?session_id=%', [fMessagesEndpoint, aSessionId]);
  payload := BuildSseEvent('endpoint', endpointUrl);
  result := result + BuildSseChunk(payload);
  for i := 0 to length(aQueued) - 1 do
  begin
    payload := BuildSseEvent('message', aQueued[i]);
    result := result + BuildSseChunk(payload);
  end;
end;

function TMcpSseTransport.TrySendSessionMessage(aSession: TMcpSseSession;
  const aMessage: RawUtf8): boolean;
var
  handle: TConnectionAsyncHandle;
  conn: TAsyncConnection;
  payload: RawUtf8;
begin
  result := false;
  if (aSession = nil) or (fHttpServer = nil) then
    exit;
  aSession.QueueLock.Lock;
  try
    handle := aSession.ConnectionHandle;
  finally
    aSession.QueueLock.UnLock;
  end;
  if handle = 0 then
    exit;
  conn := TAsyncConnection(fHttpServer.Async.ConnectionFindAndWaitLock(handle, true, 40));
  if conn = nil then
    exit;
  try
    payload := BuildSseChunk(BuildSseEvent('message', aMessage));
    result := fHttpServer.Async.WriteString(conn, payload, 1000);
  finally
    conn.UnLock(true);
  end;
end;

procedure TMcpSseTransport.AttachSessionConnection(const aSessionId: RawUtf8;
  aHandle: TConnectionAsyncHandle);
var
  session: TMcpSseSession;
begin
  if fSessions.TryGetValue(aSessionId, session) then
  begin
    session.QueueLock.Lock;
    try
      session.ConnectionHandle := aHandle;
      session.LastActivity := NowUtc;
    finally
      session.QueueLock.UnLock;
    end;
  end;
end;

procedure TMcpSseTransport.DetachSessionConnection(const aSessionId: RawUtf8);
var
  session: TMcpSseSession;
begin
  if fSessions.TryGetValue(aSessionId, session) then
  begin
    session.QueueLock.Lock;
    try
      session.ConnectionHandle := 0;
      session.LastActivity := NowUtc;
    finally
      session.QueueLock.UnLock;
    end;
  end;
end;

procedure TMcpSseTransport.RemoveSession(const aSessionId: RawUtf8);
var
  session: TMcpSseSession;
begin
  if fSessions.Extract(aSessionId, session) then
    session.Free;
end;

procedure TMcpSseTransport.ClearSessions;
begin
  if fSessions = nil then
    exit;
  // Note: TMcpSseSession objects are not freed here (pre-existing behavior).
  // IKeyValue.Clear zeroes the references but does not call .Free on TObject values.
  // A proper fix would require tracking and freeing sessions, but the async
  // connection lifecycle makes this non-trivial — sessions may still be
  // referenced by active connections during shutdown.
  fSessions.Clear;
end;

function TMcpSseTransport.GetOrCreateSession(const aSessionId: RawUtf8): TMcpSseSession;
var
  session: TMcpSseSession;
begin
  if not fSessions.TryGetValue(aSessionId, result) then
  begin
    session := TMcpSseSession.Create(aSessionId);
    fSessions.Add(aSessionId, session);
    result := session;
  end;
end;

procedure TMcpSseTransport.CleanupExpiredSessions;
begin
  fSessions.DeleteDeprecated;
end;

function TMcpSseTransport.OnSseConnect(Ctxt: THttpServerRequestAbstract): cardinal;
var
  sessionId, endpointUrl: RawUtf8;
  handshake: RawUtf8;
begin
  // Set SSE headers
  SetCorsHeaders(Ctxt);
  Ctxt.OutContentType := 'text/event-stream';
  Ctxt.OutCustomHeaders := Ctxt.OutCustomHeaders +
    'Cache-Control: no-cache' + #13#10 +
    'Connection: keep-alive' + #13#10;

  // Generate session ID
  sessionId := ToUtf8(RandomGuid);
  
  // Create session
  GetOrCreateSession(sessionId);
  
  // Send endpoint handshake
  endpointUrl := FormatUtf8('%?session_id=%', [fMessagesEndpoint, sessionId]);
  handshake := FormatUtf8('event: endpoint'#13#10'data: %'#13#10#13#10, [endpointUrl]);
  
  Ctxt.OutContent := handshake;
  Ctxt.RespStatus := HTTP_SUCCESS;
  result := HTTP_SUCCESS;
end;

function TMcpSseTransport.OnPostMessage(Ctxt: THttpServerRequestAbstract): cardinal;
var
  sessionId, requestBody, responseBody: RawUtf8;
  session: TMcpSseSession;
begin
  // Set CORS headers
  SetCorsHeaders(Ctxt);

  // Get session ID from query (accept both session_id and sessionId)
  Ctxt.UrlParam('SESSION_ID=', sessionId);
  if sessionId = '' then
    Ctxt.UrlParam('SESSIONID=', sessionId);
  if sessionId = '' then
  begin
    Ctxt.SetOutJson('{"error":"Missing session_id"}');
    exit(HTTP_BADREQUEST);
  end;

  // Find session
  if not fSessions.TryGetValue(sessionId, session) then
  begin
    Ctxt.SetOutJson('{"error":"Session not found"}');
    exit(HTTP_NOTFOUND);
  end;

  // Read request
  requestBody := Ctxt.InContent;
  
  // Execute MCP request
  responseBody := fServer.ExecuteRequest(requestBody, sessionId);
  
  // Send or enqueue response for SSE delivery
  if responseBody <> '' then
    if not TrySendSessionMessage(session, responseBody) then
      session.EnqueueMessage(responseBody);
  
  // Send accepted response
  Ctxt.SetOutJson('{"status":"accepted"}');
  result := HTTP_ACCEPTED;
end;

function TMcpSseTransport.OnSseOptions(Ctxt: THttpServerRequestAbstract): cardinal;
begin
  SetCorsHeaders(Ctxt);
  result := HTTP_NOCONTENT;
end;

function TMcpSseTransport.OnMessagesOptions(Ctxt: THttpServerRequestAbstract): cardinal;
begin
  SetCorsHeaders(Ctxt);
  result := HTTP_NOCONTENT;
end;

procedure TMcpSseTransport.PushNotification(const aSessionId, aNotification: RawUtf8);
var
  session: TMcpSseSession;
begin
  if fSessions.TryGetValue(aSessionId, session) then
    if not TrySendSessionMessage(session, aNotification) then
      session.EnqueueMessage(aNotification);
end;

procedure TMcpSseTransport.Start;
begin
  if fActive then
    exit;

  // Create and start HTTP server
  fHttpServer := TMcpSseAsyncServer.Create(
    ToUtf8(fPort), nil, nil, 'mcp', 32,
    5 * 60 * 1000,         // 5 minutes keep alive connections
    [hsoNoXPoweredHeader,  // not needed for a benchmark
     //hsoHeadersInterning,  // reduce memory contention for /plaintext and /json
     hsoNoStats,           // disable low-level statistic counters
     //hsoThreadCpuAffinity, // worse scaling on multi-servers
     hsoThreadSmooting,    // seems a good option, even if not magical
     {$ifdef WITH_LOGS}
     hsoLogVerbose,
     {$endif WITH_LOGS}
     hsoIncludeDateHeader  // required by TFB General Test Requirements #5
    ]);
  fHttpServer.ServerName := 'MMCP-SSE';

  TMcpSseAsyncServer(fHttpServer).Transport := self;

  // Register endpoints
  fHttpServer.Route.Get(fSseEndpoint, OnSseConnect);
  fHttpServer.Route.Post(fMessagesEndpoint, OnPostMessage);
  fHttpServer.Route.Options(fSseEndpoint, OnSseOptions);
  fHttpServer.Route.Options(fMessagesEndpoint, OnMessagesOptions);

  fHttpServer.WaitStarted;
  
  fActive := true;
end;

procedure TMcpSseTransport.Stop;
begin
  if not fActive then
    exit;
    
  if fHttpServer <> nil then
  begin
    fHttpServer.Shutdown;
    FreeAndNil(fHttpServer);
  end;
  
  // Clear sessions
  ClearSessions;
  
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
  FSafe := TAutoLocker.Create;
  fEndpoint := '/mcp';
  fCorsEnabled := true;
  fCorsOrigins := '*';
  // thread-safe session map with 30-minute expiration
  fSessions := Collections.NewPlainKeyValue<RawUtf8, TMcpStreamableSession>(
    [kvoThreadCriticalSection, kvoThreadSafe], 30 * 60);
end;

destructor TMcpStreamableHttpTransport.Destroy;
begin
  Stop;
  ClearSessions;
  fSessions := nil;
  inherited;
end;

procedure TMcpStreamableHttpTransport.ClearSessions;
begin
  if fSessions = nil then
    exit;
  // The IKeyValue owns its object values (no kvoValueNoFinalize), so Clear
  // frees every TMcpStreamableSession instance — no leak, no manual iteration.
  fSessions.Clear;
end;

procedure TMcpStreamableHttpTransport.Start;
begin
  if fActive then
    exit;
  fHttpServer := TMcpStreamableAsyncServer.Create(
    ToUtf8(fPort), nil, nil, 'mcp-streamable', 32,
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
  // DELETE must be registered explicitly — RunMethods does not route DELETE
  // to published methods. OnDelete delegates to mcp() via THttpServerRequestAbstract.
  fHttpServer.Route.Delete(fEndpoint, OnDelete);
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

function TMcpStreamableHttpTransport.mcp(Ctxt: THttpServerRequest): cardinal;
var
  sessionId, body, contentType: RawUtf8;
  session: TMcpStreamableSession;
  doc: TDocVariantData;
  singleItem: variant;
  item: PDocVariantData;
  i: PtrInt;
  hasRequest, isInitialize: boolean;
  method: RawUtf8;
  itemJson: RawUtf8;
begin
  // --- CORS headers on every response ---
  SetCorsHeaders(Ctxt);

  // --- OPTIONS: CORS preflight ---
  if Ctxt.Method = 'OPTIONS' then
    exit(HTTP_NOCONTENT);

  // --- GET: stubbed, return 405 ---
  if Ctxt.Method = 'GET' then
    exit(HTTP_NOTALLOWED);

  // --- Origin validation (DNS rebinding protection) ---
  if not ValidateOrigin(Ctxt) then
    exit(HTTP_FORBIDDEN);

  // --- DELETE: session termination ---
  if Ctxt.Method = 'DELETE' then
  begin
    sessionId := ExtractSessionId(Ctxt);
    if sessionId = '' then
      exit(HTTP_BADREQUEST);
    if not fSessions.ContainsKey(sessionId) then
      exit(HTTP_NOTFOUND);
    RemoveSession(sessionId);
    exit(HTTP_SUCCESS);
  end;

  // --- Only POST from here ---
  if Ctxt.Method <> 'POST' then
    exit(HTTP_NOTALLOWED);

  // --- Validate Content-Type ---
  // mORMot parses Content-Type out of headers into Ctxt.InContentType
  contentType := LowerCaseU(Ctxt.InContentType);
  if (contentType = '') or
     (PosEx('application/json', contentType) = 0) then
    exit(415); // Unsupported Media Type

  // --- Parse body: detect batch (array) vs single message (object) ---
  // Use InitJson (not InitJsonInPlace) because InContent may be a shared
  // reference-counted string — modifying it in-place causes EInvalidPointer.
  body := Ctxt.InContent;
  doc.InitJson(body, JSON_FAST);

  // Normalize: wrap single message in an array for uniform processing.
  // The transport parses batches itself and delegates individual messages
  // to TMcpServer.ExecuteRequest. This avoids changing TMcpServer's
  // interface — batch parsing and SSE framing are purely the transport's
  // responsibility.
  if doc.IsObject then
  begin
    // Copy the single object to a local variant BEFORE reinitializing doc,
    // to avoid memory corruption (doc would be destroyed while still being read)
    singleItem := variant(doc);
    doc.InitArray([singleItem], JSON_FAST);
  end;
  if not doc.IsArray or (doc.Count = 0) then
    exit(HTTP_BADREQUEST);

  // --- Classify messages and detect if any are requests ---
  hasRequest := false;
  isInitialize := false;
  for i := 0 to doc.Count - 1 do
  begin
    item := _Safe(doc.Values[i]);
    // A request has 'method' and 'id'; a notification has 'method' but no 'id'
    if item^.GetAsRawUtf8('method', method) then
    begin
      if not VarIsVoid(item^.GetValueOrNull('id')) then
      begin
        hasRequest := true;
        if method = 'initialize' then
          isInitialize := true;
      end;
    end;
  end;

  // --- Accept header validation skipped ---
  // mORMot's THttpAsyncServer filters standard headers (Accept, Content-Type,
  // Content-Length, etc.) out of InHeaders by default (HeadersUnFiltered=false).
  // The Accept header value is not reliably available in Ctxt.InHeaders.
  // FindNameValuePointer('ACCEPT: ') may match ACCEPT-ENCODING instead.
  // We follow the same lenient approach as TMcpSseTransport.AcceptsEventStream
  // which defaults to true when the header is not found.
  // The MCP spec says clients MUST send Accept: text/event-stream, but we
  // cannot enforce this at the server level with mORMot's default config.

  // --- Session validation ---
  // Initialize does not require a session ID; all other requests do
  sessionId := ExtractSessionId(Ctxt);
  session := nil;
  if not isInitialize then
  begin
    if sessionId = '' then
      exit(HTTP_BADREQUEST);
    if not fSessions.TryGetValue(sessionId, session) then
      exit(HTTP_NOTFOUND);
    session.Touch;
  end;

  // --- Notifications/responses only: process and return 202 ---
  if not hasRequest then
  begin
    for i := 0 to doc.Count - 1 do
    begin
      itemJson := _Safe(doc.Values[i])^.ToJson;
      fServer.ExecuteRequest(itemJson, sessionId);
    end;
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
  if isInitialize then
  begin
    sessionId := LowerCaseU(ToUtf8(RandomGuid));
    // Strip braces from UUID for clean ASCII header value
    // RandomGuid returns '{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}'
    if (length(sessionId) > 2) and
       (sessionId[1] = '{') and
       (sessionId[length(sessionId)] = '}') then
      sessionId := copy(sessionId, 2, length(sessionId) - 2);
    session := GetOrCreateSession(sessionId);
    session.Initialized := true;
  end;
  // Stash the resolved session id in the response headers: it is echoed to the
  // client (required for initialize) AND re-read by BuildDeferredResponse to
  // process the batch under the right session.
  Ctxt.OutCustomHeaders := Ctxt.OutCustomHeaders +
    'Mcp-Session-Id: ' + sessionId + #13#10;
  Ctxt.OutContentType := 'text/event-stream'; // marker consumed by OnRead
  Ctxt.RespStatus := HTTP_ASYNCRESPONSE;
  result := HTTP_ASYNCRESPONSE;
end;

function TMcpStreamableHttpTransport.OnDelete(
  Ctxt: THttpServerRequestAbstract): cardinal;
begin
  // Delegate to the published mcp method via THttpServerRequest downcast
  result := mcp(Ctxt as THttpServerRequest);
end;

procedure TMcpStreamableHttpTransport.SetCorsHeaders(var Ctxt: THttpServerRequest);
begin
  if not fCorsEnabled then
    exit;
  Ctxt.OutCustomHeaders := Ctxt.OutCustomHeaders +
    'Access-Control-Allow-Origin: ' + fCorsOrigins + #13#10 +
    'Access-Control-Allow-Methods: POST, GET, DELETE, OPTIONS' + #13#10 +
    'Access-Control-Allow-Headers: Content-Type, Mcp-Session-Id' + #13#10 +
    'Access-Control-Expose-Headers: Mcp-Session-Id' + #13#10 +
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

function TMcpStreamableHttpTransport.ExtractSessionId(
  var Ctxt: THttpServerRequest): RawUtf8;
var
  p: PUtf8Char;
  len: PtrInt;
begin
  result := '';
  p := FindNameValuePointer(pointer(Ctxt.InHeaders), 'MCP-SESSION-ID: ', len);
  if p = nil then
    p := FindNameValuePointer(pointer(Ctxt.InHeaders), 'MCP-SESSION-ID:', len);
  if p <> nil then
    FastSetString(result, p, len);
end;

function TMcpStreamableHttpTransport.GetOrCreateSession(
  const aSessionId: RawUtf8): TMcpStreamableSession;
begin
  if not fSessions.TryGetValue(aSessionId, result) then
  begin
    result := TMcpStreamableSession.Create(aSessionId);
    fSessions.Add(aSessionId, result);
  end;
end;

procedure TMcpStreamableHttpTransport.RemoveSession(const aSessionId: RawUtf8);
begin
  // No leak here despite the previous TODO: the IKeyValue OWNS its object values
  // (created without kvoValueNoFinalize), so Remove() frees the
  // TMcpStreamableSession instance. Freeing it again here would double-free.
  fSessions.Remove(aSessionId);
end;

function TMcpStreamableHttpTransport.FormatSseEvent(
  const aEvent, aData: RawUtf8; aId: Int64): RawUtf8;
var
  i, start: integer;
begin
  result := '';
  if aEvent <> '' then
    result := 'event: ' + aEvent + #13#10;
  if aId >= 0 then
    result := result + 'id: ' + Int64ToUtf8(aId) + #13#10;
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
    fSession: TMcpStreamableSession;
  public
    constructor Create(aTransport: TMcpStreamableHttpTransport;
      const aWrite: TMcpRawWrite; aSession: TMcpStreamableSession);
    procedure Emit(const aJsonMessage: RawUtf8);
  end;

constructor TMcpStreamEmitter.Create(aTransport: TMcpStreamableHttpTransport;
  const aWrite: TMcpRawWrite; aSession: TMcpStreamableSession);
begin
  inherited Create;
  fTransport := aTransport;
  fWrite := aWrite;
  fSession := aSession;
end;

procedure TMcpStreamEmitter.Emit(const aJsonMessage: RawUtf8);
var
  eventId: Int64;
begin
  if fSession <> nil then
    eventId := fSession.NextEventId
  else
    eventId := 0;
  fWrite(fTransport.SseChunk(
    fTransport.FormatSseEvent('message', aJsonMessage, eventId)));
end;

procedure TMcpStreamableHttpTransport.StreamDeferredResponse(
  const aWrite: TMcpRawWrite; const aBody, aOutHeaders: RawUtf8);
var
  sessionId, responseJson, itemJson, method: RawUtf8;
  doc: TDocVariantData;
  singleItem: variant;
  item: PDocVariantData;
  session: TMcpStreamableSession;
  emitter: IMcpStreamEmitter;
  handled: boolean;
  i: PtrInt;
  eventId: Int64;
  p: PUtf8Char;
  len: PtrInt;
begin
  // Resolve the session id the handler stashed in the response headers.
  sessionId := '';
  p := FindNameValuePointer(pointer(aOutHeaders), 'MCP-SESSION-ID: ', len);
  if p = nil then
    p := FindNameValuePointer(pointer(aOutHeaders), 'MCP-SESSION-ID:', len);
  if p <> nil then
    FastSetString(sessionId, p, len);
  session := nil;
  if sessionId <> '' then
    fSessions.TryGetValue(sessionId, session);

  // Re-parse the request body (same normalization as mcp()).
  doc.InitJson(aBody, JSON_FAST);
  if doc.IsObject then
  begin
    singleItem := variant(doc);
    doc.InitArray([singleItem], JSON_FAST);
  end;

  // HTTP response head: chunked, NO Content-Length. We deliberately keep the
  // connection alive (no 'Connection: close') so a client can reuse the socket
  // for subsequent requests — the terminating 0-chunk delimits this response.
  // aOutHeaders carries the CORS + Mcp-Session-Id lines (each CRLF-terminated);
  // the trailing CRLF below ends the header block.
  aWrite('HTTP/1.1 200 OK'#13#10 +
    'Content-Type: text/event-stream'#13#10 +
    'Cache-Control: no-cache'#13#10 +
    'Transfer-Encoding: chunked'#13#10 +
    aOutHeaders +
    #13#10);

  // shared emitter so a streaming tool can push intermediate token events
  emitter := TMcpStreamEmitter.Create(self, aWrite, session);

  // One SSE event (one chunk) per JSON-RPC response; each request is executed
  // at the point its event is emitted.
  if doc.IsArray then
    for i := 0 to doc.Count - 1 do
    begin
      item := _Safe(doc.Values[i]);
      if not item^.GetAsRawUtf8('method', method) then
        continue; // skip responses (no 'method' field)
      itemJson := item^.ToJson;
      if VarIsVoid(item^.GetValueOrNull('id')) then
      begin
        // notification: process silently, no SSE event
        fServer.ExecuteRequest(itemJson, sessionId);
        continue;
      end;
      // request: let a streaming hook handle it (pushing token events via the
      // emitter) and provide the final response; otherwise process normally
      handled := false;
      responseJson := '';
      if Assigned(fOnStreamCall) then
        handled := fOnStreamCall(itemJson, sessionId, emitter, responseJson);
      if not handled then
      begin
        responseJson := fServer.ExecuteRequest(itemJson, sessionId);
        if method = 'initialize' then
          responseJson := PatchProtocolVersion(responseJson);
      end;
      // final SSE event with the JSON-RPC response for this request
      if responseJson <> '' then
      begin
        if session <> nil then
          eventId := session.NextEventId
        else
          eventId := 0;
        aWrite(SseChunk(FormatSseEvent('message', responseJson, eventId)));
      end;
    end;

  // terminating zero-length chunk closes the chunked body
  aWrite('0'#13#10#13#10);
end;

function TMcpStreamableHttpTransport.PatchProtocolVersion(
  const aJson: RawUtf8): RawUtf8;
begin
  // Post-process the initialize response to replace the protocol version.
  // This is done at the transport layer to avoid changing TMcpServer or
  // TMcpJsonRpcProcessor interfaces. The server returns '2024-11-05' by default;
  // the Streamable HTTP transport patches it to '2025-03-26' because this
  // transport implements the newer spec version.
  // We anchor the replacement to the exact JSON key-value pair to avoid
  // accidentally replacing date strings elsewhere in the response.
  result := StringReplaceAll(aJson,
    '"protocolVersion":"' + MCP_PROTOCOL_VERSION + '"',
    '"protocolVersion":"' + MCP_PROTOCOL_VERSION_20250326 + '"');
end;


{ ************ TMcpStreamableSession }

constructor TMcpStreamableSession.Create(const aSessionId: RawUtf8);
begin
  inherited Create;
  SessionId := aSessionId;
  Initialized := false;
  LastActivity := NowUtc;
  LastEventId := 0;
end;

procedure TMcpStreamableSession.Touch;
begin
  LastActivity := NowUtc;
end;

destructor TMcpStreamableSession.Destroy;
begin
  ConsoleWrite('Destroying session');
      
  inherited;
end;

function TMcpStreamableSession.NextEventId: Int64;
begin
  // Simple increment — session is only accessed from one request handler at a time
  // (each POST is a synchronous response). If concurrent access is needed later,
  // add a TLightLock around this.
  inc(LastEventId);
  result := LastEventId;
end;


end.
