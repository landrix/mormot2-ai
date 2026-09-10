/// MCP Legacy-Era Bridge - the "initialize" handshake for stdio clients
// - this unit is part of the mormot2-ai project (mormot.ai.* extension)
// - licensed under MPL/GPL/LGPL three license
unit mormot.ai.mcp.legacy;

{
  *****************************************************************************

   LEGACY-ERA - a temporary compatibility bridge, meant to be REMOVED again
    - answers the pre-2026-07-28 "initialize" handshake on a stdio connection
    - then feeds requests without per-request _meta to the unchanged modern core
    - removal condition and checklist: CONCEPT.md section 6

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.core.variants,
  mormot.ai.mcp;


{ ************ LEGACY-ERA stdio bridge }

// LEGACY-ERA - why this exists at all:
// This library speaks MCP 2026-07-28 only, and that revision is stateless: every
// request carries its own _meta, there is no handshake. Clients still open stdio
// connections the old way, though. Claude Code (2.1.267) uses "initialize" on
// stdio by default, and even with negotiation switched on its server/discover
// probe gives up after 3 s - a server launched through "wsl" needs longer on a
// cold start, so the client falls back to "initialize" anyway. Without this
// bridge such a client never connects.
// The spec allows it: a server MAY implement both eras, and "an initialize
// request selects legacy semantics, scoped to the stdio process"
// (docs/specs/mcp-2026-07-28/basic/versioning.mdx, "Backward Compatibility with
// Initialization-Based Versions"). The handshake itself follows the 2025-11-25
// lifecycle ("Version Negotiation"; ping: "the receiver MUST respond promptly
// with an empty response").
// Remove when the clients we target open stdio with server/discover by default
// AND their probe survives a slow process start. Removal checklist: CONCEPT.md
// section 6.

const
  /// legacy protocol versions the bridge answers "initialize" with
  // - the requested version is echoed when it is listed here, otherwise the
  //   first (latest) one is offered: "If the server supports the requested
  //   protocol version, it MUST respond with the same version. Otherwise, the
  //   server MUST respond with another protocol version it supports."
  MCP_LEGACY_PROTOCOL_VERSIONS: array[0..3] of RawUtf8 = (
    '2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05');

type
  /// what TMcpLegacyStdioBridge.Route decided for one incoming line
  TMcpLegacyRoute = (
    /// not legacy-shaped (or not a valid JSON-RPC message): the core gets the
    // line unchanged and answers it the modern way
    lrPassThrough,
    /// answered by the bridge itself; the response is '' for a notification
    lrAnswered,
    /// rewritten into a modern request; the core's response must go through
    // AdaptResponse before it is written
    lrRewritten);

  /// LEGACY-ERA state of one stdio connection (= one process)
  // - not thread-safe: the stdio transport handles its lines one at a time
  TMcpLegacyStdioBridge = class
  private
    fServer: TMcpServer;
    fInitialized: boolean;
    fProtocolVersion: RawUtf8;
    fClientCapabilities: variant;
    function AnswerInitialize(const aId, aParams: variant): RawUtf8;
  public
    /// bind the bridge to the server whose capabilities and identity it reports
    constructor Create(aServer: TMcpServer);
    /// classify one request line: answer it, rewrite it, or leave it alone
    // - the line has to pass the core's own JSON-RPC envelope check first;
    //   whatever the core would reject it still rejects
    // - a request whose _meta carries the modern protocol version is always left
    //   alone: a dual-era server MAY serve both eras concurrently
    // - before "initialize", requests without that version are left alone too,
    //   so the core keeps rejecting them with its modern -32602
    function Route(var aRequest: RawUtf8; out aResponse: RawUtf8): TMcpLegacyRoute;
    /// turn the core's response to a rewritten request into its legacy form
    // - drops what only 2026-07-28 defines (`resultType`, `ttlMs`, `cacheScope`)
    // - an `input_required` result (Multi Round-Trip Request) has no legacy form
    //   and becomes a JSON-RPC error instead of an empty-looking success
    // - keeps tools/call and tools/list inside the legacy schemas, which accept
    //   only object-typed structuredContent, inputSchema and outputSchema
    function AdaptResponse(const aResponse: RawUtf8): RawUtf8;
    /// true once "initialize" was answered on this connection
    property Initialized: boolean read fInitialized;
    /// the legacy version agreed in the last "initialize" ('' before)
    property ProtocolVersion: RawUtf8 read fProtocolVersion;
  end;


implementation

function JsonRpcResult(const aId, aResult: variant): RawUtf8;
var
  doc: TDocVariantData;
begin
  doc.InitObject(['jsonrpc', '2.0',
                  'id', aId,
                  'result', aResult], JSON_FAST);
  result := doc.ToJson;
end;

// The capabilities server/discover reports, in their legacy form. Dropped:
// "extensions" (a 2026-07-28 concept; legacy revisions call it "experimental"),
// and "listChanged"/"subscribe" - those notifications only travel through
// subscriptions/listen, which a legacy client never opens, so announcing them
// would promise updates that never arrive.
function LegacyCapabilities(const aModern: variant): variant;
var
  src, sub: PDocVariantData;
  item: variant;
  i, j: PtrInt;
begin
  result := _ObjFast([]);
  src := _Safe(aModern);
  for i := 0 to src^.Count - 1 do
  begin
    if src^.Names[i] = 'extensions' then
      continue;
    sub := _Safe(src^.Values[i]);
    item := _ObjFast([]);
    for j := 0 to sub^.Count - 1 do
      if (sub^.Names[j] <> 'listChanged') and
         (sub^.Names[j] <> 'subscribe') then
        _Safe(item)^.AddValue(sub^.Names[j], sub^.Values[j]);
    _Safe(result)^.AddValue(src^.Names[i], item);
  end;
end;

// Legacy clients validate tools/list strictly, and one invalid tool makes them
// reject the whole list (TypeScript SDK 1.26 ToolSchema):
// - inputSchema and outputSchema MUST be objects with "type": "object". An object
//   schema without "type" gets it stated; anything else becomes an
//   accept-any-object inputSchema, or loses its outputSchema - there is nothing
//   a legacy client could validate a non-object result against.
// - every direct "properties" entry MUST be an object. JSON Schema also allows
//   the boolean forms; they are replaced by their exact equivalents, {} for true
//   and {"not": {}} for false.
procedure LegacyToolSchemas(aTools: PDocVariantData);
var
  i: PtrInt;
  tool: PDocVariantData;

  procedure FixProperties(aSchema: PDocVariantData);
  var
    props: PDocVariantData;
    k: PtrInt;
  begin
    if not aSchema^.GetAsDocVariant('properties', props) or
       not props^.IsObject then
      exit;
    for k := 0 to props^.Count - 1 do
      if not _Safe(props^.Values[k])^.IsObject then
        // TVarData, not VarType(): that one lives in Delphi's System.Variants
        if (TVarData(props^.Values[k]).VType = varBoolean) and
           not TVarData(props^.Values[k]).VBoolean then
          props^.AddOrUpdateValue(props^.Names[k], _ObjFast(['not', _ObjFast([])]))
        else
          props^.AddOrUpdateValue(props^.Names[k], _ObjFast([]));
  end;

  procedure FixSchema(const aKey: RawUtf8; aRequired: boolean);
  var
    idx: PtrInt;
    schema: PDocVariantData;
    kind: RawUtf8;
  begin
    idx := tool^.GetValueIndex(aKey);
    if idx < 0 then
      exit;
    schema := _Safe(tool^.Values[idx]);
    if schema^.IsObject and
       (schema^.GetValueIndex('type') < 0) then
      schema^.AddValue('type', 'object');
    if not schema^.IsObject or
       not schema^.GetAsRawUtf8('type', kind) or
       (kind <> 'object') then
    begin
      if aRequired then
        tool^.AddOrUpdateValue(aKey, _ObjFast(['type', 'object']))
      else
        tool^.Delete(aKey);
      exit;
    end;
    FixProperties(schema);
  end;

begin
  for i := 0 to aTools^.Count - 1 do
  begin
    tool := _Safe(aTools^.Values[i]);
    if not tool^.IsObject then
      continue;
    FixSchema('inputSchema', true);
    FixSchema('outputSchema', false);
  end;
end;


{ ************ TMcpLegacyStdioBridge }

constructor TMcpLegacyStdioBridge.Create(aServer: TMcpServer);
begin
  inherited Create;
  fServer := aServer;
  fClientCapabilities := _ObjFast([]);
end;

function TMcpLegacyStdioBridge.AnswerInitialize(const aId, aParams: variant): RawUtf8;
var
  params: PDocVariantData;
  requested: RawUtf8;
  caps, discover: variant;
  i: PtrInt;
begin
  params := _Safe(aParams);
  if not params^.GetAsRawUtf8('protocolVersion', requested) then
    requested := '';
  fProtocolVersion := MCP_LEGACY_PROTOCOL_VERSIONS[0];
  for i := 0 to high(MCP_LEGACY_PROTOCOL_VERSIONS) do
    if MCP_LEGACY_PROTOCOL_VERSIONS[i] = requested then
    begin
      fProtocolVersion := requested;
      break;
    end;
  // what the client declared once here, the modern core expects in every
  // request's _meta - Route supplies it from this copy
  caps := params^.GetValueOrNull('capabilities');
  if _Safe(caps)^.IsObject then
    fClientCapabilities := caps
  else
    fClientCapabilities := _ObjFast([]);
  fInitialized := true;
  discover := fServer.Processor.HandleDiscover;
  result := JsonRpcResult(aId, _ObjFast([
    'protocolVersion', fProtocolVersion,
    'capabilities', LegacyCapabilities(_Safe(discover)^.GetValueOrNull('capabilities')),
    'serverInfo', fServer.Processor.ServerInfo]));
end;

function TMcpLegacyStdioBridge.Route(var aRequest: RawUtf8;
  out aResponse: RawUtf8): TMcpLegacyRoute;
var
  doc: TDocVariantData;
  params, meta: PDocVariantData;
  method: RawUtf8;
  parsedParams, id, metaAdd: variant;
  idx: PtrInt;
begin
  result := lrPassThrough;
  aResponse := '';
  // the core's own envelope check ("jsonrpc":"2.0", a string method, structured
  // params, a valid id): the bridge never answers what the core would refuse
  if not fServer.Processor.ParseRequest(aRequest, method, parsedParams, id) then
    exit;
  // Positional params mean nothing to either era here: the core decides. Tested
  // by shape, not with VarIsVoid - that treats an empty [] like absent params,
  // and adding _meta to an array raises.
  params := _Safe(parsedParams);
  if params^.IsArray then
    exit;
  if params^.IsObject then
  begin
    idx := params^.GetValueIndex('_meta');
    if idx >= 0 then
    begin
      meta := _Safe(params^.Values[idx]);
      if not meta^.IsObject then
        exit;
      // modern means: the protocol version travels in _meta. A legacy _meta -
      // a progressToken, say - does not make a request modern.
      if meta^.GetValueIndex(MCP_META_PROTOCOL_VERSION) >= 0 then
        exit;
    end;
  end;
  if method = 'initialize' then
  begin
    if VarIsVoid(id) then
      exit; // an "initialize" notification means nothing; the core ignores it
    // a repeated "initialize" starts over, as a reconnect would: the last
    // negotiation wins, including the client capabilities it declares
    aResponse := AnswerInitialize(id, parsedParams);
    result := lrAnswered;
  end
  else if method = 'notifications/initialized' then
    result := lrAnswered // acknowledged by silence, like every notification
  else if method = 'ping' then
  begin
    if VarIsVoid(id) then
      exit;
    aResponse := JsonRpcResult(id, _ObjFast([]));
    result := lrAnswered;
  end
  else if fInitialized and
          not VarIsVoid(id) then
  begin
    // a request of the negotiated legacy session: add the per-request metadata
    // the modern core requires, from what "initialize" declared. Rewritten on a
    // float-safe copy - JSON_FAST would turn an argument like 0.12345678 into
    // the string "0.12345678" (the core parses with _JsonFastFloat for the same
    // reason).
    if not doc.InitJson(aRequest, JSON_FAST_FLOAT) then
      exit;
    metaAdd := _ObjFast([
      MCP_META_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION,
      MCP_META_CLIENT_CAPABILITIES, fClientCapabilities]);
    if not doc.GetAsDocVariant('params', params) then
      doc.AddOrUpdateValue('params', _ObjFast(['_meta', metaAdd]))
    else if params^.GetAsDocVariant('_meta', meta) then
    begin
      // keep what the legacy client sent (e.g. its progressToken)
      meta^.AddOrUpdateValue(MCP_META_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION);
      meta^.AddOrUpdateValue(MCP_META_CLIENT_CAPABILITIES, fClientCapabilities);
    end
    else
      params^.AddValue('_meta', metaAdd);
    aRequest := doc.ToJson;
    result := lrRewritten;
  end;
end;

function TMcpLegacyStdioBridge.AdaptResponse(const aResponse: RawUtf8): RawUtf8;
var
  doc: TDocVariantData;
  res, tools: PDocVariantData;
  resultType: RawUtf8;
  idx: PtrInt;
begin
  result := aResponse;
  // errors, and anything unexpected, travel unchanged; float-safe for the same
  // reason as in Route - structuredContent may carry numbers
  if not doc.InitJson(aResponse, JSON_FAST_FLOAT) or
     not doc.GetAsDocVariant('result', res) or
     not res^.IsObject then
    exit;
  if res^.GetAsRawUtf8('resultType', resultType) and
     (resultType = MCP_RESULT_INPUT_REQUIRED) then
  begin
    result := fServer.Processor.CreateError(doc.GetValueOrNull('id'),
      JSONRPC_INTERNAL_ERROR, 'this request needs a multi round-trip ' +
      '(input_required), which only MCP ' + MCP_PROTOCOL_VERSION +
      ' supports - the client connected with "initialize" (' +
      fProtocolVersion + ')');
    exit;
  end;
  // what only 2026-07-28 defines: the result type and the caching hints
  // (_meta stays: legacy results may carry one, unknown keys included)
  res^.Delete('resultType');
  res^.Delete('ttlMs');
  res^.Delete('cacheScope');
  // CallToolResult.structuredContent is an object in the legacy revisions; the
  // modern spec also allows an array, which a legacy client rejects outright
  idx := res^.GetValueIndex('structuredContent');
  if (idx >= 0) and
     not _Safe(res^.Values[idx])^.IsObject then
    res^.Delete('structuredContent');
  if res^.GetAsArray('tools', tools) then
    LegacyToolSchemas(tools);
  result := doc.ToJson;
end;


end.
