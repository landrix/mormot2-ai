/// LandrixAI LLM Client - MCP toolbox bridge
// - part of the mormot.ai.* extension (LandrixAI)
// - lets a TLlmAgent drive the very tools a mormot.ai.mcp server exposes: the
//   bridge talks to the in-process server over JSON-RPC, exactly as a remote
//   MCP client would, and the same RTTI-generated input schema describes both ends
// - clean-room from the MCP spec; target license MPL/GPL/LGPL (mORMot contribution)
unit mormot.ai.agent.mcp;

{
  *****************************************************************************

    TLlmMcpToolbox adapts a TMcpServer's tool registry to ILlmToolbox:
    - List   -> JSON-RPC "tools/list", projected to TLlmTool (name/description/
                inputSchema as the parameter JSON-Schema)
    - Execute -> JSON-RPC "tools/call", with the model's argument JSON embedded
                 as the params object; the text content blocks are concatenated

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.core.variants,
  mormot.ai.llm.types,
  mormot.ai.agent,
  mormot.ai.mcp;

type
  /// an ILlmToolbox backed by a mormot.ai.mcp server (the server is not owned)
  // - the server must be Active (Start called) before List/Execute are used
  TLlmMcpToolbox = class(TInterfacedObject, ILlmToolbox)
  protected
    fServer: TMcpServer;
  public
    /// wrap an existing, already-started MCP server
    constructor Create(const aServer: TMcpServer); reintroduce;
    function List: TLlmToolDynArray;
    function Execute(const aName, aArgumentsJson: RawUtf8): RawUtf8;
  end;


implementation

uses
  mormot.core.text,
  mormot.core.json;

constructor TLlmMcpToolbox.Create(const aServer: TMcpServer);
begin
  inherited Create;
  fServer := aServer;
end;

function TLlmMcpToolbox.List: TLlmToolDynArray;
var
  resp: RawUtf8;
  v: variant;
  toolsArr, t: PDocVariantData;
  i: PtrInt;
begin
  result := nil;
  // even in-process the request must carry the per-request protocol metadata:
  // the server validates it uniformly, there is no privileged internal path
  resp := fServer.ExecuteRequest(_Safe(_ObjFast([
    'jsonrpc', '2.0',
    'id', 1,
    'method', 'tools/list',
    'params', McpRequestParams(Null, 'mormot.ai.agent', '1.0.0')]))^.ToJson);
  // _JsonFastFloat: the inputSchema is re-serialized below and handed to the
  // model as its tool contract. The default parser turns a float constant it
  // cannot hold into a string, which would publish an invalid schema.
  v := _JsonFastFloat(resp);
  // result.tools[] -> name / description / inputSchema (a JSON-Schema object)
  toolsArr := _Safe(v)^.O['result']^.A['tools'];
  SetLength(result, toolsArr^.Count);
  for i := 0 to toolsArr^.Count - 1 do
  begin
    t := toolsArr^._[i];
    result[i].Name := t^.U['name'];
    result[i].Description := t^.U['description'];
    // re-serialize the nested schema object into the raw JSON the LLM tool wants
    result[i].ParametersJson := t^.O['inputSchema']^.ToJson;
  end;
end;

function TLlmMcpToolbox.Execute(const aName, aArgumentsJson: RawUtf8): RawUtf8;
var
  reqJson, resp: RawUtf8;
  args, req, v: variant;
  doc, res, content, item: PDocVariantData;
  i: PtrInt;
begin
  // the model passes arguments as a raw JSON string; embed it as a real object.
  // reject malformed JSON instead of silently calling the tool with no args, so
  // the model can correct itself
  if aArgumentsJson <> '' then
  begin
    if not IsValidJson(aArgumentsJson) then
      exit(FormatUtf8('{"error":"invalid tool arguments JSON: %"}', [aArgumentsJson]));
    // _JsonFastFloat: this goes on to a real MCP server, which validates
    // against the tool's schema - a float arriving as a string fails that
    args := _JsonFastFloat(aArgumentsJson);
  end
  else
    args := _Obj([]);
  req := _ObjFast([
    'jsonrpc', '2.0',
    'id', 1,
    'method', 'tools/call',
    'params', McpRequestParams(_ObjFast(['name', aName, 'arguments', args]),
      'mormot.ai.agent', '1.0.0')]);
  reqJson := _Safe(req)^.ToJson;
  resp := fServer.ExecuteRequest(reqJson);
  v := _JsonFastFloat(resp);
  doc := _Safe(v);
  // surface a JSON-RPC error as text so the model can recover instead of failing
  if doc^.O['error']^.Count > 0 then
    exit(FormatUtf8('{"error":"%"}', [doc^.O['error']^.U['message']]));
  res := doc^.O['result'];
  content := res^.A['content'];
  result := '';
  for i := 0 to content^.Count - 1 do
  begin
    item := content^._[i];
    if item^.U['type'] = 'text' then
      result := result + item^.U['text'];
  end;
  if result = '' then
    result := res^.ToJson; // no text blocks: hand back the raw result
  // an MCP tool-level failure (result.isError per the spec, distinct from a
  // JSON-RPC error) must be flagged, not passed off as a normal tool result
  if res^.B['isError'] then
    result := _Safe(_ObjFast(['isError', true, 'content', result]))^.ToJson;
end;

end.
