/// mORMot AI Extension - Model Context Protocol (MCP) Core Types
// - this unit is part of a proposed mormot.ai.* extension for the Open Source
// Synopse mORMot framework 2, intended for upstream contribution
// - license: MPL/GPL/LGPL three license (aligned with mORMot) - see ../LICENSE
unit mormot.ai.mcp.types;

{
  *****************************************************************************

   Model Context Protocol (MCP) - Core Types and JSON-RPC Envelope
    - Protocol version and JSON-RPC 2.0 constants
    - JSON-RPC envelope builders and request parser (TDocVariant based)
    - MCP tool descriptor record

   Built against the MCP specification revision 2025-11-25.
   Clean-room implementation: no third-party MCP code reused.

  *****************************************************************************
}

interface

{$ifdef FPC}
  {$mode Delphi}
{$endif FPC}
{$H+}

uses
  sysutils,
  classes,
  mormot.core.base,
  mormot.core.variants;

const
  /// the MCP specification revision implemented by this unit
  MCP_PROTOCOL_VERSION = '2025-11-25';

  /// JSON-RPC 2.0 protocol tag
  JSONRPC_VERSION = '2.0';

  // JSON-RPC 2.0 standard error codes - https://www.jsonrpc.org/specification
  JSONRPC_PARSE_ERROR      = -32700;
  JSONRPC_INVALID_REQUEST  = -32600;
  JSONRPC_METHOD_NOT_FOUND = -32601;
  JSONRPC_INVALID_PARAMS   = -32602;
  JSONRPC_INTERNAL_ERROR   = -32603;

  // MCP method names - the tools subset is implemented first
  MCP_METHOD_INITIALIZE     = 'initialize';
  MCP_METHOD_INITIALIZED    = 'notifications/initialized';
  MCP_METHOD_PING           = 'ping';
  MCP_METHOD_TOOLS_LIST     = 'tools/list';
  MCP_METHOD_TOOLS_CALL     = 'tools/call';
  MCP_METHOD_RESOURCES_LIST = 'resources/list';
  MCP_METHOD_RESOURCES_READ = 'resources/read';

type
  /// descriptor of a single MCP tool exposed by the server
  // - InputSchema holds a JSON Schema object as a TDocVariant (or null)
  TMcpToolMeta = record
    Name: RawUtf8;
    Description: RawUtf8;
    InputSchema: variant;
  end;
  TMcpToolMetaDynArray = array of TMcpToolMeta;

/// build a JSON-RPC 2.0 success response: {"jsonrpc":"2.0","id":..,"result":..}
function JsonRpcResult(const aId: variant; const aResult: variant): variant;

/// build a JSON-RPC 2.0 error response: {"jsonrpc":"2.0","id":..,"error":{..}}
function JsonRpcError(const aId: variant; aCode: integer;
  const aMessage: RawUtf8): variant;

/// parse a JSON-RPC 2.0 request envelope
// - returns false if the JSON is invalid or carries no "jsonrpc"/"method"
// - aId and aParams are returned as variants (null when absent)
function JsonRpcParse(const aJson: RawUtf8; out aMethod: RawUtf8;
  out aId: variant; out aParams: variant): boolean;

implementation

function JsonRpcResult(const aId: variant; const aResult: variant): variant;
begin
  result := _ObjFast([
    'jsonrpc', JSONRPC_VERSION,
    'id',      aId,
    'result',  aResult]);
end;

function JsonRpcError(const aId: variant; aCode: integer;
  const aMessage: RawUtf8): variant;
begin
  result := _ObjFast([
    'jsonrpc', JSONRPC_VERSION,
    'id',      aId,
    'error',   _ObjFast([
                 'code',    aCode,
                 'message', aMessage])]);
end;

function JsonRpcParse(const aJson: RawUtf8; out aMethod: RawUtf8;
  out aId: variant; out aParams: variant): boolean;
var
  doc: TDocVariantData;
begin
  aMethod := '';
  aId := null;
  aParams := null;
  result := doc.InitJson(aJson, JSON_FAST_FLOAT);
  if not result then
    exit;
  // a JSON-RPC 2.0 message must carry the "jsonrpc" tag
  if doc.GetValueIndex('jsonrpc') < 0 then
  begin
    result := false;
    exit;
  end;
  aMethod := doc.U['method'];
  aId := doc.GetValueOrNull('id');
  aParams := doc.GetValueOrNull('params');
  result := aMethod <> '';
end;

end.
