unit mormot.ai.mcp.types.FpcTest;

{$mode delphi}

// FPCUnit-Tests fuer mormot.ai.mcp.types: Protokollversion, JSON-RPC-Envelope
// (Result/Error-Form) und Request-Parsing inkl. Ablehnung von Nicht-JSON-RPC.
// Vergleich der JSON-Form ueber mORMot _Safe(..)^.ToJson (Schluesselreihenfolge
// ist durch _ObjFast deterministisch).

interface

uses
  sysutils,
  fpcunit,
  testregistry,
  mormot.core.base,
  mormot.core.variants,
  mormot.ai.mcp.types;

type
  TMcpTypesTests = class(TTestCase)
  published
    procedure ProtocolVersionIsCurrent;
    procedure ResultEnvelopeShape;
    procedure ErrorEnvelopeShape;
    procedure ParseRoundtrip;
    procedure ParseRejectsNonJsonRpc;
  end;

implementation

procedure TMcpTypesTests.ProtocolVersionIsCurrent;
begin
  AssertEquals('MCP revision', '2025-11-25', string(MCP_PROTOCOL_VERSION));
end;

procedure TMcpTypesTests.ResultEnvelopeShape;
var
  v: variant;
begin
  v := JsonRpcResult(1, _ObjFast(['ok', true]));
  AssertEquals('result envelope',
    '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}',
    string(_Safe(v)^.ToJson));
end;

procedure TMcpTypesTests.ErrorEnvelopeShape;
var
  v: variant;
begin
  v := JsonRpcError(7, JSONRPC_METHOD_NOT_FOUND, 'Method not found');
  AssertEquals('error envelope',
    '{"jsonrpc":"2.0","id":7,"error":{"code":-32601,"message":"Method not found"}}',
    string(_Safe(v)^.ToJson));
end;

procedure TMcpTypesTests.ParseRoundtrip;
var
  m: RawUtf8;
  id, params: variant;
begin
  AssertTrue('parse ok', JsonRpcParse(
    '{"jsonrpc":"2.0","id":42,"method":"tools/list","params":{"cursor":"x"}}',
    m, id, params));
  AssertEquals('method', 'tools/list', string(m));
  AssertEquals('id', 42, integer(id));
  AssertEquals('params.cursor', 'x', string(_Safe(params)^.U['cursor']));
end;

procedure TMcpTypesTests.ParseRejectsNonJsonRpc;
var
  m: RawUtf8;
  id, params: variant;
begin
  AssertFalse('missing jsonrpc tag', JsonRpcParse(
    '{"id":1,"method":"ping"}', m, id, params));
end;

initialization
  RegisterTest(TMcpTypesTests);

end.
