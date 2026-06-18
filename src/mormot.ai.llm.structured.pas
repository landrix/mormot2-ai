/// LandrixAI LLM Client - structured (typed) output
// - part of the mormot.ai.* extension (LandrixAI)
// - constrains a completion to a record's JSON-Schema and loads the answer back
//   into that record, reusing the very RTTI schema generator the MCP tools use
//   (mormot.ai.mcp) - the same machine describes a tool's input and a model's output
// - clean-room from the OpenAI Structured Outputs spec; target license MPL/GPL/LGPL
unit mormot.ai.llm.structured;

{
  *****************************************************************************

    - RecordJsonSchema: a record type -> JSON-Schema (via TMcpSchemaGenerator)
    - OpenAIJsonSchemaFormat / OpenAIJsonObjectFormat: response_format builders
    - ChatStructured: run a completion bound to a record schema and parse the
      JSON answer into that record with RecordLoadJson

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.core.rtti,
  mormot.core.variants,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.mcp;

const
  /// a response_format value asking the provider for any valid JSON object
  // - the broadest-compatible mode (OpenAI and Ollama); pair it with a prompt
  //   that describes the fields, then parse with RecordLoadJson
  LLM_JSON_OBJECT_FORMAT = '{"type":"json_object"}';

/// the JSON-Schema (as raw JSON) of a record type, via the shared RTTI generator
// - the record's RTTI must be available (e.g. Rtti.RegisterFromText for a packed
//   record of simple fields), exactly as for an MCP tool parameter record
function RecordJsonSchema(aTypeInfo: PRttiInfo): RawUtf8;

/// build an OpenAI response_format value for a named json_schema
// - aStrict=false (default) is the broadly-compatible mode: the schema guides the
//   model but is not strictly enforced; accepted by OpenAI and Ollama as-is
// - aStrict=true is OpenAI's strict mode: it requires additionalProperties:false
//   and all properties in "required", so it injects additionalProperties:false on
//   the top-level schema - only safe for a flat record of simple fields (a nested
//   record/array would also need a full nested schema, which is not generated)
function OpenAIJsonSchemaFormat(const aName, aSchemaJson: RawUtf8;
  aStrict: boolean = false): RawUtf8;

/// run a chat completion constrained to a record's schema and parse the result
// - generates the schema from aTypeInfo, sets aRequest.ResponseFormat to a
//   json_schema, calls ChatComplete and loads the JSON answer into aResult
// - aResult/aTypeInfo follow the RecordLoadJson convention: pass TypeInfo(TMyRec)
//   and a matching record variable; the record's RTTI must be registered (e.g.
//   Rtti.RegisterFromText) or the schema is empty
// - aResult is only defined when the function returns true; on a parse failure
//   (empty/non-JSON answer) treat it as undefined
// - aStrict defaults to false for cross-provider compatibility; pass true only
//   for a flat simple-field record talking to an OpenAI strict endpoint
function ChatStructured(const aClient: ILlmClient; var aRequest: TLlmChatRequest;
  aTypeInfo: PRttiInfo; var aResult; const aSchemaName: RawUtf8 = 'result';
  aStrict: boolean = false): boolean;


implementation

uses
  mormot.core.json;

function RecordJsonSchema(aTypeInfo: PRttiInfo): RawUtf8;
begin
  result := _Safe(TMcpSchemaGenerator.GenerateSchema(aTypeInfo))^.ToJson;
end;

function OpenAIJsonSchemaFormat(const aName, aSchemaJson: RawUtf8;
  aStrict: boolean): RawUtf8;
var
  schema: variant;
begin
  schema := _Json(aSchemaJson);
  // OpenAI strict mode additionally requires additionalProperties:false; the
  // RTTI schema generator does not emit it, so inject it on the top-level object
  if aStrict then
    _Safe(schema)^.AddValue('additionalProperties', false);
  result := _Safe(_ObjFast([
    'type', 'json_schema',
    'json_schema', _ObjFast([
      'name', aName,
      'strict', aStrict,
      'schema', schema])]))^.ToJson;
end;

function ChatStructured(const aClient: ILlmClient; var aRequest: TLlmChatRequest;
  aTypeInfo: PRttiInfo; var aResult; const aSchemaName: RawUtf8;
  aStrict: boolean): boolean;
var
  resp: TLlmChatResponse;
begin
  aRequest.ResponseFormat :=
    OpenAIJsonSchemaFormat(aSchemaName, RecordJsonSchema(aTypeInfo), aStrict);
  resp := aClient.ChatComplete(aRequest);
  result := RecordLoadJson(aResult, resp.Content, aTypeInfo);
end;

end.
