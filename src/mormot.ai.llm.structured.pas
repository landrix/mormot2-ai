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
  mormot.core.text, // VariantToUtf8 for the required-field check
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
// - aOptional names the fields a caller may leave out; everything else is
//   published as required, exactly as for an MCP tool parameter record
function RecordJsonSchema(aTypeInfo: PRttiInfo;
  const aOptional: TRawUtf8DynArray = nil): RawUtf8;

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
// - aResult is always CLEARED first, and left cleared when this returns
//   false: a caller that forgets to check the result gets zeros rather than
//   a half-filled record or the previous extraction
// - returns false unless the answer is a JSON object carrying every field the
//   generated schema declares required (see aOptional): '{}' and a wrapper
//   object both parse without error and would otherwise pass for a successful
//   extraction
// - aStrict defaults to false for cross-provider compatibility; pass true only
//   for a flat simple-field record talking to an OpenAI strict endpoint
// - aOptional names the fields the model may omit. RTTI carries no notion of
//   an optional field, so without it EVERY field is published as required AND
//   demanded back - which would make a record with a genuinely optional field
//   (a middle name, an address line 2) unusable: every otherwise-correct
//   extraction would be discarded. Same escape hatch MCP tools get from
//   TMcpToolBase.MarkOptional, and the published schema follows it, so the
//   contract we advertise and the one we enforce stay the same
function ChatStructured(const aClient: ILlmClient; var aRequest: TLlmChatRequest;
  aTypeInfo: PRttiInfo; var aResult; const aSchemaName: RawUtf8 = 'result';
  aStrict: boolean = false;
  const aOptional: TRawUtf8DynArray = nil): boolean;


implementation

uses
  mormot.core.json;

function RecordJsonSchema(aTypeInfo: PRttiInfo;
  const aOptional: TRawUtf8DynArray): RawUtf8;
begin
  result :=
    _Safe(TMcpSchemaGenerator.GenerateSchema(aTypeInfo, aOptional))^.ToJson;
end;

function OpenAIJsonSchemaFormat(const aName, aSchemaJson: RawUtf8;
  aStrict: boolean): RawUtf8;
var
  schema: variant;
begin
  // re-serialized into the request below: with the default parser a float
  // constant in the schema would turn into a string and invalidate it
  schema := _JsonFastFloat(aSchemaJson);
  // OpenAI strict mode additionally requires additionalProperties:false; the
  // RTTI schema generator does not emit it, so inject it on the top-level object
  // AddOrUpdateValue, not AddValue: a schema that already carries the key
  // would otherwise end up with it TWICE - TDocVariantData stores duplicate
  // names happily and every lookup then returns the first one
  if aStrict then
    _Safe(schema)^.AddOrUpdateValue('additionalProperties', false);
  result := _Safe(_ObjFast([
    'type', 'json_schema',
    'json_schema', _ObjFast([
      'name', aName,
      'strict', aStrict,
      'schema', schema])]))^.ToJson;
end;

function ChatStructured(const aClient: ILlmClient; var aRequest: TLlmChatRequest;
  aTypeInfo: PRttiInfo; var aResult; const aSchemaName: RawUtf8;
  aStrict: boolean; const aOptional: TRawUtf8DynArray): boolean;
var
  resp: TLlmChatResponse;
  schemaJson, field: RawUtf8;
  av, sv: variant;
  answer, required: PDocVariantData;
  i: PtrInt;
begin
  schemaJson := RecordJsonSchema(aTypeInfo, aOptional);
  aRequest.ResponseFormat :=
    OpenAIJsonSchemaFormat(aSchemaName, schemaJson, aStrict);
  resp := aClient.ChatComplete(aRequest);
  // Clear first. RecordLoadJson leaves fields it did not parse untouched
  // (jpoClearValues is in neither default option set), so a partial answer
  // used to blend into whatever the caller's variable happened to hold - and
  // for a caller that ignores the result, into the PREVIOUS extraction.
  RecordZero(@aResult, aTypeInfo);
  result := false;
  av := _JsonFastFloat(resp.Content);
  answer := _Safe(av);
  if not answer^.IsObject then
    exit; // prose, an empty answer, or a JSON array: not an extraction
  // Every field the schema declared required has to actually be there. '{}'
  // parses happily and a wrapper object like {"result":{...}} loses its
  // payload to jpoIgnoreUnknownProperty - both used to be reported as a
  // successful extraction with a record full of zeros. We publish the
  // contract in the request; enforcing it on the way back is the other half.
  required := _Safe(_JsonFastFloat(schemaJson))^.A['required'];
  for i := 0 to required^.Count - 1 do
  begin
    VariantToUtf8(required^.Values[i], field);
    if answer^.GetValueIndex(field) < 0 then
      exit;
  end;
  result := RecordLoadJson(aResult, resp.Content, aTypeInfo);
  if not result then
    // a type mismatch the presence check cannot catch: leave nothing behind
    RecordZero(@aResult, aTypeInfo);
end;

end.
