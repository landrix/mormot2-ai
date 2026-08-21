// - regression tests for mormot.ai.llm.structured (typed structured output)
unit test.llm.structured;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.core.rtti,
  mormot.core.variants, // PDocVariantData for the schema assertions
  mormot.core.test,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.structured,
  test.llm.agent; // reuse the scripted TStubLlmClient

type
  TPerson = packed record
    name: RawUtf8;
    age: integer;
  end;

  TTestLlmStructured = class(TSynTestCase)
  protected
    procedure EnsureRtti;
  published
    procedure ResponseFormatSerialized;
    procedure RecordSchemaHasFields;
    procedure ChatStructuredFillsRecord;
    procedure StrictModeInjectsAdditionalProperties;
    procedure IncompleteAnswerIsNotASuccess;
    procedure StrictModeDoesNotDuplicateAdditionalProperties;
    procedure OptionalFieldsAreNotDemandedBack;
  end;


implementation

function JsonResponse(const aContent: RawUtf8): TLlmChatResponse;
begin
  Finalize(result);
  FillCharFast(result, SizeOf(result), 0);
  result.FinishReason := lfrStop;
  result.Content := aContent;
end;

{ TTestLlmStructured }

procedure TTestLlmStructured.EnsureRtti;
begin
  if not RecordHasFields(TypeInfo(TPerson)) then
    Rtti.RegisterFromText(TypeInfo(TPerson), 'name:RawUtf8 age:integer');
end;

procedure TTestLlmStructured.ResponseFormatSerialized;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  json: RawUtf8;
begin
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'hi');
  req := LlmChatRequest('m', msgs);
  req.ResponseFormat := LLM_JSON_OBJECT_FORMAT;

  json := OpenAIChatRequestJson(req, {stream=}false);
  Check(Pos(RawUtf8('"response_format"'), json) > 0, 'response_format present');
  Check(Pos(RawUtf8('"json_object"'), json) > 0, 'json_object type embedded as object');
end;

procedure TTestLlmStructured.RecordSchemaHasFields;
var
  schema: RawUtf8;
begin
  EnsureRtti;
  schema := RecordJsonSchema(TypeInfo(TPerson));
  Check(Pos(RawUtf8('"object"'), schema) > 0, 'schema is an object');
  Check(Pos(RawUtf8('"name"'), schema) > 0, 'schema has name property');
  Check(Pos(RawUtf8('"age"'), schema) > 0, 'schema has age property');
end;

procedure TTestLlmStructured.ChatStructuredFillsRecord;
var
  stub: TStubLlmClient;
  client: ILlmClient;
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  person: TPerson;
  ok: boolean;
begin
  EnsureRtti;
  stub := TStubLlmClient.Create;
  client := stub;
  stub.Push(JsonResponse('{"name":"Alice","age":30}'));
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'Extract the person.');
  req := LlmChatRequest('test-model', msgs);

  ok := ChatStructured(client, req, TypeInfo(TPerson), person);
  Check(ok, 'JSON answer parsed into the record');
  CheckEqual(person.name, 'Alice', 'name field');
  CheckEqual(person.age, 30, 'age field');
  // the request the model received carried a strict json_schema from the record
  Check(Pos(RawUtf8('json_schema'), stub.LastRequest.ResponseFormat) > 0,
    'json_schema response_format set');
  Check(Pos(RawUtf8('"name"'), stub.LastRequest.ResponseFormat) > 0,
    'schema includes the record fields');
end;

procedure TTestLlmStructured.StrictModeInjectsAdditionalProperties;
const
  SCHEMA = '{"type":"object","properties":{"x":{"type":"string"}}}';
var
  def, strictFmt: RawUtf8;
begin
  // default is non-strict and broadly compatible (OpenAI + Ollama): no
  // additionalProperties requirement
  def := OpenAIJsonSchemaFormat('p', SCHEMA);
  Check(Pos(RawUtf8('"strict":false'), def) > 0, 'default is non-strict');
  Check(Pos(RawUtf8('additionalProperties'), def) = 0, 'no additionalProperties non-strict');
  // strict mode injects additionalProperties:false, which OpenAI requires
  strictFmt := OpenAIJsonSchemaFormat('p', SCHEMA, {strict=}true);
  Check(Pos(RawUtf8('"strict":true'), strictFmt) > 0, 'strict flag set');
  Check(Pos(RawUtf8('"additionalProperties":false'), strictFmt) > 0,
    'additionalProperties:false injected for strict');
end;


procedure TTestLlmStructured.IncompleteAnswerIsNotASuccess;
var
  stub: TStubLlmClient;
  client: ILlmClient;
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  person: TPerson;

  function Extract(const aAnswer: RawUtf8): boolean;
  begin
    stub.Push(JsonResponse(aAnswer));
    // pre-fill, so a fix that only checks the return value cannot hide behind
    // an already-zeroed variable: this is what a caller reusing a record has
    person.name := 'STALE';
    person.age := 99;
    result := ChatStructured(client, req, TypeInfo(TPerson), person);
  end;

begin
  EnsureRtti;
  stub := TStubLlmClient.Create;
  client := stub;
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'Extract the person.');
  req := LlmChatRequest('test-model', msgs);

  // '{}' parses without error and left the record untouched - reported as a
  // successful extraction of a person who is nobody
  Check(not Extract('{}'), 'an empty object is not an extraction');
  CheckEqual(person.name, '', 'and nothing of the previous one survives');
  CheckEqual(person.age, 0, 'nor of its numbers');

  // a wrapper object: jpoIgnoreUnknownProperty swallows the payload whole
  Check(not Extract('{"result":{"name":"Alice","age":30}}'),
    'a wrapped answer is not silently accepted as empty');
  CheckEqual(person.name, '', 'still nothing left behind');

  // a partial answer: the schema declares both fields required
  Check(not Extract('{"name":"Alice"}'), 'a missing required field fails');

  // prose instead of JSON
  Check(not Extract('I could not find a person.'), 'prose is not an extraction');

  // and the complete answer still works, exactly as before
  Check(Extract('{"name":"Alice","age":30}'), 'a complete answer succeeds');
  CheckEqual(person.name, 'Alice', 'name');
  CheckEqual(person.age, 30, 'age');
end;


procedure TTestLlmStructured.StrictModeDoesNotDuplicateAdditionalProperties;
var
  fmt: RawUtf8;
  d: PDocVariantData;
  i, n: PtrInt;
begin
  // a schema that already carries the key: AddValue appended a SECOND one, and
  // TDocVariantData stores duplicate names happily - every lookup then returns
  // the first, so the injected false was invisible to anyone reading it back
  fmt := OpenAIJsonSchemaFormat('r',
    '{"type":"object","additionalProperties":true,"properties":{}}', {strict=}true);
  d := _Safe(_Json(fmt))^.O['json_schema']^.O['schema'];
  n := 0;
  for i := 0 to d^.Count - 1 do
    if d^.Names[i] = 'additionalProperties' then
      inc(n);
  CheckEqual(n, 1, 'exactly one additionalProperties key');
  Check(not d^.B['additionalProperties'], 'and strict mode won');
end;


procedure TTestLlmStructured.OptionalFieldsAreNotDemandedBack;
var
  stub: TStubLlmClient;
  client: ILlmClient;
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  person: TPerson;
  optional: TRawUtf8DynArray;
begin
  EnsureRtti;
  stub := TStubLlmClient.Create;
  client := stub;
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'Extract the person.');
  req := LlmChatRequest('test-model', msgs);
  SetLength(optional, 1);
  optional[0] := 'age';

  // RTTI knows nothing about optional fields, so without this every field is
  // published as required AND demanded back - which would make any record with
  // a genuinely optional field unusable: each otherwise-correct extraction
  // discarded in full. Same escape hatch MCP tools get from MarkOptional.
  stub.Push(JsonResponse('{"name":"Alice"}'));
  person.name := 'STALE';
  person.age := 99;
  Check(ChatStructured(client, req, TypeInfo(TPerson), person, 'result',
    {strict=}false, optional), 'the optional field may be omitted');
  CheckEqual(person.name, 'Alice', 'and what came through is used');
  CheckEqual(person.age, 0, 'the omitted one is zero, not the stale value');

  // the published schema follows the same declaration - we do not advertise a
  // contract stricter than the one we enforce
  Check(Pos(RawUtf8('"required":["name"]'), stub.LastRequest.ResponseFormat) > 0,
    'the schema lists only the truly required field');

  // ...and a field that is NOT declared optional is still demanded
  stub.Push(JsonResponse('{"age":30}'));
  Check(not ChatStructured(client, req, TypeInfo(TPerson), person, 'result',
    {strict=}false, optional), 'a missing required field still fails');
end;

end.
