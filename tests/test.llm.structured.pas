// - regression tests for mormot.ai.llm.structured (typed structured output)
unit test.llm.structured;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text,
  mormot.core.rtti,
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

end.
