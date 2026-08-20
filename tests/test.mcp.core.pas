// - regression tests for mormot.ai.mcp
unit test.mcp.core;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  classes,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.buffers,
  mormot.core.rtti,
  mormot.core.variants,
  mormot.core.json,
  mormot.core.test,
  mormot.ai.mcp;

type
  TCalcParams = packed record
    A: integer;
    B: integer;
    Enabled: boolean;
    Name: RawUtf8;
  end;

  /// every tool here shares TCalcParams, where A and B are the actual
  /// arguments and Enabled/Name are decoration - exactly the situation
  /// MarkOptional exists for. Without it the generated schema would publish
  /// all four as required AND (since that is now enforced) refuse the calls
  /// these tests make, which is the honest consequence of a contract that
  /// used to be decorative.
  TCalcToolBase = class(TMcpToolBase<TCalcParams>)
  public
    constructor Create(const aName, aDescription: RawUtf8); override;
  end;

  TCalcTool = class(TCalcToolBase)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  TVersionResource = class(TMcpResourceBase)
  protected
    function GetContent: RawUtf8; override;
  end;

  /// a prompt with one required argument, returning a bare messages ARRAY to
  /// prove the server wraps it into the result envelope
  TReviewPrompt = class(TInterfacedObject, IMcpPrompt)
  protected
    fName: RawUtf8;
  public
    constructor Create(const aName: RawUtf8); reintroduce;
    function GetName: RawUtf8;
    function GetTitle: RawUtf8;
    function GetDescription: RawUtf8;
    function GetArguments: variant;
    function Render(const Args: variant;
      const AuthCtx: TMcpAuthContext): variant;
  end;

  /// a resource template that ALSO offers completion for its {path} argument
  TFilesTemplate = class(TInterfacedObject, IMcpResourceTemplate, IMcpCompletable)
  public
    function GetUriTemplate: RawUtf8;
    function GetName: RawUtf8;
    function GetTitle: RawUtf8;
    function GetDescription: RawUtf8;
    function GetMimeType: RawUtf8;
    function Complete(const ArgumentName, ArgumentValue: RawUtf8;
      const Context: variant; const AuthCtx: TMcpAuthContext): TRawUtf8DynArray;
  end;

  /// a template that SERVES its concrete URIs — the add-on that turns a mere
  /// advertisement into something resources/read can resolve
  TDbTemplate = class(TInterfacedObject, IMcpResourceTemplate,
    IMcpExpandableResourceTemplate)
  public
    function GetUriTemplate: RawUtf8;
    function GetName: RawUtf8;
    function GetTitle: RawUtf8;
    function GetDescription: RawUtf8;
    function GetMimeType: RawUtf8;
    function ReadExpanded(const aUri: RawUtf8; const aVars: variant;
      const aContext: TMcpCallContext): RawUtf8;
  end;

  /// a prompt whose completion returns MORE than the 100 the spec allows, to
  /// prove the server truncates instead of trusting the implementation
  TFloodPrompt = class(TInterfacedObject, IMcpPrompt, IMcpCompletable)
  public
    function GetName: RawUtf8;
    function GetTitle: RawUtf8;
    function GetDescription: RawUtf8;
    function GetArguments: variant;
    function Render(const Args: variant;
      const AuthCtx: TMcpAuthContext): variant;
    function Complete(const ArgumentName, ArgumentValue: RawUtf8;
      const Context: variant; const AuthCtx: TMcpAuthContext): TRawUtf8DynArray;
  end;

  /// a prompt that ELICITS its missing argument instead of failing — prompts/get
  /// is the third method allowed to answer with an InputRequiredResult
  TElicitingPrompt = class(TInterfacedObject, IMcpPrompt, IMcpInteractivePrompt)
  public
    function GetName: RawUtf8;
    function GetTitle: RawUtf8;
    function GetDescription: RawUtf8;
    function GetArguments: variant;
    function Render(const Args: variant;
      const AuthCtx: TMcpAuthContext): variant;
    function RenderInteractive(const Args: variant;
      const Context: TMcpCallContext): variant;
  end;

  /// a prompt with no arguments and no title, returning a FULL result object —
  /// the other of the two shapes a prompt may produce
  TGreetPrompt = class(TInterfacedObject, IMcpPrompt)
  public
    function GetName: RawUtf8;
    function GetTitle: RawUtf8;
    function GetDescription: RawUtf8;
    function GetArguments: variant;
    function Render(const Args: variant;
      const AuthCtx: TMcpAuthContext): variant;
  end;

  /// a tool that raises a PLAIN Exception (not ESynException) — used to prove a
  /// tool error is translated into a JSON-RPC error, never escapes the handler
  /// refuses for lack of a scope - the only way that path can be reached
  /// without an HTTP transport in front of it
  TScopeGatedTool = class(TCalcToolBase)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  TThrowingTool = class(TCalcToolBase)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  /// a tool returning a JSON ARRAY instead of the required result object
  // - the protocol has no non-object result; this must surface as an error
  //   rather than silently shipping an empty success (see ResultMustBeAnObject)
  TArrayResultTool = class(TCalcToolBase)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  /// collects what a handler pushed onto its response stream
  TRecordingSink = class(TInterfacedObject, IMcpNotificationSink)
  public
    /// every JSON-RPC message the handler sent, in order
    Sent: TRawUtf8DynArray;
    procedure Send(const aJsonMessage: RawUtf8);
  end;

  /// a tool that reports progress, driven by what the test wants to prove
  TProgressTool = class(TCalcToolBase, IMcpInteractiveTool)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  public
    /// the progress values the tool will try to report, in order
    Steps: TDoubleDynArray;
    /// what Report() answered for each attempt
    Accepted: array of boolean;
    /// what Wanted() said on entry
    SawWanted: boolean;
    function ExecuteInteractive(const Args: variant;
      const Context: TMcpCallContext): variant;
  end;

  /// a tool that records the W3C trace context it was handed
  // - the only way to observe TMcpCallContext.Trace from a test: the trace
  //   fields never appear in a result, they exist so a handler can forward them
  TTraceRecordingTool = class(TCalcToolBase, IMcpInteractiveTool)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  public
    /// what the last call saw in _meta.traceparent/tracestate/baggage
    SawTrace: TMcpTraceContext;
    function ExecuteInteractive(const Args: variant;
      const Context: TMcpCallContext): variant;
  end;

  /// a tool doing a Multi Round-Trip Request: it asks for a name on the first
  /// round and completes once the client hands one back
  TElicitingTool = class(TCalcToolBase, IMcpInteractiveTool)
  protected
    function ExecuteTyped(const aParams: TCalcParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  public
    /// what the tool asks for; the tests vary it to exercise the server gate
    InputMethod: RawUtf8;
    /// set when the second round really saw the client's answers
    SawResponses: RawUtf8;
    function ExecuteInteractive(const Args: variant;
      const Context: TMcpCallContext): variant;
  end;

  /// a token verifier the tests drive into every refusal reason
  // - the real one lives in the backend: this unit must stay free of any
  //   particular identity system, which is exactly what the interface is for
  TFakeVerifier = class(TInterfacedObject, IMcpTokenVerifier)
  public
    /// the token that verifies; anything else is mtrInvalid
    GoodToken: RawUtf8;
    /// what the good token is scoped for
    Scopes: TRawUtf8DynArray;
    /// force a specific outcome for the good token (mtrValid = no override)
    Force: TMcpTokenResult;
    /// the resource the server passed in, recorded for the audience assertion
    SeenResource: RawUtf8;
    function VerifyToken(const aToken, aResource: RawUtf8;
      out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
  end;

  /// a resource that always needs input — resources/read is the second method
  /// the spec allows an InputRequiredResult on
  TGatedResource = class(TMcpResourceBase, IMcpInteractiveResource)
  protected
    function GetContent: RawUtf8; override;
  public
    function ReadInteractive(const Context: TMcpCallContext): RawUtf8;
  end;

  TTestMcpCore = class(TSynTestCase)
  protected
    procedure EnsureCalcParamsRtti;
    function VariantToInt64Loose(const V: variant; out aValue: Int64): boolean;
    procedure CheckErrorResponse(const aResponse: RawUtf8;
      aExpectedCode: integer; const aMessageContains: RawUtf8);
    function DocPropType(const props: PDocVariantData;
      const propName: RawUtf8): RawUtf8;
    /// run a request, adding the _meta fields every request must carry
    // - since 2026-07-28 each request declares its protocol version and client
    //   capabilities; adding that here keeps the test literals about the RPC
    //   under test instead of repeating protocol boilerplate everywhere
    // - a payload that is not a JSON object, or a notification (no id), is
    //   passed through untouched so the negative tests still exercise the
    //   parser and the notification path
    function Exec(aServer: TMcpServer; const aJson: RawUtf8): RawUtf8;
    /// like Exec, but with client capabilities the caller chooses
    // - Exec goes through McpRequestParams, which declares NO capabilities; the
    //   MRTR gate is precisely about what the client did or did not declare
    function ExecCaps(aServer: TMcpServer; const aJson: RawUtf8;
      const aCapabilities: variant): RawUtf8;
  published
    procedure SchemaFromRecord;
    procedure JsonRpcProcessor;
    procedure ServerToolsResources;
    procedure ResponseBuilderTextAndFile;
    procedure ServerNotActive;
    procedure BadRequests;
    procedure ToolExceptionBecomesError;
    procedure InvalidJsonRpcEnvelope;
    procedure NotificationsNoResponse;
    procedure DiscoverReportsVersionAndIdentity;
    procedure RequestMetaIsMandatory;
    procedure ResultsCarryResultTypeAndServerInfo;
    procedure RequestParamsMergeExistingMeta;
    procedure ResultMustBeAnObject;
    procedure PreflightDecidesHttpStatus;
    procedure CacheableResultsCarryHints;
    procedure SubscriptionFilterAndFanout;
    procedure InputRequiredRoundTrip;
    procedure InputRequiredNeedsClientCapability;
    procedure InputRequiredRejectsMalformedResults;
    procedure MrtrRetryFieldsAreTypedAndCountAsPresent;
    procedure RequestStateCodecBindsAndExpires;
    procedure ScopeHierarchyAndBearerParsing;
    procedure ProtectedResourceMetadataAndChallenges;
    procedure CursorCodecRejectsForgeries;
    procedure ListsAreSortedAndPaginated;
    procedure PromptsListAndGet;
    procedure TemplatesAndCompletion;
    procedure TraceContextReachesTheHandler;
    procedure ExtensionsAreAdvertisedAndNegotiated;
    procedure ProgressIsOptInAndMonotonic;
    procedure UriTemplatesResolveOnRead;
    procedure HeaderMirroringIsConstrained;
    procedure PublishedSchemasAreBounded;
    procedure NotificationWithAbsentParamsIsSafe;
    procedure SubscriptionCapIsPerPrincipalToo;
    procedure ScopeRefusalUsesAnApplicationCode;
    procedure RequiredSchemaIsHonestAndEnforced;
    procedure TypedToolRefusesArgumentsThatDoNotParse;
  end;

implementation

{ TCalcToolBase }

constructor TCalcToolBase.Create(const aName, aDescription: RawUtf8);
begin
  inherited Create(aName, aDescription);
  MarkOptional(['Enabled', 'Name']);
end;

{ TCalcTool }

function TCalcTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
begin
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText(FormatUtf8('% + % = %', [aParams.A, aParams.B, aParams.A + aParams.B]));
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

{ TScopeGatedTool }

function TScopeGatedTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  result := Null; // never reached
  raise EMcpInsufficientScope.CreateScope('files:write files:admin');
end;

{ TVersionResource }

function TVersionResource.GetContent: RawUtf8;
begin
  result := '{"version":"1.0.0","protocol":"MCP"}';
end;

{ TReviewPrompt }

constructor TReviewPrompt.Create(const aName: RawUtf8);
begin
  inherited Create;
  fName := aName;
end;

function TReviewPrompt.GetName: RawUtf8;
begin
  result := fName;
end;

function TReviewPrompt.GetTitle: RawUtf8;
begin
  result := 'Request Code Review';
end;

function TReviewPrompt.GetDescription: RawUtf8;
begin
  result := 'Asks the LLM to analyze code quality';
end;

function TReviewPrompt.GetArguments: variant;
begin
  result := _ArrFast([
    _ObjFast(['name', 'code',
              'description', 'The code to review',
              'required', true])]);
end;

function TReviewPrompt.Render(const Args: variant;
  const AuthCtx: TMcpAuthContext): variant;
var
  code: RawUtf8;
begin
  // "Missing required arguments: -32602" — the prompt itself knows which of its
  // arguments are required, so it is the one that must refuse
  if not _Safe(Args)^.GetAsRawUtf8('code', code) or
     (code = '') then
    raise EMcpInvalidParams.CreateU('prompt argument "code" is required');
  // a BARE messages array: the server wraps it
  result := _ArrFast([
    _ObjFast(['role', 'user',
              'content', _ObjFast(['type', 'text',
                                   'text', 'Please review:'#10 + code])])]);
end;

{ TFilesTemplate }

function TDbTemplate.GetUriTemplate: RawUtf8;
begin
  result := 'db://{table}/rows/{id}';
end;

function TDbTemplate.GetName: RawUtf8;
begin
  result := 'db-row';
end;

function TDbTemplate.GetTitle: RawUtf8;
begin
  result := 'Database row';
end;

function TDbTemplate.GetDescription: RawUtf8;
begin
  result := 'One row of a table';
end;

function TDbTemplate.GetMimeType: RawUtf8;
begin
  result := 'application/json';
end;

function TDbTemplate.ReadExpanded(const aUri: RawUtf8; const aVars: variant;
  const aContext: TMcpCallContext): RawUtf8;
var
  vars: PDocVariantData;
begin
  vars := _Safe(aVars);
  // matching the shape is not proof that the thing exists
  if vars^.U['table'] = 'missing' then
    raise EMcpInvalidParams.CreateUtf8('No such table: %', [vars^.U['table']]);
  result := FormatUtf8('{"table":"%","id":"%"}',
    [vars^.U['table'], vars^.U['id']]);
end;

function TFilesTemplate.GetUriTemplate: RawUtf8;
begin
  result := 'file:///{path}'; // RFC 6570
end;

function TFilesTemplate.GetName: RawUtf8;
begin
  result := 'Project Files';
end;

function TFilesTemplate.GetTitle: RawUtf8;
begin
  // deliberately DIFFERENT from GetName: identical strings would hide a bug
  // that serializes the name under the title key (or vice versa)
  result := 'Browse project files';
end;

function TFilesTemplate.GetDescription: RawUtf8;
begin
  result := 'Access files in the project directory';
end;

function TFilesTemplate.GetMimeType: RawUtf8;
begin
  result := 'application/octet-stream';
end;

function TFilesTemplate.Complete(const ArgumentName, ArgumentValue: RawUtf8;
  const Context: variant; const AuthCtx: TMcpAuthContext): TRawUtf8DynArray;
begin
  result := nil;
  // an unknown argument has no suggestions — proves the name reaches us
  if ArgumentName <> 'path' then
    exit;
  AddRawUtf8(result, 'src/main.pas');
  AddRawUtf8(result, 'README.md');
end;

{ TFloodPrompt }

function TFloodPrompt.GetName: RawUtf8;
begin
  result := 'flood';
end;

function TFloodPrompt.GetTitle: RawUtf8;
begin
  result := '';
end;

function TFloodPrompt.GetDescription: RawUtf8;
begin
  result := 'returns too many completions on purpose';
end;

function TFloodPrompt.GetArguments: variant;
begin
  result := _ArrFast([_ObjFast(['name', 'many'])]);
end;

function TFloodPrompt.Render(const Args: variant;
  const AuthCtx: TMcpAuthContext): variant;
begin
  result := _ArrFast([]);
end;

function TFloodPrompt.Complete(const ArgumentName, ArgumentValue: RawUtf8;
  const Context: variant; const AuthCtx: TMcpAuthContext): TRawUtf8DynArray;
var
  i: integer;
begin
  result := nil;
  SetLength(result, 150); // deliberately over the 100 ceiling
  for i := 0 to 149 do
    result[i] := FormatUtf8('v%', [i]);
end;

{ TElicitingPrompt }

function TElicitingPrompt.GetName: RawUtf8;
begin
  result := 'eliciting';
end;

function TElicitingPrompt.GetTitle: RawUtf8;
begin
  result := '';
end;

function TElicitingPrompt.GetDescription: RawUtf8;
begin
  result := 'asks for its argument when it was not supplied';
end;

function TElicitingPrompt.GetArguments: variant;
begin
  result := _ArrFast([_ObjFast(['name', 'topic', 'required', true])]);
end;

function TElicitingPrompt.Render(const Args: variant;
  const AuthCtx: TMcpAuthContext): variant;
begin
  // never reached: the server prefers RenderInteractive when it is available
  result := _ArrFast([]);
end;

function TElicitingPrompt.RenderInteractive(const Args: variant;
  const Context: TMcpCallContext): variant;
var
  topic: RawUtf8;
begin
  // Round two: the client answered, so build the real prompt from its answer.
  if Context.HasInputResponses then
  begin
    topic := _Safe(_Safe(Context.InputResponses)^.GetValueOrNull('topic'))^.
      U['content'];
    result := _ArrFast([
      _ObjFast(['role', 'user',
                'content', _ObjFast(['type', 'text', 'text', 'About ' + topic])])]);
    exit;
  end;
  // Round one: the argument is missing and only the user can supply it.
  raise EMcpInputRequired.Create(
    _ObjFast(['topic', McpInputRequest(MCP_INPUT_ELICITATION,
      _ObjFast(['message', 'Which topic?',
                'requestedSchema', _ObjFast(['type', 'object'])]))]), '');
end;

{ TGreetPrompt }

function TGreetPrompt.GetName: RawUtf8;
begin
  result := 'greet';
end;

function TGreetPrompt.GetTitle: RawUtf8;
begin
  result := ''; // optional and omitted — must not appear in prompts/list
end;

function TGreetPrompt.GetDescription: RawUtf8;
begin
  result := 'Say hello';
end;

function TGreetPrompt.GetArguments: variant;
begin
  SetVariantNull(result); // no arguments at all
end;

function TGreetPrompt.Render(const Args: variant;
  const AuthCtx: TMcpAuthContext): variant;
begin
  // the FULL envelope shape, passed through untouched
  result := _ObjFast([
    'description', 'A greeting',
    'messages', _ArrFast([
      _ObjFast(['role', 'user',
                'content', _ObjFast(['type', 'text', 'text', 'Hello'])])])]);
end;

{ TThrowingTool }

function TThrowingTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  // a plain RTL Exception (NOT ESynException): the dispatcher must still catch it
  raise Exception.Create('tool blew up');
end;

{ TArrayResultTool }

function TArrayResultTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  result := _ArrFast(['not', 'an', 'object']);
end;

{ TElicitingTool }

procedure TRecordingSink.Send(const aJsonMessage: RawUtf8);
begin
  AddRawUtf8(Sent, aJsonMessage);
end;

function TProgressTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  raise Exception.Create('ExecuteTyped must not be called on an interactive tool');
end;

function TProgressTool.ExecuteInteractive(const Args: variant;
  const Context: TMcpCallContext): variant;
var
  builder: TMcpResponseBuilder;
  i: PtrInt;
begin
  SawWanted := Context.Progress.Wanted;
  SetLength(Accepted, length(Steps));
  for i := 0 to high(Steps) do
    Accepted[i] := Context.Progress.Report(Steps[i], 100, 'step');
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText('done');
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

function TTraceRecordingTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  // never reached: the server prefers ExecuteInteractive on this tool
  raise Exception.Create('ExecuteTyped must not be called on an interactive tool');
end;

function TTraceRecordingTool.ExecuteInteractive(const Args: variant;
  const Context: TMcpCallContext): variant;
var
  builder: TMcpResponseBuilder;
begin
  SawTrace := Context.Trace;
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText('ok');
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

function TElicitingTool.ExecuteTyped(const aParams: TCalcParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  // never reached: the server prefers ExecuteInteractive on this tool
  raise Exception.Create('ExecuteTyped must not be called on an interactive tool');
end;

function TElicitingTool.ExecuteInteractive(const Args: variant;
  const Context: TMcpCallContext): variant;
var
  answers, who, content: PDocVariantData;
  name: RawUtf8;
  builder: TMcpResponseBuilder;
begin
  // the value under our own key is an ElicitResult: {action, content{…}}
  answers := _Safe(Context.InputResponses);
  if not answers^.GetAsDocVariant('who', who) or
     not who^.GetAsDocVariant('content', content) or
     not content^.GetAsRawUtf8('name', name) then
  begin
    // first round: nothing to work with yet. The state carries what we already
    // know, so this handler keeps nothing server-side between the rounds.
    SawResponses := '';
    raise EMcpInputRequired.Create(
      _ObjFast(['who', McpInputRequest(InputMethod, _ObjFast([
        'mode', 'form',
        'message', 'Who is asking?']))]),
      'round-1-state');
  end;
  SawResponses := name;
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText('hello ' + name + ' (' + Context.RequestState + ')');
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

{ TFakeVerifier }

function TFakeVerifier.VerifyToken(const aToken, aResource: RawUtf8;
  out aAuthCtx: TMcpAuthContext): TMcpTokenResult;
begin
  SeenResource := aResource;
  // write a principal BEFORE deciding, on purpose: a real verifier may well do
  // that, and the server must not leak a half-filled context on a refusal
  aAuthCtx.UserID := 'someone';
  if aToken <> GoodToken then
    exit(mtrInvalid);
  if Force <> mtrValid then
    exit(Force);
  aAuthCtx.IsAuthenticated := true;
  aAuthCtx.UserID := 'user-1';
  aAuthCtx.Issuer := 'https://as.example.com';
  aAuthCtx.Scopes := Scopes;
  result := mtrValid;
end;

{ TGatedResource }

function TGatedResource.GetContent: RawUtf8;
begin
  result := '{"gated":true}';
end;

function TGatedResource.ReadInteractive(const Context: TMcpCallContext): RawUtf8;
begin
  if Context.RequestState = '' then
    // requestState only, no inputRequests: the spec allows that shape, and it
    // means "retry immediately, carrying this back"
    raise EMcpInputRequired.Create(Null, 'resource-state');
  result := '{"gated":false,"state":"' + Context.RequestState + '"}';
end;

{ TTestMcpCore }

procedure TTestMcpCore.EnsureCalcParamsRtti;
begin
  //{$ifndef HASEXTRECORDRTTI}
  if not RecordHasFields(TypeInfo(TCalcParams)) then
    Rtti.RegisterFromText(TypeInfo(TCalcParams),
      'A,B:integer Enabled:boolean Name:RawUtf8');
  //{$endif HASEXTRECORDRTTI}
end;

function TTestMcpCore.VariantToInt64Loose(const V: variant; out aValue: Int64): boolean;
var
  d: double;
  s: RawUtf8;
  wasString: boolean;
begin
  result := VariantToInt64(V, aValue);
  if result then
    exit;
  if VariantToDouble(V, d) then
  begin
    aValue := trunc(d);
    exit(true);
  end;
  VariantToUtf8(V, s, wasString);
  s := TrimU(s);
  if s = '' then
    exit(false);
  if ToInt64(s, aValue) then
    exit(true);
  if ToDouble(s, d) then
  begin
    aValue := trunc(d);
    exit(true);
  end;
  result := false;
end;

procedure TTestMcpCore.CheckErrorResponse(const aResponse: RawUtf8;
  aExpectedCode: integer; const aMessageContains: RawUtf8);
var
  doc, errDoc: PDocVariantData;
  docVar, errVar: variant;
  code: integer;
  code64: Int64;
  okCode: boolean;
  msg: RawUtf8;
begin
  docVar := _JsonFast(aResponse);
  doc := _Safe(docVar);
  errVar := doc^.GetValueOrNull('error');
  errDoc := _Safe(errVar);
  Check(errDoc^.IsObject);
  okCode := VariantToInt64Loose(errDoc^.GetValueOrDefault('code', 0), code64);
  if okCode then
    code := integer(code64)
  else
    code := 0;
  if aExpectedCode <> 0 then
    CheckEqual(code, aExpectedCode)
  else
    Check(code <> 0);
  if aMessageContains <> '' then
  begin
    errDoc^.GetAsRawUtf8('message', msg);
    Check(PosEx(aMessageContains, msg) > 0);
  end;
end;

function TTestMcpCore.DocPropType(const props: PDocVariantData;
  const propName: RawUtf8): RawUtf8;
var
  prop: variant;
  propDoc: PDocVariantData;
begin
  result := '';
  if (props = nil) or not props^.IsObject then
    exit;
  prop := props^.GetValueOrNull(propName);
  propDoc := _Safe(prop);
  if propDoc^.IsObject then
    propDoc^.GetAsRawUtf8('type', result);
end;

procedure TTestMcpCore.SchemaFromRecord;
var
  schema: variant;
  doc, props, req: PDocVariantData;
  propsVar: variant;
  typ: RawUtf8;
begin
  EnsureCalcParamsRtti;
  schema := TMcpSchemaGenerator.GenerateSchema(TypeInfo(TCalcParams));
  doc := _Safe(schema);
  Check(doc^.IsObject);
  Check(doc^.GetAsRawUtf8('type', typ));
  CheckEqual(typ, 'object');

  propsVar := doc^.GetValueOrNull('properties');
  props := _Safe(propsVar);
  Check(props^.IsObject);
  CheckEqual(DocPropType(props, 'a'), 'integer');
  CheckEqual(DocPropType(props, 'b'), 'integer');
  CheckEqual(DocPropType(props, 'enabled'), 'boolean');
  CheckEqual(DocPropType(props, 'name'), 'string');
end;

procedure TTestMcpCore.JsonRpcProcessor;
var
  proc: TMcpJsonRpcProcessor;
  method: RawUtf8;
  params, requestId: variant;
  ok: boolean;
  response: RawUtf8;
  doc, resultDoc: PDocVariantData;
  responseVar, resultVar: variant;
  jsonrpc: RawUtf8;
  v: variant;
  okFlag: boolean;
begin
  proc := TMcpJsonRpcProcessor.Create('TestServer', '1.0');
  try
    ok := proc.ParseRequest('{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}',
      method, params, requestId);
    Check(ok);
    CheckEqual(method, 'ping');
    Check(VariantToIntegerDef(requestId, 0) = 1);

    v := _ObjFast(['ok', true]);
    response := proc.CreateSuccessResponse(requestId, v);
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    Check(doc^.GetAsRawUtf8('jsonrpc', jsonrpc));
    CheckEqual(jsonrpc, '2.0');
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    Check(resultDoc^.IsObject);
    Check(resultDoc^.GetAsBoolean('ok', okFlag));
    Check(okFlag);
  finally
    proc.Free;
  end;
end;

procedure TTestMcpCore.ServerToolsResources;
var
  server: TMcpServer;
  tool: IMcpTool;
  res: IMcpResource;
  response: RawUtf8;
  doc, resultDoc, listDoc, itemDoc, contentDoc: PDocVariantData;
  responseVar, resultVar, listVar, contentVar, itemVar: variant;
  toolName: RawUtf8;
  tmp: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TCalcTool.Create('calc', 'Add two numbers');
    res := TVersionResource.Create('version://info', 'Version',
      'Server version information', 'application/json');
    server.RegisterTool(tool);
    server.RegisterResource(res);
    server.Start;

    response := Exec(server, 
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}');
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    listVar := resultDoc^.GetValueOrNull('tools');
    listDoc := _Safe(listVar);
    Check(listDoc^.IsArray);
    Check(listDoc^.Count >= 1);
    itemVar := listDoc^.Values[0];
    itemDoc := _Safe(itemVar);
    Check(itemDoc^.GetAsRawUtf8('name', toolName));
    CheckEqual(toolName, 'calc');

    response := Exec(server, 
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"a":5,"b":3,"enabled":true,"name":"x"}}}');
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    contentVar := resultDoc^.GetValueOrNull('content');
    contentDoc := _Safe(contentVar);
    Check(contentDoc^.IsArray);
    Check(contentDoc^.Count = 1);
    itemVar := contentDoc^.Values[0];
    itemDoc := _Safe(itemVar);
    Check(itemDoc^.GetAsRawUtf8('type', tmp));
    CheckEqual(tmp, 'text');
    Check(itemDoc^.GetAsRawUtf8('text', tmp));
    CheckEqual(tmp, '5 + 3 = 8');

    response := Exec(server, 
      '{"jsonrpc":"2.0","id":3,"method":"resources/list","params":{}}');
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    listVar := resultDoc^.GetValueOrNull('resources');
    listDoc := _Safe(listVar);
    Check(listDoc^.IsArray);
    Check(listDoc^.Count = 1);
    itemVar := listDoc^.Values[0];
    itemDoc := _Safe(itemVar);
    Check(itemDoc^.GetAsRawUtf8('uri', tmp));
    CheckEqual(tmp, 'version://info');

    response := Exec(server, 
      '{"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"version://info"}}');
    responseVar := _JsonFast(response);
    doc := _Safe(responseVar);
    resultVar := doc^.GetValueOrNull('result');
    resultDoc := _Safe(resultVar);
    listVar := resultDoc^.GetValueOrNull('contents');
    listDoc := _Safe(listVar);
    Check(listDoc^.IsArray);
    itemVar := listDoc^.Values[0];
    itemDoc := _Safe(itemVar);
    Check(itemDoc^.GetAsRawUtf8('uri', tmp));
    CheckEqual(tmp, 'version://info');
    Check(itemDoc^.GetAsRawUtf8('text', tmp));
    Check(tmp <> '');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ResponseBuilderTextAndFile;
var
  builder: TMcpResponseBuilder;
  response: variant;
  doc, contentDoc, itemDoc, resDoc: PDocVariantData;
  contentVar, itemVar: variant;
  tmpFile: TFileName;
  content: RawByteString;
  base64: RawUtf8;
  tmp: RawUtf8;
begin
  tmpFile := TemporaryFileName;
  content := 'mcp-test';
  Check(FileFromString(content, tmpFile));
  try
    builder := TMcpResponseBuilder.Create;
    try
      builder.AddText('hello');
      builder.AddFile(StringToUtf8(tmpFile));
      response := builder.Build;
    finally
      builder.Free;
    end;

    doc := _Safe(response);
    contentVar := doc^.GetValueOrNull('content');
    contentDoc := _Safe(contentVar);
    if CheckFailed(contentDoc^.IsArray, 'content not array') then
      exit;
    if CheckFailed(contentDoc^.Count >= 2, 'content count < 2') then
      exit;

    itemVar := contentDoc^.Values[0];
    if CheckFailed(_Safe(itemVar, itemDoc), 'content[0] not object') then
      exit;
    Check(itemDoc^.GetAsRawUtf8('type', tmp));
    CheckEqual(tmp, 'text');
    Check(itemDoc^.GetAsRawUtf8('text', tmp));
    CheckEqual(tmp, 'hello');

    itemVar := contentDoc^.Values[1];
    if CheckFailed(_Safe(itemVar, itemDoc), 'content[1] not object') then
      exit;
    Check(itemDoc^.GetAsRawUtf8('type', tmp));
    CheckEqual(tmp, 'resource');
    // EmbeddedResource: the discriminator REQUIRES a nested resource object,
    // and a binary one is BlobResourceContents { uri; mimeType?; blob }. This
    // test used to assert the flat shape - which matched no ContentBlock
    // variant in the schema - and so cemented the defect it should have caught.
    if CheckFailed(itemDoc^.GetAsDocVariant('resource', resDoc),
         'content[1].resource missing') then
      exit;
    base64 := BinToBase64(content);
    Check(resDoc^.GetAsRawUtf8('blob', tmp), 'blob, not data');
    CheckEqual(tmp, base64);
    Check(resDoc^.GetAsRawUtf8('mimeType', tmp));
    Check(tmp <> '');
    Check(resDoc^.GetAsRawUtf8('uri', tmp), 'uri is required');
    Check(IdemPChar(pointer(tmp), 'FILE:///'), 'and is a file URI');
    // the base name, never the directory we read from
    Check(PosEx('/', copy(tmp, 9, maxInt)) = 0,
      'the uri carries no server path');
    Check(not itemDoc^.Exists('data'), 'no stray data property');
    Check(not itemDoc^.Exists('fileName'), 'nor the schema-alien fileName');
  finally
    DeleteFile(tmpFile);
  end;
end;

procedure TTestMcpCore.ServerNotActive;
var
  server: TMcpServer;
  response: RawUtf8;
  doc, errDoc: PDocVariantData;
  docVar, errVar: variant;
  code: integer;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    response := Exec(server, 
      '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}');
    docVar := _JsonFast(response);
    doc := _Safe(docVar);
    errVar := doc^.GetValueOrNull('error');
    errDoc := _Safe(errVar);
    Check(errDoc^.IsObject);
    code := errDoc^.GetValueOrDefault('code', 0);
    Check(code = JSONRPC_INTERNAL_ERROR);
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.BadRequests;
var
  server: TMcpServer;
  tool: IMcpTool;
  res: IMcpResource;
  response: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TCalcTool.Create('calc', 'Add two numbers');
    res := TVersionResource.Create('version://info', 'Version',
      'Server version information', 'application/json');
    server.RegisterTool(tool);
    server.RegisterResource(res);
    server.Start;

    // invalid JSON / malformed envelope -> Invalid Request (-32600)
    response := Exec(server, '{');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');

    response := Exec(server, '{"jsonrpc":"2.0","id":1}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');

    // unknown method -> Method not found (-32601)
    response := Exec(server, '{"jsonrpc":"2.0","id":2,"method":"nope"}');
    CheckErrorResponse(response, JSONRPC_METHOD_NOT_FOUND, 'Method not found');

    // naming something that does not exist is INVALID PARAMS, not an internal
    // error: 2026-07-28 removed the dedicated -32002 "resource not found", so
    // an unknown tool/resource and a missing name are all -32602
    response := Exec(server,
      '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{}}');
    CheckErrorResponse(response, JSONRPC_INVALID_PARAMS, 'Missing tool name');

    response := Exec(server,
      '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"missing","arguments":{}}}');
    CheckErrorResponse(response, JSONRPC_INVALID_PARAMS, 'Tool not found');

    response := Exec(server,
      '{"jsonrpc":"2.0","id":5,"method":"resources/read","params":{}}');
    CheckErrorResponse(response, JSONRPC_INVALID_PARAMS, 'Missing uri');

    response := Exec(server,
      '{"jsonrpc":"2.0","id":6,"method":"resources/read","params":{"uri":"missing://info"}}');
    CheckErrorResponse(response, JSONRPC_INVALID_PARAMS, 'Resource not found');
    // -32603 stays reserved for a genuine server-side failure: a tool that
    // throws still lands there (see ToolExceptionBecomesError)
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ToolExceptionBecomesError;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TThrowingTool.Create('boom', 'Always throws'));
    server.Start;
    // a plain Exception from the tool must be turned into a JSON-RPC internal
    // error (not crash the worker, not escape ExecuteRequest)
    response := Exec(server, 
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"boom",' +
      '"arguments":{"a":1,"b":2,"enabled":true,"name":"x"}}}');
    CheckErrorResponse(response, JSONRPC_INTERNAL_ERROR, 'tool blew up');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.InvalidJsonRpcEnvelope;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // wrong jsonrpc version -> Invalid Request
    response := Exec(server, 
      '{"jsonrpc":"1.0","id":1,"method":"ping","params":{}}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
    // missing jsonrpc field -> Invalid Request
    response := Exec(server, '{"id":2,"method":"ping"}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
    // scalar params (must be object or array) -> Invalid Request
    response := Exec(server, 
      '{"jsonrpc":"2.0","id":3,"method":"ping","params":5}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.NotificationsNoResponse;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    response := Exec(server, '{"jsonrpc":"2.0","method":"ping","params":{}}');
    Check(TrimU(response) = '');
  finally
    server.Free;
  end;
end;

function TTestMcpCore.Exec(aServer: TMcpServer; const aJson: RawUtf8): RawUtf8;
var
  doc: TDocVariantData;
  params: PDocVariantData;
begin
  doc.InitJson(aJson, JSON_FAST);
  // malformed payloads and notifications go through untouched
  if not doc.IsObject or
     VarIsVoid(doc.GetValueOrNull('id')) then
    exit(aServer.ExecuteRequest(aJson));
  // params present but not an object = a deliberately malformed envelope:
  // hand it over untouched, that is exactly what such a test asserts
  if doc.GetValueIndex('params') >= 0 then
    if not doc.GetAsDocVariant('params', params) or
       not params^.IsObject then
      exit(aServer.ExecuteRequest(aJson));
  // build params through the SAME production helper a real client uses, so the
  // tests exercise McpRequestParams instead of a second, divergent copy of the
  // _meta rules (McpPost in test.mcp.transports does the same)
  doc.AddOrUpdateValue('params',
    McpRequestParams(doc.GetValueOrNull('params'), 'mcp.tests', '1.0'));
  result := aServer.ExecuteRequest(doc.ToJson);
end;

procedure TTestMcpCore.DiscoverReportsVersionAndIdentity;
var
  server: TMcpServer;
  response: RawUtf8;
  rv, resv: variant;
  rd, versions: PDocVariantData;
  tmp: RawUtf8;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // server/discover replaces `initialize`: it is the one RPC a client may
    // call to learn versions, capabilities and identity up front
    response := Exec(server, '{"jsonrpc":"2.0","id":1,"method":"server/discover"}');
    rv := _JsonFast(response);
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    Check(rd^.GetAsDocVariant('supportedVersions', versions), 'supportedVersions');
    Check(versions^.IsArray, 'supportedVersions is an array');
    CheckEqual(versions^.Count, 1, 'exactly one supported version');
    CheckEqual(VariantToUtf8(versions^.Values[0]), MCP_PROTOCOL_VERSION,
      'reports 2026-07-28');
    Check(rd^.GetValueIndex('capabilities') >= 0, 'capabilities present');
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'resultType present');
    CheckEqual(tmp, MCP_RESULT_COMPLETE, 'resultType complete');
    // the handshake methods of earlier revisions are gone, not merely ignored
    CheckErrorResponse(Exec(server,
      '{"jsonrpc":"2.0","id":2,"method":"initialize","params":{}}'),
      JSONRPC_METHOD_NOT_FOUND, 'initialize');
    CheckErrorResponse(Exec(server, '{"jsonrpc":"2.0","id":3,"method":"ping"}'),
      JSONRPC_METHOD_NOT_FOUND, 'ping');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.RequestMetaIsMandatory;
var
  server: TMcpServer;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // NOTE: deliberately NOT via Exec — these requests must stay unadorned.
    // Without a handshake the per-request fields are the only place the server
    // can learn the version, so their absence is a malformed request (-32602).
    CheckErrorResponse(server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'),
      JSONRPC_INVALID_PARAMS, MCP_META_PROTOCOL_VERSION);
    // protocolVersion present, but clientCapabilities missing
    CheckErrorResponse(server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"' + MCP_PROTOCOL_VERSION + '"}}}'),
      JSONRPC_INVALID_PARAMS, MCP_META_CLIENT_CAPABILITIES);
    // a version we do not speak is rejected with the MCP-specific code
    CheckErrorResponse(server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"2025-11-25",' +
      '"' + MCP_META_CLIENT_CAPABILITIES + '":{}}}}'),
      MCP_ERROR_UNSUPPORTED_PROTOCOL_VERSION, 'Unsupported protocol version');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ResultsCarryResultTypeAndServerInfo;
var
  server: TMcpServer;
  rv, resv, errv: variant;
  rd, meta, si: PDocVariantData;
  tmp: RawUtf8;
begin
  server := TMcpServer.Create('TestServer', '9.9');
  try
    server.Start;
    rv := _JsonFast(Exec(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'));
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    // resultType is REQUIRED on every result, not just on discover
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'resultType present');
    CheckEqual(tmp, MCP_RESULT_COMPLETE);
    // identity now travels per result instead of once in the handshake
    Check(rd^.GetAsDocVariant('_meta', meta), '_meta present');
    Check(meta^.GetAsDocVariant(MCP_META_SERVER_INFO, si), 'serverInfo present');
    Check(si^.GetAsRawUtf8('name', tmp));
    CheckEqual(tmp, 'TestServer');
    Check(si^.GetAsRawUtf8('version', tmp));
    CheckEqual(tmp, '9.9');
    // an error response carries no result, so it must not be stamped
    rv := _JsonFast(Exec(server, '{"jsonrpc":"2.0","id":2,"method":"nope"}'));
    errv := _Safe(rv)^.GetValueOrNull('error');
    Check(_Safe(errv)^.IsObject, 'error object');
    Check(VarIsVoid(_Safe(rv)^.GetValueOrNull('result')), 'no result on error');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.SubscriptionFilterAndFanout;
var
  server: TMcpServer;
  tools, res, none, extra, prompts: TMcpSubscription;
  queued: TRawUtf8DynArray;
  ack: RawUtf8;
  doc, params, meta, agreed: PDocVariantData;
  dv: variant;
  i: PtrInt;

  // JSON-RPC method of the single queued notification, '' when none
  function DrainedMethod(aSub: TMcpSubscription): RawUtf8;
  var
    d: variant;
  begin
    result := '';
    if not aSub.Drain(queued) then
      exit;
    CheckEqual(length(queued), 1, 'exactly one notification');
    d := _JsonFast(queued[0]);
    _Safe(d)^.GetAsRawUtf8('method', result);
  end;

begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // Three streams with different filters. The spec is strict: "The server
    // MUST NOT send notification types the client has not explicitly
    // requested" — so this is a whitelist, and `none` must stay empty.
    tools := server.OpenSubscription(1,
      _ObjFast(['notifications', _ObjFast(['toolsListChanged', true])]));
    res := server.OpenSubscription(2, _ObjFast(['notifications', _ObjFast([
      'resourcesListChanged', true,
      'resourceSubscriptions', _ArrFast(['version://info'])])]));
    none := server.OpenSubscription(3, _ObjFast([]));
    Check(tools <> nil, 'first stream opens');
    Check(none <> nil, 'a stream without any filter is legal');

    // acknowledgement: first message, carries the subscription id, and echoes
    // only the subset the server actually honors
    ack := server.SubscriptionAcknowledgement(res);
    dv := _JsonFast(ack);
    doc := _Safe(dv);
    Check(doc^.GetAsRawUtf8('method', ack), 'ack method');
    CheckEqual(ack, 'notifications/subscriptions/acknowledged');
    Check(doc^.GetAsDocVariant('params', params), 'ack params');
    Check(params^.GetAsDocVariant('_meta', meta), 'ack _meta');
    CheckEqual(VariantToUtf8(meta^.GetValueOrNull(MCP_META_SUBSCRIPTION_ID)),
      '2', 'ack carries the listen request id as subscription id');
    Check(params^.GetAsDocVariant('notifications', agreed), 'agreed filter');
    Check(agreed^.GetValueIndex('resourcesListChanged') >= 0, 'echoes what we honor');
    Check(agreed^.GetValueIndex('toolsListChanged') < 0,
      'does not echo a type this stream did not request');
    Check(agreed^.GetValueIndex('promptsListChanged') < 0,
      'nor one this stream did not ask for, even though the server can raise it');

    // A stream that DOES ask for prompt changes gets them echoed and delivered.
    // Both halves matter: the capability is announced in server/discover, and
    // for a while Broadcast had no route for this notification at all — it was
    // silently dropped, so the announcement was a promise the server did not
    // keep. The fan-out assertion below is what makes that impossible again.
    prompts := server.OpenSubscription(10,
      _ObjFast(['notifications', _ObjFast(['promptsListChanged', true])]));
    Check(prompts <> nil, 'a prompts stream opens');
    dv := _JsonFast(server.SubscriptionAcknowledgement(prompts));
    Check(_Safe(dv)^.GetAsDocVariant('params', params), 'ack params');
    Check(params^.GetAsDocVariant('notifications', agreed), 'agreed filter');
    Check(agreed^.GetValueIndex('promptsListChanged') >= 0,
      'the server honors promptsListChanged and says so');

    server.RegisterPrompt(TGreetPrompt.Create);
    CheckEqual(DrainedMethod(prompts), 'notifications/prompts/list_changed',
      'registering a prompt really reaches the subscriber');
    Check(not tools.Drain(queued), 'and only that stream');
    // release the slot again, so the cap assertions below still measure the cap
    // and not this test's own bookkeeping
    server.CloseSubscription(prompts);

    // fan-out follows the filters, not the connection
    server.NotifyToolsListChanged;
    CheckEqual(DrainedMethod(tools), 'notifications/tools/list_changed');
    Check(not res.Drain(queued), 'unsubscribed stream gets nothing');
    Check(not none.Drain(queued), 'empty filter gets nothing');

    server.NotifyResourcesListChanged;
    CheckEqual(DrainedMethod(res), 'notifications/resources/list_changed');
    Check(not tools.Drain(queued), 'tools stream unaffected');

    // resource updates are per URI, not per stream
    server.NotifyResourceUpdated('version://info');
    CheckEqual(DrainedMethod(res), 'notifications/resources/updated');
    server.NotifyResourceUpdated('other://thing');
    Check(not res.Drain(queued), 'a URI nobody watches notifies nobody');

    // every message carries the subscription id so a stdio client can
    // demultiplex several streams on one channel
    server.NotifyResourceUpdated('version://info');
    Check(res.Drain(queued), 'queued');
    dv := _JsonFast(queued[0]);
    Check(_Safe(dv)^.GetAsDocVariant('params', params));
    Check(params^.GetAsDocVariant('_meta', meta));
    CheckEqual(VariantToUtf8(meta^.GetValueOrNull(MCP_META_SUBSCRIPTION_ID)), '2');
    CheckEqual(params^.U['uri'], 'version://info', 'the updated URI');

    // the graceful-closure response is the JSON-RPC answer to the long-lived
    // request, so a client can tell an orderly end from a dropped connection
    dv := _JsonFast(server.SubscriptionEndResponse(tools));
    doc := _Safe(dv);
    CheckEqual(VariantToUtf8(doc^.GetValueOrNull('id')), '1', 'correlated by id');
    Check(doc^.GetValueIndex('result') >= 0, 'carries an (empty) result');

    // the cap exists so one client cannot take every HTTP worker thread
    CheckEqual(server.MaxSubscriptions, 8, 'bounded by default');
    CheckEqual(server.MaxSubscriptionsPerPrincipal, 4, 'and per caller');
    // everything here opens as the same (anonymous) principal, so take the
    // per-caller cap out of the way - it has its own test below
    server.MaxSubscriptionsPerPrincipal := 0;
    for i := 4 to 8 do
      Check(server.OpenSubscription(i, _ObjFast([])) <> nil, 'below the cap');
    extra := server.OpenSubscription(99, _ObjFast([]));
    Check(extra = nil, 'refused past the cap instead of starving the pool');

    // Backpressure: a stream nobody drains must not grow without bound —
    // MaxSubscriptions caps how MANY queues exist, not how large one gets.
    // Past the limit the subscription is dropped, which the owning stream
    // turns into a closed stream; the client reconnects and re-reads state.
    Check(not res.Cancelled, 'still live before the flood');
    for i := 0 to MCP_SUBSCRIPTION_MAX_PENDING + 10 do
      server.NotifyResourceUpdated('version://info');
    Check(res.Cancelled, 'a stream that cannot keep up is dropped');
    Check(not res.Drain(queued), 'and its queue is released, not retained');
    Check(not tools.Cancelled, 'the flood does not affect other streams');

    // a closed stream stops receiving, and closing is safe while others live
    server.CloseSubscription(tools);
    server.NotifyToolsListChanged; // must not touch the freed subscription
    Check(not res.Drain(queued), 'still nothing for the resource stream');
  finally
    server.Free; // frees whatever is still registered
  end;
end;

procedure TTestMcpCore.CacheableResultsCarryHints;
var
  server: TMcpServer;
  rv, resv: variant;
  rd: PDocVariantData;
  tmp: RawUtf8;
  ttl: Int64;

  // read the hints off one result; aTtl < 0 asserts they are ABSENT
  procedure CheckHints(const aRequest: RawUtf8; aTtl: integer;
    const aScope, aWhat: RawUtf8);
  begin
    rv := _JsonFast(Exec(server, aRequest));
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    if CheckFailed(rd^.IsObject, aWhat) then
      exit;
    if aTtl < 0 then
    begin
      Check(rd^.GetValueIndex('ttlMs') < 0, aWhat + ' must carry no ttlMs');
      Check(rd^.GetValueIndex('cacheScope') < 0,
        aWhat + ' must carry no cacheScope');
      exit;
    end;
    Check(VariantToInt64(rd^.GetValueOrDefault('ttlMs', -1), ttl),
      aWhat + ' has ttlMs');
    CheckEqual(integer(ttl), aTtl, aWhat + ' ttlMs');
    Check(rd^.GetAsRawUtf8('cacheScope', tmp), aWhat + ' has cacheScope');
    CheckEqual(tmp, aScope, aWhat + ' cacheScope');
  end;

begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TCalcTool.Create('calc', 'Add two numbers'));
    server.RegisterResource(TVersionResource.Create('version://info', 'Version',
      'Server version information', 'application/json'));
    server.Start;

    // The spec REQUIRES caching hints on exactly these four results (of the
    // methods we implement); a client that gets none must assume ttl 0 anyway,
    // but "MUST include" is not satisfied by omission.
    // Defaults are deliberately conservative: never reuse, never share.
    CheckEqual(server.ListCacheTtlMs, MCP_CACHE_TTL_DEFAULT, 'default list ttl');
    CheckEqual(MCP_CACHE_SCOPE[server.ListCacheScope], MCP_CACHE_SCOPE[mcsPrivate],
      'default scope is private: this server cannot know whether the host '  +
      'filters per caller, and public crosses authorization contexts');
    CheckHints('{"jsonrpc":"2.0","id":1,"method":"server/discover"}',
      0, MCP_CACHE_SCOPE[mcsPrivate], 'server/discover');
    CheckHints('{"jsonrpc":"2.0","id":2,"method":"tools/list"}',
      0, MCP_CACHE_SCOPE[mcsPrivate], 'tools/list');
    CheckHints('{"jsonrpc":"2.0","id":3,"method":"resources/list"}',
      0, MCP_CACHE_SCOPE[mcsPrivate], 'resources/list');
    CheckHints('{"jsonrpc":"2.0","id":4,"method":"resources/read",' +
      '"params":{"uri":"version://info"}}', 0, MCP_CACHE_SCOPE[mcsPrivate],
      'resources/read');

    // tools/call has side effects and is NOT in the cacheable list
    CheckHints('{"jsonrpc":"2.0","id":5,"method":"tools/call","params":' +
      '{"name":"calc","arguments":{"A":1,"B":2}}}', -1, '', 'tools/call');

    // configured values reach the wire, and list vs read are independent
    server.ListCacheTtlMs := 300000;
    server.ReadCacheTtlMs := 60000;
    server.ListCacheScope := mcsPublic;
    server.ReadCacheScope := mcsPublic;
    CheckHints('{"jsonrpc":"2.0","id":6,"method":"tools/list"}',
      300000, MCP_CACHE_SCOPE[mcsPublic], 'configured tools/list');
    CheckHints('{"jsonrpc":"2.0","id":7,"method":"resources/read",' +
      '"params":{"uri":"version://info"}}', 60000, MCP_CACHE_SCOPE[mcsPublic],
      'configured resources/read');

    // The two scopes are independent ON PURPOSE: publishing a static tool list
    // must not drag resource CONTENT into shared proxy caches, which is exactly
    // what the spec calls out as typically per-user. A single knob would force
    // that trade-off on the operator.
    server.ReadCacheScope := mcsPrivate;
    CheckHints('{"jsonrpc":"2.0","id":8,"method":"tools/list"}',
      300000, MCP_CACHE_SCOPE[mcsPublic], 'list stays public');
    CheckHints('{"jsonrpc":"2.0","id":9,"method":"resources/read",' +
      '"params":{"uri":"version://info"}}', 60000, MCP_CACHE_SCOPE[mcsPrivate],
      'read can stay private while the list is public');

    // a negative TTL would be a spec violation on the wire ("servers MUST
    // provide a ttlMs value that is >= 0"), so it is clamped, not forwarded
    server.ListCacheTtlMs := -1;
    CheckHints('{"jsonrpc":"2.0","id":10,"method":"tools/list"}',
      0, MCP_CACHE_SCOPE[mcsPublic], 'negative ttl is clamped');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.RequestParamsMergeExistingMeta;
var
  server: TMcpServer;
  params, wrapped: variant;
  doc, meta: PDocVariantData;
  tmp: RawUtf8;
  n: integer;
  i: PtrInt;
begin
  // A caller that already put its own key into _meta (a progressToken, say)
  // must end up with ONE merged _meta. Appending a second one would be
  // invisible: TDocVariantData stores duplicates but every lookup returns the
  // first, so the server would reject the request for missing protocol fields.
  params := _ObjFast([
    'name', 'calc',
    '_meta', _ObjFast(['progressToken', 'tok-1'])]);
  wrapped := McpRequestParams(params, 'mcp.tests', '1.0');
  doc := _Safe(wrapped);
  n := 0;
  for i := 0 to doc^.Count - 1 do
    if doc^.Names[i] = '_meta' then
      inc(n);
  CheckEqual(n, 1, 'exactly one _meta, never a second appended one');
  Check(doc^.GetAsRawUtf8('name', tmp), 'caller params preserved');
  CheckEqual(tmp, 'calc');
  Check(doc^.GetAsDocVariant('_meta', meta), '_meta is an object');
  Check(meta^.GetAsRawUtf8('progressToken', tmp), 'caller _meta key preserved');
  CheckEqual(tmp, 'tok-1');
  Check(meta^.GetAsRawUtf8(MCP_META_PROTOCOL_VERSION, tmp), 'version merged in');
  CheckEqual(tmp, MCP_PROTOCOL_VERSION);
  Check(meta^.GetValueIndex(MCP_META_CLIENT_CAPABILITIES) >= 0,
    'capabilities merged in');

  // and the server actually accepts what the helper produced
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TCalcTool.Create('calc', 'Add two numbers'));
    server.Start;
    EnsureCalcParamsRtti;
    wrapped := _ObjFast(['jsonrpc', '2.0', 'id', 1, 'method', 'tools/call',
      'params', McpRequestParams(_ObjFast([
        'name', 'calc',
        'arguments', _ObjFast(['A', 2, 'B', 3]),
        '_meta', _ObjFast(['progressToken', 'tok-1'])]))]);
    tmp := server.ExecuteRequest(_Safe(wrapped)^.ToJson);
    Check(PosEx('"error"', tmp) = 0, 'a merged _meta passes validation');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ResultMustBeAnObject;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TArrayResultTool.Create('arr', 'Returns an array'));
    server.Start;
    // Copying nothing (the old behaviour) turned a broken handler into an empty
    // success response; the caller then saw resultType:complete and no content.
    // the arguments are beside the point here, but they have to be present:
    // what the schema declares required is enforced before the tool runs
    response := Exec(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"arr",' +
      '"arguments":{"A":1,"B":2}}}');
    CheckErrorResponse(response, JSONRPC_INTERNAL_ERROR, 'must be a JSON object');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.PreflightDecidesHttpStatus;
var
  server: TMcpServer;
  errorJson: RawUtf8;
  status: integer;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // Preflight is what lets a Streamable HTTP transport pick the status BEFORE
    // it opens a stream — once the SSE head is written the answer is 200 by
    // construction, so every non-200 outcome has to be decided here.

    // valid request -> dispatchable, 200
    Check(server.PreflightRequest(
      '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"' + MCP_PROTOCOL_VERSION + '",' +
      '"' + MCP_META_CLIENT_CAPABILITIES + '":{}}}}', errorJson, status),
      'valid request passes');
    CheckEqual(status, HTTP_MCP_SUCCESS);
    CheckEqual(errorJson, '');

    // unimplemented method -> spec REQUIRES 404 (not 400), so a client can tell
    // "this method does not exist" from "this request was refused"
    Check(not server.PreflightRequest(
      '{"jsonrpc":"2.0","id":2,"method":"nope","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"' + MCP_PROTOCOL_VERSION + '",' +
      '"' + MCP_META_CLIENT_CAPABILITIES + '":{}}}}', errorJson, status),
      'unknown method rejected');
    CheckEqual(status, HTTP_MCP_NOT_FOUND);
    CheckErrorResponse(errorJson, JSONRPC_METHOD_NOT_FOUND, 'Method not found');

    // missing _meta -> 400
    Check(not server.PreflightRequest(
      '{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}',
      errorJson, status), 'missing _meta rejected');
    CheckEqual(status, HTTP_MCP_BAD_REQUEST);
    CheckErrorResponse(errorJson, JSONRPC_INVALID_PARAMS, '');
    // ...and it still tells a first-contact client which version to use: there
    // is no handshake left, and server/discover needs valid _meta itself
    Check(PosEx(MCP_PROTOCOL_VERSION, errorJson) > 0,
      'the rejection names the supported version');

    // unsupported version -> 400 with the MCP-specific code
    Check(not server.PreflightRequest(
      '{"jsonrpc":"2.0","id":4,"method":"tools/list","params":{"_meta":' +
      '{"' + MCP_META_PROTOCOL_VERSION + '":"2025-11-25",' +
      '"' + MCP_META_CLIENT_CAPABILITIES + '":{}}}}', errorJson, status),
      'unsupported version rejected');
    CheckEqual(status, HTTP_MCP_BAD_REQUEST);
    CheckErrorResponse(errorJson, MCP_ERROR_UNSUPPORTED_PROTOCOL_VERSION, '');

    // malformed envelope -> 400
    Check(not server.PreflightRequest('{', errorJson, status),
      'malformed envelope rejected');
    CheckEqual(status, HTTP_MCP_BAD_REQUEST);

    // a notification carries no _meta and is never rejected here
    Check(server.PreflightRequest(
      '{"jsonrpc":"2.0","method":"notifications/cancelled"}', errorJson, status),
      'notification passes preflight');
    CheckEqual(status, HTTP_MCP_SUCCESS);
  finally
    server.Free;
  end;
end;

function TTestMcpCore.ExecCaps(aServer: TMcpServer; const aJson: RawUtf8;
  const aCapabilities: variant): RawUtf8;
var
  doc, params, meta: TDocVariantData;
  src: PDocVariantData;
  i: PtrInt;
begin
  doc.InitJson(aJson, JSON_FAST);
  // Build a FRESH params document instead of writing through the pointer
  // GetAsDocVariant hands back: that pointer aliases doc's own storage, so
  // storing it back under 'params' frees the value while the copy source still
  // points at it. Same rule as McpRequestParams, for the same reason.
  params.InitObject([], JSON_FAST);
  if doc.GetAsDocVariant('params', src) and src^.IsObject then
    for i := 0 to src^.Count - 1 do
      if src^.Names[i] <> '_meta' then
        params.AddValue(src^.Names[i], src^.Values[i]);
  meta.InitObject([
    MCP_META_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION,
    MCP_META_CLIENT_CAPABILITIES, aCapabilities], JSON_FAST);
  params.AddValue('_meta', variant(meta));
  doc.AddOrUpdateValue('params', variant(params));
  result := aServer.ExecuteRequest(doc.ToJson);
end;

procedure TTestMcpCore.InputRequiredRoundTrip;
var
  server: TMcpServer;
  tool: TElicitingTool;
  rv, resv: variant;
  rd, requests, entry: PDocVariantData;
  tmp: RawUtf8;
  caps: variant;
begin
  EnsureCalcParamsRtti;
  caps := _ObjFast(['elicitation', _ObjFast([])]);
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TElicitingTool.Create('greet', 'Greet the caller');
    tool.InputMethod := MCP_INPUT_ELICITATION;
    server.RegisterTool(tool);
    server.RegisterResource(TGatedResource.Create('gate://one', 'Gate',
      'Needs a round trip', 'application/json'));
    // a read that WOULD be cacheable, to prove the retry is not
    server.ReadCacheTtlMs := 60000;
    server.ReadCacheScope := mcsPublic;
    server.Start;

    // --- round 1: the tool has nothing to work with and asks -----------------
    rv := _JsonFast(ExecCaps(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{}}}', caps));
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    if CheckFailed(rd^.IsObject, 'round 1 is a result, not an error') then
      exit;
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'round 1 has resultType');
    CheckEqual(tmp, MCP_RESULT_INPUT_REQUIRED, 'round 1 resultType');
    Check(rd^.GetAsRawUtf8('requestState', tmp), 'round 1 carries requestState');
    CheckEqual(tmp, 'round-1-state', 'requestState reaches the client verbatim');
    if not CheckFailed(rd^.GetAsDocVariant('inputRequests', requests),
         'round 1 carries inputRequests') then
    begin
      CheckEqual(requests^.Count, 1, 'one input request');
      CheckEqual(requests^.Names[0], 'who', 'server-assigned identifier');
      if not CheckFailed(_Safe(requests^.Values[0], entry), 'request object') then
      begin
        Check(entry^.GetAsRawUtf8('method', tmp), 'input request has a method');
        CheckEqual(tmp, MCP_INPUT_ELICITATION, 'input request method');
      end;
    end;
    // An interim result is NOT a completed one, and the caching sentence of the
    // spec is scoped to resultType 'complete'. Hints here would tell a proxy to
    // replay "I need input" — the client would then loop on its own cache.
    Check(rd^.GetValueIndex('ttlMs') < 0, 'no ttlMs on an interim result');
    Check(rd^.GetValueIndex('cacheScope') < 0, 'no cacheScope on an interim result');
    CheckEqual(tool.SawResponses, '', 'round 1 saw no answers');

    // --- round 2: the client answers and echoes the state back ---------------
    // note the different id: "The JSON-RPC id MUST be different between the
    // initial request and the retry, as they are independent requests."
    rv := _JsonFast(ExecCaps(server,
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{},"requestState":"round-1-state",' +
      '"inputResponses":{"who":{"action":"accept","content":{"name":"octocat"}}}}}',
      caps));
    resv := _Safe(rv)^.GetValueOrNull('result');
    rd := _Safe(resv);
    if CheckFailed(rd^.IsObject, 'round 2 is a result') then
      exit;
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'round 2 has resultType');
    CheckEqual(tmp, MCP_RESULT_COMPLETE, 'round 2 completes');
    CheckEqual(tool.SawResponses, 'octocat',
      'the handler received the client answers, keyed by its own identifier');
    Check(PosEx('round-1-state', VariantSaveJson(resv)) > 0,
      'the handler also received the state it issued');

    // --- resources/read: the requestState-only shape, and its cacheability ---
    rv := _JsonFast(ExecCaps(server,
      '{"jsonrpc":"2.0","id":3,"method":"resources/read",' +
      '"params":{"uri":"gate://one"}}', caps));
    rd := _Safe(_Safe(rv)^.GetValueOrNull('result'));
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'gated read has resultType');
    CheckEqual(tmp, MCP_RESULT_INPUT_REQUIRED, 'gated read asks for a retry');
    Check(rd^.GetValueIndex('inputRequests') < 0,
      'a requestState-only result omits inputRequests rather than sending {}');
    Check(rd^.GetAsRawUtf8('requestState', tmp), 'gated read carries state');

    rv := _JsonFast(ExecCaps(server,
      '{"jsonrpc":"2.0","id":4,"method":"resources/read",' +
      '"params":{"uri":"gate://one","requestState":"resource-state"}}', caps));
    rd := _Safe(_Safe(rv)^.GetValueOrNull('result'));
    Check(rd^.GetAsRawUtf8('resultType', tmp), 'retried read has resultType');
    CheckEqual(tmp, MCP_RESULT_COMPLETE, 'retried read completes');
    // The server is configured for a shareable read cache, yet this particular
    // answer was shaped by one caller's round trip: handing a proxy `public`
    // on it is the cross-authorization-context replay the spec warns about.
    Check(rd^.GetAsRawUtf8('cacheScope', tmp), 'retried read has cacheScope');
    CheckEqual(tmp, MCP_CACHE_SCOPE[mcsPrivate],
      'an MRTR retry is never shareable, whatever the server was configured for');
    CheckEqual(VariantToIntegerDef(rd^.GetValueOrDefault('ttlMs', -1), -1), 0,
      'an MRTR retry is never reusable either');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.InputRequiredNeedsClientCapability;
var
  server: TMcpServer;
  tool: TElicitingTool;
  response: RawUtf8;
  rv: variant;
  data, required: PDocVariantData;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TElicitingTool.Create('greet', 'Greet the caller');
    tool.InputMethod := MCP_INPUT_ELICITATION;
    server.RegisterTool(tool);
    server.Start;

    // "Servers MUST NOT send an inputRequests that the client has not declared
    // support for in its capabilities." The client here declared none at all,
    // so the elicitation request must never reach the wire — -32021 is the code
    // the spec reserved for exactly this.
    response := ExecCaps(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{}}}', _ObjFast([]));
    CheckErrorResponse(response, MCP_ERROR_MISSING_CLIENT_CAPABILITY,
      'elicitation');
    // "the server MUST return a MissingRequiredClientCapabilityError (-32021)
    // whose data.requiredCapabilities lists the missing capabilities" — the
    // free-text message is for humans; only this field lets a client fix itself.
    rv := _JsonFast(response);
    if not CheckFailed(_Safe(_Safe(rv)^.GetValueOrNull('error'))^.
         GetAsDocVariant('data', data), '-32021 carries data') then
      if not CheckFailed(data^.GetAsArray('requiredCapabilities', required),
           'data.requiredCapabilities is present') then
      begin
        CheckEqual(required^.Count, 1, 'one missing capability');
        CheckEqual(VariantToUtf8(required^.Values[0]), 'elicitation',
          'and it names the one the client must add');
      end;
    // the status a transport MUST send for it: "On HTTP, the response status
    // MUST be 400 Bad Request". It cannot be decided before the handler ran,
    // so it is derived from the finished response.
    CheckEqual(McpHttpStatus(response), 400, '-32021 is an HTTP 400');
    CheckEqual(McpHttpStatus('{"jsonrpc":"2.0","id":1,"result":{}}'), 200,
      'an ordinary result is 200');
    CheckEqual(McpHttpStatus(''), 200, 'a notification leaves the status alone');

    // a client declaring a DIFFERENT capability is still missing this one
    CheckErrorResponse(ExecCaps(server,
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{}}}',
      _ObjFast(['sampling', _ObjFast([])])),
      MCP_ERROR_MISSING_CLIENT_CAPABILITY, 'elicitation');

    // the same tool asking for sampling now goes through, since that IS declared
    tool.InputMethod := MCP_INPUT_SAMPLING;
    Check(PosEx('"resultType":"input_required"', ExecCaps(server,
      '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":' +
      '{"name":"greet","arguments":{}}}',
      _ObjFast(['sampling', _ObjFast([])]))) > 0,
      'a declared capability lets the input request through');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.InputRequiredRejectsMalformedResults;
var
  server: TMcpServer;
  tool: TElicitingTool;
  caps: variant;
  raised: boolean;
begin
  EnsureCalcParamsRtti;
  caps := _ObjFast(['elicitation', _ObjFast([]), 'roots', _ObjFast([])]);
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TElicitingTool.Create('greet', 'Greet the caller');
    tool.InputMethod := MCP_INPUT_ELICITATION;
    server.RegisterTool(tool);
    server.Start;

    // A typo in the method name must fail where it is written, not ship a
    // request object no client can dispatch.
    raised := false;
    try
      McpInputRequest('elicitation/created', _ObjFast([]));
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'McpInputRequest rejects a method outside the allowed three');

    // Neither field set: "Servers MUST include at least one of inputRequests or
    // requestState in every InputRequiredResult" — a client receiving neither
    // could only retry the identical request, forever. Our own handler is at
    // fault, so it surfaces as an internal error, not as a malformed result.
    raised := false;
    try
      server.ValidateInputRequests(Null, '', mcpToolsCall, caps);
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'an empty InputRequiredResult is rejected');

    // The spec lists exactly three methods that may answer with an
    // InputRequiredResult; tools/list is not one of them.
    raised := false;
    try
      server.ValidateInputRequests(
        _ObjFast(['who', McpInputRequest(MCP_INPUT_ROOTS, _ObjFast([]))]),
        '', mcpToolsList, caps);
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'tools/list may not answer with an InputRequiredResult');

    // A present-but-wrong-shaped inputRequests must NOT be silently treated as
    // "requestState-only": it would be dropped from the result and the client
    // would retry without the input the handler is waiting for — a round trip
    // that never terminates, instead of a defect anybody notices.
    raised := false;
    try
      server.ValidateInputRequests(_ArrFast(['not', 'a', 'map']),
        'some-state', mcpToolsCall, caps);
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'an inputRequests that is not an object is rejected, even '  +
      'when a requestState would have carried the result on its own');

    // params of a server-to-client request is an object; null would produce a
    // message this server's own parser rejects
    raised := false;
    try
      McpInputRequest(MCP_INPUT_ELICITATION, Null);
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'McpInputRequest rejects non-object params');

    // a well-formed one on an allowed method passes
    server.ValidateInputRequests(
      _ObjFast(['who', McpInputRequest(MCP_INPUT_ROOTS, _ObjFast([]))]),
      '', mcpResourcesRead, caps);
    Check(true, 'a valid InputRequiredResult passes the gate');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.MrtrRetryFieldsAreTypedAndCountAsPresent;
var
  server: TMcpServer;
  rv: variant;
  rd: PDocVariantData;
  tmp: RawUtf8;
  caps: variant;

  // read the cacheScope off a resources/read answer
  function ScopeOf(const aParams: RawUtf8): RawUtf8;
  begin
    rv := _JsonFast(ExecCaps(server, '{"jsonrpc":"2.0","id":1,' +
      '"method":"resources/read","params":' + aParams + '}', caps));
    rd := _Safe(_Safe(rv)^.GetValueOrNull('result'));
    result := '';
    rd^.GetAsRawUtf8('cacheScope', result);
  end;

begin
  caps := _ObjFast([]);
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterResource(TVersionResource.Create('version://info', 'Version',
      'Server version information', 'application/json'));
    server.ReadCacheTtlMs := 60000;
    server.ReadCacheScope := mcsPublic;
    server.Start;

    // baseline: a plain read really is shareable with this configuration
    CheckEqual(ScopeOf('{"uri":"version://info"}'), MCP_CACHE_SCOPE[mcsPublic],
      'a plain read follows the configured cache scope');

    // PRESENCE decides, not content. VarIsVoid() treats an EMPTY object as
    // void, so testing the value would let this exact request through as
    // shareable — and a proxy could then replay one caller's round-trip answer
    // to everyone else.
    CheckEqual(ScopeOf('{"uri":"version://info","inputResponses":{}}'),
      MCP_CACHE_SCOPE[mcsPrivate],
      'an empty inputResponses is still a retry, and still personal');
    CheckEqual(ScopeOf('{"uri":"version://info","requestState":""}'),
      MCP_CACHE_SCOPE[mcsPrivate],
      'an empty requestState is present, not absent');

    // Both fields are typed on the wire. Accepting a number as requestState
    // would stringify it and hand a handler a state this server never issued.
    CheckErrorResponse(ExecCaps(server, '{"jsonrpc":"2.0","id":2,' +
      '"method":"resources/read","params":{"uri":"version://info",' +
      '"requestState":42}}', caps), JSONRPC_INVALID_PARAMS, 'requestState');
    CheckErrorResponse(ExecCaps(server, '{"jsonrpc":"2.0","id":3,' +
      '"method":"resources/read","params":{"uri":"version://info",' +
      '"inputResponses":["nope"]}}', caps),
      JSONRPC_INVALID_PARAMS, 'inputResponses');

    // Retry fields on a method that cannot take part in a round trip are
    // meaningless — they must not quietly degrade that method's cacheability.
    rv := _JsonFast(ExecCaps(server, '{"jsonrpc":"2.0","id":4,' +
      '"method":"tools/list","params":{"requestState":"stray"}}', caps));
    rd := _Safe(_Safe(rv)^.GetValueOrNull('result'));
    Check(rd^.GetAsRawUtf8('cacheScope', tmp), 'tools/list still has a scope');
    CheckEqual(tmp, MCP_CACHE_SCOPE[server.ListCacheScope],
      'a stray requestState on tools/list changes nothing');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.RequestStateCodecBindsAndExpires;
var
  codec, other: TMcpRequestStateCodec;
  blob, tampered: RawUtf8;
  state: variant;
  dot: PtrInt;
  weak: boolean;
begin
  codec := TMcpRequestStateCodec.Create('0123456789abcdef0123456789abcdef');
  try
    blob := codec.Encode(_ObjFast(['step', 1]), 'user-42',
      'tools/call:greet');

    // the happy path: same principal, same request, inside the TTL
    Check(codec.Decode(blob, 'user-42', 'tools/call:greet', state),
      'a freshly issued state verifies');
    CheckEqual(VariantToIntegerDef(_Safe(state)^.GetValueOrDefault('step', 0), 0),
      1, 'the handler gets its own state back');

    // "servers MUST treat requestState as an attacker-controlled input" — the
    // three replay defences the spec asks for, each on its own:
    Check(not codec.Decode(blob, 'user-43', 'tools/call:greet', state),
      'state presented by a different principal is rejected');
    Check(not codec.Decode(blob, 'user-42', 'tools/call:other', state),
      'state presented on a different request is rejected');

    // flipping a single payload byte must break the signature. The payload is
    // the part before the dot; corrupting the last character of it is enough.
    dot := PosExChar('.', blob);
    tampered := copy(blob, 1, dot - 2) + 'X' + copy(blob, dot - 1, maxInt);
    Check(not codec.Decode(tampered, 'user-42', 'tools/call:greet', state),
      'a tampered payload fails verification');
    Check(not codec.Decode(copy(blob, 1, dot - 1), 'user-42',
      'tools/call:greet', state), 'a blob without a signature is rejected');
    Check(not codec.Decode('', 'user-42', 'tools/call:greet', state),
      'an empty blob is rejected');

    // a state signed with someone else's secret is not ours, however
    // well-formed it looks — this is the cross-server forgery case
    other := TMcpRequestStateCodec.Create('fedcba9876543210fedcba9876543210');
    try
      Check(not codec.Decode(other.Encode(_ObjFast(['step', 9]), 'user-42',
        'tools/call:greet'), 'user-42', 'tools/call:greet', state),
        'a state signed with a different secret is rejected');
    finally
      other.Free;
    end;
  finally
    codec.Free;
  end;

  // the deadline is inside the signed payload, so it cannot be pushed out
  codec := TMcpRequestStateCodec.Create('0123456789abcdef0123456789abcdef', 1);
  try
    blob := codec.Encode(_ObjFast(['step', 1]), 'user-42', 'tools/call:greet');
    SleepHiRes(1100);
    Check(not codec.Decode(blob, 'user-42', 'tools/call:greet', state),
      'state presented after its TTL lapsed is rejected');
  finally
    codec.Free;
  end;

  // a secret too short to be unguessable is refused outright: signing with it
  // would look like protection while providing none
  weak := false;
  try
    TMcpRequestStateCodec.Create('short').Free;
  except
    on EMcpException do
      weak := true;
  end;
  Check(weak, 'a weak secret is refused at construction');
end;

procedure TTestMcpCore.ScopeHierarchyAndBearerParsing;
var
  granted: TRawUtf8DynArray;
begin
  // "Servers MUST account for scope hierarchies, where a broader scope implies
  // narrower ones."
  granted := nil;
  AddRawUtf8(granted, 'files');
  AddRawUtf8(granted, 'tools:call');
  Check(McpScopeSatisfied(granted, 'files'), 'an exact grant satisfies');
  Check(McpScopeSatisfied(granted, 'files:read'),
    'a broader scope implies the narrower ones below it');
  Check(McpScopeSatisfied(granted, 'files:read:meta'),
    'and every level below, not just one');
  Check(McpScopeSatisfied(granted, 'tools:call'), 'a deeper exact grant works');
  // the implication runs ONE way: reading it backwards would silently turn
  // every narrow grant into the broad one it was carved out of
  Check(not McpScopeSatisfied(granted, 'tools'),
    'a narrow grant must NOT imply the broader scope above it');
  // the boundary is ':' and nothing else, or 'file' would cover 'files:write'
  Check(not McpScopeSatisfied(granted, 'filesystem'),
    'a prefix that does not end on a separator is a different scope');
  Check(not McpScopeSatisfied(granted, 'admin'), 'an unrelated scope fails');
  Check(not McpScopeSatisfied(nil, 'files'),
    'no grants satisfy nothing: an unauthorized request must fail every check');
  // The ':' hierarchy is a convention, not an OAuth rule. A deployment where
  // `admin` and `admin:delete` are unrelated permissions must be able to turn
  // it off, or the first would silently grant the second.
  Check(not McpScopeSatisfied(granted, 'files:read', {hierarchical=}false),
    'exact matching can be demanded where the convention does not hold');
  Check(McpScopeSatisfied(granted, 'files', {hierarchical=}false),
    'and an exact grant still satisfies then');
  Check(McpScopeSatisfied(nil, ''), 'an operation requiring nothing passes');

  // RFC 7235: the scheme is case-insensitive, the token is not
  CheckEqual(McpBearerToken('Bearer abc123'), 'abc123', 'a plain bearer header');
  CheckEqual(McpBearerToken('bearer abc123'), 'abc123', 'scheme is case-insensitive');
  CheckEqual(McpBearerToken('  Bearer   abc123  '), 'abc123',
    'surrounding and inner whitespace is tolerated');
  CheckEqual(McpBearerToken('Basic abc123'), '', 'another scheme yields nothing');
  CheckEqual(McpBearerToken('Bearer'), '', 'a scheme without a token is no token');
  CheckEqual(McpBearerToken('Bearer   '), '', 'nor is one with only spaces');
  CheckEqual(McpBearerToken(''), '', 'an absent header yields nothing');
  CheckEqual(McpBearerToken('BearerToken xyz'), '',
    'the space after the scheme is required');
end;

procedure TTestMcpCore.ProtectedResourceMetadataAndChallenges;
var
  server: TMcpServer;
  verifier: TFakeVerifier;
  doc: TDocVariantData;
  authCtx: TMcpAuthContext;
  scopes, servers: TRawUtf8DynArray;
  tmp: RawUtf8;
  raised: boolean;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    // --- an open server stays open: authorization is OPTIONAL in MCP --------
    Check(not server.IsProtected, 'no verifier means no protection');
    CheckEqual(ord(server.Authorize('', authCtx)), ord(mtrValid),
      'an unprotected server accepts an unauthenticated request');

    // --- switching protection on ------------------------------------------
    verifier := TFakeVerifier.Create;
    verifier.GoodToken := 'good';
    AddRawUtf8(verifier.Scopes, 'mcp:use');
    server.TokenVerifier := verifier;
    Check(server.IsProtected, 'a verifier switches protection on');

    // Without a canonical URI there is nothing to validate an audience
    // against, so the server refuses to guess rather than accept everything.
    raised := false;
    try
      server.Authorize('Bearer good', authCtx);
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'a protected server without AuthResource fails closed');

    server.AuthResource := 'https://mcp.example.com/mcp';
    scopes := nil;
    AddRawUtf8(scopes, 'mcp:use');
    server.ScopesSupported := scopes;
    servers := nil;
    AddRawUtf8(servers, 'https://as.example.com');
    server.AuthorizationServers := servers;

    // "The Protected Resource Metadata document ... MUST include the
    // authorization_servers field containing at least one authorization
    // server" — a protected server without one publishes a document no client
    // can act on, so Start refuses it while it is still a developer's problem.
    raised := false;
    try
      server.AuthorizationServers := nil;
      server.Start;
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'a protected server with no authorization server is refused');
    server.AuthorizationServers := servers;

    // --- the RFC 9728 document --------------------------------------------
    doc.InitJson(server.ProtectedResourceMetadata, JSON_FAST);
    Check(doc.IsObject, 'the metadata document is an object');
    CheckEqual(doc.U['resource'], 'https://mcp.example.com/mcp',
      'resource is the canonical URI clients put in the RFC 8707 parameter');
    Check(doc.GetValueIndex('authorization_servers') >= 0,
      'it names where to authenticate — without this a client is stuck');
    Check(doc.GetValueIndex('scopes_supported') >= 0, 'and the base scopes');
    Check(PosEx('header', doc.ToJson) > 0,
      'bearer_methods_supported says header: a token MUST NOT ride in the query');

    // "MCP Servers SHOULD NOT include offline_access in ... scopes_supported"
    raised := false;
    try
      scopes := nil;
      AddRawUtf8(scopes, 'offline_access');
      server.ScopesSupported := scopes;
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'offline_access is a client concern and is refused here');

    // --- refusals ----------------------------------------------------------
    CheckEqual(ord(server.Authorize('', authCtx)), ord(mtrMissing),
      'no header is a missing token');
    CheckEqual(ord(server.Authorize('Bearer wrong', authCtx)), ord(mtrInvalid),
      'an unknown token is invalid');
    CheckEqual(verifier.SeenResource, 'https://mcp.example.com/mcp',
      'the verifier is told which audience to check against');
    // a verifier may write a principal before it decides to refuse; the server
    // must not pass that on, or a caller checking only the context sees an
    // identity that was never authenticated
    Check(not authCtx.IsAuthenticated, 'a refused context is not authenticated');
    CheckEqual(authCtx.UserID, '',
      'and carries no principal the verifier may have written before refusing');

    // --- statuses ----------------------------------------------------------
    // "401 Unauthorized — Authorization required or token invalid",
    // "403 Forbidden — Invalid scopes or insufficient permissions"
    CheckEqual(server.AuthHttpStatus(mtrMissing), 401, 'missing token -> 401');
    CheckEqual(server.AuthHttpStatus(mtrInvalid), 401, 'invalid token -> 401');
    CheckEqual(server.AuthHttpStatus(mtrExpired), 401, 'expired token -> 401');
    // a token minted for another service is not merely too weak here — it does
    // not belong to this server at all, so it is a 401 and not a 403
    CheckEqual(server.AuthHttpStatus(mtrWrongAudience), 401,
      'a token for another resource -> 401, not 403');
    CheckEqual(server.AuthHttpStatus(mtrInsufficientScope), 403,
      'a valid token lacking a scope -> 403');
    CheckEqual(server.AuthHttpStatus(mtrValid), 200, 'a good token -> 200');

    // --- challenges --------------------------------------------------------
    tmp := server.AuthChallenge(mtrMissing);
    Check(IdemPChar(pointer(tmp), 'BEARER'), 'the challenge names the scheme');
    // RFC 9728 INSERTS the well-known segment between host and path — it does
    // not append it. Appending would send every client to a 404, and a client
    // that cannot read the metadata never finds the authorization server.
    Check(PosEx('resource_metadata="https://mcp.example.com' +
      MCP_WELL_KNOWN_RESOURCE + '/mcp"', tmp) > 0,
      'and points at the document naming the authorization server');
    CheckEqual(McpResourceMetadataUrl('https://mcp.example.com/public/mcp'),
      'https://mcp.example.com' + MCP_WELL_KNOWN_RESOURCE + '/public/mcp',
      'the resource path becomes a SUFFIX of the well-known path');
    CheckEqual(McpResourceMetadataUrl('https://mcp.example.com'),
      'https://mcp.example.com' + MCP_WELL_KNOWN_RESOURCE,
      'a resource without a path keeps the root form');
    CheckEqual(McpResourceMetadataUrl('https://mcp.example.com/'),
      'https://mcp.example.com' + MCP_WELL_KNOWN_RESOURCE,
      'and a bare trailing slash is not a path');
    CheckEqual(McpResourceMetadataPath('https://mcp.example.com/mcp'),
      MCP_WELL_KNOWN_RESOURCE + '/mcp', 'the transport routes that same path');
    // "If the request lacks any authentication information … the resource server
    // SHOULD NOT include an error code" (RFC 6750 §3.1): nothing was presented,
    // so nothing was rejected.
    Check(PosEx('error=', tmp) = 0, 'a plain 401 carries no error code');

    // A token that WAS presented and refused must say why, in the challenge —
    // the client acts on WWW-Authenticate, not on the response body. Without a
    // code it cannot tell "get a fresh token" from "ask for more scope".
    Check(PosEx('error="invalid_token"',
      server.AuthChallenge(mtrInvalid)) > 0,
      'a refused token is named invalid_token');
    Check(PosEx('error="invalid_token"',
      server.AuthChallenge(mtrExpired)) > 0,
      'an expired one too — a fresh token is what fixes it');
    // A wrong audience is not a weaker permission, it is a token for someone
    // else — also invalid_token, never insufficient_scope.
    Check(PosEx('error="invalid_token"',
      server.AuthChallenge(mtrWrongAudience)) > 0,
      'and a token minted for another resource');

    tmp := server.AuthChallenge(mtrInsufficientScope, 'files:write tools:call');
    Check(PosEx('error="insufficient_scope"', tmp) > 0,
      'a scope refusal says so');
    Check(PosEx('scope="files:write tools:call"', tmp) > 0,
      'and names ALL scopes the operation needs, in one challenge');
    Check(tmp[length(tmp)] <> ',', 'the challenge has no trailing comma');

    // --- ExecuteRequest itself is NOT gated, on purpose --------------------
    // Authorization lives in the HTTP transports, because it IS transport-level
    // ("The Model Context Protocol provides authorization capabilities at the
    // transport level"). stdio must stay usable on a protected server — the
    // spec says stdio SHOULD NOT use this scheme and take credentials from the
    // environment instead — and an in-process caller already holds the server
    // object, so there is nothing left to authorize.
    // Anyone exposing ExecuteRequest over a NEW transport must call Authorize
    // there, exactly as the two HTTP transports do.
    server.Start;
    Check(PosEx('"result"', Exec(server,
      '{"jsonrpc":"2.0","id":9,"method":"tools/list"}')) > 0,
      'in-process dispatch stays open on a protected server: authorization is ' +
      'a transport concern, and stdio SHOULD NOT use it at all');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.CursorCodecRejectsForgeries;
var
  after: RawUtf8;
begin
  CheckEqual(McpEncodeCursor('calc'), McpEncodeCursor('calc'),
    'the same position always encodes to the same token');
  Check(McpEncodeCursor('calc') <> 'calc',
    'and not to the bare name, which would invite clients to build their own');

  Check(McpDecodeCursor(McpEncodeCursor('calc'), after), 'our own token decodes');
  CheckEqual(after, 'calc', 'back to the position it named');

  // "an empty string is a valid cursor and thus MUST NOT be treated as the end
  // of results" — it names no position, so the page starts at the beginning
  Check(McpDecodeCursor('', after), 'an empty cursor is VALID, not an error');
  CheckEqual(after, '', 'and starts from the beginning');

  // Anything we did not mint must fail rather than land on a plausible
  // position: silently restarting at the top would loop a client over page one
  // forever instead of telling it the token is bad.
  Check(not McpDecodeCursor('not-base64-$$$', after), 'garbage is refused');
  Check(not McpDecodeCursor(BinToBase64uri('x:calc'), after),
    'a well-formed token without our marker is refused too');
  Check(not McpDecodeCursor('Y2FsYw', after),
    'and so is a bare base64 name someone built by hand');

  // The keyset comparison is StrComp, which stops at the first #0. A cursor
  // carrying an embedded null would be compared only up to that byte — a token
  // that looks valid but silently selects the wrong window. Names never contain
  // #0, so it cannot be one of ours.
  Check(not McpDecodeCursor(BinToBase64uri('n:foo'#0'bar'), after),
    'a cursor with an embedded null is refused, not silently truncated');
  CheckEqual(after, '', 'and yields no position');
end;

procedure TTestMcpCore.ListsAreSortedAndPaginated;
var
  server: TMcpServer;
  resp, cursor, seen: RawUtf8;
  req: RawUtf8;
  doc: TDocVariantData;
  res, arr: PDocVariantData;
  pages, i: integer;
  hasNext: boolean;
begin
  server := TMcpServer.Create('paging', '1.0');
  try
    // Registered in deliberately NON-alphabetical order: a dictionary
    // enumerates in hash order, so without sorting the sequence would depend on
    // insertion and on the hash function — and a cursor over an unordered set
    // repeats some entries while skipping others.
    server.RegisterTool(TCalcTool.Create('delta', 'd'));
    server.RegisterTool(TCalcTool.Create('alpha', 'a'));
    server.RegisterTool(TCalcTool.Create('echo', 'e'));
    server.RegisterTool(TCalcTool.Create('bravo', 'b'));
    server.RegisterTool(TCalcTool.Create('charlie', 'c'));
    server.ListPageSize := 2;
    server.Start;

    // Walk every page, collecting the names in the order they arrive.
    seen := '';
    cursor := '';
    pages := 0;
    repeat
      if pages = 0 then
        req := '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
      else
        req := FormatUtf8('{"jsonrpc":"2.0","id":1,"method":"tools/list",' +
          '"params":{"cursor":"%"}}', [cursor]);
      resp := Exec(server, req);
      doc.Clear;
      doc.InitJson(resp, JSON_FAST);
      Check(doc.GetAsDocVariant('result', res), 'a page comes back as a result');
      Check(res^.GetAsDocVariant('tools', arr), 'carrying a tools array');
      for i := 0 to arr^.Count - 1 do
        seen := seen + _Safe(arr^.Values[i])^.U['name'] + ' ';
      cursor := res^.U[MCP_RESULT_NEXT_CURSOR];
      hasNext := res^.GetValueIndex(MCP_RESULT_NEXT_CURSOR) >= 0;
      if hasNext then
        Check(arr^.Count = 2, 'a non-final page is full')
      else
        Check(arr^.Count <= 2, 'the final page holds the remainder');
      inc(pages);
      Check(pages <= 5, 'pagination terminates instead of looping');
    until not hasNext;

    CheckEqual(pages, 3, '5 tools at 2 per page take three pages');
    // The whole point: every tool exactly once, in a defined order, across
    // pages — not the hash order the dictionary would have produced.
    CheckEqual(seen, 'alpha bravo charlie delta echo ',
      'every tool appears exactly once, sorted, across all pages');
    // "Clients SHOULD treat a missing nextCursor as the end of results" — so
    // the last page must OMIT it, not send an empty one.
    Check(not hasNext, 'the last page omits nextCursor entirely');

    // An empty cursor is valid and starts from the beginning (spec: it MUST NOT
    // be read as the end of results).
    resp := Exec(server, '{"jsonrpc":"2.0","id":2,"method":"tools/list",' +
      '"params":{"cursor":""}}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'an empty cursor is served');
    Check(res^.GetAsDocVariant('tools', arr), 'with a page of tools');
    CheckEqual(_Safe(arr^.Values[0])^.U['name'], 'alpha',
      'and starts at the first entry, not at the end');

    // "Invalid cursors SHOULD result in an error with code -32602"
    CheckErrorResponse(Exec(server,
      '{"jsonrpc":"2.0","id":3,"method":"tools/list",' +
      '"params":{"cursor":"forged"}}'), -32602, 'cursor');

    // Paging off: one page, no cursor, still sorted.
    server.ListPageSize := 0;
    resp := Exec(server, '{"jsonrpc":"2.0","id":4,"method":"tools/list"}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'unpaginated list is served');
    Check(res^.GetValueIndex(MCP_RESULT_NEXT_CURSOR) < 0,
      'with no nextCursor at all');
    Check(res^.GetAsDocVariant('tools', arr), 'and every tool at once');
    CheckEqual(arr^.Count, 5, 'all five in one page');
    CheckEqual(_Safe(arr^.Values[0])^.U['name'], 'alpha', 'still sorted');

    // A negative page size cannot mean anything; it must not be read as "one
    // entry per page" or wrap into a huge window. Same harmless reading as 0.
    server.ListPageSize := -3;
    resp := Exec(server, '{"jsonrpc":"2.0","id":5,"method":"tools/list"}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'a negative page size still answers');
    Check(res^.GetAsDocVariant('tools', arr), 'with tools');
    CheckEqual(arr^.Count, 5, 'and behaves like paging off');

    // THE off-by-one: an item count that is an exact multiple of the page size.
    // The last full page must NOT carry a nextCursor, or the client fetches an
    // empty page and cannot tell that from a truncated list.
    server.UnregisterTool('echo');
    server.ListPageSize := 2;
    resp := Exec(server, '{"jsonrpc":"2.0","id":6,"method":"tools/list"}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'page one of four items');
    cursor := res^.U[MCP_RESULT_NEXT_CURSOR];
    Check(res^.GetValueIndex(MCP_RESULT_NEXT_CURSOR) >= 0,
      'four items at two per page: page one has a successor');
    resp := Exec(server, FormatUtf8('{"jsonrpc":"2.0","id":7,' +
      '"method":"tools/list","params":{"cursor":"%"}}', [cursor]));
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'page two answers');
    Check(res^.GetAsDocVariant('tools', arr), 'with the remaining tools');
    CheckEqual(arr^.Count, 2, 'exactly the rest');
    Check(res^.GetValueIndex(MCP_RESULT_NEXT_CURSOR) < 0,
      'and NO nextCursor: the list ended exactly on a page boundary');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.PromptsListAndGet;
var
  server: TMcpServer;
  resp: RawUtf8;
  doc: TDocVariantData;
  res, arr, entry, msgs: PDocVariantData;
begin
  server := TMcpServer.Create('prompts', '1.0');
  try
    server.RegisterPrompt(TReviewPrompt.Create('code_review'));
    server.RegisterPrompt(TGreetPrompt.Create);
    server.Start;

    // --- the capability must be announced, or a client never asks -----------
    resp := Exec(server, '{"jsonrpc":"2.0","id":1,"method":"server/discover"}');
    Check(PosEx('"prompts":{"listChanged":true}', resp) > 0,
      'server/discover declares the prompts capability with listChanged');

    // --- prompts/list ------------------------------------------------------
    resp := Exec(server, '{"jsonrpc":"2.0","id":2,"method":"prompts/list"}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'prompts/list answers');
    Check(res^.GetAsDocVariant('prompts', arr), 'with a prompts array');
    CheckEqual(arr^.Count, 2, 'both prompts are listed');
    // sorted, like every other list
    CheckEqual(_Safe(arr^.Values[0])^.U['name'], 'code_review', 'sorted first');
    CheckEqual(_Safe(arr^.Values[1])^.U['name'], 'greet', 'sorted second');

    entry := _Safe(arr^.Values[0]);
    CheckEqual(entry^.U['title'], 'Request Code Review', 'title is published');
    Check(entry^.GetAsDocVariant('arguments', msgs), 'arguments are published');
    CheckEqual(msgs^.Count, 1, 'one declared argument');
    CheckEqual(_Safe(msgs^.Values[0])^.U['name'], 'code', 'named code');
    Check(_Safe(msgs^.Values[0])^.B['required'], 'and marked required');

    // Optional fields are OMITTED, not emitted empty: a client must be able to
    // tell "no title" from "a title that happens to be blank".
    entry := _Safe(arr^.Values[1]);
    Check(entry^.GetValueIndex('title') < 0, 'an absent title is left out');
    Check(entry^.GetValueIndex('arguments') < 0,
      'and a prompt without arguments publishes no arguments key');

    // prompts/list is one of the methods that MUST carry caching hints
    Check(res^.GetValueIndex('ttlMs') >= 0, 'prompts/list carries ttlMs');
    Check(res^.GetValueIndex('cacheScope') >= 0, 'and cacheScope');

    // --- prompts/get, bare-array shape ------------------------------------
    resp := Exec(server, '{"jsonrpc":"2.0","id":3,"method":"prompts/get",' +
      '"params":{"name":"code_review","arguments":{"code":"x := 1;"}}}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'prompts/get answers');
    Check(res^.GetAsDocVariant('messages', msgs),
      'a prompt returning a bare array gets it wrapped into messages');
    CheckEqual(msgs^.Count, 1, 'one message');
    CheckEqual(_Safe(msgs^.Values[0])^.U['role'], 'user', 'from the user');
    Check(PosEx('x := 1;', resp) > 0, 'with the argument interpolated');
    CheckEqual(res^.U['resultType'], 'complete', 'and it is a complete result');

    // --- prompts/get, full-envelope shape ---------------------------------
    resp := Exec(server,
      '{"jsonrpc":"2.0","id":4,"method":"prompts/get","params":{"name":"greet"}}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'the envelope shape answers too');
    CheckEqual(res^.U['description'], 'A greeting',
      'and is passed through untouched');
    Check(res^.GetAsDocVariant('messages', msgs), 'with its own messages');

    // --- errors ------------------------------------------------------------
    // "Invalid prompt name: -32602"
    CheckErrorResponse(Exec(server,
      '{"jsonrpc":"2.0","id":5,"method":"prompts/get","params":{"name":"nope"}}'),
      -32602, 'Prompt not found');
    // "Missing required arguments: -32602"
    CheckErrorResponse(Exec(server,
      '{"jsonrpc":"2.0","id":6,"method":"prompts/get",' +
      '"params":{"name":"code_review"}}'), -32602, 'required');
    // a request without a name at all
    CheckErrorResponse(Exec(server,
      '{"jsonrpc":"2.0","id":7,"method":"prompts/get","params":{}}'),
      -32602, 'Missing prompt name');

    // --- prompts/get as a Multi Round-Trip Request -------------------------
    // "Servers MAY also respond to prompts/get with an InputRequiredResult":
    // the third method allowed to do so, alongside tools/call and
    // resources/read. Round one asks, round two answers.
    server.RegisterPrompt(TElicitingPrompt.Create);
    resp := ExecCaps(server,
      '{"jsonrpc":"2.0","id":8,"method":"prompts/get","params":{"name":"eliciting"}}',
      _ObjFast(['elicitation', _ObjFast([])]));
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'round one answers');
    CheckEqual(res^.U['resultType'], 'input_required',
      'an interim result, not a complete one');
    Check(res^.GetAsDocVariant('inputRequests', msgs),
      'naming what the server needs');
    Check(msgs^.GetValueIndex('topic') >= 0, 'under the key it assigned');
    // An interim result carries no caching hints — there is nothing stable yet.
    Check(res^.GetValueIndex('ttlMs') < 0, 'and no caching hints');

    resp := ExecCaps(server,
      '{"jsonrpc":"2.0","id":9,"method":"prompts/get","params":{"name":"eliciting",' +
      '"inputResponses":{"topic":{"content":"pascal"}}}}',
      _ObjFast(['elicitation', _ObjFast([])]));
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'round two answers');
    CheckEqual(res^.U['resultType'], 'complete', 'and completes this time');
    Check(res^.GetAsDocVariant('messages', msgs), 'with the rendered messages');
    Check(PosEx('About pascal', resp) > 0,
      'built from the answer the client sent back');

    // --- prompts/list paginates too ----------------------------------------
    // The page arithmetic is shared (McpPageRange), but each list wires its own
    // array key and nextCursor — so each needs proof it was wired at all.
    server.ListPageSize := 1;
    resp := Exec(server, '{"jsonrpc":"2.0","id":10,"method":"prompts/list"}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'prompts/list page one');
    Check(res^.GetAsDocVariant('prompts', arr), 'under the prompts key');
    CheckEqual(arr^.Count, 1, 'one per page');
    resp := Exec(server, FormatUtf8('{"jsonrpc":"2.0","id":11,' +
      '"method":"prompts/list","params":{"cursor":"%"}}',
      [res^.U[MCP_RESULT_NEXT_CURSOR]]));
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'prompts/list page two');
    Check(res^.GetAsDocVariant('prompts', arr), 'again under prompts');
    CheckEqual(arr^.Count, 1, 'the next one');
    CheckEqual(_Safe(arr^.Values[0])^.U['name'], 'eliciting',
      'sorted: code_review, then eliciting');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.TemplatesAndCompletion;
var
  server: TMcpServer;
  resp: RawUtf8;
  doc: TDocVariantData;
  res, arr, entry, comp: PDocVariantData;
begin
  server := TMcpServer.Create('templates', '1.0');
  try
    server.RegisterResourceTemplate(TFilesTemplate.Create);
    server.RegisterPrompt(TReviewPrompt.Create('code_review'));
    server.RegisterPrompt(TFloodPrompt.Create);
    server.Start;

    // "Servers that support completions MUST declare the completions capability"
    resp := Exec(server, '{"jsonrpc":"2.0","id":1,"method":"server/discover"}');
    Check(PosEx('"completions":{}', resp) > 0,
      'server/discover declares the completions capability');

    // --- resources/templates/list -----------------------------------------
    resp := Exec(server,
      '{"jsonrpc":"2.0","id":2,"method":"resources/templates/list"}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'templates/list answers');
    Check(res^.GetAsDocVariant('resourceTemplates', arr),
      'with a resourceTemplates array');
    CheckEqual(arr^.Count, 1, 'holding the one template');
    entry := _Safe(arr^.Values[0]);
    CheckEqual(entry^.U['uriTemplate'], 'file:///{path}',
      'the RFC 6570 template is what identifies it');
    CheckEqual(entry^.U['name'], 'Project Files', 'with its name');
    CheckEqual(entry^.U['title'], 'Browse project files',
      'and its title under the title key, not the name');
    CheckEqual(entry^.U['mimeType'], 'application/octet-stream', 'and mimeType');
    // it is a list, so it is cacheable and paginated like the others
    Check(res^.GetValueIndex('ttlMs') >= 0, 'templates/list carries ttlMs');

    // --- completion for a resource template -------------------------------
    resp := Exec(server, '{"jsonrpc":"2.0","id":3,"method":"completion/complete",' +
      '"params":{"ref":{"type":"ref/resource","uri":"file:///{path}"},' +
      '"argument":{"name":"path","value":""}}}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'completion answers');
    Check(res^.GetAsDocVariant('completion', comp), 'with a completion object');
    Check(comp^.GetAsDocVariant('values', arr), 'holding values');
    CheckEqual(arr^.Count, 2, 'both suggestions come back');
    CheckEqual(comp^.I['total'], 2, 'total is reported');
    Check(not comp^.B['hasMore'], 'and nothing was held back');

    // --- completion for a prompt, truncated at the ceiling ----------------
    resp := Exec(server, '{"jsonrpc":"2.0","id":4,"method":"completion/complete",' +
      '"params":{"ref":{"type":"ref/prompt","name":"flood"},' +
      '"argument":{"name":"many","value":"v"}}}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res), 'a flooding prompt still answers');
    Check(res^.GetAsDocVariant('completion', comp), 'with a completion object');
    Check(comp^.GetAsDocVariant('values', arr), 'holding values');
    // "Maximum 100 items per response" — the SERVER enforces it, so a handler
    // returning 150 cannot put an over-long response on the wire
    CheckEqual(arr^.Count, 100, 'truncated to the 100 the spec allows');
    CheckEqual(comp^.I['total'], 150, 'while total reports what really exists');
    Check(comp^.B['hasMore'], 'and hasMore says more were held back');

    // --- a prompt that offers no completion at all ------------------------
    // Not an error: it simply has nothing to suggest.
    resp := Exec(server, '{"jsonrpc":"2.0","id":5,"method":"completion/complete",' +
      '"params":{"ref":{"type":"ref/prompt","name":"code_review"},' +
      '"argument":{"name":"code","value":"x"}}}');
    doc.Clear;
    doc.InitJson(resp, JSON_FAST);
    Check(doc.GetAsDocVariant('result', res),
      'a prompt without IMcpCompletable answers normally');
    Check(res^.GetAsDocVariant('completion', comp), 'with a completion object');
    Check(comp^.GetAsDocVariant('values', arr), 'and an EMPTY values array');
    CheckEqual(arr^.Count, 0, 'nothing to suggest is not an error');

    // --- errors ------------------------------------------------------------
    CheckErrorResponse(Exec(server,
      '{"jsonrpc":"2.0","id":6,"method":"completion/complete",' +
      '"params":{"ref":{"type":"ref/prompt","name":"nope"},' +
      '"argument":{"name":"a","value":""}}}'), -32602, 'Prompt not found');
    CheckErrorResponse(Exec(server,
      '{"jsonrpc":"2.0","id":7,"method":"completion/complete",' +
      '"params":{"ref":{"type":"ref/nonsense","name":"x"},' +
      '"argument":{"name":"a","value":""}}}'), -32602, 'unknown completion ref');
    CheckErrorResponse(Exec(server,
      '{"jsonrpc":"2.0","id":8,"method":"completion/complete","params":{}}'),
      -32602, 'ref');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.TraceContextReachesTheHandler;
var
  server: TMcpServer;
  tool: TTraceRecordingTool;
  id: integer;

  // a tools/call carrying the given extra _meta entries verbatim
  // - ExecCaps() cannot serve here: it drops the caller's _meta and rebuilds
  //   it, which is exactly the field under test
  procedure CallWith(const aExtraMeta: RawUtf8);
  begin
    inc(id);
    tool.SawTrace := Default(TMcpTraceContext);
    server.ExecuteRequest(FormatUtf8(
      '{"jsonrpc":"2.0","id":%,"method":"tools/call","params":{' +
      '"name":"trace","arguments":{"a":1,"b":2},"_meta":{' +
      '"%":"%","%":{}%}}}',
      [id, MCP_META_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION,
       MCP_META_CLIENT_CAPABILITIES, aExtraMeta]));
  end;

const
  PARENT = '00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01';
begin
  id := 0;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TTraceRecordingTool.Create('trace', 'Records its trace context');
    server.RegisterTool(tool);
    server.Start;

    // the whole trio arrives verbatim — a server that "normalises" a
    // traceparent breaks the very correlation it is meant to preserve
    CallWith(',"' + MCP_META_TRACEPARENT + '":"' + PARENT + '"' +
             ',"' + MCP_META_TRACESTATE + '":"vendor=abc"' +
             ',"' + MCP_META_BAGGAGE + '":"tenant=acme"');
    CheckEqual(tool.SawTrace.TraceParent, PARENT, 'traceparent is passed through');
    CheckEqual(tool.SawTrace.TraceState, 'vendor=abc', 'tracestate too');
    CheckEqual(tool.SawTrace.Baggage, 'tenant=acme', 'and baggage');

    // absent stays absent: no invented trace id
    CallWith('');
    CheckEqual(tool.SawTrace.TraceParent, '', 'no traceparent, no value');

    // A CRLF-bearing value is DROPPED, not forwarded. A handler propagating it
    // into an outgoing header would otherwise splice in headers of the
    // caller's choosing — the value is worthless anyway, since it cannot be a
    // valid W3C traceparent.
    CallWith(',"' + MCP_META_TRACEPARENT + '":"00-abc\r\nX-Injected: 1"');
    CheckEqual(tool.SawTrace.TraceParent, '',
      'a traceparent carrying CRLF is dropped, not forwarded');

    // a non-string is not a trace context: stringifying 42 would hand the
    // handler a traceparent the client never sent
    CallWith(',"' + MCP_META_TRACEPARENT + '":42');
    CheckEqual(tool.SawTrace.TraceParent, '', 'a non-string traceparent is ignored');

    // DEL sits ABOVE the C0 range JSON already bars, and RFC 9110 counts it
    // as neither VCHAR (0x21-0x7E) nor obs-text (0x80-0xFF): forwarding it
    // builds an invalid header field. It is the one character that used to
    // pass the filter.
    CallWith(',"' + MCP_META_TRACEPARENT + '":"00-abc' + #$7F + 'def"');
    CheckEqual(tool.SawTrace.TraceParent, '',
      'a traceparent carrying DEL is dropped too');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ExtensionsAreAdvertisedAndNegotiated;
var
  server: TMcpServer;
  rv: variant;
  caps, ext, settings: PDocVariantData;
  raised: boolean;
begin
  // --- identifier rules: the prefix is MANDATORY for an extension id ---
  Check(McpIsValidExtensionId('io.modelcontextprotocol/tasks'), 'official id');
  Check(McpIsValidExtensionId('com.example/my-ext_v2.1'), 'vendor id');
  Check(not McpIsValidExtensionId('tasks'),
    'a bare name is a valid _meta key but NOT a valid extension id');
  Check(not McpIsValidExtensionId('/tasks'), 'empty prefix');
  Check(not McpIsValidExtensionId('com.example/'), 'empty name');
  Check(not McpIsValidExtensionId('com..example/x'), 'empty label');
  Check(not McpIsValidExtensionId('1com.example/x'), 'label starts with a digit');
  Check(not McpIsValidExtensionId('com.example-/x'), 'label ends with a hyphen');
  Check(not McpIsValidExtensionId('com.example/-x'), 'name starts with a hyphen');
  Check(not McpIsValidExtensionId('com.example/x/y'), 'slash inside the name');

  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // discover without extensions omits the field entirely
    rv := _JsonFast(Exec(server,
      '{"jsonrpc":"2.0","id":1,"method":"server/discover"}'));
    Check(_Safe(_Safe(rv)^.GetValueOrNull('result'))^.GetAsDocVariant(
      'capabilities', caps), 'discover reports capabilities');
    Check(caps^.GetValueIndex('extensions') < 0,
      'no extensions registered, no extensions field');

    // an invalid id must RAISE, not be dropped: a silently ignored registration
    // leaves handlers assuming an extension the server never advertised
    raised := false;
    try
      server.RegisterExtension('tasks');
    except
      on EMcpException do
        raised := true;
    end;
    Check(raised, 'an unprefixed extension id is refused');

    server.RegisterExtension('io.modelcontextprotocol/tasks');
    server.RegisterExtension('com.example/ui',
      _ObjFast(['mimeTypes', _ArrFast(['text/html'])]));

    rv := _JsonFast(Exec(server,
      '{"jsonrpc":"2.0","id":2,"method":"server/discover"}'));
    Check(_Safe(_Safe(rv)^.GetValueOrNull('result'))^.GetAsDocVariant(
      'capabilities', caps), 'capabilities again');
    Check(caps^.GetAsDocVariant('extensions', ext), 'extensions are advertised');
    CheckEqual(ext^.Count, 2, 'both extensions listed');
    Check(ext^.GetAsDocVariant('io.modelcontextprotocol/tasks', settings),
      'settings-free extension present');
    CheckEqual(settings^.Count, 0,
      'support without settings is spelled as an empty object, not null');
    Check(ext^.GetAsDocVariant('com.example/ui', settings), 'ui extension');
    Check(settings^.GetValueIndex('mimeTypes') >= 0, 'its settings survive');

    // --- the client side is per-request, since there is no handshake ---
    Check(McpClientSupportsExtension(
      _JsonFast('{"extensions":{"com.example/ui":{}}}'), 'com.example/ui'),
      'a declared extension is detected');
    Check(not McpClientSupportsExtension(
      _JsonFast('{"extensions":{"com.example/ui":{}}}'), 'com.example/other'),
      'an undeclared one is not');
    Check(not McpClientSupportsExtension(_JsonFast('{}'), 'com.example/ui'),
      'no extensions map at all means no support');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.ProgressIsOptInAndMonotonic;
var
  server: TMcpServer;
  tool: TProgressTool;
  sink: TRecordingSink;
  sinkRef: IMcpNotificationSink;
  auth: TMcpAuthContext;
  challenge: RawUtf8;
  note: PDocVariantData;
  params: PDocVariantData;
  id: integer;

  // one tools/call with the given extra _meta, served with (or without) a sink
  procedure CallWith(const aExtraMeta: RawUtf8; aWithSink: boolean);
  begin
    inc(id);
    sink := TRecordingSink.Create;
    sinkRef := sink; // refcount: keep it alive while the assertions read Sent
    if not aWithSink then
      sinkRef := nil;
    server.ExecuteRequest(FormatUtf8(
      '{"jsonrpc":"2.0","id":%,"method":"tools/call","params":{' +
      '"name":"work","arguments":{"a":1,"b":2},"_meta":{' +
      '"%":"%","%":{}%}}}',
      [id, MCP_META_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION,
       MCP_META_CLIENT_CAPABILITIES, aExtraMeta]), auth, challenge, sinkRef);
  end;

begin
  id := 0;
  FillCharFast(auth, SizeOf(auth), 0);
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TProgressTool.Create('work', 'Reports progress');
    server.RegisterTool(tool);
    server.Start;

    // --- opted in, values increasing: everything goes out ---
    tool.Steps := TDoubleDynArray.Create(10, 50, 100);
    CallWith(',"' + MCP_META_PROGRESS_TOKEN + '":"abc123"', true);
    Check(tool.SawWanted, 'a progressToken plus a stream means progress is wanted');
    CheckEqual(length(sink.Sent), 3, 'all three reports went out');
    note := _Safe(_JsonFast(sink.Sent[0]));
    CheckEqual(note^.U['method'], 'notifications/progress', 'method name');
    Check(note^.GetAsDocVariant('params', params), 'params');
    CheckEqual(params^.U[MCP_META_PROGRESS_TOKEN], 'abc123',
      'the token is echoed so the client can correlate');
    CheckSame(params^.D['progress'], 10, 1E-9, 'progress value');
    CheckSame(params^.D['total'], 100, 1E-9, 'total is carried when known');
    CheckEqual(params^.U['message'], 'step', 'message is carried');

    // --- "The progress value MUST increase with each notification" ---
    // A repeat and a step backwards are DROPPED rather than put on the wire:
    // a client is entitled to rely on the increase.
    tool.Steps := TDoubleDynArray.Create(10, 10, 5, 20);
    CallWith(',"' + MCP_META_PROGRESS_TOKEN + '":7', true);
    CheckEqual(length(sink.Sent), 2, 'only the two increasing values are sent');
    Check(tool.Accepted[0], 'first value accepted');
    Check(not tool.Accepted[1], 'a repeated value is refused');
    Check(not tool.Accepted[2], 'a decreasing value is refused');
    Check(tool.Accepted[3], 'an increase after a refusal is accepted again');
    note := _Safe(_JsonFast(sink.Sent[1]));
    Check(note^.GetAsDocVariant('params', params), 'params of the second');
    CheckSame(params^.D['progress'], 20, 1E-9, 'the second sent value is 20, not 5');

    // --- no token: the client did not opt in, so nothing may be emitted ---
    tool.Steps := TDoubleDynArray.Create(1, 2);
    CallWith('', true);
    Check(not tool.SawWanted, 'without a token progress is not wanted');
    CheckEqual(length(sink.Sent), 0, 'and nothing is sent');

    // --- token but nowhere to send: a no-op, never an error ---
    tool.Steps := TDoubleDynArray.Create(1, 2);
    CallWith(',"' + MCP_META_PROGRESS_TOKEN + '":"abc"', false);
    Check(not tool.SawWanted, 'no stream means progress cannot be wanted');
    Check(not tool.Accepted[0], 'Report says so instead of failing');

    // --- a token that is neither string nor integer is not an opt-in ---
    tool.Steps := TDoubleDynArray.Create(1);
    CallWith(',"' + MCP_META_PROGRESS_TOKEN + '":{"nope":1}', true);
    Check(not tool.SawWanted,
      'an object token cannot be correlated, so it is no opt-in');
    CheckEqual(length(sink.Sent), 0, 'nothing sent for a malformed token');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.UriTemplatesResolveOnRead;
var
  server: TMcpServer;
  vars: variant;
  rv: variant;
  rd, contents, item: PDocVariantData;
  caps: variant;
begin
  // --- the matcher itself ---
  Check(McpMatchUriTemplate('file:///{path}', 'file:///src/main.pas', vars),
    'a level-1 template matches');
  CheckEqual(_Safe(vars)^.U['path'], 'src/main.pas', 'and captures the value');
  Check(McpMatchUriTemplate('db://{table}/rows/{id}', 'db://users/rows/42', vars),
    'two variables with a literal in between');
  CheckEqual(_Safe(vars)^.U['table'], 'users', 'first variable');
  CheckEqual(_Safe(vars)^.U['id'], '42', 'second variable');
  Check(McpMatchUriTemplate('file:///{path}', 'file:///a%20b.txt', vars),
    'a percent-encoded value matches');
  CheckEqual(_Safe(vars)^.U['path'], 'a b.txt', 'and arrives decoded');

  // an empty capture would make the bare prefix answer for the whole family
  Check(not McpMatchUriTemplate('file:///{path}', 'file:///', vars),
    'a variable must capture something');
  Check(not McpMatchUriTemplate('file:///{path}', 'other:///x', vars),
    'the literal prefix must match');
  // the LAST variable takes the rest, reserved characters included — otherwise
  // 'file:///{path}' could never match a real file URI (see the function's own
  // comment for why strict RFC 6570 encoding is not enforced here)
  Check(McpMatchUriTemplate('db://{table}/rows/{id}', 'db://users/rows/42/x',
    vars), 'the trailing variable is greedy');
  CheckEqual(_Safe(vars)^.U['id'], '42/x', 'and takes the slash with it');
  // a template ending in a LITERAL still has to consume the whole URI
  Check(not McpMatchUriTemplate('db://{table}/rows', 'db://users/rows/extra',
    vars), 'a trailing remainder after the last literal is not a match');
  // operator forms are refused rather than half-understood: guessing would
  // resolve a URI to the WRONG resource
  Check(not McpMatchUriTemplate('file:///{+path}', 'file:///a/b', vars),
    'the reserved-expansion operator is not supported');
  Check(not McpMatchUriTemplate('file:///{path*}', 'file:///a/b', vars),
    'the explode modifier is not supported');
  Check(not McpMatchUriTemplate('file:///{path', 'file:///a', vars),
    'an unclosed expression matches nothing');

  // --- end to end through resources/read ---
  caps := _ObjFast([]);
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterResourceTemplate(TDbTemplate.Create);
    // a template WITHOUT the add-on: still a pure advertisement
    server.RegisterResourceTemplate(TFilesTemplate.Create);
    server.Start;

    rv := _JsonFast(ExecCaps(server, '{"jsonrpc":"2.0","id":1,' +
      '"method":"resources/read","params":{"uri":"db://users/rows/42"}}', caps));
    rd := _Safe(_Safe(rv)^.GetValueOrNull('result'));
    Check(rd^.GetAsDocVariant('contents', contents) and (contents^.Count = 1),
      'a URI built from an expandable template resolves');
    item := _Safe(contents^.Values[0]);
    CheckEqual(item^.U['uri'], 'db://users/rows/42', 'the concrete URI is echoed');
    CheckEqual(item^.U['mimeType'], 'application/json',
      'the mime type comes from the template');
    CheckEqual(item^.U['text'], '{"table":"users","id":"42"}',
      'the template produced the content');

    // shape matched, thing does not exist -> the handler's own -32602 stands
    CheckErrorResponse(ExecCaps(server, '{"jsonrpc":"2.0","id":2,' +
      '"method":"resources/read","params":{"uri":"db://missing/rows/1"}}', caps),
      JSONRPC_INVALID_PARAMS, 'No such table');

    // a template that cannot serve its URIs must NOT swallow the read
    CheckErrorResponse(ExecCaps(server, '{"jsonrpc":"2.0","id":3,' +
      '"method":"resources/read","params":{"uri":"file:///src/main.pas"}}', caps),
      JSONRPC_INVALID_PARAMS, 'Resource not found');

    // and a URI matching nothing at all is unchanged: -32602
    CheckErrorResponse(ExecCaps(server, '{"jsonrpc":"2.0","id":4,' +
      '"method":"resources/read","params":{"uri":"nope://x"}}', caps),
      JSONRPC_INVALID_PARAMS, 'Resource not found');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.HeaderMirroringIsConstrained;
var
  tool: TCalcTool;
  mirrored: TMcpHeaderParamDynArray;

  // does collecting the annotations of this schema raise?
  function Rejects(const aSchemaJson: RawUtf8): boolean;
  begin
    result := false;
    try
      McpCollectHeaderParams(_JsonFast(aSchemaJson));
    except
      on EMcpException do
        result := true;
    end;
  end;

begin
  EnsureCalcParamsRtti;

  // --- the header token rules (RFC 9110 1*tchar) ---
  Check(McpIsHeaderToken('Region'), 'a plain token');
  Check(McpIsHeaderToken('X-Tenant_Id.2'), 'tchar punctuation is allowed');
  Check(not McpIsHeaderToken(''), 'empty is not a token');
  Check(not McpIsHeaderToken('Re gion'), 'a space would split the header line');
  Check(not McpIsHeaderToken('Region:'), 'a colon would split the header line');
  Check(not McpIsHeaderToken('Reg'#13#10'ion'), 'CRLF would inject a header');

  // --- the base64 sentinel ---
  CheckEqual(McpDecodeHeaderValue('us-west1'), 'us-west1', 'plain value untouched');
  CheckEqual(McpDecodeHeaderValue('=?base64?SGVsbG8=?='), 'Hello', 'sentinel decoded');
  CheckEqual(McpDecodeHeaderValue('=?BASE64?SGVsbG8=?='), '=?BASE64?SGVsbG8=?=',
    'the markers are case-sensitive: an uppercase one is a literal value');

  // --- collecting from a schema ---
  mirrored := McpCollectHeaderParams(_JsonFast(
    '{"type":"object","properties":{' +
    '"region":{"type":"string","x-mcp-header":"Region"},' +
    '"query":{"type":"string"}}}'));
  CheckEqual(length(mirrored), 1, 'one annotated property');
  CheckEqual(mirrored[0].Name, 'Region', 'the header name');
  CheckEqual(RawUtf8ArrayToCsv(mirrored[0].Path, '.'), 'region', 'the path');

  // nested objects stay reachable as long as every step is a `properties` key
  mirrored := McpCollectHeaderParams(_JsonFast(
    '{"type":"object","properties":{"target":{"type":"object","properties":{' +
    '"region":{"type":"string","x-mcp-header":"Region"}}}}}'));
  CheckEqual(length(mirrored), 1, 'a nested annotation is reachable');
  CheckEqual(RawUtf8ArrayToCsv(mirrored[0].Path, '.'), 'target.region',
    'and carries its full path');

  Check(Rejects('{"type":"object","properties":{' +
    '"r":{"type":"string","x-mcp-header":"Bad Name"}}}'),
    'an invalid field-name token is refused');
  // a float has no single decimal form, so header and body could not be
  // compared reliably — the spec excludes `number` for exactly that reason
  Check(Rejects('{"type":"object","properties":{' +
    '"r":{"type":"number","x-mcp-header":"R"}}}'),
    'type number may not be mirrored');
  Check(Rejects('{"type":"object","properties":{' +
    '"a":{"type":"string","x-mcp-header":"R"},' +
    '"b":{"type":"string","x-mcp-header":"r"}}}'),
    'header names collide case-insensitively');
  // an annotation the client cannot statically reach makes the WHOLE tool
  // definition invalid — ignoring it would ship a tool whose author believes
  // a header is being mirrored
  Check(Rejects('{"type":"object","properties":{"list":{"type":"array",' +
    '"items":{"type":"object","properties":{' +
    '"r":{"type":"string","x-mcp-header":"R"}}}}}}'),
    'an annotation under `items` is not statically reachable');
  Check(Rejects('{"type":"object","properties":{"x":{"oneOf":[' +
    '{"type":"object","properties":{' +
    '"r":{"type":"string","x-mcp-header":"R"}}}]}}}'),
    'an annotation under a composition keyword is not reachable either');

  // --- the authoring API puts the annotation into the generated schema ---
  tool := TCalcTool.Create('calc', 'Add two numbers');
  try
    tool.MirrorToHeader('A', 'A-Value');
    mirrored := McpCollectHeaderParams(tool.GetInputSchema);
    CheckEqual(length(mirrored), 1, 'the generated schema carries it');
    CheckEqual(mirrored[0].Name, 'A-Value', 'with the given header name');

    // a path that is not a property of the record must fail at wiring time,
    // not silently annotate nothing
    try
      tool.MirrorToHeader('nosuchfield', 'X');
      Check(false, 'an unknown property path must raise');
    except
      on EMcpException do
        Check(true, 'an unknown property path is refused');
    end;
  finally
    tool.Free;
  end;
end;

procedure TTestMcpCore.PublishedSchemasAreBounded;
var
  reason: RawUtf8;
  deep: RawUtf8;
  i: PtrInt;
begin
  // an ordinary schema passes untouched
  Check(McpCheckSchema(_JsonFast('{"type":"object","properties":{' +
    '"a":{"type":"string"}}}'), 32, 4096, reason), 'a plain schema is fine');
  CheckEqual(reason, '', 'and reports no reason');

  // 2026-07-28 LOOSENED inputSchema to any 2020-12 keyword, so an unfamiliar
  // one must NOT be rejected — that is the opposite of the old behaviour
  Check(McpCheckSchema(_JsonFast('{"type":"object","unevaluatedProperties":false,' +
    '"dependentSchemas":{"a":{"required":["b"]}},' +
    '"patternProperties":{"^x-":{"type":"string"}}}'), 32, 4096, reason),
    'any 2020-12 keyword is allowed');

  // local $refs are fine: they resolve inside the document
  Check(McpCheckSchema(_JsonFast('{"$defs":{"x":{"type":"string"}},' +
    '"properties":{"a":{"$ref":"#/$defs/x"}}}'), 32, 4096, reason),
    'a local $ref stays inside the document');
  Check(McpCheckSchema(_JsonFast('{"properties":{"a":{"$ref":"defs.json"}}}'),
    32, 4096, reason), 'a relative pointer is resolved locally too');

  // a network $ref must never be published: the consumer MUST NOT dereference
  // it, and a schema it cannot resolve should be rejected rather than treated
  // as permissive — so we do not hand it out in the first place
  Check(not McpCheckSchema(_JsonFast(
    '{"properties":{"a":{"$ref":"https://evil.example/s.json"}}}'),
    32, 4096, reason), 'an https $ref is refused');
  Check(PosEx('outside the document', reason) > 0, 'and says why: ' + reason);
  Check(not McpCheckSchema(_JsonFast(
    '{"properties":{"a":{"$ref":"//evil.example/s.json"}}}'),
    32, 4096, reason), 'a protocol-relative $ref is the network as well');
  Check(not McpCheckSchema(_JsonFast(
    '{"properties":{"a":{"$ref":"file:///etc/passwd"}}}'),
    32, 4096, reason), 'a file:// $ref is refused too');

  // depth and node bounds: a schema is a DoS vector against every client that
  // validates against it
  deep := '{"type":"string"}';
  for i := 1 to 40 do
    deep := '{"properties":{"a":' + deep + '}}';
  Check(not McpCheckSchema(_JsonFast(deep), 32, 4096, reason),
    'a schema nested past the depth limit is refused');
  Check(PosEx('deeper', reason) > 0, 'and says why: ' + reason);
  Check(McpCheckSchema(_JsonFast(deep), 200, 4096, reason),
    'the same schema passes when the limit allows it');
  Check(not McpCheckSchema(_JsonFast(deep), 200, 5, reason),
    'the node cap bites independently of depth');
  Check(PosEx('subschemas', reason) > 0, 'and says why: ' + reason);
end;


procedure TTestMcpCore.NotificationWithAbsentParamsIsSafe;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TCalcTool.Create('calc', 'Add two numbers'));
    server.Start;
    // A notification (no id) skips BOTH guards that keep a non-object `params`
    // away from the dispatcher: ParseRequest's scalar check and
    // ValidateRequestMeta. It is therefore the only shape that reaches
    // ExecuteToolCall with nothing to read, and the two-argument _Safe leaves
    // its out-pointer unset in exactly that case. Reading `arguments` off it
    // interpreted whatever the stack held as a TDocVariantData.
    response := server.ExecuteRequest('{"jsonrpc":"2.0","method":"tools/call"}');
    CheckEqual(TrimU(response), '', 'a notification is answered with nothing');
    // the resources/read sibling reached the same overload
    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","method":"resources/read"}');
    CheckEqual(TrimU(response), '', 'resources/read notification: nothing');
    // a scalar params is NOT the gap: ParseRequest rejects that shape on the
    // notification path too, and answers -32600 with a null id. Asserted so a
    // future change to that guard cannot quietly widen the hole above.
    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","method":"tools/call","params":42}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
    // and the server still answers correctly afterwards - a corrupted heap
    // would not necessarily raise at the point of the read
    response := Exec(server, '{"jsonrpc":"2.0","id":1,"method":"tools/list"}');
    Check(PosEx('"calc"', response) > 0, 'the server is healthy afterwards');
    // the request path (with an id) keeps rejecting the same body at the
    // envelope level, which is where that check belongs
    response := server.ExecuteRequest(
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":42}');
    CheckErrorResponse(response, JSONRPC_INVALID_REQUEST, '');
  finally
    server.Free;
  end;
end;

procedure TTestMcpCore.TypedToolRefusesArgumentsThatDoNotParse;
var
  server: TMcpServer;
  response: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TCalcTool.Create('calc', 'Add two numbers'));
    server.Start;
    // `A` is declared integer and gets an array: the record parser refuses the
    // payload. Its result used to be discarded, and since the default options
    // do not include jpoClearValues the record was not even zeroed - the tool
    // ran on whatever the stack held.
    // The refusal is a TOOL error (isError), NOT a JSON-RPC error: the spec
    // files 'Input validation errors' under the half a model can self-correct
    // from (docs/specs/mcp-2026-07-28/server/tools.mdx:760-783), and reserves
    // -32602 for an unknown tool or a malformed CallToolRequest envelope.
    response := Exec(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"A":[1,2],"B":2}}}');
    Check(PosEx('"isError":true', response) > 0,
      'arguments that do not parse are refused as a tool error');
    Check(PosEx('input schema', response) > 0, 'and say why');
    Check(PosEx('"error"', response) = 0, 'never as a JSON-RPC protocol error');
    Check(PosEx('0 + 0 = 0', response) = 0, 'and the tool did not run');
    // a payload that DOES parse still works, arguments the record does not
    // declare are still tolerated (the parser runs tolerant on purpose)
    response := Exec(server,
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"A":2,"B":3,"Unknown":true}}}');
    Check(PosEx('2 + 3 = 5', response) > 0, 'a valid payload still executes');
  finally
    server.Free;
  end;
end;


procedure TTestMcpCore.SubscriptionCapIsPerPrincipalToo;
var
  server: TMcpServer;
  i: integer;
  mine, theirs, extra: TMcpSubscription;
begin
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.Start;
    // The global cap alone lets ONE caller take every slot and refuse the
    // method to everyone else - a denial of service that costs the attacker
    // one authenticated session. Each stream holds an HTTP worker for its
    // whole life, so the slots are the scarce thing.
    server.MaxSubscriptionsPerPrincipal := 2;
    mine := server.OpenSubscription(1, _ObjFast([]), 'user-1');
    Check(mine <> nil, 'first stream for this caller');
    mine := server.OpenSubscription(2, _ObjFast([]), 'user-1');
    Check(mine <> nil, 'second, still within the per-caller cap');
    extra := server.OpenSubscription(3, _ObjFast([]), 'user-1');
    Check(extra = nil, 'the third is refused, well below the global cap');

    // and the refusal is per caller, not a global freeze: everyone else still
    // gets in, which is the whole point
    theirs := server.OpenSubscription(4, _ObjFast([]), 'user-2');
    Check(theirs <> nil, 'a different caller is unaffected');

    // unauthenticated callers share one bucket: without a verifier they cannot
    // be told apart, so treating them as one is the honest reading
    Check(server.OpenSubscription(5, _ObjFast([])) <> nil, 'anonymous #1');
    Check(server.OpenSubscription(6, _ObjFast([])) <> nil, 'anonymous #2');
    Check(server.OpenSubscription(7, _ObjFast([])) = nil,
      'anonymous callers do not get a slot each');

    // 0 turns the per-caller cap off, back to the global one alone
    server.MaxSubscriptionsPerPrincipal := 0;
    for i := 8 to 9 do
      Check(server.OpenSubscription(i, _ObjFast([]), 'user-1') <> nil,
        'per-caller cap disabled');
    Check(server.SubscriptionCount <= server.MaxSubscriptions,
      'the global cap still holds');
  finally
    server.Free;
  end;
end;


procedure TTestMcpCore.ScopeRefusalUsesAnApplicationCode;
var
  server: TMcpServer;
  response: RawUtf8;
  doc, errDoc, dataDoc: PDocVariantData;
  docVar, errVar, dataVar: variant;
  code: Int64;
  scopes: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    server.RegisterTool(TScopeGatedTool.Create('gated', 'Needs a scope'));
    server.Start;
    response := Exec(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"gated",' +
      '"arguments":{"A":1,"B":2}}}');
    docVar := _JsonFast(response);
    doc := _Safe(docVar);
    errVar := doc^.GetValueOrNull('error');
    errDoc := _Safe(errVar);
    if CheckFailed(errDoc^.IsObject, 'a scope refusal is an error response') then
      exit;
    Check(VariantToInt64Loose(errDoc^.GetValueOrDefault('code', 0), code));

    // NOT -32600: that code says the request object itself was malformed, and
    // it was not - the caller simply may not do this yet. And not a code from
    // -32020..-32099 either: that sub-range belongs to the specification alone
    // (basic/index.mdx:118-128) and it defines nothing for scope, because it
    // settles insufficient scope on the HTTP layer. Application-defined codes
    // go outside the reserved range (:153-155), which is where this one sits.
    CheckEqual(code, MCP_ERROR_INSUFFICIENT_SCOPE, 'an application code');
    Check((code > -32000) or (code < -32768), 'outside the JSON-RPC reserved range');

    // and the scopes are machine-readable, not only in the prose: on stdio
    // there is no WWW-Authenticate header to carry them
    dataVar := errDoc^.GetValueOrNull('data');
    dataDoc := _Safe(dataVar);
    if CheckFailed(dataDoc^.IsObject, 'error.data carries the challenge') then
      exit;
    Check(dataDoc^.GetAsRawUtf8('requiredScopes', scopes));
    CheckEqual(scopes, 'files:write files:admin',
      'space-separated, the shape RFC 6750 uses in the header');
  finally
    server.Free;
  end;
end;


procedure TTestMcpCore.RequiredSchemaIsHonestAndEnforced;
var
  server: TMcpServer;
  tool: TCalcTool;
  response: RawUtf8;
  schema: variant;
  doc, req: PDocVariantData;
  i: PtrInt;
  names: RawUtf8;
begin
  EnsureCalcParamsRtti;
  server := TMcpServer.Create('TestServer', '1.0');
  try
    tool := TCalcTool.Create('calc', 'Add two numbers');
    server.RegisterTool(tool);
    server.Start;

    // 1. The published schema no longer over-declares. It used to list every
    // RTTI property as required - and clients SHOULD validate against it, so
    // the over-declaration made them reject calls this server would have taken.
    schema := tool.GetInputSchema;
    doc := _Safe(schema);
    if CheckFailed(doc^.GetAsDocVariant('required', req), 'required is present') then
      exit;
    names := '';
    for i := 0 to req^.Count - 1 do
      names := names + VariantToUtf8(req^.Values[i]) + ' ';
    CheckEqual(TrimU(names), 'a b',
      'only what the tool did not declare optional');

    // 2. and what it does declare is enforced. A missing argument used to parse
    // into its zero value, so the tool ran on a number nobody sent.
    response := Exec(server,
      '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"A":2}}}');
    Check(PosEx('"isError":true', response) > 0, 'a missing argument is refused');
    Check(PosEx('b', response) > 0, 'and the answer names it');
    Check(PosEx('2 + 0 = 2', response) = 0, 'the tool did not run on a zero');

    // 3. an optional one may still be left out, which is the whole point
    response := Exec(server,
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"A":2,"B":3}}}');
    Check(PosEx('2 + 3 = 5', response) > 0, 'optional arguments stay optional');

    // 4. present-but-empty is a value the caller chose, not an omission
    response := Exec(server,
      '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"calc",' +
      '"arguments":{"A":0,"B":0}}}');
    Check(PosEx('0 + 0 = 0', response) > 0, 'a zero the caller SENT is fine');

    // 5. a typo in MarkOptional is caught at wiring time, not by silently
    // leaving the field required and refusing valid calls forever
    try
      tool.MarkOptional(['NoSuchField']);
      Check(false, 'MarkOptional must reject an unknown property');
    except
      on E: EMcpException do
        Check(PosEx('NoSuchField', StringToUtf8(E.Message)) > 0, 'and names it');
    end;
  finally
    server.Free;
  end;
end;

end.
