// Shared demo helpers for MCP examples
unit demo.mcp.shared;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.rtti,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.json,
  mormot.net.client,
  mormot.ai.mcp;

type
  TAddParams = record
    a: integer;
    b: integer;
  end;

  TIpInfo = packed record
    Ip: RawUtf8;
    Date: TDateTime;
  end;

  TAddTool = class(TMcpToolBase<TAddParams>)
  protected
    function ExecuteTyped(const aParams: TAddParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  TGetIpTool = class(TInterfacedObject, IMcpTool)
  protected
    function FetchIp(out aIp: RawUtf8): boolean;
  public
    function GetName: RawUtf8;
    function GetDescription: RawUtf8;
    function GetInputSchema: variant;
    function Execute(const aArgs: variant; const aAuthCtx: TMcpAuthContext): variant;
  end;

  TVersionResource = class(TMcpResourceBase)
  protected
    function GetContent: RawUtf8; override;
  end;

procedure EnsureDemoRtti;
procedure RegisterDemoServices(const aServer: TMcpServer);
function CreateDemoServer(const aName, aVersion: RawUtf8): TMcpServer;

implementation

procedure EnsureDemoRtti;
begin
  if not RecordHasFields(TypeInfo(TAddParams)) then
    Rtti.RegisterFromText(TypeInfo(TAddParams), 'a,b:integer');
  if not RecordHasFields(TypeInfo(TIpInfo)) then
    Rtti.RegisterFromText(TypeInfo(TIpInfo), 'Ip:RawUtf8 Date:TDateTime');
end;

{ TAddTool }

function TAddTool.ExecuteTyped(const aParams: TAddParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
  resultText: RawUtf8;
begin
  resultText := FormatUtf8('% + % = %', [aParams.a, aParams.b, aParams.a + aParams.b]);
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText(resultText);
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

{ TGetIpTool }

function TGetIpTool.FetchIp(out aIp: RawUtf8): boolean;
var
  body: RawUtf8;
  doc: PDocVariantData;
begin
  aIp := '';
  body := HttpGet('https://api.ipify.org?format=json');
  if body = '' then
    exit(false);
  doc := _Safe(_JsonFast(body));
  result := doc^.GetAsRawUtf8('ip', aIp) and (aIp <> '');
end;

function TGetIpTool.GetName: RawUtf8;
begin
  result := 'public_ip';
end;

function TGetIpTool.GetDescription: RawUtf8;
begin
  result := 'Fetch public IP address and current UTC timestamp';
end;

function TGetIpTool.GetInputSchema: variant;
var
  doc, props, req: TDocVariantData;
begin
  doc.InitObject(['type', 'object'], JSON_FAST);
  props.InitObject([], JSON_FAST);
  req.InitArray([], JSON_FAST);
  doc.AddValue('properties', variant(props));
  doc.AddValue('required', variant(req));
  result := variant(doc);
end;

function TGetIpTool.Execute(const aArgs: variant; const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
  info: TIpInfo;
  json: RawUtf8;
begin
  EnsureDemoRtti;
  builder := TMcpResponseBuilder.Create;
  try
    if FetchIp(info.Ip) then
    begin
      info.Date := NowUtc;
      json := RecordSaveJson(info, TypeInfo(TIpInfo));
      builder.AddText(json);
    end
    else
      builder.AddText('{"error":"Unable to fetch public IP"}');
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

{ TVersionResource }

function TVersionResource.GetContent: RawUtf8;
begin
  result := '{"version":"1.0.0","protocol":"MCP"}';
end;

procedure RegisterDemoServices(const aServer: TMcpServer);
var
  addTool: IMcpTool;
  ipTool: IMcpTool;
  versionRes: IMcpResource;
begin
  EnsureDemoRtti;
  addTool := TAddTool.Create('add', 'Add two numbers');
  ipTool := TGetIpTool.Create;
  versionRes := TVersionResource.Create('version://info', 'Version',
    'Server version information', 'application/json');
  aServer.RegisterTool(addTool);
  aServer.RegisterTool(ipTool);
  aServer.RegisterResource(versionRes);
end;

function CreateDemoServer(const aName, aVersion: RawUtf8): TMcpServer;
begin
  result := TMcpServer.Create(aName, aVersion);
  RegisterDemoServices(result);
  result.Start;
end;

end.
