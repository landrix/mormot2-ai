program codenav.mcp;

// LandrixAI code-nav MCP server (stdio) — Pascal/TS/Kotlin code-navigation tools
// for coding agents, so they navigate the codebase without reading whole files.
//   get_outline(path)            interface outline of a .pas unit (own scanner)
//   find_definition(name)        symbol -> file:line [kind] signature (ctags)
//   search_text(pattern, glob)   compact grep over the curated source dirs
//
// Transport is line-delimited JSON-RPC over stdin/stdout: stdout carries ONLY
// protocol traffic (no status prints), per the MCP stdio transport.

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.rtti,
  mormot.core.variants,
  mormot.ai.mcp,
  mormot.ai.mcp.stdio,
  codenav.tools;

type
  TOutlineParams = packed record
    path: RawUtf8;
  end;

  TFindDefParams = packed record
    name: RawUtf8;
  end;

  TSearchParams = packed record
    pattern: RawUtf8;
    glob: RawUtf8;
  end;

  TGetOutlineTool = class(TMcpToolBase<TOutlineParams>)
  protected
    function ExecuteTyped(const aParams: TOutlineParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  TFindDefinitionTool = class(TMcpToolBase<TFindDefParams>)
  protected
    function ExecuteTyped(const aParams: TFindDefParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

  TSearchTextTool = class(TMcpToolBase<TSearchParams>)
  protected
    function ExecuteTyped(const aParams: TSearchParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

function TGetOutlineTool.ExecuteTyped(const aParams: TOutlineParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
  fn: string;
begin
  builder := TMcpResponseBuilder.Create;
  try
    fn := Utf8ToString(aParams.path);
    if (aParams.path = '') or not FileExists(fn) then
      builder.AddText(FormatUtf8('{"error":"file not found: %"}', [aParams.path]))
    else
      builder.AddText(GetOutline(fn));
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

function TFindDefinitionTool.ExecuteTyped(const aParams: TFindDefParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
begin
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText(FindDefinition(aParams.name));
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

function TSearchTextTool.ExecuteTyped(const aParams: TSearchParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
begin
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText(SearchText(aParams.pattern, aParams.glob));
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

procedure EnsureRtti;
begin
  if not RecordHasFields(TypeInfo(TOutlineParams)) then
    Rtti.RegisterFromText(TypeInfo(TOutlineParams), 'path:RawUtf8');
  if not RecordHasFields(TypeInfo(TFindDefParams)) then
    Rtti.RegisterFromText(TypeInfo(TFindDefParams), 'name:RawUtf8');
  if not RecordHasFields(TypeInfo(TSearchParams)) then
    Rtti.RegisterFromText(TypeInfo(TSearchParams), 'pattern,glob:RawUtf8');
end;

var
  server: TMcpServer;
  transport: TMcpStdioTransport;
begin
  EnsureRtti;
  server := TMcpServer.Create('landrix-codenav', '0.2.0');
  try
    server.RegisterTool(TGetOutlineTool.Create('get_outline',
      'Return a compact code outline of a source file: types/classes/interfaces ' +
      'with their members. Pascal (.pas) uses a dedicated scanner (interface part, ' +
      'de-duplicated); TypeScript/Kotlin/other files use ctags. Use this INSTEAD ' +
      'of reading large files to save tokens. Parameter: path = file path.'));
    server.RegisterTool(TFindDefinitionTool.Create('find_definition',
      'Find where a symbol (class, interface, method, function, ...) is DEFINED ' +
      'across the codebase (Pascal, TypeScript, Kotlin) via a ctags index. ' +
      'Returns "file:line [kind] signature". Use this instead of grepping for a ' +
      'definition. Parameter: name = exact symbol name.'));
    server.RegisterTool(TSearchTextTool.Create('search_text',
      'Search the source tree for a regular expression (grep), multi-language. ' +
      'Returns compact "file:line:text" matches (capped). Parameters: ' +
      'pattern = regex; glob = optional file filter like *.ts ("" = all files).'));
    server.Start;
    transport := TMcpStdioTransport.Create(server);
    try
      transport.Start;
      while transport.IsActive do
        Sleep(100);
    finally
      transport.Free;
    end;
  finally
    server.Free;
  end;
end.
