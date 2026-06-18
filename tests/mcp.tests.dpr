// - console test runner for mormot.ai.mcp
program mcp.tests;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.os,
  mormot.core.log,
  mormot.core.test,
  test.mcp.core,
  test.mcp.transports;

type
  TMcpTests = class(TSynTestsLogged)
  published
    procedure MCP;
  end;

procedure TMcpTests.MCP;
begin
  AddCase([
    TTestMcpCore,
    TTestMcpTransports,
    TTestMcpStreamableTransport
  ]);
end;

begin
  SetExecutableVersion('1.0.0');
  RunFromSynTests := true; // ensure tests don't spawn blocking stdio threads

  if ParamCount = 0 then
  begin
    with TMcpTests.Create('mORMot MCP Tests') do
    try
      Run;
    finally
      Free;
    end;
  end
  else
  begin
    TMcpTests.RunAsConsole('mORMot MCP Tests', LOG_VERBOSE, [],
      Executable.ProgramFilePath + 'data');
  end;
end.
