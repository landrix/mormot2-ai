/// MCP Server Example and Basic Tests
// - demonstrates usage of mormot.ai.mcp units
// - simple console application for testing
program mcp.examples;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}


uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.text,
  main;


begin
  try
    StartExample;
    {$ifdef FPC_X64MM}
    if (ExitCode = 0)  then
      WriteHeapStatus(' ', 16, 8, {compileflags=}true);
    {$endif FPC_X64MM}
  except
    on E: Exception do
      ConsoleShowFatalException(E);
  end;

end.
