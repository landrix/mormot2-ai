// - console test runner for mormot.ai.llm
program llm.tests;

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
  test.llm.sse;

type
  TLlmTests = class(TSynTestsLogged)
  published
    procedure LLM;
  end;

procedure TLlmTests.LLM;
begin
  AddCase([
    TTestLlmSse
  ]);
end;

begin
  SetExecutableVersion('1.0.0');

  if ParamCount = 0 then
  begin
    with TLlmTests.Create('mORMot AI LLM Tests') do
    try
      Run;
    finally
      Free;
    end;
  end
  else
    TLlmTests.RunAsConsole('mORMot AI LLM Tests', LOG_VERBOSE, [],
      Executable.ProgramFilePath + 'data');
end.
