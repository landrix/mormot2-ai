program LandrixAiTestRunner;

{$mode delphi}
{$ifdef UNIX}{$H+}{$endif}

// Aggregierter FPCUnit-Runner fuer die mormot.ai.* Extension (LandrixAI).
// Neue Test-Units einfach in der uses-Liste ergaenzen; Registrierung laeuft
// ueber RegisterTest in der jeweiligen Unit. ExitCode <> 0 bei Fehlern.
//
//   LandrixAiTestRunner -a --format=plain   # alle Tests, kompakt
//   LandrixAiTestRunner -l                  # registrierte Tests auflisten

uses
  {$IFDEF UNIX}
  cthreads,  // Thread-Treiber vor Units mit Threads (spaetere Transport-Tests)
  {$ENDIF}
  consoletestrunner,
  mormot.ai.mcp.types.FpcTest;

var
  App: TTestRunner;
begin
  App := TTestRunner.Create(nil);
  try
    App.Initialize;
    App.Title := 'mormot.ai (LandrixAI) Tests';
    App.Run;
  finally
    App.Free;
  end;
end.
