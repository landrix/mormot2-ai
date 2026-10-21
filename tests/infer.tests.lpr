// - console test runner for the native inference engine (src/infer)
// - needs no model weights: everything it reads is committed under
//   tests/data/infer-ref/ (tools/infer-ref/make_reference.py)
program infer.tests;

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
  test.infer,
  test.infer.engine,
  test.infer.xlmr,
  test.infer.tokenizer,
  test.infer.gguf,
  test.infer.fast,
  test.infer.bpe,
  test.infer.harmony,
  test.infer.gemma4,
  test.infer.qwen3,
  test.infer.llm,
  test.infer.vulkan,
  test.infer.cuda;

type
  TInferTests = class(TSynTestsLogged)
  published
    procedure Infer;
  end;

procedure TInferTests.Infer;
begin
  AddCase([
    TTestInferReference,
    TTestInferEngine,
    TTestInferXlmr,
    TTestInferTokenizer,
    TTestInferGguf,
    TTestInferFast,
    TTestInferBpe,
    TTestInferHarmony,
    TTestInferGemma4,
    TTestInferQwen3,
    TTestInferLlmNative,
    TTestInferVulkan,
    TTestInferCuda
  ]);
end;

var
  tests: TInferTests;
begin
  SetExecutableVersion('1.0.0');
  if ParamCount = 0 then
  begin
    tests := TInferTests.Create('mORMot AI Inference Tests');
    try
      tests.Run;
    finally
      tests.Free;
    end;
  end
  else
    TInferTests.RunAsConsole('mORMot AI Inference Tests', LOG_VERBOSE, [],
      Executable.ProgramFilePath + 'data');
end.
