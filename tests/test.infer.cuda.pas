/// tests of the native inference engine: the CUDA backend (backlog B3)
// - skips without an NVIDIA driver and device; LAI_REQUIRE_CUDA=1 turns
//   the skip into a failure (a run that must cover the GPU)
unit test.infer.cuda;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  classes,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.perf,
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.cuda,
  mormot.ai.infer.backend.cuda,
  mormot.ai.infer.decoder,
  mormot.ai.infer.model.qwen3,
  mormot.ai.infer.model.qwen3.cuda,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.native,
  mormot.ai.infer.sampler,
  test.infer.ref,
  test.infer.fast,
  test.infer.bpe,
  test.infer.qwen3;

type
  /// slice C1: driver, device, modules from PTX, memory and a kernel
  TTestInferCuda = class(TSynTestCase)
  protected
    fCu: TInferCuda;
    procedure Setup; override;
    procedure CleanUp; override;
    /// false (and SKIP or a failure) without a device
    function HasDevice: boolean;
  published
    /// y = a + b on the device: the bits of the CPU sum (an IEEE add is
    // correctly rounded on both), over a count no multiple of the block,
    // twice with the same kernel and other buffers
    procedure AddMatchesCpu;
    /// misuse is refused: by the checks before the driver, or by the driver
    // as an EInferCuda with its error name
    procedure RefusesMisuse;
    /// device memory: what goes up comes down, at an offset, also in pieces
    procedure UploadDownload;
    /// slice C5c: kernels and a device copy recorded as a graph run as
    // enqueued; the next capture with other parameters updates the graph, one
    // of another shape makes it again; copies and waits are refused during a
    // capture, an aborted capture leaves the stream usable; another instance
    // on another thread copies and waits meanwhile
    procedure GraphRunsCapture;
    /// slice C2: TInferCudaBackend.Linear for F32, Q8_0, Q4_K and Q6_K weights
    // against the reference's F32 product of the dequantized weights (a
    // tolerance: another summation order), with and without bias, more than
    // 65535 outputs, the weight cache, and the CPU for other types
    procedure LinearMatchesReference;
    /// slice C3 (attention: C5): rmsnorm, addnorm, swiglu, qkrope and attention of
    // tools/cuda/ops.cu against the CPU reference's RmsNorm, Add, SwiGlu, Rope
    // and CausalAttention (a tolerance: float arithmetic, another order),
    // enqueued without a wait between them
    procedure OpsMatchCpu;
    /// slice C4: the tiny Qwen3 forward on the device against the CPU
    // reference - every row of a prompt longer than one chunk, decoding, a
    // reused prefix, the weights uploaded again; F32 from safetensors
    // (relative error), Q8_0 from GGUF (cosine: the CPU quantizes the
    // activations, the device does not); each with a KV cache of floats and
    // of half floats (KvF16, a looser tolerance)
    procedure TinyQwen3MatchesCpu;
    /// the tiny Qwen3 with a head dimension of 256 (the widest qkrope takes),
    // the cache of floats, then of half floats in the same context
    procedure TinyHeadDim256MatchesCpu;
    /// Qwen3-0.6B (Q4_K_M and Q8_0 GGUF) on the device against the fast CPU
    // backend, teacher-forced - skipped without the model files
    procedure Qwen3GgufMatchesCpu;
    /// TNativeLlmClient on the device (Device := TQwen3Cuda): the greedy
    // answer of Qwen3-0.6B Q8_0 as without it, the next turn from the prompt
    // cache on the device, back to the backend with Device := nil
    procedure NativeClientOnDevice;
    /// tiny Qwen3-MoE on the device against the CPU: F32, Q8_0 from GGUF (the
    // experts of a layer in one buffer) and quantized at load, half floats
    procedure TinyMoeMatchesCpu;
    /// Qwen3-30B-A3B (Q4_K_M GGUF) on the device against the fast CPU backend,
    // teacher-forced - where the file was put
    procedure Qwen3MoeGgufMatchesCpu;
    /// D1b: the tiny Qwen2 (bias on Q, K, V, no head norm) on the device
    // against the CPU - safetensors and the Q8_0 GGUF, a prompt of two chunks,
    // decoding, the float and the half-float cache
    procedure TinyQwen2MatchesCpu;
    /// EnqueueLinearIds (linear8ids) against the reference Linear of the
    // expert each slot names: F32, Q8_0, Q5_0, Q4_K, ids repeated, x per slot
    // and shared
    procedure LinearIdsMatchesReference;
    /// Qwen2.5-0.5B-Instruct (Q8_0 and Q4_K_M GGUF) on the device against the
    // fast CPU backend, teacher-forced - skipped without the model
    procedure Qwen25GgufMatchesCpu;
    /// B14: embedding rows in place of ids on the device - the bits of the
    // ids, mixed with ids over chunks, no id in the cache there
    procedure EmbeddingRowsOnDevice;
    /// B14: the talker and the code predictor of Qwen3-TTS fed with rows on
    // the device against PyTorch (logits) - skipped without the model
    procedure Qwen3TtsOnDevice;
    /// B14: ForgetWeights - new weights at the address of forgotten ones are
    // served, other ranges kept; a freed model forgets its weights
    procedure ForgetWeightsServesNew;
    /// D4b: sliding windows and layers without rotary positions (tiny Qwen3
    // with a window, tiny SmolLM3 with both) against the CPU: a prompt in two
    // chunks, one range and a range per key, both cache formats, decode steps
    // (a CUDA graph), a branch back behind the window
    procedure SlidingWindowMatchesCpu;
    /// D3c: the route kernel against the host's choice (TDecoderModel.Route,
    // the reference below checked against it): random logits and ties, with
    // and without renorming, 8 of 128 and 2 of 8 experts
    procedure RouteMatchesHost;
  end;


implementation

/// deterministic pseudo-random values in [-aScale, aScale)
procedure Fill(var aValues: TSingleDynArray; aCount: PtrInt; aSeed: cardinal;
  aScale: single);
var
  i: PtrInt;
begin
  SetLength(aValues, aCount);
  for i := 0 to aCount - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    aValues[i] := ((aSeed shr 8) / 16777216 - 0.5) * 2 * aScale;
  end;
end;

procedure TTestInferCuda.Setup;
begin
  if TInferCuda.Available then
  begin
    fCu := TInferCuda.Create;
    AddConsole('CUDA device: % (compute %.%, % SMs, % MB, driver %)',
      [fCu.DeviceName, fCu.ComputeMajor, fCu.ComputeMinor, fCu.Multiprocessors,
       fCu.TotalMemory shr 20, fCu.DriverVersion]);
  end;
end;

procedure TTestInferCuda.CleanUp;
begin
  FreeAndNil(fCu);
end;

function TTestInferCuda.HasDevice: boolean;
var
  refused: boolean;
begin
  result := fCu <> nil;
  if result then
    exit;
  // without a device a direct Create raises EInferCuda - nothing else (no
  // release of a context it never took)
  refused := false;
  try
    TInferCuda.Create.Free;
  except
    on EInferCuda do
      refused := true;
  end;
  Check(refused, 'TInferCuda.Create without a device raises EInferCuda');
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_CUDA') <> '1',
    'LAI_REQUIRE_CUDA=1 but no CUDA device (%)', [CUDA_LIBRARY]);
  {$ifdef CPU64}
  AddConsole('SKIP CUDA: no driver or no device (%)', [CUDA_LIBRARY]);
  {$else}
  AddConsole('SKIP CUDA: a 64-bit process is needed');
  {$endif CPU64}
end;

procedure TTestInferCuda.AddMatchesCpu;
const
  COUNTS: array[0..1] of cardinal = (1000003, 17);
  BLOCK = 256;
var
  module: TInferCuModule;
  kernel: TCuHandle;
  a, b, y: TInferCuBuffer;
  va, vb, vy: TSingleDynArray;
  i, k, bad: PtrInt;
  n: cardinal; // the kernel's unsigned int
  pa, pb, py: TCuDevicePtr;
  sum: single;
begin
  if not HasDevice then
    exit;
  module := TInferCuModule.Create(fCu, @PTX_BASIC);
  try
    kernel := module.Kernel('add');
    for k := 0 to high(COUNTS) do
    begin
      n := COUNTS[k];
      SetLength(va, n);
      SetLength(vb, n);
      SetLength(vy, n);
      for i := 0 to n - 1 do
      begin
        va[i] := (i mod 1000) * 0.001 - 0.37;
        vb[i] := 1 / (i + 3);
      end;
      a := nil;
      b := nil;
      y := nil;
      try
        a := TInferCuBuffer.Create(fCu, n * SizeOf(single));
        b := TInferCuBuffer.Create(fCu, n * SizeOf(single));
        y := TInferCuBuffer.Create(fCu, n * SizeOf(single));
        a.Upload(0, pointer(va), n * SizeOf(single));
        b.Upload(0, pointer(vb), n * SizeOf(single));
        pa := a.Ptr;
        pb := b.Ptr;
        py := y.Ptr;
        fCu.Launch(kernel, [@pa, @pb, @py, @n], (n + BLOCK - 1) div BLOCK, BLOCK);
        y.Download(0, pointer(vy), n * SizeOf(single));
      finally
        y.Free;
        b.Free;
        a.Free;
      end;
      bad := 0;
      for i := 0 to n - 1 do
      begin
        sum := va[i] + vb[i];
        if PCardinal(@sum)^ <> PCardinal(@vy[i])^ then
          inc(bad);
      end;
      CheckUtf8(bad = 0, 'add of %: % of the sums differ from the CPU', [n, bad]);
    end;
  finally
    module.Free;
  end;
end;

procedure TTestInferCuda.RefusesMisuse;
var
  module: TInferCuModule;
  buf: TInferCuBuffer;
  tmp: array[0..15] of byte;
  refused: boolean;
  msg: string;
begin
  if not HasDevice then
    exit;
  refused := false;
  try
    TInferCuBuffer.Create(fCu, 0).Free;
  except
    on EInferCuda do
      refused := true;
  end;
  Check(refused, 'a buffer of no bytes');
  buf := TInferCuBuffer.Create(fCu, 16);
  try
    refused := false;
    try
      buf.Upload(8, @tmp, 16);
    except
      on EInferCuda do
        refused := true;
    end;
    Check(refused, 'an upload past the end');
    refused := false;
    try
      buf.Download(-1, @tmp, 4);
    except
      on EInferCuda do
        refused := true;
    end;
    Check(refused, 'a download before the start');
    msg := '';
    try
      // offset + size beyond Int64: the range check must not wrap around and
      // let the copy reach the driver
      buf.Upload(8, @tmp, high(Int64) - 4);
    except
      on e: EInferCuda do
        msg := e.Message;
    end;
    Check(Pos('.Upload:', msg) > 0, 'an upload whose end overflows: ' + msg);
  finally
    buf.Free;
  end;
  module := TInferCuModule.Create(fCu, @PTX_BASIC);
  try
    msg := '';
    try
      module.Kernel('no_such_kernel');
    except
      on e: EInferCuda do
        msg := e.Message;
    end;
    // the driver's error name comes with the call that failed
    Check(Pos('CUDA_ERROR_NOT_FOUND', msg) > 0, msg);
    refused := false;
    try
      fCu.Launch(module.Kernel('add'), [], 0, 256);
    except
      on EInferCuda do
        refused := true;
    end;
    Check(refused, 'an empty grid');
  finally
    module.Free;
  end;
  refused := false;
  try
    // not PTX: the driver refuses to load it
    TInferCuModule.Create(fCu, PAnsiChar('not a module')).Free;
  except
    on EInferCuda do
      refused := true;
  end;
  Check(refused, 'a module that is no PTX');
end;

procedure TTestInferCuda.UploadDownload;
const
  N = 4096;
var
  buf, other: TInferCuBuffer;
  up, down, zero: array[0..N - 1] of byte;
  i: PtrInt;
  refused: boolean;
begin
  if not HasDevice then
    exit;
  for i := 0 to N - 1 do
    up[i] := byte(i * 7 + 3);
  FillCharFast(zero, SizeOf(zero), 0);
  buf := TInferCuBuffer.Create(fCu, N);
  try
    Check(buf.Size = N, 'size');
    // in three pieces, one at an odd offset
    buf.Upload(0, @up[0], 1000);
    buf.Upload(1000, @up[1000], 1);
    buf.Upload(1001, @up[1001], N - 1001);
    FillCharFast(down, SizeOf(down), 0);
    buf.Download(0, @down, N);
    Check(CompareMem(@up, @down, N), 'whole');
    FillCharFast(down, SizeOf(down), 0);
    buf.Download(777, @down, 333);
    Check(CompareMem(@up[777], @down, 333), 'at an offset');
    // device to device, between offsets, then back
    other := TInferCuBuffer.Create(fCu, N);
    try
      other.Upload(0, @zero, N);
      buf.CopyTo(100, other, 3001, 1000);
      FillCharFast(down, SizeOf(down), 0);
      other.Download(0, @down, N);
      Check(CompareMem(@up[100], @down[3001], 1000), 'CopyTo: the bytes');
      Check(CompareMem(@zero, @down, 3001) and
        CompareMem(@zero, @down[4001], N - 4001), 'CopyTo: nothing else');
      refused := false;
      try
        buf.CopyTo(0, other, 3500, 1000); // past the end of other
      except
        on EInferCuda do
          refused := true;
      end;
      Check(refused, 'CopyTo past the end');
    finally
      other.Free;
    end;
  finally
    buf.Free;
  end;
end;



type
  /// another TInferCuda on another thread: copies, a kernel and a wait
  // while the test's instance captures a graph (the same primary context)
  TCudaSideThread = class(TThread)
  protected
    fCu: TInferCuda;
    procedure Execute; override;
  public
    Bad: integer;
    Error: string;
    constructor Create(aCu: TInferCuda);
  end;

constructor TCudaSideThread.Create(aCu: TInferCuda);
begin
  fCu := aCu; // before the thread starts
  Bad := -1;
  inherited Create(false);
end;

procedure TCudaSideThread.Execute;
const
  COUNT = 64;
var
  module: TInferCuModule;
  a, y: TInferCuBuffer;
  va, vy: array[0..COUNT - 1] of single;
  pa, py: TCuDevicePtr;
  n: cardinal;
  i: PtrInt;
  sum: single;
begin
  module := nil;
  a := nil;
  y := nil;
  try
    try
      for i := 0 to COUNT - 1 do
        va[i] := i * 0.5;
      module := TInferCuModule.Create(fCu, @PTX_BASIC);
      a := TInferCuBuffer.Create(fCu, COUNT * 4);
      y := TInferCuBuffer.Create(fCu, COUNT * 4);
      a.Upload(0, @va, COUNT * 4);
      pa := a.Ptr;
      py := y.Ptr;
      n := COUNT;
      fCu.Enqueue(module.Kernel('add'), [@pa, @pa, @py, @n], 1, COUNT);
      fCu.Synchronize;
      y.Download(0, @vy, COUNT * 4);
      Bad := 0;
      for i := 0 to COUNT - 1 do
      begin
        sum := va[i] + va[i];
        if sum <> vy[i] then
          inc(Bad);
      end;
    except
      on E: Exception do
        Error := E.Message;
    end;
  finally
    y.Free;
    a.Free;
    module.Free;
  end;
end;

procedure TTestInferCuda.GraphRunsCapture;
const
  COUNT = 1000;
  BLOCK = 256;
var
  module: TInferCuModule;
  kernel: TCuHandle;
  a, b, y: TInferCuBuffer;
  va, vb, vy: TSingleDynArray;
  exec: TCuHandle;
  pa, pb, py: TCuDevicePtr;
  n: cardinal;
  i, bad: PtrInt;
  sum: single;
  raised: boolean;
  other: TInferCuda;
  side: TCudaSideThread;
  a2: TInferCuBuffer;

  // y = a + b + b (two adds, the second over y), then a copy of y[0 .. 9]
  // to y[10 .. 19] - over the first aCount values only
  procedure Steps(aCount: cardinal);
  begin
    n := aCount;
    pa := a.Ptr;
    pb := b.Ptr;
    py := y.Ptr;
    fCu.Enqueue(kernel, [@pa, @pb, @py, @n], (n + BLOCK - 1) div BLOCK, BLOCK);
    pa := y.Ptr;
    fCu.Enqueue(kernel, [@pa, @pb, @py, @n], (n + BLOCK - 1) div BLOCK, BLOCK);
    y.CopyTo(0, y, 10 * 4, 10 * 4);
  end;

  // compare y with the CPU for the first aCount values (the copy included)
  procedure Verify(aCount: PtrInt; const aWhat: RawUtf8);
  var
    j, k: PtrInt;
  begin
    y.Download(0, pointer(vy), COUNT * 4); // waits for the graph
    bad := 0;
    for j := 0 to aCount - 1 do
    begin
      // two single additions, each rounded (Delphi Win64 evaluates a single
      // expression in double precision)
      k := j;
      if (j >= 10) and (j < 20) then
        k := j - 10;
      sum := va[k] + vb[k];
      sum := sum + vb[k];
      if PCardinal(@sum)^ <> PCardinal(@vy[j])^ then
        inc(bad);
    end;
    for j := aCount to COUNT - 1 do
      if vy[j] <> -1 then
        inc(bad);
    CheckUtf8(bad = 0, '%: % values differ', [aWhat, bad]);
  end;

  procedure Reset;
  var
    j: PtrInt;
  begin
    for j := 0 to COUNT - 1 do
      vy[j] := -1;
    y.Upload(0, pointer(vy), COUNT * 4);
  end;

begin
  if not HasDevice then
    exit;
  SetLength(va, COUNT);
  SetLength(vb, COUNT);
  SetLength(vy, COUNT);
  for i := 0 to COUNT - 1 do
  begin
    va[i] := i * 0.25 - 7;
    vb[i] := 1 / (i + 2);
  end;
  module := TInferCuModule.Create(fCu, @PTX_BASIC);
  a := nil;
  b := nil;
  y := nil;
  exec := nil;
  try
    kernel := module.Kernel('add');
    a := TInferCuBuffer.Create(fCu, COUNT * 4);
    b := TInferCuBuffer.Create(fCu, COUNT * 4);
    y := TInferCuBuffer.Create(fCu, COUNT * 4);
    a.Upload(0, pointer(va), COUNT * 4);
    b.Upload(0, pointer(vb), COUNT * 4);
    // first capture: instantiated, launched by CaptureEnd
    Reset;
    fCu.CaptureBegin;
    Steps(500);
    fCu.CaptureEnd(exec);
    Check(exec <> nil, 'executable graph');
    Verify(500, 'graph');
    // the same shape, another count: the graph takes the new parameters
    Reset;
    fCu.CaptureBegin;
    Steps(COUNT);
    fCu.CaptureEnd(exec);
    Verify(COUNT, 'updated graph');
    // another shape (no copy): the graph is made again
    Reset;
    fCu.CaptureBegin;
    n := 300;
    pa := a.Ptr;
    pb := b.Ptr;
    py := y.Ptr;
    fCu.Enqueue(kernel, [@pa, @pb, @py, @n], (n + BLOCK - 1) div BLOCK, BLOCK);
    fCu.CaptureEnd(exec);
    y.Download(0, pointer(vy), COUNT * 4);
    bad := 0;
    for i := 0 to COUNT - 1 do
    begin
      sum := va[i] + vb[i];
      if ((i < 300) and (PCardinal(@sum)^ <> PCardinal(@vy[i])^)) or
         ((i >= 300) and (vy[i] <> -1)) then
        inc(bad);
    end;
    CheckUtf8(bad = 0, 'graph of another shape: % values differ', [bad]);
    // copies and waits are refused during a capture; an abort ends it
    fCu.CaptureBegin;
    Check(fCu.Capturing, 'capturing');
    raised := false;
    try
      y.Upload(0, pointer(vy), 4);
    except
      on EInferCuda do
        raised := true;
    end;
    Check(raised, 'no Upload during a capture');
    raised := false;
    try
      fCu.Synchronize;
    except
      on EInferCuda do
        raised := true;
    end;
    Check(raised, 'no Synchronize during a capture');
    raised := false;
    try
      fCu.CaptureBegin;
    except
      on EInferCuda do
        raised := true;
    end;
    Check(raised, 'no nested capture');
    fCu.CaptureAbort;
    Check(not fCu.Capturing, 'aborted');
    // the stream works on: a launch and the earlier graph again
    Reset;
    fCu.CaptureBegin;
    Steps(COUNT);
    fCu.CaptureEnd(exec);
    Verify(COUNT, 'after an abort');
    // another instance on another thread works while this one captures:
    // neither waits for the other's stream
    other := TInferCuda.Create;
    try
      // no copy between the buffers of two instances (two streams)
      a2 := TInferCuBuffer.Create(other, COUNT * 4);
      try
        raised := false;
        try
          y.CopyTo(0, a2, 0, 4);
        except
          on EInferCuda do
            raised := true;
        end;
        Check(raised, 'no CopyTo into another instance');
      finally
        a2.Free;
      end;
      Reset;
      fCu.CaptureBegin;
      Steps(COUNT);
      side := TCudaSideThread.Create(other);
      try
        side.WaitFor;
        CheckUtf8((side.Bad = 0) and (side.Error = ''),
          'another instance during a capture: % differ %', [side.Bad, side.Error]);
      finally
        side.Free;
      end;
      fCu.CaptureEnd(exec);
      Verify(COUNT, 'graph beside another instance');
    finally
      other.Free;
    end;
  finally
    fCu.CaptureAbort;
    fCu.GraphFree(exec);
    y.Free;
    b.Free;
    a.Free;
    module.Free;
  end;
  Check(exec = nil, 'GraphFree');
end;

procedure TTestInferCuda.LinearMatchesReference;
const
  // rows, in, out: one row, partial tiles, more than 65535 outputs; gemm8
  // from 16 rows on, linear8 below; then three F32 only: inputs no multiple
  // of 32 (linear8 at 20 rows) or of 8 (linear); the last four not for the
  // K-quants: whole blocks of 32 but no multiple of 256 (Qwen2.5's 896), an
  // odd number of blocks (rows not 4-aligned for Q8_0 and Q5_0)
  SHAPES: array[0..12, 0..2] of integer = (
    (1, 256, 1), (3, 512, 17), (37, 1024, 33), (5, 3072, 20), (1, 256, 70001),
    (70, 512, 130), (20, 520, 9), (3, 100, 9), (16, 36, 66), (1, 32, 3),
    (3, 96, 17), (1, 896, 65), (20, 896, 33));
  TYPES: array[0..4] of TInferDType = (idtF32, idtQ8_0, idtQ4_K, idtQ6_K,
    idtQ5_0);
var
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  fast: TInferCpuFastBackend;
  t, i, o, m, rows, nin, nout: PtrInt;
  kind: TInferDType;
  x, bias, wf, wd, y1, y2, y3: TSingleDynArray;
  wq: RawByteString;
  data: pointer;
  b: PSingleArray;
  e: double;
  calls: Int64;
begin
  if not HasDevice then
    exit;
  gpu := TInferCudaBackend.Create;
  ref := TInferCpuBackend.Create;
  fast := TInferCpuFastBackend.Create;
  try
    // the plain linear kernel, then the fast ones
    for m := 0 to 1 do
    for t := 0 to high(TYPES) do
    begin
      gpu.FastLinear := m = 1;
      kind := TYPES[t];
      for i := 0 to high(SHAPES) do
      begin
        if (i in [6..8]) and (kind <> idtF32) then
          continue;
        if (i > 8) and (kind in [idtQ4_K, idtQ6_K]) then
          continue;
        rows := SHAPES[i, 0];
        nin := SHAPES[i, 1];
        nout := SHAPES[i, 2];
        Fill(x, rows * nin, 11 + i, 3);
        Fill(bias, nout, 21 + i, 1);
        b := nil;
        if not odd(i) then
          b := pointer(bias);
        case kind of
          idtF32:
            begin
              Fill(wf, nout * nin, 31 + i, 0.1);
              data := pointer(wf);
            end;
          idtQ8_0:
            begin
              wq := RandomQ8(nout, nin, 41 + i);
              data := pointer(wq);
            end;
          idtQ5_0:
            begin
              wq := RandomQ5(nout, nin, 141 + i);
              data := pointer(wq);
            end;
        else
          begin
            wq := RandomK(kind, nout, nin, 51 + i + 10 * t);
            data := pointer(wq);
          end;
        end;
        SetLength(wd, nout * nin);
        for o := 0 to nout - 1 do
          InferRowToF32(data, kind, o, nin, @wd[o * nin]);
        SetLength(y1, rows * nout);
        SetLength(y2, rows * nout);
        ref.Linear(pointer(x), rows, nin, InferWeight(pointer(wd), idtF32), nout,
          b, pointer(y1));
        // the weights of the previous shape are freed: their address may come
        // back for other data of the same size (the cache is keyed by it)
        gpu.ClearWeights;
        calls := gpu.GpuLinears;
        gpu.Linear(pointer(x), rows, nin, InferWeight(data, kind), nout, b,
          pointer(y2));
        CheckUtf8(gpu.GpuLinears = calls + 1, '% on the device', [INFER_DTYPE_NAME[kind]]);
        e := RelError(pointer(y2), pointer(y1), rows * nout);
        CheckUtf8(e <= 1e-5, '% % x % x % (fast %): relative error %',
          [INFER_DTYPE_NAME[kind], rows, nin, nout, gpu.FastLinear, e]);
        // again, from the weight cache: the same bits
        SetLength(y3, rows * nout);
        gpu.Linear(pointer(x), rows, nin, InferWeight(data, kind), nout, b,
          pointer(y3));
        CheckUtf8(SameBits(pointer(y2), pointer(y3), rows * nout),
          '% cached weights', [INFER_DTYPE_NAME[kind]]);
      end;
    end;
    // a weight type without a kernel: the CPU, bit for bit as the fast backend
    wq := RandomQ4(8, 64, 5);
    Fill(x, 2 * 64, 3, 2);
    SetLength(y1, 2 * 8);
    SetLength(y2, 2 * 8);
    calls := gpu.GpuLinears;
    gpu.Linear(pointer(x), 2, 64, InferWeight(pointer(wq), idtQ4_0), 8, nil,
      pointer(y1));
    fast.Linear(pointer(x), 2, 64, InferWeight(pointer(wq), idtQ4_0), 8, nil,
      pointer(y2));
    Check(gpu.GpuLinears = calls, 'Q4_0 stays on the CPU');
    Check(SameBits(pointer(y1), pointer(y2), 2 * 8), 'Q4_0 as the fast backend');
  finally
    fast.Free;
    ref.Free;
    gpu.Free;
  end;
end;


procedure TTestInferCuda.OpsMatchCpu;
var
  ops: TInferCuModule;
  ref: TInferCpuBackend;
  kNorm, kAdd, kSwi, kQk, kAtt, kMerge: TCuHandle;
  bufs: array of TInferCuBuffer;

  // a device copy of v (freed at the end)
  function Up(const v: TSingleDynArray): TCuDevicePtr;
  var
    b: TInferCuBuffer;
  begin
    b := TInferCuBuffer.Create(fCu, length(v) * 4 + 4);
    SetLength(bufs, length(bufs) + 1);
    bufs[high(bufs)] := b;
    if v <> nil then
      b.Upload(0, pointer(v), length(v) * 4);
    result := b.Ptr;
  end;

  // the buffer of aPtr back into v (its length), after every enqueued kernel
  procedure Down(aPtr: TCuDevicePtr; var v: TSingleDynArray);
  var
    i: PtrInt;
  begin
    fCu.Synchronize;
    for i := 0 to high(bufs) do
      if bufs[i].Ptr = aPtr then
      begin
        bufs[i].Download(0, pointer(v), length(v) * 4);
        exit;
      end;
  end;

  procedure Near(const a, b: TSingleDynArray; aTol: double; const aWhat: RawUtf8);
  var
    e: double;
  begin
    e := RelError(pointer(a), pointer(b), length(b));
    CheckUtf8(e <= aTol, '%: relative error %', [aWhat, e]);
  end;

  procedure NormAndAdd(aRows, aCols: PtrInt);
  var
    x, t, w, y, xc, yc: TSingleDynArray;
    px, pt, pw, py: TCuDevicePtr;
    cols: cardinal;
    eps: single;
  begin
    Fill(x, aRows * aCols, 7 + aCols, 3);
    Fill(t, aRows * aCols, 8 + aCols, 2);
    Fill(w, aCols, 9 + aCols, 1);
    SetLength(y, aRows * aCols);
    SetLength(yc, aRows * aCols);
    cols := aCols;
    eps := 1e-6;
    // rmsnorm
    px := Up(x);
    pw := Up(w);
    py := Up(y);
    fCu.Enqueue(kNorm, [@px, @pw, @py, @cols, @eps], aRows, 128);
    Down(py, yc);
    ref.RmsNorm(pointer(x), aRows, aCols, pointer(w), eps, pointer(y));
    Near(yc, y, 1e-5, FormatUtf8('rmsnorm % x %', [aRows, aCols]));
    // addnorm: x += t, then the norm of x
    pt := Up(t);
    fCu.Enqueue(kAdd, [@px, @pt, @pw, @py, @cols, @eps], aRows, 128);
    SetLength(xc, aRows * aCols);
    Down(px, xc);
    Down(py, yc);
    ref.Add(pointer(x), pointer(t), aRows * aCols);
    ref.RmsNorm(pointer(x), aRows, aCols, pointer(w), eps, pointer(y));
    Near(xc, x, 1e-6, FormatUtf8('addnorm residual % x %', [aRows, aCols]));
    Near(yc, y, 1e-5, FormatUtf8('addnorm % x %', [aRows, aCols]));
  end;

  procedure SwiGluCase(aCount: PtrInt);
  var
    g, u, gc: TSingleDynArray;
    pg, pu: TCuDevicePtr;
    n: cardinal;
  begin
    Fill(g, aCount, 21, 6);
    Fill(u, aCount, 22, 2);
    pg := Up(g);
    pu := Up(u);
    n := aCount;
    fCu.Enqueue(kSwi, [@pg, @pu, @n], (n + 127) div 128, 128);
    SetLength(gc, aCount);
    Down(pg, gc);
    ref.SwiGlu(pointer(g), pointer(u), aCount);
    Near(gc, g, 1e-5, FormatUtf8('swiglu %', [aCount]));
  end;

  // qkrope of aRows new rows after aPast cached ones, then the attention of
  // those rows over the whole cache - both against the CPU; the attention
  // blocks hold aHeadBlock heads times 8 div aHeadBlock rows, the keys of a
  // row cut into aSplits ranges (merged by attnmerge when more than one);
  // aNorm false: no RMSNorm per head (Qwen2), qn and kn not read
  procedure AttentionCase(aRows, aHeads, aKvHeads, aHeadDim, aPast, aHeadBlock,
    aSplits: PtrInt; aNorm: boolean = true; aWindow: PtrInt = 0;
    aRope: boolean = true);
  var
    q, k, v, qn, kn, cs, cosv, sinv, keys, values, y, kc, vc, qc, yc,
      part: TSingleDynArray;
    pq, pk, pv, pqn, pkn, pcs, pkc, pvc, py, ppart: TCuDevicePtr;
    rows, heads, kvHeads, headDim, past, splits, hb, rb, norm, rope,
      window: cardinal;
    eps: single;
    half, kvw, cp, r, i: PtrInt;
  begin
    half := aHeadDim shr 1;
    kvw := aKvHeads * aHeadDim;
    cp := aPast + aRows + 3; // a cache larger than used
    Fill(q, aRows * aHeads * aHeadDim, 31, 2);
    Fill(k, aRows * kvw, 32, 2);
    Fill(v, aRows * kvw, 33, 2);
    Fill(qn, aHeadDim, 34, 1);
    Fill(kn, aHeadDim, 35, 1);
    // rotary factors of the new rows: cos, then sin (as TQwen3Model.Rotary)
    SetLength(cosv, aRows * half);
    SetLength(sinv, aRows * half);
    for r := 0 to aRows - 1 do
      for i := 0 to half - 1 do
      begin
        cosv[r * half + i] := Cos((aPast + r) * (i + 1) * 0.01);
        sinv[r * half + i] := Sin((aPast + r) * (i + 1) * 0.01);
      end;
    cs := copy(cosv);
    SetLength(cs, 2 * aRows * half);
    MoveFast(pointer(sinv)^, cs[aRows * half], aRows * half * 4);
    // the cached rows: random keys and values, the new ones written by qkrope;
    // with a window, the rows no query sees are NaN - read, they would show
    Fill(keys, cp * kvw, 36, 1);
    Fill(values, cp * kvw, 37, 1);
    if aWindow > 0 then
      for i := 0 to (aPast - aWindow + 1) * kvw - 1 do
      begin
        PCardinal(@keys[i])^ := $7fc00000;
        PCardinal(@values[i])^ := $7fc00000;
      end;
    pq := Up(q);
    pk := Up(k);
    pv := Up(v);
    pqn := Up(qn);
    pkn := Up(kn);
    pcs := Up(cs);
    pkc := Up(keys);
    pvc := Up(values);
    rows := aRows;
    heads := aHeads;
    kvHeads := aKvHeads;
    headDim := aHeadDim;
    past := aPast;
    splits := aSplits;
    hb := aHeadBlock;
    rb := 8 div aHeadBlock;
    eps := 1e-6;
    norm := ord(aNorm);
    rope := ord(aRope);
    window := aWindow;
    // every buffer up first: the two kernels then run back to back, with no
    // copy (which waits for the stream) between them
    SetLength(y, aRows * aHeads * aHeadDim);
    py := Up(y);
    SetLength(part, aHeads * aRows * aSplits * (aHeadDim + 2));
    ppart := Up(part);
    fCu.Enqueue(kQk, [@pq, @pk, @pv, @pqn, @pkn, @pcs, @pkc, @pvc, @rows, @heads,
      @kvHeads, @headDim, @past, @eps, @norm, @rope], aHeads + aKvHeads, 128, aRows);
    fCu.Enqueue(kAtt, [@pq, @pkc, @pvc, @py, @ppart, @rows, @heads, @headDim,
      @kvHeads, @past, @splits, @hb, @rb, @window], aHeads div aHeadBlock, 128,
      (aRows + PtrInt(rb) - 1) div PtrInt(rb), aSplits);
    if aSplits > 1 then
      fCu.Enqueue(kMerge, [@ppart, @py, @rows, @heads, @headDim, @splits],
        aHeads, 128, aRows);
    SetLength(qc, length(q));
    SetLength(kc, length(keys));
    SetLength(vc, length(values));
    SetLength(yc, length(y));
    Down(pq, qc);
    Down(pkc, kc);
    Down(pvc, vc);
    Down(py, yc);
    // the CPU: per-head norms, rotary positions, the store into the cache
    if aNorm then
    begin
      ref.RmsNorm(pointer(q), aRows * aHeads, aHeadDim, pointer(qn), eps, pointer(q));
      ref.RmsNorm(pointer(k), aRows * aKvHeads, aHeadDim, pointer(kn), eps, pointer(k));
    end;
    if aRope then
    begin
      ref.Rope(pointer(q), aRows, aHeads, aHeadDim, pointer(cosv), pointer(sinv));
      ref.Rope(pointer(k), aRows, aKvHeads, aHeadDim, pointer(cosv), pointer(sinv));
    end;
    MoveFast(pointer(k)^, keys[aPast * kvw], aRows * kvw * 4);
    MoveFast(pointer(v)^, values[aPast * kvw], aRows * kvw * 4);
    Near(qc, q, 1e-5, FormatUtf8('qkrope q % rows, % heads of %',
      [aRows, aHeads, aHeadDim]));
    // the NaN rows before a window: not compared (NaN is never near)
    i := 0;
    if aWindow > 0 then
      i := MaxPtrInt(aPast - aWindow + 1, 0) * kvw;
    Near(copy(kc, i, length(kc) - i), copy(keys, i, length(keys) - i), 1e-5,
      FormatUtf8('qkrope keys after %', [aPast]));
    Check(CompareMem(pointer(vc), pointer(values), length(values) * 4),
      'qkrope values copied, the rest of the cache untouched');
    ref.CausalAttention(pointer(q), aRows, aHeads, aHeadDim, pointer(keys),
      pointer(values), aPast, aKvHeads, pointer(y), aWindow);
    Near(yc, y, 2e-5, FormatUtf8(
      'attention % rows after %, % heads / % KV heads of %, % per block, % ' +
      'splits, window %, rope %', [aRows, aPast, aHeads, aKvHeads, aHeadDim,
      aHeadBlock, aSplits, aWindow, aRope]));
  end;

var
  i: PtrInt;
begin
  if not HasDevice then
    exit;
  ref := TInferCpuBackend.Create;
  ops := TInferCuModule.Create(fCu, @PTX_OPS);
  try
    kNorm := ops.Kernel('rmsnorm');
    kAdd := ops.Kernel('addnorm');
    kSwi := ops.Kernel('swiglu');
    kQk := ops.Kernel('qkrope');
    kAtt := ops.Kernel('attention');
    kMerge := ops.Kernel('attnmerge');
    NormAndAdd(5, 1024);
    NormAndAdd(3, 100); // no multiple of the block
    SwiGluCase(100003);
    AttentionCase(7, 4, 2, 64, 0, 2, 1);     // a prompt, 4 rows per block
    AttentionCase(7, 4, 2, 64, 0, 1, 3);     // early rows: empty late ranges
    AttentionCase(3, 16, 8, 128, 20, 2, 1);  // more rows after a cached prefix
    AttentionCase(1, 16, 8, 128, 300, 2, 1); // decoding: more keys than a tile
    AttentionCase(1, 16, 8, 128, 300, 2, 5); // decoding, the keys in 5 ranges
    AttentionCase(2, 4, 4, 256, 5, 1, 1);    // the widest head
    AttentionCase(2, 4, 4, 256, 5, 1, 2);
    AttentionCase(9, 8, 1, 64, 130, 8, 4);   // 8 heads of one KV head
    AttentionCase(5, 14, 2, 64, 10, 7, 1, false); // no head norm (Qwen2.5-0.5B)
    // D4b: sliding windows (the keys before them NaN) and NoPE
    AttentionCase(7, 4, 2, 64, 0, 2, 1, true, 3);      // a prompt, window 3
    AttentionCase(7, 4, 2, 64, 0, 1, 3, true, 3);      // in ranges
    AttentionCase(3, 16, 8, 128, 20, 2, 1, true, 5);   // after a cached prefix
    AttentionCase(1, 16, 8, 128, 300, 2, 5, true, 64); // decoding, 5 ranges
    AttentionCase(1, 16, 8, 128, 300, 2, 1, true, 400); // a window over all keys
    AttentionCase(9, 8, 1, 64, 130, 8, 4, true, 1);    // a row sees itself only
    AttentionCase(5, 4, 2, 64, 10, 2, 1, true, 0, false); // NoPE
    AttentionCase(5, 4, 2, 64, 10, 2, 2, false, 4, false); // NoPE, window, no norm
  finally
    for i := 0 to high(bufs) do
      bufs[i].Free;
    ops.Free;
    ref.Free;
  end;
end;


procedure TTestInferCuda.TinyQwen3MatchesCpu;
const
  PROMPT = 300; // more than one chunk of the device forward pass (256)
  /// the relative error of the F32 logits: a cache of floats, of half floats
  // (the rounding of 11 bits, as TTestInferVulkan)
  TOL: array[boolean] of double = (1e-4, 4e-3);
var
  dir: TFileName;
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  model: TQwen3Model;
  vk: TQwen3Cuda;
  c1: TQwen3Context;
  c2: TQwen3CuContext;
  ids: TIntegerDynArray;
  l1, l2: TSingleDynArray;
  i, k, v: PtrInt;
  seed: cardinal;
  e: double;
  cmp: TInferCompare;
  id: integer;
  out1, out2: TIntegerDynArray;
begin
  if not HasDevice then
    exit;
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  SetLength(ids, PROMPT);
  seed := 7;
  for i := 0 to PROMPT - 1 do
  begin
    seed := seed * 1103515245 + 12345;
    ids[i] := 3 + (seed shr 16) mod 93;
  end;
  // k and 1: the F32 or the Q8_0 model; k >= 2: the cache in half floats
  for k := 0 to 3 do
  begin
    gpu := TInferCudaBackend.Create;
    ref := TInferCpuBackend.Create;
    model := nil;
    vk := nil;
    c1 := nil;
    c2 := nil;
    try
      if k and 1 = 0 then
        model := TQwen3Model.Create(dir, ref)
      else
        model := TQwen3Model.CreateFromGguf(IncludeTrailingPathDelimiter(FindInferRefDir('tiny-qwen3-gguf')) +
          'q8_0.gguf', ref);
      v := model.Config.VocabSize;
      vk := TQwen3Cuda.Create(model, gpu);
      vk.KvF16 := k >= 2;
      c1 := model.NewContext;
      c2 := vk.NewContext;
      // the prompt: every row's logits
      SetLength(l1, PROMPT * v);
      SetLength(l2, PROMPT * v);
      model.Forward(c1, pointer(ids), PROMPT, pointer(l1), true);
      vk.Forward(c2, pointer(ids), PROMPT, pointer(l2), true);
      Check(c2.Length = PROMPT, 'context length');
      Check(c2.KvF16 = vk.KvF16, 'the cache format of KvF16');
      if k and 1 = 0 then
      begin
        e := RelError(pointer(l2), pointer(l1), PROMPT * v);
        CheckUtf8(e <= TOL[vk.KvF16], 'F32 prompt logits (KvF16 %): relative error %',
          [vk.KvF16, e]);
      end
      else
      begin
        cmp := InferCompare(pointer(l2), pointer(l1), PROMPT * v, 0.05, 0.02);
        CheckUtf8(cmp.Cosine >= 0.999, 'Q8_0 prompt logits: %',
          [InferCompareText(cmp)]);
      end;
      // decoding, one token at a time - the keys of a row in one range or
      // cut into ranges of 64; kernel by kernel, then as a graph
      for i := 0 to 4 do
      begin
        id := ids[i];
        vk.SplitKeys := (i and 1) * 64;
        vk.UseGraphs := i >= 2;
        model.Forward(c1, @id, 1, pointer(l1));
        vk.Forward(c2, @id, 1, pointer(l2));
        if k and 1 = 0 then
        begin
          e := RelError(pointer(l2), pointer(l1), v);
          CheckUtf8(e <= TOL[vk.KvF16], 'F32 decode step % (KvF16 %): relative error %',
            [i, vk.KvF16, e]);
        end
        else
        begin
          cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
          CheckUtf8(cmp.Cosine >= 0.999, 'Q8_0 decode step %: %',
            [i, InferCompareText(cmp)]);
        end;
      end;
      // a reused prefix: the cache cut back, then other tokens
      Check(vk.Reuse(c2, pointer(ids), 100) = 99, 'reuse');
      c1.Truncate(99);
      model.Forward(c1, @ids[150], 20, pointer(l1));
      vk.Forward(c2, @ids[150], 20, pointer(l2));
      cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
      CheckUtf8(cmp.Cosine >= 0.999, 'after reuse: %', [InferCompareText(cmp)]);
      // the backend forgot its weights: the next forward uploads them again
      gpu.ClearWeights;
      id := ids[200];
      model.Forward(c1, @id, 1, pointer(l1));
      vk.Forward(c2, @id, 1, pointer(l2));
      cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
      CheckUtf8(cmp.Cosine >= 0.999, 'after ClearWeights: %', [InferCompareText(cmp)]);
      // greedy: every returned token is cached, as on the CPU
      c1.Reset;
      c2.Reset;
      out1 := model.Greedy(c1, copy(ids, 0, 10), 6, []);
      out2 := vk.Greedy(c2, copy(ids, 0, 10), 6, []);
      Check((length(out2) = 6) and (c2.Length = 10 + 6), 'greedy cached its tokens');
      Check((length(out1) = 6) and (c1.Length = c2.Length), 'greedy lengths');
      // a fresh context (64 positions) grows with 10 tokens cached: they are
      // copied into the larger cache
      FreeAndNil(c2);
      c2 := vk.NewContext;
      c1.Reset;
      model.Forward(c1, pointer(ids), 10, pointer(l1));
      vk.Forward(c2, pointer(ids), 10, pointer(l2));
      model.Forward(c1, @ids[10], 100, pointer(l1));
      vk.Forward(c2, @ids[10], 100, pointer(l2));
      cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
      CheckUtf8(cmp.Cosine >= 0.999, 'after the cache grew: %', [InferCompareText(cmp)]);
    finally
      c2.Free;
      c1.Free;
      vk.Free;
      model.Free;
      ref.Free;
      gpu.Free;
    end;
  end;
end;

procedure TTestInferCuda.Qwen3GgufMatchesCpu;
const
  FILES: array[0..1] of TFileName = ('qwen3-0.6b-q4_k_m.gguf', 'qwen3-0.6b-q8_0.gguf');
  // "Wie verlege ich ein Kupferrohr?" - any ids do; these are ordinary words
  PROMPT: array[0..11] of integer = (
    151644, 872, 198, 54, 645, 2739, 273, 10109, 12153, 3105, 63084, 30);
  STEPS = 16;
var
  dir, gguf: TFileName;
  f, i, v, agree: PtrInt;
  gpu: TInferCudaBackend;
  cpu: TInferCpuFastBackend;
  model: TQwen3Model;
  vk: TQwen3Cuda;
  c1: TQwen3Context;
  c2: TQwen3CuContext;
  l1, l2: TSingleDynArray;
  cmp: TInferCompare;
  worst: double;
  id: integer;
begin
  if not HasDevice then
    exit;
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3-0.6B on the device: a 32-bit process');
  exit;
  {$endif CPU32}
  dir := FindQwen3Folder('model.safetensors');
  for f := 0 to high(FILES) do
  begin
    gguf := '';
    if dir <> '' then
      gguf := ExpandFileName(IncludeTrailingPathDelimiter(dir) + '..' + PathDelim +
        FILES[f]);
    if (gguf = '') or not FileExists(gguf) then
    begin
      CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
        'LAI_REQUIRE_MODEL=1 but no vendor/models/%', [FILES[f]]);
      AddConsole('SKIP Qwen3-0.6B on the device: no vendor/models/%', [FILES[f]]);
      continue;
    end;
    gpu := TInferCudaBackend.Create;
    cpu := TInferCpuFastBackend.Create;
    model := nil;
    vk := nil;
    c1 := nil;
    c2 := nil;
    try
      model := TQwen3Model.CreateFromGguf(gguf, cpu);
      v := model.Config.VocabSize;
      vk := TQwen3Cuda.Create(model, gpu);
      c1 := model.NewContext;
      c2 := vk.NewContext;
      SetLength(l1, v);
      SetLength(l2, v);
      model.Forward(c1, @PROMPT[0], length(PROMPT), pointer(l1));
      vk.Forward(c2, @PROMPT[0], length(PROMPT), pointer(l2));
      worst := 1;
      agree := 0;
      for i := 0 to STEPS - 1 do
      begin
        cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
        if cmp.Cosine < worst then
          worst := cmp.Cosine;
        id := InferArgMax(pointer(l1), v);
        if InferArgMax(pointer(l2), v) = id then
          inc(agree);
        // the CPU's token to both: the comparison stays on one path
        model.Forward(c1, @id, 1, pointer(l1));
        vk.Forward(c2, @id, 1, pointer(l2));
      end;
      AddConsole('% on %: lowest logits cosine %, argmax agrees % of %',
        [FILES[f], fCu.DeviceName, worst, agree, STEPS]);
      CheckUtf8(worst >= 0.99, '% logits cosine %', [FILES[f], worst]);
      CheckUtf8(agree >= STEPS - 2, '% argmax agrees % of %', [FILES[f], agree, STEPS]);
    finally
      c2.Free;
      c1.Free;
      vk.Free;
      model.Free;
      cpu.Free;
      gpu.Free;
    end;
  end;
end;

procedure TTestInferCuda.NativeClientOnDevice;
var
  dir, gguf: TFileName;
  gpu: TInferCudaBackend;
  client: TNativeLlmClient;
  llm: ILlmClient;
  msgs: TLlmMessageDynArray;
  first, req: TLlmChatRequest;
  r1, r2, r3: TLlmChatResponse;
  hits, calls: Int64;
  ctx: TQwen3Context;
begin
  if not HasDevice then
    exit;
  {$ifdef CPU32}
  AddConsole('SKIP the native client on the device: a 32-bit process');
  exit;
  {$endif CPU32}
  dir := FindQwen3Folder('model.safetensors');
  gguf := '';
  if dir <> '' then
    gguf := ExpandFileName(IncludeTrailingPathDelimiter(dir) + '..' + PathDelim +
      'qwen3-0.6b-q8_0.gguf');
  if (gguf = '') or not FileExists(gguf) then
  begin
    Check(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b-q8_0.gguf');
    AddConsole('SKIP the native client on the device: no vendor/models/' +
      'qwen3-0.6b-q8_0.gguf');
    exit;
  end;
  gpu := TInferCudaBackend.Create;
  try
    client := TNativeLlmClient.CreateFromGguf(gguf, '', gpu);
    llm := client; // the interface frees the client, and with it its Device
    client.EnableThinking := false;
    client.MaxTokens := 24;
    SetLength(msgs, 1);
    msgs[0] := LlmMessage(lrUser, 'Name three colors.');
    first := LlmChatRequest('', msgs);
    // on the backend (the model's own forward pass: TInferCudaBackend.Linear
    // per matrix), then on the device (kernels per chunk, no Linear call)
    calls := gpu.GpuLinears;
    r1 := llm.ChatComplete(first);
    Check(gpu.GpuLinears > calls, 'the backend computed the first answer');
    client.Device := TQwen3Cuda.Create(client.Model, gpu);
    // the float cache: the same words as the backend over a whole answer (the
    // half floats of KvF16 may flip a close greedy choice; Qwen3GgufMatchesCpu
    // checks them)
    TQwen3Cuda(client.Device).KvF16 := false;
    Check(client.Model.Device = client.Device, 'the model computes on the device');
    ctx := client.Model.NewContext;
    try
      Check((ctx is TQwen3CuContext) and (ctx.Device = client.Device),
        'a new context is on the device');
    finally
      ctx.Free;
    end;
    calls := gpu.GpuLinears;
    r2 := llm.ChatComplete(first);
    Check(gpu.GpuLinears = calls, 'the device computed the answer, not the backend');
    Check(r2.Content <> '', 'an answer on the device');
    CheckUtf8(r2.Content = r1.Content, 'the device "%" as the backend "%"',
      [r2.Content, r1.Content]);
    // the next turn starts with the cached prompt: computed from there
    hits := client.Cache.Hits;
    SetLength(msgs, 3);
    msgs[1] := LlmMessage(lrAssistant, r2.Content);
    msgs[2] := LlmMessage(lrUser, 'One more, please.');
    req := LlmChatRequest('', msgs);
    r3 := llm.ChatComplete(req);
    Check(r3.Content <> '', 'the next turn on the device');
    Check(client.Cache.Hits = hits + 1, 'the next turn reused the device cache');
    AddConsole('native client on %: "%" / "%"', [fCu.DeviceName, r2.Content,
      r3.Content]);
    // back on the backend: the device and its contexts go
    client.Device := nil;
    Check(client.Model.Device = nil, 'no device any more');
    r1 := llm.ChatComplete(first);
    CheckUtf8(r1.Content = r2.Content, 'the backend again: "%"', [r1.Content]);
    llm := nil; // before the backend it computes with
  finally
    llm := nil;
    gpu.Free;
  end;
end;

procedure TTestInferCuda.TinyHeadDim256MatchesCpu;
const
  PROMPT = 300; // more than one chunk (256)
  /// the relative error: a cache of floats, of half floats
  TOL: array[boolean] of double = (1e-4, 4e-3);
var
  dir: TFileName;
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  model: TQwen3Model;
  cu: TQwen3Cuda;
  c1: TQwen3Context;
  c2: TQwen3CuContext;
  ids: TIntegerDynArray;
  l1, l2: TSingleDynArray;
  i, v: PtrInt;
  seed: cardinal;
  e: double;
  id: integer;
  f16: boolean;
begin
  if not HasDevice then
    exit;
  dir := FindInferRefDir('tiny-qwen3-hd256');
  if CheckFailed(dir <> '', 'tiny-qwen3-hd256 found') then
    exit;
  SetLength(ids, PROMPT);
  seed := 11;
  for i := 0 to PROMPT - 1 do
  begin
    seed := seed * 1103515245 + 12345;
    ids[i] := 3 + (seed shr 16) mod 93;
  end;
  gpu := TInferCudaBackend.Create;
  ref := TInferCpuBackend.Create;
  model := nil;
  cu := nil;
  c1 := nil;
  c2 := nil;
  try
    model := TQwen3Model.Create(dir, ref);
    v := model.Config.VocabSize;
    cu := TQwen3Cuda.Create(model, gpu);
    c1 := model.NewContext;
    c2 := cu.NewContext;
    SetLength(l1, PROMPT * v);
    SetLength(l2, PROMPT * v);
    // a cache of floats, then of half floats: after Reset, the context takes
    // the format of KvF16
    for f16 := false to true do
    begin
      cu.KvF16 := f16;
      c1.Reset;
      c2.Reset;
      model.Forward(c1, pointer(ids), PROMPT, pointer(l1), true);
      cu.Forward(c2, pointer(ids), PROMPT, pointer(l2), true);
      Check(c2.KvF16 = f16, 'the cache format of KvF16');
      e := RelError(pointer(l2), pointer(l1), PROMPT * v);
      CheckUtf8(e <= TOL[f16], 'head dimension 256 (KvF16 %), prompt: relative error %',
        [f16, e]);
      for i := 0 to 3 do
      begin
        id := ids[i];
        model.Forward(c1, @id, 1, pointer(l1));
        cu.Forward(c2, @id, 1, pointer(l2));
        e := RelError(pointer(l2), pointer(l1), v);
        CheckUtf8(e <= TOL[f16],
          'head dimension 256 (KvF16 %), decode step %: relative error %', [f16, i, e]);
      end;
    end;
  finally
    c2.Free;
    c1.Free;
    cu.Free;
    model.Free;
    ref.Free;
    gpu.Free;
  end;
end;


procedure TTestInferCuda.TinyMoeMatchesCpu;
const
  PROMPT = 300; // more than one chunk of the device forward pass (256)
  /// the router of the tiny model is sharp (as in TTestInferQwen3): the
  // rounding of a half-float cache comes out of three layers at up to 4.6e-3
  // (measured 2026-10-17), where the dense tiny-qwen3 stays within 4e-3
  TOL: array[boolean] of double = (1e-4, 6e-3);
var
  dir: TFileName;
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  model: TDecoderModel;
  vk: TQwen3Cuda;
  c1: TQwen3Context;
  c2, c3: TQwen3CuContext;
  ids: TIntegerDynArray;
  l1, l2, l3: TSingleDynArray;
  i, k, v: PtrInt;
  seed: cardinal;
  e: double;
  cmp: TInferCompare;
  id: integer;
begin
  if not HasDevice then
    exit;
  dir := FindInferRefDir('tiny-qwen3moe');
  if CheckFailed(dir <> '', 'tiny-qwen3moe found') then
    exit;
  SetLength(ids, PROMPT);
  seed := 11;
  for i := 0 to PROMPT - 1 do
  begin
    seed := seed * 1103515245 + 12345;
    ids[i] := 3 + (seed shr 16) mod 93;
  end;
  // 0: F32 (each expert a buffer of its own), 1: the Q8_0 GGUF (the experts
  // of a layer one buffer, every expert an offset), 2: Q8_0 quantized at load
  // (each expert its own buffer), 3: F32 with the cache in half floats
  for k := 0 to 3 do
  begin
    gpu := TInferCudaBackend.Create;
    ref := TInferCpuBackend.Create;
    model := nil;
    vk := nil;
    c1 := nil;
    c2 := nil;
    c3 := nil;
    try
      case k of
        1:
          model := NewDecoderModelFromGguf(MakePath([dir, 'gguf', 'q8_0.gguf']), ref);
        2:
          model := TQwen3MoeModel.Create(dir, ref, idtQ8_0);
      else
        model := TQwen3MoeModel.Create(dir, ref);
      end;
      Check(model is TQwen3MoeModel, 'a Qwen3-MoE model');
      v := model.Config.VocabSize;
      vk := TQwen3Cuda.Create(model, gpu);
      vk.KvF16 := k = 3;
      c1 := model.NewContext;
      c2 := vk.NewContext;
      SetLength(l1, PROMPT * v);
      SetLength(l2, PROMPT * v);
      SetLength(l3, v);
      model.Forward(c1, pointer(ids), PROMPT, pointer(l1), true);
      vk.Forward(c2, pointer(ids), PROMPT, pointer(l2), true);
      Check(c2.Length = PROMPT, 'context length');
      if k in [0, 3] then
      begin
        e := RelError(pointer(l2), pointer(l1), PROMPT * v);
        CheckUtf8(e <= TOL[vk.KvF16], 'MoE F32 prompt logits (KvF16 %): relative error %',
          [vk.KvF16, e]);
      end
      else
      begin
        // Q8_0 at load: the CPU also quantizes the activations, the device
        // not - over 300 tokens measured cosine 0.9988 (the router passes the
        // difference on), the Q8_0 GGUF 0.999
        cmp := InferCompare(pointer(l2), pointer(l1), PROMPT * v, 0.5, 0.5);
        CheckUtf8((cmp.NonFinite = 0) and (cmp.Cosine >= 0.998), 'MoE Q8_0 (%) prompt logits: %',
          [k, InferCompareText(cmp)]);
      end;
      // the GGUF has the experts of a layer in one buffer: a decode step
      // chooses them on the device, the others on the host
      CheckUtf8(vk.DeviceRoutable = (k = 1), 'case %: routable on the device', [k]);
      if k = 1 then
      begin
        // the same steps once more with the choice on the host: the same
        // arithmetic (linear8ids is linear8, moecombine the scatteradds)
        c3 := vk.NewContext;
        vk.RouteOnDevice := false;
        vk.Forward(c3, pointer(ids), PROMPT, nil);
        vk.CheckRoute := true;
      end;
      // decoding: one token, two experts of it in a layer
      for i := 0 to 4 do
      begin
        id := ids[i];
        vk.SplitKeys := (i and 1) * 64;
        vk.UseGraphs := true;
        model.Forward(c1, @id, 1, pointer(l1));
        if k = 1 then
        begin
          vk.RouteOnDevice := false;
          vk.Forward(c3, @id, 1, pointer(l3));
          vk.RouteOnDevice := true;
        end;
        vk.Forward(c2, @id, 1, pointer(l2));
        if k = 1 then
        begin
          e := RelError(pointer(l2), pointer(l3), v);
          CheckUtf8(e <= 1e-6, 'MoE decode step %: the choice on the device against ' +
            'the host: relative error %', [i, e]);
        end;
        if k in [0, 3] then
        begin
          e := RelError(pointer(l2), pointer(l1), v);
          CheckUtf8(e <= TOL[vk.KvF16], 'MoE F32 decode step % (KvF16 %): relative error %',
            [i, vk.KvF16, e]);
        end
        else
        begin
          cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
          CheckUtf8((cmp.NonFinite = 0) and (cmp.Cosine >= 0.999), 'MoE Q8_0 (%) decode step %: %',
            [k, i, InferCompareText(cmp)]);
        end;
      end;
      if k = 1 then
      begin
        // CheckRoute: every step of every layer with experts compared
        CheckUtf8((vk.RouteChecks = 5 * 2) and (vk.RouteMismatches = 0) and
          (vk.RouteMaxDiff <= 1e-7), 'the device routed % rows, % otherwise, weights ' +
          'within %', [vk.RouteChecks, vk.RouteMismatches, vk.RouteMaxDiff]);
        vk.CheckRoute := false;
        // and the choice inside a graph: steps decoded as graphs against the
        // host's choice on the device (not the CPU: it quantizes the
        // activations too, a near tie of the router may tip there)
        for i := 5 to 9 do
        begin
          id := ids[i];
          model.Forward(c1, @id, 1, pointer(l1));
          vk.RouteOnDevice := false;
          vk.Forward(c3, @id, 1, pointer(l3));
          vk.RouteOnDevice := true;
          vk.Forward(c2, @id, 1, pointer(l2));
          e := RelError(pointer(l2), pointer(l3), v);
          CheckUtf8(e <= 1e-6, 'MoE decode step % as a graph against the host''s ' +
            'choice: relative error %', [i, e]);
        end;
      end;
      // the backend forgot its weights: the experts go up again
      gpu.ClearWeights;
      id := ids[200];
      model.Forward(c1, @id, 1, pointer(l1));
      vk.Forward(c2, @id, 1, pointer(l2));
      cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
      CheckUtf8((cmp.NonFinite = 0) and (cmp.Cosine >= 0.999), 'MoE after ClearWeights: %', [InferCompareText(cmp)]);
    finally
      c3.Free;
      c2.Free;
      c1.Free;
      vk.Free;
      model.Free;
      ref.Free;
      gpu.Free;
    end;
  end;
end;

procedure TTestInferCuda.Qwen3MoeGgufMatchesCpu;
const
  // "Wie verlege ich ein Kupferrohr?" as for Qwen3-0.6B: the same tokenizer
  PROMPT: array[0..11] of integer = (
    151644, 872, 198, 54, 645, 2739, 273, 10109, 12153, 3105, 63084, 30);
  STEPS = 16;
  /// the largest router gap at which the device may choose other experts
  // than the CPU: a near tie of the 8th and 9th of 128 experts - measured on
  // 2026-10-19 (RTX 5090, CUDA and Vulkan, FPC): 162 of 1344 routes chosen
  // otherwise, at a gap of 0.052 at most (Delphi 13 the same since its CPU
  // computes as FPC, backlog B13); a routing error gives gaps of whole units
  ROUTE_GAP = 0.1;
var
  dir, gguf: TFileName;
  i, v, agree: PtrInt;
  tape: TDecoderRouteTape;
  gpu: TInferCudaBackend;
  cpu: TInferCpuFastBackend;
  model: TDecoderModel;
  vk: TQwen3Cuda;
  c1: TQwen3Context;
  c2: TQwen3CuContext;
  l1, l2: TSingleDynArray;
  cmp: TInferCompare;
  worst: double;
  id: integer;
  timer: TPrecisionTimer;
begin
  if not HasDevice then
    exit;
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3-30B-A3B on the device: a 32-bit process');
  exit;
  {$endif CPU32}
  // a large model of its own: only where it was put (not LAI_REQUIRE_MODEL)
  dir := FindQwen3Folder('model.safetensors');
  gguf := '';
  if dir <> '' then
    gguf := ExpandFileName(IncludeTrailingPathDelimiter(dir) + '..' + PathDelim +
      'qwen3-30b-a3b-q4_k_m.gguf');
  if (gguf = '') or not FileExists(gguf) then
  begin
    AddConsole('SKIP Qwen3-30B-A3B on the device: no vendor/models/qwen3-30b-a3b-q4_k_m.gguf');
    exit;
  end;
  gpu := TInferCudaBackend.Create;
  cpu := TInferCpuFastBackend.Create;
  model := nil;
  vk := nil;
  c1 := nil;
  c2 := nil;
  tape := nil;
  try
    model := NewDecoderModelFromGguf(gguf, cpu);
    v := model.Config.VocabSize;
    timer.Start;
    vk := TQwen3Cuda.Create(model, gpu);
    AddConsole('Qwen3-30B-A3B uploaded in %', [timer.Stop]);
    c1 := model.NewContext;
    c2 := vk.NewContext;
    SetLength(l1, v);
    SetLength(l2, v);
    // the CPU's expert choice handed to the device (TDecoderRouteTape): the
    // logits then differ by arithmetic only; the device's own choice is
    // compared - it may differ at a near tie of the router, never by more
    tape := TDecoderRouteTape.Create;
    model.RouteTape := tape;
    tape.Mode := drmRecord;
    model.Forward(c1, @PROMPT[0], length(PROMPT), pointer(l1));
    tape.Mode := drmReplay;
    vk.Forward(c2, @PROMPT[0], length(PROMPT), pointer(l2));
    worst := 1;
    agree := 0;
    for i := 0 to STEPS - 1 do
    begin
      cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
      if cmp.Cosine < worst then
        worst := cmp.Cosine;
      // InferCompare leaves a NaN or an infinity out of the cosine
      if cmp.NonFinite <> 0 then
        worst := -1;
      id := InferArgMax(pointer(l1), v);
      if InferArgMax(pointer(l2), v) = id then
        inc(agree);
      // the CPU's token to both: the comparison stays on one path
      tape.Mode := drmRecord;
      model.Forward(c1, @id, 1, pointer(l1));
      tape.Mode := drmReplay;
      vk.Forward(c2, @id, 1, pointer(l2));
    end;
    AddConsole('Qwen3-30B-A3B on %: lowest logits cosine %, argmax agrees % of %; ' +
      'the device chose otherwise % of % times, largest router gap %',
      [fCu.DeviceName, worst, agree, STEPS, tape.Differs, tape.Routes, tape.DiffersGap]);
    CheckUtf8(tape.Pending = 0, '% routes not replayed', [tape.Pending]);
    CheckUtf8(worst >= 0.99, 'Qwen3-30B-A3B logits cosine %', [worst]);
    // with the routing given, 16 of 16 at a cosine of 0.9995 - Delphi 13 too
    // since its CPU computes single expressions in single (backlog B13)
    CheckUtf8(agree = STEPS, 'Qwen3-30B-A3B argmax agrees % of %', [agree, STEPS]);
    // a near tie is rare: 12 pct of the routes - the router logits of the
    // wrong row give 47 pct, at small gaps each
    CheckUtf8(tape.Differs * 4 <= tape.Routes, 'Qwen3-30B-A3B: the device chose ' +
      'otherwise % of % times', [tape.Differs, tape.Routes]);
    CheckUtf8(tape.DiffersGap <= ROUTE_GAP, 'Qwen3-30B-A3B: the device chose ' +
      'otherwise at a router gap of %', [tape.DiffersGap]);
    // without the tape a decode step chooses on the device (D3c): the route
    // kernel against TDecoderModel.Route on the same router logits
    model.RouteTape := nil;
    Check(vk.DeviceRoutable, 'Qwen3-30B-A3B routable on the device');
    vk.CheckRoute := true;
    for i := 0 to 7 do
    begin
      id := InferArgMax(pointer(l2), v);
      vk.Forward(c2, @id, 1, pointer(l2));
    end;
    vk.CheckRoute := false;
    AddConsole('Qwen3-30B-A3B routed on the device: % rows, % chosen otherwise ' +
      'than the host, weights within %', [vk.RouteChecks, vk.RouteMismatches,
      vk.RouteMaxDiff]);
    CheckUtf8((vk.RouteChecks = 8 * model.Config.Layers) and (vk.RouteMismatches = 0) and
      (vk.RouteMaxDiff <= 1e-6), 'Qwen3-30B-A3B route kernel: % rows, % otherwise, %',
      [vk.RouteChecks, vk.RouteMismatches, vk.RouteMaxDiff]);
  finally
    if model <> nil then
      model.RouteTape := nil;
    tape.Free;
    c2.Free;
    c1.Free;
    vk.Free;
    model.Free;
    cpu.Free;
    gpu.Free;
  end;
end;

procedure TTestInferCuda.TinyQwen2MatchesCpu;
const
  PROMPT = 300; // more than one chunk of the device forward pass (256)
  /// float cache as for tiny-qwen3; a half-float cache as AttentionRangesMatchCpu
  TOL: array[boolean] of double = (1e-4, 4e-3);
var
  dir: TFileName;
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  model: TDecoderModel;
  vk: TQwen3Cuda;
  c1: TQwen3Context;
  c2: TQwen3CuContext;
  ids: TIntegerDynArray;
  l1, l2: TSingleDynArray;
  i, k, v: PtrInt;
  seed: cardinal;
  e: double;
  cmp: TInferCompare;
  id: integer;
begin
  if not HasDevice then
    exit;
  dir := FindInferRefDir('tiny-qwen2');
  if CheckFailed(dir <> '', 'tiny-qwen2 found') then
    exit;
  SetLength(ids, PROMPT);
  seed := 13;
  for i := 0 to PROMPT - 1 do
  begin
    seed := seed * 1103515245 + 12345;
    ids[i] := 3 + (seed shr 16) mod 93;
  end;
  // 0: F32 with a float cache, 1: F32 with a half-float cache, 2: Q8_0 GGUF
  for k := 0 to 2 do
  begin
    gpu := TInferCudaBackend.Create;
    ref := TInferCpuBackend.Create;
    model := nil;
    vk := nil;
    c1 := nil;
    c2 := nil;
    try
      if k = 2 then
        model := NewDecoderModelFromGguf(MakePath([dir, 'gguf', 'q8_0.gguf']), ref)
      else
        model := NewDecoderModel(dir, ref);
      Check(model is TQwen2Model, 'a Qwen2 model');
      v := model.Config.VocabSize;
      vk := TQwen3Cuda.Create(model, gpu);
      vk.KvF16 := k = 1;
      c1 := model.NewContext;
      c2 := vk.NewContext;
      SetLength(l1, PROMPT * v);
      SetLength(l2, PROMPT * v);
      model.Forward(c1, pointer(ids), PROMPT, pointer(l1), true);
      vk.Forward(c2, pointer(ids), PROMPT, pointer(l2), true);
      if k < 2 then
      begin
        e := RelError(pointer(l2), pointer(l1), PROMPT * v);
        CheckUtf8(e <= TOL[k = 1], 'Qwen2 F32 prompt logits (KvF16 %): relative error %',
          [k = 1, e]);
      end
      else
      begin
        // Q8_0: the CPU also quantizes the activations, the device not
        cmp := InferCompare(pointer(l2), pointer(l1), PROMPT * v, 0.5, 0.5);
        CheckUtf8((cmp.NonFinite = 0) and (cmp.Cosine >= 0.999),
          'Qwen2 Q8_0 prompt logits: %', [InferCompareText(cmp)]);
      end;
      for i := 0 to 3 do
      begin
        id := ids[i];
        model.Forward(c1, @id, 1, pointer(l1));
        vk.Forward(c2, @id, 1, pointer(l2));
        if k < 2 then
        begin
          e := RelError(pointer(l2), pointer(l1), v);
          CheckUtf8(e <= TOL[k = 1], 'Qwen2 F32 decode step % (KvF16 %): relative ' +
            'error %', [i, k = 1, e]);
        end
        else
        begin
          cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
          CheckUtf8((cmp.NonFinite = 0) and (cmp.Cosine >= 0.999),
            'Qwen2 Q8_0 decode step %: %', [i, InferCompareText(cmp)]);
        end;
      end;
    finally
      c2.Free;
      c1.Free;
      vk.Free;
      model.Free;
      ref.Free;
      gpu.Free;
    end;
  end;
end;

procedure TTestInferCuda.LinearIdsMatchesReference;
const
  EXPERTS = 5;
  SLOTS = 4;
  NOUT = 33;
  IDS: array[0..SLOTS - 1] of integer = (3, 0, 4, 3);
  // type, inputs: the K-quant at 256, the others also at 896
  KINDS: array[0..6] of TInferDType = (
    idtF32, idtQ8_0, idtQ5_0, idtQ4_K, idtQ8_0, idtQ5_0, idtF32);
  INPUTS: array[0..6] of integer = (256, 256, 256, 256, 896, 896, 896);
var
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  c, s, o, nin, xr: PtrInt;
  kind: TInferDType;
  perSlot: boolean;
  x, wf, wd, y1, y2: TSingleDynArray;
  wq: RawByteString;
  data: pointer;
  stride: Int64;
  xb, yb, ib: TInferCuBuffer;
  e: double;
begin
  if not HasDevice then
    exit;
  gpu := TInferCudaBackend.Create;
  ref := TInferCpuBackend.Create;
  try
    for c := 0 to high(KINDS) do
      for perSlot := false to true do
      begin
        kind := KINDS[c];
        nin := INPUTS[c];
        case kind of
          idtF32:
            begin
              Fill(wf, EXPERTS * NOUT * nin, 81 + c, 0.1);
              data := pointer(wf);
            end;
          idtQ8_0:
            begin
              wq := RandomQ8(EXPERTS * NOUT, nin, 82 + c);
              data := pointer(wq);
            end;
          idtQ5_0:
            begin
              wq := RandomQ5(EXPERTS * NOUT, nin, 83 + c);
              data := pointer(wq);
            end;
        else
          begin
            wq := RandomK(kind, EXPERTS * NOUT, nin, 84 + c);
            data := pointer(wq);
          end;
        end;
        stride := Int64(NOUT) * (nin div INFER_DTYPE_BLOCK[kind]) *
          INFER_DTYPE_SIZE[kind];
        SetLength(wd, EXPERTS * NOUT * nin);
        for o := 0 to EXPERTS * NOUT - 1 do
          InferRowToF32(data, kind, o, nin, @wd[o * nin]);
        Fill(x, SLOTS * nin, 85 + c, 2);
        // the reference: slot s through expert IDS[s], x row s or row 0
        SetLength(y1, SLOTS * NOUT);
        SetLength(y2, SLOTS * NOUT);
        for s := 0 to SLOTS - 1 do
        begin
          xr := 0;
          if perSlot then
            xr := s;
          ref.Linear(@x[xr * nin], 1, nin, InferWeight(@wd[IDS[s] * NOUT * nin],
            idtF32), NOUT, nil, @y1[s * NOUT]);
        end;
        CheckUtf8(gpu.GpuLinearIds(kind, nin), '% x % by ids',
          [INFER_DTYPE_NAME[kind], nin]);
        xb := nil;
        yb := nil;
        ib := nil;
        gpu.GpuLock;
        try
          xb := TInferCuBuffer.Create(gpu.Cuda, SLOTS * nin * 4);
          yb := TInferCuBuffer.Create(gpu.Cuda, SLOTS * NOUT * 4);
          ib := TInferCuBuffer.Create(gpu.Cuda, SLOTS * 4);
          xb.Upload(0, pointer(x), SLOTS * nin * 4);
          ib.Upload(0, @IDS, SLOTS * 4);
          gpu.EnqueueLinearIds(xb, nin, kind, gpu.DeviceWeight(data, kind,
            stride * EXPERTS).Ptr, stride, NOUT, ib.Ptr, SLOTS, perSlot, yb);
          yb.Download(0, pointer(y2), SLOTS * NOUT * 4);
        finally
          ib.Free;
          yb.Free;
          xb.Free;
          gpu.GpuUnLock;
        end;
        e := RelError(pointer(y2), pointer(y1), SLOTS * NOUT);
        CheckUtf8(e <= 1e-5, '% x % by ids (x per slot %): relative error %',
          [INFER_DTYPE_NAME[kind], nin, perSlot, e]);
      end;
  finally
    ref.Free;
    gpu.Free;
  end;
end;

procedure TTestInferCuda.Qwen25GgufMatchesCpu;
const
  STEPS = 16;
  FILES: array[0..1] of TFileName = (
    'qwen2.5-0.5b-instruct-q8_0.gguf', 'qwen2.5-0.5b-instruct-q4_k_m.gguf');
  KINDS: array[0..1] of RawUtf8 = ('Q8_0', 'Q4_K_M');
var
  gguf: TFileName;
  i, v, agree, f: PtrInt;
  gpu: TInferCudaBackend;
  cpu: TInferCpuFastBackend;
  model: TDecoderModel;
  vk: TQwen3Cuda;
  c1: TQwen3Context;
  c2: TQwen3CuContext;
  l1, l2: TSingleDynArray;
  cmp: TInferCompare;
  worst: double;
  id: integer;
  prompt: TIntegerDynArray;
  ref: TInferRefSet;
  refdir: TFileName;
  p: PIntegerArray;
  n: PtrInt;
begin
  if not HasDevice then
    exit;
  {$ifdef CPU32}
  AddConsole('SKIP Qwen2.5 on the device: a 32-bit process');
  exit;
  {$endif CPU32}
  refdir := FindInferRefDir('qwen25');
  if CheckFailed(refdir <> '', 'qwen25 reference found') then
    exit;
  // the chat_de prompt of the reference (43 tokens through the template)
  ref := TInferRefSet.CreateFromFolder(refdir);
  try
    p := ref.I32('chat_de.ids', n);
    SetLength(prompt, n);
    MoveFast(p^, prompt[0], n * SizeOf(integer));
  finally
    ref.Free;
  end;
  for f := 0 to high(FILES) do
  begin
    // Q4_K_M: Q5_0 for most matrices (hidden 896), Q4_K, Q6_K and Q8_0
    gguf := FindVendorModel(FILES[f]);
    if gguf = '' then
    begin
      CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
        'LAI_REQUIRE_MODEL=1 but no vendor/models/%', [FILES[f]]);
      AddConsole('SKIP Qwen2.5 on the device: no vendor/models/%', [FILES[f]]);
      continue;
    end;
    gpu := TInferCudaBackend.Create;
    cpu := TInferCpuFastBackend.Create;
    model := nil;
    vk := nil;
    c1 := nil;
    c2 := nil;
    try
      model := NewDecoderModelFromGguf(gguf, cpu);
      Check(model is TQwen2Model, 'a Qwen2 model');
      v := model.Config.VocabSize;
      vk := TQwen3Cuda.Create(model, gpu);
      c1 := model.NewContext;
      c2 := vk.NewContext;
      SetLength(l1, v);
      SetLength(l2, v);
      model.Forward(c1, pointer(prompt), length(prompt), pointer(l1));
      vk.Forward(c2, pointer(prompt), length(prompt), pointer(l2));
      worst := 1;
      agree := 0;
      for i := 0 to STEPS - 1 do
      begin
        cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
        if cmp.Cosine < worst then
          worst := cmp.Cosine;
        // InferCompare leaves a NaN or an infinity out of the cosine
        if cmp.NonFinite <> 0 then
          worst := -1;
        id := InferArgMax(pointer(l1), v);
        if InferArgMax(pointer(l2), v) = id then
          inc(agree);
        // the CPU's token to both: the comparison stays on one path
        model.Forward(c1, @id, 1, pointer(l1));
        vk.Forward(c2, @id, 1, pointer(l2));
      end;
      AddConsole('Qwen2.5-0.5B % on %: lowest logits cosine %, argmax agrees % of %',
        [KINDS[f], fCu.DeviceName, worst, agree, STEPS]);
      CheckUtf8(worst >= 0.99, 'Qwen2.5 % logits cosine %', [KINDS[f], worst]);
      CheckUtf8(agree >= STEPS - 2, 'Qwen2.5 % argmax agrees % of %',
        [KINDS[f], agree, STEPS]);
    finally
      c2.Free;
      c1.Free;
      vk.Free;
      model.Free;
      cpu.Free;
      gpu.Free;
    end;
  end;
end;



procedure TTestInferCuda.SlidingWindowMatchesCpu;
const
  NAMES: array[0..2] of RawUtf8 = ('tiny-qwen3-sliding', 'tiny-smollm3',
    'tiny-qwen3-sliding');
  /// the third: 16 query heads on one KV head (make_reference_d4b.py)
  SUBS: array[0..2] of TFileName = ('', '', 'g16');
  PROMPT = 300; // more than one chunk of the device (256)
  TOL: array[boolean] of double = (1e-4, 4e-3);
var
  dir: TFileName;
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  model: TDecoderModel;
  dev: TQwen3Cuda;
  c1, c3: TQwen3Context;
  c2: TQwen3CuContext;
  ids: TIntegerDynArray;
  l1, l2, b1, b2: TSingleDynArray;
  m, k, i, v: PtrInt;
  seed: cardinal;
  e: double;
  id: integer;
begin
  if not HasDevice then
    exit;
  SetLength(ids, PROMPT);
  seed := 13;
  for i := 0 to PROMPT - 1 do
  begin
    seed := seed * 1103515245 + 12345;
    ids[i] := 3 + (seed shr 16) mod 93;
  end;
  for m := 0 to high(NAMES) do
  begin
    dir := FindInferRefDir(NAMES[m]);
    if CheckFailed(dir <> '', Utf8ToString(NAMES[m] + ' found')) then
      continue;
    if SUBS[m] <> '' then
      dir := IncludeTrailingPathDelimiter(dir + SUBS[m]);
    gpu := TInferCudaBackend.Create;
    ref := TInferCpuBackend.Create;
    model := nil;
    dev := nil;
    c1 := nil;
    c2 := nil;
    c3 := nil;
    try
      model := NewDecoderModel(dir, ref);
      v := model.Config.VocabSize;
      dev := TQwen3Cuda.Create(model, gpu);
      c1 := model.NewContext;
      c1.FullCache := true; // Truncate(PROMPT) after the decode steps
      model.Forward(c1, pointer(ids), PROMPT, nil);
      SetLength(l1, PROMPT * v);
      SetLength(l2, PROMPT * v);
      c3 := model.NewContext;
      model.Forward(c3, pointer(ids), PROMPT, pointer(l1), true);
      FreeAndNil(c3);
      // k bit 0: one range or a range per key; bit 1: the cache in half floats
      for k := 0 to 3 do
      begin
        dev.KvF16 := k >= 2;
        dev.SplitKeys := k and 1;
        FreeAndNil(c2);
        c2 := dev.NewContext;
        dev.Forward(c2, pointer(ids), PROMPT, pointer(l2), true);
        e := RelError(pointer(l2), pointer(l1), PROMPT * v);
        CheckUtf8(e <= TOL[k >= 2], '% KvF16=% SplitKeys=% prompt: relative error %',
          [NAMES[m], k >= 2, k and 1, e]);
        // decode steps (one CUDA graph each)
        SetLength(b1, v);
        SetLength(b2, v);
        for i := 0 to 3 do
        begin
          id := ids[i];
          model.Forward(c1, @id, 1, pointer(b1));
          dev.Forward(c2, @id, 1, pointer(b2));
          e := RelError(pointer(b2), pointer(b1), v);
          CheckUtf8(e <= TOL[k >= 2], '% KvF16=% SplitKeys=% decode step %: ' +
            'relative error %', [NAMES[m], k >= 2, k and 1, i, e]);
        end;
        c1.Truncate(PROMPT);
        // back to 3 tokens, behind the window: the device keeps every row
        c2.Truncate(3);
        SetLength(b1, 20 * v);
        SetLength(b2, 20 * v);
        dev.Forward(c2, @ids[100], 20, pointer(b2), true);
        c3 := model.NewContext;
        model.Forward(c3, pointer(ids), 3, nil);
        model.Forward(c3, @ids[100], 20, pointer(b1), true);
        FreeAndNil(c3);
        e := RelError(pointer(b2), pointer(b1), 20 * v);
        CheckUtf8(e <= TOL[k >= 2], '% KvF16=% SplitKeys=% branch at 3: ' +
          'relative error %', [NAMES[m], k >= 2, k and 1, e]);
      end;
    finally
      c3.Free;
      c2.Free;
      c1.Free;
      dev.Free;
      model.Free;
      ref.Free;
      gpu.Free;
    end;
  end;
end;


procedure TTestInferCuda.RouteMatchesHost;
const
  // experts, chosen, renormed
  CASES: array[0..3, 0..2] of integer = ((128, 8, 1), (128, 8, 0), (8, 2, 1), (64, 4, 1));
  ROWS = 24;
var
  dir: TFileName;
  ref: TInferCpuBackend;
  model: TDecoderModel;
  ops: TInferCuModule;
  kRoute: TCuHandle;
  bl, bi, bw: TInferCuBuffer;
  logits, w, ww, mw: TSingleDynArray;
  ids, wids, mids: TIntegerDynArray;
  c, r, j, ne, k, bad: PtrInt;
  seed: cardinal;
  pl, pid, pw: TCuDevicePtr;
  experts, used, norm: cardinal;
  worst: double;
begin
  if not HasDevice then
    exit;
  // the reference is the model's Route (tiny-qwen3moe: 2 of 8, renormed)
  dir := FindInferRefDir('tiny-qwen3moe');
  if CheckFailed(dir <> '', 'tiny-qwen3moe found') then
    exit;
  ref := TInferCpuBackend.Create;
  model := nil;
  ops := nil;
  bl := nil;
  bi := nil;
  bw := nil;
  try
    model := TQwen3MoeModel.Create(dir, ref);
    SetLength(logits, 128);
    SetLength(ids, 8);
    SetLength(w, 8);
    SetLength(mids, 8);
    SetLength(mw, 8);
    seed := 5;
    bad := 0;
    for r := 0 to 99 do
    begin
      for j := 0 to 7 do
      begin
        seed := seed * 1103515245 + 12345;
        logits[j] := ((seed shr 9) and 1023) / 128 - 4;
      end;
      if r mod 10 = 0 then
        logits[3] := logits[5]; // a tie
      RouteRef(pointer(logits), 8, 2, true, pointer(ids), pointer(w));
      model.Route(pointer(logits), pointer(mids), pointer(mw));
      // the model's order is by logit: compare as sets with their weights
      if mids[0] = mids[1] then
        inc(bad);
      for j := 0 to 1 do
        if not (((mids[j] = ids[0]) and (PCardinal(@mw[j])^ = PCardinal(@w[0])^)) or
                ((mids[j] = ids[1]) and (PCardinal(@mw[j])^ = PCardinal(@w[1])^))) then
          inc(bad);
    end;
    CheckEqual(bad, 0, 'the reference is TDecoderModel.Route');
    // the route kernel against the reference
    ops := TInferCuModule.Create(fCu, @PTX_OPS);
    kRoute := ops.Kernel('route');
    bl := TInferCuBuffer.Create(fCu, 128 * 4);
    bi := TInferCuBuffer.Create(fCu, 64 * 4);
    bw := TInferCuBuffer.Create(fCu, 64 * 4);
    SetLength(wids, 64);
    SetLength(ww, 64);
    for c := 0 to high(CASES) do
    begin
      ne := CASES[c, 0];
      k := CASES[c, 1];
      bad := 0;
      worst := 0;
      for r := 0 to ROWS - 1 do
      begin
        for j := 0 to ne - 1 do
        begin
          seed := seed * 1103515245 + 12345;
          logits[j] := ((seed shr 9) and 4095) / 256 - 8;
        end;
        case r mod 4 of
          1: // the k-th and the (k+1)-th equal: the lower id wins
            for j := 0 to ne - 1 do
              logits[j] := (j mod 5) * 0.5;
          2: // every logit the same
            for j := 0 to ne - 1 do
              logits[j] := 1.25;
          3: // large and very negative values
            begin
              logits[0] := 80;
              logits[ne - 1] := -80;
            end;
        end;
        RouteRef(pointer(logits), ne, k, CASES[c, 2] <> 0, pointer(ids), pointer(w));
        bl.Upload(0, pointer(logits), ne * 4);
        pl := bl.Ptr;
        pid := bi.Ptr;
        pw := bw.Ptr;
        experts := ne;
        used := k;
        norm := CASES[c, 2];
        fCu.Enqueue(kRoute, [@pl, @pid, @pw, @experts, @used, @norm], 1, 32);
        bi.Download(0, pointer(wids), k * 4);
        bw.Download(0, pointer(ww), k * 4);
        for j := 0 to k - 1 do
        begin
          if wids[j] <> ids[j] then
            inc(bad);
          // a NaN would pass a plain maximum: counted as a failure
          if not (Abs(ww[j] - w[j]) <= 1) then
            worst := 1e30
          else if Abs(ww[j] - w[j]) > worst then
            worst := Abs(ww[j] - w[j]);
        end;
      end;
      CheckUtf8((bad = 0) and (worst <= 1e-7), 'route % of % (renormed %): % ids ' +
        'otherwise, weights within %', [k, ne, CASES[c, 2], bad, worst]);
    end;
  finally
    bw.Free;
    bi.Free;
    bl.Free;
    ops.Free;
    model.Free;
    ref.Free;
  end;
end;


procedure TTestInferCuda.EmbeddingRowsOnDevice;
var
  dir: TFileName;
  ref: TInferRefSet;
  gpu: TInferCudaBackend;
  cpu: TInferCpuBackend;
  model: TQwen3Model;
  dev: TQwen3Cuda;
  c1, c2: TQwen3Context;
  ids: PIntegerArray;
  n, h, vocab, cut1, cut2, i: PtrInt;
  rows, all, mine: TSingleDynArray;
  long: TIntegerDynArray;
  seed: cardinal;
begin
  if not HasDevice then
    exit;
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  gpu := TInferCudaBackend.Create;
  cpu := TInferCpuBackend.Create;
  model := nil;
  dev := nil;
  c1 := nil;
  c2 := nil;
  try
    model := TQwen3Model.Create(dir, cpu);
    dev := TQwen3Cuda.Create(model, gpu);
    ids := ref.I32('case2.ids', n);
    h := model.Config.Hidden;
    vocab := model.Config.VocabSize;
    SetLength(all, n * vocab);
    SetLength(mine, n * vocab);
    SetLength(rows, n * h);
    model.Embed(ids, n, pointer(rows));
    c1 := dev.NewContext;
    c2 := dev.NewContext;
    // on one device the rows of Embed give the bits of their ids - also
    // through TDecoderModel.ForwardEmbeds, which hands the context over
    dev.Forward(c1, ids, n, pointer(all), true);
    model.ForwardEmbeds(c2, pointer(rows), n, pointer(mine), true);
    Check(CompareMem(pointer(all), pointer(mine), n * vocab * SizeOf(single)),
      'rows on the device give the bits of their ids');
    Check((c2.Length = n) and (c2.Id[0] = -1), 'no id where a row was fed');
    // the same chunks with ids, and with rows in the middle
    cut1 := 3;
    cut2 := n - 4;
    c1.Reset;
    c2.Reset;
    dev.Forward(c1, ids, cut1, @all[0], true);
    dev.Forward(c1, @ids^[cut1], cut2 - cut1, @all[cut1 * vocab], true);
    dev.Forward(c1, @ids^[cut2], n - cut2, @all[cut2 * vocab], true);
    dev.Forward(c2, ids, cut1, @mine[0], true);
    dev.ForwardEmbeds(c2, @rows[cut1 * h], cut2 - cut1, @mine[cut1 * vocab], true);
    dev.Forward(c2, @ids^[cut2], n - cut2, @mine[cut2 * vocab], true);
    Check(CompareMem(pointer(all), pointer(mine), n * vocab * SizeOf(single)),
      'ids and rows mixed over chunks');
    CheckEqual(dev.Reuse(c2, ids, n), cut1, 'Reuse stops at the first row');
    // 300 rows in one call: two chunks of the device, the rows of the second
    // taken from the right offset
    SetLength(long, 300);
    seed := 11;
    for i := 0 to high(long) do
    begin
      seed := seed * 1103515245 + 12345;
      long[i] := 3 + (seed shr 16) mod 93;
    end;
    n := length(long);
    SetLength(all, n * vocab);
    SetLength(mine, n * vocab);
    SetLength(rows, n * h);
    model.Embed(pointer(long), n, pointer(rows));
    c1.Reset;
    c2.Reset;
    dev.Forward(c1, pointer(long), n, pointer(all), true);
    dev.ForwardEmbeds(c2, pointer(rows), n, pointer(mine), true);
    Check(CompareMem(pointer(all), pointer(mine), n * vocab * SizeOf(single)),
      '300 rows over two chunks');
  finally
    c2.Free;
    c1.Free;
    dev.Free;
    model.Free;
    cpu.Free;
    gpu.Free;
    ref.Free;
  end;
end;

procedure TTestInferCuda.Qwen3TtsOnDevice;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  gpu: TInferCudaBackend;
  cpu: TInferCpuFastBackend;
  model: TQwen3Model;
  dev: TQwen3Cuda;
  ctx: TQwen3Context;
  logits: TSingleDynArray;
  x: PSingleArray;
  n, m, h, vocab: PtrInt;
  cmp: TInferCompare;
begin
  if not HasDevice then
    exit;
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3-TTS on the device: a 32-bit process');
  exit;
  {$endif CPU32}
  dir := TtsFolder;
  if dir = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/%', [TTS_MODEL]);
    AddConsole('SKIP Qwen3-TTS on the device: no vendor/models/%', [TTS_MODEL]);
    exit;
  end;
  refdir := FindInferRefDir('qwen3-tts-lm');
  if CheckFailed(refdir <> '', 'qwen3-tts-lm found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  gpu := TInferCudaBackend.Create;
  cpu := TInferCpuFastBackend.Create;
  model := nil;
  dev := nil;
  ctx := nil;
  try
    // the talker: the last logits of its 22 rows
    model := NewTtsTalker(dir, cpu);
    dev := TQwen3Cuda.Create(model, gpu);
    dev.KvF16 := false; // against float32: the cache in floats too
    h := model.Config.Hidden;
    x := ref.F32('talker.embed', m);
    n := m div h;
    SetLength(logits, model.Config.VocabSize);
    ctx := dev.NewContext;
    dev.ForwardEmbeds(ctx, x, n, pointer(logits));
    FreeAndNil(ctx);
    cmp := InferCompare(pointer(logits), ref.F32('talker.logits_last', m), m, 1e-3, 1e-3);
    CheckUtf8(InferCompareOk(cmp), 'talker logits on %: %', [fCu.DeviceName,
      InferCompareText(cmp)]);
    AddConsole(FormatString('Qwen3-TTS talker on %: %', [fCu.DeviceName, InferCompareText(cmp)]));
    FreeAndNil(dev);
    FreeAndNil(model);
    // the code predictor: head 0 on row 1 is the logits of code group 0
    model := NewTtsPredictor(dir, cpu);
    dev := TQwen3Cuda.Create(model, gpu);
    dev.KvF16 := false;
    vocab := model.Config.VocabSize;
    x := ref.F32('predictor.embed', m);
    n := m div h;
    SetLength(logits, n * vocab);
    ctx := dev.NewContext;
    dev.ForwardEmbeds(ctx, x, n, pointer(logits), true);
    FreeAndNil(ctx);
    cmp := InferCompare(@logits[vocab], ref.F32('predictor.logits', m), vocab, 1e-3, 1e-3);
    CheckUtf8(InferCompareOk(cmp), 'predictor group 0 on %: %', [fCu.DeviceName,
      InferCompareText(cmp)]);
    CheckEqual(InferArgMax(@logits[vocab], vocab), ref.I32('predictor.codes', m)^[1],
      'the code of group 0');
  finally
    ctx.Free;
    dev.Free;
    model.Free;
    cpu.Free;
    gpu.Free;
    ref.Free;
  end;
end;

procedure TTestInferCuda.ForgetWeightsServesNew;
const
  NIN = 64;
  NOUT = 8;
var
  gpu: TInferCudaBackend;
  cpu: TInferCpuBackend;
  model: TQwen3Model;
  dev: TQwen3Cuda;
  ctx: TQwen3Context;
  dir: TFileName;
  x, w, w2, wb, y1, y2, yb, ydev: TSingleDynArray;
  epoch: integer;
  id: integer;
begin
  if not HasDevice then
    exit;
  gpu := TInferCudaBackend.Create;
  cpu := TInferCpuBackend.Create;
  model := nil;
  dev := nil;
  ctx := nil;
  try
    Fill(x, NIN, 91, 2);
    Fill(w, NOUT * NIN, 92, 0.5);
    Fill(w2, NOUT * NIN, 93, 0.5);
    Fill(wb, NOUT * NIN, 94, 0.5);
    SetLength(yb, NOUT);
    SetLength(y1, NOUT);
    SetLength(y2, NOUT);
    SetLength(ydev, NOUT);
    gpu.Linear(pointer(x), 1, NIN, InferWeight(pointer(w), idtF32), NOUT, nil,
      pointer(ydev));
    // wb cached after w: forgetting w moves it down in the cache
    gpu.Linear(pointer(x), 1, NIN, InferWeight(pointer(wb), idtF32), NOUT, nil,
      pointer(yb));
    // new weights at the same address, of the same type and size: the stale
    // copy (the trap ForgetWeights is for)
    MoveFast(w2[0], w[0], NOUT * NIN * SizeOf(single));
    cpu.Linear(pointer(x), 1, NIN, InferWeight(pointer(w), idtF32), NOUT, nil,
      pointer(y2));
    gpu.Linear(pointer(x), 1, NIN, InferWeight(pointer(w), idtF32), NOUT, nil,
      pointer(y1));
    Check(CompareMem(pointer(y1), pointer(ydev), NOUT * SizeOf(single)),
      'without ForgetWeights the old copy');
    // another range: kept, the epoch stays
    epoch := gpu.WeightEpoch;
    gpu.ForgetWeights(pointer(x), @x[NIN]);
    CheckEqual(gpu.WeightEpoch, epoch, 'nothing forgotten');
    // the range of the weights, a byte of it is enough
    gpu.ForgetWeights(@w[NOUT * NIN - 1], @w[NOUT * NIN]);
    Check(gpu.WeightEpoch <> epoch, 'the epoch moves on');
    gpu.Linear(pointer(x), 1, NIN, InferWeight(pointer(w), idtF32), NOUT, nil,
      pointer(y1));
    CheckUtf8(RelError(pointer(y1), pointer(y2), NOUT) <= 1e-5,
      'the new weights after ForgetWeights: %', [RelError(pointer(y1), pointer(y2), NOUT)]);
    // wb stayed cached: changed in place, still its old copy
    MoveFast(w2[0], wb[0], NOUT * NIN * SizeOf(single));
    gpu.Linear(pointer(x), 1, NIN, InferWeight(pointer(wb), idtF32), NOUT, nil,
      pointer(y1));
    Check(CompareMem(pointer(y1), pointer(yb), NOUT * SizeOf(single)),
      'the other weight kept');
    // a device forgets the weights of its model when freed (the model may
    // compute with another backend)
    dir := FindInferRefDir('tiny-qwen3');
    if CheckFailed(dir <> '', 'tiny-qwen3 found') then
      exit;
    model := TQwen3Model.Create(dir, cpu);
    dev := TQwen3Cuda.Create(model, gpu);
    ctx := dev.NewContext;
    id := 5;
    dev.Forward(ctx, @id, 1, nil);
    FreeAndNil(ctx);
    epoch := gpu.WeightEpoch;
    FreeAndNil(dev);
    Check(gpu.WeightEpoch <> epoch, 'the device forgot the weights of its model');
    FreeAndNil(model);
    // a model computing with the GPU backend forgets its weights when freed
    model := TQwen3Model.Create(dir, gpu);
    ctx := model.NewContext;
    model.Forward(ctx, @id, 1, nil); // its matrices through gpu.Linear
    FreeAndNil(ctx);
    epoch := gpu.WeightEpoch;
    FreeAndNil(model);
    Check(gpu.WeightEpoch <> epoch, 'the model forgot its weights');
  finally
    ctx.Free;
    dev.Free;
    model.Free;
    cpu.Free;
    gpu.Free;
  end;
end;

end.
