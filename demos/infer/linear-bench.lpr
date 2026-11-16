// LandrixAI - speed of the one-row matrix products of a GGUF model on the GPU
// (backlog B20: the decode step of a large model is mostly these products)
// - uploads every 2-D weight the device multiplies (not the token embedding,
//   nor a convolution kernel of fewer than MIN_IN inputs), groups the matrices
//   by type and shape, and runs each group as a decode step does: one row
//   through every matrix of the group, several rounds - CUDA as one graph,
//   Vulkan as one batch with a barrier between dispatches
// - reports per group the time per matrix and the bytes per second, then the
//   sum of one pass over all matrices: the floor of a decode step - without
//   the experts of a mixture (3-D tensors, read per chosen expert); a group
//   smaller than the L2 cache of the GPU is read from there after the first
//   round, unlike in a decode step (on an RTX 5090: the 48-row Q8_0 matrices
//   of Qwen3.8-27B, which are bound by latency anyway)
// - CUDA: the wall clock around launching the graph and waiting for it (an
//   event per matrix would cost more than the small ones take); Vulkan: the
//   GPU time of its profile (a timestamp after each dispatch, about 1 % of a
//   decode step) - its wall clock would count the device idling while the
//   host records the batch, which a decode step overlaps (BatchFlush); a
//   pass over every group first brings the clocks of the GPU up
// - LAI_PAIRS=<n> in the environment: from how many outputs matvec8 takes two
//   features per warp (MatvecPairsFrom of either backend; 999999: never)
//
//   usage: linear-bench file.gguf [vulkan | cuda] [rounds=5]
//   e.g.   linear-bench qwen3.8-27b-q4_k_m.gguf cuda
program linear.bench;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.loader.gguf,
  mormot.ai.infer.vulkan,
  mormot.ai.infer.backend.vulkan,
  mormot.ai.infer.cuda,
  mormot.ai.infer.backend.cuda;

type
  /// the matrices of one type and shape
  TGroup = record
    DType: TInferDType;
    InCount, OutCount: PtrInt;
    Bytes: Int64; // of one matrix
    Data: array of pointer;
    Exec: TCuHandle; // CUDA: the graph of all rounds over the group
    Ns: Int64; // time of one round over the group, the best of TIMED_RUNS
  end;
  TGroupDynArray = array of TGroup;

const
  /// timed runs per group: the fastest counts (a name of its own: Pascal
  // would take a local variable "tries" for it)
  TIMED_RUNS = 3;
  /// fewer inputs: not a matrix of a decode step (ssm_conv1d of Qwen3.5 has 4)
  MIN_IN = 256;

var
  gguf: TGgufFile;
  groups: TGroupDynArray;
  mode: RawUtf8;
  rounds: integer;

procedure Collect(aVulkan: TInferVulkanBackend; aCuda: TInferCudaBackend);
var
  i, g: PtrInt;
  t: PInferTensor;
  ok: boolean;
begin
  for i := 0 to high(gguf.Tensors) do
  begin
    t := @gguf.Tensors[i];
    if (length(t^.Shape) <> 2) or (t^.Shape[1] < MIN_IN) or
       IdemPChar(pointer(t^.Name), 'TOKEN_EMBD') then
      continue;
    if aVulkan <> nil then
      ok := aVulkan.GpuLinear(t^.DType, t^.Shape[1])
    else
      ok := aCuda.GpuLinear(t^.DType, t^.Shape[1]);
    if not ok then
      continue;
    g := 0;
    while (g < length(groups)) and ((groups[g].DType <> t^.DType) or
          (groups[g].InCount <> t^.Shape[1]) or (groups[g].OutCount <> t^.Shape[0])) do
      inc(g);
    if g = length(groups) then
    begin
      SetLength(groups, g + 1);
      groups[g].DType := t^.DType;
      groups[g].InCount := t^.Shape[1];
      groups[g].OutCount := t^.Shape[0];
      groups[g].Bytes := t^.Bytes;
    end;
    SetLength(groups[g].Data, length(groups[g].Data) + 1);
    groups[g].Data[high(groups[g].Data)] := t^.Data;
  end;
  if groups = nil then
    raise EInferError.Create('linear-bench: no matrix the device multiplies');
end;

function MaxOf(aIn: boolean): PtrInt;
var
  g: PtrInt;
begin
  result := 1;
  for g := 0 to high(groups) do
    if aIn and (groups[g].InCount > result) then
      result := groups[g].InCount
    else if not aIn and (groups[g].OutCount > result) then
      result := groups[g].OutCount;
end;

procedure FillX(var aX: TSingleDynArray);
var
  i: PtrInt;
begin
  SetLength(aX, MaxOf(true));
  for i := 0 to high(aX) do
    aX[i] := ((i * 7919) mod 2001 - 1000) / 1000;
end;

procedure RunCuda;
var
  be: TInferCudaBackend;
  x, y: TInferCuBuffer;
  host: TSingleDynArray;
  w: TInferWeight;
  g, r, i, pass, runs: PtrInt;
  t0, t1: Int64;
begin
  be := TInferCudaBackend.Create;
  if GetEnvironmentVariable('LAI_PAIRS') <> '' then
    be.MatvecPairsFrom := StrToInt(GetEnvironmentVariable('LAI_PAIRS'));
  x := nil;
  y := nil;
  try
    ConsoleWrite('cuda: % (compute %.%)', [be.Cuda.DeviceName,
      be.Cuda.ComputeMajor, be.Cuda.ComputeMinor]);
    Collect(nil, be);
    FillX(host);
    x := TInferCuBuffer.Create(be.Cuda, length(host) * 4);
    y := TInferCuBuffer.Create(be.Cuda, MaxOf(false) * 4);
    be.GpuLock;
    try
      x.Upload(0, pointer(host), length(host) * 4);
      for g := 0 to high(groups) do
        for i := 0 to high(groups[g].Data) do
          be.DeviceWeight(groups[g].Data[i], groups[g].DType, groups[g].Bytes);
      // one graph per group: every round over its matrices
      for g := 0 to high(groups) do
      begin
        w.DType := groups[g].DType;
        be.Cuda.CaptureBegin;
        try
          for r := 1 to rounds do
            for i := 0 to high(groups[g].Data) do
            begin
              w.Data := groups[g].Data[i];
              be.EnqueueLinear(x, 1, groups[g].InCount, w, groups[g].OutCount, y);
            end;
          be.Cuda.CaptureEnd(groups[g].Exec); // and runs it once
        except
          be.Cuda.CaptureAbort;
          raise;
        end;
        be.Cuda.Synchronize;
      end;
      // pass 0 warms the GPU up, pass 1 counts
      for pass := 0 to 1 do
        for g := 0 to high(groups) do
        begin
          runs := 1;
          if pass = 1 then
            runs := TIMED_RUNS;
          groups[g].Ns := high(Int64);
          for i := 1 to runs do
          begin
            QueryPerformanceMicroSeconds(t0);
            be.Cuda.GraphLaunch(groups[g].Exec);
            be.Cuda.Synchronize;
            QueryPerformanceMicroSeconds(t1);
            if (t1 - t0) * 1000 < groups[g].Ns then
              groups[g].Ns := (t1 - t0) * 1000;
          end;
        end;
    finally
      for g := 0 to high(groups) do
        be.Cuda.GraphFree(groups[g].Exec);
      be.GpuUnLock;
    end;
  finally
    y.Free;
    x.Free;
    be.Free;
  end;
  for g := 0 to high(groups) do
    groups[g].Ns := groups[g].Ns div rounds;
end;

procedure RunVulkan;
var
  be: TInferVulkanBackend;
  vk: TInferVulkan;
  x, y, temp: TInferVkBuffer;
  host: TSingleDynArray;
  w: TInferWeight;
  g, r, i, pass, runs, part, parts: PtrInt;
  ns: Int64;
begin
  be := TInferVulkanBackend.Create;
  if GetEnvironmentVariable('LAI_PAIRS') <> '' then
    be.MatvecPairsFrom := StrToInt(GetEnvironmentVariable('LAI_PAIRS'));
  x := nil;
  y := nil;
  temp := nil;
  try
    vk := be.Vulkan;
    ConsoleWrite('vulkan: % (type %)', [vk.DeviceName, vk.DeviceType]);
    if not vk.HasTimestamps then
      raise EInferError.Create('linear-bench: no timestamps on this device');
    Collect(be, nil);
    FillX(host);
    x := TInferVkBuffer.Create(vk, length(host) * 4, true);
    y := TInferVkBuffer.Create(vk, MaxOf(false) * 4, true);
    be.GpuLock;
    try
      vk.Upload(x, 0, pointer(host), length(host) * 4);
      part := 1;
      for g := 0 to high(groups) do
      begin
        w.DType := groups[g].DType;
        for i := 0 to high(groups[g].Data) do
        begin
          w.Data := groups[g].Data[i];
          be.UploadLinear(w, groups[g].InCount, groups[g].OutCount);
        end;
        r := be.LinearPart(w.DType, groups[g].InCount, groups[g].OutCount);
        if r > part then
          part := r;
      end;
      // a matrix larger than one binding goes part by part through temp
      temp := TInferVkBuffer.Create(vk, part * 4, true);
      vk.Profile := true;
      // pass 0 warms the GPU up, pass 1 counts
      for pass := 0 to 1 do
        for g := 0 to high(groups) do
        begin
          w.DType := groups[g].DType;
          // the dispatches of one matrix: one per part, then a copy of each
          part := be.LinearPart(w.DType, groups[g].InCount, groups[g].OutCount);
          parts := (groups[g].OutCount + part - 1) div part;
          if parts > 1 then
            parts := parts * 2;
          runs := 1;
          if pass = 1 then
            runs := TIMED_RUNS;
          groups[g].Ns := high(Int64);
          for i := 1 to runs do
          begin
            vk.ProfileReset;
            vk.BatchBegin;
            try
              for r := 0 to rounds * length(groups[g].Data) - 1 do
              begin
                if vk.BatchRoom < parts then
                begin
                  vk.BatchSubmit;
                  vk.BatchBegin;
                end;
                w.Data := groups[g].Data[r mod length(groups[g].Data)];
                be.RecordLinear(x, 1, groups[g].InCount, w, groups[g].OutCount, y,
                  true, temp);
              end;
              vk.BatchSubmit;
            except
              vk.BatchAbort;
              raise;
            end;
            // a portability device may turn out to stamp whole batches only
            // (TInferVulkan.HasTimestamps false from then on)
            ns := vk.ProfileNs;
            if not vk.HasTimestamps or (ns <= 0) then
              raise EInferError.Create('linear-bench: no timestamps on this device');
            if ns < groups[g].Ns then
              groups[g].Ns := ns;
          end;
        end;
    finally
      be.GpuUnLock;
    end;
  finally
    temp.Free;
    y.Free;
    x.Free;
    be.Free;
  end;
  for g := 0 to high(groups) do
    groups[g].Ns := groups[g].Ns div rounds;
end;

procedure Report;
var
  g: PtrInt;
  total, bytes: Int64;
begin
  total := 0;
  bytes := 0;
  for g := 0 to high(groups) do
  begin
    ConsoleWrite('% % x % (% x %): % us per matrix, % GB/s', [
      INFER_DTYPE_NAME[groups[g].DType], groups[g].OutCount, groups[g].InCount,
      length(groups[g].Data), KB(groups[g].Bytes), groups[g].Ns / 1000 /
      length(groups[g].Data), groups[g].Bytes * length(groups[g].Data) /
      groups[g].Ns]);
    inc(total, groups[g].Ns);
    inc(bytes, groups[g].Bytes * length(groups[g].Data));
  end;
  ConsoleWrite('all matrices once: % ms, % GB, % GB/s', [total / 1000000,
    bytes / 1E9, bytes / total]);
end;

begin
  if ParamCount < 1 then
  begin
    ConsoleWrite('usage: linear-bench file.gguf [vulkan | cuda] [rounds=5]');
    exit;
  end;
  mode := 'vulkan';
  if ParamCount >= 2 then
    mode := LowerCase(StringToUtf8(ParamStr(2)));
  rounds := 5;
  if ParamCount >= 3 then
    rounds := StrToIntDef(ParamStr(3), 5);
  if rounds < 1 then
    rounds := 1;
  try
    gguf := TGgufFile.Create(ParamStr(1));
    try
      if mode = 'cuda' then
        RunCuda
      else
        RunVulkan;
      Report;
    finally
      gguf.Free;
    end;
  except
    on E: Exception do
      ConsoleWrite('% %', [E.ClassType, E.Message]);
  end;
end.
