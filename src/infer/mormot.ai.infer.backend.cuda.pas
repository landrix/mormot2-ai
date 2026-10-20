/// native inference engine - the CUDA backend: Linear on an NVIDIA GPU
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - backlog B3, slice C2: TInferCudaBackend is the fast CPU backend whose
//   Linear runs on a CUDA device for F32, Q8_0, Q5_0, Q4_K, Q6_K and MXFP4 weights;
//   every other operation, every other weight type and every call while the
//   device is busy stay on the CPU (as TInferVulkanBackend)
// - weights and biases are uploaded once, on first use, and kept on the
//   device, keyed by their address, type and size: they must neither change
//   nor be freed while the backend lives, unless their owner forgets them
//   first (ForgetWeights for an address range - a TDecoderModel does it for
//   its own in its destructor; a temporary bias would be served stale from
//   its reused address) - ClearWeights forgets them all
// - activations go up and results come down on every call, and each call
//   waits for the device: a correctness step, the forward pass on the device
//   comes with slice C4
// - arithmetic: the weights dequantized in float, as InferRowToF32 does, times
//   F32 activations, summed in another order than the CPU - a tolerance
unit mormot.ai.infer.backend.cuda;

{
  *****************************************************************************

    - TInferCudaBackend: TInferCpuFastBackend with Linear on a CUDA device

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

{$ifndef FPC}{$ifdef CPUX64}
  {$EXCESSPRECISION OFF} // single expressions in single, as FPC computes them
{$endif}{$endif}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.cuda;

type
  /// the fast CPU backend with Linear on a CUDA device
  TInferCudaBackend = class(TInferCpuFastBackend)
  protected
    fCu: TInferCuda;
    fLinearModule: TInferCuModule;
    fLinear, fLinear8, fGemm8, fLinear8Ids: TCuHandle;
    fFastLinear: boolean;
    fGpu: TLightLock;
    // the weight cache: parallel arrays, searched linearly (a model has a few
    // hundred matrices)
    fWeightData: array of pointer;
    fWeightType: array of TInferDType;
    fWeightBytes: TInt64DynArray;
    fWeightBuffer: array of TInferCuBuffer;
    fWeightCount: PtrInt;
    fX, fY: TInferCuBuffer;
    fGpuLinears: Int64;
    fWeightEpoch: integer;
    function Weight(aData: pointer; aType: TInferDType;
      aBytes: Int64): TInferCuBuffer;
    function Scratch(var aBuffer: TInferCuBuffer; aBytes: Int64): TInferCuBuffer;
    // the kernel and grid of a Linear: gemm8 for GEMM_ROWS rows or more,
    // linear8 when the inputs allow, linear otherwise
    procedure LinearShape(aType: TInferDType; aIn, aRows, aOut: PtrInt;
      out aKernel: TCuHandle; out aX, aY, aBlock: cardinal);
  public
    /// the CPU part as TInferCpuFastBackend.Create, plus the CUDA device
    // (TInferCuda.Create: raises EInferCuda without one)
    constructor Create(aThreads: integer = 0); reintroduce;
    destructor Destroy; override;
    /// y = x * w^T + bias on the device for F32, Q8_0, Q5_0, Q4_K and Q6_K
    // weights, on the CPU otherwise (and while another thread uses the device)
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); override;
    /// release every weight kept on the device
    procedure ClearWeights;
    /// release the device copy of every weight whose host memory overlaps
    // aFrom .. aTo - 1 (see TInferBackend.ForgetWeights); WeightEpoch moves
    // on when one went, so a holder of uploaded weights looks them up again
    procedure ForgetWeights(aFrom, aTo: pointer); override;
    /// true when Linear runs weights of aType with aIn inputs on the device
    function GpuLinear(aType: TInferDType; aIn: PtrInt): boolean;
    /// take the device for a sequence of enqueued operations (a forward pass
    // on the GPU): Linear computes on the CPU meanwhile
    procedure GpuLock;
    procedure GpuUnLock;
    /// the device copy of a weight (uploaded on first use) - GpuLock held
    function DeviceWeight(aData: pointer; aType: TInferDType;
      aBytes: Int64): TInferCuBuffer;
    /// enqueue y = x * w^T (+ aBias), the weights from the cache - GpuLock
    // held; x and y are [aRows, aIn] and [aRows, aOut] at offset 0; no wait
    // - aBias ([aOut] floats) from the cache too: upload it beforehand
    // (DeviceWeight) when the call is captured into a graph
    procedure EnqueueLinear(x: TInferCuBuffer; aRows, aIn: PtrInt;
      const w: TInferWeight; aOut: PtrInt; y: TInferCuBuffer;
      aBias: PSingleArray = nil);
    /// the same with weights of aType already on the device at aWeight - e.g.
    // one expert inside the buffer of all experts of a layer (DeviceWeight of
    // the whole tensor, then an offset); aRows up to 65535; aBias: a device
    // address of [aOut] floats, or 0
    procedure EnqueueLinearAt(x: TInferCuBuffer; aRows, aIn: PtrInt;
      aType: TInferDType; aWeight: TCuDevicePtr; aOut: PtrInt; y: TInferCuBuffer;
      aBias: TCuDevicePtr = 0);
    /// true when EnqueueLinearIds runs weights of aType with aIn inputs (the
    // form of linear8: any block type, F32 with aIn a multiple of 8)
    function GpuLinearIds(aType: TInferDType; aIn: PtrInt): boolean;
    /// enqueue one token row through aSlots experts chosen on the device:
    // slot s reads its expert id from aIds[s] (int32 at a device address), the
    // expert's [aOut, aIn] matrix of aType at aWeight + id * aStride bytes (the
    // experts of a layer in one buffer); x is row s of x when aXPerSlot, else
    // row 0; y receives row s - linear8ids, the arithmetic of linear8
    // - aBias: a device address of [experts, aOut] floats (row id added), or 0
    procedure EnqueueLinearIds(x: TInferCuBuffer; aIn: PtrInt; aType: TInferDType;
      aWeight: TCuDevicePtr; aStride: Int64; aOut: PtrInt; aIds: TCuDevicePtr;
      aSlots: PtrInt; aXPerSlot: boolean; y: TInferCuBuffer;
      aBias: TCuDevicePtr = 0);
    /// the device context
    property Cuda: TInferCuda read fCu;
    /// how many Linear calls ran on the device
    property GpuLinears: Int64 read fGpuLinears;
    /// incremented by ClearWeights: a holder of uploaded weights uploads again
    property WeightEpoch: integer read fWeightEpoch;
    /// the fast kernels of tools/cuda/linear.cu (linear8, gemm8) - default
    // true; false: the plain linear kernel only (the tests compare both)
    property FastLinear: boolean read fFastLinear write fFastLinear;
  end;


implementation

const
  /// the dtype parameter of tools/cuda/linear.cu
  // - Unknown, F32, F16, BF16, I32, I64, U8, I8, Q8_0, Q4_0, Q4_K, Q6_K, Q5_0,
  // MXFP4;
  // 0 = CPU
  LINEAR_DTYPE: array[TInferDType] of cardinal = (
    0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 3, 4, 5, 6);

  /// the threads of a block of linear and linear8 in tools/cuda/linear.cu
  LINEAR_BLOCK = 128;
  /// the threads of a gemm8 block (a tile of 64 rows x 64 features)
  GEMM_BLOCK = 256;
  /// from this many rows on a Linear runs gemm8 (as GEMM_ROWS on Vulkan)
  GEMM_ROWS = 16;


{ TInferCudaBackend }

constructor TInferCudaBackend.Create(aThreads: integer);
begin
  inherited Create(aThreads);
  fCu := TInferCuda.Create;
  fLinearModule := TInferCuModule.Create(fCu, @PTX_LINEAR);
  fLinear := fLinearModule.Kernel('linear');
  fLinear8 := fLinearModule.Kernel('linear8');
  fGemm8 := fLinearModule.Kernel('gemm8');
  fLinear8Ids := fLinearModule.Kernel('linear8ids');
  fFastLinear := true;
end;

procedure TInferCudaBackend.LinearShape(aType: TInferDType; aIn, aRows,
  aOut: PtrInt; out aKernel: TCuHandle; out aX, aY, aBlock: cardinal);
begin
  if fFastLinear and (aRows >= GEMM_ROWS) and (aIn and 31 = 0) then
  begin
    // 64 rows x 64 features per block
    aKernel := fGemm8;
    aX := (aOut + 63) shr 6;
    aY := (aRows + 63) shr 6;
    aBlock := GEMM_BLOCK;
  end
  else if fFastLinear and ((aType <> idtF32) or (aIn and 7 = 0)) then
  begin
    // 8 weights per step: every block type qualifies, F32 when aIn allows;
    // 4 features (one per warp) x 8 rows per block
    aKernel := fLinear8;
    aX := (aOut + 3) shr 2;
    aY := (aRows + 7) shr 3;
    aBlock := LINEAR_BLOCK;
  end
  else
  begin
    aKernel := fLinear;
    aX := aOut;
    aY := aRows;
    aBlock := LINEAR_BLOCK;
  end;
end;

destructor TInferCudaBackend.Destroy;
begin
  if fCu <> nil then
  begin
    ClearWeights;
    FreeAndNil(fX);
    FreeAndNil(fY);
    FreeAndNil(fLinearModule);
    FreeAndNil(fCu); // last: the buffers and the module live in its context
  end;
  inherited Destroy;
end;

procedure TInferCudaBackend.ClearWeights;
var
  i: PtrInt;
begin
  fGpu.Lock;
  try
    for i := 0 to fWeightCount - 1 do
      FreeAndNil(fWeightBuffer[i]);
    fWeightCount := 0;
    inc(fWeightEpoch);
  finally
    fGpu.UnLock;
  end;
end;

procedure TInferCudaBackend.ForgetWeights(aFrom, aTo: pointer);
var
  i, n: PtrInt;
begin
  if PtrUInt(aTo) <= PtrUInt(aFrom) then
    exit;
  fGpu.Lock;
  try
    n := 0;
    for i := 0 to fWeightCount - 1 do
      if (PtrUInt(fWeightData[i]) < PtrUInt(aTo)) and
         (PtrUInt(fWeightData[i]) + PtrUInt(fWeightBytes[i]) > PtrUInt(aFrom)) then
        FreeAndNil(fWeightBuffer[i])
      else
      begin
        // kept: moved down over the forgotten ones
        fWeightData[n] := fWeightData[i];
        fWeightType[n] := fWeightType[i];
        fWeightBytes[n] := fWeightBytes[i];
        fWeightBuffer[n] := fWeightBuffer[i];
        inc(n);
      end;
    if n <> fWeightCount then
    begin
      for i := n to fWeightCount - 1 do
        fWeightBuffer[i] := nil;
      fWeightCount := n;
      inc(fWeightEpoch);
    end;
  finally
    fGpu.UnLock;
  end;
end;

function TInferCudaBackend.GpuLinear(aType: TInferDType; aIn: PtrInt): boolean;
begin
  result := (LINEAR_DTYPE[aType] <> 0) and (aIn > 0) and
    (aIn mod INFER_DTYPE_BLOCK[aType] = 0);
end;

procedure TInferCudaBackend.GpuLock;
begin
  fGpu.Lock;
end;

procedure TInferCudaBackend.GpuUnLock;
begin
  fGpu.UnLock;
end;

function TInferCudaBackend.DeviceWeight(aData: pointer; aType: TInferDType;
  aBytes: Int64): TInferCuBuffer;
begin
  result := Weight(aData, aType, aBytes);
end;

procedure TInferCudaBackend.EnqueueLinear(x: TInferCuBuffer; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; y: TInferCuBuffer; aBias: PSingleArray);
var
  pb: TCuDevicePtr;
begin
  if not GpuLinear(w.DType, aIn) or (aOut < 1) or (Int64(aOut) > high(integer)) or
     (Int64(aIn) > high(integer)) then
    raise EInferCuda.CreateUtf8('%.EnqueueLinear: % rows of % % into %',
      [self, aRows, aIn, INFER_DTYPE_NAME[w.DType], aOut]);
  pb := 0;
  if aBias <> nil then
    pb := Weight(aBias, idtF32, Int64(aOut) * 4).Ptr;
  EnqueueLinearAt(x, aRows, aIn, w.DType, Weight(w.Data, w.DType, Int64(aOut) *
    (aIn div INFER_DTYPE_BLOCK[w.DType]) * INFER_DTYPE_SIZE[w.DType]).Ptr, aOut, y, pb);
end;

procedure TInferCudaBackend.EnqueueLinearAt(x: TInferCuBuffer; aRows, aIn: PtrInt;
  aType: TInferDType; aWeight: TCuDevicePtr; aOut: PtrInt; y: TInferCuBuffer;
  aBias: TCuDevicePtr);
var
  pw, px, pb, py: TCuDevicePtr;
  rows, nin, nout, dtype, hasBias, gx, gy, block: cardinal;
  kernel: TCuHandle;
begin
  if not GpuLinear(aType, aIn) or (aRows < 1) or (aRows > 65535) or
     (aOut < 1) or (Int64(aOut) > high(integer)) or
     (Int64(aIn) > high(integer)) or (aWeight = 0) then
    raise EInferCuda.CreateUtf8('%.EnqueueLinearAt: % rows of % % into %',
      [self, aRows, aIn, INFER_DTYPE_NAME[aType], aOut]);
  pw := aWeight;
  px := x.Ptr;
  pb := aBias; // never read without a bias
  py := y.Ptr;
  rows := aRows;
  nin := aIn;
  nout := aOut;
  dtype := LINEAR_DTYPE[aType];
  hasBias := ord(aBias <> 0);
  LinearShape(aType, aIn, aRows, aOut, kernel, gx, gy, block);
  fCu.Enqueue(kernel, [@pw, @px, @pb, @py, @rows, @nin, @nout, @dtype,
    @hasBias], gx, block, gy);
end;

function TInferCudaBackend.GpuLinearIds(aType: TInferDType; aIn: PtrInt): boolean;
begin
  result := GpuLinear(aType, aIn) and ((aType <> idtF32) or (aIn and 7 = 0));
end;

procedure TInferCudaBackend.EnqueueLinearIds(x: TInferCuBuffer; aIn: PtrInt;
  aType: TInferDType; aWeight: TCuDevicePtr; aStride: Int64; aOut: PtrInt;
  aIds: TCuDevicePtr; aSlots: PtrInt; aXPerSlot: boolean; y: TInferCuBuffer;
  aBias: TCuDevicePtr);
var
  pw, px, pids, py, pb: TCuDevicePtr;
  stride: Int64;
  nin, nout, dtype, xslot: cardinal;
begin
  if not GpuLinearIds(aType, aIn) or (aSlots < 1) or (aSlots > 65535) or
     (aOut < 1) or (Int64(aOut) > high(integer)) or (aStride <= 0) or
     (aWeight = 0) or (aIds = 0) then
    raise EInferCuda.CreateUtf8('%.EnqueueLinearIds: % slots of % % into %',
      [self, aSlots, aIn, INFER_DTYPE_NAME[aType], aOut]);
  pw := aWeight;
  px := x.Ptr;
  pids := aIds;
  py := y.Ptr;
  stride := aStride;
  nin := aIn;
  nout := aOut;
  dtype := LINEAR_DTYPE[aType];
  xslot := ord(aXPerSlot);
  pb := aBias;
  fCu.Enqueue(fLinear8Ids, [@pw, @px, @pids, @py, @stride, @nin, @nout, @dtype,
    @xslot, @pb], (aOut + 3) shr 2, LINEAR_BLOCK, aSlots);
end;

function TInferCudaBackend.Weight(aData: pointer; aType: TInferDType;
  aBytes: Int64): TInferCuBuffer;
var
  i: PtrInt;
begin
  for i := 0 to fWeightCount - 1 do
    if (fWeightData[i] = aData) and (fWeightType[i] = aType) and
       (fWeightBytes[i] = aBytes) then
    begin
      result := fWeightBuffer[i];
      exit;
    end;
  // rounded up to whole 32-bit words: linear8 and gemm8 read the weights as
  // words, the last one of a Q8_0 or Q6_K row (34 and 210 bytes per block)
  // reaches up to 2 bytes beyond the data (both sizes are even)
  result := TInferCuBuffer.Create(fCu, (aBytes + 3) and not Int64(3));
  try
    result.Upload(0, aData, aBytes);
  except
    result.Free;
    raise;
  end;
  if fWeightCount = length(fWeightData) then
  begin
    i := NextGrow(fWeightCount);
    SetLength(fWeightData, i);
    SetLength(fWeightType, i);
    SetLength(fWeightBytes, i);
    SetLength(fWeightBuffer, i);
  end;
  fWeightData[fWeightCount] := aData;
  fWeightType[fWeightCount] := aType;
  fWeightBytes[fWeightCount] := aBytes;
  fWeightBuffer[fWeightCount] := result;
  inc(fWeightCount);
end;

function TInferCudaBackend.Scratch(var aBuffer: TInferCuBuffer;
  aBytes: Int64): TInferCuBuffer;
begin
  if (aBuffer = nil) or (aBuffer.Size < aBytes) then
  begin
    FreeAndNil(aBuffer);
    // 25 % more: a growing context does not reallocate on every token - and
    // the exact size where the device has no room for the margin
    try
      aBuffer := TInferCuBuffer.Create(fCu, aBytes + aBytes shr 2);
    except
      on EInferCuda do
        aBuffer := TInferCuBuffer.Create(fCu, aBytes);
    end;
  end;
  result := aBuffer;
end;

procedure TInferCudaBackend.Linear(x: PSingleArray; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
var
  wbytes: Int64;
  wb, bb, xb, yb: TInferCuBuffer;
  pw, px, pb, py: TCuDevicePtr;
  rows, nin, nout, dtype, hasBias, gx, gy, block: cardinal;
  kernel: TCuHandle;
begin
  // grid y holds the rows (at most 65535), grid x the outputs
  if not GpuLinear(w.DType, aIn) or (aRows < 1) or (aRows > 65535) or
     (aOut < 1) or (Int64(aOut) > high(integer)) or (Int64(aIn) > high(integer)) or
     not fGpu.TryLock then
  begin
    inherited Linear(x, aRows, aIn, w, aOut, bias, y);
    exit;
  end;
  try
    wbytes := Int64(aOut) * (aIn div INFER_DTYPE_BLOCK[w.DType]) *
      INFER_DTYPE_SIZE[w.DType];
    wb := Weight(w.Data, w.DType, wbytes);
    bb := nil;
    if bias <> nil then
      bb := Weight(bias, idtF32, Int64(aOut) * 4);
    xb := Scratch(fX, Int64(aRows) * aIn * 4);
    yb := Scratch(fY, Int64(aRows) * aOut * 4);
    xb.Upload(0, x, Int64(aRows) * aIn * 4);
    pw := wb.Ptr;
    px := xb.Ptr;
    pb := 0; // never read without a bias
    if bb <> nil then
      pb := bb.Ptr;
    py := yb.Ptr;
    rows := aRows;
    nin := aIn;
    nout := aOut;
    dtype := LINEAR_DTYPE[w.DType];
    hasBias := ord(bias <> nil);
    LinearShape(w.DType, aIn, aRows, aOut, kernel, gx, gy, block);
    fCu.Launch(kernel, [@pw, @px, @pb, @py, @rows, @nin, @nout, @dtype,
      @hasBias], gx, block, gy);
    yb.Download(0, y, Int64(aRows) * aOut * 4);
    inc(fGpuLinears);
  finally
    fGpu.UnLock;
  end;
end;

end.
