/// native inference engine - the Vulkan backend: Linear on the GPU
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - phase 4b of docs/native-inference.md: TInferVulkanBackend is the fast CPU
//   backend whose Linear runs on a Vulkan device for F32, Q8_0, Q5_0, Q4_K and
//   Q6_K weights; every other operation, every other weight type and every
//   call while the device is busy stay on the CPU
// - weights and biases are uploaded once, on first use, and kept on the
//   device, keyed by their address, type and size: they must neither change
//   nor be freed while the backend lives (model weights and biases do not; a
//   temporary bias would be served stale from its reused address) -
//   ClearWeights forgets them
// - activations go up and results come down on every call, and each call
//   waits for the device: a correctness step, not yet a speed-up for decoding
//   (that needs the whole forward pass on the device, phase 4d)
// - arithmetic: the weights dequantized in float, as InferRowToF32 does, times
//   F32 activations, summed in another order than the CPU - a tolerance
unit mormot.ai.infer.backend.vulkan;

{
  *****************************************************************************

    - TInferVulkanBackend: TInferCpuFastBackend with Linear on a Vulkan device

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.vulkan;

type
  /// the fast CPU backend with Linear on a Vulkan device
  TInferVulkanBackend = class(TInferCpuFastBackend)
  protected
    fVk: TInferVulkan;
    fLinear, fLinear8, fLinear8Sg, fGemm8: TInferVkPipeline;
    // linear8.comp for the experts chosen on the device (-DIDS), both builds
    fLinear8Ids, fLinear8IdsSg: TInferVkPipeline;
    // matvec8.comp per LINEAR_DTYPE (2 Q8_0, 3 Q4_K, 4 Q6_K), plain and with
    // subgroup reductions
    fMatvec, fMatvecSg: array[2..4] of TInferVkPipeline;
    fUseSubgroups, fUseMatvec, fUseGemm: boolean;
    fGpu: TLightLock;
    // the weight cache: parallel arrays, searched linearly (a model has a few
    // hundred matrices)
    fWeightData: array of pointer;
    fWeightType: array of TInferDType;
    fWeightBytes: TInt64DynArray;
    fWeightBuffer: array of TInferVkBuffer;
    fWeightCount: PtrInt;
    fX, fY, fNoBias: TInferVkBuffer;
    fGpuLinears: Int64;
    fWeightEpoch: integer;
    fMaxLinearBytes: Int64;
    function Weight(aData: pointer; aType: TInferDType;
      aBytes: Int64): TInferVkBuffer;
    function Scratch(var aBuffer: TInferVkBuffer; aBytes: Int64): TInferVkBuffer;
    // the shader and work groups of a Linear: matvec8.comp for one row of
    // Q8_0, Q4_K or Q6_K weights when UseMatvec (4 features per work
    // group), else
    // linear8.comp (4 features and 8 rows per work group) when the inputs
    // allow it, else linear.comp - the subgroup builds when UseSubgroups
    function LinearShape(aType: TInferDType; aIn, aRows, aOut: PtrInt;
      out aX, aY: cardinal): TInferVkPipeline;
  public
    /// the CPU part as TInferCpuFastBackend.Create, plus the Vulkan device
    // (TInferVulkan.Create: raises EInferVulkan without one)
    constructor Create(aThreads: integer = 0); reintroduce;
    destructor Destroy; override;
    /// y = x * w^T + bias on the device for F32, Q8_0, Q5_0, Q4_K and Q6_K
    // weights, on the CPU otherwise (and while another thread uses the device)
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); override;
    /// release every weight kept on the device
    procedure ClearWeights;
    /// take the device for a sequence of batch operations (a forward pass on
    // the GPU): Linear computes on the CPU meanwhile
    procedure GpuLock;
    procedure GpuUnLock;
    /// the device copy of a weight (uploaded on first use) - GpuLock held
    function DeviceWeight(aData: pointer; aType: TInferDType;
      aBytes: Int64): TInferVkBuffer;
    /// true when RecordLinear takes weights of aType with aIn inputs
    function GpuLinear(aType: TInferDType; aIn: PtrInt): boolean;
    /// record y = x * w^T (+ aBias) into the running batch of Vulkan, the
    // weights from the cache - GpuLock held, BatchBegin called; x and y are
    // [aRows, aIn] and [aRows, aOut] at offset 0
    // - aBias ([aOut] floats) comes from the cache too: uploaded beforehand
    // (DeviceWeight), as a batch cannot upload; no bias for a matrix in parts
    // - a matrix larger than one binding (LinearPart < aOut) goes part by
    // part into aTemp - at least [aRows, LinearPart] floats, given by the
    // caller (no buffer is made or freed while a batch is recorded) - whose
    // rows are then copied into y; the weights uploaded by UploadLinear
    procedure RecordLinear(x: TInferVkBuffer; aRows, aIn: PtrInt;
      const w: TInferWeight; aOut: PtrInt; y: TInferVkBuffer;
      aBarrier: boolean = true; aTemp: TInferVkBuffer = nil;
      aBias: PSingleArray = nil);
    /// record y = x * w^T (no bias) as RecordLinear, the aOut x aIn weights of
    // aType at byte aOffset of aWeight (e.g. one expert of a buffer which holds
    // every expert of a layer): aOffset a multiple of the device's
    // MinStorageAlignment, the matrix within one binding (LinearPart = aOut)
    procedure RecordLinearAt(x: TInferVkBuffer; aRows, aIn: PtrInt;
      aType: TInferDType; aWeight: TInferVkBuffer; aOffset: Int64; aOut: PtrInt;
      y: TInferVkBuffer; aBarrier: boolean = true);
    /// true when RecordLinearIds runs an aOut x aIn matrix of aType per expert
    // from aWeight: the form of linear8 (any block type, F32 with aIn a
    // multiple of 8) and the whole buffer in one binding
    function GpuLinearIds(aType: TInferDType; aIn: PtrInt;
      aWeight: TInferVkBuffer): boolean;
    /// record one token row through aSlots experts chosen on the device: slot
    // s reads its expert id from aIds[s] (int32), the expert's aOut x aIn
    // matrix of aType at byte id * aStride of aWeight (the experts of a layer
    // one buffer, bound whole); x is row s of x when aXPerSlot, else row 0;
    // y receives row s - linear8.comp built with -DIDS
    procedure RecordLinearIds(x: TInferVkBuffer; aIn: PtrInt; aType: TInferDType;
      aWeight: TInferVkBuffer; aStride: Int64; aOut: PtrInt; aIds: TInferVkBuffer;
      aSlots: PtrInt; aXPerSlot: boolean; y: TInferVkBuffer; aBarrier: boolean = true);
    /// the output features of one part of an aOut x aIn matrix of aType on
    // the device: aOut when its bytes fit one binding (maxStorageBufferRange,
    // or MaxLinearBytes), else as many as do - e.g. the 165 MB head of
    // Qwen3-0.6B Q8_0 on a GPU binding 128 MB (Adreno X1-85)
    function LinearPart(aType: TInferDType; aIn, aOut: PtrInt): PtrInt;
    /// upload the weights of an aOut x aIn matrix, in the parts of
    // LinearPart - GpuLock held, outside a batch
    procedure UploadLinear(const w: TInferWeight; aIn, aOut: PtrInt);
    /// the device context
    property Vulkan: TInferVulkan read fVk;
    /// true when the device runs the subgroup build of linear8.comp: Vulkan
    // 1.1, subgroups of at least 32 invocations (a power of two) with basic,
    // arithmetic and clustered operations in compute shaders
    function SubgroupLinear: boolean;
    /// use the subgroup build when SubgroupLinear (the default) - false: the
    // shared-memory build, for comparisons
    property UseSubgroups: boolean read fUseSubgroups write fUseSubgroups;
    /// one row of Q8_0, Q4_K or Q6_K weights (decoding) through matvec8.comp
    // instead of linear8.comp - by default on an Apple GPU (VendorId $106B)
    // only: there Qwen3-0.6B decodes at 98 instead of 66 tokens/s (Q8_0) and
    // at 84 instead of 56 (Q4_K_M), while an RTX 5090 decodes 5 % slower with
    // it (backlog B8)
    property UseMatvec: boolean read fUseMatvec write fUseMatvec;
    /// many rows (prefill) through the tiled gemm8.comp instead of linear8.comp
    // - by default where the device runs work groups of 256 invocations
    property UseGemm: boolean read fUseGemm write fUseGemm;
    /// how many Linear calls ran on the device
    property GpuLinears: Int64 read fGpuLinears;
    /// the bytes of a weight binding at most, below maxStorageBufferRange -
    // 0 (default): the device limit; for tests of LinearPart - change it,
    // then ClearWeights (a TQwen3Vulkan uploads in the new parts)
    property MaxLinearBytes: Int64 read fMaxLinearBytes write fMaxLinearBytes;
    /// incremented by ClearWeights: a holder of uploaded weights uploads again
    property WeightEpoch: integer read fWeightEpoch;
  end;


implementation

const
  /// the dtype push constant of tools/vk/linear.comp
  // - Unknown, F32, F16, BF16, I32, I64, U8, I8, Q8_0, Q4_0, Q4_K, Q6_K, Q5_0;
  // 0 = CPU
  LINEAR_DTYPE: array[TInferDType] of cardinal = (
    0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 3, 4, 5);

  /// from this many rows on a Linear runs the tiled gemm8.comp (UseGemm)
  GEMM_ROWS = 16;

type
  /// the push constants of tools/vk/linear.comp
  TLinearPush = record
    Rows, InCount, OutCount, DType, HasBias: cardinal;
  end;

  /// the push constants of linear8.comp built with -DIDS
  TLinearIdsPush = record
    Rows, InCount, OutCount, DType, HasBias, Stride, XSlot: cardinal;
  end;


{ TInferVulkanBackend }

constructor TInferVulkanBackend.Create(aThreads: integer);
var
  i: PtrInt;
begin
  inherited Create(aThreads);
  fVk := TInferVulkan.Create;
  fLinear := TInferVkPipeline.Create(fVk, @SPV_LINEAR, length(SPV_LINEAR), 4,
    SizeOf(TLinearPush));
  fLinear.Name := 'linear';
  fLinear8 := TInferVkPipeline.Create(fVk, @SPV_LINEAR8, length(SPV_LINEAR8), 4,
    SizeOf(TLinearPush));
  fLinear8.Name := 'linear';
  fLinear8Ids := TInferVkPipeline.Create(fVk, @SPV_LINEAR8_IDS,
    length(SPV_LINEAR8_IDS), 5, SizeOf(TLinearIdsPush));
  fLinear8Ids.Name := 'linear';
  fMatvec[2] := TInferVkPipeline.Create(fVk, @SPV_MATVEC8_Q8,
    length(SPV_MATVEC8_Q8), 4, SizeOf(TLinearPush));
  fMatvec[3] := TInferVkPipeline.Create(fVk, @SPV_MATVEC8_Q4K,
    length(SPV_MATVEC8_Q4K), 4, SizeOf(TLinearPush));
  fMatvec[4] := TInferVkPipeline.Create(fVk, @SPV_MATVEC8_Q6K,
    length(SPV_MATVEC8_Q6K), 4, SizeOf(TLinearPush));
  for i := low(fMatvec) to high(fMatvec) do
    fMatvec[i].Name := 'linear';
  // 256 invocations per work group: more than Vulkan guarantees (128)
  if (fVk.MaxInvocations >= 256) and (fVk.MaxGroupSizeX >= 256) then
  begin
    fGemm8 := TInferVkPipeline.Create(fVk, @SPV_GEMM8, length(SPV_GEMM8), 4,
      SizeOf(TLinearPush));
    fGemm8.Name := 'linear';
    fUseGemm := true;
  end;
  if (fVk.ApiVersion >= VK_API_VERSION_1_1) and
     (fVk.SubgroupStages and VK_SHADER_STAGE_COMPUTE_BIT <> 0) and
     (fVk.SubgroupOperations and (VK_SUBGROUP_FEATURE_BASIC_BIT or
       VK_SUBGROUP_FEATURE_ARITHMETIC_BIT or VK_SUBGROUP_FEATURE_CLUSTERED_BIT) =
       VK_SUBGROUP_FEATURE_BASIC_BIT or VK_SUBGROUP_FEATURE_ARITHMETIC_BIT or
       VK_SUBGROUP_FEATURE_CLUSTERED_BIT) and
     (fVk.SubgroupSize >= 32) and (fVk.SubgroupSize <= 128) and
     (fVk.SubgroupSize and (fVk.SubgroupSize - 1) = 0) then
  begin
    fLinear8Sg := TInferVkPipeline.Create(fVk, @SPV_LINEAR8_SG,
      length(SPV_LINEAR8_SG), 4, SizeOf(TLinearPush));
    fLinear8Sg.Name := 'linear';
    fLinear8IdsSg := TInferVkPipeline.Create(fVk, @SPV_LINEAR8_IDS_SG,
      length(SPV_LINEAR8_IDS_SG), 5, SizeOf(TLinearIdsPush));
    fLinear8IdsSg.Name := 'linear';
    fMatvecSg[2] := TInferVkPipeline.Create(fVk, @SPV_MATVEC8_Q8_SG,
      length(SPV_MATVEC8_Q8_SG), 4, SizeOf(TLinearPush));
    fMatvecSg[3] := TInferVkPipeline.Create(fVk, @SPV_MATVEC8_Q4K_SG,
      length(SPV_MATVEC8_Q4K_SG), 4, SizeOf(TLinearPush));
    fMatvecSg[4] := TInferVkPipeline.Create(fVk, @SPV_MATVEC8_Q6K_SG,
      length(SPV_MATVEC8_Q6K_SG), 4, SizeOf(TLinearPush));
    for i := low(fMatvecSg) to high(fMatvecSg) do
      fMatvecSg[i].Name := 'linear';
    fUseSubgroups := true;
  end;
  fUseMatvec := fVk.VendorId = $106B; // Apple
  fNoBias := TInferVkBuffer.Create(fVk, 4, true);
end;

destructor TInferVulkanBackend.Destroy;
var
  i: PtrInt;
begin
  if fVk <> nil then
  begin
    ClearWeights;
    FreeAndNil(fX);
    FreeAndNil(fY);
    FreeAndNil(fNoBias);
    FreeAndNil(fGemm8);
    for i := low(fMatvec) to high(fMatvec) do
    begin
      FreeAndNil(fMatvecSg[i]);
      FreeAndNil(fMatvec[i]);
    end;
    FreeAndNil(fLinear8IdsSg);
    FreeAndNil(fLinear8Ids);
    FreeAndNil(fLinear8Sg);
    FreeAndNil(fLinear8);
    FreeAndNil(fLinear);
    FreeAndNil(fVk);
  end;
  inherited Destroy;
end;

procedure TInferVulkanBackend.ClearWeights;
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

procedure TInferVulkanBackend.GpuLock;
begin
  fGpu.Lock;
end;

procedure TInferVulkanBackend.GpuUnLock;
begin
  fGpu.UnLock;
end;

function TInferVulkanBackend.DeviceWeight(aData: pointer; aType: TInferDType;
  aBytes: Int64): TInferVkBuffer;
begin
  result := Weight(aData, aType, aBytes);
end;

function TInferVulkanBackend.GpuLinear(aType: TInferDType; aIn: PtrInt): boolean;
begin
  result := (LINEAR_DTYPE[aType] <> 0) and (aIn > 0) and
    (aIn mod INFER_DTYPE_BLOCK[aType] = 0);
end;

function TInferVulkanBackend.SubgroupLinear: boolean;
begin
  result := fLinear8Sg <> nil;
end;

function TInferVulkanBackend.LinearShape(aType: TInferDType; aIn, aRows,
  aOut: PtrInt; out aX, aY: cardinal): TInferVkPipeline;
var
  n: PtrInt;
begin
  if fUseMatvec and (aType in [idtQ8_0, idtQ4_K, idtQ6_K]) and (aRows = 1) then
  begin
    if fUseSubgroups and (fMatvecSg[LINEAR_DTYPE[aType]] <> nil) then
      result := fMatvecSg[LINEAR_DTYPE[aType]]
    else
      result := fMatvec[LINEAR_DTYPE[aType]];
    n := (aOut + 3) shr 2; // one feature per 32 invocations
    aY := 1;
  end
  else if fUseGemm and (fGemm8 <> nil) and (aRows >= GEMM_ROWS) and
          (aIn and 31 = 0) and
          ((aOut + 63) shr 6 <= 65535) then
  begin
    // 64 rows x 64 features per work group
    result := fGemm8;
    aX := (aOut + 63) shr 6;
    aY := (aRows + 63) shr 6;
    exit;
  end
  else if (aType <> idtF32) or (aIn and 7 = 0) then
  begin
    // 8 weights per step: every block type qualifies, F32 when aIn allows
    if fUseSubgroups and (fLinear8Sg <> nil) then
      result := fLinear8Sg
    else
      result := fLinear8;
    n := (aOut + 3) shr 2;
    aY := (aRows + 7) shr 3;
  end
  else
  begin
    result := fLinear;
    n := aOut;
    aY := aRows;
  end;
  if n > 65535 then
    n := 65535; // both shaders loop over the features
  aX := n;
end;

function TInferVulkanBackend.LinearPart(aType: TInferDType; aIn, aOut: PtrInt): PtrInt;
var
  rowBytes, most: Int64;
begin
  rowBytes := Int64(aIn div INFER_DTYPE_BLOCK[aType]) * INFER_DTYPE_SIZE[aType];
  most := fVk.MaxStorageRange;
  if (fMaxLinearBytes > 0) and (fMaxLinearBytes < most) then
    most := fMaxLinearBytes;
  most := most and not Int64(3); // allocated in whole 4-byte words
  if rowBytes * aOut <= most then
    result := aOut
  else
    result := most div rowBytes;
  if result < 1 then
    raise EInferVulkan.CreateUtf8('%.LinearPart: one row of % % binds more than % bytes',
      [self, aIn, INFER_DTYPE_NAME[aType], most]);
end;

procedure TInferVulkanBackend.UploadLinear(const w: TInferWeight; aIn, aOut: PtrInt);
var
  rowBytes: Int64;
  part, o, n: PtrInt;
begin
  rowBytes := Int64(aIn div INFER_DTYPE_BLOCK[w.DType]) * INFER_DTYPE_SIZE[w.DType];
  part := LinearPart(w.DType, aIn, aOut);
  o := 0;
  while o < aOut do
  begin
    n := aOut - o;
    if n > part then
      n := part;
    DeviceWeight(PAnsiChar(w.Data) + Int64(o) * rowBytes, w.DType, n * rowBytes);
    inc(o, n);
  end;
end;

procedure TInferVulkanBackend.RecordLinear(x: TInferVkBuffer; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; y: TInferVkBuffer; aBarrier: boolean;
  aTemp: TInferVkBuffer; aBias: PSingleArray);
var
  push: TLinearPush;
  gx, gy: cardinal;
  pipe: TInferVkPipeline;
  bias: TInferVkBuffer;
  rowBytes: Int64;
  part, o, n: PtrInt;
begin
  if not GpuLinear(w.DType, aIn) or (aRows < 1) or (aRows > 65535) or (aOut < 1) then
    raise EInferVulkan.CreateUtf8('%.RecordLinear: % rows of % % into %',
      [self, aRows, aIn, INFER_DTYPE_NAME[w.DType], aOut]);
  rowBytes := Int64(aIn div INFER_DTYPE_BLOCK[w.DType]) * INFER_DTYPE_SIZE[w.DType];
  push.Rows := aRows;
  push.InCount := aIn;
  push.DType := LINEAR_DTYPE[w.DType];
  push.HasBias := ord(aBias <> nil);
  part := LinearPart(w.DType, aIn, aOut);
  if part >= aOut then
  begin
    push.OutCount := aOut;
    pipe := LinearShape(w.DType, aIn, aRows, aOut, gx, gy);
    if aBias = nil then
      bias := fNoBias
    else
      bias := DeviceWeight(aBias, idtF32, Int64(aOut) * 4);
    fVk.BatchDispatch(pipe, [DeviceWeight(w.Data, w.DType, aOut * rowBytes), x,
      bias, y], @push, gx, gy, 1, aBarrier);
    exit;
  end;
  if aBias <> nil then
    raise EInferVulkan.CreateUtf8('%.RecordLinear: a bias for a matrix in parts',
      [self]);
  // part by part into aTemp, its rows copied to their columns of y
  if (aTemp = nil) or (aTemp.Size < Int64(aRows) * part * 4) then
    raise EInferVulkan.CreateUtf8('%.RecordLinear: % % x % in parts of % ' +
      'needs a buffer of % x % floats', [self, INFER_DTYPE_NAME[w.DType], aOut,
      aIn, part, aRows, part]);
  o := 0;
  while o < aOut do
  begin
    n := aOut - o;
    if n > part then
      n := part;
    push.OutCount := n;
    pipe := LinearShape(w.DType, aIn, aRows, n, gx, gy);
    fVk.BatchDispatch(pipe, [DeviceWeight(PAnsiChar(w.Data) + Int64(o) * rowBytes,
      w.DType, n * rowBytes), x, fNoBias, aTemp], @push, gx, gy, 1, true);
    fVk.BatchCopyRows(aTemp, 0, Int64(n) * 4, y, Int64(o) * 4, Int64(aOut) * 4,
      Int64(n) * 4, aRows);
    inc(o, n);
  end;
end;

procedure TInferVulkanBackend.RecordLinearAt(x: TInferVkBuffer; aRows,
  aIn: PtrInt; aType: TInferDType; aWeight: TInferVkBuffer; aOffset: Int64;
  aOut: PtrInt; y: TInferVkBuffer; aBarrier: boolean);
var
  push: TLinearPush;
  gx, gy: cardinal;
  pipe: TInferVkPipeline;
  bytes: Int64;
begin
  if not GpuLinear(aType, aIn) or (aRows < 1) or (aRows > 65535) or (aOut < 1) or
     (aWeight = nil) or (LinearPart(aType, aIn, aOut) < aOut) then
    raise EInferVulkan.CreateUtf8('%.RecordLinearAt: % rows of % % into %',
      [self, aRows, aIn, INFER_DTYPE_NAME[aType], aOut]);
  bytes := Int64(aOut) * (aIn div INFER_DTYPE_BLOCK[aType]) * INFER_DTYPE_SIZE[aType];
  if (aOffset < 0) or (aOffset > aWeight.Size) or (bytes > aWeight.Size - aOffset) then
    raise EInferVulkan.CreateUtf8('%.RecordLinearAt: % bytes at % of a buffer of %',
      [self, bytes, aOffset, aWeight.Size]);
  // the shaders read whole 4-byte words: the last one may reach up to 3 bytes
  // beyond the matrix (into the next one, or into the rounding of the buffer)
  bytes := (bytes + 3) and not Int64(3);
  if aOffset + bytes > aWeight.Size then
    bytes := aWeight.Size - aOffset;
  push.Rows := aRows;
  push.InCount := aIn;
  push.OutCount := aOut;
  push.DType := LINEAR_DTYPE[aType];
  push.HasBias := 0;
  pipe := LinearShape(aType, aIn, aRows, aOut, gx, gy);
  fVk.BatchDispatchAt(pipe, [aWeight, x, fNoBias, y], [aOffset, 0, 0, 0],
    [bytes, 0, 0, 0], @push, gx, gy, 1, aBarrier);
end;

function TInferVulkanBackend.GpuLinearIds(aType: TInferDType; aIn: PtrInt;
  aWeight: TInferVkBuffer): boolean;
begin
  // the whole buffer in one binding - below MaxLinearBytes too, as LinearPart
  result := GpuLinear(aType, aIn) and ((aType <> idtF32) or (aIn and 7 = 0)) and
    (aWeight <> nil) and (aWeight.Size <= fVk.MaxStorageRange) and
    ((fMaxLinearBytes <= 0) or (aWeight.Size <= fMaxLinearBytes));
end;

procedure TInferVulkanBackend.RecordLinearIds(x: TInferVkBuffer; aIn: PtrInt;
  aType: TInferDType; aWeight: TInferVkBuffer; aStride: Int64; aOut: PtrInt;
  aIds: TInferVkBuffer; aSlots: PtrInt; aXPerSlot: boolean; y: TInferVkBuffer;
  aBarrier: boolean);
var
  push: TLinearIdsPush;
  pipe: TInferVkPipeline;
  n: PtrInt;
begin
  // the byte address of a weight is a 32-bit value in the shader
  if not GpuLinearIds(aType, aIn, aWeight) or (aSlots < 1) or (aSlots > 65535) or
     (aOut < 1) or (aStride <= 0) or (aStride > aWeight.Size) or
     (aStride and 3 <> 0) or (aIds = nil) then
    raise EInferVulkan.CreateUtf8('%.RecordLinearIds: % slots of % % into %',
      [self, aSlots, aIn, INFER_DTYPE_NAME[aType], aOut]);
  push.Rows := 1;
  push.InCount := aIn;
  push.OutCount := aOut;
  push.DType := LINEAR_DTYPE[aType];
  push.HasBias := 0;
  push.Stride := aStride;
  push.XSlot := ord(aXPerSlot);
  if fUseSubgroups and (fLinear8IdsSg <> nil) then
    pipe := fLinear8IdsSg
  else
    pipe := fLinear8Ids;
  n := (aOut + 3) shr 2;
  if n > 65535 then
    n := 65535; // the shader loops over the features
  fVk.BatchDispatch(pipe, [aWeight, x, fNoBias, y, aIds], @push, n, aSlots, 1,
    aBarrier);
end;

function TInferVulkanBackend.Weight(aData: pointer; aType: TInferDType;
  aBytes: Int64): TInferVkBuffer;
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
  result := TInferVkBuffer.Create(fVk, aBytes, true);
  try
    fVk.Upload(result, 0, aData, aBytes);
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

function TInferVulkanBackend.Scratch(var aBuffer: TInferVkBuffer;
  aBytes: Int64): TInferVkBuffer;
var
  grown: Int64;
begin
  if (aBuffer = nil) or (aBuffer.Size < aBytes) then
  begin
    FreeAndNil(aBuffer);
    // 25 % more: a growing context does not reallocate on every token - but
    // never beyond what Run may bind (aBytes itself was checked by Linear)
    grown := aBytes + aBytes shr 2;
    if (grown + 3) and not Int64(3) > fVk.MaxStorageRange then // as allocated
      grown := aBytes;
    aBuffer := TInferVkBuffer.Create(fVk, grown, true);
  end;
  result := aBuffer;
end;

procedure TInferVulkanBackend.Linear(x: PSingleArray; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
var
  wbytes: Int64;
  wb, bb, xb, yb: TInferVkBuffer;
  push: TLinearPush;
  gx, gy: cardinal;
  pipe: TInferVkPipeline;
begin
  if (LINEAR_DTYPE[w.DType] = 0) or (aRows < 1) or (aRows > 65535) or
     (aIn < 1) or (aOut < 1) or (aIn mod INFER_DTYPE_BLOCK[w.DType] <> 0) then
  begin
    inherited Linear(x, aRows, aIn, w, aOut, bias, y);
    exit;
  end;
  wbytes := Int64(aOut) * (aIn div INFER_DTYPE_BLOCK[w.DType]) *
    INFER_DTYPE_SIZE[w.DType];
  // the sizes as allocated: whole 4-byte words (see TInferVkBuffer.Create)
  if ((wbytes + 3) and not Int64(3) > fVk.MaxStorageRange) or
     (Int64(aRows) * aIn * 4 > fVk.MaxStorageRange) or
     (Int64(aRows) * aOut * 4 > fVk.MaxStorageRange) or
     not fGpu.TryLock then
  begin
    inherited Linear(x, aRows, aIn, w, aOut, bias, y);
    exit;
  end;
  try
    wb := Weight(w.Data, w.DType, wbytes);
    if bias <> nil then
      bb := Weight(bias, idtF32, Int64(aOut) * 4)
    else
      bb := fNoBias;
    xb := Scratch(fX, Int64(aRows) * aIn * 4);
    yb := Scratch(fY, Int64(aRows) * aOut * 4);
    fVk.Upload(xb, 0, x, Int64(aRows) * aIn * 4);
    push.Rows := aRows;
    push.InCount := aIn;
    push.OutCount := aOut;
    push.DType := LINEAR_DTYPE[w.DType];
    push.HasBias := ord(bias <> nil);
    pipe := LinearShape(w.DType, aIn, aRows, aOut, gx, gy);
    fVk.Run(pipe, [wb, xb, bb, yb], @push, gx, gy);
    fVk.Download(yb, 0, y, Int64(aRows) * aOut * 4);
    inc(fGpuLinears);
  finally
    fGpu.UnLock;
  end;
end;

end.
