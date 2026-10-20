/// native inference engine - the Qwen3 forward pass on a CUDA device
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - backlog B3, slice C4: TQwen3Cuda runs the layers of a TQwen3Model on the
//   device of a TInferCudaBackend - RMSNorm, the projections, rotary positions,
//   causal attention over a KV cache that lives on the device, SwiGLU, the
//   residuals, the final norm and the head - as kernels enqueued on one
//   stream per chunk of tokens: the embeddings and rotary factors go up, the
//   logits come down, and the host waits once (the download)
// - the arithmetic is float throughout (the weights dequantized as
//   InferRowToF32 does): logits match the CPU within a tolerance, not in bits
// - the KV cache holds half floats by default (KvF16), as on Vulkan
// - the model's weights are uploaded once, at Create; the model and the
//   backend must outlive the TQwen3Cuda and its contexts
// - a mixture of experts (dfMoe, phase D slice D3): the router logits on the
//   device, the choice of the experts on the host (TDecoderModel.Route, one
//   wait per layer with experts), then expert by expert in id order its tokens
//   gathered, its SwiGLU block by the Linear kernels, its output added back
//   with the router weights (gatherrows, scatteradd); the experts of a layer
//   stay one buffer when they are one tensor in the file (GGUF), each expert
//   an offset in it
// - a decode step (one token, slice D3c) chooses on the device when every
//   layer with experts has them in one buffer: the route kernel writes the
//   expert ids and weights (as TDecoderModel.Route, sorted by id), linear8ids
//   computes gate, up and down of the k chosen experts at once, moecombine
//   sums them in id order - no wait for the host, the step stays one CUDA
//   graph (RouteOnDevice); a prefill and a run with a RouteTape choose on the
//   host
// - sliding-window layers and layers without rotary positions (NoPE, backlog
//   D4b): the attention reads only the keys of the window, qkrope leaves q and
//   k unrotated; the cache on the device keeps every row (as
//   TDecoderContext.FullCache on the CPU), so a context can go back to any
//   length - the narrow cache of the CPU only saves memory
// - gpt-oss (TDecoderModel.Extras, slice D5d1): attention sinks in the
//   attention kernels (or attnmerge), the output, router and expert biases
//   (expert e at row e of its bias), the clamped SwiGLU (swiglu with alpha and
//   limit) and its MXFP4 experts through the Linear kernels
unit mormot.ai.infer.model.qwen3.cuda;

{
  *****************************************************************************

    - TQwen3CuContext: a sequence and its KV cache on the device
    - TQwen3Cuda: Forward and Greedy of a TQwen3Model on the device

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
  mormot.ai.infer.cuda,
  mormot.ai.infer.backend.cuda,
  mormot.ai.infer.decoder,
  mormot.ai.infer.model.qwen3;

type
  TQwen3Cuda = class;

  /// one device buffer per layer
  TInferCuBufferDynArray = array of TInferCuBuffer;

  /// the device address of every expert of a layer
  TCuDevicePtrDynArray = array of TCuDevicePtr;

  /// one sequence: its token ids and its KV cache on the device
  // - a TQwen3Context of TQwen3Cuda: Length, Common, Truncate, Reset and the
  // ids at every position as there (Capacity: of the device cache); the CPU
  // arrays stay empty
  TQwen3CuContext = class(TQwen3Context)
  protected
    fRows: PtrInt;
    fDevKeys, fDevValues: TInferCuBufferDynArray; // per layer [capacity, kvWidth]
    fKvF16: boolean; // the cache holds half floats (TQwen3Cuda.KvF16)
    // device scratch for up to fRows token rows
    fDevX, fDevH, fDevQ, fDevK, fDevV, fDevAttn, fDevTmp, fDevGate, fDevUp,
      fDevCos: TInferCuBuffer;
    fPart, fLogits: TInferCuBuffer; // the key ranges of the attention, logits
    // a layer with experts: router logits, the rows of one expert (gathered
    // input, output), the token rows and weights of every expert in id order
    fDevRoute, fDevGather, fDevExpOut, fDevIdx, fDevW: TInferCuBuffer;
    // a decode step routed on the device: the k expert ids and weights, the
    // gate / up and down rows of the k experts
    fDevKIds, fDevKW, fDevKGate, fDevKUp, fDevKDown: TInferCuBuffer;
    fKIds: TIntegerDynArray; // CheckRoute: the choice read back
    fKW: TSingleDynArray;
    fRoute, fChosenW, fW: TSingleDynArray;
    fChosen, fIdx, fStart: TIntegerDynArray;
    // the inputs of a chunk on the host: embeddings, then cos, then sin
    fIn: TSingleDynArray;
    // the decode step as a graph (TQwen3Cuda.UseGraphs), on fCu
    fCu: TInferCuda;
    fGraph: TCuHandle;
    procedure FreeScratch;
    procedure FreeCache;
  public
    destructor Destroy; override;
    /// true when the cache holds half floats (TQwen3Cuda.KvF16 when it was
    // allocated)
    property KvF16: boolean
      read fKvF16;
  end;

  /// the forward pass of a TQwen3Model on the device of a TInferCudaBackend
  // - a TQwen3Device: Model.Device := this instance makes Model.NewContext
  // (and so a TNativeLlmClient) decode here
  TQwen3Cuda = class(TQwen3Device)
  protected
    fBackend: TInferCudaBackend;
    fCu: TInferCuda;
    fOps, fBasic: TInferCuModule;
    fNorm, fAddNorm, fSwiGlu, fQkRope, fAttention, fMerge, fAdd: TCuHandle;
    fFill0, fGather, fScatter: TCuHandle; // a mixture of experts
    fRoute, fCombine: TCuHandle; // routed on the device (D3c)
    fMoe: boolean; // a layer with experts
    // every layer with experts can route on the device (one buffer each)
    fDeviceRoutable, fRouteOnDevice, fCheckRoute: boolean;
    fRouteChecks, fRouteMismatches: Int64;
    fRouteMaxDiff: double;
    // per layer with experts: where every expert's gate, up and down are, and
    // the bytes from one expert to the next when they are one buffer (else 0)
    fExpertGate, fExpertUp, fExpertDown: array of TCuDevicePtrDynArray;
    fStrideGate, fStrideUp, fStrideDown: TInt64DynArray;
    fQkRope16, fAttention16: TCuHandle; // for a KV cache of half floats
    fEpoch: integer; // the backend's WeightEpoch of the uploaded weights
    fSplitKeys: integer;
    fUseGraphs, fKvF16: boolean;
    procedure UploadWeights;
    function Vector(v: PSingleArray; aCount: PtrInt): TCuDevicePtr;
    function ChunkRows(aPast, aCount: PtrInt; aAllLogits: boolean): PtrInt;
    procedure Prepare(aCtx: TQwen3CuContext; aTokens, aRows, aOutRows: PtrInt);
    /// aCount rows of aIds or of aEmbeds (the other one nil) in one pass;
    // aNormed <> nil receives the normed last row
    procedure Chunk(aCtx: TQwen3CuContext; aIds: PIntegerArray; aEmbeds: PSingleArray;
      aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean;
      aNormed: PSingleArray);
    /// Forward of aIds or of aEmbeds (the other one nil), chunk by chunk
    procedure ForwardRows(aCtx: TQwen3Context; aIds: PIntegerArray; aEmbeds: PSingleArray;
      aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean;
      aNormed: PSingleArray);
    procedure EnqueueNorm(x: TInferCuBuffer; aRows, aCols: PtrInt;
      w: PSingleArray; y: TInferCuBuffer);
    procedure EnqueueAddNorm(x, t: TInferCuBuffer; aRows, aCols: PtrInt;
      w: PSingleArray; y: TInferCuBuffer);
    /// the dfMoe block of layer aLayer: fDevH in, fDevTmp out
    procedure EnqueueMoe(aCtx: TQwen3CuContext; aLayer, aRows: PtrInt);
    /// the same for one token row, the experts chosen on the device
    procedure EnqueueMoeDevice(aCtx: TQwen3CuContext; aLayer: PtrInt);
    /// CheckRoute: the device's choice against TDecoderModel.Route
    procedure CheckDeviceRoute(aCtx: TQwen3CuContext);
  public
    /// upload every weight of aModel to the device of aBackend; raises
    // EInferError when a matrix type has no kernel (F32, Q8_0, Q5_0, Q4_K and
    // Q6_K have one)
    // - aModel and aBackend are not owned and must outlive this instance
    constructor Create(aModel: TDecoderModel; aBackend: TInferCudaBackend);
    destructor Destroy; override;
    /// a context for this model on this device (owned by the caller)
    function NewContext: TQwen3CuContext;
    /// NewContext, for TQwen3Model.NewContext
    function NewDeviceContext: TQwen3Context; override;
    /// TQwen3Model.Forward on the device: append aCount tokens to aCtx (a
    // context of this instance) and compute them; aLogits: nil, the last
    // token's [VocabSize] or, when aAllLogits, [aCount, VocabSize]
    procedure Forward(aCtx: TQwen3Context; aIds: PIntegerArray; aCount: PtrInt;
      aLogits: PSingleArray; aAllLogits: boolean = false); override;
    /// TQwen3Model.ForwardEmbeds on the device: aCount embedding rows
    // ([aCount, Hidden] floats) in place of the embeddings of ids, as Forward;
    // aNormed <> nil receives the last row after the final norm ([Hidden])
    procedure ForwardEmbeds(aCtx: TQwen3Context; aEmbeds: PSingleArray;
      aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean = false;
      aNormed: PSingleArray = nil); override;
    /// keep the cached tokens of aCtx that start aIds, at most aCount - 1
    function Reuse(aCtx: TQwen3Context; aIds: PIntegerArray;
      aCount: PtrInt): PtrInt; override;
    /// TQwen3Model.Greedy on the device
    function Greedy(aCtx: TQwen3Context; const aPrompt: array of integer;
      aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
    property Backend: TInferCudaBackend
      read fBackend;
    /// the attention cuts the keys of a row into ranges of about SplitKeys
    // when there are few blocks (decoding), merged afterwards - 0: never
    // (the tests compare both)
    property SplitKeys: integer
      read fSplitKeys write fSplitKeys;
    /// a decode step (one token) is recorded as a CUDA graph and launched at
    // once instead of some 370 kernel launches - default true; the graph of
    // the previous step takes the new parameters (TInferCuda.CaptureEnd)
    // - with experts only when they are chosen on the device (RouteOnDevice)
    property UseGraphs: boolean
      read fUseGraphs write fUseGraphs;
    /// the KV cache in half floats (default) instead of floats: half the
    // memory, and half the bytes the attention reads per decoded token -
    // the keys and values rounded to 11 significant bits, the logits then
    // within a looser tolerance of the CPU's
    // - what it buys grows with the context: Qwen3-0.6B Q8_0 on an RTX 5090
    // decodes 2-3 % faster at 200 tokens, 6-8 % at 4000, 16-32 % at 16000
    // - what it costs: logits about 1e-3 off instead of 1e-4, harmless where
    // every step ends in a token id; set it false where floats are fed back
    // step after step (ForwardEmbeds loops of a speech model: the error grows,
    // measured 7e-3 on the Qwen3-TTS talker against 3e-5) and to compare
    // with a reference
    // - a context takes it when its cache is allocated, and again on its
    // next Forward while it is empty (after Reset)
    property KvF16: boolean
      read fKvF16 write fKvF16;
    /// a decode step chooses the experts on the device (default true) when
    // every layer with experts holds them in one buffer (DeviceRoutable) and
    // no RouteTape is set - one CUDA graph per step; false: on the host
    property RouteOnDevice: boolean
      read fRouteOnDevice write fRouteOnDevice;
    /// true when the experts of every layer are one buffer and their types
    // have a linear8ids form: a decode step can route on the device
    property DeviceRoutable: boolean
      read fDeviceRoutable;
    /// diagnosis and tests: after every route on the device the step waits,
    // reads the router logits and the choice back and compares them with
    // TDecoderModel.Route on the same logits (no graph meanwhile)
    property CheckRoute: boolean
      read fCheckRoute write fCheckRoute;
    /// CheckRoute: rows compared, rows whose ids differ, the largest weight
    // difference of the rows with the same ids
    property RouteChecks: Int64
      read fRouteChecks;
    property RouteMismatches: Int64
      read fRouteMismatches;
    property RouteMaxDiff: double
      read fRouteMaxDiff;
  end;


implementation

const
  /// token rows per chunk at most
  CHUNK_MAX = 256;
  /// the logits of every row of a chunk (rows * vocabulary floats): the
  // chunk rows are halved down to this size
  LOGITS_MAX = 256 shl 20;
  /// the queries of one attention block at most (MAXQ in tools/cuda/ops.cu):
  // the heads of one KV head times rows, which share every key and value read
  ATTENTION_QUERIES = 8;
  /// the attention cuts the keys of a row into ranges of about SPLIT_KEYS
  // when there are few blocks - decoding one token - at most SPLIT_GROUPS
  // blocks in all (then attnmerge combines them); as on Vulkan
  SPLIT_KEYS = 64;
  SPLIT_GROUPS = 2048;
  /// the threads of a block of tools/cuda/ops.cu
  OPS_BLOCK = 128;

/// a device buffer of at least aBytes, reallocated (content lost) when smaller
procedure NeedDevice(aCu: TInferCuda; var aBuffer: TInferCuBuffer; aBytes: Int64);
begin
  if (aBuffer <> nil) and (aBuffer.Size >= aBytes) then
    exit;
  FreeAndNil(aBuffer);
  aBuffer := TInferCuBuffer.Create(aCu, aBytes);
end;


{ TQwen3CuContext }

destructor TQwen3CuContext.Destroy;
begin
  if fCu <> nil then
    fCu.GraphFree(fGraph);
  FreeScratch;
  FreeCache;
  inherited Destroy;
end;

procedure TQwen3CuContext.FreeScratch;
begin
  FreeAndNil(fDevX);
  FreeAndNil(fDevH);
  FreeAndNil(fDevQ);
  FreeAndNil(fDevK);
  FreeAndNil(fDevV);
  FreeAndNil(fDevAttn);
  FreeAndNil(fDevTmp);
  FreeAndNil(fDevGate);
  FreeAndNil(fDevUp);
  FreeAndNil(fDevCos);
  FreeAndNil(fDevRoute);
  FreeAndNil(fDevGather);
  FreeAndNil(fDevExpOut);
  FreeAndNil(fDevIdx);
  FreeAndNil(fDevW);
  FreeAndNil(fDevKIds);
  FreeAndNil(fDevKW);
  FreeAndNil(fDevKGate);
  FreeAndNil(fDevKUp);
  FreeAndNil(fDevKDown);
  fRows := 0;
end;

procedure TQwen3CuContext.FreeCache;
var
  i: PtrInt;
begin
  for i := 0 to high(fDevKeys) do
  begin
    FreeAndNil(fDevKeys[i]);
    FreeAndNil(fDevValues[i]);
  end;
  fDevKeys := nil;
  fDevValues := nil;
  FreeAndNil(fPart);
  FreeAndNil(fLogits);
  fCapacity := 0;
end;


{ TQwen3Cuda }

constructor TQwen3Cuda.Create(aModel: TDecoderModel; aBackend: TInferCudaBackend);
var
  l: PtrInt;
begin
  inherited Create;
  if (aModel = nil) or (aBackend = nil) then
    EInferError.RaiseUtf8('%.Create: no model or no backend', [self]);
  // the layers it runs: attention, and a dense SwiGLU block or experts
  for l := 0 to high(aModel.Config.Spec) do
    if aModel.Config.Spec[l].Mixer <> dmAttention then
      EInferError.RaiseUtf8('%: layers of another kind not supported', [self]);
  // attention sinks, the output, router and expert biases and the clamped
  // SwiGLU of gpt-oss run here too (slice D5d)
  fMoe := not aModel.HasOnly(dmAttention, dfDense);
  // qkrope keeps one normalized head in shared memory, the attention reads
  // the keys as float4
  if (aModel.Config.HeadDim > 256) or (aModel.Config.HeadDim and 3 <> 0) then
    EInferError.RaiseUtf8('%: head dimension % not supported', [self,
      aModel.Config.HeadDim]);
  fModel := aModel;
  fBackend := aBackend;
  fCu := aBackend.Cuda;
  fOps := TInferCuModule.Create(fCu, @PTX_OPS);
  fBasic := TInferCuModule.Create(fCu, @PTX_BASIC);
  fNorm := fOps.Kernel('rmsnorm');
  fAddNorm := fOps.Kernel('addnorm');
  fSwiGlu := fOps.Kernel('swiglu');
  fQkRope := fOps.Kernel('qkrope');
  fAttention := fOps.Kernel('attention');
  fQkRope16 := fOps.Kernel('qkrope16');
  fAttention16 := fOps.Kernel('attention16');
  fMerge := fOps.Kernel('attnmerge');
  fFill0 := fOps.Kernel('fill0');
  fGather := fOps.Kernel('gatherrows');
  fScatter := fOps.Kernel('scatteradd');
  fRoute := fOps.Kernel('route');
  fCombine := fOps.Kernel('moecombine');
  fRouteOnDevice := true;
  fSplitKeys := SPLIT_KEYS;
  fUseGraphs := true;
  fKvF16 := true;
  fAdd := fBasic.Kernel('add');
  fBackend.GpuLock;
  try
    UploadWeights;
  finally
    fBackend.GpuUnLock;
  end;
end;

destructor TQwen3Cuda.Destroy;
begin
  // the model no longer hands out contexts of this device
  if (fModel <> nil) and (fModel.Device = self) then
    fModel.Device := nil;
  // the copies of the model's weights on the backend: the model may compute
  // with another backend (it forgets only there when freed), and memory it
  // frees later may come back at the same address with other weights
  if (fModel <> nil) and (fBackend <> nil) then
    fModel.ForgetWeights(fBackend);
  fBasic.Free;
  fOps.Free;
  inherited Destroy;
end;

function TQwen3Cuda.Vector(v: PSingleArray; aCount: PtrInt): TCuDevicePtr;
begin
  result := fBackend.DeviceWeight(v, idtF32, Int64(aCount) * 4).Ptr;
end;

procedure TQwen3Cuda.UploadWeights;
var
  c: TQwen3Config;
  l, qw, kvw: PtrInt;
  ly: ^TQwen3Layer;

  procedure Matrix(const w: TInferWeight; aRows, aCols: PtrInt);
  begin
    if not fBackend.GpuLinear(w.DType, aCols) then
      EInferError.RaiseUtf8('%: no kernel for a % matrix of % columns',
        [self, INFER_DTYPE_NAME[w.DType], aCols]);
    fBackend.DeviceWeight(w.Data, w.DType, Int64(aRows) *
      (aCols div INFER_DTYPE_BLOCK[w.DType]) * INFER_DTYPE_SIZE[w.DType]);
  end;

  // the experts of a layer: one buffer when they lie one after the other in
  // host memory (a tensor of a GGUF file), else one buffer per expert
  procedure Experts(const w: TInferWeightDynArray; aRows, aCols: PtrInt;
    out aPtr: TCuDevicePtrDynArray; out aStride: Int64);
  var
    e: PtrInt;
    slice: Int64;
    one: boolean;
    base: TCuDevicePtr;
  begin
    SetLength(aPtr, length(w));
    aStride := 0;
    one := true;
    for e := 0 to high(w) do
    begin
      if not fBackend.GpuLinear(w[e].DType, aCols) then
        EInferError.RaiseUtf8('%: no kernel for a % expert of % columns',
          [self, INFER_DTYPE_NAME[w[e].DType], aCols]);
      if (w[e].DType <> w[0].DType) or (PtrUInt(w[e].Data) <> PtrUInt(w[0].Data) +
         PtrUInt(e) * PtrUInt(Int64(aRows) * (aCols div INFER_DTYPE_BLOCK[w[0].DType]) *
         INFER_DTYPE_SIZE[w[0].DType])) then
        one := false;
    end;
    if one then
    begin
      slice := Int64(aRows) * (aCols div INFER_DTYPE_BLOCK[w[0].DType]) *
        INFER_DTYPE_SIZE[w[0].DType];
      base := fBackend.DeviceWeight(w[0].Data, w[0].DType, slice * length(w)).Ptr;
      for e := 0 to high(w) do
        aPtr[e] := base + e * slice;
      // linear8ids finds an expert at id * stride: whole words, one type
      if fBackend.GpuLinearIds(w[0].DType, aCols) and (slice and 3 = 0) then
        aStride := slice;
    end
    else
      for e := 0 to high(w) do
        aPtr[e] := fBackend.DeviceWeight(w[e].Data, w[e].DType, Int64(aRows) *
          (aCols div INFER_DTYPE_BLOCK[w[e].DType]) * INFER_DTYPE_SIZE[w[e].DType]).Ptr;
  end;

begin
  // everything up front: the first token would otherwise carry the transfer
  // - the caller holds GpuLock
  c := fModel.Config;
  qw := c.Heads * c.HeadDim;
  kvw := c.KvHeads * c.HeadDim;
  fExpertGate := nil;
  fExpertUp := nil;
  fExpertDown := nil;
  SetLength(fExpertGate, c.Layers);
  SetLength(fExpertUp, c.Layers);
  SetLength(fExpertDown, c.Layers);
  fStrideGate := nil;
  fStrideUp := nil;
  fStrideDown := nil;
  SetLength(fStrideGate, c.Layers);
  SetLength(fStrideUp, c.Layers);
  SetLength(fStrideDown, c.Layers);
  // the route kernel keeps the choice of up to 64 of 4096 experts
  fDeviceRoutable := fMoe and (c.ExpertsUsed <= 64) and (c.Experts <= 4096);
  for l := 0 to c.Layers - 1 do
  begin
    ly := @fModel.Layers[l];
    Matrix(ly^.Q, qw, c.Hidden);
    Matrix(ly^.K, kvw, c.Hidden);
    Matrix(ly^.V, kvw, c.Hidden);
    Matrix(ly^.O, c.Hidden, qw);
    if c.Spec[l].Ffn = dfMoe then
    begin
      Matrix(ly^.Router, c.Experts, c.Hidden);
      Experts(ly^.ExpertGate, c.ExpertIntermediate, c.Hidden, fExpertGate[l],
        fStrideGate[l]);
      Experts(ly^.ExpertUp, c.ExpertIntermediate, c.Hidden, fExpertUp[l],
        fStrideUp[l]);
      Experts(ly^.ExpertDown, c.Hidden, c.ExpertIntermediate, fExpertDown[l],
        fStrideDown[l]);
      if (fStrideGate[l] = 0) or (fStrideUp[l] = 0) or (fStrideDown[l] = 0) then
        fDeviceRoutable := false;
    end
    else
    begin
      Matrix(ly^.Gate, c.Intermediate, c.Hidden);
      Matrix(ly^.Up, c.Intermediate, c.Hidden);
      Matrix(ly^.Down, c.Hidden, c.Intermediate);
    end;
    Vector(ly^.InputNorm, c.Hidden);
    Vector(ly^.PostNorm, c.Hidden);
    if ly^.QNorm <> nil then
    begin
      Vector(ly^.QNorm, c.HeadDim);
      Vector(ly^.KNorm, c.HeadDim);
    end;
    // the biases and sinks up front: a captured graph cannot upload
    if ly^.QBias <> nil then
    begin
      Vector(ly^.QBias, qw);
      Vector(ly^.KBias, kvw);
      Vector(ly^.VBias, kvw);
    end;
    if ly^.OBias <> nil then
      Vector(ly^.OBias, c.Hidden);
    if ly^.Sinks <> nil then
      Vector(ly^.Sinks, c.Heads);
    if ly^.RouterBias <> nil then
      Vector(ly^.RouterBias, c.Experts);
    // each bias of the experts on its own, as TDecoderModel.MoeBlock
    if ly^.ExpertGateBias <> nil then
      Vector(ly^.ExpertGateBias, c.Experts * c.ExpertIntermediate);
    if ly^.ExpertUpBias <> nil then
      Vector(ly^.ExpertUpBias, c.Experts * c.ExpertIntermediate);
    if ly^.ExpertDownBias <> nil then
      Vector(ly^.ExpertDownBias, c.Experts * c.Hidden);
  end;
  Vector(fModel.FinalNorm, c.Hidden);
  if fModel.HasHead then // a stack without one: the normed state only
    Matrix(fModel.Head, c.VocabSize, c.Hidden);
  fEpoch := fBackend.WeightEpoch;
end;

function TQwen3Cuda.NewContext: TQwen3CuContext;
begin
  result := TQwen3CuContext.Create;
  result.fDevice := self;
  result.fCu := fCu;
end;

function TQwen3Cuda.NewDeviceContext: TQwen3Context;
begin
  result := NewContext;
end;

function TQwen3Cuda.ChunkRows(aPast, aCount: PtrInt; aAllLogits: boolean): PtrInt;
var
  c: TQwen3Config;
begin
  c := fModel.Config;
  result := aCount;
  if result > CHUNK_MAX then
    result := CHUNK_MAX;
  // the logits of every row of the chunk in one buffer
  if aAllLogits then
    while (result > 1) and (Int64(result) * c.VocabSize * 4 > LOGITS_MAX) do
      result := result shr 1;
end;

procedure TQwen3Cuda.Prepare(aCtx: TQwen3CuContext; aTokens, aRows,
  aOutRows: PtrInt);
var
  c: TQwen3Config;
  cap, kvw, l, half, es: PtrInt;
  keys, values: TInferCuBufferDynArray;
begin
  c := fModel.Config;
  kvw := c.KvHeads * c.HeadDim;
  half := c.HeadDim shr 1;
  // an empty context takes the cache format of KvF16
  if (aCtx.fLength = 0) and (aCtx.fCapacity > 0) and (aCtx.fKvF16 <> fKvF16) then
    aCtx.FreeCache;
  if aCtx.fCapacity = 0 then
    aCtx.fKvF16 := fKvF16;
  es := 4; // bytes per cached value
  if aCtx.fKvF16 then
    es := 2;
  // the KV cache: doubled, a quarter beyond the tokens wanted (as on the CPU),
  // the cached tokens copied into the new buffers
  if aTokens > aCtx.fCapacity then
  begin
    cap := aCtx.fCapacity * 2;
    if cap < aTokens + aTokens shr 2 then
      cap := aTokens + aTokens shr 2;
    if cap < 64 then
      cap := 64;
    if cap > c.MaxPositions then
      cap := c.MaxPositions;
    SetLength(keys, c.Layers);
    SetLength(values, c.Layers);
    try
      for l := 0 to c.Layers - 1 do
      begin
        keys[l] := TInferCuBuffer.Create(fCu, Int64(cap) * kvw * es);
        values[l] := TInferCuBuffer.Create(fCu, Int64(cap) * kvw * es);
        if aCtx.fLength > 0 then
        begin
          aCtx.fDevKeys[l].CopyTo(0, keys[l], 0, Int64(aCtx.fLength) * kvw * es);
          aCtx.fDevValues[l].CopyTo(0, values[l], 0, Int64(aCtx.fLength) * kvw * es);
        end;
      end;
    except
      for l := 0 to high(keys) do
      begin
        keys[l].Free;
        values[l].Free;
      end;
      raise;
    end;
    for l := 0 to high(aCtx.fDevKeys) do
    begin
      aCtx.fDevKeys[l].Free;
      aCtx.fDevValues[l].Free;
    end;
    aCtx.fDevKeys := keys;
    aCtx.fDevValues := values;
    SetLength(aCtx.fIds, cap);
    aCtx.fCapacity := cap;
  end;
  // the scratch of a chunk
  if aRows > aCtx.fRows then
  begin
    aCtx.FreeScratch;
    NeedDevice(fCu, aCtx.fDevX, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fCu, aCtx.fDevH, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fCu, aCtx.fDevTmp, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fCu, aCtx.fDevQ, Int64(aRows) * c.Heads * c.HeadDim * 4);
    NeedDevice(fCu, aCtx.fDevAttn, Int64(aRows) * c.Heads * c.HeadDim * 4);
    NeedDevice(fCu, aCtx.fDevK, Int64(aRows) * kvw * 4);
    NeedDevice(fCu, aCtx.fDevV, Int64(aRows) * kvw * 4);
    NeedDevice(fCu, aCtx.fDevGate, Int64(aRows) *
      MaxPtrInt(c.Intermediate, c.ExpertIntermediate) * 4);
    NeedDevice(fCu, aCtx.fDevUp, Int64(aRows) *
      MaxPtrInt(c.Intermediate, c.ExpertIntermediate) * 4);
    NeedDevice(fCu, aCtx.fDevCos, Int64(aRows) * half * 2 * 4); // cos, then sin
    SetLength(aCtx.fIn, aRows * (c.Hidden + 2 * half));
    if fMoe then
    begin
      NeedDevice(fCu, aCtx.fDevRoute, Int64(aRows) * c.Experts * 4);
      NeedDevice(fCu, aCtx.fDevGather, Int64(aRows) * c.Hidden * 4);
      NeedDevice(fCu, aCtx.fDevExpOut, Int64(aRows) * c.Hidden * 4);
      NeedDevice(fCu, aCtx.fDevIdx, Int64(aRows) * c.ExpertsUsed * 4);
      NeedDevice(fCu, aCtx.fDevW, Int64(aRows) * c.ExpertsUsed * 4);
      SetLength(aCtx.fRoute, aRows * c.Experts);
      SetLength(aCtx.fChosen, aRows * c.ExpertsUsed);
      SetLength(aCtx.fChosenW, aRows * c.ExpertsUsed);
      SetLength(aCtx.fIdx, aRows * c.ExpertsUsed);
      SetLength(aCtx.fW, aRows * c.ExpertsUsed);
      SetLength(aCtx.fStart, c.Experts + 1);
      NeedDevice(fCu, aCtx.fDevKIds, Int64(c.ExpertsUsed) * 4);
      NeedDevice(fCu, aCtx.fDevKW, Int64(c.ExpertsUsed) * 4);
      NeedDevice(fCu, aCtx.fDevKGate, Int64(c.ExpertsUsed) * c.ExpertIntermediate * 4);
      NeedDevice(fCu, aCtx.fDevKUp, Int64(c.ExpertsUsed) * c.ExpertIntermediate * 4);
      NeedDevice(fCu, aCtx.fDevKDown, Int64(c.ExpertsUsed) * c.Hidden * 4);
      SetLength(aCtx.fKIds, c.ExpertsUsed);
      SetLength(aCtx.fKW, c.ExpertsUsed);
    end;
    aCtx.fRows := aRows;
  end;
  if aOutRows > 0 then
    NeedDevice(fCu, aCtx.fLogits, Int64(aOutRows) * c.VocabSize * 4);
end;

procedure TQwen3Cuda.EnqueueNorm(x: TInferCuBuffer; aRows, aCols: PtrInt;
  w: PSingleArray; y: TInferCuBuffer);
var
  px, pw, py: TCuDevicePtr;
  cols: cardinal;
  eps: single;
begin
  px := x.Ptr;
  pw := Vector(w, aCols);
  py := y.Ptr;
  cols := aCols;
  eps := fModel.Config.RmsEps;
  fCu.Enqueue(fNorm, [@px, @pw, @py, @cols, @eps], aRows, OPS_BLOCK);
end;

procedure TQwen3Cuda.EnqueueAddNorm(x, t: TInferCuBuffer; aRows, aCols: PtrInt;
  w: PSingleArray; y: TInferCuBuffer);
var
  px, pt, pw, py: TCuDevicePtr;
  cols: cardinal;
  eps: single;
begin
  px := x.Ptr;
  pt := t.Ptr;
  pw := Vector(w, aCols);
  py := y.Ptr;
  cols := aCols;
  eps := fModel.Config.RmsEps;
  fCu.Enqueue(fAddNorm, [@px, @pt, @pw, @py, @cols, @eps], aRows, OPS_BLOCK);
end;

// the device address of row aRow of aWidth floats in aBias, 0 for no bias
function BiasRow(aBias: TCuDevicePtr; aRow, aWidth: PtrInt): TCuDevicePtr;
begin
  if aBias = 0 then
    result := 0
  else
    result := aBias + TCuDevicePtr(Int64(aRow) * aWidth * 4);
end;

procedure TQwen3Cuda.EnqueueMoe(aCtx: TQwen3CuContext; aLayer, aRows: PtrInt);
var
  ly: ^TQwen3Layer;
  n, h, fi, k, ne, r, j, e, cnt, total: PtrInt;
  ph, pg, pe, pt, pidx, pw, pu, gb, ub, db: TCuDevicePtr;
  cols, count: cardinal;
  alpha, limit: single;
begin
  ly := @fModel.Layers[aLayer];
  n := aRows;
  h := fModel.Config.Hidden;
  fi := fModel.Config.ExpertIntermediate;
  k := fModel.Config.ExpertsUsed;
  ne := fModel.Config.Experts;
  alpha := fModel.Config.SwiGluAlpha;
  limit := fModel.Config.SwiGluLimit;
  // the biases of the experts (gpt-oss), expert e at row e
  gb := 0;
  ub := 0;
  db := 0;
  if ly^.ExpertGateBias <> nil then
    gb := Vector(ly^.ExpertGateBias, ne * fi);
  if ly^.ExpertUpBias <> nil then
    ub := Vector(ly^.ExpertUpBias, ne * fi);
  if ly^.ExpertDownBias <> nil then
    db := Vector(ly^.ExpertDownBias, ne * h);
  // the router logits down (waits for the layer so far), the choice on the host
  fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.Router, ne, aCtx.fDevRoute,
    ly^.RouterBias);
  aCtx.fDevRoute.Download(0, pointer(aCtx.fRoute), Int64(n) * ne * 4);
  for r := 0 to n - 1 do
    fModel.Route(@aCtx.fRoute[r * ne], @aCtx.fChosen[r * k], @aCtx.fChosenW[r * k]);
  // the token rows and weights of every expert, expert by expert, up at once
  total := 0;
  for e := 0 to ne - 1 do
  begin
    aCtx.fStart[e] := total;
    for r := 0 to n - 1 do
      for j := 0 to k - 1 do
        if aCtx.fChosen[r * k + j] = e then
        begin
          aCtx.fIdx[total] := r;
          aCtx.fW[total] := aCtx.fChosenW[r * k + j];
          inc(total);
        end;
  end;
  aCtx.fStart[ne] := total;
  aCtx.fDevIdx.Upload(0, pointer(aCtx.fIdx), Int64(total) * 4);
  aCtx.fDevW.Upload(0, pointer(aCtx.fW), Int64(total) * 4);
  pt := aCtx.fDevTmp.Ptr;
  count := n * h;
  fCu.Enqueue(fFill0, [@pt, @count], (count + OPS_BLOCK - 1) div OPS_BLOCK, OPS_BLOCK);
  ph := aCtx.fDevH.Ptr;
  pg := aCtx.fDevGather.Ptr;
  pe := aCtx.fDevExpOut.Ptr;
  pu := aCtx.fDevUp.Ptr;
  cols := h;
  for e := 0 to ne - 1 do
  begin
    cnt := aCtx.fStart[e + 1] - aCtx.fStart[e];
    if cnt = 0 then
      continue;
    pidx := aCtx.fDevIdx.Ptr + Int64(aCtx.fStart[e]) * 4;
    pw := aCtx.fDevW.Ptr + Int64(aCtx.fStart[e]) * 4;
    fCu.Enqueue(fGather, [@ph, @pidx, @pg, @cols], cnt, OPS_BLOCK);
    fBackend.EnqueueLinearAt(aCtx.fDevGather, cnt, h, ly^.ExpertGate[e].DType,
      fExpertGate[aLayer][e], fi, aCtx.fDevGate, BiasRow(gb, e, fi));
    fBackend.EnqueueLinearAt(aCtx.fDevGather, cnt, h, ly^.ExpertUp[e].DType,
      fExpertUp[aLayer][e], fi, aCtx.fDevUp, BiasRow(ub, e, fi));
    pg := aCtx.fDevGate.Ptr;
    count := cnt * fi;
    fCu.Enqueue(fSwiGlu, [@pg, @pu, @count, @alpha, @limit],
      (count + OPS_BLOCK - 1) div OPS_BLOCK, OPS_BLOCK);
    fBackend.EnqueueLinearAt(aCtx.fDevGate, cnt, fi, ly^.ExpertDown[e].DType,
      fExpertDown[aLayer][e], h, aCtx.fDevExpOut, BiasRow(db, e, h));
    fCu.Enqueue(fScatter, [@pe, @pidx, @pw, @pt, @cols], cnt, OPS_BLOCK);
    pg := aCtx.fDevGather.Ptr;
  end;
end;

procedure TQwen3Cuda.EnqueueMoeDevice(aCtx: TQwen3CuContext; aLayer: PtrInt);
var
  ly: ^TQwen3Layer;
  h, fi, k: PtrInt;
  pr, pids, pw, pg, pu, pd, pt, gb, ub, db: TCuDevicePtr;
  experts, used, norm, count: cardinal;
  alpha, limit: single;
begin
  ly := @fModel.Layers[aLayer];
  h := fModel.Config.Hidden;
  fi := fModel.Config.ExpertIntermediate;
  k := fModel.Config.ExpertsUsed;
  alpha := fModel.Config.SwiGluAlpha;
  limit := fModel.Config.SwiGluLimit;
  gb := 0;
  ub := 0;
  db := 0;
  if ly^.ExpertGateBias <> nil then
    gb := Vector(ly^.ExpertGateBias, fModel.Config.Experts * fi);
  if ly^.ExpertUpBias <> nil then
    ub := Vector(ly^.ExpertUpBias, fModel.Config.Experts * fi);
  if ly^.ExpertDownBias <> nil then
    db := Vector(ly^.ExpertDownBias, fModel.Config.Experts * h);
  // the router logits, the choice into fDevKIds / fDevKW - no host involved
  fBackend.EnqueueLinear(aCtx.fDevH, 1, h, ly^.Router, fModel.Config.Experts,
    aCtx.fDevRoute, ly^.RouterBias);
  pr := aCtx.fDevRoute.Ptr;
  pids := aCtx.fDevKIds.Ptr;
  pw := aCtx.fDevKW.Ptr;
  experts := fModel.Config.Experts;
  used := k;
  norm := ord(fModel.Config.NormTopK);
  fCu.Enqueue(fRoute, [@pr, @pids, @pw, @experts, @used, @norm], 1, 32);
  if fCheckRoute then
    CheckDeviceRoute(aCtx);
  // gate and up of the k experts from the same row, SwiGLU, down per slot
  fBackend.EnqueueLinearIds(aCtx.fDevH, h, ly^.ExpertGate[0].DType,
    fExpertGate[aLayer][0], fStrideGate[aLayer], fi, pids, k, false, aCtx.fDevKGate,
    gb);
  fBackend.EnqueueLinearIds(aCtx.fDevH, h, ly^.ExpertUp[0].DType,
    fExpertUp[aLayer][0], fStrideUp[aLayer], fi, pids, k, false, aCtx.fDevKUp, ub);
  pg := aCtx.fDevKGate.Ptr;
  pu := aCtx.fDevKUp.Ptr;
  count := k * fi;
  fCu.Enqueue(fSwiGlu, [@pg, @pu, @count, @alpha, @limit],
    (count + OPS_BLOCK - 1) div OPS_BLOCK, OPS_BLOCK);
  fBackend.EnqueueLinearIds(aCtx.fDevKGate, fi, ly^.ExpertDown[0].DType,
    fExpertDown[aLayer][0], fStrideDown[aLayer], h, pids, k, true, aCtx.fDevKDown,
    db);
  // the weighted sum in id order into fDevTmp
  pd := aCtx.fDevKDown.Ptr;
  pt := aCtx.fDevTmp.Ptr;
  count := h;
  fCu.Enqueue(fCombine, [@pd, @pw, @pt, @used, @count],
    (count + OPS_BLOCK - 1) div OPS_BLOCK, OPS_BLOCK);
end;

procedure TQwen3Cuda.CheckDeviceRoute(aCtx: TQwen3CuContext);
var
  k, j, t, id: PtrInt;
  v: single;
  d: double;
  same: boolean;
begin
  k := fModel.Config.ExpertsUsed;
  // the downloads wait for the route kernel
  aCtx.fDevRoute.Download(0, pointer(aCtx.fRoute), Int64(fModel.Config.Experts) * 4);
  aCtx.fDevKIds.Download(0, pointer(aCtx.fKIds), Int64(k) * 4);
  aCtx.fDevKW.Download(0, pointer(aCtx.fKW), Int64(k) * 4);
  fModel.Route(pointer(aCtx.fRoute), pointer(aCtx.fChosen), pointer(aCtx.fChosenW));
  // the host's choice by id, as the kernel writes it
  for j := 1 to k - 1 do
  begin
    id := aCtx.fChosen[j];
    v := aCtx.fChosenW[j];
    t := j;
    while (t > 0) and (aCtx.fChosen[t - 1] > id) do
    begin
      aCtx.fChosen[t] := aCtx.fChosen[t - 1];
      aCtx.fChosenW[t] := aCtx.fChosenW[t - 1];
      dec(t);
    end;
    aCtx.fChosen[t] := id;
    aCtx.fChosenW[t] := v;
  end;
  inc(fRouteChecks);
  same := true;
  for j := 0 to k - 1 do
    if aCtx.fChosen[j] <> aCtx.fKIds[j] then
      same := false;
  if not same then
  begin
    inc(fRouteMismatches);
    exit;
  end;
  for j := 0 to k - 1 do
  begin
    d := Abs(aCtx.fChosenW[j] - aCtx.fKW[j]);
    if not (d <= 1) then
      d := 1e30; // a NaN or an infinity of the device: never within a tolerance
    if d > fRouteMaxDiff then
      fRouteMaxDiff := d;
  end;
end;

procedure TQwen3Cuda.Chunk(aCtx: TQwen3CuContext; aIds: PIntegerArray;
  aEmbeds: PSingleArray; aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean;
  aNormed: PSingleArray);
var
  c: TQwen3Config;
  n, h, f, qw, kvw, half, past, l, outRows, hb, rb, blocks, splits, most,
    wsplits, last: PtrInt;
  ly: ^TQwen3Layer;
  p: PSingleArray;
  pq, pk, pv, pqn, pkn, pcs, pkc, pvc, pa, ppart, px, pt, psk: TCuDevicePtr;
  rows, heads, kvHeads, headDim, ppast, nsplits, hblock, rblock, count,
    norm, rope, window: cardinal;
  eps, alpha, limit: single;
  capture, device: boolean;
  qkrope, attention: TCuHandle;
begin
  c := fModel.Config;
  n := aCount;
  h := c.Hidden;
  f := c.Intermediate;
  qw := c.Heads * c.HeadDim;
  kvw := c.KvHeads * c.HeadDim;
  half := c.HeadDim shr 1;
  past := aCtx.fLength;
  outRows := 0;
  if aLogits <> nil then
    if aAllLogits then
      outRows := n
    else
      outRows := 1;
  Prepare(aCtx, past + n, n, outRows);
  // the kernels of the cache format of aCtx (set by Prepare)
  qkrope := fQkRope;
  attention := fAttention;
  if aCtx.fKvF16 then
  begin
    qkrope := fQkRope16;
    attention := fAttention16;
  end;
  // the attention blocks: hb heads of one KV head (hb divides the group)
  // times rb rows, the keys of a row cut into splits ranges unless there are
  // enough blocks without
  hb := c.Heads div c.KvHeads;
  while (hb > ATTENTION_QUERIES) or (c.Heads div c.KvHeads mod hb <> 0) do
    dec(hb);
  rb := ATTENTION_QUERIES div hb;
  blocks := (n + rb - 1) div rb;
  splits := 1;
  if fSplitKeys > 0 then
    splits := (past + n + fSplitKeys - 1) div fSplitKeys;
  most := SPLIT_GROUPS div ((c.Heads div hb) * blocks);
  if splits > most then
    splits := most;
  if splits < 1 then
    splits := 1;
  // a layer with a sliding window: one token row per block, so all its
  // queries share one key range - the others' rows would sum values outside
  // their window with a weight of 0, and 0 times a NaN or infinite value is
  // NaN
  wsplits := splits;
  if n > 1 then
  begin
    wsplits := 1;
    if fSplitKeys > 0 then
      wsplits := (past + n + fSplitKeys - 1) div fSplitKeys;
    most := SPLIT_GROUPS div ((c.Heads div hb) * n);
    if wsplits > most then
      wsplits := most;
    if wsplits < 1 then
      wsplits := 1;
  end;
  ppart := 0;
  if (splits > 1) or (wsplits > 1) then
  begin
    NeedDevice(fCu, aCtx.fPart, Int64(c.Heads) * n * MaxPtrInt(splits, wsplits) *
      (c.HeadDim + 2) * 4);
    ppart := aCtx.fPart.Ptr;
  end;
  // the inputs: embeddings, then the rotary factors (cos, then sin)
  p := pointer(aCtx.fIn);
  if aIds <> nil then
    fModel.Embed(aIds, n, p)
  else
    MoveFast(aEmbeds^[0], p^[0], n * h * SizeOf(single));
  fModel.Rotary(past, n, @p^[n * h], @p^[n * h + n * half]);
  aCtx.fDevX.Upload(0, p, Int64(n) * h * 4);
  aCtx.fDevCos.Upload(0, @p^[n * h], Int64(n) * half * 2 * 4);
  rows := n;
  heads := c.Heads;
  kvHeads := c.KvHeads;
  headDim := c.HeadDim;
  ppast := past;
  nsplits := splits;
  hblock := hb;
  rblock := rb;
  eps := c.RmsEps;
  pq := aCtx.fDevQ.Ptr;
  pk := aCtx.fDevK.Ptr;
  pv := aCtx.fDevV.Ptr;
  pcs := aCtx.fDevCos.Ptr;
  pa := aCtx.fDevAttn.Ptr;
  // a single token: the whole step recorded as a graph, launched at once -
  // with experts only when they are chosen on the device; a RouteTape needs
  // the host's choice (TDecoderModel.Route), CheckRoute waits in every layer
  device := fMoe and (n = 1) and fRouteOnDevice and fDeviceRoutable and
    (fModel.RouteTape = nil);
  capture := fUseGraphs and (n = 1) and (not fMoe or device) and not fCheckRoute;
  if capture then
    fCu.CaptureBegin;
  try
    // the input norm of layer 0; every later one comes with the residual of the
    // layer before (addnorm)
    EnqueueNorm(aCtx.fDevX, n, h, fModel.Layers[0].InputNorm, aCtx.fDevH);
    for l := 0 to c.Layers - 1 do
    begin
      ly := @fModel.Layers[l];
      // attention block: the projections, per head the norms, rotary positions
      // and the store of k and v into the cache, the attention, the output
      fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.Q, qw, aCtx.fDevQ, ly^.QBias);
      fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.K, kvw, aCtx.fDevK, ly^.KBias);
      fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.V, kvw, aCtx.fDevV, ly^.VBias);
      // without head norms (Qwen2) the kernel reads no qn or kn
      norm := ord(ly^.QNorm <> nil);
      rope := ord(not c.Spec[l].NoRope);
      window := c.Spec[l].Window;
      nsplits := splits;
      rblock := rb;
      if window > 0 then
      begin
        nsplits := wsplits;
        rblock := 1;
      end;
      pqn := 0;
      pkn := 0;
      if ly^.QNorm <> nil then
      begin
        pqn := Vector(ly^.QNorm, c.HeadDim);
        pkn := Vector(ly^.KNorm, c.HeadDim);
      end;
      pkc := aCtx.fDevKeys[l].Ptr;
      pvc := aCtx.fDevValues[l].Ptr;
      // the attention sinks of gpt-oss, 0 without
      psk := 0;
      if ly^.Sinks <> nil then
        psk := Vector(ly^.Sinks, c.Heads);
      fCu.Enqueue(qkrope, [@pq, @pk, @pv, @pqn, @pkn, @pcs, @pkc, @pvc, @rows,
        @heads, @kvHeads, @headDim, @ppast, @eps, @norm, @rope],
        c.Heads + c.KvHeads, OPS_BLOCK, n);
      fCu.Enqueue(attention, [@pq, @pkc, @pvc, @pa, @ppart, @rows, @heads,
        @headDim, @kvHeads, @ppast, @nsplits, @hblock, @rblock, @window, @psk],
        c.Heads div hb, OPS_BLOCK, (n + PtrInt(rblock) - 1) div PtrInt(rblock),
        nsplits);
      if nsplits > 1 then
        fCu.Enqueue(fMerge, [@ppart, @pa, @rows, @heads, @headDim, @nsplits, @psk],
          c.Heads, OPS_BLOCK, n);
      fBackend.EnqueueLinear(aCtx.fDevAttn, n, qw, ly^.O, h, aCtx.fDevTmp, ly^.OBias);
      // residual and the feed-forward pre-norm, gate and up, SwiGLU, down
      EnqueueAddNorm(aCtx.fDevX, aCtx.fDevTmp, n, h, ly^.PostNorm, aCtx.fDevH);
      if c.Spec[l].Ffn = dfMoe then
      begin
        if device then
          EnqueueMoeDevice(aCtx, l)
        else
          EnqueueMoe(aCtx, l, n);
      end
      else
      begin
        fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.Gate, f, aCtx.fDevGate);
        fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.Up, f, aCtx.fDevUp);
        px := aCtx.fDevGate.Ptr;
        pt := aCtx.fDevUp.Ptr;
        count := n * f;
        alpha := c.SwiGluAlpha;
        limit := c.SwiGluLimit;
        fCu.Enqueue(fSwiGlu, [@px, @pt, @count, @alpha, @limit],
          (count + OPS_BLOCK - 1) div OPS_BLOCK, OPS_BLOCK);
        fBackend.EnqueueLinear(aCtx.fDevGate, n, f, ly^.Down, h, aCtx.fDevTmp);
      end;
      // the residual, and the input norm of the next layer
      if l < c.Layers - 1 then
        EnqueueAddNorm(aCtx.fDevX, aCtx.fDevTmp, n, h, fModel.Layers[l + 1].InputNorm,
          aCtx.fDevH)
      else
      begin
        px := aCtx.fDevX.Ptr;
        pt := aCtx.fDevTmp.Ptr;
        count := n * h;
        fCu.Enqueue(fAdd, [@px, @pt, @px, @count],
          (count + OPS_BLOCK - 1) div OPS_BLOCK, OPS_BLOCK);
      end;
    end;
    last := 0;
    if (outRows > 0) or (aNormed <> nil) then
    begin
      if outRows <= 1 then
      begin
        // the last row only: to the start of the scratch, its norm, the head
        aCtx.fDevX.CopyTo(Int64(n - 1) * h * 4, aCtx.fDevTmp, 0, Int64(h) * 4);
        EnqueueNorm(aCtx.fDevTmp, 1, h, fModel.FinalNorm, aCtx.fDevH);
      end
      else
      begin
        EnqueueNorm(aCtx.fDevX, n, h, fModel.FinalNorm, aCtx.fDevH);
        last := n - 1;
      end;
      if outRows > 0 then
        fBackend.EnqueueLinear(aCtx.fDevH, outRows, h, fModel.Head, c.VocabSize,
          aCtx.fLogits);
    end;
    if capture then
      fCu.CaptureEnd(aCtx.fGraph);
  except
    fCu.CaptureAbort;
    raise;
  end;
  // waits for every kernel before it (one stream)
  if outRows > 0 then
    aCtx.fLogits.Download(0, aLogits, Int64(outRows) * c.VocabSize * 4);
  if aNormed <> nil then
    aCtx.fDevH.Download(Int64(last) * h * 4, aNormed, Int64(h) * 4);
  if (outRows = 0) and (aNormed = nil) then
    fCu.Synchronize;
  if aIds <> nil then
    MoveFast(aIds^[0], aCtx.fIds[past], n * SizeOf(integer))
  else
    FillCharFast(aCtx.fIds[past], n * SizeOf(integer), $ff); // -1: no id
  aCtx.fLength := past + n;
end;

procedure TQwen3Cuda.ForwardRows(aCtx: TQwen3Context; aIds: PIntegerArray;
  aEmbeds: PSingleArray; aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean;
  aNormed: PSingleArray);
var
  i, n, done, vocab, h: PtrInt;
  last: boolean;
  ids: PIntegerArray;
  embeds, normed: PSingleArray;
  ctx: TQwen3CuContext;
begin
  if (aCtx = nil) or (aCtx.Device <> self) then
    EInferError.RaiseUtf8('%.Forward: no context of this instance', [self]);
  ctx := TQwen3CuContext(aCtx); // only NewContext gives this device
  if aCount <= 0 then
    EInferError.RaiseUtf8('%.Forward: no tokens', [self]);
  if aCount > fModel.Config.MaxPositions - ctx.fLength then
    EInferError.RaiseUtf8('%.Forward: % + % tokens exceed the % positions',
      [self, ctx.fLength, aCount, fModel.Config.MaxPositions]);
  if (aLogits <> nil) and not fModel.HasHead then
    EInferError.RaiseUtf8('%.Forward: no head for logits (the normed state only)',
      [self]);
  vocab := fModel.Config.VocabSize;
  h := fModel.Config.Hidden;
  if aIds <> nil then
    for i := 0 to aCount - 1 do
      if (aIds^[i] < 0) or (aIds^[i] >= vocab) then
        EInferError.RaiseUtf8('%.Forward: token id % outside the vocabulary',
          [self, aIds^[i]]);
  fBackend.GpuLock;
  try
    // the backend forgot the weights (ClearWeights): up again
    if fEpoch <> fBackend.WeightEpoch then
      UploadWeights;
    done := 0;
    while done < aCount do
    begin
      n := ChunkRows(ctx.fLength, aCount - done, aAllLogits and (aLogits <> nil));
      last := done + n = aCount;
      ids := nil;
      embeds := nil;
      if aIds <> nil then
        ids := @aIds^[done]
      else
        embeds := @aEmbeds^[done * h];
      normed := nil;
      if last then
        normed := aNormed;
      if aAllLogits and (aLogits <> nil) then
        Chunk(ctx, ids, embeds, n, @aLogits^[done * vocab], true, normed)
      else if last then
        Chunk(ctx, ids, embeds, n, aLogits, false, normed)
      else
        Chunk(ctx, ids, embeds, n, nil, false, nil);
      inc(done, n);
    end;
  finally
    fBackend.GpuUnLock;
  end;
end;

procedure TQwen3Cuda.Forward(aCtx: TQwen3Context; aIds: PIntegerArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean);
begin
  if (aIds = nil) and (aCount > 0) then
    EInferError.RaiseUtf8('%.Forward: no ids', [self]);
  if not fModel.HasEmbedding then
    EInferError.RaiseUtf8('%.Forward: no token embedding (ForwardEmbeds only)', [self]);
  ForwardRows(aCtx, aIds, nil, aCount, aLogits, aAllLogits, nil);
end;

procedure TQwen3Cuda.ForwardEmbeds(aCtx: TQwen3Context; aEmbeds: PSingleArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean; aNormed: PSingleArray);
begin
  if (aEmbeds = nil) and (aCount > 0) then
    EInferError.RaiseUtf8('%.ForwardEmbeds: no rows', [self]);
  ForwardRows(aCtx, nil, aEmbeds, aCount, aLogits, aAllLogits, aNormed);
end;

function TQwen3Cuda.Reuse(aCtx: TQwen3Context; aIds: PIntegerArray;
  aCount: PtrInt): PtrInt;
begin
  if (aCtx = nil) or (aCtx.Device <> self) then
    EInferError.RaiseUtf8('%.Reuse: no context of this instance', [self]);
  result := 0;
  if aCount > 0 then
    result := aCtx.Common(aIds, aCount - 1);
  aCtx.Truncate(result);
end;

function TQwen3Cuda.Greedy(aCtx: TQwen3Context; const aPrompt: array of integer;
  aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
begin
  if (aCtx = nil) or (aCtx.Device <> self) then
    EInferError.RaiseUtf8('%.Greedy: no context of this instance', [self]);
  // the model hands a context of this device to Forward here
  result := fModel.Greedy(aCtx, aPrompt, aMaxNew, aStop);
end;

end.
