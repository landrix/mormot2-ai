/// native inference engine - the decoder (causal language model) of every family
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - what the families share: the settings with the kind of every layer, the
//   KV cache of a sequence (TDecoderContext), the weights read from a
//   safetensors or a GGUF file, the forward pass over the layers, the hand-over
//   to a device (GPU) - a family (mormot.ai.infer.model.qwen3) reads its own
//   settings and tensor names and registers its class
// - the layers today: pre-norm self-attention (RMSNorm, Q/K/V projections -
//   with a bias where the family has one -, RMSNorm of every Q and K head where
//   it has one, rotary positions, grouped-query causal attention, output
//   projection) and a pre-norm SwiGLU feed-forward block,
//   both with a residual; a final RMSNorm and the language-model head (tied to
//   the embedding or its own matrix) - TDecoderLayerSpec names the kind of
//   every layer, so another mixer (linear attention) docks in there; an
//   attention layer may see a sliding window of keys only (Window) and may go
//   without rotary positions (NoRope, NoPE); the feed-forward block is dense or
//   a mixture of experts (dfMoe: a softmax router picks the top k of the
//   experts per token, their SwiGLU outputs are summed with the router
//   weights, renormed or not)
// - what gpt-oss adds (TDecoderModel.Extras): attention sinks (a learned
//   logit per head in every softmax), a bias on the output projection, the
//   router and the experts, the clamped SwiGLU (SwiGluLimit); and YaRN rotary
//   positions (RopeFactor: frequencies between interpolation and extrapolation,
//   cos and sin scaled by an attention factor), computed in Rotary for every
//   executor
// - a sliding-window layer keeps the keys a next token may see, not the whole
//   sequence: its cache rows start at a base position that moves on when the
//   buffer is full (FullCache keeps every key and only masks, the reference);
//   a cached prefix is then a valid restore point only while its window is
//   still there (TDecoderContext.Restorable) - Reuse computes again otherwise
// - weights: F32 is used in place (memory-mapped), BF16 widened to F32 once at
//   load (exact: BF16 is the upper half of an F32); or every matrix quantized
//   to Q8_0 (or Q4_0) at load, byte for byte as a GGUF writer does it; the
//   embedding rows are then read per token; or a llama.cpp GGUF file: its
//   blocks used in place
// - a decoder may also be one stack of a larger checkpoint (CreateStack: its
//   settings from an object inside config.json, its tensors under a prefix,
//   its own embedding and head or none of either), and take embedding rows
//   instead of token ids (ForwardEmbeds, which also hands back the normed last
//   row, on a device too) - the language model of a speech, vision or
//   audio model; a model forgets its weights on a GPU backend when it is
//   freed (TInferBackend.ForgetWeights)
// - arithmetic: matrix products through the backend; norms, rotary positions
//   and attention in double, stored as single. Rotary angles follow the float32
//   steps of the reference (inverse frequency and angle rounded to single)
// - the model is immutable after loading and may serve several threads; every
//   sequence brings its own TDecoderContext with its KV cache and scratch buffers
unit mormot.ai.infer.decoder;

{
  *****************************************************************************

    - TDecoderConfig: the hyper-parameters and the kind of every layer
    - TDecoderContext: KV cache and scratch buffers of one sequence
    - TDecoderModel: weights validated at load, Forward over new tokens, Greedy
    - TDecoderDevice: computes a TDecoderModel elsewhere (a GPU)
    - TDecoderRouteTape: the expert choices of one run handed to another
    - RegisterDecoderModel / NewDecoderModel: a family by its model type

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
  mormot.core.variants,
  mormot.ai.infer.tensor,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.infer.loader.gguf,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.sampler; // InferArgMax

type
  /// how a layer mixes the tokens of the sequence
  // - dmAttention: causal grouped-query attention over every cached token
  TDecoderMixer = (
    dmAttention);

  /// the feed-forward block of a layer
  // - dfDense: one SwiGLU block (gate, up, down)
  // - dfMoe: a mixture of experts - a router (softmax over Experts logits)
  // picks ExpertsUsed SwiGLU blocks of ExpertIntermediate per token, their
  // outputs are summed with the router probabilities (renormed to a sum of 1
  // when NormTopK)
  TDecoderFfn = (
    dfDense,
    dfMoe);

  /// the kind of one layer
  // - all zero (the default of a new array) is attention over every cached
  // token with rotary positions and a dense block
  TDecoderLayerSpec = record
    Mixer: TDecoderMixer;
    Ffn: TDecoderFfn;
    /// a sliding window of dmAttention: a token sees the last Window tokens,
    // itself included (key j for query i when i - Window < j <= i, as
    // transformers masks it); 0: every token before it
    Window: PtrInt;
    /// no rotary positions on Q and K in this layer (NoPE)
    NoRope: boolean;
  end;
  TDecoderLayerSpecDynArray = array of TDecoderLayerSpec;

  /// decoder hyper-parameters
  TDecoderConfig = record
    VocabSize: PtrInt;
    Hidden: PtrInt;
    Layers: PtrInt;
    Heads: PtrInt;
    KvHeads: PtrInt;
    HeadDim: PtrInt;
    Intermediate: PtrInt;
    MaxPositions: PtrInt;
    RmsEps: double;
    RopeTheta: double;
    TiedEmbeddings: boolean;
    /// the experts of a dfMoe layer (0 without one)
    Experts: PtrInt;
    /// the experts a token goes through (top k)
    ExpertsUsed: PtrInt;
    /// the hidden width of one expert
    ExpertIntermediate: PtrInt;
    /// the weights of the chosen experts renormed to a sum of 1
    NormTopK: boolean;
    /// > 0: the clamped SwiGLU of gpt-oss in every feed-forward block, with
    // SwiGluAlpha (TInferBackend.SwiGlu); 0: the plain one
    SwiGluLimit: single;
    SwiGluAlpha: single;
    /// > 0: YaRN rotary positions (rope_type yarn) with this scaling factor;
    // 0: plain RoPE (the other Rope* fields unused)
    RopeFactor: double;
    /// YaRN: original_max_position_embeddings, beta_fast, beta_slow, truncate
    RopeOriginal: PtrInt;
    RopeBetaFast, RopeBetaSlow: double;
    RopeTruncate: boolean;
    /// YaRN: the factor of cos and sin (attention_factor); 0 derives it from
    // RopeFactor as transformers does (0.1 ln factor + 1)
    RopeAttention: double;
    /// the kind of every layer ([Layers]) - a family that leaves it empty
    // gets attention and a dense block in every layer
    Spec: TDecoderLayerSpecDynArray;
  end;

  /// weights by expert
  TInferWeightDynArray = array of TInferWeight;

  /// the weights of one decoder layer - F32 views
  // - Gate, Up and Down for a dfDense layer; Router ([Experts, Hidden], F32)
  // and ExpertGate, ExpertUp, ExpertDown ([Experts] each) for a dfMoe one
  // - QNorm and KNorm ([HeadDim]): the RMSNorm of every Q and K head (Qwen3),
  // nil without one; QBias, KBias and VBias ([Heads or KvHeads * HeadDim]):
  // nil without a bias (Qwen2 has one); OBias ([Hidden]) likewise (gpt-oss)
  // - Sinks ([Heads]): the attention sinks (gpt-oss), nil without
  // - RouterBias ([Experts]), ExpertGateBias and ExpertUpBias ([Experts *
  // ExpertIntermediate], expert by expert) and ExpertDownBias ([Experts *
  // Hidden]): the biases of a dfMoe layer (gpt-oss), nil without
  TDecoderLayer = record
    Q, K, V, O, Gate, Up, Down: TInferWeight;
    QNorm, KNorm, InputNorm, PostNorm: PSingleArray;
    QBias, KBias, VBias, OBias: PSingleArray;
    Sinks: PSingleArray;
    Router: TInferWeight;
    RouterBias: PSingleArray;
    ExpertGate, ExpertUp, ExpertDown: TInferWeightDynArray;
    ExpertGateBias, ExpertUpBias, ExpertDownBias: PSingleArray;
  end;
  TDecoderLayerDynArray = array of TDecoderLayer;

  /// what a model computes beyond attention, RMSNorm and a plain SwiGLU
  // (TDecoderModel.Extras): a device refuses a model with one it lacks
  // - dxSinks: attention sinks; dxOutputBias: a bias on the attention output;
  // dxRouterBias, dxExpertBias: biases of a mixture of experts; dxSwiGluLimit:
  // the clamped SwiGLU of gpt-oss
  TDecoderExtra = (
    dxSinks,
    dxOutputBias,
    dxRouterBias,
    dxExpertBias,
    dxSwiGluLimit);
  TDecoderExtras = set of TDecoderExtra;

  /// called with intermediate states of a Forward, for diagnosis and tests
  // - aName: 'embed' (input of layer 0), 'layerNN' (output of layer NN),
  // 'layer00_attn' / 'layer00_mlp' (attention and MLP output of layer 0, before
  // the residual), 'norm' (final RMSNorm); aValues is [aRows, aCols] and only
  // valid during the call
  TOnDecoderState = procedure(const aName: RawUtf8; aValues: PSingleArray;
    aRows, aCols: PtrInt) of object;

  TDecoderDevice = class;

  /// what TDecoderModel.Route does with its TDecoderRouteTape
  // - drmOff: routes from the logits; drmRecord: routes from the logits and
  // appends the choice; drmReplay: routes from the logits, compares, and hands
  // out the next recorded choice instead
  TDecoderRouteMode = (
    drmOff,
    drmRecord,
    drmReplay);

  /// the expert choices (ids and weights) of a run, recorded token by token
  // and layer by layer, handed in the same order to another run of the same
  // tokens - for tests: a device given the CPU's choice computes the same
  // experts, so its logits differ by arithmetic only, not by a near tie of the
  // router; the choice it would have made is compared (Differs, DiffersGap)
  // - both runs must route the tokens in the same order: layer by layer, the
  // rows of a chunk in order - so the same chunks (the CPU takes all tokens
  // of a Forward at once, a device up to its chunk size: a short prompt)
  // - not thread-safe: one sequence at a time on its model
  TDecoderRouteTape = class
  protected
    fMode: TDecoderRouteMode;
    fChosen: TIntegerDynArray;
    fWeights: TSingleDynArray;
    fCount, fPos, fRoutes, fDiffers: PtrInt; // in routed tokens
    fDiffersGap: double;
    procedure Put(aChosen: PIntegerArray; aWeights: PSingleArray; aUsed: PtrInt);
    procedure Take(aLogits: PSingleArray; aChosen: PIntegerArray;
      aWeights: PSingleArray; aExperts, aUsed: PtrInt);
  public
    /// forget the recorded choices and the counters
    procedure Clear;
    /// drmRecord before the reference run, drmReplay before the other one -
    // a recording starts over once everything recorded was replayed
    property Mode: TDecoderRouteMode
      read fMode write fMode;
    /// routed tokens recorded and not yet replayed
    function Pending: PtrInt;
    /// tokens replayed since Clear
    property Routes: PtrInt
      read fRoutes;
    /// replayed tokens whose own choice (from the logits of the replaying
    // run) was another set of experts than the recorded one
    property Differs: PtrInt
      read fDiffers;
    /// the largest gap, among Differs, between the k-th highest router logit
    // of the replaying run and the highest one it left out: a near tie gives
    // a small gap, a wrong routing a large one
    property DiffersGap: double
      read fDiffersGap;
  end;

  /// the KV cache and scratch buffers of one sequence
  // - one per concurrent sequence; may be reused for another sequence after
  // Reset, and for another model of the same shape
  // - remembers the id at every cached position: a prompt that starts with
  // the same tokens keeps their keys and values (Common, then Truncate) - the
  // cache of position i depends on the tokens 0..i only; a position fed an
  // embedding row (ForwardEmbeds) has no id: -1, which Common never matches,
  // so no common prefix reaches past it
  // - a layer with a sliding window keeps the rows from Base[layer] on, enough
  // for the next token (FullCache: all of them); going back is only possible
  // to a length whose window is still there (Restorable)
  TDecoderContext = class
  protected
    fDevice: TDecoderDevice; // nil: the keys and values are in the arrays below
    fLength, fCapacity: PtrInt;
    fKvWidth, fLayers: PtrInt;
    fOwner: integer; // serial of the model whose keys and values are cached
    fFullCache: boolean;
    fKeys, fValues: array of TSingleDynArray; // per layer [rows, kvWidth]
    // per layer: the sliding window (0: none, or kept in full) and the
    // position of row 0 of fKeys / fValues (0 without a window)
    fWindow, fBase: array of PtrInt;
    fIds: TIntegerDynArray; // [capacity] the token at every cached position
    fX, fH, fQ, fK, fV, fAttn, fTmp, fGate, fUp: TSingleDynArray;
    fCos, fSin: TSingleDynArray; // [new tokens, HeadDim / 2] rotary factors
    // a dfMoe layer: router logits [new tokens, Experts], the chosen experts
    // and their weights [new tokens, ExpertsUsed], the rows of one expert
    // (gathered input, output, row index, weight)
    fRoute, fChosenW, fGather, fExpertOut, fRowW: TSingleDynArray;
    fChosen, fRow: TIntegerDynArray;
    function GetId(aPosition: PtrInt): integer;
    procedure SetFullCache(aValue: boolean);
  public
    /// forget the cached tokens - the buffers stay allocated
    procedure Reset;
    /// keep the first aLength cached tokens, forget the others
    // - raises EInferError unless 0 <= aLength <= Length and Restorable(aLength)
    procedure Truncate(aLength: PtrInt);
    /// true when the cache can go back to its first aLength tokens: every
    // sliding window still holds the keys position aLength sees (always for 0
    // and Length, and without a sliding window)
    function Restorable(aLength: PtrInt): boolean;
    /// how many of the first tokens of aIds[0..aCount-1] the cache holds:
    // the length of the common prefix
    function Common(aIds: PIntegerArray; aCount: PtrInt): PtrInt;
    /// the longest common prefix with aIds[0..aCount-1] that the cache can go
    // back to (Common, then 0 unless Restorable)
    function Usable(aIds: PIntegerArray; aCount: PtrInt): PtrInt;
    /// the cache rows layer aLayer holds allocated (0 before the first
    // Forward) - a sliding window keeps about twice its size
    function CachedRows(aLayer: PtrInt): PtrInt;
    /// the position of the first cache row of layer aLayer (0 unless its
    // sliding window has moved on)
    function CachedBase(aLayer: PtrInt): PtrInt;
    /// keep the keys of every token also in sliding-window layers, which then
    // only mask (the reference, and every length restorable) - set it on an
    // empty context; false by default
    property FullCache: boolean
      read fFullCache write SetFullCache;
    /// tokens in the KV cache = position of the next token
    property Length: PtrInt
      read fLength;
    /// the token cached at aPosition (0 .. Length - 1), -1 where an
    // embedding row was fed (ForwardEmbeds)
    property Id[aPosition: PtrInt]: integer
      read GetId;
    /// the device that keeps the keys and values of this context (a context
    // of TDecoderDevice.NewDeviceContext), nil for one of TDecoderModel
    property Device: TDecoderDevice
      read fDevice;
  end;

  /// a causal language model over a safetensors folder or a GGUF file
  // - abstract: a family (TQwen3Model) reads its settings and maps its tensor
  // names, then registers its class (RegisterDecoderModel)
  TDecoderModel = class
  protected
    fSerial: integer; // unique per instance: an address may be reused after Free
    fDevice: TDecoderDevice;
    fConfig: TDecoderConfig;
    fFile: TSafeTensorsFile; // Create: a Hugging Face folder
    fGguf: TGgufFile; // CreateFromGguf: one llama.cpp file
    fBackend: TInferBackend;
    fOwned: array of TSingleDynArray; // BF16 tensors widened at load
    fOwnedQ8: array of array of TInferBlockQ8_0; // matrices quantized at load
    fOwnedQ4: array of array of TInferBlockQ4_0;
    fOwnedMx: array of array of TInferBlockMxfp4; // repacked MXFP4 experts
    fWeights: TInferDType;
    fEmbedSrc: PInferTensor; // the embedding as stored: read per token (nil: none)
    // CreateStack: the object of config.json (dotted path), the prefix that
    // replaces "model." in every tensor name, the tensors of the embedding
    // ('' = none) and of the head
    fStackConfig, fStackPrefix, fStackEmbed, fStackHead: RawUtf8;
    fRouteTape: TDecoderRouteTape;
    fEmbed, fHead: TInferWeight;
    fNorm: PSingleArray;
    fLayer: TDecoderLayerDynArray;
    fInvFreq: TSingleDynArray;
    fRopeScale: single; // YaRN: the factor of cos and sin (1 without)
    // the conversion running on Backend.RunItems (ConvertRows, WidenTensor)
    fCvSrc: pointer;
    fCvSrcType, fCvType: TInferDType;
    fCvCols, fCvOwned, fCvCount: PtrInt;
    fCvDest: PSingleArray;
    fCvName: RawUtf8;
    procedure ConvertItem(aItem: integer);
    procedure WidenItem(aItem: integer);
    /// the aRows rows of aSrc (aCols values of aSrcType each) quantized into
    // the owned matrix aOwned of aType, the rows on the backend's threads
    procedure ConvertRows(aSrc: pointer; aSrcType: TInferDType; aRows, aCols: PtrInt;
      aType: TInferDType; aOwned: PtrInt; const aName: RawUtf8);
    procedure Init(aBackend: TInferBackend);
    /// read config.json into aDoc and return the object of the settings:
    // aDoc itself (its model_type checked against ModelType), or for a stack
    // the object at its path (the model_type of a stack is not checked)
    function ReadConfig(const aConfigFile: TFileName;
      var aDoc: TDocVariantData): PDocVariantData;
    /// the name in the file of a tensor of the Hugging Face layout - for a
    // stack under its prefix, '' when the stack has no such tensor
    function SourceName(const aName: RawUtf8): RawUtf8;
    /// the tensors of the file whose name starts with aPrefix
    function TensorCountOf(const aPrefix: RawUtf8): PtrInt;
    /// Forward of aIds or of aEmbeds (the other one nil)
    procedure ForwardRows(aCtx: TDecoderContext; aIds: PIntegerArray;
      aEmbeds: PSingleArray; aCount: PtrInt; aLogits: PSingleArray;
      aAllLogits: boolean; const aOnState: TOnDecoderState; aNormed: PSingleArray);
    /// fConfig from a Hugging Face config.json - raises EInferError for a
    // setting the family does not support; ends with CheckConfig
    procedure LoadConfig(const aConfigFile: TFileName); virtual; abstract;
    /// fConfig from the metadata of fGguf - as LoadConfig
    procedure LoadGgufConfig; virtual; abstract;
    /// bound every setting, fill an empty Spec, the inverse rotary frequencies
    procedure CheckConfig(const aSource: TFileName);
    /// fInvFreq and fRopeScale of YaRN, from the plain frequencies
    procedure YarnFrequencies;
    /// fEmbedSrc, fEmbed, fNorm, fHead and fLayer from the file
    procedure LoadWeights; virtual; abstract;
    /// the llama.cpp name of a Hugging Face tensor name, '' if there is none
    function GgufName(const aName: RawUtf8): RawUtf8; virtual; abstract;
    /// a tensor by its Hugging Face name - in a GGUF file by llama.cpp's name
    function Find(const aName: RawUtf8): PInferTensor;
    function TensorCount: PtrInt;
    function Checked(const aName: RawUtf8; const aDims: array of Int64;
      aBlocks: boolean = false): PInferTensor;
    function Weight(const aName: RawUtf8; const aDims: array of Int64): PSingleArray;
    function Matrix(const aName: RawUtf8; aRows, aCols: Int64;
      aType: TInferDType): TInferWeight;
    /// the experts of a GGUF tensor [aExperts, aRows, aCols] (by its llama.cpp
    // name): one view per expert into the blocks as stored - Q8_0, Q4_0, Q4_K
    // or Q6_K
    function GgufExperts(const aGgufName: RawUtf8;
      aExperts, aRows, aCols: Int64): TInferWeightDynArray;
    /// a BF16 or F16 tensor widened to F32, kept with the model
    function WidenTensor(t: PInferTensor; const aName: RawUtf8): PSingleArray;
    /// aValues kept with the model (moved: aValues is nil afterwards)
    function Own(var aValues: TSingleDynArray): PSingleArray;
    /// aBlocks zeroed MXFP4 blocks kept with the model - for a family that
    // repacks the blocks of a checkpoint
    function OwnMxfp4(aBlocks: PtrInt): PInferBlockMxfp4Array;
    /// a matrix of aType from the F32 rows of aSrc ([aRows, aCols]) - for a
    // family that reorders a tensor first: idtF32 keeps aSrc with the model
    // (moved, nil afterwards), Q8_0 or Q4_0 quantize it row by row as Matrix
    // does (aSrc left as it is, for the next matrix)
    function MatrixOf(var aSrc: TSingleDynArray; aRows, aCols: Int64;
      aType: TInferDType; const aName: RawUtf8): TInferWeight;
    /// row aRow of aSrc (rows of aCols values of aSrcType) quantized into the
    // blocks of that row of the owned matrix aOwned of aType, one block of 32
    // after the other - raises for a NaN, an infinity or a weight beyond the
    // range
    procedure QuantizeRow(aSrc: pointer; aSrcType: TInferDType; aCols: PtrInt;
      aType: TInferDType; aOwned, aRow: PtrInt; const aName: RawUtf8);
    /// raise unless every FP16 scale of the blocks of aTensor is finite
    procedure CheckBlockScales(aTensor: PInferTensor; const aName: RawUtf8);
    procedure Grow(aCtx: TDecoderContext; aTokens, aNew: PtrInt);
    /// room in the rows of sliding-window layer aLayer for aRows new tokens
    // at aPast: older keys than the window of aPast moved out first
    procedure Slide(aCtx: TDecoderContext; aLayer, aPast, aRows: PtrInt);
    procedure RopeFactors(aCtx: TDecoderContext; aRows, aFirstPos: PtrInt);
    procedure Attention(aCtx: TDecoderContext; aLayer, aRows: PtrInt);
    /// the dmAttention mixer of layer aLayer over aRows new tokens, its output
    // added to fX
    procedure AttentionBlock(aCtx: TDecoderContext; aLayer, aRows, aPast: PtrInt;
      const aOnState: TOnDecoderState);
    /// the dfDense block of layer aLayer, its output added to fX
    procedure DenseBlock(aCtx: TDecoderContext; aLayer, aRows: PtrInt;
      const aOnState: TOnDecoderState);
    /// the dfMoe block of layer aLayer, its output added to fX
    procedure MoeBlock(aCtx: TDecoderContext; aLayer, aRows: PtrInt;
      const aOnState: TOnDecoderState);
    procedure SetDevice(aDevice: TDecoderDevice);
  public
    /// the model_type of a Hugging Face config.json this class loads
    class function ModelType: RawUtf8; virtual; abstract;
    /// the general.architecture of a GGUF file this class loads
    class function Architecture: RawUtf8; virtual; abstract;
    /// load <aFolder>config.json and <aFolder>model.safetensors (or the files
    // of <aFolder>model.safetensors.index.json)
    // - raises EInferError when a setting is unsupported or a weight is missing
    // or shaped otherwise; aBackend is not owned
    // - aWeights: idtF32 (the reference arithmetic), idtQ8_0 (every matrix
    // quantized at load) or idtQ4_0 (the layers in Q4_0, the head in Q8_0);
    // both need the hidden, attention and feed-forward widths to be
    // multiples of 32
    constructor Create(const aFolder: TFileName; aBackend: TInferBackend;
      aWeights: TInferDType = idtF32); virtual;
    /// load a llama.cpp GGUF file of the architecture: the settings from its
    // metadata, every matrix as stored (Q8_0, Q4_0, Q5_0, Q4_K and Q6_K used
    // in place, F16 and BF16 widened to F32), a head tied to token_embd when the
    // file has no output.weight
    // - raises EInferError for another architecture, a tensor of another type
    // or shape, a scale that is not finite, or a tensor not understood
    constructor CreateFromGguf(const aFileName: TFileName;
      aBackend: TInferBackend); virtual;
    /// load one decoder stack of a larger checkpoint in a Hugging Face folder
    // (e.g. the talker of a speech model): its settings from the object at
    // aConfigPath of <aFolder>config.json (dotted, e.g. 'talker_config' or
    // 'talker_config.code_predictor_config'), its tensors from
    // <aFolder>model.safetensors with aPrefix in place of "model." (e.g.
    // 'talker.model.'), the token embedding aEmbedding ('' = none: such a
    // model takes embedding rows only, ForwardEmbeds) and the head aHead
    // ('' = none: such a model gives no logits, only the normed state of
    // ForwardEmbeds - for a stack whose caller applies a head per step) - full
    // tensor names, [VocabSize, Hidden] each
    // - the other tensors of the file are left alone; every tensor under
    // aPrefix + 'layers.' must be understood
    // - aWeights as Create
    constructor CreateStack(const aFolder: TFileName; aBackend: TInferBackend;
      const aConfigPath, aPrefix, aEmbedding, aHead: RawUtf8;
      aWeights: TInferDType = idtF32);
    /// unmap the weights, and forget them on the backend first
    // (ForgetWeights) - the backend must still live
    destructor Destroy; override;
    /// aBackend.ForgetWeights for every byte of weights this model holds (the
    // mapped file, what was widened or quantized at load): a GPU backend
    // drops its copies - called by Destroy for Backend, and by a device for
    // its own backend when it is freed
    procedure ForgetWeights(aBackend: TInferBackend);
    /// a context sized for this model (KV cache grows on demand) - on Device
    // when one is set
    function NewContext: TDecoderContext;
    /// append aCount tokens to the sequence of aCtx and compute them
    // - their positions are aCtx.Length .. aCtx.Length + aCount - 1; their ids,
    // keys and values enter the KV cache - Length only grows once every layer
    // is computed, so an exception in the layers leaves the tokens it held
    // - aLogits: nil computes no logits; else it receives the logits of the
    // last token ([VocabSize]), or of every new token when aAllLogits
    // ([aCount, VocabSize])
    procedure Forward(aCtx: TDecoderContext; aIds: PIntegerArray; aCount: PtrInt;
      aLogits: PSingleArray; aAllLogits: boolean = false;
      const aOnState: TOnDecoderState = nil);
    /// Forward with aCount embedding rows aEmbeds ([aCount, Hidden] floats)
    // in place of the embeddings of token ids - for text interleaved with
    // image or audio rows, or a stack fed by another model; rows made by
    // Embed give the very logits of Forward with those ids
    // - their positions have no id in the cache (-1): Reuse never keeps them
    // - the rows are copied before the computation: the caller may reuse
    // aEmbeds at once
    // - a loop that feeds floats back (a speech model's talker) adds up
    // rounding: on a GPU device, keep its KV cache in floats (KvF16 false)
    // - aNormed <> nil receives the last row after the final RMSNorm
    // ([Hidden]): what the head multiplies, for a model that feeds it on
    // (a talker's code predictor) or applies a head of its own - on a device
    // too, and with aLogits nil also without a head
    procedure ForwardEmbeds(aCtx: TDecoderContext; aEmbeds: PSingleArray;
      aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean = false;
      const aOnState: TOnDecoderState = nil; aNormed: PSingleArray = nil);
    /// true when the model has a token embedding: Forward, Embed and Greedy
    // need one (a stack without one takes ForwardEmbeds only)
    function HasEmbedding: boolean;
    /// true when the model has a head: logits need one (a stack without one
    // gives the normed state of ForwardEmbeds only)
    function HasHead: boolean;
    /// keep the cached tokens of aCtx that start aIds[0..aCount-1], forget the
    // others, and return how many stayed: the rest of aIds is to be computed
    // - at most aCount - 1 stay, so a Forward of the rest has a token to give
    // logits for; none when aCtx holds a sequence of another model, or when a
    // sliding window no longer holds what the common prefix needs
    // (TDecoderContext.Usable)
    function Reuse(aCtx: TDecoderContext; aIds: PIntegerArray; aCount: PtrInt): PtrInt;
    /// greedy continuation: the prompt is fed after aCtx's cached tokens, then
    // the highest logit is taken until one of aStop or aMaxNew tokens
    // - returns the new ids, the stop token included when one was produced;
    // ties go to the lowest id (like torch.argmax)
    // - afterwards aCtx holds the prompt and every returned token - except the
    // last one when the context reached MaxPositions
    function Greedy(aCtx: TDecoderContext; const aPrompt: array of integer;
      aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
    /// the F32 embeddings of aIds[0..aCount-1] into aDest ([aCount, Hidden]),
    // as Forward takes them (for another executor, e.g. the GPU)
    // - raises EInferError for an id outside the vocabulary
    procedure Embed(aIds: PIntegerArray; aCount: PtrInt; aDest: PSingleArray);
    /// the rotary factors of aRows positions from aFirstPos into aCos and aSin
    // ([aRows, HeadDim / 2] each), as Forward computes them
    procedure Rotary(aFirstPos, aRows: PtrInt; aCos, aSin: PSingleArray);
    /// the experts of one token from its [Experts] router logits: aChosen and
    // aWeights ([ExpertsUsed] each), highest probability first - as MoeBlock
    // routes, also for a device that routes on the host
    // - with a RouteTape: recorded (drmRecord), or the recorded choice handed
    // out instead (drmReplay)
    procedure Route(aLogits: PSingleArray; aChosen: PIntegerArray;
      aWeights: PSingleArray);
    /// true when every layer is of the kind aMixer and aFfn - what a device
    // checks before it takes a model
    function HasOnly(aMixer: TDecoderMixer; aFfn: TDecoderFfn): boolean;
    /// what the layers compute beyond attention, RMSNorm and a plain SwiGLU
    // - a device checks it too: [] for the Qwen families
    function Extras: TDecoderExtras;
    /// the weights of the final RMSNorm, as used
    property FinalNorm: PSingleArray
      read fNorm;
    /// the hyper-parameters
    property Config: TDecoderConfig
      read fConfig;
    /// the backend the model computes with (not owned)
    property Backend: TInferBackend
      read fBackend;
    /// the type of the layer matrices: idtF32, idtQ8_0 or idtQ4_0 (for a GGUF
    // file: the type of the first attention matrix)
    property Weights: TInferDType
      read fWeights;
    /// the weights of every layer, as used - for diagnosis and tests
    property Layers: TDecoderLayerDynArray
      read fLayer;
    /// the language-model head, as used
    property Head: TInferWeight
      read fHead;
    /// the GGUF file of CreateFromGguf (nil after Create): its metadata holds
    // the tokenizer too
    property Gguf: TGgufFile
      read fGguf;
    /// where NewContext puts new contexts: nil (the default) computes with
    // Backend, a TDecoderDevice of this model on that device (not owned)
    // - Forward, Reuse and Greedy follow the context: one of a device is
    // computed there whatever Device is now
    property Device: TDecoderDevice
      read fDevice write SetDevice;
    /// the expert choices of Route recorded or replayed (tests); nil (the
    // default): Route only routes - not owned, one sequence at a time
    property RouteTape: TDecoderRouteTape
      read fRouteTape write fRouteTape;
  end;

  /// a decoder family, as RegisterDecoderModel takes it
  TDecoderModelClass = class of TDecoderModel;

  /// computes a TDecoderModel elsewhere - on a GPU (TQwen3Vulkan): its contexts
  // keep their keys and values there, and TDecoderModel.Forward, Reuse and
  // Greedy hand them over (a TNativeLlmClient decodes on it through
  // TDecoderModel.Device)
  TDecoderDevice = class
  protected
    fModel: TDecoderModel;
  public
    /// a new, empty context on this device (owned by the caller)
    function NewDeviceContext: TDecoderContext; virtual; abstract;
    /// TDecoderModel.Forward for a context of this device (no states)
    procedure Forward(aCtx: TDecoderContext; aIds: PIntegerArray; aCount: PtrInt;
      aLogits: PSingleArray; aAllLogits: boolean = false); virtual; abstract;
    /// TDecoderModel.ForwardEmbeds for a context of this device (no states)
    // - raises EInferError here: a device that takes embedding rows overrides it
    procedure ForwardEmbeds(aCtx: TDecoderContext; aEmbeds: PSingleArray;
      aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean = false;
      aNormed: PSingleArray = nil); virtual;
    /// TDecoderModel.Reuse for a context of this device
    function Reuse(aCtx: TDecoderContext; aIds: PIntegerArray;
      aCount: PtrInt): PtrInt; virtual; abstract;
    /// the model it computes (not owned)
    property Model: TDecoderModel
      read fModel;
  end;


/// make a decoder family known to NewDecoderModel and NewDecoderModelFromGguf
// - call it from the initialization section of the family's unit: the list is
// not locked, so it must not change while models are opened
// - the first class registered for a model type (or an architecture) is the
// one found
procedure RegisterDecoderModel(aClass: TDecoderModelClass);

/// the registered class of a Hugging Face model_type, nil if none
function DecoderModelClass(const aModelType: RawUtf8): TDecoderModelClass;

/// the registered class of a GGUF general.architecture, nil if none
function DecoderModelClassOfArchitecture(const aArchitecture: RawUtf8): TDecoderModelClass;

/// the registered class for <aFolder>config.json, by its model_type
// - raises EInferError when the file cannot be read or no family loads it
function DecoderModelClassOfFolder(const aFolder: TFileName): TDecoderModelClass;

/// load a Hugging Face folder with the family its config.json names
// - see TDecoderModel.Create
function NewDecoderModel(const aFolder: TFileName; aBackend: TInferBackend;
  aWeights: TInferDType = idtF32): TDecoderModel;

/// load a GGUF file with the family its general.architecture names
// - see TDecoderModel.CreateFromGguf
function NewDecoderModelFromGguf(const aFileName: TFileName;
  aBackend: TInferBackend): TDecoderModel;


implementation


/// aValue rounded to the nearest single - a cast of a double is no conversion
// on every compiler, an assignment is
function RoundToSingle(aValue: double): single;
begin
  result := aValue;
end;


{ TDecoderContext }

procedure TDecoderContext.Reset;
var
  i: PtrInt;
begin
  fLength := 0;
  for i := 0 to high(fBase) do
    fBase[i] := 0;
end;

procedure TDecoderContext.Truncate(aLength: PtrInt);
begin
  if (aLength < 0) or (aLength > fLength) then
    EInferError.RaiseUtf8('%.Truncate(%) of % tokens', [self, aLength, fLength]);
  if not Restorable(aLength) then
    EInferError.RaiseUtf8('%.Truncate(%): a sliding window no longer holds ' +
      'the keys of that position', [self, aLength]);
  if aLength = 0 then
    Reset
  else
    fLength := aLength;
end;

function TDecoderContext.Restorable(aLength: PtrInt): boolean;
var
  i, first: PtrInt;
begin
  result := false;
  if (aLength < 0) or (aLength > fLength) then
    exit;
  result := true;
  if aLength = 0 then
    exit; // a new sequence: every window starts again
  for i := 0 to high(fWindow) do
    if fWindow[i] > 0 then
    begin
      // the next token at aLength sees aLength - Window + 1 .. aLength
      first := aLength - fWindow[i] + 1;
      if first < 0 then
        first := 0;
      if fBase[i] > first then
      begin
        result := false;
        exit;
      end;
    end;
end;

function TDecoderContext.Usable(aIds: PIntegerArray; aCount: PtrInt): PtrInt;
begin
  result := Common(aIds, aCount);
  if not Restorable(result) then
    result := 0;
end;

function TDecoderContext.CachedRows(aLayer: PtrInt): PtrInt;
begin
  result := 0;
  if (fKvWidth > 0) and (aLayer >= 0) and (aLayer <= high(fKeys)) then
    result := system.length(fKeys[aLayer]) div fKvWidth;
end;

function TDecoderContext.CachedBase(aLayer: PtrInt): PtrInt;
begin
  result := 0;
  if (aLayer >= 0) and (aLayer <= high(fBase)) then
    result := fBase[aLayer];
end;

procedure TDecoderContext.SetFullCache(aValue: boolean);
begin
  if aValue = fFullCache then
    exit;
  if fLength <> 0 then
    EInferError.RaiseUtf8('%.FullCache on % cached tokens', [self, fLength]);
  fFullCache := aValue;
  fLayers := 0; // the next Forward lays the rows out again
end;

function TDecoderContext.Common(aIds: PIntegerArray; aCount: PtrInt): PtrInt;
begin
  result := 0;
  if aCount > fLength then
    aCount := fLength;
  // a position fed an embedding row holds -1: no id matches it, not even a
  // negative one a caller might pass
  while (result < aCount) and (fIds[result] >= 0) and (fIds[result] = aIds^[result]) do
    inc(result);
end;

function TDecoderContext.GetId(aPosition: PtrInt): integer;
begin
  if (aPosition < 0) or (aPosition >= fLength) then
    EInferError.RaiseUtf8('%.Id[%] of % tokens', [self, aPosition, fLength]);
  result := fIds[aPosition];
end;


{ TDecoderDevice }

procedure TDecoderDevice.ForwardEmbeds(aCtx: TDecoderContext; aEmbeds: PSingleArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean; aNormed: PSingleArray);
begin
  EInferError.RaiseUtf8('%.ForwardEmbeds: embedding rows not supported', [self]);
end;


{ TDecoderModel }

const
  /// the values WidenTensor converts per item of RunItems: small enough
  // that the tiny test models take several (and a rest), large enough that
  // claiming an item costs nothing next to it
  WIDEN_ITEM = 1 shl 12;

var
  /// the last model serial handed out (0 = none, an empty context's owner)
  DecoderModelSerial: integer;

procedure TDecoderModel.Init(aBackend: TInferBackend);
begin
  fSerial := InterlockedIncrement(DecoderModelSerial);
  if aBackend = nil then
    EInferError.RaiseUtf8('%.Create: no backend', [self]);
  fBackend := aBackend;
  if not fBackend.Supports(ioCausalAttention, idtF32) or
     not fBackend.Supports(ioRmsNorm, idtF32) or
     not fBackend.Supports(ioSwiGlu, idtF32) or
     not fBackend.Supports(ioRope, idtF32) then
    EInferError.RaiseUtf8('%: the backend lacks a decoder operation', [self]);
end;

constructor TDecoderModel.Create(const aFolder: TFileName; aBackend: TInferBackend;
  aWeights: TInferDType);
begin
  inherited Create;
  Init(aBackend);
  if not (aWeights in [idtF32, idtQ8_0, idtQ4_0]) then
    EInferError.RaiseUtf8('%.Create: weights as % not supported',
      [self, INFER_DTYPE_NAME[aWeights]]);
  fWeights := aWeights;
  if not fBackend.Supports(ioLinear, aWeights) then
    EInferError.RaiseUtf8('%: the backend has no % Linear',
      [self, INFER_DTYPE_NAME[aWeights]]);
  LoadConfig(IncludeTrailingPathDelimiter(aFolder) + 'config.json');
  if (aWeights <> idtF32) and
     ((fConfig.Hidden and 31 <> 0) or (fConfig.Intermediate and 31 <> 0) or
      (fConfig.ExpertIntermediate and 31 <> 0) or
      ((fConfig.Heads * fConfig.HeadDim) and 31 <> 0)) then
    EInferError.RaiseUtf8('%: % needs widths that are multiples of 32',
      [self, INFER_DTYPE_NAME[aWeights]]);
  fFile := TSafeTensorsFile.OpenFolder(aFolder);
  LoadWeights;
end;

constructor TDecoderModel.CreateFromGguf(const aFileName: TFileName;
  aBackend: TInferBackend);
begin
  inherited Create;
  Init(aBackend);
  fGguf := TGgufFile.Create(aFileName);
  LoadGgufConfig;
  LoadWeights;
  fWeights := fLayer[0].Q.DType;
end;

constructor TDecoderModel.CreateStack(const aFolder: TFileName;
  aBackend: TInferBackend; const aConfigPath, aPrefix, aEmbedding,
  aHead: RawUtf8; aWeights: TInferDType);
begin
  if (aConfigPath = '') or (aPrefix = '') then
    EInferError.RaiseUtf8('%.CreateStack: a config path and a prefix are needed',
      [self]);
  fStackConfig := aConfigPath;
  fStackPrefix := aPrefix;
  fStackEmbed := aEmbedding;
  fStackHead := aHead;
  Create(aFolder, aBackend, aWeights);
end;

destructor TDecoderModel.Destroy;
begin
  // a GPU backend keeps copies by address: gone with the memory they mirror
  ForgetWeights(fBackend);
  fFile.Free;
  fGguf.Free;
  inherited Destroy;
end;

procedure TDecoderModel.ForgetWeights(aBackend: TInferBackend);
var
  i: PtrInt;
begin
  if aBackend = nil then
    exit;
  if fFile <> nil then
    InferForgetTensors(aBackend, fFile.Tensors);
  if fGguf <> nil then
    InferForgetTensors(aBackend, fGguf.Tensors);
  for i := 0 to high(fOwned) do
    InferForgetBytes(aBackend, pointer(fOwned[i]), length(fOwned[i]) * SizeOf(single));
  for i := 0 to high(fOwnedQ8) do
    InferForgetBytes(aBackend, pointer(fOwnedQ8[i]),
      length(fOwnedQ8[i]) * SizeOf(TInferBlockQ8_0));
  for i := 0 to high(fOwnedQ4) do
    InferForgetBytes(aBackend, pointer(fOwnedQ4[i]),
      length(fOwnedQ4[i]) * SizeOf(TInferBlockQ4_0));
  for i := 0 to high(fOwnedMx) do
    InferForgetBytes(aBackend, pointer(fOwnedMx[i]),
      length(fOwnedMx[i]) * SizeOf(TInferBlockMxfp4));
end;

function TDecoderModel.ReadConfig(const aConfigFile: TFileName;
  var aDoc: TDocVariantData): PDocVariantData;
var
  p: PUtf8Char;
  part: RawUtf8;
  sub: PDocVariantData;
begin
  if not aDoc.InitJsonFromFile(aConfigFile, JSON_FAST_FLOAT) then
    EInferError.RaiseUtf8('%: cannot read %', [self, aConfigFile]);
  result := @aDoc;
  if fStackPrefix = '' then
  begin
    if aDoc.U['model_type'] <> ModelType then
      EInferError.RaiseUtf8('%: model_type "%" is not %',
        [self, aDoc.U['model_type'], ModelType]);
    exit;
  end;
  // a stack: the object at the dotted path
  p := pointer(fStackConfig);
  while p <> nil do
  begin
    GetNextItem(p, '.', part);
    if (part = '') or not result^.GetAsObject(part, sub) then
      EInferError.RaiseUtf8('%: % has no object %', [self, aConfigFile, fStackConfig]);
    result := sub;
  end;
end;

function TDecoderModel.SourceName(const aName: RawUtf8): RawUtf8;
begin
  result := aName;
  if fStackPrefix = '' then
    exit;
  if aName = 'lm_head.weight' then
    result := fStackHead
  else if aName = 'model.embed_tokens.weight' then
    result := fStackEmbed
  else if IdemPChar(pointer(aName), 'MODEL.') then
    result := fStackPrefix + copy(aName, 7, MaxInt)
  else
    result := ''; // a stack holds no other tensor of the layout
end;

function TDecoderModel.TensorCountOf(const aPrefix: RawUtf8): PtrInt;
var
  i: PtrInt;
  t: PInferTensor;
begin
  result := 0;
  if fFile = nil then
    exit;
  t := pointer(fFile.Tensors);
  for i := 1 to length(fFile.Tensors) do
  begin
    if (aPrefix = '') or
       ((length(t^.Name) >= length(aPrefix)) and
        CompareMem(pointer(t^.Name), pointer(aPrefix), length(aPrefix))) then
      inc(result);
    inc(t);
  end;
end;

function TDecoderModel.HasEmbedding: boolean;
begin
  result := fEmbedSrc <> nil;
end;

function TDecoderModel.HasHead: boolean;
begin
  result := fHead.Data <> nil;
end;

procedure TDecoderModel.CheckConfig(const aSource: TFileName);
var
  i: PtrInt;
  moe: boolean;
begin
  // every dimension bounded on its own before any product: a product of
  // unbounded values could wrap around and pass
  if (fConfig.VocabSize <= 0) or (fConfig.VocabSize > 1 shl 22) or
     (fConfig.Hidden <= 0) or (fConfig.Hidden > 1 shl 16) or
     (fConfig.Layers <= 0) or (fConfig.Layers > 1024) or
     (fConfig.Heads <= 0) or (fConfig.Heads > 1 shl 12) or
     (fConfig.KvHeads <= 0) or (fConfig.KvHeads > fConfig.Heads) or
     (fConfig.Heads mod fConfig.KvHeads <> 0) or
     (fConfig.HeadDim <= 0) or (fConfig.HeadDim > 1024) or odd(fConfig.HeadDim) or
     (fConfig.Heads * fConfig.HeadDim > 1 shl 16) or
     (fConfig.Intermediate <= 0) or (fConfig.Intermediate > 1 shl 18) or
     (fConfig.MaxPositions <= 0) or (fConfig.MaxPositions > 1 shl 24) or
     not (fConfig.RmsEps > 0) or (fConfig.RmsEps > 1) or
     not (fConfig.RopeTheta > 1) or (fConfig.RopeTheta > 1e12) then
    EInferError.RaiseUtf8('%: % has an invalid or unsupported shape', [self, aSource]);
  // the kind of every layer: zero-filled is attention and a dense block
  if fConfig.Spec = nil then
    SetLength(fConfig.Spec, fConfig.Layers)
  else if length(fConfig.Spec) <> fConfig.Layers then
    EInferError.RaiseUtf8('%: % layer kinds for % layers',
      [self, length(fConfig.Spec), fConfig.Layers]);
  // experts only with a dfMoe layer, and then bounded
  moe := false;
  for i := 0 to high(fConfig.Spec) do
    if fConfig.Spec[i].Ffn = dfMoe then
      moe := true;
  if moe then
  begin
    if (fConfig.Experts <= 0) or (fConfig.Experts > 1 shl 12) or
       (fConfig.ExpertsUsed <= 0) or (fConfig.ExpertsUsed > fConfig.Experts) or
       (fConfig.ExpertIntermediate <= 0) or (fConfig.ExpertIntermediate > 1 shl 18) then
      EInferError.RaiseUtf8('%: % has invalid experts (% of %, width %)', [self,
        aSource, fConfig.ExpertsUsed, fConfig.Experts, fConfig.ExpertIntermediate]);
  end
  else if (fConfig.Experts <> 0) or (fConfig.ExpertsUsed <> 0) or
          (fConfig.ExpertIntermediate <> 0) then
    EInferError.RaiseUtf8('%: % has experts but no layer uses them', [self, aSource]);
  for i := 0 to high(fConfig.Spec) do
    if (fConfig.Spec[i].Window < 0) or (fConfig.Spec[i].Window > 1 shl 24) or
       ((fConfig.Spec[i].Window <> 0) and (fConfig.Spec[i].Mixer <> dmAttention)) then
      EInferError.RaiseUtf8('%: % has a window of % in layer %',
        [self, aSource, fConfig.Spec[i].Window, i]);
  if (fConfig.SwiGluLimit < 0) or (fConfig.SwiGluLimit > 1e6) or
     ((fConfig.SwiGluLimit > 0) and
      (not (fConfig.SwiGluAlpha > 0) or (fConfig.SwiGluAlpha > 1e6))) then
    EInferError.RaiseUtf8('%: % has a SwiGLU limit of % (alpha %)',
      [self, aSource, fConfig.SwiGluLimit, fConfig.SwiGluAlpha]);
  if (fConfig.RopeFactor <> 0) and
     (not (fConfig.RopeFactor >= 1) or (fConfig.RopeFactor > 1e4) or
      (fConfig.RopeOriginal <= 0) or (fConfig.RopeOriginal > 1 shl 24) or
      not (fConfig.RopeBetaFast > 0) or (fConfig.RopeBetaFast > 1e6) or
      not (fConfig.RopeBetaSlow > 0) or (fConfig.RopeBetaSlow > 1e6) or
      not (fConfig.RopeAttention >= 0) or (fConfig.RopeAttention > 1e3)) then
    EInferError.RaiseUtf8('%: % has invalid YaRN settings (factor %)',
      [self, aSource, fConfig.RopeFactor]);
  // inverse rotary frequencies as the reference forms them in float32:
  // 1 / theta ^ (2i / d), each step rounded to single
  SetLength(fInvFreq, fConfig.HeadDim shr 1);
  for i := 0 to high(fInvFreq) do
    fInvFreq[i] := 1 / RoundToSingle(Exp(Ln(fConfig.RopeTheta) *
      RoundToSingle((2 * i) / fConfig.HeadDim)));
  fRopeScale := 1;
  if fConfig.RopeFactor <> 0 then
    YarnFrequencies;
end;

procedure TDecoderModel.YarnFrequencies;
var
  i, d: PtrInt;
  lo, hi, span, pi2: double;
  loS, spanS, ramp, keep, pos, inter, extra, f: single;

  // the dimension that turns aRotations times over the original context
  function CorrectionDim(aRotations: double): double;
  begin
    result := (d * Ln(fConfig.RopeOriginal / (aRotations * pi2))) /
      (2 * Ln(fConfig.RopeTheta));
  end;

begin
  // as _compute_yarn_parameters of transformers: the frequencies below the
  // correction range keep their value (extrapolation), those above it are
  // divided by the factor (interpolation), a linear ramp between - every
  // tensor step in float32, the bounds in double as Python computes them
  d := fConfig.HeadDim;
  pi2 := 2 * PI;
  lo := CorrectionDim(fConfig.RopeBetaFast);
  hi := CorrectionDim(fConfig.RopeBetaSlow);
  if fConfig.RopeTruncate then
  begin
    // math.floor / math.ceil
    if Trunc(lo) > lo then
      lo := Trunc(lo) - 1
    else
      lo := Trunc(lo);
    if Trunc(hi) < hi then
      hi := Trunc(hi) + 1
    else
      hi := Trunc(hi);
  end;
  if lo < 0 then
    lo := 0;
  if hi > d - 1 then
    hi := d - 1;
  if lo = hi then
    hi := hi + 0.001; // no singularity
  span := hi - lo;
  loS := lo;
  spanS := span;
  f := fConfig.RopeFactor;
  for i := 0 to high(fInvFreq) do
  begin
    // the extrapolation 1 / pos is the plain frequency, the interpolation
    // 1 / (factor * pos)
    pos := Exp(Ln(fConfig.RopeTheta) * RoundToSingle((2 * i) / d));
    pos := f * pos;
    inter := 1 / pos;
    extra := fInvFreq[i];
    ramp := i - loS;
    ramp := ramp / spanS;
    if ramp < 0 then
      ramp := 0
    else if ramp > 1 then
      ramp := 1;
    keep := 1 - ramp; // the share of the extrapolation
    ramp := 1 - keep;
    inter := inter * ramp;
    extra := extra * keep;
    fInvFreq[i] := inter + extra;
  end;
  if fConfig.RopeAttention > 0 then
    fRopeScale := fConfig.RopeAttention
  else if f <= 1 then
    fRopeScale := 1
  else
    fRopeScale := 0.1 * Ln(f) + 1;
end;

function TDecoderModel.HasOnly(aMixer: TDecoderMixer; aFfn: TDecoderFfn): boolean;
var
  i: PtrInt;
begin
  result := false;
  for i := 0 to high(fConfig.Spec) do
    if (fConfig.Spec[i].Mixer <> aMixer) or (fConfig.Spec[i].Ffn <> aFfn) then
      exit;
  result := true;
end;

function TDecoderModel.Extras: TDecoderExtras;
var
  i: PtrInt;
begin
  result := [];
  if fConfig.SwiGluLimit > 0 then
    include(result, dxSwiGluLimit);
  for i := 0 to high(fLayer) do
  begin
    if fLayer[i].Sinks <> nil then
      include(result, dxSinks);
    if fLayer[i].OBias <> nil then
      include(result, dxOutputBias);
    if fLayer[i].RouterBias <> nil then
      include(result, dxRouterBias);
    if (fLayer[i].ExpertGateBias <> nil) or (fLayer[i].ExpertUpBias <> nil) or
       (fLayer[i].ExpertDownBias <> nil) then
      include(result, dxExpertBias);
  end;
end;

function TDecoderModel.Find(const aName: RawUtf8): PInferTensor;
begin
  if fGguf <> nil then
  begin
    result := nil;
    if GgufName(aName) <> '' then
      result := fGguf.FindTensor(GgufName(aName));
  end
  else if fStackPrefix <> '' then
  begin
    result := nil;
    if SourceName(aName) <> '' then
      result := fFile.Find(SourceName(aName));
  end
  else
    result := fFile.Find(aName);
end;

function TDecoderModel.TensorCount: PtrInt;
begin
  if fGguf <> nil then
    result := length(fGguf.Tensors)
  else
    result := length(fFile.Tensors);
end;

function TDecoderModel.Checked(const aName: RawUtf8;
  const aDims: array of Int64; aBlocks: boolean): PInferTensor;
var
  i: PtrInt;
  ok: boolean;
  want: TInt64DynArray;
  t: PInferTensor;
begin
  t := Find(aName);
  if t = nil then
  begin
    if (fGguf = nil) and (SourceName(aName) <> '') then
      fFile.Tensor(SourceName(aName)); // raises with the file's message
    if fGguf = nil then
      EInferError.RaiseUtf8('%: no tensor % in the stack %', [self, aName, fStackPrefix]);
    EInferError.RaiseUtf8('%: no tensor % (%) in %',
      [self, GgufName(aName), aName, fGguf.FileName]);
  end;
  ok := length(t^.Shape) = length(aDims);
  if ok then
    for i := 0 to high(aDims) do
      if t^.Shape[i] <> aDims[i] then
        ok := false;
  if not ok then
  begin
    SetLength(want, length(aDims));
    for i := 0 to high(aDims) do
      want[i] := aDims[i];
    EInferError.RaiseUtf8('%: % is shaped %, expected %',
      [self, aName, InferShapeText(t^.Shape), InferShapeText(want)]);
  end;
  // F32 and BF16 from safetensors; a GGUF file adds F16, and the block types
  // for a matrix (aBlocks)
  if not ((t^.DType in [idtF32, idtBF16]) or
          ((fGguf <> nil) and ((t^.DType = idtF16) or
           (aBlocks and (t^.DType in INFER_BLOCK_DTYPES))))) then
    EInferError.RaiseUtf8('%: % is %, not supported here',
      [self, aName, INFER_DTYPE_NAME[t^.DType]]);
  result := t;
end;

function TDecoderModel.Weight(const aName: RawUtf8;
  const aDims: array of Int64): PSingleArray;
var
  t: PInferTensor;
begin
  t := Checked(aName, aDims);
  if t^.DType = idtF32 then
    result := t^.Data
  else
    result := WidenTensor(t, aName);
end;

function TDecoderModel.WidenTensor(t: PInferTensor;
  const aName: RawUtf8): PSingleArray;
var
  n: PtrInt;
begin
  // BF16 is the upper half of an F32, F16 a subset of it: widening is exact
  if t^.Count > High(PtrInt) div 4 then
    EInferError.RaiseUtf8('%: % too large', [self, aName]);
  n := length(fOwned);
  SetLength(fOwned, n + 1);
  SetLength(fOwned[n], t^.Count);
  result := pointer(fOwned[n]);
  // in pieces of WIDEN_ITEM values on the backend's threads
  fCvSrc := t^.Data;
  fCvSrcType := t^.DType;
  fCvCount := t^.Count;
  fCvDest := result;
  fBackend.RunItems(WidenItem, (t^.Count + WIDEN_ITEM - 1) div WIDEN_ITEM);
end;

procedure TDecoderModel.WidenItem(aItem: integer);
var
  first, n: PtrInt;
begin
  first := PtrInt(aItem) * WIDEN_ITEM;
  n := fCvCount - first;
  if n > WIDEN_ITEM then
    n := WIDEN_ITEM;
  // the types widened here (BF16, F16) hold one value per element
  InferRowToF32(@PByteArray(fCvSrc)^[first * INFER_DTYPE_SIZE[fCvSrcType]],
    fCvSrcType, 0, n, @fCvDest^[first]);
end;

function TDecoderModel.Own(var aValues: TSingleDynArray): PSingleArray;
var
  n: PtrInt;
begin
  n := length(fOwned);
  SetLength(fOwned, n + 1);
  fOwned[n] := aValues; // a reference: the values are not copied
  aValues := nil;
  result := pointer(fOwned[n]);
end;

function TDecoderModel.OwnMxfp4(aBlocks: PtrInt): PInferBlockMxfp4Array;
var
  n: PtrInt;
begin
  n := length(fOwnedMx);
  SetLength(fOwnedMx, n + 1);
  SetLength(fOwnedMx[n], aBlocks);
  result := pointer(fOwnedMx[n]);
end;

function TDecoderModel.MatrixOf(var aSrc: TSingleDynArray; aRows, aCols: Int64;
  aType: TInferDType; const aName: RawUtf8): TInferWeight;
var
  n: PtrInt;
begin
  if Int64(length(aSrc)) <> aRows * aCols then
    EInferError.RaiseUtf8('%: % has % values for [%,%]',
      [self, aName, length(aSrc), aRows, aCols]);
  case aType of
    idtF32:
      result := InferWeight(Own(aSrc), idtF32);
    idtQ8_0:
      begin
        n := length(fOwnedQ8);
        SetLength(fOwnedQ8, n + 1);
        SetLength(fOwnedQ8[n], aRows * (aCols shr 5));
        ConvertRows(pointer(aSrc), idtF32, aRows, aCols, aType, n, aName);
        result := InferWeight(pointer(fOwnedQ8[n]), idtQ8_0);
      end;
    idtQ4_0:
      begin
        n := length(fOwnedQ4);
        SetLength(fOwnedQ4, n + 1);
        SetLength(fOwnedQ4[n], aRows * (aCols shr 5));
        ConvertRows(pointer(aSrc), idtF32, aRows, aCols, aType, n, aName);
        result := InferWeight(pointer(fOwnedQ4[n]), idtQ4_0);
      end;
  else
    EInferError.RaiseUtf8('%: % as % not supported', [self, aName,
      INFER_DTYPE_NAME[aType]]);
  end;
end;

procedure TDecoderModel.QuantizeRow(aSrc: pointer; aSrcType: TInferDType;
  aCols: PtrInt; aType: TInferDType; aOwned, aRow: PtrInt; const aName: RawUtf8);
var
  nb, b, c, blk: PtrInt;
  d: word;
  x: array[0..31] of single; // one block: no heap on the threads
begin
  nb := aCols shr 5;
  for b := 0 to nb - 1 do
  begin
    // Q8_0 and Q4_0 quantize every block of 32 on its own
    blk := aRow * nb + b;
    InferRowToF32(aSrc, aSrcType, blk, 32, @x);
    // a NaN or an infinity has no quantized form: refused, by its bits (no FPU)
    for c := 0 to 31 do
      if PCardinal(@x[c])^ and $7f800000 = $7f800000 then
        EInferError.RaiseUtf8('%: % has a NaN or infinite weight at row % column %',
          [self, aName, aRow, b * 32 + c]);
    if aType = idtQ8_0 then
    begin
      InferQuantizeRowQ8_0Ref(@x, 32, @fOwnedQ8[aOwned][blk]);
      d := fOwnedQ8[aOwned][blk].d;
    end
    else
    begin
      InferQuantizeRowQ4_0(@x, 32, @fOwnedQ4[aOwned][blk]);
      d := fOwnedQ4[aOwned][blk].d;
    end;
    // a finite weight beyond the range still gets the FP16 scale +Inf (ggml
    // writes it too): every product with it would be Inf or NaN
    if d and $7fff >= $7c00 then
      EInferError.RaiseUtf8('%: % has a weight beyond the % range in row %',
        [self, aName, INFER_DTYPE_NAME[aType], aRow]);
  end;
end;

procedure TDecoderModel.ConvertRows(aSrc: pointer; aSrcType: TInferDType;
  aRows, aCols: PtrInt; aType: TInferDType; aOwned: PtrInt; const aName: RawUtf8);
begin
  fCvSrc := aSrc;
  fCvSrcType := aSrcType;
  fCvCols := aCols;
  fCvType := aType;
  fCvOwned := aOwned;
  fCvName := aName;
  try
    fBackend.RunItems(ConvertItem, aRows);
  finally
    fCvName := '';
  end;
end;

procedure TDecoderModel.ConvertItem(aItem: integer);
begin
  QuantizeRow(fCvSrc, fCvSrcType, fCvCols, fCvType, fCvOwned, aItem, fCvName);
end;

/// raise unless every MXFP4 block of t has an exponent the CPU computes
// exactly (INFER_MXFP4_EXACT_MIN..MAX) or only zeros as values
procedure CheckMxfp4Scales(t: PInferTensor; const aName: RawUtf8; aModel: TObject);
var
  b, j: PtrInt;
  m: PInferBlockMxfp4;
begin
  m := t^.Data;
  for b := 0 to t^.Count div 32 - 1 do
  begin
    if (m^.e < INFER_MXFP4_EXACT_MIN) or (m^.e > INFER_MXFP4_EXACT_MAX) then
      // +0 and -0 are the nibbles 0 and 8
      for j := 0 to 15 do
        if m^.qs[j] and $77 <> 0 then
          EInferError.RaiseUtf8('%: % has the MXFP4 exponent % in block %, ' +
            'beyond the range computed exactly', [aModel, aName, m^.e, b]);
    inc(m);
  end;
end;

/// raise unless every FP16 scale of the blocks of t is finite
procedure CheckScales(t: PInferTensor; const aName: RawUtf8; aModel: TObject);
var
  b, size, first, second: PtrInt;
  p: PByteArray;
begin
  if t^.DType = idtMXFP4 then
  begin
    CheckMxfp4Scales(t, aName, aModel);
    exit;
  end;
  size := INFER_DTYPE_SIZE[t^.DType];
  // where the half-float scales sit: d first (Q8_0, Q4_0, Q5_0), d and dmin first
  // (Q4_K), d last (Q6_K)
  first := 0;
  second := -1;
  case t^.DType of
    idtQ4_K:
      second := 2;
    idtQ6_K:
      first := SizeOf(TInferBlockQ6_K) - 2;
  end;
  p := t^.Data;
  for b := 0 to t^.Count div INFER_DTYPE_BLOCK[t^.DType] - 1 do
  begin
    if (PWord(@p^[first])^ and $7fff >= $7c00) or
       ((second >= 0) and (PWord(@p^[second])^ and $7fff >= $7c00)) then
      EInferError.RaiseUtf8('%: % has a scale that is no finite number in block %',
        [aModel, aName, b]);
    p := @p^[size];
  end;
end;

procedure TDecoderModel.CheckBlockScales(aTensor: PInferTensor;
  const aName: RawUtf8);
begin
  CheckScales(aTensor, aName, self);
end;

function TDecoderModel.GgufExperts(const aGgufName: RawUtf8;
  aExperts, aRows, aCols: Int64): TInferWeightDynArray;
var
  t: PInferTensor;
  e: PtrInt;
  size: Int64;
  p: PByteArray;
begin
  t := fGguf.FindTensor(aGgufName);
  if t = nil then
    EInferError.RaiseUtf8('%: no tensor % in %', [self, aGgufName, fGguf.FileName]);
  if not InferShapeIs(t^, [aExperts, aRows, aCols]) then
    EInferError.RaiseUtf8('%: % is shaped %, expected [%,%,%]',
      [self, aGgufName, InferShapeText(t^.Shape), aExperts, aRows, aCols]);
  if not (t^.DType in INFER_BLOCK_DTYPES) then
    EInferError.RaiseUtf8('%: % is %, not supported for experts here',
      [self, aGgufName, INFER_DTYPE_NAME[t^.DType]]);
  if not fBackend.Supports(ioLinear, t^.DType) then
    EInferError.RaiseUtf8('%: the backend has no % Linear for %',
      [self, INFER_DTYPE_NAME[t^.DType], aGgufName]);
  CheckScales(t, aGgufName, self);
  // one expert: aRows rows of whole blocks, right after the one before
  p := t^.Data;
  size := (aRows * aCols div INFER_DTYPE_BLOCK[t^.DType]) * INFER_DTYPE_SIZE[t^.DType];
  SetLength(result, aExperts);
  for e := 0 to aExperts - 1 do
    result[e] := InferWeight(@p^[e * size], t^.DType);
end;

function TDecoderModel.Matrix(const aName: RawUtf8; aRows, aCols: Int64;
  aType: TInferDType): TInferWeight;
var
  t: PInferTensor;
  nb, n: PtrInt;
begin
  if fGguf <> nil then
  begin
    // a GGUF file decides: blocks in place, F32 in place, F16 / BF16 widened
    t := Checked(aName, [aRows, aCols], true);
    case t^.DType of
      idtQ8_0, idtQ4_0, idtQ5_0, idtQ4_K, idtQ6_K, idtMXFP4:
        begin
          CheckScales(t, aName, self);
          result := InferWeight(t^.Data, t^.DType);
        end;
    else
      result := InferWeight(Weight(aName, [aRows, aCols]), idtF32);
    end;
    if not fBackend.Supports(ioLinear, result.DType) then
      EInferError.RaiseUtf8('%: the backend has no % Linear for %',
        [self, INFER_DTYPE_NAME[result.DType], aName]);
    exit;
  end;
  if aType = idtF32 then
  begin
    result := InferWeight(Weight(aName, [aRows, aCols]), idtF32);
    exit;
  end;
  // Q8_0 or Q4_0: block by block through single precision, as ggml quantizes
  t := Checked(aName, [aRows, aCols]);
  nb := aCols shr 5;
  if aType = idtQ8_0 then
  begin
    n := length(fOwnedQ8);
    SetLength(fOwnedQ8, n + 1);
    SetLength(fOwnedQ8[n], aRows * nb);
    result := InferWeight(pointer(fOwnedQ8[n]), idtQ8_0);
  end
  else
  begin
    n := length(fOwnedQ4);
    SetLength(fOwnedQ4, n + 1);
    SetLength(fOwnedQ4[n], aRows * nb);
    result := InferWeight(pointer(fOwnedQ4[n]), idtQ4_0);
  end;
  ConvertRows(t^.Data, t^.DType, aRows, aCols, aType, n, aName);
end;

function TDecoderModel.NewContext: TDecoderContext;
begin
  if fDevice <> nil then
    result := fDevice.NewDeviceContext
  else
    result := TDecoderContext.Create;
end;

procedure TDecoderModel.SetDevice(aDevice: TDecoderDevice);
begin
  if (aDevice <> nil) and (aDevice.fModel <> self) then
    EInferError.RaiseUtf8('%.Device: % computes another model', [self, aDevice]);
  fDevice := aDevice;
end;

procedure Need(var aBuffer: TSingleDynArray; aCount: PtrInt); overload;
begin
  if length(aBuffer) < aCount then
    SetLength(aBuffer, aCount);
end;

procedure TDecoderModel.Grow(aCtx: TDecoderContext; aTokens, aNew: PtrInt);
var
  i, cap, kvw: PtrInt;
begin
  // the cache belongs to one model: a context holding another model's
  // sequence is refused (same shape or not), an empty one is taken over
  kvw := fConfig.KvHeads * fConfig.HeadDim;
  if aCtx.fOwner <> fSerial then
  begin
    if aCtx.fLength <> 0 then
      EInferError.RaiseUtf8('%.Forward: context holds a sequence of another model',
        [self]);
    aCtx.fOwner := fSerial;
  end;
  if (aCtx.fKvWidth <> kvw) or (aCtx.fLayers <> fConfig.Layers) or
     (length(aCtx.fWindow) <> fConfig.Layers) then
  begin
    // the shape is only taken once its arrays are there: a failed SetLength
    // leaves a context that tries again
    aCtx.fCapacity := 0;
    aCtx.fKeys := nil; // a sliding window layer may become a full one
    aCtx.fValues := nil;
    SetLength(aCtx.fKeys, fConfig.Layers);
    SetLength(aCtx.fValues, fConfig.Layers);
    SetLength(aCtx.fWindow, fConfig.Layers);
    SetLength(aCtx.fBase, fConfig.Layers);
    for i := 0 to fConfig.Layers - 1 do
    begin
      aCtx.fBase[i] := 0;
      if aCtx.fFullCache then
        aCtx.fWindow[i] := 0 // every row kept: the window only masks
      else
        aCtx.fWindow[i] := fConfig.Spec[i].Window;
    end;
    aCtx.fKvWidth := kvw;
    aCtx.fLayers := fConfig.Layers;
  end
  else
    // the same shape but another model: its windows
    for i := 0 to fConfig.Layers - 1 do
      if aCtx.fFullCache then
        aCtx.fWindow[i] := 0
      else if aCtx.fWindow[i] <> fConfig.Spec[i].Window then
      begin
        if aCtx.fLength <> 0 then
          EInferError.RaiseUtf8('%.Forward: context holds other windows', [self]);
        aCtx.fWindow[i] := fConfig.Spec[i].Window;
        aCtx.fBase[i] := 0;
        aCtx.fKeys[i] := nil;
        aCtx.fValues[i] := nil;
        aCtx.fCapacity := 0; // a full layer again: sized by the loop below
      end;
  if aTokens > aCtx.fCapacity then
  begin
    // doubled, and a quarter beyond the tokens wanted: a prefill is followed
    // by decoding, which must not copy the whole cache at its first token
    cap := aCtx.fCapacity * 2;
    if cap < aTokens + aTokens shr 2 then
      cap := aTokens + aTokens shr 2;
    if cap < 64 then
      cap := 64;
    if cap > fConfig.MaxPositions then
      cap := fConfig.MaxPositions;
    // the byte size must fit PtrInt (a 32-bit process): SetLength would wrap
    if cap > (High(PtrInt) div SizeOf(single)) div kvw then
      EInferError.RaiseUtf8('%.Forward: a KV cache of % tokens is too large for ' +
        'this process', [self, cap]);
    // a sliding window layer sizes its own rows (Slide)
    for i := 0 to fConfig.Layers - 1 do
      if aCtx.fWindow[i] = 0 then
      begin
        SetLength(aCtx.fKeys[i], cap * kvw);
        SetLength(aCtx.fValues[i], cap * kvw);
      end;
    SetLength(aCtx.fIds, cap);
    aCtx.fCapacity := cap;
  end;
  Need(aCtx.fX, aNew * fConfig.Hidden);
  Need(aCtx.fH, aNew * fConfig.Hidden);
  Need(aCtx.fTmp, aNew * fConfig.Hidden);
  Need(aCtx.fQ, aNew * fConfig.Heads * fConfig.HeadDim);
  Need(aCtx.fAttn, aNew * fConfig.Heads * fConfig.HeadDim);
  Need(aCtx.fK, aNew * kvw);
  Need(aCtx.fV, aNew * kvw);
  Need(aCtx.fGate, aNew * MaxPtrInt(fConfig.Intermediate, fConfig.ExpertIntermediate));
  Need(aCtx.fUp, aNew * MaxPtrInt(fConfig.Intermediate, fConfig.ExpertIntermediate));
  Need(aCtx.fCos, aNew * (fConfig.HeadDim shr 1));
  Need(aCtx.fSin, aNew * (fConfig.HeadDim shr 1));
  if fConfig.Experts > 0 then
  begin
    Need(aCtx.fRoute, aNew * fConfig.Experts);
    Need(aCtx.fChosenW, aNew * fConfig.ExpertsUsed);
    Need(aCtx.fGather, aNew * fConfig.Hidden);
    Need(aCtx.fExpertOut, aNew * fConfig.Hidden);
    Need(aCtx.fRowW, aNew);
    if length(aCtx.fChosen) < aNew * fConfig.ExpertsUsed then
      SetLength(aCtx.fChosen, aNew * fConfig.ExpertsUsed);
    if length(aCtx.fRow) < aNew then
      SetLength(aCtx.fRow, aNew);
  end;
end;

procedure TDecoderModel.Slide(aCtx: TDecoderContext; aLayer, aPast, aRows: PtrInt);
var
  w, base, first, keep, need, rows, cap, kvw: PtrInt;
begin
  w := aCtx.fWindow[aLayer];
  kvw := aCtx.fKvWidth;
  base := aCtx.fBase[aLayer];
  // both buffers: a failed SetLength may have grown only the keys
  rows := MinPtrInt(length(aCtx.fKeys[aLayer]), length(aCtx.fValues[aLayer])) div kvw;
  // the new rows fit behind the kept ones - unless a large prefill left far
  // more rows than decoding needs: then they shrink back now
  if (aPast + aRows - base <= rows) and ((rows <= 4 * w) or (aRows > w)) then
    exit;
  // the first new token sees aPast - w + 1 .. aPast: older keys are dropped,
  // the newer ones moved to row 0 - the new rows never overwrite a key that
  // one of them still needs
  first := aPast - w + 1;
  if first < base then
    first := base;
  keep := aPast - first;
  if (first > base) and (keep > 0) then
  begin
    MoveFast(aCtx.fKeys[aLayer][(first - base) * kvw], aCtx.fKeys[aLayer][0],
      keep * kvw * SizeOf(single));
    MoveFast(aCtx.fValues[aLayer][(first - base) * kvw], aCtx.fValues[aLayer][0],
      keep * kvw * SizeOf(single));
  end;
  aCtx.fBase[aLayer] := first;
  need := keep + aRows;
  if (need > rows) or (rows > 4 * w) then
  begin
    // twice the window: a decode step moves the rows once every w tokens;
    // growing leaves a quarter more, shrinking (need <= 2w - 1) none
    if need > rows then
      cap := need + need shr 2
    else
      cap := need;
    if cap < 2 * w then
      cap := 2 * w;
    if cap > fConfig.MaxPositions then
      cap := fConfig.MaxPositions;
    if cap < need then
      cap := need;
    if cap > (High(PtrInt) div SizeOf(single)) div kvw then
      EInferError.RaiseUtf8('%.Forward: a KV cache of % tokens is too large for ' +
        'this process', [self, cap]);
    SetLength(aCtx.fKeys[aLayer], cap * kvw);
    SetLength(aCtx.fValues[aLayer], cap * kvw);
  end;
end;

procedure TDecoderModel.RopeFactors(aCtx: TDecoderContext; aRows, aFirstPos: PtrInt);
begin
  // once per Forward: every layer and both Q and K use the same factors
  Rotary(aFirstPos, aRows, pointer(aCtx.fCos), pointer(aCtx.fSin));
end;

procedure TDecoderModel.Rotary(aFirstPos, aRows: PtrInt; aCos, aSin: PSingleArray);
var
  r, i, half: PtrInt;
  angle, c, s: single;
begin
  half := fConfig.HeadDim shr 1;
  for r := 0 to aRows - 1 do
    for i := 0 to half - 1 do
    begin
      // the angle in single, as the reference's float32 product
      angle := fInvFreq[i] * (aFirstPos + r);
      c := Cos(angle);
      s := Sin(angle);
      if fRopeScale <> 1 then
      begin
        // YaRN: both scaled in float32, after the rounding of cos and sin
        c := c * fRopeScale;
        s := s * fRopeScale;
      end;
      aCos^[r * half + i] := c;
      aSin^[r * half + i] := s;
    end;
end;

procedure TDecoderModel.Embed(aIds: PIntegerArray; aCount: PtrInt; aDest: PSingleArray);
var
  i, id: PtrInt;
begin
  if fEmbedSrc = nil then
    EInferError.RaiseUtf8('%.Embed: no token embedding (embedding rows only)', [self]);
  for i := 0 to aCount - 1 do
  begin
    id := aIds^[i];
    if (id < 0) or (id >= fConfig.VocabSize) then
      EInferError.RaiseUtf8('%.Embed: token id % outside the vocabulary', [self, id]);
    InferRowToF32(fEmbedSrc^.Data, fEmbedSrc^.DType, id, fConfig.Hidden,
      @aDest^[i * fConfig.Hidden]);
  end;
end;

procedure TDecoderModel.Attention(aCtx: TDecoderContext; aLayer, aRows: PtrInt);
begin
  // the reference backend computes in double; a fast one splits by head and
  // row block - the cache rows start at the base position of the layer, a
  // sliding window masks the older ones (they may be gone)
  fBackend.CausalAttention(pointer(aCtx.fQ), aRows, fConfig.Heads, fConfig.HeadDim,
    pointer(aCtx.fKeys[aLayer]), pointer(aCtx.fValues[aLayer]),
    aCtx.fLength - aCtx.fBase[aLayer], fConfig.KvHeads, pointer(aCtx.fAttn),
    fConfig.Spec[aLayer].Window, fLayer[aLayer].Sinks);
end;

procedure TDecoderModel.AttentionBlock(aCtx: TDecoderContext;
  aLayer, aRows, aPast: PtrInt; const aOnState: TOnDecoderState);
var
  n, h, qw, kvw: PtrInt;
  ly: ^TDecoderLayer;
begin
  ly := @fLayer[aLayer];
  n := aRows;
  h := fConfig.Hidden;
  qw := fConfig.Heads * fConfig.HeadDim;
  kvw := fConfig.KvHeads * fConfig.HeadDim;
  // pre-norm, projections, per-head norms, rotary positions
  fBackend.RmsNorm(pointer(aCtx.fX), n, h, ly^.InputNorm, fConfig.RmsEps,
    pointer(aCtx.fH));
  fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Q, qw, ly^.QBias, pointer(aCtx.fQ));
  fBackend.Linear(pointer(aCtx.fH), n, h, ly^.K, kvw, ly^.KBias, pointer(aCtx.fK));
  fBackend.Linear(pointer(aCtx.fH), n, h, ly^.V, kvw, ly^.VBias, pointer(aCtx.fV));
  if ly^.QNorm <> nil then
  begin
    fBackend.RmsNorm(pointer(aCtx.fQ), n * fConfig.Heads, fConfig.HeadDim,
      ly^.QNorm, fConfig.RmsEps, pointer(aCtx.fQ));
    fBackend.RmsNorm(pointer(aCtx.fK), n * fConfig.KvHeads, fConfig.HeadDim,
      ly^.KNorm, fConfig.RmsEps, pointer(aCtx.fK));
  end;
  if not fConfig.Spec[aLayer].NoRope then
  begin
    fBackend.Rope(pointer(aCtx.fQ), n, fConfig.Heads, fConfig.HeadDim,
      pointer(aCtx.fCos), pointer(aCtx.fSin));
    fBackend.Rope(pointer(aCtx.fK), n, fConfig.KvHeads, fConfig.HeadDim,
      pointer(aCtx.fCos), pointer(aCtx.fSin));
  end;
  if aCtx.fWindow[aLayer] > 0 then
    Slide(aCtx, aLayer, aPast, n);
  MoveFast(pointer(aCtx.fK)^, aCtx.fKeys[aLayer][(aPast - aCtx.fBase[aLayer]) * kvw],
    n * kvw * SizeOf(single));
  MoveFast(pointer(aCtx.fV)^, aCtx.fValues[aLayer][(aPast - aCtx.fBase[aLayer]) * kvw],
    n * kvw * SizeOf(single));
  Attention(aCtx, aLayer, n);
  fBackend.Linear(pointer(aCtx.fAttn), n, qw, ly^.O, h, ly^.OBias, pointer(aCtx.fTmp));
  if Assigned(aOnState) and (aLayer = 0) then
    aOnState('layer00_attn', pointer(aCtx.fTmp), n, h);
  fBackend.Add(pointer(aCtx.fX), pointer(aCtx.fTmp), n * h);
end;

procedure TDecoderModel.DenseBlock(aCtx: TDecoderContext; aLayer, aRows: PtrInt;
  const aOnState: TOnDecoderState);
var
  n, h, f: PtrInt;
  ly: ^TDecoderLayer;
begin
  ly := @fLayer[aLayer];
  n := aRows;
  h := fConfig.Hidden;
  f := fConfig.Intermediate;
  // pre-norm, SwiGLU, residual
  fBackend.RmsNorm(pointer(aCtx.fX), n, h, ly^.PostNorm, fConfig.RmsEps,
    pointer(aCtx.fH));
  fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Gate, f, nil, pointer(aCtx.fGate));
  fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Up, f, nil, pointer(aCtx.fUp));
  fBackend.SwiGlu(pointer(aCtx.fGate), pointer(aCtx.fUp), n * f,
    fConfig.SwiGluAlpha, fConfig.SwiGluLimit);
  fBackend.Linear(pointer(aCtx.fGate), n, f, ly^.Down, h, nil, pointer(aCtx.fTmp));
  if Assigned(aOnState) and (aLayer = 0) then
    aOnState('layer00_mlp', pointer(aCtx.fTmp), n, h);
  fBackend.Add(pointer(aCtx.fX), pointer(aCtx.fTmp), n * h);
end;

procedure TDecoderModel.Route(aLogits: PSingleArray; aChosen: PIntegerArray;
  aWeights: PSingleArray);
var
  i, j, t, best, e, k: PtrInt;
  m, sum, picked: double;
  taken: boolean;
begin
  e := fConfig.Experts;
  k := fConfig.ExpertsUsed;
  // softmax in double: the probabilities of all experts
  m := aLogits^[0];
  for i := 1 to e - 1 do
    if aLogits^[i] > m then
      m := aLogits^[i];
  sum := 0;
  for i := 0 to e - 1 do
    sum := sum + Exp(aLogits^[i] - m);
  // the k highest logits (the highest probabilities), the lower id on a tie
  picked := 0;
  for j := 0 to k - 1 do
  begin
    best := -1;
    for i := 0 to e - 1 do
    begin
      taken := false;
      for t := 0 to j - 1 do
        if aChosen^[t] = i then
          taken := true;
      if not taken and ((best < 0) or (aLogits^[i] > aLogits^[best])) then
        best := i;
    end;
    aChosen^[j] := best;
    aWeights^[j] := Exp(aLogits^[best] - m) / sum;
    picked := picked + aWeights^[j];
  end;
  if fConfig.NormTopK then
    for j := 0 to k - 1 do
      aWeights^[j] := aWeights^[j] / picked;
  if fRouteTape <> nil then
    case fRouteTape.Mode of
      drmRecord:
        fRouteTape.Put(aChosen, aWeights, k);
      drmReplay:
        fRouteTape.Take(aLogits, aChosen, aWeights, e, k);
    end;
end;


{ TDecoderRouteTape }

procedure TDecoderRouteTape.Clear;
begin
  fCount := 0;
  fPos := 0;
  fRoutes := 0;
  fDiffers := 0;
  fDiffersGap := 0;
end;

function TDecoderRouteTape.Pending: PtrInt;
begin
  result := fCount - fPos;
end;

procedure TDecoderRouteTape.Put(aChosen: PIntegerArray; aWeights: PSingleArray;
  aUsed: PtrInt);
begin
  if fPos = fCount then
  begin
    // everything recorded was replayed: start over
    fCount := 0;
    fPos := 0;
  end;
  if (fCount + 1) * aUsed > length(fChosen) then
  begin
    SetLength(fChosen, NextGrow((fCount + 1) * aUsed));
    SetLength(fWeights, length(fChosen));
  end;
  MoveFast(aChosen^[0], fChosen[fCount * aUsed], aUsed * SizeOf(integer));
  MoveFast(aWeights^[0], fWeights[fCount * aUsed], aUsed * SizeOf(single));
  inc(fCount);
end;

procedure TDecoderRouteTape.Take(aLogits: PSingleArray; aChosen: PIntegerArray;
  aWeights: PSingleArray; aExperts, aUsed: PtrInt);
var
  i, j: PtrInt;
  same, mine: boolean;
  kth, best: double;
begin
  if fPos >= fCount then
    EInferError.RaiseUtf8('%: no recorded choice left (% replayed)', [self, fRoutes]);
  // the same set of experts? (both lists hold distinct ids)
  same := true;
  for j := 0 to aUsed - 1 do
  begin
    mine := false;
    for i := 0 to aUsed - 1 do
      if fChosen[fPos * aUsed + i] = aChosen^[j] then
        mine := true;
    if not mine then
      same := false;
  end;
  if not same then
  begin
    inc(fDiffers);
    // the k-th highest logit of this run, and the highest one it left out
    kth := aLogits^[aChosen^[aUsed - 1]];
    best := -1E38;
    for i := 0 to aExperts - 1 do
    begin
      mine := false;
      for j := 0 to aUsed - 1 do
        if aChosen^[j] = i then
          mine := true;
      if not mine and (aLogits^[i] > best) then
        best := aLogits^[i];
    end;
    if kth - best > fDiffersGap then
      fDiffersGap := kth - best;
  end;
  MoveFast(fChosen[fPos * aUsed], aChosen^[0], aUsed * SizeOf(integer));
  MoveFast(fWeights[fPos * aUsed], aWeights^[0], aUsed * SizeOf(single));
  inc(fPos);
  inc(fRoutes);
end;

procedure TDecoderModel.MoeBlock(aCtx: TDecoderContext; aLayer, aRows: PtrInt;
  const aOnState: TOnDecoderState);
var
  n, h, fi, k, e, r, j, c, cnt, i: PtrInt;
  w: single;
  ly: ^TDecoderLayer;
  src, dst, gb, ub, db: PSingleArray;
begin
  ly := @fLayer[aLayer];
  n := aRows;
  h := fConfig.Hidden;
  fi := fConfig.ExpertIntermediate;
  k := fConfig.ExpertsUsed;
  // pre-norm, the router: its logits, then the experts of every token
  fBackend.RmsNorm(pointer(aCtx.fX), n, h, ly^.PostNorm, fConfig.RmsEps,
    pointer(aCtx.fH));
  fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Router, fConfig.Experts,
    ly^.RouterBias, pointer(aCtx.fRoute));
  for r := 0 to n - 1 do
    Route(@aCtx.fRoute[r * fConfig.Experts], @aCtx.fChosen[r * k],
      @aCtx.fChosenW[r * k]);
  FillCharFast(pointer(aCtx.fTmp)^, n * h * SizeOf(single), 0);
  // expert by expert, in id order (as the reference adds them): its tokens
  // gathered, one SwiGLU block over them, weighted into their rows
  for e := 0 to fConfig.Experts - 1 do
  begin
    cnt := 0;
    for r := 0 to n - 1 do
      for j := 0 to k - 1 do
        if aCtx.fChosen[r * k + j] = e then
        begin
          aCtx.fRow[cnt] := r;
          aCtx.fRowW[cnt] := aCtx.fChosenW[r * k + j];
          MoveFast(aCtx.fH[r * h], aCtx.fGather[cnt * h], h * SizeOf(single));
          inc(cnt);
        end;
    if cnt = 0 then
      continue;
    gb := nil;
    ub := nil;
    db := nil;
    if ly^.ExpertGateBias <> nil then
      gb := @ly^.ExpertGateBias^[e * fi];
    if ly^.ExpertUpBias <> nil then
      ub := @ly^.ExpertUpBias^[e * fi];
    if ly^.ExpertDownBias <> nil then
      db := @ly^.ExpertDownBias^[e * h];
    fBackend.Linear(pointer(aCtx.fGather), cnt, h, ly^.ExpertGate[e], fi, gb,
      pointer(aCtx.fGate));
    fBackend.Linear(pointer(aCtx.fGather), cnt, h, ly^.ExpertUp[e], fi, ub,
      pointer(aCtx.fUp));
    fBackend.SwiGlu(pointer(aCtx.fGate), pointer(aCtx.fUp), cnt * fi,
      fConfig.SwiGluAlpha, fConfig.SwiGluLimit);
    fBackend.Linear(pointer(aCtx.fGate), cnt, fi, ly^.ExpertDown[e], h, db,
      pointer(aCtx.fExpertOut));
    for c := 0 to cnt - 1 do
    begin
      w := aCtx.fRowW[c];
      src := @aCtx.fExpertOut[c * h];
      dst := @aCtx.fTmp[aCtx.fRow[c] * h];
      for i := 0 to h - 1 do
        dst^[i] := dst^[i] + w * src^[i];
    end;
  end;
  if Assigned(aOnState) and (aLayer = 0) then
    aOnState('layer00_mlp', pointer(aCtx.fTmp), n, h);
  fBackend.Add(pointer(aCtx.fX), pointer(aCtx.fTmp), n * h);
end;

/// 'layerNN' - two digits at least, like the reference ('layer%02d')
function LayerName(aLayer: PtrInt): RawUtf8;
begin
  if aLayer < 10 then
    result := FormatUtf8('layer0%', [aLayer])
  else
    result := FormatUtf8('layer%', [aLayer]);
end;

procedure TDecoderModel.Forward(aCtx: TDecoderContext; aIds: PIntegerArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean;
  const aOnState: TOnDecoderState);
begin
  if (aIds = nil) and (aCount > 0) then
    EInferError.RaiseUtf8('%.Forward: no ids', [self]);
  ForwardRows(aCtx, aIds, nil, aCount, aLogits, aAllLogits, aOnState, nil);
end;

procedure TDecoderModel.ForwardEmbeds(aCtx: TDecoderContext; aEmbeds: PSingleArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean;
  const aOnState: TOnDecoderState; aNormed: PSingleArray);
begin
  if (aEmbeds = nil) and (aCount > 0) then
    EInferError.RaiseUtf8('%.ForwardEmbeds: no rows', [self]);
  ForwardRows(aCtx, nil, aEmbeds, aCount, aLogits, aAllLogits, aOnState, aNormed);
end;

procedure TDecoderModel.ForwardRows(aCtx: TDecoderContext; aIds: PIntegerArray;
  aEmbeds: PSingleArray; aCount: PtrInt; aLogits: PSingleArray;
  aAllLogits: boolean; const aOnState: TOnDecoderState; aNormed: PSingleArray);
var
  i, l, n, h, f, qw, past, id: PtrInt;
begin
  if aCtx = nil then
    EInferError.RaiseUtf8('%.Forward: no context', [self]);
  if (aIds <> nil) and (fEmbedSrc = nil) then
    EInferError.RaiseUtf8('%.Forward: no token embedding (ForwardEmbeds only)', [self]);
  // before the cache changes
  if (aLogits <> nil) and (fHead.Data = nil) then
    EInferError.RaiseUtf8('%.Forward: no head for logits (the normed state only)',
      [self]);
  if aCtx.fDevice <> nil then
  begin
    // the keys and values are on a device: computed there
    if aCtx.fDevice.fModel <> self then
      EInferError.RaiseUtf8('%.Forward: a context of % for another model',
        [self, aCtx.fDevice]);
    if Assigned(aOnState) then
      EInferError.RaiseUtf8('%.Forward: no states from %', [self, aCtx.fDevice]);
    if aIds <> nil then
      aCtx.fDevice.Forward(aCtx, aIds, aCount, aLogits, aAllLogits)
    else
      aCtx.fDevice.ForwardEmbeds(aCtx, aEmbeds, aCount, aLogits, aAllLogits,
        aNormed);
    exit;
  end;
  if aCount <= 0 then
    EInferError.RaiseUtf8('%.Forward: no tokens', [self]);
  past := aCtx.fLength;
  if aCount > fConfig.MaxPositions - past then
    EInferError.RaiseUtf8('%.Forward: % + % tokens exceed the % positions',
      [self, past, aCount, fConfig.MaxPositions]);
  n := aCount;
  h := fConfig.Hidden;
  f := MaxPtrInt(fConfig.Intermediate,
    MaxPtrInt(fConfig.ExpertIntermediate, fConfig.Experts));
  qw := fConfig.Heads * fConfig.HeadDim;
  if n > (High(PtrInt) div SizeOf(single)) div MaxPtrInt(f, MaxPtrInt(h, qw)) then
    EInferError.RaiseUtf8('%.Forward: % tokens too many at once', [self, n]);
  if aIds <> nil then
    for i := 0 to n - 1 do
    begin
      id := aIds^[i];
      if (id < 0) or (id >= fConfig.VocabSize) then
        EInferError.RaiseUtf8('%.Forward: token id % outside the vocabulary', [self, id]);
    end;
  Grow(aCtx, past + n, n);
  RopeFactors(aCtx, n, past);
  // embeddings: of the ids, or the rows as given
  if aIds <> nil then
    for i := 0 to n - 1 do
      InferRowToF32(fEmbedSrc^.Data, fEmbedSrc^.DType, aIds^[i], h,
        @aCtx.fX[i * h])
  else
    MoveFast(aEmbeds^[0], aCtx.fX[0], n * h * SizeOf(single));
  if Assigned(aOnState) then
    aOnState('embed', pointer(aCtx.fX), n, h);
  for l := 0 to fConfig.Layers - 1 do
  begin
    case fConfig.Spec[l].Mixer of
      dmAttention:
        AttentionBlock(aCtx, l, n, past, aOnState);
    end;
    case fConfig.Spec[l].Ffn of
      dfDense:
        DenseBlock(aCtx, l, n, aOnState);
      dfMoe:
        MoeBlock(aCtx, l, n, aOnState);
    end;
    if Assigned(aOnState) then
      aOnState(LayerName(l), pointer(aCtx.fX), n, h);
  end;
  if aIds <> nil then
    MoveFast(aIds^[0], aCtx.fIds[past], n * SizeOf(integer))
  else
    FillCharFast(aCtx.fIds[past], n * SizeOf(integer), $ff); // -1: no id
  aCtx.fLength := past + n;
  if (aLogits = nil) and (aNormed = nil) and not Assigned(aOnState) then
    exit;
  fBackend.RmsNorm(pointer(aCtx.fX), n, h, fNorm, fConfig.RmsEps, pointer(aCtx.fH));
  if Assigned(aOnState) then
    aOnState('norm', pointer(aCtx.fH), n, h);
  if aNormed <> nil then
    MoveFast(aCtx.fH[(n - 1) * h], aNormed^[0], h * SizeOf(single));
  if aLogits = nil then
    exit;
  if aAllLogits then
    fBackend.Linear(pointer(aCtx.fH), n, h, fHead, fConfig.VocabSize, nil, aLogits)
  else
    fBackend.Linear(@aCtx.fH[(n - 1) * h], 1, h, fHead, fConfig.VocabSize, nil,
      aLogits);
end;

function TDecoderModel.Reuse(aCtx: TDecoderContext; aIds: PIntegerArray;
  aCount: PtrInt): PtrInt;
begin
  if aCtx = nil then
    EInferError.RaiseUtf8('%.Reuse: no context', [self]);
  if aCtx.fDevice <> nil then
  begin
    if aCtx.fDevice.fModel <> self then
      EInferError.RaiseUtf8('%.Reuse: a context of % for another model',
        [self, aCtx.fDevice]);
    result := aCtx.fDevice.Reuse(aCtx, aIds, aCount);
    exit;
  end;
  result := 0;
  if (aCtx.fOwner = fSerial) and (aCount > 0) then
    result := aCtx.Usable(aIds, aCount - 1);
  aCtx.Truncate(result);
end;

function TDecoderModel.Greedy(aCtx: TDecoderContext; const aPrompt: array of integer;
  aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
var
  logits: TSingleDynArray;
  n, i, next: PtrInt;
  id: integer;
  stop: boolean;
begin
  result := nil;
  if length(aPrompt) = 0 then
    EInferError.RaiseUtf8('%.Greedy: empty prompt', [self]);
  if aMaxNew <= 0 then
    EInferError.RaiseUtf8('%.Greedy: % new tokens', [self, aMaxNew]);
  SetLength(logits, fConfig.VocabSize);
  Forward(aCtx, @aPrompt[0], length(aPrompt), pointer(logits));
  SetLength(result, aMaxNew);
  n := 0;
  repeat
    next := InferArgMax(pointer(logits), fConfig.VocabSize);
    result[n] := next;
    inc(n);
    stop := n = aMaxNew;
    for i := 0 to high(aStop) do
      if aStop[i] = next then
        stop := true;
    // every returned token enters the cache, so a conversation can go on;
    // only a full context leaves the last one out
    if aCtx.fLength >= fConfig.MaxPositions then
      break;
    id := next;
    if stop then
      Forward(aCtx, @id, 1, nil)
    else
      Forward(aCtx, @id, 1, pointer(logits));
  until stop;
  SetLength(result, n);
end;


{ registration }

var
  DecoderClasses: array of TDecoderModelClass;

procedure RegisterDecoderModel(aClass: TDecoderModelClass);
var
  n: PtrInt;
begin
  if aClass = nil then
    exit;
  n := length(DecoderClasses);
  SetLength(DecoderClasses, n + 1);
  DecoderClasses[n] := aClass;
end;

function DecoderModelClass(const aModelType: RawUtf8): TDecoderModelClass;
var
  i: PtrInt;
begin
  for i := 0 to high(DecoderClasses) do
  begin
    result := DecoderClasses[i];
    if result.ModelType = aModelType then
      exit;
  end;
  result := nil;
end;

function DecoderModelClassOfArchitecture(const aArchitecture: RawUtf8): TDecoderModelClass;
var
  i: PtrInt;
begin
  for i := 0 to high(DecoderClasses) do
  begin
    result := DecoderClasses[i];
    if result.Architecture = aArchitecture then
      exit;
  end;
  result := nil;
end;

function DecoderModelClassOfFolder(const aFolder: TFileName): TDecoderModelClass;
var
  doc: TDocVariantData;
  fn: TFileName;
begin
  fn := IncludeTrailingPathDelimiter(aFolder) + 'config.json';
  if not doc.InitJsonFromFile(fn, JSON_FAST_FLOAT) then
    EInferError.RaiseUtf8('DecoderModelClassOfFolder: cannot read %', [fn]);
  result := DecoderModelClass(doc.U['model_type']);
  if result = nil then
    EInferError.RaiseUtf8('DecoderModelClassOfFolder: model_type "%" of % is ' +
      'not supported', [doc.U['model_type'], fn]);
end;

function NewDecoderModel(const aFolder: TFileName; aBackend: TInferBackend;
  aWeights: TInferDType): TDecoderModel;
begin
  result := DecoderModelClassOfFolder(aFolder).Create(aFolder, aBackend, aWeights);
end;

function NewDecoderModelFromGguf(const aFileName: TFileName;
  aBackend: TInferBackend): TDecoderModel;
var
  gguf: TGgufFile;
  arch: RawUtf8;
  c: TDecoderModelClass;
begin
  // the file is mapped twice: once for its architecture, once by the family
  gguf := TGgufFile.Create(aFileName);
  try
    arch := gguf.Str('general.architecture');
  finally
    gguf.Free;
  end;
  c := DecoderModelClassOfArchitecture(arch);
  if c = nil then
    EInferError.RaiseUtf8('NewDecoderModelFromGguf: architecture "%" of % is ' +
      'not supported', [arch, aFileName]);
  result := c.CreateFromGguf(aFileName, aBackend);
end;

end.
