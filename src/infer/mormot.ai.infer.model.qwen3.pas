/// native inference engine - Qwen3 decoder (causal language model)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room from the published architecture (looked up in Hugging Face's
//   modeling_qwen3.py, Apache-2.0, no code taken; llama2.c, MIT, for the shape
//   of a decoder loop): token embedding, then per layer pre-norm self-attention
//   (RMSNorm, Q/K/V projections, RMSNorm of every Q and K head, rotary
//   positions, grouped-query causal attention, output projection) and a
//   pre-norm SwiGLU feed-forward block, both with a residual; a final RMSNorm
//   and the language-model head (tied to the embedding or its own matrix)
// - the family of mormot.ai.infer.decoder: this unit reads the settings of
//   config.json or of the GGUF metadata, maps the tensor names and loads the
//   weights; the forward pass, the KV cache and the device hand-over are
//   TDecoderModel's
// - head_dim comes from the configuration: Qwen3-0.6B has 16 heads of 128 for
//   a hidden size of 1024, not 1024 / 16
// - or a llama.cpp GGUF file of the qwen3 architecture (CreateFromGguf): its
//   metadata for the settings, its Q8_0 / Q4_0 / Q5_0 / Q4_K / Q6_K matrices
//   used in place
// - TQwen2Model: Qwen2 and Qwen2.5 (model_type and GGUF architecture qwen2,
//   looked up in modeling_qwen2.py): the same decoder with a bias on the Q, K
//   and V projections and no RMSNorm per head; config.json has no head_dim
//   (hidden_size / num_attention_heads), nor has the GGUF attention.key_length
// - a sliding window (config.json use_sliding_window, sliding_window, and
//   max_window_layers or layer_types, as Qwen3Config and Qwen2Config derive
//   them) on the layers transformers gives one; GGUF files of these families
//   carry no window, one that names attention.sliding_window is refused
// - TSmolLM3Model: SmolLM3 (model_type smollm3, looked up in
//   modeling_smollm3.py and configuration_smollm3.py): no bias, no head norm,
//   no rotary positions in the layers no_rope_layers marks 0 (NoPE; by default
//   every no_rope_layer_interval-th layer), a sliding window on the layers
//   layer_types names (by default the NoPE layers when use_sliding_window);
//   config.json only - llama.cpp's smollm3 GGUF hard-codes the NoPE layers and
//   is refused until checked against it
// - TQwen3MoeModel: Qwen3-MoE (model_type qwen3_moe, GGUF architecture
//   qwen3moe, e.g. Qwen3-30B-A3B) - the same attention, a mixture of experts
//   as feed-forward block (looked up in modeling_qwen3_moe.py: a softmax router,
//   the top num_experts_per_tok experts, renormed when norm_topk_prob; dense
//   layers where mlp_only_layers or decoder_sparse_step say so)
unit mormot.ai.infer.model.qwen3;

{
  *****************************************************************************

    - TQwen3Model: the qwen3 family of TDecoderModel
    - TQwen3MoeModel: Qwen3-MoE, the experts as feed-forward block
    - TQwen2Model: Qwen2 / Qwen2.5, a bias on Q, K and V instead of the head norms
    - TSmolLM3Model: SmolLM3, layers without rotary positions (NoPE)
    - TQwen3Config, TQwen3Context, TQwen3Device...: the decoder types under
      their former names

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
  mormot.ai.infer.loader.gguf,
  mormot.ai.infer.backend,
  mormot.ai.infer.decoder;

type
  /// the decoder types under their former names
  TQwen3Config = TDecoderConfig;
  TQwen3Layer = TDecoderLayer;
  TQwen3LayerDynArray = TDecoderLayerDynArray;
  TOnQwen3State = TOnDecoderState;
  TQwen3Context = TDecoderContext;
  TQwen3Device = TDecoderDevice;

  /// a Qwen3 causal language model over a safetensors folder or a GGUF file
  // - Create (config.json and model.safetensors) and CreateFromGguf as
  // TDecoderModel's; model_type and architecture qwen3
  TQwen3Model = class(TDecoderModel)
  protected
    procedure LoadConfig(const aConfigFile: TFileName); override;
    procedure LoadGgufConfig; override;
    procedure LoadWeights; override;
    function GgufName(const aName: RawUtf8): RawUtf8; override;
    /// a positive count of the GGUF metadata, bounded before it becomes a
    // PtrInt: a value beyond 32 bits must not wrap into a valid one
    function GgufCount(const aKey: RawUtf8): PtrInt;
    /// the feed-forward settings of config.json (aDoc), before CheckConfig -
    // nothing here: every layer dense
    procedure LoadFfnConfig(const aDoc: TDocVariantData;
      const aConfigFile: TFileName); virtual;
    /// the same from the GGUF metadata - nothing here
    procedure LoadGgufFfnConfig; virtual;
    /// the feed-forward weights of layer aLayer (tensor names from aPrefix,
    // 'model.layers.N.'); returns how many tensors they took - here the dense
    // gate, up and down
    function LoadFfn(aLayer: PtrInt; const aPrefix: RawUtf8): PtrInt; virtual;
    /// the attention settings of config.json (aDoc) per layer, before
    // CheckConfig - here the sliding window of Qwen3Config / Qwen2Config
    procedure LoadMixerConfig(const aDoc: TDocVariantData;
      const aConfigFile: TFileName); virtual;
    /// the window of every layer from layer_types, when aDoc has them: true
    // when it had; a sliding_attention layer needs aWindow > 0
    function LoadLayerTypes(const aDoc: TDocVariantData; aWindow: PtrInt): boolean;
    /// fConfig.Spec sized to the layers - false while their count is invalid
    // (left to CheckConfig)
    function SpecSized: boolean;
    /// true: an RMSNorm of every Q and K head (Qwen3); false: none, head_dim
    // from hidden / heads when the settings do not name it (Qwen2, SmolLM3)
    class function HeadNorm: boolean; virtual;
    /// true: a bias on the Q, K and V projections (Qwen2); false: none
    class function QkvBias: boolean; virtual;
  public
    /// 'qwen3'
    class function ModelType: RawUtf8; override;
    /// 'qwen3'
    class function Architecture: RawUtf8; override;
  end;

  /// a Qwen2 or Qwen2.5 causal language model (e.g. Qwen2.5-0.5B-Instruct)
  // - config.json (model_type qwen2) or GGUF (architecture qwen2); the Q, K and
  // V projections with a bias (q_proj.bias ... / attn_q.bias ...), no q_norm
  // or k_norm
  TQwen2Model = class(TQwen3Model)
  protected
    class function HeadNorm: boolean; override;
    class function QkvBias: boolean; override;
  public
    /// 'qwen2'
    class function ModelType: RawUtf8; override;
    /// 'qwen2'
    class function Architecture: RawUtf8; override;
  end;

  /// a SmolLM3 causal language model (e.g. SmolLM3-3B)
  // - config.json (model_type smollm3): no bias (attention_bias and mlp_bias
  // false), no head norm, head_dim hidden / heads; no_rope_layers (1: rotary
  // positions, 0: none) or every no_rope_layer_interval-th layer without;
  // layer_types, or by default a sliding window on the layers without rotary
  // positions when use_sliding_window
  // - no GGUF: llama.cpp fixes the NoPE layers instead of reading them
  TSmolLM3Model = class(TQwen3Model)
  protected
    procedure LoadGgufConfig; override;
    procedure LoadMixerConfig(const aDoc: TDocVariantData;
      const aConfigFile: TFileName); override;
    class function HeadNorm: boolean; override;
  public
    /// 'smollm3'
    class function ModelType: RawUtf8; override;
    /// 'smollm3'
    class function Architecture: RawUtf8; override;
  end;

  /// a Qwen3-MoE causal language model (e.g. Qwen3-30B-A3B)
  // - config.json: num_experts, num_experts_per_tok, moe_intermediate_size,
  // norm_topk_prob, decoder_sparse_step and mlp_only_layers; the experts one by
  // one (model.layers.N.mlp.experts.E.gate_proj ...), the router mlp.gate
  // - GGUF (architecture qwen3moe): expert_count, expert_used_count,
  // expert_feed_forward_length; the experts of a layer as one 3-D tensor each
  // (ffn_gate_exps, ffn_up_exps, ffn_down_exps), the router ffn_gate_inp; a
  // layer without ffn_gate_inp is dense; the weights always renormed, as
  // llama.cpp computes qwen3moe
  // - the router stays F32 when the matrices are quantized at load
  TQwen3MoeModel = class(TQwen3Model)
  protected
    procedure LoadFfnConfig(const aDoc: TDocVariantData;
      const aConfigFile: TFileName); override;
    procedure LoadGgufFfnConfig; override;
    function LoadFfn(aLayer: PtrInt; const aPrefix: RawUtf8): PtrInt; override;
    /// Qwen3MoeConfig knows no max_window_layers nor layer_types: with
    // use_sliding_window every layer slides
    procedure LoadMixerConfig(const aDoc: TDocVariantData;
      const aConfigFile: TFileName); override;
  public
    /// 'qwen3_moe'
    class function ModelType: RawUtf8; override;
    /// 'qwen3moe'
    class function Architecture: RawUtf8; override;
  end;


implementation


{ TQwen3Model }

class function TQwen3Model.ModelType: RawUtf8;
begin
  result := 'qwen3';
end;

class function TQwen3Model.Architecture: RawUtf8;
begin
  result := 'qwen3';
end;

class function TQwen3Model.HeadNorm: boolean;
begin
  result := true;
end;

class function TQwen3Model.QkvBias: boolean;
begin
  result := false;
end;

const
  REQUIRED_KEYS: array[0..8] of RawUtf8 = (
    'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
    'num_key_value_heads', 'intermediate_size', 'max_position_embeddings',
    'rms_norm_eps', 'hidden_act');

/// hidden / heads, for a family whose settings do not name the head size -
// raises unless it divides
function HeadDimOf(aSelf: TObject; aHidden, aHeads: PtrInt): PtrInt;
begin
  if (aHeads <= 0) or (aHidden <= 0) or (aHidden mod aHeads <> 0) then
    EInferError.RaiseUtf8('%: a hidden size of % for % heads', [aSelf, aHidden, aHeads]);
  result := aHidden div aHeads;
end;

procedure TQwen3Model.LoadConfig(const aConfigFile: TFileName);
var
  doc: TDocVariantData;
  d, rope, mrope: PDocVariantData;
  sections: PDocVariantData;
  i, sum: PtrInt;
  k: RawUtf8;
begin
  d := ReadConfig(aConfigFile, doc);
  // a missing key would read as 0: refused, also head_dim for Qwen3 (no
  // hidden / heads fallback - that is exactly the value Qwen3 does not have)
  for i := 0 to high(REQUIRED_KEYS) do
    if d^.GetValueIndex(REQUIRED_KEYS[i]) < 0 then
      EInferError.RaiseUtf8('%: % has no "%"', [self, aConfigFile, REQUIRED_KEYS[i]]);
  if HeadNorm and (d^.GetValueIndex('head_dim') < 0) then
    EInferError.RaiseUtf8('%: % has no "head_dim"', [self, aConfigFile]);
  if d^.U['hidden_act'] <> 'silu' then
    EInferError.RaiseUtf8('%: hidden_act "%" not supported', [self, d^.U['hidden_act']]);
  if d^.B['attention_bias'] or d^.B['mlp_bias'] then
    EInferError.RaiseUtf8('%: attention_bias or mlp_bias not supported', [self]);
  // rope_theta at the top (transformers 4) or in rope_parameters (5), and
  // plain RoPE only: a scaling would change every angle
  if d^.GetAsObject('rope_parameters', rope) then
  begin
    if (rope^.U['rope_type'] <> '') and (rope^.U['rope_type'] <> 'default') then
      EInferError.RaiseUtf8('%: rope_type "%" not supported', [self, rope^.U['rope_type']]);
    fConfig.RopeTheta := rope^.D['rope_theta'];
  end
  else
    fConfig.RopeTheta := d^.D['rope_theta'];
  mrope := nil;
  if d^.GetAsObject('rope_scaling', rope) and (rope^.Count > 0) then
    // a stack (the talker of Qwen3-TTS) may name a multimodal RoPE: taken as
    // plain RoPE - the same angles while the positions of its three sections
    // are equal, as for one sequence fed in order (checked below)
    if (fStackPrefix <> '') and rope^.GetAsArray('mrope_section', sections) then
      mrope := rope
    else
      EInferError.RaiseUtf8('%: rope_scaling not supported', [self]);
  fConfig.VocabSize := d^.I['vocab_size'];
  fConfig.Hidden := d^.I['hidden_size'];
  fConfig.Layers := d^.I['num_hidden_layers'];
  fConfig.Heads := d^.I['num_attention_heads'];
  fConfig.KvHeads := d^.I['num_key_value_heads'];
  if d^.GetValueIndex('head_dim') >= 0 then
    fConfig.HeadDim := d^.I['head_dim']
  else
    fConfig.HeadDim := HeadDimOf(self, fConfig.Hidden, fConfig.Heads);
  if mrope <> nil then
  begin
    // the sections cover every frequency, nothing else in the object
    sum := 0;
    for i := 0 to sections^.Count - 1 do
    begin
      k := VariantToUtf8(sections^.Values[i]);
      if Utf8ToInteger(k, 0) <= 0 then
        EInferError.RaiseUtf8('%: an mrope_section of "%"', [self, k]);
      inc(sum, Utf8ToInteger(k, 0));
    end;
    if sum <> fConfig.HeadDim shr 1 then
      EInferError.RaiseUtf8('%: mrope_section sums to %, not % frequencies',
        [self, sum, fConfig.HeadDim shr 1]);
    for i := 0 to mrope^.Count - 1 do
    begin
      k := mrope^.Names[i];
      if (k = 'rope_type') or (k = 'type') then
      begin
        if VariantToUtf8(mrope^.Values[i]) <> 'default' then
          EInferError.RaiseUtf8('%: rope_scaling % "%" not supported',
            [self, k, VariantToUtf8(mrope^.Values[i])]);
      end
      else if (k <> 'mrope_section') and (k <> 'interleaved') then
        EInferError.RaiseUtf8('%: rope_scaling.% not supported', [self, k]);
    end;
  end;
  fConfig.Intermediate := d^.I['intermediate_size'];
  fConfig.MaxPositions := d^.I['max_position_embeddings'];
  fConfig.RmsEps := d^.D['rms_norm_eps'];
  fConfig.TiedEmbeddings := d^.B['tie_word_embeddings'];
  LoadFfnConfig(d^, aConfigFile);
  LoadMixerConfig(d^, aConfigFile);
  CheckConfig(aConfigFile);
end;

procedure TQwen3Model.LoadFfnConfig(const aDoc: TDocVariantData;
  const aConfigFile: TFileName);
begin
end;

function TQwen3Model.SpecSized: boolean;
begin
  result := (fConfig.Layers > 0) and (fConfig.Layers <= 1024);
  if result and (fConfig.Spec = nil) then
    SetLength(fConfig.Spec, fConfig.Layers);
end;

function TQwen3Model.LoadLayerTypes(const aDoc: TDocVariantData;
  aWindow: PtrInt): boolean;
var
  types: PDocVariantData;
  t: RawUtf8;
  i: PtrInt;
begin
  result := aDoc.GetAsArray('layer_types', types);
  if not result then
    exit;
  if types^.Count <> fConfig.Layers then
    EInferError.RaiseUtf8('%: % layer types for % layers',
      [self, types^.Count, fConfig.Layers]);
  for i := 0 to types^.Count - 1 do
  begin
    t := VariantToUtf8(types^.Values[i]);
    if t = 'sliding_attention' then
    begin
      // transformers masks it with sliding_window: there must be one
      if aWindow <= 0 then
        EInferError.RaiseUtf8('%: layer % slides without a sliding window', [self, i]);
      fConfig.Spec[i].Window := aWindow;
    end
    else if t <> 'full_attention' then
      EInferError.RaiseUtf8('%: layer type "%" not supported', [self, t]);
  end;
end;

/// sliding_window as the configuration class leaves it: 0 when aGated and
// not use_sliding_window (Qwen3Config / Qwen2Config set it to None then), or
// when it is null; aDefault when the key is missing (the class default) -
// raises for a size that is not positive
function SlidingWindowOf(aSelf: TObject; const aDoc: TDocVariantData;
  aDefault: PtrInt; aGated: boolean): PtrInt;
var
  i: PtrInt;
  v: Int64;
begin
  result := 0;
  if aGated and not aDoc.B['use_sliding_window'] then
    exit;
  i := aDoc.GetValueIndex('sliding_window');
  if i < 0 then
  begin
    result := aDefault;
    exit;
  end;
  if VarIsEmptyOrNull(aDoc.Values[i]) then
    exit;
  v := VariantToInt64Def(aDoc.Values[i], 0);
  if (v <= 0) or (v > 1 shl 24) then
    EInferError.RaiseUtf8('%: a sliding_window of %', [aSelf, v]);
  result := v;
end;

procedure TQwen3Model.LoadMixerConfig(const aDoc: TDocVariantData;
  const aConfigFile: TFileName);
var
  w, first, i: PtrInt;
begin
  if not SpecSized then
    exit;
  // as Qwen3Config / Qwen2Config: no window unless use_sliding_window (both
  // default to a sliding_window of 4096 and max_window_layers of 28); then
  // layer_types, or the layers from max_window_layers on - a sliding layer
  // type without a window is refused (transformers cannot mask it either)
  w := SlidingWindowOf(self, aDoc, 4096, true);
  if LoadLayerTypes(aDoc, w) or (w = 0) then
    exit;
  first := 28;
  if aDoc.GetValueIndex('max_window_layers') >= 0 then
    first := aDoc.I['max_window_layers'];
  if first < 0 then
    first := 0;
  for i := first to fConfig.Layers - 1 do
    fConfig.Spec[i].Window := w;
end;

function TQwen3Model.GgufCount(const aKey: RawUtf8): PtrInt;
var
  v: Int64;
begin
  v := fGguf.Int(aKey);
  if (v <= 0) or (v > 1 shl 30) then
    EInferError.RaiseUtf8('%: % = % out of range', [self, aKey, v]);
  result := v;
end;

procedure TQwen3Model.LoadGgufFfnConfig;
begin
end;

procedure TQwen3Model.LoadGgufConfig;
var
  emb: PInferTensor;
  a: RawUtf8;
begin
  if fGguf.Str('general.architecture') <> Architecture then
    EInferError.RaiseUtf8('%: architecture "%" is not %',
      [self, fGguf.Str('general.architecture'), Architecture]);
  a := Architecture + '.';
  // a scaled RoPE or a sliding window would change the arithmetic: refused
  if fGguf.Has(a + 'rope.scaling.type') and
     (fGguf.Str(a + 'rope.scaling.type') <> 'none') then
    EInferError.RaiseUtf8('%: rope scaling "%" not supported',
      [self, fGguf.Str(a + 'rope.scaling.type')]);
  if fGguf.Has(a + 'attention.sliding_window') then
    EInferError.RaiseUtf8('%: sliding window attention not supported', [self]);
  emb := fGguf.Tensor('token_embd.weight');
  if length(emb^.Shape) <> 2 then
    EInferError.RaiseUtf8('%: token_embd.weight is shaped %',
      [self, InferShapeText(emb^.Shape)]);
  if (emb^.Shape[0] <= 0) or (emb^.Shape[0] > 1 shl 22) then
    EInferError.RaiseUtf8('%: a vocabulary of % tokens', [self, emb^.Shape[0]]);
  fConfig.VocabSize := emb^.Shape[0];
  fConfig.Hidden := GgufCount(a + 'embedding_length');
  fConfig.Layers := GgufCount(a + 'block_count');
  fConfig.Heads := GgufCount(a + 'attention.head_count');
  fConfig.KvHeads := GgufCount(a + 'attention.head_count_kv');
  // qwen2 files name no key length: hidden / heads
  if HeadNorm or fGguf.Has(a + 'attention.key_length') then
    fConfig.HeadDim := GgufCount(a + 'attention.key_length')
  else
    fConfig.HeadDim := HeadDimOf(self, fConfig.Hidden, fConfig.Heads);
  // RoPE over a part of each head only would be another arithmetic
  if fGguf.Has(a + 'rope.dimension_count') and
     (GgufCount(a + 'rope.dimension_count') <> fConfig.HeadDim) then
    EInferError.RaiseUtf8('%: RoPE over % of % dimensions not supported',
      [self, GgufCount(a + 'rope.dimension_count'), fConfig.HeadDim]);
  if fGguf.Has(a + 'attention.value_length') and
     (GgufCount(a + 'attention.value_length') <> fConfig.HeadDim) then
    EInferError.RaiseUtf8('%: key and value heads of different sizes', [self]);
  fConfig.Intermediate := GgufCount(a + 'feed_forward_length');
  fConfig.MaxPositions := GgufCount(a + 'context_length');
  fConfig.RmsEps := fGguf.Float(a + 'attention.layer_norm_rms_epsilon');
  fConfig.RopeTheta := fGguf.Float(a + 'rope.freq_base');
  // llama.cpp ties the head to the embedding by leaving output.weight out
  fConfig.TiedEmbeddings := fGguf.FindTensor('output.weight') = nil;
  LoadGgufFfnConfig;
  CheckConfig(fGguf.FileName);
end;

const
  /// Hugging Face names and llama.cpp's for the tensors of one layer
  LAYER_NAMES: array[0..14, 0..1] of RawUtf8 = (
    ('self_attn.q_proj.weight', 'attn_q.weight'),
    ('self_attn.k_proj.weight', 'attn_k.weight'),
    ('self_attn.v_proj.weight', 'attn_v.weight'),
    ('self_attn.o_proj.weight', 'attn_output.weight'),
    ('self_attn.q_norm.weight', 'attn_q_norm.weight'),
    ('self_attn.k_norm.weight', 'attn_k_norm.weight'),
    ('mlp.gate_proj.weight', 'ffn_gate.weight'),
    ('mlp.up_proj.weight', 'ffn_up.weight'),
    ('mlp.down_proj.weight', 'ffn_down.weight'),
    ('input_layernorm.weight', 'attn_norm.weight'),
    ('post_attention_layernorm.weight', 'ffn_norm.weight'),
    ('mlp.gate.weight', 'ffn_gate_inp.weight'), // the router of Qwen3-MoE
    ('self_attn.q_proj.bias', 'attn_q.bias'), // the biases of Qwen2
    ('self_attn.k_proj.bias', 'attn_k.bias'),
    ('self_attn.v_proj.bias', 'attn_v.bias'));

function TQwen3Model.GgufName(const aName: RawUtf8): RawUtf8;
var
  p, i: PtrInt;
  layer, rest: RawUtf8;
begin
  result := '';
  if aName = 'model.embed_tokens.weight' then
    result := 'token_embd.weight'
  else if aName = 'model.norm.weight' then
    result := 'output_norm.weight'
  else if aName = 'lm_head.weight' then
    result := 'output.weight'
  else if IdemPChar(pointer(aName), 'MODEL.LAYERS.') then
  begin
    rest := copy(aName, 14, maxInt);
    p := PosEx('.', rest);
    if p <= 1 then
      exit;
    layer := copy(rest, 1, p - 1);
    rest := copy(rest, p + 1, maxInt);
    for i := 0 to high(LAYER_NAMES) do
      if rest = LAYER_NAMES[i, 0] then
      begin
        result := 'blk.' + layer + '.' + LAYER_NAMES[i, 1];
        exit;
      end;
  end;
end;

procedure TQwen3Model.LoadWeights;
var
  i, used, before: PtrInt;
  h, qw, kvw: Int64;
  p: RawUtf8;
  l: ^TDecoderLayer;
  head, emb: PInferTensor;
  headType: TInferDType;
begin
  // with Q4_0 the head stays Q8_0, as llama.cpp keeps the output matrix finer
  // than the layers: it decides every token (a GGUF file brings its own)
  headType := fWeights;
  if headType = idtQ4_0 then
    headType := idtQ8_0;
  h := fConfig.Hidden;
  qw := fConfig.Heads * fConfig.HeadDim;
  kvw := fConfig.KvHeads * fConfig.HeadDim;
  used := 1;
  if (fStackPrefix <> '') and (fStackEmbed = '') then
  begin
    // a stack fed with embedding rows only: no token embedding, no tied head
    fEmbedSrc := nil;
    if fConfig.TiedEmbeddings and (fStackHead <> '') then
      EInferError.RaiseUtf8('%: a head tied to no embedding', [self]);
  end
  else
  begin
    fEmbedSrc := Checked('model.embed_tokens.weight', [fConfig.VocabSize, h],
      fGguf <> nil);
    // read per token: its scales checked here, also when the head is untied
    if fEmbedSrc^.DType in INFER_BLOCK_DTYPES then
      CheckBlockScales(fEmbedSrc, 'model.embed_tokens.weight');
    if (fGguf = nil) and (fWeights = idtF32) then
      fEmbed := InferWeight(Weight('model.embed_tokens.weight', [fConfig.VocabSize, h]),
        idtF32);
    inc(used);
  end;
  fNorm := Weight('model.norm.weight', [h]);
  head := Find('lm_head.weight');
  if (fStackPrefix <> '') and (fStackHead = '') then
    // a stack without a head (CreateStack): no logits, the normed state only
  else if fConfig.TiedEmbeddings then
  begin
    // a tied head may still be stored: only when it is the very same matrix
    if head <> nil then
    begin
      emb := Find('model.embed_tokens.weight');
      if (head^.DType <> emb^.DType) or (head^.Bytes <> emb^.Bytes) or
         not CompareMem(head^.Data, emb^.Data, head^.Bytes) then
        EInferError.RaiseUtf8('%: tie_word_embeddings but lm_head.weight differs ' +
          'from the embedding', [self]);
      inc(used);
    end;
    if (fGguf = nil) and (fWeights = idtF32) then
      fHead := fEmbed
    else
      fHead := Matrix('model.embed_tokens.weight', fConfig.VocabSize, h, headType);
  end
  else
  begin
    fHead := Matrix('lm_head.weight', fConfig.VocabSize, h, headType);
    inc(used);
  end;
  SetLength(fLayer, fConfig.Layers);
  before := used;
  for i := 0 to fConfig.Layers - 1 do
  begin
    p := FormatUtf8('model.layers.%.', [i]);
    l := @fLayer[i];
    l^.Q := Matrix(p + 'self_attn.q_proj.weight', qw, h, fWeights);
    l^.K := Matrix(p + 'self_attn.k_proj.weight', kvw, h, fWeights);
    l^.V := Matrix(p + 'self_attn.v_proj.weight', kvw, h, fWeights);
    l^.O := Matrix(p + 'self_attn.o_proj.weight', h, qw, fWeights);
    if HeadNorm then
    begin
      l^.QNorm := Weight(p + 'self_attn.q_norm.weight', [fConfig.HeadDim]);
      l^.KNorm := Weight(p + 'self_attn.k_norm.weight', [fConfig.HeadDim]);
      inc(used, 2);
    end;
    if QkvBias then
    begin
      l^.QBias := Weight(p + 'self_attn.q_proj.bias', [qw]);
      l^.KBias := Weight(p + 'self_attn.k_proj.bias', [kvw]);
      l^.VBias := Weight(p + 'self_attn.v_proj.bias', [kvw]);
      inc(used, 3);
    end;
    l^.InputNorm := Weight(p + 'input_layernorm.weight', [h]);
    l^.PostNorm := Weight(p + 'post_attention_layernorm.weight', [h]);
    inc(used, 6 + LoadFfn(i, p));
  end;
  // an unknown tensor means a model we do not understand - in a stack, only
  // its layers are its own (the file holds other models too)
  if fStackPrefix <> '' then
  begin
    if used - before <> TensorCountOf(fStackPrefix + 'layers.') then
      EInferError.RaiseUtf8('%: % tensors under %layers., % understood',
        [self, TensorCountOf(fStackPrefix + 'layers.'), fStackPrefix, used - before]);
  end
  else if used <> TensorCount then
    EInferError.RaiseUtf8('%: % tensors in the file, % understood',
      [self, TensorCount, used]);
end;

function TQwen3Model.LoadFfn(aLayer: PtrInt; const aPrefix: RawUtf8): PtrInt;
var
  l: ^TDecoderLayer;
  h, f: Int64;
begin
  l := @fLayer[aLayer];
  h := fConfig.Hidden;
  f := fConfig.Intermediate;
  l^.Gate := Matrix(aPrefix + 'mlp.gate_proj.weight', f, h, fWeights);
  l^.Up := Matrix(aPrefix + 'mlp.up_proj.weight', f, h, fWeights);
  l^.Down := Matrix(aPrefix + 'mlp.down_proj.weight', h, f, fWeights);
  result := 3;
end;


{ TQwen2Model }

class function TQwen2Model.ModelType: RawUtf8;
begin
  result := 'qwen2';
end;

class function TQwen2Model.Architecture: RawUtf8;
begin
  result := 'qwen2';
end;

class function TQwen2Model.HeadNorm: boolean;
begin
  result := false;
end;

class function TQwen2Model.QkvBias: boolean;
begin
  result := true;
end;


{ TSmolLM3Model }

class function TSmolLM3Model.ModelType: RawUtf8;
begin
  result := 'smollm3';
end;

class function TSmolLM3Model.Architecture: RawUtf8;
begin
  result := 'smollm3';
end;

class function TSmolLM3Model.HeadNorm: boolean;
begin
  result := false;
end;

procedure TSmolLM3Model.LoadGgufConfig;
begin
  EInferError.RaiseUtf8('%: SmolLM3 from GGUF not supported', [self]);
end;

procedure TSmolLM3Model.LoadMixerConfig(const aDoc: TDocVariantData;
  const aConfigFile: TFileName);
var
  rope: PDocVariantData;
  w, every, i: PtrInt;
  v: Int64;
begin
  if not SpecSized then
    exit;
  // as SmolLM3Config: no_rope_layers (1 = rotary positions), else every
  // no_rope_layer_interval-th layer without
  if aDoc.GetAsArray('no_rope_layers', rope) then
  begin
    if rope^.Count <> fConfig.Layers then
      EInferError.RaiseUtf8('%: % no_rope_layers for % layers',
        [self, rope^.Count, fConfig.Layers]);
    for i := 0 to rope^.Count - 1 do
    begin
      v := VariantToInt64Def(rope^.Values[i], -1);
      if (v <> 0) and (v <> 1) then
        EInferError.RaiseUtf8('%: no_rope_layers[%] = %', [self, i, v]);
      fConfig.Spec[i].NoRope := v = 0;
    end;
  end
  else
  begin
    every := 4;
    if aDoc.GetValueIndex('no_rope_layer_interval') >= 0 then
      every := aDoc.I['no_rope_layer_interval'];
    if every <= 0 then
      EInferError.RaiseUtf8('%: no_rope_layer_interval = %', [self, every]);
    for i := 0 to fConfig.Layers - 1 do
      fConfig.Spec[i].NoRope := (i + 1) mod every = 0;
  end;
  // SmolLM3Config keeps sliding_window whatever use_sliding_window says, and
  // the model masks by layer_types: explicit layer types slide with it; else
  // a window on the layers without rotary positions when use_sliding_window
  if LoadLayerTypes(aDoc, SlidingWindowOf(self, aDoc, 0, false)) then
    exit;
  w := SlidingWindowOf(self, aDoc, 0, true);
  if w = 0 then
    exit;
  for i := 0 to fConfig.Layers - 1 do
    if fConfig.Spec[i].NoRope then
      fConfig.Spec[i].Window := w;
end;


{ TQwen3MoeModel }

class function TQwen3MoeModel.ModelType: RawUtf8;
begin
  result := 'qwen3_moe';
end;

class function TQwen3MoeModel.Architecture: RawUtf8;
begin
  result := 'qwen3moe';
end;

const
  MOE_KEYS: array[0..4] of RawUtf8 = (
    'num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
    'norm_topk_prob', 'decoder_sparse_step');

procedure TQwen3MoeModel.LoadFfnConfig(const aDoc: TDocVariantData;
  const aConfigFile: TFileName);
var
  only: PDocVariantData;
  i, j, step: PtrInt;
  dense: boolean;
begin
  for i := 0 to high(MOE_KEYS) do
    if aDoc.GetValueIndex(MOE_KEYS[i]) < 0 then
      EInferError.RaiseUtf8('%: % has no "%"', [self, aConfigFile, MOE_KEYS[i]]);
  fConfig.Experts := aDoc.I['num_experts'];
  fConfig.ExpertsUsed := aDoc.I['num_experts_per_tok'];
  fConfig.ExpertIntermediate := aDoc.I['moe_intermediate_size'];
  fConfig.NormTopK := aDoc.B['norm_topk_prob'];
  step := aDoc.I['decoder_sparse_step'];
  if step <= 0 then
    EInferError.RaiseUtf8('%: decoder_sparse_step = %', [self, step]);
  // as Qwen3MoeDecoderLayer: experts unless the layer is in mlp_only_layers
  // or (index + 1) is no multiple of decoder_sparse_step - an invalid layer
  // count is left to CheckConfig
  if (fConfig.Layers <= 0) or (fConfig.Layers > 1024) then
    exit;
  if not aDoc.GetAsArray('mlp_only_layers', only) then
    only := nil;
  SetLength(fConfig.Spec, fConfig.Layers);
  for i := 0 to fConfig.Layers - 1 do
  begin
    dense := (i + 1) mod step <> 0;
    if only <> nil then
      for j := 0 to only^.Count - 1 do
        if VariantToInt64Def(only^.Values[j], -1) = i then
          dense := true;
    if not dense then
      fConfig.Spec[i].Ffn := dfMoe;
  end;
end;

procedure TQwen3MoeModel.LoadMixerConfig(const aDoc: TDocVariantData;
  const aConfigFile: TFileName);
var
  w, i: PtrInt;
begin
  if not SpecSized then
    exit;
  // as modeling_qwen3_moe.py: one mask for all layers, sliding when
  // config.sliding_window is set (only with use_sliding_window)
  w := SlidingWindowOf(self, aDoc, 4096, true);
  for i := 0 to fConfig.Layers - 1 do
    fConfig.Spec[i].Window := w;
end;

procedure TQwen3MoeModel.LoadGgufFfnConfig;
var
  a: RawUtf8;
  i: PtrInt;
begin
  a := Architecture + '.';
  fConfig.Experts := GgufCount(a + 'expert_count');
  fConfig.ExpertsUsed := GgufCount(a + 'expert_used_count');
  fConfig.ExpertIntermediate := GgufCount(a + 'expert_feed_forward_length');
  fConfig.NormTopK := true;
  if (fConfig.Layers <= 0) or (fConfig.Layers > 1024) then
    exit;
  SetLength(fConfig.Spec, fConfig.Layers);
  for i := 0 to fConfig.Layers - 1 do
    if fGguf.FindTensor(FormatUtf8('blk.%.ffn_gate_inp.weight', [i])) <> nil then
      fConfig.Spec[i].Ffn := dfMoe;
end;

function TQwen3MoeModel.LoadFfn(aLayer: PtrInt; const aPrefix: RawUtf8): PtrInt;
var
  l: ^TDecoderLayer;
  h, fi, n, e: PtrInt;
  b: RawUtf8;
begin
  if fConfig.Spec[aLayer].Ffn = dfDense then
  begin
    result := inherited LoadFfn(aLayer, aPrefix);
    exit;
  end;
  l := @fLayer[aLayer];
  h := fConfig.Hidden;
  fi := fConfig.ExpertIntermediate;
  n := fConfig.Experts;
  l^.Router := Matrix(aPrefix + 'mlp.gate.weight', n, h, idtF32);
  if fGguf <> nil then
  begin
    b := FormatUtf8('blk.%.', [aLayer]);
    l^.ExpertGate := GgufExperts(b + 'ffn_gate_exps.weight', n, fi, h);
    l^.ExpertUp := GgufExperts(b + 'ffn_up_exps.weight', n, fi, h);
    l^.ExpertDown := GgufExperts(b + 'ffn_down_exps.weight', n, h, fi);
    result := 4;
    exit;
  end;
  SetLength(l^.ExpertGate, n);
  SetLength(l^.ExpertUp, n);
  SetLength(l^.ExpertDown, n);
  for e := 0 to n - 1 do
  begin
    b := FormatUtf8('%mlp.experts.%.', [aPrefix, e]);
    l^.ExpertGate[e] := Matrix(b + 'gate_proj.weight', fi, h, fWeights);
    l^.ExpertUp[e] := Matrix(b + 'up_proj.weight', fi, h, fWeights);
    l^.ExpertDown[e] := Matrix(b + 'down_proj.weight', h, fi, fWeights);
  end;
  result := 1 + 3 * n;
end;


initialization
  RegisterDecoderModel(TQwen3Model);
  RegisterDecoderModel(TQwen3MoeModel);
  RegisterDecoderModel(TQwen2Model);
  RegisterDecoderModel(TSmolLM3Model);

end.
