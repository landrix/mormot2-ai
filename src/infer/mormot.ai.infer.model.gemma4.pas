/// native inference engine - Gemma 4 decoder (causal language model)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room from the published architecture (looked up in Hugging Face's
//   modeling_gemma4.py, configuration_gemma4.py and modeling_rope_utils.py,
//   transformers 5.19, Apache-2.0, no code taken): token embedding times
//   sqrt(hidden), then per layer self-attention between two norms (RMSNorm,
//   Q/K/V projections, an RMSNorm over every Q and K head and one without
//   weight over every V head, rotary positions, grouped-query causal attention
//   with a score factor of 1, output projection, RMSNorm, residual) and a
//   feed-forward block between two norms (GeGLU with the tanh GELU, RMSNorm,
//   residual), the whole state times the layer's scalar; a final RMSNorm, the
//   head tied to the embedding, the logits soft-capped (tanh(x / c) * c)
// - two kinds of layers, as layer_types says (by default five sliding to one
//   full): sliding_attention with a window, head_dim and num_key_value_heads,
//   the rotary set of rope_parameters.sliding_attention; full_attention over
//   every token with global_head_dim and num_global_key_value_heads (with
//   attention_k_eq_v: V is the K projection, no v_proj), the rotary set of
//   rope_parameters.full_attention - "proportional": the frequencies over the
//   whole head, only the first partial_rotary_factor of them rotate
// - enable_moe_block (26B-A4B): beside the dense block a mixture of experts on
//   the same input - a router (an RMSNorm without weight, times router.scale and
//   hidden^-0.5, softmax over every expert, the top_k_experts renormed, each
//   times its per_expert_scale) and experts with gate and up in one tensor
//   gate_up_proj [experts, 2 * width, hidden] (the gate rows first) and
//   down_proj [experts, hidden, width]; dense and expert outputs normed on
//   their own, summed, normed again
// - the checkpoints of Gemma4ForConditionalGeneration (model_type gemma4): the
//   settings in text_config, the tensors under model.language_model. (the
//   vision and audio towers are left alone); Gemma4ForCausalLM (gemma4_text):
//   the settings at the top, the tensors under model.
// - the family of mormot.ai.infer.decoder: this unit reads config.json and
//   maps the tensors; the forward pass is TDecoderModel's
// - not yet (backlog D6): per-layer embeddings and shared KV layers (E2B,
//   E4B), GGUF, the tokenizer and chat template, a device (Vulkan, CUDA)
unit mormot.ai.infer.model.gemma4;

{
  *****************************************************************************

    - TGemma4Model: the gemma4 family of TDecoderModel (Gemma 4 checkpoints)
    - TGemma4TextModel: the same over a gemma4_text checkpoint

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
  mormot.ai.infer.backend,
  mormot.ai.infer.decoder;

type
  /// a Gemma 4 causal language model (gemma-4-31B, gemma-4-26B-A4B) over a
  // safetensors folder (one file or several with an index) with its weights in
  // BF16 or F32
  // - config.json of Gemma4ForConditionalGeneration (model_type gemma4, the
  // text settings in text_config): head_dim, global_head_dim,
  // num_key_value_heads, num_global_key_value_heads, attention_k_eq_v,
  // sliding_window, layer_types, rope_parameters (per layer type),
  // final_logit_softcapping, enable_moe_block with num_experts,
  // top_k_experts and moe_intermediate_size
  // - refused: per-layer embeddings (hidden_size_per_layer_input > 0) and
  // shared KV layers (num_kv_shared_layers > 0) - E2B and E4B -, an untied
  // head, attention_bias, bidirectional attention over all tokens, another
  // activation than gelu_pytorch_tanh, another rope_type
  // - the router stays F32 when the matrices are quantized at load
  TGemma4Model = class(TDecoderModel)
  protected
    procedure LoadConfig(const aConfigFile: TFileName); override;
    procedure LoadGgufConfig; override;
    procedure LoadWeights; override;
    function GgufName(const aName: RawUtf8): RawUtf8; override;
    /// the settings of the text model (aDoc: its object)
    procedure LoadTextConfig(const aDoc: TDocVariantData; const aSource: TFileName);
    /// the rotary sets of rope_parameters
    procedure LoadRope(const aDoc: TDocVariantData);
    /// the experts of layer aLayer: one gate, up and down matrix per expert,
    // sliced from the 3-D tensors
    procedure LoadExperts(aLayer: PtrInt; const aPrefix: RawUtf8);
    /// a slice of aRows rows of aCols values starting at row aFirst of the
    // tensor aTensor (aWide: its F32 values, nil unless the weights are F32)
    function Slice(aTensor: PInferTensor; aWide: PSingleArray;
      aFirst, aRows, aCols: Int64; const aName: RawUtf8): TInferWeight;
    /// the tensor prefix of the layout in place of "model."
    class function Prefix: RawUtf8; virtual;
  public
    /// 'gemma4'
    class function ModelType: RawUtf8; override;
    /// 'gemma4', as llama.cpp names it (GGUF: backlog D6)
    class function Architecture: RawUtf8; override;
  end;

  /// a Gemma 4 text model saved as Gemma4ForCausalLM (model_type gemma4_text):
  // the settings at the top of config.json, the tensors under model.
  TGemma4TextModel = class(TGemma4Model)
  protected
    class function Prefix: RawUtf8; override;
  public
    /// 'gemma4_text'
    class function ModelType: RawUtf8; override;
  end;


implementation


{ TGemma4Model }

class function TGemma4Model.ModelType: RawUtf8;
begin
  result := 'gemma4';
end;

class function TGemma4Model.Architecture: RawUtf8;
begin
  result := 'gemma4';
end;

class function TGemma4Model.Prefix: RawUtf8;
begin
  result := 'model.language_model.';
end;

const
  REQUIRED_KEYS: array[0..6] of RawUtf8 = (
    'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
    'num_attention_heads', 'num_key_value_heads', 'head_dim');

  /// the keys of a rope dictionary of one layer type we read
  ROPE_KEYS: array[0..3] of RawUtf8 = (
    'rope_type', 'rope_theta', 'partial_rotary_factor', 'factor');

/// a float of aDoc, aDefault when the key is missing or null
function FloatOr(const aDoc: TDocVariantData; const aKey: RawUtf8;
  aDefault: double): double;
var
  i: PtrInt;
begin
  result := aDefault;
  i := aDoc.GetValueIndex(aKey);
  if (i < 0) or VarIsEmptyOrNull(aDoc.Values[i]) then
    exit;
  if not VariantToDouble(aDoc.Values[i], result) then
    EInferError.RaiseUtf8('TGemma4Model: % is no number', [aKey]);
end;

/// an integer of aDoc, aDefault when the key is missing or null
function IntOr(const aDoc: TDocVariantData; const aKey: RawUtf8;
  aDefault: Int64): Int64;
var
  i: PtrInt;
begin
  result := aDefault;
  i := aDoc.GetValueIndex(aKey);
  if (i < 0) or VarIsEmptyOrNull(aDoc.Values[i]) then
    exit;
  if not VariantToInt64(aDoc.Values[i], result) then
    EInferError.RaiseUtf8('TGemma4Model: % is no integer', [aKey]);
end;

/// a boolean of aDoc, aDefault when the key is missing or null
function BoolOr(const aDoc: TDocVariantData; const aKey: RawUtf8;
  aDefault: boolean): boolean;
var
  i: PtrInt;
begin
  result := aDefault;
  i := aDoc.GetValueIndex(aKey);
  if (i >= 0) and not VarIsEmptyOrNull(aDoc.Values[i]) then
    result := aDoc.B[aKey];
end;

procedure TGemma4Model.LoadRope(const aDoc: TDocVariantData);
var
  rope, one: PDocVariantData;
  i, j, t: PtrInt;
  k, kind: RawUtf8;
  known: boolean;
begin
  // Gemma4TextConfig: sliding default theta 10000; full proportional, a
  // quarter of the head, theta 1e6
  fConfig.RopeTheta := 10000;
  fConfig.Rope2Theta := 1000000;
  fConfig.Rope2Partial := 0.25;
  if not aDoc.GetAsObject('rope_parameters', rope) then
  begin
    if aDoc.GetValueIndex('rope_parameters') >= 0 then
      if not VarIsEmptyOrNull(aDoc.Value['rope_parameters']) then
        EInferError.RaiseUtf8('%: rope_parameters is no object', [self]);
    exit;
  end;
  for t := 0 to rope^.Count - 1 do
  begin
    kind := rope^.Names[t];
    if (kind <> 'sliding_attention') and (kind <> 'full_attention') then
      EInferError.RaiseUtf8('%: rope_parameters for "%" not supported', [self, kind]);
    // null: transformers builds no rotary set for that kind and fails in
    // the forward pass - refused here (GetAsObject)
    if not rope^.GetAsObject(kind, one) then
      EInferError.RaiseUtf8('%: rope_parameters.% is no object', [self, kind]);
    for i := 0 to one^.Count - 1 do
    begin
      k := one^.Names[i];
      known := false;
      for j := 0 to high(ROPE_KEYS) do
        if k = ROPE_KEYS[j] then
          known := true;
      if not known then
        EInferError.RaiseUtf8('%: rope setting %.% not supported', [self, kind, k]);
    end;
    if FloatOr(one^, 'factor', 1) <> 1 then
      EInferError.RaiseUtf8('%: a rope factor of % not supported',
        [self, one^.U['factor']]);
    k := one^.U['rope_type'];
    if kind = 'sliding_attention' then
    begin
      // the sliding layers rotate their whole head
      if (k <> 'default') or (FloatOr(one^, 'partial_rotary_factor', 1) <> 1) then
        EInferError.RaiseUtf8('%: sliding rope_type "%" not supported', [self, k]);
      fConfig.RopeTheta := FloatOr(one^, 'rope_theta', 10000);
    end
    else
    begin
      // default = proportional over the whole head
      fConfig.Rope2Theta := FloatOr(one^, 'rope_theta', 1000000);
      if k = 'default' then
        fConfig.Rope2Partial := 1
      else if k = 'proportional' then
        fConfig.Rope2Partial := FloatOr(one^, 'partial_rotary_factor', 1)
      else
        EInferError.RaiseUtf8('%: full rope_type "%" not supported', [self, k]);
    end;
  end;
end;

procedure TGemma4Model.LoadTextConfig(const aDoc: TDocVariantData;
  const aSource: TFileName);
var
  d: PDocVariantData;
  types: PDocVariantData;
  i, w, ghd, gkv, p, layer: PtrInt;
  kv, perGiven: boolean;
  t, k: RawUtf8;
  per, one: PDocVariantData;
  kind: array of RawUtf8;
begin
  d := @aDoc;
  for i := 0 to high(REQUIRED_KEYS) do
    if d^.GetValueIndex(REQUIRED_KEYS[i]) < 0 then
      EInferError.RaiseUtf8('%: % has no "%"', [self, aSource, REQUIRED_KEYS[i]]);
  // what this family does not compute (yet)
  if not fBackend.Supports(ioGeluTanhGlu, idtF32) then
    EInferError.RaiseUtf8('%: the backend has no GeluTanhGlu', [self]);
  t := d^.U['hidden_activation'];
  if (t <> '') and (t <> 'gelu_pytorch_tanh') then
    EInferError.RaiseUtf8('%: hidden_activation "%" not supported', [self, t]);
  if IntOr(d^, 'hidden_size_per_layer_input', 256) <> 0 then
    EInferError.RaiseUtf8('%: per-layer embeddings not supported', [self]);
  if IntOr(d^, 'num_kv_shared_layers', 0) <> 0 then
    EInferError.RaiseUtf8('%: shared KV layers not supported', [self]);
  if BoolOr(d^, 'attention_bias', false) then
    EInferError.RaiseUtf8('%: attention_bias not supported', [self]);
  if d^.U['use_bidirectional_attention'] = 'all' then
    EInferError.RaiseUtf8('%: bidirectional attention not supported', [self]);
  if not BoolOr(d^, 'tie_word_embeddings', true) then
    EInferError.RaiseUtf8('%: an untied head not supported', [self]);
  fConfig.VocabSize := d^.I['vocab_size'];
  fConfig.Hidden := d^.I['hidden_size'];
  fConfig.Intermediate := d^.I['intermediate_size'];
  fConfig.Layers := d^.I['num_hidden_layers'];
  fConfig.Heads := d^.I['num_attention_heads'];
  fConfig.KvHeads := d^.I['num_key_value_heads'];
  fConfig.HeadDim := d^.I['head_dim'];
  fConfig.MaxPositions := IntOr(d^, 'max_position_embeddings', 131072);
  fConfig.RmsEps := FloatOr(d^, 'rms_norm_eps', 1e-6);
  fConfig.TiedEmbeddings := true;
  fConfig.Activation := daGeluTanh;
  fConfig.AttentionScale := 1;
  fConfig.SandwichNorm := true;
  fConfig.ValueNorm := true;
  fConfig.LogitSoftcap := FloatOr(d^, 'final_logit_softcapping', 0);
  // sqrt(hidden) as a float32 tensor (Gemma4TextScaledWordEmbedding)
  if fConfig.Hidden > 0 then
    fConfig.EmbedScale := Sqrt(fConfig.Hidden);
  LoadRope(d^);
  if BoolOr(d^, 'enable_moe_block', false) then
  begin
    fConfig.Experts := IntOr(d^, 'num_experts', 0);
    fConfig.ExpertsUsed := IntOr(d^, 'top_k_experts', 0);
    fConfig.ExpertIntermediate := IntOr(d^, 'moe_intermediate_size', 0);
    // the top-k probabilities divided by their sum
    fConfig.NormTopK := true;
    if (fConfig.Experts <= 0) or (fConfig.ExpertsUsed <= 0) or
       (fConfig.ExpertIntermediate <= 0) then
      EInferError.RaiseUtf8('%: enable_moe_block without its experts', [self]);
  end;
  // the full layers: wider heads, their own KV heads when K is V
  ghd := IntOr(d^, 'global_head_dim', 512);
  kv := BoolOr(d^, 'attention_k_eq_v', false);
  gkv := fConfig.KvHeads;
  if kv and (d^.GetValueIndex('num_global_key_value_heads') >= 0) and
     not VarIsEmptyOrNull(d^.Value['num_global_key_value_heads']) then
    gkv := d^.I['num_global_key_value_heads'];
  w := IntOr(d^, 'sliding_window', 512);
  // per_layer_config (what save_pretrained writes): the overrides of
  // head_dim and num_key_value_heads by layer index or layer type, in place
  // of global_head_dim and num_global_key_value_heads (transformers then
  // derives nothing from those)
  // - the key alone turns the global keys off, null included
  // (Gemma4TextConfig: "if 'per_layer_config' not in kwargs")
  per := nil;
  perGiven := d^.GetValueIndex('per_layer_config') >= 0;
  if perGiven and not VarIsEmptyOrNull(d^.Value['per_layer_config']) and
     not d^.GetAsObject('per_layer_config', per) then
    EInferError.RaiseUtf8('%: per_layer_config is no object', [self]);
  if (fConfig.Layers > 0) and (fConfig.Layers <= 1024) then
  begin
    SetLength(fConfig.Spec, fConfig.Layers);
    SetLength(kind, fConfig.Layers);
    if not d^.GetAsArray('layer_types', types) then
      types := nil
    else if types^.Count <> fConfig.Layers then
      EInferError.RaiseUtf8('%: % layer types for % layers',
        [self, types^.Count, fConfig.Layers]);
    for i := 0 to fConfig.Layers - 1 do
    begin
      if fConfig.Experts > 0 then
        fConfig.Spec[i].Ffn := dfDenseMoe;
      // five sliding layers, one full one by default; the last one full
      if types = nil then
      begin
        if (i + 1) mod 6 <> 0 then
          t := 'sliding_attention'
        else
          t := 'full_attention';
      end
      else
        t := VariantToUtf8(types^.Values[i]);
      if i = fConfig.Layers - 1 then
        t := 'full_attention'; // as Gemma4TextConfig forces it
      kind[i] := t;
      if t = 'sliding_attention' then
      begin
        if (w <= 0) or (w > 1 shl 24) then
          EInferError.RaiseUtf8('%: a sliding_window of %', [self, w]);
        fConfig.Spec[i].Window := w;
      end
      else if t = 'full_attention' then
      begin
        if not perGiven then
        begin
          fConfig.Spec[i].HeadDim := ghd;
          fConfig.Spec[i].KvHeads := gkv;
        end;
        fConfig.Spec[i].KIsV := kv;
        fConfig.Spec[i].Rope2 := true;
      end
      else
        EInferError.RaiseUtf8('%: layer type "%" not supported', [self, t]);
    end;
    if per <> nil then
      for p := 0 to per^.Count - 1 do
      begin
        k := per^.Names[p];
        if not per^.GetAsObject(k, one) then
          EInferError.RaiseUtf8('%: per_layer_config.% is no object', [self, k]);
        for i := 0 to one^.Count - 1 do
          if (one^.Names[i] <> 'head_dim') and
             (one^.Names[i] <> 'num_key_value_heads') then
            EInferError.RaiseUtf8('%: per_layer_config.%.% not supported',
              [self, k, one^.Names[i]]);
        layer := Utf8ToInteger(k, -1);
        for i := 0 to fConfig.Layers - 1 do
          if (i = layer) or (kind[i] = k) then
          begin
            fConfig.Spec[i].HeadDim := IntOr(one^, 'head_dim', fConfig.HeadDim);
            fConfig.Spec[i].KvHeads := IntOr(one^, 'num_key_value_heads',
              fConfig.KvHeads);
          end;
        if ((layer < 0) or (layer >= fConfig.Layers) or (k <> Int32ToUtf8(layer))) and
           (k <> 'sliding_attention') and (k <> 'full_attention') then
          EInferError.RaiseUtf8('%: per_layer_config.% names no layer', [self, k]);
      end;
  end;
  CheckConfig(aSource);
end;

procedure TGemma4Model.LoadConfig(const aConfigFile: TFileName);
var
  doc: TDocVariantData;
  d, text: PDocVariantData;
begin
  d := ReadConfig(aConfigFile, doc);
  if fStackPrefix = '' then
  begin
    // the tensors of the text model under their prefix (a stack of the file)
    fStackPrefix := Prefix;
    fStackEmbed := Prefix + 'embed_tokens.weight';
    if ModelType = 'gemma4' then
    begin
      if not BoolOr(d^, 'tie_word_embeddings', true) then
        EInferError.RaiseUtf8('%: an untied head not supported', [self]);
      if not d^.GetAsObject('text_config', text) then
        EInferError.RaiseUtf8('%: % has no text_config', [self, aConfigFile]);
      d := text;
    end;
  end;
  LoadTextConfig(d^, aConfigFile);
end;

procedure TGemma4Model.LoadGgufConfig;
begin
  EInferError.RaiseUtf8('%: GGUF not supported yet', [self]);
end;

function TGemma4Model.GgufName(const aName: RawUtf8): RawUtf8;
begin
  result := '';
end;

function TGemma4Model.Slice(aTensor: PInferTensor; aWide: PSingleArray;
  aFirst, aRows, aCols: Int64; const aName: RawUtf8): TInferWeight;
begin
  if fWeights = idtF32 then
    result := InferWeight(@aWide^[aFirst * aCols], idtF32)
  else
    // the source rows: BF16 or F32, one value per element
    result := QuantizedRows(
      @PByteArray(aTensor^.Data)^[aFirst * aCols * INFER_DTYPE_SIZE[aTensor^.DType]],
      aTensor^.DType, aRows, aCols, fWeights, aName);
end;

procedure TGemma4Model.LoadExperts(aLayer: PtrInt; const aPrefix: RawUtf8);
var
  l: ^TDecoderLayer;
  h, fi, n, e: PtrInt;
  gu, down: PInferTensor;
  wgu, wdown: PSingleArray;
  b: RawUtf8;
begin
  l := @fLayer[aLayer];
  h := fConfig.Hidden;
  fi := fConfig.ExpertIntermediate;
  n := fConfig.Experts;
  b := aPrefix + 'experts.';
  gu := Checked(b + 'gate_up_proj', [n, 2 * fi, h]);
  down := Checked(b + 'down_proj', [n, h, fi]);
  // F32: the tensors once in F32, every expert a view into them
  wgu := nil;
  wdown := nil;
  if fWeights = idtF32 then
  begin
    wgu := Weight(b + 'gate_up_proj', [n, 2 * fi, h]);
    wdown := Weight(b + 'down_proj', [n, h, fi]);
  end;
  SetLength(l^.ExpertGate, n);
  SetLength(l^.ExpertUp, n);
  SetLength(l^.ExpertDown, n);
  for e := 0 to n - 1 do
  begin
    // the gate rows of an expert first, then its up rows ([width, hidden] each)
    l^.ExpertGate[e] := Slice(gu, wgu, e * 2 * fi, fi, h, b + 'gate_up_proj');
    l^.ExpertUp[e] := Slice(gu, wgu, e * 2 * fi + fi, fi, h, b + 'gate_up_proj');
    l^.ExpertDown[e] := Slice(down, wdown, e * h, h, fi, b + 'down_proj');
  end;
end;

procedure TGemma4Model.LoadWeights;
var
  i, j, used: PtrInt;
  h, hd, qw, kvw: Int64;
  p: RawUtf8;
  l: ^TDecoderLayer;
  sp: ^TDecoderLayerSpec;
  headType: TInferDType;
  scale: PSingleArray;
  folded: TSingleDynArray;
  root: single;
begin
  // with Q4_0 the head stays Q8_0, as for Qwen3
  headType := fWeights;
  if headType = idtQ4_0 then
    headType := idtQ8_0;
  h := fConfig.Hidden;
  fEmbedSrc := Checked('model.embed_tokens.weight', [fConfig.VocabSize, h]);
  if fWeights = idtF32 then
  begin
    fEmbed := InferWeight(Weight('model.embed_tokens.weight', [fConfig.VocabSize, h]),
      idtF32);
    fHead := fEmbed; // tied
  end
  else
    fHead := Matrix('model.embed_tokens.weight', fConfig.VocabSize, h, headType);
  fNorm := Weight('model.norm.weight', [h]);
  used := 2;
  // hidden^-0.5 as the reference multiplies it: a float32 scalar
  root := 1 / Sqrt(1.0 * h); // Delphi 7: no Sqrt of an Int64
  SetLength(fLayer, fConfig.Layers);
  for i := 0 to fConfig.Layers - 1 do
  begin
    p := FormatUtf8('model.layers.%.', [i]);
    l := @fLayer[i];
    sp := @fConfig.Spec[i];
    hd := sp^.HeadDim;
    qw := fConfig.Heads * hd;
    kvw := sp^.KvHeads * hd;
    l^.Q := Matrix(p + 'self_attn.q_proj.weight', qw, h, fWeights);
    l^.K := Matrix(p + 'self_attn.k_proj.weight', kvw, h, fWeights);
    if not sp^.KIsV then
    begin
      l^.V := Matrix(p + 'self_attn.v_proj.weight', kvw, h, fWeights);
      inc(used);
    end;
    l^.O := Matrix(p + 'self_attn.o_proj.weight', h, qw, fWeights);
    l^.QNorm := Weight(p + 'self_attn.q_norm.weight', [hd]);
    l^.KNorm := Weight(p + 'self_attn.k_norm.weight', [hd]);
    l^.InputNorm := Weight(p + 'input_layernorm.weight', [h]);
    l^.AttnNorm := Weight(p + 'post_attention_layernorm.weight', [h]);
    l^.PostNorm := Weight(p + 'pre_feedforward_layernorm.weight', [h]);
    l^.FfnNorm := Weight(p + 'post_feedforward_layernorm.weight', [h]);
    l^.Scalar := Weight(p + 'layer_scalar', [1]);
    l^.Gate := Matrix(p + 'mlp.gate_proj.weight', fConfig.Intermediate, h, fWeights);
    l^.Up := Matrix(p + 'mlp.up_proj.weight', fConfig.Intermediate, h, fWeights);
    l^.Down := Matrix(p + 'mlp.down_proj.weight', h, fConfig.Intermediate, fWeights);
    inc(used, 13);
    if sp^.Ffn = dfDenseMoe then
    begin
      l^.DenseOutNorm := Weight(p + 'post_feedforward_layernorm_1.weight', [h]);
      l^.MoeInNorm := Weight(p + 'pre_feedforward_layernorm_2.weight', [h]);
      l^.MoeOutNorm := Weight(p + 'post_feedforward_layernorm_2.weight', [h]);
      l^.Router := Matrix(p + 'router.proj.weight', fConfig.Experts, h, idtF32);
      // the router input: an RMSNorm without weight, times scale and root -
      // folded into one weight of the norm
      scale := Weight(p + 'router.scale', [h]);
      SetLength(folded, h);
      for j := 0 to h - 1 do
        folded[j] := scale^[j] * root;
      l^.RouterScale := Own(folded);
      l^.ExpertScale := Weight(p + 'router.per_expert_scale', [fConfig.Experts]);
      LoadExperts(i, p);
      inc(used, 8);
    end;
  end;
  // an unknown tensor of the text model means one we do not understand
  if used <> TensorCountOf(fStackPrefix) then
    EInferError.RaiseUtf8('%: % tensors under %, % understood',
      [self, TensorCountOf(fStackPrefix), fStackPrefix, used]);
end;


{ TGemma4TextModel }

class function TGemma4TextModel.ModelType: RawUtf8;
begin
  result := 'gemma4_text';
end;

class function TGemma4TextModel.Prefix: RawUtf8;
begin
  result := 'model.';
end;


initialization
  RegisterDecoderModel(TGemma4Model);
  RegisterDecoderModel(TGemma4TextModel);

end.
