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
//   the settings at the top, the tensors under model.; that of
//   gemma-4-12B (Gemma4UnifiedForConditionalGeneration, model_type
//   gemma4_unified, modeling_gemma4_unified.py): the layout of gemma4, its text
//   model a dense Gemma 4
// - or a llama.cpp GGUF file of the gemma4 architecture (CreateFromGguf), as
//   convert_hf_to_gguf.py writes it (looked up in conversion/gemma.py -
//   Gemma4Model -, src/llama-arch.cpp and src/models/gemma4.cpp of the
//   llama.cpp commit 2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c, no code taken):
//   the norms as they are (no 1 + weight), the router and the scales
//   (ffn_gate_inp.scale, ffn_down_exps.scale) as they are, gate and up of the
//   experts in one tensor (the gate rows first), K=V where a full layer has no
//   attn_v, the proportional RoPE as the factors of rope_freqs (1 for a
//   rotating pair, 1e30 for the others)
// - the family of mormot.ai.infer.decoder: this unit reads config.json and
//   maps the tensors; the forward pass is TDecoderModel's
// - not yet (backlog D6): per-layer embeddings and shared KV layers (E2B,
//   E4B), a device (Vulkan, CUDA)
unit mormot.ai.infer.model.gemma4;

{
  *****************************************************************************

    - TGemma4Model: the gemma4 family of TDecoderModel (Gemma 4 checkpoints)
    - TGemma4TextModel: the same over a gemma4_text checkpoint
    - TGemma4UnifiedModel: the same over a gemma4_unified checkpoint (12B)

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
  mormot.ai.infer.loader.gguf,
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
  // head, attention_bias, bidirectional attention over all tokens, the
  // double-wide MLP (use_double_wide_mlp), another activation than
  // gelu_pytorch_tanh, another rope_type
  // - the router stays F32 when the matrices are quantized at load
  // - CreateFromGguf: a gemma4 file of llama.cpp's converter (gemma-4-26B-A4B,
  // 31B and 12B, whose GGUF all name the architecture gemma4), the types the
  // file holds (block types in place, BF16 / F16 widened); refused as from a
  // folder, and a feed-forward width per layer, a rope scaling, an output.weight
  TGemma4Model = class(TDecoderModel)
  protected
    procedure LoadConfig(const aConfigFile: TFileName); override;
    procedure LoadGgufConfig; override;
    procedure LoadWeights; override;
    function GgufName(const aName: RawUtf8): RawUtf8; override;
    /// a positive integer of the GGUF metadata
    function GgufCount(const aKey: RawUtf8): PtrInt;
    /// a positive integer of layer aLayer: an array per layer, or one value
    function GgufPerLayer(const aKey: RawUtf8; aLayer: PtrInt): PtrInt;
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
    /// true when config.json keeps the text settings in text_config
    class function InTextConfig: boolean; virtual;
  public
    /// 'gemma4'
    class function ModelType: RawUtf8; override;
    /// 'gemma4', as llama.cpp names it
    class function Architecture: RawUtf8; override;
  end;

  /// a Gemma 4 text model saved as Gemma4ForCausalLM (model_type gemma4_text):
  // the settings at the top of config.json, the tensors under model.
  TGemma4TextModel = class(TGemma4Model)
  protected
    class function Prefix: RawUtf8; override;
    class function InTextConfig: boolean; override;
  public
    /// 'gemma4_text'
    class function ModelType: RawUtf8; override;
  end;

  /// gemma-4-12B (Gemma4UnifiedForConditionalGeneration, model_type
  // gemma4_unified): the layout of TGemma4Model, a dense text model (its
  // config has no experts and no per-layer embeddings)
  TGemma4UnifiedModel = class(TGemma4Model)
  public
    /// 'gemma4_unified'
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

class function TGemma4Model.InTextConfig: boolean;
begin
  result := true;
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
  if BoolOr(d^, 'use_double_wide_mlp', false) then
    EInferError.RaiseUtf8('%: the double-wide MLP not supported', [self]);
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
    if InTextConfig then
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

function TGemma4Model.GgufCount(const aKey: RawUtf8): PtrInt;
var
  v: Int64;
begin
  v := fGguf.Int(aKey);
  if (v <= 0) or (v > 1 shl 30) then
    EInferError.RaiseUtf8('%: % = % out of range', [self, aKey, v]);
  result := v;
end;

function TGemma4Model.GgufPerLayer(const aKey: RawUtf8; aLayer: PtrInt): PtrInt;
var
  v: PGgufValue;
  x: Int64;
begin
  // a number for every layer, or one for all of them
  v := fGguf.Find(aKey);
  if (v = nil) or (v^.VType <> gtArray) then
  begin
    result := GgufCount(aKey);
    exit;
  end;
  if v^.Count <> fConfig.Layers then
    EInferError.RaiseUtf8('%: % has % values for % layers',
      [self, aKey, v^.Count, fConfig.Layers]);
  case v^.ElemType of
    gtInt32:
      x := PIntegerArray(v^.Data)^[aLayer];
    gtUInt32:
      x := PCardinalArray(v^.Data)^[aLayer];
  else
    begin
      EInferError.RaiseUtf8('%: % is an array of %, expected integers',
        [self, aKey, GGUF_TYPE_NAME[v^.ElemType]]);
      x := 0;
    end;
  end;
  if (x <= 0) or (x > 1 shl 30) then
    EInferError.RaiseUtf8('%: %[%] = % out of range', [self, aKey, aLayer, x]);
  result := x;
end;

procedure TGemma4Model.LoadGgufConfig;
var
  a: RawUtf8;
  emb, freqs: PInferTensor;
  pattern: PGgufValue;
  full, window, i, r: PtrInt;
  f: PSingleArray;
begin
  if fGguf.Str('general.architecture') <> Architecture then
    EInferError.RaiseUtf8('%: architecture "%" is not %',
      [self, fGguf.Str('general.architecture'), Architecture]);
  a := Architecture + '.';
  if not fBackend.Supports(ioGeluTanhGlu, idtF32) then
    EInferError.RaiseUtf8('%: the backend has no GeluTanhGlu', [self]);
  // what this family does not compute (yet): per-layer embeddings and shared
  // KV layers (E2B, E4B), the double-wide MLP (a width per layer), a scaled
  // RoPE, an untied head, prediction layers after the model
  if fGguf.Has(a + 'embedding_length_per_layer_input') and
     (fGguf.Int(a + 'embedding_length_per_layer_input') <> 0) then
    EInferError.RaiseUtf8('%: per-layer embeddings not supported', [self]);
  if fGguf.Has(a + 'attention.shared_kv_layers') and
     (fGguf.Int(a + 'attention.shared_kv_layers') <> 0) then
    EInferError.RaiseUtf8('%: shared KV layers not supported', [self]);
  if fGguf.Has(a + 'nextn_predict_layers') and
     (fGguf.Int(a + 'nextn_predict_layers') <> 0) then
    EInferError.RaiseUtf8('%: prediction layers not supported', [self]);
  if fGguf.Has(a + 'rope.scaling.type') and
     (fGguf.Str(a + 'rope.scaling.type') <> 'none') then
    EInferError.RaiseUtf8('%: rope scaling "%" not supported',
      [self, fGguf.Str(a + 'rope.scaling.type')]);
  if (fGguf.Find(a + 'feed_forward_length') <> nil) and
     (fGguf.Find(a + 'feed_forward_length')^.VType = gtArray) then
    EInferError.RaiseUtf8('%: a feed-forward width per layer (the double-wide ' +
      'MLP) not supported', [self]);
  if fGguf.FindTensor('output.weight') <> nil then
    EInferError.RaiseUtf8('%: an untied head not supported', [self]);
  emb := fGguf.Tensor('token_embd.weight');
  if length(emb^.Shape) <> 2 then
    EInferError.RaiseUtf8('%: token_embd.weight is shaped %',
      [self, InferShapeText(emb^.Shape)]);
  if (emb^.Shape[0] <= 0) or (emb^.Shape[0] > 1 shl 22) then
    EInferError.RaiseUtf8('%: a vocabulary of % tokens', [self, emb^.Shape[0]]);
  fConfig.VocabSize := emb^.Shape[0];
  fConfig.Hidden := GgufCount(a + 'embedding_length');
  fConfig.Layers := GgufCount(a + 'block_count');
  if fConfig.Layers > 1024 then
    EInferError.RaiseUtf8('%: % layers', [self, fConfig.Layers]);
  fConfig.Heads := GgufCount(a + 'attention.head_count');
  fConfig.Intermediate := GgufCount(a + 'feed_forward_length');
  fConfig.MaxPositions := GgufCount(a + 'context_length');
  fConfig.RmsEps := fGguf.Float(a + 'attention.layer_norm_rms_epsilon');
  // the heads: key_length_swa for the sliding layers, key_length for the
  // full ones; values of the same widths
  fConfig.HeadDim := GgufCount(a + 'attention.key_length_swa');
  full := GgufCount(a + 'attention.key_length');
  if (GgufCount(a + 'attention.value_length_swa') <> fConfig.HeadDim) or
     (GgufCount(a + 'attention.value_length') <> full) then
    EInferError.RaiseUtf8('%: key and value heads of different sizes', [self]);
  // the sliding layers rotate their whole head; the full ones too, by
  // rope_freqs only the first pairs (the proportional RoPE, see below)
  if (GgufCount(a + 'rope.dimension_count_swa') <> fConfig.HeadDim) or
     (GgufCount(a + 'rope.dimension_count') <> full) then
    EInferError.RaiseUtf8('%: rotating % and % values of heads of % and %',
      [self, fGguf.Int(a + 'rope.dimension_count_swa'),
       fGguf.Int(a + 'rope.dimension_count'), fConfig.HeadDim, full]);
  fConfig.RopeTheta := fGguf.Float(a + 'rope.freq_base_swa');
  fConfig.Rope2Theta := fGguf.Float(a + 'rope.freq_base');
  // rope_freqs: a factor per pair of the full head, 1 for the rotating pairs,
  // 1e30 for the others (the angle divided by it: none) - as the converter
  // writes the proportional RoPE of transformers (generate_extra_tensors)
  freqs := fGguf.Tensor('rope_freqs.weight');
  if (freqs^.DType <> idtF32) or not InferShapeIs(freqs^, [full shr 1]) then
    EInferError.RaiseUtf8('%: rope_freqs.weight is % %, expected F32 [%]',
      [self, INFER_DTYPE_NAME[freqs^.DType], InferShapeText(freqs^.Shape), full shr 1]);
  f := freqs^.Data;
  r := 0;
  while (r < full shr 1) and (f^[r] = 1) do
    inc(r);
  for i := r to (full shr 1) - 1 do
    if not (f^[i] >= 1e30) then
      EInferError.RaiseUtf8('%: rope_freqs.weight[%] = % is neither 1 before ' +
        'the others nor 1e30 after them', [self, i, f^[i]]);
  if r = 0 then
    EInferError.RaiseUtf8('%: rope_freqs.weight rotates no pair', [self]);
  // the share of the head whose Trunc(share * head) shr 1 (CheckConfig) is r
  // under any rounding: 2r + 1 values, not 2r (2r / 44 * 44 may give
  // 29.999..., one pair less); the whole head when every pair rotates
  if 2 * r = full then
    fConfig.Rope2Partial := 1
  else
    fConfig.Rope2Partial := (2 * r + 1) / full;
  if Trunc(fConfig.Rope2Partial * full) shr 1 <> r then
    EInferError.RaiseUtf8('%: % rotating pairs of a head of % not representable',
      [self, r, full]);
  fConfig.TiedEmbeddings := true;
  fConfig.Activation := daGeluTanh;
  fConfig.AttentionScale := 1;
  fConfig.SandwichNorm := true;
  fConfig.ValueNorm := true;
  if fGguf.Has(a + 'final_logit_softcapping') then
    fConfig.LogitSoftcap := fGguf.Float(a + 'final_logit_softcapping');
  fConfig.EmbedScale := Sqrt(fConfig.Hidden);
  if fGguf.Has(a + 'expert_count') and (fGguf.Int(a + 'expert_count') <> 0) then
  begin
    fConfig.Experts := GgufCount(a + 'expert_count');
    fConfig.ExpertsUsed := GgufCount(a + 'expert_used_count');
    fConfig.ExpertIntermediate := GgufCount(a + 'expert_feed_forward_length');
    fConfig.NormTopK := true;
  end;
  // the kind of every layer: sliding_window_pattern (true = sliding)
  pattern := fGguf.Find(a + 'attention.sliding_window_pattern');
  if (pattern = nil) or (pattern^.VType <> gtArray) or
     (pattern^.ElemType <> gtBool) or (pattern^.Count <> fConfig.Layers) then
    EInferError.RaiseUtf8('%: attention.sliding_window_pattern is no array of % ' +
      'booleans', [self, fConfig.Layers]);
  window := GgufCount(a + 'attention.sliding_window');
  SetLength(fConfig.Spec, fConfig.Layers);
  fConfig.KvHeads := 0;
  for i := 0 to fConfig.Layers - 1 do
  begin
    if fConfig.Experts > 0 then
      fConfig.Spec[i].Ffn := dfDenseMoe;
    if PByteArray(pattern^.Data)^[i] <> 0 then
    begin
      fConfig.Spec[i].Window := window;
      // the KV heads of the config: those of the first sliding layer
      if fConfig.KvHeads = 0 then
        fConfig.KvHeads := GgufPerLayer(a + 'attention.head_count_kv', i);
      fConfig.Spec[i].KvHeads := GgufPerLayer(a + 'attention.head_count_kv', i);
    end
    else
    begin
      fConfig.Spec[i].HeadDim := full;
      fConfig.Spec[i].KvHeads := GgufPerLayer(a + 'attention.head_count_kv', i);
      fConfig.Spec[i].Rope2 := true;
      // attention_k_eq_v: the converter writes no attn_v (V is the K
      // projection); a sliding layer always has its own
      fConfig.Spec[i].KIsV :=
        fGguf.FindTensor(FormatUtf8('blk.%.attn_v.weight', [i])) = nil;
    end;
  end;
  if fConfig.KvHeads = 0 then
    EInferError.RaiseUtf8('%: no sliding layer', [self]);
  CheckConfig(fGguf.FileName);
end;

const
  /// Hugging Face names and llama.cpp's for the tensors of one layer
  LAYER_NAMES: array[0..21, 0..1] of RawUtf8 = (
    ('input_layernorm.weight', 'attn_norm.weight'),
    ('post_attention_layernorm.weight', 'post_attention_norm.weight'),
    ('pre_feedforward_layernorm.weight', 'ffn_norm.weight'),
    ('post_feedforward_layernorm.weight', 'post_ffw_norm.weight'),
    ('post_feedforward_layernorm_1.weight', 'post_ffw_norm_1.weight'),
    ('pre_feedforward_layernorm_2.weight', 'pre_ffw_norm_2.weight'),
    ('post_feedforward_layernorm_2.weight', 'post_ffw_norm_2.weight'),
    ('layer_scalar', 'layer_output_scale.weight'),
    ('self_attn.q_proj.weight', 'attn_q.weight'),
    ('self_attn.k_proj.weight', 'attn_k.weight'),
    ('self_attn.v_proj.weight', 'attn_v.weight'),
    ('self_attn.o_proj.weight', 'attn_output.weight'),
    ('self_attn.q_norm.weight', 'attn_q_norm.weight'),
    ('self_attn.k_norm.weight', 'attn_k_norm.weight'),
    ('mlp.gate_proj.weight', 'ffn_gate.weight'),
    ('mlp.up_proj.weight', 'ffn_up.weight'),
    ('mlp.down_proj.weight', 'ffn_down.weight'),
    ('router.proj.weight', 'ffn_gate_inp.weight'),
    ('router.scale', 'ffn_gate_inp.scale'),
    ('router.per_expert_scale', 'ffn_down_exps.scale'),
    ('experts.gate_up_proj', 'ffn_gate_up_exps.weight'),
    ('experts.down_proj', 'ffn_down_exps.weight'));

function TGemma4Model.GgufName(const aName: RawUtf8): RawUtf8;
var
  p, i: PtrInt;
  layer, rest: RawUtf8;
begin
  result := '';
  if aName = 'model.embed_tokens.weight' then
    result := 'token_embd.weight'
  else if aName = 'model.norm.weight' then
    result := 'output_norm.weight'
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

function TGemma4Model.Slice(aTensor: PInferTensor; aWide: PSingleArray;
  aFirst, aRows, aCols: Int64; const aName: RawUtf8): TInferWeight;
var
  t: TInferDType;
begin
  t := aTensor^.DType;
  if aWide <> nil then
    result := InferWeight(@aWide^[aFirst * aCols], idtF32)
  else if t in INFER_BLOCK_DTYPES then
    // a GGUF file: whole rows of blocks in place (a row holds whole blocks)
    result := InferWeight(@PByteArray(aTensor^.Data)^[
      (aFirst * aCols div INFER_DTYPE_BLOCK[t]) * INFER_DTYPE_SIZE[t]], t)
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
  tgu, tdown: TInferDType;
  b: RawUtf8;
begin
  l := @fLayer[aLayer];
  h := fConfig.Hidden;
  fi := fConfig.ExpertIntermediate;
  n := fConfig.Experts;
  b := aPrefix + 'experts.';
  // a GGUF file: [experts, 2 * width, hidden] and [experts, hidden, width] as
  // in the folder (ffn_gate_up_exps, ffn_down_exps), blocks or floats
  gu := Checked(b + 'gate_up_proj', [n, 2 * fi, h], fGguf <> nil);
  down := Checked(b + 'down_proj', [n, h, fi], fGguf <> nil);
  // F32: the tensors once in F32, every expert a view into them; a GGUF file
  // keeps its blocks in place and widens its floats
  wgu := nil;
  wdown := nil;
  if fGguf <> nil then
  begin
    tgu := idtF32;
    tdown := idtF32;
    if gu^.DType in INFER_BLOCK_DTYPES then
    begin
      CheckBlockScales(gu, b + 'gate_up_proj');
      tgu := gu^.DType;
    end
    else
      wgu := Weight(b + 'gate_up_proj', [n, 2 * fi, h]);
    if down^.DType in INFER_BLOCK_DTYPES then
    begin
      CheckBlockScales(down, b + 'down_proj');
      tdown := down^.DType;
    end
    else
      wdown := Weight(b + 'down_proj', [n, h, fi]);
    // the types computed: the blocks as they are, the floats widened
    if not fBackend.Supports(ioLinear, tgu) or
       not fBackend.Supports(ioLinear, tdown) then
      EInferError.RaiseUtf8('%: the backend has no Linear for the experts of ' +
        'layer % (% and %)', [self, aLayer, INFER_DTYPE_NAME[tgu],
        INFER_DTYPE_NAME[tdown]]);
  end
  else if fWeights = idtF32 then
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
  fEmbedSrc := Checked('model.embed_tokens.weight', [fConfig.VocabSize, h],
    fGguf <> nil);
  // read per token: its scales checked here
  if fEmbedSrc^.DType in INFER_BLOCK_DTYPES then
    CheckBlockScales(fEmbedSrc, 'model.embed_tokens.weight');
  if (fGguf = nil) and (fWeights = idtF32) then
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
  // an unknown tensor of the text model means one we do not understand (in
  // a GGUF file: rope_freqs too, read with the settings)
  if fGguf <> nil then
  begin
    inc(used);
    if used <> length(fGguf.Tensors) then
      EInferError.RaiseUtf8('%: % tensors in %, % understood',
        [self, length(fGguf.Tensors), fGguf.FileName, used]);
  end
  else if used <> TensorCountOf(fStackPrefix) then
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

class function TGemma4TextModel.InTextConfig: boolean;
begin
  result := false;
end;


{ TGemma4UnifiedModel }

class function TGemma4UnifiedModel.ModelType: RawUtf8;
begin
  result := 'gemma4_unified';
end;


initialization
  RegisterDecoderModel(TGemma4Model);
  RegisterDecoderModel(TGemma4TextModel);
  RegisterDecoderModel(TGemma4UnifiedModel);

end.
