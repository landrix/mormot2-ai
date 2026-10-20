/// native inference engine - gpt-oss decoder (causal language model)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room from the published architecture (looked up in Hugging Face's
//   modeling_gpt_oss.py and configuration_gpt_oss.py, transformers 5.19,
//   Apache-2.0, no code taken): token embedding, then per layer pre-norm
//   self-attention (RMSNorm, Q/K/V projections with a bias, rotary positions
//   with YaRN, grouped-query causal attention with a learned sink logit per
//   head, output projection with a bias) and a pre-norm mixture of experts
//   (a router with a bias picks the top num_experts_per_tok experts by logit,
//   a softmax over their logits weighs them; each expert a clamped SwiGLU
//   with biases), both with a residual; a final RMSNorm and an untied head
// - the layers alternate a sliding window (sliding_window, 128 for the
//   published models) and full attention, as layer_types says (by default the
//   even layers slide)
// - the family of mormot.ai.infer.decoder: this unit reads config.json and
//   maps the tensors; the forward pass is TDecoderModel's
// - the experts of the transformers 5 layout: gate_up_proj [experts, hidden,
//   2 * width] with the gate and up columns interleaved (gate even, up odd),
//   down_proj [experts, width, hidden] - both transposed against nn.Linear:
//   reordered into one matrix per expert at load
// - the MXFP4 experts of the published checkpoints (quantization_config
//   mxfp4): gate_up_proj_blocks [experts, 2 * width, hidden / 32, 16] and
//   gate_up_proj_scales [experts, 2 * width, hidden / 32] - rows are outputs,
//   gate and up rows interleaved, byte j of a block holds value 2j in its low
//   and 2j + 1 in its high nibble, the scale an E8M0 exponent; down_proj the
//   same over [experts, hidden, width / 32]. Repacked at load into ggml's
//   block layout (TInferBlockMxfp4), never widened
// - GGUF (llama.cpp's architecture gpt-oss): gate and up already apart as
//   ffn_gate_exps / ffn_up_exps [experts, width, hidden], down as ffn_down_exps,
//   used in place (MXFP4 or another block type), their biases as *.bias, the
//   sinks as attn_sinks.weight; the settings from gpt-oss.* metadata - the
//   SwiGLU limit 7 and alpha 1.702 as llama.cpp fixes them, YaRN with the betas
//   of the file (32 and 1 without) and truncate false as the checkpoint says
//   (llama.cpp itself rounds the correction range, as truncate true)
// - not yet (backlog D5): a device (Vulkan, CUDA)
unit mormot.ai.infer.model.gptoss;

{
  *****************************************************************************

    - TGptOssModel: the gpt_oss family of TDecoderModel

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
  /// a gpt-oss causal language model (e.g. gpt-oss-20b) over a safetensors
  // folder (one file or several with an index) with its weights in BF16 or
  // F32, the experts also in MXFP4
  // - config.json (model_type gpt_oss): head_dim, num_local_experts,
  // num_experts_per_tok, intermediate_size (the width of an expert),
  // sliding_window and layer_types, rope_parameters (transformers 5) or
  // rope_scaling and rope_theta (transformers 4) of rope_type yarn,
  // swiglu_limit and swiglu_alpha (default 7 and 1.702)
  // - a quantization_config must be quant_method mxfp4: the experts are then
  // read from their *_blocks and *_scales and stay MXFP4 whatever aWeights
  // asks for (the other matrices follow aWeights); a block whose exponent is
  // outside INFER_MXFP4_EXACT_MIN..MAX must hold only zeros
  // - refused: attention_bias false, tied embeddings, another quantization,
  // another rope_type
  // - the router stays F32 when the matrices are quantized at load
  TGptOssModel = class(TDecoderModel)
  protected
    fMxfp4: boolean; // quantization_config mxfp4: the experts as blocks
    procedure LoadConfig(const aConfigFile: TFileName); override;
    procedure LoadGgufConfig; override;
    procedure LoadWeights; override;
    function GgufName(const aName: RawUtf8): RawUtf8; override;
    /// the YaRN settings of config.json
    procedure LoadRope(const aDoc: TDocVariantData);
    /// the experts of layer aLayer from the transformers 5 tensors
    procedure LoadExperts(aLayer: PtrInt; const aPrefix: RawUtf8);
    /// the experts of layer aLayer of a GGUF file, in place
    procedure LoadGgufExperts(aLayer: PtrInt; const aPrefix: RawUtf8);
    /// a positive count of the GGUF metadata
    function GgufCount(const aKey: RawUtf8): PtrInt;
    /// the gate and up biases of layer aLayer: the even and odd columns of
    // gate_up_proj_bias, and down_proj_bias
    procedure LoadExpertBiases(aLayer: PtrInt; const aPrefix: RawUtf8);
    /// the MXFP4 experts of layer aLayer: the gate, up and down matrices of
    // every expert repacked from the *_blocks and *_scales tensors
    procedure LoadMxfp4Experts(aLayer: PtrInt; const aPrefix: RawUtf8);
    /// a U8 tensor shaped aDims (MXFP4 blocks and scales)
    function ByteTensor(const aName: RawUtf8; const aDims: array of Int64): PByteArray;
  public
    /// 'gpt_oss'
    class function ModelType: RawUtf8; override;
    /// 'gpt-oss', as llama.cpp names it
    class function Architecture: RawUtf8; override;
  end;


implementation


{ TGptOssModel }

class function TGptOssModel.ModelType: RawUtf8;
begin
  result := 'gpt_oss';
end;

class function TGptOssModel.Architecture: RawUtf8;
begin
  result := 'gpt-oss';
end;

const
  REQUIRED_KEYS: array[0..9] of RawUtf8 = (
    'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
    'num_key_value_heads', 'head_dim', 'intermediate_size', 'num_local_experts',
    'num_experts_per_tok', 'max_position_embeddings');

  /// the keys of a YaRN rope dictionary we read - any other one is refused
  // (mscale, mscale_all_dim, partial_rotary_factor... would change the angles)
  YARN_KEYS: array[0..8] of RawUtf8 = (
    'rope_type', 'type', 'rope_theta', 'factor', 'beta_fast', 'beta_slow',
    'truncate', 'original_max_position_embeddings', 'attention_factor');

/// a float of aDoc, aDefault when the key is missing, null, or (aZeroDefault)
// zero - as Python's "get(key) or default"
function FloatOr(const aDoc: TDocVariantData; const aKey: RawUtf8;
  aDefault: double; aZeroDefault: boolean): double;
var
  i: PtrInt;
begin
  result := aDefault;
  i := aDoc.GetValueIndex(aKey);
  if (i < 0) or VarIsEmptyOrNull(aDoc.Values[i]) then
    exit;
  if not VariantToDouble(aDoc.Values[i], result) then
    EInferError.RaiseUtf8('TGptOssModel: % is no number', [aKey]);
  if aZeroDefault and (result = 0) then
    result := aDefault;
end;

procedure TGptOssModel.LoadRope(const aDoc: TDocVariantData);
var
  rope: PDocVariantData;
  i, j: PtrInt;
  k, t: RawUtf8;
  known: boolean;
begin
  // transformers 5 writes rope_parameters, 4 wrote rope_scaling with
  // rope_theta at the top level; with neither, GptOssConfig sets YaRN with a
  // factor of 32 over 4096 positions, betas 32 / 1, truncate false
  if not aDoc.GetAsObject('rope_parameters', rope) and
     not aDoc.GetAsObject('rope_scaling', rope) then
  begin
    fConfig.RopeTheta := FloatOr(aDoc, 'rope_theta', 150000, false);
    fConfig.RopeFactor := 32;
    fConfig.RopeOriginal := 4096;
    fConfig.RopeBetaFast := 32;
    fConfig.RopeBetaSlow := 1;
    fConfig.RopeTruncate := false;
    fConfig.RopeAttention := 0;
    exit;
  end;
  for i := 0 to rope^.Count - 1 do
  begin
    k := rope^.Names[i];
    known := false;
    for j := 0 to high(YARN_KEYS) do
      if k = YARN_KEYS[j] then
        known := true;
    if not known then
      EInferError.RaiseUtf8('%: rope setting % not supported', [self, k]);
  end;
  t := rope^.U['rope_type'];
  if t = '' then
    t := rope^.U['type'];
  if t <> 'yarn' then
    EInferError.RaiseUtf8('%: rope_type "%" not supported', [self, t]);
  fConfig.RopeTheta := FloatOr(rope^, 'rope_theta',
    FloatOr(aDoc, 'rope_theta', 150000, false), false);
  if rope^.GetValueIndex('factor') < 0 then
    EInferError.RaiseUtf8('%: YaRN without a factor', [self]);
  if rope^.GetValueIndex('original_max_position_embeddings') < 0 then
    EInferError.RaiseUtf8('%: YaRN without original_max_position_embeddings', [self]);
  fConfig.RopeOriginal := rope^.I['original_max_position_embeddings'];
  // a factor of null: max_position_embeddings / original, as transformers
  // derives it; 0 or less would divide by zero - refused
  fConfig.RopeFactor := FloatOr(rope^, 'factor', -1, false);
  if VarIsEmptyOrNull(rope^.Value['factor']) and (fConfig.RopeOriginal > 0) then
    fConfig.RopeFactor := fConfig.MaxPositions / fConfig.RopeOriginal;
  if not (fConfig.RopeFactor > 0) then
    EInferError.RaiseUtf8('%: a YaRN factor of %', [self, rope^.U['factor']]);
  fConfig.RopeBetaFast := FloatOr(rope^, 'beta_fast', 32, true);
  fConfig.RopeBetaSlow := FloatOr(rope^, 'beta_slow', 1, true);
  // transformers: rope_parameters.get("truncate", True)
  fConfig.RopeTruncate := true;
  if rope^.GetValueIndex('truncate') >= 0 then
    fConfig.RopeTruncate := rope^.B['truncate'];
  // attention_factor null is no attention_factor
  fConfig.RopeAttention := FloatOr(rope^, 'attention_factor', 0, false);
  if (rope^.GetValueIndex('attention_factor') >= 0) and
     not VarIsEmptyOrNull(rope^.Value['attention_factor']) and
     not (fConfig.RopeAttention > 0) then
    EInferError.RaiseUtf8('%: an attention_factor of %', [self, fConfig.RopeAttention]);
end;

procedure TGptOssModel.LoadConfig(const aConfigFile: TFileName);
var
  doc: TDocVariantData;
  d, types, q: PDocVariantData;
  i: PtrInt;
  w: Int64;
  t: RawUtf8;
begin
  d := ReadConfig(aConfigFile, doc);
  for i := 0 to high(REQUIRED_KEYS) do
    if d^.GetValueIndex(REQUIRED_KEYS[i]) < 0 then
      EInferError.RaiseUtf8('%: % has no "%"', [self, aConfigFile, REQUIRED_KEYS[i]]);
  // MXFP4 experts: transformers takes quant_method mxfp4 alone (no settings)
  fMxfp4 := d^.GetValueIndex('quantization_config') >= 0;
  if fMxfp4 then
    if not d^.GetAsObject('quantization_config', q) then
      EInferError.RaiseUtf8('%: quantization_config is no object', [self])
    else if q^.U['quant_method'] <> 'mxfp4' then
      EInferError.RaiseUtf8('%: quantization "%" not supported',
        [self, q^.U['quant_method']]);
  // GptOssConfig defaults: attention_bias true, untied head
  if (d^.GetValueIndex('attention_bias') >= 0) and not d^.B['attention_bias'] then
    EInferError.RaiseUtf8('%: attention_bias false not supported', [self]);
  if d^.B['tie_word_embeddings'] then
    EInferError.RaiseUtf8('%: tied embeddings not supported', [self]);
  fConfig.VocabSize := d^.I['vocab_size'];
  fConfig.Hidden := d^.I['hidden_size'];
  fConfig.Layers := d^.I['num_hidden_layers'];
  fConfig.Heads := d^.I['num_attention_heads'];
  // null as GptOssConfig derives it: every head its own KV head, hidden / heads
  if VarIsEmptyOrNull(d^.Value['num_key_value_heads']) then
    fConfig.KvHeads := fConfig.Heads
  else
    fConfig.KvHeads := d^.I['num_key_value_heads'];
  if VarIsEmptyOrNull(d^.Value['head_dim']) and (fConfig.Heads > 0) then
    fConfig.HeadDim := fConfig.Hidden div fConfig.Heads
  else
    fConfig.HeadDim := d^.I['head_dim'];
  fConfig.Intermediate := d^.I['intermediate_size'];
  fConfig.MaxPositions := d^.I['max_position_embeddings'];
  fConfig.RmsEps := FloatOr(d^, 'rms_norm_eps', 1e-5, false);
  fConfig.TiedEmbeddings := false;
  fConfig.Experts := d^.I['num_local_experts'];
  fConfig.ExpertsUsed := d^.I['num_experts_per_tok'];
  fConfig.ExpertIntermediate := fConfig.Intermediate;
  // a softmax over the top-k logits = the top-k probabilities renormed
  fConfig.NormTopK := true;
  fConfig.SwiGluLimit := FloatOr(d^, 'swiglu_limit', 7, false);
  fConfig.SwiGluAlpha := FloatOr(d^, 'swiglu_alpha', 1.702, false);
  if not (fConfig.SwiGluLimit > 0) then
    EInferError.RaiseUtf8('%: a swiglu_limit of %', [self, fConfig.SwiGluLimit]);
  LoadRope(d^);
  // every layer a mixture of experts; the window by layer_types, by default
  // on the even layers (GptOssConfig) - a sliding layer needs a window
  if (fConfig.Layers > 0) and (fConfig.Layers <= 1024) then
  begin
    SetLength(fConfig.Spec, fConfig.Layers);
    // GptOssConfig.sliding_window: 128 by default, null for none
    i := d^.GetValueIndex('sliding_window');
    if i < 0 then
      w := 128
    else if VarIsEmptyOrNull(d^.Values[i]) then
      w := 0
    else
    begin
      w := VariantToInt64Def(d^.Values[i], 0);
      if (w <= 0) or (w > 1 shl 24) then
        EInferError.RaiseUtf8('%: a sliding_window of %', [self, w]);
    end;
    if not d^.GetAsArray('layer_types', types) then
      types := nil
    else if types^.Count <> fConfig.Layers then
      EInferError.RaiseUtf8('%: % layer types for % layers',
        [self, types^.Count, fConfig.Layers]);
    for i := 0 to fConfig.Layers - 1 do
    begin
      fConfig.Spec[i].Ffn := dfMoe;
      if types = nil then
      begin
        if odd(i + 1) then
          t := 'sliding_attention'
        else
          t := 'full_attention';
      end
      else
        t := VariantToUtf8(types^.Values[i]);
      if t = 'sliding_attention' then
      begin
        if w <= 0 then
          EInferError.RaiseUtf8('%: layer % slides without a sliding window', [self, i]);
        fConfig.Spec[i].Window := w;
      end
      else if t <> 'full_attention' then
        EInferError.RaiseUtf8('%: layer type "%" not supported', [self, t]);
    end;
  end;
  CheckConfig(aConfigFile);
end;

function TGptOssModel.GgufCount(const aKey: RawUtf8): PtrInt;
var
  v: Int64;
begin
  v := fGguf.Int(aKey);
  if (v <= 0) or (v > 1 shl 30) then
    EInferError.RaiseUtf8('%: % = % out of range', [self, aKey, v]);
  result := v;
end;

procedure TGptOssModel.LoadGgufConfig;
var
  a: RawUtf8;
  emb: PInferTensor;
  i, w: PtrInt;
begin
  if fGguf.Str('general.architecture') <> Architecture then
    EInferError.RaiseUtf8('%: architecture "%" is not %',
      [self, fGguf.Str('general.architecture'), Architecture]);
  a := Architecture + '.';
  emb := fGguf.Tensor('token_embd.weight');
  if (length(emb^.Shape) <> 2) or (emb^.Shape[0] <= 0) or
     (emb^.Shape[0] > 1 shl 22) then
    EInferError.RaiseUtf8('%: token_embd.weight is shaped %',
      [self, InferShapeText(emb^.Shape)]);
  fConfig.VocabSize := emb^.Shape[0];
  fConfig.Hidden := GgufCount(a + 'embedding_length');
  fConfig.Layers := GgufCount(a + 'block_count');
  fConfig.Heads := GgufCount(a + 'attention.head_count');
  fConfig.KvHeads := GgufCount(a + 'attention.head_count_kv');
  fConfig.HeadDim := GgufCount(a + 'attention.key_length');
  if fGguf.Has(a + 'attention.value_length') and
     (GgufCount(a + 'attention.value_length') <> fConfig.HeadDim) then
    EInferError.RaiseUtf8('%: key and value heads of different sizes', [self]);
  if fGguf.Has(a + 'rope.dimension_count') and
     (GgufCount(a + 'rope.dimension_count') <> fConfig.HeadDim) then
    EInferError.RaiseUtf8('%: RoPE over % of % dimensions not supported',
      [self, GgufCount(a + 'rope.dimension_count'), fConfig.HeadDim]);
  fConfig.Intermediate := GgufCount(a + 'feed_forward_length');
  fConfig.ExpertIntermediate := GgufCount(a + 'expert_feed_forward_length');
  fConfig.Experts := GgufCount(a + 'expert_count');
  fConfig.ExpertsUsed := GgufCount(a + 'expert_used_count');
  fConfig.MaxPositions := GgufCount(a + 'context_length');
  fConfig.RmsEps := fGguf.Float(a + 'attention.layer_norm_rms_epsilon');
  fConfig.RopeTheta := fGguf.Float(a + 'rope.freq_base');
  fConfig.NormTopK := true;
  // llama.cpp: an untied head, the SwiGLU of gpt-oss with fixed settings
  if fGguf.FindTensor('output.weight') = nil then
    EInferError.RaiseUtf8('%: tied embeddings not supported', [self]);
  fConfig.TiedEmbeddings := false;
  fConfig.SwiGluLimit := 7;
  fConfig.SwiGluAlpha := 1.702;
  // YaRN, as the converter writes it: type, factor and the original context
  if fGguf.Str(a + 'rope.scaling.type') <> 'yarn' then
    EInferError.RaiseUtf8('%: rope scaling "%" not supported',
      [self, fGguf.Str(a + 'rope.scaling.type')]);
  // an attention factor of llama.cpp's own would change the scale: refused,
  // not guessed
  if fGguf.Has(a + 'rope.scaling.attn_factor') or
     fGguf.Has(a + 'rope.scaling.yarn_log_multiplier') then
    EInferError.RaiseUtf8('%: a YaRN attention factor not supported', [self]);
  fConfig.RopeFactor := fGguf.Float(a + 'rope.scaling.factor');
  if not (fConfig.RopeFactor > 0) then
    EInferError.RaiseUtf8('%: a YaRN factor of %', [self, fConfig.RopeFactor]);
  fConfig.RopeOriginal := GgufCount(a + 'rope.scaling.original_context_length');
  // the betas as the published file writes them (a converter newer than the
  // llama.cpp commit pinned in UPSTREAM_BASE adds the keys), else 32 and 1,
  // the defaults llama.cpp takes then
  fConfig.RopeBetaFast := 32;
  if fGguf.Has(a + 'rope.scaling.yarn_beta_fast') then
    fConfig.RopeBetaFast := fGguf.Float(a + 'rope.scaling.yarn_beta_fast');
  fConfig.RopeBetaSlow := 1;
  if fGguf.Has(a + 'rope.scaling.yarn_beta_slow') then
    fConfig.RopeBetaSlow := fGguf.Float(a + 'rope.scaling.yarn_beta_slow');
  fConfig.RopeTruncate := false;
  fConfig.RopeAttention := 0;
  // the even layers slide (llama.cpp: swa pattern 2), every layer has experts
  w := GgufCount(a + 'attention.sliding_window');
  if (w > 1 shl 24) or (fConfig.Layers > 1024) then
    EInferError.RaiseUtf8('%: a sliding window of % over % layers',
      [self, w, fConfig.Layers]);
  SetLength(fConfig.Spec, fConfig.Layers);
  for i := 0 to fConfig.Layers - 1 do
  begin
    fConfig.Spec[i].Ffn := dfMoe;
    if not odd(i) then
      fConfig.Spec[i].Window := w;
  end;
  CheckConfig(fGguf.FileName);
end;

const
  /// Hugging Face names and llama.cpp's for the tensors of one layer; the
  // expert biases under names of our own (the checkpoint interleaves gate and up)
  LAYER_NAMES: array[0..15, 0..1] of RawUtf8 = (
    ('self_attn.q_proj.weight', 'attn_q.weight'),
    ('self_attn.k_proj.weight', 'attn_k.weight'),
    ('self_attn.v_proj.weight', 'attn_v.weight'),
    ('self_attn.o_proj.weight', 'attn_output.weight'),
    ('self_attn.q_proj.bias', 'attn_q.bias'),
    ('self_attn.k_proj.bias', 'attn_k.bias'),
    ('self_attn.v_proj.bias', 'attn_v.bias'),
    ('self_attn.o_proj.bias', 'attn_output.bias'),
    ('self_attn.sinks', 'attn_sinks.weight'),
    ('input_layernorm.weight', 'attn_norm.weight'),
    ('post_attention_layernorm.weight', 'post_attention_norm.weight'),
    ('mlp.router.weight', 'ffn_gate_inp.weight'),
    ('mlp.router.bias', 'ffn_gate_inp.bias'),
    ('mlp.experts.gate_bias', 'ffn_gate_exps.bias'),
    ('mlp.experts.up_bias', 'ffn_up_exps.bias'),
    ('mlp.experts.down_proj_bias', 'ffn_down_exps.bias'));

function TGptOssModel.GgufName(const aName: RawUtf8): RawUtf8;
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

procedure TGptOssModel.LoadGgufExperts(aLayer: PtrInt; const aPrefix: RawUtf8);
var
  l: ^TDecoderLayer;
  h, fi, n: PtrInt;
  b: RawUtf8;
begin
  l := @fLayer[aLayer];
  h := fConfig.Hidden;
  fi := fConfig.ExpertIntermediate;
  n := fConfig.Experts;
  b := FormatUtf8('blk.%.', [aLayer]);
  l^.ExpertGate := GgufExperts(b + 'ffn_gate_exps.weight', n, fi, h);
  l^.ExpertUp := GgufExperts(b + 'ffn_up_exps.weight', n, fi, h);
  l^.ExpertDown := GgufExperts(b + 'ffn_down_exps.weight', n, h, fi);
  l^.ExpertGateBias := Weight(aPrefix + 'mlp.experts.gate_bias', [n, fi]);
  l^.ExpertUpBias := Weight(aPrefix + 'mlp.experts.up_bias', [n, fi]);
  l^.ExpertDownBias := Weight(aPrefix + 'mlp.experts.down_proj_bias', [n, h]);
end;

procedure TGptOssModel.LoadExpertBiases(aLayer: PtrInt; const aPrefix: RawUtf8);
var
  l: ^TDecoderLayer;
  fi, n, e, o: PtrInt;
  gub: PInferTensor;
  row, gb, ub: TSingleDynArray;
  b: RawUtf8;
begin
  l := @fLayer[aLayer];
  fi := fConfig.ExpertIntermediate;
  n := fConfig.Experts;
  b := aPrefix + 'mlp.experts.';
  gub := Checked(b + 'gate_up_proj_bias', [n, 2 * fi]);
  l^.ExpertDownBias := Weight(b + 'down_proj_bias', [n, fConfig.Hidden]);
  SetLength(row, 2 * fi);
  SetLength(gb, n * fi);
  SetLength(ub, n * fi);
  for e := 0 to n - 1 do
  begin
    InferRowToF32(gub^.Data, gub^.DType, e, 2 * fi, pointer(row));
    for o := 0 to fi - 1 do
    begin
      gb[e * fi + o] := row[2 * o];
      ub[e * fi + o] := row[2 * o + 1];
    end;
  end;
  l^.ExpertGateBias := Own(gb);
  l^.ExpertUpBias := Own(ub);
end;

function TGptOssModel.ByteTensor(const aName: RawUtf8;
  const aDims: array of Int64): PByteArray;
var
  t: PInferTensor;
  want: TInt64DynArray;
  i: PtrInt;
begin
  t := Find(aName);
  if t = nil then
    EInferError.RaiseUtf8('%: no tensor %', [self, aName]);
  if t^.DType <> idtU8 then
    EInferError.RaiseUtf8('%: % is %, expected U8',
      [self, aName, INFER_DTYPE_NAME[t^.DType]]);
  if not InferShapeIs(t^, aDims) then
  begin
    SetLength(want, length(aDims));
    for i := 0 to high(aDims) do
      want[i] := aDims[i];
    EInferError.RaiseUtf8('%: % is shaped %, expected %',
      [self, aName, InferShapeText(t^.Shape), InferShapeText(want)]);
  end;
  result := t^.Data;
end;

/// block aBlock of aRows rows of aBlocks MXFP4 blocks: from the checkpoint
// layout (scale apart, value 2j low and 2j + 1 high nibble of byte j) into
// ggml's (value j low and j + 16 high nibble of byte j) - false for a block
// whose exponent is outside the exact range and that holds a value
function RepackMxfp4(aSrc, aScales: PByteArray; aBlock: PtrInt;
  aDst: PInferBlockMxfp4): boolean;
var
  j: PtrInt;
  v: array[0..31] of byte;
  s: PByteArray;
begin
  s := @aSrc^[aBlock * 16];
  for j := 0 to 15 do
  begin
    v[2 * j] := s^[j] and 15;
    v[2 * j + 1] := s^[j] shr 4;
  end;
  aDst^.e := aScales^[aBlock];
  for j := 0 to 15 do
    aDst^.qs[j] := v[j] or (v[j + 16] shl 4);
  result := true;
  if (aDst^.e < INFER_MXFP4_EXACT_MIN) or (aDst^.e > INFER_MXFP4_EXACT_MAX) then
    // +0 and -0 are the nibbles 0 and 8
    for j := 0 to 15 do
      if aDst^.qs[j] and $77 <> 0 then
        result := false;
end;

procedure TGptOssModel.LoadMxfp4Experts(aLayer: PtrInt; const aPrefix: RawUtf8);
var
  l: ^TDecoderLayer;
  h, fi, n, e, r, k, hb, fb: PtrInt;
  gub, gus, db, ds: PByteArray;
  gate, up, down: PInferBlockMxfp4Array;
  b: RawUtf8;

  procedure Repack(aSrc, aScales: PByteArray; aSrcRow, aBlocks: PtrInt;
    aDst: PInferBlockMxfp4Array; aDstRow: PtrInt; const aName: RawUtf8);
  var
    c: PtrInt;
  begin
    for c := 0 to aBlocks - 1 do
      if not RepackMxfp4(aSrc, aScales, aSrcRow * aBlocks + c,
               @aDst^[aDstRow * aBlocks + c]) then
        EInferError.RaiseUtf8('%: % has the MXFP4 exponent % in row % block %, ' +
          'beyond the range computed exactly',
          [self, aName, aScales^[aSrcRow * aBlocks + c], aSrcRow, c]);
  end;

begin
  l := @fLayer[aLayer];
  h := fConfig.Hidden;
  fi := fConfig.ExpertIntermediate;
  n := fConfig.Experts;
  if (h and 31 <> 0) or (fi and 31 <> 0) then
    EInferError.RaiseUtf8('%: MXFP4 experts need widths that are multiples of 32',
      [self]);
  if not fBackend.Supports(ioLinear, idtMXFP4) then
    EInferError.RaiseUtf8('%: the backend has no MXFP4 Linear', [self]);
  hb := h shr 5;
  fb := fi shr 5;
  b := aPrefix + 'mlp.experts.';
  gub := ByteTensor(b + 'gate_up_proj_blocks', [n, 2 * fi, hb, 16]);
  gus := ByteTensor(b + 'gate_up_proj_scales', [n, 2 * fi, hb]);
  db := ByteTensor(b + 'down_proj_blocks', [n, h, fb, 16]);
  ds := ByteTensor(b + 'down_proj_scales', [n, h, fb]);
  // one owned array per matrix kind for all experts, one view per expert
  gate := OwnMxfp4(n * fi * hb);
  up := OwnMxfp4(n * fi * hb);
  down := OwnMxfp4(n * h * fb);
  SetLength(l^.ExpertGate, n);
  SetLength(l^.ExpertUp, n);
  SetLength(l^.ExpertDown, n);
  for e := 0 to n - 1 do
  begin
    // gate: the even rows of gate_up_proj, up: the odd ones
    for r := 0 to fi - 1 do
    begin
      k := e * fi + r;
      Repack(gub, gus, e * 2 * fi + 2 * r, hb, gate, k, b + 'gate_up_proj');
      Repack(gub, gus, e * 2 * fi + 2 * r + 1, hb, up, k, b + 'gate_up_proj');
    end;
    for r := 0 to h - 1 do
      Repack(db, ds, e * h + r, fb, down, e * h + r, b + 'down_proj');
    l^.ExpertGate[e] := InferWeight(@gate^[e * fi * hb], idtMXFP4);
    l^.ExpertUp[e] := InferWeight(@up^[e * fi * hb], idtMXFP4);
    l^.ExpertDown[e] := InferWeight(@down^[e * h * fb], idtMXFP4);
  end;
  LoadExpertBiases(aLayer, aPrefix);
end;

procedure TGptOssModel.LoadExperts(aLayer: PtrInt; const aPrefix: RawUtf8);
var
  l: ^TDecoderLayer;
  h, fi, n, e, i, o: PtrInt;
  gu, down: PInferTensor;
  row, gate, up, dn: TSingleDynArray;
  b: RawUtf8;
begin
  if fGguf <> nil then
  begin
    LoadGgufExperts(aLayer, aPrefix);
    exit;
  end;
  if fMxfp4 then
  begin
    LoadMxfp4Experts(aLayer, aPrefix);
    exit;
  end;
  l := @fLayer[aLayer];
  h := fConfig.Hidden;
  fi := fConfig.ExpertIntermediate;
  n := fConfig.Experts;
  b := aPrefix + 'mlp.experts.';
  gu := Checked(b + 'gate_up_proj', [n, h, 2 * fi]);
  down := Checked(b + 'down_proj', [n, fi, h]);
  LoadExpertBiases(aLayer, aPrefix);
  // row holds a row of gate_up_proj (2 * width) or of down_proj (hidden)
  SetLength(row, MaxPtrInt(2 * fi, h));
  // one nn.Linear matrix per expert: gate and up [width, hidden] from the
  // rows of gate_up_proj (one per input), down [hidden, width] from those of
  // down_proj - transposed while read
  SetLength(l^.ExpertGate, n);
  SetLength(l^.ExpertUp, n);
  SetLength(l^.ExpertDown, n);
  for e := 0 to n - 1 do
  begin
    // MatrixOf takes an F32 matrix over: a new one for every expert then
    if length(gate) <> fi * h then
      SetLength(gate, fi * h);
    if length(up) <> fi * h then
      SetLength(up, fi * h);
    if length(dn) <> h * fi then
      SetLength(dn, h * fi);
    for i := 0 to h - 1 do
    begin
      InferRowToF32(gu^.Data, gu^.DType, e * h + i, 2 * fi, pointer(row));
      for o := 0 to fi - 1 do
      begin
        gate[o * h + i] := row[2 * o];
        up[o * h + i] := row[2 * o + 1];
      end;
    end;
    for i := 0 to fi - 1 do
    begin
      InferRowToF32(down^.Data, down^.DType, e * fi + i, h, pointer(row));
      for o := 0 to h - 1 do
        dn[o * fi + i] := row[o];
    end;
    l^.ExpertGate[e] := MatrixOf(gate, fi, h, fWeights,
      FormatUtf8('%gate_up_proj[%].gate', [b, e]));
    l^.ExpertUp[e] := MatrixOf(up, fi, h, fWeights,
      FormatUtf8('%gate_up_proj[%].up', [b, e]));
    l^.ExpertDown[e] := MatrixOf(dn, h, fi, fWeights,
      FormatUtf8('%down_proj[%]', [b, e]));
  end;
end;

procedure TGptOssModel.LoadWeights;
var
  i, used: PtrInt;
  h, qw, kvw: Int64;
  p: RawUtf8;
  l: ^TDecoderLayer;
  headType: TInferDType;
begin
  // with Q4_0 the head stays Q8_0, as for Qwen3
  headType := fWeights;
  if headType = idtQ4_0 then
    headType := idtQ8_0;
  h := fConfig.Hidden;
  qw := fConfig.Heads * fConfig.HeadDim;
  kvw := fConfig.KvHeads * fConfig.HeadDim;
  fEmbedSrc := Checked('model.embed_tokens.weight', [fConfig.VocabSize, h],
    fGguf <> nil);
  // read per token: its block scales checked here
  if fEmbedSrc^.DType in INFER_BLOCK_DTYPES then
    CheckBlockScales(fEmbedSrc, 'model.embed_tokens.weight');
  if (fGguf = nil) and (fWeights = idtF32) then
    fEmbed := InferWeight(Weight('model.embed_tokens.weight', [fConfig.VocabSize, h]),
      idtF32);
  fNorm := Weight('model.norm.weight', [h]);
  fHead := Matrix('lm_head.weight', fConfig.VocabSize, h, headType);
  used := 3;
  SetLength(fLayer, fConfig.Layers);
  for i := 0 to fConfig.Layers - 1 do
  begin
    p := FormatUtf8('model.layers.%.', [i]);
    l := @fLayer[i];
    l^.Q := Matrix(p + 'self_attn.q_proj.weight', qw, h, fWeights);
    l^.K := Matrix(p + 'self_attn.k_proj.weight', kvw, h, fWeights);
    l^.V := Matrix(p + 'self_attn.v_proj.weight', kvw, h, fWeights);
    l^.O := Matrix(p + 'self_attn.o_proj.weight', h, qw, fWeights);
    l^.QBias := Weight(p + 'self_attn.q_proj.bias', [qw]);
    l^.KBias := Weight(p + 'self_attn.k_proj.bias', [kvw]);
    l^.VBias := Weight(p + 'self_attn.v_proj.bias', [kvw]);
    l^.OBias := Weight(p + 'self_attn.o_proj.bias', [h]);
    l^.Sinks := Weight(p + 'self_attn.sinks', [fConfig.Heads]);
    l^.InputNorm := Weight(p + 'input_layernorm.weight', [h]);
    l^.PostNorm := Weight(p + 'post_attention_layernorm.weight', [h]);
    l^.Router := Matrix(p + 'mlp.router.weight', fConfig.Experts, h, idtF32);
    l^.RouterBias := Weight(p + 'mlp.router.bias', [fConfig.Experts]);
    LoadExperts(i, p);
    // MXFP4: blocks and scales in place of each expert matrix; GGUF: gate and
    // up apart, with a bias each
    if fGguf <> nil then
      inc(used, 19)
    else
      inc(used, 17 + 2 * ord(fMxfp4));
  end;
  // an unknown tensor means a model we do not understand
  if used <> TensorCount then
    EInferError.RaiseUtf8('%: % tensors in the file, % understood',
      [self, TensorCount, used]);
end;


initialization
  RegisterDecoderModel(TGptOssModel);

end.
