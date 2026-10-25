/// native inference engine - Qwen3.5 decoder (causal language model)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room from the published architecture (looked up in Hugging Face's
//   modeling_qwen3_5.py and configuration_qwen3_5.py, transformers 5.19,
//   Apache-2.0, no code taken): token embedding, then per layer a mixer
//   between an RMSNorm and a residual and a SwiGLU block between an RMSNorm and
//   a residual; a final RMSNorm, the head tied to the embedding or its own
// - two kinds of mixer, as layer_types says (by default three linear_attention
//   to one full_attention): the Gated DeltaNet (in_proj_qkv through a depthwise
//   causal convolution of linear_conv_kernel_dim with silu, Q and K L2-normed,
//   beta = sigmoid(in_proj_b), the decay -exp(A_log) * softplus(in_proj_a +
//   dt_bias), the delta rule per value head, an RMSNorm of the head times
//   silu(in_proj_z), out_proj) or full attention whose q_proj gives per head the
//   query and a gate (Q and K normed per head, partial_rotary_factor of every
//   head rotating, the output times sigmoid(gate) before o_proj)
// - every RMSNorm but the gated one multiplies by (1 + weight): folded into the
//   weight at load, in float32 as the reference adds it
// - the checkpoints of Qwen3_5ForConditionalGeneration (model_type qwen3_5):
//   the settings in text_config, the tensors under model.language_model. (the
//   vision tower model.visual. and the multi-token prediction head mtp. are left
//   alone); Qwen3_5ForCausalLM (qwen3_5_text): the settings at the top, the
//   tensors under model.
// - the family of mormot.ai.infer.decoder: this unit reads config.json and
//   maps the tensors; the forward pass is TDecoderModel's
// - the tokenizer, chat template and parser: mormot.ai.infer.tokenizer.bpe,
//   mormot.ai.infer.chat and mormot.ai.llm.native (D7c); restore points of
//   the recurrent state: TDecoderContext.CheckpointId (D7d)
// - not yet: GGUF, a device (backlog D7e)
unit mormot.ai.infer.model.qwen35;

{
  *****************************************************************************

    - TQwen35Model: the qwen3_5 family of TDecoderModel (Qwen3.5 checkpoints)
    - TQwen35TextModel: the same over a qwen3_5_text checkpoint

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
  /// a Qwen3.5 causal language model (Qwen3.5-0.8B .. 27B, dense) over a
  // safetensors folder (one file or several with an index) with its weights in
  // BF16 or F32
  // - config.json of Qwen3_5ForConditionalGeneration (model_type qwen3_5, the
  // text settings in text_config): layer_types (or full_attention_interval),
  // head_dim, linear_num_key_heads, linear_num_value_heads,
  // linear_key_head_dim, linear_value_head_dim, linear_conv_kernel_dim,
  // rope_parameters (rope_theta, partial_rotary_factor; the multimodal
  // sections change nothing for text), tie_word_embeddings
  // - the rotary settings as transformers reads them: rope_scaling in place
  // of rope_parameters when not empty, type for rope_type, rope_theta and
  // partial_rotary_factor from the top when missing there (factor 0.25)
  // - refused: another hidden_act than silu, attention_bias, attn_output_gate
  // false, mlp_only_layers, another rope_type, an unknown rotary key, no
  // rope_theta at all, a rotating part that is odd
  // - in_proj_b and in_proj_a stay F32 when the matrices are quantized at
  // load: they feed the exponentials of beta and the decay
  TQwen35Model = class(TDecoderModel)
  protected
    /// the settings of the text model (aDoc: its object)
    procedure LoadTextConfig(const aDoc: TDocVariantData; const aSource: TFileName);
    procedure LoadConfig(const aConfigFile: TFileName); override;
    procedure LoadGgufConfig; override;
    procedure LoadWeights; override;
    function GgufName(const aName: RawUtf8): RawUtf8; override;
    /// a norm of (1 + weight): the sum kept with the model
    function PlusOne(const aName: RawUtf8; aCount: PtrInt): PSingleArray;
    /// the tensor prefix of the layout in place of "model."
    class function Prefix: RawUtf8; virtual;
  public
    /// 'qwen3_5'
    class function ModelType: RawUtf8; override;
    /// 'qwen35', as llama.cpp names it (GGUF: backlog D7)
    class function Architecture: RawUtf8; override;
  end;

  /// a Qwen3.5 text model saved as Qwen3_5ForCausalLM (model_type
  // qwen3_5_text): the settings at the top of config.json, the tensors under
  // model.
  TQwen35TextModel = class(TQwen35Model)
  protected
    class function Prefix: RawUtf8; override;
  public
    /// 'qwen3_5_text'
    class function ModelType: RawUtf8; override;
  end;


implementation


{ TQwen35Model }

class function TQwen35Model.ModelType: RawUtf8;
begin
  result := 'qwen3_5';
end;

class function TQwen35Model.Architecture: RawUtf8;
begin
  result := 'qwen35';
end;

class function TQwen35Model.Prefix: RawUtf8;
begin
  result := 'model.language_model.';
end;

const
  REQUIRED_KEYS: array[0..5] of RawUtf8 = (
    'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
    'num_attention_heads', 'num_key_value_heads');

  /// the keys of rope_parameters we read (the sections only place the
  // positions of an image: text has one position per token)
  ROPE_KEYS: array[0..5] of RawUtf8 = (
    'rope_type', 'type', 'rope_theta', 'partial_rotary_factor', 'mrope_section',
    'mrope_interleaved');

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
    EInferError.RaiseUtf8('TQwen35Model: % is no number', [aKey]);
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
    EInferError.RaiseUtf8('TQwen35Model: % is no integer', [aKey]);
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

procedure TQwen35Model.LoadTextConfig(const aDoc: TDocVariantData;
  const aSource: TFileName);
var
  d, rope, types, only: PDocVariantData;
  i, j, interval: PtrInt;
  t, k: RawUtf8;
  known, lin: boolean;
  dims: double;
begin
  d := @aDoc;
  for i := 0 to high(REQUIRED_KEYS) do
    if d^.GetValueIndex(REQUIRED_KEYS[i]) < 0 then
      EInferError.RaiseUtf8('%: % has no "%"', [self, aSource, REQUIRED_KEYS[i]]);
  // what this family does not compute
  t := d^.U['hidden_act'];
  if (t <> '') and (t <> 'silu') then
    EInferError.RaiseUtf8('%: hidden_act "%" not supported', [self, t]);
  if BoolOr(d^, 'attention_bias', false) then
    EInferError.RaiseUtf8('%: attention_bias not supported', [self]);
  // transformers gates every attention layer: a checkpoint that says it does
  // not would be computed otherwise than it was trained
  if not BoolOr(d^, 'attn_output_gate', true) then
    EInferError.RaiseUtf8('%: attention without its output gate not supported', [self]);
  if d^.GetAsArray('mlp_only_layers', only) and (only^.Count > 0) then
    EInferError.RaiseUtf8('%: mlp_only_layers not supported', [self]);
  fConfig.VocabSize := d^.I['vocab_size'];
  fConfig.Hidden := d^.I['hidden_size'];
  fConfig.Intermediate := d^.I['intermediate_size'];
  fConfig.Layers := d^.I['num_hidden_layers'];
  fConfig.Heads := d^.I['num_attention_heads'];
  fConfig.KvHeads := d^.I['num_key_value_heads'];
  // the defaults of Qwen3_5TextConfig
  fConfig.HeadDim := IntOr(d^, 'head_dim', 256);
  fConfig.MaxPositions := IntOr(d^, 'max_position_embeddings', 32768);
  fConfig.RmsEps := FloatOr(d^, 'rms_norm_eps', 1e-6);
  fConfig.LinKeyHeads := IntOr(d^, 'linear_num_key_heads', 16);
  fConfig.LinValueHeads := IntOr(d^, 'linear_num_value_heads', 32);
  fConfig.LinKeyDim := IntOr(d^, 'linear_key_head_dim', 128);
  fConfig.LinValueDim := IntOr(d^, 'linear_value_head_dim', 128);
  fConfig.LinConv := IntOr(d^, 'linear_conv_kernel_dim', 4);
  // rope_scaling (the older name) in place of rope_parameters when given; a
  // value missing there comes from the top (transformers' convert_rope_params_
  // to_dict): rope_theta without a default here - a checkpoint names its own
  // (an empty rope_scaling is false in Python: rope_parameters then)
  if not d^.GetAsObject('rope_scaling', rope) or (rope^.Count = 0) then
    if not d^.GetAsObject('rope_parameters', rope) then
      rope := nil;
  if rope <> nil then
    for i := 0 to rope^.Count - 1 do
    begin
      k := rope^.Names[i];
      known := false;
      for j := 0 to high(ROPE_KEYS) do
        if k = ROPE_KEYS[j] then
          known := true;
      if not known then
        EInferError.RaiseUtf8('%: rope setting % not supported', [self, k]);
    end;
  t := '';
  if rope <> nil then
    if rope^.GetValueIndex('rope_type') >= 0 then
      t := rope^.U['rope_type']
    else
      t := rope^.U['type']; // the older name, as standardize_rope_params
  if (t <> '') and (t <> 'default') then
    EInferError.RaiseUtf8('%: rope_type "%" not supported', [self, t]);
  if (rope <> nil) and (rope^.GetValueIndex('rope_theta') >= 0) then
    fConfig.RopeTheta := FloatOr(rope^, 'rope_theta', 0)
  else if d^.GetValueIndex('rope_theta') >= 0 then
    fConfig.RopeTheta := FloatOr(d^, 'rope_theta', 0)
  else
    EInferError.RaiseUtf8('%: % has no rope_theta', [self, aSource]);
  // int(head_dim * partial_rotary_factor) values of every head rotate
  dims := FloatOr(d^, 'partial_rotary_factor', 0.25);
  if rope <> nil then
    dims := FloatOr(rope^, 'partial_rotary_factor', dims);
  if not (dims > 0) or (dims > 1) then
    EInferError.RaiseUtf8('%: a partial_rotary_factor of %', [self, dims]);
  fConfig.RopeDims := Trunc(fConfig.HeadDim * dims);
  if (fConfig.RopeDims <= 0) or odd(fConfig.RopeDims) then
    EInferError.RaiseUtf8('%: % rotating values of a head of %',
      [self, fConfig.RopeDims, fConfig.HeadDim]);
  // the kind of every layer: layer_types, else every interval-th is full
  if (fConfig.Layers > 0) and (fConfig.Layers <= 1024) then
  begin
    SetLength(fConfig.Spec, fConfig.Layers);
    if not d^.GetAsArray('layer_types', types) then
      types := nil
    else if types^.Count <> fConfig.Layers then
      EInferError.RaiseUtf8('%: % layer types for % layers',
        [self, types^.Count, fConfig.Layers]);
    interval := IntOr(d^, 'full_attention_interval', 4);
    if (types = nil) and (interval <= 0) then
      EInferError.RaiseUtf8('%: a full_attention_interval of %', [self, interval]);
    for i := 0 to fConfig.Layers - 1 do
    begin
      if types <> nil then
        t := VariantToUtf8(types^.Values[i])
      else if (i + 1) mod interval <> 0 then
        t := 'linear_attention'
      else
        t := 'full_attention';
      if t = 'linear_attention' then
        fConfig.Spec[i].Mixer := dmDeltaNet
      else if t = 'full_attention' then
        fConfig.Spec[i].OutputGate := true
      else
        EInferError.RaiseUtf8('%: layer type "%" not supported', [self, t]);
    end;
    // full attention only: the DeltaNet shapes (defaults included) unused
    lin := false;
    for i := 0 to fConfig.Layers - 1 do
      if fConfig.Spec[i].Mixer = dmDeltaNet then
        lin := true;
    if not lin then
    begin
      fConfig.LinKeyHeads := 0;
      fConfig.LinValueHeads := 0;
      fConfig.LinKeyDim := 0;
      fConfig.LinValueDim := 0;
      fConfig.LinConv := 0;
    end;
  end;
  CheckConfig(aSource);
end;

procedure TQwen35Model.LoadConfig(const aConfigFile: TFileName);
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
    // the head: tied as tie_word_embeddings at the top says (transformers
    // ignores the key of text_config there), false by default
    fConfig.TiedEmbeddings := BoolOr(d^, 'tie_word_embeddings', false);
    if not fConfig.TiedEmbeddings then
      fStackHead := 'lm_head.weight';
    if ModelType = 'qwen3_5' then
    begin
      if not d^.GetAsObject('text_config', text) then
        EInferError.RaiseUtf8('%: % has no text_config', [self, aConfigFile]);
      d := text;
    end;
  end
  else
    fConfig.TiedEmbeddings := fStackHead = fStackEmbed;
  LoadTextConfig(d^, aConfigFile);
end;

procedure TQwen35Model.LoadGgufConfig;
begin
  EInferError.RaiseUtf8('%: GGUF not supported yet', [self]);
end;

function TQwen35Model.GgufName(const aName: RawUtf8): RawUtf8;
begin
  result := '';
end;

function TQwen35Model.PlusOne(const aName: RawUtf8; aCount: PtrInt): PSingleArray;
var
  w: PSingleArray;
  v: TSingleDynArray;
  i: PtrInt;
begin
  // output * (1.0 + weight.float()): the sum in float32, then the product -
  // the folded weight gives the very products of the reference
  w := Weight(aName, [aCount]);
  SetLength(v, aCount);
  for i := 0 to aCount - 1 do
    v[i] := 1 + w^[i];
  result := Own(v);
end;

procedure TQwen35Model.LoadWeights;
var
  i, j, used, vh, kd, vd, cd, kw: PtrInt;
  h, hd, qw, kvw: Int64;
  p: RawUtf8;
  l: ^TDecoderLayer;
  headType: TInferDType;
  alog: PSingleArray;
  negA: TSingleDynArray;
  e: single;
begin
  // with Q4_0 the head stays Q8_0, as for Qwen3
  headType := fWeights;
  if headType = idtQ4_0 then
    headType := idtQ8_0;
  h := fConfig.Hidden;
  fEmbedSrc := Checked('model.embed_tokens.weight', [fConfig.VocabSize, h]);
  used := 2;
  if not fConfig.TiedEmbeddings then
    // outside the prefix: not counted below
    fHead := Matrix('lm_head.weight', fConfig.VocabSize, h, headType)
  else if fWeights = idtF32 then
  begin
    fEmbed := InferWeight(Weight('model.embed_tokens.weight', [fConfig.VocabSize, h]),
      idtF32);
    fHead := fEmbed;
  end
  else
    fHead := Matrix('model.embed_tokens.weight', fConfig.VocabSize, h, headType);
  fNorm := PlusOne('model.norm.weight', h);
  hd := fConfig.HeadDim;
  qw := fConfig.Heads * hd;
  kvw := fConfig.KvHeads * hd;
  vh := fConfig.LinValueHeads;
  kd := fConfig.LinKeyHeads * fConfig.LinKeyDim;
  vd := vh * fConfig.LinValueDim;
  cd := 2 * kd + vd;
  kw := fConfig.LinConv;
  SetLength(fLayer, fConfig.Layers);
  for i := 0 to fConfig.Layers - 1 do
  begin
    p := FormatUtf8('model.layers.%.', [i]);
    l := @fLayer[i];
    l^.InputNorm := PlusOne(p + 'input_layernorm.weight', h);
    l^.PostNorm := PlusOne(p + 'post_attention_layernorm.weight', h);
    l^.Gate := Matrix(p + 'mlp.gate_proj.weight', fConfig.Intermediate, h, fWeights);
    l^.Up := Matrix(p + 'mlp.up_proj.weight', fConfig.Intermediate, h, fWeights);
    l^.Down := Matrix(p + 'mlp.down_proj.weight', h, fConfig.Intermediate, fWeights);
    inc(used, 5);
    if fConfig.Spec[i].Mixer = dmDeltaNet then
    begin
      p := p + 'linear_attn.';
      l^.LinQkv := Matrix(p + 'in_proj_qkv.weight', cd, h, fWeights);
      l^.LinZ := Matrix(p + 'in_proj_z.weight', vd, h, fWeights);
      l^.LinB := Matrix(p + 'in_proj_b.weight', vh, h, idtF32);
      l^.LinA := Matrix(p + 'in_proj_a.weight', vh, h, idtF32);
      l^.LinOut := Matrix(p + 'out_proj.weight', h, vd, fWeights);
      // [channels, 1, kernel] as nn.Conv1d stores a depthwise one
      l^.LinConvW := Weight(p + 'conv1d.weight', [cd, 1, kw]);
      l^.LinDtBias := Weight(p + 'dt_bias', [vh]);
      // -A_log.float().exp(): the factor of the decay, in float32
      alog := Weight(p + 'A_log', [vh]);
      SetLength(negA, vh);
      for j := 0 to vh - 1 do
      begin
        e := Exp(alog^[j]);
        negA[j] := -e;
      end;
      l^.LinNegA := Own(negA);
      // the gated RMSNorm multiplies by its weight as it is
      l^.LinNorm := Weight(p + 'norm.weight', [fConfig.LinValueDim]);
      inc(used, 9);
    end
    else
    begin
      p := p + 'self_attn.';
      // per head the query rows, then the gate rows
      l^.Q := Matrix(p + 'q_proj.weight', 2 * qw, h, fWeights);
      l^.K := Matrix(p + 'k_proj.weight', kvw, h, fWeights);
      l^.V := Matrix(p + 'v_proj.weight', kvw, h, fWeights);
      l^.O := Matrix(p + 'o_proj.weight', h, qw, fWeights);
      l^.QNorm := PlusOne(p + 'q_norm.weight', hd);
      l^.KNorm := PlusOne(p + 'k_norm.weight', hd);
      inc(used, 6);
    end;
  end;
  // an unknown tensor of the text model means one we do not understand
  if used <> TensorCountOf(fStackPrefix) then
    EInferError.RaiseUtf8('%: % tensors under %, % understood',
      [self, TensorCountOf(fStackPrefix), fStackPrefix, used]);
end;


{ TQwen35TextModel }

class function TQwen35TextModel.ModelType: RawUtf8;
begin
  result := 'qwen3_5_text';
end;

class function TQwen35TextModel.Prefix: RawUtf8;
begin
  result := 'model.';
end;


initialization
  RegisterDecoderModel(TQwen35Model);
  RegisterDecoderModel(TQwen35TextModel);

end.
