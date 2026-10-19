/// native inference engine - text to dense embedding (BGE-M3)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - tokenizer + encoder + pooling: CLS of the last hidden state, L2-normalised
//   (BGE-M3 dense, verified against the revision's 1_Pooling/config.json by the
//   reference generator; a GGUF file must say bert.pooling_type = 2, CLS)
// - two sources: a Hugging Face folder (tokenizer.json, config.json,
//   model.safetensors) or one GGUF file holding tokenizer and weights - the
//   Q8_0 file of the lembed oracle (reference contract v1)
// - a text longer than MaxTokens (512 incl. <s> </s>, reference contract v1) is
//   refused, never cut silently: the caller chunks before
// - immutable after loading: Embed may run on several threads, each call uses its
//   own context
unit mormot.ai.infer.pipeline.embed;

{
  *****************************************************************************

    - TInferEmbedder: Embed texts into unit-length vectors

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
  mormot.core.text,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.loader.gguf,
  mormot.ai.infer.tokenizer.ugm,
  mormot.ai.infer.model.xlmr;

const
  /// the token limit of reference contract v1, <s> and </s> included
  INFER_EMBED_MAX_TOKENS = 512;

type
  /// text -> dense vector with a Unigram tokenizer and an XLM-R encoder
  TInferEmbedder = class
  protected
    fTokenizer: TUgmTokenizer;
    fModel: TXlmrModel;
    fMaxTokens: PtrInt;
    procedure Check;
  public
    /// load <aFolder>tokenizer.json, config.json and model.safetensors
    // - aBackend is not owned
    constructor Create(const aFolder: TFileName; aBackend: TInferBackend);
    /// load tokenizer and weights from one GGUF file (e.g. bge-m3-q8_0.gguf)
    // - aBackend is not owned
    constructor CreateFromGguf(const aFileName: TFileName; aBackend: TInferBackend);
    /// release tokenizer and weights
    destructor Destroy; override;
    /// the unit-length vectors of aTexts, one row of Dim singles per text
    // - the texts run as one right-padded batch; raises EInferError for a text
    // longer than MaxTokens
    function Embed(const aTexts: array of RawUtf8): TSingleDynArray;
    /// the vector size
    function Dim: PtrInt;
    /// the longest text accepted, in tokens (default INFER_EMBED_MAX_TOKENS)
    property MaxTokens: PtrInt
      read fMaxTokens write fMaxTokens;
    /// the tokenizer
    property Tokenizer: TUgmTokenizer
      read fTokenizer;
    /// the encoder
    property Model: TXlmrModel
      read fModel;
  end;


implementation

{ TInferEmbedder }

constructor TInferEmbedder.Create(const aFolder: TFileName; aBackend: TInferBackend);
var
  folder: TFileName;
begin
  inherited Create;
  folder := IncludeTrailingPathDelimiter(aFolder);
  fTokenizer := TUgmTokenizer.Create(folder + 'tokenizer.json');
  fModel := TXlmrModel.Create(folder, aBackend);
  Check;
end;

constructor TInferEmbedder.CreateFromGguf(const aFileName: TFileName;
  aBackend: TInferBackend);
begin
  inherited Create;
  fModel := TXlmrModel.CreateFromGguf(aFileName, aBackend);
  // CLS pooling is what Embed computes: mean or last-token pooling is refused
  if fModel.Gguf.Int('bert.pooling_type') <> 2 then
    EInferError.RaiseUtf8('%: bert.pooling_type % is not CLS (2)',
      [self, fModel.Gguf.Int('bert.pooling_type')]);
  fTokenizer := TUgmTokenizer.CreateFromGguf(fModel.Gguf);
  Check;
end;

procedure TInferEmbedder.Check;
begin
  if fTokenizer.VocabSize <> fModel.Config.VocabSize then
    EInferError.RaiseUtf8('%: tokenizer has % pieces, the model % rows',
      [self, fTokenizer.VocabSize, fModel.Config.VocabSize]);
  // padding and masking use the model's pad id: both files must agree on it
  if fTokenizer.PadId <> fModel.Config.PadId then
    EInferError.RaiseUtf8('%: tokenizer <pad> is %, the model pad_token_id %',
      [self, fTokenizer.PadId, fModel.Config.PadId]);
  fMaxTokens := MinPtrInt(INFER_EMBED_MAX_TOKENS, fModel.MaxTokens);
end;

destructor TInferEmbedder.Destroy;
begin
  fModel.Free;
  fTokenizer.Free;
  inherited Destroy;
end;

function TInferEmbedder.Dim: PtrInt;
begin
  result := fModel.Config.Hidden;
end;

function TInferEmbedder.Embed(const aTexts: array of RawUtf8): TSingleDynArray;
var
  ids: array of TIntegerDynArray;
  batch: TIntegerDynArray;
  ctx: TXlmrContext;
  rows, width, r, c, h: PtrInt;
begin
  result := nil;
  rows := length(aTexts);
  if rows = 0 then
    exit;
  SetLength(ids, rows);
  width := 0;
  for r := 0 to rows - 1 do
  begin
    // refuse a huge text before the tokenizer allocates per byte: no text of
    // MaxTokens pieces is longer than this, even before normalization shrinks it
    if length(aTexts[r]) > fMaxTokens * fTokenizer.MaxPieceLen * 4 then
      EInferError.RaiseUtf8('%.Embed: text % has % bytes, far more than % ' +
        'tokens - split it before', [self, r, length(aTexts[r]), fMaxTokens]);
    ids[r] := fTokenizer.Encode(aTexts[r]);
    if length(ids[r]) > fMaxTokens then
      EInferError.RaiseUtf8('%.Embed: text % has % tokens, more than % - ' +
        'split it before', [self, r, length(ids[r]), fMaxTokens]);
    if length(ids[r]) > width then
      width := length(ids[r]);
  end;
  // right-padded batch: padding is invisible to the real tokens
  SetLength(batch, rows * width);
  for r := 0 to rows - 1 do
    for c := 0 to width - 1 do
      if c < length(ids[r]) then
        batch[r * width + c] := ids[r][c]
      else
        batch[r * width + c] := fModel.Config.PadId;
  h := fModel.Config.Hidden;
  SetLength(result, rows * h);
  ctx := TXlmrContext.Create;
  try
    fModel.Forward(ctx, pointer(batch), rows, width);
    for r := 0 to rows - 1 do
    begin
      MoveFast(ctx.Hidden[r * width * h], result[r * h], h * SizeOf(single));
      fModel.Backend.L2Normalize(@result[r * h], h);
    end;
  finally
    ctx.Free;
  end;
end;

end.
