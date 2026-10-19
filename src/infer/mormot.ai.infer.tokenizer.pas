/// native inference engine - the tokenizer contract of a decoder family
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - what a chat template and an output parser need from a tokenizer: text to
//   ids, the id of a control token, the bytes of an id - so that a family
//   brings its own tokenizer (byte-level BPE with another pre-tokenizer,
//   SentencePiece) without touching them
unit mormot.ai.infer.tokenizer;

{
  *****************************************************************************

    - TInferTokenizer: the abstract tokenizer of a decoder family
    - TInferTokenizerClass: a family names its tokenizer by this class

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
  mormot.ai.infer.loader.gguf;

type
  /// the tokenizer of a decoder family
  // - thread-safe after loading: encoding works on local state only
  // - Encode treats every text as text: control tokens come from TokenId only
  TInferTokenizer = class
  public
    /// load a Hugging Face tokenizer.json - the inherited constructor does
    // nothing but TObject.Create, a descendant does the loading
    constructor Create(const aFileName: TFileName); virtual;
    /// load the tokenizer.ggml.* metadata of a GGUF file - as Create
    constructor CreateFromGguf(aGguf: TGgufFile); virtual;
    /// token ids of aText
    function Encode(const aText: RawUtf8): TIntegerDynArray; virtual; abstract;
    /// append the token ids of aText to aIds[0..aCount-1]
    procedure EncodeAppend(const aText: RawUtf8; var aIds: TIntegerDynArray;
      var aCount: PtrInt); virtual; abstract;
    /// the id of a control or added token by its content, -1 if none
    function TokenId(const aContent: RawUtf8): integer; virtual; abstract;
    /// the bytes of one token ('' for an id without a token)
    function TokenBytes(aId: integer): RawByteString; virtual; abstract;
    /// the concatenated bytes of the ids
    function Decode(const aIds: array of integer): RawUtf8; virtual; abstract;
    /// highest id + 1
    function VocabSize: integer; virtual; abstract;
  end;

  /// a tokenizer class, as a decoder family registers it
  TInferTokenizerClass = class of TInferTokenizer;


implementation

{ TInferTokenizer }

constructor TInferTokenizer.Create(const aFileName: TFileName);
begin
  inherited Create;
end;

constructor TInferTokenizer.CreateFromGguf(aGguf: TGgufFile);
begin
  inherited Create;
end;

end.
