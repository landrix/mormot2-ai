/// native inference engine - Unigram (SentencePiece "UGM") tokenizer
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - ported from llama.cpp src/llama-vocab.cpp (llm_tokenizer_ugm), commit
//   6868a52cb337 - see UPSTREAM_BASE and NOTICE. Its license, kept with the code:
//     MIT License
//     Copyright (c) 2023-2026 The ggml authors
//     Permission is hereby granted, free of charge, to any person obtaining a copy
//     of this software and associated documentation files (the "Software"), to
//     deal in the Software without restriction, including without limitation the
//     rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//     sell copies of the Software, and to permit persons to whom the Software is
//     furnished to do so, subject to the following conditions:
//     The above copyright notice and this permission notice shall be included in
//     all copies or substantial portions of the Software.
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//     IN THE SOFTWARE.
// - the algorithm is SentencePiece's: charsmap normalization
//   (longest match in an XOR-compressed double array), whitespace folded to one
//   U+2581 before each word, Viterbi over the piece scores, unknown code points
//   at min score - 10, consecutive unknowns merged into one
// - behaviour = SentencePiece with parse_special=false: control and special
//   pieces (<s> </s> <pad> <unk> <mask>) never match text; this differs from the
//   Hugging Face tokenizer, which reads "<s>" in text as a control token
// - data from tokenizer.json (Hugging Face) or from the metadata of a GGUF file
//   (tokenizer.ggml.model "t5", as llama.cpp reads it): both carry the precompiled
//   charsmap, vocabulary and scores of the SentencePiece model; the source must
//   describe exactly this pipeline, otherwise it is refused
// - thread-safe after loading: Encode uses only local state
unit mormot.ai.infer.tokenizer.ugm;

{
  *****************************************************************************

    - TUgmTokenizer: load tokenizer.json or GGUF metadata, Normalize and
      Encode text to ids

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
  mormot.core.buffers,
  mormot.core.variants,
  mormot.ai.infer.tensor,
  mormot.ai.infer.loader.gguf;

type
  /// a SentencePiece Unigram tokenizer
  TUgmTokenizer = class
  protected
    fPieces: TRawUtf8DynArray;
    fScores: TSingleDynArray;
    fMatchable: TBooleanDynArray;
    fHashHead, fHashNext: TIntegerDynArray;
    fHashMask: PtrInt;
    fMaxPieceLen: PtrInt;
    fUnkId, fBosId, fEosId, fPadId: integer;
    fUnknownScore: double;
    fCharsMap: RawByteString;
    fXcda: PCardinalArray;
    fXcdaCount: PtrInt;
    fReplacements: PUtf8Char;
    fReplacementsLen: PtrInt;
    procedure LoadCharsMap(const aCharsMap: RawByteString);
    procedure BuildIndex;
    procedure Finish(const aCharsMap: RawByteString);
    function PieceId(p: PUtf8Char; aLen: PtrInt): integer;
    function NormalizePrefix(p: PUtf8Char; aLen: PtrInt;
      out aRepl: PUtf8Char; out aReplLen: PtrInt): PtrInt;
  public
    /// load and validate a Hugging Face tokenizer.json of this pipeline:
    // Precompiled + Replace(" {2,}"), Metaspace(add_prefix_space), Unigram
    // - raises EInferError otherwise
    constructor Create(const aFileName: TFileName);
    /// load and validate the tokenizer.ggml.* metadata of a GGUF file: model
    // "t5" (Unigram) with space prefix, folded whitespace, <s> ... </s> added
    // - raises EInferError otherwise; aGguf is only read during the call
    constructor CreateFromGguf(aGguf: TGgufFile);
    /// the SentencePiece-normalized text: charsmap applied, whitespace folded,
    // U+2581 before each word (what Encode segments)
    function Normalize(const aText: RawUtf8): RawUtf8;
    /// token ids of aText; aAddSpecial wraps them in <s> ... </s>
    // - invalid UTF-8 bytes become U+FFFD, like SentencePiece
    function Encode(const aText: RawUtf8; aAddSpecial: boolean = true): TIntegerDynArray;
    /// number of pieces, special ones included
    function VocabSize: integer;
    /// the text of a piece id
    function Piece(aId: integer): RawUtf8;
    /// id of <unk>
    property UnkId: integer
      read fUnkId;
    /// id of <s>
    property BosId: integer
      read fBosId;
    /// id of </s>
    property EosId: integer
      read fEosId;
    /// id of <pad>, -1 if the vocabulary has none
    property PadId: integer
      read fPadId;
    /// the longest piece, in bytes of normalized text
    property MaxPieceLen: PtrInt
      read fMaxPieceLen;
  end;


implementation

const
  /// the UTF-8 bytes of U+2581 LOWER ONE EIGHTH BLOCK (SentencePiece's escaped
  // space) and of U+FFFD REPLACEMENT CHARACTER - kept as bytes: a non-ASCII
  // string literal would be re-encoded by Unicode Delphi
  ESCAPED_SPACE_BYTES: array[0..2] of byte = ($E2, $96, $81);
  REPLACEMENT_BYTES: array[0..2] of byte = ($EF, $BF, $BD);

var
  ESCAPED_SPACE, REPLACEMENT_CHAR: RawUtf8;

/// length of the UTF-8 sequence starting at p (1..4), 1 for an invalid one
// - aValid tells whether the sequence is a well-formed code point, as
// SentencePiece decodes it: no overlong form, no surrogate, at most U+10FFFF
function Utf8SeqLen(p: PByteArray; aLen: PtrInt; out aValid: boolean): PtrInt;
var
  b: byte;
  i: PtrInt;
  cp: cardinal;
begin
  b := p^[0];
  if b < $80 then
  begin
    aValid := true;
    result := 1;
    exit;
  end
  else if b and $e0 = $c0 then
  begin
    result := 2;
    cp := b and $1f;
  end
  else if b and $f0 = $e0 then
  begin
    result := 3;
    cp := b and $0f;
  end
  else if b and $f8 = $f0 then
  begin
    result := 4;
    cp := b and $07;
  end
  else
  begin
    aValid := false;
    result := 1;
    exit;
  end;
  aValid := result <= aLen;
  if aValid then
    for i := 1 to result - 1 do
      if p^[i] and $c0 <> $80 then
      begin
        aValid := false;
        break;
      end
      else
        cp := (cp shl 6) or (p^[i] and $3f);
  if aValid then
    case result of
      2:
        aValid := cp >= $80;
      3:
        aValid := (cp >= $800) and ((cp < $d800) or (cp > $dfff));
      4:
        aValid := (cp >= $10000) and (cp <= $10ffff);
    end;
  if not aValid then
    result := 1;
end;


{ TUgmTokenizer }

constructor TUgmTokenizer.Create(const aFileName: TFileName);
var
  doc: TDocVariantData;
  norm, list, step, pre, model, vocab, entry, added: PDocVariantData;
  i, n, id: PtrInt;
  replaceOk: boolean;
  post, single: PDocVariantData;
  charsmap: RawUtf8;
begin
  inherited Create;
  if not doc.InitJsonFromFile(aFileName, JSON_FAST_FLOAT) then
    EInferError.RaiseUtf8('%: cannot read %', [self, aFileName]);
  // the pipeline this class implements, and nothing else
  if not doc.GetAsObject('normalizer', norm) or (norm^.U['type'] <> 'Sequence') or
     not norm^.GetAsArray('normalizers', list) or (list^.Count <> 2) then
    EInferError.RaiseUtf8('%: normalizer is not Sequence[Precompiled, Replace]', [self]);
  step := _Safe(list^.Values[0]);
  if step^.U['type'] <> 'Precompiled' then
    EInferError.RaiseUtf8('%: first normalizer is not Precompiled', [self]);
  charsmap := step^.U['precompiled_charsmap'];
  step := _Safe(list^.Values[1]);
  replaceOk := (step^.U['type'] = 'Replace') and (step^.U['content'] = ' ') and
    (step^.O['pattern']^.U['Regex'] = ' {2,}');
  if not replaceOk then
    EInferError.RaiseUtf8('%: second normalizer is not Replace(" {2,}", " ")', [self]);
  if not doc.GetAsObject('pre_tokenizer', pre) or (pre^.U['type'] <> 'Metaspace') or
     (pre^.U['replacement'] <> ESCAPED_SPACE) or not pre^.B['add_prefix_space'] or
     ((pre^.GetValueIndex('prepend_scheme') >= 0) and
      (pre^.U['prepend_scheme'] <> 'always')) then
    EInferError.RaiseUtf8('%: pre_tokenizer is not Metaspace(U+2581, prefix)', [self]);
  // <s> A </s> is hard-wired below: the file must say exactly that
  if not doc.GetAsObject('post_processor', post) or
     (post^.U['type'] <> 'TemplateProcessing') or
     not post^.GetAsArray('single', single) or (single^.Count <> 3) or
     (_Safe(single^.Values[0])^.O['SpecialToken']^.U['id'] <> '<s>') or
     (_Safe(single^.Values[1])^.O['Sequence']^.U['id'] <> 'A') or
     (_Safe(single^.Values[2])^.O['SpecialToken']^.U['id'] <> '</s>') then
    EInferError.RaiseUtf8('%: post_processor is not the <s> A </s> template', [self]);
  if not doc.GetAsObject('model', model) or (model^.U['type'] <> 'Unigram') or
     model^.B['byte_fallback'] or not model^.GetAsArray('vocab', vocab) or
     (vocab^.Count = 0) then
    EInferError.RaiseUtf8('%: model is not a Unigram vocabulary', [self]);
  if model^.GetValueIndex('unk_id') < 0 then
    EInferError.RaiseUtf8('%: Unigram model without unk_id', [self]);
  fUnkId := model^.I['unk_id'];
  // pieces and their scores, kept as single like SentencePiece stores them
  n := vocab^.Count;
  SetLength(fPieces, n);
  SetLength(fScores, n);
  SetLength(fMatchable, n);
  for i := 0 to n - 1 do
  begin
    entry := _Safe(vocab^.Values[i]);
    if entry^.Count <> 2 then
      EInferError.RaiseUtf8('%: vocab entry % is not [piece, score]', [self, i]);
    VariantToUtf8(entry^.Values[0], fPieces[i]);
    fScores[i] := entry^.Values[1];
    fMatchable[i] := fPieces[i] <> '';
  end;
  if (fUnkId < 0) or (fUnkId >= n) or (fPieces[fUnkId] <> '<unk>') then
    EInferError.RaiseUtf8('%: unk_id % is not the <unk> piece', [self, fUnkId]);
  // special pieces never match text (parse_special=false)
  fBosId := -1;
  fEosId := -1;
  fPadId := -1;
  if doc.GetAsArray('added_tokens', added) then
    for i := 0 to added^.Count - 1 do
    begin
      entry := _Safe(added^.Values[i]);
      id := entry^.I['id'];
      if (id < 0) or (id >= n) then
        EInferError.RaiseUtf8('%: added token id % outside the vocabulary', [self, id]);
      // a non-special added token is a user-defined piece (score 0, matched in
      // the normalizer): not implemented, so refused instead of mis-tokenized
      if not entry^.B['special'] then
        EInferError.RaiseUtf8('%: user-defined token "%" not supported',
          [self, entry^.U['content']]);
      fMatchable[id] := false;
      if entry^.U['content'] = '<s>' then
        fBosId := id
      else if entry^.U['content'] = '</s>' then
        fEosId := id
      else if entry^.U['content'] = '<pad>' then
        fPadId := id;
    end;
  fMatchable[fUnkId] := false;
  if (fBosId < 0) or (fEosId < 0) then
    EInferError.RaiseUtf8('%: no <s> / </s> special tokens', [self]);
  Finish(Base64ToBin(charsmap));
end;

const
  // llama.cpp's token types (LLAMA_TOKEN_TYPE_*)
  TT_NORMAL = 1;
  TT_UNKNOWN = 2;
  TT_CONTROL = 3;
  TT_USER_DEFINED = 4;
  TT_UNUSED = 5;
  TT_BYTE = 6;

constructor TUgmTokenizer.CreateFromGguf(aGguf: TGgufFile);
var
  i, n: PtrInt;
  count: Int64;
  scores: PSingleArray;
  types: PIntegerArray;

  function Id(const aKey: RawUtf8): integer;
  var
    v: Int64;
  begin
    v := aGguf.Int(aKey);
    if (v < 0) or (v >= n) then
      EInferError.RaiseUtf8('%: % = % outside the vocabulary', [self, aKey, v]);
    result := v;
  end;

begin
  inherited Create;
  if aGguf = nil then
    EInferError.RaiseUtf8('%.CreateFromGguf: no file', [self]);
  // the pipeline this class implements, and nothing else - flags llama.cpp
  // defaults when absent are required explicitly where the default differs
  if aGguf.Str('tokenizer.ggml.model') <> 't5' then
    EInferError.RaiseUtf8('%: tokenizer.ggml.model "%" is not t5 (Unigram)',
      [self, aGguf.Str('tokenizer.ggml.model')]);
  if aGguf.Has('tokenizer.ggml.pre') and
     (aGguf.Str('tokenizer.ggml.pre') <> 'default') then
    EInferError.RaiseUtf8('%: pre-tokenizer "%" not supported',
      [self, aGguf.Str('tokenizer.ggml.pre')]);
  if not aGguf.BoolDef('tokenizer.ggml.add_space_prefix', true) or
     not aGguf.Bool('tokenizer.ggml.remove_extra_whitespaces') or
     not aGguf.BoolDef('tokenizer.ggml.escape_whitespaces', true) or
     aGguf.BoolDef('tokenizer.ggml.treat_whitespace_as_suffix', false) then
    EInferError.RaiseUtf8('%: whitespace handling is not prefix + folded', [self]);
  // <s> A </s> is hard-wired in Encode: the file must say exactly that
  if not aGguf.Bool('tokenizer.ggml.add_bos_token') or
     not aGguf.Bool('tokenizer.ggml.add_eos_token') then
    EInferError.RaiseUtf8('%: <s> and </s> are not both added', [self]);
  fPieces := aGguf.Strings('tokenizer.ggml.tokens');
  n := length(fPieces);
  if n = 0 then
    EInferError.RaiseUtf8('%: empty vocabulary', [self]);
  count := n;
  scores := aGguf.Floats32('tokenizer.ggml.scores', count);
  types := aGguf.Ints32('tokenizer.ggml.token_type', count);
  fUnkId := Id('tokenizer.ggml.unknown_token_id');
  fBosId := Id('tokenizer.ggml.bos_token_id');
  fEosId := Id('tokenizer.ggml.eos_token_id');
  fPadId := -1;
  if aGguf.Has('tokenizer.ggml.padding_token_id') then
    fPadId := Id('tokenizer.ggml.padding_token_id');
  SetLength(fScores, n);
  SetLength(fMatchable, n);
  for i := 0 to n - 1 do
  begin
    fScores[i] := scores^[i];
    case types^[i] of
      TT_NORMAL:
        fMatchable[i] := fPieces[i] <> '';
      TT_UNKNOWN:
        if i <> fUnkId then
          EInferError.RaiseUtf8('%: piece % is a second unknown token', [self, i]);
      TT_CONTROL, TT_UNUSED:
        // never matched in text: control pieces by parse_special=false, unused
        // ones as the oracle does (lembed reads "[PAD250000]" of BGE-M3 as
        // plain text - lembed-tokens, extra cases)
        ;
      TT_USER_DEFINED, TT_BYTE:
        // matched in the normalizer, or byte fallback: not implemented, so
        // refused instead of mis-tokenized
        EInferError.RaiseUtf8('%: piece "%" of type % not supported',
          [self, fPieces[i], types^[i]]);
    else
      EInferError.RaiseUtf8('%: piece % has unknown type %', [self, i, types^[i]]);
    end;
  end;
  if types^[fUnkId] <> TT_UNKNOWN then
    EInferError.RaiseUtf8('%: unknown_token_id % is no unknown piece', [self, fUnkId]);
  if (types^[fBosId] <> TT_CONTROL) or (types^[fEosId] <> TT_CONTROL) or
     ((fPadId >= 0) and (types^[fPadId] <> TT_CONTROL)) then
    EInferError.RaiseUtf8('%: <s> </s> <pad> must be control pieces', [self]);
  Finish(aGguf.Bytes('tokenizer.ggml.precompiled_charsmap'));
end;

procedure TUgmTokenizer.Finish(const aCharsMap: RawByteString);
var
  i: PtrInt;
  minScore, unknown: single;
begin
  // unknown code points score below every normal piece - computed in single,
  // like llama.cpp (min_score - 10.0f), then widened: an extra double digit
  // could resolve a near-tie differently
  minScore := 3.4e38;
  for i := 0 to high(fScores) do
    if fMatchable[i] and (fScores[i] < minScore) then
      minScore := fScores[i];
  unknown := minScore - 10;
  fUnknownScore := unknown;
  LoadCharsMap(aCharsMap);
  BuildIndex;
end;

procedure TUgmTokenizer.LoadCharsMap(const aCharsMap: RawByteString);
var
  xcdaBytes: cardinal;
begin
  fCharsMap := aCharsMap;
  if length(fCharsMap) < 4 then
    EInferError.RaiseUtf8('%: precompiled charsmap missing or too short', [self]);
  MoveFast(pointer(fCharsMap)^, xcdaBytes, 4);
  if (xcdaBytes = 0) or (xcdaBytes and 3 <> 0) or
     (Int64(xcdaBytes) > length(fCharsMap) - 4) then
    EInferError.RaiseUtf8('%: charsmap trie size % invalid', [self, xcdaBytes]);
  fXcda := @PByteArray(fCharsMap)^[4];
  fXcdaCount := xcdaBytes shr 2;
  fReplacements := @PByteArray(fCharsMap)^[4 + PtrInt(xcdaBytes)];
  fReplacementsLen := length(fCharsMap) - 4 - PtrInt(xcdaBytes);
end;

procedure TUgmTokenizer.BuildIndex;
var
  i, n, h: PtrInt;
begin
  // open hash of the matchable pieces: crc32c of the bytes, chained by index -
  // a lookup costs no allocation, so the Viterbi loop stays allocation-free
  n := length(fPieces);
  fHashMask := 1;
  while fHashMask < n * 2 do
    fHashMask := fHashMask shl 1;
  dec(fHashMask);
  SetLength(fHashHead, fHashMask + 1);
  FillCharFast(pointer(fHashHead)^, length(fHashHead) * SizeOf(integer), $ff); // -1
  SetLength(fHashNext, n);
  fMaxPieceLen := 0;
  for i := 0 to n - 1 do
  begin
    fHashNext[i] := -1;
    if not fMatchable[i] then
      continue;
    if length(fPieces[i]) > fMaxPieceLen then
      fMaxPieceLen := length(fPieces[i]);
    h := crc32c(0, pointer(fPieces[i]), length(fPieces[i])) and fHashMask;
    fHashNext[i] := fHashHead[h];
    fHashHead[h] := i;
  end;
end;

function TUgmTokenizer.PieceId(p: PUtf8Char; aLen: PtrInt): integer;
begin
  result := fHashHead[crc32c(0, pointer(p), aLen) and fHashMask];
  while result >= 0 do
  begin
    if (length(fPieces[result]) = aLen) and
       CompareMem(pointer(fPieces[result]), p, aLen) then
      exit;
    result := fHashNext[result];
  end;
end;

function TUgmTokenizer.NormalizePrefix(p: PUtf8Char; aLen: PtrInt;
  out aRepl: PUtf8Char; out aReplLen: PtrInt): PtrInt;
var
  node, value, best: cardinal;
  i, matched: PtrInt;
  c: byte;
  unitValue: cardinal;
  valid: boolean;
begin
  // longest prefix in the XOR-compressed double array: bits 10-30 base (shifted
  // by bit 9), bits 0-7 + 31 label check, bit 8 leaf, value = low 31 bits
  matched := 0;
  best := 0;
  unitValue := fXcda^[0];
  node := (unitValue shr 10) shl ((unitValue and (1 shl 9)) shr 6);
  for i := 0 to aLen - 1 do
  begin
    c := PByteArray(p)^[i];
    if c = 0 then
      break;
    node := node xor c;
    if node >= cardinal(fXcdaCount) then
      break;
    unitValue := fXcda^[node];
    if unitValue and (cardinal(1 shl 31) or $ff) <> c then
      break;
    value := unitValue; // leaf flag of the matched node
    node := node xor ((unitValue shr 10) shl ((unitValue and (1 shl 9)) shr 6));
    if (value shr 8) and 1 <> 0 then
    begin
      if node >= cardinal(fXcdaCount) then
        EInferError.RaiseUtf8('%: charsmap trie index out of range', [self]);
      matched := i + 1;
      best := fXcda^[node] and (cardinal(1 shl 31) - 1);
    end;
  end;
  if matched > 0 then
  begin
    if best >= cardinal(fReplacementsLen) then
      EInferError.RaiseUtf8('%: charsmap replacement out of range', [self]);
    aRepl := fReplacements + best;
    aReplLen := 0;
    while (PtrInt(best) + aReplLen < fReplacementsLen) and (aRepl[aReplLen] <> #0) do
      inc(aReplLen);
    if PtrInt(best) + aReplLen >= fReplacementsLen then
      EInferError.RaiseUtf8('%: unterminated charsmap replacement', [self]);
    result := matched;
    exit;
  end;
  // no rule: one code point unchanged, or U+FFFD for an invalid byte
  result := Utf8SeqLen(pointer(p), aLen, valid);
  if valid then
  begin
    aRepl := p;
    aReplLen := result;
  end
  else
  begin
    aRepl := pointer(REPLACEMENT_CHAR);
    aReplLen := 3;
  end;
end;

function TUgmTokenizer.Normalize(const aText: RawUtf8): RawUtf8;
var
  w: TTextWriter;
  tmp: TTextWriterStackBuffer;
  p, repl: PUtf8Char;
  len, consumed, rlen, i: PtrInt;
  inWord: boolean;
begin
  // SentencePiece flags of this model: add_dummy_prefix, remove_extra_whitespaces,
  // escape_whitespaces - every run of spaces becomes one U+2581 before the next
  // word, leading and trailing spaces vanish
  w := TTextWriter.CreateOwnedStream(tmp);
  try
    p := pointer(aText);
    len := length(aText);
    inWord := false;
    while len > 0 do
    begin
      consumed := NormalizePrefix(p, len, repl, rlen);
      for i := 0 to rlen - 1 do
        if repl[i] <> ' ' then
        begin
          if not inWord then
          begin
            inWord := true;
            w.AddNoJsonEscape(pointer(ESCAPED_SPACE), 3);
          end;
          w.Add(repl[i]);
        end
        else
          inWord := false;
      inc(p, consumed);
      dec(len, consumed);
    end;
    w.SetText(result);
  finally
    w.Free;
  end;
end;

type
  TUgmBest = record
    Id: integer;
    From: PtrInt;
    Score: double;
  end;
  TUgmBestDynArray = array of TUgmBest;

function TUgmTokenizer.Encode(const aText: RawUtf8;
  aAddSpecial: boolean): TIntegerDynArray;
var
  norm: RawUtf8;
  p: PByteArray;
  n, pos, cp, len, maxLen, id, count, i, j: PtrInt;
  best: TUgmBestDynArray;
  score: double;
  valid, wholeCodePoint, prevUnknown, unknown: boolean;
  rev: TIntegerDynArray;
begin
  norm := Normalize(aText);
  n := length(norm);
  p := pointer(norm);
  count := 0;
  if n > 0 then
  begin
    // Viterbi: best[k] = best segmentation of the first k bytes
    SetLength(best, n + 1);
    for i := 1 to n do
    begin
      best[i].Id := fUnkId;
      best[i].From := 0;
      best[i].Score := -1e300;
    end;
    best[0].Id := fUnkId;
    best[0].Score := 0;
    pos := 0;
    while pos < n do
    begin
      cp := Utf8SeqLen(@p^[pos], n - pos, valid); // normalized text is valid UTF-8
      wholeCodePoint := false;
      maxLen := fMaxPieceLen;
      if maxLen > n - pos then
        maxLen := n - pos;
      // every piece starting here, shortest first (the order of a trie walk)
      for len := 1 to maxLen do
      begin
        id := PieceId(@p^[pos], len);
        if id < 0 then
          continue;
        if len = cp then
          wholeCodePoint := true;
        score := best[pos].Score + fScores[id]; // single score, double sum
        if score > best[pos + len].Score then
        begin
          best[pos + len].Id := id;
          best[pos + len].From := pos;
          best[pos + len].Score := score;
        end;
      end;
      // no piece covers this code point: it becomes <unk>
      if not wholeCodePoint then
      begin
        score := best[pos].Score + fUnknownScore;
        if score > best[pos + cp].Score then
        begin
          best[pos + cp].Id := fUnkId;
          best[pos + cp].From := pos;
          best[pos + cp].Score := score;
        end;
      end;
      inc(pos, cp);
    end;
    // backtrack, merging consecutive unknowns into one
    SetLength(rev, n);
    pos := n;
    prevUnknown := false;
    while true do
    begin
      unknown := best[pos].Id = fUnkId;
      if not (prevUnknown and unknown) then
      begin
        rev[count] := best[pos].Id;
        inc(count);
      end;
      if best[pos].From = 0 then
        break;
      prevUnknown := unknown;
      pos := best[pos].From;
    end;
  end;
  j := 0;
  if aAddSpecial then
    SetLength(result, count + 2)
  else
    SetLength(result, count);
  if aAddSpecial then
  begin
    result[0] := fBosId;
    j := 1;
  end;
  for i := count - 1 downto 0 do
  begin
    result[j] := rev[i];
    inc(j);
  end;
  if aAddSpecial then
    result[j] := fEosId;
end;

function TUgmTokenizer.VocabSize: integer;
begin
  result := length(fPieces);
end;

function TUgmTokenizer.Piece(aId: integer): RawUtf8;
begin
  if (aId >= 0) and (aId < length(fPieces)) then
    result := fPieces[aId]
  else
    result := '';
end;


initialization
  FastSetString(ESCAPED_SPACE, @ESCAPED_SPACE_BYTES, 3);
  FastSetString(REPLACEMENT_CHAR, @REPLACEMENT_BYTES, 3);

end.
