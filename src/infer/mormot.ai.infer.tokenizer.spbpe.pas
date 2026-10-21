/// native inference engine - SentencePiece-style BPE tokenizer (Gemma 4)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room implementation of the pipeline a Hugging Face tokenizer.json
//   describes; the reference is the tokenizers library (Apache-2.0, read only
//   for the documented behaviour) and the gate tests/data/infer-ref/
//   gemma4-tokens: every space becomes U+2581 (a Replace normalizer), no
//   pre-tokenizer cut (the Split on " " of Gemma 4 finds no space left after
//   the normalizer), so the whole text is one BPE word; its symbols are the
//   characters of the vocabulary, a character outside it its UTF-8 bytes as
//   the tokens <0xNN> (byte fallback), merged by rank (leftmost first on equal
//   rank); decoding turns U+2581 back into a space and <0xNN> into its byte
// - behaviour = parse_special=false: Encode treats every text as text, also
//   "<|turn>" or "<|tool_call>"; control tokens come only from TokenId (the
//   chat template), never from user text
// - invalid UTF-8 bytes become U+FFFD before normalization
// - thread-safe after loading: Encode uses only local state
unit mormot.ai.infer.tokenizer.spbpe;

{
  *****************************************************************************

    - TSpBpeTokenizer: load the BPE tokenizer.json of Gemma 4 (SentencePiece
      style: U+2581 for a space, byte fallback), Normalize, Encode text to ids
      and Decode ids to bytes

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
  mormot.ai.infer.tokenizer,
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.unicode;

type
  /// a BPE tokenizer in the style of SentencePiece: U+2581 for a space, the
  // whole text one word, byte fallback - the tokenizer of Gemma 4
  TSpBpeTokenizer = class(TInferTokenizer)
  protected
    fTokens: TRawUtf8DynArray;       // id -> text in the vocabulary ('' = none)
    fBytes: TRawByteStringDynArray;  // id -> the bytes it decodes to
    fVocabCount: integer;            // ids below are BPE tokens, above added ones
    fByteId: array[byte] of integer; // the tokens <0x00> .. <0xFF>
    fHashHead, fHashNext: TIntegerDynArray;
    fHashMask: PtrInt;
    fMergeKey: TInt64DynArray;       // open addressing, -1 = empty slot
    fMergeRank, fMergeNew: TIntegerDynArray;
    fMergeMask: PtrInt;
    fMergeCount: integer;
    fAddedText: TRawUtf8DynArray;
    fAddedId: TIntegerDynArray;
    /// the tables from the vocabulary (texts by id), the merges (left and
    // right by rank) and the added tokens
    procedure Build(const aVocab, aLeft, aRight, aAddedText: TRawUtf8DynArray;
      const aAddedId: TIntegerDynArray);
    procedure AddMerge(aLeft, aRight, aRank, aNew: integer);
    function FindMerge(aLeft, aRight: integer; out aRank, aNew: integer): boolean;
    /// the id of a vocabulary text, -1 if none
    function TextId(p: pointer; aLen: PtrInt): integer;
    /// the merges over one normalized text, appended to aIds
    procedure Bpe(p: PByteArray; aLen: PtrInt; var aIds: TIntegerDynArray;
      var aCount: PtrInt; var aSym: TBpeSymbolDynArray; var aQueue: TBpeQueue);
  public
    /// load and validate a Hugging Face tokenizer.json of this pipeline:
    // normalizer Replace(" " -> U+2581); no truncation, no padding; pre_tokenizer none or Split(" ", not
    // inverted) - it finds no space after the normalizer; BPE with
    // byte_fallback (no dropout, no prefix/suffix, ignore_merges off) and the
    // 256 tokens <0x00> .. <0xFF> in the vocabulary (so no character is ever
    // unknown and unk_token / fuse_unk never apply); decoder Sequence[
    // Replace(U+2581 -> " "), ByteFallback, Fuse] - raises EInferError otherwise
    // - the post_processor is not read: Encode never adds special tokens (the
    // chat template writes <bos>, as transformers renders it); added tokens,
    // special or not, are never matched in text (the tokenizers library would
    // match both)
    constructor Create(const aFileName: TFileName); override;
    /// not supported yet (D6d): raises EInferError
    constructor CreateFromGguf(aGguf: TGgufFile); override;
    /// the normalized text: invalid UTF-8 replaced, every space U+2581
    function Normalize(const aText: RawUtf8): RawUtf8;
    /// token ids of aText - every text is text, no control token is matched
    function Encode(const aText: RawUtf8): TIntegerDynArray; override;
    /// append the token ids of aText to aIds[0..aCount-1]
    procedure EncodeAppend(const aText: RawUtf8; var aIds: TIntegerDynArray;
      var aCount: PtrInt); override;
    /// the id of an added token by its content (e.g. '<|turn>'), -1 if none
    function TokenId(const aContent: RawUtf8): integer; override;
    /// the bytes of one token: U+2581 as a space, <0xNN> as its byte (part of
    // a UTF-8 sequence only), '' for an id inside the vocabulary without token
    function TokenBytes(aId: integer): RawByteString; override;
    /// the concatenated bytes of the ids - valid UTF-8 only when no multi-byte
    // character is cut at the ends of the list
    function Decode(const aIds: array of integer): RawUtf8; override;
    /// highest id + 1 (BPE tokens and added tokens)
    function VocabSize: integer; override;
    /// number of BPE tokens (ids 0..BpeCount-1)
    property BpeCount: integer
      read fVocabCount;
    /// number of merges
    property MergeCount: integer
      read fMergeCount;
  end;

var
  /// U+2581 LOWER ONE EIGHTH BLOCK, the space of SentencePiece, as UTF-8
  // - set at unit initialization from its bytes: a #$e2#$96#$81 literal
  // would be three UTF-16 characters on Unicode Delphi
  SP_SPACE: RawUtf8;


implementation

const
  HEX_UPPER: array[0..15] of AnsiChar = '0123456789ABCDEF';

function MergeHash(aKey: Int64): PtrUInt;
  {$ifdef HASINLINE}inline;{$endif}
begin
  // Fibonacci hashing of the (left, right) pair - the low bits are masked
  result := PtrUInt(QWord(aKey) * QWord($9e3779b97f4a7c15) shr 32);
end;

function HexNibble(c: AnsiChar): integer;
begin
  case c of
    '0'..'9':
      result := ord(c) - ord('0');
    'A'..'F':
      result := ord(c) - ord('A') + 10;
    'a'..'f':
      result := ord(c) - ord('a') + 10;
  else
    result := -1;
  end;
end;

/// the bytes a token text decodes to, as the decoder of the pipeline does it:
// U+2581 -> space, then a text "<0xNN>" (hex in either case) -> that byte
function TokenTextBytes(const aText: RawUtf8): RawByteString;
var
  hi, lo: integer;
begin
  if (length(aText) = 6) and (aText[1] = '<') and (aText[2] = '0') and
     (aText[3] = 'x') and (aText[6] = '>') then
  begin
    hi := HexNibble(aText[4]);
    lo := HexNibble(aText[5]);
    if (hi >= 0) and (lo >= 0) then
    begin
      SetLength(result, 1);
      PByteArray(result)^[0] := hi shl 4 + lo;
      exit;
    end;
  end;
  result := StringReplaceAll(aText, SP_SPACE, ' ');
end;

function IsReplace(aDoc: PDocVariantData; const aFrom, aTo: RawUtf8): boolean;
begin
  result := (aDoc^.U['type'] = 'Replace') and
    (aDoc^.O['pattern']^.Count = 1) and
    (aDoc^.O['pattern']^.U['String'] = aFrom) and (aDoc^.U['content'] = aTo);
end;


{ TSpBpeTokenizer }

constructor TSpBpeTokenizer.Create(const aFileName: TFileName);
var
  doc: TDocVariantData;
  norm, pre, model, vocab, merges, pair, added, entry, decoder,
  list: PDocVariantData;
  i, id: PtrInt;
  texts, a, b, addedText: TRawUtf8DynArray;
  addedId: TIntegerDynArray;
  text: RawUtf8;
begin
  inherited Create(aFileName);
  if not doc.InitJsonFromFile(aFileName, JSON_FAST_FLOAT) then
    EInferError.RaiseUtf8('%: cannot read %', [self, aFileName]);
  // the pipeline this class implements, and nothing else
  if not doc.GetAsObject('normalizer', norm) or
     not IsReplace(norm, ' ', SP_SPACE) then
    EInferError.RaiseUtf8('%: normalizer is not Replace(" ", U+2581)', [self]);
  // a saved truncation or padding would change what Encode returns
  if not VarIsEmptyOrNull(doc.Value['truncation']) or
     not VarIsEmptyOrNull(doc.Value['padding']) then
    EInferError.RaiseUtf8('%: truncation or padding is set', [self]);
  // a Split on " " has nothing to cut once every space is U+2581
  if doc.GetAsObject('pre_tokenizer', pre) then
  begin
    if (pre^.U['type'] <> 'Split') or (pre^.O['pattern']^.Count <> 1) or
       (pre^.O['pattern']^.U['String'] <> ' ') or pre^.B['invert'] then
      EInferError.RaiseUtf8('%: pre_tokenizer is neither none nor Split(" ")',
        [self]);
  end
  else if not VarIsEmptyOrNull(doc.Value['pre_tokenizer']) then
    EInferError.RaiseUtf8('%: pre_tokenizer is neither none nor Split(" ")', [self]);
  if not doc.GetAsObject('decoder', decoder) or
     (decoder^.U['type'] <> 'Sequence') or
     not decoder^.GetAsArray('decoders', list) or (list^.Count <> 3) or
     not IsReplace(_Safe(list^.Values[0]), SP_SPACE, ' ') or
     (_Safe(list^.Values[1])^.U['type'] <> 'ByteFallback') or
     (_Safe(list^.Values[2])^.U['type'] <> 'Fuse') then
    EInferError.RaiseUtf8('%: decoder is not Sequence[Replace(U+2581, " "), ' +
      'ByteFallback, Fuse]', [self]);
  if not doc.GetAsObject('model', model) or (model^.U['type'] <> 'BPE') or
     not VarIsEmptyOrNull(model^.Value['dropout']) or
     (model^.U['continuing_subword_prefix'] <> '') or
     (model^.U['end_of_word_suffix'] <> '') or
     not model^.B['byte_fallback'] or model^.B['ignore_merges'] then
    EInferError.RaiseUtf8('%: model is not a BPE with byte fallback', [self]);
  // GetAsArray is false for an empty array: no merges is a valid BPE
  if not model^.GetAsObject('vocab', vocab) or (vocab^.Count = 0) or
     not model^.GetAsDocVariant('merges', merges) or not merges^.IsArray then
    EInferError.RaiseUtf8('%: BPE without vocab or merges', [self]);
  SetLength(texts, vocab^.Count);
  for i := 0 to vocab^.Count - 1 do
  begin
    id := VariantToIntegerDef(vocab^.Values[i], -1);
    if (id < 0) or (id >= vocab^.Count) or (texts[id] <> '') then
      EInferError.RaiseUtf8('%: vocab id % of "%" invalid or used twice',
        [self, id, vocab^.Names[i]]);
    texts[id] := vocab^.Names[i];
  end;
  // the merges: ["a", "b"] (current format) or "a b" (older files; a token
  // text holds no space here, U+2581 stands for it)
  SetLength(a, merges^.Count);
  SetLength(b, merges^.Count);
  for i := 0 to merges^.Count - 1 do
    if _Safe(merges^.Values[i])^.IsArray then
    begin
      pair := _Safe(merges^.Values[i]);
      if pair^.Count <> 2 then
        EInferError.RaiseUtf8('%: merge % is not a pair', [self, i]);
      VariantToUtf8(pair^.Values[0], a[i]);
      VariantToUtf8(pair^.Values[1], b[i]);
    end
    else
    begin
      VariantToUtf8(merges^.Values[i], text);
      if not Split(text, ' ', a[i], b[i]) or (a[i] = '') or (b[i] = '') or
         (PosEx(' ', b[i]) > 0) then
        EInferError.RaiseUtf8('%: merge % is not "a b"', [self, i]);
    end;
  // added tokens: absent, or an array
  added := nil;
  if doc.GetValueIndex('added_tokens') >= 0 then
    if not doc.GetAsDocVariant('added_tokens', added) or not added^.IsArray then
      EInferError.RaiseUtf8('%: added_tokens is not an array', [self]);
  if added <> nil then
  begin
    SetLength(addedText, added^.Count);
    SetLength(addedId, added^.Count);
    for i := 0 to added^.Count - 1 do
    begin
      entry := _Safe(added^.Values[i]);
      addedId[i] := entry^.I['id'];
      addedText[i] := entry^.U['content'];
    end;
  end;
  Build(texts, a, b, addedText, addedId);
end;

constructor TSpBpeTokenizer.CreateFromGguf(aGguf: TGgufFile);
begin
  inherited CreateFromGguf(aGguf);
  EInferError.RaiseUtf8('%: the tokenizer of a GGUF file is not supported yet',
    [self]);
end;

procedure TSpBpeTokenizer.Build(const aVocab, aLeft, aRight,
  aAddedText: TRawUtf8DynArray; const aAddedId: TIntegerDynArray);
var
  i, n, id, maxId, h: PtrInt;
  text: RawUtf8;
  l, r, m, oldRank, oldNew: integer;
begin
  n := length(aVocab);
  if n = 0 then
    EInferError.RaiseUtf8('%: BPE without vocab', [self]);
  fVocabCount := n;
  maxId := n - 1;
  for i := 0 to high(aAddedId) do
  begin
    id := aAddedId[i];
    if (id < 0) or (id > n + 1 shl 20) then
      EInferError.RaiseUtf8('%: added token id % out of range', [self, id]);
    if id > maxId then
      maxId := id;
  end;
  SetLength(fTokens, maxId + 1);
  SetLength(fBytes, maxId + 1);
  for i := 0 to n - 1 do
  begin
    if aVocab[i] = '' then
      EInferError.RaiseUtf8('%: vocab id % has no text', [self, i]);
    fTokens[i] := aVocab[i];
  end;
  // hash of the texts, chained by id: lookups allocate nothing
  fHashMask := 1;
  while fHashMask < n * 2 do
    fHashMask := fHashMask shl 1;
  dec(fHashMask);
  SetLength(fHashHead, fHashMask + 1);
  FillCharFast(pointer(fHashHead)^, length(fHashHead) * SizeOf(integer), $ff); // -1
  SetLength(fHashNext, n);
  for i := 0 to n - 1 do
  begin
    // a text twice: the tokenizers library keeps the last id, refuse instead
    if TextId(pointer(fTokens[i]), length(fTokens[i])) >= 0 then
      EInferError.RaiseUtf8('%: vocab text "%" has two ids', [self, fTokens[i]]);
    h := crc32c(0, pointer(fTokens[i]), length(fTokens[i])) and fHashMask;
    fHashNext[i] := fHashHead[h];
    fHashHead[h] := i;
  end;
  // byte fallback: the tokenizers library looks up "<0x%02X>"
  for i := 0 to 255 do
  begin
    text := '<0x' + RawUtf8(HEX_UPPER[i shr 4]) + RawUtf8(HEX_UPPER[i and 15]) + '>';
    fByteId[i] := TextId(pointer(text), length(text));
    if fByteId[i] < 0 then
      EInferError.RaiseUtf8('%: no token % for byte fallback', [self, text]);
  end;
  // the merges: (left, right) -> rank and merged id
  fMergeCount := length(aLeft);
  fMergeMask := 1;
  while fMergeMask < fMergeCount * 2 + 2 do
    fMergeMask := fMergeMask shl 1;
  dec(fMergeMask);
  SetLength(fMergeKey, fMergeMask + 1);
  FillCharFast(pointer(fMergeKey)^, length(fMergeKey) * SizeOf(Int64), $ff); // -1
  SetLength(fMergeRank, fMergeMask + 1);
  SetLength(fMergeNew, fMergeMask + 1);
  for i := 0 to fMergeCount - 1 do
  begin
    l := TextId(pointer(aLeft[i]), length(aLeft[i]));
    r := TextId(pointer(aRight[i]), length(aRight[i]));
    text := aLeft[i] + aRight[i];
    m := TextId(pointer(text), length(text));
    if (l < 0) or (r < 0) or (m < 0) then
      EInferError.RaiseUtf8('%: merge % "% %" outside the vocabulary',
        [self, i, aLeft[i], aRight[i]]);
    if FindMerge(l, r, oldRank, oldNew) then
      // the tokenizers library would keep the last rank: refuse instead
      EInferError.RaiseUtf8('%: merge % "% %" repeats merge %',
        [self, i, aLeft[i], aRight[i], oldRank]);
    AddMerge(l, r, i, m);
  end;
  // added tokens: never matched in text, reachable by TokenId
  SetLength(fAddedText, length(aAddedId));
  SetLength(fAddedId, length(aAddedId));
  for i := 0 to high(aAddedId) do
  begin
    id := aAddedId[i];
    if (aAddedText[i] = '') or
       ((id < n) and (fTokens[id] <> aAddedText[i])) or
       ((id >= n) and (fTokens[id] <> '')) then
      EInferError.RaiseUtf8('%: added token % "%" clashes with the vocabulary',
        [self, id, aAddedText[i]]);
    fTokens[id] := aAddedText[i];
    fAddedText[i] := aAddedText[i];
    fAddedId[i] := id;
  end;
  // what each id decodes to, added tokens through the same decoder
  for i := 0 to maxId do
    if fTokens[i] <> '' then
      fBytes[i] := TokenTextBytes(fTokens[i]);
end;

procedure TSpBpeTokenizer.AddMerge(aLeft, aRight, aRank, aNew: integer);
var
  key: Int64;
  h: PtrInt;
begin
  key := (Int64(aLeft) shl 32) or cardinal(aRight);
  h := MergeHash(key) and fMergeMask;
  while fMergeKey[h] <> -1 do
    h := (h + 1) and fMergeMask;
  fMergeKey[h] := key;
  fMergeRank[h] := aRank;
  fMergeNew[h] := aNew;
end;

function TSpBpeTokenizer.FindMerge(aLeft, aRight: integer;
  out aRank, aNew: integer): boolean;
var
  key: Int64;
  h: PtrInt;
begin
  key := (Int64(aLeft) shl 32) or cardinal(aRight);
  h := MergeHash(key) and fMergeMask;
  repeat
    if fMergeKey[h] = key then
    begin
      aRank := fMergeRank[h];
      aNew := fMergeNew[h];
      result := true;
      exit;
    end;
    if fMergeKey[h] = -1 then
    begin
      result := false;
      exit;
    end;
    h := (h + 1) and fMergeMask;
  until false;
end;

function TSpBpeTokenizer.TextId(p: pointer; aLen: PtrInt): integer;
begin
  result := -1;
  if aLen <= 0 then
    exit;
  result := fHashHead[crc32c(0, p, aLen) and fHashMask];
  while result >= 0 do
  begin
    if (length(fTokens[result]) = aLen) and
       CompareMem(pointer(fTokens[result]), p, aLen) then
      exit;
    result := fHashNext[result];
  end;
end;

procedure TSpBpeTokenizer.Bpe(p: PByteArray; aLen: PtrInt;
  var aIds: TIntegerDynArray; var aCount: PtrInt; var aSym: TBpeSymbolDynArray;
  var aQueue: TBpeQueue);
var
  i, k, len, count, pos, next, prev, rank, newId, cur: integer;
begin
  if aLen <= 0 then
    exit;
  // one symbol per character of the vocabulary, else one per byte of it
  if length(aSym) < aLen then
    SetLength(aSym, aLen);
  count := 0;
  i := 0;
  while i < aLen do
  begin
    // the length of the character: Normalize wrote valid UTF-8
    if p^[i] < $c0 then
      len := 1
    else if p^[i] < $e0 then
      len := 2
    else if p^[i] < $f0 then
      len := 3
    else
      len := 4;
    if i + len > aLen then
      len := aLen - i;
    newId := TextId(@p^[i], len);
    if newId >= 0 then
    begin
      aSym[count].Id := newId;
      inc(count);
    end
    else
      for k := 0 to len - 1 do
      begin
        aSym[count].Id := fByteId[p^[i + k]];
        inc(count);
      end;
    inc(i, len);
  end;
  for i := 0 to count - 1 do
  begin
    aSym[i].Prev := i - 1;
    aSym[i].Next := i + 1;
    aSym[i].Len := 1;
  end;
  aSym[count - 1].Next := -1;
  // the merges of the tokenizers library: lowest rank first, leftmost first
  // on equal rank; a queue entry whose pair no longer produces its merged id
  // is stale and skipped
  aQueue.Count := 0;
  for i := 0 to count - 2 do
    if FindMerge(aSym[i].Id, aSym[i + 1].Id, rank, newId) then
      aQueue.Push(rank, i, newId);
  while aQueue.Pop(pos, newId) do
  begin
    if (aSym[pos].Len = 0) or (aSym[pos].Next < 0) then
      continue;
    next := aSym[pos].Next;
    if not FindMerge(aSym[pos].Id, aSym[next].Id, rank, cur) or (cur <> newId) then
      continue;
    // merge the right symbol into this one
    aSym[pos].Id := newId;
    inc(aSym[pos].Len, aSym[next].Len);
    aSym[pos].Next := aSym[next].Next;
    aSym[next].Len := 0;
    if aSym[pos].Next >= 0 then
      aSym[aSym[pos].Next].Prev := pos;
    prev := aSym[pos].Prev;
    if (prev >= 0) and FindMerge(aSym[prev].Id, aSym[pos].Id, rank, cur) then
      aQueue.Push(rank, prev, cur);
    next := aSym[pos].Next;
    if (next >= 0) and FindMerge(aSym[pos].Id, aSym[next].Id, rank, cur) then
      aQueue.Push(rank, pos, cur);
  end;
  i := 0;
  while i >= 0 do
  begin
    if aCount = length(aIds) then
      SetLength(aIds, aCount * 2 + 16);
    aIds[aCount] := aSym[i].Id;
    inc(aCount);
    i := aSym[i].Next;
  end;
end;

function TSpBpeTokenizer.Normalize(const aText: RawUtf8): RawUtf8;
var
  cp: TCardinalDynArray;
  n, i: PtrInt;
begin
  n := Utf8ToCodePoints(pointer(aText), length(aText), cp);
  for i := 0 to n - 1 do
    if cp[i] = 32 then
      cp[i] := $2581;
  result := CodePointsToUtf8(pointer(cp), n);
end;

procedure TSpBpeTokenizer.EncodeAppend(const aText: RawUtf8;
  var aIds: TIntegerDynArray; var aCount: PtrInt);
var
  text: RawUtf8;
  sym: TBpeSymbolDynArray;
  queue: TBpeQueue;
begin
  queue.Count := 0;
  text := Normalize(aText);
  Bpe(pointer(text), length(text), aIds, aCount, sym, queue);
end;

function TSpBpeTokenizer.Encode(const aText: RawUtf8): TIntegerDynArray;
var
  n: PtrInt;
begin
  result := nil;
  n := 0;
  EncodeAppend(aText, result, n);
  SetLength(result, n);
end;

function TSpBpeTokenizer.TokenId(const aContent: RawUtf8): integer;
var
  i: PtrInt;
begin
  for i := 0 to high(fAddedText) do
    if fAddedText[i] = aContent then
    begin
      result := fAddedId[i];
      exit;
    end;
  result := -1;
end;

function TSpBpeTokenizer.TokenBytes(aId: integer): RawByteString;
begin
  if (aId < 0) or (aId > high(fBytes)) then
    EInferError.RaiseUtf8('%.TokenBytes: id % outside 0..%',
      [self, aId, high(fBytes)]);
  result := fBytes[aId];
end;

function TSpBpeTokenizer.Decode(const aIds: array of integer): RawUtf8;
var
  i, n: PtrInt;
begin
  n := 0;
  for i := 0 to high(aIds) do
    inc(n, length(TokenBytes(aIds[i])));
  FastSetString(result, n);
  n := 0;
  for i := 0 to high(aIds) do
  begin
    MoveFast(pointer(fBytes[aIds[i]])^, PByteArray(result)^[n], length(fBytes[aIds[i]]));
    inc(n, length(fBytes[aIds[i]]));
  end;
end;

function TSpBpeTokenizer.VocabSize: integer;
begin
  result := length(fTokens);
end;

procedure InitSpSpace;
const
  BYTES: array[0..2] of byte = ($e2, $96, $81);
begin
  FastSetString(SP_SPACE, @BYTES, 3);
end;


initialization
  InitSpSpace;

end.
