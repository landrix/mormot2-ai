/// native inference engine - byte-level BPE tokenizer (Qwen2 / Qwen3)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room implementation of the pipeline a Hugging Face tokenizer.json
//   describes; the reference is the tokenizers library (Apache-2.0, read only
//   for the documented behaviour) and the gate tests/data/infer-ref/qwen3-tokens:
//   NFC normalizer, the Qwen2 Split regex (isolated), ByteLevel without prefix
//   space, BPE merges by rank (leftmost first on equal rank)
// - the split regex is not run by a regex engine: TBpeTokenizer.Split is the
//   leftmost-first match of its seven alternatives, written out by hand
//     (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
//      ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//   with the code point classes of mormot.ai.infer.unicode (measured against
//   the regex of the tokenizers library, also the case folding of (?i))
// - behaviour = parse_special=false: Encode treats every text as text, also
//   "<|im_end|>" or "<think>"; control tokens come only from TokenId (the chat
//   template), never from user text
// - invalid UTF-8 bytes become U+FFFD before normalization
// - thread-safe after loading: Encode uses only local state
unit mormot.ai.infer.tokenizer.bpe;

{
  *****************************************************************************

    - TBpeTokenizer: load a byte-level BPE tokenizer.json, the older pair
      vocab.json + merges.txt or the tokenizer of a GGUF file (Qwen2 pipeline),
      Normalize, Split, Encode text to ids and Decode ids to bytes

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

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
  mormot.ai.infer.unicode;

type
  /// one symbol of a word being merged
  TBpeSymbol = record
    Id: integer;
    Prev, Next: integer; // -1 at the ends
    Len: integer;        // 0 = merged into its left neighbour
  end;
  TBpeSymbolDynArray = array of TBpeSymbol;

  /// a min-heap of candidate merges: rank first, then position
  TBpeQueue = object
    Key: TInt64DynArray; // rank shl 32 + position
    New: TIntegerDynArray;
    Count: PtrInt;
    procedure Push(aRank, aPos, aNew: integer);
    function Pop(out aPos, aNew: integer): boolean;
  end;

  /// a byte-level BPE tokenizer with the Qwen2 pre-tokenizer
  TBpeTokenizer = class(TInferTokenizer)
  protected
    fTokens: TRawByteStringDynArray; // id -> bytes ('' = no token)
    fVocabCount: integer;            // ids below are BPE tokens, above added ones
    fByteId: array[byte] of integer;
    fHashHead, fHashNext: TIntegerDynArray;
    fHashMask: PtrInt;
    fMergeKey: TInt64DynArray;       // open addressing, -1 = empty slot
    fMergeRank, fMergeNew: TIntegerDynArray;
    fMergeMask: PtrInt;
    fMergeCount: integer;
    fAddedText: TRawUtf8DynArray;
    fAddedId: TIntegerDynArray;
    /// the tables from the vocabulary (byte-level texts by id), the merges
    // (left and right by rank) and the added tokens
    procedure Build(const aVocab, aLeft, aRight, aAddedText: TRawUtf8DynArray;
      const aAddedId: TIntegerDynArray);
    /// the texts by id of a {"text": id} object: ids 0..n-1, each exactly once
    function VocabTexts(aVocab: PDocVariantData): TRawUtf8DynArray;
    /// "a b" (one space) split into a and b, raising otherwise
    procedure SplitMerge(const aText: RawUtf8; aIndex: PtrInt;
      out aLeft, aRight: RawUtf8);
    /// raise unless every special token aDoc names (any "..._token" key with a
    // string or {"content": ...}, extra_special_tokens,
    // additional_special_tokens, model_specific_special_tokens) is an added
    // token (see CheckToken)
    procedure CheckSpecialTokens(aDoc: PDocVariantData; const aFile: TFileName);
    /// raise unless aText is an added token
    procedure CheckToken(const aText, aKey: RawUtf8; const aFile: TFileName);
    /// true when aText is a key of vocab.json (byte-level text)
    function InVocab(const aText: RawUtf8): boolean;
    procedure AddMerge(aLeft, aRight, aRank, aNew: integer);
    function FindMerge(aLeft, aRight: integer; out aRank, aNew: integer): boolean;
    function BytesId(p: pointer; aLen: PtrInt): integer;
    procedure Bpe(p: PByteArray; aLen: PtrInt; var aIds: TIntegerDynArray;
      var aCount: PtrInt; var aSym: TBpeSymbolDynArray; var aQueue: TBpeQueue);
  public
    /// load and validate a Hugging Face tokenizer.json of this pipeline:
    // NFC, Split(Qwen2 regex, Isolated) + ByteLevel(no prefix space, no regex),
    // BPE (no dropout, no unk, no prefix/suffix, no byte fallback), ByteLevel
    // decoder - raises EInferError otherwise
    constructor Create(const aFileName: TFileName); override;
    /// the same tokenizer from the tokenizer.ggml.* metadata of a GGUF file:
    // model gpt2, pre-tokenizer qwen2; normal tokens are the BPE vocabulary,
    // control and user-defined ones the added tokens, unused ones no token
    // - GGUF stores no normalizer: NFC is applied, as Qwen's tokenizer.json has
    // it (llama.cpp itself does not normalize)
    constructor CreateFromGguf(aGguf: TGgufFile); override;
    /// the same tokenizer from the older pair vocab.json + merges.txt, the files
    // of transformers' Qwen2Tokenizer; the pipeline is not in them, so the one
    // of this class is assumed (NFC, Qwen2 Split, ByteLevel) and checked where
    // the folder of aVocabFile tells: a tokenizer_config.json there must name
    // Qwen2Tokenizer(Fast) and no prefix space
    // - the added tokens come, as in transformers 5.19, from the
    // added_tokens_decoder of that tokenizer_config.json, or without one (older
    // folders) from an added_tokens.json beside it; a special token named in
    // the config (or, without a decoder, in special_tokens_map.json) must be an
    // added token - transformers would append a missing one with a new id, and
    // would match one of the vocabulary as a whole in text, which Encode does
    // not; every name in these files is checked, without
    // transformers' rules of which one wins, so a folder whose files
    // disagree may be refused although transformers loads it, but none is
    // accepted whose ids differ; as with tokenizer.json, an added token is
    // never matched in text (see Encode)
    // - as transformers assigns them: the added ids must follow the
    // vocabulary without a gap, each content new; an absent unk_token,
    // eos_token or pad_token is Qwen2Tokenizer's default <|endoftext|>, which
    // must then exist
    // - refused, because transformers would then read something else or add
    // tokens Encode does not: a tokenizer.json (use Create on it),
    // tokenizer.model*, tiktoken.model or tekken.json beside; in the config
    // fast_tokenizer_files, an inline vocab or merges, add_bos_token or
    // add_eos_token true; two added tokens with the same content
    // - merges.txt as the tokenizers library 0.23 reads it for transformers
    // 5.19: one "a b" per line, LF or CRLF, every line starting with
    // "#version" skipped; an empty line or one with more spaces is refused
    constructor CreateFromVocabMerges(const aVocabFile, aMergesFile: TFileName);
    /// the NFC-normalized text (invalid UTF-8 replaced first)
    function Normalize(const aText: RawUtf8): RawUtf8;
    /// the pre-tokenized pieces of the normalized text, as UTF-8
    function Split(const aText: RawUtf8): TRawUtf8DynArray;
    /// token ids of aText - every text is text, no control token is matched
    function Encode(const aText: RawUtf8): TIntegerDynArray; override;
    /// append the token ids of aText to aIds[0..aCount-1]
    procedure EncodeAppend(const aText: RawUtf8; var aIds: TIntegerDynArray;
      var aCount: PtrInt); override;
    /// the id of an added token by its content (e.g. '<|im_end|>'), -1 if none
    function TokenId(const aContent: RawUtf8): integer; override;
    /// the bytes of one token: '' for an id inside the vocabulary that has no
    // token; a BPE token may be part of a UTF-8 sequence only
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


const
  /// the Split regex of Qwen2 / Qwen3 (tokenizer.json pre_tokenizer), verbatim
  QWEN2_SPLIT_REGEX =
    '(?i:''s|''t|''re|''ve|''m|''ll|''d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|' +
    ' ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+';


implementation

var
  /// GPT-2 byte-level alphabet: byte -> code point, and back (-1 = none)
  ByteToCp: array[byte] of word;
  CpToByte: array[0..511] of smallint;

procedure InitByteLevel;
var
  b, n: PtrInt;
begin
  // printable bytes map to themselves, the others to 256.. in byte order
  for b := 0 to high(CpToByte) do
    CpToByte[b] := -1;
  n := 0;
  for b := 0 to 255 do
  begin
    if ((b >= ord('!')) and (b <= ord('~'))) or ((b >= $a1) and (b <= $ac)) or
       ((b >= $ae) and (b <= $ff)) then
      ByteToCp[b] := b
    else
    begin
      ByteToCp[b] := 256 + n;
      inc(n);
    end;
    CpToByte[ByteToCp[b]] := b;
  end;
end;

/// the raw bytes of a byte-level token text, false if it has a foreign char
function ByteLevelDecode(const aText: RawUtf8; out aBytes: RawByteString): boolean;
var
  cp: TCardinalDynArray;
  n, i: PtrInt;
begin
  result := false;
  n := Utf8ToCodePoints(pointer(aText), length(aText), cp);
  SetLength(aBytes, n);
  for i := 0 to n - 1 do
  begin
    if (cp[i] > high(CpToByte)) or (CpToByte[cp[i]] < 0) then
      exit;
    PByteArray(aBytes)^[i] := CpToByte[cp[i]];
  end;
  result := n > 0;
end;

function MergeHash(aKey: Int64): PtrUInt;
  {$ifdef HASINLINE}inline;{$endif}
begin
  // Fibonacci hashing of the (left, right) pair - the low bits are masked
  result := PtrUInt(QWord(aKey) * QWord($9e3779b97f4a7c15) shr 32);
end;


{ TBpeTokenizer }

constructor TBpeTokenizer.Create(const aFileName: TFileName);
var
  doc: TDocVariantData;
  norm, pre, list, split, level, model, vocab, merges, entry, added,
  decoder: PDocVariantData;
  pair: PDocVariantData;
  i: PtrInt;
  texts, a, b, addedText: TRawUtf8DynArray;
  addedId: TIntegerDynArray;
  text: RawUtf8;
begin
  inherited Create(aFileName);
  if not doc.InitJsonFromFile(aFileName, JSON_FAST_FLOAT) then
    EInferError.RaiseUtf8('%: cannot read %', [self, aFileName]);
  // the pipeline this class implements, and nothing else
  if not doc.GetAsObject('normalizer', norm) or (norm^.U['type'] <> 'NFC') then
    EInferError.RaiseUtf8('%: normalizer is not NFC', [self]);
  if not doc.GetAsObject('pre_tokenizer', pre) or (pre^.U['type'] <> 'Sequence') or
     not pre^.GetAsArray('pretokenizers', list) or (list^.Count <> 2) then
    EInferError.RaiseUtf8('%: pre_tokenizer is not Sequence[Split, ByteLevel]', [self]);
  split := _Safe(list^.Values[0]);
  if (split^.U['type'] <> 'Split') or (split^.U['behavior'] <> 'Isolated') or
     split^.B['invert'] or (split^.O['pattern']^.U['Regex'] <> QWEN2_SPLIT_REGEX) then
    EInferError.RaiseUtf8('%: first pre-tokenizer is not the Qwen2 Split', [self]);
  level := _Safe(list^.Values[1]);
  if (level^.U['type'] <> 'ByteLevel') or level^.B['add_prefix_space'] or
     level^.B['use_regex'] then
    EInferError.RaiseUtf8('%: second pre-tokenizer is not ByteLevel(no prefix, ' +
      'no regex)', [self]);
  if not doc.GetAsObject('decoder', decoder) or (decoder^.U['type'] <> 'ByteLevel') then
    EInferError.RaiseUtf8('%: decoder is not ByteLevel', [self]);
  if not doc.GetAsObject('model', model) or (model^.U['type'] <> 'BPE') or
     not VarIsEmptyOrNull(model^.Value['dropout']) or
     not VarIsEmptyOrNull(model^.Value['unk_token']) or
     (model^.U['continuing_subword_prefix'] <> '') or
     (model^.U['end_of_word_suffix'] <> '') or model^.B['fuse_unk'] or
     model^.B['byte_fallback'] or model^.B['ignore_merges'] then
    EInferError.RaiseUtf8('%: model is not a plain BPE', [self]);
  if not model^.GetAsObject('vocab', vocab) or (vocab^.Count = 0) or
     not model^.GetAsArray('merges', merges) then
    EInferError.RaiseUtf8('%: BPE without vocab or merges', [self]);
  texts := VocabTexts(vocab);
  // the merges: ["a", "b"] (current format) or "a b" (older files)
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
      SplitMerge(text, i, a[i], b[i]);
    end;
  // added tokens: absent, or an array - anything else is no file of ours
  // (GetAsArray is false for an empty array too, hence GetAsDocVariant)
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

constructor TBpeTokenizer.CreateFromGguf(aGguf: TGgufFile);
var
  tokens, merges, a, b, texts, addedText: TRawUtf8DynArray;
  addedId: TIntegerDynArray;
  types: PIntegerArray;
  count: Int64;
  i, n: PtrInt;
begin
  inherited CreateFromGguf(aGguf);
  // llama.cpp's names of the Qwen2 pipeline: a GPT-2 byte-level BPE with the
  // qwen2 pre-tokenizer (the Split regex of tokenizer.json); GGUF stores no
  // normalizer - Qwen's tokenizer.json has NFC, which Normalize applies
  if aGguf.Str('tokenizer.ggml.model') <> 'gpt2' then
    EInferError.RaiseUtf8('%: tokenizer model "%" is no byte-level BPE',
      [self, aGguf.Str('tokenizer.ggml.model')]);
  if aGguf.Str('tokenizer.ggml.pre') <> 'qwen2' then
    EInferError.RaiseUtf8('%: pre-tokenizer "%" is not qwen2',
      [self, aGguf.Str('tokenizer.ggml.pre')]);
  tokens := aGguf.Strings('tokenizer.ggml.tokens');
  count := length(tokens);
  types := aGguf.Ints32('tokenizer.ggml.token_type', count);
  merges := aGguf.Strings('tokenizer.ggml.merges');
  // normal tokens (1) first: the BPE vocabulary; then control (3) and
  // user-defined (4) tokens, the added ones; unused padding (5) is no token
  n := 0;
  while (n < length(tokens)) and (types^[n] = 1) do
    inc(n);
  if n = 0 then
    EInferError.RaiseUtf8('%: no normal token', [self]);
  texts := copy(tokens, 0, n);
  for i := n to high(tokens) do
    case types^[i] of
      3, 4:
        begin
          SetLength(addedText, length(addedText) + 1);
          SetLength(addedId, length(addedId) + 1);
          addedText[high(addedText)] := tokens[i];
          addedId[high(addedId)] := i;
        end;
      5:
        ; // unused: a padding row of the embedding, never produced
    else
      EInferError.RaiseUtf8('%: token % "%" of type % after the vocabulary',
        [self, i, tokens[i], types^[i]]);
    end;
  SetLength(a, length(merges));
  SetLength(b, length(merges));
  for i := 0 to high(merges) do
    SplitMerge(merges[i], i, a[i], b[i]);
  Build(texts, a, b, addedText, addedId);
end;

const
  /// the special tokens Qwen2Tokenizer sets to <|endoftext|> when none is given
  DEFAULTED: array[0..2] of RawUtf8 = ('unk_token', 'eos_token', 'pad_token');
  /// config keys that make transformers read another vocabulary
  REFUSED_KEYS: array[0..2] of RawUtf8 = ('fast_tokenizer_files', 'vocab', 'merges');

constructor TBpeTokenizer.CreateFromVocabMerges(const aVocabFile, aMergesFile: TFileName);
var
  doc, cfg, map: TDocVariantData;
  decoder: PDocVariantData;
  dir, fn: TFileName;
  texts, a, b, addedText: TRawUtf8DynArray;
  addedId: TIntegerDynArray;
  merges, line: RawUtf8;
  p, e, n, i: PtrInt;
  hasConfig, hasMap, known: boolean;
  sr: TSearchRec;
begin
  inherited Create(aVocabFile);
  hasConfig := false;
  decoder := nil;
  if not doc.InitJsonFromFile(aVocabFile, JSON_FAST_FLOAT) or not doc.IsObject or
     (doc.Count = 0) then
    EInferError.RaiseUtf8('%: % is no vocabulary', [self, aVocabFile]);
  texts := VocabTexts(@doc);
  // the merges, one per line
  merges := StringFromFile(aMergesFile);
  if merges = '' then
    EInferError.RaiseUtf8('%: cannot read %', [self, aMergesFile]);
  n := 0;
  p := 1;
  while p <= length(merges) do
  begin
    e := PosEx(#10, merges, p);
    if e = 0 then
      e := length(merges) + 1;
    line := copy(merges, p, e - p);
    if (line <> '') and (line[length(line)] = #13) then
      SetLength(line, length(line) - 1);
    if copy(line, 1, 8) <> '#version' then
    begin
      if n = length(a) then
      begin
        SetLength(a, NextGrow(n));
        SetLength(b, length(a));
      end;
      SplitMerge(line, n, a[n], b[n]);
      inc(n);
    end;
    p := e + 1;
  end;
  SetLength(a, n);
  SetLength(b, n);
  // what the folder tells about the pipeline, and the added tokens
  dir := ExtractFilePath(aVocabFile);
  if FileExists(dir + 'tokenizer.json') then
    EInferError.RaiseUtf8('%: % has a tokenizer.json, which transformers ' +
      'reads instead - use Create on it', [self, dir]);
  // what transformers picks without a tokenizer.json: a name containing
  // tokenizer.model, tiktoken.model or tekken.json
  if FindFirst(dir + '*', faAnyFile, sr) = 0 then
  try
    repeat
      if (Pos('tokenizer.model', sr.Name) > 0) or (Pos('tiktoken.model', sr.Name) > 0) or
         (Pos('tekken.json', sr.Name) > 0) then
        EInferError.RaiseUtf8('%: % has %, which transformers may read instead',
          [self, dir, sr.Name]);
    until FindNext(sr) <> 0;
  finally
    FindClose(sr);
  end;
  fn := dir + 'tokenizer_config.json';
  if FileExists(fn) then
  begin
    if not cfg.InitJsonFromFile(fn, JSON_FAST_FLOAT) or not cfg.IsObject then
      EInferError.RaiseUtf8('%: cannot read %', [self, fn]);
    hasConfig := true;
    if (cfg.U['tokenizer_class'] <> 'Qwen2Tokenizer') and
       (cfg.U['tokenizer_class'] <> 'Qwen2TokenizerFast') then
      EInferError.RaiseUtf8('%: tokenizer_class "%" is not Qwen2Tokenizer',
        [self, cfg.U['tokenizer_class']]);
    if cfg.B['add_prefix_space'] then
      EInferError.RaiseUtf8('%: add_prefix_space is set', [self]);
    for i := 0 to high(REFUSED_KEYS) do
      if cfg.GetValueIndex(REFUSED_KEYS[i]) >= 0 then
        EInferError.RaiseUtf8('%: % is set in the config', [self, REFUSED_KEYS[i]]);
    if cfg.B['add_bos_token'] or cfg.B['add_eos_token'] then
      EInferError.RaiseUtf8('%: add_bos_token or add_eos_token is set', [self]);
    if cfg.GetValueIndex('added_tokens_decoder') >= 0 then
      if not cfg.GetAsDocVariant('added_tokens_decoder', decoder) or
         not decoder^.IsObject then
        EInferError.RaiseUtf8('%: added_tokens_decoder is not an object', [self]);
    if decoder <> nil then
    begin
      SetLength(addedText, decoder^.Count);
      SetLength(addedId, decoder^.Count);
      for i := 0 to decoder^.Count - 1 do
      begin
        addedId[i] := Utf8ToInteger(decoder^.Names[i], -1);
        addedText[i] := _Safe(decoder^.Values[i])^.U['content'];
      end;
    end;
  end;
  // without a decoder (older folders) the added tokens of added_tokens.json -
  // with one, transformers reads neither it nor special_tokens_map.json
  fn := dir + 'added_tokens.json';
  if (decoder = nil) and FileExists(fn) then
  begin
    doc.Clear;
    if not doc.InitJsonFromFile(fn, JSON_FAST_FLOAT) or not doc.IsObject then
      EInferError.RaiseUtf8('%: cannot read %', [self, fn]);
    SetLength(addedText, doc.Count);
    SetLength(addedId, doc.Count);
    for i := 0 to doc.Count - 1 do
    begin
      addedText[i] := doc.Names[i];
      addedId[i] := VariantToIntegerDef(doc.Values[i], -1);
    end;
  end;
  // transformers appends the added tokens in the order of their ids right
  // after the vocabulary, whatever the ids say: only that layout keeps them
  for i := 0 to high(addedId) do
  begin
    known := false;
    for p := 0 to high(addedId) do
      if addedId[p] = length(texts) + i then
        known := true;
    if not known then
      EInferError.RaiseUtf8('%: the added ids do not follow the vocabulary ' +
        '(% .. %) without a gap', [self, length(texts), length(texts) + high(addedId)]);
  end;
  for i := 1 to high(addedText) do
    for p := 0 to i - 1 do
      if addedText[p] = addedText[i] then
        EInferError.RaiseUtf8('%: added token "%" is listed twice',
          [self, addedText[i]]);
  Build(texts, a, b, addedText, addedId);
  for i := 0 to high(addedText) do
    if InVocab(addedText[i]) then
      EInferError.RaiseUtf8('%: added token "%" is in the vocabulary',
        [self, addedText[i]]);
  // the special tokens named must exist; without a decoder (older folders)
  // those of special_tokens_map.json too
  hasMap := false;
  if hasConfig then
    CheckSpecialTokens(@cfg, 'tokenizer_config.json');
  fn := dir + 'special_tokens_map.json';
  if (decoder = nil) and FileExists(fn) then
  begin
    if not map.InitJsonFromFile(fn, JSON_FAST_FLOAT) or not map.IsObject then
      EInferError.RaiseUtf8('%: cannot read %', [self, fn]);
    CheckSpecialTokens(@map, 'special_tokens_map.json');
    hasMap := true;
  end;
  // Qwen2Tokenizer's default for these three, when no file names them
  for i := 0 to high(DEFAULTED) do
    if not (hasConfig and (cfg.GetValueIndex(DEFAULTED[i]) >= 0)) and
       not (hasMap and (map.GetValueIndex(DEFAULTED[i]) >= 0)) then
      CheckToken('<|endoftext|>', DEFAULTED[i], 'Qwen2Tokenizer (default)');
end;

procedure TBpeTokenizer.CheckToken(const aText, aKey: RawUtf8;
  const aFile: TFileName);
begin
  if TokenId(aText) >= 0 then
    exit;
  if InVocab(aText) then
    EInferError.RaiseUtf8('%: % of % is "%", a vocabulary entry: transformers ' +
      'would match it in text', [self, aKey, aFile, aText]);
  EInferError.RaiseUtf8('%: % of % is "%", which is no token',
    [self, aKey, aFile, aText]);
end;

function TBpeTokenizer.InVocab(const aText: RawUtf8): boolean;
var
  bytes: RawByteString;
begin
  result := ByteLevelDecode(aText, bytes) and
    (BytesId(pointer(bytes), length(bytes)) >= 0);
end;

procedure TBpeTokenizer.CheckSpecialTokens(aDoc: PDocVariantData;
  const aFile: TFileName);

  procedure Check(const aValue: variant; const aKey: RawUtf8);
  var
    text: RawUtf8;
  begin
    // a string or an AddedToken object {"content": ...}; null, a boolean
    // (add_bos_token) or a number names no token
    if _Safe(aValue)^.IsObject then
      text := _Safe(aValue)^.U['content']
    else if VarIsString(aValue) then
      VariantToUtf8(aValue, text)
    else
      exit;
    CheckToken(text, aKey, aFile);
  end;

  procedure CheckList(const aKey: RawUtf8);
  var
    list: PDocVariantData;
    j: PtrInt;
  begin
    // an array of tokens or an object of named tokens
    if aDoc^.GetAsDocVariant(aKey, list) then
      for j := 0 to list^.Count - 1 do
        Check(list^.Values[j], aKey);
  end;

var
  i, n: PtrInt;
begin
  // bos_token .. mask_token and any model-specific "..._token" key
  for i := 0 to aDoc^.Count - 1 do
  begin
    n := length(aDoc^.Names[i]);
    if (n >= 6) and (copy(aDoc^.Names[i], n - 5, 6) = '_token') then
      Check(aDoc^.Values[i], aDoc^.Names[i]);
  end;
  CheckList('extra_special_tokens');
  CheckList('additional_special_tokens');
  CheckList('model_specific_special_tokens');
end;

function TBpeTokenizer.VocabTexts(aVocab: PDocVariantData): TRawUtf8DynArray;
var
  i, n, id: PtrInt;
begin
  n := aVocab^.Count;
  SetLength(result, n);
  for i := 0 to n - 1 do
  begin
    id := VariantToIntegerDef(aVocab^.Values[i], -1);
    if (id < 0) or (id >= n) or (result[id] <> '') then
      EInferError.RaiseUtf8('%: vocab id % of "%" invalid or used twice',
        [self, id, aVocab^.Names[i]]);
    result[id] := aVocab^.Names[i];
  end;
end;

procedure TBpeTokenizer.SplitMerge(const aText: RawUtf8; aIndex: PtrInt;
  out aLeft, aRight: RawUtf8);
var
  sp: PtrInt;
begin
  sp := PosEx(' ', aText);
  if (sp <= 1) or (sp = length(aText)) or (PosEx(' ', aText, sp + 1) > 0) then
    EInferError.RaiseUtf8('%: merge % "%" is not "a b"', [self, aIndex, aText]);
  aLeft := copy(aText, 1, sp - 1);
  aRight := copy(aText, sp + 1, maxInt);
end;

procedure TBpeTokenizer.Build(const aVocab, aLeft, aRight, aAddedText: TRawUtf8DynArray;
  const aAddedId: TIntegerDynArray);
var
  i, n, id, maxId, h: PtrInt;
  bytes, left, right: RawByteString;
  l, r, m, oldRank, oldNew: integer;
  one: byte;
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
  for i := 0 to n - 1 do
  begin
    if not ByteLevelDecode(aVocab[i], bytes) then
      EInferError.RaiseUtf8('%: vocab entry % "%" is not byte-level text',
        [self, i, aVocab[i]]);
    fTokens[i] := bytes;
  end;
  // hash of the byte strings, chained by id: lookups allocate nothing
  fHashMask := 1;
  while fHashMask < n * 2 do
    fHashMask := fHashMask shl 1;
  dec(fHashMask);
  SetLength(fHashHead, fHashMask + 1);
  FillCharFast(pointer(fHashHead)^, length(fHashHead) * SizeOf(integer), $ff); // -1
  SetLength(fHashNext, n);
  for i := 0 to n - 1 do
  begin
    h := crc32c(0, pointer(fTokens[i]), length(fTokens[i])) and fHashMask;
    fHashNext[i] := fHashHead[h];
    fHashHead[h] := i;
  end;
  for i := 0 to 255 do
  begin
    one := i;
    fByteId[i] := BytesId(@one, 1);
    if fByteId[i] < 0 then
      EInferError.RaiseUtf8('%: no token for byte %', [self, i]);
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
    if not ByteLevelDecode(aLeft[i], left) or not ByteLevelDecode(aRight[i], right) then
      EInferError.RaiseUtf8('%: merge % is not byte-level text', [self, i]);
    l := BytesId(pointer(left), length(left));
    r := BytesId(pointer(right), length(right));
    bytes := left + right;
    m := BytesId(pointer(bytes), length(bytes));
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
end;

procedure TBpeTokenizer.AddMerge(aLeft, aRight, aRank, aNew: integer);
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

function TBpeTokenizer.FindMerge(aLeft, aRight: integer;
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

function TBpeTokenizer.BytesId(p: pointer; aLen: PtrInt): integer;
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

procedure TBpeQueue.Push(aRank, aPos, aNew: integer);
var
  i, parent: PtrInt;
  k: Int64;
begin
  if Count = length(Key) then
  begin
    SetLength(Key, Count * 2 + 16);
    SetLength(New, Count * 2 + 16);
  end;
  k := (Int64(aRank) shl 32) or cardinal(aPos);
  i := Count;
  inc(Count);
  while i > 0 do
  begin
    parent := (i - 1) shr 1;
    if Key[parent] <= k then
      break;
    Key[i] := Key[parent];
    New[i] := New[parent];
    i := parent;
  end;
  Key[i] := k;
  New[i] := aNew;
end;

function TBpeQueue.Pop(out aPos, aNew: integer): boolean;
var
  i, child: PtrInt;
  k: Int64;
  n: integer;
begin
  result := Count > 0;
  if not result then
    exit;
  aPos := integer(Key[0] and $ffffffff);
  aNew := New[0];
  dec(Count);
  if Count = 0 then
    exit;
  k := Key[Count];
  n := New[Count];
  i := 0;
  repeat
    child := i * 2 + 1;
    if child >= Count then
      break;
    if (child + 1 < Count) and (Key[child + 1] < Key[child]) then
      inc(child);
    if Key[child] >= k then
      break;
    Key[i] := Key[child];
    New[i] := New[child];
    i := child;
  until false;
  Key[i] := k;
  New[i] := n;
end;

procedure TBpeTokenizer.Bpe(p: PByteArray; aLen: PtrInt;
  var aIds: TIntegerDynArray; var aCount: PtrInt; var aSym: TBpeSymbolDynArray;
  var aQueue: TBpeQueue);
var
  i, pos, next, prev, rank, newId, cur: integer;
begin
  if aLen <= 0 then
    exit;
  // one symbol per byte, then the merges of the tokenizers library: lowest
  // rank first, leftmost first on equal rank; a queue entry whose pair no
  // longer produces its merged id is stale and skipped
  if length(aSym) < aLen then
    SetLength(aSym, aLen);
  for i := 0 to aLen - 1 do
  begin
    aSym[i].Id := fByteId[p^[i]];
    aSym[i].Prev := i - 1;
    aSym[i].Next := i + 1;
    aSym[i].Len := 1;
  end;
  aSym[aLen - 1].Next := -1;
  aQueue.Count := 0; // the arrays are kept between the pieces of one text
  for i := 0 to aLen - 2 do
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

/// ASCII-only case folding of the contraction letters, plus U+017F -> s
// (measured: (?i) matches exactly these, see make_reference_qwen3.py)
function FoldContraction(aCp: cardinal): cardinal;
  {$ifdef HASINLINE}inline;{$endif}
begin
  if (aCp >= ord('A')) and (aCp <= ord('Z')) then
    result := aCp + 32
  else if aCp = $17f then
    result := ord('s')
  else
    result := aCp;
end;

/// end of the regex match starting at aPos (exclusive) - see the unit header
function SplitEnd(cp: PCardinalArray; aPos, aCount: PtrInt): PtrInt;
var
  c0, c1, c2: cardinal;
  k0: TUnicodeClass;
  j, k: PtrInt;

  function IsCrLf(c: cardinal): boolean;
  begin
    result := (c = 10) or (c = 13);
  end;

  function IsOther(i: PtrInt): boolean; // [^\s\p{L}\p{N}]
  begin
    result := UnicodeClassOf(cp^[i]) = ucOther;
  end;

begin
  c0 := cp^[aPos];
  k0 := UnicodeClassOf(c0);
  // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
  if (c0 = ord('''')) and (aPos + 1 < aCount) then
  begin
    c1 := FoldContraction(cp^[aPos + 1]);
    if (c1 = ord('s')) or (c1 = ord('t')) or (c1 = ord('m')) or (c1 = ord('d')) then
    begin
      result := aPos + 2;
      exit;
    end;
    if aPos + 2 < aCount then
    begin
      c2 := FoldContraction(cp^[aPos + 2]);
      if (((c1 = ord('r')) or (c1 = ord('v'))) and (c2 = ord('e'))) or
         ((c1 = ord('l')) and (c2 = ord('l'))) then
      begin
        result := aPos + 3;
        exit;
      end;
    end;
  end;
  // 2. [^\r\n\p{L}\p{N}]?\p{L}+
  j := -1;
  if k0 = ucLetter then
    j := aPos
  else if (k0 <> ucNumber) and not IsCrLf(c0) and (aPos + 1 < aCount) and
          (UnicodeClassOf(cp^[aPos + 1]) = ucLetter) then
    j := aPos + 1;
  if j >= 0 then
  begin
    while (j < aCount) and (UnicodeClassOf(cp^[j]) = ucLetter) do
      inc(j);
    result := j;
    exit;
  end;
  // 3. \p{N}
  if k0 = ucNumber then
  begin
    result := aPos + 1;
    exit;
  end;
  // 4. ' '?[^\s\p{L}\p{N}]+[\r\n]*
  k := aPos;
  if (c0 = 32) and (aPos + 1 < aCount) and IsOther(aPos + 1) then
    k := aPos + 1;
  if IsOther(k) then
  begin
    j := k + 1;
    while (j < aCount) and IsOther(j) do
      inc(j);
    while (j < aCount) and IsCrLf(cp^[j]) do
      inc(j);
    result := j;
    exit;
  end;
  // only white space is left: its run is aPos..j-1
  j := aPos + 1;
  while (j < aCount) and (UnicodeClassOf(cp^[j]) = ucSpace) do
    inc(j);
  // 5. \s*[\r\n]+ - up to the last CR/LF of the run
  k := j - 1;
  while (k >= aPos) and not IsCrLf(cp^[k]) do
    dec(k);
  if k >= aPos then
    result := k + 1
  // 6. \s+(?!\S) - the run up to the end, or all but its last char
  else if j = aCount then
    result := j
  else if j - aPos >= 2 then
    result := j - 1
  // 7. \s+ - a single white space before a non-space
  else
    result := j;
end;

function TBpeTokenizer.Normalize(const aText: RawUtf8): RawUtf8;
begin
  result := NfcNormalize(aText);
end;

function TBpeTokenizer.Split(const aText: RawUtf8): TRawUtf8DynArray;
var
  cp: TCardinalDynArray;
  n, pos, e, count: PtrInt;
begin
  result := nil;
  n := Utf8ToCodePoints(pointer(aText), length(aText), cp);
  NfcCodePoints(cp, n);
  count := 0;
  pos := 0;
  while pos < n do
  begin
    e := SplitEnd(pointer(cp), pos, n);
    if count = length(result) then
      SetLength(result, count * 2 + 8);
    result[count] := CodePointsToUtf8(@cp[pos], e - pos);
    inc(count);
    pos := e;
  end;
  SetLength(result, count);
end;

procedure TBpeTokenizer.EncodeAppend(const aText: RawUtf8;
  var aIds: TIntegerDynArray; var aCount: PtrInt);
var
  cp: TCardinalDynArray;
  n, pos, e, len: PtrInt;
  bytes: TByteDynArray;
  sym: TBpeSymbolDynArray;
  queue: TBpeQueue;
begin
  // one set of buffers for all pieces of the text: no allocation per piece
  queue.Count := 0;
  n := Utf8ToCodePoints(pointer(aText), length(aText), cp);
  NfcCodePoints(cp, n);
  SetLength(bytes, n * 4 + 4);
  pos := 0;
  while pos < n do
  begin
    e := SplitEnd(pointer(cp), pos, n);
    len := CodePointsToUtf8Buffer(@cp[pos], e - pos, pointer(bytes));
    Bpe(pointer(bytes), len, aIds, aCount, sym, queue);
    pos := e;
  end;
end;

function TBpeTokenizer.Encode(const aText: RawUtf8): TIntegerDynArray;
var
  n: PtrInt;
begin
  result := nil;
  n := 0;
  EncodeAppend(aText, result, n);
  SetLength(result, n);
end;

function TBpeTokenizer.TokenId(const aContent: RawUtf8): integer;
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

function TBpeTokenizer.TokenBytes(aId: integer): RawByteString;
begin
  if (aId < 0) or (aId > high(fTokens)) then
    EInferError.RaiseUtf8('%.TokenBytes: id % outside 0..%',
      [self, aId, high(fTokens)]);
  result := fTokens[aId];
end;

function TBpeTokenizer.Decode(const aIds: array of integer): RawUtf8;
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
    MoveFast(pointer(fTokens[aIds[i]])^, PByteArray(result)^[n], length(fTokens[aIds[i]]));
    inc(n, length(fTokens[aIds[i]]));
  end;
end;

function TBpeTokenizer.VocabSize: integer;
begin
  result := length(fTokens);
end;


initialization
  InitByteLevel;

end.
