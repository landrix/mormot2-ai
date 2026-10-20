// LandrixAI - throughput of the native embedder (phase 1c of
// docs/native-inference.md)
// - embeds the embed-de corpus (documents, then queries) one text at a time, as
//   the lembed oracle does when its reference was made, and reports texts/s,
//   tokens/s, the time per backend operation, the peak memory and the cosine
//   against the lembed vectors of tests/data/embed-de/reference
//
//   usage: embed-bench [gguf] [texts=all] [threads=0 (default) | ref | lembed]
//   - ref: the scalar reference backend, otherwise TInferCpuFastBackend
//   - lembed: the same loop through sqlite-lembed (vendor/sqlite-ext/<cpu>-<os>),
//     for a comparison on one machine with one program
//   - texts = soak<minutes>, e.g. soak90: a soak test - the corpus over and over
//     for that long, a report every LAI_SOAK_INTERVAL seconds (default 600) with
//     throughput, current and peak memory, and every vector compared bit for bit
//     with the one of the first pass; a file soak.stop next to the program ends
//     it after the current text (native engine only)
//   e.g.   embed-bench vendor/models/bge-m3-q8_0.gguf 40 1
program embed.bench;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  {$ifdef OSWINDOWS}
  windows,
  {$endif OSWINDOWS}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.buffers,
  mormot.core.variants,
  mormot.core.perf,
  mormot.ai.llm.types,
  mormot.ai.embeddings,
  mormot.db.raw.sqlite3.static, // the SQLite that hosts lembed
  mormot.ai.sqlite,
  mormot.ai.embed.lembed,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.threads,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.kernels,
  mormot.ai.infer.pipeline.embed;

type
  /// forwards every operation to an inner backend and sums its time per op
  TTimedBackend = class(TInferBackend)
  protected
    fInner: TInferBackend;
    fTime: array[TInferOp] of Int64;
    fCalls: array[TInferOp] of Int64;
    fStart: Int64;
    procedure Enter;
    procedure Leave(aOp: TInferOp);
  public
    constructor Create(aInner: TInferBackend);
    function Supports(aOp: TInferOp; aWeights: TInferDType): boolean; override;
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); override;
    procedure LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
      gamma, beta: PSingleArray; aEps: double; y: PSingleArray); override;
    procedure Gelu(x: PSingleArray; aCount: PtrInt); override;
    procedure Add(x, y: PSingleArray; aCount: PtrInt); override;
    procedure SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
      aKeep: PIntegerArray); override;
    procedure Attention(q, k, v: PSingleArray; aSeq, aHeads, aHeadDim: PtrInt;
      aKeep: PIntegerArray; aScratch, y: PSingleArray); override;
    procedure L2Normalize(x: PSingleArray; aCount: PtrInt); override;
    procedure CausalAttention(q: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      keys, values: PSingleArray; aPast, aKvHeads: PtrInt; y: PSingleArray;
      aWindow: PtrInt = 0; aSinks: PSingleArray = nil;
      aScale: double = 0); override;
    procedure RmsNorm(x: PSingleArray; aRows, aCols: PtrInt; w: PSingleArray;
      aEps: double; y: PSingleArray); override;
    procedure SwiGlu(gate, up: PSingleArray; aCount: PtrInt;
      aAlpha: single = 1; aLimit: single = 0); override;
    procedure GeluTanhGlu(gate, up: PSingleArray; aCount: PtrInt); override;
    procedure RunItems(const aJob: TOnInferItem; aCount: integer); override;
    procedure Rope(x: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      aCos, aSin: PSingleArray); override;
    function Report(aTotal: Int64): RawUtf8;
  end;

const
  OP_NAME: array[TInferOp] of RawUtf8 = (
    'linear', 'layernorm', 'gelu', 'add', 'softmax', 'attention', 'l2', 'causal',
    'rmsnorm', 'swiglu', 'rope', 'geglu');

function NotZero(v: Int64): Int64;
begin
  if v < 1 then
    result := 1
  else
    result := v;
end;

constructor TTimedBackend.Create(aInner: TInferBackend);
begin
  inherited Create;
  fInner := aInner;
end;

procedure TTimedBackend.Enter;
begin
  QueryPerformanceMicroSeconds(fStart);
end;

procedure TTimedBackend.Leave(aOp: TInferOp);
var
  now: Int64;
begin
  QueryPerformanceMicroSeconds(now);
  inc(fTime[aOp], now - fStart);
  inc(fCalls[aOp]);
end;

function TTimedBackend.Supports(aOp: TInferOp; aWeights: TInferDType): boolean;
begin
  result := fInner.Supports(aOp, aWeights);
end;

procedure TTimedBackend.Linear(x: PSingleArray; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
begin
  Enter;
  fInner.Linear(x, aRows, aIn, w, aOut, bias, y);
  Leave(ioLinear);
end;

procedure TTimedBackend.LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
  gamma, beta: PSingleArray; aEps: double; y: PSingleArray);
begin
  Enter;
  fInner.LayerNorm(x, aRows, aCols, gamma, beta, aEps, y);
  Leave(ioLayerNorm);
end;

procedure TTimedBackend.Gelu(x: PSingleArray; aCount: PtrInt);
begin
  Enter;
  fInner.Gelu(x, aCount);
  Leave(ioGelu);
end;

procedure TTimedBackend.Add(x, y: PSingleArray; aCount: PtrInt);
begin
  Enter;
  fInner.Add(x, y, aCount);
  Leave(ioAdd);
end;

procedure TTimedBackend.SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
  aKeep: PIntegerArray);
begin
  Enter;
  fInner.SoftmaxMasked(x, aRows, aCols, aKeep);
  Leave(ioSoftmaxMasked);
end;

procedure TTimedBackend.Attention(q, k, v: PSingleArray; aSeq, aHeads,
  aHeadDim: PtrInt; aKeep: PIntegerArray; aScratch, y: PSingleArray);
begin
  Enter;
  fInner.Attention(q, k, v, aSeq, aHeads, aHeadDim, aKeep, aScratch, y);
  Leave(ioAttention);
end;

procedure TTimedBackend.L2Normalize(x: PSingleArray; aCount: PtrInt);
begin
  Enter;
  fInner.L2Normalize(x, aCount);
  Leave(ioL2Normalize);
end;

procedure TTimedBackend.CausalAttention(q: PSingleArray; aRows, aHeads,
  aHeadDim: PtrInt; keys, values: PSingleArray; aPast, aKvHeads: PtrInt;
  y: PSingleArray; aWindow: PtrInt; aSinks: PSingleArray; aScale: double);
begin
  fInner.CausalAttention(q, aRows, aHeads, aHeadDim, keys, values, aPast, aKvHeads, y,
    aWindow, aSinks, aScale);
end;

procedure TTimedBackend.RmsNorm(x: PSingleArray; aRows, aCols: PtrInt;
  w: PSingleArray; aEps: double; y: PSingleArray);
begin
  Enter;
  fInner.RmsNorm(x, aRows, aCols, w, aEps, y);
  Leave(ioRmsNorm);
end;

procedure TTimedBackend.SwiGlu(gate, up: PSingleArray; aCount: PtrInt;
  aAlpha, aLimit: single);
begin
  Enter;
  fInner.SwiGlu(gate, up, aCount, aAlpha, aLimit);
  Leave(ioSwiGlu);
end;

procedure TTimedBackend.GeluTanhGlu(gate, up: PSingleArray; aCount: PtrInt);
begin
  Enter;
  fInner.GeluTanhGlu(gate, up, aCount);
  Leave(ioGeluTanhGlu);
end;

procedure TTimedBackend.RunItems(const aJob: TOnInferItem; aCount: integer);
begin
  fInner.RunItems(aJob, aCount); // the load on the threads of the inner one
end;

procedure TTimedBackend.Rope(x: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
  aCos, aSin: PSingleArray);
begin
  Enter;
  fInner.Rope(x, aRows, aHeads, aHeadDim, aCos, aSin);
  Leave(ioRope);
end;

function TTimedBackend.Report(aTotal: Int64): RawUtf8;
var
  op: TInferOp;
  sum: Int64;
begin
  result := '';
  sum := 0;
  for op := low(op) to high(op) do
  begin
    inc(sum, fTime[op]);
    if fCalls[op] > 0 then
      result := FormatUtf8('%  %: % ms (% %%)'#10, [result, OP_NAME[op],
        fTime[op] div 1000, (fTime[op] * 1000 div NotZero(aTotal)) / 10]);
  end;
  result := FormatUtf8('%  rest (tokenizer, embeddings, copies): % ms'#10,
    [result, (aTotal - sum) div 1000]);
end;

{$ifdef OSDARWIN}
/// the libc call: struct rusage starts with two timevals (16 bytes each on a
// 64-bit Darwin), then ru_maxrss in bytes
function getrusage(aWho: integer; aUsage: pointer): integer; cdecl;
  external 'c' name 'getrusage';
{$endif OSDARWIN}

/// peak resident memory in MB, -1 if unknown
function PeakRssMB: integer;
{$ifdef OSDARWIN}
var
  usage: array[0..17] of Int64; // struct rusage: 144 bytes
{$endif OSDARWIN}
{$ifdef OSLINUX}
var
  s: RawUtf8;
  p: PUtf8Char;
{$endif OSLINUX}
{$ifdef OSWINDOWS}
var
  info: TMemoryInfo;
{$endif OSWINDOWS}
begin
  result := -1;
  {$ifdef OSDARWIN}
  if getrusage(0 {RUSAGE_SELF}, @usage) = 0 then
    result := usage[4] shr 20;
  {$endif OSDARWIN}
  {$ifdef OSLINUX}
  s := StringFromFileNoSize('/proc/self/status');
  p := StrPosI('VMHWM:', pointer(s));
  if p <> nil then
    result := GetCardinal(PUtf8Char(GotoNextNotSpace(p + 6))) div 1024;
  {$endif OSLINUX}
  {$ifdef OSWINDOWS}
  if GetMemoryInfo(info, true) then
    result := info.allocreserved shr 20; // the peak working set
  {$endif OSWINDOWS}
end;

{$ifdef OSWINDOWS}
/// opt out of Windows power throttling (EcoQoS): a process without a visible
// window - e.g. started over SSH or as a service - is otherwise slowed down
// after a while, which spoils any long measurement
procedure NoPowerThrottling;
type
  TPowerThrottling = record
    Version, ControlMask, StateMask: cardinal;
  end;
var
  setInfo: function(aProcess: THandle; aClass: integer; aInfo: pointer;
    aSize: cardinal): LongBool; stdcall;
  state: TPowerThrottling;
begin
  setInfo := GetProcAddress(GetModuleHandle('kernel32.dll'), 'SetProcessInformation');
  if not Assigned(setInfo) then
    exit;
  state.Version := 1;            // PROCESS_POWER_THROTTLING_CURRENT_VERSION
  state.ControlMask := 1;        // PROCESS_POWER_THROTTLING_EXECUTION_SPEED
  state.StateMask := 0;          // ... switched off
  if setInfo(GetCurrentProcess, 4 {ProcessPowerThrottling}, @state,
       SizeOf(state)) then
    ConsoleWrite('power throttling: off');
end;
{$endif OSWINDOWS}

const
  /// the largest text of each length bucket, in tokens (512 = the model limit)
  BUCKET_MAX: array[0..3] of integer = (16, 64, 256, 512);
  /// the folder of the sqlite-lembed build for this platform
  EXT_PLATFORM =
    {$ifdef CPUX64} 'x86_64-' {$else} 'aarch64-' {$endif CPUX64} +
    {$ifdef OSWINDOWS} 'win64' {$else} 'linux' {$endif OSWINDOWS};

/// current resident memory in MB, -1 if unknown
function RssMB: integer;
{$ifdef OSLINUX}
var
  s: RawUtf8;
  p: PUtf8Char;
{$endif OSLINUX}
{$ifdef OSWINDOWS}
var
  info: TMemoryInfo;
{$endif OSWINDOWS}
begin
  result := -1;
  {$ifdef OSLINUX}
  s := StringFromFileNoSize('/proc/self/status');
  p := StrPosI('VMRSS:', pointer(s));
  if p <> nil then
    result := GetCardinal(PUtf8Char(GotoNextNotSpace(p + 6))) div 1024;
  {$endif OSLINUX}
  {$ifdef OSWINDOWS}
  if GetMemoryInfo(info, true) then
    result := info.allocused shr 20; // the working set
  {$endif OSWINDOWS}
end;

/// <dir>/<aPath> for the first of the program folder and its parents holding it
function FindUp(const aPath: TFileName): TFileName;
var
  dir: TFileName;
  i: integer;
begin
  dir := ExcludeTrailingPathDelimiter(Executable.ProgramFilePath);
  for i := 1 to 6 do
  begin
    result := MakePath([dir, aPath]);
    if FileExists(result) then
      exit;
    dir := ExtractFileDir(dir);
  end;
  result := '';
end;

var
  gguf, corpus, queries, refDocs, refQueries: TFileName;
  texts, lines: TRawUtf8DynArray;
  ref, v: TSingleDynArray;
  raw: RawUtf8;
  row: TDocVariantData;
  cpu: TInferCpuBackend;
  lembed: IEmbedder;
  extDir: TFileName;
  timed: TTimedBackend;
  emb: TInferEmbedder;
  i, n, limit, tokens, h, below: PtrInt;
  t0, t1: Int64;
  cos, minCos, a: double;
  j: PtrInt;
  t2, t3: Int64;
  // soak test
  soakMin, interval, pass, done, mism, tokTotal, lastTok: Int64;
  tLast: Int64;
  first: TSingleDynArray;
  tokc: TIntegerDynArray;
  stopFile: TFileName;
  stop: boolean;
  bucket: integer;
  // per text length bucket (up to 16, 64, 256, 512 tokens): texts, tokens, us
  bTexts, bTokens, bTime: array[0..3] of Int64;
begin
  gguf := '';
  if ParamCount >= 1 then
    gguf := ParamStr(1);
  if gguf = '' then
    gguf := FindUp(MakePath(['vendor', 'models', 'bge-m3-q8_0.gguf']));
  limit := MaxInt;
  soakMin := 0;
  if (ParamCount >= 2) and (copy(ParamStr(2), 1, 4) = 'soak') then
    soakMin := GetInteger(pointer(StringToUtf8(copy(ParamStr(2), 5, 10))))
  else if (ParamCount >= 2) and (ParamStr(2) <> 'all') then
    limit := GetInteger(pointer(StringToUtf8(ParamStr(2))));
  corpus := FindUp(MakePath(['tests', 'data', 'embed-de', 'corpus.jsonl']));
  queries := FindUp(MakePath(['tests', 'data', 'embed-de', 'queries.jsonl']));
  refDocs := FindUp(MakePath(['tests', 'data', 'embed-de', 'reference', 'docs.f32']));
  refQueries := FindUp(MakePath(['tests', 'data', 'embed-de', 'reference', 'queries.f32']));
  if (gguf = '') or (corpus = '') then
  begin
    ConsoleWrite('usage: embed-bench [gguf] [texts=all] - model or corpus not found');
    ExitCode := 2;
    exit;
  end;
  // documents then queries, in file order: the order of the lembed vectors
  texts := nil;
  ref := nil;
  for i := 0 to 1 do
  begin
    if i = 0 then
      raw := StringFromFile(corpus)
    else
      raw := StringFromFile(queries);
    lines := nil; // CsvToRawUtf8DynArray appends
    CsvToRawUtf8DynArray(pointer(raw), lines, #10);
    for j := 0 to high(lines) do
      if TrimU(lines[j]) <> '' then
      begin
        row.Clear;
        row.InitJson(lines[j], JSON_FAST);
        AddRawUtf8(texts, row.U['text']);
      end;
  end;
  raw := StringFromFile(refDocs) + StringFromFile(refQueries);
  if length(raw) = length(texts) * 1024 * 4 then
  begin
    SetLength(ref, length(texts) * 1024);
    MoveFast(pointer(raw)^, pointer(ref)^, length(raw));
  end;
  n := MinPtrInt(limit, length(texts));
  if (soakMin > 0) and ((n = 0) or
     ((ParamCount >= 3) and (ParamStr(3) = 'lembed'))) then
  begin
    ConsoleWrite('soak: needs a non-empty corpus and the native engine');
    ExitCode := 2;
    exit;
  end;
  {$ifdef OSWINDOWS}
  if GetEnvironmentVariable('LAI_THROTTLE') = '' then
    NoPowerThrottling;
  {$endif OSWINDOWS}
  if (ParamCount >= 3) and (ParamStr(3) = 'lembed') then
  begin
    extDir := FindUp(MakePath(['vendor', 'sqlite-ext', EXT_PLATFORM,
      'lembed0' + SqliteExtSuffix]));
    if extDir = '' then
    begin
      ConsoleWrite('lembed0 for % not found', [EXT_PLATFORM]);
      ExitCode := 2;
      exit;
    end;
    QueryPerformanceMicroSeconds(t0);
    lembed := TLembedEmbedder.Create(StringToUtf8(ExtractFileDir(extDir)),
      StringToUtf8(gguf), 'bench');
    QueryPerformanceMicroSeconds(t1);
    ConsoleWrite('backend: lembed %, loaded in % ms', [EXT_PLATFORM, (t1 - t0) div 1000]);
    minCos := 1;
    below := 0;
    QueryPerformanceMicroSeconds(t0);
    for i := 0 to n - 1 do
    begin
      v := lembed.Embed(texts[i]);
      if ref <> nil then
      begin
        cos := 0;
        for j := 0 to high(v) do
        begin
          a := v[j];
          cos := cos + a * ref[i * length(v) + j];
        end;
        if cos < minCos then
          minCos := cos;
        if cos < 0.999 then
          inc(below);
      end;
    end;
    QueryPerformanceMicroSeconds(t1);
    ConsoleWrite('peak RSS: % MB', [PeakRssMB]);
    lembed := nil;
    // the token count from the native tokenizer, after the timed run
    cpu := TInferCpuBackend.Create;
    emb := TInferEmbedder.CreateFromGguf(gguf, cpu);
    try
      tokens := 0;
      for i := 0 to n - 1 do
        inc(tokens, length(emb.Tokenizer.Encode(texts[i])));
    finally
      emb.Free;
      cpu.Free;
    end;
    ConsoleWrite('% texts, % tokens (mean %) in % ms: % texts/s, % tokens/s',
      [n, tokens, tokens / MaxPtrInt(n, 1), (t1 - t0) div 1000,
       n * 1e6 / NotZero(t1 - t0), tokens * 1e6 / NotZero(t1 - t0)]);
    if ref <> nil then
      ConsoleWrite('cosine against the lembed reference: min %, % below 0.999',
        [minCos, below]);
    exit;
  end;
  if (ParamCount >= 3) and (ParamStr(3) = 'ref') then
  begin
    cpu := TInferCpuBackend.Create;
    ConsoleWrite('backend: reference (scalar, one thread)');
  end
  else
  begin
    j := 0;
    if ParamCount >= 3 then
      j := GetInteger(pointer(StringToUtf8(ParamStr(3))));
    cpu := TInferCpuFastBackend.Create(j);
    ConsoleWrite('backend: fast, % threads, Q8_0 kernel %',
      [TInferCpuFastBackend(cpu).Threads, InferQ8KernelName]);
  end;
  timed := TTimedBackend.Create(cpu);
  QueryPerformanceMicroSeconds(t0);
  emb := TInferEmbedder.CreateFromGguf(gguf, timed);
  try
    QueryPerformanceMicroSeconds(t1);
    ConsoleWrite('loaded % in % ms', [ExtractFileName(gguf), (t1 - t0) div 1000]);
    h := emb.Dim;
    if soakMin > 0 then
    begin
      interval := GetInteger(pointer(StringToUtf8(
        GetEnvironmentVariable('LAI_SOAK_INTERVAL'))));
      if interval <= 0 then
        interval := 600;
      stopFile := MakePath([Executable.ProgramFilePath, 'soak.stop']);
      SetLength(tokc, n);
      for i := 0 to n - 1 do
        tokc[i] := length(emb.Tokenizer.Encode(texts[i]));
      SetLength(first, n * h);
      ConsoleWrite('soak: % min, % texts per pass, report every % s, stop file %',
        [soakMin, n, interval, stopFile]);
      pass := 0;
      done := 0;
      mism := 0;
      tokTotal := 0;
      lastTok := 0;
      stop := false;
      QueryPerformanceMicroSeconds(t0);
      tLast := t0;
      repeat
        for i := 0 to n - 1 do
        begin
          v := emb.Embed([texts[i]]);
          if pass = 0 then
            MoveFast(pointer(v)^, first[i * h], h * SizeOf(single))
          else if not CompareMem(pointer(v), @first[i * h], h * SizeOf(single)) then
            inc(mism);
          inc(done);
          inc(tokTotal, tokc[i]);
          QueryPerformanceMicroSeconds(t1);
          if t1 - tLast >= interval * 1000000 then
          begin
            ConsoleWrite('% min: pass %, % texts, % tokens/s in the last interval, ' +
              'RSS % MB (peak %), % vectors differing from pass 1',
              [(t1 - t0) div 60000000, pass + 1, done,
               (tokTotal - lastTok) * 1000000 div NotZero(t1 - tLast), RssMB,
               PeakRssMB, mism]);
            tLast := t1;
            lastTok := tokTotal;
          end;
          if (t1 - t0 >= soakMin * 60000000) or FileExists(stopFile) then
          begin
            stop := true;
            break;
          end;
        end;
        inc(pass);
      until stop;
      ConsoleWrite('soak done: % min, % full passes + % texts, % texts, % tokens, ' +
        '% tokens/s overall, RSS % MB (peak %), % vectors differing from pass 1',
        [(t1 - t0) div 60000000, done div n, done mod n, done, tokTotal,
         tokTotal * 1000000 div NotZero(t1 - t0), RssMB, PeakRssMB, mism]);
      if mism <> 0 then
        ExitCode := 1;
      exit;
    end;
    tokens := 0;
    minCos := 1;
    below := 0;
    FillCharFast(bTexts, SizeOf(bTexts), 0);
    FillCharFast(bTokens, SizeOf(bTokens), 0);
    FillCharFast(bTime, SizeOf(bTime), 0);
    QueryPerformanceMicroSeconds(t0);
    for i := 0 to n - 1 do
    begin
      j := length(emb.Tokenizer.Encode(texts[i]));
      inc(tokens, j);
      if j <= 16 then
        bucket := 0
      else if j <= 64 then
        bucket := 1
      else if j <= 256 then
        bucket := 2
      else
        bucket := 3;
      QueryPerformanceMicroSeconds(t2);
      v := emb.Embed([texts[i]]);
      QueryPerformanceMicroSeconds(t3);
      inc(bTexts[bucket]);
      inc(bTokens[bucket], j);
      inc(bTime[bucket], t3 - t2);
      if ref <> nil then
      begin
        cos := 0;
        for j := 0 to h - 1 do
        begin
          a := v[j];
          cos := cos + a * ref[i * h + j];
        end;
        if cos < minCos then
          minCos := cos;
        if cos < 0.999 then
          inc(below);
      end;
    end;
    QueryPerformanceMicroSeconds(t1);
    ConsoleWrite('% texts, % tokens (mean %) in % ms: % texts/s, % tokens/s',
      [n, tokens, tokens / MaxPtrInt(n, 1), (t1 - t0) div 1000,
       n * 1e6 / NotZero(t1 - t0), tokens * 1e6 / NotZero(t1 - t0)]);
    if ref <> nil then
      ConsoleWrite('cosine against lembed: min %, % below 0.999', [minCos, below]);
    for bucket := 0 to 3 do
      if bTexts[bucket] > 0 then
        ConsoleWrite('  up to % tokens: % texts, % tokens, % ms per text, % tokens/s',
          [BUCKET_MAX[bucket], bTexts[bucket], bTokens[bucket],
           bTime[bucket] / 1000 / bTexts[bucket],
           bTokens[bucket] * 1000000 div NotZero(bTime[bucket])]);
    ConsoleWrite('peak RSS: % MB', [PeakRssMB]);
    ConsoleWrite(timed.Report(t1 - t0));
  finally
    emb.Free;
    timed.Free;
    cpu.Free;
  end;
end.
