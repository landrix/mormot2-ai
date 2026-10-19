/// native inference engine - a low-latency thread pool for the CPU backend
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - a forward pass issues a few hundred short parallel operations per text
//   (every Linear, Attention, GELU): a pool that puts its threads to sleep after
//   each one, or sleeps a millisecond while acquiring them, costs more than the
//   work. Here the workers spin for a short while after a job and only then wait
//   on their event; the calling thread works too
// - dynamic scheduling: every thread takes the next ticket from one monotonic
//   counter; Run returns once every ITEM is done and never waits for a worker
//   the OS has not scheduled yet (in a VM that happens all the time)
// - a worker may wake late, while the next job is being set up: a job owns the
//   tickets Base .. Base + Count - 1, published under a seqlock (odd while Run
//   writes Job, Base and Count). A worker reads the sequence, Count and Base,
//   then the sequence again, and uses the snapshot only if the sequence was
//   even and unchanged (two equal Counts would not do: a job of 8, then of 2,
//   then of 8 items could mix one's Count with another's Base), then claims a
//   ticket by compare-exchange only while it lies inside the pair's range. An
//   exhausted (old) job lets nobody claim, a set-up job neither, so the counter
//   does not move until the new job is published - no ticket is ever taken and
//   dropped, no item runs twice or never. Before the counter could wrap, Run
//   restarts it at 0 - with the sequence odd and no worker inside Work, so no
//   snapshot from before the restart can ever match a ticket after it
// - one Run at a time (the caller holds a lock); an exception in an item is
//   re-raised by Run once every item is done
// - an item should not allocate on the heap: with Delphi's memory manager the
//   threads then wait at its lock (measured, see RunWorkers); an item that
//   needs scratch memory takes it per worker - RunWorkers passes the index of
//   the thread that runs it
unit mormot.ai.infer.threads;

{
  *****************************************************************************

    - TInferThreadPool: run aCount independent items on all cores, with or
      without the index of the thread that runs each

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

{$ifndef FPC}{$ifdef CPUX64}
  {$EXCESSPRECISION OFF} // single expressions in single, as FPC computes them
{$endif}{$endif}

uses
  classes,
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.ai.infer.tensor;

type
  /// one work item of TInferThreadPool.Run
  TOnInferItem = procedure(aItem: integer) of object;

  /// one work item of TInferThreadPool.RunWorkers
  // - aWorker is 0 for the thread that called RunWorkers, 1..Workers for the
  // pool's threads: one index is one thread during a run, so scratch memory
  // allocated per index before the run is never shared
  TOnInferWorkerItem = procedure(aItem, aWorker: integer) of object;

  TInferThreadPool = class;

  /// a worker of TInferThreadPool
  TInferWorker = class(TThread)
  protected
    fPool: TInferThreadPool;
    fWake: TSynEvent;
    fIndex: integer;
    procedure Execute; override;
  public
    constructor Create(aPool: TInferThreadPool; aIndex: integer);
    destructor Destroy; override;
  end;

  /// a fixed set of worker threads for short parallel operations
  TInferThreadPool = class
  protected
    fWorkers: array of TInferWorker;
    fJob: TOnInferItem;
    fJobWorker: TOnInferWorkerItem; // set instead of fJob by RunWorkers
    fTicket, fBase, fCount, fDone: integer;
    // seqlock of the (Job, Base, Count) snapshot: odd while Run sets it up
    fSeq: integer;
    // workers inside Work: the ticket counter is reset only when it is 0
    fActive: integer;
    // the generation the idle workers spin on: on a line of its own (128 bytes
    // cover the largest cache line we run on), so ticket claims and done counts
    // do not invalidate it for every spinner
    fPad1: array[0..31] of integer;
    fGeneration: integer;
    fPad2: array[0..31] of integer;
    fTerminated: boolean;
    fError: RawUtf8;
    fErrorLock: TLightLock;
    procedure Work(aWorker: integer);
    procedure Start(const aJob: TOnInferItem; const aJobWorker: TOnInferWorkerItem;
      aCount: integer);
  public
    /// start aWorkers threads (the caller of Run is one more)
    constructor Create(aWorkers: integer);
    /// stop and free the workers
    destructor Destroy; override;
    /// aJob(0) .. aJob(aCount - 1) on the workers and the calling thread
    // - returns when every item is done; not reentrant, one Run at a time
    procedure Run(const aJob: TOnInferItem; aCount: integer);
    /// aJob(0, w) .. aJob(aCount - 1, w), w the index of the running thread:
    // 0 the caller, 1..Workers the pool's threads
    // - for items that need scratch memory: allocate Workers + 1 buffers once,
    // before the run, and let each item use the one of its index - measured
    // with 32 threads and 20000 items of 16 KB scratch: Delphi 13 Win64 took
    // 67 ms allocating per item and 12 ms with a buffer per index; FPC 7 ms
    // either way (its heap is per thread)
    procedure RunWorkers(const aJob: TOnInferWorkerItem; aCount: integer);
    /// the worker threads, the caller of Run not counted
    function Workers: integer;
    /// the next ticket - for tests of the counter restart near 2^30
    property Ticket: integer
      read fTicket write fTicket;
  end;


implementation

{$ifdef FPC}
  {$Q-} // tickets wrap modulo 2^32 on purpose
{$else}
  {$OVERFLOWCHECKS OFF}
{$endif FPC}

const
  /// spin iterations before a worker sleeps on its event: long enough to
  // bridge the serial steps between two parallel operations of a forward pass
  SPIN_COUNT = 200000;

/// a plain read the compiler cannot keep in a register: a call per spin, and no
// write - spinning threads must not fight over the cache line
function PlainRead(var aValue: integer): integer;
begin
  result := aValue;
end;

/// a read with a full barrier: a compare-exchange of the value with itself
function AtomicRead(var aValue: integer): integer;
begin
  repeat
    result := aValue;
  until LockedExc32(PCardinal(@aValue)^, cardinal(result), cardinal(result));
end;

/// a write with a full barrier
procedure AtomicWrite(var aTarget: integer; aValue: integer);
var
  old: integer;
begin
  repeat
    old := aTarget;
  until LockedExc32(PCardinal(@aTarget)^, cardinal(aValue), cardinal(old));
end;


{ TInferWorker }

constructor TInferWorker.Create(aPool: TInferThreadPool; aIndex: integer);
begin
  fPool := aPool;
  fIndex := aIndex;
  fWake := TSynEvent.Create;
  inherited Create(false);
end;

destructor TInferWorker.Destroy;
begin
  inherited Destroy; // waits for Execute to end
  fWake.Free;
end;

procedure TInferWorker.Execute;
var
  seen, gen, spin: integer;
begin
  seen := 0;
  repeat
    // spin a little, then sleep until the pool wakes us
    spin := 0;
    repeat
      gen := PlainRead(fPool.fGeneration);
      if (gen <> seen) or fPool.fTerminated then
        break;
      inc(spin);
      if spin > SPIN_COUNT then
      begin
        fWake.WaitFor(100);
        spin := 0;
      end;
    until false;
    if fPool.fTerminated then
      break;
    seen := gen;
    fPool.Work(fIndex);
  until false;
end;


{ TInferThreadPool }

constructor TInferThreadPool.Create(aWorkers: integer);
var
  i: PtrInt;
begin
  inherited Create;
  SetLength(fWorkers, aWorkers);
  for i := 0 to aWorkers - 1 do
    fWorkers[i] := TInferWorker.Create(self, i + 1); // 0 is the caller
end;

destructor TInferThreadPool.Destroy;
var
  i: PtrInt;
begin
  fTerminated := true;
  for i := 0 to high(fWorkers) do
    fWorkers[i].fWake.SetEvent;
  for i := 0 to high(fWorkers) do
    fWorkers[i].Free;
  inherited Destroy;
end;

function TInferThreadPool.Workers: integer;
begin
  result := length(fWorkers);
end;

procedure TInferThreadPool.Work(aWorker: integer);
var
  ticket, count, base, done, seq: integer;
  item: cardinal;
begin
  // a consistent (Count, Base) snapshot by seqlock: the sequence is even and
  // the same before and after reading - two equal Counts would not do (jobs of
  // 8, 2, 8 items could mix the Base of one with the Count of another). The
  // snapshot stays valid for every claim below: a job is replaced only once all
  // its items are done, and then no ticket lies in its range any more
  // registered before the snapshot is read: Run resets the ticket counter only
  // while no worker can hold a snapshot (see Run)
  InterlockedIncrement(fActive);
  try
    seq := AtomicRead(fSeq);
    if odd(seq) then
      exit; // being set up: Run wakes the workers again once published
    count := AtomicRead(fCount);
    base := AtomicRead(fBase);
    if (count = 0) or (AtomicRead(fSeq) <> seq) then
      exit;
    done := 0;
    repeat
      // claim the next ticket only while it is an item of this pair's job: a
      // ticket is never taken and then dropped, so no item can get lost - one
      // atomic operation per item
      repeat
        ticket := PlainRead(fTicket);
        item := cardinal(ticket - base);
        if item >= cardinal(count) then
        begin
          // every item of this job is taken: publish what this thread did
          if done > 0 then
            LockedAdd32(PCardinal(@fDone)^, done);
          exit;
        end;
      until LockedExc32(PCardinal(@fTicket)^, cardinal(ticket + 1), cardinal(ticket));
      try
        if Assigned(fJobWorker) then
          fJobWorker(item, aWorker)
        else
          fJob(item);
      except
        on E: Exception do
        begin
          fErrorLock.Lock;
          if fError = '' then
            fError := FormatUtf8('%: %', [E, E.Message]);
          fErrorLock.UnLock;
        end;
      end;
      inc(done); // published at exit, with the results (locked add = barrier)
    until false;
  finally
    InterlockedDecrement(fActive);
  end;
end;

procedure TInferThreadPool.Run(const aJob: TOnInferItem; aCount: integer);
begin
  Start(aJob, nil, aCount);
end;

procedure TInferThreadPool.RunWorkers(const aJob: TOnInferWorkerItem;
  aCount: integer);
begin
  Start(nil, aJob, aCount);
end;

procedure TInferThreadPool.Start(const aJob: TOnInferItem;
  const aJobWorker: TOnInferWorkerItem; aCount: integer);
var
  i, spin: integer;
  err: RawUtf8;
begin
  if aCount <= 0 then
    exit;
  if (aCount = 1) or (fWorkers = nil) then
  begin
    for i := 0 to aCount - 1 do
      if Assigned(aJobWorker) then
        aJobWorker(i, 0)
      else
        aJob(i);
    exit;
  end;
  // the snapshot becomes invalid first (odd sequence): a worker entering from
  // now on leaves at once, one already inside counts in fActive
  InterlockedIncrement(fSeq);
  // tickets wrap modulo 2^32: long before that, once no worker is inside Work
  // (so none holds a snapshot - the odd sequence keeps new ones out until the
  // job below is published), restart the counter at 0
  if cardinal(AtomicRead(fTicket)) >= $40000000 then
  begin
    spin := 0;
    while AtomicRead(fActive) <> 0 do
    begin
      inc(spin);
      if spin > SPIN_COUNT then
        SwitchToThread;
    end;
    AtomicWrite(fTicket, 0);
  end;
  // set up while the sequence is odd: nobody claims meanwhile (an older
  // snapshot's range is exhausted), so the counter is still and Base is the
  // next ticket
  AtomicWrite(fCount, 0);
  fJob := aJob;
  fJobWorker := aJobWorker;
  fError := '';
  AtomicWrite(fDone, 0);
  AtomicWrite(fBase, AtomicRead(fTicket));
  AtomicWrite(fCount, aCount);
  InterlockedIncrement(fSeq); // even again: published
  InterlockedIncrement(fGeneration); // wakes the spinning workers
  for i := 0 to high(fWorkers) do
    fWorkers[i].fWake.SetEvent; // only matters for a sleeping worker
  Work(0);
  // the last items finish on other threads: wait for them, not for workers
  spin := 0;
  while PlainRead(fDone) < aCount do
  begin
    inc(spin);
    if spin > SPIN_COUNT then
      SwitchToThread;
  end;
  AtomicRead(fDone); // barrier: every item's results are visible
  if fError <> '' then
  begin
    err := fError;
    fError := '';
    EInferError.RaiseUtf8('TInferThreadPool.Run: %', [err]);
  end;
end;

end.
