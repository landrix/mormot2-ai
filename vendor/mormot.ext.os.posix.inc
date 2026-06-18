// mormot.ext.os.posix.inc - POSIX implementation of TExternalProcess
// TODO: test Delphi's Linux compiler (LLVM)

{$ifdef FPC}

{ local helper: poll a file descriptor for available data }

function ExtWaitReadPending(fd, timeoutms: integer): boolean;
var
  p: TPollFD;
begin
  p.fd := fd;
  p.events := POLLIN or POLLPRI;
  p.revents := 0;
  result := FpPoll(@p, 1, timeoutms) > 0;
end;

{ background reader thread }

function ExtProcessReaderThread(p: pointer): PtrInt;
var
  proc: TExternalProcess;
  tmp: TBuffer4K;
  n: ssize_t;
  new: RawUtf8;
begin
  result := 0;
  proc := TExternalProcess(p);
  try
    repeat
      if proc.fStdoutRead = 0 then
        break;
      if not ExtWaitReadPending(proc.fStdoutRead, 50) then
        continue;
      n := fpread(proc.fStdoutRead, @tmp, SizeOf(tmp));
      if n <= 0 then
        break;
      proc.fOutputSafe.Lock;
      try
        SetLength(proc.fOutput, Length(proc.fOutput) + n);
        MoveFast(tmp, PByteArray(proc.fOutput)[Length(proc.fOutput) - n], n);
      finally
        proc.fOutputSafe.UnLock;
      end;
      if Assigned(proc.fOnOutput) then
      begin
        FastSetString(new, @tmp, n);
        proc.fOnOutput(new, proc.fPid);
      end;
    until false;
  finally
    proc.fReaderFinished := true;
  end;
end;

{ TExternalProcess }

function TExternalProcess.Start(const cmd: TRunArg; const env: TRunArg;
  const wrkdir: TFileName; options: TExtProcessOptions): boolean;
var
  stdin_fds, stdout_fds: tfildes;
  args: array[0..3] of PAnsiChar;
  cmdstr: RawUtf8;
  cpid: TPid;
begin
  result := false;
  if fStarted then
    exit;
  fCommand := cmd;
  fWorkDir := wrkdir;
  fOptions := options;
  fExitCode := -1;
  fTerminated := false;
  fReaderFinished := false;
  fOutput := '';
  // create pipes
  if fppipe(stdin_fds) <> 0 then
    exit;
  if fppipe(stdout_fds) <> 0 then
  begin
    fpclose(stdin_fds[0]);
    fpclose(stdin_fds[1]);
    exit;
  end;
  if wrkdir <> '' then
    ChDir(wrkdir);
  cpid := FpFork;
  if cpid < 0 then
  begin
    fpclose(stdin_fds[0]);
    fpclose(stdin_fds[1]);
    fpclose(stdout_fds[0]);
    fpclose(stdout_fds[1]);
    exit;
  end;
  if cpid = 0 then
  begin
    // child process
    fpclose(stdin_fds[1]);
    fpclose(stdout_fds[0]);
    fpdup2(stdin_fds[0], 0);    // stdin
    fpdup2(stdout_fds[1], 1);   // stdout
    fpdup2(stdout_fds[1], 2);   // stderr -> stdout
    fpclose(stdin_fds[0]);
    fpclose(stdout_fds[1]);
    cmdstr := cmd;
    args[0] := '/bin/sh';
    args[1] := '-c';
    args[2] := pointer(cmdstr);
    args[3] := nil;
    FpExecve('/bin/sh', @args[0], envp);
    FpExit(127);
  end;
  // parent
  fpclose(stdin_fds[0]);
  fpclose(stdout_fds[1]);
  fStdinWrite := stdin_fds[1];
  fStdoutRead := stdout_fds[0];
  fPid := cpid;
  fStarted := true;
  result := true;
  // start background reader
  BeginThread(@ExtProcessReaderThread, self, fReaderTid);
end;

function TExternalProcess.Write(const data: RawByteString): boolean;
begin
  result := Write(pointer(data), length(data));
end;

function TExternalProcess.Write(p: pointer; len: PtrInt): boolean;
var
  written: ssize_t;
  total: PtrInt;
begin
  result := false;
  if (fStdinWrite = 0) or
     (p = nil) or
     (len <= 0) then
    exit;
  total := 0;
  repeat
    written := fpwrite(fStdinWrite, PAnsiChar(p) + total, len - total);
    if written <= 0 then
      exit;
    inc(total, written);
  until total >= len;
  result := true;
end;

function TExternalProcess.WriteAndCloseStdin(const data: RawByteString): boolean;
begin
  result := Write(data);
  CloseStdin;
end;

procedure TExternalProcess.CloseStdin;
begin
  if fStdinWrite <> 0 then
  begin
    fpclose(fStdinWrite);
    fStdinWrite := 0;
  end;
end;

function TExternalProcess.ReadAvailable: RawByteString;
begin
  fOutputSafe.Lock;
  try
    result := fOutput;
    fOutput := '';
  finally
    fOutputSafe.UnLock;
  end;
end;

function TExternalProcess.HasDataAvailable: boolean;
begin
  fOutputSafe.Lock;
  try
    result := fOutput <> '';
  finally
    fOutputSafe.UnLock;
  end;
end;

function TExternalProcess.GetRunning: boolean;
var
  res, wr: cint;
begin
  result := false;
  if not fStarted or
     fTerminated then
    exit;
  wr := FpWaitPid(fPid, @res, WNOHANG);
  if wr = 0 then
    result := true
  else if wr = cint(fPid) then
  begin
    fExitCode := res shr 8; // WEXITSTATUS
    fTerminated := true;
  end;
end;

function TExternalProcess.GetPid: cardinal;
begin
  result := fPid;
end;

function TExternalProcess.Terminate(waitms: cardinal): boolean;
var
  endtix: Int64;
  res: cint;
begin
  result := not Running;
  if result then
    exit;
  if fpkill(fPid, SIGTERM) = 0 then
  begin
    endtix := GetTickCount64 + waitms;
    while GetTickCount64 < endtix do
    begin
      if FpWaitPid(fPid, @res, WNOHANG) <> 0 then
      begin
        fExitCode := res shr 8;
        fTerminated := true;
        result := true;
        exit;
      end;
      SleepHiRes(50);
    end;
  end;
  Kill;
  result := true;
end;

procedure TExternalProcess.Kill;
var
  res: cint;
begin
  if fStarted and
     not fTerminated then
  begin
    fpkill(fPid, SIGKILL);
    FpWaitPid(fPid, @res, 0); // reap zombie
    fExitCode := -1;
    fTerminated := true;
  end;
end;

function TExternalProcess.WaitFor(waitms: cardinal): integer;
var
  endtix: Int64;
  res: cint;
begin
  result := -1;
  if not fStarted then
    exit;
  if fTerminated then
  begin
    result := fExitCode;
    exit;
  end;
  if waitms = INFINITE then
  begin
    FpWaitPid(fPid, @res, 0);
    fExitCode := res shr 8;
    fTerminated := true;
    result := fExitCode;
    exit;
  end;
  endtix := GetTickCount64 + waitms;
  repeat
    if FpWaitPid(fPid, @res, WNOHANG) <> 0 then
    begin
      fExitCode := res shr 8;
      fTerminated := true;
      result := fExitCode;
      exit;
    end;
    SleepHiRes(50);
  until GetTickCount64 > endtix;
end;

destructor TExternalProcess.Destroy;
var
  endtix: Int64;
begin
  if fStarted then
  begin
    if Running then
      Terminate(2000);
    CloseStdin;
    if fStdoutRead <> 0 then
    begin
      fpclose(fStdoutRead);
      fStdoutRead := 0;
    end;
    endtix := GetTickCount64 + 3000;
    while not fReaderFinished and
          (GetTickCount64 < endtix) do
      SleepHiRes(10);
  end;
  inherited Destroy;
end;

{$else}

// Delphi POSIX: stub (fork/pipe not available without FPC's BaseUnix)

function TExternalProcess.Start(const cmd: TRunArg; const env: TRunArg;
  const wrkdir: TFileName; options: TExtProcessOptions): boolean;
begin
  result := false; // not implemented on Delphi POSIX
end;

function TExternalProcess.Write(const data: RawByteString): boolean;
begin
  result := false;
end;

function TExternalProcess.Write(p: pointer; len: PtrInt): boolean;
begin
  result := false;
end;

function TExternalProcess.WriteAndCloseStdin(const data: RawByteString): boolean;
begin
  result := false;
end;

procedure TExternalProcess.CloseStdin;
begin
end;

function TExternalProcess.ReadAvailable: RawByteString;
begin
  result := '';
end;

function TExternalProcess.HasDataAvailable: boolean;
begin
  result := false;
end;

function TExternalProcess.GetRunning: boolean;
begin
  result := false;
end;

function TExternalProcess.GetPid: cardinal;
begin
  result := 0;
end;

function TExternalProcess.Terminate(waitms: cardinal): boolean;
begin
  result := false;
end;

procedure TExternalProcess.Kill;
begin
end;

function TExternalProcess.WaitFor(waitms: cardinal): integer;
begin
  result := -1;
end;

destructor TExternalProcess.Destroy;
begin
  inherited Destroy;
end;

{$endif FPC}
