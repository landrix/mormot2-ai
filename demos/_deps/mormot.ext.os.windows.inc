// mormot.ext.os.windows.inc - Windows implementation of TExternalProcess

{ background reader thread }

function ExtProcessReaderThread(p: pointer):
  {$ifdef FPC} PtrInt {$else} Integer {$endif};
var
  proc: TExternalProcess;
  tmp: TBuffer4K;
  new: RawByteString;
  pending, n: cardinal;
begin
  result := 0;
  proc := TExternalProcess(p);
  try
    repeat
      if proc.fStdoutRead = 0 then
        break;
      pending := 0;
      if not PeekNamedPipe(proc.fStdoutRead, nil, 0, nil, @pending, nil) then
        break;
      if pending = 0 then
      begin
        SleepHiRes(10);
        continue;
      end;
      if pending > SizeOf(tmp) then
        pending := SizeOf(tmp);
      n := 0;
      if not ReadFile(proc.fStdoutRead, tmp, pending, n, nil) or
         (n = 0) then
        break;
      proc.fOutputSafe.Lock;
      try
        SetLength(proc.fOutput, Length(proc.fOutput) + PtrInt(n));
        MoveFast(tmp, PByteArray(proc.fOutput)[Length(proc.fOutput) - PtrInt(n)], n);
      finally
        proc.fOutputSafe.UnLock;
      end;
      if Assigned(proc.fOnOutput) then
      begin
        FastSetRawByteString(new, @tmp, n);
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
  startupinfo: TStartupInfoW;
  processinfo: TProcessInformation;
  procsec, pipesec: TSecurityAttributes;
  wcmd, wenv, wpath: SynUnicode;
  in_rd, out_wr: THandle;
  flags: cardinal;
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
  // security attributes for inheritable handles
  FillCharFast(procsec, SizeOf(procsec), 0);
  procsec.nLength := SizeOf(procsec);
  procsec.bInheritHandle := true;
  FillCharFast(pipesec, SizeOf(pipesec), 0);
  pipesec.nLength := SizeOf(pipesec);
  pipesec.bInheritHandle := true;
  // create stdin pipe
  if not CreatePipe(in_rd, fStdinWrite, @pipesec, 0) then
    exit;
  SetHandleInformation(fStdinWrite, HANDLE_FLAG_INHERIT, 0);
  // create stdout pipe
  if not CreatePipe(fStdoutRead, out_wr, @pipesec, 0) then
  begin
    CloseHandle(in_rd);
    CloseHandle(fStdinWrite);
    fStdinWrite := 0;
    exit;
  end;
  SetHandleInformation(fStdoutRead, HANDLE_FLAG_INHERIT, 0);
  // prepare startup info
  FillCharFast(startupinfo, SizeOf(startupinfo), 0);
  startupinfo.cb := SizeOf(startupinfo);
  startupinfo.hStdInput := in_rd;
  startupinfo.hStdOutput := out_wr;
  startupinfo.hStdError := out_wr;
  startupinfo.dwFlags := STARTF_USESTDHANDLES or STARTF_USESHOWWINDOW;
  startupinfo.wShowWindow := SW_HIDE;
  // prepare command and path
  wcmd := SynUnicode(cmd);
  UniqueString(wcmd);
  wpath := SynUnicode(wrkdir);
  if env <> '' then
  begin
    wenv := SynUnicode(env);
    UniqueString(wenv);
  end;
  // process creation flags
  flags := CREATE_UNICODE_ENVIRONMENT or CREATE_DEFAULT_ERROR_MODE;
  if epoNewConsole in options then
    flags := flags or CREATE_NEW_CONSOLE
  else
    flags := flags or CREATE_NEW_PROCESS_GROUP; // default: own group for isolation
  if epoNewProcessGroup in options then
    flags := flags or CREATE_NEW_PROCESS_GROUP;
  if epoJobCloseChildren in options then
    flags := flags or CREATE_BREAKAWAY_FROM_JOB;
  // create the process
  FillCharFast(processinfo, SizeOf(processinfo), 0);
  result := CreateProcessW(nil, pointer(wcmd), @procsec, @procsec, true,
    flags, pointer({%H-}wenv), pointer(wpath), startupinfo, processinfo);
  // close child-side pipe ends
  CloseHandle(in_rd);
  CloseHandle(out_wr);
  if not result then
  begin
    CloseHandle(fStdinWrite);
    CloseHandle(fStdoutRead);
    fStdinWrite := 0;
    fStdoutRead := 0;
    exit;
  end;
  fProcess := processinfo.hProcess;
  fProcessThread := processinfo.hThread;
  fPid := processinfo.dwProcessID;
  fStarted := true;
  // optional job object
  if epoJobCloseChildren in options then
  begin
    fJob := CreateJobToClose(fPid, 'ExtProcess', true);
    if (fJob <> 0) and
       not AssignJobToProcess(fJob, fProcess, 'ExtProcess') then
    begin
      CloseHandle(fJob);
      fJob := 0;
    end;
  end;
  // start background reader thread
  {$ifdef FPC}
  BeginThread(@ExtProcessReaderThread, self, fReaderTid);
  {$else}
  fReaderHandle := BeginThread(nil, 0, @ExtProcessReaderThread, self, 0, fReaderTid);
  {$endif FPC}
end;

function TExternalProcess.Write(const data: RawByteString): boolean;
begin
  result := Write(pointer(data), length(data));
end;

function TExternalProcess.Write(p: pointer; len: PtrInt): boolean;
begin
  result := (fStdinWrite <> 0) and
            (p <> nil) and
            (len > 0) and
            FileWriteAll(fStdinWrite, p, len);
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
    CloseHandle(fStdinWrite);
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
  c: cardinal;
begin
  result := false;
  if not fStarted or
     fTerminated then
    exit;
  if not GetExitCodeProcess(fProcess, c) then
    exit;
  if c = STILL_ACTIVE then
    result := true
  else
  begin
    fExitCode := c;
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
  c: cardinal;
begin
  result := not Running; // Running also captures fExitCode on natural exit
  if result then
    exit;
  endtix := GetTickCount64 + waitms;
  // try WM_QUIT on all threads
  QuitProcess(fPid, 0);
  while GetTickCount64 < endtix do
  begin
    if GetExitCodeProcess(fProcess, c) and
       (c <> STILL_ACTIVE) then
    begin
      fExitCode := c;
      fTerminated := true;
      result := true;
      exit;
    end;
    SleepHiRes(50);
  end;
  // hard kill
  Kill;
  result := true;
end;

procedure TExternalProcess.Kill;
begin
  if fStarted and
     not fTerminated then
  begin
    TerminateProcess(fProcess, 255);
    fExitCode := 255;
    fTerminated := true;
  end;
end;

function TExternalProcess.WaitFor(waitms: cardinal): integer;
var
  endtix: Int64;
  c: cardinal;
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
    endtix := 0
  else
    endtix := GetTickCount64 + waitms;
  repeat
    if GetExitCodeProcess(fProcess, c) and
       (c <> STILL_ACTIVE) then
    begin
      fExitCode := c;
      fTerminated := true;
      result := fExitCode;
      exit;
    end;
    SleepHiRes(50);
  until (endtix <> 0) and
        (GetTickCount64 > endtix);
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
    // close stdout pipe to force reader thread exit
    if fStdoutRead <> 0 then
    begin
      CloseHandle(fStdoutRead);
      fStdoutRead := 0;
    end;
    // wait for reader thread
    endtix := GetTickCount64 + 3000;
    while not fReaderFinished and
          (GetTickCount64 < endtix) do
      SleepHiRes(10);
    if fReaderHandle <> 0 then
      CloseHandle(fReaderHandle);
    if fProcess <> 0 then
      CloseHandle(fProcess);
    if fProcessThread <> 0 then
      CloseHandle(fProcessThread);
    if fJob <> 0 then
      CloseHandle(fJob);
  end;
  inherited Destroy;
end;
