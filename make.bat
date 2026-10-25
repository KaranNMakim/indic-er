@echo off
REM Windows stand-in for GNU make: `make demo` / `make test` / `make eval`
REM from cmd.exe in this folder. From PowerShell use `.\make.bat demo`.
setlocal
where py >nul 2>nul
if %errorlevel%==0 (
  py "%~dp0tasks.py" %*
) else (
  python "%~dp0tasks.py" %*
)
endlocal
