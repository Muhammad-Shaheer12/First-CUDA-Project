@echo off
setlocal

for %%I in ("%ProgramFiles(x86)%") do set PF86=%%~sI
set VSWHERE=%PF86%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" (
  echo ERROR: vswhere.exe not found. Install Visual Studio Build Tools 2022.
  exit /b 1
)

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSINSTALL=%%I

for %%I in ("%VSINSTALL%") do set VSINSTALL=%%~sI

if "%VSINSTALL%"=="" (
  echo ERROR: Visual Studio C++ Build Tools not found.
  exit /b 1
)

set VCVARS=%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
  echo ERROR: vcvars64.bat not found at: %VCVARS%
  exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b 1

cd /d "%~dp0"

mingw32-make clean
if errorlevel 1 exit /b 1

mingw32-make
if errorlevel 1 exit /b 1

echo Build OK.
