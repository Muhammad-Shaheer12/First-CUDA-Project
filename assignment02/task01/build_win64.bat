@echo off
setlocal

for %%I in ("%ProgramFiles(x86)%") do set PF86=%%~sI
set VSWHERE=%PF86%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" (
  echo ERROR: vswhere.exe not found.
  exit /b 1
)

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSINSTALL=%%I
for %%I in ("%VSINSTALL%") do set VSINSTALL=%%~sI

set VCVARS=%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
  echo ERROR: vcvars64.bat not found at: %VCVARS%
  exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b 1

cd /d "%~dp0"
nvcc -std=c++17 -O2 -Xcompiler="/W4 /wd4211" src/device_query.cu -o device_query.exe
if errorlevel 1 exit /b 1

echo Build OK. Run: .\device_query.exe
