@echo off
rem Build helper for Windows. Adds MSYS2 MinGW gcc to PATH for this shell only.
rem Usage:
rem   build.bat              -> builds libvc.dll
rem   build.bat test         -> builds all test_*.exe
rem   build.bat clean        -> removes generated binaries

set "MINGW=C:\tools\msys64\mingw64\bin"
if not exist "%MINGW%\gcc.exe" (
    echo ERROR: gcc not found at %MINGW%
    echo Install MSYS2 and run `pacman -S mingw-w64-x86_64-gcc`, or edit this script.
    exit /b 1
)
set "PATH=%MINGW%;%PATH%"

if "%1"=="" (
    gcc --version | findstr /b "gcc"
    mingw32-make libvc.dll
    if errorlevel 1 exit /b 1
    if not exist libopenblas.dll copy /y openblas\bin\libopenblas.dll . >nul
    echo.
    echo built: libvc.dll + libopenblas.dll ^(copied next to libvc.dll^)
    goto :eof
)

if "%1"=="test" (
    mingw32-make all
    if errorlevel 1 exit /b 1
    if not exist libopenblas.dll copy /y openblas\bin\libopenblas.dll . >nul
    goto :eof
)

if "%1"=="bench" (
    mingw32-make bench.exe
    if errorlevel 1 exit /b 1
    if not exist libopenblas.dll copy /y openblas\bin\libopenblas.dll . >nul
    goto :eof
)

if "%1"=="clean" (
    mingw32-make clean
    goto :eof
)

echo Unknown target: %1
exit /b 1
