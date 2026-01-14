@echo off
REM Build script for SYCL test (Windows)

setlocal enabledelayedexpansion

echo Building SYCL preprocessor kernel test...

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Get project root (parent of script directory)
for %%I in ("%SCRIPT_DIR%") do set PROJECT_ROOT=%%~dpI
set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

REM Source and output paths
set SOURCE_FILE=%SCRIPT_DIR%\test_sycl.cpp
set COMPAT_SOURCE=%PROJECT_ROOT%\src\gpujpeg_device_compat_sycl.cpp
set OUTPUT_BINARY=%SCRIPT_DIR%\test_sycl.exe

REM Include directories
set INCLUDE_DIRS=-I%PROJECT_ROOT% -I%PROJECT_ROOT%\libgpujpeg -I%PROJECT_ROOT%\src

REM Compiler flags
set SYCL_FLAGS=-fsycl
set CXX_FLAGS=-std=c++17 -O2 -Wall
set DEFINES=-DGPUJPEG_USE_SYCL -DGPUJPEG_INTERNAL_BUILD -DTEST_MAIN

REM Build with icpx (Intel oneAPI DPC++/SYCL compiler)
echo Compiler: icpx
echo Source: %SOURCE_FILE%
echo SYCL Compat: %COMPAT_SOURCE%
echo Output: %OUTPUT_BINARY%
echo.

icpx %SYCL_FLAGS% %CXX_FLAGS% %DEFINES% %INCLUDE_DIRS% "%SOURCE_FILE%" "%COMPAT_SOURCE%" -o "%OUTPUT_BINARY%"

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo Build successful!
echo Executable: %OUTPUT_BINARY%
echo.
echo To run the test, execute:
echo   %OUTPUT_BINARY%

set ONEAPI_DEVICE_SELECTOR=level_zero:gpu
REM set ONEAPI_DEVICE_SELECTOR=*:cpu

echo Running SYCL preprocessor kernel test...
"%OUTPUT_BINARY%"

endlocal
