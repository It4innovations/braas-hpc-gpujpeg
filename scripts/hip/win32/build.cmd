@echo off
REM Build script for GPUJPEG using HIP/ROCm clang++ directly
setlocal enabledelayedexpansion

set CLANG="C:\AMD\ROCm\5.7\bin\clang++.exe"
set ROCM_INC="C:\AMD\ROCm\5.7\include"
set ROOT_DIR=%~dp0
set BUILD_DIR=build_hipcc

REM Create build directory
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%BUILD_DIR%\obj_gpujpeg" mkdir "%BUILD_DIR%\obj_gpujpeg"
if not exist "%BUILD_DIR%\obj_create_rgb_gradient" mkdir "%BUILD_DIR%\obj_create_rgb_gradient"
if not exist "%BUILD_DIR%\obj_gpujpegtool" mkdir "%BUILD_DIR%\obj_gpujpegtool"
if not exist "%BUILD_DIR%\libgpujpeg" mkdir "%BUILD_DIR%\libgpujpeg"

REM Generate version header
echo Generating version header...
(
echo #ifndef GPUJPEG_VERSION_H
echo #define GPUJPEG_VERSION_H
echo #define GPUJPEG_VERSION_MAJOR 0
echo #define GPUJPEG_VERSION_MINOR 27
echo #define GPUJPEG_VERSION_PATCH 10
echo #define GPUJPEG_MK_VERSION_INT^(GPUJPEG_VERSION_MAJOR, GPUJPEG_VERSION_MINOR, GPUJPEG_VERSION_PATCH^) ^(^(GPUJPEG_VERSION_MAJOR^) ^<^< 16U ^| ^(GPUJPEG_VERSION_MINOR^) ^<^< 8U ^| ^(GPUJPEG_VERSION_PATCH^)^)
echo #define GPUJPEG_VERSION_INT GPUJPEG_MK_VERSION_INT^(GPUJPEG_VERSION_MAJOR, GPUJPEG_VERSION_MINOR, GPUJPEG_VERSION_PATCH^)
echo #define LIBGPUJPEG_API_VERSION ^(^(GPUJPEG_VERSION_MAJOR ^<^< 8U^) ^| GPUJPEG_VERSION_MINOR^)
echo #endif // GPUJPEG_VERSION_H
) > "%BUILD_DIR%\libgpujpeg\gpujpeg_version.h"

rem --offload-arch=gfx900
rem -D__HIP_PLATFORM_AMD__
REM Common compiler flags - shorter to avoid command line length issues
set CFLAGS=-std=c++20 -O2 -DGPUJPEG_USE_HIP -DHAVE_GPUJPEG_VERSION_H -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -DGPUJPEG_EXPORTS -x hip --offload-arch=gfx900
set IFLAGS=-I %ROOT_DIR% -I libgpujpeg -I . -I src -I %BUILD_DIR% -I %ROCM_INC%

echo Building GPUJPEG Library...

REM Compile source files one by one
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_dct_cpu.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_dct_cpu.obj
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_decoder.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_decoder.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_encoder.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_encoder.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_exif.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_exif.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_huffman_cpu_decoder.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_huffman_cpu_decoder.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_huffman_cpu_encoder.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_huffman_cpu_encoder.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_reader.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_reader.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_table.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_table.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_writer.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_writer.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\utils\image_delegate.cpp -o %BUILD_DIR%\obj_gpujpeg\image_delegate.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\utils\pam.cpp -o %BUILD_DIR%\obj_gpujpeg\pam.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\utils\y4m.cpp -o %BUILD_DIR%\obj_gpujpeg\y4m.obj

REM COMMON
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_common.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_common.obj

REM GPU
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_dct_gpu.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_dct_gpu.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_huffman_gpu_decoder.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_huffman_gpu_decoder.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_huffman_gpu_encoder.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_huffman_gpu_encoder.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_postprocessor.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_postprocessor.obj 
%CLANG% %CFLAGS% %IFLAGS% -c src\gpujpeg_preprocessor.cpp -o %BUILD_DIR%\obj_gpujpeg\gpujpeg_preprocessor.obj


echo.
echo Linking gpujpeg.dll...
%CLANG% -shared %BUILD_DIR%\obj_gpujpeg\*.obj -L"C:\AMD\ROCm\5.7\lib" -lamdhip64 -o %BUILD_DIR%\gpujpeg.dll

echo.
echo Building gpujpegtool...
%CLANG% %CFLAGS% %IFLAGS% -c src\main.cpp -o %BUILD_DIR%\obj_gpujpegtool\main.obj
%CLANG% %CFLAGS% %IFLAGS% -c src\utils\getopt.cpp -o %BUILD_DIR%\obj_gpujpegtool\getopt.obj
%CLANG% %BUILD_DIR%\obj_gpujpegtool\main.obj %BUILD_DIR%\obj_gpujpegtool\getopt.obj %BUILD_DIR%\obj_gpujpeg\*.obj -L"C:\AMD\ROCm\5.7\lib" -lamdhip64 -o %BUILD_DIR%\gpujpegtool.exe


echo.
echo Building create_rgb_gradient...
%CLANG% %CFLAGS% %IFLAGS% -c src\create_rgb_gradient.cpp -o %BUILD_DIR%\obj_create_rgb_gradient\create_rgb_gradient.obj
%CLANG% %BUILD_DIR%\obj_create_rgb_gradient\create_rgb_gradient.obj -o %BUILD_DIR%\create_rgb_gradient.exe


echo.
echo ============================================================================
echo Build Complete!
echo ============================================================================
echo Output directory: %BUILD_DIR%
echo   - gpujpeg.dll
echo   - gpujpegtool.exe
echo   - create_rgb_gradient.exe
echo.

endlocal
