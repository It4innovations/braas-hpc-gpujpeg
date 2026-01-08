cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
# change version also in configure.ac

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options for GPU backend selection
option(GPUJPEG_USE_HIP "Build with HIP backend for AMD GPUs" OFF)
option(GPUJPEG_USE_SYCL "Build with SYCL backend for Intel/other GPUs" OFF)

# Determine project languages based on GPU backend
if(GPUJPEG_USE_HIP)
    project(gpujpeg VERSION 0.27.10 LANGUAGES CXX)
elseif(GPUJPEG_USE_SYCL)
    project(gpujpeg VERSION 0.27.10 LANGUAGES CXX)
else()
    project(gpujpeg VERSION 0.27.10 LANGUAGES CXX CUDA)
endif()

# options
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for (needs CMake 3.18)")
option(HUFFMAN_GPU_CONST_TABLES "Enable Huffman GPU decoder quick const tables" OFF)

# Configure GPU Backend
if(GPUJPEG_USE_HIP)
    set(GPU_BACKEND "HIP")
    message(STATUS "Building GPUJPEG with HIP backend for AMD GPUs")
    
    # Enable HIP language - CMake will find the appropriate Clang compiler
    # Don't set CMAKE_HIP_COMPILER to hipcc wrapper, let CMake find Clang
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
        enable_language(HIP)
    else()
        message(WARNING "CMake 3.21+ recommended for HIP support. Using CXX with HIP flags.")
        # Fall back to C++ compilation with HIP flags for older CMake
        find_program(HIPCC_EXECUTABLE NAMES hipcc REQUIRED)
        if(NOT HIPCC_EXECUTABLE)
            message(FATAL_ERROR "hipcc not found. Please install ROCm.")
        endif()
    endif()
    
elseif(GPUJPEG_USE_SYCL)
    set(GPU_BACKEND "SYCL")
    message(STATUS "Building GPUJPEG with SYCL backend for Intel GPUs")
    
    # Find Intel DPC++/SYCL compiler
    if(NOT CMAKE_CXX_COMPILER MATCHES "icpx|dpcpp")
        message(WARNING "SYCL backend requires Intel DPC++ compiler (icpx). "
                        "Please set CMAKE_CXX_COMPILER to icpx or dpcpp.")
    endif()
    
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fno-omit-frame-pointer")
    endif()
    
else()
    set(GPU_BACKEND "CUDA")
    message(STATUS "Building GPUJPEG with CUDA backend (default)")

    if (CMAKE_CUDA_ARCHITECTURES MATCHES OFF)
        message(NOTICE "You may want to set CUDA architecture, suggested `-DCMAKE_CUDA_ARCHITECTURES=native`")
    endif()    
endif()

if(WIN32)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/${CMAKE_PROJECT_NAME}")
    endif()
    string(REPLACE "\\" "/" CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(NOTICE "Suggesting setting build type expicitly (eg. "
        "\"-DCMAKE_BUILD_TYPE=Release\" or \"Debug\") for single-config "
        "generators.")
endif()

cmake_policy(SET CMP0010 NEW) # error on undefined variable

set(CPACK_PACKAGE_INSTALL_DIRECTORY "gpujpeg") # may be removed when version will be explicitly defined
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
include(CPack)

include(CTest)

execute_process(COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions(-DGIT_REV=\"${GIT_REV}\")

if(NOT BUILD_SHARED_LIBS)
    add_definitions("-DGPUJPEG_STATIC")
    set(PC_CFLAGS "${PC_CFLAGS} -DGPUJPEG_STATIC")
    set(PC_LIBS "${PC_LIBS} -lcudart")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libgpujpeg/gpujpeg_version.h.cmakein
               ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg/gpujpeg_version.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libgpujpeg.pc.cmakein
               ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg.pc @ONLY)
add_definitions("-DHAVE_GPUJPEG_VERSION_H")
add_definitions("-DGPUJPEG_INTERNAL_BUILD")

# Include cmake modules in "cmake" directory (FindGLFW.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# Common settings
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(COMPILED_OPTIONS)

if (MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /W4")
    if (MSVC_VERSION GREATER_EQUAL 1914)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Zc:__cplusplus")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /Zc:__cplusplus")
        # Suppress C4127 warnings for CUDA code (conditional expression is constant in template code)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /wd4127")
    endif()
else()
    if(GPU_BACKEND STREQUAL "HIP")
        # For HIP, avoid -pedantic to prevent assembler flag issues
        # Use DWARF-2 to avoid compatibility issues with older GNU assemblers
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -gdwarf-2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-2")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra")
endif()

if(HUFFMAN_GPU_CONST_TABLES)
    add_definitions("-DHUFFMAN_GPU_CONST_TABLES")
    set(COMPILED_OPTIONS "${COMPILED_OPTIONS} Huffman_const_tables")
endif()

message(STATUS "Configured options: ${COMPILED_OPTIONS}")

# GPUJPEG library
set(H_FILES
    libgpujpeg/gpujpeg_common.h
    libgpujpeg/gpujpeg_decoder.h
    libgpujpeg/gpujpeg_encoder.h
    libgpujpeg/gpujpeg_type.h
    libgpujpeg/gpujpeg.h
    ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg/gpujpeg_version.h
)

# Common source files (.cpp files)
set(GPUJPEG_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_dct_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_exif.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_huffman_cpu_decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_huffman_cpu_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_postprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_preprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_table.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_writer.cpp
)
list(REMOVE_ITEM GPUJPEG_COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM GPUJPEG_COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/getopt.cpp")

# For CUDA backend, remove GPU-related .cpp files (these are wrappers for HIP/SYCL only)
# For HIP/SYCL backends, keep these .cpp files as they are the actual implementation
if(GPU_BACKEND STREQUAL "CUDA")
    list(REMOVE_ITEM GPUJPEG_COMMON_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_dct_gpu.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_huffman_gpu_decoder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_huffman_gpu_encoder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_postprocessor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_preprocessor.cpp"
    )
endif()

# For non-SYCL backends, always remove SYCL-specific compatibility layer
if(NOT GPU_BACKEND STREQUAL "SYCL")
    list(REMOVE_ITEM GPUJPEG_COMMON_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpujpeg_device_compat_sycl.cpp"
    )
endif()

# Add utility source files
list(APPEND GPUJPEG_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/image_delegate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/pam.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/y4m.cpp
)

# GPU kernel sources - different per backend
if(GPU_BACKEND STREQUAL "CUDA")
    set(GPUJPEG_GPU_SOURCES
        src/gpujpeg_dct_gpu.cu
        src/gpujpeg_huffman_gpu_decoder.cu
        src/gpujpeg_huffman_gpu_encoder.cu
        src/gpujpeg_postprocessor.cu
        src/gpujpeg_preprocessor.cu
    )
    
    # CUDA-specific compiler flags
    if(NOT WIN32)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
    endif()
    
    # Set CUDA architecture (adjust as needed)
    if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES MATCHES OFF)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
        message(STATUS "Setting default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    
elseif(GPU_BACKEND STREQUAL "HIP")
    set(GPUJPEG_GPU_SOURCES
        src/gpujpeg_dct_gpu.cpp
        src/gpujpeg_huffman_gpu_decoder.cpp
        src/gpujpeg_huffman_gpu_encoder.cpp
        src/gpujpeg_postprocessor.cpp
        src/gpujpeg_preprocessor.cpp
    )
    
    # HIP-specific compiler flags
    if(NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
    
elseif(GPU_BACKEND STREQUAL "SYCL")
    set(GPUJPEG_GPU_SOURCES
        src/gpujpeg_device_compat_sycl.cpp
        src/gpujpeg_dct_gpu.cpp
        src/gpujpeg_huffman_gpu_decoder.cpp
        src/gpujpeg_huffman_gpu_encoder.cpp
        src/gpujpeg_postprocessor.cpp
        src/gpujpeg_preprocessor.cpp
    )
    
    # SYCL-specific compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    if(NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
endif()

add_library(gpujpeg ${H_FILES} ${GPUJPEG_COMMON_SOURCES} ${GPUJPEG_GPU_SOURCES})

# Set language for GPU source files
if(GPU_BACKEND STREQUAL "HIP")
    # Mark GPU sources as HIP language so they're compiled with hipcc
    set_source_files_properties(${GPUJPEG_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

# Add compile definitions for GPU backend to all source files
if(GPU_BACKEND STREQUAL "CUDA")
    target_compile_definitions(${PROJECT_NAME} PRIVATE GPUJPEG_USE_CUDA)
elseif(GPU_BACKEND STREQUAL "HIP")
    target_compile_definitions(${PROJECT_NAME} PRIVATE GPUJPEG_USE_HIP)
elseif(GPU_BACKEND STREQUAL "SYCL")
    target_compile_definitions(${PROJECT_NAME} PRIVATE GPUJPEG_USE_SYCL)
endif()

# Include directories based on backend
if(GPU_BACKEND STREQUAL "CUDA")
    target_include_directories(${PROJECT_NAME}
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
elseif(GPU_BACKEND STREQUAL "HIP")
    # HIP include directories - find ROCm installation
    if(NOT DEFINED ROCM_PATH AND DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    endif()
    if(NOT DEFINED ROCM_PATH)
        # Try common ROCm installation paths
        if(EXISTS "/opt/rocm")
            set(ROCM_PATH "/opt/rocm")
        elseif(EXISTS "/opt/rocm-6.0.2")
            set(ROCM_PATH "/opt/rocm-6.0.2")
        endif()
    endif()
    
    if(ROCM_PATH)
        message(STATUS "Using ROCm path: ${ROCM_PATH}")
        target_include_directories(${PROJECT_NAME}
            PRIVATE 
                ${ROCM_PATH}/include
                ${ROCM_PATH}/hip/include
        )
    elseif(CMAKE_HIP_TOOLKIT_INCLUDE_DIRECTORIES)
        target_include_directories(${PROJECT_NAME}
            PRIVATE ${CMAKE_HIP_TOOLKIT_INCLUDE_DIRECTORIES}
        )
    else()
        message(WARNING "Could not determine HIP include directories. You may need to set ROCM_PATH.")
    endif()
elseif(GPU_BACKEND STREQUAL "SYCL")
    # SYCL include directories are handled by compiler
endif()

# Link GPU runtime libraries
if(GPU_BACKEND STREQUAL "CUDA")
    if(UNIX)
        target_link_libraries(${PROJECT_NAME} m)
    endif()
    
elseif(GPU_BACKEND STREQUAL "HIP")
    # HIP libraries are automatically linked when using HIP language
    # No need to explicitly find_package(hip) when using enable_language(HIP)
    if(UNIX)
        target_link_libraries(${PROJECT_NAME} m)
    endif()
    
elseif(GPU_BACKEND STREQUAL "SYCL")
    # SYCL runtime is linked via compiler flags
    if(UNIX)
        target_link_libraries(${PROJECT_NAME} m)
    endif()
endif()

# GPUJPEG command-line application
if(NOT IS_SUBMODULE)
    set(GPUJPEGTOOL_SOURCES src/main.cpp)
    if(WIN32)
        list(APPEND GPUJPEGTOOL_SOURCES src/utils/getopt.cpp)
    endif()
    add_executable(gpujpegtool ${GPUJPEGTOOL_SOURCES})
    target_link_libraries(gpujpegtool gpujpeg)

    if(WIN32)
        target_compile_definitions(gpujpegtool PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()

    install(TARGETS gpujpegtool DESTINATION bin)
    
    add_executable(create_rgb_gradient src/create_rgb_gradient.cpp)
    install(TARGETS create_rgb_gradient DESTINATION bin)
endif()

install(TARGETS ${PROJECT_NAME}
        EXPORT GPUJPEGTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/libgpujpeg
        RUNTIME DESTINATION bin)

install(FILES AUTHORS CONTRIBUTING.md COPYING FAQ.md INSTALL NEWS.md README.md
        DESTINATION doc)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg.pc DESTINATION share/pkgconfig)

# CMake config
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfigVersion.cmake"
    VERSION ${Upstream_VERSION}
    COMPATIBILITY AnyNewerVersion)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/GPUJPEG"
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
    NO_SET_AND_CHECK_MACRO)
install(EXPORT GPUJPEGTargets
    DESTINATION lib/cmake/GPUJPEG)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfigVersion.cmake
    DESTINATION lib/cmake/GPUJPEG)

# Print configuration summary
message(STATUS "")
message(STATUS "GPUJPEG Configuration Summary:")
message(STATUS "  GPU Backend: ${GPU_BACKEND}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
if(GPU_BACKEND STREQUAL "CUDA")
    message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
elseif(GPU_BACKEND STREQUAL "HIP")
    message(STATUS "  HIP Compiler: ${CMAKE_HIP_COMPILER}")
    if(CMAKE_HIP_ARCHITECTURES)
        message(STATUS "  HIP Architectures: ${CMAKE_HIP_ARCHITECTURES}")
    endif()
elseif(GPU_BACKEND STREQUAL "SYCL")
    message(STATUS "  SYCL Compiler: ${CMAKE_CXX_COMPILER}")
endif()
message(STATUS "")

# vim: sw=4
