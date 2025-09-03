
cmake_minimum_required(VERSION 3.16)
project(native_client_min LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Build CUDA NVRTC executor" ON)
option(ENABLE_OPENCL "Build OpenCL executor (stub)" OFF)
option(ENABLE_VULKAN "Build Vulkan executor (stub)" OFF)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Boost::beast comes with Boost.System
find_package(Boost REQUIRED COMPONENTS system asio)

add_executable(native_client
    src/main.cpp
    src/websocket_client.hpp
    src/websocket_client.cpp
    src/server_client.hpp
    src/server_client.cpp
    src/local_ws_server.hpp
    src/local_ws_server.cpp
    src/base64.hpp
    src/base64.cpp
    src/executors/iexecutor.hpp
    src/executors/binary_executor.hpp
    src/executors/binary_executor.cpp
    src/executors/cuda_executor.hpp
    src/executors/cuda_executor.cpp
    src/executors/opencl_executor.hpp
    src/executors/opencl_executor.cpp
    src/executors/vulkan_executor.hpp
    src/executors/vulkan_executor.cpp
)

target_include_directories(native_client PRIVATE src)

target_link_libraries(native_client PRIVATE
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
    Boost::system
    Boost::asio
    nlohmann_json::nlohmann_json
)

if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_libraries(native_client PRIVATE CUDA::cuda_driver CUDA::nvrtc)
        target_compile_definitions(native_client PRIVATE HAVE_CUDA=1)
    else()
        message(WARNING "CUDAToolkit not found, CUDA mode disabled")
    endif()
endif()

# Keep build small: no install rules, no extra targets


if(ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        target_include_directories(native_client PRIVATE ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(native_client PRIVATE ${OpenCL_LIBRARIES})
        target_compile_definitions(native_client PRIVATE HAVE_OPENCL=1)
    else()
        message(WARNING "OpenCL not found, OpenCL mode disabled")
    endif()
endif()

if(ENABLE_VULKAN)
    # find Vulkan normally
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        target_link_libraries(native_client PRIVATE Vulkan::Vulkan)
        target_compile_definitions(native_client PRIVATE HAVE_VULKAN=1)
    else()
        message(WARNING "Vulkan not found, Vulkan mode disabled")
    endif()

    # Prefer the vcpkg-provided config for shaderc (vcpkg installs unofficial-shaderc)
    message(STATUS "Trying to find shaderc via vcpkg/config (unofficial-shaderc)...")
    find_package(unofficial-shaderc CONFIG QUIET)

    # fallbacks: plain shaderc config, Vulkan component, pkg-config
    if(NOT unofficial-shaderc_FOUND)
        message(STATUS "unofficial-shaderc not found; trying shaderc (other vcpkg layouts)...")
        find_package(shaderc CONFIG QUIET)
    endif()

    if(NOT (unofficial-shaderc_FOUND OR shaderc_FOUND))
        message(STATUS "shaderc config mode not found; trying Vulkan component shaderc_combined (if available)...")
        find_package(Vulkan QUIET COMPONENTS shaderc_combined)
    endif()

    if(NOT (unofficial-shaderc_FOUND OR shaderc_FOUND OR Vulkan_shaderc_combined_FOUND))
        message(STATUS "Trying pkg-config (last-resort) for shaderc...")
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(SHADERC_PKG QUIET shaderc)
        endif()
    endif()

    # Link the right target / libs
    if(unofficial-shaderc_FOUND)
        message(STATUS "Linking shaderc via vcpkg config: unofficial::shaderc::shaderc")
        target_link_libraries(native_client PRIVATE unofficial::shaderc::shaderc)
        target_compile_definitions(native_client PRIVATE HAVE_SHADERC=1)
    elseif(shaderc_FOUND)
        message(STATUS "Linking shaderc via CMake config: shaderc::shaderc")
        target_link_libraries(native_client PRIVATE shaderc::shaderc)
        target_compile_definitions(native_client PRIVATE HAVE_SHADERC=1)
    elseif(DEFINED Vulkan_shaderc_combined_FOUND AND Vulkan_shaderc_combined_FOUND)
        message(STATUS "Linking Vulkan::shaderc_combined")
        target_link_libraries(native_client PRIVATE Vulkan::shaderc_combined)
        target_compile_definitions(native_client PRIVATE HAVE_SHADERC=1)
    elseif(SHADERC_PKG_FOUND)
        message(STATUS "Linking shaderc via pkg-config libs")
        target_include_directories(native_client PRIVATE ${SHADERC_PKG_INCLUDE_DIRS})
        target_link_libraries(native_client PRIVATE ${SHADERC_PKG_LIBRARIES})
        target_compile_definitions(native_client PRIVATE HAVE_SHADERC=1)
    else()
        message(WARNING "shaderc not found; will expect SPIR-V binaries only")
    endif()
endif()
