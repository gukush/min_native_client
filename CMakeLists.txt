cmake_minimum_required(VERSION 3.16)
project(native_client_min LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Build CUDA NVRTC executor" ON)
option(ENABLE_OPENCL "Build OpenCL executor (stub)" ON)
option(ENABLE_VULKAN "Build Vulkan executor (stub)" OFF)
option(ENABLE_LUA "Enable Lua host scripting via sol2 + LuaJIT" OFF)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Boost::beast comes with Boost.System
find_package(Boost REQUIRED COMPONENTS system asio)

add_executable(native_client
    src/main.cpp
    src/websocket_client.hpp
    src/websocket_client.cpp

    # Renamed client
    src/orchestrator_client.hpp
    src/orchestrator_client.cpp

    src/base64.hpp
    src/base64.cpp

    # executors
    src/executors/iexecutor.hpp
    src/executors/binary_executor.hpp
    src/executors/binary_executor.cpp
    src/executors/cuda_executor.hpp
    src/executors/cuda_executor.cpp
    src/executors/opencl_executor.hpp
    src/executors/opencl_executor.cpp
    src/executors/vulkan_executor.hpp
    src/executors/vulkan_executor.cpp

    # Lua host (conditional)
    src/lua_host.hpp
    src/lua_host.cpp
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

# ---------------- CUDA ----------------
if(ENABLE_CUDA)
    # Try to find CUDA toolkit
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        # Check if the required targets exist before linking
        if(TARGET CUDA::cuda_driver AND TARGET CUDA::nvrtc)
            target_link_libraries(native_client PRIVATE CUDA::cuda_driver CUDA::nvrtc)
            target_compile_definitions(native_client PRIVATE HAVE_CUDA=1)
            message(STATUS "CUDA: Using CUDAToolkit imported targets (CUDA::cuda_driver, CUDA::nvrtc)")
        else()
            # Fallback: try to find CUDA libraries manually
            message(STATUS "CUDA: CUDAToolkit found but imported targets not available, trying manual detection")

            # Find CUDA libraries manually
            find_library(CUDA_DRIVER_LIBRARY NAMES cuda PATHS ${CUDAToolkit_LIBRARY_DIR})
            find_library(CUDA_NVRTC_LIBRARY NAMES nvrtc PATHS ${CUDAToolkit_LIBRARY_DIR})

            if(CUDA_DRIVER_LIBRARY AND CUDA_NVRTC_LIBRARY)
                target_link_libraries(native_client PRIVATE ${CUDA_DRIVER_LIBRARY} ${CUDA_NVRTC_LIBRARY})
                target_include_directories(native_client PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
                target_compile_definitions(native_client PRIVATE HAVE_CUDA=1)
                message(STATUS "CUDA: Using manual library detection (${CUDA_DRIVER_LIBRARY}, ${CUDA_NVRTC_LIBRARY})")
            else()
                message(WARNING "CUDA libraries not found, CUDA support disabled")
            endif()
        endif()
    else()
        message(WARNING "CUDAToolkit not found, CUDA support disabled")
    endif()
endif()

# ---------------- OpenCL ----------------
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

# ---------------- Vulkan (+ shaderc) ----------------
if(ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        target_link_libraries(native_client PRIVATE Vulkan::Vulkan)
        target_compile_definitions(native_client PRIVATE HAVE_VULKAN=1)
    else()
        message(WARNING "Vulkan not found, Vulkan mode disabled")
    endif()

    message(STATUS "Trying to find shaderc via vcpkg/config (unofficial-shaderc)...")
    find_package(unofficial-shaderc CONFIG QUIET)

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

# ---------------- Lua host (sol2 + LuaJIT) ----------------
if(ENABLE_LUA)
    # sol2 itself (header-only target)
    find_package(sol2 CONFIG REQUIRED)  # vcpkg: sol2::sol2

    # Try to find LuaJIT (preferred) or Lua (any 5.x) in several ways.
    set(_LUA_FOUND FALSE)

    # -- Config-mode (vcpkg-style) --
    find_package(LuaJIT CONFIG QUIET)         # exports luajit::luajit on vcpkg
    if(LuaJIT_FOUND)
        set(_LUA_FOUND TRUE)
        set(_LUA_TGT luajit::luajit)
    endif()

    if(NOT _LUA_FOUND)
        # Some ports export "lua::lua" or variables via FindLua.cmake
        find_package(lua CONFIG QUIET)        # non-standard, but cheap to try
        if(lua_FOUND AND TARGET lua::lua)
            set(_LUA_FOUND TRUE)
            set(_LUA_TGT lua::lua)
        endif()
    endif()

    if(NOT _LUA_FOUND)
        # Built-in FindLua (may locate system Lua 5.x)
        find_package(Lua QUIET)               # sets LUA_INCLUDE_DIR / LUA_LIBRARIES
        if(Lua_FOUND)
            set(_LUA_FOUND TRUE)
        endif()
    endif()

    if(NOT _LUA_FOUND)
        # -- pkg-config fallback --
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(LUAJIT_PKG QUIET luajit)
            if(LUAJIT_PKG_FOUND)
                set(_LUA_FOUND TRUE)
                set(_LUA_PKG_INCS ${LUAJIT_PKG_INCLUDE_DIRS})
                set(_LUA_PKG_LIBS ${LUAJIT_PKG_LIBRARIES})
            endif()
        endif()
    endif()

    if(NOT _LUA_FOUND)
        # -- Manual last-resort (typical vcpkg locations) --
        # Covers headers living in subdirs like: luajit-2.1/, luajit/, lua5.4/, etc.
        set(_VCPKG_INC_HINTS
            "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}/include"
            "${CMAKE_SOURCE_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include"
            "/opt/vcpkg/installed/${VCPKG_TARGET_TRIPLET}/include")
        set(_VCPKG_LIB_HINTS
            "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}/lib"
            "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}/debug/lib"
            "/opt/vcpkg/installed/${VCPKG_TARGET_TRIPLET}/lib"
            "/opt/vcpkg/installed/${VCPKG_TARGET_TRIPLET}/debug/lib")

        find_path(LUA_INCLUDE_DIR
            NAMES lua.h luajit.h
            HINTS ${_VCPKG_INC_HINTS}
            PATH_SUFFIXES luajit-2.1 luajit lua5.4 lua5.3 lua5.2 lua5.1 lua)

        find_library(LUA_LIBRARY
            NAMES luajit-5.1 luajit lua5.4 lua5.3 lua5.2 lua5.1 lua
            HINTS ${_VCPKG_LIB_HINTS})

        if(LUA_INCLUDE_DIR AND LUA_LIBRARY)
            set(_LUA_FOUND TRUE)
        endif()
    endif()

    # ---- Wire everything up ----
    if(TARGET sol2::sol2)
        target_link_libraries(native_client PRIVATE sol2::sol2)
    endif()

    if(_LUA_FOUND)
        # Prefer a proper target if we have one
        if(DEFINED _LUA_TGT AND TARGET ${_LUA_TGT})
            target_link_libraries(native_client PRIVATE ${_LUA_TGT})
            # Try to harvest include dirs from the target
            get_target_property(_LUA_INC_FROM_TGT ${_LUA_TGT} INTERFACE_INCLUDE_DIRECTORIES)
            if(_LUA_INC_FROM_TGT)
                target_include_directories(native_client PRIVATE ${_LUA_INC_FROM_TGT})
            endif()
        elseif(Lua_FOUND)
            # From FindLua.cmake
            target_include_directories(native_client PRIVATE ${LUA_INCLUDE_DIR})
            target_link_libraries(native_client PRIVATE ${LUA_LIBRARIES})
        elseif(LUAJIT_PKG_FOUND)
            # From pkg-config
            target_include_directories(native_client PRIVATE ${_LUA_PKG_INCS})
            target_link_libraries(native_client PRIVATE ${_LUA_PKG_LIBS})
        elseif(LUA_INCLUDE_DIR AND LUA_LIBRARY)
            # Manual fallback
            target_include_directories(native_client PRIVATE ${LUA_INCLUDE_DIR})
            target_link_libraries(native_client PRIVATE ${LUA_LIBRARY})
        endif()

        target_compile_definitions(native_client PRIVATE HAVE_LUA=1)
        # If we specifically matched LuaJIT, add HAVE_LUAJIT too
        if(LuaJIT_FOUND OR LUAJIT_PKG_FOUND OR (LUA_LIBRARY MATCHES "luajit"))
            target_compile_definitions(native_client PRIVATE HAVE_LUAJIT=1)
        endif()
    else()
        message(FATAL_ERROR
            "LuaJIT/Lua not found. Install via vcpkg: `vcpkg install sol2 luajit` "
            "or provide -DLUA_INCLUDE_DIR and -DLUA_LIBRARY.")
    endif()
endif()

# Keep build small: no install rules
