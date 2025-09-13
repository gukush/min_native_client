#include "cuda_executor.hpp"
#include "../base64.hpp"
#include <iostream>
#include <chrono>
#include <cstring>

#ifdef HAVE_CUDA

// GPU timing accumulators (thread-local)
static thread_local double g_cuda_compile_ms = 0.0;
static thread_local double g_cuda_h2d_ms = 0.0;
static thread_local double g_cuda_kernel_ms = 0.0;
static thread_local double g_cuda_d2h_ms = 0.0;

// Static member definition
KernelCache<CudaExecutor::CudaKernel> CudaExecutor::kernel_cache_;

bool CudaExecutor::check(CUresult res, const char* what){
    if(res==CUDA_SUCCESS) return true;
    const char* errName=nullptr; const char* errStr=nullptr;
    cuGetErrorName(res, &errName); cuGetErrorString(res, &errStr);
    std::cerr << what << " failed: " << (errName?errName:"?") << " - " << (errStr?errStr:"?") << std::endl;
    return false;
}

bool CudaExecutor::checkNVRTC(nvrtcResult res, const char* what){
    if(res==NVRTC_SUCCESS) return true;
    std::cerr << what << " failed: " << nvrtcGetErrorString(res) << std::endl;
    return false;
}

bool CudaExecutor::ensureDriver(){
    static bool inited=false;
    static bool ok=false;
    if(!inited){
        ok = (cuInit(0)==CUDA_SUCCESS);
        inited=true;
    }
    return ok;
}

bool CudaExecutor::initialize(const json& cfg){
    (void)cfg;
    if(!ensureDriver()){
        std::cerr << "[CUDA] cuInit failed\n";
        return false;
    }
    int count=0;
    if(cuDeviceGetCount(&count)!=CUDA_SUCCESS || count<=0){
        std::cerr << "[CUDA] No CUDA device found\n";
        return false;
    }
    if(cuDeviceGet(&device_, devId)!=CUDA_SUCCESS){
        std::cerr << "[CUDA] cuDeviceGet failed for device " << devId << "\n";
        return false;
    }
    if(cuDevicePrimaryCtxRetain(&ctx, device_)!=CUDA_SUCCESS){
        std::cerr << "[CUDA] cuDevicePrimaryCtxRetain failed\n";
        return false;
    }
    return true;
}

bool CudaExecutor::compileNVRTC(const std::string& src, const std::string& entry, std::string& ptx){
    (void)entry;
    nvrtcProgram prog{};
    if(!checkNVRTC(nvrtcCreateProgram(&prog, src.c_str(), "kernel.cu", 0, nullptr, nullptr), "nvrtcCreateProgram")) return false;

    int major=8, minor=9;  // Default to RTX 4070 Ti compute capability
    if(ctx){
        CUdevice dev; cuCtxGetDevice(&dev);
        if(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) == CUDA_SUCCESS &&
           cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) == CUDA_SUCCESS) {
            std::cout << "[CUDA] Detected compute capability: " << major << "." << minor << std::endl;
        } else {
            std::cout << "[CUDA] Failed to detect compute capability, using default 8.9" << std::endl;
        }
    } else {
        std::cout << "[CUDA] No CUDA context, using default compute capability 8.9" << std::endl; us
    }

    // Use sm_XX format instead of compute_XX for better compatibility
    std::string archOpt = std::string("--gpu-architecture=sm_") + std::to_string(major) + std::to_string(minor);
    std::cout << "[CUDA] NVRTC compilation options: --std=c++14 " << archOpt << std::endl;
    const char* opts[] = {"--std=c++14", archOpt.c_str()};
    auto __nvrtc_t0 = std::chrono::high_resolution_clock::now();
    auto r = nvrtcCompileProgram(prog, int(std::size(opts)), opts);
    auto __nvrtc_t1 = std::chrono::high_resolution_clock::now();
    g_cuda_compile_ms = std::chrono::duration<double, std::milli>(__nvrtc_t1-__nvrtc_t0).count();

    size_t logSize=0; nvrtcGetProgramLogSize(prog, &logSize);
    if(logSize>1){ std::string log; log.resize(logSize); nvrtcGetProgramLog(prog, log.data()); std::cout << log << std::endl; }
    if(!checkNVRTC(r,"nvrtcCompileProgram")){ nvrtcDestroyProgram(&prog); return false; }

    size_t ptxSize=0; nvrtcGetPTXSize(prog,&ptxSize); ptx.resize(ptxSize); nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);
    return true;
}

std::shared_ptr<KernelCache<CudaExecutor::CudaKernel>::CachedKernel>
CudaExecutor::get_or_compile_kernel(const std::string& src, const std::string& entry) {
    std::string key = KernelCache<CudaKernel>::computeHash(src + entry);

    auto cached = kernel_cache_.get(key);
    if(cached) {
        std::cout << "[CUDA] Using cached kernel for entry: " << entry << std::endl;
        g_cuda_compile_ms = 0.0;
        return cached;
    }

    std::cout << "[CUDA] Compiling new kernel for entry: " << entry << std::endl;
    std::string ptx;
    if(!compileNVRTC(src, entry, ptx)) {
        return nullptr;
    }

    auto kernel = std::make_shared<KernelCache<CudaKernel>::CachedKernel>();
    kernel->kernel.ptx = ptx;
    kernel->kernel.entry = entry;
    kernel->lastUsed = std::chrono::steady_clock::now();

    kernel_cache_.put(key, kernel);
    std::cout << "[CUDA] Kernel compiled and cached\n";
    return kernel;
}

bool CudaExecutor::launch(const std::string& ptx, const std::string& entry,
                const std::vector<uint64_t>& uniforms,
                const std::vector<std::vector<uint8_t>>& inputs,
                const std::vector<size_t>& outputSizes,
                const std::vector<int>& grid, const std::vector<int>& block,
                std::vector<std::vector<uint8_t>>& outputs){

    CUcontext pushed = nullptr;
    if(!check(cuCtxPushCurrent(ctx), "cuCtxPushCurrent")) return false;

    auto pop_ctx = [&](){
        CUcontext dummy=nullptr;
        cuCtxPopCurrent(&dummy);
    };

    // CUDA events for timing
    cudaEvent_t evH2D0, evH2D1, evK0, evK1, evD2H0, evD2H1;
    cudaEventCreate(&evH2D0); cudaEventCreate(&evH2D1);
    cudaEventCreate(&evK0);   cudaEventCreate(&evK1);
    cudaEventCreate(&evD2H0); cudaEventCreate(&evD2H1);

    CUmodule mod=nullptr; CUfunction fun=nullptr;

    // JIT options for driver (for PTX -> SASS)
    CUjit_option options[6];
    void* optionVals[6];
    char error_log[8192]; error_log[0]=0;
    char info_log[8192];  info_log[0]=0;
    unsigned int logSize=sizeof(error_log), infoSize=sizeof(info_log);

    options[0] = CU_JIT_ERROR_LOG_BUFFER;       optionVals[0] = error_log;
    options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; optionVals[1] = (void*)(uintptr_t)logSize;
    options[2] = CU_JIT_INFO_LOG_BUFFER;        optionVals[2] = info_log;
    options[3] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES; optionVals[3] = (void*)(uintptr_t)infoSize;
    options[4] = CU_JIT_LOG_VERBOSE;            optionVals[4] = (void*)1;
    options[5] = CU_JIT_FALLBACK_STRATEGY;      optionVals[5] = (void*)CU_PREFER_PTX;

    if(!check(cuModuleLoadDataEx(&mod, ptx.c_str(), 6, options, optionVals), "cuModuleLoadDataEx")){
        if (error_log[0]) std::cerr << "[JIT error] " << error_log << std::endl;
        if (info_log[0])  std::cerr << "[JIT info]  " << info_log  << std::endl;
        pop_ctx();
        return false;
    }

    if(!check(cuModuleGetFunction(&fun, mod, entry.c_str()), "cuModuleGetFunction")){
        cuModuleUnload(mod);
        pop_ctx();
        return false;
    }

    std::vector<CUdeviceptr> dIn(inputs.size());
    // H2D begin
    cudaEventRecord(evH2D0, 0);
    for(size_t i=0;i<inputs.size();++i){
        if(!check(cuMemAlloc(&dIn[i], inputs[i].size()),"cuMemAlloc(in)")) {
            for(size_t j=0; j<i; ++j) cuMemFree(dIn[j]);
            cuModuleUnload(mod);
            pop_ctx();
            return false;
        }
        if(!check(cuMemcpyHtoD(dIn[i], inputs[i].data(), inputs[i].size()),"cuMemcpyHtoD(in)")) {
            for(size_t j=0; j<=i; ++j) cuMemFree(dIn[j]);
            cuModuleUnload(mod);
            pop_ctx();
            return false;
        }
    }
    // H2D end
    cudaEventRecord(evH2D1, 0);
    cudaEventSynchronize(evH2D1);
    { float ms=0.0f; cudaEventElapsedTime(&ms, evH2D0, evH2D1); g_cuda_h2d_ms = (double)ms; }

    std::vector<CUdeviceptr> dOut(outputSizes.size());
    for(size_t i=0;i<outputSizes.size();++i){
        if(outputSizes[i]==0){ dOut[i]=0; continue; }
        if(!check(cuMemAlloc(&dOut[i], outputSizes[i]),"cuMemAlloc(out)")){
            for(auto d: dIn) cuMemFree(d);
            for(size_t j=0;j<i;++j) cuMemFree(dOut[j]);
            cuModuleUnload(mod);
            pop_ctx();
            return false;
        }
        check(cuMemsetD8(dOut[i], 0, outputSizes[i]), "cuMemsetD8(out)");
    }

    // Build args: uniforms..., inputs..., outputs...
    std::vector<void*> args; args.reserve(uniforms.size() + dIn.size() + dOut.size());
    for(auto& u: uniforms) args.push_back((void*)&u);
    for(auto& d: dIn) args.push_back((void*)&d);
    for(auto& d: dOut) if(d) args.push_back((void*)&d);

    dim3 g(grid.size()>0?grid[0]:1, grid.size()>1?grid[1]:1, grid.size()>2?grid[2]:1);
    dim3 b(block.size()>0?block[0]:1, block.size()>1?block[1]:1, block.size()>2?block[2]:1);

    std::cout << "[CUDA] Launch parameters:" << std::endl;
    std::cout << "[CUDA]   Grid: (" << g.x << ", " << g.y << ", " << g.z << ")" << std::endl;
    std::cout << "[CUDA]   Block: (" << b.x << ", " << b.y << ", " << b.z << ")" << std::endl;
    std::cout << "[CUDA]   Total threads: " << (g.x * g.y * g.z * b.x * b.y * b.z) << std::endl;
    std::cout << "[CUDA]   Arguments: " << args.size() << " (uniforms: " << uniforms.size()
              << ", inputs: " << dIn.size() << ", outputs: " << dOut.size() << ")" << std::endl;

    cudaEventRecord(evK0, 0);
    if(!check(cuLaunchKernel(fun, g.x,g.y,g.z, b.x,b.y,b.z, 0, 0, args.data(), nullptr), "cuLaunchKernel")){
        std::cerr << "[CUDA] Kernel launch failed with parameters:" << std::endl;
        std::cerr << "[CUDA]   Grid: (" << g.x << ", " << g.y << ", " << g.z << ")" << std::endl;
        std::cerr << "[CUDA]   Block: (" << b.x << ", " << b.y << ", " << b.z << ")" << std::endl;
        std::cerr << "[CUDA]   Entry point: " << entry << std::endl;
        std::cerr << "[CUDA]   Arguments count: " << args.size() << std::endl;
        for(auto d: dIn) cuMemFree(d);
        for(auto d: dOut) cuMemFree(d);
        cuModuleUnload(mod);
        pop_ctx();
        return false;
    }
    cudaEventRecord(evK1, 0);
    cudaEventSynchronize(evK1);
    { float ms=0.0f; cudaEventElapsedTime(&ms, evK0, evK1); g_cuda_kernel_ms = (double)ms; }

    if(!check(cuCtxSynchronize(), "cuCtxSynchronize")){
        std::cerr << "[CUDA] Context synchronization failed after kernel launch" << std::endl;
        std::cerr << "[CUDA] This usually indicates a kernel execution error (e.g., illegal memory access)" << std::endl;
        std::cerr << "[CUDA] Kernel parameters were:" << std::endl;
        std::cerr << "[CUDA]   Grid: (" << g.x << ", " << g.y << ", " << g.z << ")" << std::endl;
        std::cerr << "[CUDA]   Block: (" << b.x << ", " << b.y << ", " << b.z << ")" << std::endl;
        std::cerr << "[CUDA]   Entry point: " << entry << std::endl;
        for(auto d: dIn) cuMemFree(d);
        for(auto d: dOut) cuMemFree(d);
        cuModuleUnload(mod);
        pop_ctx();
        return false;
    }

    outputs.resize(dOut.size());
    // D2H begin
    cudaEventRecord(evD2H0, 0);
    for(size_t i=0;i<dOut.size();++i){
        outputs[i].resize(outputSizes[i]);
        if(!check(cuMemcpyDtoH(outputs[i].data(), dOut[i], outputSizes[i]), "cuMemcpyDtoH")){
            for(auto d: dIn) cuMemFree(d);
            for(auto d: dOut) cuMemFree(d);
            cuModuleUnload(mod);
            pop_ctx();
            return false;
        }
        cuMemFree(dOut[i]);
    }
    // D2H end
    cudaEventRecord(evD2H1, 0);
    cudaEventSynchronize(evD2H1);
    { float ms=0.0f; cudaEventElapsedTime(&ms, evD2H0, evD2H1); g_cuda_d2h_ms = (double)ms; }

    for(auto d: dIn) cuMemFree(d);
    cuModuleUnload(mod);
    // destroy events
    cudaEventDestroy(evH2D0); cudaEventDestroy(evH2D1);
    cudaEventDestroy(evK0);   cudaEventDestroy(evK1);
    cudaEventDestroy(evD2H0); cudaEventDestroy(evD2H1);
    pop_ctx();
    return true;
}

ExecResult CudaExecutor::run_task(const json& task){
    ExecResult r;
    std::cout << "[CUDA] Task JSON: " << task.dump(2) << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    try {
        std::string src = task.value("source","");
        std::string entry = task.value("entry","main");

        std::vector<uint64_t> uniforms;
        if(task.contains("uniforms") && task["uniforms"].is_array()){
            for(auto& v: task["uniforms"]){
                if(v.is_number_unsigned()) uniforms.push_back(v.get<uint64_t>());
                else if(v.is_number_integer()) uniforms.push_back((uint64_t)v.get<long long>());
                else if(v.is_number_float()){ double d=v.get<double>(); uint64_t u; std::memcpy(&u,&d,sizeof(double)); uniforms.push_back(u); }
            }
        }

        std::vector<std::vector<uint8_t>> inputs;
        if(task.contains("inputs") && task["inputs"].is_array()){
            for(auto& it: task["inputs"]){
                std::string b64 = it.value("data","");
                extern std::vector<uint8_t> base64_decode(const std::string& s);
                inputs.push_back(base64_decode(b64));
            }
        }

        std::vector<size_t> outputSizes;
        if(task.contains("outputSizes") && task["outputSizes"].is_array()){
            for(auto& x: task["outputSizes"]) if(x.is_number_unsigned()) outputSizes.push_back(x.get<size_t>());
        }

        std::vector<int> grid = task.value("grid", std::vector<int>{});
        std::vector<int> block = task.value("block", std::vector<int>{});

        // If grid/block are missing, try to derive from global dimensions
        if (grid.empty() || block.empty()) {
            if (task.contains("global")) {
                if (task["global"].is_array()) {
                    // Global is an array - use it as grid and set reasonable block default
                    grid = task["global"].get<std::vector<int>>();
                    if (block.empty()) block = {16, 16, 1};
                } else if (task["global"].is_number_integer()) {
                    // Global is a scalar - derive 1D grid
                    const auto G = task["global"].get<long long>();
                    const int TPB = 256;
                    grid = { int(std::max<long long>(1, (G + TPB - 1)/TPB)), 1, 1 };
                    block = { TPB, 1, 1 };
                }
            }
            // Set defaults if still empty
            if (grid.empty()) grid = {1, 1, 1};
            if (block.empty()) block = {16, 16, 1};
        }

        std::cout << "[CUDA] Derived launch parameters:" << std::endl;
        std::cout << "[CUDA]   Grid: [" << (grid.size() > 0 ? std::to_string(grid[0]) : "1")
                  << ", " << (grid.size() > 1 ? std::to_string(grid[1]) : "1")
                  << ", " << (grid.size() > 2 ? std::to_string(grid[2]) : "1") << "]" << std::endl;
        std::cout << "[CUDA]   Block: [" << (block.size() > 0 ? std::to_string(block[0]) : "1")
                  << ", " << (block.size() > 1 ? std::to_string(block[1]) : "1")
                  << ", " << (block.size() > 2 ? std::to_string(block[2]) : "1") << "]" << std::endl;

        // Use cached kernel if available
        auto cached_kernel = get_or_compile_kernel(src, entry);
        if(!cached_kernel){
            r.error="CUDA kernel compilation failed";
            return r;
        }

        std::vector<std::vector<uint8_t>> outputs;
        if(!launch(cached_kernel->kernel.ptx, entry, uniforms, inputs, outputSizes, grid, block, outputs)){
            r.error="CUDA kernel launch failed";
            return r;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        // set kernel GPU time in ms; wall time is still t1-t0 if you need it for logs
        r.ms = g_cuda_kernel_ms;
        r.ok = true;
        r.outputs = std::move(outputs);
        return r;
    } catch (const std::runtime_error& e) {
        std::cerr << "[CUDA] Runtime error in run_task: " << e.what() << std::endl;
        r.error = std::string("CUDA runtime error: ") + e.what();
        return r;
    } catch (const std::exception& e) {
        std::cerr << "[CUDA] Standard exception in run_task: " << e.what() << std::endl;
        std::cerr << "[CUDA] Exception type: " << typeid(e).name() << std::endl;
        r.error = std::string("CUDA standard exception: ") + e.what();
        return r;
    } catch (...) {
        std::cerr << "[CUDA] Unknown exception in run_task" << std::endl;
        r.error = "CUDA unknown exception";
        return r;
    }
}

CudaExecutor::~CudaExecutor(){
    if(ctx){
        CUcontext current = nullptr;
        cuCtxGetCurrent(&current);
        if(current != ctx){
            cuCtxPushCurrent(ctx);
            CUcontext dummy = nullptr;
            cuCtxPopCurrent(&dummy);
        }
        cuDevicePrimaryCtxRelease(device_);
        ctx = nullptr;
    }
}
#else

bool CudaExecutor::initialize(const json& cfg){ (void)cfg; return false; }
ExecResult CudaExecutor::run_task(const json& task){ (void)task; return ExecResult{false,{},0.0, "CUDA disabled"}; }
CudaExecutor::~CudaExecutor(){};

#endif
