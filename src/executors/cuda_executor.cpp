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
        std::cout << "[CUDA] No CUDA context, using default compute capability 8.9" << std::endl;
    }

    // Use sm_XX format instead of compute_XX for better compatibility
    std::string archOpt = std::string("--gpu-architecture=sm_") + std::to_string(major) + std::to_string(minor);

    // Add include paths for CUDA headers (avoid system headers that cause issues with NVRTC)
    const char* opts[] = {
        "--std=c++14",
        archOpt.c_str(),
        "-I/usr/local/cuda/include",
        "-I/opt/cuda/include",
        "-I/usr/local/cuda/targets/x86_64-linux/include",
        "-I/opt/cuda/targets/x86_64-linux/include"
    };
    std::cout << "[CUDA] NVRTC compilation options: --std=c++14 " << archOpt << " -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include -I/opt/cuda/targets/x86_64-linux/include" << std::endl;
    auto __nvrtc_t0 = std::chrono::high_resolution_clock::now();
    auto r = nvrtcCompileProgram(prog, 6, opts);
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
                std::vector<std::vector<uint8_t>>& outputs,
                const std::vector<bool>& inputInPlace){

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
        std::cout << "[CUDA] Allocating input buffer " << i << " with size: " << inputs[i].size() << " bytes" << std::endl;
        if(!check(cuMemAlloc(&dIn[i], inputs[i].size()),"cuMemAlloc(in)")) {
            std::cerr << "[CUDA] Failed to allocate input buffer " << i << " with size " << inputs[i].size() << " bytes" << std::endl;
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

    // Check if this is ECM Stage1 kernel (unified buffer)
    bool is_ecm_stage1 = (entry == "ecm_stage1_v3_optimized" || entry == "ecm_stage1_v3");

    std::vector<CUdeviceptr> dOut(outputSizes.size());
    for(size_t i=0;i<outputSizes.size();++i){
        if(outputSizes[i]==0){ dOut[i]=0; continue; }
        std::cout << "[CUDA] Allocating output buffer " << i << " with size: " << outputSizes[i] << " bytes" << std::endl;
        if(!check(cuMemAlloc(&dOut[i], outputSizes[i]),"cuMemAlloc(out)")){
            std::cerr << "[CUDA] Failed to allocate output buffer " << i << " with size " << outputSizes[i] << " bytes" << std::endl;
            for(auto d: dIn) cuMemFree(d);
            for(size_t j=0;j<i;++j) cuMemFree(dOut[j]);
            cuModuleUnload(mod);
            pop_ctx();
            return false;
        }
        // Only zero out output buffer if not using unified buffer (ECM Stage1)
        if (!is_ecm_stage1) {
            check(cuMemsetD8(dOut[i], 0, outputSizes[i]), "cuMemsetD8(out)");
        }
    }

    // Check if any inputs are marked as in-place
    bool inPlace = false;
    for (size_t i = 0; i < inputInPlace.size(); ++i) {
        if (inputInPlace[i]) {
            inPlace = true;
            std::cout << "[CUDA] Input " << i << " marked as in-place" << std::endl;
        }
    }

    if (!inPlace) {
        std::cout << "[CUDA] No in-place inputs detected, using separate buffer mode" << std::endl;
    }

    // Build args: uniforms..., inputs..., outputs...
    std::vector<void*> args; args.reserve(uniforms.size() + dIn.size() + dOut.size());
    for(auto& u: uniforms) args.push_back((void*)&u);
    for(auto& d: dIn) args.push_back((void*)&d);

    // Special handling for ECM Stage1 kernel (unified buffer)
    if (is_ecm_stage1 && dIn.size() == 1 && dOut.size() == 1 && dIn[0] != 0 && dOut[0] != 0) {
        // ECM Stage1 expects unified buffer - pass only input buffer, don't pass output buffer
        std::cout << "[CUDA] ECM Stage1 detected: using unified buffer (input only)" << std::endl;
        args.push_back((void*)&dIn[0]); // Only pass input buffer
    } else {
        // Handle outputs based on in-place flags
        for(size_t i = 0; i < dOut.size(); ++i) {
            if (dOut[i] == 0) continue; // Skip null output buffers

            // If this output corresponds to an in-place input, use the input buffer
            if (i < inputInPlace.size() && inputInPlace[i] && i < dIn.size()) {
                args.push_back((void*)&dIn[i]); // Use input buffer as output
                std::cout << "[CUDA] Output " << i << " in-place: using input buffer as output buffer" << std::endl;
            } else {
                args.push_back((void*)&dOut[i]); // Use separate output buffer
            }
        }
    }

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
        CUdeviceptr src = dOut[i];

        // Special handling for ECM Stage1 kernel (unified buffer)
        if (is_ecm_stage1 && i == 0 && dIn.size() > 0) {
            src = dIn[0]; // Read from input buffer for ECM Stage1
            std::cout << "[CUDA] ECM Stage1: reading result from input buffer" << std::endl;
        }
        // For in-place inputs, read from input buffer instead of output buffer
        else if (i < inputInPlace.size() && inputInPlace[i] && i < dIn.size()) {
            src = dIn[i];
            std::cout << "[CUDA] Output " << i << " in-place: reading result from input buffer" << std::endl;
        }

        if(!check(cuMemcpyDtoH(outputs[i].data(), src, outputSizes[i]), "cuMemcpyDtoH")){
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

    // Use the debug utilities from kernel_cache.hpp
    std::cout << "[CUDA] Task summary: " << json_summary(task) << std::endl;
    std::cout << "[CUDA] Task keys: ";
    for (auto& [key, value] : task.items()) {
        std::cout << key << " ";
    }
    std::cout << std::endl;

    // Show truncated structure for detailed debugging
    if (task.size() > 0) {
        auto truncated = truncate_json_for_debug(task, 100, 2);  // 100 chars max, 2 levels deep
        std::cout << "[CUDA] Task structure (truncated): " << truncated.dump(2) << std::endl;
    }

    // Check for workload schema information
    if (task.contains("workload")) {
        std::cout << "[CUDA] Workload found in task" << std::endl;
        auto workload = task["workload"];
        if (workload.contains("schema")) {
            std::cout << "[CUDA] Schema found in workload: " << workload["schema"].dump(2) << std::endl;
        } else {
            std::cout << "[CUDA] No schema in workload" << std::endl;
            std::cout << "[CUDA] Workload keys: ";
            for (auto& [key, value] : workload.items()) {
                std::cout << key << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "[CUDA] No workload in task" << std::endl;
    }

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
        std::cout << "[CUDA] Parsed " << uniforms.size() << " uniform values" << std::endl;

        // Improved input processing with better size reporting
        std::vector<std::vector<uint8_t>> inputs;
        size_t total_input_bytes = 0;

        if(task.contains("inputs") && task["inputs"].is_array()){
            std::cout << "[CUDA] Processing " << task["inputs"].size() << " input buffer(s):" << std::endl;

            for(size_t idx = 0; idx < task["inputs"].size(); ++idx){
                auto& it = task["inputs"][idx];
                std::string b64 = it.value("b64", it.value("data", ""));

                // Show base64 size info
                std::cout << "[CUDA]   Input " << idx << ": base64 string " << b64.length() << " chars";

                // Decode and show actual size
                extern std::vector<uint8_t> base64_decode(const std::string& s);
                auto decoded = base64_decode(b64);
                std::cout << " -> " << decoded.size() << " bytes";

                // Show first few bytes as hex for verification
                if (!decoded.empty()) {
                    std::cout << " (first 8 bytes: ";
                    for (size_t i = 0; i < std::min(size_t(8), decoded.size()); ++i) {
                        printf("%02x", decoded[i]);
                    }
                    std::cout << ")";
                }
                std::cout << std::endl;

                total_input_bytes += decoded.size();
                inputs.push_back(std::move(decoded));
            }

            std::cout << "[CUDA] Total input data: " << total_input_bytes << " bytes ("
                      << (total_input_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
        } else {
            std::cout << "[CUDA] No inputs found in task" << std::endl;
        }

        std::vector<size_t> outputSizes;
        if(task.contains("outputSizes") && task["outputSizes"].is_array()){
            for(auto& x: task["outputSizes"]) {
                if(x.is_number_unsigned()) {
                    outputSizes.push_back(x.get<size_t>());
                } else if(x.is_number_integer()) {
                    outputSizes.push_back(static_cast<size_t>(x.get<long long>()));
                } else if(x.is_number_float()) {
                    outputSizes.push_back(static_cast<size_t>(x.get<double>()));
                }
            }
        }

        // Better output size reporting
        std::cout << "[CUDA] Output buffers (" << outputSizes.size() << "): ";
        size_t total_output_bytes = 0;
        for(size_t i = 0; i < outputSizes.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "[" << i << "]=" << outputSizes[i] << "B";
            total_output_bytes += outputSizes[i];
        }
        if (!outputSizes.empty()) {
            std::cout << " (total: " << total_output_bytes << " bytes, "
                      << (total_output_bytes / 1024.0 / 1024.0) << " MB)";
        }
        std::cout << std::endl;

        std::vector<int> grid = task.value("grid", std::vector<int>{});
        std::vector<int> block = task.value("block", std::vector<int>{});

        std::cout << "[CUDA] Launch config - grid: " << grid.size() << " dims, block: " << block.size() << " dims" << std::endl;

        // If grid/block are missing, try to derive from global dimensions
        if (grid.empty() || block.empty()) {
            if (task.contains("global")) {
                if (task["global"].is_array()) {
                    grid = task["global"].get<std::vector<int>>();
                    if (block.empty()) block = {16, 16, 1};
                    std::cout << "[CUDA] Derived grid from global array" << std::endl;
                } else if (task["global"].is_number_integer()) {
                    const auto G = task["global"].get<long long>();
                    const int TPB = 256;
                    grid = { int(std::max<long long>(1, (G + TPB - 1)/TPB)), 1, 1 };
                    block = { TPB, 1, 1 };
                    std::cout << "[CUDA] Derived 1D grid from global=" << G << std::endl;
                }
            }
            // Set defaults if still empty
            if (grid.empty()) grid = {1, 1, 1};
            if (block.empty()) block = {16, 16, 1};
        }

        // Concise launch parameters display
        auto format_dims = [](const std::vector<int>& dims) {
            return "[" + std::to_string(dims.size() > 0 ? dims[0] : 1) +
                   "," + std::to_string(dims.size() > 1 ? dims[1] : 1) +
                   "," + std::to_string(dims.size() > 2 ? dims[2] : 1) + "]";
        };

        int total_threads = (grid.size() > 0 ? grid[0] : 1) * (grid.size() > 1 ? grid[1] : 1) * (grid.size() > 2 ? grid[2] : 1) *
                           (block.size() > 0 ? block[0] : 1) * (block.size() > 1 ? block[1] : 1) * (block.size() > 2 ? block[2] : 1);

        std::cout << "[CUDA] Launch: Grid" << format_dims(grid) << " × Block" << format_dims(block)
                  << " = " << total_threads << " threads" << std::endl;

        // Parse schema to check for in-place flags on inputs
        std::vector<bool> inputInPlace(inputs.size(), false);
        bool found_schema = false;

        if (task.contains("workload") && task["workload"].is_object()) {
            auto workload = task["workload"];
            if (workload.contains("schema") && workload["schema"].is_object()) {
                auto schema = workload["schema"];
                if (schema.contains("inputs") && schema["inputs"].is_array()) {
                    auto inputs_schema = schema["inputs"];
                    found_schema = true;
                    std::cout << "[CUDA] Schema processing: found " << inputs_schema.size() << " input schema(s)" << std::endl;

                    for (size_t i = 0; i < inputs_schema.size() && i < inputs.size(); ++i) {
                        if (inputs_schema[i].contains("inPlace") && inputs_schema[i]["inPlace"].is_boolean()) {
                            inputInPlace[i] = inputs_schema[i]["inPlace"].get<bool>();
                            std::cout << "[CUDA]   Input " << i << ": inPlace=" << (inputInPlace[i] ? "true" : "false") << std::endl;
                        }
                    }
                }
            }
        }

        if (!found_schema) {
            std::cout << "[CUDA] No schema found - using separate input/output buffers" << std::endl;
        }

        // Show final in-place configuration
        bool has_inplace = false;
        for (size_t i = 0; i < inputInPlace.size(); ++i) {
            if (inputInPlace[i]) {
                has_inplace = true;
                break;
            }
        }
        std::cout << "[CUDA] Buffer mode: " << (has_inplace ? "in-place" : "separate") << " processing" << std::endl;

        // Use cached kernel if available
        auto cached_kernel = get_or_compile_kernel(src, entry);
        if(!cached_kernel){
            r.error="CUDA kernel compilation failed";
            return r;
        }

        std::vector<std::vector<uint8_t>> outputs;
        if(!launch(cached_kernel->kernel.ptx, entry, uniforms, inputs, outputSizes, grid, block, outputs, inputInPlace)){
            r.error="CUDA kernel launch failed";
            return r;
        }

        // Report output results
        std::cout << "[CUDA] Execution completed, " << outputs.size() << " output buffer(s) returned" << std::endl;
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::cout << "[CUDA]   Output " << i << ": " << outputs[i].size() << " bytes";
            if (!outputs[i].empty()) {
                // Show first few bytes for verification
                std::cout << " (first 8 bytes: ";
                for (size_t j = 0; j < std::min(size_t(8), outputs[i].size()); ++j) {
                    printf("%02x", outputs[i][j]);
                }
                std::cout << ")";
            }
            std::cout << std::endl;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        r.ms = g_cuda_kernel_ms;  // GPU time from timing events
        r.ok = true;
        r.outputs = std::move(outputs);

        std::cout << "[CUDA] Task completed successfully in " << r.ms << " ms GPU time" << std::endl;
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

// Enhanced methods for efficient batch processing
ExecResult CudaExecutor::run_batch_task(const json& task) {
    std::cout << "[CUDA-ENHANCED] Running batch task" << std::endl;

    // Check if this is a batch bitonic sort task
    if (task.contains("batchType") && task["batchType"] == "bitonic_sort") {
        return run_bitonic_sort_batch(task);
    }

    // Fall back to regular task processing for other tasks
    return run_task(task);
}

ExecResult CudaExecutor::create_gpu_buffer(const json& task) {
    std::cout << "[CUDA-ENHANCED] Creating GPU buffer" << std::endl;

    if (!ensureDriver()) {
        return ExecResult{false, {}, 0.0, "CUDA driver not available"};
    }

    CUcontext pushed = nullptr;
    if (!check(cuCtxPushCurrent(ctx), "cuCtxPushCurrent")) {
        return ExecResult{false, {}, 0.0, "Failed to push CUDA context"};
    }

    auto pop_ctx = [&]() {
        CUcontext dummy = nullptr;
        cuCtxPopCurrent(&dummy);
    };

    try {
        std::string bufferId = task["bufferId"];
        size_t bufferSize = task["size"];

        CUdeviceptr d_ptr;
        if (!check(cuMemAlloc(&d_ptr, bufferSize), "cuMemAlloc")) {
            pop_ctx();
            return ExecResult{false, {}, 0.0, "Failed to allocate GPU memory"};
        }

        // Store buffer info
        gpuBuffers_[bufferId] = {d_ptr, bufferSize, bufferId, true};

        pop_ctx();
        return ExecResult{true, {}, 0.0, "GPU buffer created"};

    } catch (const std::exception& e) {
        pop_ctx();
        return ExecResult{false, {}, 0.0, std::string("Error creating GPU buffer: ") + e.what()};
    }
}

ExecResult CudaExecutor::destroy_gpu_buffer(const json& task) {
    std::cout << "[CUDA-ENHANCED] Destroying GPU buffer" << std::endl;

    try {
        std::string bufferId = task["bufferId"];

        auto it = gpuBuffers_.find(bufferId);
        if (it != gpuBuffers_.end()) {
            cuMemFree(it->second.ptr);
            gpuBuffers_.erase(it);
            return ExecResult{true, {}, 0.0, "GPU buffer destroyed"};
        } else {
            return ExecResult{false, {}, 0.0, "Buffer not found"};
        }
    } catch (const std::exception& e) {
        return ExecResult{false, {}, 0.0, std::string("Error destroying GPU buffer: ") + e.what()};
    }
}

ExecResult CudaExecutor::run_kernel_on_gpu_buffer(const json& task) {
    std::cout << "[CUDA-ENHANCED] Running kernel on GPU buffer" << std::endl;

    if (!ensureDriver()) {
        return ExecResult{false, {}, 0.0, "CUDA driver not available"};
    }

    CUcontext pushed = nullptr;
    if (!check(cuCtxPushCurrent(ctx), "cuCtxPushCurrent")) {
        return ExecResult{false, {}, 0.0, "Failed to push CUDA context"};
    }

    auto pop_ctx = [&]() {
        CUcontext dummy = nullptr;
        cuCtxPopCurrent(&dummy);
    };

    try {
        std::string bufferId = task["bufferId"];
        std::string source = task["source"];
        std::string entry = task["entry"];
        std::vector<uint64_t> uniforms = task["uniforms"];
        std::vector<int> grid = task["grid"];
        std::vector<int> block = task["block"];

        auto it = gpuBuffers_.find(bufferId);
        if (it == gpuBuffers_.end()) {
            pop_ctx();
            return ExecResult{false, {}, 0.0, "GPU buffer not found"};
        }

        // Get or compile kernel
        auto kernel = get_or_compile_kernel(source, entry);
        if (!kernel) {
            pop_ctx();
            return ExecResult{false, {}, 0.0, "Failed to compile kernel"};
        }

        // Launch kernel directly on GPU buffer
        CUmodule mod = nullptr;
        CUfunction fun = nullptr;

        // JIT options for driver
        CUjit_option options[6];
        void* optionVals[6];
        char error_log[8192]; error_log[0] = 0;
        char info_log[8192]; info_log[0] = 0;
        unsigned int logSize = sizeof(error_log), infoSize = sizeof(info_log);

        options[0] = CU_JIT_ERROR_LOG_BUFFER; optionVals[0] = error_log;
        options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; optionVals[1] = (void*)(uintptr_t)logSize;
        options[2] = CU_JIT_INFO_LOG_BUFFER; optionVals[2] = info_log;
        options[3] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES; optionVals[3] = (void*)(uintptr_t)infoSize;
        options[4] = CU_JIT_LOG_VERBOSE; optionVals[4] = (void*)1;
        options[5] = CU_JIT_FALLBACK_STRATEGY; optionVals[5] = (void*)CU_PREFER_PTX;

        if (!check(cuModuleLoadDataEx(&mod, kernel->kernel.ptx.c_str(), 6, options, optionVals), "cuModuleLoadDataEx")) {
            if (error_log[0]) std::cerr << "[JIT error] " << error_log << std::endl;
            if (info_log[0]) std::cerr << "[JIT info] " << info_log << std::endl;
            pop_ctx();
            return ExecResult{false, {}, 0.0, "Failed to load module"};
        }

        if (!check(cuModuleGetFunction(&fun, mod, entry.c_str()), "cuModuleGetFunction")) {
            cuModuleUnload(mod);
            pop_ctx();
            return ExecResult{false, {}, 0.0, "Failed to get function"};
        }

        // Build arguments: uniforms..., gpu_buffer
        std::vector<void*> args;
        args.reserve(uniforms.size() + 1);
        for (auto& u : uniforms) args.push_back((void*)&u);
        args.push_back((void*)&it->second.ptr);

        dim3 g(grid.size() > 0 ? grid[0] : 1, grid.size() > 1 ? grid[1] : 1, grid.size() > 2 ? grid[2] : 1);
        dim3 b(block.size() > 0 ? block[0] : 1, block.size() > 1 ? block[1] : 1, block.size() > 2 ? block[2] : 1);

        // Launch kernel
        if (!check(cuLaunchKernel(fun, g.x, g.y, g.z, b.x, b.y, b.z, 0, 0, args.data(), nullptr), "cuLaunchKernel")) {
            cuModuleUnload(mod);
            pop_ctx();
            return ExecResult{false, {}, 0.0, "Kernel launch failed"};
        }

        if (!check(cuCtxSynchronize(), "cuCtxSynchronize")) {
            cuModuleUnload(mod);
            pop_ctx();
            return ExecResult{false, {}, 0.0, "Context synchronization failed"};
        }

        cuModuleUnload(mod);
        pop_ctx();

        return ExecResult{true, {}, 0.0, "Kernel executed successfully"};

    } catch (const std::exception& e) {
        pop_ctx();
        return ExecResult{false, {}, 0.0, std::string("Error running kernel: ") + e.what()};
    }
}

ExecResult CudaExecutor::run_bitonic_sort_batch(const json& task) {
    std::cout << "[CUDA-ENHANCED] Running bitonic sort batch" << std::endl;

    // This will be implemented to process multiple sort stages efficiently
    // For now, fall back to regular processing
    return run_task(task);
}

#else

bool CudaExecutor::initialize(const json& cfg){ (void)cfg; return false; }
ExecResult CudaExecutor::run_task(const json& task){ (void)task; return ExecResult{false,{},0.0, "CUDA disabled"}; }
CudaExecutor::~CudaExecutor(){};

#endif
