
#include "cuda_executor.hpp"
#include <iostream>
#include <chrono>

#ifdef HAVE_CUDA

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
    if(inited) return true;
    if(!check(cuInit(0),"cuInit")) return false;
    inited=true; return true;
}

bool CudaExecutor::initialize(const json& cfg){
    (void)cfg;
    if(!ensureDriver()) return false;
    int cnt=0; if(!check(cuDeviceGetCount(&cnt),"cuDeviceGetCount")) return false;
    if(cnt<=0){ std::cerr<<"no cuda devices"<<std::endl; return false; }
    CUdevice dev; if(!check(cuDeviceGet(&dev, devId),"cuDeviceGet")) return false;
    if(!check(cuCtxCreate(&ctx, /*ctxCreateParams=*/nullptr,/*flags=*/0, dev),"cuCtxCreate")) return false;
    return true;
}

bool CudaExecutor::compileNVRTC(const std::string& src, const std::string& entry, std::string& ptx){
    (void)entry;
    nvrtcProgram prog=nullptr;
    if(!checkNVRTC(nvrtcCreateProgram(&prog, src.c_str(), "k.cu", 0, nullptr, nullptr), "nvrtcCreateProgram")) return false;
    const char* opts[] = {"--std=c++14"};
    auto r = nvrtcCompileProgram(prog, 1, opts);
    size_t logSize=0; nvrtcGetProgramLogSize(prog, &logSize);
    if(logSize>1){ std::string log; log.resize(logSize); nvrtcGetProgramLog(prog, log.data()); std::cout << log << std::endl; }
    if(!checkNVRTC(r,"nvrtcCompileProgram")){ nvrtcDestroyProgram(&prog); return false; }
    size_t ptxSize=0; nvrtcGetPTXSize(prog,&ptxSize); ptx.resize(ptxSize); nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);
    return true;
}

bool CudaExecutor::launch(const std::string& ptx, const std::string& entry,
                const std::vector<uint64_t>& uniforms,
                const std::vector<std::vector<uint8_t>>& inputs,
                const std::vector<size_t>& outputSizes,
                const std::vector<int>& grid, const std::vector<int>& block,
                std::vector<std::vector<uint8_t>>& outputs){
    CUcontext prev_ctx = nullptr;
    check(cuCtxGetCurrent(&prev_ctx), "cuCtxGetCurrent");
    if(prev_ctx != ctx){
        if(!check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")) return false;
    }
    CUmodule mod=nullptr; CUfunction fun=nullptr;
    if(!check(cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr), "cuModuleLoadDataEx")) return false;
    if(!check(cuModuleGetFunction(&fun, mod, entry.c_str()), "cuModuleGetFunction")){ cuModuleUnload(mod); return false; }

    // device buffers
    std::vector<CUdeviceptr> dIn(inputs.size());
    for(size_t i=0;i<inputs.size();++i){
        if(!check(cuMemAlloc(&dIn[i], inputs[i].size()),"cuMemAlloc(in)")) return false;
        if(!check(cuMemcpyHtoD(dIn[i], inputs[i].data(), inputs[i].size()),"cuMemcpyHtoD(in)")) return false;
    }
    std::vector<CUdeviceptr> dOut(outputSizes.size());
    for(size_t i=0;i<outputSizes.size();++i){
        if(!check(cuMemAlloc(&dOut[i], outputSizes[i]),"cuMemAlloc(out)")) return false;
    }

    // build args: uniforms first (as 64-bit values), then inputs, then outputs
    std::vector<void*> args;
    std::vector<uint64_t> uniformsCopy = uniforms;
    for(auto& u: uniformsCopy) args.push_back(&u);
    for(auto& d: dIn) args.push_back(&d);
    for(auto& d: dOut) args.push_back(&d);

    dim3 g(grid.size()>0?grid[0]:1, grid.size()>1?grid[1]:1, grid.size()>2?grid[2]:1);
    dim3 b(block.size()>0?block[0]:1, block.size()>1?block[1]:1, block.size()>2?block[2]:1);

    if(!check(cuLaunchKernel(fun, g.x,g.y,g.z, b.x,b.y,b.z, 0, 0, args.data(), nullptr), "cuLaunchKernel")){
        cuModuleUnload(mod); return false;
    }
    if(!check(cuCtxSynchronize(), "cuCtxSynchronize")){ cuModuleUnload(mod); return false; }

    outputs.resize(dOut.size());
    for(size_t i=0;i<dOut.size();++i){
        outputs[i].resize(outputSizes[i]);
        if(!check(cuMemcpyDtoH(outputs[i].data(), dOut[i], outputSizes[i]), "cuMemcpyDtoH")){ cuModuleUnload(mod); return false; }
        cuMemFree(dOut[i]);
    }
    for(auto d: dIn) cuMemFree(d);
    cuModuleUnload(mod);
    if (prev_ctx != ctx){
        check(cuCtxSetCurrent(prev_ctx), "cuCtxSetCurrent(restore)");
    }
    return true;
}

ExecResult CudaExecutor::run_task(const json& task){
    ExecResult r;
    auto t0 = std::chrono::high_resolution_clock::now();
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
            // Expect base64
            extern std::vector<uint8_t> base64_decode(const std::string& s);
            inputs.push_back(base64_decode(b64));
        }
    }
    std::vector<size_t> outputSizes;
    if(task.contains("outputSizes") && task["outputSizes"].is_array()){
        for(auto& x: task["outputSizes"]) if(x.is_number_unsigned()) outputSizes.push_back(x.get<size_t>());
    }
    std::vector<int> grid = task.value("grid", std::vector<int>{1,1,1});
    std::vector<int> block = task.value("block", std::vector<int>{16,16,1});

    std::string ptx;
    if(!compileNVRTC(src, entry, ptx)){ r.error="compile failed"; return r; }
    std::vector<std::vector<uint8_t>> outputs;
    if(!launch(ptx, entry, uniforms, inputs, outputSizes, grid, block, outputs)){ r.error="launch failed"; return r; }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    r.ok = true;
    r.outputs = std::move(outputs);
    return r;
}

CudaExecutor::~CudaExecutor(){
    if(ctx) {
        cuCtxDestroy(ctx);
    }
}

#else

bool CudaExecutor::initialize(const json& cfg){ (void)cfg; return false; }
ExecResult CudaExecutor::run_task(const json& task){ (void)task; return ExecResult{false,{},{}, "CUDA disabled"}; }
CudaExecutor::~CudaExecutor(){};
#endif
