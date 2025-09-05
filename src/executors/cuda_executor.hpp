#pragma once
#include "iexecutor.hpp"
#include "kernel_cache.hpp"
#ifdef HAVE_CUDA
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>
#endif
#include <string>
#include <vector>
#include <memory>

class CudaExecutor : public IExecutor {
public:
    explicit CudaExecutor(int deviceId=0) : devId(deviceId) {}
    bool initialize(const json& cfg) override;
    ExecResult run_task(const json& task) override;
    ~CudaExecutor();

private:
#ifdef HAVE_CUDA
    struct CudaKernel {
        std::string ptx;
        std::string entry;
    };
    
    // Static kernel cache shared across all executor instances
    static KernelCache<CudaKernel> kernel_cache_;
    
    bool ensureDriver();
    bool compileNVRTC(const std::string& src, const std::string& entry, std::string& ptx);
    std::shared_ptr<KernelCache<CudaKernel>::CachedKernel> 
        get_or_compile_kernel(const std::string& src, const std::string& entry);
    bool launch(const std::string& ptx, const std::string& entry,
                const std::vector<uint64_t>& uniforms,
                const std::vector<std::vector<uint8_t>>& inputs,
                const std::vector<size_t>& outputSizes,
                const std::vector<int>& grid, const std::vector<int>& block,
                std::vector<std::vector<uint8_t>>& outputs);
    bool check(CUresult res, const char* what);
    bool checkNVRTC(nvrtcResult res, const char* what);
    CUcontext ctx=nullptr;
#endif
    int devId{0};
};
