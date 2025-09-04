
#pragma once
#include "iexecutor.hpp"
#ifdef HAVE_CUDA
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>
#endif
#include <string>
#include <vector>

class CudaExecutor : public IExecutor {
public:
    explicit CudaExecutor(int deviceId=0) : devId(deviceId) {}
    bool initialize(const json& cfg) override;
    ExecResult run_task(const json& task) override;

private:
#ifdef HAVE_CUDA
    bool ensureDriver();
    bool compileNVRTC(const std::string& src, const std::string& entry, std::string& ptx);
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
