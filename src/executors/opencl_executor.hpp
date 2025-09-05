#pragma once
#include "iexecutor.hpp"
#include "kernel_cache.hpp"
#ifdef HAVE_OPENCL
#include <CL/cl.h>
#endif
#include <string>
#include <vector>
#include <memory>
#include <mutex>

class OpenCLExecutor : public IExecutor {
public:
    OpenCLExecutor() = default;
    bool initialize(const json& cfg) override;
    ExecResult run_task(const json& task) override;
    ~OpenCLExecutor();

private:
#ifdef HAVE_OPENCL
    struct CLKernel {
        cl_program program;
        cl_kernel kernel;
        std::string buildLog;
    };
    
    // Static kernel cache shared across all executor instances
    static KernelCache<CLKernel> kernel_cache_;
    static std::mutex queue_mutex_;
    
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    
    bool pick_first_available();
    std::shared_ptr<KernelCache<CLKernel>::CachedKernel> 
        get_or_build_kernel(const std::string& source, const std::string& entry);
    bool build_kernel(const std::string& source, const std::string& entry, 
                     cl_program& prog, cl_kernel& kernel, std::string& buildLog);
#endif
};
