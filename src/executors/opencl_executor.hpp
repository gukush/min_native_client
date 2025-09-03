
#pragma once
#include "iexecutor.hpp"
#ifdef HAVE_OPENCL
#include <CL/cl.h>
#endif
#include <string>
#include <vector>

class OpenCLExecutor : public IExecutor {
public:
    OpenCLExecutor() = default;
    bool initialize(const json& cfg) override;
    ExecResult run_task(const json& task) override;

private:
#ifdef HAVE_OPENCL
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;

    bool pick_first_available();
    bool build_kernel(const std::string& source, const std::string& entry, cl_program& prog, cl_kernel& kernel, std::string& buildLog);
#endif
};
