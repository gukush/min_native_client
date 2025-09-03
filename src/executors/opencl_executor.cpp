#include "opencl_executor.hpp"
#include <iostream>
#include <chrono>
#include <cstring>
#include <vector>
#include <cstdint>

#ifdef HAVE_OPENCL

bool OpenCLExecutor::pick_first_available(){
    cl_uint numPlatforms=0;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if(err!=CL_SUCCESS || numPlatforms==0){ std::cerr<<"No OpenCL platforms\n"; return false; }
    std::vector<cl_platform_id> plats(numPlatforms);
    clGetPlatformIDs(numPlatforms, plats.data(), nullptr);

    for(auto p: plats){
        cl_uint numDevs=0;
        if(clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, &numDevs)==CL_SUCCESS && numDevs>0){
            platform = p;
            break;
        }
    }
    if(!platform || !device){
        // fallback to CPU
        for(auto p: plats){
            cl_uint numDevs=0;
            if(clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &device, &numDevs)==CL_SUCCESS && numDevs>0){
                platform = p; break;
            }
        }
    }
    return platform && device;
}

bool OpenCLExecutor::initialize(const json& cfg){
    (void)cfg;
    std::cerr << "[OpenCL] Initializing OpenCL executor..." << std::endl;
    if(!pick_first_available()) {
        std::cerr << "[OpenCL] Failed to pick available platform/device" << std::endl;
        return false;
    }
    std::cerr << "[OpenCL] Platform and device selected successfully" << std::endl;

    cl_int err=0;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if(err!=CL_SUCCESS){
        std::cerr<<"[OpenCL] clCreateContext failed with error: " << err << std::endl;
        return false;
    }
    std::cerr << "[OpenCL] Context created successfully" << std::endl;

#if CL_TARGET_OPENCL_VERSION >= 200
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
#else
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    if(err!=CL_SUCCESS){
        std::cerr<<"[OpenCL] clCreateCommandQueue failed with error: " << err << std::endl;
        return false;
    }
    std::cerr << "[OpenCL] Command queue created successfully" << std::endl;
    return true;
}

bool OpenCLExecutor::build_kernel(const std::string& source, const std::string& entry, cl_program& prog, cl_kernel& kernel, std::string& buildLog){
    const char* src = source.c_str();
    size_t len = source.size();
    cl_int err=0;

    std::cerr << "[OpenCL] Creating program from source (length: " << len << ")" << std::endl;
    prog = clCreateProgramWithSource(context, 1, &src, &len, &err);
    if(err!=CL_SUCCESS){
        std::cerr<<"[OpenCL] clCreateProgramWithSource failed with error: " << err << std::endl;
        return false;
    }

    std::cerr << "[OpenCL] Building program..." << std::endl;
    err = clBuildProgram(prog, 1, &device, "", nullptr, nullptr);
    size_t logSize=0;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    if(logSize>1){
        buildLog.resize(logSize);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
    }
    if(err!=CL_SUCCESS){
        std::cerr<<"[OpenCL] clBuildProgram failed with error: " << err << std::endl;
        std::cerr<<"[OpenCL] Build log: " << buildLog << std::endl;
        clReleaseProgram(prog);
        return false;
    }

    std::cerr << "[OpenCL] Creating kernel with entry point: '" << entry << "'" << std::endl;
    kernel = clCreateKernel(prog, entry.c_str(), &err);
    if(err!=CL_SUCCESS){
        std::cerr<<"[OpenCL] clCreateKernel failed with error: " << err << std::endl;
        std::cerr<<"[OpenCL] Entry point: '" << entry << "'" << std::endl;
        std::cerr<<"[OpenCL] Build log: " << buildLog << std::endl;
        clReleaseProgram(prog);
        return false;
    }

    std::cerr << "[OpenCL] Kernel created successfully" << std::endl;
    return true;
}

ExecResult OpenCLExecutor::run_task(const json& task){
    ExecResult r;
    auto t0 = std::chrono::high_resolution_clock::now();

    // helper to attach elapsed time before returning
    auto finish = [&](ExecResult res)->ExecResult {
        auto t1 = std::chrono::high_resolution_clock::now();
        res.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return res;
    };

    std::cerr << "[OpenCL] Starting task execution..." << std::endl;
    std::cerr << "[OpenCL] Task JSON: " << task.dump(2) << std::endl;

    std::string source = task.value("source","");
    std::string entry = task.value("entry","main");

    std::cerr << "[OpenCL] Source length: " << source.length() << std::endl;
    std::cerr << "[OpenCL] Entry point: '" << entry << "'" << std::endl;

    // Parse uniforms (packed 64-bit values)
    std::vector<uint64_t> uniforms;
    if(task.contains("uniforms") && task["uniforms"].is_array()){
        std::cerr << "[OpenCL] Parsing uniforms..." << std::endl;
        for(auto& v: task["uniforms"]){
            if(v.is_number_unsigned()) uniforms.push_back(v.get<uint64_t>());
            else if(v.is_number_integer()) uniforms.push_back((uint64_t)v.get<long long>());
            else if(v.is_number_float()){ double d=v.get<double>(); uint64_t u; std::memcpy(&u,&d,sizeof(double)); uniforms.push_back(u); }
        }
        std::cerr << "[OpenCL] Parsed " << uniforms.size() << " uniforms" << std::endl;
    }

    // Inputs
    std::vector<std::vector<uint8_t>> inputs;
    if(task.contains("inputs") && task["inputs"].is_array()){
        std::cerr << "[OpenCL] Parsing inputs..." << std::endl;
        for(auto& it: task["inputs"]){
            extern std::vector<uint8_t> base64_decode(const std::string&);
            inputs.push_back(base64_decode(it.value("data","")));
        }
        std::cerr << "[OpenCL] Parsed " << inputs.size() << " inputs" << std::endl;
        for(size_t i = 0; i < inputs.size(); ++i) {
            std::cerr << "[OpenCL] Input " << i << " size: " << inputs[i].size() << " bytes" << std::endl;
        }
    }

    std::vector<size_t> outputSizes;
    if(task.contains("outputSizes") && task["outputSizes"].is_array()){
        std::cerr << "[OpenCL] Parsing output sizes..." << std::endl;
        for(auto& x: task["outputSizes"]) if(x.is_number_unsigned()) outputSizes.push_back(x.get<size_t>());
        std::cerr << "[OpenCL] Parsed " << outputSizes.size() << " output sizes" << std::endl;
        for(size_t i = 0; i < outputSizes.size(); ++i) {
            std::cerr << "[OpenCL] Output " << i << " size: " << outputSizes[i] << " bytes" << std::endl;
        }
    }

    // Small RAII guard that releases OpenCL resources on scope exit.
    struct CLGuard {
        cl_program prog = nullptr;
        cl_kernel kernel = nullptr;
        std::vector<cl_mem> inBufs;
        std::vector<cl_mem> outBufs;
        cl_command_queue queue = nullptr;

        ~CLGuard(){
            for(auto m: inBufs) if(m) clReleaseMemObject(m);
            for(auto m: outBufs) if(m) clReleaseMemObject(m);
            if(kernel) clReleaseKernel(kernel);
            if(prog) clReleaseProgram(prog);
            // Do not release queue/context here - they are owned by the executor
        }
    } guard;

    // Build kernel (guard.prog / guard.kernel will be released by CLGuard destructor)
    std::string buildLog;
    if(!build_kernel(source, entry, guard.prog, guard.kernel, buildLog)){
        r.error = std::string("build failed: ") + buildLog;
        return finish(r);
    }

    guard.queue = queue; // not owned, just for clarity (do not release)

    cl_int err = CL_SUCCESS;
    int argIndex = 0;

    std::cerr << "[OpenCL] Setting kernel arguments..." << std::endl;

    // Create input buffers and set kernel args for them (first 2 arguments)
    std::cerr << "[OpenCL] Creating input buffers..." << std::endl;
    guard.inBufs.resize(inputs.size(), nullptr);
    for(size_t i=0;i<inputs.size();++i){
        std::cerr << "[OpenCL] Creating input buffer " << i << " with size " << inputs[i].size() << std::endl;
        guard.inBufs[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputs[i].size(), (void*)inputs[i].data(), &err);
        if(err!=CL_SUCCESS){
            std::cerr << "[OpenCL] clCreateBuffer input failed with error: " << err << std::endl;
            r.error = "clCreateBuffer input failed";
            return finish(r);
        }
        std::cerr << "[OpenCL] Setting input buffer arg " << argIndex << std::endl;
        err = clSetKernelArg(guard.kernel, argIndex++, sizeof(cl_mem), &guard.inBufs[i]);
        if(err!=CL_SUCCESS){
            std::cerr << "[OpenCL] clSetKernelArg input failed with error: " << err << std::endl;
            r.error = "clSetKernelArg input failed";
            return finish(r);
        }
    }

    // Create output buffers and set kernel args for them (3rd argument)
    std::cerr << "[OpenCL] Creating output buffers..." << std::endl;
    guard.outBufs.resize(outputSizes.size(), nullptr);
    for(size_t i=0;i<outputSizes.size();++i){
        std::cerr << "[OpenCL] Creating output buffer " << i << " with size " << outputSizes[i] << std::endl;
        guard.outBufs[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSizes[i], nullptr, &err);
        if(err!=CL_SUCCESS){
            std::cerr << "[OpenCL] clCreateBuffer output failed with error: " << err << std::endl;
            r.error = "clCreateBuffer output failed";
            return finish(r);
        }
        std::cerr << "[OpenCL] Setting output buffer arg " << argIndex << std::endl;
        err = clSetKernelArg(guard.kernel, argIndex++, sizeof(cl_mem), &guard.outBufs[i]);
        if(err!=CL_SUCCESS){
            std::cerr << "[OpenCL] clSetKernelArg output failed with error: " << err << std::endl;
            r.error = "clSetKernelArg output failed";
            return finish(r);
        }
    }

    // Set uniform args (arguments 4, 5, 6: rows, K, cols)
    for(size_t i=0;i<uniforms.size();++i){
        uint64_t u = uniforms[i];
        int intVal = static_cast<int>(u);  // Convert uint64_t to int for kernel
        std::cerr << "[OpenCL] Setting uniform arg " << i << " = " << u << " (as int: " << intVal << ")" << std::endl;
        err = clSetKernelArg(guard.kernel, argIndex++, sizeof(int), &intVal);
        if(err!=CL_SUCCESS){
            std::cerr << "[OpenCL] clSetKernelArg uniform failed with error: " << err << std::endl;
            r.error = "clSetKernelArg uniform failed";
            return finish(r);
        }
    }

    // Global/local sizes (default 1D with 3 entries)
    std::vector<size_t> global = {1,1,1}, local = {1,1,1};
    if(task.contains("global") && task["global"].is_array()){
        auto a=task["global"]; for(size_t i=0;i<a.size()&&i<3;i++) global[i]=a[i].get<size_t>();
    }
    if(task.contains("local") && task["local"].is_array()){
        auto a=task["local"]; for(size_t i=0;i<a.size()&&i<3;i++) local[i]=a[i].get<size_t>();
    }

    std::cerr << "[OpenCL] Global work size: [" << global[0] << ", " << global[1] << ", " << global[2] << "]" << std::endl;
    std::cerr << "[OpenCL] Local work size: [" << local[0] << ", " << local[1] << ", " << local[2] << "]" << std::endl;

    std::cerr << "[OpenCL] Enqueuing kernel execution..." << std::endl;
    err = clEnqueueNDRangeKernel(queue, guard.kernel, 3, nullptr, global.data(), local.data(), 0, nullptr, nullptr);
    if(err!=CL_SUCCESS){
        std::cerr << "[OpenCL] clEnqueueNDRangeKernel failed with error: " << err << std::endl;
        r.error = "clEnqueueNDRangeKernel failed";
        return finish(r);
    }

    std::cerr << "[OpenCL] Waiting for kernel completion..." << std::endl;
    clFinish(queue);
    std::cerr << "[OpenCL] Kernel execution completed" << std::endl;

    // Read outputs back
    std::cerr << "[OpenCL] Reading output buffers..." << std::endl;
    r.outputs.resize(guard.outBufs.size());
    for(size_t i=0;i<guard.outBufs.size();++i){
        std::cerr << "[OpenCL] Reading output buffer " << i << " with size " << outputSizes[i] << std::endl;
        r.outputs[i].resize(outputSizes[i]);
        err = clEnqueueReadBuffer(queue, guard.outBufs[i], CL_TRUE, 0, outputSizes[i], r.outputs[i].data(), 0, nullptr, nullptr);
        if(err!=CL_SUCCESS){
            std::cerr << "[OpenCL] clEnqueueReadBuffer failed with error: " << err << std::endl;
            r.error = "clEnqueueReadBuffer failed";
            return finish(r);
        }
    }

    std::cerr << "[OpenCL] Task execution completed successfully!" << std::endl;
    r.ok = true;
    return finish(r);
}

#else

bool OpenCLExecutor::initialize(const json& cfg){ (void)cfg; return false; }
ExecResult OpenCLExecutor::run_task(const json& task){ (void)task; return ExecResult{false,{},0.0,"OpenCL disabled"}; }

#endif
