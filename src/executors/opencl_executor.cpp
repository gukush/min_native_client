#include "opencl_executor.hpp"

#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cstring>

#ifdef HAVE_OPENCL
    #ifndef CL_TARGET_OPENCL_VERSION
    #define CL_TARGET_OPENCL_VERSION 200
    #endif
#  include <CL/cl.h>
#endif

// base64.cpp is compiled but there is no header: forward-declare the symbols.
extern std::vector<unsigned char> base64_decode(const std::string&);
extern std::string base64_encode(const unsigned char*, size_t);

using json = IExecutor::json;

#ifdef HAVE_OPENCL

// Turn OpenCL error code to string (helper function - moved outside anonymous namespace)
const char* cl_err(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        default: return "CL_???";
    }
}

namespace {

struct ClProgram {
    cl_program h = nullptr;
    ~ClProgram() { if (h) clReleaseProgram(h); }
};

struct ClKernel {
    cl_kernel h = nullptr;
    ~ClKernel() { if (h) clReleaseKernel(h); }
};

struct ClBuf {
    cl_mem h = nullptr;
    ~ClBuf() { if (h) clReleaseMemObject(h); }
    // movable, not copyable
    ClBuf() = default;
    ClBuf(const ClBuf&) = delete;
    ClBuf& operator=(const ClBuf&) = delete;
    ClBuf(ClBuf&& o) noexcept { h = o.h; o.h = nullptr; }
    ClBuf& operator=(ClBuf&& o) noexcept {
        if (this != &o) {
            if (h) clReleaseMemObject(h);
            h = o.h;
            o.h = nullptr;
        }
        return *this;
    }
};

} // namespace

// Public static from header
std::mutex OpenCLExecutor::queue_mutex_;

// Simple: pick first available platform/device (header already declares pick_first_available()).
bool OpenCLExecutor::pick_first_available() {
    cl_int err = CL_SUCCESS;

    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "[OpenCL] clGetPlatformIDs failed: " << cl_err(err) << "\n";
        return false;
    }
    std::vector<cl_platform_id> plats(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, plats.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL] clGetPlatformIDs(vector) failed: " << cl_err(err) << "\n";
        return false;
    }

    for (auto p : plats) {
        cl_uint numDevs = 0;
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevs);
        if (err != CL_SUCCESS || numDevs == 0) continue;
        std::vector<cl_device_id> devs(numDevs);
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, numDevs, devs.data(), nullptr);
        if (err != CL_SUCCESS) continue;

        platform = p;
        device = devs[0];
        return true;
    }
    std::cerr << "[OpenCL] No devices found.\n";
    return false;
}

bool OpenCLExecutor::initialize(const json& cfg) {
    (void)cfg;
    if (!pick_first_available()) return false;

    cl_int err = CL_SUCCESS;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS) {
        std::cerr << "[OpenCL] clCreateContext failed: " << cl_err(err) << "\n";
        return false;
    }
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!queue || err != CL_SUCCESS) {
        std::cerr << "[OpenCL] clCreateCommandQueue failed: " << cl_err(err) << "\n";
        return false;
    }
    return true;
}

OpenCLExecutor::~OpenCLExecutor() {
#ifdef HAVE_OPENCL
    if (queue) {
        clReleaseCommandQueue(queue);
        queue = nullptr;
    }
    if (context) {
        clReleaseContext(context);
        context = nullptr;
    }
    // Note: device and platform don't need to be released
#endif
}

bool OpenCLExecutor::build_kernel(const std::string& source, const std::string& entry,
                                 cl_program& prog, cl_kernel& kernel,
                                 std::string& buildLog) {
    cl_int err = CL_SUCCESS;
    const char* src_ptr = source.c_str();
    size_t src_len = source.length();

    // Create program from source
    prog = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    if (!prog || err != CL_SUCCESS) {
        buildLog = "clCreateProgramWithSource failed: " + std::string(cl_err(err));
        return false;
    }

    // Build program
    err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            buildLog = std::string(log.data());
        }
        clReleaseProgram(prog);
        prog = nullptr;
        return false;
    }

    // Create kernel
    kernel = clCreateKernel(prog, entry.c_str(), &err);
    if (!kernel || err != CL_SUCCESS) {
        buildLog = "clCreateKernel failed: " + std::string(cl_err(err));
        clReleaseProgram(prog);
        prog = nullptr;
        return false;
    }

    return true;
}

ExecResult OpenCLExecutor::run_task(const json& task) {
    ExecResult r; r.ok = false; r.ms = 0.0;

    const std::string source = task.value("source", "");
    const std::string entry  = task.value("entry",  "execute_task");
    if (source.empty()) { r.error = "OpenCL: missing 'source'"; return r; }

    const bool dbg = (std::getenv("NATIVE_DEBUG") != nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1) Build / fetch kernel
    ClProgram program;
    ClKernel  kernel;
    {
        std::string buildLog;
        cl_program p = nullptr; cl_kernel k = nullptr;
        if (!build_kernel(source, entry, p, k, buildLog)) {
            r.error = "OpenCL: build failed" + (buildLog.empty() ? "" : (": " + buildLog));
            return r;
        }
        program.h = p; kernel.h = k;
    }

    // 2) Parse uniforms (pack as 32-bit ints)
    std::vector<int32_t> uniforms;
    if (task.contains("uniforms") && task["uniforms"].is_array()) {
        uniforms.reserve(task["uniforms"].size());
        for (const auto& u : task["uniforms"]) {
            int32_t v = 0;
            if (u.is_number_integer())      v = static_cast<int32_t>(u.get<int64_t>());
            else if (u.is_number_unsigned()) v = static_cast<int32_t>(u.get<uint64_t>());
            else if (u.is_number_float())    v = static_cast<int32_t>(u.get<double>());
            else { r.error = "OpenCL: uniform not numeric"; return r; }
            uniforms.push_back(v);
        }
    }

    // 3) Parse inputs (A,B) and allocate device buffers
    std::vector<ClBuf> inputs;
    if (task.contains("inputs") && task["inputs"].is_array()) {
        inputs.reserve(task["inputs"].size());
        for (const auto& it : task["inputs"]) {
            std::vector<unsigned char> host;
            if      (it.contains("b64")  && it["b64"].is_string())  host = base64_decode(it["b64"].get<std::string>());
            else if (it.contains("data") && it["data"].is_string()) host = base64_decode(it["data"].get<std::string>());

            cl_int err = CL_SUCCESS;
            ClBuf buf;
            buf.h = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | (host.empty() ? 0 : CL_MEM_COPY_HOST_PTR),
                                   host.size(),
                                   host.empty() ? nullptr : host.data(),
                                   &err);
            if (!buf.h || err != CL_SUCCESS) {
                r.error = "OpenCL: clCreateBuffer(input) failed (" + std::to_string(err) + ")";
                return r;
            }
            inputs.emplace_back(std::move(buf));
            if (dbg) std::cerr << "[OpenCL] input[" << (inputs.size()-1) << "] bytes=" << host.size() << "\n";
        }
    }

    // 4) Parse outputs and allocate device buffers
    auto get_u64 = [](const nlohmann::json& j, uint64_t& out) -> bool {
        if (j.is_number_unsigned()) { out = j.get<uint64_t>(); return true; }
        if (j.is_number_integer())  { int64_t v = j.get<int64_t>(); if (v < 0) return false; out = static_cast<uint64_t>(v); return true; }
        if (j.is_number_float())    { double  v = j.get<double>();  if (v < 0) return false; out = static_cast<uint64_t>(v); return true; }
        return false;
    };

    std::vector<size_t> outSizes;
    if (task.contains("outputSizes") && task["outputSizes"].is_array()) {
        outSizes.reserve(task["outputSizes"].size());
        for (const auto& o : task["outputSizes"]) {
            uint64_t v = 0;
            if (!get_u64(o, v)) { r.error = "OpenCL: outputSizes entries must be non-negative numbers"; return r; }
            outSizes.push_back(static_cast<size_t>(v));
        }
    }


    std::vector<ClBuf> outputs;
    outputs.reserve(outSizes.size());
    for (size_t i = 0; i < outSizes.size(); ++i) {
        size_t sz = outSizes[i];
        cl_int err = CL_SUCCESS;
        ClBuf buf;
        buf.h = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sz, nullptr, &err);
        if (!buf.h || err != CL_SUCCESS) {
            r.error = "OpenCL: clCreateBuffer(output) failed (" + std::to_string(err) + ")";
            return r;
        }
        outputs.emplace_back(std::move(buf));
    }

    // 5) Bind kernel args: UNIFORMS -> INPUTS -> OUTPUTS
    {
        cl_uint arg = 0;
        for (int32_t v : uniforms) {
            cl_int err = clSetKernelArg(kernel.h, arg++, sizeof(int32_t), &v);
            if (err != CL_SUCCESS) { r.error = "OpenCL: clSetKernelArg(uniform) failed (" + std::to_string(err) + ")";
                cl_uint numArgs = 0;
                clGetKernelInfo(kernel.h, CL_KERNEL_NUM_ARGS, sizeof(numArgs), &numArgs, nullptr);
                r.error = "OpenCL: clSetKernelArg(output) failed (" + std::to_string(err) +
                        "), arg_index=" + std::to_string(arg-1) +
                        ", kernel_num_args=" + std::to_string(numArgs);
                return r; }
        }
        for (auto& b : inputs) {
            cl_int err = clSetKernelArg(kernel.h, arg++, sizeof(cl_mem), &b.h);
            if (err != CL_SUCCESS) { r.error = "OpenCL: clSetKernelArg(input) failed (" + std::to_string(err) + ")";
                cl_uint numArgs = 0;
                clGetKernelInfo(kernel.h, CL_KERNEL_NUM_ARGS, sizeof(numArgs), &numArgs, nullptr);
                r.error = "OpenCL: clSetKernelArg(output) failed (" + std::to_string(err) +
                        "), arg_index=" + std::to_string(arg-1) +
                        ", kernel_num_args=" + std::to_string(numArgs);
                return r; }
        }
        for (auto& b : outputs) {
            cl_int err = clSetKernelArg(kernel.h, arg++, sizeof(cl_mem), &b.h);
            if (err != CL_SUCCESS) { r.error = "OpenCL: clSetKernelArg(output) failed (" + std::to_string(err) + ")";
                cl_uint numArgs = 0;
                clGetKernelInfo(kernel.h, CL_KERNEL_NUM_ARGS, sizeof(numArgs), &numArgs, nullptr);
                r.error = "OpenCL: clSetKernelArg(output) failed (" + std::to_string(err) +
                        "), arg_index=" + std::to_string(arg-1) +
                        ", kernel_num_args=" + std::to_string(numArgs);
                return r; }
        }
        if (dbg) std::cerr << "[OpenCL] args: uniforms=" << uniforms.size()
                           << " inputs=" << inputs.size()
                           << " outputs=" << outputs.size() << "\n";
    }

    // 6) Global / local sizes
    size_t g[3] = {0,0,0}, l[3] = {0,0,0};
    size_t work_dim = 1;
    bool have_local = false;

    if (task.contains("global")) {
         if (task["global"].is_array()) {
             work_dim = std::min<size_t>(3, task["global"].size());
             for (size_t i=0; i<work_dim; ++i) {
                 uint64_t v=0; if (!get_u64(task["global"][i], v)) { r.error="OpenCL: global dims must be non-negative numbers"; return r; }
                 g[i] = static_cast<size_t>(v);
             }
         } else if (task["global"].is_number()) {
             uint64_t v=0; if (!get_u64(task["global"], v)) { r.error="OpenCL: global must be a non-negative number"; return r; }
             g[0] = static_cast<size_t>(v);
            work_dim = 1;
        }
    } else if (!outSizes.empty()) {
        // fallback: 1D over first output byte-count (caller should set global explicitly)
        g[0] = outSizes[0];
        work_dim = 1;
    }

    if (task.contains("local") && task["local"].is_array()) {
        have_local = true;
        for (size_t i=0; i<std::min<size_t>(3, task["local"].size()); ++i)
            l[i] = static_cast<size_t>(task["local"][i].get<uint64_t>());
    }

    if (dbg) {
        std::cerr << "[OpenCL] work_dim=" << work_dim
                  << " global=(" << g[0] << "," << g[1] << "," << g[2] << ")"
                  << " local=("  << l[0] << "," << l[1] << "," << l[2] << ")\n";
    }

    // 7) Enqueue
    {
        cl_int err = clEnqueueNDRangeKernel(queue, kernel.h, work_dim, nullptr,
                                            g, have_local ? l : nullptr,
                                            0, nullptr, nullptr);
        if (err != CL_SUCCESS) { r.error = "OpenCL: clEnqueueNDRangeKernel failed (" + std::to_string(err) + ")"; return r; }
        clFinish(queue);
    }

    // 8) Read back
    r.outputs.clear();
    r.outputs.resize(outSizes.size());
    for (size_t i=0; i<outSizes.size(); ++i) {
        const size_t sz = outSizes[i];
        r.outputs[i].resize(sz);
        if (sz == 0) continue;
        cl_int err = clEnqueueReadBuffer(queue, outputs[i].h, CL_TRUE, 0, sz, r.outputs[i].data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { r.error = "OpenCL: clEnqueueReadBuffer failed (" + std::to_string(err) + ")"; return r; }
    }

    // Done
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.ok = true;
    return r;
}

#else // !HAVE_OPENCL

bool OpenCLExecutor::initialize(const json& cfg) { (void)cfg; return false; }
ExecResult OpenCLExecutor::run_task(const json& task) { (void)task; return ExecResult{false, {}, 0.0, "OpenCL disabled"}; }
OpenCLExecutor::~OpenCLExecutor() {}
#endif // HAVE_OPENCL