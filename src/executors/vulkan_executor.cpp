#include "vulkan_executor.hpp"
#include "../base64.hpp"

#include <iostream>
#include <mutex>
#include <vector>
#include <cstring>
#include <chrono>

#ifdef HAVE_SHADERC
#include <shaderc/shaderc.hpp>
#endif

#ifdef HAVE_VULKAN

using json = IExecutor::json;

// Static member definition
KernelCache<VulkanExecutor::VulkanKernel> VulkanExecutor::kernel_cache_;

namespace {
inline const char* vk_err(VkResult r){
    switch(r){
#define C(x) case x: return #x
        C(VK_SUCCESS); C(VK_NOT_READY); C(VK_TIMEOUT); C(VK_EVENT_SET); C(VK_EVENT_RESET);
        C(VK_INCOMPLETE); C(VK_ERROR_OUT_OF_HOST_MEMORY); C(VK_ERROR_OUT_OF_DEVICE_MEMORY);
        C(VK_ERROR_INITIALIZATION_FAILED); C(VK_ERROR_DEVICE_LOST); C(VK_ERROR_MEMORY_MAP_FAILED);
        C(VK_ERROR_LAYER_NOT_PRESENT); C(VK_ERROR_EXTENSION_NOT_PRESENT); C(VK_ERROR_FEATURE_NOT_PRESENT);
        C(VK_ERROR_INCOMPATIBLE_DRIVER); C(VK_ERROR_FORMAT_NOT_SUPPORTED); C(VK_ERROR_FRAGMENTED_POOL);
#undef C
        default: return "VK_ERROR_UNKNOWN";
    }
}

inline std::vector<uint8_t> b64dec(const std::string& s){
    return base64_decode(s);
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props){
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;++i){
        if( (typeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & props) == props ) return i;
    }
    return UINT32_MAX;
}

static bool create_buffer(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem){
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if(vkCreateBuffer(dev, &bi, nullptr, &buf)!=VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, buf, &req);
    uint32_t type = find_memory_type(phys, req.memoryTypeBits, props);
    if(type==UINT32_MAX) return false;

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size; ai.memoryTypeIndex = type;
    if(vkAllocateMemory(dev, &ai, nullptr, &mem)!=VK_SUCCESS) return false;
    vkBindBufferMemory(dev, buf, mem, 0);
    return true;
}

static std::vector<uint32_t> b64_to_spirv(const std::string& b64) {
    auto bytes = b64dec(b64);
    size_t words = bytes.size()/4;
    std::vector<uint32_t> out(words);
    std::memcpy(out.data(), bytes.data(), words*4);
    return out;
}
} // namespace

// ---------- Shared context (singleton per process) ----------
struct VkCtx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;

    ~VkCtx(){
        if (dev) {
            if (pool) vkDestroyCommandPool(dev, pool, nullptr);
            vkDestroyDevice(dev, nullptr);
        }
        if (inst) vkDestroyInstance(inst, nullptr);
    }
};

static std::mutex g_ctx_mtx;
static std::unique_ptr<VkCtx> g_ctx;

// Helper function to check if device supports an extension
static bool check_extension_support(VkPhysicalDevice device, const char* extension_name) {
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, extensions.data());
    
    for (const auto& ext : extensions) {
        if (strcmp(ext.extensionName, extension_name) == 0) {
            return true;
        }
    }
    return false;
}

static bool ensure_context() {
    std::lock_guard<std::mutex> lk(g_ctx_mtx);
    if (g_ctx && g_ctx->dev) return true;

    g_ctx = std::make_unique<VkCtx>();

    // Instance
    {
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        ai.pApplicationName = "native-client";
        ai.apiVersion = VK_API_VERSION_1_1;
        
        // Enable instance extensions for compute performance
        const char* instance_extensions[] = {
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
        };
        
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &ai;
        ci.enabledExtensionCount = 1;
        ci.ppEnabledExtensionNames = instance_extensions;
        if (vkCreateInstance(&ci, nullptr, &g_ctx->inst) != VK_SUCCESS) {
            std::cerr << "[Vulkan] vkCreateInstance failed\n";
            return false;
        }
    }

    // Pick device with compute queue
    uint32_t ndev=0; vkEnumeratePhysicalDevices(g_ctx->inst, &ndev, nullptr);
    if (ndev == 0) { std::cerr << "[Vulkan] No devices\n"; return false; }

    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(g_ctx->inst, &ndev, devs.data());
    bool found=false;
    for (auto d : devs) {
        uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qf.data());
        for (uint32_t i=0;i<nq;++i) {
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                g_ctx->phys = d; g_ctx->qfam = i; found=true; break;
            }
        }
        if (found) break;
    }
    if (!found) { std::cerr << "[Vulkan] No compute queue\n"; return false; }

    // Logical device
    {
        float prio=1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = g_ctx->qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        // Check and enable device extensions for compute and matmul acceleration
        std::vector<const char*> enabled_extensions;
        const char* candidate_extensions[] = {
            VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
            VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
            VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
            VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME,
            VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
            VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME
        };
        
        for (const char* ext : candidate_extensions) {
            if (check_extension_support(g_ctx->phys, ext)) {
                enabled_extensions.push_back(ext);
                std::cout << "[Vulkan] Enabling extension: " << ext << std::endl;
            } else {
                std::cout << "[Vulkan] Extension not supported: " << ext << std::endl;
            }
        }

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
        dci.ppEnabledExtensionNames = enabled_extensions.empty() ? nullptr : enabled_extensions.data();
        if (vkCreateDevice(g_ctx->phys, &dci, nullptr, &g_ctx->dev) != VK_SUCCESS) {
            std::cerr << "[Vulkan] vkCreateDevice failed\n";
            return false;
        }
        vkGetDeviceQueue(g_ctx->dev, g_ctx->qfam, 0, &g_ctx->queue);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = g_ctx->qfam;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(g_ctx->dev, &pci, nullptr, &g_ctx->pool) != VK_SUCCESS) {
            std::cerr << "[Vulkan] vkCreateCommandPool failed\n";
            return false;
        }
    }
    return true;
}

// ---------- Shader caching methods ----------
std::shared_ptr<KernelCache<VulkanExecutor::VulkanKernel>::CachedKernel>
VulkanExecutor::get_or_compile_shader(const std::string& glsl_source, const std::string& spirv_b64) {
    // Create cache key based on the input (either GLSL source or SPIR-V base64)
    std::string cache_input = !spirv_b64.empty() ? spirv_b64 : glsl_source;
    std::string key = KernelCache<VulkanKernel>::computeHash(cache_input);

    auto cached = kernel_cache_.get(key);
    if(cached) {
        std::cout << "[Vulkan] Using cached shader" << std::endl;
        return cached;
    }

    std::cout << "[Vulkan] Compiling new shader" << std::endl;

    auto kernel_wrapper = std::make_shared<KernelCache<VulkanKernel>::CachedKernel>();

    if (!spirv_b64.empty()) {
        // Use provided SPIR-V
        kernel_wrapper->kernel.spirv = decode_spirv_b64(spirv_b64);
        if (kernel_wrapper->kernel.spirv.empty()) {
            std::cerr << "[Vulkan] Failed to decode SPIR-V from base64" << std::endl;
            return nullptr;
        }
        kernel_wrapper->kernel.buildLog = "SPIR-V loaded from base64";
    } else if (!glsl_source.empty()) {
        // Compile GLSL to SPIR-V
        if (!compile_glsl_to_spirv(glsl_source, kernel_wrapper->kernel.spirv, kernel_wrapper->kernel.buildLog)) {
            std::cerr << "[Vulkan] Shader compilation failed: " << kernel_wrapper->kernel.buildLog << std::endl;
            return nullptr;
        }
    } else {
        std::cerr << "[Vulkan] No shader source provided" << std::endl;
        return nullptr;
    }

    kernel_wrapper->lastUsed = std::chrono::steady_clock::now();
    kernel_cache_.put(key, kernel_wrapper);
    std::cout << "[Vulkan] Shader compiled and cached" << std::endl;

    return kernel_wrapper;
}

bool VulkanExecutor::compile_glsl_to_spirv(const std::string& glsl, std::vector<uint32_t>& spirv, std::string& buildLog) {
#ifdef HAVE_SHADERC
    try {
        shaderc::Compiler compiler;
        shaderc::CompileOptions options;
        options.SetOptimizationLevel(shaderc::OptimizationLevel::Performance);

        auto result = compiler.CompileGlslToSpv(glsl, shaderc_compute_shader, "shader.comp", options);

        if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
            buildLog = result.GetErrorMessage();
            return false;
        }

        spirv = {result.cbegin(), result.cend()};
        buildLog = "Compilation successful";
        return true;
    } catch (const std::exception& e) {
        buildLog = std::string("Exception during compilation: ") + e.what();
        return false;
    }
#else
    buildLog = "shaderc not available";
    return false;
#endif
}

std::vector<uint32_t> VulkanExecutor::decode_spirv_b64(const std::string& spirv_b64) {
    return b64_to_spirv(spirv_b64);
}

// ---------- Executor ----------
VulkanExecutor::VulkanExecutor() = default;
VulkanExecutor::~VulkanExecutor() = default;

bool VulkanExecutor::initialize(const json& cfg) {
    (void)cfg;
    return ensure_context();
}

ExecResult VulkanExecutor::run_task(const json& task) {
    ExecResult r;
    r.ok = false;
    r.ms = 0.0;
    r.error.clear();
    r.outputs.clear();

    if (!ensure_context()) { r.error = "Vulkan init failed"; return r; }
    auto& C = *g_ctx;

    // Get SPIR-V using cached compilation
    std::string spirv_b64 = task.value("spirv_b64", std::string());
    std::string glsl_source = task.value("source_glsl", std::string());

    auto cached_shader = get_or_compile_shader(glsl_source, spirv_b64);
    if (!cached_shader) {
        r.error = "Vulkan: shader compilation/loading failed";
        return r;
    }

    const std::vector<uint32_t>& spirv = cached_shader->kernel.spirv;

    // Inputs/outputs
    std::vector<std::vector<uint8_t>> inputs;
    if (task.contains("inputs") && task["inputs"].is_array()) {
        inputs.reserve(task["inputs"].size());
        for (const auto& in : task["inputs"]) inputs.emplace_back(b64dec(in.value("data","")));
    }

    std::vector<size_t> outputSizes;
    if (task.contains("outputSizes") && task["outputSizes"].is_array()) {
        outputSizes.reserve(task["outputSizes"].size());
        for (const auto& s : task["outputSizes"]) outputSizes.push_back(s.get<size_t>());
    }

    // Optional push constants (uniforms: array of u64 -> raw bytes)
    std::vector<uint8_t> uniforms_bytes;
    if (task.contains("uniforms") && task["uniforms"].is_array()) {
        uniforms_bytes.resize(task["uniforms"].size() * sizeof(uint64_t));
        size_t o = 0;
        for (const auto& u : task["uniforms"]) {
            uint64_t v = u.get<uint64_t>();
            std::memcpy(uniforms_bytes.data() + o, &v, sizeof(uint64_t));
            o += sizeof(uint64_t);
        }
    }

    // Groups (dispatch)
    auto groups = task.value("groups", std::vector<uint32_t>{1,1,1});
    uint32_t gx = groups.size()>0 ? groups[0] : 1;
    uint32_t gy = groups.size()>1 ? groups[1] : 1;
    uint32_t gz = groups.size()>2 ? groups[2] : 1;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkQueryPool qp = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;

    std::vector<VkBuffer> in_bufs;
    std::vector<VkDeviceMemory> in_mem;

    // Shader module
    {
        VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sci.codeSize = spirv.size() * sizeof(uint32_t);
        sci.pCode = spirv.data();
        if (vkCreateShaderModule(C.dev, &sci, nullptr, &shader) != VK_SUCCESS) {
            r.error = "vkCreateShaderModule failed";
            // Clean up and return
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
    }

    // Descriptor set layout: N inputs + M outputs
    const uint32_t numIn  = static_cast<uint32_t>(inputs.size());
    const uint32_t numOut = static_cast<uint32_t>(outputSizes.size());

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(numIn + numOut);

    uint32_t bindex = 0;
    for (uint32_t i=0;i<numIn;++i) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = bindex++;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
    }
    for (uint32_t i=0;i<numOut;++i) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = bindex++;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
    }

    {
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = static_cast<uint32_t>(bindings.size());
        lci.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(C.dev, &lci, nullptr, &setLayout) != VK_SUCCESS) {
            r.error="vkCreateDescriptorSetLayout failed";
            // Clean up and return
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
    }

    // Pipeline layout (with optional push constants)
    {
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &setLayout;

        VkPushConstantRange pcr{};
        if (!uniforms_bytes.empty()) {
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.offset = 0;
            pcr.size = static_cast<uint32_t>(uniforms_bytes.size());
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcr;
        }

        if (vkCreatePipelineLayout(C.dev, &plci, nullptr, &pipeLayout) != VK_SUCCESS) {
            r.error="vkCreatePipelineLayout failed";
            // Clean up and return
            if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
    }

    // Pipeline
    {
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = stage;
        cpi.layout = pipeLayout;

        if (vkCreateComputePipelines(C.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline) != VK_SUCCESS) {
            r.error = "vkCreateComputePipelines failed";
            // Clean up and return
            if (pipeline) vkDestroyPipeline(C.dev, pipeline, nullptr);
            if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
    }

    // Buffers: host-visible for simplicity
    in_bufs.resize(numIn);
    in_mem.resize(numIn);
    for (uint32_t i=0;i<numIn;++i) {
        const auto& h = inputs[i];
        if (!::create_buffer(C.dev, C.phys, h.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           in_bufs[i], in_mem[i])) {
            r.error="input buffer alloc failed";
            // Clean up and return
            for (auto m : in_mem) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : in_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            if (pipeline) vkDestroyPipeline(C.dev, pipeline, nullptr);
            if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
        void* p = nullptr; vkMapMemory(C.dev, in_mem[i], 0, h.size(), 0, &p);
        std::memcpy(p, h.data(), h.size());
        vkUnmapMemory(C.dev, in_mem[i]);
    }

    // Outputs: allocate all (if multiple)
    std::vector<VkBuffer> out_bufs(numOut, VK_NULL_HANDLE);
    std::vector<VkDeviceMemory> out_mems(numOut, VK_NULL_HANDLE);
    for (uint32_t i=0;i<numOut;++i) {
        const VkDeviceSize sz = static_cast<VkDeviceSize>(outputSizes[i]);
        if (!::create_buffer(C.dev, C.phys, sz,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           out_bufs[i], out_mems[i])) {
            r.error="output buffer alloc failed";
            // Clean up and return
            for (auto m : out_mems) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : out_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            for (auto m : in_mem) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : in_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            if (pipeline) vkDestroyPipeline(C.dev, pipeline, nullptr);
            if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
        if (sz) {
            void* p = nullptr; vkMapMemory(C.dev, out_mems[i], 0, sz, 0, &p);
            std::memset(p, 0, static_cast<size_t>(sz));
            vkUnmapMemory(C.dev, out_mems[i]);
        }
    }

    // Descriptor pool + set
    {
        VkDescriptorPoolSize psize{};
        psize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        psize.descriptorCount = static_cast<uint32_t>(bindings.size());

        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &psize;

        if (vkCreateDescriptorPool(C.dev, &dpci, nullptr, &dpool) != VK_SUCCESS) {
            r.error="vkCreateDescriptorPool failed";
            // Clean up and return
            if (dpool) vkDestroyDescriptorPool(C.dev, dpool, nullptr);
            for (auto m : out_mems) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : out_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            for (auto m : in_mem) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : in_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            if (pipeline) vkDestroyPipeline(C.dev, pipeline, nullptr);
            if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }

        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &setLayout;

        if (vkAllocateDescriptorSets(C.dev, &dsai, &dset) != VK_SUCCESS) {
            r.error="vkAllocateDescriptorSets failed";
            // Clean up and return
            if (dpool) vkDestroyDescriptorPool(C.dev, dpool, nullptr);
            for (auto m : out_mems) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : out_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            for (auto m : in_mem) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : in_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            if (pipeline) vkDestroyPipeline(C.dev, pipeline, nullptr);
            if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
    }

    // Update descriptors
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(bindings.size());
    std::vector<VkDescriptorBufferInfo> infos;
    infos.reserve(bindings.size());

    uint32_t bindIdx = 0;
    for (uint32_t i=0;i<numIn;++i) {
        VkDescriptorBufferInfo bi{ in_bufs[i], 0, VK_WHOLE_SIZE };
        infos.push_back(bi);
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = dset; w.dstBinding = bindIdx++;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo = &infos.back();
        writes.push_back(w);
    }
    for (uint32_t i=0;i<numOut;++i) {
        VkDescriptorBufferInfo bi{ out_bufs[i], 0, VK_WHOLE_SIZE };
        infos.push_back(bi);
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = dset; w.dstBinding = bindIdx++;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo = &infos.back();
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(C.dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Command buffer
    {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = C.pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(C.dev, &cbai, &cmd) != VK_SUCCESS) {
            r.error="vkAllocateCommandBuffers failed";
            // Clean up and return
            if (cmd) vkFreeCommandBuffers(C.dev, C.pool, 1, &cmd);
            if (dpool) vkDestroyDescriptorPool(C.dev, dpool, nullptr);
            for (auto m : out_mems) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : out_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            for (auto m : in_mem) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : in_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            if (pipeline) vkDestroyPipeline(C.dev, pipeline, nullptr);
            if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
    }

    // Timestamp query pool
    {
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = 2;
        if (vkCreateQueryPool(C.dev, &qci, nullptr, &qp) != VK_SUCCESS) {
            r.error="vkCreateQueryPool failed";
            // Clean up and return
            if (qp) vkDestroyQueryPool(C.dev, qp, nullptr);
            if (cmd) vkFreeCommandBuffers(C.dev, C.pool, 1, &cmd);
            if (dpool) vkDestroyDescriptorPool(C.dev, dpool, nullptr);
            for (auto m : out_mems) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : out_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            for (auto m : in_mem) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : in_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            if (pipeline) vkDestroyPipeline(C.dev, pipeline, nullptr);
            if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
    }

    // Record
    {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &dset, 0, nullptr);

        if (!uniforms_bytes.empty()) {
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               static_cast<uint32_t>(uniforms_bytes.size()), uniforms_bytes.data());
        }

        vkCmdResetQueryPool(cmd, qp, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp, 0);
        vkCmdDispatch(cmd, gx, gy, gz);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp, 1);
        vkEndCommandBuffer(cmd);
    }

    // Submit and wait
    {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(C.dev, &fci, nullptr, &fence) != VK_SUCCESS) {
            r.error="vkCreateFence failed";
            // Clean up and return
            if (fence) vkDestroyFence(C.dev, fence, nullptr);
            if (qp) vkDestroyQueryPool(C.dev, qp, nullptr);
            if (cmd) vkFreeCommandBuffers(C.dev, C.pool, 1, &cmd);
            if (dpool) vkDestroyDescriptorPool(C.dev, dpool, nullptr);
            for (auto m : out_mems) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : out_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            for (auto m : in_mem) if (m) vkFreeMemory(C.dev, m, nullptr);
            for (auto b : in_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
            if (pipeline) vkDestroyPipeline(C.dev, pipeline, nullptr);
            if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
            if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);
            return r;
        }
        vkQueueSubmit(C.queue, 1, &si, fence);
        vkWaitForFences(C.dev, 1, &fence, VK_TRUE, 5000000000ULL); // 5s, fixed from UINT64_C(5e9)
        vkDestroyFence(C.dev, fence, nullptr);
    }

    // Kernel GPU time from timestamps
    {
        VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(C.phys, &props);
        uint64_t ts[2] = {0,0};
        VkResult gr = vkGetQueryPoolResults(C.dev, qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (gr == VK_SUCCESS && ts[1] > ts[0]) {
            double ns = double(ts[1] - ts[0]) * double(props.limits.timestampPeriod);
            r.ms = ns / 1.0e6;
        }
    }

    // Read back outputs
    r.outputs.clear();
    r.outputs.resize(numOut);
    for (uint32_t i=0;i<numOut;++i) {
        const size_t sz = outputSizes[i];
        r.outputs[i].resize(sz);
        if (sz == 0) continue;
        void* p = nullptr; vkMapMemory(C.dev, out_mems[i], 0, sz, 0, &p);
        std::memcpy(r.outputs[i].data(), p, sz);
        vkUnmapMemory(C.dev, out_mems[i]);
    }

    r.ok = true;
    r.timings = {
        {"compileMs", 0.0},
        {"h2dMs", 0.0},
        {"kernelMs", r.ms},
        {"d2hMs", 0.0}
    };
    r.error.clear();

    // Clean up
    if (qp) vkDestroyQueryPool(C.dev, qp, nullptr);
    if (cmd) vkFreeCommandBuffers(C.dev, C.pool, 1, &cmd);
    if (dpool) vkDestroyDescriptorPool(C.dev, dpool, nullptr);
    for (auto m : out_mems) if (m) vkFreeMemory(C.dev, m, nullptr);
    for (auto b : out_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
    for (auto m : in_mem) if (m) vkFreeMemory(C.dev, m, nullptr);
    for (auto b : in_bufs) if (b) vkDestroyBuffer(C.dev, b, nullptr);
    if (pipeline) vkDestroyPipeline(C.dev, pipeline, nullptr);
    if (pipeLayout) vkDestroyPipelineLayout(C.dev, pipeLayout, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(C.dev, setLayout, nullptr);
    if (shader) vkDestroyShaderModule(C.dev, shader, nullptr);

    return r;
}

#else // !HAVE_VULKAN

VulkanExecutor::VulkanExecutor() = default;
VulkanExecutor::~VulkanExecutor() = default;
bool VulkanExecutor::initialize(const json& cfg){ (void)cfg; return false; }
ExecResult VulkanExecutor::run_task(const json& task){ (void)task; return ExecResult{false, {}, 0.0, "Vulkan disabled"}; }

#endif // HAVE_VULKAN