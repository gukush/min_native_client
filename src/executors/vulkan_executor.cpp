
#include "vulkan_executor.hpp"
#include "base64.hpp"
#include <iostream>
#include <chrono>
#include <cstring>

#ifdef HAVE_SHADERC
#include <shaderc/shaderc.hpp>
#endif

#ifdef HAVE_VULKAN

KernelCache<VulkanExecutor::VulkanKernel> VulkanExecutor::kernel_cache_;
thread_local VkCommandPool VulkanExecutor::thread_cmd_pool = VK_NULL_HANDLE;

static const char* vkErr(VkResult r){
#define C(x) case x: return #x;
    switch(r){
        C(VK_SUCCESS) C(VK_ERROR_OUT_OF_HOST_MEMORY) C(VK_ERROR_OUT_OF_DEVICE_MEMORY)
        C(VK_ERROR_INITIALIZATION_FAILED) C(VK_ERROR_DEVICE_LOST) C(VK_ERROR_MEMORY_MAP_FAILED)
        C(VK_ERROR_LAYER_NOT_PRESENT) C(VK_ERROR_EXTENSION_NOT_PRESENT) C(VK_ERROR_FEATURE_NOT_PRESENT)
        C(VK_ERROR_INCOMPATIBLE_DRIVER) default: return "VK_ERROR_UNKNOWN";
    }
#undef C
}

VkCommandPool VulkanExecutor::get_thread_cmd_pool() {
    if (thread_cmd_pool == VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> lock(device_mutex_);
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.queueFamilyIndex = queueFamily;
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        auto r = vkCreateCommandPool(device, &cpi, nullptr, &thread_cmd_pool);
        if(r != VK_SUCCESS) {
            std::cerr<<"[Vulkan] Thread command pool creation failed: "<<vkErr(r)<<"\n";
            return VK_NULL_HANDLE;
        }
    }
    return thread_cmd_pool;
}


std::shared_ptr<KernelCache<VulkanExecutor::VulkanKernel>::CachedKernel>
VulkanExecutor::get_or_build_shader(const std::string& glsl, const std::string& spirv_b64) {
    std::string key = KernelCache<VulkanKernel>::computeHash(glsl + spirv_b64);

    auto cached = kernel_cache_.get(key);
    if(cached) {
        std::cout << "[Vulkan] Using cached shader module" << std::endl;
        return cached;
    }

    std::cout << "[Vulkan] Building new shader module" << std::endl;

    VkShaderModule module;
    std::vector<uint32_t> spirv;

    if(!glsl.empty()) {
        if(!build_pipeline_from_glsl(glsl, "main", module, spirv)) {
            return nullptr;
        }
    } else if(!spirv_b64.empty()) {
        auto spirvb = base64_decode(spirv_b64);
        spirv.resize((spirvb.size()+3)/4);
        std::memcpy(spirv.data(), spirvb.data(), spirvb.size());
        if(!build_pipeline_from_spirv(spirv, module)) {
            return nullptr;
        }
    } else {
        return nullptr;
    }

    auto cachedKernel = std::make_shared<KernelCache<VulkanKernel>::CachedKernel>();
    cachedKernel->kernel.module = module;
    cachedKernel->kernel.spirv = spirv;
    cachedKernel->lastUsed = std::chrono::steady_clock::now();

    kernel_cache_.put(key, cachedKernel);
    std::cout << "[Vulkan] Shader cached (cache size: " << kernel_cache_.size() << ")" << std::endl;
    return cachedKernel;
}

bool VulkanExecutor::create_instance(){
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "native-client-min";
    app.applicationVersion = VK_MAKE_VERSION(1,0,0);
    app.pEngineName = "none";
    app.engineVersion = VK_MAKE_VERSION(1,0,0);
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;

    VkResult r = vkCreateInstance(&ci, nullptr, &instance);
    if(r!=VK_SUCCESS){ std::cerr<<"vkCreateInstance: "<<vkErr(r)<<"\n"; return false; }
    return true;
}

bool VulkanExecutor::pick_device(){
    uint32_t count=0; vkEnumeratePhysicalDevices(instance,&count,nullptr);
    if(count==0){ std::cerr<<"No Vulkan devices\n"; return false; }
    std::vector<VkPhysicalDevice> devs(count); vkEnumeratePhysicalDevices(instance,&count,devs.data());
    for(auto d: devs){
        uint32_t qf=0; vkGetPhysicalDeviceQueueFamilyProperties(d,&qf,nullptr);
        std::vector<VkQueueFamilyProperties> qfp(qf); vkGetPhysicalDeviceQueueFamilyProperties(d,&qf,qfp.data());
        for(uint32_t i=0;i<qf;i++){
            if(qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT){
                phys=d; queueFamily=i; return true;
            }
        }
    }
    return false;
}

bool VulkanExecutor::create_device(){
    float prio=1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    auto r = vkCreateDevice(phys, &dci, nullptr, &device);
    if(r!=VK_SUCCESS){ std::cerr<<"vkCreateDevice: "<<vkErr(r)<<"\n"; return false; }
    vkGetDeviceQueue(device, queueFamily, 0, &queue);
    return true;
}

bool VulkanExecutor::create_pools(){
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 }
    };
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 16;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = sizes;
    auto r = vkCreateDescriptorPool(device, &pci, nullptr, &descPool);
    if(r!=VK_SUCCESS){ std::cerr<<"vkCreateDescriptorPool: "<<vkErr(r)<<"\n"; return false; }

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = queueFamily;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    r = vkCreateCommandPool(device, &cpi, nullptr, &cmdPool);
    if(r!=VK_SUCCESS){ std::cerr<<"vkCreateCommandPool: "<<vkErr(r)<<"\n"; return false; }
    return true;
}

bool VulkanExecutor::initialize(const json& cfg){
    (void)cfg;
    if(!create_instance()) return false;
    if(!pick_device()) return false;
    if(!create_device()) return false;
    if(!create_pools()) return false;
    return true;
}

uint32_t VulkanExecutor::find_memory_type(uint32_t typeBits, VkMemoryPropertyFlags props){
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++){
        if((typeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    return UINT32_MAX;
}

bool VulkanExecutor::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem){
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto r = vkCreateBuffer(device, &bi, nullptr, &buf);
    if(r!=VK_SUCCESS){ std::cerr<<"vkCreateBuffer: "<<vkErr(r)<<"\n"; return false; }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, buf, &mr);
    uint32_t idx = find_memory_type(mr.memoryTypeBits, props);
    if(idx==UINT32_MAX){ std::cerr<<"No suitable memory type\n"; return false; }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = idx;
    r = vkAllocateMemory(device, &ai, nullptr, &mem);
    if(r!=VK_SUCCESS){ std::cerr<<"vkAllocateMemory: "<<vkErr(r)<<"\n"; return false; }
    vkBindBufferMemory(device, buf, mem, 0);
    return true;
}

bool VulkanExecutor::build_pipeline_from_glsl(const std::string& glsl, const std::string& entry, VkShaderModule& module){
#ifndef HAVE_SHADERC
    std::cerr<<"shaderc not available; provide SPIR-V instead\n";
    return false;
#else
    shaderc::Compiler comp;
    shaderc::CompileOptions opts;
    auto res = comp.CompileGlslToSpv(glsl, shaderc_compute_shader, "shader.comp", opts);
    if(res.GetCompilationStatus()!=shaderc_compilation_status_success){
        std::cerr<<"shaderc compile error: "<<res.GetErrorMessage()<<std::endl;
        return false;
    }
    std::vector<uint32_t> spirv(res.cbegin(), res.cend());
    return build_pipeline_from_spirv(spirv, module);
#endif
}

bool VulkanExecutor::build_pipeline_from_spirv(const std::vector<uint32_t>& spirv, VkShaderModule& module){
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize = spirv.size()*sizeof(uint32_t);
    sci.pCode = spirv.data();
    auto r = vkCreateShaderModule(device, &sci, nullptr, &module);
    if(r!=VK_SUCCESS){ std::cerr<<"vkCreateShaderModule: "<<vkErr(r)<<"\n"; return false; }
    return true;
}

bool VulkanExecutor::submit_and_wait(VkCommandBuffer cmd){
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    auto r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    if(r!=VK_SUCCESS){ std::cerr<<"vkQueueSubmit: "<<vkErr(r)<<"\n"; return false; }
    vkQueueWaitIdle(queue);
    return true;
}

ExecResult VulkanExecutor::run_task(const json& task){
    ExecResult r;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::string entry = task.value("entry","main");
    std::string glsl = task.value("source_glsl","");
    std::vector<uint32_t> spirv;

    if(glsl.empty() && task.contains("spirv") && task["spirv"].is_string()){
        auto spirvb = base64_decode(task["spirv"].get<std::string>());
        spirv.resize((spirvb.size()+3)/4);
        std::memcpy(spirv.data(), spirvb.data(), spirvb.size());
    } else if(glsl.empty()) {
        r.error="No shader provided";
        return r;
    }

    // Parse buffers
    std::vector<std::vector<uint8_t>> inputs;
    if(task.contains("inputs") && task["inputs"].is_array()){
        for(auto& it: task["inputs"]){
            extern std::vector<uint8_t> base64_decode(const std::string&);
            inputs.push_back(base64_decode(it.value("data","")));
        }
    }
    std::vector<size_t> outputSizes;
    if(task.contains("outputSizes") && task["outputSizes"].is_array()){
        for(auto& x: task["outputSizes"]) if(x.is_number_unsigned()) outputSizes.push_back(x.get<size_t>());
    }

    std::vector<uint8_t> uniformsBuf;
    if(task.contains("uniforms") && task["uniforms"].is_array()){
        // For std140 layout with uint values:
        // - Each uint is 4 bytes
        // - Struct must be 16-byte aligned
        // - Need padding to reach 16 bytes total

        std::vector<uint32_t> uniforms32;
        for(auto& v: task["uniforms"]){
            uint32_t u=0;
            if(v.is_number_unsigned()) u = static_cast<uint32_t>(v.get<uint64_t>());
            else if(v.is_number_integer()) u = static_cast<uint32_t>(v.get<long long>());
            else if(v.is_number_float()) u = static_cast<uint32_t>(v.get<double>());
            uniforms32.push_back(u);
        }

        while(uniforms32.size() < 4) {
            uniforms32.push_back(0); // padding
        }

        uniformsBuf.resize(uniforms32.size() * sizeof(uint32_t));
        std::memcpy(uniformsBuf.data(), uniforms32.data(), uniformsBuf.size());
    }

    // Create shader module using cache
    std::string spirv_b64 = "";
    if(!glsl.empty()) {
        // For GLSL, we'll pass empty spirv_b64 and let get_or_build_shader handle GLSL compilation
        auto cached_shader = get_or_build_shader(glsl, spirv_b64);
        if(!cached_shader) {
            r.error = "shader compilation/caching failed";
            return r;
        }
        shader = cached_shader->kernel.module;
    } else {
        // For SPIR-V, encode to base64 and use cache
        spirv_b64 = base64_encode(std::string(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(uint32_t)));
        auto cached_shader = get_or_build_shader("", spirv_b64);
        if(!cached_shader) {
            r.error = "shader compilation/caching failed";
            return r;
        }
        shader = cached_shader->kernel.module;
    }

    // Descriptor set layout: 1 uniform buffer + N inputs + M outputs
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    uint32_t bindingIndex = 0;
    if(uniformsBuf.size()){
        VkDescriptorSetLayoutBinding b{};
        b.binding = bindingIndex++;
        b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
    }
    for(size_t i=0;i<inputs.size();++i){
        VkDescriptorSetLayoutBinding b{};
        b.binding = bindingIndex++;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
    }
    for(size_t i=0;i<outputSizes.size();++i){
        VkDescriptorSetLayoutBinding b{};
        b.binding = bindingIndex++;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
    }

    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = (uint32_t)bindings.size();
    lci.pBindings = bindings.data();
    VkDescriptorSetLayout dsl;
    if(vkCreateDescriptorSetLayout(device, &lci, nullptr, &dsl)!=VK_SUCCESS){ r.error="dsl create failed"; return r; }

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout layout;
    if(vkCreatePipelineLayout(device, &plci, nullptr, &layout)!=VK_SUCCESS){ r.error="pipeline layout failed"; return r; }

    VkPipelineShaderStageCreateInfo ssi{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssi.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssi.module = shader;
    ssi.pName = entry.c_str();

    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pci.stage = ssi;
    pci.layout = layout;
    VkPipeline pipe;
    if(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe)!=VK_SUCCESS){ r.error="pipeline create failed"; return r; }

    // Buffers
    VkBuffer uniB=VK_NULL_HANDLE; VkDeviceMemory uniM=VK_NULL_HANDLE;
    if(uniformsBuf.size()){
        if(!create_buffer(uniformsBuf.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniB, uniM)){ r.error="uniform buffer failed"; return r; }
        void* map=nullptr; vkMapMemory(device, uniM, 0, uniformsBuf.size(), 0, &map); std::memcpy(map, uniformsBuf.data(), uniformsBuf.size()); vkUnmapMemory(device, uniM);
    }
    std::vector<VkBuffer> inB(inputs.size());
    std::vector<VkDeviceMemory> inM(inputs.size());
    for(size_t i=0;i<inputs.size();++i){
        if(!create_buffer(inputs[i].size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, inB[i], inM[i])){ r.error="input buffer failed"; return r; }
        void* map=nullptr; vkMapMemory(device, inM[i], 0, inputs[i].size(), 0, &map); std::memcpy(map, inputs[i].data(), inputs[i].size()); vkUnmapMemory(device, inM[i]);
    }
    std::vector<VkBuffer> outB(outputSizes.size());
    std::vector<VkDeviceMemory> outM(outputSizes.size());
    for(size_t i=0;i<outputSizes.size();++i){
        if(!create_buffer(outputSizes[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, outB[i], outM[i])){ r.error="output buffer failed"; return r; }
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = descPool;
    dai.descriptorSetCount = 1;
    VkDescriptorSetLayout setLayouts[] = { dsl };
    dai.pSetLayouts = setLayouts;
    VkDescriptorSet dset;
    if(vkAllocateDescriptorSets(device, &dai, &dset)!=VK_SUCCESS){ r.error="alloc dset failed"; return r; }

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> infos;
    infos.reserve(bindings.size());
    uint32_t bind=0;
    if(uniformsBuf.size()){
        VkDescriptorBufferInfo bi{uniB,0,uniformsBuf.size()}; infos.push_back(bi);
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = dset; w.dstBinding = bind++; w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w.descriptorCount = 1; w.pBufferInfo = &infos.back();
        writes.push_back(w);
    }
    for(size_t i=0;i<inB.size();++i){
        VkDescriptorBufferInfo bi{inB[i],0,VK_WHOLE_SIZE}; infos.push_back(bi);
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = dset; w.dstBinding = bind++; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.descriptorCount = 1; w.pBufferInfo = &infos.back();
        writes.push_back(w);
    }
    for(size_t i=0;i<outB.size();++i){
        VkDescriptorBufferInfo bi{outB[i],0,VK_WHOLE_SIZE}; infos.push_back(bi);
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = dset; w.dstBinding = bind++; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.descriptorCount = 1; w.pBufferInfo = &infos.back();
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);

    // Command buffer - use thread-local command pool
    VkCommandPool threadPool = get_thread_cmd_pool();
    if(threadPool == VK_NULL_HANDLE) {
        r.error = "thread command pool creation failed";
        return r;
    }

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = threadPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    if(vkAllocateCommandBuffers(device, &cai, &cmd) != VK_SUCCESS) {
        r.error = "command buffer allocation failed";
        return r;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);

    // Dispatch
    uint32_t gx=1,gy=1,gz=1;
    if(task.contains("groups") && task["groups"].is_array()){
        auto g = task["groups"];
        if(g.size()>0) gx = g[0].get<uint32_t>();
        if(g.size()>1) gy = g[1].get<uint32_t>();
        if(g.size()>2) gz = g[2].get<uint32_t>();
    }
    vkCmdDispatch(cmd, gx, gy, gz);
    vkEndCommandBuffer(cmd);

    if(!submit_and_wait(cmd)){ r.error="submit failed"; return r; }

    // Read back
    r.outputs.resize(outB.size());
    for(size_t i=0;i<outB.size();++i){
        r.outputs[i].resize(outputSizes[i]);
        void* map=nullptr; vkMapMemory(device, outM[i], 0, outputSizes[i], 0, &map); std::memcpy(r.outputs[i].data(), map, outputSizes[i]); vkUnmapMemory(device, outM[i]);
    }
    r.ok=true;

    // Cleanup per-run resources (shader is cached, don't destroy)
    if(pipe) vkDestroyPipeline(device, pipe, nullptr);
    if(layout) vkDestroyPipelineLayout(device, layout, nullptr);
    if(dsl) vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    if(uniB) vkDestroyBuffer(device, uniB, nullptr);
    if(uniM) vkFreeMemory(device, uniM, nullptr);
    for(size_t i=0;i<inB.size();++i){ if(inB[i]) vkDestroyBuffer(device, inB[i], nullptr); if(inM[i]) vkFreeMemory(device, inM[i], nullptr); }
    for(size_t i=0;i<outB.size();++i){ if(outB[i]) vkDestroyBuffer(device, outB[i], nullptr); if(outM[i]) vkFreeMemory(device, outM[i], nullptr); }

    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

#else
bool VulkanExecutor::initialize(const json& cfg){ (void)cfg; return false; }
ExecResult VulkanExecutor::run_task(const json& task){ (void)task; return ExecResult{false,{},0.0,"Vulkan disabled"}; }
#endif
