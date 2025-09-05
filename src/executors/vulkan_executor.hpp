#pragma once
#include "iexecutor.hpp"
#include "kernel_cache.hpp"
#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif
#include <string>
#include <vector>
#include <memory>
#include <mutex>

class VulkanExecutor : public IExecutor {
public:
    VulkanExecutor() = default;
    bool initialize(const json& cfg) override;
    ExecResult run_task(const json& task) override;
    ~VulkanExecutor();

private:
#ifdef HAVE_VULKAN
    struct VulkanKernel {
        VkShaderModule module;
        std::vector<uint32_t> spirv;
    };
    
    // Static kernel cache
    static KernelCache<VulkanKernel> kernel_cache_;
    
    // Per-thread command pools
    thread_local static VkCommandPool thread_cmd_pool;
    std::mutex device_mutex_;
    
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;

    bool create_instance();
    bool pick_device();
    bool create_device();
    bool create_pools();
    VkCommandPool get_thread_cmd_pool();

    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem);
    uint32_t find_memory_type(uint32_t typeBits, VkMemoryPropertyFlags props);
    bool submit_and_wait(VkCommandBuffer cmd);

    std::shared_ptr<KernelCache<VulkanKernel>::CachedKernel> 
        get_or_build_shader(const std::string& glsl, const std::string& spirv_b64);
    bool build_pipeline_from_glsl(const std::string& glsl, const std::string& entry, VkShaderModule& module, std::vector<uint32_t>& spirv);
    bool build_pipeline_from_spirv(const std::vector<uint32_t>& spirv, VkShaderModule& module);
#endif
};
