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
    VulkanExecutor();
    bool initialize(const json& cfg) override;
    ExecResult run_task(const json& task) override;
    ~VulkanExecutor();

private:
#ifdef HAVE_VULKAN
    struct VulkanKernel {
        std::vector<uint32_t> spirv;
        std::string buildLog;
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
        get_or_compile_shader(const std::string& glsl_source, const std::string& spirv_b64);
    bool compile_glsl_to_spirv(const std::string& glsl, std::vector<uint32_t>& spirv, std::string& buildLog);
    std::vector<uint32_t> decode_spirv_b64(const std::string& spirv_b64);
#endif
};