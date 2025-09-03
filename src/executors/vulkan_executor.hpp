
#pragma once
#include "iexecutor.hpp"
#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif
#include <string>
#include <vector>

class VulkanExecutor : public IExecutor {
public:
    VulkanExecutor() = default;
    bool initialize(const json& cfg) override;
    ExecResult run_task(const json& task) override;

private:
#ifdef HAVE_VULKAN
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

    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem);
    uint32_t find_memory_type(uint32_t typeBits, VkMemoryPropertyFlags props);
    bool submit_and_wait(VkCommandBuffer cmd);

    bool build_pipeline_from_glsl(const std::string& glsl, const std::string& entry, VkShaderModule& module);
    bool build_pipeline_from_spirv(const std::vector<uint32_t>& spirv, VkShaderModule& module);
#endif
};
