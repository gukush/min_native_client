#pragma once
#include <thread>
#include <functional>
#include <nlohmann/json.hpp>
#include <boost/asio/ssl.hpp>

class CudaExecutor;
class OpenCLExecutor;
class VulkanExecutor;

class LocalWSServer {
public:
    using json = nlohmann::json;
    LocalWSServer();
    ~LocalWSServer();

    bool start(const std::string& address="127.0.0.1", unsigned short port=8787,
               const std::string& target="/native", bool use_ssl=true);
    void stop();

private:
    void run(const std::string& address, unsigned short port,
             const std::string& target, bool use_ssl);
    void run_ssl(const std::string& address, unsigned short port, const std::string& target);
    void run_plain(const std::string& address, unsigned short port, const std::string& target);

    bool running=false;
    std::thread th;
#ifdef HAVE_CUDA
    std::unique_ptr<CudaExecutor> cuda_executor_;
#endif
#ifdef HAVE_OPENCL
    std::unique_ptr<OpenCLExecutor> opencl_executor_;
#endif
#ifdef HAVE_VULKAN
    std::unique_ptr<VulkanExecutor> vulkan_executor_;
#endif
};