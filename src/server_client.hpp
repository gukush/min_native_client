#pragma once
#include "websocket_client.hpp"
#include "executors/binary_executor.hpp"
#include <memory>
#include <queue>
#include <mutex>
#include <atomic>

class ThreadPool;

class ServerBinaryClient {
public:
    ServerBinaryClient(bool insecure, int concurrency = 1);
    ~ServerBinaryClient();
    bool connect(const std::string& url);
    void run();

private:
    void handle_message(const nlohmann::json& j);
    void handle_workload(const nlohmann::json& w);
    void handle_chunk(const nlohmann::json& c);
    void process_chunk_concurrent(const nlohmann::json& chunk);
    void process_workload_concurrent(const nlohmann::json& workload);

    std::tuple<std::string,std::string> split_host_port(const std::string& url);

    std::unique_ptr<WebSocketClient> ws_;
    std::unique_ptr<BinaryExecutor> bin_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::atomic<bool> connected_{false};
    int max_concurrency_;
    std::atomic<int> active_chunks_{0};
};
