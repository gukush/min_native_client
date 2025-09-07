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
    ServerBinaryClient(bool insecure, int concurrency = 1, bool enable_listener = false);
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

    // Listener functionality
    void connect_to_listener();
    void notify_listener_chunk_arrival(const std::string& chunk_id, const std::string& task_id);
    void notify_listener_chunk_complete(const std::string& chunk_id, const std::string& status);

    std::unique_ptr<WebSocketClient> ws_;
    std::unique_ptr<BinaryExecutor> bin_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::atomic<bool> connected_{false};
    int max_concurrency_;
    std::atomic<int> active_chunks_{0};

    // Listener state
    bool enable_listener_;
    std::unique_ptr<WebSocketClient> listener_ws_;
};
