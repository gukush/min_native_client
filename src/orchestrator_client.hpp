#pragma once
#include <memory>
#include <queue>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <functional>
#include <optional>

#include <nlohmann/json.hpp>

#include "websocket_client.hpp"
#include "base64.hpp"

#include "executors/iexecutor.hpp"
#include "executors/binary_executor.hpp"
#include "executors/cuda_executor.hpp"
#include "executors/opencl_executor.hpp"
#include "executors/vulkan_executor.hpp"

#ifdef HAVE_LUA
#include "lua_host.hpp"
#endif

class OrchestratorClient {
public:
    using json = nlohmann::json;

    OrchestratorClient(bool insecure, int concurrency = 1);
    ~OrchestratorClient();

    bool connect(const std::string& url);
    void run(); // blocking
    void run(const std::atomic<bool>& interrupt_flag); // blocking with interrupt support

private:
    // Message handling
    void handle_message(const json& j);
    void handle_workload(const json& workload);
    void handle_chunk(const json& chunk);

    // Execution paths
    void process_chunk_concurrent(json chunk);
    void process_workload_concurrent(json workload);
    void process_chunk_impl(const json& chunk);
    void register_with_server();

    // Results
    static std::string sha256(const std::vector<uint8_t>& data);
    void send_chunk_done_enhanced(const nlohmann::json& chunk, const std::vector<uint8_t>& bytes, double processing_ms,
                                  const std::string& strategy, const nlohmann::json& executor_meta, const nlohmann::json& timings_meta,
                                  int64_t t_recv_ms, int64_t t_send_ms);

    void send_chunk_error_enhanced(const nlohmann::json& chunk,
                                   const std::string& err,
                                   int64_t t_recv_ms,
                                   int64_t t_send_ms);

    // Artifacts / Lua
#ifdef HAVE_LUA
    void maybe_load_host_lua_from_workload(const json& workload);
    std::optional<std::string> extract_lua_script(const json& workload);
#endif

    // Executor registry
    IExecutor* get_executor(const std::string& name);
    json run_executor(const std::string& name, const json& task);

    // Threading
    void schedule_or_run(json item, bool is_chunk);
    void on_task_finished();

private:
    std::unique_ptr<WebSocketClient> ws_;
    bool insecure_;
    int max_concurrency_;
    std::atomic<int> active_{0};

    // simple FIFO to throttle tasks if needed
    std::mutex q_mtx_;
    std::queue<std::pair<bool, json>> pending_; // (is_chunk, data)

    // executors
    std::unordered_map<std::string, std::unique_ptr<IExecutor>> executors_;

    // host Lua
#ifdef HAVE_LUA
    std::unique_ptr<LuaHost> lua_;
    bool lua_ready_ = false;
#endif

    // connection state
    std::atomic<bool> connected_{false};

    // current workload for artifact processing
    json current_workload_;
};
