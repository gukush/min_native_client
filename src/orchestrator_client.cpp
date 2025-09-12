#include "orchestrator_client.hpp"
#include "base64.hpp"
#include <openssl/sha.h>
#include <iomanip>
#include <sstream>
#include <thread>
#include <future>
#include <chrono>
#include <iostream>

using json = nlohmann::json;

// --------------------- helpers ---------------------

std::string OrchestratorClient::sha256(const std::vector<uint8_t>& data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, data.data(), data.size());
    SHA256_Final(hash, &ctx);
    std::ostringstream ss;
    for (unsigned char c : hash) ss << std::hex << std::setw(2) << std::setfill('0') << (int)c;
    return ss.str();
}

static inline int64_t now_ms_epoch() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// --------------------- ctor/dtor ---------------------

OrchestratorClient::OrchestratorClient(bool insecure, int concurrency)
    : ws_(std::make_unique<WebSocketClient>()),
      insecure_(insecure),
      max_concurrency_(std::max(1, concurrency)) {}

OrchestratorClient::~OrchestratorClient() = default;

// --------------------- connection + loop ---------------------

bool OrchestratorClient::connect(const std::string& url) {
    // Set up callbacks using the correct WebSocket client interface
    ws_->onConnected = [this]() {
        connected_ = true;
        std::cout << "[ws] open\n";
        // Register with server immediately after connection
        register_with_server();
    };
    ws_->onDisconnected = [this]() { connected_ = false; std::cout << "[ws] closed\n"; };
    ws_->onJson = [this](const json& j) { handle_message(j); };

    // Parse URL to extract host, port, and path
    // Simple URL parsing - assumes format like "wss://host:port/path" or "ws://host:port/path"
    std::string protocol, host, port, path;
    if (url.find("://") != std::string::npos) {
        protocol = url.substr(0, url.find("://"));
        std::string rest = url.substr(url.find("://") + 3);
        if (rest.find("/") != std::string::npos) {
            std::string hostport = rest.substr(0, rest.find("/"));
            path = rest.substr(rest.find("/"));
            if (hostport.find(":") != std::string::npos) {
                host = hostport.substr(0, hostport.find(":"));
                port = hostport.substr(hostport.find(":") + 1);
            } else {
                host = hostport;
                port = (protocol == "wss" || protocol == "https") ? "443" : "80";
            }
        } else {
            host = rest;
            port = (protocol == "wss" || protocol == "https") ? "443" : "80";
            path = "/";
        }
    } else {
        // Fallback for simple host:port format
        if (url.find(":") != std::string::npos) {
            host = url.substr(0, url.find(":"));
            port = url.substr(url.find(":") + 1);
        } else {
            host = url;
            port = "80";
        }
        path = "/";
        protocol = insecure_ ? "ws" : "wss";
    }

    bool use_ssl = (protocol == "wss" || protocol == "https") && !insecure_;
    return ws_->connect(host, port, path, use_ssl);
}

void OrchestratorClient::run() {
    // The WebSocket client runs its event loop automatically in a separate thread
    // when connect() is called. We just need to keep the main thread alive.
    // In a real application, you might want to handle other events here.
    while (connected_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void OrchestratorClient::run(const std::atomic<bool>& interrupt_flag) {
    // The WebSocket client runs its event loop automatically in a separate thread
    // when connect() is called. We just need to keep the main thread alive.
    // Check for interrupt signal periodically.
    while (connected_ && !interrupt_flag.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (interrupt_flag.load()) {
        std::cout << "[Client] Shutting down due to interrupt signal..." << std::endl;
        // Disconnect gracefully
        if (ws_) {
            ws_->disconnect();
        }
    }
}

void OrchestratorClient::register_with_server() {
    // Send client:join message to register with the server
    json capabilities = {
        {"frameworks", {"opencl", "vulkan", "cuda", "exe"}},
        {"strategies", {"native-block-matmul", "native-block-matmul-flex","exe-block-matmul-flex","native-ecm-stage1"}},
        {"capacity", max_concurrency_},
        {"clientType", "native"}
    };

    json message = {
        {"type", "client:join"},
        {"data", capabilities}
    };

    std::cout << "[client] Registering with server: " << message.dump() << std::endl;
    ws_->send_json(message);
}

// --------------------- message dispatch ---------------------

void OrchestratorClient::handle_message(const json& j) {
    const auto type = j.value("type", std::string());
    const auto data = j.value("data", json::object());

    std::cout << "[client] Received message type: " << type << std::endl;

    if (type == "welcome") {
        std::cout << "[client] Server welcome: " << data.dump() << std::endl;
    } else if (type == "client:join:ack") {
        std::cout << "[client] Registration acknowledged: " << data.dump() << std::endl;
    } else if (type == "workload:new") {
        std::cout << "[client] New workload received" << std::endl;
        handle_workload(data);
    } else if (type == "chunk:assign") {
        std::cout << "[client] Chunk assigned" << std::endl;
        handle_chunk(data);
    } else if (type == "workload:ready") {
        std::cout << "[client] workload ready\n";
        // Send workload:ready acknowledgment back to server
        json ready_ack = {
            {"type", "workload:ready"},
            {"data", {"id", data.value("taskId", "")}}
        };
        ws_->send_json(ready_ack);
    } else if (type == "ping") {
        json pong = { {"type","pong"}, {"data", {}} };
        ws_->send_json(pong);
    } else {
        std::cout << "[client] unhandled type: " << type << "\n";
    }
}

// --------------------- workload / chunk ---------------------

void OrchestratorClient::handle_workload(const json& workload) {
#ifdef HAVE_LUA
    maybe_load_host_lua_from_workload(workload);
#endif
    schedule_or_run(workload, /*is_chunk*/false);
}

void OrchestratorClient::handle_chunk(const json& chunk) {
    schedule_or_run(chunk, /*is_chunk*/true);
}

void OrchestratorClient::schedule_or_run(json item, bool is_chunk) {
    // Throttle by max_concurrency_ (chunks only).
    if (is_chunk) {
        int running = active_.load();
        if (running >= max_concurrency_) {
            std::lock_guard<std::mutex> lk(q_mtx_);
            pending_.push({true, std::move(item)});
            return;
        }
        active_.fetch_add(1);
        std::async(std::launch::async, [this, c = std::move(item)]{
            process_chunk_impl(c);
            on_task_finished();
        }).wait(); // If you want non-blocking here, remove .wait(); keeping .wait() makes it simple/serial per thread
    } else {
        // workloads are light bookkeeping, run inline or async as you prefer
        std::async(std::launch::async, [this, w = std::move(item)]{
            process_workload_concurrent(w);
        });
    }
}

void OrchestratorClient::on_task_finished() {
    // pull next pending chunk if any
    std::pair<bool, json> next;
    {
        std::lock_guard<std::mutex> lk(q_mtx_);
        if (pending_.empty()) {
            active_.fetch_sub(1);
            return;
        }
        next = std::move(pending_.front());
        pending_.pop();
    }
    // run next chunk
    std::async(std::launch::async, [this, c = std::move(next.second)]{
        process_chunk_impl(c);
        on_task_finished();
    });
}

// --------------------- artifact / Lua host ---------------------

#ifdef HAVE_LUA
std::optional<std::string> OrchestratorClient::extract_lua_script(const json& workload) {
    // Expected shapes:
    // - data.artifacts[] objects with { type: "lua", name: "host.lua", data: "<base64 or text>" }
    // - or artifacts[].type=="text" and name=="host.lua"
    // - or payload.hostScript (string)
    const auto& artifacts = workload.value("artifacts", json::array());
    for (const auto& a : artifacts) {
        const auto t = a.value("type", std::string());
        const auto name = a.value("name", std::string());
        if (name == "host.lua" || t == "lua") {
            const std::string *p = nullptr;
            if (a.contains("data")  && a["data"].is_string())  p = &a["data"].get_ref<const std::string&>();
            else if (a.contains("bytes") && a["bytes"].is_string()) p = &a["bytes"].get_ref<const std::string&>();
            if (p) {
                const std::string& d = *p;
                // try base64 decode; if fails, treat as plain text
                try {
                    auto bin = base64_decode(d);
                    return std::string(bin.begin(), bin.end());
                } catch (...) {
                    return d;
                }
            }
        }
    }
    if (workload.contains("payload") && workload["payload"].contains("hostScript")) {
        return workload["payload"]["hostScript"].get<std::string>();
    }
    return std::nullopt;
}
#endif // HAVE_LUA

#ifdef HAVE_LUA
void OrchestratorClient::maybe_load_host_lua_from_workload(const json& workload) {
    auto src = extract_lua_script(workload);
    if (!src) { return; }
    lua_ = std::make_unique<LuaHost>();

    // Provide the callback that actually runs an executor
    auto cb = [this](const std::string& fw, const json& task) -> json {
        return this->run_executor(fw, task);
    };

    if (!lua_->load(*src, cb)) {
        std::cerr << "[lua] failed to load host.lua\n";
        lua_.reset();
        lua_ready_ = false;
        return;
    }

    // NEW: Pass artifacts to Lua host
    if (workload.contains("artifacts")) {
        lua_->set_artifacts(workload["artifacts"]);
    }

    if (!lua_->has_compile_and_run()) {
        std::cerr << "[lua] host.lua missing compile_and_run(chunk)\n";
        lua_.reset();
        lua_ready_ = false;
        return;
    }
    lua_ready_ = true;
    std::cout << "[lua] Loaded host lua from workload" <<std::endl;
}
#endif // HAVE_LUA

// --------------------- executors ---------------------

IExecutor* OrchestratorClient::get_executor(const std::string& name) {
    auto it = executors_.find(name);
    if (it != executors_.end()) return it->second.get();

    std::unique_ptr<IExecutor> exe;
    if (name == "binary") exe = std::make_unique<BinaryExecutor>();
#ifdef HAVE_CUDA
    else if (name == "cuda") exe = std::make_unique<CudaExecutor>();
#endif
#ifdef HAVE_OPENCL
    else if (name == "opencl") exe = std::make_unique<OpenCLExecutor>();
#endif
#ifdef HAVE_VULKAN
    else if (name == "vulkan") exe = std::make_unique<VulkanExecutor>();
#endif
    else return nullptr;

    if (!exe->initialize(json::object())) return nullptr;
    auto* raw = exe.get();
    executors_.emplace(name, std::move(exe));
    return raw;
}

json OrchestratorClient::run_executor(const std::string& name, const json& task) {
    auto* exe = get_executor(name);
    if (!exe) throw std::runtime_error("Executor not available: " + name);

    ExecResult result = exe->run_task(task);

    // Convert ExecResult to JSON format
    json result_json;
    result_json["ok"] = result.ok;
    result_json["error"] = result.error;
    result_json["ms"] = result.ms;
    result_json["timings"] = result.timings;

    // Convert outputs to base64 encoded strings
    if (!result.outputs.empty()) {
        json outputs_array = json::array();
        for (const auto& output : result.outputs) {
            outputs_array.push_back(base64_encode(output));
        }
        result_json["outputs"] = outputs_array;

        // For backward compatibility, also include the first output as "result"
        if (!result.outputs.empty() && !result.outputs[0].empty()) {
            result_json["result"] = base64_encode(result.outputs[0]);
            result_json["result_b64"] = base64_encode(result.outputs[0]);
        }
    }

    return result_json;
}

// --------------------- per-chunk processing ---------------------

void OrchestratorClient::process_workload_concurrent(json workload) {
    std::cout << "[client] Processing workload: " << workload.dump() << std::endl;

    // Extract workload information
    std::string taskId = workload.value("taskId", workload.value("id", ""));
    std::string framework = workload.value("framework", "");

    std::cout << "[client] Workload taskId: '" << taskId << "', framework: '" << framework << "'" << std::endl;
    std::cout << "[client] Workload keys: ";
    for (auto& [key, value] : workload.items()) {
        std::cout << key << " ";
    }
    std::cout << std::endl;

    // Process artifacts if present
    if (workload.contains("artifacts")) {
        auto artifacts = workload["artifacts"];
        std::cout << "[client] Found " << artifacts.size() << " artifacts" << std::endl;

        // Process artifacts immediately using the binary executor
        auto* exe = get_executor("binary");
        if (exe) {
            // Create a task for artifact processing
            json artifact_task = workload;
            artifact_task["id"] = taskId;

            // Process artifacts (this will download and cache them)
            exe->run_task(artifact_task);
            std::cout << "[client] Processed artifacts for task " << taskId << std::endl;
        }

        // Store the workload for the binary executor to process
        current_workload_ = workload;
    }

    // Send workload:ready message back to server
    // The server expects us to send the taskId back (even though it already knows it)
    json ready_msg = {
        {"type", "workload:ready"},
        {"data", {{"id", taskId}}}
    };

    std::cout << "[client] Sending workload:ready for task '" << taskId << "'" << std::endl;
    ws_->send_json(ready_msg);
}

void OrchestratorClient::process_chunk_impl(const json& chunk) {
    const auto t_recv_ms = now_ms_epoch();
    const auto started = std::chrono::high_resolution_clock::now();

    try {
        const auto meta    = chunk.value("meta",    json::object());
        const auto payload = chunk.value("payload", json::object());

        auto get_str = [](const json& j, const char* key) -> std::string {
        if (j.contains(key) && j[key].is_string()) return j[key].get<std::string>();
        return {};
        };
        auto norm = [](std::string s) {
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s.rfind("native-", 0) == 0) s.erase(0, 7);
        return s;
        };

        std::string framework = get_str(payload, "framework");
        if (framework.empty()) framework = get_str(chunk, "framework");
        if (framework.empty()) framework = get_str(meta,   "framework");
        if (framework.empty()) framework = get_str(meta,   "backend");
        framework = norm(framework);

        std::string action = norm(get_str(payload, "action")); // e.g., "exec", "compile_and_run", "cpu_matmul"

        std::string workload_framework;
        if (!current_workload_.is_null()) {
        workload_framework = norm(get_str(current_workload_, "framework"));
        }

        // Final inference
        if (framework.empty()) {
        if (action == "exec") framework = "exe";
        else if (action == "cpu_matmul") framework = "cpu";
        else if (!workload_framework.empty()) framework = workload_framework;
        }

        json exec_result;

#ifdef HAVE_LUA
        if (lua_ready_) {
            std::cout<< "[client] executing chunk with Lua" <<std::endl;
            exec_result = lua_->compile_and_run(chunk);
        } else
#endif
        {
            // Determine execution path based on workload framework
            std::cout << "[client] Effective framework: '" << framework << "'" << std::endl;
            // For exe framework, ALWAYS use binary executor regardless of backend in meta
            if (framework == "exe") {
                std::cout << "[client] Using binary executor for exe framework" << std::endl;

                const auto& payload = chunk.value("payload", json::object());
                const auto& meta = chunk.value("meta", json::object());

                // For exe-block-matmul-flex strategy, always use binary executor
                IExecutor* exe = get_executor("binary");
                if (!exe) throw std::runtime_error("Binary executor not available");

                // Prepare task for binary executor - merge payload into task properly
                json task_for_executor = payload; // This contains buffers, outputs etc.

                // Add additional metadata
                task_for_executor["taskId"] = chunk.value("taskId", "");
                task_for_executor["chunkId"] = chunk.value("chunkId", "");
                task_for_executor["meta"] = meta;

                // Add workload information if available for binary path resolution
                if (!current_workload_.is_null()) {
                    task_for_executor["workload"] = current_workload_;
                    // Extract task ID for binary path resolution
                    task_for_executor["id"] = current_workload_.value("taskId", current_workload_.value("id", ""));
                }

                std::cout << "[client] Executing chunk with binary executor" << std::endl;
                std::cout << "[client] Task data keys: ";
                for (auto& [key, value] : task_for_executor.items()) {
                    std::cout << key << " ";
                }
                std::cout << std::endl;

                ExecResult executor_result = exe->run_task(task_for_executor);

                // Convert ExecResult to JSON format
                exec_result["ok"] = executor_result.ok;
                exec_result["error"] = executor_result.error;
                exec_result["ms"] = executor_result.ms;
                exec_result["timings"] = executor_result.timings;

                std::cout << "[client] Binary executor result: ok=" << executor_result.ok
                         << ", error='" << executor_result.error << "', outputs.size=" << executor_result.outputs.size() << std::endl;

                // Convert outputs to base64 encoded strings
                if (!executor_result.outputs.empty()) {
                    json outputs_array = json::array();
                    for (const auto& output : executor_result.outputs) {
                        outputs_array.push_back(base64_encode(output));
                    }
                    exec_result["outputs"] = outputs_array;

                    // For backward compatibility, also include the first output as "result"
                    if (!executor_result.outputs.empty() && !executor_result.outputs[0].empty()) {
                        exec_result["result"] = base64_encode(executor_result.outputs[0]);
                        exec_result["result_b64"] = base64_encode(executor_result.outputs[0]);
                        std::cout << "[client] Added result fields, result size=" << executor_result.outputs[0].size() << std::endl;
                    }
                } else {
                    std::cout << "[client] No outputs from binary executor!" << std::endl;
                }

            } else if (framework == "opencl" || framework == "cuda" || framework == "vulkan" || framework == "cpu") {
                // Use framework-specific executor (opencl, cuda, vulkan) for non-exe frameworks
                std::cout << "[client] Using framework executor: " << framework << std::endl;
                const auto& payload = chunk.value("payload", json::object());
                exec_result = run_executor(framework, payload);
            } else {
                // Fallback to binary executor
                std::cout << "[client] Using binary executor (fallback)" << std::endl;
                const auto& payload = chunk.value("payload", json::object());
                exec_result = run_executor("binary", payload);
            }
        }

        // Extract result bytes - add better error reporting
        std::vector<uint8_t> bytes;
        if (exec_result.contains("result_b64")) {
            auto b = base64_decode(exec_result["result_b64"].get<std::string>());
            bytes.assign(b.begin(), b.end());
        } else if (exec_result.contains("result") && exec_result["result"].is_string()) {
            auto b = base64_decode(exec_result["result"].get<std::string>());
            bytes.assign(b.begin(), b.end());
        } else if (exec_result.contains("results") && exec_result["results"].is_array() && !exec_result["results"].empty()) {
            auto b = base64_decode(exec_result["results"][0].get<std::string>());
            bytes.assign(b.begin(), b.end());
        } else if (exec_result.contains("raw") && exec_result["raw"].is_array()) {
            for (auto& v : exec_result["raw"]) bytes.push_back(static_cast<uint8_t>(v.get<int>()));
        } else {
            // Better error reporting
            std::cout << "[client] exec_result keys: ";
            for (auto& [key, value] : exec_result.items()) {
                std::cout << key << " ";
            }
            std::cout << std::endl;
            std::cout << "[client] exec_result dump: " << exec_result.dump() << std::endl;
            throw std::runtime_error("Executor result missing expected fields (result, result_b64, results, or raw)");
        }

        const auto ended = std::chrono::high_resolution_clock::now();
        double processing_ms = std::chrono::duration<double, std::milli>(ended - started).count();
        const auto t_send_ms = now_ms_epoch();

        // Pull through any detailed timing the executor provided:
        json timings_meta = json::object();
        if (exec_result.contains("timings")) timings_meta = exec_result["timings"];
        else if (exec_result.contains("profile")) timings_meta = exec_result["profile"];

        timings_meta["tClientRecv"] = t_recv_ms;
        timings_meta["tClientSend"] = t_send_ms;
        timings_meta["processingTimeMs"] = processing_ms;

        json executor_meta = json::object();
        if (exec_result.contains("device")) executor_meta["device"] = exec_result["device"];
        if (exec_result.contains("executor")) executor_meta.update(exec_result["executor"]);
        if (!framework.empty()) executor_meta["framework"] = framework;
#ifdef HAVE_LUA
        executor_meta["strategy"] = lua_ready_ ? "lua-host" : (!framework.empty() ? framework : "binary");
#else
        executor_meta["strategy"] = !framework.empty() ? framework : "binary";
#endif

        send_chunk_done_enhanced(chunk, bytes, processing_ms,
                                 executor_meta["strategy"].get<std::string>(),
                                 executor_meta, timings_meta,
                                 t_recv_ms, t_send_ms);
    } catch (const std::exception& e) {
        std::cout << "[client] Chunk processing error: " << e.what() << std::endl;
        const auto t_send_ms = now_ms_epoch();
        send_chunk_error_enhanced(chunk, e.what(), t_recv_ms, t_send_ms);
    } catch (...) {
        std::cout << "[client] Unknown chunk processing error" << std::endl;
        const auto t_send_ms = now_ms_epoch();
        send_chunk_error_enhanced(chunk, "unknown error", t_recv_ms, t_send_ms);
    }
}

// --------------------- result sending ---------------------

void OrchestratorClient::send_chunk_done_enhanced(const json& chunk,
                                                  const std::vector<uint8_t>& bytes,
                                                  double /*processing_ms*/,
                                                  const std::string& strategy,
                                                  const json& executor_meta,
                                                  const json& timings_meta,
                                                  int64_t t_recv_ms,
                                                  int64_t t_send_ms)
{
    const auto taskId  = chunk.value("taskId", chunk.value("parentId", std::string()));
    const auto chunkId = chunk.value("chunkId", std::string());
    const auto replica = chunk.value("replica", 0);

    const std::string b64 = base64_encode(bytes);
    const std::string sum = sha256(bytes);

    json data = {
        {"taskId", taskId},
        {"chunkId", chunkId},
        {"replica", replica},
        {"status", "ok"},
        {"checksum", sum},
        {"reportedChecksum", sum},
        {"result", b64},
        {"strategy", strategy},
        {"timings", timings_meta},
        {"executor", executor_meta},
        {"clientTimes", {
            {"tClientRecv", t_recv_ms},
            {"tClientSend", t_send_ms}
        }}
    };
    ws_->send_json({{"type","workload:chunk_done_enhanced"},{"data", data}});
}

void OrchestratorClient::send_chunk_error_enhanced(const json& chunk,
                                                   const std::string& err,
                                                   int64_t t_recv_ms,
                                                   int64_t t_send_ms)
{
    const auto taskId  = chunk.value("taskId", chunk.value("parentId", std::string()));
    const auto chunkId = chunk.value("chunkId", std::string());
    const auto replica = chunk.value("replica", 0);
    json data = {
        {"taskId", taskId},
        {"chunkId", chunkId},
        {"replica", replica},
        {"status", "error"},
        {"error",  err},
        {"timings", {
            {"tClientRecv", t_recv_ms},
            {"tClientSend", t_send_ms}
        }}
    };
    ws_->send_json({{"type","workload:chunk_done_enhanced"},{"data", data}});
}