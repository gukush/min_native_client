#include "server_client.hpp"
#include "thread_pool.hpp"
#include "base64.hpp"
#include <iostream>
#include <thread>
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

// Helper function to calculate SHA256 checksum of binary data
std::string calculate_checksum(const std::vector<uint8_t>& data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data.data(), data.size());
    SHA256_Final(hash, &sha256);

    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

ServerBinaryClient::ServerBinaryClient(bool insecure, int concurrency, bool enable_listener)
    : max_concurrency_(concurrency), enable_listener_(enable_listener) {
    std::cout << "[client] Initializing with concurrency: " << concurrency << std::endl;
    if (enable_listener_) {
        std::cout << "[client] Listener mode enabled" << std::endl;
    }
    thread_pool_ = std::make_unique<ThreadPool>(concurrency);

    ws_ = std::make_unique<WebSocketClient>();
    bin_ = std::make_unique<BinaryExecutor>();
    ws_->onConnected = [&]{
        connected_=true;
        std::cout << "[client] Connected with " << concurrency << " worker threads" << std::endl;
        if (enable_listener_) {
            connect_to_listener();
        }
    };
    ws_->onDisconnected = [&]{ connected_=false; };
    ws_->onJson = [&](const nlohmann::json& j){ handle_message(j); };
}

ServerBinaryClient::~ServerBinaryClient() {
    // ThreadPool destructor will wait for pending tasks
}

std::tuple<std::string,std::string> ServerBinaryClient::split_host_port(const std::string& url){
    std::string s;
    std::string default_port = "443";

    if(url.find("wss://") == 0) {
        s = url.substr(6);
        default_port = "443";
    } else if(url.find("ws://") == 0) {
        s = url.substr(5);
        default_port = "80";
    } else {
        // Fallback: assume it's already just host:port
        s = url;
        default_port = "80";
    }

    // Remove path part if present (everything after the first '/')
    auto path_pos = s.find('/');
    if(path_pos != std::string::npos) {
        s = s.substr(0, path_pos);
    }

    auto pos = s.find(':');
    if(pos==std::string::npos) return {s, default_port};
    return {s.substr(0,pos), s.substr(pos+1)};
}

bool ServerBinaryClient::connect(const std::string& url){
    auto [host,port] = split_host_port(url);
    // Determine if we should use SSL based on the URL scheme
    bool use_ssl = (url.find("wss://") == 0);

    // Try connecting with the original host first
    bool connected = ws_->connect(host, port, "/ws-native", use_ssl);

    // If connection failed and we're trying to connect to localhost/127.0.0.1,
    // try alternative approaches
    if (!connected && (host == "127.0.0.1" || host == "localhost")) {
        std::cout << "[client] Connection failed, trying alternative host resolution..." << std::endl;

        // Try with host.docker.internal (if available)
        if (host == "127.0.0.1") {
            std::cout << "[client] Trying host.docker.internal..." << std::endl;
            connected = ws_->connect("host.docker.internal", port, "/ws-native", use_ssl);
        }

        // If still failed, try with the actual host IP
        if (!connected) {
            std::cout << "[client] Trying to resolve host IP directly..." << std::endl;
            // For now, just try the original host again with different resolver settings
            connected = ws_->connect(host, port, "/ws-native", use_ssl);
        }
    }

    if (connected) {
        // Send client registration message after successful connection
        nlohmann::json capabilities = {
            {"device", {
                {"name", "Native Multi-Framework Client"},
                {"type", "native"},
                {"vendor", "MultiFramework"},
                {"memory", 8192},
                {"computeUnits", 32}
            }},
            {"supportedFrameworks", {"vulkan", "opencl", "cuda"}},
            {"clientType", "native"},
            {"hasWebGPU", false}
        };

        ws_->joinComputation(capabilities);
        std::cout << "[client] Sent client registration with capabilities" << std::endl;
    }

    return connected;
}

void ServerBinaryClient::run(){
    while(connected_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if(active_chunks_ > 0) {
            std::cout << "[client] Active chunks: " << active_chunks_.load() << "/" << max_concurrency_ << std::endl;
        }
    }
}

void ServerBinaryClient::handle_message(const json& j){
    auto type = j.value("type", std::string());
    auto data = j.value("data", json::object());

    std::cout << "[client] Received message type: " << type << std::endl;

    if(type=="workload:new"){
        handle_workload(data);
    } else if(type=="chunk:assign"){
        handle_chunk(data);
    } else if(type=="workload:ready"){
        // Server indicates workload is ready, we can start processing
        std::cout << "[client] Workload ready, starting processing..." << std::endl;
    } else if(type=="client:join:ack"){
        std::cout << "[client] Client join acknowledged" << std::endl;
    } else if(type=="welcome"){
        std::cout << "[client] Server welcome: " << data.dump() << std::endl;
    } else if(type=="error"){
        std::cerr << "[client] Server error: " << data.dump() << std::endl;
    } else if(type=="register"){
        // optional ack
    } else {
        std::cout << "[client] Unknown message type: " << type << std::endl;
    }
}

void ServerBinaryClient::handle_workload(const json& w){
    thread_pool_->enqueue([this, w]() {
        process_workload_concurrent(w);
    });
}

void ServerBinaryClient::handle_chunk(const json& c){
    if(active_chunks_ >= max_concurrency_) {
        std::cout << "[client] Queue full, waiting..." << std::endl;
    }

    // Notify listener of chunk arrival
    std::string chunkId = c.value("chunkId", "");
    std::string taskId = c.value("taskId", "");
    notify_listener_chunk_arrival(chunkId, taskId);

    active_chunks_++;
    thread_pool_->enqueue([this, c]() {
        process_chunk_concurrent(c);
        active_chunks_--;
    });
}

void ServerBinaryClient::process_chunk_concurrent(const json& c){
    // Extract chunk information from server format
    std::string taskId = c.value("taskId", "");
    std::string chunkId = c.value("chunkId", "");
    int replica = c.value("replica", 0);
    auto payload = c.value("payload", json::object());
    auto meta = c.value("meta", json::object());

    std::cout << "[client] Processing chunk " << chunkId << " (replica " << replica << ") for task " << taskId << std::endl;

    // Convert server payload format to binary executor format
    json chunk_task = {
        {"id", chunkId},
        {"parentId", taskId},
        {"replica", replica},
        {"meta", meta}
    };

    // Set the program name for binary execution
    if (meta.contains("program")) {
        chunk_task["program"] = meta["program"];
    }

    // Handle payload buffers if present
    if (payload.contains("buffers") && payload["buffers"].is_array()) {
        // Convert buffers to stdin format for binary executor
        std::string stdin_data;
        std::cout << "[client] Processing " << payload["buffers"].size() << " buffers" << std::endl;
        for (size_t i = 0; i < payload["buffers"].size(); i++) {
            const auto& buffer = payload["buffers"][i];
            std::cout << "[client] Buffer " << i << " type: " << buffer.type_name() << std::endl;
            if (buffer.is_string()) {
                // Assume base64 encoded (legacy format)
                std::string base64_str = buffer.get<std::string>();
                std::cout << "[client] Buffer " << i << " is base64 string, length: " << base64_str.length() << std::endl;
                std::vector<uint8_t> decoded = base64_decode(base64_str);
                std::cout << "[client] Decoded buffer " << i << " to " << decoded.size() << " bytes" << std::endl;
                stdin_data.append(reinterpret_cast<const char*>(decoded.data()), decoded.size());
            } else if (buffer.is_array()) {
                // Handle raw byte array (new format)
                std::cout << "[client] Buffer " << i << " is array, size: " << buffer.size() << std::endl;
                for (const auto& byte : buffer) {
                    if (byte.is_number()) {
                        stdin_data.push_back(static_cast<char>(byte.get<uint8_t>()));
                    }
                }
            } else {
                std::cout << "[client] Buffer " << i << " is unknown type: " << buffer.type_name() << std::endl;
                std::cout << "[client] Buffer " << i << " content: " << buffer.dump() << std::endl;
            }
        }
        std::cout << "[client] Total stdin data size: " << stdin_data.size() << " bytes" << std::endl;
        if (!stdin_data.empty()) {
            chunk_task["stdin"] = stdin_data;
        }
    }

    // Add backend information to meta if available
    if (meta.contains("backend")) {
        chunk_task["meta"]["backend"] = meta["backend"];
    }

    auto res = bin_->execute(chunk_task);
    nlohmann::json arr = nlohmann::json::array();
    std::string combined_checksum;

    for(auto& o: res.outputs) {
        arr.push_back(base64_encode(o));
        // Calculate checksum for each output and combine them
        std::string output_checksum = calculate_checksum(o);
        if (combined_checksum.empty()) {
            combined_checksum = output_checksum;
        } else {
            // Combine checksums by hashing them together
            std::vector<uint8_t> combined_data;
            combined_data.insert(combined_data.end(), combined_checksum.begin(), combined_checksum.end());
            combined_data.insert(combined_data.end(), output_checksum.begin(), output_checksum.end());
            combined_checksum = calculate_checksum(combined_data);
        }
    }

    nlohmann::json reply = {
        {"type","workload:chunk_done_enhanced"},
        {"data",{
            {"taskId", taskId},
            {"chunkId", chunkId},
            {"replica", replica},
            {"status", res.ok ? "ok" : "error"},
            {"result", arr},
            {"checksum", combined_checksum},
            {"processingTime", res.ms}
        }}
    };

    ws_->send_json(reply);
    std::cout << "[client] Sent chunk result for " << chunkId << " (status: " << (res.ok ? "ok" : "error") << ")" << std::endl;

    // Notify listener of chunk completion
    std::string status = res.ok ? "completed" : "error";
    notify_listener_chunk_complete(chunkId, status);
}

void ServerBinaryClient::process_workload_concurrent(const json& w){
    std::string workload_id = w.value("id", "");
    std::cout << "[client] Processing workload " << workload_id << std::endl;

    // Handle artifacts if present (this is the binary and input files)
    if (w.contains("artifacts") && w["artifacts"].is_array()) {
        std::cout << "[client] Processing " << w["artifacts"].size() << " artifacts for workload " << workload_id << std::endl;
        bin_->handle_workload_artifacts(w);
    }

    // For native-block-matmul-flex strategy, we don't execute the workload directly
    // Instead, we wait for individual chunks to be assigned
    std::cout << "[client] Workload " << workload_id << " artifacts processed, waiting for chunks..." << std::endl;

    // Send acknowledgment that we're ready to process chunks
    nlohmann::json reply = {
        {"type","workload:ready"},
        {"data",{
            {"id", workload_id},
            {"status", "ready"}
        }}
    };

    ws_->send_json(reply);
    std::cout << "[client] Sent workload ready acknowledgment for " << workload_id << std::endl;
}

// Listener functionality implementation
void ServerBinaryClient::connect_to_listener() {
    if (!enable_listener_) return;

    try {
        listener_ws_ = std::make_unique<WebSocketClient>();

        listener_ws_->onConnected = [&]{
            std::cout << "[listener] Connected to listener at ws://127.0.0.1:8765" << std::endl;
        };

        listener_ws_->onDisconnected = [&]{
            std::cout << "[listener] Disconnected from listener" << std::endl;
            listener_ws_.reset();
        };

        listener_ws_->onMessage = [&](const std::string& message){
            try {
                auto response = json::parse(message);
                std::cout << "[listener] Response: " << response.dump() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[listener] Failed to parse response: " << e.what() << std::endl;
            }
        };

        if (!listener_ws_->connect("127.0.0.1", "8765", "/", false)) {
            std::cout << "[listener] Failed to connect to listener" << std::endl;
            listener_ws_.reset();
        }
    } catch (const std::exception& e) {
        std::cout << "[listener] Exception connecting to listener: " << e.what() << std::endl;
        listener_ws_.reset();
    }
}

void ServerBinaryClient::notify_listener_chunk_arrival(const std::string& chunk_id, const std::string& task_id) {
    if (!enable_listener_ || !listener_ws_) return;

    try {
        json message = {
            {"type", "chunk_status"},
            {"chunk_id", chunk_id},
            {"task_id", task_id},
            {"status", 0}  // 0 = chunk arrival/start
        };

        listener_ws_->send_json(message);
        std::cout << "[listener] Notified chunk arrival: " << chunk_id << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[listener] Failed to notify chunk arrival: " << e.what() << std::endl;
    }
}

void ServerBinaryClient::notify_listener_chunk_complete(const std::string& chunk_id, const std::string& status) {
    if (!enable_listener_ || !listener_ws_) return;

    try {
        int status_code = (status == "completed") ? 1 : -1;  // 1 = success, -1 = error

        json message = {
            {"type", "chunk_status"},
            {"chunk_id", chunk_id},
            {"status", status_code}
        };

        listener_ws_->send_json(message);
        std::cout << "[listener] Notified chunk completion: " << chunk_id << " status: " << status << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[listener] Failed to notify chunk completion: " << e.what() << std::endl;
    }
}
