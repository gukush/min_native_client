
#pragma once
#include "websocket_client.hpp"
#include "executors/binary_executor.hpp"
#include "base64.hpp"
#include <memory>

class ServerBinaryClient {
public:
    using json = nlohmann::json;
    ServerBinaryClient(bool insecure=true);

    bool connect(const std::string& url);
    void run(); // blocks

private:
    void handle_message(const json& j);
    void handle_workload(const json& j);
    void handle_chunk(const json& j);

    static std::tuple<std::string,std::string> split_host_port(const std::string& url);

    std::unique_ptr<WebSocketClient> ws_;
    std::unique_ptr<BinaryExecutor> bin_;
    bool connected_=false;
};
