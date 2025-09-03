#pragma once
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/ssl.hpp>
#include <nlohmann/json.hpp>
#include <thread>
#include <functional>

class WebSocketClient {
public:
    using json = nlohmann::json;
    WebSocketClient();
    ~WebSocketClient();

    bool connect(const std::string& host, const std::string& port, const std::string& target="/ws-native");
    void disconnect();
    void send_json(const json& j);
    void send(const std::string& message);
    void sendEvent(const std::string& eventType, const json& data);

    // Additional methods for the working implementation
    void joinComputation(const json& capabilities);
    void requestTask();
    void submitTaskResult(const std::string& assignmentId, const std::string& taskId,
                         const json& result, double processingTime, const std::string& checksum);
    void submitWorkloadResult(const std::string& workloadId, const std::string& result,
                             double processingTime, const std::string& checksum);
    void submitChunkResult(const std::string& parentId, const std::string& chunkId,
                          const json& results, double processingTime,
                          const std::string& strategy, const json& metadata,
                          const std::string& checksum);
    void reportError(const std::string& workloadId, const std::string& message);
    void reportChunkError(const std::string& parentId, const std::string& chunkId,
                         const std::string& message);

    // Callback functions
    std::function<void()> onConnected;
    std::function<void()> onDisconnected;
    std::function<void(const json&)> onJson;
    std::function<void(const std::string&)> onMessage;
    std::function<void(const json&)> onRegister;
    std::function<void(const json&)> onTaskAssigned;
    std::function<void(const json&)> onWorkloadAssigned;
    std::function<void(const json&)> onChunkAssigned;
    std::function<void(const json&)> onTaskVerified;
    std::function<void(const json&)> onTaskSubmitted;
    std::function<void(const json&)> onWorkloadComplete;

private:
    void runEventLoop();

    boost::asio::io_context ioc;
    boost::asio::ip::tcp::resolver resolver{ioc};
    boost::asio::ssl::context ctx{boost::asio::ssl::context::tls_client};
    boost::beast::websocket::stream<boost::asio::ssl::stream<boost::beast::tcp_stream>> ws{ioc, ctx};
    std::thread ioThread;
    bool shouldStop = false;
};
