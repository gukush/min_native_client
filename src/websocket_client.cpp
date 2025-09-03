#include "websocket_client.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/url.hpp>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

WebSocketClient::WebSocketClient() {
    // Configure SSL context to accept self-signed certificates
    ctx.set_verify_mode(ssl::verify_none);
    ctx.set_options(ssl::context::default_workarounds |
                   ssl::context::no_sslv2 |
                   ssl::context::no_sslv3 |
                   ssl::context::single_dh_use);
}

WebSocketClient::~WebSocketClient() {
    disconnect();
}

bool WebSocketClient::connect(const std::string& host, const std::string& port, const std::string& target) {
    try {
        // Resolve hostname
        auto const results = resolver.resolve(host, port);

        // Get the underlying socket
        auto& socket = beast::get_lowest_layer(ws);

        // Make the connection on the IP address we get from a lookup
        socket.connect(results);

        // Update the host string for SNI
        std::string hostWithPort = host + ':' + port;

        // Set SNI Hostname (many hosts need this to handshake successfully)
        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), host.c_str())) {
            beast::error_code ec{static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()};
            throw beast::system_error{ec};
        }

        // Perform the SSL handshake
        ws.next_layer().handshake(ssl::stream_base::client);

        // Set a decorator to change the User-Agent of the handshake
        ws.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req) {
                req.set(beast::http::field::user_agent, "MultiFramework-Native-Client/1.0");
            }));

        // Connect to the native WebSocket endpoint
        std::string native_target = "/ws-native";
        ws.handshake(hostWithPort, native_target);

        // Start the event loop in a separate thread
        shouldStop = false;
        ioThread = std::thread(&WebSocketClient::runEventLoop, this);

        std::cout << " Connected to native WebSocket endpoint: " << native_target << std::endl;

        if (onConnected) {
            onConnected();
        }

        return true;

    } catch (std::exception const& e) {
        std::cerr << "WebSocket connection error: " << e.what() << std::endl;
        return false;
    }
}

void WebSocketClient::disconnect() {
    shouldStop = true;

    if (ws.is_open()) {
        try {
            ws.close(websocket::close_code::normal);
        } catch (...) {
            // Ignore errors during close
        }
    }

    if (ioThread.joinable()) {
        ioThread.join();
    }

    if (onDisconnected) {
        onDisconnected();
    }
}

void WebSocketClient::send_json(const json& j) {
    try {
        auto s = j.dump();
        ws.write(net::buffer(s));
    } catch (std::exception const& e) {
        std::cerr << "WS send error: " << e.what() << std::endl;
    }
}

void WebSocketClient::sendEvent(const std::string& eventType, const json& data) {
    if (!ws.is_open()) {
        std::cerr << "WebSocket not connected, cannot send event: " << eventType << std::endl;
        return;
    }

    try {
        json message = {
            {"type", eventType},
            {"data", data}
        };

        std::string messageStr = message.dump();
        std::cout << "[WS-SEND] " << eventType << ": " << messageStr << std::endl;

        ws.write(net::buffer(messageStr));
    } catch (std::exception const& e) {
        std::cerr << "WebSocket send error for event " << eventType << ": " << e.what() << std::endl;
    }
}

void WebSocketClient::send(const std::string& message) {
    if (!ws.is_open()) {
        std::cerr << "WebSocket not connected, cannot send message" << std::endl;
        return;
    }

    try {
        ws.write(net::buffer(message));
    } catch (std::exception const& e) {
        std::cerr << "WebSocket send error: " << e.what() << std::endl;
    }
}

void WebSocketClient::joinComputation(const json& capabilities) {
    json joinData = {
        {"gpuInfo", capabilities.value("device", json::object())},
        {"hasWebGPU", false},
        {"supportedFrameworks", capabilities.value("supportedFrameworks", json::array({"vulkan"}))},
        {"clientType", "native"}
    };

    sendEvent("client:join", joinData);
    std::cout << " Sent client:join with capabilities" << std::endl;
}

void WebSocketClient::requestTask() {
    sendEvent("task:request", json::object());
    std::cout << " Requested matrix task" << std::endl;
}

void WebSocketClient::submitTaskResult(const std::string& assignmentId, const std::string& taskId,
                                      const json& result, double processingTime, const std::string& checksum) {
    json resultData = {
        {"assignmentId", assignmentId},
        {"taskId", taskId},
        {"result", result},
        {"processingTime", processingTime},
        {"reportedChecksum", checksum}
    };

    sendEvent("task:complete", resultData);
    std::cout << " Submitted task result for " << taskId << std::endl;
}

void WebSocketClient::submitWorkloadResult(const std::string& workloadId, const std::string& result,
                                          double processingTime, const std::string& checksum) {
    json resultData = {
        {"id", workloadId},
        {"result", result},
        {"processingTime", processingTime},
        {"reportedChecksum", checksum}
    };

    sendEvent("workload:done", resultData);
    std::cout << " Submitted workload result for " << workloadId << std::endl;
}

void WebSocketClient::submitChunkResult(const std::string& parentId, const std::string& chunkId,
                                       const json& results, double processingTime,
                                       const std::string& strategy, const json& metadata,
                                       const std::string& checksum) {
    json resultData = {
        {"parentId", parentId},
        {"chunkId", chunkId},
        {"results", results},
        {"result", results.is_array() && !results.empty() ? results[0] : results},
        {"processingTime", processingTime},
        {"strategy", strategy},
        {"metadata", metadata},
        {"reportedChecksum", checksum}
    };

    sendEvent("workload:chunk_done_enhanced", resultData);
    std::cout << " Submitted enhanced chunk result for " << chunkId << std::endl;
}

void WebSocketClient::reportError(const std::string& workloadId, const std::string& message) {
    json errorData = {
        {"id", workloadId},
        {"message", message}
    };

    sendEvent("workload:error", errorData);
    std::cout << " Reported error for " << workloadId << ": " << message << std::endl;
}

void WebSocketClient::reportChunkError(const std::string& parentId, const std::string& chunkId,
                                      const std::string& message) {
    json errorData = {
        {"parentId", parentId},
        {"chunkId", chunkId},
        {"message", message}
    };

    sendEvent("workload:chunk_error", errorData);
    std::cout << " Reported chunk error for " << chunkId << ": " << message << std::endl;
}

void WebSocketClient::runEventLoop() {
    beast::flat_buffer buffer;

    while (!shouldStop && ws.is_open()) {
        try {
            // Read a message
            ws.read(buffer);

            // Convert to string
            std::string messageStr = beast::buffers_to_string(buffer.data());
            buffer.clear();

            // Parse JSON message
            try {
                json message = json::parse(messageStr);
                std::string eventType = message.value("type", "");
                json eventData = message.value("data", json::object());

                std::cout << "[WS-RECV] " << eventType << std::endl;

                // Handle different event types
                if (eventType == "register") {
                    if (onRegister) {
                        onRegister(eventData);
                    }
                } else if (eventType == "task:assign") {
                    if (onTaskAssigned) {
                        onTaskAssigned(eventData);
                    }
                } else if (eventType == "workload:new") {
                    if (onWorkloadAssigned) {
                        onWorkloadAssigned(eventData);
                    }
                } else if (eventType == "workload:chunk_assign") {
                    if (onChunkAssigned) {
                        onChunkAssigned(eventData);
                    }
                } else if (eventType == "task:verified") {
                    if (onTaskVerified) {
                        onTaskVerified(eventData);
                    }
                } else if (eventType == "task:submitted") {
                    if (onTaskSubmitted) {
                        onTaskSubmitted(eventData);
                    }
                } else if (eventType == "workload:complete") {
                    if (onWorkloadComplete) {
                        onWorkloadComplete(eventData);
                    }
                } else if (eventType == "admin:k_update") {
                    std::cout << " K parameter updated to: " << eventData << std::endl;
                } else if (eventType == "clients:update") {
                    // Optional: handle client list updates
                } else {
                    std::cout << " Unknown event type: " << eventType << std::endl;
                }

                // Call the generic JSON handler
                if (onJson) {
                    onJson(message);
                }

            } catch (json::parse_error const& e) {
                std::cerr << "JSON parse error: " << e.what() << std::endl;
                std::cerr << "Raw message: " << messageStr << std::endl;
            }

            // Call generic message handler if set
            if (onMessage) {
                onMessage(messageStr);
            }

        } catch (beast::system_error const& se) {
            if (se.code() != websocket::error::closed) {
                std::cerr << "WebSocket read error: " << se.code().message() << std::endl;
            }
            break;
        } catch (std::exception const& e) {
            std::cerr << "WebSocket event loop error: " << e.what() << std::endl;
            break;
        }
    }

    std::cout << " WebSocket event loop ended" << std::endl;
}
