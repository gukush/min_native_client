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
    ssl_ctx.set_verify_mode(ssl::verify_none);
    ssl_ctx.set_options(ssl::context::default_workarounds |
                       ssl::context::no_sslv2 |
                       ssl::context::no_sslv3 |
                       ssl::context::single_dh_use);
}

WebSocketClient::~WebSocketClient() {
    disconnect();
}

bool WebSocketClient::connect(const std::string& host, const std::string& port, const std::string& target, bool use_ssl) {
    try {
        // Resolve hostname with better error handling
        boost::system::error_code ec;
        auto results = resolver.resolve(host, port, ec);

        if (ec) {
            std::cerr << "DNS resolution failed for " << host << ":" << port << " - " << ec.message() << std::endl;

            // Try fallback for localhost addresses
            if (host == "127.0.0.1" || host == "localhost") {
                std::cerr << "Trying fallback resolution for localhost..." << std::endl;
                // Try resolving with different approach - use the modern API
                results = resolver.resolve(tcp::v4(), host, port, ec);
                if (ec) {
                    std::cerr << "Fallback DNS resolution also failed: " << ec.message() << std::endl;
                    return false;
                }
            } else {
                return false;
            }
        }

        use_ssl_connection = use_ssl;

        if (use_ssl) {
            // SSL WebSocket connection
            ssl_ws = std::make_unique<boost::beast::websocket::stream<boost::asio::ssl::stream<boost::beast::tcp_stream>>>(ioc, ssl_ctx);

            // Get the underlying socket
            auto& socket = beast::get_lowest_layer(*ssl_ws);

            // Make the connection on the IP address we get from a lookup
            socket.connect(results);

            // Update the host string for SNI
            std::string hostWithPort = host + ':' + port;

            // Set SNI Hostname (many hosts need this to handshake successfully)
            if (!SSL_set_tlsext_host_name(ssl_ws->next_layer().native_handle(), host.c_str())) {
                beast::error_code ec{static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()};
                throw beast::system_error{ec};
            }

            // Perform the SSL handshake
            ssl_ws->next_layer().handshake(ssl::stream_base::client);

            // Set a decorator to change the User-Agent of the handshake
            ssl_ws->set_option(websocket::stream_base::decorator(
                [](websocket::request_type& req) {
                    req.set(beast::http::field::user_agent, "MultiFramework-Native-Client/1.0");
                }));

            // Set message size limit to 18x the default (16MB * 18 = 288MB)
            ssl_ws->read_message_max(288 * 1024 * 1024);

            // Connect to the native WebSocket endpoint
            ssl_ws->handshake(hostWithPort, target);

            std::cout << " Connected to SSL WebSocket endpoint: " << target << std::endl;
        } else {
            // Plain WebSocket connection
            plain_ws = std::make_unique<boost::beast::websocket::stream<boost::beast::tcp_stream>>(ioc);

            // Get the underlying socket
            auto& socket = plain_ws->next_layer();

            // Make the connection on the IP address we get from a lookup
            socket.connect(results);

            // Set a decorator to change the User-Agent of the handshake
            plain_ws->set_option(websocket::stream_base::decorator(
                [](websocket::request_type& req) {
                    req.set(beast::http::field::user_agent, "MultiFramework-Native-Client/1.0");
                }));

            // Set message size limit to 18x the default (16MB * 18 = 288MB)
            plain_ws->read_message_max(288 * 1024 * 1024);

            // Connect to the native WebSocket endpoint
            plain_ws->handshake(host, target);

            std::cout << " Connected to plain WebSocket endpoint: " << target << std::endl;
        }

        // Start the event loop in a separate thread
        shouldStop = false;
        ioThread = std::thread(&WebSocketClient::runEventLoop, this);

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

    if (use_ssl_connection && ssl_ws && ssl_ws->is_open()) {
        try {
            ssl_ws->close(websocket::close_code::normal);
        } catch (...) {
            // Ignore errors during close
        }
    } else if (!use_ssl_connection && plain_ws && plain_ws->is_open()) {
        try {
            plain_ws->close(websocket::close_code::normal);
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

bool WebSocketClient::isConnected() const {
    if (use_ssl_connection) {
        return ssl_ws && ssl_ws->is_open();
    } else {
        return plain_ws && plain_ws->is_open();
    }
}

void WebSocketClient::send_json(const json& j) {
    try {
        auto s = j.dump();
        if (use_ssl_connection && ssl_ws) {
            ssl_ws->write(net::buffer(s));
        } else if (!use_ssl_connection && plain_ws) {
            plain_ws->write(net::buffer(s));
        }
    } catch (std::exception const& e) {
        std::cerr << "WS send error: " << e.what() << std::endl;
    }
}

void WebSocketClient::sendEvent(const std::string& eventType, const json& data) {
    if (!isConnected()) {
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

        if (use_ssl_connection && ssl_ws) {
            ssl_ws->write(net::buffer(messageStr));
        } else if (!use_ssl_connection && plain_ws) {
            plain_ws->write(net::buffer(messageStr));
        }
    } catch (std::exception const& e) {
        std::cerr << "WebSocket send error for event " << eventType << ": " << e.what() << std::endl;
    }
}

void WebSocketClient::send(const std::string& message) {
    if (!isConnected()) {
        std::cerr << "WebSocket not connected, cannot send message" << std::endl;
        return;
    }

    try {
        if (use_ssl_connection && ssl_ws) {
            ssl_ws->write(net::buffer(message));
        } else if (!use_ssl_connection && plain_ws) {
            plain_ws->write(net::buffer(message));
        }
    } catch (std::exception const& e) {
        std::cerr << "WebSocket send error: " << e.what() << std::endl;
    }
}

void WebSocketClient::joinComputation(const json& capabilities) {
    json joinData = {
        {"frameworks", capabilities.value("supportedFrameworks", json::array({"vulkan", "opencl", "cuda"}))},
        {"strategies", json::array({"native-block-matmul-flex"})},
        {"clientType", "native"},
        {"device", capabilities.value("device", json::object())},
        {"hasWebGPU", false}
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

    while (!shouldStop && isConnected()) {
        try {
            // Read a message
            if (use_ssl_connection && ssl_ws) {
                ssl_ws->read(buffer);
            } else if (!use_ssl_connection && plain_ws) {
                plain_ws->read(buffer);
            } else {
                break;
            }

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
                } else if (eventType == "client:join:ack") {
                    std::cout << " Client join acknowledged: " << eventData.dump() << std::endl;
                } else if (eventType == "task:assign") {
                    if (onTaskAssigned) {
                        onTaskAssigned(eventData);
                    }
                } else if (eventType == "workload:new") {
                    if (onWorkloadAssigned) {
                        onWorkloadAssigned(eventData);
                    }
                } else if (eventType == "chunk:assign") {
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
                } else if (eventType == "message") {
                    // Simple message/acknowledgment from server
                    std::cout << " Server message: " << eventData.dump() << std::endl;
                } else if (eventType == "pong") {
                    // WebSocket pong response
                    std::cout << " Received pong from server" << std::endl;
                } else if (eventType == "ping") {
                    // WebSocket ping request
                    std::cout << " Received ping from server" << std::endl;
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
