// local_ws_server.cpp
#include "local_ws_server.hpp"
#include "thread_pool.hpp"
#include "executors/cuda_executor.hpp"
#include "executors/opencl_executor.hpp"
#include "executors/vulkan_executor.hpp"
#include "base64.hpp"
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/error.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

using tcp = boost::asio::ip::tcp;
namespace websocket = boost::beast::websocket;
namespace beast = boost::beast;
namespace ssl = boost::asio::ssl;

LocalWSServer::LocalWSServer(int concurrency) : max_concurrency(concurrency) {
    std::cout << "[local-ws] Initializing with concurrency: " << concurrency << std::endl;
    thread_pool = std::make_unique<ThreadPool>(concurrency);
    
    std::cout << "[local-ws] Initializing GPU executors..." << std::endl;
    #ifdef HAVE_CUDA
    cuda_executor_ = std::make_unique<CudaExecutor>(0);
    if (!cuda_executor_->initialize({})) {
        std::cerr << "[local-ws] FATAL: CUDA executor initialization failed!" << std::endl;
        cuda_executor_.reset(); // Set to null if failed
    } else {
        std::cout << "[local-ws] ✓ CUDA executor initialized." << std::endl;
    }
    #endif
    #ifdef HAVE_OPENCL
    opencl_executor_ = std::make_unique<OpenCLExecutor>();
    if (!opencl_executor_->initialize({})) {
        std::cerr << "[local-ws] FATAL: OpenCL executor initialization failed!" << std::endl;
        opencl_executor_.reset(); // Set to null if failed
    } else {
        std::cout << "[local-ws] ✓ OpenCL executor initialized." << std::endl;
    }
    #endif
    #ifdef HAVE_VULKAN
    vulkan_executor_ = std::make_unique<VulkanExecutor>();
    if (!vulkan_executor_->initialize({})) {
        std::cerr << "[local-ws] FATAL: Vulkan executor initialization failed!" << std::endl;
        vulkan_executor_.reset(); // Set to null if failed
    } else {
        std::cout << "[local-ws] ✓ Vulkan executor initialized." << std::endl;
    }
    #endif
}
LocalWSServer::~LocalWSServer(){ stop(); }

json LocalWSServer::process_request(const json& req) {
    nlohmann::json resp;
    
    if(req.is_discarded() || req.value("action","")!="compile_and_run"){
        resp = nlohmann::json::object();
        resp["ok"] = false;
        resp["error"] = "invalid request";
        return resp;
    }
    
    const std::string framework = req.value("framework","cuda");
    
    auto make_ok = [](const auto& result){
        nlohmann::json outs = nlohmann::json::array();
        for(const auto& o : result.outputs) outs.push_back(base64_encode(o));
        return nlohmann::json{{"ok",true},{"outputs",outs},{"processingTimeMs",result.ms}};
    };
    
    if(framework=="cuda"){
    #ifdef HAVE_CUDA
        if(!cuda_executor_){
            resp = nlohmann::json::object();
            resp["ok"] = false;
            resp["error"] = "CUDA executor is not available or failed to initialize";
        } else {
            auto result = cuda_executor_->run_task(req);
            if(result.ok){
                std::cerr << "[local-ws] CUDA task completed successfully" << std::endl;
                resp = make_ok(result);
            } else {
                std::cerr << "[local-ws] CUDA task failed: " << result.error << std::endl;
                resp = nlohmann::json::object();
                resp["ok"] = false;
                resp["error"] = result.error;
            }
        }
    #else
        resp = nlohmann::json::object();
        resp["ok"] = false;
        resp["error"] = "cuda disabled at build time";
    #endif
    } else if(framework=="opencl"){
    #ifdef HAVE_OPENCL
        if(!opencl_executor_){
            resp = nlohmann::json::object();
            resp["ok"] = false;
            resp["error"] = "OpenCL executor is not available or failed to initialize";
        } else {
            auto result = opencl_executor_->run_task(req);
            if(result.ok){
                std::cerr << "[local-ws] OpenCL task completed successfully" << std::endl;
                resp = make_ok(result);
            } else {
                std::cerr << "[local-ws] OpenCL task failed: " << result.error << std::endl;
                resp = nlohmann::json::object();
                resp["ok"] = false;
                resp["error"] = result.error;
            }
        }
    #else
        resp = nlohmann::json::object();
        resp["ok"] = false;
        resp["error"] = "opencl disabled at build time";
    #endif
    } else if(framework=="vulkan"){
    #ifdef HAVE_VULKAN
        if(!vulkan_executor_){
            resp = nlohmann::json::object();
            resp["ok"] = false;
            resp["error"] = "Vulkan executor is not available or failed to initialize";
        } else {
            auto result = vulkan_executor_->run_task(req);
            if(result.ok){
                std::cerr << "[local-ws] Vulkan task completed successfully" << std::endl;
                resp = make_ok(result);
            } else {
                std::cerr << "[local-ws] Vulkan task failed: " << result.error << std::endl;
                resp = nlohmann::json::object();
                resp["ok"] = false;
                resp["error"] = result.error;
            }
        }
    #else
        resp = nlohmann::json::object();
        resp["ok"] = false;
        resp["error"] = "vulkan disabled at build time";
    #endif
    } else {
        resp = nlohmann::json::object();
        resp["ok"] = false;
        resp["error"] = "framework must be cuda|opencl|vulkan";
    }
    
    return resp;
}

bool LocalWSServer::start(const std::string& address, unsigned short port,
                         const std::string& target, bool use_ssl){
    if(running) return false;
    running=true;
    th = std::thread(&LocalWSServer::run, this, address, port, target, use_ssl);
    return true;
}

void LocalWSServer::stop(){
    running=false;
    if(th.joinable()) th.join();
}

void LocalWSServer::run(const std::string& address, unsigned short port,
                       const std::string& target, bool use_ssl){
    if(use_ssl) {
        run_ssl(address, port, target);
    } else {
        run_plain(address, port, target);
    }
}

// Helper: treat these errors as normal client disconnects (not server fatal)
static bool is_normal_disconnect(const boost::system::error_code& ec){
    if(!ec) return false;
    return ec == boost::asio::error::eof
        || ec == boost::asio::error::connection_reset
        || ec == boost::asio::error::connection_aborted
        || ec == boost::asio::ssl::error::stream_truncated;
}

void LocalWSServer::run_ssl(const std::string& address, unsigned short port, const std::string& target){
    try{
        boost::asio::io_context ioc;

        // SSL context setup
        ssl::context ctx{ssl::context::tlsv12_server};
        ctx.use_certificate_chain_file("server.crt");
        ctx.use_private_key_file("server.key", ssl::context::pem);

        tcp::endpoint endpoint{boost::asio::ip::make_address(address), port};
        tcp::acceptor acceptor{ioc};
        boost::system::error_code ec;
        acceptor.open(endpoint.protocol(), ec);
        if(ec){
            std::cerr << "[local-ws] acceptor.open error: " << ec.message() << std::endl;
            return;
        }
        acceptor.set_option(boost::asio::socket_base::reuse_address(true), ec);
        if(ec){
            std::cerr << "[local-ws] setting reuse_address failed: " << ec.message() << std::endl;
        }
        acceptor.bind(endpoint, ec);
        if(ec){
            std::cerr << "[local-ws] bind error: " << ec.message() << std::endl;
            return;
        }
        acceptor.listen(boost::asio::socket_base::max_listen_connections, ec);
        if(ec){
            std::cerr << "[local-ws] listen error: " << ec.message() << std::endl;
            return;
        }

        std::cout << "[local-ws] SSL WebSocket server listening on wss://" << address << ":" << port << target << std::endl;
        std::cout << "[local-ws] Concurrent processing enabled: " << max_concurrency << " threads" << std::endl;

        while(running){
            tcp::socket socket{ioc};
            boost::system::error_code accept_ec;
            acceptor.accept(socket, accept_ec);
            if(accept_ec){
                if(running) std::cerr << "[local-ws] accept error: " << accept_ec.message() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            // Process each connection in thread pool
            thread_pool->enqueue([this, socket = std::move(socket), &ctx, target]() mutable {
                try{
                    active_requests++;
                    std::cout << "[local-ws] Active requests: " << active_requests.load() << "/" << max_concurrency << std::endl;
                    
                    ssl::stream<tcp::socket> ssl_socket{std::move(socket), ctx};
                    
                    boost::system::error_code hs_ec;
                    ssl_socket.handshake(ssl::stream_base::server, hs_ec);
                    if(hs_ec){
                        if(is_normal_disconnect(hs_ec)){
                            if(running) std::cerr << "[local-ws] SSL handshake aborted by client: " << hs_ec.message() << std::endl;
                        } else {
                            std::cerr << "[local-ws] SSL handshake error: " << hs_ec.message() << std::endl;
                        }
                        boost::system::error_code close_ec;
                        ssl_socket.lowest_layer().close(close_ec);
                        active_requests--;
                        return;
                    }

                    websocket::stream<ssl::stream<tcp::socket>&> ws{ssl_socket};

                    boost::system::error_code ws_accept_ec;
                    ws.accept(ws_accept_ec);
                    if(ws_accept_ec){
                        if(is_normal_disconnect(ws_accept_ec)){
                            if(running) std::cerr << "[local-ws] WebSocket accept aborted by client: " << ws_accept_ec.message() << std::endl;
                        } else {
                            std::cerr << "[local-ws] WebSocket accept error: " << ws_accept_ec.message() << std::endl;
                        }
                        boost::system::error_code close_ec;
                        ssl_socket.lowest_layer().close(close_ec);
                        active_requests--;
                        return;
                    }

                    beast::flat_buffer buffer;
                    boost::system::error_code read_ec;
                    ws.read(buffer, read_ec);
                    if(read_ec){
                        if(is_normal_disconnect(read_ec)){
                            if(running) std::cerr << "[local-ws] client disconnected (read): " << read_ec.message() << std::endl;
                        } else {
                            std::cerr << "[local-ws] WebSocket read error: " << read_ec.message() << std::endl;
                        }
                        boost::system::error_code close_ec;
                        ws.next_layer().lowest_layer().close(close_ec);
                        active_requests--;
                        return;
                    }

                    std::string s = beast::buffers_to_string(buffer.data());
                    buffer.clear();

                    nlohmann::json req = nlohmann::json::parse(s, nullptr, false);
                    
                    // Process request
                    auto resp = process_request(req);
                    
                    // Send response
                    auto out = resp.dump();
                    ws.text(true);
                    boost::system::error_code write_ec;
                    ws.write(boost::asio::buffer(out), write_ec);
                    if(write_ec){
                        if(is_normal_disconnect(write_ec)){
                            if(running) std::cerr << "[local-ws] client disconnected (write): " << write_ec.message() << std::endl;
                        } else {
                            std::cerr << "[local-ws] WebSocket write error: " << write_ec.message() << std::endl;
                        }
                    }

                    boost::system::error_code shutdown_ec;
                    ws.close(websocket::close_code::normal, shutdown_ec);
                    if(shutdown_ec && !is_normal_disconnect(shutdown_ec)){
                        std::cerr << "[local-ws] WebSocket close error: " << shutdown_ec.message() << std::endl;
                    }

                    boost::system::error_code ssl_shutdown_ec;
                    if (ssl_socket.lowest_layer().is_open()) {
                        ssl_socket.shutdown(ssl_shutdown_ec);
                        if (ssl_shutdown_ec && !is_normal_disconnect(ssl_shutdown_ec) && ssl_shutdown_ec.value() != EBADF) {
                            std::cerr << "[local-ws] SSL shutdown error: " << ssl_shutdown_ec.message() << std::endl;
                        }
                    }
                    boost::system::error_code close_ec;
                    ssl_socket.lowest_layer().close(close_ec);
                    
                    active_requests--;
                    
                }catch(std::exception const& e){
                    if(running) std::cerr << "[local-ws] connection handler exception: " << e.what() << std::endl;
                    active_requests--;
                }
            });
        }

    }catch(std::exception const& e){
        if(running) std::cerr << "[local-ws] SSL server fatal error: " << e.what() << std::endl;
    }
}

void LocalWSServer::run_plain(const std::string& address, unsigned short port, const std::string& target){
    try{
        boost::asio::io_context ioc;
        tcp::endpoint endpoint{boost::asio::ip::make_address(address), port};
        tcp::acceptor acceptor{ioc};
        boost::system::error_code ec;
        acceptor.open(endpoint.protocol(), ec);
        if(ec){
            std::cerr << "[local-ws] acceptor.open error: " << ec.message() << std::endl;
            return;
        }
        acceptor.set_option(boost::asio::socket_base::reuse_address(true), ec);
        if(ec){
            std::cerr << "[local-ws] setting reuse_address failed: " << ec.message() << std::endl;
        }
        acceptor.bind(endpoint, ec);
        if(ec){
            std::cerr << "[local-ws] bind error: " << ec.message() << std::endl;
            return;
        }
        acceptor.listen(boost::asio::socket_base::max_listen_connections, ec);
        if(ec){
            std::cerr << "[local-ws] listen error: " << ec.message() << std::endl;
            return;
        }

        std::cout << "[local-ws] Plain WebSocket server listening on ws://" << address << ":" << port << target << std::endl;
        std::cout << "[local-ws] Concurrent processing enabled: " << max_concurrency << " threads" << std::endl;

        while(running){
            tcp::socket socket{ioc};
            boost::system::error_code accept_ec;
            acceptor.accept(socket, accept_ec);
            if(accept_ec){
                if(running) std::cerr << "[local-ws] accept error: " << accept_ec.message() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            // Process each connection in thread pool
            thread_pool->enqueue([this, socket = std::move(socket)]() mutable {
                try{
                    active_requests++;
                    std::cout << "[local-ws] Active requests: " << active_requests.load() << "/" << max_concurrency << std::endl;
                    
                    websocket::stream<tcp::socket> ws{std::move(socket)};
                    boost::system::error_code ws_accept_ec;
                    ws.accept(ws_accept_ec);
                    if(ws_accept_ec){
                        if(is_normal_disconnect(ws_accept_ec)){
                            if(running) std::cerr << "[local-ws] WebSocket accept aborted by client: " << ws_accept_ec.message() << std::endl;
                        } else {
                            std::cerr << "[local-ws] WebSocket accept error: " << ws_accept_ec.message() << std::endl;
                        }
                        boost::system::error_code close_ec;
                        ws.next_layer().close(close_ec);
                        active_requests--;
                        return;
                    }

                    beast::flat_buffer buffer;
                    boost::system::error_code read_ec;
                    ws.read(buffer, read_ec);
                    if(read_ec){
                        if(is_normal_disconnect(read_ec)){
                            if(running) std::cerr << "[local-ws] client disconnected (read): " << read_ec.message() << std::endl;
                        } else {
                            std::cerr << "[local-ws] WebSocket read error: " << read_ec.message() << std::endl;
                        }
                        boost::system::error_code close_ec;
                        ws.next_layer().close(close_ec);
                        active_requests--;
                        return;
                    }

                    std::string s = beast::buffers_to_string(buffer.data());
                    buffer.clear();

                    nlohmann::json req = nlohmann::json::parse(s, nullptr, false);
                    
                    // Process request
                    auto resp = process_request(req);
                    
                    // Send response
                    auto out = resp.dump();
                    ws.text(true);
                    boost::system::error_code write_ec;
                    ws.write(boost::asio::buffer(out), write_ec);
                    if(write_ec){
                        if(is_normal_disconnect(write_ec)){
                            if(running) std::cerr << "[local-ws] client disconnected (write): " << write_ec.message() << std::endl;
                        } else {
                            std::cerr << "[local-ws] WebSocket write error: " << write_ec.message() << std::endl;
                        }
                    }

                    boost::system::error_code close_ec;
                    ws.close(websocket::close_code::normal, close_ec);
                    if(close_ec && !is_normal_disconnect(close_ec)){
                        std::cerr << "[local-ws] WebSocket close error: " << close_ec.message() << std::endl;
                    }
                    boost::system::error_code underlying_close_ec;
                    ws.next_layer().close(underlying_close_ec);
                    
                    active_requests--;

                }catch(std::exception const& e){
                    if(running) std::cerr << "[local-ws] connection handler exception (plain): " << e.what() << std::endl;
                    active_requests--;
                }
            });
        }

    }catch(std::exception const& e){
        if(running) std::cerr << "[local-ws] error: " << e.what() << std::endl;
    }
}
