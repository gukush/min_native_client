#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <thread>
#include "server_client.hpp"
#include "local_ws_server.hpp"

// Simple logging utility
class Logger {
private:
    static std::string getCurrentTime() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

public:
    static void info(const std::string& message) {
        std::cout << "[" << getCurrentTime() << "] [INFO] " << message << std::endl;
    }

    static void warn(const std::string& message) {
        std::cout << "[" << getCurrentTime() << "] [WARN] " << message << std::endl;
    }

    static void error(const std::string& message) {
        std::cerr << "[" << getCurrentTime() << "] [ERROR] " << message << std::endl;
    }

    static void debug(const std::string& message) {
        std::cout << "[" << getCurrentTime() << "] [DEBUG] " << message << std::endl;
    }
};

static void usage(const char* exe){
    std::cout << "Usage: " << exe << " --mode <server|local> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --url wss://localhost:3001    Server URL for server mode\n";
    std::cout << "  --concurrency <n>             Number of concurrent chunks (1-32, default: 1)\n";
    std::cout << "  --insecure                    Allow insecure connections\n";
    std::cout << "  --secure                      Force secure connections\n";
    std::cout << "  --ssl                         Enable SSL for local mode (default: true)\n";
    std::cout << "  --no-ssl                      Disable SSL for local mode\n";
    std::cout << "  --port <port>                 Port for local mode (default: 8787)\n";
    std::cout << "  --listener                    Enable listener mode for metrics collection\n";
    std::cout << "  --verbose                     Enable verbose logging\n";
    std::cout << "  --help                        Show this help\n";
}

int main(int argc, char** argv){
    std::string mode="server";
    std::string url="wss://localhost:3001";
    bool insecure=true;
    bool use_ssl=true;  // Default to SSL for local mode
    unsigned short port=8787;
    bool verbose=false;
    int concurrency=1;
    bool enable_listener=false;

    Logger::info("Native WebSocket Client starting up...");
    Logger::info("Build timestamp: " + std::string(__DATE__) + " " + std::string(__TIME__));

    // Parse command line arguments
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--mode" && i+1<argc){
            mode=argv[++i];
            Logger::debug("Mode set to: " + mode);
        }
        else if(a=="--url" && i+1<argc){
            url=argv[++i];
            Logger::debug("Server URL set to: " + url);
        }
        else if(a=="--concurrency" && i+1<argc){
            concurrency=std::atoi(argv[++i]);
            if(concurrency < 1) concurrency = 1;
            if(concurrency > 32) concurrency = 32;
            Logger::debug("Concurrency set to: " + std::to_string(concurrency));
        }
        else if(a=="--secure"){
            insecure=false;
            Logger::debug("Secure mode enabled");
        }
        else if(a=="--insecure"){
            insecure=true;
            Logger::debug("Insecure mode enabled");
            // Change default URL to use ws:// instead of wss://
            if(url.find("wss://") == 0) {
                url = "ws://" + url.substr(6);
                Logger::debug("Changed URL to insecure: " + url);
            }
        }
        else if(a=="--ssl"){
            use_ssl=true;
            Logger::debug("SSL enabled for local mode");
        }
        else if(a=="--no-ssl"){
            use_ssl=false;
            Logger::debug("SSL disabled for local mode");
        }
        else if(a=="--port" && i+1<argc){
            port=std::stoi(argv[++i]);
            Logger::debug("Port set to: " + std::to_string(port));
        }
        else if(a=="--verbose"){
            verbose=true;
            Logger::debug("Verbose logging enabled");
        }
        else if(a=="--listener"){
            enable_listener=true;
            Logger::debug("Listener mode enabled");
        }
        else if(a=="--help"){
            usage(argv[0]);
            return 0;
        }
        else {
            Logger::warn("Unknown argument: " + a);
        }
    }

    Logger::info("Configuration:");
    Logger::info("  Mode: " + mode);
    Logger::info("  Concurrency: " + std::to_string(concurrency) + " threads");
    Logger::info("  Listener: " + std::string(enable_listener ? "enabled" : "disabled"));

    if(mode == "server") {
        Logger::info("  Server URL: " + url);
        Logger::info("  Insecure: " + std::string(insecure ? "true" : "false"));
    } else {
        Logger::info("  Port: " + std::to_string(port));
        Logger::info("  SSL: " + std::string(use_ssl ? "enabled" : "disabled"));
    }
    Logger::info("  Verbose: " + std::string(verbose ? "enabled" : "disabled"));

    if(concurrency > 1) {
        Logger::info("=== CONCURRENT MODE ===");
        Logger::info("Running with " + std::to_string(concurrency) + " worker threads");
        Logger::info("Multiple chunks will be processed in parallel");
        Logger::info("Kernel compilation caching is enabled");
        Logger::warn("Note: GPU resources are shared across threads");
    }

    if(mode=="server"){
        Logger::info("=== SERVER MODE ===");
        Logger::info("Starting server-mode native client (binary execution)...");
        Logger::info("This mode connects TO a remote server, not FROM browsers");

        try {
            ServerBinaryClient client(insecure, concurrency, enable_listener);
            Logger::info("ServerBinaryClient created successfully");
            Logger::info("Thread pool initialized with " + std::to_string(concurrency) + " workers");

            Logger::info("Attempting to connect to: " + url);
            if(!client.connect(url)){
                Logger::error("Failed to connect to " + url);
                Logger::error("Check that the remote server is running and accessible");
                return 1;
            }
            Logger::info("Successfully connected to server!");
            Logger::info("Ready to process up to " + std::to_string(concurrency) + " chunks concurrently");

            Logger::info("Starting client run loop...");
            client.run();
            Logger::info("Client run loop ended");
        } catch (const std::exception& e) {
            Logger::error("Server mode exception: " + std::string(e.what()));
            return 1;
        }

    } else if(mode=="local"){
        Logger::info("=== LOCAL MODE ===");
        Logger::info("Starting local WebSocket server for browser connections");
        Logger::info("Browsers will connect TO this server FROM web pages");

        if(use_ssl) {
            Logger::info("SSL WebSocket server mode enabled");
            Logger::info("Browsers should connect to: wss://127.0.0.1:" + std::to_string(port) + "/native");
            Logger::warn("SSL Certificate Requirements:");
            Logger::warn("  - server.crt and server.key must exist in current directory");
            Logger::warn("  - Generate with: ./generate_cert.sh");
            Logger::warn("  - Or use: openssl req -x509 -newkey rsa:2048 -keyout server.key -out server.crt -days 365 -nodes -subj '/CN=localhost'");
            Logger::warn("Browser Setup:");
            Logger::warn("  - Use Chrome with: --ignore-certificate-errors --ignore-ssl-errors");
            Logger::warn("  - Or install server.crt as trusted certificate");
        } else {
            Logger::info("Plain WebSocket server mode enabled");
            Logger::info("Browsers should connect to: ws://127.0.0.1:" + std::to_string(port) + "/native");
            Logger::warn("Plain WebSocket mode - only works with HTTP pages (not HTTPS)");
        }

        // Check for SSL certificates if SSL is enabled
        if(use_ssl) {
            Logger::debug("Checking for SSL certificate files...");
            std::ifstream cert_file("server.crt");
            std::ifstream key_file("server.key");

            if(!cert_file.good()) {
                Logger::error("server.crt not found in current directory!");
                Logger::error("SSL server will fail to start without certificates");
                Logger::info("Generate certificates with: ./generate_cert.sh");
            } else {
                Logger::info("✓ server.crt found");
            }

            if(!key_file.good()) {
                Logger::error("server.key not found in current directory!");
                Logger::error("SSL server will fail to start without private key");
            } else {
                Logger::info("✓ server.key found");
            }
        }

        try {
            Logger::info("Creating LocalWSServer instance with " + std::to_string(concurrency) + " workers...");
            LocalWSServer srv(concurrency);

            Logger::info("Starting WebSocket server...");
            Logger::info("  Address: 127.0.0.1");
            Logger::info("  Port: " + std::to_string(port));
            Logger::info("  Path: /native");
            Logger::info("  SSL: " + std::string(use_ssl ? "enabled" : "disabled"));
            Logger::info("  Concurrency: " + std::to_string(concurrency) + " concurrent chunks");

            if(!srv.start("127.0.0.1", port, "/native", use_ssl)){
                Logger::error("Failed to start local WebSocket server");
                Logger::error("Possible causes:");
                Logger::error("  - Port " + std::to_string(port) + " already in use");
                Logger::error("  - Missing SSL certificates (if SSL enabled)");
                Logger::error("  - Permission denied to bind port");
                Logger::error("  - Invalid SSL certificate format");
                return 1;
            }

            Logger::info("✓ WebSocket server started successfully!");
            Logger::info("Server is now listening for browser connections...");
            Logger::info("Server will run indefinitely (press Ctrl+C to stop)");

            if(use_ssl) {
                Logger::info("");
                Logger::info("🌐 Browser Connection Info:");
                Logger::info("  URL: wss://127.0.0.1:" + std::to_string(port) + "/native");
                Logger::info("  Protocol: Secure WebSocket (WSS)");
                Logger::info("  Self-signed cert: Browsers will show security warning");
            } else {
                Logger::info("");
                Logger::info("🌐 Browser Connection Info:");
                Logger::info("  URL: ws://127.0.0.1:" + std::to_string(port) + "/native");
                Logger::info("  Protocol: Plain WebSocket (WS)");
                Logger::info("  Limitation: Only works with HTTP pages (not HTTPS/WebGPU)");
            }

            Logger::info("");
            Logger::info("📊 Concurrency Settings:");
            Logger::info("  Max concurrent requests: " + std::to_string(concurrency));
            Logger::info("  Kernel caching: ENABLED (shared across threads)");
            Logger::info("  Thread pool: " + std::to_string(concurrency) + " workers");
            Logger::info("  GPU frameworks: CUDA, OpenCL, Vulkan");

            Logger::info("");
            Logger::info("⏳ Waiting for connections... (will run for 24 hours)");

            // Block forever (or until interrupted)
            std::this_thread::sleep_for(std::chrono::hours(24));

            Logger::info("Server shutdown after 24 hours");

        } catch (const std::exception& e) {
            Logger::error("Local mode exception: " + std::string(e.what()));
            return 1;
        }

    } else {
        Logger::error("Invalid mode: " + mode);
        Logger::error("Valid modes are: server, local");
        usage(argv[0]);
        return 1;
    }

    Logger::info("Application exiting normally");
    return 0;
}
