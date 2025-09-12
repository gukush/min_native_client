#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <csignal>
#include <atomic>
#include "orchestrator_client.hpp"

// Global flag for signal handling
static std::atomic<bool> g_interrupted{false};

// Signal handler
static void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\n[Signal] Received interrupt signal, shutting down gracefully..." << std::endl;
        g_interrupted = true;
    }
}

// Very small CLI parser
struct Args {
    std::string url = "wss://localhost:3001";
    bool insecure = true;          // allow self-signed by default
    int concurrency = 1;
    bool verbose = true;
};

static void print_help(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--url wss://host:port/path] [--secure|--insecure] [--concurrency N]\n";
    std::cout << "\nThe client can be stopped gracefully with Ctrl+C (SIGINT) or SIGTERM.\n";
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--help" || s == "-h") { print_help(argv[0]); std::exit(0); }
        else if (s == "--url" && i + 1 < argc) { a.url = argv[++i]; }
        else if (s == "--secure") { a.insecure = false; }
        else if (s == "--insecure") { a.insecure = true; }
        else if (s == "--concurrency" && i + 1 < argc) { a.concurrency = std::max(1, std::atoi(argv[++i])); }
        else if (s == "--quiet") { a.verbose = false; }
        else {
            std::cerr << "Unknown arg: " << s << "\n";
            print_help(argv[0]);
            std::exit(2);
        }
    }
    return a;
}

int main(int argc, char** argv) {
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto args = parse_args(argc, argv);

    OrchestratorClient client{args.insecure, args.concurrency};
    if (!client.connect(args.url)) {
        std::cerr << "Failed to connect to " << args.url << "\n";
        return 2;
    }

    // Pass the interrupt flag to the client
    client.run(g_interrupted); // blocking event loop with interrupt support
    return 0;
}
