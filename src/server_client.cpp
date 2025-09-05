#include "server_client.hpp"
#include "thread_pool.hpp"
#include <iostream>
#include <thread>

ServerBinaryClient::ServerBinaryClient(bool insecure, int concurrency) 
    : max_concurrency_(concurrency) {
    std::cout << "[client] Initializing with concurrency: " << concurrency << std::endl;
    thread_pool_ = std::make_unique<ThreadPool>(concurrency);
    
    ws_ = std::make_unique<WebSocketClient>();
    bin_ = std::make_unique<BinaryExecutor>();
    ws_->onConnected = [&]{ 
        connected_=true;
        std::cout << "[client] Connected with " << concurrency << " worker threads" << std::endl;
    };
    ws_->onDisconnected = [&]{ connected_=false; };
    ws_->onJson = [&](const nlohmann::json& j){ handle_message(j); };
}

ServerBinaryClient::~ServerBinaryClient() {
    // ThreadPool destructor will wait for pending tasks
}

std::tuple<std::string,std::string> ServerBinaryClient::split_host_port(const std::string& url){
    auto s = url.substr(6);
    auto pos = s.find(':');
    if(pos==std::string::npos) return {s,"443"};
    return {s.substr(0,pos), s.substr(pos+1)};
}

bool ServerBinaryClient::connect(const std::string& url){
    auto [host,port] = split_host_port(url);
    return ws_->connect(host, port, "/ws-native");
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
    if(type=="workload:new"){
        handle_workload(data);
    } else if(type=="workload:chunk_assign"){
        handle_chunk(data);
    } else if(type=="register"){
        // optional ack
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
    
    active_chunks_++;
    thread_pool_->enqueue([this, c]() {
        process_chunk_concurrent(c);
        active_chunks_--;
    });
}

void ServerBinaryClient::process_chunk_concurrent(const json& c){
    auto res = bin_->execute(c);
    nlohmann::json arr = nlohmann::json::array();
    for(auto& o: res.outputs) arr.push_back(base64_encode(o));
    
    nlohmann::json reply = {
        {"type","workload:chunk_done_enhanced"},
        {"data",{
            {"parentId", c.value("parentId","")},
            {"chunkId", c.value("chunkId","")},
            {"results", arr},
            {"processingTime", res.ms}
        }}
    };
    
    ws_->send_json(reply);
}

void ServerBinaryClient::process_workload_concurrent(const json& w){
    auto res = bin_->execute(w);
    nlohmann::json reply = {
        {"type","workload:done"},
        {"data",{
            {"id", w.value("id","")},
            {"result", res.ok && !res.outputs.empty() ? base64_encode(res.outputs[0]) : ""},
            {"processingTime", res.ms}
        }}
    };
    ws_->send_json(reply);
}
