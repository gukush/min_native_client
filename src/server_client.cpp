
#include "server_client.hpp"
#include <iostream>
#include <thread>

ServerBinaryClient::ServerBinaryClient(bool insecure){
    ws_ = std::make_unique<WebSocketClient>();
    bin_ = std::make_unique<BinaryExecutor>();
    ws_->onConnected = [&]{ connected_=true; };
    ws_->onDisconnected = [&]{ connected_=false; };
    ws_->onJson = [&](const nlohmann::json& j){ handle_message(j); };
}

std::tuple<std::string,std::string> ServerBinaryClient::split_host_port(const std::string& url){
    // expects wss://host:port
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
    while(connected_) std::this_thread::sleep_for(std::chrono::milliseconds(200));
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

void ServerBinaryClient::handle_chunk(const json& c){
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
