#pragma once
#include <nlohmann/json.hpp>
#include <vector>
#include <string>

struct ExecResult {
    bool ok{false};
    std::vector<std::vector<uint8_t>> outputs;
    double ms{0.0};
    std::string error;
    nlohmann::json timings = nlohmann::json::object();
};

class IExecutor {
public:
    using json = nlohmann::json;
    virtual ~IExecutor() = default;
    virtual bool initialize(const json& cfg) = 0;
    virtual ExecResult run_task(const json& task) = 0;
};
