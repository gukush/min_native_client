
#pragma once
#include "iexecutor.hpp"

class BinaryExecutor : public IExecutor {
public:
    bool initialize(const json& cfg) override { (void)cfg; return true; }
    ExecResult run_task(const json& task) override;
    ExecResult execute(const json& task){ return run_task(task); }
};
