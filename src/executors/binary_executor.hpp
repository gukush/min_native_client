
#pragma once
#include "iexecutor.hpp"
#include <unordered_map>
#include <string>
#include <filesystem>

class BinaryExecutor : public IExecutor {
public:
    BinaryExecutor();
    ~BinaryExecutor();

    bool initialize(const json& cfg) override { (void)cfg; return true; }
    ExecResult run_task(const json& task) override;
    ExecResult execute(const json& task){ return run_task(task); }

    // New methods for artifact management
    void handle_workload_artifacts(const json& workload);
    void cleanup_task_artifacts(const std::string& task_id);
    std::string get_binary_path(const std::string& program_name, const std::string& task_id = "");

private:
    std::unordered_map<std::string, std::string> program_paths_; // program_name -> absolute_path
    std::unordered_map<std::string, std::string> task_artifacts_; // task_id -> temp_dir
    std::string base_cache_dir_;

    std::string create_task_directory(const std::string& task_id, const std::string& client_id = "");
    void write_artifact(const std::string& task_id, const json& artifact);
    void make_executable(const std::string& file_path);
};
