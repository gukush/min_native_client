
#include "binary_executor.hpp"
#include "../base64.hpp"
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>
extern char **environ;

static std::vector<std::string> get_args(const nlohmann::json& j){
    std::vector<std::string> a;
    if(j.contains("args") && j["args"].is_array()){
        for(auto& x: j["args"]) if(x.is_string()) a.push_back(x.get<std::string>());
    }
    return a;
}

BinaryExecutor::BinaryExecutor() {
    // Set up base cache directory
    const char* home = getenv("HOME");
    if (home) {
        base_cache_dir_ = std::string(home) + "/.cache/volunteer";
    } else {
        base_cache_dir_ = "/tmp/volunteer";
    }

    // Create base cache directory if it doesn't exist
    std::filesystem::create_directories(base_cache_dir_);
    std::cout << "[BinaryExecutor] Cache directory: " << base_cache_dir_ << std::endl;
}

BinaryExecutor::~BinaryExecutor() {
    // Clean up all task artifacts
    for (const auto& [task_id, temp_dir] : task_artifacts_) {
        try {
            std::filesystem::remove_all(temp_dir);
            std::cout << "[BinaryExecutor] Cleaned up task artifacts for " << task_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[BinaryExecutor] Error cleaning up " << task_id << ": " << e.what() << std::endl;
        }
    }
}

ExecResult BinaryExecutor::run_task(const json& task){
    ExecResult r;

    // Handle artifacts if this is a workload:new message
    if (task.contains("artifacts") && task["artifacts"].is_array()) {
        handle_workload_artifacts(task);
    }

    // Handle artifacts from workload if this is a chunk with workload info
    if (task.contains("workload") && task["workload"].contains("artifacts")) {
        // Use the task ID from the chunk, not the workload
        json workload_with_id = task["workload"];
        workload_with_id["id"] = task.value("id", "");
        handle_workload_artifacts(workload_with_id);
    }

    // Determine executable path
    std::string exe;
    std::string task_id = task.value("id", "");

    // Check if this is a chunk with specific program in meta
    if (task.contains("meta") && task["meta"].contains("program")) {
        std::string program_name = task["meta"]["program"];
        exe = get_binary_path(program_name, task_id);
        std::cout << "[BinaryExecutor] Using binary from meta.program: " << program_name << " -> " << exe << std::endl;
    } else if (task.contains("workload") && task["workload"].contains("config") && task["workload"]["config"].contains("program")) {
        // Get program name from workload config
        std::string program_name = task["workload"]["config"]["program"];
        exe = get_binary_path(program_name, task_id);
        std::cout << "[BinaryExecutor] Using binary from workload.config.program: " << program_name << " -> " << exe << std::endl;
    } else if (task.contains("program")) {
        // Fall back to workload program
        std::string program_name = task["program"];
        exe = get_binary_path(program_name, task_id);
        std::cout << "[BinaryExecutor] Using binary from task.program: " << program_name << " -> " << exe << std::endl;
    } else {
        // Legacy: direct executable path
        exe = task.value("executable", "");
        std::cout << "[BinaryExecutor] Using direct executable path: " << exe << std::endl;
    }

    if(exe.empty()){
        r.error="no executable found";
        return r;
    }

    // Handle chunked data for block matmul
    std::vector<uint8_t> stdin_data;

    std::cout << "[BinaryExecutor] Task keys: ";
    for (auto& [key, value] : task.items()) {
        std::cout << key << " ";
    }
    std::cout << std::endl;

    if(task.contains("stdin") && task["stdin"].is_string()){
        // Raw binary data from stdin
        auto s = task["stdin"].get<std::string>();
        stdin_data.assign(s.begin(), s.end());
        std::cout << "[BinaryExecutor] Got stdin data: " << stdin_data.size() << " bytes" << std::endl;
    } else if (task.contains("buffers") && task["buffers"].is_array()) {
        // Handle buffers array format from native-block-matmul-flex strategy
        auto buffers = task["buffers"];
        std::cout << "[BinaryExecutor] Processing " << buffers.size() << " buffers" << std::endl;

        for (size_t i = 0; i < buffers.size(); i++) {
            const auto& buffer = buffers[i];
            size_t buffer_start = stdin_data.size();

            if (buffer.is_array()) {
                // Convert array of bytes to vector
                for (const auto& byte : buffer) {
                    if (byte.is_number()) {
                        stdin_data.push_back(static_cast<uint8_t>(byte.get<int>()));
                    }
                }
            } else if (buffer.is_string()) {
                // Handle base64 encoded data
                auto b = base64_decode(buffer.get<std::string>());
                stdin_data.insert(stdin_data.end(), b.begin(), b.end());
            }

            std::cout << "[BinaryExecutor] Buffer " << i << ": " << (stdin_data.size() - buffer_start) << " bytes" << std::endl;
        }
        std::cout << "[BinaryExecutor] Total stdin data: " << stdin_data.size() << " bytes" << std::endl;
    } else {
        std::cout << "[BinaryExecutor] No stdin or buffers data found in task" << std::endl;
    }

    // Build command line arguments for the binary
    std::vector<std::string> arg_strings;

    // Add matrix dimensions from meta if available
    if (task.contains("meta")) {
        auto meta = task["meta"];
        if (meta.contains("uniforms") && meta["uniforms"].is_array()) {
            auto uniforms = meta["uniforms"];
            if (uniforms.size() >= 3) {
                // uniforms: [rows, K, cols]
                arg_strings.push_back("--rows");
                arg_strings.push_back(std::to_string(uniforms[0].get<int>()));
                arg_strings.push_back("--k");
                arg_strings.push_back(std::to_string(uniforms[1].get<int>()));
                arg_strings.push_back("--cols");
                arg_strings.push_back(std::to_string(uniforms[2].get<int>()));

                std::cout << "[BinaryExecutor] Matrix dimensions: "
                         << uniforms[0].get<int>() << " x "
                         << uniforms[1].get<int>() << " x "
                         << uniforms[2].get<int>() << std::endl;
            }
        }

        // Add backend information
        if (meta.contains("backend")) {
            arg_strings.push_back("--backend");
            arg_strings.push_back(meta["backend"].get<std::string>());
        }
    }

    // Add any additional args from task
    auto additional_args = get_args(task);
    arg_strings.insert(arg_strings.end(), additional_args.begin(), additional_args.end());

    // Convert to char* array for execve
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe.c_str()));
    for(auto& a: arg_strings) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    // Debug: print the command being executed
    std::cout << "[BinaryExecutor] Executing: " << exe;
    for(auto& a: arg_strings) std::cout << " " << a;
    std::cout << std::endl;

    int inpipe[2], outpipe[2];
    if (pipe(inpipe) == -1 || pipe(outpipe) == -1) {
        r.error = "failed to create pipes";
        return r;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Debug: print the executable path
    std::cout << "[BinaryExecutor] Attempting to execute: " << exe << std::endl;
    std::cout << "[BinaryExecutor] Working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "[BinaryExecutor] File exists: " << std::filesystem::exists(exe) << std::endl;
    if (std::filesystem::exists(exe)) {
        std::cout << "[BinaryExecutor] Is regular file: " << std::filesystem::is_regular_file(exe) << std::endl;
        std::cout << "[BinaryExecutor] File size: " << std::filesystem::file_size(exe) << " bytes" << std::endl;
    }

    pid_t pid = fork();
    if(pid==0){
        dup2(inpipe[0], STDIN_FILENO);
        dup2(outpipe[1], STDOUT_FILENO);
        close(inpipe[1]); close(outpipe[0]);
        execve(exe.c_str(), argv.data(), environ);
        std::perror("execve");
        _exit(127);
    } else if(pid>0){
        close(inpipe[0]); close(outpipe[1]);

        if(!stdin_data.empty()) {
            std::cout << "[BinaryExecutor] Writing " << stdin_data.size() << " bytes to stdin" << std::endl;
            ssize_t written = write(inpipe[1], stdin_data.data(), stdin_data.size());
            if(written < 0) {
                r.error = "failed to write to stdin";
                close(inpipe[1]); close(outpipe[0]);
                return r;
            }
            std::cout << "[BinaryExecutor] Successfully wrote " << written << " bytes to stdin" << std::endl;
        } else {
            std::cout << "[BinaryExecutor] No data to write to stdin" << std::endl;
        }
        close(inpipe[1]);

        std::vector<uint8_t> out;
        uint8_t buf[4096];
        ssize_t n;
        while((n=read(outpipe[0], buf, sizeof(buf)))>0){
            out.insert(out.end(), buf, buf+n);
        }
        close(outpipe[0]);

        std::cout << "[BinaryExecutor] Read " << out.size() << " bytes from stdout" << std::endl;

        int status=0;
        waitpid(pid, &status, 0);

        auto t1 = std::chrono::high_resolution_clock::now();
        r.ms = std::chrono::duration<double, std::milli>(t1-t0).count();

        if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            std::cout << "[BinaryExecutor] Process exited with code: " << exit_code << std::endl;
            r.ok = (exit_code == 0);
            if (!r.ok) {
                r.error = "process exited with code " + std::to_string(exit_code);
            }
        } else if (WIFSIGNALED(status)) {
            int signal = WTERMSIG(status);
            std::cout << "[BinaryExecutor] Process killed by signal: " << signal << std::endl;
            r.ok = false;
            r.error = "process killed by signal " + std::to_string(signal);
        } else {
            std::cout << "[BinaryExecutor] Process terminated abnormally" << std::endl;
            r.ok = false;
            r.error = "process terminated abnormally";
        }

        r.outputs = { std::move(out) };
        return r;
    } else {
        r.error="fork failed";
        return r;
    }
}

void BinaryExecutor::handle_workload_artifacts(const json& workload) {
    std::string task_id = workload.value("id", "");
    if (task_id.empty()) {
        std::cerr << "[BinaryExecutor] No task ID in workload artifacts" << std::endl;
        return;
    }

    // Extract client ID from workload (sent once per task, not per chunk)
    std::string client_id = workload.value("clientId", "");

    // Create task directory with client ID to prevent conflicts
    std::string task_dir = create_task_directory(task_id, client_id);
    if (task_dir.empty()) {
        std::cerr << "[BinaryExecutor] Failed to create task directory for " << task_id << std::endl;
        return;
    }

    // Process each artifact
    const auto& artifacts = workload["artifacts"];
    for (const auto& artifact : artifacts) {
        write_artifact(task_id, artifact);
    }

    std::cout << "[BinaryExecutor] Processed " << artifacts.size() << " artifacts for task " << task_id << std::endl;
}

void BinaryExecutor::cleanup_task_artifacts(const std::string& task_id) {
    auto it = task_artifacts_.find(task_id);
    if (it != task_artifacts_.end()) {
        try {
            std::filesystem::remove_all(it->second);
            std::cout << "[BinaryExecutor] Cleaned up artifacts for task " << task_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[BinaryExecutor] Error cleaning up task " << task_id << ": " << e.what() << std::endl;
        }
        task_artifacts_.erase(it);
    }
}

std::string BinaryExecutor::get_binary_path(const std::string& program_name, const std::string& task_id) {
    // First check if we have a cached path for this program
    auto it = program_paths_.find(program_name);
    if (it != program_paths_.end()) {
        return it->second;
    }

    // If we have a task_id, look in the task's artifact directory
    if (!task_id.empty()) {
        auto task_it = task_artifacts_.find(task_id);
        if (task_it != task_artifacts_.end()) {
            std::string task_dir = task_it->second;
            std::string exe_path = task_dir + "/" + program_name;

            // Check if the executable exists
            if (std::filesystem::exists(exe_path) && std::filesystem::is_regular_file(exe_path)) {
                program_paths_[program_name] = exe_path;
                return exe_path;
            }
        }
    }

    // Fall back to system PATH
    return program_name;
}

std::string BinaryExecutor::create_task_directory(const std::string& task_id, const std::string& client_id) {
    // Create unique directory name using task_id + client_id to prevent conflicts
    std::string unique_id = task_id;
    if (!client_id.empty()) {
        unique_id = task_id + "_" + client_id;
    }

    std::string task_dir = base_cache_dir_ + "/" + unique_id;

    try {
        std::filesystem::create_directories(task_dir);
        task_artifacts_[task_id] = task_dir; // Still use task_id as key for cleanup
        std::cout << "[BinaryExecutor] Created task directory: " << task_dir << " (task_id: " << task_id << ", client_id: " << client_id << ")" << std::endl;
        return task_dir;
    } catch (const std::exception& e) {
        std::cerr << "[BinaryExecutor] Failed to create task directory " << task_dir << ": " << e.what() << std::endl;
        return "";
    }
}

void BinaryExecutor::write_artifact(const std::string& task_id, const json& artifact) {
    auto task_it = task_artifacts_.find(task_id);
    if (task_it == task_artifacts_.end()) {
        std::cerr << "[BinaryExecutor] No task directory for " << task_id << std::endl;
        return;
    }

    std::string task_dir = task_it->second;
    std::string name = artifact.value("name", "");
    std::string type = artifact.value("type", "binary");
    bool is_executable = artifact.value("exec", false);

    if (name.empty()) {
        std::cerr << "[BinaryExecutor] Artifact missing name" << std::endl;
        return;
    }

    std::string file_path = task_dir + "/" + name;

    try {
        if (type == "text" && artifact.contains("content")) {
            // Text file
            std::string content = artifact["content"];
            std::ofstream file(file_path);
            file << content;
            file.close();
        } else if (artifact.contains("bytes")) {
            // Binary file (base64 encoded)
            std::string base64_data = artifact["bytes"];
            std::vector<uint8_t> binary_data = base64_decode(base64_data);

            std::ofstream file(file_path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(binary_data.data()), binary_data.size());
            file.close();
        } else {
            std::cerr << "[BinaryExecutor] Unknown artifact type or missing data for " << name << std::endl;
            return;
        }

        // Make executable if specified
        if (is_executable) {
            make_executable(file_path);
        }

        // Cache the path if it's an executable
        if (is_executable) {
            program_paths_[name] = file_path;
        }

        std::cout << "[BinaryExecutor] Wrote artifact " << name << " to " << file_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[BinaryExecutor] Error writing artifact " << name << ": " << e.what() << std::endl;
    }
}

void BinaryExecutor::make_executable(const std::string& file_path) {
#ifdef _WIN32
    // On Windows, we don't need to set executable permissions
    // The file extension (.exe) is what matters
#else
    // On POSIX systems, set executable permissions
    struct stat st;
    if (stat(file_path.c_str(), &st) == 0) {
        chmod(file_path.c_str(), st.st_mode | S_IXUSR | S_IXGRP | S_IXOTH);
    }
#endif
}
