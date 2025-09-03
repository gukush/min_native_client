
#include "binary_executor.hpp"
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <iostream>
#include <cstring>
#include <chrono>

static std::vector<std::string> get_args(const nlohmann::json& j){
    std::vector<std::string> a;
    if(j.contains("args") && j["args"].is_array()){
        for(auto& x: j["args"]) if(x.is_string()) a.push_back(x.get<std::string>());
    }
    return a;
}

ExecResult BinaryExecutor::run_task(const json& task){
    ExecResult r;
    std::string exe = task.value("executable","");
    if(exe.empty()){ r.error="no executable"; return r; }

    std::vector<uint8_t> stdin_data;
    if(task.contains("stdin") && task["stdin"].is_string()){
        // Assume base64 in upstream, but here we expect raw for simplicity
        // Accept raw hex-like strings? keep raw minimal
        auto s = task["stdin"].get<std::string>();
        stdin_data.assign(s.begin(), s.end());
    }

    auto args = get_args(task);
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe.c_str()));
    for(auto& a: args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    int inpipe[2], outpipe[2];
	if (pipe(inpipe) == -1 || pipe(outpipe) == -1) {
    	// error
    	return ExecResult{false, {}, 0};
	}

    auto t0 = std::chrono::high_resolution_clock::now();
    pid_t pid = fork();
    if(pid==0){
        dup2(inpipe[0], STDIN_FILENO);
        dup2(outpipe[1], STDOUT_FILENO);
        close(inpipe[1]); close(outpipe[0]);
        execve(exe.c_str(), argv.data(), nullptr);
        std::perror("execve");
        _exit(127);
    } else if(pid>0){
        close(inpipe[0]); close(outpipe[1]);
        if(!stdin_data.empty()) {
            ssize_t written = write(inpipe[1], stdin_data.data(), stdin_data.size());
            if(written < 0) {
                r.error = "failed to write to stdin";
                return r;
            }
        }
        close(inpipe[1]);

        std::vector<uint8_t> out;
        uint8_t buf[4096];
        ssize_t n;
        while((n=read(outpipe[0], buf, sizeof(buf)))>0){
            out.insert(out.end(), buf, buf+n);
        }
        close(outpipe[0]);
        int status=0; waitpid(pid, &status, 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        r.ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        r.ok = (WIFEXITED(status) && WEXITSTATUS(status)==0);
        if(!r.ok) r.error="process failed";
        r.outputs = { std::move(out) };
        return r;
    } else {
        r.error="fork failed";
        return r;
    }
}
