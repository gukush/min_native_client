#pragma once
#ifdef HAVE_LUA
#include <sol/sol.hpp>
#include <nlohmann/json.hpp>
#include <functional>

class LuaHost {
public:
    using json = nlohmann::json;
    using ExecCallback = std::function<json(const std::string&, const json&)>;

    LuaHost();
    ~LuaHost();

    // Provide Lua source and the callback to run real executors
    bool load(const std::string& lua_source, ExecCallback cb);

    bool has_compile_and_run() const;
    // Calls compile_and_run(chunk) in Lua. Chunk is passed as a Lua table.
    json compile_and_run(const json& chunk);
    // Calls compile_and_run(chunk) in Lua with workload framework and config available.
    json compile_and_run(const json& chunk, const std::string& workload_framework, const json& workload_config);

    // conversions (public for helper functions)
    static json to_json(sol::object v);
    static sol::object to_lua(sol::state& L, const json& j);
    void set_artifacts(const json& artifacts);
private:

private:
    sol::state L_;
    json artifacts_;
    ExecCallback exec_;
};

#endif // HAVE_LUA
