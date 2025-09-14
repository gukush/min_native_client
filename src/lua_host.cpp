#include "lua_host.hpp"

#ifdef HAVE_LUA
#include <iostream>

LuaHost::LuaHost() {
    // nothing yet
}

LuaHost::~LuaHost() = default;

bool LuaHost::load(const std::string& lua_source, ExecCallback cb) {
    exec_ = std::move(cb);
    L_.open_libraries(sol::lib::base, sol::lib::math, sol::lib::string, sol::lib::table, sol::lib::package);

    // Bind executor.run(framework, task_table)
    L_["executor"] = L_.create_table();
    L_["executor"]["run"] = [this](const std::string& fw, sol::object task_obj) -> sol::object {
        if (!exec_) throw std::runtime_error("Executor callback not set");
        // Convert task table -> json
        json task = to_json(task_obj);
        json result = exec_(fw, task);
        return to_lua(L_, result);
    };

    // Load the script
    sol::protected_function_result r = L_.safe_script(lua_source, &sol::script_pass_on_error);
    if (!r.valid()) {
        sol::error err = r;
        std::cerr << "[lua] load error: " << err.what() << "\n";
        return false;
    }
    return true;
}

bool LuaHost::has_compile_and_run() const {
    sol::optional<sol::function> f = L_["compile_and_run"];
    return f.has_value();
}

LuaHost::json LuaHost::compile_and_run(const json& chunk) {
    sol::function f = L_["compile_and_run"];
    if (!f.valid()) throw std::runtime_error("compile_and_run not found");

    // Only make artifacts available if they exist
    if (!artifacts_.is_null()) {
        L_["artifacts"] = to_lua(L_, artifacts_);
    }

    sol::object arg = to_lua(L_, chunk);
    sol::protected_function_result r = f(arg);
    if (!r.valid()) {
        sol::error err = r;
        throw std::runtime_error(std::string("lua runtime error: ") + err.what());
    }
    sol::object out = r;
    return to_json(out);
}

// Method 2: Enhanced version with workload parameters
LuaHost::json LuaHost::compile_and_run(const json& chunk, const std::string& workload_framework, const json& workload_config) {
    sol::function f = L_["compile_and_run"];
    if (!f.valid()) throw std::runtime_error("compile_and_run not found");

    // Make workload framework and config available to Lua script
    L_["workload_framework"] = workload_framework;
    L_["workload_config"] = to_lua(L_, workload_config);

    // Also make artifacts available if they exist
    if (!artifacts_.is_null()) {
        L_["artifacts"] = to_lua(L_, artifacts_);
    }

    sol::object arg = to_lua(L_, chunk);
    sol::protected_function_result r = f(arg);
    if (!r.valid()) {
        sol::error err = r;
        throw std::runtime_error(std::string("lua runtime error: ") + err.what());
    }
    sol::object out = r;
    return to_json(out);
}

// ------------------ conversions ------------------

static LuaHost::json table_to_json(sol::table t);

LuaHost::json LuaHost::to_json(sol::object v) {
    if (!v.valid() || v.get_type() == sol::type::nil) return nullptr;

    switch (v.get_type()) {
        case sol::type::number: {
            // Lua numbers are doubles. Try to cast to integer if exact.
            double d = v.as<double>();
            if (std::floor(d) == d) return static_cast<long long>(d);
            return d;
        }
        case sol::type::string: return v.as<std::string>();
        case sol::type::boolean: return v.as<bool>();
        case sol::type::table: return table_to_json(v.as<sol::table>());
        default: return nullptr;
    }
}

static bool is_array_like(const sol::table& t) {
    // If keys are 1..N only, treat as array.
    std::size_t n = 0;
    for (auto& kv : t) {
        if (kv.first.get_type() != sol::type::number) return false;
        double d = kv.first.as<double>();
        if (std::floor(d) != d) return false;
        std::size_t idx = static_cast<std::size_t>(d);
        if (idx == 0) return false;
        if (idx > n) n = idx;
    }
    // allow sparse a bit but keep it simple
    return n > 0;
}

static LuaHost::json table_to_json(sol::table t) {
    if (is_array_like(t)) {
        LuaHost::json arr = LuaHost::json::array();
        std::size_t i = 1;
        while (t[i].valid()) {
            arr.push_back(LuaHost::to_json(t[i]));
            ++i;
        }
        return arr;
    } else {
        LuaHost::json obj = LuaHost::json::object();
        for (auto& kv : t) {
            if (kv.first.get_type() == sol::type::string) {
                std::string k = kv.first.as<std::string>();
                obj[k] = LuaHost::to_json(kv.second);
            }
        }
        return obj;
    }
}

sol::object LuaHost::to_lua(sol::state& L, const json& j) {
    switch (j.type()) {
        case json::value_t::null: return sol::make_object(L, sol::nil);
        case json::value_t::boolean: return sol::make_object(L, j.get<bool>());
        case json::value_t::number_integer: return sol::make_object(L, j.get<long long>());
        case json::value_t::number_unsigned: return sol::make_object(L, j.get<unsigned long long>());
        case json::value_t::number_float: return sol::make_object(L, j.get<double>());
        case json::value_t::string: return sol::make_object(L, j.get<std::string>());
        case json::value_t::array: {
            sol::table t = L.create_table();
            int idx = 1;
            for (const auto& v : j) {
                t[idx++] = to_lua(L, v);
            }
            return t;
        }
        case json::value_t::object: {
            sol::table t = L.create_table();
            for (auto it = j.begin(); it != j.end(); ++it) {
                t[it.key()] = to_lua(L, it.value());
            }
            return t;
        }
        default: return sol::make_object(L, sol::nil);
    }
}

void LuaHost::set_artifacts(const json& artifacts) {
    artifacts_ = artifacts;
    std::cout << "[Lua] Set Artifacts";
    // Make artifacts available to Lua script
    L_["artifacts"] = to_lua(L_, artifacts);
}

#endif // HAVE_LUA
