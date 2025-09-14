#pragma once
#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>
#include <chrono>
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>

template<typename KernelType>
class KernelCache {
public:
    struct CachedKernel {
        KernelType kernel;
        std::string buildLog;
        std::chrono::steady_clock::time_point lastUsed;
    };

    std::shared_ptr<CachedKernel> get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if(it != cache_.end()) {
            it->second->lastUsed = std::chrono::steady_clock::now();
            return it->second;
        }
        return nullptr;
    }

    void put(const std::string& key, std::shared_ptr<CachedKernel> kernel) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[key] = kernel;

        if(cache_.size() > max_size_) {
            evictOldest();
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    static std::string computeHash(const std::string& data) {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256_CTX sha256;
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, data.c_str(), data.length());
        SHA256_Final(hash, &sha256);

        std::stringstream ss;
        for(int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        return ss.str();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<CachedKernel>> cache_;
    size_t max_size_ = 100;

    void evictOldest() {
        auto oldest = cache_.begin();
        auto oldestTime = oldest->second->lastUsed;

        for(auto it = cache_.begin(); it != cache_.end(); ++it) {
            if(it->second->lastUsed < oldestTime) {
                oldest = it;
                oldestTime = it->second->lastUsed;
            }
        }

        if(oldest != cache_.end()) {
            cache_.erase(oldest);
        }
    }
};


// Helper function to create a truncated/summarized version of JSON for debugging
inline nlohmann::json truncate_json_for_debug(const nlohmann::json& j, size_t max_string_length = 100, int max_depth = 3, int current_depth = 0) {
    if (current_depth >= max_depth) {
        return "[MAX_DEPTH_REACHED]";
    }

    if (j.is_null()) {
        return nullptr;
    } else if (j.is_boolean() || j.is_number()) {
        return j;
    } else if (j.is_string()) {
        std::string str = j.get<std::string>();
        if (str.length() > max_string_length) {
            return str.substr(0, max_string_length) + "...[" + std::to_string(str.length()) + " chars total]";
        }
        return j;
    } else if (j.is_array()) {
        nlohmann::json result = nlohmann::json::array();
        size_t count = 0;
        for (const auto& item : j) {
            if (count < 3) {  // Show first 3 items
                result.push_back(truncate_json_for_debug(item, max_string_length, max_depth, current_depth + 1));
            } else {
                result.push_back("[..." + std::to_string(j.size() - 3) + " more items]");
                break;
            }
            count++;
        }
        return result;
    } else if (j.is_object()) {
        nlohmann::json result = nlohmann::json::object();
        size_t count = 0;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (count < 10) {  // Show first 10 keys
                result[it.key()] = truncate_json_for_debug(it.value(), max_string_length, max_depth, current_depth + 1);
            } else {
                result["..."] = "[" + std::to_string(j.size() - 10) + " more keys]";
                break;
            }
            count++;
        }
        return result;
    }
    return j;
}

inline std::string json_summary(const nlohmann::json& j) {
        std::ostringstream ss;

        if (j.is_object()) {
            ss << "{";
            bool first = true;
            for (auto it = j.begin(); it != j.end(); ++it) {
                if (!first) ss << ", ";
                ss << "\"" << it.key() << "\":";

                if (it.value().is_string()) {
                    std::string str = it.value().get<std::string>();
                    if (str.length() > 50) {
                        ss << "\"[string:" << str.length() << "chars]\"";
                    } else {
                        ss << "\"" << str << "\"";
                    }
                } else if (it.value().is_array()) {
                    ss << "[array:" << it.value().size() << "items]";
                } else if (it.value().is_object()) {
                    ss << "{object:" << it.value().size() << "keys}";
                } else {
                    ss << it.value();
                }
                first = false;
            }
            ss << "}";
        } else {
            ss << j.type_name();
        }

        return ss.str();
}
