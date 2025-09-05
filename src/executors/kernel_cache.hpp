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
