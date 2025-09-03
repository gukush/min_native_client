
#include "base64.hpp"
#include <string>
#include <vector>

static const std::string b64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::vector<uint8_t>& data) {
    std::string out;
    int val=0, valb=-6;
    for (uint8_t c : data) {
        val = (val<<8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(b64_chars[(val>>valb)&0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(b64_chars[((val<<8)>>(valb+8))&0x3F]);
    while (out.size()%4) out.push_back('=');
    return out;
}

std::vector<uint8_t> base64_decode(const std::string& s) {
    std::vector<int> T(256, -1);
    for (int i=0; i<64; i++) T[b64_chars[i]] = i;
    std::vector<uint8_t> out;
    int val=0, valb=-8;
    for (uint8_t c : s) {
        if (T[c] == -1) break;
        val = (val<<6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(uint8_t((val>>valb)&0xFF));
            valb -= 8;
        }
    }
    return out;
}
