#include "network/json_utils.h"

#include <cctype>
#include <cstdint>
#include <cstdio>

namespace digital_human {
namespace network {
namespace json {
namespace {

void AppendUtf8(uint32_t codepoint, std::string& out) {
    if (codepoint <= 0x7F) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

bool Hex4(const std::string& input, size_t pos, uint32_t& value) {
    if (pos + 4 > input.size()) return false;
    value = 0;
    for (size_t i = 0; i < 4; ++i) {
        char c = input[pos + i];
        value <<= 4;
        if (c >= '0' && c <= '9') value |= static_cast<uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(c - 'A' + 10);
        else return false;
    }
    return true;
}

size_t FindValue(const std::string& input, const std::string& field) {
    const std::string needle = "\"" + field + "\"";
    size_t pos = input.find(needle);
    if (pos == std::string::npos) return pos;
    pos = input.find(':', pos + needle.size());
    if (pos == std::string::npos) return pos;
    ++pos;
    while (pos < input.size()
           && std::isspace(static_cast<unsigned char>(input[pos]))) {
        ++pos;
    }
    return pos;
}

}  // namespace

std::string Escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

bool ExtractString(const std::string& input,
                   const std::string& field,
                   std::string& value) {
    size_t pos = FindValue(input, field);
    if (pos == std::string::npos || pos >= input.size() || input[pos] != '\"') {
        return false;
    }
    ++pos;
    std::string out;
    while (pos < input.size()) {
        char c = input[pos++];
        if (c == '\"') {
            value = std::move(out);
            return true;
        }
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (pos >= input.size()) return false;
        char escaped = input[pos++];
        switch (escaped) {
            case '\"': out.push_back('\"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                uint32_t cp = 0;
                if (!Hex4(input, pos, cp)) return false;
                pos += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF
                    && pos + 6 <= input.size()
                    && input[pos] == '\\' && input[pos + 1] == 'u') {
                    uint32_t low = 0;
                    if (Hex4(input, pos + 2, low)
                        && low >= 0xDC00 && low <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10)
                           + (low - 0xDC00);
                        pos += 6;
                    }
                }
                AppendUtf8(cp, out);
                break;
            }
            default: return false;
        }
    }
    return false;
}

bool ExtractBool(const std::string& input,
                 const std::string& field,
                 bool& value) {
    size_t pos = FindValue(input, field);
    if (pos == std::string::npos) return false;
    if (input.compare(pos, 4, "true") == 0) {
        value = true;
        return true;
    }
    if (input.compare(pos, 5, "false") == 0) {
        value = false;
        return true;
    }
    return false;
}

}  // namespace json
}  // namespace network
}  // namespace digital_human
