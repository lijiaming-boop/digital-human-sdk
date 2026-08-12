#pragma once

#include <string>

namespace digital_human {
namespace network {
namespace json {

std::string Escape(const std::string& value);
bool ExtractString(const std::string& input,
                   const std::string& field,
                   std::string& value);
bool ExtractBool(const std::string& input,
                 const std::string& field,
                 bool& value);

}  // namespace json
}  // namespace network
}  // namespace digital_human
