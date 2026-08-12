#include "dialog/sentence_segmenter.h"

#include <algorithm>

namespace digital_human {
namespace dialog {
namespace {

size_t Utf8CharLength(unsigned char lead) {
    if ((lead & 0x80) == 0) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;
}

size_t Utf8Count(const std::string& text, size_t bytes) {
    size_t count = 0;
    for (size_t pos = 0; pos < std::min(bytes, text.size());) {
        pos += std::min(Utf8CharLength(
                            static_cast<unsigned char>(text[pos])),
                        text.size() - pos);
        ++count;
    }
    return count;
}

bool IsStrongBoundary(const std::string& cp) {
    return cp == "。" || cp == "！" || cp == "？"
        || cp == "." || cp == "!" || cp == "?"
        || cp == "\n";
}

bool IsWeakBoundary(const std::string& cp) {
    return cp == "，" || cp == "；" || cp == "," || cp == ";";
}

std::string Trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

}  // namespace

SentenceSegmenter::SentenceSegmenter(SentenceSegmenterConfig config)
    : config_(config) {}

std::vector<std::string> SentenceSegmenter::Push(const std::string& delta) {
    buffer_ += delta;
    std::vector<std::string> clauses;

    size_t clause_start = 0;
    size_t pos = 0;
    while (pos < buffer_.size()) {
        const size_t len = std::min(
            Utf8CharLength(static_cast<unsigned char>(buffer_[pos])),
            buffer_.size() - pos);
        const std::string cp = buffer_.substr(pos, len);
        const bool strong = IsStrongBoundary(cp);
        const bool weak = IsWeakBoundary(cp);
        const size_t end = pos + len;
        const size_t chars = Utf8Count(
            buffer_.substr(clause_start, end - clause_start),
            end - clause_start);

        if (strong || (weak && chars >= config_.min_weak_boundary_chars)) {
            auto clause = Trim(buffer_.substr(clause_start, end - clause_start));
            if (!clause.empty()) clauses.push_back(std::move(clause));
            clause_start = end;
        }
        pos = end;
    }

    if (clause_start > 0) buffer_.erase(0, clause_start);
    return clauses;
}

std::string SentenceSegmenter::Flush() {
    std::string result = Trim(buffer_);
    buffer_.clear();
    return result;
}

void SentenceSegmenter::Reset() {
    buffer_.clear();
}

}  // namespace dialog
}  // namespace digital_human
