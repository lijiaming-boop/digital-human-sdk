#pragma once
#include "digital_human/export.h"

#include <cstddef>
#include <string>
#include <vector>

namespace digital_human {
namespace dialog {

struct DH_API SentenceSegmenterConfig {
    size_t min_weak_boundary_chars = 8;
};

/// Incrementally turns UTF-8 model output into TTS-friendly clauses.
class DH_API SentenceSegmenter {
public:
    explicit SentenceSegmenter(
        SentenceSegmenterConfig config = SentenceSegmenterConfig{});

    std::vector<std::string> Push(const std::string& delta);
    std::string Flush();
    void Reset();

private:
    SentenceSegmenterConfig config_;
    std::string buffer_;
};

}  // namespace dialog
}  // namespace digital_human
