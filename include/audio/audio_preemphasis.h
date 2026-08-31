#pragma once

#include <vector>
#include <memory>
#include "digital_human/export.h"

namespace digital_human {
namespace audio {

class DH_API PreEmphasis {
public:
    explicit PreEmphasis(float alpha = 0.97f);
    ~PreEmphasis();
    PreEmphasis(const PreEmphasis&) = delete;
    PreEmphasis& operator=(const PreEmphasis&) = delete;
    PreEmphasis(PreEmphasis&&) noexcept;
    PreEmphasis& operator=(PreEmphasis&&) noexcept;

    std::vector<float> process(const std::vector<float>& pcm) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
}  // namespace digital_human
