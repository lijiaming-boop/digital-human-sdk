#pragma once
#include<vector>
#include<memory>

namespace digital_human{
    namespace core{
        class FaceAlignigner{
            public:
            FaceAlignigner();
            ~FaceAlignigner();
            FaceAlignigner& operator=(const FaceAlignigner& other) = delete;
            FaceAlignigner(const FaceAlignigner& other) = delete;
            FaceAlignigner& operator=(FaceAlignigner&& other) noexcept;
            FaceAlignigner(FaceAlignigner&& other) noexcept;
        private:
            std::unique_ptr<FaceAlignignerImpl> impl_;
            struct FaceAlignignerImpl;
        };
        struct FaceAlignerResult{
            std::vector<float> landmarks;
        };
    }
}