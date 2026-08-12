#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace digital_human {
namespace avatar {

enum class AvatarImageFormat {
    UNKNOWN,
    JPEG,
    PNG,
};

/// EXIF orientation 策略（P0 资源限制）：手机拍摄的 JPEG 通常携带 EXIF
/// orientation 标记，浏览器与社交平台默认按该标记旋转显示。SDK 必须明确
/// 该策略，避免头像方向不一致导致人脸检测失败或画面倒置。
/// - Apply（默认）：按 EXIF orientation 自动校正方向，与浏览器行为一致。
/// - Ignore：忽略 EXIF orientation，使用原始像素方向（用于调用方已在外层
///   完成方向校正或要求像素级一致的场景，便于精确尺寸探测与画布契约）。
enum class ExifOrientationPolicy {
    Apply,
    Ignore,
};

struct AvatarUploadLimits {
    size_t max_encoded_bytes = 10U * 1024U * 1024U;
    int max_width = 4096;
    int max_height = 4096;
    int64_t max_pixels = 16LL * 1024LL * 1024LL;
    /// EXIF orientation 处理策略，默认 Apply 与浏览器显示一致。
    ExifOrientationPolicy exif_orientation = ExifOrientationPolicy::Apply;
};

struct AvatarImage {
    cv::Mat bgr;
    AvatarImageFormat format = AvatarImageFormat::UNKNOWN;
};

/// Decodes untrusted JPEG/PNG upload bytes into an independently-owned BGR frame.
bool DecodeAvatarUpload(const std::vector<uint8_t>& encoded,
                        const std::string& content_type,
                        const AvatarUploadLimits& limits,
                        AvatarImage& image,
                        std::string& error);

/// Loads a persisted avatar file through the same validation path as uploads.
bool LoadAvatarImage(const std::string& path,
                     const AvatarUploadLimits& limits,
                     AvatarImage& image,
                     std::string& error);

}  // namespace avatar
}  // namespace digital_human
