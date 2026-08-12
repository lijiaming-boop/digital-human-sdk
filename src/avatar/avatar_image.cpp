#include "avatar/avatar_image.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace digital_human {
namespace avatar {
namespace {

constexpr uint8_t kPngSignature[] = {
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
};

bool IsValidLimits(const AvatarUploadLimits& limits) {
    return limits.max_encoded_bytes > 0 && limits.max_width > 0
        && limits.max_height > 0 && limits.max_pixels > 0;
}

AvatarImageFormat DetectFormat(const std::vector<uint8_t>& encoded) {
    if (encoded.size() >= 3 && encoded[0] == 0xFF && encoded[1] == 0xD8
        && encoded[2] == 0xFF) {
        return AvatarImageFormat::JPEG;
    }
    if (encoded.size() >= sizeof(kPngSignature)
        && std::equal(std::begin(kPngSignature), std::end(kPngSignature),
                      encoded.begin())) {
        return AvatarImageFormat::PNG;
    }
    return AvatarImageFormat::UNKNOWN;
}

/// 解析 JPEG/PNG header 获取尺寸（P0 资源限制）：在调用 cv::imdecode 完整解码前
/// 先校验宽高与像素总量，避免解压炸弹（小体积编码 → 巨大像素）造成内存峰值。
/// 失败时返回 false（无法探测尺寸，调用方应回退到完整解码后再校验）。
bool ProbeDimensions(const std::vector<uint8_t>& encoded,
                     AvatarImageFormat format, int& width, int& height) {
    width = 0;
    height = 0;
    if (format == AvatarImageFormat::PNG) {
        // PNG: 8 字节签名 + 4 字节长度 + 4 字节 "IHDR" + 4 字节宽 + 4 字节高（大端）
        if (encoded.size() < 24) return false;
        width = (static_cast<int>(encoded[16]) << 24)
              | (static_cast<int>(encoded[17]) << 16)
              | (static_cast<int>(encoded[18]) << 8)
              | static_cast<int>(encoded[19]);
        height = (static_cast<int>(encoded[20]) << 24)
               | (static_cast<int>(encoded[21]) << 16)
               | (static_cast<int>(encoded[22]) << 8)
               | static_cast<int>(encoded[23]);
        return width > 0 && height > 0;
    }
    if (format == AvatarImageFormat::JPEG) {
        // JPEG: 扫描 SOF0(0xFF 0xC0)/SOF1/SOF2 标记，其后 2 字节精度、2 字节高、2 字节宽。
        const size_t n = encoded.size();
        size_t i = 2;  // 跳过 SOI(0xFF 0xD8)
        while (i + 9 < n) {
            if (encoded[i] != 0xFF) { ++i; continue; }
            const uint8_t marker = encoded[i + 1];
            // SOF0/1/2/3 ... SOF15 (0xC0-0xCF)，排除 DHT(0xC4)/JPG(0xC8)/DAC(0xCC)
            if (marker >= 0xC0 && marker <= 0xCF
                && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
                height = (static_cast<int>(encoded[i + 5]) << 8)
                       | static_cast<int>(encoded[i + 6]);
                width = (static_cast<int>(encoded[i + 7]) << 8)
                      | static_cast<int>(encoded[i + 8]);
                return width > 0 && height > 0;
            }
            // 跳过本段：2 字节段长（大端，含长度自身，不含 0xFF marker）
            if (i + 3 >= n) return false;
            const size_t seg_len = (static_cast<size_t>(encoded[i + 2]) << 8)
                                 | static_cast<size_t>(encoded[i + 3]);
            if (seg_len < 2) return false;
            i += 2 + seg_len;
        }
        return false;
    }
    return false;
}

std::string NormalizeContentType(std::string content_type) {
    const size_t separator = content_type.find(';');
    if (separator != std::string::npos) content_type.resize(separator);
    content_type.erase(
        std::remove_if(content_type.begin(), content_type.end(),
                       [](unsigned char ch) { return std::isspace(ch) != 0; }),
        content_type.end());
    std::transform(content_type.begin(), content_type.end(),
                   content_type.begin(), [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return content_type;
}

bool ContentTypeMatches(const std::string& content_type,
                        AvatarImageFormat format) {
    if (content_type.empty()) return true;
    if (format == AvatarImageFormat::JPEG) {
        return content_type == "image/jpeg" || content_type == "image/jpg";
    }
    if (format == AvatarImageFormat::PNG) return content_type == "image/png";
    return false;
}

bool NormalizeBGR(const cv::Mat& decoded, cv::Mat& bgr, std::string& error) {
    if (decoded.empty()) {
        error = "avatar image decoder returned an empty image";
        return false;
    }
    if (decoded.depth() != CV_8U) {
        error = "avatar image must use 8-bit channels";
        return false;
    }

    switch (decoded.channels()) {
        case 1:
            cv::cvtColor(decoded, bgr, cv::COLOR_GRAY2BGR);
            break;
        case 3:
            bgr = decoded.clone();
            break;
        case 4:
            cv::cvtColor(decoded, bgr, cv::COLOR_BGRA2BGR);
            break;
        default:
            error = "avatar image must have 1, 3, or 4 channels";
            return false;
    }
    if (!bgr.isContinuous()) bgr = bgr.clone();
    return true;
}

}  // namespace

bool DecodeAvatarUpload(const std::vector<uint8_t>& encoded,
                        const std::string& content_type,
                        const AvatarUploadLimits& limits,
                        AvatarImage& image,
                        std::string& error) {
    image = AvatarImage{};
    error.clear();
    if (!IsValidLimits(limits)) {
        error = "invalid avatar upload limits";
        return false;
    }
    if (encoded.empty()) {
        error = "avatar upload is empty";
        return false;
    }
    if (encoded.size() > limits.max_encoded_bytes) {
        error = "avatar upload exceeds the encoded byte limit";
        return false;
    }

    const AvatarImageFormat format = DetectFormat(encoded);
    if (format == AvatarImageFormat::UNKNOWN) {
        error = "avatar upload is not a supported JPEG or PNG image";
        return false;
    }
    if (!ContentTypeMatches(NormalizeContentType(content_type), format)) {
        error = "avatar Content-Type does not match the encoded image";
        return false;
    }
    if (encoded.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "avatar upload is too large for the image decoder";
        return false;
    }

    // 解码前先探测尺寸（P0 资源限制）：在 imdecode 分配完整像素缓冲前，
    // 用 header 解析结果校验宽高/像素上限，拒绝解压炸弹。
    // 注意：探测得到的是编码像素尺寸（不含 EXIF 旋转），Apply 策略下 imdecode
    // 可能输出与探测尺寸互换（90°/270° 旋转）的帧，因此解码后必须再次校验。
    int probe_w = 0;
    int probe_h = 0;
    if (ProbeDimensions(encoded, format, probe_w, probe_h)) {
        const int64_t probe_pixels = static_cast<int64_t>(probe_w)
                                   * static_cast<int64_t>(probe_h);
        if (probe_w > limits.max_width || probe_h > limits.max_height
            || probe_pixels > limits.max_pixels) {
            error = "avatar dimensions exceed the configured limits (pre-decode probe)";
            return false;
        }
    }

    // EXIF orientation 策略（P0）：Apply 时按 EXIF 标记自动旋转，与浏览器一致；
    // Ignore 时使用原始像素方向。
    // 注意：cv::IMREAD_UNCHANGED = -1（所有位置 1），其中包含
    // IMREAD_IGNORE_ORIENTATION(128)，会导致 EXIF 永远不被应用。因此 Apply
    // 策略必须用 ANYCOLOR|ANYDEPTH 显式组合，而非 IMREAD_UNCHANGED。
    int read_flags = cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH;
    if (limits.exif_orientation == ExifOrientationPolicy::Ignore) {
        read_flags |= cv::IMREAD_IGNORE_ORIENTATION;
    }
    // OpenCV's external-buffer Mat constructor takes a mutable pointer even
    // though imdecode treats this encoded byte view as read-only.
    const cv::Mat encoded_view(1, static_cast<int>(encoded.size()), CV_8UC1,
                               const_cast<uint8_t*>(encoded.data()));
    const cv::Mat decoded = cv::imdecode(encoded_view, read_flags);
    if (decoded.empty()) {
        error = "avatar JPEG/PNG decoding failed";
        return false;
    }

    const int64_t pixel_count = static_cast<int64_t>(decoded.cols)
                              * static_cast<int64_t>(decoded.rows);
    if (decoded.cols <= 0 || decoded.rows <= 0
        || decoded.cols > limits.max_width || decoded.rows > limits.max_height
        || pixel_count > limits.max_pixels) {
        error = "avatar dimensions exceed the configured limits";
        return false;
    }

    cv::Mat bgr;
    if (!NormalizeBGR(decoded, bgr, error)) return false;
    image.bgr = std::move(bgr);
    image.format = format;
    return true;
}

bool LoadAvatarImage(const std::string& path,
                     const AvatarUploadLimits& limits,
                     AvatarImage& image,
                     std::string& error) {
    image = AvatarImage{};
    error.clear();
    if (path.empty()) {
        error = "avatar path is empty";
        return false;
    }
    if (!IsValidLimits(limits)) {
        error = "invalid avatar upload limits";
        return false;
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "failed to open avatar file";
        return false;
    }
    const std::streamoff size = input.tellg();
    if (size <= 0) {
        error = "avatar file is empty";
        return false;
    }
    if (static_cast<uint64_t>(size) > limits.max_encoded_bytes) {
        error = "avatar file exceeds the encoded byte limit";
        return false;
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> encoded(static_cast<size_t>(size));
    if (!input.read(reinterpret_cast<char*>(encoded.data()),
                    static_cast<std::streamsize>(size))) {
        error = "failed to read avatar file";
        return false;
    }
    return DecodeAvatarUpload(encoded, {}, limits, image, error);
}

}  // namespace avatar
}  // namespace digital_human
