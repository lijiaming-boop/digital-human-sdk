#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "avatar/avatar_image.h"
#include "dialog/conversation_session.h"

using namespace digital_human;
namespace fs = std::filesystem;

namespace {

bool Check(bool condition, const std::string& message) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << message << '\n';
    return condition;
}

std::vector<uint8_t> Encode(const cv::Mat& image, const std::string& extension) {
    std::vector<uint8_t> encoded;
    cv::imencode(extension, image, encoded);
    return encoded;
}

/// 在 JPEG 字节流的 SOI(FF D8) 之后插入一个最小 EXIF APP1 段，仅包含
/// Orientation 标签（tag 0x0112）。用于验证 EXIF orientation 策略：
/// orientation=6 表示相对原始像素顺时针旋转 90°，解码后宽高应互换。
/// EXIF 结构：APP1 marker + length + "Exif\0\0" + TIFF header + IFD0。
std::vector<uint8_t> InjectExifOrientation(const std::vector<uint8_t>& jpeg,
                                            uint16_t orientation) {
    if (jpeg.size() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) return jpeg;
    std::vector<uint8_t> out;
    out.reserve(jpeg.size() + 64);
    out.push_back(0xFF);
    out.push_back(0xD8);  // SOI

    // APP1 marker + length（JPEG APP 段长度包含 2 字节长度字段自身，不含 marker）。
    // payload = 6(Exif\0\0) + 8(TIFF header) + 2(IFD count) + 12(one entry)
    // + 4(next IFD) = 32，length = 32 + 2 = 34。
    out.push_back(0xFF);
    out.push_back(0xE1);
    out.push_back(0x00);
    out.push_back(0x22);  // length = 34
    // "Exif\0\0"
    out.insert(out.end(), {'E', 'x', 'i', 'f', 0x00, 0x00});
    // TIFF header (little-endian "II")
    out.insert(out.end(), {'I', 'I', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00});
    // IFD0: 1 entry
    out.push_back(0x01);
    out.push_back(0x00);
    // Entry: tag=0x0112(Orientation), type=0x0003(SHORT), count=1, value=orientation
    out.push_back(0x12);
    out.push_back(0x01);
    out.push_back(0x03);
    out.push_back(0x00);
    out.push_back(0x01);
    out.push_back(0x00);
    out.push_back(0x00);
    out.push_back(0x00);
    out.push_back(static_cast<uint8_t>(orientation & 0xFF));
    out.push_back(static_cast<uint8_t>(orientation >> 8));
    out.push_back(0x00);
    out.push_back(0x00);
    // Next IFD offset = 0
    out.insert(out.end(), {0x00, 0x00, 0x00, 0x00});

    // 剩余 JPEG 字节（跳过已写入的 SOI）
    out.insert(out.end(), jpeg.begin() + 2, jpeg.end());
    return out;
}

class FixedTextClient final : public dialog::ITextGenerationClient {
public:
    bool Generate(const dialog::GenerateRequest&,
                  const dialog::TextDeltaCallback& on_delta,
                  const dialog::CancelCheck& cancelled,
                  std::string&) override {
        if (cancelled && cancelled()) return false;
        on_delta("ok.");
        return true;
    }
};

class DelayedTTSClient final : public tts::ITTSClient {
public:
    bool Synthesize(const std::string&,
                    const tts::PCMCallback& on_audio,
                    const tts::CancelCheck& cancelled,
                    std::string&) override {
        for (int index = 0; index < 4; ++index) {
            if (cancelled && cancelled()) return false;
            tts::PCMChunk chunk;
            chunk.sample_rate = 16000;
            chunk.channels = 1;
            chunk.samples.assign(1600, 0.01F);
            if (!on_audio(std::move(chunk))) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return true;
    }
};

class RecordingSink final : public dialog::IDigitalHumanSink {
public:
    bool PushAudio(const std::vector<float>&,
                   int64_t,
                   std::string&) override {
        return true;
    }

    bool PushVideo(const cv::Mat& frame,
                   int64_t,
                   std::string&) override {
        std::lock_guard<std::mutex> lock(mutex_);
        colors_.push_back(frame.at<cv::Vec3b>(0, 0));
        return true;
    }

    void Finish() override {}

    std::vector<cv::Vec3b> Colors() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return colors_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<cv::Vec3b> colors_;
};

bool HasColor(const std::vector<cv::Vec3b>& colors, const cv::Vec3b& expected) {
    for (const cv::Vec3b& color : colors) {
        if (color == expected) return true;
    }
    return false;
}

}  // namespace

int main() {
    bool ok = true;
    avatar::AvatarUploadLimits limits;

    avatar::AvatarImage decoded;
    std::string error;
    ok &= Check(!avatar::DecodeAvatarUpload({}, "image/jpeg", limits,
                                             decoded, error),
                "empty upload is rejected");
    ok &= Check(!error.empty(), "empty upload reports an error");

    error.clear();
    std::vector<uint8_t> garbage{0x01, 0x02, 0x03, 0x04};
    ok &= Check(!avatar::DecodeAvatarUpload(garbage, "image/jpeg", limits,
                                             decoded, error),
                "invalid JPEG bytes are rejected");

    cv::Mat gray(32, 48, CV_8UC1, cv::Scalar(127));
    const std::vector<uint8_t> png = Encode(gray, ".png");
    error.clear();
    ok &= Check(avatar::DecodeAvatarUpload(png, "image/png", limits,
                                            decoded, error),
                "PNG upload is decoded");
    ok &= Check(decoded.bgr.type() == CV_8UC3
                    && decoded.bgr.cols == 48 && decoded.bgr.rows == 32,
                "uploaded image is normalized to BGR");
    ok &= Check(decoded.format == avatar::AvatarImageFormat::PNG,
                "PNG format is detected");

    error.clear();
    ok &= Check(!avatar::DecodeAvatarUpload(png, "image/jpeg", limits,
                                             decoded, error),
                "declared content type must match encoded image");

    avatar::AvatarUploadLimits byte_limits = limits;
    byte_limits.max_encoded_bytes = png.size() - 1;
    error.clear();
    ok &= Check(!avatar::DecodeAvatarUpload(png, "image/png", byte_limits,
                                             decoded, error),
                "encoded byte limit is enforced");

    avatar::AvatarUploadLimits dimension_limits = limits;
    dimension_limits.max_width = 40;
    error.clear();
    ok &= Check(!avatar::DecodeAvatarUpload(png, "image/png", dimension_limits,
                                             decoded, error),
                "decoded dimension limit is enforced");

    cv::Mat bgra(20, 24, CV_8UC4, cv::Scalar(10, 20, 30, 128));
    const std::vector<uint8_t> alpha_png = Encode(bgra, ".png");
    error.clear();
    ok &= Check(avatar::DecodeAvatarUpload(alpha_png, "", limits,
                                            decoded, error),
                "content type can be auto-detected");
    ok &= Check(decoded.bgr.type() == CV_8UC3
                    && decoded.bgr.at<cv::Vec3b>(0, 0) == cv::Vec3b(10, 20, 30),
                "BGRA upload is normalized without aliasing input bytes");

    // EXIF orientation 策略（P0 资源限制）：构造一个 20(宽)x40(高) 的非对称
    // JPEG，注入 EXIF orientation=6（顺时针 90°）。Apply 策略应将帧旋转为
    // 40x20，Ignore 策略应保持原始 20x40 像素方向。
    {
        cv::Mat tall(40, 20, CV_8UC3, cv::Scalar(30, 60, 90));
        const std::vector<uint8_t> raw_jpeg = Encode(tall, ".jpg");
        const std::vector<uint8_t> exif_jpeg =
            InjectExifOrientation(raw_jpeg, 6);

        avatar::AvatarUploadLimits apply_limits = limits;
        apply_limits.exif_orientation = avatar::ExifOrientationPolicy::Apply;
        error.clear();
        ok &= Check(avatar::DecodeAvatarUpload(exif_jpeg, "image/jpeg",
                                                apply_limits, decoded, error),
                    "EXIF Apply policy decodes JPEG with orientation marker");
        ok &= Check(decoded.bgr.cols == 40 && decoded.bgr.rows == 20,
                    "EXIF Apply policy rotates frame to 40x20 (orientation=6)");

        avatar::AvatarUploadLimits ignore_limits = limits;
        ignore_limits.exif_orientation = avatar::ExifOrientationPolicy::Ignore;
        error.clear();
        ok &= Check(avatar::DecodeAvatarUpload(exif_jpeg, "image/jpeg",
                                                ignore_limits, decoded, error),
                    "EXIF Ignore policy decodes JPEG with orientation marker");
        ok &= Check(decoded.bgr.cols == 20 && decoded.bgr.rows == 40,
                    "EXIF Ignore policy keeps original 20x40 pixel layout");
    }

    const fs::path temp_path = fs::temp_directory_path()
        / "digital_human_avatar_upload_test.png";
    {
        std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(png.data()),
                     static_cast<std::streamsize>(png.size()));
    }
    error.clear();
    ok &= Check(avatar::LoadAvatarImage(temp_path.string(), limits,
                                         decoded, error),
                "uploaded image can be loaded from a persisted file");
    std::error_code ignored;
    fs::remove(temp_path, ignored);

    FixedTextClient text_client;
    DelayedTTSClient tts_client;
    RecordingSink sink;
    dialog::ConversationSession session(text_client, tts_client, sink);
    dialog::ConversationConfig config;
    config.min_tts_clause_chars = 1;
    config.mel_lookahead_ms = 0;
    config.reply_tail_silence_ms = 0;

    cv::Mat first_avatar(16, 16, CV_8UC3, cv::Scalar(1, 2, 3));
    ok &= Check(session.Start(config, first_avatar),
                "session starts with first uploaded avatar");
    first_avatar.setTo(cv::Scalar(99, 99, 99));
    ok &= Check(session.SubmitUserText("turn one") != 0
                    && session.WaitUntilIdle(std::chrono::seconds(3)),
                "first avatar turn completes");

    cv::Mat second_avatar(16, 16, CV_8UC3, cv::Scalar(7, 8, 9));
    ok &= Check(session.UpdateAvatar(second_avatar),
                "running session accepts a new uploaded avatar");
    second_avatar.setTo(cv::Scalar(88, 88, 88));
    ok &= Check(session.SubmitUserText("turn two") != 0
                    && session.WaitUntilIdle(std::chrono::seconds(3)),
                "updated avatar turn completes");
    ok &= Check(!session.UpdateAvatar(cv::Mat()),
                "empty avatar update is rejected");
    session.Stop(true);

    const std::vector<cv::Vec3b> colors = sink.Colors();
    ok &= Check(HasColor(colors, cv::Vec3b(1, 2, 3)),
                "session clones the initial avatar");
    ok &= Check(HasColor(colors, cv::Vec3b(7, 8, 9)),
                "subsequent frames use the updated avatar");
    ok &= Check(!HasColor(colors, cv::Vec3b(99, 99, 99))
                    && !HasColor(colors, cv::Vec3b(88, 88, 88)),
                "session never aliases caller-owned avatar memory");

    return ok ? 0 : 1;
}
