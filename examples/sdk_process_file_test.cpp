#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "digital_human_sdk.h"

namespace fs = std::filesystem;

namespace {

template <typename T>
void WriteLE(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

bool WriteTestWav(const fs::path& path, int sample_rate, double seconds) {
    const int64_t sample_count = static_cast<int64_t>(sample_rate * seconds);
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t data_size = static_cast<uint32_t>(
        sample_count * channels * bits_per_sample / 8);
    const uint32_t riff_size = 36 + data_size;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    out.write("RIFF", 4);
    WriteLE(out, riff_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    WriteLE<uint32_t>(out, 16);
    WriteLE<uint16_t>(out, 1);
    WriteLE(out, channels);
    WriteLE<uint32_t>(out, sample_rate);
    WriteLE(out, byte_rate);
    WriteLE(out, block_align);
    WriteLE(out, bits_per_sample);
    out.write("data", 4);
    WriteLE(out, data_size);

    constexpr double kPi = 3.14159265358979323846;
    for (int64_t i = 0; i < sample_count; ++i) {
        const double t = static_cast<double>(i) / sample_rate;
        const double envelope = 0.55 + 0.45 * std::sin(2.0 * kPi * 2.0 * t);
        const auto sample = static_cast<int16_t>(
            9000.0 * envelope * std::sin(2.0 * kPi * 220.0 * t));
        WriteLE(out, sample);
    }
    return static_cast<bool>(out);
}

}  // namespace

int main() {
    const fs::path project_dir(PROJECT_SOURCE_DIR);
    const fs::path lipsync_param =
        project_dir / "models" / "Wav2Lip-SD-GAN-opt.param";
    const fs::path lipsync_bin =
        project_dir / "models" / "Wav2Lip-SD-GAN-opt.bin";
    const fs::path face_dir = project_dir / "models" / "face";
    const fs::path image_path = project_dir / "assets" / "face.jpg";

    if (!fs::exists(lipsync_param) || !fs::exists(lipsync_bin)
        || !fs::exists(face_dir) || !fs::exists(image_path)) {
        std::cout << "[SKIP] ProcessFile 回归测试所需模型或图片不存在\n";
        return 0;
    }

    const fs::path wav_path = fs::temp_directory_path()
        / "digital_human_sdk_process_file_test.wav";
    if (!WriteTestWav(wav_path, 16000, 2.0)) {
        std::cerr << "[FAIL] 无法创建测试 WAV: " << wav_path << '\n';
        return 1;
    }

    digital_human::SDKConfig config;
    config.lipsync_model_dir = (project_dir / "models").string();
    config.face_model_dir = face_dir.string();
    config.enable_frame_pacing = false;
    config.audio_raw_queue_size = 2;
    config.mel_queue_size = 4;
    config.video_raw_queue_size = 2;
    config.face_queue_size = 2;
    config.infer_queue_size = 2;
    config.output_queue_size = 2;
    config.file_audio_lead_ms = 300;
    config.file_stall_timeout_ms = 20000;

    digital_human::DigitalHumanSDK sdk;
    const auto init = sdk.Init(config);
    if (init != digital_human::SDKError::OK) {
        std::cerr << "[FAIL] SDK 初始化失败: " << sdk.GetLastError() << '\n';
        fs::remove(wav_path);
        return 1;
    }

    int64_t frame_count = 0;
    int64_t previous_pts = -1;
    bool pts_monotonic = true;
    const auto started = std::chrono::steady_clock::now();
    const auto result = sdk.ProcessFile(
        wav_path.string(), image_path.string(),
        [&](const cv::Mat& frame, int64_t pts_ms) {
            if (frame.empty() || pts_ms < previous_pts) {
                pts_monotonic = false;
            }
            previous_pts = pts_ms;
            ++frame_count;
        });
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    fs::remove(wav_path);

    if (result != digital_human::SDKError::OK) {
        std::cerr << "[FAIL] ProcessFile 返回 "
                  << digital_human::SDKErrorToString(result)
                  << ": " << sdk.GetLastError() << '\n';
        return 1;
    }
    if (frame_count <= 0 || !pts_monotonic) {
        std::cerr << "[FAIL] 输出无效: frames=" << frame_count
                  << ", pts_monotonic=" << pts_monotonic << '\n';
        return 1;
    }

    std::cout << "[PASS] 小队列反压下 ProcessFile 正常结束: frames="
              << frame_count << ", elapsed_ms=" << elapsed_ms << '\n';
    return 0;
}
