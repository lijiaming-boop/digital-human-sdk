/**
 * @file offline_audio_test.cpp
 * @brief 离线音频测试 — 加载 WAV 文件并运行完整音频处理流水线
 *
 * 用法: ./bin/offline_audio_test <path/to/audio.wav>
 *
 * 处理流程:
 *   WAV → AudioLoader → NoiseReduction → AudioFramer
 *   → VAD → PreEmphasis → RMSNormalize → MelFeatureExtract
 *   → CMVN → 统计输出
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdint>

#include "audio/audio_loader.h"
#include "audio/audio_noise_reduction.h"
#include "audio/audio_framer.h"
#include "audio/audio_vad.h"
#include "audio/audio_preemphasis.h"
#include "audio/audio_rms_normalize.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_cmvn.h"

using namespace digital_human::audio;

struct AudioStats {
    double duration_ms;
    int    sample_rate;
    int    channels;
    int    samples;
    int    frames;
    int    voiced_frames;
    int    mel_rows;
    int    mel_cols;
    double load_ms;
    double process_ms;
};

static AudioStats processAudio(const std::vector<float>& pcm, int sr, int ch) {
    AudioStats s;
    s.samples     = static_cast<int>(pcm.size());
    s.sample_rate = sr;
    s.channels    = ch;
    s.duration_ms = static_cast<double>(pcm.size()) / ch / sr * 1000.0;

    auto t0 = std::chrono::steady_clock::now();

    // 1. 降噪
    NoiseReduction nr(10, 0.02f);
    auto denoised = nr.process(pcm, sr);

    // 2. RMS 归一化
    RMSNormalize rms_norm(0.056f);
    auto normalized = rms_norm.process(denoised);

    // 3. 预加重
    PreEmphasis pe(0.97f);
    auto emphasized = pe.process(normalized);

    // 4. 分帧
    AudioFramer framer;
    FrameConfig fcfg{400, 160};
    auto frames = framer.frame(emphasized, fcfg);
    s.frames = static_cast<int>(frames.size());

    // 5. VAD
    VoiceActivityDetector vad(0.01f, 0.0f, 0.5f, 3);
    auto voiced = vad.filter(frames);
    s.voiced_frames = static_cast<int>(voiced.size());

    // 6. Mel 频谱
    MelFeatureExtract mel;
    MelConfig mcfg{512, 80, sr, 0.0f, 8000.0f};
    auto mel_spec = mel.extract(voiced.empty() ? frames : voiced, mcfg);
    s.mel_rows = mel_spec.rows;
    s.mel_cols = mel_spec.cols;

    // 7. CMVN
    CMVN cmvn;
    auto feat = cmvn.process(mel_spec);

    s.process_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    return s;
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << "  离线音频处理测试" << std::endl;
    std::cout << "==============================================" << std::endl;

    // 确定音频文件路径
    std::string wav_path;
    if (argc > 1) {
        wav_path = argv[1];
    } else {
        // 默认使用 assets 中的测试音频
        wav_path = ASSETS_DIR "/test_16k_mono.wav";
    }

    // 尝试 WAV 加载，失败则回退到 Raw PCM
    std::cout << "\n[加载] " << wav_path << std::endl;

    std::vector<float> pcm_data;
    int sample_rate = 16000;
    int channels = 1;
    double duration_s = 0.0;

    auto t0 = std::chrono::steady_clock::now();

    // 尝试 AudioLoader (FFmpeg)
    AudioLoader loader;
    AudioData data;
    bool loaded = false;
    try {
        data = loader.load(wav_path);
        loaded = true;
    } catch (const std::exception& e) {
        std::cout << "  [WARN] AudioLoader 失败: " << e.what() << std::endl;
        std::cout << "  [INFO] 尝试加载 raw PCM 回退..." << std::endl;
    }

    if (loaded) {
        pcm_data   = std::move(data.samples);
        sample_rate = data.sampleRate;
        channels   = data.channels;
        duration_s = data.duration;
    } else {
        // 回退: 加载 raw PCM (s16le, 16kHz mono)
        std::string raw_path = wav_path;
        // 将 .wav 替换为 .raw
        auto pos = raw_path.rfind(".wav");
        if (pos != std::string::npos) {
            raw_path.replace(pos, 4, ".raw");
        } else {
            raw_path += ".raw";
        }

        std::ifstream raw_file(raw_path, std::ios::binary);
        if (!raw_file) {
            // 尝试 assets 目录下的 .raw
            raw_path = ASSETS_DIR "/test_16k_mono.raw";
            raw_file.open(raw_path, std::ios::binary);
        }

        if (!raw_file) {
            std::cerr << "[FAIL] 无法打开 raw PCM 文件" << std::endl;
            return 1;
        }

        raw_file.seekg(0, std::ios::end);
        size_t raw_size = static_cast<size_t>(raw_file.tellg());
        raw_file.seekg(0, std::ios::beg);

        std::vector<int16_t> raw_samples(raw_size / 2);
        raw_file.read(reinterpret_cast<char*>(raw_samples.data()),
                      static_cast<std::streamsize>(raw_size));

        // s16le → float
        pcm_data.resize(raw_samples.size());
        for (size_t i = 0; i < raw_samples.size(); ++i) {
            pcm_data[i] = static_cast<float>(raw_samples[i]) / 32768.0f;
        }

        sample_rate = 16000;
        channels = 1;
        duration_s = static_cast<double>(pcm_data.size()) / sample_rate;

        std::cout << "  [INFO] Raw PCM 加载成功: " << pcm_data.size()
                  << " samples, " << duration_s << "s" << std::endl;
    }

    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    if (pcm_data.empty()) {
        std::cerr << "[FAIL] 空音频数据" << std::endl;
        return 1;
    }

    std::cout << "  duration: " << duration_s << "s" << std::endl;
    std::cout << "  sample_rate: " << sample_rate << " Hz" << std::endl;
    std::cout << "  channels: " << channels << std::endl;
    std::cout << "  samples: " << pcm_data.size() << std::endl;
    std::cout << "  load_time: " << std::fixed << std::setprecision(2)
              << load_ms << " ms" << std::endl;

    // 处理
    std::cout << "\n[处理] 音频流水线" << std::endl;
    AudioStats stats = processAudio(pcm_data, sample_rate, channels);
    stats.load_ms = load_ms;

    // 输出统计
    std::cout << "\n==============================================" << std::endl;
    std::cout << "  处理统计" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  输入:" << std::endl;
    std::cout << "    时长:  " << std::fixed << std::setprecision(1)
              << stats.duration_ms << " ms" << std::endl;
    std::cout << "    采样率: " << stats.sample_rate << " Hz" << std::endl;
    std::cout << "    声道数: " << stats.channels << std::endl;
    std::cout << "    样本数: " << stats.samples << std::endl;

    std::cout << "  音频处理:" << std::endl;
    std::cout << "    分帧:  " << stats.frames << " 帧"
              << " (frame=400, hop=160)" << std::endl;
    std::cout << "    VAD:   " << stats.voiced_frames << " 有声帧"
              << " (过滤 " << (stats.frames - stats.voiced_frames) << " 帧)"
              << std::endl;
    std::cout << "    Mel:   " << stats.mel_rows << " × "
              << stats.mel_cols << " (T × mel_bins)" << std::endl;

    std::cout << "  性能:" << std::endl;
    std::cout << "    加载:  " << stats.load_ms << " ms" << std::endl;
    std::cout << "    处理:  " << std::fixed << std::setprecision(2)
              << stats.process_ms << " ms" << std::endl;
    double ratio = stats.duration_ms / stats.process_ms;
    std::cout << "    实时比: " << std::fixed << std::setprecision(1)
              << ratio << "x 实时" << std::endl;

    // 结果判定
    bool pass = true;
    if (stats.frames <= 0) {
        std::cout << "\n  [FAIL] 无输出帧" << std::endl;
        pass = false;
    } else if (stats.mel_rows <= 0 || stats.mel_cols <= 0) {
        std::cout << "\n  [FAIL] Mel 特征无效" << std::endl;
        pass = false;
    } else if (ratio < 0.5) {
        std::cout << "\n  [WARN] 处理速度低于 0.5x 实时" << std::endl;
    } else {
        std::cout << "\n  [PASS] 音频处理流水线正常" << std::endl;
    }

    std::cout << "\n==============================================" << std::endl;
    return pass ? 0 : 1;
}
