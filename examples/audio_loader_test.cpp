#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <filesystem>
#include "audio/audio_loader.h"

namespace fs = std::filesystem;
using namespace digital_human::audio;

constexpr double PI = 3.14159265358979323846;

struct WavHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t fileSize;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmtSize = 16;
    uint16_t audioFormat = 1;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize;
};

void writeWav(const std::string& path, const std::vector<int16_t>& samples,
              int sampleRate, int channels) {
    std::ofstream f(path, std::ios::binary);

    WavHeader hdr;
    hdr.numChannels = static_cast<uint16_t>(channels);
    hdr.sampleRate = static_cast<uint32_t>(sampleRate);
    hdr.bitsPerSample = 16;
    hdr.byteRate = hdr.sampleRate * hdr.numChannels * (hdr.bitsPerSample / 8);
    hdr.blockAlign = hdr.numChannels * (hdr.bitsPerSample / 8);
    hdr.dataSize = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    hdr.fileSize = 36 + hdr.dataSize;

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(samples.data()), hdr.dataSize);
}

std::vector<int16_t> generateSine(float freq, float durationSec,
                                   int sampleRate, int channels, float amplitude) {
    int totalSamples = static_cast<int>(sampleRate * durationSec);
    std::vector<int16_t> samples(totalSamples * channels);

    for (int i = 0; i < totalSamples; i++) {
        double t = static_cast<double>(i) / sampleRate;
        int16_t val = static_cast<int16_t>(amplitude * 32767.0 * std::sin(2.0 * PI * freq * t));
        for (int ch = 0; ch < channels; ch++) {
            samples[i * channels + ch] = val;
        }
    }
    return samples;
}

int main() {
    std::cout << "========== AudioLoader Test ==========" << std::endl;

    std::string testWav = "test_audio.wav";
    std::string invalidPath = "non_existent_audio.wav";

    // Generate a test WAV: 440Hz sine, 1.5s, 44100Hz, stereo, 16-bit
    std::cout << "\nGenerating test WAV (440Hz, 1.5s, 44100Hz, stereo, 16-bit)..." << std::endl;
    auto rawSamples = generateSine(440.0f, 1.5f, 44100, 2, 0.8f);
    writeWav(testWav, rawSamples, 44100, 2);
    std::cout << "  WAV file created: " << testWav << " (" << rawSamples.size() * 2 << " bytes)" << std::endl;

    AudioLoader loader;

    // ==========================================
    // Test 1: Load valid audio file
    // ==========================================
    std::cout << "\n[Test 1] Load valid audio file..." << std::endl;
    try {
        AudioData audio = loader.load(testWav);

        std::cout << "  [OK] Sample rate: " << audio.sampleRate << " Hz (expected 16000)" << std::endl;
        std::cout << "  [OK] Channels: " << audio.channels << " (expected 1)" << std::endl;
        std::cout << "  [OK] Duration: " << audio.duration << " s" << std::endl;
        std::cout << "  [OK] Total samples: " << audio.samples.size() << std::endl;

        bool srOk = (audio.sampleRate == 16000);
        bool chOk = (audio.channels == 1);
        bool dataOk = (!audio.samples.empty());

        std::cout << (srOk ? "  [PASS]" : "  [FAIL]") << " Sample rate check" << std::endl;
        std::cout << (chOk ? "  [PASS]" : "  [FAIL]") << " Channel check" << std::endl;
        std::cout << (dataOk ? "  [PASS]" : "  [FAIL]") << " Data not empty" << std::endl;

    } catch (const AudioLoaderException& e) {
        std::cout << "  [FAIL] Unexpected exception: " << e.what() << std::endl;
    }

    // ==========================================
    // Test 2: Verify normalization (peak ≈ 1.0)
    // ==========================================
    std::cout << "\n[Test 2] Verify normalization..." << std::endl;
    try {
        AudioData audio = loader.load(testWav);

        float maxAbs = 0.0f;
        for (float s : audio.samples) {
            float absVal = std::fabs(s);
            if (absVal > maxAbs) maxAbs = absVal;
        }

        std::cout << "  Max absolute sample value: " << maxAbs << std::endl;
        bool normOk = (maxAbs >= 0.99f && maxAbs <= 1.01f);
        std::cout << (normOk ? "  [PASS]" : "  [FAIL]") << " Peak normalization (expected ~1.0)" << std::endl;

        float sumSq = 0.0f;
        for (float s : audio.samples) {
            sumSq += s * s;
        }
        float rms = std::sqrt(sumSq / audio.samples.size());
        std::cout << "  RMS value: " << rms << std::endl;

    } catch (const AudioLoaderException& e) {
        std::cout << "  [FAIL] Unexpected exception: " << e.what() << std::endl;
    }

    // ==========================================
    // Test 3: Load non-existent file (error path)
    // ==========================================
    std::cout << "\n[Test 3] Load non-existent file..." << std::endl;
    try {
        loader.load(invalidPath);
        std::cout << "  [FAIL] Should have thrown AudioLoaderException" << std::endl;
    } catch (const AudioLoaderException& e) {
        std::cout << "  [PASS] Caught expected exception: " << e.what() << std::endl;
    }

    // ==========================================
    // Test 4: Resampling correctness (output ~= 16000Hz)
    // ==========================================
    std::cout << "\n[Test 4] Resampling correctness..." << std::endl;
    try {
        AudioData audio = loader.load(testWav);
        int expectedSamples = static_cast<int>(1.5 * 16000);
        int actualSamples = static_cast<int>(audio.samples.size());
        std::cout << "  Expected ~" << expectedSamples << " samples for 1.5s @ 16000Hz" << std::endl;
        std::cout << "  Actual: " << actualSamples << " samples" << std::endl;
        double ratio = static_cast<double>(actualSamples) / expectedSamples;
        bool resampleOk = (ratio > 0.9 && ratio < 1.3);
        std::cout << (resampleOk ? "  [PASS]" : "  [FAIL]") << " Sample count within expected range" << std::endl;
    } catch (const AudioLoaderException& e) {
        std::cout << "  [FAIL] Unexpected exception: " << e.what() << std::endl;
    }

    // Cleanup
    if (fs::exists(testWav)) fs::remove(testWav);

    std::cout << "\n========== Test Complete ==========" << std::endl;
    return 0;
}
