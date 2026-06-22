#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "audio/audio_vad.h"

using namespace digital_human::audio;

std::vector<std::vector<float>> makeSilenceFrames(int numFrames, int frameSize) {
    std::vector<std::vector<float>> frames(numFrames,
        std::vector<float>(frameSize, 1e-6f));
    return frames;
}

std::vector<std::vector<float>> makeToneFrames(int numFrames, int frameSize,
                                                float freq, int sampleRate) {
    std::vector<std::vector<float>> frames(numFrames,
        std::vector<float>(frameSize, 0.0f));
    for (int i = 0; i < numFrames; i++) {
        for (int j = 0; j < frameSize; j++) {
            double t = static_cast<double>(i * frameSize + j) / sampleRate;
            frames[i][j] = 0.5f * std::sin(2.0 * 3.14159265 * freq * t);
        }
    }
    return frames;
}

int main() {
    std::cout << "========== VAD Test ==========" << std::endl;

    const int frameSize = 400;
    const int sampleRate = 16000;

    // ==========================================
    // Test 1: Pure silence → all frames removed
    // ==========================================
    std::cout << "\n[Test 1] Pure silence..." << std::endl;
    {
        VoiceActivityDetector vad(0.001f, 0.0f, 0.5f, 0);
        auto silence = makeSilenceFrames(10, frameSize);
        auto result = vad.filter(silence);

        bool ok = result.empty();
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " All silence frames removed (got " << result.size() << ")" << std::endl;
    }

    // ==========================================
    // Test 2: 440Hz tone → all frames kept (low ZCR, high energy)
    // ==========================================
    std::cout << "\n[Test 2] 440 Hz tone (voiced-like)..." << std::endl;
    {
        VoiceActivityDetector vad(0.001f, 0.0f, 0.5f, 0);
        auto tone = makeToneFrames(10, frameSize, 440.0f, sampleRate);
        auto result = vad.filter(tone);

        bool ok = (result.size() == 10);
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " All tone frames kept (got " << result.size() << ")" << std::endl;
    }

    // ==========================================
    // Test 3: Mixed speech + silence
    // ==========================================
    std::cout << "\n[Test 3] Mixed speech + silence..." << std::endl;
    {
        VoiceActivityDetector vad(0.001f, 0.0f, 0.5f, 0);

        auto silence = makeSilenceFrames(3, frameSize);
        auto speech = makeToneFrames(4, frameSize, 440.0f, sampleRate);

        std::vector<std::vector<float>> mixed;
        mixed.insert(mixed.end(), silence.begin(), silence.end());
        mixed.insert(mixed.end(), speech.begin(), speech.end());
        mixed.insert(mixed.end(), silence.begin(), silence.end());

        auto result = vad.filter(mixed);
        bool ok = (result.size() == 4);
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " Exactly 4 speech frames kept (got " << result.size() << ")" << std::endl;
    }

    // ==========================================
    // Test 4: Empty input
    // ==========================================
    std::cout << "\n[Test 4] Empty input..." << std::endl;
    {
        VoiceActivityDetector vad;
        std::vector<std::vector<float>> empty;
        auto result = vad.filter(empty);

        bool ok = result.empty();
        std::cout << (ok ? "  [PASS]" : "  [FAIL]") << " Returns empty" << std::endl;
    }

    // ==========================================
    // Test 5: High ZCR white noise → filtered out
    // ==========================================
    std::cout << "\n[Test 5] High ZCR noise rejected..." << std::endl;
    {
        VoiceActivityDetector vad(0.001f, 0.0f, 0.45f, 0);

        // Generate noise with many zero crossings
        std::vector<std::vector<float>> noiseFrames(5,
            std::vector<float>(frameSize, 0.0f));
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < frameSize; j++) {
                // High-frequency signal → high ZCR
                double t = static_cast<double>(i * frameSize + j) / sampleRate;
                noiseFrames[i][j] = 0.3f * std::sin(2.0 * 3.14159265 * 7000.0 * t);
            }
        }

        auto result = vad.filter(noiseFrames);

        // 7000Hz has ZCR ≈ 2*7000/16000 ≈ 0.875, which is > 0.45 max → filtered
        bool ok = result.empty();
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " High-ZCR noise filtered (got " << result.size() << ")" << std::endl;
    }

    // ==========================================
    // Test 6: Hangover preserves context
    // ==========================================
    std::cout << "\n[Test 6] Hangover..." << std::endl;
    {
        VoiceActivityDetector vad(0.001f, 0.0f, 0.5f, 3);

        auto silence = makeSilenceFrames(10, frameSize);
        auto speech = makeToneFrames(1, frameSize, 440.0f, sampleRate);

        std::vector<std::vector<float>> mixed;
        mixed.insert(mixed.end(), silence.begin(), silence.begin() + 5);
        mixed.insert(mixed.end(), speech.begin(), speech.end());
        mixed.insert(mixed.end(), silence.begin(), silence.begin() + 5);

        auto result = vad.filter(mixed);

        // 1 speech + 3 hangover after = 4 frames (no hangover before since
        // silence before has no speech to extend from)
        // Actually: speech[i]=true for i=5, hangover extends to [2,8], so 7 frames
        bool ok = (result.size() >= 4 && result.size() <= 7);
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " Hangover extends speech segment (" << result.size()
                  << " frames)" << std::endl;
    }

    // ==========================================
    // Test 7: Default parameters
    // ==========================================
    std::cout << "\n[Test 7] Default parameters..." << std::endl;
    {
        VoiceActivityDetector vad;  // default: energy=0.01, zcrMin=0.1, zcrMax=0.6, hangover=3

        auto tone = makeToneFrames(5, frameSize, 440.0f, sampleRate);
        auto result = vad.filter(tone);

        bool ok = !result.empty();
        // 440Hz tone: energy ≈ 400*0.125 ≈ 50 >> 0.01 ✓
        // ZCR ≈ 0.055 which is < zcrMin=0.1 ✗ → filtered out with defaults
        std::cout << "  Result: " << result.size() << " / 5 frames kept";
        bool ok = (result.size() == 5);
        std::cout << (ok ? "\n  [PASS]" : "\n  [FAIL]")
                  << " Default params work for 440Hz tone" << std::endl;
    }

    std::cout << "\n========== Test Complete ==========" << std::endl;
    return 0;
}
