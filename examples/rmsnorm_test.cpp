#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "audio/audio_rms_normalize.h"

using namespace digital_human::audio;

int main() {
    std::cout << "========== RMSNormalize Test ==========" << std::endl;

    const float eps = 1e-4f;

    // ==========================================
    // Test 1: Target RMS achieved
    // ==========================================
    std::cout << "\n[Test 1] Target RMS..." << std::endl;
    {
        RMSNormalize norm(0.1f);  // target RMS = 0.1

        // Generate signal with known RMS
        std::vector<float> pcm(16000);
        for (size_t i = 0; i < pcm.size(); i++) {
            pcm[i] = 0.05f * std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f);
        }

        auto result = norm.process(pcm);

        // Calculate actual RMS
        float sumSq = 0.0f;
        for (float x : result) {
            sumSq += x * x;
        }
        float actualRMS = std::sqrt(sumSq / result.size());

        // For a sine wave 0.05 amplitude: RMS = 0.05/sqrt(2) ≈ 0.0354
        // After normalization to 0.1: should be close to 0.1
        bool ok = std::fabs(actualRMS - 0.1f) < 0.01f;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " RMS = " << actualRMS << " (target 0.1)" << std::endl;
    }

    // ==========================================
    // Test 2: -25dB FS target (default)
    // ==========================================
    std::cout << "\n[Test 2] -25dB FS default..." << std::endl;
    {
        RMSNormalize norm;  // default -25dB FS

        float targetRMS = std::pow(10.0f, -25.0f / 20.0f);  // ≈ 0.0562

        std::vector<float> pcm(16000);
        for (size_t i = 0; i < pcm.size(); i++) {
            pcm[i] = 0.1f * std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f);
        }

        auto result = norm.process(pcm);

        float sumSq = 0.0f;
        for (float x : result) {
            sumSq += x * x;
        }
        float actualRMS = std::sqrt(sumSq / result.size());

        bool ok = std::fabs(actualRMS - targetRMS) < 0.01f;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " RMS = " << actualRMS << " (target " << targetRMS << ")" << std::endl;
    }

    // ==========================================
    // Test 3: Empty input
    // ==========================================
    std::cout << "\n[Test 3] Empty input..." << std::endl;
    {
        RMSNormalize norm;
        std::vector<float> empty;
        auto result = norm.process(empty);

        bool ok = result.empty();
        std::cout << (ok ? "  [PASS]" : "  [FAIL]") << " Returns empty" << std::endl;
    }

    // ==========================================
    // Test 4: Silent signal (all zeros) → unchanged
    // ==========================================
    std::cout << "\n[Test 4] Silent signal..." << std::endl;
    {
        RMSNormalize norm(0.1f);
        std::vector<float> silence(100, 0.0f);
        auto result = norm.process(silence);

        bool allZero = true;
        for (float x : result) {
            if (std::fabs(x) > 1e-10f) allZero = false;
        }
        std::cout << (allZero ? "  [PASS]" : "  [FAIL]")
                  << " Silence unchanged" << std::endl;
    }

    // ==========================================
    // Test 5: Scale factor correctness
    // ==========================================
    std::cout << "\n[Test 5] Scale factor..." << std::endl;
    {
        RMSNormalize norm(1.0f);

        // Signal with RMS = 0.5, should be scaled by 1.0/0.5 = 2.0
        std::vector<float> pcm(1000, 0.5f);
        auto result = norm.process(pcm);

        float expectedRMS = 1.0f;
        float sumSq = 0.0f;
        for (float x : result) {
            sumSq += x * x;
        }
        float actualRMS = std::sqrt(sumSq / result.size());

        // Each sample was 0.5, scaled by 2 → 1.0
        bool valOk = std::fabs(result[0] - 1.0f) < eps;
        bool rmsOk = std::fabs(actualRMS - expectedRMS) < eps;

        std::cout << (valOk ? "  [PASS]" : "  [FAIL]") << " Sample scaled to 1.0" << std::endl;
        std::cout << (rmsOk ? "  [PASS]" : "  [FAIL]") << " RMS = 1.0" << std::endl;
    }

    // ==========================================
    // Test 6: Already at target → minimal change
    // ==========================================
    std::cout << "\n[Test 6] Already at target..." << std::endl;
    {
        float target = 0.1f;
        RMSNormalize norm(target);

        // Create signal with RMS exactly 0.1
        std::vector<float> pcm(1000, 0.1f);
        auto result = norm.process(pcm);

        float sumSq = 0.0f;
        for (float x : result) {
            sumSq += x * x;
        }
        float actualRMS = std::sqrt(sumSq / result.size());

        bool ok = std::fabs(actualRMS - target) < 0.001f;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " RMS unchanged (" << actualRMS << ")" << std::endl;
    }

    std::cout << "\n========== Test Complete ==========" << std::endl;
    return 0;
}
