#include <iostream>
#include <vector>
#include <cmath>
#include "audio/audio_noise_reduction.h"

using namespace digital_human::audio;

std::vector<float> generateSine(float freq, float durationSec, int sampleRate) {
    int totalSamples = static_cast<int>(sampleRate * durationSec);
    std::vector<float> samples(totalSamples);
    for (int i = 0; i < totalSamples; i++) {
        double t = static_cast<double>(i) / sampleRate;
        samples[i] = static_cast<float>(std::sin(2.0 * 3.14159265 * freq * t));
    }
    return samples;
}

int main() {
    std::cout << "========== NoiseReduction Test ==========" << std::endl;

    const int sampleRate = 16000;
    int failures = 0;

    // ==========================================
    // Test 1: Silent input → unchanged
    // ==========================================
    std::cout << "\n[Test 1] Silent input..." << std::endl;
    {
        NoiseReduction nr(10, 0.02f);
        std::vector<float> silence(8000, 0.0f);
        auto result = nr.process(silence, sampleRate);

        // Result should be silent or near-silent
        float maxAbs = 0.0f;
        for (float x : result) maxAbs = std::max(maxAbs, std::fabs(x));

        bool ok = (maxAbs < 0.01f);
        failures += !ok;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " Silent remains silent (max=" << maxAbs << ")" << std::endl;
    }

    // ==========================================
    // Test 2: Clean tone passes through largely unchanged
    // ==========================================
    std::cout << "\n[Test 2] Clean tone..." << std::endl;
    {
        // Low oversubtraction factor → clean signal passes
        NoiseReduction nr(10, 0.01f);

        auto clean = generateSine(440.0f, 0.5f, sampleRate);
        auto result = nr.process(clean, sampleRate);

        // Clean tone should have similar RMS
        float originalRMS = 0.0f, resultRMS = 0.0f;
        for (size_t i = 0; i < clean.size(); i++) {
            originalRMS += clean[i] * clean[i];
            resultRMS += result[i] * result[i];
        }
        originalRMS = std::sqrt(originalRMS / clean.size());
        resultRMS = std::sqrt(resultRMS / result.size());

        float ratio = resultRMS / std::max(originalRMS, 1e-10f);
        bool ok = (ratio > 0.7f && ratio < 1.3f);
        failures += !ok;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " RMS ratio = " << ratio << " (expected ~1.0)" << std::endl;
    }

    // ==========================================
    // Test 3: Noisy signal → energy reduced
    // ==========================================
    std::cout << "\n[Test 3] Noisy signal energy reduced..." << std::endl;
    {
        NoiseReduction nr(5, 1.0f);  // aggressive reduction

        // Signal: 440Hz tone + white noise (low SNR)
        std::vector<float> noisy(16000);
        for (size_t i = 0; i < noisy.size(); i++) {
            double t = static_cast<double>(i) / sampleRate;
            float tone = 0.3f * std::sin(2.0 * 3.14159265 * 440.0 * t);
            float noise = 0.3f * (std::sin(static_cast<float>(i) * 13.7f) +
                                   std::sin(static_cast<float>(i) * 47.1f));
            noise /= 2.0f;
            noisy[i] = tone + noise;
        }

        auto result = nr.process(noisy, sampleRate);

        float originalEnergy = 0.0f, resultEnergy = 0.0f;
        for (size_t i = 0; i < noisy.size(); i++) {
            originalEnergy += noisy[i] * noisy[i];
            resultEnergy += result[i] * result[i];
        }

        bool ok = (resultEnergy < originalEnergy);
        failures += !ok;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " Energy reduced (before=" << originalEnergy
                  << ", after=" << resultEnergy << ")" << std::endl;
    }

    // ==========================================
    // Test 4: Empty input
    // ==========================================
    std::cout << "\n[Test 4] Empty input..." << std::endl;
    {
        NoiseReduction nr;
        std::vector<float> empty;
        auto result = nr.process(empty, sampleRate);

        bool ok = result.empty();
        failures += !ok;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]") << " Returns empty" << std::endl;
    }

    // ==========================================
    // Test 5: Output same size as input
    // ==========================================
    std::cout << "\n[Test 5] Output size..." << std::endl;
    {
        NoiseReduction nr(10, 0.02f);
        auto pcm = generateSine(440.0f, 0.3f, sampleRate);
        auto result = nr.process(pcm, sampleRate);

        bool ok = (result.size() == pcm.size());
        failures += !ok;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " Size preserved (" << result.size() << ")" << std::endl;
    }

    // ==========================================
    // Test 6: All values finite
    // ==========================================
    std::cout << "\n[Test 6] All values finite..." << std::endl;
    {
        NoiseReduction nr(10, 0.5f);
        auto pcm = generateSine(440.0f, 0.5f, sampleRate);
        auto result = nr.process(pcm, sampleRate);

        bool allFinite = true;
        for (float x : result) {
            if (!std::isfinite(x)) { allFinite = false; break; }
        }
        failures += !allFinite;
        std::cout << (allFinite ? "  [PASS]" : "  [FAIL]") << " All finite" << std::endl;
    }

    // ==========================================
    // Test 7: Long input remains bounded and valid
    // ==========================================
    std::cout << "\n[Test 7] Long input..." << std::endl;
    {
        NoiseReduction nr(10, 0.5f);
        auto pcm = generateSine(440.0f, 60.0f, sampleRate);
        auto result = nr.process(pcm, sampleRate);

        bool ok = result.size() == pcm.size();
        for (float x : result) {
            if (!std::isfinite(x)) {
                ok = false;
                break;
            }
        }
        failures += !ok;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]")
                  << " 60-second input processed (" << result.size()
                  << " samples)" << std::endl;
    }

    std::cout << "\n========== Test Complete: "
              << failures << " failure(s) ==========" << std::endl;
    return failures == 0 ? 0 : 1;
}
