#include <iostream>
#include <vector>
#include <cmath>
#include "audio/audio_preemphasis.h"

using namespace digital_human::audio;

constexpr double PI = 3.14159265358979323846;

int main() {
    std::cout << "========== PreEmphasis Test ==========" << std::endl;

    // ==========================================
    // Test 1: Formula correctness y[n] = x[n] - alpha * x[n-1]
    // ==========================================
    std::cout << "\n[Test 1] Formula y[n] = x[n] - alpha * x[n-1]..." << std::endl;
    {
        PreEmphasis pre(0.5f);  // use 0.5 for easy verification
        std::vector<float> pcm = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto result = pre.process(pcm);

        // y[0] = x[0] = 1.0
        // y[1] = 2.0 - 0.5*1.0 = 1.5
        // y[2] = 3.0 - 0.5*2.0 = 2.0
        // y[3] = 4.0 - 0.5*3.0 = 2.5
        // y[4] = 5.0 - 0.5*4.0 = 3.0
        float expected[] = {1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
        bool allOk = true;
        for (size_t i = 0; i < 5; i++) {
            if (std::fabs(result[i] - expected[i]) > 1e-6f) {
                std::cout << "  [FAIL] index " << i << ": got " << result[i]
                          << " expected " << expected[i] << std::endl;
                allOk = false;
            }
        }
        std::cout << (allOk ? "  [PASS]" : "  [FAIL]") << " All values match" << std::endl;
    }

    // ==========================================
    // Test 2: Default alpha = 0.97
    // ==========================================
    std::cout << "\n[Test 2] Default alpha = 0.97..." << std::endl;
    {
        PreEmphasis pre;
        std::vector<float> pcm = {1.0f, 2.0f, 3.0f};
        auto result = pre.process(pcm);

        float y0 = 1.0f;
        float y1 = 2.0f - 0.97f * 1.0f;
        float y2 = 3.0f - 0.97f * 2.0f;

        bool ok = std::fabs(result[0] - y0) < 1e-6f
               && std::fabs(result[1] - y1) < 1e-6f
               && std::fabs(result[2] - y2) < 1e-6f;
        std::cout << (ok ? "  [PASS]" : "  [FAIL]") << " Default alpha values" << std::endl;
    }

    // ==========================================
    // Test 3: Single sample — no filtering
    // ==========================================
    std::cout << "\n[Test 3] Single sample..." << std::endl;
    {
        PreEmphasis pre(0.97f);
        std::vector<float> pcm = {3.14f};
        auto result = pre.process(pcm);

        bool sizeOk = (result.size() == 1);
        bool valOk = (std::fabs(result[0] - 3.14f) < 1e-6f);
        std::cout << (sizeOk ? "  [PASS]" : "  [FAIL]") << " Size = 1" << std::endl;
        std::cout << (valOk ? "  [PASS]" : "  [FAIL]") << " Value unchanged" << std::endl;
    }

    // ==========================================
    // Test 4: Empty input
    // ==========================================
    std::cout << "\n[Test 4] Empty input..." << std::endl;
    {
        PreEmphasis pre;
        std::vector<float> empty;
        auto result = pre.process(empty);

        bool emptyOk = result.empty();
        std::cout << (emptyOk ? "  [PASS]" : "  [FAIL]") << " Returns empty vector" << std::endl;
    }

    // ==========================================
    // Test 5: All zeros → all zeros
    // ==========================================
    std::cout << "\n[Test 5] All zeros..." << std::endl;
    {
        PreEmphasis pre(0.95f);
        std::vector<float> zeros = {0.0f, 0.0f, 0.0f, 0.0f};
        auto result = pre.process(zeros);

        bool allZero = true;
        for (float v : result) {
            if (std::fabs(v) > 1e-10f) allZero = false;
        }
        std::cout << (allZero ? "  [PASS]" : "  [FAIL]") << " Output all zeros" << std::endl;
    }

    // ==========================================
    // Test 6: Alpha boundary — alpha=0.0 (passthrough)
    // ==========================================
    std::cout << "\n[Test 6] Alpha = 0.0 (passthrough)..." << std::endl;
    {
        PreEmphasis pre(0.0f);
        std::vector<float> pcm = {1.0f, 2.0f, 3.0f, 4.0f};
        auto result = pre.process(pcm);

        bool passthrough = true;
        for (size_t i = 0; i < pcm.size(); i++) {
            if (std::fabs(result[i] - pcm[i]) > 1e-6f) passthrough = false;
        }
        std::cout << (passthrough ? "  [PASS]" : "  [FAIL]") << " Passthrough" << std::endl;
    }

    // ==========================================
    // Test 7: Alpha clamped to [0, 1]
    // ==========================================
    std::cout << "\n[Test 7] Alpha clamped to [0, 1]..." << std::endl;
    {
        PreEmphasis preLow(-0.5f);   // should clamp to 0
        PreEmphasis preHigh(1.5f);   // should clamp to 1

        std::vector<float> pcm = {1.0f, 2.0f, 3.0f};

        auto resultLow = preLow.process(pcm);
        auto resultHigh = preHigh.process(pcm);

        // low: alpha=0 → passthrough
        bool lowOk = true;
        for (size_t i = 0; i < pcm.size() && lowOk; i++) {
            if (std::fabs(resultLow[i] - pcm[i]) > 1e-6f) lowOk = false;
        }

        // high: alpha=1 → y[n] = x[n] - x[n-1], y[0] = x[0]
        float expHigh[] = {1.0f, 1.0f, 1.0f};  // 2-1=1, 3-2=1
        bool highOk = true;
        for (size_t i = 0; i < 3 && highOk; i++) {
            if (std::fabs(resultHigh[i] - expHigh[i]) > 1e-6f) highOk = false;
        }

        std::cout << (lowOk ? "  [PASS]" : "  [FAIL]") << " Negative alpha clamped to 0" << std::endl;
        std::cout << (highOk ? "  [PASS]" : "  [FAIL]") << " Alpha > 1 clamped to 1" << std::endl;
    }

    std::cout << "\n========== Test Complete ==========" << std::endl;
    return 0;
}
