#include <iostream>
#include <vector>
#include <cmath>
#include "audio/audio_framer.h"

using namespace digital_human::audio;

constexpr double PI = 3.14159265358979323846;

double hamming(int n, int L) {
    return 0.54 - 0.46 * std::cos(2.0 * PI * n / (L - 1));
}

int main() {
    std::cout << "========== AudioFramer Test ==========" << std::endl;

    AudioFramer framer;

    // ==========================================
    // Test 1: Exact fit (no padding needed)
    // ==========================================
    std::cout << "\n[Test 1] Exact fit — no padding..." << std::endl;
    {
        // frameSize=4, hopSize=2, N=10 → 4 frames:
        //   Frame 0: idx [0,4), Frame 1: [2,6), Frame 2: [4,8), Frame 3: [6,10)
        //   Last frame ends at 10 = N → exact fit
        FrameConfig cfg{4, 2};
        std::vector<float> pcm = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

        auto fm = framer.frame(pcm, cfg);

        int expectedFrames = 4;
        bool countOk = (static_cast<int>(fm.size()) == expectedFrames);
        std::cout << (countOk ? "  [PASS]" : "  [FAIL]") << " Frame count: " << fm.size()
                  << " (expected " << expectedFrames << ")" << std::endl;

        // Check frame[0] = {1*w0, 2*w1, 3*w2, 4*w3}
        bool valueOk = true;
        for (int j = 0; j < 4; j++) {
            float expected = static_cast<float>(pcm[j] * hamming(j, 4));
            if (std::fabs(fm[0][j] - expected) > 1e-4f) {
                valueOk = false;
                break;
            }
        }
        std::cout << (valueOk ? "  [PASS]" : "  [FAIL]") << " Frame 0 windowed values" << std::endl;

        // Check frame[3] = {7*w0, 8*w1, 9*w2, 10*w3}
        valueOk = true;
        for (int j = 0; j < 4; j++) {
            float expected = static_cast<float>(pcm[6 + j] * hamming(j, 4));
            if (std::fabs(fm[3][j] - expected) > 1e-4f) {
                valueOk = false;
                break;
            }
        }
        std::cout << (valueOk ? "  [PASS]" : "  [FAIL]") << " Frame 3 windowed values" << std::endl;
    }

    // ==========================================
    // Test 2: Padding needed
    // ==========================================
    std::cout << "\n[Test 2] Last frame needs padding..." << std::endl;
    {
        // frameSize=4, hopSize=2, N=9 → 4 frames
        //   Frame 3: [6,10) but N=9 → last element padded with 0
        FrameConfig cfg{4, 2};
        std::vector<float> pcm = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

        auto fm = framer.frame(pcm, cfg);

        int expectedFrames = 4;
        bool countOk = (static_cast<int>(fm.size()) == expectedFrames);
        std::cout << (countOk ? "  [PASS]" : "  [FAIL]") << " Frame count: " << fm.size()
                  << " (expected " << expectedFrames << ")" << std::endl;

        // Frame 3, last element (idx 9 = N, out of bounds → 0 * w3 = 0)
        bool padOk = (std::fabs(fm[3][3] - 0.0f) < 1e-6f);
        std::cout << (padOk ? "  [PASS]" : "  [FAIL]") << " Last element of last frame is zero-padded" << std::endl;

        // Frame 3, element 2 (idx 8 → pcm[8]=9.0 * w2)
        float expected = static_cast<float>(9.0f * hamming(2, 4));
        bool prePadOk = (std::fabs(fm[3][2] - expected) < 1e-4f);
        std::cout << (prePadOk ? "  [PASS]" : "  [FAIL]") << " Second-to-last element uses real sample" << std::endl;
    }

    // ==========================================
    // Test 3: PCM shorter than one frame
    // ==========================================
    std::cout << "\n[Test 3] PCM shorter than frameSize..." << std::endl;
    {
        // frameSize=4, hopSize=2, N=2 → 1 frame, mostly padding
        FrameConfig cfg{4, 2};
        std::vector<float> pcm = {0.5f, 0.8f};

        auto fm = framer.frame(pcm, cfg);

        bool countOk = (fm.size() == 1);
        std::cout << (countOk ? "  [PASS]" : "  [FAIL]") << " Single frame output" << std::endl;

        bool dimOk = (fm[0].size() == 4);
        std::cout << (dimOk ? "  [PASS]" : "  [FAIL]") << " Frame has correct size" << std::endl;

        // fm[0] = {0.5*w0, 0.8*w1, 0*w2, 0*w3}
        bool firstOk = (std::fabs(fm[0][0] - 0.5f * hamming(0, 4)) < 1e-4f);
        bool secondOk = (std::fabs(fm[0][1] - 0.8f * hamming(1, 4)) < 1e-4f);
        bool thirdOk = (std::fabs(fm[0][2] - 0.0f) < 1e-6f);
        bool fourthOk = (std::fabs(fm[0][3] - 0.0f) < 1e-6f);

        std::cout << (firstOk ? "  [PASS]" : "  [FAIL]") << " Element 0 uses sample" << std::endl;
        std::cout << (secondOk ? "  [PASS]" : "  [FAIL]") << " Element 1 uses sample" << std::endl;
        std::cout << (thirdOk ? "  [PASS]" : "  [FAIL]") << " Element 2 zero-padded" << std::endl;
        std::cout << (fourthOk ? "  [PASS]" : "  [FAIL]") << " Element 3 zero-padded" << std::endl;
    }

    // ==========================================
    // Test 4: Hamming window formula verification
    // ==========================================
    std::cout << "\n[Test 4] Hamming window formula..." << std::endl;
    {
        int L = 8;
        // w[0] = 0.54 - 0.46*cos(0) = 0.54 - 0.46 = 0.08
        // w[L-1] = 0.54 - 0.46*cos(2π) = 0.54 - 0.46 = 0.08
        // w[L/2] ≈ 0.54 - 0.46*cos(π) = 0.54 + 0.46 = 1.0 (peak at center)

        double w0 = hamming(0, L);
        double wEnd = hamming(L - 1, L);
        double wMid = hamming(L / 2, L);

        bool edgeOk = (std::fabs(w0 - 0.08) < 1e-6 && std::fabs(wEnd - 0.08) < 1e-6);
        bool peakOk = (std::fabs(wMid - 1.0) < 1e-6);

        std::cout << (edgeOk ? "  [PASS]" : "  [FAIL]") << " Edges = 0.08 ("
                  << w0 << ", " << wEnd << ")" << std::endl;
        std::cout << (peakOk ? "  [PASS]" : "  [FAIL]") << " Center = 1.0 (" << wMid << ")" << std::endl;
    }

    // ==========================================
    // Test 5: Error cases
    // ==========================================
    std::cout << "\n[Test 5] Error cases..." << std::endl;
    {
        // Empty PCM
        try {
            std::vector<float> empty;
            framer.frame(empty);
            std::cout << "  [FAIL] Should have thrown on empty pcm" << std::endl;
        } catch (const AudioFramerException& e) {
            std::cout << "  [PASS] Empty pcm throws: " << e.what() << std::endl;
        }

        // Invalid config
        try {
            FrameConfig badCfg{0, 160};
            std::vector<float> pcm = {1.0f};
            framer.frame(pcm, badCfg);
            std::cout << "  [FAIL] Should have thrown on frameSize=0" << std::endl;
        } catch (const AudioFramerException& e) {
            std::cout << "  [PASS] Invalid frameSize throws: " << e.what() << std::endl;
        }

        try {
            FrameConfig badCfg{400, -1};
            std::vector<float> pcm = {1.0f};
            framer.frame(pcm, badCfg);
            std::cout << "  [FAIL] Should have thrown on negative hopSize" << std::endl;
        } catch (const AudioFramerException& e) {
            std::cout << "  [PASS] Negative hopSize throws: " << e.what() << std::endl;
        }
    }

    // ==========================================
    // Test 6: Window cache reuse
    // ==========================================
    std::cout << "\n[Test 6] Window cache reuse..." << std::endl;
    {
        // Same frameSize should reuse cached window
        FrameConfig cfg{4, 2};
        std::vector<float> pcm(20, 1.0f);

        auto fm1 = framer.frame(pcm, cfg);
        auto fm2 = framer.frame(pcm, cfg);

        // Both should produce identical results (window reused)
        bool identical = true;
        for (size_t i = 0; i < fm1.size() && identical; i++) {
            for (size_t j = 0; j < fm1[i].size() && identical; j++) {
                if (std::fabs(fm1[i][j] - fm2[i][j]) > 1e-6f) identical = false;
            }
        }
        std::cout << (identical ? "  [PASS]" : "  [FAIL]") << " Same config produces identical results" << std::endl;
    }

    std::cout << "\n========== Test Complete ==========" << std::endl;
    return 0;
}
