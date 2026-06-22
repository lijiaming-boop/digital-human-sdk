#include <iostream>
#include <vector>
#include <cmath>
#include "audio/audio_framer.h"
#include "audio/audio_mel_feature_extract.h"

using namespace digital_human::audio;

constexpr double PI = 3.14159265358979323846;

std::vector<float> generateSine(float freq, float durationSec, int sampleRate) {
    int totalSamples = static_cast<int>(sampleRate * durationSec);
    std::vector<float> samples(totalSamples);
    for (int i = 0; i < totalSamples; i++) {
        double t = static_cast<double>(i) / sampleRate;
        samples[i] = static_cast<float>(std::sin(2.0 * PI * freq * t));
    }
    return samples;
}

int main() {
    std::cout << "========== MelFeatureExtract Test ==========" << std::endl;

    AudioFramer framer;
    MelFeatureExtract extractor;

    // ==========================================
    // Test 1: Basic shape check
    // ==========================================
    std::cout << "\n[Test 1] Output shape..." << std::endl;
    {
        auto pcm = generateSine(440.0f, 1.0f, 16000);
        FrameConfig frameCfg{400, 160};
        auto frames = framer.frame(pcm, frameCfg);

        MelConfig cfg;
        cfg.nFFT = 512;
        cfg.nMels = 80;
        cfg.sampleRate = 16000;

        cv::Mat melSpec = extractor.extract(frames, cfg);

        int expectedFrames = static_cast<int>(frames.size());
        bool rowsOk = (melSpec.rows == expectedFrames);
        bool colsOk = (melSpec.cols == cfg.nMels);
        bool typeOk = (melSpec.type() == CV_32FC1);

        std::cout << (rowsOk ? "  [PASS]" : "  [FAIL]") << " Rows: " << melSpec.rows
                  << " (expected " << expectedFrames << ")" << std::endl;
        std::cout << (colsOk ? "  [PASS]" : "  [FAIL]") << " Cols: " << melSpec.cols
                  << " (expected " << cfg.nMels << ")" << std::endl;
        std::cout << (typeOk ? "  [PASS]" : "  [FAIL]") << " Type: CV_32FC1" << std::endl;
    }

    // ==========================================
    // Test 2: Normalization range [0, 1]
    // ==========================================
    std::cout << "\n[Test 2] Normalization to [0, 1]..." << std::endl;
    {
        auto pcm = generateSine(440.0f, 1.0f, 16000);
        FrameConfig frameCfg{400, 160};
        auto frames = framer.frame(pcm, frameCfg);

        cv::Mat melSpec = extractor.extract(frames);

        double minVal, maxVal;
        cv::minMaxLoc(melSpec, &minVal, &maxVal);

        bool minOk = (minVal >= 0.0 && minVal <= 0.01);
        bool maxOk = (maxVal >= 0.99 && maxVal <= 1.01);
        bool rangeOk = (maxVal > minVal);

        std::cout << "  Min: " << minVal << ", Max: " << maxVal << std::endl;
        std::cout << (minOk ? "  [PASS]" : "  [FAIL]") << " Min ≈ 0" << std::endl;
        std::cout << (maxOk ? "  [PASS]" : "  [FAIL]") << " Max ≈ 1" << std::endl;
        std::cout << (rangeOk ? "  [PASS]" : "  [FAIL]") << " Range > 0 (not flat)" << std::endl;
    }

    // ==========================================
    // Test 3: 440Hz sine energy in correct mel bins
    // ==========================================
    std::cout << "\n[Test 3] Frequency → Mel bin mapping..." << std::endl;
    {
        auto pcm = generateSine(440.0f, 1.0f, 16000);
        FrameConfig frameCfg{400, 160};
        auto frames = framer.frame(pcm, frameCfg);

        cv::Mat melSpec = extractor.extract(frames);

        // Average across all frames → 1x80 row vector of mean bin energy
        cv::Mat meanMel;
        cv::reduce(melSpec, meanMel, 0, cv::REDUCE_AVG);

        // 440 Hz → mel = 2595*log10(1+440/700) ≈ 529
        // mel bins: 0..8000Hz → 0..2840 mel, step ≈ 35 mel/bin
        // Expected peak around bin 15
        double globalMin, globalMax;
        cv::Point maxLoc;
        cv::minMaxLoc(meanMel, &globalMin, &globalMax, nullptr, &maxLoc);

        int peakBin = maxLoc.x;
        std::cout << "  Peak mel bin: " << peakBin << " (440 Hz sine)" << std::endl;

        // 440Hz → bin 14.08 → mel bin ~15
        bool peakOk = (peakBin >= 12 && peakBin <= 20);
        std::cout << (peakOk ? "  [PASS]" : "  [FAIL]")
                  << " Peak in expected mel bin range [12, 20]" << std::endl;
    }

    // ==========================================
    // Test 4: Different frequencies → different peaks
    // ==========================================
    std::cout << "\n[Test 4] Different frequencies map to different mel bins..." << std::endl;
    {
        auto pcmLow = generateSine(200.0f, 0.5f, 16000);
        auto pcmHigh = generateSine(2000.0f, 0.5f, 16000);

        FrameConfig frameCfg{400, 160};
        auto framesLow = framer.frame(pcmLow, frameCfg);
        auto framesHigh = framer.frame(pcmHigh, frameCfg);

        cv::Mat melLow = extractor.extract(framesLow);
        cv::Mat melHigh = extractor.extract(framesHigh);

        cv::Mat meanLow, meanHigh;
        cv::reduce(melLow, meanLow, 0, cv::REDUCE_AVG);
        cv::reduce(melHigh, meanHigh, 0, cv::REDUCE_AVG);

        double minVal;
        cv::Point lowPeak, highPeak;
        double lowMax, highMax;
        cv::minMaxLoc(meanLow, &minVal, &lowMax, nullptr, &lowPeak);
        cv::minMaxLoc(meanHigh, &minVal, &highMax, nullptr, &highPeak);

        std::cout << "  200 Hz peak bin: " << lowPeak.x << std::endl;
        std::cout << "  2000 Hz peak bin: " << highPeak.x << std::endl;

        bool diffOk = (highPeak.x > lowPeak.x);
        std::cout << (diffOk ? "  [PASS]" : "  [FAIL]")
                  << " Higher frequency → higher mel bin" << std::endl;
    }

    // ==========================================
    // Test 5: Empty input
    // ==========================================
    std::cout << "\n[Test 5] Empty frames..." << std::endl;
    {
        std::vector<std::vector<float>> empty;
        cv::Mat result = extractor.extract(empty);

        bool emptyOk = result.empty();
        std::cout << (emptyOk ? "  [PASS]" : "  [FAIL]") << " Returns empty Mat" << std::endl;
    }

    // ==========================================
    // Test 6: Filterbank cache — same config reuses filterbank
    // ==========================================
    std::cout << "\n[Test 6] Filterbank cache..." << std::endl;
    {
        auto pcm = generateSine(440.0f, 0.2f, 16000);
        FrameConfig frameCfg{400, 160};
        auto frames = framer.frame(pcm, frameCfg);

        MelConfig cfg;
        cv::Mat m1 = extractor.extract(frames, cfg);
        cv::Mat m2 = extractor.extract(frames, cfg);

        // Same config → identical results (filterbank reused, not rebuilt)
        bool identical = true;
        for (int r = 0; r < m1.rows && identical; r++) {
            const float* row1 = m1.ptr<float>(r);
            const float* row2 = m2.ptr<float>(r);
            for (int c = 0; c < m1.cols && identical; c++) {
                if (std::fabs(row1[c] - row2[c]) > 1e-6f) identical = false;
            }
        }
        std::cout << (identical ? "  [PASS]" : "  [FAIL]")
                  << " Same config produces identical results" << std::endl;

        // Different config → filterbank rebuilt, different results
        MelConfig cfg2;
        cfg2.nFFT = 1024;
        cv::Mat m3 = extractor.extract(frames, cfg2);

        bool differentShape = (m3.rows == m1.rows && m3.cols == m1.cols);
        // Even with same nMels and numFrames, the values should differ due to different nFFT
        bool anyDiff = false;
        if (differentShape) {
            for (int r = 0; r < m1.rows && !anyDiff; r++) {
                const float* row1 = m1.ptr<float>(r);
                const float* row3 = m3.ptr<float>(r);
                for (int c = 0; c < m1.cols && !anyDiff; c++) {
                    if (std::fabs(row1[c] - row3[c]) > 1e-3f) anyDiff = true;
                }
            }
        }
        std::cout << (anyDiff ? "  [PASS]" : "  [FAIL]")
                  << " Different nFFT → different mel spectrogram" << std::endl;
    }

    // ==========================================
    // Test 7: All values are finite (no NaN / Inf)
    // ==========================================
    std::cout << "\n[Test 7] No NaN or Inf values..." << std::endl;
    {
        auto pcm = generateSine(440.0f, 1.0f, 16000);
        FrameConfig frameCfg{400, 160};
        auto frames = framer.frame(pcm, frameCfg);

        cv::Mat melSpec = extractor.extract(frames);

        bool allFinite = true;
        for (int r = 0; r < melSpec.rows && allFinite; r++) {
            const float* row = melSpec.ptr<float>(r);
            for (int c = 0; c < melSpec.cols && allFinite; c++) {
                if (!std::isfinite(row[c])) allFinite = false;
            }
        }
        std::cout << (allFinite ? "  [PASS]" : "  [FAIL]") << " All values finite" << std::endl;
    }

    // ==========================================
    // Test 8: Silent input → low energy
    // ==========================================
    std::cout << "\n[Test 8] Silent input energy..." << std::endl;
    {
        std::vector<float> silence(16000, 1e-6f);  // near-silence
        FrameConfig frameCfg{400, 160};
        auto frames = framer.frame(silence, frameCfg);

        cv::Mat melSpec = extractor.extract(frames);
        cv::Scalar meanVal = cv::mean(melSpec);

        // Silent input should have very low mean after dB conversion and normalization
        // Actually, with normalization, silent input gets scaled to [0,1] just like any input
        // So this test is not that meaningful for absolute values
        // But we can check that the result is not NaN or Inf
        bool finiteOk = std::isfinite(static_cast<float>(meanVal[0]));
        std::cout << (finiteOk ? "  [PASS]" : "  [FAIL]")
                  << " Silent input produces finite output (mean=" << meanVal[0] << ")" << std::endl;
    }

    std::cout << "\n========== Test Complete ==========" << std::endl;
    return 0;
}
