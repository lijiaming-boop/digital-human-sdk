#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "audio/audio_cmvn.h"

using namespace digital_human::audio;

int main() {
    std::cout << "========== CMVN Test ==========" << std::endl;

    const float eps = 1e-4f;

    // ==========================================
    // Test 1: Per-dimension mean ≈ 0, std ≈ 1
    // ==========================================
    std::cout << "\n[Test 1] Mean ≈ 0, std ≈ 1 per dimension..." << std::endl;
    {
        // Create a non-trivial mel spectrogram: 100 frames × 80 mel bins
        int nFrames = 100;
        int nMels = 80;
        cv::Mat melSpec(nFrames, nMels, CV_32F);
        for (int i = 0; i < nFrames; i++) {
            float* row = melSpec.ptr<float>(i);
            for (int j = 0; j < nMels; j++) {
                // Different mean per mel bin
                row[j] = 0.1f * j + 0.5f * std::sin(static_cast<float>(i) * 0.1f);
            }
        }

        CMVN cmvn;
        cv::Mat normalized = cmvn.process(melSpec);

        // Check shape
        bool shapeOk = (normalized.rows == nFrames && normalized.cols == nMels);

        // Check mean of first dimension
        cv::Scalar mean = cv::mean(normalized.col(0));
        bool meanOk = std::fabs(static_cast<float>(mean[0]) - 0.0f) < 0.01f;

        // Check std of first dimension
        cv::Mat col0 = normalized.col(0);
        cv::Scalar colMean, colStd;
        cv::meanStdDev(col0, colMean, colStd);
        float stdVal = static_cast<float>(colStd[0]);
        bool stdOk = std::fabs(stdVal - 1.0f) < 0.01f;

        std::cout << (shapeOk ? "  [PASS]" : "  [FAIL]") << " Shape preserved" << std::endl;
        std::cout << (meanOk ? "  [PASS]" : "  [FAIL]")
                  << " Mean ≈ 0 (actual " << mean[0] << ")" << std::endl;
        std::cout << (stdOk ? "  [PASS]" : "  [FAIL]")
                  << " Std ≈ 1 (actual " << stdVal << ")" << std::endl;
    }

    // ==========================================
    // Test 2: Constant input → NaN handling
    // ==========================================
    std::cout << "\n[Test 2] Constant value input..." << std::endl;
    {
        cv::Mat melSpec(10, 5, CV_32F, cv::Scalar(0.5f));
        CMVN cmvn;
        cv::Mat result = cmvn.process(melSpec);

        // Constant input: mean = 0.5, std = 0 → normalization gives 0 (or epsilon)
        bool finiteOk = true;
        for (int i = 0; i < result.rows && finiteOk; i++) {
            const float* row = result.ptr<float>(i);
            for (int j = 0; j < result.cols && finiteOk; j++) {
                if (!std::isfinite(row[j])) finiteOk = false;
            }
        }
        std::cout << (finiteOk ? "  [PASS]" : "  [FAIL]") << " All values finite" << std::endl;
    }

    // ==========================================
    // Test 3: Empty input
    // ==========================================
    std::cout << "\n[Test 3] Empty input..." << std::endl;
    {
        CMVN cmvn;
        cv::Mat empty;
        cv::Mat result = cmvn.process(empty);

        bool ok = result.empty();
        std::cout << (ok ? "  [PASS]" : "  [FAIL]") << " Returns empty" << std::endl;
    }

    // ==========================================
    // Test 4: Single frame
    // ==========================================
    std::cout << "\n[Test 4] Single frame..." << std::endl;
    {
        cv::Mat melSpec(1, 4, CV_32F);
        float* row = melSpec.ptr<float>(0);
        row[0] = 1.0f; row[1] = 2.0f; row[2] = 3.0f; row[3] = 4.0f;

        CMVN cmvn;
        cv::Mat result = cmvn.process(melSpec);

        // Single frame: each dimension's mean = value, std = 0
        // → result should be 0 (with epsilon protection)
        bool finiteOk = true;
        for (int j = 0; j < result.cols; j++) {
            if (!std::isfinite(result.at<float>(0, j))) finiteOk = false;
        }
        std::cout << (finiteOk ? "  [PASS]" : "  [FAIL]") << " Single frame finite output" << std::endl;
    }

    // ==========================================
    // Test 5: Identity — check that normalization changes values
    // ==========================================
    std::cout << "\n[Test 5] Normalization actually changes values..." << std::endl;
    {
        int nFrames = 50;
        int nMels = 10;
        cv::Mat melSpec(nFrames, nMels, CV_32F);
        for (int i = 0; i < nFrames; i++) {
            float* row = melSpec.ptr<float>(i);
            for (int j = 0; j < nMels; j++) {
                row[j] = static_cast<float>(i * (j + 1)) / 100.0f;
            }
        }

        CMVN cmvn;
        cv::Mat result = cmvn.process(melSpec);

        // Values should differ from input (since we have variance)
        double diff = cv::norm(melSpec, result);
        bool changed = (diff > 0.01);
        std::cout << (changed ? "  [PASS]" : "  [FAIL]")
                  << " Normalization changed values (diff=" << diff << ")" << std::endl;
    }

    std::cout << "\n========== Test Complete ==========" << std::endl;
    return 0;
}
