#include "audio/audio_noise_reduction.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

namespace digital_human {
namespace audio {

struct NoiseReduction::Impl {
    int noiseFrames;
    float oversubtraction;
    float floorFactor;

    static constexpr int kFrameSize = 512;
    static constexpr int kHopSize = 128;
    static constexpr float kFloor = 0.01f;

    std::vector<float> hammingWindow(int size) {
        std::vector<float> w(size);
        for (int i = 0; i < size; i++) {
            w[i] = 0.54f - 0.46f * std::cos(2.0f * 3.14159265f * i / (size - 1));
        }
        return w;
    }

    std::vector<float> process(const std::vector<float>& pcm, int sampleRate) {
        (void)sampleRate;
        if (pcm.empty()) return {};

        int N = static_cast<int>(pcm.size());
        int nFreqBins = kFrameSize / 2 + 1;

        // Frame the signal
        int numFrames = std::max(1, (N - kFrameSize + kHopSize - 1) / kHopSize + 1);

        // Pad to full frames
        int paddedLen = (numFrames - 1) * kHopSize + kFrameSize;
        std::vector<float> padded(paddedLen, 0.0f);
        std::copy(pcm.begin(), pcm.end(), padded.begin());

        auto window = hammingWindow(kFrameSize);

        // Compute FFT for each frame
        std::vector<std::vector<float>> magnitudeSpectra(numFrames,
            std::vector<float>(nFreqBins, 0.0f));
        std::vector<std::vector<float>> phaseSpectra(numFrames,
            std::vector<float>(nFreqBins, 0.0f));

        cv::Mat fftInput(1, kFrameSize, CV_32F);
        for (int f = 0; f < numFrames; f++) {
            float* row = fftInput.ptr<float>(0);
            for (int j = 0; j < kFrameSize; j++) {
                int idx = f * kHopSize + j;
                row[j] = (idx < paddedLen) ? padded[idx] * window[j] : 0.0f;
            }

            cv::Mat fftResult;
            cv::dft(fftInput, fftResult, cv::DFT_COMPLEX_OUTPUT);

            const float* fftRow = fftResult.ptr<float>(0);
            for (int k = 0; k < nFreqBins; k++) {
                float real = fftRow[2 * k];
                float imag = fftRow[2 * k + 1];
                magnitudeSpectra[f][k] = std::sqrt(real * real + imag * imag);
                phaseSpectra[f][k] = std::atan2(imag, real);
            }
        }

        // Estimate noise from first noiseFrames (average magnitude)
        int noiseCount = std::min(noiseFrames, numFrames);
        std::vector<float> noiseMag(nFreqBins, 0.0f);
        for (int f = 0; f < noiseCount; f++) {
            for (int k = 0; k < nFreqBins; k++) {
                noiseMag[k] += magnitudeSpectra[f][k];
            }
        }
        for (int k = 0; k < nFreqBins; k++) {
            noiseMag[k] /= noiseCount;
        }

        // Spectral subtraction
        for (int f = 0; f < numFrames; f++) {
            for (int k = 0; k < nFreqBins; k++) {
                float cleanMag = magnitudeSpectra[f][k]
                    - oversubtraction * noiseMag[k];
                cleanMag = std::max(cleanMag, kFloor * magnitudeSpectra[f][k]);
                magnitudeSpectra[f][k] = cleanMag;
            }
        }

        // IFFT and overlap-add
        std::vector<float> output(paddedLen, 0.0f);
        std::vector<float> weightSum(paddedLen, 0.0f);

        for (int f = 0; f < numFrames; f++) {
            // Reconstruct complex spectrum
            cv::Mat complexSpec(1, kFrameSize, CV_32FC2);
            float* cRow = complexSpec.ptr<float>(0);
            for (int k = 0; k < kFrameSize; k++) {
                float mag, phase;
                if (k < nFreqBins) {
                    mag = magnitudeSpectra[f][k];
                    phase = phaseSpectra[f][k];
                } else {
                    // Mirror conjugate
                    int mirror = kFrameSize - k;
                    mag = magnitudeSpectra[f][mirror];
                    phase = -phaseSpectra[f][mirror];
                }
                cRow[2 * k] = mag * std::cos(phase);
                cRow[2 * k + 1] = mag * std::sin(phase);
            }

            cv::Mat timeOut;
            cv::idft(complexSpec, timeOut,
                     cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

            float* tRow = timeOut.ptr<float>(0);
            for (int j = 0; j < kFrameSize; j++) {
                int idx = f * kHopSize + j;
                if (idx < paddedLen) {
                    output[idx] += tRow[j] * window[j];
                    weightSum[idx] += window[j] * window[j];
                }
            }
        }

        // Normalize by window overlap weights
        std::vector<float> result(pcm.size());
        for (size_t i = 0; i < result.size(); i++) {
            if (weightSum[i] > 1e-10f) {
                result[i] = output[i] / weightSum[i];
            } else {
                result[i] = output[i];
            }
        }

        return result;
    }
};

NoiseReduction::NoiseReduction(int noiseFrames, float oversubtraction)
    : impl_(std::make_unique<Impl>()) {
    impl_->noiseFrames = std::max(1, noiseFrames);
    impl_->oversubtraction = std::max(0.0f, oversubtraction);
    impl_->floorFactor = 0.01f;
}

NoiseReduction::~NoiseReduction() = default;

NoiseReduction::NoiseReduction(NoiseReduction&&) noexcept = default;

NoiseReduction& NoiseReduction::operator=(NoiseReduction&&) noexcept = default;

std::vector<float> NoiseReduction::process(const std::vector<float>& pcm, int sampleRate) {
    return impl_->process(pcm, sampleRate);
}

}  // namespace audio
}  // namespace digital_human
