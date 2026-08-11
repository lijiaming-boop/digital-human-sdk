#include "audio/audio_loader.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/version.h>
}

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

namespace digital_human {
namespace audio {

struct AudioLoader::Impl {
    AudioData load(const std::string& filePath) {
        if (!fs::exists(filePath)) {
            throw AudioLoaderException("file not exists: " + filePath);
        }

        AVFormatContext* formatCtx = nullptr;
        if (avformat_open_input(&formatCtx, filePath.c_str(), nullptr, nullptr) < 0) {
            throw AudioLoaderException("failed to open file: " + filePath);
        }

        if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
            avformat_close_input(&formatCtx);
            throw AudioLoaderException("failed to find stream info");
        }

        int audioStreamIdx = -1;
        const AVCodec* codec = nullptr;
        for (unsigned int i = 0; i < formatCtx->nb_streams; i++) {
            if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStreamIdx = i;
                codec = avcodec_find_decoder(formatCtx->streams[i]->codecpar->codec_id);
                break;
            }
        }

        if (audioStreamIdx == -1 || !codec) {
            avformat_close_input(&formatCtx);
            throw AudioLoaderException("no audio stream found");
        }

        AVCodecParameters* codecParams = formatCtx->streams[audioStreamIdx]->codecpar;
        AVCodecContext* codecCtx = avcodec_alloc_context3(codec);
        if (!codecCtx) {
            avformat_close_input(&formatCtx);
            throw AudioLoaderException("failed to allocate codec context");
        }

        if (avcodec_parameters_to_context(codecCtx, codecParams) < 0) {
            avcodec_free_context(&codecCtx);
            avformat_close_input(&formatCtx);
            throw AudioLoaderException("failed to copy codec parameters");
        }

        if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
            avcodec_free_context(&codecCtx);
            avformat_close_input(&formatCtx);
            throw AudioLoaderException("failed to open codec");
        }

        // 修复: 当 channel_layout 未设置时（某些 WAV 文件），根据 channels 推导
        SwrContext* swrCtx = nullptr;
#if LIBAVUTIL_VERSION_MAJOR >= 57
        AVChannelLayout source_layout = codecCtx->ch_layout;
        bool owns_source_layout = false;
        if (source_layout.nb_channels == 0) {
            av_channel_layout_default(&source_layout, 1);
            owns_source_layout = true;
        }
        AVChannelLayout mono_layout = AV_CHANNEL_LAYOUT_MONO;
        const int swr_options_result = swr_alloc_set_opts2(
            &swrCtx,
            &mono_layout, AV_SAMPLE_FMT_FLT, 16000,
            &source_layout, codecCtx->sample_fmt, codecCtx->sample_rate,
            0, nullptr);
        if (owns_source_layout) {
            av_channel_layout_uninit(&source_layout);
        }
#else
        int64_t source_layout = codecCtx->channel_layout;
        if (source_layout == 0) {
            source_layout = av_get_default_channel_layout(codecCtx->channels);
        }
        swrCtx = swr_alloc_set_opts(
            nullptr,
            AV_CH_LAYOUT_MONO, AV_SAMPLE_FMT_FLT, 16000,
            source_layout, codecCtx->sample_fmt, codecCtx->sample_rate,
            0, nullptr);
        const int swr_options_result = swrCtx ? 0 : AVERROR(ENOMEM);
#endif
        if (swr_options_result < 0 || !swrCtx || swr_init(swrCtx) < 0) {
            avcodec_free_context(&codecCtx);
            avformat_close_input(&formatCtx);
            swr_free(&swrCtx);
            throw AudioLoaderException("failed to initialize resampler");
        }

        std::vector<float> outputSamples;
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();

        double duration = 0.0;
        if (formatCtx->duration != AV_NOPTS_VALUE) {
            duration = static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
        }

        while (av_read_frame(formatCtx, packet) >= 0) {
            if (packet->stream_index == audioStreamIdx) {
                if (avcodec_send_packet(codecCtx, packet) < 0) {
                    av_packet_unref(packet);
                    continue;
                }

                while (true) {
                    int ret = avcodec_receive_frame(codecCtx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    }
                    if (ret < 0) {
                        break;
                    }

                    int outSamples = swr_get_out_samples(swrCtx, frame->nb_samples);
                    std::vector<float> buf(outSamples);

                    uint8_t* outBuf = reinterpret_cast<uint8_t*>(buf.data());
                    int converted = swr_convert(swrCtx, &outBuf, outSamples,
                                                const_cast<const uint8_t**>(frame->data),
                                                frame->nb_samples);
                    if (converted > 0) {
                        outputSamples.insert(outputSamples.end(), buf.data(), buf.data() + converted);
                    }
                }
            }
            av_packet_unref(packet);
        }

        av_frame_free(&frame);
        av_packet_free(&packet);
        swr_free(&swrCtx);
        avcodec_free_context(&codecCtx);
        avformat_close_input(&formatCtx);

        if (outputSamples.empty()) {
            throw AudioLoaderException("no audio data decoded");
        }

        normalize(outputSamples);

        AudioData result;
        result.samples = std::move(outputSamples);
        result.sampleRate = 16000;
        result.channels = 1;
        result.duration = duration;
        return result;
    }

    void normalize(std::vector<float>& samples) {
        float maxVal = 0.0f;
        for (float s : samples) {
            maxVal = std::max(maxVal, std::fabs(s));
        }
        if (maxVal > 0.0f) {
            float scale = 1.0f / maxVal;
            for (float& s : samples) {
                s *= scale;
            }
        }
    }
};

AudioLoader::AudioLoader() : impl_(std::make_unique<Impl>()) {}

AudioLoader::~AudioLoader() = default;
AudioLoader::AudioLoader(AudioLoader&& other) noexcept = default;
AudioLoader& AudioLoader::operator=(AudioLoader&& other) noexcept = default;

AudioData AudioLoader::load(const std::string& filePath) {
    return impl_->load(filePath);
}

}  // namespace audio
}  // namespace digital_human
