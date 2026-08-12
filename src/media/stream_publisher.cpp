#include "media/stream_publisher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libavutil/time.h>
#include <libavutil/version.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace digital_human {
namespace media {
namespace {

std::string AvError(int code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, buffer, sizeof(buffer));
    return buffer;
}

void EnsureNetworkInitialized() {
    static std::once_flag flag;
    std::call_once(flag, []() { avformat_network_init(); });
}

StreamProtocol ResolveProtocol(const StreamPublisherConfig& config) {
    if (config.protocol != StreamProtocol::AUTO) return config.protocol;
    if (config.url.rfind("rtmp://", 0) == 0
        || config.url.rfind("rtmps://", 0) == 0) {
        return StreamProtocol::RTMP;
    }
    if (config.url.rfind("rtsp://", 0) == 0
        || config.url.rfind("rtsps://", 0) == 0) {
        return StreamProtocol::RTSP;
    }
    return StreamProtocol::FILE;
}

const char* FormatName(StreamProtocol protocol) {
    if (protocol == StreamProtocol::RTMP) return "flv";
    if (protocol == StreamProtocol::RTSP) return "rtsp";
    return nullptr;
}

bool SupportsPixelFormat(const AVCodec* codec, AVPixelFormat wanted) {
#if LIBAVCODEC_VERSION_MAJOR >= 61
    const void* configs = nullptr;
    int count = 0;
    if (avcodec_get_supported_config(nullptr, codec,
            AV_CODEC_CONFIG_PIX_FORMAT, 0, &configs, &count) < 0
        || !configs) {
        return true;
    }
    const auto* formats = static_cast<const AVPixelFormat*>(configs);
    return std::find(formats, formats + count, wanted) != formats + count;
#else
    if (!codec->pix_fmts) return true;
    for (const AVPixelFormat* fmt = codec->pix_fmts;
         *fmt != AV_PIX_FMT_NONE; ++fmt) {
        if (*fmt == wanted) return true;
    }
    return false;
#endif
}

AVSampleFormat SelectSampleFormat(const AVCodec* codec) {
#if LIBAVCODEC_VERSION_MAJOR >= 61
    const void* configs = nullptr;
    int count = 0;
    if (avcodec_get_supported_config(nullptr, codec,
            AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, &configs, &count) >= 0
        && configs && count > 0) {
        const auto* formats = static_cast<const AVSampleFormat*>(configs);
        for (int i = 0; i < count; ++i) {
            if (formats[i] == AV_SAMPLE_FMT_FLTP) return formats[i];
        }
        return formats[0];
    }
#else
    if (codec->sample_fmts) {
        for (const AVSampleFormat* fmt = codec->sample_fmts;
             *fmt != AV_SAMPLE_FMT_NONE; ++fmt) {
            if (*fmt == AV_SAMPLE_FMT_FLTP) return *fmt;
        }
        return codec->sample_fmts[0];
    }
#endif
    return AV_SAMPLE_FMT_FLTP;
}

int ChannelCount(const AVCodecContext* context) {
#if LIBAVUTIL_VERSION_MAJOR >= 57
    return context->ch_layout.nb_channels;
#else
    return context->channels;
#endif
}

void SetChannelLayout(AVCodecContext* context, int channels) {
#if LIBAVUTIL_VERSION_MAJOR >= 57
    av_channel_layout_default(&context->ch_layout, channels);
#else
    context->channels = channels;
    context->channel_layout = av_get_default_channel_layout(channels);
#endif
}

void CopyChannelLayout(AVFrame* frame, const AVCodecContext* context) {
#if LIBAVUTIL_VERSION_MAJOR >= 57
    av_channel_layout_copy(&frame->ch_layout, &context->ch_layout);
#else
    frame->channels = context->channels;
    frame->channel_layout = context->channel_layout;
#endif
}

}  // namespace

struct StreamPublisher::Impl {
    struct VideoItem {
        cv::Mat frame;
        int64_t pts_ms = 0;
    };
    struct AudioItem {
        std::vector<float> samples;
        int64_t pts_ms = 0;
    };

    StreamPublisherConfig config;
    StreamProtocol protocol = StreamProtocol::FILE;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<VideoItem> video_queue;
    std::deque<AudioItem> audio_queue;
    std::thread worker;
    bool opened = false;
    bool closing = false;
    bool failed = false;
    std::string last_error;

    AVFormatContext* format_context = nullptr;
    AVCodecContext* video_context = nullptr;
    AVCodecContext* audio_context = nullptr;
    AVStream* video_stream = nullptr;
    AVStream* audio_stream = nullptr;
    SwsContext* sws_context = nullptr;
    SwrContext* swr_context = nullptr;
    AVAudioFifo* audio_fifo = nullptr;
    AVFrame* video_frame = nullptr;
    int64_t last_video_pts = AV_NOPTS_VALUE;
    int64_t audio_next_pts = 0;
    bool audio_pts_initialized = false;
    bool header_written = false;

    std::atomic<int64_t> video_frames_in{0};
    std::atomic<int64_t> video_frames_encoded{0};
    std::atomic<int64_t> video_frames_dropped{0};
    std::atomic<int64_t> audio_samples_in{0};
    std::atomic<int64_t> audio_frames_encoded{0};
    std::atomic<int64_t> packets_written{0};
    std::atomic<int64_t> io_deadline_us{0};

    static int InterruptIo(void* opaque) {
        const auto* self = static_cast<const Impl*>(opaque);
        const int64_t deadline = self->io_deadline_us.load(
            std::memory_order_relaxed);
        return deadline > 0 && av_gettime_relative() >= deadline;
    }

    void ArmIoTimeout() {
        if (config.io_timeout_ms <= 0) {
            io_deadline_us.store(0, std::memory_order_relaxed);
            return;
        }
        io_deadline_us.store(
            av_gettime_relative()
                + static_cast<int64_t>(config.io_timeout_ms) * 1000,
            std::memory_order_relaxed);
    }

    void DisarmIoTimeout() {
        io_deadline_us.store(0, std::memory_order_relaxed);
    }

    void SetFailure(const std::string& error) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!failed) last_error = error;
            failed = true;
        }
        cv.notify_all();
    }

    bool OpenVideo(std::string& error) {
        std::vector<std::string> candidates;
        if (!config.video_encoder.empty()) {
            candidates.push_back(config.video_encoder);
        } else {
            candidates = {"h264_nvenc", "h264_qsv", "h264_amf",
                          "libx264", "h264"};
        }

        std::string attempts;
        for (const auto& name : candidates) {
            const AVCodec* codec = avcodec_find_encoder_by_name(name.c_str());
            if (!codec || !SupportsPixelFormat(codec, AV_PIX_FMT_YUV420P)) {
                continue;
            }
            AVCodecContext* candidate = avcodec_alloc_context3(codec);
            if (!candidate) continue;
            candidate->codec_id = codec->id;
            candidate->codec_type = AVMEDIA_TYPE_VIDEO;
            candidate->width = config.width;
            candidate->height = config.height;
            candidate->pix_fmt = AV_PIX_FMT_YUV420P;
            candidate->bit_rate = config.video_bitrate;
            candidate->gop_size = config.gop_size;
            candidate->max_b_frames = 0;
            candidate->time_base = av_d2q(1.0 / config.fps, 100000);
            candidate->framerate = av_d2q(config.fps, 100000);
            if (format_context->oformat->flags & AVFMT_GLOBALHEADER) {
                candidate->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            AVDictionary* options = nullptr;
            if (name == "libx264") {
                av_dict_set(&options, "preset", config.encoder_preset.c_str(), 0);
                av_dict_set(&options, "tune", "zerolatency", 0);
            }
            const int result = avcodec_open2(candidate, codec, &options);
            av_dict_free(&options);
            if (result >= 0) {
                video_context = candidate;
                break;
            }
            attempts += name + "=" + AvError(result) + "; ";
            avcodec_free_context(&candidate);
        }
        if (!video_context) {
            error = "no usable H.264 encoder: " + attempts;
            return false;
        }

        video_stream = avformat_new_stream(format_context, nullptr);
        if (!video_stream) {
            error = "avformat_new_stream(video) failed";
            return false;
        }
        video_stream->time_base = video_context->time_base;
        const int copy_result = avcodec_parameters_from_context(
            video_stream->codecpar, video_context);
        if (copy_result < 0) {
            error = "copy video codec parameters: " + AvError(copy_result);
            return false;
        }
        video_stream->codecpar->codec_tag = 0;

        sws_context = sws_getContext(
            config.width, config.height, AV_PIX_FMT_BGR24,
            config.width, config.height, video_context->pix_fmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_context) {
            error = "sws_getContext failed";
            return false;
        }
        video_frame = av_frame_alloc();
        if (!video_frame) {
            error = "av_frame_alloc(video) failed";
            return false;
        }
        video_frame->format = video_context->pix_fmt;
        video_frame->width = config.width;
        video_frame->height = config.height;
        const int buffer_result = av_frame_get_buffer(video_frame, 32);
        if (buffer_result < 0) {
            error = "video frame buffer: " + AvError(buffer_result);
            return false;
        }
        return true;
    }

    bool OpenAudio(std::string& error) {
        const AVCodec* codec = avcodec_find_encoder_by_name(
            config.audio_encoder.c_str());
        if (!codec) {
            error = "AAC encoder not found: " + config.audio_encoder;
            return false;
        }
        audio_context = avcodec_alloc_context3(codec);
        if (!audio_context) {
            error = "avcodec_alloc_context3(audio) failed";
            return false;
        }
        audio_context->codec_type = AVMEDIA_TYPE_AUDIO;
        audio_context->codec_id = codec->id;
        audio_context->sample_fmt = SelectSampleFormat(codec);
        audio_context->sample_rate = config.output_audio_sample_rate;
        audio_context->bit_rate = config.audio_bitrate;
        audio_context->time_base = AVRational{1, config.output_audio_sample_rate};
        SetChannelLayout(audio_context, config.output_audio_channels);
        if (format_context->oformat->flags & AVFMT_GLOBALHEADER) {
            audio_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        const int open_result = avcodec_open2(audio_context, codec, nullptr);
        if (open_result < 0) {
            error = "open AAC encoder: " + AvError(open_result);
            return false;
        }

        audio_stream = avformat_new_stream(format_context, nullptr);
        if (!audio_stream) {
            error = "avformat_new_stream(audio) failed";
            return false;
        }
        audio_stream->time_base = audio_context->time_base;
        const int copy_result = avcodec_parameters_from_context(
            audio_stream->codecpar, audio_context);
        if (copy_result < 0) {
            error = "copy audio codec parameters: " + AvError(copy_result);
            return false;
        }
        audio_stream->codecpar->codec_tag = 0;

#if LIBAVUTIL_VERSION_MAJOR >= 57
        AVChannelLayout input_layout;
        av_channel_layout_default(&input_layout, config.input_audio_channels);
        const int swr_result = swr_alloc_set_opts2(
            &swr_context,
            &audio_context->ch_layout,
            audio_context->sample_fmt,
            audio_context->sample_rate,
            &input_layout,
            AV_SAMPLE_FMT_FLT,
            config.input_audio_sample_rate,
            0, nullptr);
        av_channel_layout_uninit(&input_layout);
        if (swr_result < 0) {
            error = "swr_alloc_set_opts2: " + AvError(swr_result);
            return false;
        }
#else
        swr_context = swr_alloc_set_opts(
            nullptr,
            audio_context->channel_layout,
            audio_context->sample_fmt,
            audio_context->sample_rate,
            av_get_default_channel_layout(config.input_audio_channels),
            AV_SAMPLE_FMT_FLT,
            config.input_audio_sample_rate,
            0, nullptr);
#endif
        if (!swr_context) {
            error = "swr allocation failed";
            return false;
        }
        const int swr_init_result = swr_init(swr_context);
        if (swr_init_result < 0) {
            error = "swr_init: " + AvError(swr_init_result);
            return false;
        }
        const int fifo_size = std::max(1024, audio_context->frame_size * 4);
        audio_fifo = av_audio_fifo_alloc(
            audio_context->sample_fmt,
            ChannelCount(audio_context), fifo_size);
        if (!audio_fifo) {
            error = "av_audio_fifo_alloc failed";
            return false;
        }
        return true;
    }

    bool Initialize(std::string& error) {
        EnsureNetworkInitialized();
        protocol = ResolveProtocol(config);
        const int alloc_result = avformat_alloc_output_context2(
            &format_context, nullptr, FormatName(protocol), config.url.c_str());
        if (alloc_result < 0 || !format_context) {
            error = "allocate output context: " + AvError(alloc_result);
            return false;
        }
        format_context->interrupt_callback.callback = &Impl::InterruptIo;
        format_context->interrupt_callback.opaque = this;
        if (!OpenVideo(error) || !OpenAudio(error)) return false;

        AVDictionary* io_options = nullptr;
        if (config.io_timeout_ms > 0) {
            const auto timeout_us = std::to_string(
                static_cast<int64_t>(config.io_timeout_ms) * 1000);
            av_dict_set(&io_options, "rw_timeout", timeout_us.c_str(), 0);
        }
        if (!(format_context->oformat->flags & AVFMT_NOFILE)) {
            ArmIoTimeout();
            const int io_result = avio_open2(
                &format_context->pb, config.url.c_str(), AVIO_FLAG_WRITE,
                &format_context->interrupt_callback, &io_options);
            DisarmIoTimeout();
            if (io_result < 0) {
                av_dict_free(&io_options);
                error = "open output URL: " + AvError(io_result);
                return false;
            }
        }
        av_dict_free(&io_options);

        AVDictionary* header_options = nullptr;
        if (protocol == StreamProtocol::RTSP && config.rtsp_tcp) {
            av_dict_set(&header_options, "rtsp_transport", "tcp", 0);
        }
        if (protocol == StreamProtocol::RTMP) {
            av_dict_set(&header_options, "flvflags", "no_duration_filesize", 0);
        }
        ArmIoTimeout();
        const int header_result = avformat_write_header(
            format_context, &header_options);
        DisarmIoTimeout();
        av_dict_free(&header_options);
        if (header_result < 0) {
            error = "write stream header: " + AvError(header_result);
            return false;
        }
        header_written = true;
        return true;
    }

    bool WritePackets(AVCodecContext* codec_context,
                      AVStream* stream,
                      bool video,
                      std::string& error) {
        AVPacket* packet = av_packet_alloc();
        if (!packet) {
            error = "av_packet_alloc failed";
            return false;
        }
        while (true) {
            const int result = avcodec_receive_packet(codec_context, packet);
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) break;
            if (result < 0) {
                error = "receive encoded packet: " + AvError(result);
                av_packet_free(&packet);
                return false;
            }
            av_packet_rescale_ts(packet, codec_context->time_base,
                                 stream->time_base);
            packet->stream_index = stream->index;
            ArmIoTimeout();
            const int write_result = av_interleaved_write_frame(
                format_context, packet);
            DisarmIoTimeout();
            av_packet_unref(packet);
            if (write_result < 0) {
                error = "write muxed packet: " + AvError(write_result);
                av_packet_free(&packet);
                return false;
            }
            packets_written.fetch_add(1, std::memory_order_relaxed);
            if (video) {
                video_frames_encoded.fetch_add(1, std::memory_order_relaxed);
            } else {
                audio_frames_encoded.fetch_add(1, std::memory_order_relaxed);
            }
        }
        av_packet_free(&packet);
        return true;
    }

    bool SendFrame(AVCodecContext* codec_context,
                   AVStream* stream,
                   AVFrame* frame,
                   bool video,
                   std::string& error) {
        const int send_result = avcodec_send_frame(codec_context, frame);
        if (send_result < 0 && send_result != AVERROR_EOF) {
            error = "send frame to encoder: " + AvError(send_result);
            return false;
        }
        return WritePackets(codec_context, stream, video, error);
    }

    bool EncodeVideo(const VideoItem& item, std::string& error) {
        cv::Mat resized;
        const cv::Mat* source = &item.frame;
        if (item.frame.cols != config.width || item.frame.rows != config.height) {
            cv::resize(item.frame, resized, cv::Size(config.width, config.height));
            source = &resized;
        }
        if (source->type() != CV_8UC3) {
            error = "publisher video frame must be CV_8UC3 BGR";
            return false;
        }
        const int writable = av_frame_make_writable(video_frame);
        if (writable < 0) {
            error = "video frame is not writable: " + AvError(writable);
            return false;
        }
        const uint8_t* input_data[1] = {source->data};
        int input_linesize[1] = {static_cast<int>(source->step)};
        sws_scale(sws_context, input_data, input_linesize, 0,
                  config.height, video_frame->data, video_frame->linesize);
        int64_t pts = av_rescale_q(item.pts_ms, AVRational{1, 1000},
                                   video_context->time_base);
        if (last_video_pts != AV_NOPTS_VALUE && pts <= last_video_pts) {
            pts = last_video_pts + 1;
        }
        last_video_pts = pts;
        video_frame->pts = pts;
        return SendFrame(video_context, video_stream, video_frame, true, error);
    }

    bool EncodeAudioFrames(bool flush_partial, std::string& error) {
        const int frame_size = audio_context->frame_size > 0
            ? audio_context->frame_size : 1024;
        while (av_audio_fifo_size(audio_fifo) >= frame_size
               || (flush_partial && av_audio_fifo_size(audio_fifo) > 0)) {
            const int available = av_audio_fifo_size(audio_fifo);
            const int read_samples = std::min(available, frame_size);
            AVFrame* frame = av_frame_alloc();
            if (!frame) {
                error = "av_frame_alloc(audio) failed";
                return false;
            }
            frame->nb_samples = frame_size;
            frame->format = audio_context->sample_fmt;
            frame->sample_rate = audio_context->sample_rate;
            CopyChannelLayout(frame, audio_context);
            const int buffer_result = av_frame_get_buffer(frame, 0);
            if (buffer_result < 0) {
                error = "audio frame buffer: " + AvError(buffer_result);
                av_frame_free(&frame);
                return false;
            }
            av_samples_set_silence(frame->data, 0, frame_size,
                ChannelCount(audio_context), audio_context->sample_fmt);
            const int read_result = av_audio_fifo_read(
                audio_fifo, reinterpret_cast<void**>(frame->data), read_samples);
            if (read_result < read_samples) {
                error = "av_audio_fifo_read returned too few samples";
                av_frame_free(&frame);
                return false;
            }
            frame->pts = audio_next_pts;
            audio_next_pts += frame_size;
            const bool ok = SendFrame(
                audio_context, audio_stream, frame, false, error);
            av_frame_free(&frame);
            if (!ok) return false;
        }
        return true;
    }

    bool EncodeAudio(const AudioItem& item, std::string& error) {
        const int input_samples = static_cast<int>(
            item.samples.size()
            / static_cast<size_t>(config.input_audio_channels));
        if (input_samples <= 0) return true;
        if (!audio_pts_initialized) {
            audio_next_pts = av_rescale_q(item.pts_ms, AVRational{1, 1000},
                AVRational{1, audio_context->sample_rate});
            audio_pts_initialized = true;
        }
        const int output_capacity = static_cast<int>(av_rescale_rnd(
            swr_get_delay(swr_context, config.input_audio_sample_rate)
                + input_samples,
            audio_context->sample_rate,
            config.input_audio_sample_rate,
            AV_ROUND_UP));
        uint8_t** converted = nullptr;
        int converted_linesize = 0;
        const int alloc_result = av_samples_alloc_array_and_samples(
            &converted, &converted_linesize, ChannelCount(audio_context),
            output_capacity, audio_context->sample_fmt, 0);
        if (alloc_result < 0) {
            error = "allocate resampled audio: " + AvError(alloc_result);
            return false;
        }
        const uint8_t* input_data[1] = {
            reinterpret_cast<const uint8_t*>(item.samples.data())};
        const int converted_samples = swr_convert(
            swr_context, converted, output_capacity,
            input_data, input_samples);
        if (converted_samples < 0) {
            error = "audio resample: " + AvError(converted_samples);
            av_freep(&converted[0]);
            av_freep(&converted);
            return false;
        }
        const int required = av_audio_fifo_size(audio_fifo) + converted_samples;
        if (av_audio_fifo_realloc(audio_fifo, required) < 0) {
            error = "grow audio FIFO failed";
            av_freep(&converted[0]);
            av_freep(&converted);
            return false;
        }
        const int written = av_audio_fifo_write(
            audio_fifo, reinterpret_cast<void**>(converted), converted_samples);
        av_freep(&converted[0]);
        av_freep(&converted);
        if (written < converted_samples) {
            error = "audio FIFO write returned too few samples";
            return false;
        }
        return EncodeAudioFrames(false, error);
    }

    void Run() {
        while (true) {
            VideoItem video;
            AudioItem audio;
            bool use_video = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]() {
                    return failed || closing
                        || (!video_queue.empty() && !audio_queue.empty());
                });
                if (failed) break;
                if (closing && video_queue.empty() && audio_queue.empty()) break;
                if (!closing
                    && (video_queue.empty() || audio_queue.empty())) {
                    continue;
                }
                use_video = audio_queue.empty()
                    || (!video_queue.empty()
                        && video_queue.front().pts_ms
                           < audio_queue.front().pts_ms);
                if (use_video) {
                    video = std::move(video_queue.front());
                    video_queue.pop_front();
                } else {
                    audio = std::move(audio_queue.front());
                    audio_queue.pop_front();
                }
            }
            cv.notify_all();
            std::string error;
            const bool ok = use_video
                ? EncodeVideo(video, error)
                : EncodeAudio(audio, error);
            if (!ok) {
                SetFailure(error);
                break;
            }
        }

        if (!failed) {
            std::string error;
            if (!EncodeAudioFrames(true, error)
                || !SendFrame(audio_context, audio_stream, nullptr, false, error)
                || !SendFrame(video_context, video_stream, nullptr, true, error)) {
                SetFailure(error);
            } else if (header_written) {
                ArmIoTimeout();
                const int result = av_write_trailer(format_context);
                DisarmIoTimeout();
                if (result < 0) SetFailure("write trailer: " + AvError(result));
            }
        }
    }

    void Cleanup() {
        av_frame_free(&video_frame);
        if (audio_fifo) av_audio_fifo_free(audio_fifo);
        audio_fifo = nullptr;
        swr_free(&swr_context);
        sws_freeContext(sws_context);
        sws_context = nullptr;
        avcodec_free_context(&audio_context);
        avcodec_free_context(&video_context);
        if (format_context) {
            if (!(format_context->oformat->flags & AVFMT_NOFILE)
                && format_context->pb) {
                avio_closep(&format_context->pb);
            }
            avformat_free_context(format_context);
            format_context = nullptr;
        }
        video_stream = nullptr;
        audio_stream = nullptr;
        header_written = false;
    }
};

StreamPublisher::StreamPublisher() : impl_(std::make_unique<Impl>()) {}

StreamPublisher::~StreamPublisher() {
    std::string ignored;
    Close(false, ignored);
}

bool StreamPublisher::Open(const StreamPublisherConfig& config,
                           std::string& error) {
    error.clear();
    if (config.url.empty() || config.width <= 0 || config.height <= 0
        || config.width % 2 != 0 || config.height % 2 != 0
        || config.fps <= 0.0 || config.input_audio_sample_rate <= 0
        || config.output_audio_sample_rate <= 0
        || config.input_audio_channels <= 0
        || config.output_audio_channels <= 0
        || config.max_video_queue == 0 || config.max_audio_queue == 0) {
        error = "invalid stream publisher configuration";
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->opened) {
            error = "stream publisher is already open";
            return false;
        }
        impl_->config = config;
        impl_->closing = false;
        impl_->failed = false;
        impl_->last_error.clear();
        impl_->video_queue.clear();
        impl_->audio_queue.clear();
        impl_->last_video_pts = AV_NOPTS_VALUE;
        impl_->audio_next_pts = 0;
        impl_->audio_pts_initialized = false;
        impl_->video_frames_in.store(0, std::memory_order_relaxed);
        impl_->video_frames_encoded.store(0, std::memory_order_relaxed);
        impl_->video_frames_dropped.store(0, std::memory_order_relaxed);
        impl_->audio_samples_in.store(0, std::memory_order_relaxed);
        impl_->audio_frames_encoded.store(0, std::memory_order_relaxed);
        impl_->packets_written.store(0, std::memory_order_relaxed);
        impl_->io_deadline_us.store(0, std::memory_order_relaxed);
    }
    if (!impl_->Initialize(error)) {
        impl_->Cleanup();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->last_error = error;
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->opened = true;
    }
    impl_->worker = std::thread([this]() { impl_->Run(); });
    return true;
}

bool StreamPublisher::PushVideo(const cv::Mat& bgr_frame,
                                int64_t pts_ms,
                                std::string& error) {
    error.clear();
    if (bgr_frame.empty() || bgr_frame.type() != CV_8UC3 || pts_ms < 0) {
        error = "invalid BGR video frame or PTS";
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->opened || impl_->closing || impl_->failed) {
        error = impl_->last_error.empty()
            ? "stream publisher is not accepting video" : impl_->last_error;
        return false;
    }
    if (impl_->video_queue.size() >= impl_->config.max_video_queue) {
        impl_->video_queue.pop_front();
        impl_->video_frames_dropped.fetch_add(1, std::memory_order_relaxed);
    }
    impl_->video_queue.push_back(Impl::VideoItem{bgr_frame.clone(), pts_ms});
    impl_->video_frames_in.fetch_add(1, std::memory_order_relaxed);
    impl_->cv.notify_all();
    return true;
}

bool StreamPublisher::PushAudio(const std::vector<float>& interleaved_pcm,
                                int64_t pts_ms,
                                std::string& error) {
    error.clear();
    if (interleaved_pcm.empty() || pts_ms < 0) {
        error = "invalid PCM audio or PTS";
        return false;
    }
    std::unique_lock<std::mutex> lock(impl_->mutex);
    const auto wait_deadline = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(
            impl_->config.max_audio_push_wait_ms > 0
                ? impl_->config.max_audio_push_wait_ms : 0);
    const bool ready = impl_->cv.wait_until(lock, wait_deadline, [&]() {
        return !impl_->opened || impl_->closing || impl_->failed
            || impl_->audio_queue.size() < impl_->config.max_audio_queue;
    });
    // ready==false 表示 wait_until 超时：队列持续满且未收到停止信号，
    // 视为慢消费者不可恢复，置 failed 并返回，避免会话停止时无限阻塞。
    if (!ready
        || !impl_->opened || impl_->closing || impl_->failed) {
        if (!ready && impl_->opened && !impl_->closing && !impl_->failed) {
            std::string timeout_error = "audio push timed out after "
                + std::to_string(impl_->config.max_audio_push_wait_ms)
                + "ms (slow consumer)";
            lock.unlock();
            impl_->SetFailure(timeout_error);
            error = timeout_error;
            return false;
        }
        error = impl_->last_error.empty()
            ? "stream publisher is not accepting audio" : impl_->last_error;
        return false;
    }
    if (interleaved_pcm.size()
        % static_cast<size_t>(impl_->config.input_audio_channels) != 0) {
        error = "PCM sample count is not divisible by input channel count";
        return false;
    }
    impl_->audio_queue.push_back(
        Impl::AudioItem{interleaved_pcm, pts_ms});
    impl_->audio_samples_in.fetch_add(
        static_cast<int64_t>(interleaved_pcm.size()
            / static_cast<size_t>(impl_->config.input_audio_channels)),
        std::memory_order_relaxed);
    lock.unlock();
    impl_->cv.notify_all();
    return true;
}

bool StreamPublisher::Close(bool drain, std::string& error) {
    error.clear();
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->opened) {
            if (impl_->failed) error = impl_->last_error;
            return !impl_->failed;
        }
        impl_->closing = true;
        if (!drain) {
            impl_->video_queue.clear();
            impl_->audio_queue.clear();
        }
    }
    impl_->cv.notify_all();
    if (impl_->worker.joinable()) impl_->worker.join();
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->opened = false;
        if (impl_->failed) error = impl_->last_error;
    }
    impl_->Cleanup();
    return error.empty();
}

bool StreamPublisher::IsOpen() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->opened && !impl_->closing && !impl_->failed;
}

std::string StreamPublisher::GetLastError() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->last_error;
}

StreamPublisherMetrics StreamPublisher::GetMetrics() const {
    StreamPublisherMetrics metrics;
    metrics.video_frames_in = impl_->video_frames_in.load();
    metrics.video_frames_encoded = impl_->video_frames_encoded.load();
    metrics.video_frames_dropped = impl_->video_frames_dropped.load();
    metrics.audio_samples_in = impl_->audio_samples_in.load();
    metrics.audio_frames_encoded = impl_->audio_frames_encoded.load();
    metrics.packets_written = impl_->packets_written.load();
    return metrics;
}

}  // namespace media
}  // namespace digital_human
