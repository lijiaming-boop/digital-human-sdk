#include <iostream>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <audio_file_path>" << std::endl;
        return -1;
    }

    const char* filename = argv[1];

    // 1. 打开输入文件 (Open Input)
    // ------------------------------------
    AVFormatContext* format_ctx = nullptr;
    if (avformat_open_input(&format_ctx, filename, nullptr, nullptr) < 0) {
        std::cerr << "[Error] Could not open file: " << filename << std::endl;
        return -1;
    }
    std::cout << "[Success] File opened. Duration: " << format_ctx->duration / AV_TIME_BASE << "s" << std::endl;

    // 检索流信息
    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        std::cerr << "[Error] Could not find stream info" << std::endl;
        avformat_close_input(&format_ctx);
        return -1;
    }

    // 2. 查找音频流 (Find Audio Stream)
    // ------------------------------------
    int audio_stream_idx = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
            break;
        }
    }

    if (audio_stream_idx == -1) {
        std::cerr << "[Error] No audio stream found!" << std::endl;
        avformat_close_input(&format_ctx);
        return -1;
    }
    std::cout << "[Success] Found audio stream at index: " << audio_stream_idx << std::endl;

    // 3. 初始化解码器 (Init Decoder)
    // ------------------------------------
    AVCodecParameters* codec_params = format_ctx->streams[audio_stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) {
        std::cerr << "[Error] Decoder not found!" << std::endl;
        return -1;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codec_params);

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "[Error] Could not open codec" << std::endl;
        return -1;
    }

    std::cout << "[Success] Decoder initialized: " << codec->name << std::endl;
    std::cout << "    -> Sample Rate: " << codec_ctx->sample_rate << " Hz" << std::endl;
    std::cout << "    -> Channels: " << codec_ctx->channels << std::endl;

    // 4. 读取与解码循环 (Decode Loop)
    // ------------------------------------
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    int frame_count = 0;
    int max_frames_to_decode = 10; // 为了测试，我们只解码前10帧

    std::cout << "\n--- Start Decoding (First 10 frames) ---" << std::endl;

    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == audio_stream_idx) {
            // 发送 Packet 到解码器
            int ret = avcodec_send_packet(codec_ctx, packet);
            if (ret < 0) {
                std::cerr << "Error sending packet for decoding" << std::endl;
                break;
            }

            // 从解码器接收 Frame
            while (ret >= 0) {
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    std::cerr << "Error during decoding" << std::endl;
                    break;
                }
                
                // 打印信息
                std::cout << "Frame " << ++frame_count << ": samples=" << frame->nb_samples 
                          << ", pts=" << frame->pts << std::endl;
            }
        }
        av_packet_unref(packet);
        if (frame_count >= max_frames_to_decode) {
            break;
        }
    }
    
    // 资源释放
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);

    std::cout << "\n=== Test Finished Successfully ===" << std::endl;
    return 0;
}