// ============================================================================
// perf_benchmark.cpp
//
// 端到端性能基准：用 face.jpg + zw.mp3 跑完整 Pipeline，
// 测量：
//   - 实际输出帧率（fps）
//   - 端到端延迟分布（p50/p95/p99）
//   - VideoProcessor 缓存命中率
//   - 推理/渲染平均耗时
// 用法: ./perf_benchmark [assets_dir] [target_fps]
// ============================================================================
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/pipeline.h"
#include "core/packet.h"
#include "audio/audio_loader.h"

using namespace digital_human;
using namespace std::chrono;

struct PerfStats {
    int64_t frames_out       = 0;
    int64_t frames_in        = 0;
    double   wall_time_ms    = 0.0;
    double   avg_infer_ms    = 0.0;
    double   avg_render_ms   = 0.0;
    double   avg_video_ms    = 0.0;
    double   avg_audio_ms    = 0.0;
    double   actual_fps      = 0.0;
    std::vector<double> latencies_ms;  // 端到端延迟：output_pts - input_pts
};

// 计算 p 分位
double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}

int main(int argc, char* argv[]) {
    std::string assets_dir = (argc > 1) ? argv[1] : ASSETS_DIR;
    double target_fps = (argc > 2) ? std::stod(argv[2]) : 30.0;

    std::cout << "==============================================\n";
    std::cout << "  Pipeline 端到端性能基准\n";
    std::cout << "==============================================\n";
    std::cout << "  assets_dir = " << assets_dir << "\n";
    std::cout << "  target_fps = " << target_fps << "\n";
    std::cout << "==============================================\n\n";

    // ---- 加载音频 ----
    std::string mp3_path = assets_dir + "/zw.mp3";
    audio::AudioLoader loader;
    audio::AudioData audio_data;
    try {
        audio_data = loader.load(mp3_path);
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] 加载音频失败: " << e.what() << "\n";
        return 1;
    }
    std::cout << "[INFO] 音频: " << audio_data.duration << "s @ "
              << audio_data.sampleRate << "Hz, samples="
              << audio_data.samples.size() << "\n";

    // ---- 加载人脸图像 ----
    std::string face_path = assets_dir + "/face.jpg";
    cv::Mat face_img = cv::imread(face_path);
    if (face_img.empty()) {
        std::cerr << "[FAIL] 加载人脸失败: " << face_path << "\n";
        return 1;
    }
    std::cout << "[INFO] 人脸: " << face_img.cols << "x" << face_img.rows << "\n\n";

    // ---- 配置 Pipeline ----
    core::PipelineConfig cfg;
    cfg.target_fps = target_fps;
    cfg.audio_sample_rate = audio_data.sampleRate;
    cfg.audio_frame_size  = 400;
    cfg.audio_hop_size    = 160;

    core::Pipeline pipeline;
    if (!pipeline.Init(cfg)) {
        std::cerr << "[FAIL] Pipeline Init 失败\n";
        return 1;
    }

    // 设置 dlib 人脸关键点模型路径
    std::string model_dir = std::string(PROJECT_SOURCE_DIR) + "/models";
    std::string landmark_path = model_dir + "/face";
    pipeline.SetLandmarkModelPath(landmark_path);
    std::cout << "[INFO] 人脸关键点模型: " << landmark_path << "\n";

    // 初始化 Wav2Lip 推理模型
    if (!pipeline.InitModelInferencer(model_dir)) {
        std::cerr << "[FAIL] ModelInferencer Init 失败\n";
        return 1;
    }
    std::cout << "[INFO] Wav2Lip 模型目录: " << model_dir << "\n\n";

    // ---- 输出帧统计 ----
    PerfStats stats;
    auto t_start = steady_clock::now();
    auto last_report = t_start;

    // 启动 Pipeline
    if (!pipeline.Start()) {
        std::cerr << "[FAIL] Pipeline Start 失败\n";
        return 1;
    }

    // ---- 喂入视频帧（以 30fps 重复喂同一张人脸，模拟拟合图片场景） ----
    // 同时喂入音频
    const double video_fps = target_fps;
    const double video_interval_ms = 1000.0 / video_fps;
    const int64_t total_video_frames =
        static_cast<int64_t>(audio_data.duration * video_fps);

    // 视频生产者线程
    std::thread video_thread([&]() {
        for (int64_t i = 0; i < total_video_frames; ++i) {
            int64_t pts_ms = static_cast<int64_t>(i * video_interval_ms);
            pipeline.PushVideo(face_img, pts_ms);
            stats.frames_in++;
            // 模拟按帧率生产，给 Pipeline 充足时间消费
            std::this_thread::sleep_for(
                microseconds(static_cast<int>(video_interval_ms * 1000 * 0.5)));
        }
        pipeline.MarkVideoEOS();
    });

    // 音频生产者：非重叠分块推送（AudioProcessor 环形窗按 hop 重分帧，
    // 重叠推送会导致样本在窗内重复、音频时间轴被拉长 ~2.5×）
    std::thread audio_thread([&]() {
        const int chunk = 1600;  // 100ms @16kHz
        int64_t offset = 0;
        int64_t total  = static_cast<int64_t>(audio_data.samples.size());
        while (offset < total) {
            int64_t end = std::min(offset + chunk, total);
            std::vector<float> data(
                audio_data.samples.begin() + offset,
                audio_data.samples.begin() + end);
            int64_t pts_ms = static_cast<int64_t>(
                offset * 1000.0 / cfg.audio_sample_rate);
            pipeline.PushAudio(data, pts_ms);
            offset = end;
        }
        pipeline.MarkAudioEOS();
    });

    // ---- 消费者：测量端到端延迟和实际 fps ----
    std::thread consumer_thread([&]() {
        core::OutputFramePacket out;
        while (true) {
            if (!pipeline.GetOutputFrame(out, 500)) {
                if (!pipeline.IsRunning()) break;
                continue;
            }
            if (out.header.IsEOS()) break;
            if (out.header.IsSkip() || out.payload.empty()) continue;

            stats.frames_out++;
            // 端到端延迟：当前墙钟时间 - 帧的 PTS
            double now_ms = duration<double, std::milli>(
                steady_clock::now() - t_start).count();
            double e2e = now_ms - static_cast<double>(out.header.pts_ms);
            stats.latencies_ms.push_back(e2e);

            // 每 50 帧打印一次进度
            if (stats.frames_out % 50 == 0) {
                auto now = steady_clock::now();
                double elapsed = duration<double, std::milli>(now - last_report).count();
                double fps = 50000.0 / std::max(elapsed, 0.001);
                std::cout << "  [PROGRESS] 输出 " << stats.frames_out
                          << " 帧, 最近 50 帧 fps=" << fps
                          << ", e2e=" << e2e << "ms"
                          << ", cost=" << out.header.cost_ms << "ms\n";
                last_report = now;
            }
        }
    });

    // 等待所有线程
    video_thread.join();
    audio_thread.join();
    consumer_thread.join();
    pipeline.Stop();

    auto t_end = steady_clock::now();
    stats.wall_time_ms = duration<double, std::milli>(t_end - t_start).count();

    // ---- 聚合 Pipeline metrics ----
    auto m = pipeline.GetMetrics();
    stats.avg_infer_ms  = m.avg_inference_ms;
    stats.avg_render_ms = m.avg_output_ms;
    stats.avg_video_ms  = m.avg_video_process_ms;
    stats.avg_audio_ms  = m.avg_audio_process_ms;
    stats.actual_fps    = (stats.wall_time_ms > 0)
                          ? stats.frames_out * 1000.0 / stats.wall_time_ms
                          : 0.0;

    // ---- 输出报告 ----
    std::cout << "\n==============================================\n";
    std::cout << "  性能基准报告\n";
    std::cout << "==============================================\n";
    std::cout << "  目标帧率:        " << target_fps << " fps\n";
    std::cout << "  输入帧数:        " << stats.frames_in << "\n";
    std::cout << "  输出帧数:        " << stats.frames_out << "\n";
    std::cout << "  丢弃/跳过帧:     " << m.frames_dropped << " / " << m.frames_skipped << "\n";
    std::cout << "  总墙钟耗时:      " << stats.wall_time_ms << " ms ("
              << stats.wall_time_ms / 1000.0 << " s)\n";
    std::cout << "  实际输出帧率:    " << stats.actual_fps << " fps\n";
    std::cout << "  ------------------------------------------\n";
    std::cout << "  平均音频处理:    " << stats.avg_audio_ms << " ms\n";
    std::cout << "  平均视频处理:    " << stats.avg_video_ms << " ms\n";
    std::cout << "  平均推理耗时:    " << stats.avg_infer_ms << " ms\n";
    std::cout << "  平均渲染耗时:    " << stats.avg_render_ms << " ms\n";
    std::cout << "  ------------------------------------------\n";

    if (!stats.latencies_ms.empty()) {
        std::cout << "  端到端延迟分布 (ms):\n";
        std::cout << "    min:   " << stats.latencies_ms.front() << "\n";
        std::cout << "    p50:   " << percentile(stats.latencies_ms, 0.50) << "\n";
        std::cout << "    p95:   " << percentile(stats.latencies_ms, 0.95) << "\n";
        std::cout << "    p99:   " << percentile(stats.latencies_ms, 0.99) << "\n";
        std::cout << "    max:   " << stats.latencies_ms.back() << "\n";
        double sum = 0;
        for (double v : stats.latencies_ms) sum += v;
        std::cout << "    avg:   " << sum / stats.latencies_ms.size() << "\n";
    }

    std::cout << "==============================================\n";

    // 与音频时长比较：实际帧率 = 输出帧数 / 音频时长
    double audio_dur_s = audio_data.duration;
    double fps_by_audio = stats.frames_out / audio_dur_s;
    std::cout << "  按音频时长计算帧率: " << fps_by_audio << " fps\n";
    std::cout << "  (音频时长: " << audio_dur_s << " s)\n";
    std::cout << "==============================================\n";

    return 0;
}
