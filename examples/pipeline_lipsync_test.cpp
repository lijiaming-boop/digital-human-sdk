/**
 * @file pipeline_lipsync_test.cpp
 * @brief Pipeline 端到端口型同步 + 帧率验证测试
 *
 * 走完整多线程 Pipeline（face.jpg + 音频 → 输出视频帧），验证：
 *   1. 帧率：输出内容帧率 / 生成速度（相对实时倍率）/ 各阶段耗时
 *   2. 口型：嘴部 ROI 帧间变化量（mouth openness）与音频能量的相关性
 *   3. 产物：输出帧序列 → MP4（ffmpeg 合成音轨）
 *
 * 用法:
 *   ./bin/pipeline_lipsync_test [assets_dir] [seconds] [target_fps] [audio_name]
 *   默认: assets_dir=ASSETS_DIR, seconds=30, target_fps=25, audio=zw.mp3
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/pipeline.h"
#include "core/packet.h"
#include "core/face_detector.h"
#include "audio/audio_loader.h"

using namespace digital_human;
using namespace std::chrono;

// ============================================================================
// 工具函数
// ============================================================================

/// @brief Pearson 相关系数
static double PearsonCorr(const std::vector<double>& x,
                          const std::vector<double>& y) {
    size_t n = std::min(x.size(), y.size());
    if (n < 3) return 0.0;
    double mx = 0, my = 0;
    for (size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sxy = 0, sxx = 0, syy = 0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mx, dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if (sxx < 1e-12 || syy < 1e-12) return 0.0;
    return sxy / std::sqrt(sxx * syy);
}

/// @brief 嘴部 ROI 灰度图与原图的平均绝对差（mouth openness 代理指标）
static double MouthOpenness(const cv::Mat& frame_gray,
                            const cv::Mat& orig_gray_roi,
                            const cv::Rect& roi) {
    cv::Mat diff;
    cv::absdiff(frame_gray(roi), orig_gray_roi, diff);
    return cv::mean(diff)[0];
}

int main(int argc, char* argv[]) {
    std::string assets_dir = (argc > 1) ? argv[1] : ASSETS_DIR;
    double      seconds    = (argc > 2) ? std::stod(argv[2]) : 30.0;
    double      target_fps = (argc > 3) ? std::stod(argv[3]) : 25.0;
    std::string audio_name = (argc > 4) ? argv[4] : "zw.mp3";
    int         infer_threads = (argc > 5) ? std::stoi(argv[5]) : 0;
    // 帧输出根目录（默认 assets/output/pipeline_lipsync；
    // WSL 下建议指向原生文件系统路径，避免 drvfs 写盘拖慢消费线程）
    std::string out_dir = (argc > 6) ? argv[6]
                                     : assets_dir + "/output/pipeline_lipsync";

    std::string model_dir = std::string(PROJECT_SOURCE_DIR) + "/models";
    std::string frames_dir = out_dir + "/frames";
    std::filesystem::create_directories(frames_dir);

    std::cout << "==============================================\n";
    std::cout << "  Pipeline 端到端口型同步 + 帧率验证\n";
    std::cout << "==============================================\n";
    std::cout << "  assets:  " << assets_dir << "\n";
    std::cout << "  时长上限: " << seconds << "s, 目标帧率: " << target_fps
              << "fps, 音频: " << audio_name << "\n\n";

    // ====================================================================
    // 1. 加载音频（截到 seconds）
    // ====================================================================
    audio::AudioLoader loader;
    audio::AudioData audio;
    try {
        audio = loader.load(assets_dir + "/" + audio_name);
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] 音频加载失败: " << e.what() << "\n";
        return 1;
    }
    size_t max_samples = static_cast<size_t>(seconds * audio.sampleRate)
                       * audio.channels;
    if (audio.samples.size() > max_samples) {
        audio.samples.resize(max_samples);
        audio.duration = seconds;
    }
    double content_s = static_cast<double>(audio.samples.size())
                     / audio.channels / audio.sampleRate;
    std::cout << "[音频] " << content_s << "s @ " << audio.sampleRate
              << "Hz, ch=" << audio.channels << "\n";

    // ====================================================================
    // 2. 加载人脸 + 嘴部 ROI（68 关键点的 48-67 号点包围盒）
    // ====================================================================
    cv::Mat face_img = cv::imread(assets_dir + "/face.jpg");
    if (face_img.empty()) {
        std::cerr << "[FAIL] 人脸加载失败\n";
        return 1;
    }
    std::cout << "[人脸] " << face_img.cols << "x" << face_img.rows << "\n";

    core::FaceDetector detector;
    detector.loadModel(model_dir + "/shape_predictor_68_face_landmarks.dat");
    auto faces = detector.detect(face_img);
    if (faces.empty()) {
        std::cerr << "[FAIL] 未检测到人脸\n";
        return 1;
    }
    auto landmarks = detector.getLandmarks(face_img, faces[0]);
    int mx0 = face_img.cols, my0 = face_img.rows, mx1 = 0, my1 = 0;
    for (int i = 48; i <= 67 && i < (int)landmarks.size(); ++i) {
        mx0 = std::min(mx0, landmarks[i].x); my0 = std::min(my0, landmarks[i].y);
        mx1 = std::max(mx1, landmarks[i].x); my1 = std::max(my1, landmarks[i].y);
    }
    // 外扩 40% 覆盖唇部活动区
    int mw = mx1 - mx0, mh = my1 - my0;
    cv::Rect mouth_roi(mx0 - (int)(mw * 0.4), my0 - (int)(mh * 0.6),
                       (int)(mw * 1.8), (int)(mh * 2.2));
    mouth_roi &= cv::Rect(0, 0, face_img.cols, face_img.rows);
    std::cout << "[嘴部 ROI] " << mouth_roi << "\n";

    cv::Mat orig_gray;
    cv::cvtColor(face_img, orig_gray, cv::COLOR_BGR2GRAY);
    cv::Mat orig_mouth = orig_gray(mouth_roi).clone();

    // ====================================================================
    // 3. 配置并启动 Pipeline（批处理模式：关闭帧间隔调节跑满吞吐）
    // ====================================================================
    core::PipelineConfig cfg;
    cfg.target_fps        = target_fps;
    cfg.audio_sample_rate = audio.sampleRate;
    cfg.audio_frame_size  = 400;
    cfg.audio_hop_size    = 160;
    cfg.enable_frame_pacing = false;   // 离线生成：最大吞吐

    core::Pipeline pipeline;
    if (!pipeline.Init(cfg)) {
        std::cerr << "[FAIL] Pipeline Init\n";
        return 1;
    }
    pipeline.SetLandmarkModelPath(model_dir
                                  + "/shape_predictor_68_face_landmarks.dat");
    if (!pipeline.InitModelInferencer(model_dir)) {
        std::cerr << "[FAIL] 模型初始化\n";
        return 1;
    }
    if (infer_threads > 0) {
        pipeline.SetInferenceThreads(infer_threads);
        std::cout << "[配置] 推理线程数: " << infer_threads << "\n";
    }

    // ====================================================================
    // 4. 生产者 / 消费者线程
    // ====================================================================
    const int64_t total_video_frames =
        static_cast<int64_t>(content_s * target_fps);
    const double interval_ms = 1000.0 / target_fps;

    int64_t frames_in = 0, frames_out = 0;
    std::vector<double> openness_series;

    auto t_start = steady_clock::now();

    if (!pipeline.Start()) {
        std::cerr << "[FAIL] Pipeline Start\n";
        return 1;
    }

    std::thread video_thread([&]() {
        for (int64_t i = 0; i < total_video_frames; ++i) {
            int64_t pts = static_cast<int64_t>(i * interval_ms);
            pipeline.PushVideo(face_img, pts);
            frames_in++;
        }
        pipeline.MarkVideoEOS();
    });

    std::thread audio_thread([&]() {
        // 非重叠分块推送：AudioProcessor 内部环形窗按 hop 重分帧，
        // 重叠推送会导致数据在环形窗内重复、时间轴被拉长
        const int chunk = 1600;   // 100ms @16kHz
        int64_t offset = 0;
        int64_t total  = static_cast<int64_t>(audio.samples.size());
        while (offset < total) {
            int64_t end = std::min(offset + chunk, total);
            std::vector<float> data(
                audio.samples.begin() + offset,
                audio.samples.begin() + end);
            int64_t pts_ms = static_cast<int64_t>(
                offset * 1000.0 / cfg.audio_sample_rate);
            pipeline.PushAudio(data, pts_ms);
            offset = end;
        }
        pipeline.MarkAudioEOS();
    });

    int saved = 0;
    std::thread consumer_thread([&]() {
        core::OutputFramePacket out;
        while (true) {
            if (!pipeline.GetOutputFrame(out, 1000)) {
                if (!pipeline.IsRunning()) break;
                continue;
            }
            if (out.header.IsEOS()) break;
            if (out.header.IsSkip() || out.payload.empty()) continue;

            frames_out++;

            // 口型开合度（与原图静态背景的差异）
            cv::Mat gray;
            cv::cvtColor(out.payload, gray, cv::COLOR_BGR2GRAY);
            openness_series.push_back(
                MouthOpenness(gray, orig_mouth, mouth_roi));

            // 保存帧用于合成视频
            char fname[64];
            std::snprintf(fname, sizeof(fname), "f_%05d.jpg", saved);
            cv::imwrite(frames_dir + "/" + fname, out.payload,
                        {cv::IMWRITE_JPEG_QUALITY, 85});
            saved++;

            if (frames_out % 50 == 0) {
                double wall = duration<double, std::milli>(
                    steady_clock::now() - t_start).count();
                std::cout << "  [进度] 输出 " << frames_out << " 帧, wall="
                          << (wall / 1000.0) << "s, 内容="
                          << (frames_out / target_fps) << "s\n";
            }
        }
    });

    video_thread.join();
    audio_thread.join();
    consumer_thread.join();
    pipeline.Stop();

    double wall_s = duration<double>(steady_clock::now() - t_start).count();

    // ====================================================================
    // 5. 音频能量序列（与输出帧对齐）
    //    窗口与 mel 时序窗同宽（window*hop = 160ms），消除开窗错位导致的
    //    相关性稀释；同时输出滑动平均平滑后的相关性（音素尺度 ~200ms）
    // ====================================================================
    std::vector<double> energy_series;
    {
        const int64_t mel_span_samples = static_cast<int64_t>(
            0.160 * audio.sampleRate);   // 与 16 帧 mel 窗同宽 (160ms)
        int64_t hop_samples = static_cast<int64_t>(audio.sampleRate / target_fps);
        for (int64_t i = 0; i < frames_out; ++i) {
            int64_t b = i * hop_samples * audio.channels;
            int64_t e = std::min<int64_t>(b + mel_span_samples * audio.channels,
                                          audio.samples.size());
            if (b >= e) { energy_series.push_back(0.0); continue; }
            double sq = 0;
            for (int64_t k = b; k < e; ++k)
                sq += (double)audio.samples[k] * audio.samples[k];
            energy_series.push_back(std::sqrt(sq / (e - b)));
        }
    }

    // 滑动平均平滑（窗口 5 帧 = 200ms，贴近音素持续时间）
    auto smooth5 = [](const std::vector<double>& v) {
        std::vector<double> out(v.size(), 0.0);
        for (size_t i = 0; i < v.size(); ++i) {
            double s = 0; int n = 0;
            for (int k = -2; k <= 2; ++k) {
                int64_t j = static_cast<int64_t>(i) + k;
                if (j >= 0 && j < (int64_t)v.size()) { s += v[j]; ++n; }
            }
            out[i] = s / n;
        }
        return out;
    };

    // 导出序列 CSV（供离线分析）
    {
        std::string csv_path = out_dir + "/openness_energy.csv";
        FILE* fp = std::fopen(csv_path.c_str(), "w");
        if (fp) {
            std::fprintf(fp, "frame,openness,energy\n");
            for (size_t i = 0; i < openness_series.size(); ++i)
                std::fprintf(fp, "%zu,%.4f,%.6f\n",
                             i, openness_series[i],
                             i < energy_series.size() ? energy_series[i] : 0.0);
            std::fclose(fp);
        }
    }

    // ====================================================================
    // 6. 指标汇总
    // ====================================================================
    auto m = pipeline.GetMetrics();

    double open_mean = 0, open_max = 0;
    for (double v : openness_series) {
        open_mean += v; open_max = std::max(open_max, v);
    }
    if (!openness_series.empty()) open_mean /= openness_series.size();

    double corr = PearsonCorr(openness_series, energy_series);
    double corr_smooth = PearsonCorr(smooth5(openness_series),
                                     smooth5(energy_series));
    double content_fps = (content_s > 0) ? frames_out / content_s : 0.0;
    double speed_x     = (wall_s > 0) ? content_s / wall_s : 0.0;

    std::cout << "\n==============================================\n";
    std::cout << "  验证报告\n";
    std::cout << "==============================================\n";
    std::cout << "  [帧率]\n";
    std::cout << "    输入视频帧:      " << frames_in << "\n";
    std::cout << "    输出帧数:        " << frames_out << "\n";
    std::cout << "    推理次数:        " << m.inference_count
              << " (理想=" << frames_out << ")\n";
    std::cout << "    丢弃/跳过:       " << m.frames_dropped << " / "
              << m.frames_skipped << "\n";
    std::cout << "    内容时长:        " << content_s << " s\n";
    std::cout << "    墙钟耗时:        " << wall_s << " s\n";
    std::cout << "    内容帧率:        " << content_fps << " fps (目标 "
              << target_fps << ")\n";
    std::cout << "    生成速度:        " << speed_x << "x 实时\n";
    std::cout << "    ------------------------------------------\n";
    std::cout << "    平均音频处理:    " << m.avg_audio_process_ms << " ms\n";
    std::cout << "    平均视频处理:    " << m.avg_video_process_ms << " ms\n";
    std::cout << "    平均推理耗时:    " << m.avg_inference_ms << " ms\n";
    std::cout << "    平均渲染耗时:    " << m.avg_output_ms << " ms\n";
    std::cout << "  [口型]\n";
    std::cout << "    嘴部平均变化量:  " << open_mean << " (0-255 灰度差)\n";
    std::cout << "    嘴部最大变化量:  " << open_max << "\n";
    std::cout << "    口型-能量相关性: " << corr << " (Pearson r, 帧级)\n";
    std::cout << "    口型-能量相关性: " << corr_smooth
              << " (Pearson r, 200ms 平滑)\n";
    std::cout << "==============================================\n";

    // ====================================================================
    // 7. 合成 MP4（ffmpeg 叠加音轨）
    // ====================================================================
    if (saved > 0) {
        std::string silent = out_dir + "/silent.mp4";
        std::string finalv = out_dir + "/lipsync_with_audio.mp4";
        std::string cmd1 = "ffmpeg -y -v error -framerate "
            + std::to_string((int)target_fps) + " -i " + frames_dir
            + "/f_%05d.jpg -c:v libx264 -pix_fmt yuv420p " + silent;
        std::string cmd2 = "ffmpeg -y -v error -i " + silent + " -i "
            + assets_dir + "/" + audio_name
            + " -c:v copy -c:a aac -shortest " + finalv;
        std::cout << "[视频] 合成中...\n";
        if (std::system(cmd1.c_str()) == 0
            && std::system(cmd2.c_str()) == 0) {
            auto sz = std::filesystem::file_size(finalv) / 1024;
            std::cout << "[视频] " << finalv << " (" << sz << "KB, "
                      << saved << "帧)\n";
        } else {
            std::cout << "[视频] ffmpeg 合成失败，帧序列保留在 "
                      << frames_dir << "\n";
        }
    }

    // 退出码：帧率与口型双重判定
    bool fps_ok   = content_fps >= 24.0;
    bool mouth_ok = (open_mean > 1.0)
                 && (corr > 0.15 || corr_smooth > 0.2);
    std::cout << "\n[结论] 帧率" << (fps_ok ? "达标(>=24fps)" : "未达标")
              << ", 口型" << (mouth_ok ? "驱动有效" : "驱动无效") << "\n";
    return (fps_ok && mouth_ok) ? 0 : 2;
}
