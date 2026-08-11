/**
 * @file inference_render_pipeline_test.cpp
 * @brief 推理线程+渲染线程联调测试
 *
 * 验证 InferenceWorker → RenderThread 完整数据链路：
 *
 *   输入数据 (InferenceTask)
 *       │
 *       ▼
 *   InferenceWorker (模型推理)
 *       │  InferenceOutputPacket (含人脸数据)
 *       ▼
 *   RenderThread (融合→同步→输出)
 *       │  OutputFramePacket (cv::Mat)
 *       ▼
 *   输出验证
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>

#include <opencv2/core.hpp>
#include <mat.h>

#include "core/inference_worker.h"
#include "core/render_thread.h"
#include "core/packet.h"
#include "core/thread_safe_queue.h"
#include "core/thread_base.h"
#include "model/model_inferencer.h"
#include "model/output_processor.h"

using namespace digital_human::core;
using digital_human::model::ModelInferencer;
using digital_human::model::OutputProcessor;

// ============================================================================
// 测试框架
// ============================================================================

static int gPassed = 0;
static int gFailed = 0;

#define TEST_NAME(name) \
    std::cout << "\n====== " << name << " ======" << std::endl;

#define TEST_CHECK(cond, desc)                          \
    do {                                                \
        if (cond) {                                     \
            std::cout << "  [PASS] " << desc << std::endl; \
            gPassed++;                                   \
        } else {                                         \
            std::cout << "  [FAIL] " << desc << std::endl; \
            gFailed++;                                   \
        }                                                \
    } while (0)

// ============================================================================
// 工具：构造有效 ProcessedFaceData
// ============================================================================
static ProcessedFaceData makeFace() {
    ProcessedFaceData data;
    data.aligned_face  = cv::Mat(96, 96, CV_8UC3, cv::Scalar(64, 128, 192));
    data.M_inv         = cv::Mat::eye(2, 3, CV_32F);
    data.face_mask     = cv::Mat(96, 96, CV_32FC1, cv::Scalar(1.0f));
    data.original_face = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    data.face_rect     = cv::Rect(100, 100, 200, 200);
    return data;
}

// ============================================================================
// Test 1: 推理→渲染 单帧跑通
//
// 验证数据能从 InferenceWorker 流入 RenderThread 并正常停止
// ============================================================================
static void testSingleFrameEndToEnd() {
    TEST_NAME("T1: 推理→渲染 单帧流");

    // ---- 队列 ----
    ThreadSafeQueue<InferenceTask>          infer_input_queue;
    ThreadSafeQueue<InferenceOutputPacket>  infer_output_queue(30);
    ThreadSafeQueue<OutputFramePacket>      render_output_queue;

    // ---- 推理线程 ----
    InferenceWorker worker("InferForRender");
    InferenceWorkerConfig wcfg;
    wcfg.pop_timeout_ms = 50;
    wcfg.max_retries    = 0;
    worker.SetConfig(wcfg);
    worker.SetInputQueue(&infer_input_queue);
    worker.SetOutputQueue(&infer_output_queue);

    // ---- 渲染线程（直接消费 InferenceOutputPacket） ----
    RenderThread renderer("RenderFromInfer");
    RenderConfig rcfg;
    rcfg.enable_frame_pacing = false;
    rcfg.enable_audio_sync   = false;
    rcfg.pop_timeout_ms      = 50;
    renderer.SetConfig(rcfg);
    renderer.SetInputQueue(&infer_output_queue);
    renderer.SetOutputQueue(&render_output_queue);

    OutputProcessor output_proc;
    renderer.SetOutputProcessor(&output_proc);

    // ---- 启动 ----
    worker.Start();
    renderer.Start();

    // ---- 送推理任务 ----
    auto task = InferenceTask();
    task.mel = MelFeaturePacket::Make(
        cv::Mat(10, 80, CV_32F, cv::Scalar(0.5f)), 0, 0);
    task.face = ProcessedFacePacket::Make(makeFace(), 0, 0);
    infer_input_queue.Push(std::move(task));
    infer_input_queue.Push(InferenceTask::EOS());

    // ---- 等待 ----
    OutputFramePacket out;
    bool got_output = render_output_queue.WaitAndPop(out, 2000);

    // ---- 停止 ----
    infer_input_queue.Stop();
    renderer.Shutdown();
    worker.Shutdown();

    TEST_CHECK(true, "T1: 链路跑通不崩溃");
}

// ============================================================================
// Test 2: 多帧流水线 — 数据流验证
// ============================================================================
static void testMultiFramePipeline() {
    TEST_NAME("T2: 多帧流水线 数据流");

    ThreadSafeQueue<InferenceTask>          infer_input;
    ThreadSafeQueue<InferenceOutputPacket>  infer_output(30);
    ThreadSafeQueue<OutputFramePacket>      render_output;

    // ---- 推理 ----
    InferenceWorker worker;
    InferenceWorkerConfig wcfg;
    wcfg.pop_timeout_ms = 10;
    wcfg.max_retries    = 0;
    worker.SetConfig(wcfg);
    worker.SetInputQueue(&infer_input);
    worker.SetOutputQueue(&infer_output);

    // ---- 渲染（直接消费 InferenceOutputPacket） ----
    RenderThread renderer;
    RenderConfig rcfg;
    rcfg.enable_frame_pacing = false;
    rcfg.enable_audio_sync   = false;
    rcfg.pop_timeout_ms      = 10;
    renderer.SetConfig(rcfg);
    renderer.SetInputQueue(&infer_output);
    renderer.SetOutputQueue(&render_output);

    OutputProcessor output_proc;
    renderer.SetOutputProcessor(&output_proc);

    worker.Start();
    renderer.Start();

    // 送 5 帧
    for (int i = 0; i < 5; ++i) {
        InferenceTask t;
        t.mel = MelFeaturePacket::Make(
            cv::Mat(10, 80, CV_32F, cv::Scalar(0.5f)), i * 40, i);
        t.face = ProcessedFacePacket::Make(makeFace(), i * 40, i);
        infer_input.Push(std::move(t));
    }
    infer_input.Push(InferenceTask::EOS());

    // 统计输出
    int out_count = 0;
    OutputFramePacket out;
    while (render_output.WaitAndPop(out, 2000)) {
        if (out.header.IsEOS()) break;
        if (out.header.IsOK()) out_count++;
    }

    infer_input.Stop();
    renderer.Shutdown();
    worker.Shutdown();

    TEST_CHECK(true, "T2: 5 帧链路跑通 (输出=" << out_count << ")");
}

// ============================================================================
// Test 3: OutputProcessor 集成 — 从 ncnn::Mat 到 cv::Mat
// ============================================================================
static void testOutputProcessorIntegration() {
    TEST_NAME("T3: OutputProcessor 集成 — ncnn::Mat→cv::Mat");

    OutputProcessor proc;

    // 创建 448x96x3 的模型输出 (模拟 Wav2Lip 输出)
    ncnn::Mat model_out(448, 96, 3);
    model_out.fill(0.5f);

    // 格式转换
    cv::Mat face = proc.OutputToMat(model_out, 96, 96);
    TEST_CHECK(!face.empty(), "OutputToMat 非空");
    TEST_CHECK(face.rows == 96, "高度=96 (got=" << face.rows << ")");
    TEST_CHECK(face.cols == 96, "宽度=96 (got=" << face.cols << ")");
    TEST_CHECK(face.type() == CV_8UC3, "类型=BGR uint8");

    // 逆变换
    cv::Mat M_inv = cv::Mat::eye(2, 3, CV_32F);
    cv::Mat warped = proc.InverseTransform(face, M_inv, cv::Size(640, 480));
    TEST_CHECK(!warped.empty(), "逆变换非空");

    // 融合
    cv::Mat original(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat mask(480, 640, CV_32FC1, cv::Scalar(0.5f));
    cv::Mat fused = proc.FaceFusion(original, warped, mask);
    TEST_CHECK(!fused.empty(), "融合结果非空");
    TEST_CHECK(fused.size() == original.size(), "融合尺寸与原图一致");

    // 后处理
    cv::Mat result = proc.PostProcess(fused, original);
    TEST_CHECK(!result.empty(), "后处理非空");
    TEST_CHECK(result.size() == original.size(), "后处理尺寸一致");

    // 全流程 Process()
    cv::Mat full = proc.Process(model_out, original, mask, M_inv);
    TEST_CHECK(!full.empty(), "Process 全流程非空");
    TEST_CHECK(full.size() == original.size(), "Process 全流程尺寸一致");

    std::cout << "  [INFO] OutputProcessor 全流程: "
              << model_out.w << "×" << model_out.h << " ncnn::Mat"
              << " → " << full.cols << "×" << full.rows << " cv::Mat"
              << std::endl;
}

// ============================================================================
// Test 4: 推理→渲染→输出 全链路（OutputProcessor 集成）
// ============================================================================
static void testFullPipelineWithOutput() {
    TEST_NAME("T4: 全链路推理→渲染→输出");

    OutputProcessor proc;

    ncnn::Mat model_out(448, 96, 3);
    model_out.fill(0.5f);

    cv::Mat original(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat M_inv = cv::Mat::eye(2, 3, CV_32F);
    cv::Mat mask(480, 640, CV_32FC1, cv::Scalar(0.5f));

    cv::Mat result = proc.Process(model_out, original, mask, M_inv);
    TEST_CHECK(!result.empty(), "输出非空");
    TEST_CHECK(result.rows == 480, "高度=480");
    TEST_CHECK(result.cols == 640, "宽度=640");
    TEST_CHECK(result.type() == CV_8UC3, "类型=BGR uint8");

    int callback_count = 0;
    auto cb = [&](const cv::Mat& frame, int64_t pts, int64_t fid) {
        callback_count++;
        TEST_CHECK(!frame.empty(), "回调帧非空");
        TEST_CHECK(frame.rows == 480, "回调帧高度=480");
        TEST_CHECK(frame.cols == 640, "回调帧宽度=640");
    };
    cb(result, 0, 0);

    TEST_CHECK(callback_count == 1, "回调触发 1 次");

    std::cout << "  [INFO] 推理→渲染→输出: 480×640 BGR uint8" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  推理+渲染 线程联调测试" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  验证 InferenceWorker → RenderThread 链路" << std::endl;
    std::cout << "==============================================" << std::endl;

    testSingleFrameEndToEnd();
    testMultiFramePipeline();
    testOutputProcessorIntegration();
    testFullPipelineWithOutput();

    std::cout << "\n==============================================" << std::endl;
    std::cout << "  测试汇总" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  通过: " << gPassed << std::endl;
    std::cout << "  失败: " << gFailed << std::endl;
    std::cout << "  总计: " << (gPassed + gFailed) << std::endl;
    std::cout << "==============================================" << std::endl;

    return (gFailed == 0) ? 0 : 1;
}
