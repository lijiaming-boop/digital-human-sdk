# Digital Human SDK

基于 Wav2Lip-SD-GAN 的数字人口型同步 SDK，提供完整的音视频处理流水线：音频特征提取、人脸检测对齐、模型推理、口唇融合渲染。

---

## 目录

- [快速开始](#快速开始)
- [系统要求](#系统要求)
- [依赖安装](#依赖安装)
- [构建](#构建)
- [模块架构](#模块架构)
- [运行测试](#运行测试)
- [API 概览](#api-概览)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/lijiaming-boop/digital-human-sdk.git
cd digital-human-sdk

# 2. 安装依赖（Ubuntu 22.04）
sudo apt install -y build-essential cmake pkg-config \
    libopencv-dev libavformat-dev libavcodec-dev \
    libavutil-dev libswresample-dev libswscale-dev \
    libportaudio-dev

# 3. 安装 ncnn（手动）
git clone https://github.com/Tencent/ncnn.git
cd ncnn && mkdir build && cd build
cmake .. -DNCNN_VULKAN=OFF && make -j$(nproc) && sudo make install
cd ../..

# 4. 构建
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 5. 运行测试
./bin/full_pipeline_test
```

---

## 系统要求

| 项目 | 最低要求 |
|------|---------|
| 操作系统 | Ubuntu 22.04 / Windows WSL2 |
| 编译器 | GCC 11+ / Clang 14+ |
| CMake | 3.16+ |
| CPU | 4 核 x86_64 (支持 AVX2) |
| 内存 | 8 GB |
| 音频设备 | ALSA 兼容 (PortAudio) |
| GPU (可选) | Vulkan 1.1+ |

---

## 依赖安装

### Ubuntu 22.04

```bash
# 基础工具
sudo apt install -y build-essential cmake pkg-config git

# OpenCV 4
sudo apt install -y libopencv-dev

# FFmpeg
sudo apt install -y libavformat-dev libavcodec-dev \
    libavutil-dev libswresample-dev libswscale-dev

# PortAudio
sudo apt install -y portaudio19-dev

# ncnn（源码安装）
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git checkout 20240820
mkdir build && cd build
cmake .. -DNCNN_VULKAN=OFF \
         -DNCNN_BUILD_EXAMPLES=OFF \
         -DNCNN_BUILD_TOOLS=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../..
```

---

## 构建

### 完整构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 仅构建核心库

```bash
make -j$(nproc) digital_human_core
```

### 构建指定测试

```bash
make -j$(nproc) full_pipeline_test
make -j$(nproc) pipeline_test
make -j$(nproc) inference_worker_test
```

### 输出目录

```
build/
├── lib/libdigital_human_core.so   # 核心共享库
└── bin/                            # 测试可执行文件
    ├── full_pipeline_test
    ├── pipeline_test
    ├── inference_worker_test
    ├── render_thread_test
    ├── audio_processor_test
    ├── frame_scheduler_test
    └── audio_sync_test
```

---

## 模块架构

### 数据流全景

```
音频: PCM float → NoiseReduction → AudioFramer → VAD
      → PreEmphasis → RMSNormalize → MelFeatureExtract
      → CMVN → ncnn::Mat

视频: cv::Mat BGR → FaceDetect → FaceAlign → FaceMaskGen
      → ProcessedFaceData

推理: Mel + AlignedFace → ModelInferencer(Wav2Lip) → ncnn::Mat

渲染: ncnn::Mat → OutputToMat → InverseTransform
      → FaceFusion → Sharpen → ColorBlend → cv::Mat
```

### 线程模型

```
AudioProducer ──► AudioProcessor ──► ┌─ SyncPoint ─┐
                                     │  (AV匹配)    │
VideoProducer ──► VideoProcessor ──►└──────────────┘
                                              │
                                              ▼
                                        InferenceWorker
                                              │
                                              ▼
                                        OutputProcessor
                                              │
                                              ▼
                                          RenderThread
                                              │
                                              ▼
                                     FrameCallback/OutputQueue
```

| 线程 | 职责 | 优先级 |
|------|------|--------|
| AudioProducer | 读取音频数据 | 高 |
| AudioProcessor | PCM → Mel 特征提取（7阶段流水线） | 中 |
| VideoProducer | 读取视频帧 | 高 |
| VideoProcessor | 人脸检测→对齐→遮罩 | 中 |
| InferenceWorker | Wav2Lip 模型推理 + 失败重试 | 中 |
| OutputProcessor | 模型输出后处理 | 中 |
| RenderThread | 融合 + 音频同步 + 帧间隔调节 | 高 |
| AudioPlayback | PortAudio 音频输出（系统回调） | 最高（实时） |

### 核心库结构

```
include/
├── audio/                          # 音频处理模块
│   ├── audio_loader.h             # 音频文件加载
│   ├── audio_framer.h             # 分帧（frame/hop）
│   ├── audio_noise_reduction.h    # 谱减法降噪
│   ├── audio_vad.h                # 语音活动检测
│   ├── audio_preemphasis.h        # 预加重
│   ├── audio_rms_normalize.h      # RMS 归一化
│   ├── audio_mel_feature_extract.h# Mel 频谱 (FFT→滤波→Log)
│   ├── audio_cmvn.h               # 倒谱归一化
│   ├── audio_ring_buffer.h        # 无锁 SPSC 环形缓冲区
│   └── audio_player.h             # PortAudio 播放器
│
├── core/                           # 核心模块
│   ├── image_loader.h             # 图像加载
│   ├── face_detector.h            # 人脸检测
│   ├── face_aligner.h             # 人脸对齐 (仿射变换)
│   ├── face_mask_generator.h      # 口唇遮罩生成
│   ├── frame_scheduler.h          # 帧调度 (DROP/DUPLICATE/DISPLAY)
│   ├── av_sync.h                  # 音视频同步 (Audio Master Clock)
│   ├── audio_sync_scheduler.h     # 音频同步调度器
│   ├── audio_processor.h          # 音频处理线程
│   ├── inference_worker.h         # 推理线程
│   ├── render_thread.h            # 渲染线程
│   ├── pipeline.h                 # 流水线编排
│   ├── packet.h                   # 数据包定义
│   ├── thread_safe_queue.h        # 线程安全队列
│   └── thread_base.h              # 线程基类 (生命周期管理)
│
└── model/                          # 模型模块
    ├── model_loader.h             # 异步模型加载 (回调通知)
    ├── model_inferencer.h         # Wav2Lip 推理器 (auto-tune)
    └── output_processor.h         # 输出后处理 (融合/锐化/色彩)
```

---

## 运行测试

### 全量测试

```bash
cd build

# 全链路集成测试（162 测试，无需 Wav2Lip 模型）
./bin/full_pipeline_test

# 推理+渲染联调（21 测试）
./bin/inference_render_pipeline_test

# 线程基础设施（76 测试）
./bin/pipeline_test

# 各模块独立测试
./bin/audio_processor_test      # 23 测试
./bin/inference_worker_test     # 45 测试
./bin/render_thread_test        # 28 测试
./bin/frame_scheduler_test      # 62 测试
./bin/audio_sync_test           # 37 测试
```

### 快速验证脚本

```bash
# run_all_tests.sh
cd build
tests=(
    full_pipeline_test
    inference_render_pipeline_test
    pipeline_test
    audio_processor_test
    inference_worker_test
    render_thread_test
    frame_scheduler_test
    audio_sync_test
)
for t in "${tests[@]}"; do
    echo "=== $t ==="
    ./bin/$t 2>/dev/null
done
```

### 测试层级

| 层级 | 覆盖 | 命令 |
|------|------|------|
| Level 0 | 模块独立测试（无需模型） | 各 `*_test.cpp` |
| Level 1 | 线程基础设施（队列/线程基类） | `pipeline_test` |
| Level 2 | 音频/推理/渲染线程 | `*_worker_test` |
| Level 3 | 推理+渲染联调 | `inference_render_pipeline_test` |
| Level 4 | 全链路端到端 | `full_pipeline_test` |

---

## API 概览

### 音频处理

```cpp
#include "audio/audio_loader.h"
#include "audio/audio_framer.h"
#include "audio/audio_mel_feature_extract.h"

using namespace digital_human::audio;

// 加载音频
AudioLoader loader;
AudioData data = loader.load("input.wav");

// 降噪 → 分帧
NoiseReduction nr;
auto denoised = nr.process(data.samples, data.sampleRate);

AudioFramer framer;
auto frames = framer.frame(denoised, {400, 160});

// Mel 特征
MelFeatureExtract mel;
MelConfig mc{512, 80, 16000, 0, 8000};
auto spec = mel.extract(frames, mc);  // cv::Mat (T×80)
```

### 视频处理

```cpp
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"

using namespace digital_human::core;

// 人脸检测
FaceDetector detector;
auto faces = detector.detect(image);

// 对齐
FaceAligner aligner;
auto result = aligner.alignByRect(image, landmarks, 96, faces[0]);

// 遮罩
FaceMaskGenerator mask_gen;
auto mask = mask_gen.generateMouthMask(image.size(), landmarks);
```

### 模型推理

```cpp
#include "model/model_inferencer.h"

digital_human::model::ModelInferencer model;
model.Init("models/Wav2Lip-SD-GAN-opt");

ncnn::Mat output = model.Infer(audio_feat_ncnn, face_input_ncnn);
```

### 流水线

```cpp
#include "core/pipeline.h"

using namespace digital_human::core;

PipelineConfig config;
config.audio_sample_rate = 16000;
config.target_fps        = 25.0;

Pipeline pipeline;
pipeline.Init(config);
pipeline.Start();

// 输入数据
pipeline.PushAudio(pcm_data, pts_ms);
pipeline.PushVideo(frame, pts_ms);

// 获取输出
OutputFramePacket result;
while (pipeline.GetOutputFrame(result)) {
    cv::imshow("Output", result.payload);
}

pipeline.Stop();
```

### 音视频同步

```cpp
#include "core/audio_sync_scheduler.h"

using namespace digital_human::core;

AudioSyncScheduler sched;
AudioSyncConfig cfg;
cfg.audio_sample_rate = 48000;
cfg.target_fps        = 25.0;
sched.Init(cfg);
sched.LoadAudio(pcm_data, channels);
sched.Play();

// 帧调度
auto decision = sched.ScheduleFrame(frame_id, pts_ms);
switch (decision.action) {
    case FrameAction::DISPLAY:   break;  // 正常显示
    case FrameAction::DROP:      break;  // 丢弃滞后帧
    case FrameAction::DUPLICATE: break;  // 重复超前帧
    case FrameAction::WAIT:      break;  // 等待音频追赶
}
```

---

## 配置说明

### PipelineConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `audio_sample_rate` | 16000 | 音频采样率 (Hz) |
| `audio_frame_size` | 400 | 帧长 (samples @16kHz=25ms) |
| `audio_hop_size` | 160 | 帧移 (samples @16kHz=10ms) |
| `target_fps` | 25.0 | 目标帧率 |
| `sync_threshold_ms` | 30.0 | 同步阈值 (ms) |
| `max_drift_ms` | 100.0 | 最大允许漂移 (ms) |

### InferenceWorkerConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `max_retries` | 3 | 推理失败最大重试次数 |
| `pop_timeout_ms` | 100 | 队列弹出超时 (ms) |
| `input_queue_warn_threshold` | 10 | 队列深度警告阈值 |
| `input_queue_error_threshold` | 30 | 队列深度错误阈值 |
| `backlog_check_interval` | 1000 | 积压检测间隔 (ms) |
| `latency_warn_threshold_ms` | 100.0 | 推理延迟警告阈值 (ms) |

### RenderConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `target_fps` | 25.0 | 目标帧率 |
| `sync_threshold_ms` | 30.0 | 同步阈值 (ms) |
| `max_drift_ms` | 100.0 | 最大允许漂移 (ms) |
| `enable_audio_sync` | true | 启用音频同步 |
| `enable_frame_pacing` | true | 启用帧间隔调节 |
| `drain_max_frames` | 30 | 退出时最大排空帧数 |
| `pop_timeout_ms` | 100 | 队列弹出超时 (ms) |

---

## 常见问题

### 编译

**Q: ncnn 找不到**
```bash
export CMAKE_PREFIX_PATH=/usr/local/lib/cmake:$CMAKE_PREFIX_PATH
cmake ..
```

**Q: OpenCV 版本不匹配** — Ubuntu 22.04 默认安装 OpenCV 4.5.4

**人脸模型** — 请按 [深度学习人脸模型](docs/models/face_models.md) 准备 SCRFD 与 2D106 权重。

### 运行时

**Q: PortAudio 报"No audio devices"**
- WSL2 默认无音频设备，不影响功能测试
- 生产环境需 ALSA 驱动 (`sudo apt install alsa-utils`)

**Q: 推理线程一直重试**
- 确认模型文件路径正确
- 检查 `models/` 下有 `.param` 和 `.bin` 文件

**Q: 渲染帧率不稳定**
- 调整 `RenderConfig::target_fps`
- 确保 `enable_frame_pacing = true`
- 检查队列深度是否积压（`InferenceMetrics` 日志）

### 调试

```bash
# Valgrind 检测内存
valgrind --leak-check=full ./bin/full_pipeline_test 2>/dev/null

# 指定模型目录
export MODEL_DIR=/path/to/models
```

---

## 贡献

1. 从 `master` 创建功能分支：`feature/your-feature`
2. 遵循 PIMPL 模式编写新模块
3. 所有公有 API 需中文 Doxygen 注释
4. 为每个模块编写独立测试（覆盖正常/边界/异常路径）
5. 提交前确保全量测试通过：`./run_all_tests.sh`

---



## 许可证

MIT License
