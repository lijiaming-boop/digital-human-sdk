# 音频处理线程技术方案

## 1. 概述

音频处理线程 (`AudioProcessor`) 负责从音频缓冲区读取 PCM 数据，提取梅尔频谱特征，并将特征推送到推理队列。它必须与音频播放线程（PortAudio 回调）解耦，确保不阻塞实时音频播放。

## 2. 设计目标

| 维度 | 目标 |
|------|------|
| 实时性 | 每 10ms 音频处理延迟 < 5ms（避免特征落后于播放） |
| 非阻塞 | PortAudio 回调路径零锁等待 |
| 流式处理 | 支持增量音频输入，无需完整文件加载 |
| 帧精确 | 与 Wav2Lip 模型的 160-hop / 400-frame 对齐 |

## 3. 架构

```
音频源 (文件/麦克风)
    │
    ▼
AudioRingBuffer (SPSC, 锁自由)
    │
    ├──── AudioPlayer (PortAudio 回调线程)  →  扬声器
    │         [实时优先级，不可阻塞]
    │
    └──── AudioProcessor (处理线程)
              [普通优先级，尽力实时]
              │
              ├── 1. 从 RingBuffer 读取 ≥ 1 hop 的新数据
              ├── 2. 追加到内部滑动窗口 (overlap 240 samples)
              ├── 3. 窗口满 400 samples 时触发处理:
              │     NoiseReduction → AudioFramer → VAD
              │     → PreEmphasis → RMSNormalize
              │     → MelFeatureExtract → CMVN
              └── 4. 推送 MelFeaturePacket → InferenceQueue
```

### 3.1 数据流

```
时间轴 (16kHz):
sample: |---240---|----160----|----160----|----160----|...
                 frame 0 (400 samples)
                           frame 1 (hop=160, overlap=240)
                                      frame 2
                    
处理节奏: 每收到 160samples (10ms) → 输出 1 帧 mel 特征
```

### 3.2 线程边界

```
PortAudio 回调线程:
  - 从 AudioRingBuffer 读取 → 填充 DMA 缓冲区
  - 更新 consumed_frames (atomic)
  - 禁止任何锁操作
  - 路径延迟: < 1ms (必须满足)

AudioProcessor 线程:
  - 从 AudioRingBuffer 读取 → 处理 pipeline
  - 写入 MelFeatureQueue (ThreadSafeQueue)
  - 允许有限阻塞 (mutex + condition_variable)
  - 目标延迟: < 5ms / 每 hop
```

## 4. 滑动窗口设计

```cpp
struct SlidingWindow {
    std::vector<float> buffer;       // 累积的音频数据
    int64_t            frame_size;   // 400 samples
    int64_t            hop_size;     // 160 samples
    int64_t            read_cursor;  // 已处理的 sample 位置

    // 追加新数据并返回可提取的帧数
    int Append(const float* data, size_t count) {
        buffer.insert(buffer.end(), data, data + count);
        // 计算可提取的完整帧数
        if (buffer.size() < frame_size) return 0;
        return (buffer.size() - frame_size) / hop_size + 1;
    }

    // 取出下一帧（移动语义，避免拷贝）
    std::vector<float> PopFrame() {
        std::vector<float> frame(buffer.begin(), buffer.begin() + frame_size);
        // 滑动窗口：丢弃 hop_size 个 samples
        buffer.erase(buffer.begin(), buffer.begin() + hop_size);
        read_cursor += hop_size;
        return frame;
    }
};
```

## 5. 实时性保证

### 5.1 分层优先级

```
高优先级 (实时):
  PortAudio 回调 ← RingBuffer (无锁 SPSC)

中优先级 (尽力实时):
  AudioProcessor  ← RingBuffer (批量读取)

低优先级 (非实时):
  InferenceWorker ← MelFeatureQueue (有界)
  VideoProcessor  ← VideoRawQueue (有界)
```

### 5.2 背压保护

- MelFeatureQueue 有界 (60 帧 ≈ 600ms 音频)
- AudioProcessor 检测队列堆积时自动调节处理速度
- AudioPlayer 不受影响（独立 RingBuffer）

### 5.3 时钟同步

AudioProcessor 不依赖系统时钟，而是以 **音频 sample 位置** 驱动：

```
pts = read_cursor / sample_rate * 1000  (毫秒)

每处理一个 hop，pts 增加 10ms (160/16000*1000)
```

## 6. 模块接口

### AudioProcessor

```cpp
class AudioProcessor : public ThreadBase {
public:
    AudioProcessor();
    ~AudioProcessor();

    // 配置
    void SetAudioSource(const float* data, size_t samples,
                        int sample_rate, int channels);
    void SetRingBuffer(RingBuffer* buffer);  // 流式模式
    void SetOutputQueue(ThreadSafeQueue<MelFeaturePacket>* queue);
    void SetConfig(const AudioProcessorConfig& config);

    // 线程主循环
    void Run() override;

    // 状态查询
    int64_t GetProcessedSamples() const;  // 已处理的 sample 数
    double  GetProcessedDurationMs() const;
    int64_t GetPendingFrames() const;     // 等待处理的帧数

    // 重置
    void Reset();

private:
    // 核心处理 (const 方法，无状态副作用)
    MelFeaturePacket ProcessOneFrame(
        const std::vector<float>& frame_window) const;

    // 滑动窗口管理
    int FillWindow();            // 从 RingBuffer 读入新数据
    std::vector<float> NextFrame();  // 提取下一帧并滑窗

    // 音频处理模块（每个线程独立实例）
    mutable NoiseReduction      noise_reduction_;
    mutable AudioFramer         framer_;
    mutable VoiceActivityDetector vad_;
    mutable PreEmphasis         pre_emphasis_;
    mutable RMSNormalize        rms_normalize_;
    mutable MelFeatureExtract   mel_extract_;
    mutable CMVN                cmvn_;

    // 状态
    std::vector<float> window_buffer_;   // 滑动窗口
    int64_t            sample_rate_ = 16000;
    int64_t            frame_size_ = 400;
    int64_t            hop_size_ = 160;
    int64_t            read_cursor_ = 0;    // 已处理的 sample 位置
    int64_t            seq_id_ = 0;

    // 外部依赖
    const float*       fixed_audio_ = nullptr;  // 文件模式
    size_t             fixed_audio_size_ = 0;
    RingBuffer*        ring_buffer_ = nullptr;
    ThreadSafeQueue<MelFeaturePacket>* output_queue_ = nullptr;
};
```

### AudioProcessorConfig

```cpp
struct AudioProcessorConfig {
    int    sample_rate       = 16000;
    int    channels          = 1;
    int    frame_size        = 400;    // 帧长 (25ms @16kHz)
    int    hop_size          = 160;    // 帧移 (10ms @16kHz)
    int    mel_bins          = 80;     // Mel 滤波器组数
    int    nfft              = 512;    // FFT 点数
    int    output_queue_capacity = 60; // 输出队列容量
    int    window_capacity   = 4800;   // 滑动窗口最大 (300ms)
    int    idle_sleep_ms     = 5;      // 空闲时休眠
};
```

## 7. 文件结构

```
include/core/
├── audio_processor.h           # AudioProcessor 线程
├── audio_processor_config.h    # 配置结构

src/core/
├── audio_processor.cpp         # 实现
```

## 8. 测试策略

| 测试 | 场景 |
|------|------|
| 滑动窗口正确性 | 输入 1600 samples → 验证输出 (1600-400)/160+1 = 8 帧 |
| 空缓冲区等待 | RingBuffer 空时 WaitAndPop 超时，不空转 |
| 实时性 | 处理 1s 音频，验证延迟 < 10ms |
| 与 AudioPlayer 并行 | 同时播放和处理，验证无干扰 |
| 流式输入 | 持续追加音频，验证逐帧输出 |
