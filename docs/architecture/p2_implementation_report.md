# P2 可靠性、质量门禁与依赖解耦实施报告

## 1. 目标与边界

本轮实现对应 `project_completion_roadmap.md` 的 P2 SDK 侧工作：弱网恢复、媒体时间轴、真实队列延迟、编码器回退、故障注入、质量门禁，以及 FFmpeg/PortAudio 解耦。网关集群、真实 RTMP/RTSP 服务器压测、跨平台 GPU 矩阵和 30 分钟长稳属于部署环境验收项，不在 SDK 单仓代码中伪造结果。

## 2. 已实现能力

| 领域 | 实现 | 关键行为 |
|---|---|---|
| 音频输出解耦 | `IAudioPlayer` + 构造注入 | `dh_runtime` 不再直接持有 PortAudio 实现；关闭 `DIGITAL_HUMAN_ENABLE_AUDIO_IO` 仍可构建，调用方可注入平台后端或测试时钟 |
| 音频文件解码解耦 | `dh_audio_loader` | 新增 `DIGITAL_HUMAN_ENABLE_AUDIO_LOADER`；纯流式推理构建不再需要 FFmpeg，精简构建中的 `ProcessFile` 返回明确能力不可用错误 |
| 网络发布恢复 | `StreamPublisher` 指数退避重连 | RTMP/RTSP 写失败后，在最大重连窗口内按上限退避；暴露尝试、成功、失败指标 |
| 重连数据策略 | `ReconnectAudioPolicy` | 视频只保留最新帧；音频可选择 `DROP`、有界 `RETAIN` 或 `FAIL_SESSION`，避免恢复后回放陈旧口型 |
| 编码器能力回退 | 硬件候选 + 软件回退 | 自动尝试 NVENC/QSV/AMF，再回退 libx264/h264；即使指定硬件编码器失败也允许软件回退，并暴露实际选中编码器 |
| 真实延迟背压 | `max_video_queue_latency_ms` | RTMP/RTSP 视频队列同时受帧数和 PTS 延迟约束；文件离线编码保持完整帧，不应用实时丢帧策略 |
| 独立媒体时钟 | `MediaClock` | 音频主时钟、PTS 单调性检测、p50/p95/p99 抖动、最大漂移、断点计数和有界长期视频校正 |
| 发布质量门禁 | `QualityGate` | 对 FPS、帧延迟、丢帧率、AV 同步、音视频时长差、嘴部 PSNR/SSIM 和 PTS 单调性输出逐项违规原因 |
| 故障测试 | `publisher_resilience_test` | 确定性模拟 ENOSPC/写盘失败，并覆盖慢消费者导致的音频入队超时和会话失败 |

## 3. 配置建议

实时发布建议从以下基线开始，再按网络 RTT 和 TTS 分片大小调优：

```cpp
StreamPublisherConfig config;
config.max_video_queue_latency_ms = 500;
config.enable_reconnect = true;
config.reconnect_initial_backoff_ms = 250;
config.reconnect_max_backoff_ms = 4000;
config.reconnect_window_ms = 30000;
config.reconnect_audio_policy = ReconnectAudioPolicy::RETAIN;
config.max_retained_audio_ms = 2000;
```

生产代码不得设置 `debug_fail_after_packets`；该字段只用于 CI 故障路径验证。

## 4. 构建与测试分层

- 全功能构建：HTTP、媒体、PortAudio、音频文件解码全部启用。
- 精简流式构建：`BUILD_EXAMPLES=OFF`，并将四个可选模块全部关闭；该路径验证运行时不再隐式链接 FFmpeg、libcurl 或 PortAudio。
- P2 快速门禁：`ctest -L p2 --output-on-failure`。
- 故障门禁：`ctest -L fault --output-on-failure`。
- 质量门禁：`ctest -L quality --output-on-failure`。

## 5. 尚需环境验收的项目

以下工作需要真实基础设施或发布数据，不能仅由单元测试宣称完成：

1. 在可控 RTMP/RTSP 服务端执行断网、拒绝写入、半开连接和恢复测试，确认不同音频策略符合产品预期。
2. 建立 Windows/Linux、CPU/Vulkan/NVIDIA/Intel/AMD 编码器矩阵，记录实际回退路径。
3. 使用真实模型和素材生成嘴部 ROI 的 PSNR/SSIM 基线，并为 INT8/FP16 分别设置阈值。
4. 执行至少 30 分钟多轮会话长稳，采集 RSS、队列深度、FPS p50/p95/p99、漂移和重连指标。
5. 将 `QualityGate` 输入接入基准测试产物，在 CI 中以非零退出码阻止不达标版本发布。

## 6. 完成判定

SDK 侧 P2 实现完成的判定是：全功能构建、精简构建、P2/故障测试全部通过；环境侧 P2 完成则还必须满足第 5 节的真实网络、硬件、模型质量和长稳报告。二者应分开记录，避免把“代码路径存在”误报为“生产指标已经达标”。
