# 音视频编码与 RTMP/RTSP 推流

当前媒体层接受数字人 SDK 既有的 float PCM 和 BGR 输出，在外围完成 AAC、H.264 编码及网络发布，不修改口型推理 Pipeline。

## 1. 组件职责

- `StreamPublisher`：BGR→H.264、PCM→AAC、音视频交错封装和发布；
- `ConversationStreamBridge`：把 TTS PCM 同时送往数字人 SDK 和 AAC 输入，并把 SDK 渲染 BGR 送往 H.264 输入；
- `ConversationSession`：文本生成、分句、TTS 和音视频供料。

RTMP/RTSP 服务端不在 SDK 内。调用方需先部署 MediaMTX、SRS、ZLMediaKit 或其他兼容服务，并提供可发布 URL。

## 2. 直接发布 PCM/BGR

```cpp
#include "media/stream_publisher.h"

using namespace digital_human::media;

StreamPublisherConfig config;
config.url = "rtmp://127.0.0.1:1935/live/avatar";
config.protocol = StreamProtocol::RTMP;  // AUTO 也可根据 URL 判断
config.width = 640;
config.height = 360;
config.fps = 25.0;
config.input_audio_sample_rate = 16000;
config.input_audio_channels = 1;
config.output_audio_sample_rate = 48000;
config.output_audio_channels = 1;
config.video_bitrate = 2'000'000;
config.audio_bitrate = 96'000;

StreamPublisher publisher;
std::string error;
if (!publisher.Open(config, error)) {
    // 记录 error
}

// frame: CV_8UC3，尺寸必须为 640x360；pcm: interleaved float
publisher.PushAudio(pcm, audio_pts_ms, error);
publisher.PushVideo(frame, video_pts_ms, error);

publisher.Close(true, error);  // true 表示排空队列并 flush
```

RTSP 只需替换地址和协议：

```cpp
config.url = "rtsp://127.0.0.1:8554/live/avatar";
config.protocol = StreamProtocol::RTSP;
config.rtsp_tcp = true;
```

本地文件模式用于模块测试；输出扩展名应对应容器，例如 `.flv`：

```cpp
config.url = "output.flv";
config.protocol = StreamProtocol::FILE;
```

## 3. 接入实时会话闭环

初始化顺序必须是：启动 `DigitalHumanSDK`，打开 `StreamPublisher`，启动 `ConversationStreamBridge`，最后启动 `ConversationSession`。

```cpp
#include "dialog/conversation_session.h"
#include "digital_human_sdk.h"
#include "media/conversation_stream_bridge.h"
#include "media/stream_publisher.h"

DigitalHumanSDK sdk;
sdk.Init(sdk_config);
sdk.Start();

media::StreamPublisher publisher;
publisher.Open(publisher_config, error);

media::ConversationStreamBridge bridge(sdk, publisher);
bridge.Start(error);

dialog::ConversationSession session(text_client, tts_client, bridge);
session.Start(conversation_config, avatar_bgr);
session.SubmitUserText("你好，请介绍一下自己");

// 退出时由 Session 触发 bridge.Finish()，排空 SDK 和发布器。
session.Stop(true);
sdk.Stop();
```

不要把数字人底图直接提交给 `StreamPublisher`。发布器的视频输入应该是 `DigitalHumanSDK::GetOutputFrame` 返回的口型融合结果；`ConversationStreamBridge` 已完成该路由。

## 4. 编码器选择

`video_encoder` 留空时按以下顺序选择本机可用编码器：

1. `h264_nvenc`
2. `h264_qsv`
3. `h264_amf`
4. `libx264`
5. `h264`

部署时建议显式指定已经验证的编码器，避免不同机器选择结果不一致。例如 CPU 环境使用 `libx264`，NVIDIA 环境使用 `h264_nvenc`。如果编码器存在但设备初始化失败，发布器会继续尝试下一个候选编码器。

AAC 默认使用 FFmpeg 的 `aac` 编码器。媒体层会把 16 kHz float PCM 重采样为编码器配置的 48 kHz 格式，并通过 Audio FIFO 对齐 AAC frame size。

## 5. 队列与实时性

- `max_video_queue` 默认 12；队列满时丢最旧视频帧，优先保证低延迟；
- `max_audio_queue` 默认 64；队列满时反压 `PushAudio`，避免语音样本缺失；
- `Close(true)` 会排空队列，补齐最后一个 AAC frame 并写容器 trailer；
- `Close(false)` 用于故障快速退出，未编码的队列数据会丢弃；
- `io_timeout_ms` 通过 FFmpeg 中断回调限制建连、写包和收尾 I/O；生产环境还应在上层实现重连策略。

可通过 `GetMetrics()` 读取：

```cpp
const auto metrics = publisher.GetMetrics();
// video_frames_in / video_frames_encoded / video_frames_dropped
// audio_samples_in / audio_frames_encoded / packets_written
```

如果 `video_frames_dropped` 持续增长，应降低输出分辨率/帧率、降低编码复杂度或启用硬件编码器。

## 6. 测试

本地封装测试：

```bash
./bin/stream_publisher_test
```

真实数字人闭环测试（需要模型和测试底图）：

```bash
./bin/conversation_stream_integration_test
```

网络发布冒烟测试：

```bash
./bin/stream_network_publish_test rtmp rtmp://127.0.0.1:1935/live/test
./bin/stream_network_publish_test rtsp rtsp://127.0.0.1:8554/live/test
```

测试程序会实时生成约 1.2 秒 H.264/AAC 流。成功仅表示发布连接和 packet 写入正常；生产验收还应由接收端同时校验视频帧、音频轨和持续播放。

## 7. 常见错误

| 错误 | 检查项 |
|---|---|
| `Connection refused` | 服务端是否启动、端口和发布路径是否正确 |
| `no usable H.264 encoder` | FFmpeg 是否包含 libx264 或硬件编码器，驱动是否可用 |
| `BGR frame size does not match publisher config` | SDK 输出尺寸与 publisher 的 width/height 是否一致 |
| `PCM sample count is not divisible...` | interleaved PCM 样本数是否按声道对齐 |
| 视频丢帧持续增加 | 编码吞吐不足或网络阻塞，检查 metrics 并调小码率/分辨率 |
| 有画面无声音 | 是否将原始 TTS PCM 同时送入 publisher，输入采样率配置是否正确 |
