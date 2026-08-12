# 全链路闭环验收

`full_conversation_chain_test` 用于验证从用户文本到最终音视频文件或网络流的完整工程链路：

```text
用户文本
  → llama.cpp OpenAI Chat Completions（真实 Qwen3）
  → SSE 增量分句
  → HTTP TTS（16 kHz mono PCM）
  → ConversationSession
  → DigitalHumanSDK / Vulkan Wav2Lip
  → rendered BGR
  → H.264 + AAC
  → FLV 文件或 RTMP/RTSP
```

## 1. 前置服务

启动 llama.cpp：

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_llama_cpp_server.ps1
```

当前测试环境尚未部署生产 TTS 模型。可用协议 mock 验证真实 HTTP/PCM 边界：

```powershell
python tools/mock_dialog_service.py --host 127.0.0.1 --port 18080
```

mock TTS 返回确定性测试音 PCM，因此能验证网络请求、PCM 解码、口型驱动、AAC 编码和同步，但不能验收真实语音音色或文本读音。部署真实 TTS 时只需将测试命令中的 TTS URL 替换为兼容接口。

## 2. 构建

```powershell
cmake --build D:\digital-human-dev\build\digital-human-sdk-windows-ucrt64-vulkan `
  --target full_conversation_chain_test -j 8
```

## 3. 文件闭环

```powershell
D:\digital-human-dev\build\digital-human-sdk-windows-ucrt64-vulkan\bin\full_conversation_chain_test.exe `
  http://127.0.0.1:8090/v1/chat/completions `
  http://127.0.0.1:18080/tts `
  D:\digital-human-dev\build\digital-human-sdk-windows-ucrt64-vulkan\artifacts\full_conversation_chain.flv `
  file `
  Qwen3-4B-Q4_K_M.gguf
```

测试会自行解封装输出并检查：

- llama.cpp 返回非空回复；
- ConversationSession 无错误并正常结束；
- Wav2Lip 至少生成一个视频帧；
- 输出包含 H.264 视频和 AAC 音频；
- 两条轨道均包含有效 packet。

本次验收结果：回复“数字人全链路测试正常。”，H.264 7 packets，AAC 20 packets，640×360@25fps，48 kHz 单声道。

## 4. RTSP 网络闭环

测试环境可用 FFmpeg 临时接收发布：

```powershell
ffmpeg -rtsp_flags listen -listen_timeout 30 `
  -i rtsp://127.0.0.1:18554/live/full-chain -f null NUL
```

另一个终端运行：

```powershell
D:\digital-human-dev\build\digital-human-sdk-windows-ucrt64-vulkan\bin\full_conversation_chain_test.exe `
  http://127.0.0.1:8090/v1/chat/completions `
  http://127.0.0.1:18080/tts `
  rtsp://127.0.0.1:18554/live/full-chain `
  rtsp `
  Qwen3-4B-Q4_K_M.gguf
```

接收端必须识别 H.264 High、AAC-LC 48 kHz mono，并实际解码出视频帧。RTMP 测试方式相同，但目标必须是可接受发布的 RTMP 服务端。

## 5. 本次发现并修复的问题

短回复只有约 400 ms 音频时，`AVMatcher` 曾把“原始视频输入 EOS 且已处理队列暂时为空”误判为“视频处理完成”。此时 `VideoProcessor` 仍在处理已入队 BGR，导致提前向推理线程发送 EOS，最终只有 AAC、没有 H.264。

修复后，`AVMatcher` 不再根据瞬时空队列推断完成，而是等待 `VideoProcessor` 排空原始视频队列后显式发出的 `ProcessedFacePacket::EOS`。短回复完整闭环已连续运行三次通过。

## 6. 验收结论边界

当前已验证：

- 真实本地 Qwen3 文本生成；
- HTTP TTS 协议、PCM 数据和取消/超时边界；
- 真实 Vulkan Wav2Lip 推理和口型渲染；
- H.264/AAC 文件封装；
- RTSP 发布、接收和解码。

生产部署前仍需验证：

- 选定的真实 TTS 服务及其音色、读音和长文本稳定性；
- 目标 RTMP/RTSP 服务端的鉴权、重连和长时间运行；
- 麦克风/ASR 输入以及用户打断；
- 多轮长会话的历史窗口和内存/显存稳定性。
