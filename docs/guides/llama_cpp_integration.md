# llama.cpp 接入指南

项目通过 llama.cpp server 的 OpenAI 兼容 HTTP 接口生成回复，不链接 llama.cpp 库，也不在 SDK 进程内加载 GGUF 模型。这样模型进程、GPU 参数和 SDK 生命周期可以独立管理。

当前本机配置：

- llama.cpp 目录：`E:\llama.cpp`
- 服务程序：`E:\llama.cpp\llama-server.exe`
- 模型：`E:\llama.cpp\models\Qwen3-4B-Q4_K_M.gguf`
- 默认监听：`127.0.0.1:8090`
- Chat Completions：`http://127.0.0.1:8090/v1/chat/completions`

## 1. 启动 llama.cpp server

项目提供前台启动脚本：

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_llama_cpp_server.ps1
```

等价的直接命令：

```powershell
E:\llama.cpp\llama-server.exe `
  -m E:\llama.cpp\models\Qwen3-4B-Q4_K_M.gguf `
  --host 127.0.0.1 `
  --port 8090 `
  -c 8192 `
  -ngl 99 `
  --parallel 1
```

检查服务：

```powershell
Invoke-RestMethod http://127.0.0.1:8090/health
Invoke-RestMethod http://127.0.0.1:8090/v1/models
```

默认只监听环回地址。若改为局域网地址或 `0.0.0.0`，必须通过脚本的 `-ApiKey` 参数设置密钥，并配置防火墙访问范围：

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_llama_cpp_server.ps1 `
  -ListenHost 0.0.0.0 -ApiKey "replace-with-a-secret"
```

SDK 侧把相同密钥写入 `LlamaCppTextGenerationConfig::api_key`。

## 2. 创建文本生成 Client

```cpp
#include "dialog/llama_cpp_text_generation_client.h"

digital_human::dialog::LlamaCppTextGenerationConfig config;
config.endpoint =
    "http://127.0.0.1:8090/v1/chat/completions";
config.model = "Qwen3-4B-Q4_K_M.gguf";
config.temperature = 0.7F;
config.top_p = 0.9F;
config.max_tokens = 256;
config.stream = true;
config.enable_thinking = false;

digital_human::dialog::LlamaCppTextGenerationClient text_client(config);
```

`enable_thinking` 默认关闭。数字人需要把回复实时送入 TTS，关闭 Qwen3 thinking 可避免将思考过程送去朗读，也能降低首句延迟。

## 3. 接入 ConversationSession

llama.cpp Client 实现现有的 `ITextGenerationClient`，会话、TTS 和推流代码不需要了解模型类型：

```cpp
dialog::ConversationSession session(text_client, tts_client, media_sink);

dialog::ConversationConfig conversation;
conversation.session_id = "user-001";
conversation.system_prompt =
    "你是数字人讲解员。回答简洁、自然，每句话适合直接朗读。";

session.Start(conversation, avatar_bgr);
session.SubmitUserText("你好，请介绍一下自己");
```

每次请求自动组装：

1. `system_prompt`；
2. `ConversationSession` 保存的历史 user/assistant 消息；
3. 当前用户消息；
4. `stream`、采样参数和 `chat_template_kwargs.enable_thinking`。

服务端 SSE 的 `choices[].delta.content` 会作为增量文本送入分句器，完整句子随即进入 TTS，无需等待整段回答结束。

## 4. 完整媒体闭环

```text
用户文本
  → LlamaCppTextGenerationClient
  → llama.cpp /v1/chat/completions
  → SSE 增量分句
  → TTS PCM
  → ConversationStreamBridge
      ├─→ DigitalHumanSDK → 渲染 BGR → H.264
      └─→ AAC
  → RTMP / RTSP
```

媒体层使用方式见[音视频编码与 RTMP/RTSP 推流](stream_publishing.md)。

## 5. 配置说明

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `endpoint` | `http://127.0.0.1:8090/v1/chat/completions` | llama.cpp Chat API |
| `model` | `Qwen3-4B-Q4_K_M.gguf` | OpenAI 请求中的模型名 |
| `temperature` | `0.7` | 采样温度，设为 0 可用于确定性测试 |
| `top_p` | `0.9` | nucleus sampling |
| `max_tokens` | `256` | 单轮最大生成 token 数 |
| `stream` | `true` | 使用 SSE 增量回复 |
| `enable_thinking` | `false` | 是否启用 Qwen3 thinking |
| `cache_prompt` | `true` | 允许服务端复用公共 prompt 前缀 |
| `connect_timeout_ms` | `2000` | 建连超时 |
| `request_timeout_ms` | `120000` | 整个生成请求超时，0 表示不限制 |

`model` 字段仅作为协议参数；实际加载哪个 GGUF 由 `llama-server.exe -m` 决定。

## 6. 测试

真实服务测试：

```powershell
D:\digital-human-dev\build\digital-human-sdk-windows-ucrt64-vulkan\bin\llama_cpp_client_test.exe `
  http://127.0.0.1:8090/v1/chat/completions `
  Qwen3-4B-Q4_K_M.gguf
```

该测试覆盖：

- SSE 增量响应；
- 非流式 JSON 响应；
- system prompt、历史消息和当前用户消息；
- llama.cpp → `ConversationSession` → Fake TTS →媒体 Sink。

无模型环境可先启动 `tools/mock_dialog_service.py`，再把 endpoint 改为 `http://127.0.0.1:18080/v1/chat/completions`。

## 7. 常见问题

| 现象 | 检查项 |
|---|---|
| `Connection refused` | llama-server 是否启动，端口是否为 8090 |
| HTTP 401 | SDK 与 server 的 API key 是否一致 |
| 回复超时 | 提高 `request_timeout_ms`，检查 GPU offload 和上下文大小 |
| TTS 朗读思考过程 | 确保 `enable_thinking=false` |
| 首句延迟较高 | 使用 SSE、缩短 system prompt、限制 `max_tokens`，确认 `-ngl` 已启用 GPU |
| 多轮上下文过长 | 停止并重建 ConversationSession；后续可增加历史窗口配置 |
