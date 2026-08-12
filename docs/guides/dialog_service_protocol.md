# 会话服务接口协议

第一阶段使用两个独立 HTTP POST 服务。协议刻意保持简单，不绑定文本模型或 TTS 引擎。

## 文本生成服务

请求头：

```http
Content-Type: application/json
Accept: application/json, text/event-stream
Authorization: Bearer <可选 API Key>
```

请求体：

```json
{
  "session_id": "session-001",
  "system_prompt": "你是数字人助手",
  "user_text": "你好",
  "history": [
    {"role": "user", "content": "上一轮问题"},
    {"role": "assistant", "content": "上一轮回复"}
  ],
  "stream": true
}
```

完整 JSON 响应：

```json
{"reply": "您好，请问有什么可以帮助您？"}
```

也兼容字段名 `text`。纯文本响应也可被接受，但建议始终返回 JSON。

SSE 响应：

```text
data: {"delta":"您好，"}

data: {"delta":"请问有什么可以帮助您？"}

data: {"done":true}
```

也可以用 `data: [DONE]` 结束。第一阶段只消费 `delta` 或 `reply`，其他字段被忽略。

## TTS 服务

请求头：

```http
Content-Type: application/json
Accept: application/octet-stream
Authorization: Bearer <可选 API Key>
```

请求体：

```json
{
  "text": "您好，请问有什么可以帮助您？",
  "sample_rate": 16000,
  "channels": 1,
  "format": "pcm_s16le"
}
```

响应体为不带文件头的裸 PCM，支持：

- `pcm_s16le`：小端有符号 16 位整数，默认。
- `pcm_f32le`：小端 IEEE 754 float32。

HTTP Client 会将响应转换为 `[-1, 1]` float PCM，并按 `chunk_samples` 切块。第一阶段要求输出 16kHz、单声道；重采样属于后续适配器扩展。

## 错误约定

- HTTP 2xx 表示成功。
- 非 2xx 状态会转换为会话错误回调。
- 文本 JSON 缺少 `reply/text/delta` 时视为协议错误。
- TTS 响应为空或不是完整 PCM 样本时视为协议错误。
- 连接、请求和取消错误都通过接口的 `error` 字符串返回。

## 本地验证

```powershell
python tools/mock_dialog_service.py --port 18080

http_service_client_test.exe `
  http://127.0.0.1:18080/text `
  http://127.0.0.1:18080/tts

# SSE 文本响应
http_service_client_test.exe `
  http://127.0.0.1:18080/text-sse `
  http://127.0.0.1:18080/tts
```

mock 服务只用于接口测试，不是生产服务实现。
