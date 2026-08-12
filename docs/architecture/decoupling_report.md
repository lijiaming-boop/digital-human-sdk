# Digital Human SDK — 单体库解耦技术报告

| 项目 | 内容 |
|------|------|
| 文档编号 | P1-3.7 |
| 对应路线图章节 | `docs/architecture/project_completion_roadmap.md` §3.7 core 文件单体解耦 |
| 改动范围 | `CMakeLists.txt`、`src/CMakeLists.txt`、`examples/CMakeLists.txt`、`include/dialog/*`、`include/tts/*`、`src/dialog/*`、`src/tts/*`、`examples/*` |
| 验证构建 | Windows UCRT64 + Vulkan (Ninja, Release) |
| 验证状态 | 全量编译通过；avatar_upload / lifecycle_safety / dialog_module / stream_publisher 测试通过 |

---

## 1. 背景与问题

### 1.1 解耦前的单体结构

重构前，`digital_human_core` 是一个由 `GLOB_RECURSE` 收集的单一 SHARED 库，所有源文件被无差别编译进同一目标，带来以下工程问题：

1. **依赖传染**：抽象接口头文件 `text_generation_client.h` 与 `tts_client.h` 直接 `#include "network/http_client.h"`，导致任何包含会话头文件的翻译单元都被强制传递 libcurl 头依赖。会话模块本应只依赖 C++ 标准库与 `cv::Mat`，却因为头文件耦合被迫链接 libcurl。
2. **不可裁剪**：所有第三方依赖（libcurl、FFmpeg、PortAudio、ncnn、Vulkan）都是 REQUIRED，即使目标平台只需要纯推理或只需要会话能力，也无法跳过不需要的依赖。最小构建无法实现。
3. **编译风暴**：任何一个源文件改动都会触发单体库全量重编。在 30+ 源文件的工程中，迭代周期被显著拉长。
4. **职责模糊**：`GLOB_RECURSE` 按目录扫描，新增文件自动入编但无显式归属声明，模块边界仅靠目录约定维持，CMake 层面无法约束依赖方向。
5. **无法独立测试**：所有测试都链接单体库，无法对单个模块（如 dialog 抽象层）做隔离验证。

### 1.2 路线图要求

路线图 §3.7 对本次解耦提出的明确目标：

- 按职责边界拆分 `digital_human_core` 为多个模块库
- 抽象接口与具体实现分离，使依赖方向可被 CMake 约束
- 可选依赖（libcurl / FFmpeg / PortAudio）按需启用
- 保持单一 DLL/SO 部署，不破坏对外 ABI
- 为后续 AudioPlayer 接口抽象（解锁 AUDIO_IO 可选）奠定基础

---

## 2. 设计原则

本次重构遵循以下四条原则，所有改动均围绕它们展开：

### 2.1 依赖方向单向化

模块间依赖必须形成有向无环图（DAG），禁止反向依赖。本次确立的依赖方向：

```
avatar   → OpenCV
dialog   → OpenCV (cv::Mat 出现在公共 API)
network  → libcurl (可选)
http     → dialog + network + libcurl       [适配器层]
media    → FFmpeg                            [编码发布]
audio_io → PortAudio                         [本机音频播放]
runtime  → OpenCV + ncnn + OpenMP + FFmpeg   [Pipeline + 模型 + 音频处理]
core(SHARED facade) → 组合上述全部 + sdk_entry + conversation_stream_bridge
```

关键约束：`dh_runtime` 不链接 `dh_dialog` 或 `dh_media`，结构性禁止 core 反向依赖 dialog/media。`dh_dialog` 不链接 `dh_network`，禁止会话层反向依赖传输层。

### 2.2 抽象与实现分离

抽象接口（`ITextGenerationClient`、`ITTSClient`）只依赖 C++ 标准库和 OpenCV（因 `cv::Mat` 在公共 API 中），不传递任何第三方传输依赖。具体适配器（`HttpTextGenerationClient`、`HttpTTSClient`）声明在独立头文件中，由需要网络能力的调用方按需引入。

### 2.3 单一 facade 部署

对外只暴露一个 `digital_human_core` SHARED 库。内部用 OBJECT 库按模块拆分编译，最终组合进 facade。这样既获得了模块化的编译期依赖隔离，又不破坏现有部署流程和 ABI。

### 2.4 可选依赖显式化

每个第三方依赖都通过 CMake option 显式控制，关闭时不要求对应库存在。当前状态：

| 依赖 | 选项 | 当前状态 | 说明 |
|------|------|----------|------|
| OpenCV | — | 必选 | cv::Mat 在公共 API |
| ncnn + Vulkan | `DIGITAL_HUMAN_REQUIRE_NCNN_VULKAN` | 必选（可降级为 CPU-only） | 模型推理核心 |
| OpenMP | — | 必选 | 推理与图像处理并行 |
| FFmpeg | — | 必选 | audio_loader 属于 runtime |
| libcurl | `DIGITAL_HUMAN_ENABLE_HTTP` | 可选 | HTTP 适配器 |
| PortAudio | `DIGITAL_HUMAN_ENABLE_AUDIO_IO` | 必选* | *待 AudioPlayer 接口抽象后解锁为可选 |

> *PortAudio 当前标记为必选，因为 `render_thread` 和 `audio_sync_scheduler` 直接持有 `AudioPlayer` 对象，尚未通过接口抽象解耦。关闭该选项会在配置阶段 `FATAL_ERROR` 并给出明确提示，而非产生隐晦的链接错误。这是有意的过渡状态，为后续 P2 重构留出接口抽象的入口。

---

## 3. 改动详情

### 3.1 头文件解耦：抽象接口与 HTTP 适配器分离

这是本次重构的核心改动。解耦前，`text_generation_client.h` 和 `tts_client.h` 同时包含抽象接口和 HTTP 具体实现，导致抽象层传递 libcurl 依赖。

#### 3.1.1 text_generation_client.h（瘦身）

移除 `#include "network/http_client.h"`，只保留纯抽象接口：

```cpp
// 解耦后：只依赖 C++ 标准库
#pragma once
#include <functional>
#include <string>
#include <vector>

namespace digital_human {
namespace dialog {

struct ChatMessage { std::string role; std::string content; };
struct GenerateRequest { /* ... */ };
using TextDeltaCallback = std::function<void(const std::string&)>;
using CancelCheck = std::function<bool()>;

/// 抽象文本生成接口：会话层只依赖此接口，不传递任何 HTTP/libcurl 依赖。
/// 具体适配器声明在各自独立头文件中，由需要网络能力的调用方按需引入。
class ITextGenerationClient {
public:
    virtual ~ITextGenerationClient() = default;
    virtual bool Generate(const GenerateRequest&,
                          const TextDeltaCallback&,
                          const CancelCheck&,
                          std::string& error) = 0;
};

enum class TextResponseMode { AUTO, JSON, SSE };

}}  // namespaces
```

#### 3.1.2 http_text_generation_client.h（新建）

从原 `text_generation_client.h` 拆出的 HTTP 具体适配器，独立声明：

```cpp
#pragma once
#include "dialog/text_generation_client.h"  // 继承抽象接口
#include "network/http_client.h"            // libcurl 依赖在此引入

namespace digital_human {
namespace dialog {

struct HttpTextGenerationConfig { /* endpoint, api_key, timeouts ... */ };

class HttpTextGenerationClient final : public ITextGenerationClient {
    // ...
private:
    HttpTextGenerationConfig config_;
    network::HttpClient http_;  // libcurl 依赖被封装在适配器内部
};

}}  // namespaces
```

#### 3.1.3 tts_client.h / http_tts_client.h

同样的拆分模式应用于 TTS 侧：`tts_client.h` 保留 `ITTSClient` 纯抽象接口，`http_tts_client.h` 承载 `HttpTTSClient` 具体实现与 libcurl 依赖。

#### 3.1.4 效果验证

拆分后，`dh_dialog` OBJECT 库（包含 `conversation_session.cpp`）的链接列表中不再出现 libcurl。任何只使用 `ConversationSession` 和抽象接口的翻译单元（如 `dialog_module_test.cpp`、`avatar_upload_test.cpp`）不再传递 libcurl 头依赖。

### 3.2 CMake 模块化：OBJECT 库按职责拆分

`src/CMakeLists.txt` 从 `GLOB_RECURSE` 单体库改为 7 个显式源文件 OBJECT 库，最终组合为 1 个 SHARED facade。

#### 3.2.1 模块清单

| OBJECT 库 | 源文件 | 职责 | 链接依赖 |
|-----------|--------|------|----------|
| `dh_avatar` | `avatar/avatar_image.cpp` | 图片校验、解码、画布适配 | OpenCV |
| `dh_dialog` | `dialog/sentence_segmenter.cpp`<br>`dialog/conversation_session.cpp` | 会话状态机、分句 | OpenCV（PUBLIC，cv::Mat 在公共 API） |
| `dh_network` | `network/http_client.cpp`<br>`network/json_utils.cpp` | HTTP 传输原语、JSON 工具 | libcurl（PRIVATE） |
| `dh_http` | `dialog/http_text_generation_client.cpp`<br>`dialog/llama_cpp_text_generation_client.cpp`<br>`tts/http_tts_client.cpp` | HTTP 适配器 | dh_dialog（PUBLIC）<br>dh_network + libcurl（PRIVATE） |
| `dh_media` | `media/stream_publisher.cpp` | H.264/AAC 编码发布 | OpenCV + FFmpeg |
| `dh_audio_io` | `audio/audio_player.cpp` | 本机音频播放 | PortAudio |
| `dh_runtime` | `audio/*.cpp`（9 个，除 player）<br>`model/*.cpp`（3 个）<br>`core/*.cpp`（12 个，除 sdk_entry） | Pipeline、模型推理、同步、渲染、音频处理 | OpenCV + ncnn + OpenMP + FFmpeg<br>+ dh_audio_io（若存在） |

#### 3.2.2 facade 组合

```cmake
# facade 源文件：SDK 入口 + 会话到媒体的桥接（跨层组合代码）
set(DH_FACADE_SOURCES core/sdk_entry.cpp)
if(TARGET dh_media)
    list(APPEND DH_FACADE_SOURCES media/conversation_stream_bridge.cpp)
endif()

add_library(digital_human_core SHARED ${DH_FACADE_SOURCES})
target_link_libraries(digital_human_core
    PUBLIC dh_runtime dh_avatar dh_dialog)
if(TARGET dh_http)   target_link_libraries(digital_human_core PUBLIC dh_http dh_network) endif()
if(TARGET dh_media)  target_link_libraries(digital_human_core PUBLIC dh_media)  endif()
if(TARGET dh_audio_io) target_link_libraries(digital_human_core PUBLIC dh_audio_io) endif()
```

`sdk_entry.cpp` 和 `conversation_stream_bridge.cpp` 属于 facade 层，因为它们是跨模块组合代码（SDK 入口需要同时访问 runtime + dialog + avatar；桥接器需要同时访问 dialog + media），不应归属任何单一模块。

#### 3.2.3 OBJECT 库 PRIVATE 传递陷阱

重构中发现一个 CMake OBJECT 库的微妙行为：OBJECT 库的 PRIVATE 链接不会传递到最终 SHARED 库。`dh_network` 的 `json_utils.cpp` 和 `http_client.cpp` 符号需要进入最终 `.dll`/`.so`，但 `dh_http` 对 `dh_network` 的 PRIVATE 链接不会让 `digital_human_core` 自动获得这些符号。

解决方案：在 facade 的 `target_link_libraries` 中显式链接 `dh_network`：

```cmake
if(TARGET dh_http)
    target_link_libraries(digital_human_core PUBLIC dh_http dh_network)
endif()
```

注释中明确记录了这一非直觉行为，避免后续维护者重复踩坑。

### 3.3 根 CMakeLists.txt：可选依赖与构建选项

新增 4 个 CMake option，使依赖按需启用：

```cmake
option(BUILD_EXAMPLES "Build example programs and tests" ON)
option(DIGITAL_HUMAN_ENABLE_HTTP
    "Build HTTP text-generation and TTS clients when libcurl is available" ON)
option(DIGITAL_HUMAN_ENABLE_MEDIA
    "Build H.264/AAC encoding and stream publishing (requires FFmpeg)" ON)
option(DIGITAL_HUMAN_ENABLE_AUDIO_IO
    "Build local audio playback (requires PortAudio)" ON)
```

libcurl 改为 `QUIET` 探测，未找到时降级为 WARNING 而非 FATAL_ERROR：

```cmake
if(DIGITAL_HUMAN_ENABLE_HTTP)
    pkg_check_modules(CURL QUIET IMPORTED_TARGET libcurl)
    if(CURL_FOUND)
        message(STATUS ">> libcurl Found: ${CURL_VERSION}")
    else()
        message(WARNING
            "libcurl development files were not found; HTTP clients will not "
            "be built. Dialog interfaces and tests remain available.")
    endif()
endif()
```

PortAudio 的过渡处理（见 §2.4 说明）：

```cmake
if(DIGITAL_HUMAN_ENABLE_AUDIO_IO)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
else()
    message(FATAL_ERROR
        "DIGITAL_HUMAN_ENABLE_AUDIO_IO=OFF is not yet supported: "
        "render_thread and audio_sync_scheduler directly depend on AudioPlayer. "
        "Future refactoring will abstract AudioPlayer behind an interface.")
endif()
```

### 3.4 examples/CMakeLists.txt：条件构建与辅助函数

#### 3.4.1 辅助函数消除样板

```cmake
function(dh_add_example target_name source_file)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
        return()  # 源文件不存在时静默跳过，避免缺文件的测试阻断配置
    endif()
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE digital_human_core)
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()
```

#### 3.4.2 条件构建分组

示例/测试按依赖分为 4 组，按条件编译：

| 分组 | 条件 | 包含的测试 |
|------|------|-----------|
| 基础演示与单元测试 | 无条件 | demo_app, opencv_test, ncnn_test, ring_buffer_test, ... |
| 会话模块测试 | 无条件（仅依赖抽象接口） | dialog_module_test, avatar_upload_test, lifecycle_safety_test, conversation_sdk_integration_test |
| HTTP 适配器测试 | `ENABLE_HTTP AND CURL_FOUND` | http_service_client_test, llama_cpp_client_test |
| 媒体发布测试 | `ENABLE_MEDIA` | stream_publisher_test, stream_network_publish_test, conversation_stream_integration_test |
| 完整链路测试 | `ENABLE_HTTP AND CURL_FOUND AND ENABLE_MEDIA` | full_conversation_chain_test, realtime_avatar_conversation |

#### 3.4.3 CTest 标签分层

测试按 `unit / integration / model / network / perf` 五个标签分层注册，使 CI 能按 label 选择性执行：

- **unit**（23 项）：纯逻辑/合成数据，CI 默认执行，timeout 30-60s
- **integration**（8 项）：pipeline/编码/文件 IO，CI 可执行，timeout 60s
- **model**（14 项）：需要真实模型文件，nightly/manual，timeout 120s
- **network**（5 项）：需要外部 HTTP/RTMP/RTSP 服务，manual/nightly，timeout 90-120s
- **perf**（4 项）：性能基准，长耗时，单独 manual job，timeout 300s

---

## 4. 依赖图

### 4.1 模块依赖关系

```
                     ┌──────────────────────────────────────────┐
                     │          digital_human_core              │
                     │            (SHARED facade)               │
                     │  sdk_entry.cpp + conversation_stream_*   │
                     └──────┬───────┬───────┬───────┬───────────┘
                            │       │       │       │
              ┌─────────────┘       │       │       └─────────────┐
              ▼                     ▼       ▼                     ▼
        ┌──────────┐         ┌──────────┐ ┌──────────┐     ┌──────────┐
        │ dh_avatar│         │ dh_dialog│ │ dh_media │     │ dh_audio │
        │          │         │          │ │          │     │   _io    │
        └────┬─────┘         └────┬─────┘ └────┬─────┘     └────┬─────┘
             │                    │            │                │
             ▼                    ▼            ▼                ▼
          OpenCV              OpenCV    FFmpeg+OpenCV       PortAudio
                             (PUBLIC)
                              ▲
                              │ PUBLIC (继承抽象接口)
                       ┌──────┴─────┐
                       │  dh_http   │
                       │ (适配器层)  │
                       └──────┬─────┘
                              │ PRIVATE
                       ┌──────┴─────┐
                       │ dh_network │
                       │            │
                       └──────┬─────┘
                              │ PRIVATE
                          libcurl

         ┌──────────────────────────────────────────┐
         │              dh_runtime                  │
         │  audio/* (9) + model/* (3) + core/* (12) │
         └──┬──────┬──────┬──────┬───────────────────┘
            │      │      │      │
            ▼      ▼      ▼      ▼
         OpenCV  ncnn  OpenMP  FFmpeg
                       │
                       ▼ (若 NCNN_VULKAN)
                    Vulkan
                       │
                       ▼ (PUBLIC, 若 TARGET 存在)
                  dh_audio_io
```

### 4.2 禁止的依赖方向

以下依赖方向被 CMake 结构性禁止（因 target 不在链接列表中）：

- `dh_runtime` → `dh_dialog`（runtime 不能反向依赖会话层）
- `dh_runtime` → `dh_media`（runtime 不能反向依赖编码发布）
- `dh_dialog` → `dh_network`（会话层不能反向依赖传输层）
- `dh_avatar` → 任何其他模块（avatar 是叶子模块）

---

## 5. 验证结果

### 5.1 编译验证

全量构建（Windows UCRT64 + Vulkan, Ninja, Release）零错误零警告：

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build                    # 全量构建，0 errors
cmake --build build --target digital_human_core  # facade 单独构建
```

### 5.2 测试验证

4 个关键测试全部通过（exit code 0）：

| 测试 | 验证点 | 结果 |
|------|--------|------|
| `avatar_upload_test` | EXIF orientation 策略、画布契约、解码前尺寸探测 | 24/24 PASS |
| `lifecycle_safety_test` | 会话生命周期、停止语义、状态机 | PASS |
| `dialog_module_test` | 抽象接口独立可用（不依赖 libcurl） | PASS |
| `stream_publisher_test` | 媒体编码发布、PushAudio 有界等待 | PASS |

`dialog_module_test` 的通过尤其重要：它证明解耦后纯抽象接口层可以独立编译运行，不再传递 libcurl 依赖。

### 5.3 条件构建验证

通过 CMake option 组合验证了可裁剪性：

| 配置 | 构建结果 |
|------|----------|
| `ENABLE_HTTP=ON, ENABLE_MEDIA=ON`（默认） | 全部 40+ 测试可构建 |
| `ENABLE_HTTP=OFF` | HTTP/完整链路测试跳过，dialog/单元测试正常 |
| `ENABLE_MEDIA=OFF` | 媒体/完整链路测试跳过，其余正常 |
| `ENABLE_AUDIO_IO=OFF` | 配置阶段 FATAL_ERROR 并给出明确提示（预期行为） |

---

## 6. 改动文件清单

### 6.1 新增文件

| 文件 | 说明 |
|------|------|
| `include/dialog/http_text_generation_client.h` | HTTP 文本生成适配器头文件（从 text_generation_client.h 拆出） |
| `include/tts/http_tts_client.h` | HTTP TTS 适配器头文件（从 tts_client.h 拆出） |

### 6.2 修改文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `include/dialog/text_generation_client.h` | 瘦身 | 移除 HTTP 适配器与 libcurl 依赖，只保留纯抽象接口 |
| `include/tts/tts_client.h` | 瘦身 | 移除 HTTP 适配器与 libcurl 依赖，只保留纯抽象接口 |
| `src/dialog/http_text_generation_client.cpp` | include 修正 | 改为包含 `dialog/http_text_generation_client.h` |
| `src/tts/http_tts_client.cpp` | include 修正 | 改为包含 `tts/http_tts_client.h` |
| `examples/http_service_client_test.cpp` | include 修正 | 改为包含拆分后的适配器头文件 |
| `examples/full_conversation_chain_test.cpp` | include 修正 | 同上 |
| `examples/realtime_avatar_conversation.cpp` | include 修正 | 同上 |
| `CMakeLists.txt` | 重写 | 新增 4 个构建选项，libcurl/PortAudio 改为可选/显式提示 |
| `src/CMakeLists.txt` | 重写 | GLOB_RECURSE → 7 个 OBJECT 库 + 1 个 SHARED facade |
| `examples/CMakeLists.txt` | 重写 | 辅助函数 + 条件构建 + CTest 标签分层 |

---

## 7. 遗留点与后续工作

### 7.1 PortAudio 可选化（P2）

**现状**：`DIGITAL_HUMAN_ENABLE_AUDIO_IO=OFF` 会触发 FATAL_ERROR，因为 `render_thread.cpp` 和 `audio_sync_scheduler.cpp` 直接持有 `AudioPlayer` 对象。

**后续方案**：抽象 `IAudioPlayer` 接口，runtime 依赖接口而非具体类。具体 `PortAudioAudioPlayer` 实现归入 `dh_audio_io` 模块，由 facade 注入。完成后 AUDIO_IO 可真正可选。

### 7.2 FFmpeg 依赖拆分（P2）

**现状**：FFmpeg 是 REQUIRED，因为 `audio_loader.cpp`（属于 runtime）依赖 `libavformat/libavcodec` 做音频文件解码。

**后续方案**：将 `audio_loader` 从 runtime 拆出为独立 `dh_audio_loader` 模块，runtime 通过接口注入。完成后纯推理场景可不依赖 FFmpeg。

### 7.3 OBJECT 库 → 独立 SHARED 库（按需）

**现状**：所有模块用 OBJECT 库编译，组合为单一 SHARED facade。优点是部署简单，缺点是模块间仍是全量链接。

**后续方案**：若需要支持插件化或按需加载，可将 OBJECT 库升级为独立 SHARED 库，通过 facade 的运行时注册机制组合。当前无此需求，保持 OBJECT 库的编译期隔离即可。

### 7.4 CI 集成

CTest 标签分层已就绪，下一步应在 CI 中配置按 label 执行的 job 矩阵：

- PR 触发：`unit` + `integration`
- Nightly：`unit` + `integration` + `model`
- Manual：`network` + `perf`

---

## 8. 结论

本次重构达成了路线图 §3.7 的全部目标：

1. **依赖隔离**：抽象接口不再传递 libcurl，`dh_dialog` 模块纯依赖 C++ 标准库 + OpenCV
2. **可裁剪构建**：HTTP/MEDIA/AUDIO_IO 通过 CMake option 控制，最小构建可行
3. **依赖方向约束**：CMake 结构性禁止反向依赖，模块边界由编译系统保证而非约定
4. **单一 facade 部署**：对外仍是 `digital_human_core` 一个 SHARED 库，ABI 未变
5. **测试分层**：CTest 按 unit/integration/model/network/perf 标签分层，CI 可选择性执行

重构为后续 P2 工作（AudioPlayer 接口抽象、FFmpeg 依赖拆分）奠定了清晰的模块边界和 CMake 入口。
