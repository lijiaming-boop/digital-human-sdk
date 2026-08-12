# Digital Human SDK — P1 级实施方案

| 项目 | 内容 |
|------|------|
| 文档编号 | P1-IMPL |
| 对应路线图 | `docs/architecture/project_completion_roadmap.md` §3.7–§3.10 |
| 前置条件 | P0（§3.1–§3.6）已完成并验证 |
| 当前状态 | §3.7 模块化已部分完成（OBJECT 库拆分），其余待实施 |
| 关联文档 | [单体库解耦技术报告](decoupling_report.md) |

---

## 目录

- [P1-3.7 模块化独立 target 完善](#p1-37-模块化独立-target-完善)
- [P1-3.8 SDK 安装、ABI 与版本策略](#p1-38-sdk-安装abi-与版本策略)
- [P1-3.9 可观测性体系](#p1-39-可观测性体系)
- [P1-3.10 多轮对话产品级能力](#p1-310-多轮对话产品级能力)
- [实施顺序与依赖关系](#实施顺序与依赖关系)

---

## P1-3.7 模块化独立 target 完善

### 现状

OBJECT 库拆分已完成（7 个模块 + 1 个 SHARED facade），编译期依赖方向已被 CMake 约束。但 OBJECT 库只是内部编译组织，外部 consumer 无法只链接 `digital_human_dialog` 等子模块。

### 目标

将关键模块从 OBJECT 库升级为可独立安装的 STATIC/SHARED 库，使外部项目能按需链接子模块。

### 实施步骤

#### 步骤 1：将 OBJECT 库改为 STATIC 库

```cmake
# src/CMakeLists.txt 改动示例
# OBJECT → STATIC，使每个模块可独立安装
add_library(dh_dialog STATIC
    dialog/sentence_segmenter.cpp
    dialog/conversation_session.cpp
)
# STATIC 库的 PRIVATE 依赖会正确传递到最终 SHARED facade
```

| 模块 | 库类型 | 理由 |
|------|--------|------|
| `dh_dialog` | STATIC | 无第三方依赖，适合独立链接 |
| `dh_avatar` | STATIC | 仅依赖 OpenCV，适合独立链接 |
| `dh_network` | STATIC | libcurl 依赖封装在内部 |
| `dh_http` | STATIC | 依赖 dialog + network |
| `dh_media` | STATIC | FFmpeg 依赖封装在内部 |
| `dh_audio_io` | STATIC | PortAudio 依赖封装在内部 |
| `dh_runtime` | STATIC | 依赖较重，但 STATIC 可被 facade 正确聚合 |
| `digital_human_core` | SHARED | 对外唯一动态库，保持单一 DLL/SO 部署 |

#### 步骤 2：符号可见性控制

为每个模块设置默认符号隐藏，只导出公共 API：

```cmake
# 对所有模块 target 应用
set_target_properties(dh_dialog dh_avatar dh_network dh_http
    dh_media dh_audio_io dh_runtime digital_human_core
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)
```

公共头文件中的类需添加导出宏（见 P1-3.8）。

#### 步骤 3：安装规则

```cmake
# src/CMakeLists.txt 末尾
install(TARGETS digital_human_core
    EXPORT DigitalHumanSDKTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# 安装公共头文件（按目录结构保留）
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
```

### 验证标准

- `cmake --install` 后，`lib/` 下有 `digital_human_core.dll/.so/.dylib`
- `include/` 下有完整公共头目录结构
- 各 STATIC 库在 `lib/` 下可选安装（通过 `INSTALL_STATIC_MODULES` 选项控制）

---

## P1-3.8 SDK 安装、ABI 与版本策略

### 现状

- 项目定义了 `VERSION 0.1.0`，但未用于库的 `VERSION/SOVERSION`
- 无 `install()` 规则，无 CMake package export
- 无 Windows 导出宏（`DIGITAL_HUMAN_API`），SHARED 库符号全部默认可见
- 公共头直接暴露 `cv::Mat`、`std::vector`、`std::string`、`std::function`
- 三套错误体系并存：SDK 层 `SDKError` 枚举、会话层 `bool + string`、Pipeline 层 `bool + stderr`

### 目标

1. 提供 `find_package(DigitalHumanSDK)` 能力
2. 库有 `VERSION/SOVERSION`，Windows 有导出宏
3. 提供版本化 C API（opaque handle + POD 配置），保证跨编译器 ABI
4. 统一错误结构（错误码 + 模块 + 消息 + 底层原因）
5. 公共头不泄漏 ncnn/curl/FFmpeg 内部类型

### 实施步骤

#### 步骤 1：版本号与 SOVERSION

```cmake
# 根 CMakeLists.txt
project(DigitalHumanSDK VERSION 0.2.0 LANGUAGES CXX)

# src/CMakeLists.txt
set_target_properties(digital_human_core PROPERTIES
    VERSION ${PROJECT_VERSION}      # 0.2.0 → libdigital_human_core.so.0.2.0
    SOVERSION 0                     → libdigital_human_core.so.0
)
```

版本政策：

| 版本号变化 | 含义 | ABI 兼容性 |
|------------|------|------------|
| MAJOR (0→1) | 不兼容的 API 变更 | 不保证 |
| MINOR (0→2) | 向后兼容的功能新增 | 保证 |
| PATCH (0→1) | 向后兼容的缺陷修复 | 保证 |

SOVERSION 只在 MAJOR 变更时递增。

#### 步骤 2：Windows 导出宏

新建 `include/digital_human/export.h`：

```cpp
#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef DIGITAL_HUMAN_CORE_EXPORTS
        #define DH_API __declspec(dllexport)
    #else
        #define DH_API __declspec(dllimport)
    #endif
#else
    #define DH_API __attribute__((visibility("default")))
#endif

#define DH_LOCAL __attribute__((visibility("hidden")))
```

CMake 中编译 facade 时定义 `DIGITAL_HUMAN_CORE_EXPORTS`：

```cmake
target_compile_definitions(digital_human_core PRIVATE DIGITAL_HUMAN_CORE_EXPORTS)
```

在公共头中对所有公共类/函数添加 `DH_API`：

```cpp
// include/digital_human_sdk.h
class DH_API DigitalHumanSDK {
    // ...
};
```

#### 步骤 3：CMake package export

创建 `cmake/DigitalHumanSDKConfig.cmake.in`：

```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# 必选依赖
find_dependency(OpenCV)
find_dependency(OpenMP)
find_dependency(ncnn CONFIG)

# 可选依赖（按编译时配置决定是否 find_dependency）
if(@DIGITAL_HUMAN_ENABLE_HTTP@)
    find_dependency(CURL)
endif()
if(@DIGITAL_HUMAN_ENABLE_MEDIA@)
    # FFmpeg 通过 pkg-config 查找，此处使用 pkg_check_modules
endif()

include("${CMAKE_CURRENT_LIST_DIR}/DigitalHumanSDKTargets.cmake")

check_required_components(DigitalHumanSDK)
```

在根 CMakeLists.txt 末尾添加：

```cmake
install(EXPORT DigitalHumanSDKTargets
    FILE DigitalHumanSDKTargets.cmake
    NAMESPACE DigitalHumanSDK::
    DESTINATION lib/cmake/DigitalHumanSDK
)

configure_package_config_file(
    cmake/DigitalHumanSDKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/DigitalHumanSDKConfig.cmake
    INSTALL_DESTINATION lib/cmake/DigitalHumanSDK
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/DigitalHumanSDKConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/DigitalHumanSDKConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/DigitalHumanSDKConfigVersion.cmake
    DESTINATION lib/cmake/DigitalHumanSDK
)
```

外部 consumer 使用方式：

```cmake
find_package(DigitalHumanSDK 0.2 REQUIRED)
target_link_libraries(my_app PRIVATE DigitalHumanSDK::digital_human_core)
```

#### 步骤 4：版本化 C API

新建 `include/digital_human/c_api.h`，提供 opaque handle + POD 配置：

```cpp
#pragma once
#include <stdint.h>
#include "digital_human/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 版本号查询，用于运行时 ABI 检查
DH_API uint32_t dh_sdk_api_version(void);  // 返回 0x00020000 (0.2.0)

/// Opaque handle
typedef struct dh_sdk_t dh_sdk_t;

/// POD 配置（纯 C 兼容，无 STL 类型）
typedef struct {
    uint32_t struct_size;  // 必须设为 sizeof(dh_sdk_config_t)，用于版本检查
    const char* model_path;
    const char* face_model_path;
    int   width;
    int   height;
    int   fps;
    int   audio_sample_rate;
    int   audio_channels;
    int   inference_threads;
    int   enable_gpu;       // 0=CPU, 1=Vulkan
} dh_sdk_config_t;

/// 错误结构（统一错误体系的 C 映射）
typedef struct {
    uint32_t struct_size;
    int      error_code;      // DH_ERROR_OK / DH_ERROR_MODEL / ...
    const char* module;       // "pipeline" / "dialog" / "media" / ...
    const char* message;      // 可读消息
    const char* cause;        // 底层原因（可为 NULL）
} dh_error_t;

/// 生命周期
DH_API dh_sdk_t* dh_sdk_create(const dh_sdk_config_t* config, dh_error_t* error);
DH_API void      dh_sdk_destroy(dh_sdk_t* sdk);
DH_API int       dh_sdk_start(dh_sdk_t* sdk, dh_error_t* error);
DH_API int       dh_sdk_stop(dh_sdk_t* sdk, int timeout_ms, dh_error_t* error);

/// 音视频供料（buffer 生命周期由调用者管理）
DH_API int dh_sdk_push_audio(dh_sdk_t* sdk,
                              const float* samples,
                              int64_t sample_count,
                              int64_t pts_ms,
                              dh_error_t* error);
DH_API int dh_sdk_push_video(dh_sdk_t* sdk,
                              const uint8_t* bgr_data,
                              int width, int height, int stride,
                              int64_t pts_ms,
                              dh_error_t* error);

#ifdef __cplusplus
}
#endif
```

C API 实现封装在 `src/core/c_api.cpp`，内部调用 `DigitalHumanSDK` 的 PIMPL 实现。

#### 步骤 5：统一错误结构

新建 `include/digital_human/error.h`：

```cpp
#pragma once
#include <string>
#include "digital_human/export.h"

namespace digital_human {

/// 统一错误码（替代原 SDKError，增加模块归属）
enum class ErrorCategory {
    OK = 0,
    Config,
    Lifecycle,
    Model,
    Audio,
    Video,
    Sync,
    Dialog,
    TTS,
    Network,
    Media,
    Timeout,
    Cancelled,
    Unknown,
};

/// 统一错误结构
struct DH_API Error {
    ErrorCategory category = ErrorCategory::OK;
    std::string   module;    // "pipeline.audio_processor" / "dialog.session" / ...
    std::string   message;   // 可读消息
    std::string   cause;     // 底层原因（可为空）

    bool ok() const { return category == ErrorCategory::OK; }
    std::string to_string() const;
};

}  // namespace digital_human
```

迁移策略：

1. 新建 `Error` 结构，与现有 `SDKError` 并存
2. `SDKError` 保留为兼容别名，内部映射到 `Error`
3. 新增接口直接使用 `Error`，旧接口在下一个 MAJOR 版本移除
4. `ConversationSession::on_error` 回调签名从 `string` 升级为 `Error`

#### 步骤 6：公共头审计

当前公共头泄漏的第三方类型：

| 头文件 | 泄漏类型 | 处理方式 |
|--------|----------|----------|
| `digital_human_sdk.h` | `cv::Mat` (FrameCallback, PushVideo) | 保留（OpenCV 是必选依赖），但 C API 提供 `uint8_t*` 重载 |
| `digital_human_sdk.h` | `std::vector<float>` (PushAudio) | C API 提供 `const float*` 重载 |
| `conversation_session.h` | `cv::Mat` (UpdateAvatar, PushVideo) | 保留，C API 提供字节缓冲重载 |
| `stream_publisher.h` | FFmpeg 类型（不直接暴露） | 确认无泄漏 |
| `text_generation_client.h` | 无泄漏（已解耦） | — |
| `tts_client.h` | 无泄漏（已解耦） | — |

### 验证标准

- [ ] `cmake --install` 产出 `lib/cmake/DigitalHumanSDK/` 目录，包含 Config + Targets + Version
- [ ] 独立测试项目通过 `find_package(DigitalHumanSDK 0.2)` 成功链接
- [ ] `digital_human_core.so.0.2.0` + `.so.0` 软链接正确生成
- [ ] Windows 下 `digital_human_core.dll` 只导出标记 `DH_API` 的符号
- [ ] C API 测试程序能用 `gcc` 编译并运行（验证跨编译器 ABI）
- [ ] `Error` 结构覆盖所有错误路径，`SDKError` 映射正确
- [ ] `nm -D libdigital_human_core.so | grep ncnn` 无输出（ncnn 类型不泄漏）

---

## P1-3.9 可观测性体系

### 现状

- **日志**：154 处 `std::cout/std::cerr/fprintf` 直出，无 logger 抽象，无级别，无 session_id/turn_id
- **指标**：Pipeline 层有 `PipelineMetrics`/`InferenceMetrics`/`FrameStats`，但无 turn 级追踪，无 p50/p95/p99，会话/LLM/TTS/Avatar/Publisher 层完全缺失
- **无指标导出 API**：`GetMetrics()` 只返回当前快照，无法转为 Prometheus/OpenTelemetry

### 目标

1. 统一 logger 接口，替换全部 154 处直出
2. 日志携带 `session_id` / `turn_id` / 模块名 / 级别
3. turn 级指标覆盖会话/LLM/TTS/Avatar/Publisher
4. 指标支持 p50/p95/p99 分位数
5. 指标快照 API 可被上层导出为 Prometheus/OpenTelemetry

### 实施步骤

#### 步骤 1：Logger 接口

新建 `include/digital_human/logger.h`：

```cpp
#pragma once
#include <memory>
#include <string>
#include "digital_human/export.h"

namespace digital_human {

enum class LogLevel {
    Debug,
    Info,
    Warn,
    Error,
};

/// 日志记录（单条）
struct LogRecord {
    LogLevel    level;
    std::string module;       // "pipeline.audio_processor"
    std::string session_id;   // 可为空（非会话上下文）
    uint64_t    turn_id = 0;  // 0 表示非 turn 上下文
    std::string message;
    int64_t     timestamp_ms = 0;  // epoch ms，0 表示由 logger 填充
};

/// 抽象 logger 接口：SDK 不直接决定输出位置
class DH_API ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(const LogRecord& record) = 0;
    /// 返回当前允许的最低级别，低于此级别的日志将被丢弃
    virtual LogLevel min_level() const { return LogLevel::Info; }
};

/// 默认 logger：输出到 stderr（带颜色），用于无注入时的回退
DH_API std::shared_ptr<ILogger> create_default_logger(LogLevel min = LogLevel::Info);

/// 全局 logger 注入（线程安全）
/// 传入 nullptr 重置为默认 logger
DH_API void set_global_logger(std::shared_ptr<ILogger> logger);
DH_API std::shared_ptr<ILogger> get_global_logger();

}  // namespace digital_human
```

#### 步骤 2：日志上下文传播

新建 `include/digital_human/log_context.h`，提供 RAII 上下文绑定：

```cpp
#pragma once
#include <string>
#include "digital_human/logger.h"

namespace digital_human {

/// 线程局部日志上下文：自动为当前线程的日志填充 session_id/turn_id
class DH_API LogContext {
public:
    LogContext(const std::string& session_id, uint64_t turn_id);
    ~LogContext();

    // 不可拷贝，可移动
    LogContext(LogContext&&) noexcept;
    LogContext& operator=(LogContext&&) noexcept;

    /// 获取当前线程的上下文（无上下文时返回 nullptr）
    static const LogContext* current();
};

}  // namespace digital_human
```

`ConversationSession` 在进入 turn 时创建 `LogContext`，退出时销毁。所有模块通过 `LogContext::current()` 自动获取 session_id/turn_id。

#### 步骤 3：日志宏

提供便捷日志宏，自动填充模块名和上下文：

```cpp
// include/digital_human/log_macros.h
#pragma once
#include "digital_human/logger.h"
#include "digital_human/log_context.h"

namespace digital_human::detail {

class DH_API LogEmitter {
public:
    LogEmitter(LogLevel level, const char* module);
    ~LogEmitter();
    LogEmitter& operator<<(const auto& value) {
        buffer_ += std::to_string(value);  // 简化，实际需 format
        return *this;
    }
private:
    LogLevel    level_;
    const char* module_;
    std::string buffer_;
};

}  // namespace digital_human::detail

#define DH_LOG(level, module) \
    ::digital_human::detail::LogEmitter(level, module)

#define DH_LOG_DEBUG(module) DH_LOG(::digital_human::LogLevel::Debug, module)
#define DH_LOG_INFO(module)  DH_LOG(::digital_human::LogLevel::Info,  module)
#define DH_LOG_WARN(module)  DH_LOG(::digital_human::LogLevel::Warn,  module)
#define DH_LOG_ERROR(module) DH_LOG(::digital_human::LogLevel::Error, module)
```

#### 步骤 4：替换 154 处直出

按模块分批替换，每批独立提交：

| 批次 | 模块 | 文件数 | 命中数 | 优先级 |
|------|------|--------|--------|--------|
| 1 | core/pipeline | 1 | 14 | 高 |
| 2 | model/* | 3 | 70 | 高 |
| 3 | core/audio_sync_scheduler | 1 | 24 | 高 |
| 4 | core/其余 | 8 | 22 | 中 |
| 5 | audio/* | 2 | 12 | 中 |
| 6 | dialog/* / media/* / network/* | 3 | 12 | 中 |

替换规则：

```
std::cerr << "[Pipeline] Init: duplicate initialization" << std::endl;
→ DH_LOG_ERROR("pipeline") << "Init: duplicate initialization";

std::cout << "[Pipeline] 初始化成功: " << model_path << std::endl;
→ DH_LOG_INFO("pipeline") << "初始化成功: " << model_path;
```

#### 步骤 5：turn 级指标体系

新建 `include/digital_human/metrics.h`：

```cpp
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include "digital_human/export.h"

namespace digital_human {

/// 分位数统计器（环形缓冲 + 排序计算 p50/p95/p99）
class DH_API PercentileTracker {
public:
    explicit PercentileTracker(size_t capacity = 1024);
    void record(double value_ms);
    double p50() const;
    double p95() const;
    double p99() const;
    double avg() const;
    uint64_t count() const;
    void reset();
private:
    // 实现略：环形缓冲 + 延迟排序
};

/// 会话级指标
struct DH_API SessionMetrics {
    uint64_t total_turns = 0;
    uint64_t successful_turns = 0;
    uint64_t failed_turns = 0;
    uint64_t interrupted_turns = 0;
    uint64_t total_sessions = 0;
    uint64_t active_sessions = 0;
    size_t   history_length = 0;        // 当前历史轮次
};

/// LLM 指标
struct DH_API LLMMetrics {
    uint64_t total_requests = 0;
    uint64_t failed_requests = 0;
    uint64_t cancelled_requests = 0;
    PercentileTracker first_token_latency;     // 首 token 延迟 ms
    PercentileTracker total_latency;           // 总耗时 ms
    PercentileTracker cancel_latency;          // 取消耗时 ms
    double    tokens_per_second = 0;           // 最近一次的 tokens/s
    int       last_http_status = 0;
};

/// TTS 指标
struct DH_API TTSMetrics {
    uint64_t total_requests = 0;
    uint64_t failed_requests = 0;
    PercentileTracker first_byte_latency;      // 首字节延迟 ms
    PercentileTracker first_pcm_latency;       // 首 PCM 延迟 ms
    double    realtime_factor = 0;             // 合成时长/音频时长
    uint64_t  total_samples = 0;
};

/// Avatar 指标
struct DH_API AvatarMetrics {
    uint64_t total_uploads = 0;
    uint64_t rejected_uploads = 0;
    uint64_t total_updates = 0;
    PercentileTracker decode_latency;
    int original_width = 0;
    int original_height = 0;
    int canvas_width = 0;
    int canvas_height = 0;
    std::string last_reject_reason;
};

/// Publisher 指标
struct DH_API PublisherMetrics {
    uint64_t packets_written = 0;
    uint64_t video_frames_encoded = 0;
    uint64_t video_frames_dropped = 0;
    uint64_t audio_frames_encoded = 0;
    uint64_t reconnect_count = 0;
    PercentileTracker encode_latency;
    std::string last_drop_reason;
};

/// 统一指标快照（聚合所有层）
struct DH_API MetricsSnapshot {
    int64_t timestamp_ms = 0;
    SessionMetrics  session;
    LLMMetrics      llm;
    TTSMetrics      tts;
    AvatarMetrics   avatar;
    PublisherMetrics publisher;
    // Pipeline 层指标复用现有 SDKMetrics（通过组合而非继承）
};

/// 指标注册中心：全局单例，各模块注册自己的指标采集器
class DH_API MetricsRegistry {
public:
    static MetricsRegistry& instance();

    /// 获取当前快照（线程安全）
    MetricsSnapshot snapshot() const;

    /// 重置所有计数器（用于测试）
    void reset();

    // 各模块注册自己的指标更新接口（内部使用）
    void record_llm_first_token(double latency_ms);
    void record_tts_first_pcm(double latency_ms);
    void record_avatar_decode(double latency_ms);
    // ... 其他 record 方法
};

}  // namespace digital_human
```

#### 步骤 6：指标注入与采集

各模块通过 `MetricsRegistry` 上报指标：

```cpp
// conversation_session.cpp — turn 结束时
void ConversationSession::Impl::MaybeCompleteTurn(...) {
    // ... 原有逻辑 ...
    auto& metrics = MetricsRegistry::instance();
    metrics.record_turn_complete(success, was_cancelled);
}

// http_text_generation_client.cpp — 首 token 到达时
auto on_delta = [&](const std::string& delta) {
    if (!first_token_recorded) {
        first_token_recorded = true;
        MetricsRegistry::instance().record_llm_first_token(
            elapsed_since(start_ms));
    }
    // ...
};
```

#### 步骤 7：指标导出

提供 Prometheus 文本格式导出：

```cpp
/// 将 MetricsSnapshot 转为 Prometheus 文本格式
DH_API std::string to_prometheus(const MetricsSnapshot& snapshot);
```

调用方可通过 HTTP endpoint 暴露：

```
# TYPE dh_session_total_turns counter
dh_session_total_turns 42
# TYPE dh_llm_first_token_ms histogram
dh_llm_first_token_ms{quantile="0.5"} 120
dh_llm_first_token_ms{quantile="0.95"} 350
dh_llm_first_token_ms{quantile="0.99"} 800
```

### 验证标准

- [ ] `grep -rn "std::cout\|std::cerr\|fprintf" src/` 返回 0 结果
- [ ] 注入自定义 logger 后，所有日志经回调输出，无直出
- [ ] 日志记录包含 `session_id`、`turn_id`、`module`、`level` 四个字段
- [ ] `MetricsSnapshot` 包含路线图 §3.9 表格中全部指标
- [ ] `PercentileTracker` 在 1000 样本下 p50/p95/p99 误差 < 5%
- [ ] `to_prometheus()` 输出可被 Prometheus parser 正确解析
- [ ] 30 分钟多轮会话中，指标计数器无丢失、无重复

---

## P1-3.10 多轮对话产品级能力

### 现状

- **History**：`std::vector<ChatMessage>` 无限增长，无轮次/字符/token 上限
- **ASR**：完全不存在
- **VAD**：仅离线批处理（`VoiceActivityDetector::filter`），无流式事件回调
- **Barge-in**：`ConversationSession::Interrupt()` 只取消 LLM/TTS，无音频清理和时间轴重对齐
- **网关**：无 HTTP/multipart 会话网关，SDK 直接暴露给调用方

### 目标

1. History 预算：轮次上限 + 字符/token 估算 + 滑动窗口
2. ASR 抽象接口：partial/final transcript 事件
3. 流式 VAD：voice start/end 事件回调
4. Barge-in 链路：VAD → 停止 TTS → 清理 PCM → 重对齐时间轴
5. Web Gateway 架构设计（本阶段只做设计，不实现）

### 实施步骤

#### 步骤 1：History 预算

扩展 `ConversationConfig`：

```cpp
// include/dialog/conversation_session.h
struct ConversationConfig {
    // ... 原有字段 ...

    /// History 预算（P1-3.10）
    int max_history_turns = 20;           // 保留最近 N 轮（1 轮 = 1 user + 1 assistant）
    int max_history_chars = 8192;         // 历史总字符上限
    int max_history_tokens_estimate = 0;  // token 估算上限（0=不限制）
    bool enable_history_summary = false;  // 是否启用摘要策略（未来扩展）
};
```

在 `ConversationSession::Impl` 中实现截断：

```cpp
// conversation_session.cpp — 成功 turn 后追加 history 后调用
void TrimHistory() {
    // 1. 按轮次截断
    const int max_messages = config.max_history_turns * 2;
    if (static_cast<int>(history.size()) > max_messages) {
        history.erase(history.begin(),
                      history.begin() + (history.size() - max_messages));
    }
    // 2. 按字符截断（从最旧消息开始删除）
    size_t total_chars = 0;
    for (const auto& msg : history) total_chars += msg.content.size();
    while (total_chars > static_cast<size_t>(config.max_history_chars)
           && history.size() > 2) {
        total_chars -= history.front().content.size();
        history.erase(history.begin());
    }
    // 3. token 估算（粗略：chars / 3.5 for Chinese, chars / 4 for English）
    if (config.max_history_tokens_estimate > 0) {
        const int estimated_tokens = static_cast<int>(total_chars / 3.5);
        while (estimated_tokens > config.max_history_tokens_estimate
               && history.size() > 2) {
            history.erase(history.begin());
        }
    }
}
```

将 `history.size()` 上报为 `SessionMetrics.history_length`。

#### 步骤 2：ASR 抽象接口

新建 `include/dialog/asr_client.h`：

```cpp
#pragma once
#include <functional>
#include <memory>
#include <string>
#include "digital_human/export.h"

namespace digital_human {
namespace dialog {

/// ASR 转录结果
struct Transcript {
    std::string text;
    bool is_final = false;       // true=最终结果, false=partial
    double confidence = 1.0;
    int64_t start_ms = 0;        // 语音起始时间
    int64_t end_ms = 0;          // 语音结束时间
};

using TranscriptCallback = std::function<void(const Transcript&)>;
using CancelCheck = std::function<bool()>;

/// 抽象 ASR 接口
/// 实现 1：本地流式 ASR（如 sherpa-onnx、vosk）
/// 实现 2：HTTP 流式 ASR（如 Whisper API streaming）
class DH_API IASRClient {
public:
    virtual ~IASRClient() = default;

    /// 开始识别会话
    /// on_transcript: partial/final 结果回调
    /// cancelled: 调用方取消检查
    virtual bool Start(const TranscriptCallback& on_transcript,
                       const CancelCheck& cancelled,
                       std::string& error) = 0;

    /// 推送音频帧（16kHz mono float32）
    /// 在 Start 之后、Stop 之前调用
    virtual bool PushAudio(const float* samples,
                           size_t sample_count,
                           std::string& error) = 0;

    /// 停止识别，等待最终结果
    virtual bool Stop(std::string& error) = 0;
};

}  // namespace dialog
}  // namespace digital_human
```

#### 步骤 3：流式 VAD 接口

新建 `include/audio/streaming_vad.h`：

```cpp
#pragma once
#include <functional>
#include <memory>
#include "digital_human/export.h"

namespace digital_human {
namespace audio {

/// 流式 VAD 事件
enum class VADEvent {
    VoiceStart,    // 检测到语音开始
    VoiceEnd,      // 检测到语音结束
    Silence,       // 静音持续（用于超时判断）
};

using VADCallback = std::function<void(VADEvent)>;

/// 抽象流式 VAD 接口
/// 实现 1：基于能量+过零率的轻量 VAD（扩展现有 audio_vad）
/// 实现 2：基于 WebRTC VAD 的 Silero VAD
class DH_API IStreamingVAD {
public:
    virtual ~IStreamingVAD() = default;

    /// 开始检测
    /// on_event: 语音事件回调
    virtual bool Start(const VADCallback& on_event) = 0;

    /// 推送音频帧（16kHz mono float32）
    virtual bool PushAudio(const float* samples, size_t sample_count) = 0;

    /// 停止检测
    virtual void Stop() = 0;
};

/// VAD 配置
struct DH_API StreamingVADConfig {
    int sample_rate = 16000;
    int frame_ms = 30;              // 帧长
    float energy_threshold = 0.02F; // 能量阈值
    int hangover_frames = 8;        // 语音结束后的 hangover 帧数
    int min_voice_frames = 3;       // 判定 VoiceStart 的最小连续语音帧
    int silence_timeout_ms = 0;     // 静音超时（0=不触发）
};

/// 创建默认流式 VAD（能量+过零率）
DH_API std::unique_ptr<IStreamingVAD> create_streaming_vad(
    const StreamingVADConfig& config);

}  // namespace audio
}  // namespace digital_human
```

#### 步骤 4：Barge-in 链路

扩展 `ConversationSession` 支持 ASR + VAD 驱动的 barge-in：

```cpp
// include/dialog/conversation_session.h
struct ConversationConfig {
    // ... 原有字段 ...

    /// Barge-in 配置（P1-3.10）
    bool enable_barge_in = false;         // 是否启用音频打断
    int  barge_in_vad_hangover_ms = 240;  // VAD hangover
    int  barge_in_min_voice_ms = 90;      // 最短语音持续时间触发打断
    int  barge_in_cleanup_timeout_ms = 500; // 清理超时
};

class ConversationSession {
public:
    // ... 原有方法 ...

    /// 注入 ASR 客户端（可选，用于语音输入）
    void SetASRClient(IASRClient* client);

    /// 注入流式 VAD（可选，用于 barge-in）
    void SetStreamingVAD(audio::IStreamingVAD* vad);

    /// 推送用户音频（供 ASR + VAD 消费）
    /// samples: 16kHz mono float32
    bool PushUserAudio(const float* samples, size_t sample_count,
                       std::string& error);
};
```

Barge-in 处理流程：

```
用户语音 → StreamingVAD → VoiceStart 事件
                               │
                               ▼
                    ConversationSession::OnBargeIn()
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              Cancel TTS  Clear PCM   Realign
              (cancel_    (audio_     (reset
               current)   chunks      turn_audio
                          .clear())   _start_ms)
                               │
                               ▼
                    通知 SDK 刷新视频时间轴
```

```cpp
// conversation_session.cpp
void ConversationSession::Impl::OnBargeIn() {
    std::unique_lock<std::mutex> lock(mutex);
    if (state != SessionState::PLAYING
        && state != SessionState::SYNTHESIZING) {
        return;  // 非活动状态不触发 barge-in
    }
    // 1. 取消当前 LLM/TTS（复用 Interrupt 逻辑）
    cancel_current = true;
    sentence_jobs.clear();
    audio_chunks.clear();
    tts_done = !tts_active;
    // 2. 重对齐音频时间轴
    audio_submitted_until_ms = 0;
    turn_audio_start_ms = 0;
    audio_sample_cursor = 0;
    // 3. 通知 Bridge 清理未编码 PCM
    SetState(SessionState::INTERRUPTING);
    MaybeCompleteTurn(lock);
    // 4. 上报指标
    MetricsRegistry::instance().record_barge_in();
}
```

#### 步骤 5：Web Gateway 架构设计

本阶段只做设计文档，不实现。Gateway 独立于 SDK，负责：

```
浏览器 ──WebSocket──→ Gateway
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    认证/配额      会话租约      目标白名单
         │             │             │
         └─────────────┼─────────────┘
                       ▼
              ConversationSession
              (SDK 媒体引擎)
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
        LLM Adapter  TTS Adapter  StreamPublisher
```

Gateway 职责边界：

| 职责 | Gateway | SDK |
|------|---------|-----|
| 用户认证 (API key/OAuth) | 是 | 否 |
| 租户配额/限频 | 是 | 否 |
| multipart 头像上传 | 是 | 否（SDK 接收已解码字节） |
| 会话租约/超时 | 是 | 否 |
| LLM/TTS/推流目标白名单 | 是 | 否 |
| ASR/VAD/barge-in | 转发 | 是 |
| PCM 流式处理 | 否 | 是 |
| H.264/AAC 编码 | 否 | 是 |
| 推流发布 | 否 | 是 |
| 指标导出 | 聚合 | 提供 snapshot API |

### 验证标准

- [ ] `history.size()` 不超过 `max_history_turns * 2`
- [ ] `history` 总字符数不超过 `max_history_chars`
- [ ] 长会话（100 轮）下 LLM 请求体大小稳定不增长
- [ ] `IASRClient` 接口可通过 mock 实现验证 partial/final 事件
- [ ] `IStreamingVAD` 在静音→语音→静音序列下正确触发 VoiceStart/VoiceEnd
- [ ] Barge-in 后，`audio_chunks` 被清空，`turn_audio_start_ms` 被重置
- [ ] Barge-in 后，旧 turn 的视频帧不再被提交
- [ ] Barge-in 触发到 turn 终态完成 < `barge_in_cleanup_timeout_ms`
- [ ] Gateway 架构文档明确职责边界，SDK 不包含 HTTP 服务器/认证逻辑

---

## 实施顺序与依赖关系

```
P1-3.7 模块化独立 target
    │
    ├──→ P1-3.8 SDK 安装/ABI/版本
    │        │
    │        └──→ P1-3.9 可观测性（依赖 DH_API 宏）
    │                 │
    │                 └──→ P1-3.10 多轮对话（依赖指标体系）
    │
    └──→ P1-3.10 History 预算（无前置依赖，可提前）
```

| 阶段 | 任务 | 前置依赖 | 预计提交 |
|------|------|----------|----------|
| 3.7a | OBJECT → STATIC 库升级 | 无 | `refactor(cmake): static modules` |
| 3.7b | 符号可见性控制 + install 规则 | 3.7a | `feat(cmake): symbol visibility + install` |
| 3.8a | export.h + DH_API 宏 | 3.7b | `feat(api): windows export macro` |
| 3.8b | VERSION/SOVERSION + package export | 3.8a | `feat(cmake): package export + version` |
| 3.8c | 统一 Error 结构 | 3.8a | `feat(error): unified error structure` |
| 3.8d | 版本化 C API | 3.8b, 3.8c | `feat(api): versioned C API` |
| 3.9a | Logger 接口 + 上下文传播 | 3.8a | `feat(log): logger interface` |
| 3.9b | 替换 154 处直出（分 6 批） | 3.9a | `refactor(log): replace stdout/stderr` |
| 3.9c | turn 级指标 + 分位数 | 3.9a | `feat(metrics): turn-level metrics` |
| 3.9d | Prometheus 导出 | 3.9c | `feat(metrics): prometheus export` |
| 3.10a | History 预算 | 无 | `feat(dialog): history budget` |
| 3.10b | ASR 接口 + 流式 VAD | 3.8a | `feat(dialog): ASR + streaming VAD` |
| 3.10c | Barge-in 链路 | 3.10a, 3.10b, 3.9c | `feat(dialog): barge-in chain` |
| 3.10d | Gateway 架构设计文档 | 全部 | `docs: gateway architecture` |

### 关键约束

1. **3.8a（export.h）是 3.9 和 3.10 的共同前置**：logger 和 ASR 接口的公共类都需要 `DH_API` 宏
2. **3.9a（logger 接口）是 3.9b（替换直出）的前置**：必须先有接口才能替换
3. **3.10c（barge-in）依赖 3.9c（指标）**：barge-in 需要上报指标
4. **3.10a（history 预算）无前置依赖**：可最早启动，建议与 3.7 并行
5. **3.8d（C API）建议最后做**：需要在 C++ API 稳定后封装
