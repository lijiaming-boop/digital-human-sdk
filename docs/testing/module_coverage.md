# 项目模块覆盖测试

## 1. 目标与边界

本轮测试补充以“可离线、可重复、可纳入 CI”为原则，覆盖核心模块的正常路径、参数边界、失败路径、生命周期和可选依赖裁剪。真实模型精度、真实 HTTP 服务、RTMP/RTSP 服务端兼容性、硬件编码器和声卡行为仍属于集成或验收测试，不应伪装成离线单元测试。

新增测试统一位于 `tests/`，即使 `BUILD_EXAMPLES=OFF` 也会在 `BUILD_TESTING=ON` 时构建。HTTP 和媒体测试分别跟随 `dh_network`、`dh_media` 目标条件注册，最小依赖构建不会产生悬空测试。

## 2. 模块覆盖矩阵

| 模块 | 新增覆盖 | 既有回归补充 | CTest 标签 |
| --- | --- | --- | --- |
| core / concurrency | 有界队列容量、溢出指标、批量操作、阻塞唤醒、停止排空；线程启动/超时/异常；WorkerRegistry 顺序回收 | `ring_buffer_test`、`lifecycle_safety_test` | `unit;coverage;core` |
| observability | Logger 上下文与级别过滤、Metrics 快照/Prometheus、百分位统计、QualityGate 通过与违规 | `p2_resilience_test` | `unit;coverage;observability` |
| media clock / sync | 时钟配置、校正、抖动、回退与重置 | `frame_scheduler_test`、`av_sync_test`、`audio_sync_test` | `unit;coverage;core` |
| audio | StreamingVAD 配置校验、分帧、去抖、hangover、停止、重启隔离 | `preemphasis_test`、`rmsnorm_test`、`noise_reduction_test`、`vad_test`、`audio_framer_test`、`cmvn_test`、`mel_feature_test`、`audio_processor_test` | `unit;coverage;audio` |
| public API | C/C++ 空指针、结构体版本、状态查询、指标、初始化、重复初始化、缺模型启动、终止态 | `sdk_model_error_test`、`lifecycle_safety_test` | `unit;coverage;api` |
| avatar / image | PNG 正常解码、格式和 MIME 校验、字节/尺寸/像素限制、损坏输入、批量失败隔离、move 语义 | `avatar_upload_test`、`image_load_test` | `unit;coverage;avatar` |
| dialog | 增量分句、强弱边界、UTF-8 计数、flush/reset | `dialog_module_test` | `unit;coverage;dialog` |
| model | 未加载状态、缺失模型路径、回调契约、等待和 move 语义 | `ncnn_test` 及 model 标签测试 | `unit;coverage;model` |
| network / HTTP | JSON 转义/解析、Unicode 代理对、非法值；空 URL/回调；文本生成和 TTS 参数失败路径 | `http_service_client_test`、`llama_cpp_client_test` | `unit;coverage;network`，条件启用 |
| media publisher | 发布配置全边界、关闭幂等、未打开输入、指标初值、桥接器启动/输入边界 | `stream_publisher_test`、`publisher_resilience_test` | `unit;coverage;media`，条件启用 |
| pipeline / render / inference | 本轮不复制已有场景，继续由合成输入、文件链路和真实模型分层覆盖 | `pipeline_test`、`render_thread_test`、`inference_render_pipeline_test`、model 标签测试 | `unit`、`integration`、`model` |

## 3. 执行方式

完整依赖的日常离线门禁：

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DBUILD_EXAMPLES=ON
cmake --build build -j
ctest --test-dir build --output-on-failure -L unit
```

只运行本轮覆盖集合：

```bash
ctest --test-dir build --output-on-failure -L coverage
```

最小依赖裁剪验证：

```bash
cmake -S . -B build-minimal \
  -DBUILD_TESTING=ON -DBUILD_EXAMPLES=OFF \
  -DDIGITAL_HUMAN_ENABLE_HTTP=OFF \
  -DDIGITAL_HUMAN_ENABLE_MEDIA=OFF \
  -DDIGITAL_HUMAN_ENABLE_AUDIO_IO=OFF \
  -DDIGITAL_HUMAN_ENABLE_AUDIO_LOADER=OFF
cmake --build build-minimal -j
ctest --test-dir build-minimal --output-on-failure -L coverage
```

## 4. 代码覆盖率报告

GCC/Clang 构建可通过 `DIGITAL_HUMAN_ENABLE_COVERAGE` 启用 gcov 兼容插桩。安装 `gcovr` 后，`coverage_report` 会先运行所有 `coverage` 标签测试，再生成 HTML 和 Cobertura XML：

```bash
cmake -S . -B build-coverage \
  -DBUILD_TESTING=ON -DBUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DDIGITAL_HUMAN_ENABLE_COVERAGE=ON
cmake --build build-coverage --target coverage_report -j
```

产物：

- `build-coverage/coverage/index.html`
- `build-coverage/coverage/coverage.xml`

覆盖率只统计 `src/`，排除 `tests/`。建议 CI 初期记录基线而不立即设置过高阈值；稳定后按模块逐步设置 line/branch 门禁，避免为了数字制造无意义断言。

## 5. 本轮验证结果

验证日期：2026-08-12。

- 完整依赖：`ctest -L coverage`，6/6 通过。
- 完整依赖单元回归：`ctest -L unit`，29/29 通过。
- 最小依赖且关闭 examples：`ctest -L coverage`，4/4 通过。
- VAD 重启隔离测试发现 `Start()` 未清理 residual buffer；实现已在启动时清零，避免上一次会话的尾部采样污染下一次会话。

## 6. 后续环境测试

以下场景需要专用 fixture 或服务，保留在 integration/model/network/nightly 层：

1. 真实 ncnn 模型加载、推理数值与 GPU/CPU 一致性。
2. HTTP/SSE 分片响应、超限中止、取消和慢速服务端。
3. RTMP/RTSP 断线重连、音频保留策略、重连窗口和关闭竞争。
4. FFmpeg 硬件编码器选择、回退及不同版本兼容性。
5. PortAudio 真声卡播放、设备切换与拔插。
6. 端到端实时头像链路的音画质量和长稳运行。
