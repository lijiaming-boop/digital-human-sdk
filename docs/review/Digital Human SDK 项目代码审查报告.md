# Digital Human SDK 项目代码审查报告

## 一、严重缺陷（Bug 级，应优先修复）

### 1.1 音视频同步时钟计算错误（功能性 Bug）

**(1) Pipeline 音频时钟二次方增长** — [src/core/pipeline.cpp:742-749](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/pipeline.cpp#L742-L749)

C++



```cpp
int64_t estimated_samples = elapsed_ms / 1000.0 * ctx.config.audio_sample_rate;
ctx.av_sync.UpdateAudioClock(estimated_samples);  // 每次传入累计值，但 AVSync 内部累加
```

`AVSync::UpdateAudioClock` 内部是累加（`audio_clock_ms += delta_ms`），而这里传入的是"自启动以来的累计采样数"，导致时钟按 O(N²) 增长，几秒钟后所有帧都会被判为 `VIDEO_BEHIND` 而被丢弃。

**(2) AudioSyncScheduler 立体声时钟翻倍** — [src/audio/audio_sync_scheduler.cpp:68](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_sync_scheduler.cpp#L68)

C++



```cpp
int64_t samplesConsumed = deltaFrames * config.audio_channels;  // 多乘了声道数
av_sync.UpdateAudioClock(samplesConsumed);
```

`AudioPlayer::GetConsumedFrames()` 返回的是帧数（每声道一份），但 AVSync 内部用 `samples / sample_rate` 计算（`sample_rate=48000`，未乘声道），导致立体声下时钟以 2× 速度推进，drift 永远为负，同步逻辑持续丢帧。

### 1.2 未定义行为 / 数据竞争

| #    | 位置                                                         | 问题                                                         |
| :--- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| B1   | [src/audio/audio_player.cpp:104,116-122](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_player.cpp) | PortAudio 回调读取 `read_frame_pos`/`total_frames`/`audio_data`（非原子），主线程 `LoadAudio`/`Stop` 同时写入 → UB |
| B2   | [src/audio/audio_player.cpp:442-444](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_player.cpp#L442-L444) | `total_paused_duration.store(total_paused_duration.load() + x)` 非原子 RMW，应使用 `fetch_add`，会丢失更新 |
| B3   | [src/core/audio_processor.cpp:68](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/audio_processor.cpp#L68) | `bool eos_marked_` 由 `MarkEOS()` 写、`Run()` 读，无同步 → 数据竞争 |
| B4   | [src/core/inference_worker.cpp:85](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/inference_worker.cpp#L85) 与 [render_thread.cpp:68](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/render_thread.cpp#L68) | `bool input_eos_` 同样存在竞争，且实际从未被 `Run()` 读取（死代码） |
| B5   | [src/model/model_loader.cpp:138](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/model_loader.cpp#L138) | `loading_thread = std::thread(...)` 未先 join 旧线程，若上一次未 join 会触发 `std::terminate` |
| B6   | [src/model/model_loader.cpp:110](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/model_loader.cpp#L110) | `LoadCallback` 在工作线程执行无 try/catch，回调抛异常 → `std::terminate` |
| B7   | [include/core/face_aligner.h:36-43](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/face_aligner.h#L36-L43) | `FaceAlignerResult::valid` 未初始化，默认构造后读取为 UB     |

### 1.3 数值除零 / 精度问题

| 位置                                                         | 问题                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [src/audio/audio_framer.cpp:14-16](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_framer.cpp#L14-L16) | `denom = frameSize - 1`，调用方传 `frameSize==1` 时除零      |
| [src/core/face_aligner.cpp:57](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/face_aligner.cpp#L57) | `scale = desired_dist / current_dist`，两眼重合时 `current_dist==0` → Inf |
| [src/audio/audio_cmvn.cpp:32](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_cmvn.cpp#L32) | `1.0f / max(std, kEps)` 在 std 极小（静音段）时放大 ~80dB 噪声 |
| [src/audio/audio_mel_feature_extract.cpp:43,45](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_mel_feature_extract.cpp#L43-L45) | Mel 滤波器分母 `+1e-10f` 在 `center==left` 时权重爆炸        |
| [src/model/model_inferencer.cpp:392](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/model_inferencer.cpp#L392) | `std::thread::hardware_concurrency()` 可能返回 0，`std::clamp(n,1,0)` 是 UB |

### 1.4 渲染/推理流水线数据丢失

- [src/core/pipeline.cpp:733-739](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/pipeline.cpp#L733-L739) `FrameAction::WAIT` 注释掉了 `Push`，实际丢弃了帧，违反 WAIT 语义
- [src/core/pipeline.cpp:566-610](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/pipeline.cpp#L566-L610) `MatchFacePacket` 当视频超前时直接丢弃 face 包且无回退队列，PTS 错位时系统性丢帧
- [src/core/pipeline.cpp:639-664](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/pipeline.cpp#L639-L664) `OutputThread` 缺少 `M_inv`/`face_mask`/`original_face`，无法做逆变换与融合，注释承认"完整实现需要 InferenceOutputPacket 携带 ProcessedFaceData"
- [src/core/pipeline.cpp:354](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/pipeline.cpp#L354) `mel_pkt.header.cost_ms = mel_pkt.header.cost_ms;` 自赋值，cost 实际从未记录
- [src/core/pipeline.cpp:969-972](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/pipeline.cpp#L969-L972) `GetDriftMs()` 直接 `return 0.0;` 是桩函数

### 1.5 OpenCV 资源未初始化

[src/model/output_processor.cpp:135](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/output_processor.cpp#L135) `cv::warpAffine(..., cv::BORDER_TRANSPARENT)` 在新分配的 `dst` 上使用，无源对应的像素保持未初始化，下游 `FaceFusion` 会读到垃圾像素。应使用 `BORDER_CONSTANT` 或预先清零。

[src/core/face_mask_generator.cpp:168-188](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/face_mask_generator.cpp#L168-L188) `to3ChannelMask` 仅检查 `CV_32FC1`，若输入 `CV_32FC3` 跳过分支后 `merge(3, alpha_f)` 会生成 9 通道 Mat，与文档声明的 `CV_32FC3` 不符。

------

## 二、架构与设计问题

### 2.1 双份分叉的实现（最严重的架构问题）

[src/core/pipeline.cpp:209-754](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/pipeline.cpp#L209-L754) 在 `Pipeline::Impl` 内部重新定义了 `AudioProcessorThread`/`InferenceThread`/`RenderThread` 等嵌套类，**完全没用** [audio_processor.cpp](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/audio_processor.cpp)/[inference_worker.cpp](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/inference_worker.cpp)/[render_thread.cpp](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/render_thread.cpp) 中的独立类。两套实现严重分叉：

- 独立 `InferenceWorker` 有重试/EWMA/积压检测；pipeline 内部版本完全没有，推理输出为空时直接 `continue` 丢帧
- 独立 `RenderThread::Run()` 从不调用 `frame_scheduler_`（初始化了但不用），`actual_fps` 永远为 0；WAIT 分支是死代码
- 嵌套 `RenderThread` 遮蔽了 `core::RenderThread`，`InferenceThread` 遮蔽了 `core::InferenceWorker`，命名冲突

建议：明确哪一套是规范的，删除/合并另一套。

### 2.2 `ThreadBase` 超时是"假"的

[include/core/thread_base.h:112-118](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/thread_base.h#L112-L118) `Wait(int timeout_ms)` 注释承认"当前无效（始终完全等待）"，但 [pipeline.h:142](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/pipeline.h#L142) 文档却声明"超时后强制终止"。一旦工作线程卡死（如 ncnn 推理无内部取消），`Pipeline::Stop()` 和析构函数会无限阻塞。需要补充强制终止路径。

### 2.3 `ThreadSafeQueue` 违反 Rule of Five

[include/core/thread_safe_queue.h:88-102](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/thread_safe_queue.h#L88-L102) 有移动构造，但**未声明移动赋值**，导致对象可移动构造却不可移动赋值（API 不对称）。且移动构造漏掉了 `peak_size_`/`total_overflows_`/`last_push_time_` 等统计字段，造成静默数据丢失。

### 2.4 `RingBuffer` SPSC 约束未文档化也未校验

[src/audio/audio_ring_buffer.cpp](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_ring_buffer.cpp) 实现是 SPSC 无锁环形缓冲，但头文件未声明此约束。多生产者并发调用 `write()` 会读取同一 `w`、互相覆盖数据。`reset()` 在生产/消费活跃时调用也是竞争。`capacity==0` 时 `w % capacity` 是除零 UB。

### 2.5 错误处理策略混乱

项目中三种策略并存，无统一规范：

- 抛异常：`AudioLoader::load`、`ImageLoader::loadImageFromFile`
- cerr + 返回 false/空：`FaceDetector`、`ModelInferencer::Init`、`AudioPlayer`
- 静默返回空：`AudioVad::filter`、`MelFeatureExtract::extract`、`PreEmphasis::process`

调用方无法程序化区分"文件不存在"、"解码失败"、"无音频流"。建议统一为 `Result<T, ErrorCode>` 或异常体系。

------

## 三、Pimpl 模式与命名一致性

### 3.1 命名空间违规（违反项目约束）

项目约定 `digital_human::<module>`（snake_case），但以下三个模块违规：

| 文件                                                         | 当前命名空间            | 应为                  |
| :----------------------------------------------------------- | :---------------------- | :-------------------- |
| [include/core/face_detector.h:9](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/face_detector.h#L9) | `DigitalHuman::core`    | `digital_human::core` |
| [include/core/face_mask_generator.h:7](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/face_mask_generator.h#L7) | `DigitalHuman::Core`    | `digital_human::core` |
| [include/core/face_aligner.h:6](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/face_aligner.h#L6) | `digital_human::core` ✓ | —                     |

这导致 [pipeline.cpp:86-88](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/pipeline.cpp#L86-L88) 必须同时使用三种命名空间限定。

### 3.2 类名拼写错误

[include/core/face_aligner.h:11](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/face_aligner.h#L11) `class FaceAlignigner`（多了个 `g`），拼写错误传播到所有使用点（face_aligner.cpp、pipeline.cpp、Impl 结构体名）。

### 3.3 Impl 命名不一致

| 文件                                                         | Impl 结构体名        | 成员名              |
| :----------------------------------------------------------- | :------------------- | :------------------ |
| 多数类（20+）                                                | `struct Impl`        | `impl_`             |
| [image_loader.h:26](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/image_loader.h#L26) | `ImageLoaderImpl`    | `impl_`             |
| [face_detector.h:27](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/face_detector.h#L27) | `FaceDetectorImpl`   | `impl_`             |
| [face_aligner.h:32](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/face_aligner.h#L32) | `FaceAlignignerImpl` | `impl_`             |
| [face_mask_generator.h:48](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/face_mask_generator.h#L48) | `Impl`               | `pImpl`（唯一异常） |

### 3.4 移动操作策略不一致

[pipeline.h:105-108](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/pipeline.h#L105-L108)、[inference_worker.h:79-82](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/inference_worker.h#L79-L82)、[audio_processor.h:72-75](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/audio_processor.h#L72-L75)、[render_thread.h:88-91](file:///C:/Users/27013/Desktop/digital-human-sdk/include/core/render_thread.h#L88-L91) 这 4 个继承 `ThreadBase` 的类 **删除**了移动操作，违反项目约定"声明移动操作"。技术上因 `ThreadBase` 持有 `std::thread` 不能移动是合理的，但应更新项目约定，或改为组合而非继承。

------

## 四、构建系统问题

### 4.1 根 [CMakeLists.txt](file:///C:/Users/27013/Desktop/digital-human-sdk/CMakeLists.txt) 反模式

| 行     | 问题                                                         |
| :----- | :----------------------------------------------------------- |
| 10     | `add_compile_options(-O3)` 全局应用，**Debug 构建也被 -O3**  |
| 13     | `add_compile_options(-march=native)` 编入构建主机 CPU 指令集，**二进制不可移植** |
| 14     | `add_definitions(-DNDEBUG)` 强制 Debug 也定义 NDEBUG，**摧毁 debug 构建** |
| 17     | `-Wall -Wextra` 是 GCC/Clang 专用，MSVC 不识别               |
| 21-23  | `add_compile_options(${OpenMP_CXX_FLAGS})` 旧式，应链接 `OpenMP::OpenMP_CXX` imported target |
| 31, 82 | 全局 `include_directories(${CMAKE_SOURCE_DIR}/include)` 出现两次（重复） |
| 34     | `add_definitions(-DASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")` **把构建主机绝对路径编入二进制**，不可重定位 |
| 35     | `add_definitions(-DPROJECT_SOURCE_DIR=...)` **与 CMake 内置变量 `PROJECT_SOURCE_DIR` 同名**，易混淆 |
| 39     | `link_directories(BEFORE "$ENV{HOME}/.local/lib")` 硬编码家目录，Linux 限定，极脆弱 |
| 69-72  | `OpenCV_LIBS`、`dlib_INCLUDE_DIRS` 等用全局 `include_directories`，应改为 `target_link_libraries(... PUBLIC ...)` |

### 4.2 [src/CMakeLists.txt](file:///C:/Users/27013/Desktop/digital-human-sdk/src/CMakeLists.txt)

第 3 行 `file(GLOB_RECURSE SRC_FILES ...)` **未加 `CONFIGURE_DEPENDS`** — 新增/删除 .cpp 文件不会触发重新 configure，是 CMake 官方明确警告的反模式。

### 4.3 [examples/CMakeLists.txt](file:///C:/Users/27013/Desktop/digital-human-sdk/examples/CMakeLists.txt)

- **36 个近乎完全相同的 `if(EXISTS...) add_executable ... set_target_properties` 块**，应改为 `foreach` 循环或 helper 函数。385 行可压缩到 ~10 行。
- 前 5 个 target 漏写 `set_target_properties`，后面都有 — 不一致。
- `set_target_properties(... RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")` **冗余**，根 CMakeLists 已全局设置 `CMAKE_RUNTIME_OUTPUT_DIRECTORY`。
- **没有 `enable_testing()`/`add_test()`**，30+ 测试可执行文件未注册到 CTest，`ctest` 无法发现/运行。

### 4.4 [build.sh](file:///C:/Users/27013/Desktop/digital-human-sdk/build.sh) 与 [CMakePresets.json](file:///C:/Users/27013/Desktop/digital-human-sdk/CMakePresets.json)

- `build.sh:12-13` 硬编码 `x86_64-linux-gnu` 路径，仅 Linux x86_64 可用
- `build.sh:19-24` 用 `sed -i` 改生成的 `build.make`，每次 re-configure 会被覆盖
- `CMakePresets.json:7` 硬编码 `$env{HOME}/vcpkg/...`，仅特定用户机器可用

------

## 五、测试基础设施问题

### 5.1 无测试框架

没有 GoogleTest/Catch2/doctest。所有测试是手写 `main()`。`TEST_CHECK(cond, desc)` 宏在 **12 个文件中复制粘贴**（audio_processor_test:38, pipeline_test:43, ...），未提取为共享头文件，违反项目 `code-style.md §11.2` 的规定。

### 5.2 断言风格分裂

- 12 个文件用宏 + `gPassed`/`gFailed` 计数
- ~15 个文件用 `std::cout << (cond ? "[PASS]" : "[FAIL]")` 内联三元，**无失败计数、无退出码**

### 5.3 退出码不可靠（破坏 CI）

约半数测试始终 `return 0`，shell/CI 无法检测失败：

| 始终返回 0（无法报告失败）                                   | 正确返回非零                                                 |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| audio_framer_test, audio_loader_test, cmvn_test, dlib_test, face_detector_test, face_mask_generator_test, ffmpeg_audio_test, image_load_test, mel_feature_test, ncnn_test, noise_reduction_test, opencv_test, preemphasis_test, ring_buffer_test, rmsnorm_test, vad_test, diagnose_output | audio_processor_test, audio_sync_test, av_sync_test, fit_test, frame_scheduler_test, full_pipeline_test, inference_render_pipeline_test, inference_worker_test, model_inferencer_test, output_processor_test, pipeline_test, render_thread_test |

### 5.4 无 CI/CD

无 `.github/`、`.gitlab-ci.yml`、`Jenkinsfile` 等。结合 5.3，**项目没有任何自动化测试门禁**。

------

## 六、版本控制卫生

### 6.1 `build_wsl/` 499 个未跟踪构建产物

`git ls-files -- build_wsl` 返回 0 个跟踪文件（好），但 `build_wsl/` 不在 `.gitignore` 中，`git status` 显示为 `??`，一次误 `git add .` 就会被提交。包含 `CMakeCache.txt`、`CMakeFiles/`（含 `a.out`、`.o`）、30+ 编译好的可执行文件。

### 6.2 [.gitignore](file:///C:/Users/27013/Desktop/digital-human-sdk/.gitignore) 太窄

当前仅忽略 `build/`、`out/`、`.vs/`、`.vscode/`、`*.jpg`、`*.png`、`*.dat`、`.DS_Store`。缺少：`build_wsl/`、`*.o`、`*.so`、`*.a`、`*.exe`、`CMakeCache.txt`、`CMakeFiles/`、`compile_commands.json`、`*.mp4`、`*.mp3`、`*.zip`。

### 6.3 其他未跟踪产物

- `assets/ffmpeg.zip`、`assets/zw.mp3`、`assets/output/digital_human.mp4` 未忽略
- `models/Wav2Lip-SD-GAN-opt.{bin,param}` 模型权重未忽略
- `assets/gen_test_audio.py`、`assets/test_16k_mono.{raw,wav}` 已删除但未提交（staged as `D`）

------

## 七、性能问题

| #    | 位置                                                         | 问题                                                         |
| :--- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| P1   | [audio_framer.cpp:47-57](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_framer.cpp#L47-L57) | `vector<vector<float>>` 每帧一次堆分配，下游 FFT 缓存不友好。应扁平化为 `vector<float>` + stride |
| P2   | [audio_noise_reduction.cpp:45-48](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_noise_reduction.cpp#L45-L48) | 同上，且逐帧 `cv::dft` 可批量用 `cv::DFT_ROWS`               |
| P3   | [audio_mel_feature_extract.cpp:109](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_mel_feature_extract.cpp#L109) | `melFilterbank_ * powerSpec.t()` 每次调用都分配转置副本，应用 `cv::gemm` + `GEMM_2_T` |
| P4   | [audio_cmvn.cpp:23-38](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_cmvn.cpp#L23-L38) | 列主序遍历行主序 Mat，每次内层访问 cache miss，应交换循环顺序 |
| P5   | [output_processor.cpp:189-204](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/output_processor.cpp#L189-L204) | 逐像素 C++ 循环做 alpha blend，应用 `cv::addWeighted`/`cv::multiply` 向量化，10-50× 提速 |
| P6   | [output_processor.cpp:401](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/output_processor.cpp#L401) | `result = fused_image.clone()` 即使不锐化/混色也总是克隆     |
| P7   | [model_inferencer.cpp:319-328](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/model_inferencer.cpp#L319-L328) | 每次 `Infer()` 都持 `stats_mutex` 更新延迟统计，热路径串行化，应改 atomic |
| P8   | [model_inferencer.cpp:300-302](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/model_inferencer.cpp#L300-L302) | 每次 `doInfer` 重写 `net.opt.num_threads`/`use_vulkan_compute`，ncnn 每次拷贝 opt 是浪费 |
| P9   | [face_detector.cpp:9](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/face_detector.cpp#L9) | `#include <opencv2/opencv.hpp>` 全模块头文件，编译期膨胀     |

------

## 八、其他代码异味

- **头文件膨胀**：5 个头文件用 `<opencv2/opencv.hpp>`（全模块），仅用 `cv::Mat` 应改 `<opencv2/core.hpp>`；4 个头文件不必要的 `<atomic>`；[audio_loader.h:3,7](file:///C:/Users/27013/Desktop/digital-human-sdk/include/audio/audio_loader.h#L3-L7) 重复 `#include <string>`
- **魔数遍地**：模型名 `"Wav2Lip-SD-GAN-opt"` 在 [model_loader.cpp:27](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/model_loader.cpp#L27) 和 [model_inferencer.cpp:351-352](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/model_inferencer.cpp#L351-L352) 两处定义；采样率 16000 在 audio_loader.cpp 两处硬编码；dlib 68 点关键点索引（36/42/48/67 等）散落各处
- **死代码**：[audio_framer.cpp:41](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_framer.cpp#L41) `bool needPad` 算了不用；[output_processor.cpp:237-238](file:///C:/Users/27013/Desktop/digital-human-sdk/src/model/output_processor.cpp#L237-L238) `if (ksize % 2 == 0)` 对字面量 5 永远为 false
- **冗余 nullptr**：[audio_sync_scheduler.cpp:104-116](file:///C:/Users/27013/Desktop/digital-human-sdk/src/core/audio_sync_scheduler.cpp#L104-L116) 与 [audio_player.cpp:181-193](file:///C:/Users/27013/Desktop/digital-human-sdk/src/audio/audio_player.cpp#L181-L193) 在 `unique_ptr` move 后显式 `other.impl_ = nullptr`（多余，move 后已为 null）
- **const 正确性缺失**：[output_processor.h](file:///C:/Users/27013/Desktop/digital-human-sdk/include/model/output_processor.h) 所有公有方法均非 const（Impl 实际无状态）；`FaceDetector::detect`/`getLandmarks`/`isModelLoaded`、`ImageLoader` 三个 load 方法、`FaceAligner::align`/`alignByRect` 同样
- **API 设计**：`ImageLoader` Impl 完全无状态，类无意义，应为自由函数或静态方法；`AudioPlayer::LoadAudio` 用 `int numSamples`，内部却用 `int64_t total_frames`，类型不一致
- **`using namespace std`**：[examples/demo.cpp:3](file:///C:/Users/27013/Desktop/digital-human-sdk/examples/demo.cpp#L3) 违反 `code-style.md §10.3/§13`，且 demo.cpp 实际是个空壳（仅 `cout<<"this is demo.cpp"<<endl;`）

------

## 九、修复优先级建议

**P0（影响功能正确性，应立即修复）**

1. 音视频时钟 Bug（§1.1 两处）— 当前立体声/全链路同步基本是坏的
2. `WAIT` 帧丢弃、`MatchFacePacket` 丢帧（§1.4）— 流水线会卡死
3. `AudioPlayer` 数据竞争（§1.2 B1/B2）— 实时回调 UB
4. `ModelLoader` 线程生命周期（B5/B6）— 可能 `std::terminate`
5. `output_processor.cpp:135` `BORDER_TRANSPARENT` 未初始化像素

**P1（架构层面，影响可维护性）** 6. 决定 pipeline.cpp 嵌套类 vs 独立 worker 类哪套是规范的，删除另一套（§2.1） 7. `ThreadBase::Wait` 实现真正的超时/强制终止（§2.2） 8. 统一错误处理策略（§2.5） 9. 修复命名空间/类名拼写/Impl 命名一致性（§3.1-3.3）

**P2（工程基础设施）** 10. 把 `build_wsl/` 与标准 CMake 产物加入 `.gitignore`（§6） 11. 测试用 CTest 注册，修复退出码，提取 `TEST_CHECK` 共享头（§5） 12. 例子 CMakeLists 用 `foreach` 去重（§4.3） 13. 根 CMakeLists 改用 target-scoped 现代 CMake（§4.1） 14. 加 CI 流水线（§5.4）

**P3（性能与质量提升）** 15. 数值除零/精度保护（§1.3） 16. 性能优化：扁平化帧缓冲、向量化 blend、避免转置、atomic 统计（§7） 17. 头文件瘦身（§8） 18. 集中魔数为命名常量（§8）