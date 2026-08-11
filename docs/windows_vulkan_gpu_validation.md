# Windows 原生 Vulkan 验证记录

验证日期：2026-08-07。目标为在 Windows 原生环境中运行项目的
`Wav2Lip-SD-GAN-opt` NCNN 模型，验证 Vulkan、GPU 选择、数值正确性和
CPU 可用性；它不替代完整音视频流水线验收。

## D 盘环境布局

开发产物和缓存均放在 `D:\digital-human-dev`：

- 构建：`D:\digital-human-dev\build`
- 模型/下载缓存：`D:\digital-human-dev\cache`
- NCNN 官方 Windows 共享库：
  `D:\digital-human-dev\ncnn\vs2022-shared\ncnn-20260526-windows-vs2022-shared`
- 临时目录：`D:\digital-human-dev\temp`

`windows-ucrt64-vulkan` CMake 预设把构建目录、临时目录和 NCNN 前缀均指向
D 盘；仓库源代码仍位于用户已有的 C 盘工作区，未移动或复制模型文件。

## 实际验证方法

`tools/windows_ncnn_gpu_validation.cpp` 是一个最小 Windows 原生验证程序。
它通过 NCNN C API 加载项目模型，并调用官方共享库的 Vulkan 初始化导出，使用
项目同样的双输入：`audio_sequences`（16×80×1）与
`face_sequences`（96×96×6）。每个模式先热身一次，再统计四次推理。

输出必须满足：`extract_result=0` 且形状为 `96×96×3`。
该工具只用于环境验收；主项目仍经 CMake 直接使用 NCNN C++ API，不依赖工具中
为最小化验证而使用的动态导出查询。

## 结果

| 项目 | 结果 |
| --- | --- |
| Vulkan 实例 | 创建成功，返回 0 |
| NCNN 可枚举设备 | 2 块 |
| 默认设备 | GPU 0：NVIDIA GeForce RTX 4050 Laptop GPU |
| 备选设备 | GPU 1：AMD Radeon(TM) 780M |
| CPU 推理 | 成功，输出 96×96×3，median 26.299 ms |
| RTX 4050 Vulkan FP16 | 成功，输出 96×96×3，median 2.572 ms |
| RTX 4050 加速比 | 约 10.2×（同一最小模型推理口径） |
| AMD 780M Vulkan FP16 | 成功，输出 96×96×3，median 11.246 ms |
| RTX 4050 重复启动 | 6/6 成功，median 2.552–2.815 ms |
| 正式 `model_inferencer_test` | 51/51 通过；30 次连续 GPU 推理平均 12.022 ms、最大 34.852 ms |

CPU 与 RTX 4050 的输出逐元素对比（27,648 个 FP32 输出值）：

- 最大绝对误差：0.00218204
- 平均绝对误差：0.00004385
- RMSE：0.00011966

这是启用 Vulkan FP16 storage/packed/arithmetic 后的预期小量化误差，输出形状、
非空值和稳定性均正常。需要将口型视觉质量作为发布门槛时，应继续使用真实静态
人像和音频片段进行端到端 SSIM/LPIPS 与人工审片，而不是只依赖零填充输入。

## CPU 回退策略

应用的默认策略应为：尝试 `EnableGPU(true)`；若 Vulkan 设备枚举、模型加载、
预热或首次 `extract` 失败，则重新加载 CPU 网络并记录原因，继续处理请求。
项目中的 `ModelInferencer` 已将这些检查封装为 GPU 预热失败后回退 CPU。
本机已独立验证 CPU 路径和两块 Vulkan GPU 均可产生合法输出；完整 SDK 原生构建后
还需执行一次“人为禁用 Vulkan”的进程级回退回归测试。

## 完整 SDK 构建状态

已在 Windows/UCRT64 完成完整 SDK 构建。OpenCV 4.13.0、FFmpeg 8.1.1、PortAudio
19 和 Vulkan loader 均安装在 `D:\MSYC64`。因为官方 MSVC 预编译 NCNN 与
UCRT64/MinGW 的 C++ ABI 不兼容，已使用同一 UCRT64 编译器在 D 盘构建带 Vulkan 的
NCNN，并安装在 `D:\digital-human-dev\ncnn\ucrt64-vulkan`；正式预设已指向该位置。

构建过程中同步修复了 Windows/FFmpeg 8 兼容性：非标准 `M_PI`、缺少 `<cstdint>`、
Win32 `ERROR` 宏冲突，以及已移除的旧 FFmpeg 声道布局 API。最终项目核心 DLL 与全部
示例/测试二进制均生成于
`D:\digital-human-dev\build\digital-human-sdk-windows-ucrt64-vulkan\bin`。

日后在此机器重建可执行：

```powershell
D:\MSYC64\ucrt64\bin\cmake.exe --preset windows-ucrt64-vulkan
D:\MSYC64\ucrt64\bin\cmake.exe --build --preset windows-ucrt64-vulkan
```

当前正式模型回归 `model_inferencer_test` 为 51/51 通过：Vulkan 可用时自动选择 GPU 0，
30 次连续推理全部成功（平均 12.022 ms，最大 34.852 ms），关闭 GPU 的状态切换也已通过。
仍应以项目的 `pipeline_lipsync_test` 对真实静态人脸图片、音频和视频输出完成视觉口型
端到端验收，并保留 CPU 与 Vulkan 两套结果。

## 资源目录端到端验收（2026-08-07）

实际使用 `assets/face.jpg`（1920×1384）与
`assets/voice_30s_16k_mono.wav`（30 秒、16 kHz、单声道），模型为
`Wav2Lip-SD-GAN-opt`。输出帧、MP4 与机器可读报告均保存到 D 盘，避免输出 I/O 影响
C 盘工作区：`D:\digital-human-dev\reports`。

| 验收项 | Vulkan RTX 4050 | 显式 CPU 回退 |
| --- | ---: | ---: |
| GPU 状态 | 已启用（device 0） | 已关闭 |
| 输入/输出帧 | 750 / 750 | 750 / 750 |
| 丢弃/跳过帧 | 0 / 0 | 0 / 0 |
| 内容帧率 | 25 fps | 25 fps |
| 相对实时速度 | 3.011× | 1.215× |
| 平均模型推理 | 5.084 ms | 32.736 ms |
| 嘴部平均变化量 | 5.270 | 5.273 |
| 嘴部-音频相关（平滑） | 0.683 | 0.683 |
| 自动验收 | 通过 | 通过 |

验收门槛为内容帧率不低于 24 fps、嘴部平均变化量大于 1，且帧级相关大于 0.15 或
平滑相关大于 0.2；两种执行路径均满足。人工抽检 GPU 中间帧未发现黑帧、错位或明显
面部渲染异常。
