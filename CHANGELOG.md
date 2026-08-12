# Changelog

本文件记录 Digital Human SDK 的版本变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

发布产物应同时记录对应的 Git commit、CMake 配置、编译器和依赖版本，便于来源追溯。

## [Unreleased]

### Added
- 补充根目录 LICENSE 文件（MIT），与 README 声明一致。
- 新增 CHANGELOG.md，建立版本变更追溯基线。
- CMakePresets.json 提供可移植的 `linux-cpu-only` 与 `windows-vulkan` preset，
  机器专用配置移入不入库的 `CMakeUserPresets.json`。

### Changed
- `.gitignore` 通配覆盖全部 CMake 构建树（`build/`、`build-*/`、`build_wsl/`、
  `build-wsl-*/`、`build-windows-*/`），并忽略 `artifacts/` 与 `CMakeUserPresets.json`。
- `build.sh` 不再修改用户系统目录或生成的 `build.make`，改为通过 `CMAKE_LIBRARY_PATH`
  注入本地依赖链接路径，保持构建可复现且无副作用。
- `run_tests.sh` 在出现失败或超时时以非零状态退出，确保可作为 CI 门禁。

### Fixed
- 修复 `run_tests.sh` 统计失败后仍返回 0 导致 CI 误判成功的问题。

## [0.1.0] - 2026-08-11

### Added
- 完成从音频、静态头像到 Wav2Lip 推理、画面融合和 H.264/AAC 输出的核心闭环。
- OpenAI-compatible / llama.cpp 增量文本生成与增量分句。
- HTTP TTS 适配与多线程会话编排。
- JPEG/PNG 上传校验、BGR 归一化和运行中头像热更新。
- TTS PCM、数字人视频帧到 FLV、RTMP、RTSP 的编码发布。
- SDK / Pipeline 一次性生命周期、Worker Registry、统一停止时限和运行指标。
- 单轮与多轮真实链路验收。

[Unreleased]: https://github.com/lijiaming-boop/digital-human-sdk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lijiaming-boop/digital-human-sdk/releases/tag/v0.1.0
