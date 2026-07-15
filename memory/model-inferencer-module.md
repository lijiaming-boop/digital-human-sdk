---
name: model-inferencer-module
description: ModelInferencer (PIMPL模式) - Wav2Lip-SD-GAN 模型推理引擎
metadata:
  type: project
  tags: [wav2lip, inference, ncnn, performance]
---

完成 ModelInferencer 模块的开发与验收测试。

**核心设计**
- `ncnn::Net` 在 Init 时一次性加载，完整生命周期复用
- `ncnn::Extractor` 每次 Infer 时创建（轻量级，可多次创建）
- 自动 benchmark 线程数 [1,2,4,8,16] 选最优
- 若 CPU 不达标且 Vulkan 可用，自动 GPU 回退
- 内置性能计数器（推理次数、平均/最小/最大延迟）

**API**
- `Init(model_dir)` / `Init(param_path, bin_path)` — 初始化 + auto tune
- `Infer(audio_feat, face_input)` — 单帧推理，返回 3 通道输出
- `SetThreadCount(n)` — 手动设置线程数
- `EnableGPU(bool)` — GPU 开关
- `GetAvgLatencyMs()` / `GetMinLatencyMs()` / `GetMaxLatencyMs()` — 性能统计
- `ResetStats()` / `PrintStats()` — 统计管理

**性能数据 (Ryzen 7 7840HS)**
- 最佳 CPU 延迟: ~75ms (8 线程) — 未达 50ms 目标
- Vulkan GPU: ~80ms — 当前 ncnn 无真正的 GPU 加速
- 50ms 目标需 GPU 或更优硬件

**模型输出**
- Wav2Lip-SD-GAN 输出形状: (w=448, h=96, c=3)
- 输入音频: (w=80, h=80, c=1)
- 输入人脸: (w=96, h=96, c=6)

**测试覆盖**: 51 个用例，含模型加载、推理验证、延迟测量、GPU 检测、Move 语义、空输入
