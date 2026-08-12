# 文档索引

文档按用途分类。架构总览以当前代码为准；`reference/`、`review/` 和 `bugfix/` 中包含历史材料，不应替代当前架构说明。

## 架构

- [架构总览](architecture/overview.md)
- [第一阶段稳定性改造：生命周期、线程回收与配置一致性](architecture/phase1_stability_refactor.md)
- [第二阶段改造：生命周期收敛、Worker Registry 与可观测性](architecture/phase2_lifecycle_observability_refactor.md)
- [用户图片驱动的实时数字人会话实现方案](architecture/realtime_avatar_conversation_implementation.md)
- [项目完善方案与实施路线](architecture/project_completion_roadmap.md)
- [多线程 Pipeline](architecture/multi_thread_pipeline.md)
- [音频处理线程](architecture/audio_processor_thread.md)
- [推理线程](architecture/inference_worker.md)
- [渲染线程](architecture/render_thread.md)

## 使用与部署

- [会话服务接口协议](guides/dialog_service_protocol.md)
- [llama.cpp 接入指南](guides/llama_cpp_integration.md)
- [实时头像数字人会话运行指南](guides/realtime_avatar_conversation.md)
- [全链路闭环验收](guides/end_to_end_validation.md)
- [音视频编码与 RTMP/RTSP 推流](guides/stream_publishing.md)
- [Windows Vulkan GPU 验证](guides/windows_vulkan_gpu_validation.md)

## 模型

- [人脸模型](models/face_models.md)
- [模型精度与量化](models/model_precision_quantization_guide.md)

## 性能

- [流水线性能优化报告](perf/pipeline_optimization_report.md)
- [性能优化计划](perf/performance_optimization_plan.md)
- [火焰图分析](perf/perf_flamegraph_analysis.md)
- `perf/` 下的 SVG、folded、data 和日志是对应分析的原始证据。

## 维护记录

- `bugfix/`：历史缺陷修复说明。
- `review/`：阶段性代码审查报告。

## 历史参考

- [Code Wiki](reference/code_wiki.md)
- [项目分析](reference/project_analysis.md)
- [框架与推理面试参考](reference/interview_faq_framework_inference.md)
