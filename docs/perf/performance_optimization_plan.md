# 数字人性能优化方案

## 已确认的瓶颈与本次处置

基准为 16 核 WSL2 上的 `perf_benchmark`。原始火焰图中，`libgomp` 的两个
barrier 自旋栈合计约 60.3% self CPU；除去该噪声后，主要计算为 Wav2Lip 的
ncnn Winograd/GEMM（约 25%）以及 OpenCV 的 resize/alpha blend（约 5%）。

本次代码变更：

1. 在第一次 ncnn/OpenMP 执行前设置 `OMP_WAIT_POLICY=PASSIVE` 与
   `GOMP_SPINCOUNT=0` 的默认值；若进程已设置同名变量，SDK 不覆盖。
2. 将 `ncnn::Net::opt.num_threads` 移至 `load_param/load_model` 之前，保证
   Winograd/GEMM pipeline 按目标线程数创建。
3. 取消初始化后仅修改 `net.opt` 的伪自动调优；线程数切换会重载模型，避免
   日志中的 `convolution winograd will use load-time value` 警告。

## 推荐部署配置

先从 2 个 ncnn 推理线程开始，并保持 OpenCV 至多 4 线程；在 `Pipeline::Start()`
前调用 `SetInferenceThreads(2)` 或 `SetInferenceThreads(4)`。使用环境变量可
显式覆盖 SDK 默认的等待策略：

```bash
OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0 OMP_NUM_THREADS=4 ./perf_benchmark
```

`OMP_NUM_THREADS` 是部署上限，不应大于 ncnn 线程配置；流水线还有音频、视频、
匹配和渲染 worker，16 核机器不建议让单次推理独占全部核心。

## 验收方法与目标

在同一模型、音频、帧率和机器上，分别采集基线与优化版各三次 25 秒样本。记录：

| 指标 | 目标 |
| --- | --- |
| `libgomp` self CPU | 从 60.3% 降至 < 3% |
| Winograd load-time 警告 | 0 条 |
| p95 推理耗时 / 端到端延迟 | 不高于基线 |
| 输出 FPS、丢帧率 | 不低于 / 不高于基线 |
| 常驻内存 | 增幅 < 5% |

命令：

```bash
bash scripts/profile_flamegraph.sh build-profile/bin/perf_benchmark 999 fp
```

同时保存 `perf report --sort dso`、基准程序输出的 p50/p95/p99 和线程数，避免
只依据总 CPU 利用率作判断。

## 后续优化路线

| 优先级 | 方向 | 预期收益 | 完成标准 |
| --- | --- | --- | --- |
| P1 | 线程预算 sweep：ncnn 1/2/4，OpenCV 1/2/4，按实际流水线负载测试 | 降尾延迟、避免过订阅 | 选出每个部署规格的固定配置 |
| P1 | GPU 路径端到端验证（含上传、同步、回退） | CPU 紧张设备的吞吐提升 | GPU p95 和失败回退均达标 |
| P2 | 合成 ROI 优化：只处理嘴部 ROI，合并 resize/blend | 降低约 5% OpenCV 热点 | 图像回归通过且 render p95 降低 |
| P2 | 静态人脸预计算：复用对齐结果、mask、变换矩阵 | 降视频前处理和分配 | 命中率、内存上限、场景切换正确 |
| P3 | ncnn 模型部署优化：INT8/FP16、目标 CPU 编译、算子融合 | 降真实推理算力 | 口型质量回归与性能目标同时通过 |
| P3 | 端到端持续性能门禁 | 防止性能回退 | CI 保存基线并对 p95/FPS 设阈值 |

P1 完成后再投入 P2：当前首要问题是并行等待浪费，先把 CPU 时间还给真实推理，
再优化合成细节才具有可量化的收益。
