# 模型精度与量化入门：为数字人 SDK 安全地部署和升级 Wav2Lip

这份文档面向第一次接触模型部署和量化的同学。目标不是“把模型文件变小”，而是在**口型质量、延迟、内存和硬件兼容性**之间做可验证的取舍。

本文以当前 SDK 的 Wav2Lip NCNN 模型为例：

```text
输入 1：audio_sequences，16 x 80 x 1   （mel 频谱）
输入 2：face_sequences， 96 x 96 x 6   （人脸与掩码拼接特征）
输出： output
```

这些名字和尺寸是模型契约的一部分。升级模型时，不得仅以文件名或文件大小判断兼容性。

---

## 1. 先建立正确的心智模型

一次神经网络推理包含两类数：

- **权重（weight）**：训练完成后固化在 `.bin` 内的参数；模型“更大”通常首先意味着它更多。
- **激活（activation）**：每次输入人脸、音频后，层与层之间临时产生的数据。

精度就是用多少位二进制数保存和计算这些数。数位少，数据移动和存储通常更少；但能表达的数值范围与精细度也更小，可能造成质量下降。

`INT8` 的意思并不是“结果只有 8 位”，而是让可量化的权重和中间激活以 8 位整数参与计算，再按量化比例还原到可用的数值范围。

> 结论：量化是一次有损的近似。必须拿真实的人脸与音频验证，不能只看模型能否加载。

---

## 2. FP32、FP16、INT8 与混合精度分别是什么

### 2.1 FP32：32 位浮点数，质量基准

FP32 是最常见的训练和部署基线。它数值范围大、误差小，通常最容易复现训练结果。

- 优点：最稳妥；适合做回归对照和排查数值问题。
- 缺点：权重与激活占用较大；CPU 内存带宽压力较高。
- 本项目定位：**永远保留一份 FP32 参考模型**，它不是可随意删除的“旧文件”。

如果量化版出现嘴部抖动、口型偏移或颜色异常，先与同一输入下的 FP32 输出比较，才能判断问题来自量化、预处理还是渲染。

### 2.2 FP16：16 位浮点数，GPU 友好候选

FP16 仍是浮点数，只是每个数占用从 4 字节降至 2 字节。相较 INT8，它通常保留更多连续数值信息；相较 FP32，它可以明显减少模型和中间数据的搬运。

- 优点：常适合支持半精度的 GPU；改变量较温和，质量风险通常低于全 INT8。
- 缺点：不是每个 Vulkan 设备都能因此更快；在普通 CPU 上也未必有明显收益。
- 本项目定位：**Vulkan 路径的优先候选之一**，必须在目标显卡实测后选用。

不要把“GPU”自动等同于“FP16 一定最快”。小模型、频繁 CPU/GPU 传输、Vulkan 同步或渲染耗时，都可能掩盖模型侧收益。

### 2.3 INT8：8 位整数，CPU 部署的重点候选

INT8 把浮点值映射为整数。一个常见的线性量化关系是：

```text
q = round(x / scale) + zero_point
x ≈ (q - zero_point) * scale
```

其中 `x` 是原浮点数，`q` 是存储/计算用的整数；`scale` 和 `zero_point` 保存映射规则。实际 NCNN 的具体层格式由量化工具生成，无需在业务代码手写这个公式。

- 优点：权重通常约为 FP32 的四分之一；可降低 CPU 的内存带宽压力，许多卷积网络能获益。
- 缺点：激活范围估计不准会放大量化误差；口型生成对时序和局部细节敏感，全模型 INT8 可能带来闪烁或嘴形失真。
- 本项目定位：**CPU 路径的首选试验方案**，通过质量门禁后才可作为默认版本。

NCNN 的后训练量化（PTQ）工具可把 FP32 模型转换为 INT8；运行时加载量化后的 `.param/.bin` 即可，推理 API 不需要改成另一套调用方式。[NCNN 官方量化说明](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference)

### 2.4 混合精度：只量化适合的层

混合精度不是第四种单独的数据类型，而是一种部署策略：大部分卷积使用 INT8，少数误差敏感的层仍使用 FP32（或在 GPU 上用 FP16）。

对 Wav2Lip，建议按下面顺序尝试，而不是一开始就猜测哪些层必须保留：

1. 得到全 INT8 候选模型；
2. 在固定人脸与语音集合上跑质量回归；
3. 用层级误差或失败样本定位敏感层；
4. 让这些层回退到 FP32，再生成混合精度版本；
5. 重新测质量、p95 延迟和内存。

NCNN 允许在量化表中注释某一层的权重 scale，使其保持 FP32 推理，这是实现混合精度的直接方式。[NCNN 官方量化说明](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference)

---

## 3. 为什么“模型更大”不等于“做 INT8 就安全”

较大模型可能增加三种压力：

| 压力 | 常见表现 | INT8 能否解决 |
| --- | --- | --- |
| 权重存储与加载 | 模型文件大、初始化慢、内存高 | 通常有帮助 |
| 单帧计算量 | 推理延迟上升 | 取决于算子、CPU/GPU 与驱动 |
| 激活/显存 | 高分辨率或宽网络导致运行时内存高 | 部分有帮助 |

所以新模型的性能不能由“量化成功”决定，必须由端到端基准决定。特别是当前 SDK 中，模型推理和 OpenCV 融合渲染都占用时间；即便模型缩短，端到端 FPS 也不一定同比增长。

---

## 4. 本项目建议维护的模型版本

推荐让模型版本在目录与元数据上明确区分，不覆盖旧模型：

```text
models/wav2lip/<model-id>/
  fp32/
    model.param
    model.bin
  int8/
    model.param
    model.bin
    calibration.table
  fp16/                 # 仅在目标 GPU 实测获胜时提供
    model.param
    model.bin
  manifest.json
```

`manifest.json` 至少记录：

```json
{
  "model_id": "wav2lip-sd-gan-v1",
  "source_precision": "fp32",
  "variant": "int8-mixed",
  "sha256": "<param/bin 的校验值>",
  "inputs": {
    "audio_sequences": [16, 80, 1],
    "face_sequences": [96, 96, 6]
  },
  "output": "output",
  "benchmark_profile": "desktop-vulkan-a",
  "quality_baseline": "wav2lip-sd-gan-v1-fp32"
}
```

`sha256` 防止“同名文件被悄悄替换”；输入、输出信息防止模型加载成功却因 blob 名称或张量尺寸改变而产生错误结果。

---

## 5. 从 FP32 制作 INT8：可复现流程

### 5.1 前置条件

1. 保留原始 FP32 `.param/.bin`，不要就地覆盖。
2. 使用与 SDK 运行时同版本或兼容版本的 NCNN 工具。
3. 收集**有授权、可长期保留**的校准样本；校准样本不能来自线上用户私密数据。
4. 样本应覆盖真实情形：不同人脸、肤色、姿态、光照、嘴部开合、语速和静音段。

官方示例建议使用验证集做校准；对图像模型建议规模达到数千张。对于本项目的双输入模型，更关键的是覆盖真实的“音频特征—人脸帧”分布，而非只追求数量。[NCNN 官方量化说明](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference)

### 5.2 生成真实的双输入校准数据

不要把 JPEG、视频帧或原始 WAV 直接交给 `ncnn2table`。模型收到的是 SDK 预处理后的两个张量：

```text
calibration/
  audio/000001.npy   # NPY 的 NCHW 形状为 [1,80,16]；NCNN 输入仍是 [16,80,1]
  face/000001.npy    # NPY 的 NCHW 形状为 [6,96,96]；NCNN 输入仍是 [96,96,6]
  audio.list
  face.list
```

两个 list 的第 N 行必须是一对来自同一时刻的样本；顺序不能错位。因为 `audio_sequences` 在 `.param` 中排在前面，命令中也应先传 `audio.list`，再传 `face.list`。项目的 `pipeline_lipsync_test` 可直接用 `calibration_dir` 参数导出这些真实 NPY；文件本身按 NCHW 保存，而 `ncnn2table` 的 `shape` 参数按 NCNN 的 `W,H,C` 填写。

### 5.3 转换命令

以下命令是建议的可复现模板；工具路径按实际 NCNN 安装位置替换：

```bash
# 1) 图优化：产物独立保存
ncnnoptimize \
  models/wav2lip/source/model.param models/wav2lip/source/model.bin \
  models/wav2lip/int8/model-opt.param models/wav2lip/int8/model-opt.bin 0

# 2) 以真实双输入 NPY 生成校准表。shape 顺序为 NCNN 的 W,H,C。
ncnn2table \
  models/wav2lip/int8/model-opt.param models/wav2lip/int8/model-opt.bin \
  calibration/audio.list,calibration/face.list \
  models/wav2lip/int8/calibration.table \
  shape=[16,80,1],[96,96,6] type=1 thread=4 method=kl

# 3) 生成 INT8 模型
ncnn2int8 \
  models/wav2lip/int8/model-opt.param models/wav2lip/int8/model-opt.bin \
  models/wav2lip/int8/model.param models/wav2lip/int8/model.bin \
  models/wav2lip/int8/calibration.table
```

`type=1` 表示以 NPY 输入；NCNN 对多输入模型接受多个 list 文件，`shape` 使用 `W,H,C` 参数顺序。`kl` 与 `aciq` 是可选的校准算法，应以质量和性能实测来选择，不应凭名称判断优劣。[NCNN 官方量化说明](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference)

### 5.4 制作混合精度版本

如果全 INT8 未通过质量门禁：

1. 复制 `calibration.table` 为 `calibration-mixed.table`；
2. 注释需要保留 FP32 的层权重 scale 行，例如 `#<layer_name>_param_0 ...`；
3. 重新执行 `ncnn2int8`；
4. 将输出标记为 `int8-mixed`，不得误称为全 INT8。

不能在没有证据时武断地注释一批层。每一次排除都应对应一条失败样本、一个质量指标或一份误差分析记录。

---

## 6. 性能与质量门禁：更换高权重模型时如何保证不退化

“保证性能”的正确含义是：在上线前用固定、可复现的测试拒绝不合格版本，并在运行时保留回退路径；不是承诺每个新模型都更快。

### 6.1 结构门禁（必须为 0 错误）

- `.param/.bin` 可加载；校验值与清单一致。
- 输入 blob 为 `audio_sequences` 与 `face_sequences`，输出 blob 为 `output`。
- 输入尺寸符合 `[16,80,1]` 与 `[96,96,6]`，或代码和 manifest 已同时完成兼容升级。
- CPU 推理、Vulkan 推理（若设备可用）均至少完成一次真实提取，输出非空且无 NaN/Inf。

### 6.2 性能门禁（同一机器、同一数据集）

不要只测一帧。每个候选版本在预热后至少测 100 帧，分别报告：

| 指标 | 解释 |
| --- | --- |
| 首帧时间 | 模型加载、Vulkan 管线创建及缓存预热的体验 |
| p50 / p95 / p99 推理延迟 | 正常、尾延迟和极端抖动 |
| 端到端 FPS 与丢帧率 | 用户真正看到的吞吐结果 |
| 峰值 RSS / 显存 | 大模型是否超过设备预算 |
| CPU 与 GPU 版本的相对结果 | 决定默认后端，而非依赖理论 |

若目标是 25 FPS，端到端预算是 40 ms/帧。但该预算包含推理、融合渲染、队列和同步；因此模型本身应预留余量，具体阈值以当前 FP32 基线和目标机器测得的渲染时间制定。

SDK 已有 `pipeline_lipsync_test` 和性能测试样例。应固定测试命令、机器规格、线程数、模型版本与样本集，再将测量结果作为候选模型的发布记录。

### 6.3 质量门禁（INT8 最容易被遗漏）

每个 FP16、INT8、混合精度版本都要与同输入的 FP32 输出比较：

- **口型同步**：音素出现时，张嘴/闭嘴的时机是否一致；
- **嘴部局部质量**：牙齿、唇缘、张口形状是否异常；
- **时间稳定性**：连续帧是否闪烁、跳动或颜色漂移；
- **人脸稳定性**：嘴部融合是否破坏身份特征、边缘或肤色；
- **人工盲测**：对主观视觉瑕疵尤其重要。

像素 MSE/PSNR 可以辅助定位，但不能单独作为口型模型的发布条件：它们无法可靠表达“声音与嘴形是否同步”。

### 6.4 运行时选择与回退

建议按顺序尝试已通过验收的变体：

```text
目标 Vulkan GPU 的已验证最快版本
        ↓ 不可用、失败或超时
已验证的 CPU INT8/混合精度版本
        ↓ 不可用或质量/结构校验失败
FP32 基准版本
```

Vulkan 初始化失败时必须如当前实现一样明确回退 CPU；不能把“请求开启 GPU”当作“GPU 已真实推理”。同样，不能因为 INT8 文件存在，就跳过 FP32 回退包。

---

## 7. 新手常见误区

| 误区 | 正确做法 |
| --- | --- |
| 模型小 4 倍，延迟一定小 4 倍 | 分别实测权重加载、推理、渲染与同步；通常不会线性缩短。 |
| 校准时输入全零也可以 | 用运行时真实预处理结果；全零无法代表音频和人脸激活分布。 |
| INT8 只影响模型文件 | 激活量化同样会影响输出，常是质量差异来源。 |
| GPU 上 INT8 必然最佳 | 在目标 Vulkan 设备比较 FP32、FP16、INT8 和混合精度。 |
| 能加载、能输出图片就说明成功 | 还必须通过时序口型质量、p95 延迟、内存和回退测试。 |
| 直接覆盖旧 `.bin` 最省事 | 保留不可变版本、hash 和 FP32 基线，才能定位和回退。 |

---

## 8. 一份可执行的上线清单

1. [ ] 固化 FP32 参考模型及其 `manifest` 和 hash。
2. [ ] 从真实 SDK 预处理结果导出成对的 `audio.npy`、`face.npy`。
3. [ ] 用 `ncnnoptimize`、`ncnn2table`、`ncnn2int8` 生成全 INT8 候选。
4. [ ] 通过结构门禁：blob 名、shape、非空输出、NaN/Inf、hash。
5. [ ] 在固定 CPU 与 Vulkan 机器上跑预热后的多帧基准。
6. [ ] 与 FP32 完成口型、嘴部、时序稳定性和人工盲测比较。
7. [ ] 若质量失败，基于证据生成混合精度候选并重复测试。
8. [ ] 将获胜版本、测试环境、原始结果和回退版本写入发布记录。
9. [ ] 只让通过全部门禁的版本成为默认；保留 FP32 与 CPU 回退。

---

## 9. 推荐的决策顺序

```text
先修复/验证 Vulkan 真实可用性
        ↓
为 CPU 制作并评测 INT8
        ↓
为目标 GPU 比较 FP32、FP16、INT8
        ↓
若全 INT8 质量不足，尝试混合精度
        ↓
由端到端 p95、内存和口型质量共同决定默认版本
```

这套顺序的核心是：**精度格式是候选方案，测量和质量门禁才是决策机制。**

---

## 10. 本项目的一键式复现命令

先构建端到端测试程序，再用真实图片与音频生成校准集、构建变体并选择版本：

```bash
cmake --build build --target pipeline_lipsync_test -j2

# 生成至少 100 对真实 audio/face 张量；这里 5 秒、25 FPS 产生 125 对。
build/bin/pipeline_lipsync_test assets 5 25 zw_trimmed.mp3 4 \
  build/precision-eval/fp32 \
  models/Wav2Lip-SD-GAN-opt.param models/Wav2Lip-SD-GAN-opt.bin 0 \
  build/precision-eval/calibration 125 || true

# 生成 FP16 与 CPU INT8 候选。ncnn-tools-dir 常为 /usr/local/bin。
bash scripts/quantize_wav2lip.sh \
  models/Wav2Lip-SD-GAN-opt.param models/Wav2Lip-SD-GAN-opt.bin \
  build/precision-eval/calibration build/precision-eval/variants /usr/local/bin

# 分别运行 fp32、fp16、int8；第 7、8 参数替换为对应 .param/.bin。
build/bin/pipeline_lipsync_test assets 5 25 zw_trimmed.mp3 4 \
  build/precision-eval/reports/int8 \
  build/precision-eval/variants/int8/model.param \
  build/precision-eval/variants/int8/model.bin 0 || true

# 根据端到端 JSON 报告选择；fp32 始终作为回退。
python3 scripts/select_wav2lip_variant.py \
  build/precision-eval/reports build/precision-eval/selection.json
```

第 9 个参数设为 `1` 才会请求 Vulkan。结果文件会记录 `gpu_enabled`；只有它为 `true` 才代表设备完成了真实模型推理。若全 INT8 未通过质量门禁，设置 `MIXED_EXCLUDE`（由失败样本定位的层前缀）后再执行量化脚本，即会构建 `int8-mixed` 候选。
