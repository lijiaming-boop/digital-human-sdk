# Wav2Lip 口型驱动失效根因分析与 Mel 预处理修复

## 问题现象

端到端管线 `pipeline_lipsync_test` 运行 30s 音频 + `face.jpg` 推理后，验证报告显示：

```
口型-能量相关性: -0.169376 (Pearson r, 帧级)
[结论] 帧率达标(>=24fps), 口型驱动无效
```

嘴部变化量与音频能量几乎无相关性（甚至为负），模型输出未随音频驱动口型。

---

## 诊断方法

### 诊断脚本

新增 [lip_sync_response_test.cpp](../../examples/lip_sync_response_test.cpp)，用 `assets/silence_30s_16k_mono.wav`（静音）与 `voice_30s_16k_mono.wav`（语音）两组截然不同的音频，对同一张脸推理 N=8 个窗口，量化对比：

- Mel 输入差异（基线，证明输入确实不同）
- 模型输出全图差异 / 嘴部 ROI 差异
- 帧间差异（输出是否随时间变化）
- 响应比 = 输出差异 / 输入差异

### 诊断结果（修复前）

| 指标 | 数值 | 含义 |
|------|------|------|
| Mel 输入差异（静音 vs 语音） | 0.646 | 输入差异正常 |
| 模型输出全图差异 | 0.0026 | 输出几乎不变 |
| 嘴部 ROI 差异 | 0.0056 | 嘴部微弱变化 |
| 静音组帧间差异 | 0 | 静音时输出完全恒定 |
| 语音组帧间差异 | 0.0022 | 语音时仅微小变化 |
| 响应比 | 0.4% | **几乎不响应音频** |

**判定：弱响应**。模型输出 mean≈0.467（接近 sigmoid 0.5），静音时完全不变——模型实际把人脸输入直接重建，忽略了音频条件。

---

## 根因分析

### 根因一：Mel 预处理与 Wav2Lip 官方实现严重不匹配（已修复）

对比 [audio_mel_feature_extract.cpp](../../src/audio/audio_mel_feature_extract.cpp) 旧实现与 [Wav2Lip 官方 audio.py](https://github.com/Rudrabha/Wav2Lip/blob/master/audio.py)，发现 **7 处不匹配**：

| 参数 | Wav2Lip 官方 | 旧实现 | 影响 |
|------|------------|--------|------|
| `n_fft` | **800** | 512 | 频谱分辨率错 |
| `win_size` | **800** | 等于 nFFT | 窗长错 |
| `fmin` | **55** | 0 | 低频噪声进入 |
| `fmax` | **7600** | 8000 | 高频范围错 |
| 窗函数 | **Hann** | 无 | 频谱泄漏 |
| dB 公式 | `20·log10(|D|)` 振幅谱 | `10·log10(power)` 功率谱 | 差 2 倍 |
| 归一化 | symmetric **[-4,4]** + `ref_level_db=20` + `min_level_db=-100` | min-max **[0,1]** 逐次独立统计 | **致命**，分布完全错 |
| Mel 滤波器 | librosa **Slaney area-normalized** | 自实现三角滤波器无归一化 | 权重不同 |

**最致命的问题**：`audio_processor.cpp` 调用 `mel_extract.extract(..., apply_minmax=false)`，输出原始 dB 值（约 [-100,-20]），完全没做 Wav2Lip 的 symmetric 归一化。模型期望 [-4,4] 归一化输入，收到的是原始 dB 值——输入分布严重偏移（OOD），模型无法正确响应音频条件。

### 根因二：模型权重可疑（待验证）

修复 Mel 预处理后重新诊断：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Mel 输入差异 | 0.646 | **7.28**（归一化后值域更宽） |
| 模型输出全图差异 | 0.0026 | 0.0029 |
| 输出 mean | 0.467 | 0.467（**完全不变**） |
| 响应比 | 0.4% | 0.04% |

Mel 输入差异从 0.646 升到 7.28（证明归一化生效），但**模型输出 mean 仍是 0.467 完全不变**——说明模型本身对音频输入无响应。

进一步检查：
- [.param 文件](../../models/Wav2Lip-SD-GAN-opt.param) 结构正确：有 `audio_sequences` 输入 + 完整 `audio_encoder` 卷积分支
- [model_inferencer.cpp](../../src/model/model_inferencer.cpp) 推理代码正确：音频/人脸输入都正确注入 ncnn Extractor
- **.bin 文件仅 72MB**，而 Wav2Lip 完整权重通常 400MB+——权重可能不完整或不是正确的 Wav2Lip 权重

**判定**：Mel 预处理已修复（必要前置工作），但当前 `Wav2Lip-SD-GAN-opt.bin` 权重本身可疑，需要重新从 PyTorch 原始权重转换或更换模型。

---

## 解决方案

### 已完成：Mel 预处理对齐 Wav2Lip 官方

#### 1. [audio_mel_feature_extract.h](../../include/audio/audio_mel_feature_extract.h) — MelConfig 扩展

新增字段，默认值对齐 Wav2Lip 官方：

```cpp
struct MelConfig {
    int   nFFT        = 800;     // FFT 点数（旧 512）
    int   nMels       = 80;
    int   sampleRate  = 16000;
    float fMin        = 55.0f;   // 旧 0.0f
    float fMax        = 7600.0f; // 旧 8000.0f
    int   winSize     = 800;     // 新增：窗长
    float refLevelDb  = 20.0f;   // 新增：参考电平
    float minLevelDb  = -100.0f; // 新增：最低电平
    float maxAbsNorm  = 4.0f;    // 新增：symmetric 归一化上界
};
```

`apply_minmax=true` 现在做 Wav2Lip symmetric 归一化（而非旧的 min-max [0,1]）。

#### 2. [audio_mel_feature_extract.cpp](../../src/audio/audio_mel_feature_extract.cpp) — 实现重写

| 步骤 | 旧实现 | 新实现 |
|------|--------|--------|
| 窗函数 | 无 | **Hann 窗**（对齐 numpy.hanning） |
| 频谱 | 功率谱 `real²+imag²` | **振幅谱** `sqrt(real²+imag²)` |
| dB 转换 | `10·log10(power)` | **`20·log10(|D|) - ref_level_db`** |
| Mel 滤波器 | 三角滤波器无归一化 | **Slaney area-normalized**（每个滤波器除以面积） |
| Mel 公式 | `2595·log10(1+hz/700)` | **Slaney 风格**（对齐 librosa） |
| 归一化 | min-max [0,1] 逐次独立 | **symmetric [-4,4]**：`(dB - min_level_db)·10/(-min_level_db)` + clip |

#### 3. [audio_processor.cpp](../../src/core/audio_processor.cpp) — 调用方修复

- `UpdateMelConfig()`：fMin=55, fMax=7600, winSize=nFFT, 传入 symmetric 归一化参数
- `mel_extract.extract(...)` 调用从 `apply_minmax=false` 改为 `true`（做 symmetric 归一化）
- 移除 CMVN 后处理（Wav2Lip symmetric 归一化已是模型期望的最终输入）

#### 4. [audio_processor.h](../../include/core/audio_processor.h) — 配置默认值

`AudioProcessorConfig::nfft` 默认值从 512 改为 800，`AutoConfigure()` 同步更新。

#### 5. 诊断脚本修复

- [lip_sync_response_test.cpp](../../examples/lip_sync_response_test.cpp)
- [lip_sync_diagnose_test.cpp](../../examples/lip_sync_diagnose_test.cpp)

移除 CMVN，改用 Wav2Lip 官方 Mel 参数 + symmetric 归一化。

### 待办：模型权重验证 / 替换

Mel 预处理修复是必要前置工作，但当前模型权重仍无响应。后续路径：

1. **验证权重**：从 [Wav2Lip 官方仓库](https://github.com/Rudrabha/Wav2Lip) 下载 `Wav2Lip.pth`，用 `onnx2ncnn` 重新转换为 ncnn 格式，替换 `models/Wav2Lip-SD-GAN-opt.bin/.param`
2. **若仍无响应**：考虑替换为 MuseTalk（256×256 实时）或 DINet（纯 CNN 友好 ncnn），详见模型调研报告

---

## 验证方法

### 重新运行诊断

```bash
cd /mnt/c/Users/27013/Desktop/digital-human-sdk
./build/bin/lip_sync_response_test
```

期望结果（权重修复后）：
- 响应比 > 5%
- 嘴部 ROI 差异 > 0.02
- 输出 mean 随音频变化

### 重新运行端到端管线

```bash
./build/bin/pipeline_lipsync_test assets 30 25 zw_trimmed.mp3 0 /home/$USER/dh_lipsync_run
```

期望结果（权重修复后）：
- 口型-能量相关性 Pearson r > 0.3
- 结论变为"口型驱动有效"

---

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| [include/audio/audio_mel_feature_extract.h](../../include/audio/audio_mel_feature_extract.h) | MelConfig 新增 winSize/refLevelDb/minLevelDb/maxAbsNorm 字段，默认对齐 Wav2Lip |
| [src/audio/audio_mel_feature_extract.cpp](../../src/audio/audio_mel_feature_extract.cpp) | 重写：Hann 窗 + 振幅谱 + 20·log10 + symmetric 归一化 + Slaney 滤波器 |
| [include/core/audio_processor.h](../../include/core/audio_processor.h) | AudioProcessorConfig::nfft 默认 512→800 |
| [src/core/audio_processor.cpp](../../src/core/audio_processor.cpp) | UpdateMelConfig 对齐 Wav2Lip；apply_minmax=true；移除 CMVN |
| [examples/lip_sync_response_test.cpp](../../examples/lip_sync_response_test.cpp) | 移除 CMVN，用 symmetric 归一化 |
| [examples/lip_sync_diagnose_test.cpp](../../examples/lip_sync_diagnose_test.cpp) | 移除 CMVN，用 symmetric 归一化 |

---

## 参考资料

- [Wav2Lip 官方 hparams.py](https://github.com/Rudrabha/Wav2Lip/blob/master/hparams.py)
- [Wav2Lip 官方 audio.py](https://github.com/Rudrabha/Wav2Lip/blob/master/audio.py)
- [librosa.filters.mel Slaney 归一化](https://librosa.org/doc/latest/generated/librosa.filters.mel.html)
