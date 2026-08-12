# 渲染线程架构

## 1. 概述

渲染线程 (`RenderThread`) 位于流水线末端，负责从渲染队列获取推理产出的口型特征图，将其融合到原始高清背景中，并根据音频时钟进行帧调度同步，最终输出或显示视频帧。

## 2. 数据流

```
推理输出 (ncnn::Mat 448×96)         ProcessedFaceData
         │                                  │
         └──────────┬───────────────────────┘
                    ▼
            RenderTask (组合包)
                    │
                    ▼
           RenderThread 主循环
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
   1. 格式转换   2. 逆变换    3. 口唇融合
   ncnn→cv::Mat  M_inv映射   alpha blending
         │
         ▼
   4. 后处理 (锐化+色彩融合)
         │
         ▼
   5. 同步调度
   AudioPlayer时钟 → FrameScheduler决策
         │
         ├── DISPLAY  → 显示/输出帧
         ├── DROP     → 丢弃
         ├── DUPLICATE→ 重复上一帧
         └── WAIT     → 等待
```

## 3. 关键设计

### 3.1 RenderTask 数据包

```cpp
struct RenderTaskData {
    ncnn::Mat          model_output;    ///< 模型输出 448×96 RGB float
    cv::Mat            original_face;   ///< 原始人脸 (BGR uint8)
    cv::Mat            M_inv;           ///< 逆仿射变换矩阵 2×3
    cv::Mat            face_mask;       ///< 口唇遮罩 (CV_32FC1)
};
using RenderPacket = Packet<RenderTaskData>;
```

### 3.2 帧同步策略

采用 **音频主时钟** 策略，与现有 `AudioSyncScheduler` 兼容：

```
每帧处理:
  1. 获取音频播放位置 (来自 AudioPlayer)
  2. 计算 drift = video_pts - audio_clock
  3. |drift| ≥ max_drift → DROP
  4. drift > threshold  → DUPLICATE (视频超前)
  5. drift < -threshold → DROP (视频滞后)
  6. 否则 DISPLAY
```

### 3.3 帧率稳定

使用帧间隔计时器稳定输出帧率：

```
目标帧间隔 = 1000 / target_fps (ms)

每帧渲染后:
  实际耗时 = now - 上一帧时间
  剩余等待 = 目标帧间隔 - 实际耗时
  如果剩余等待 > 0 → sleep(剩余等待)
  否则 → 延迟警告 (渲染跟不上)
```

### 3.4 队列排空与优雅退出

```
停止流程:
  1. 设置停止标志
  2. 继续处理队列中的剩余帧 (最多 drain_max)
  3. 清空队列
  4. 输出 EOS
  5. 释放资源
```

## 4. 性能优化

| 优化 | 措施 |
|------|------|
| 零拷贝 | RenderTask 通过 move 语义传递 cv::Mat |
| 预分配 | OutputProcessor 内部缓冲区预分配 |
| 帧间隔计时 | 使用 steady_clock 精确测量帧间隔 |
| 跳过积压帧 | 队列深度 > 3 时主动丢帧 |
