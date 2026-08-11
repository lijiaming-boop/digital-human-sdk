# 深度学习人脸模型

`FaceDetector` 不再依赖 dlib。它使用以下模型组合：

| 职责 | 模型 | 运行时 | 原因 |
| --- | --- | --- | --- |
| 人脸检测 | SCRFD-2.5G-KPS | ncnn | 对小脸、侧脸比 HOG 稳定；模型小，且项目已链接 ncnn。 |
| 稠密关键点 | 2D106 | OpenCV DNN (ONNX) | 保留现有 68 点对齐与口部遮罩的输出契约；106 点会在 SDK 内映射为 iBUG 68 点。 |

将模型放在同一个目录（例如 `models/face/`）：

```text
models/face/
  scrfd_2.5g_kps-opt2.param
  scrfd_2.5g_kps-opt2.bin
  2d106det.onnx
```

然后将该**目录**传给 `FaceDetector::loadModel` 或
`Pipeline::SetLandmarkModelPath`（为保持源代码兼容性，后者沿用了旧名称）。

SCRFD 的 ncnn 权重必须是带 5 点输出、输出 blob 名为
`score_8/16/32` 与 `bbox_8/16/32` 的 `*-opt2` 版本。官方 ncnn SCRFD 示例提供
该命名约定；2D106 模型的输入为 RGB `192x192`，输出为 212 个归一化坐标。

模型文件被 `.gitignore` 排除，避免把大体积权重和其许可义务混入 SDK 源码。上线前必须单独
审查所选权重及训练数据的许可证；InsightFace 发布的部分预训练模型仅限非商业研究，商业 SDK
应使用已获授权或自行训练并导出的同结构模型。
