---
name: output-processor-module
description: OutputProcessor (PIMPL模式) - 模型输出后处理、人脸融合、锐化与色彩校正
metadata:
  type: project
  tags: [wav2lip, post-processing, face-fusion, image-blending]
---

完成 OutputProcessor 模块的开发与验收测试。

**核心功能**
- `OutputToMat(ncnn::Mat)` — 模型输出转 cv::Mat (RGB float → BGR uint8)
- `extractFaceRegion()` — 从 448×96 输出中提取有效人脸区域
- `InverseTransform(face, M_inv)` — 逆仿射变换回原始图像坐标
- `FaceFusion(orig, gen, mask)` — alpha 遮罩口唇融合
- `Sharpen(image, strength)` — USM 非锐化遮罩增强
- `ColorBlend(gen, orig, alpha)` — 加权色彩融合
- `Process()` — 全流程管道（转换→逆变换→融合→后处理）

**测试覆盖**: 67 个用例，涵盖格式转换、逆变换、融合逻辑、锐化、色彩混合、空输入、Move 语义、极端值
