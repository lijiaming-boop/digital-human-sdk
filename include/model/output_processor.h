#pragma once

#include <memory>
#include <mat.h>
#include <opencv2/core.hpp>

namespace digital_human {
namespace model {

/**
 * @brief Wav2Lip 模型输出后处理器
 *
 * 负责将模型推理输出的 ncnn::Mat (RGB float) 转换为 OpenCV 图像，
 * 逆变换回原始图像坐标空间，与原始人脸进行口唇区域融合，
 * 最后进行锐化和色彩融合后处理。
 *
 * 处理流程：
 *   ncnn::Mat → cv::Mat (BGR uint8) → 逆变换 → 口唇融合 → 锐化 → 色彩融合
 *
 * 采用 PIMPL（Pointer to Implementation）模式。
 */
class OutputProcessor {
public:
    OutputProcessor();
    ~OutputProcessor();
    OutputProcessor(const OutputProcessor&) = delete;
    OutputProcessor& operator=(const OutputProcessor&) = delete;
    OutputProcessor(OutputProcessor&&) noexcept;
    OutputProcessor& operator=(OutputProcessor&&) noexcept;

    // ==================== 格式转换 ====================

    /**
     * @brief 将模型输出 ncnn::Mat 转换为 OpenCV cv::Mat
     *
     * 转换流程：
     *   1. 从 ncnn::Mat (w, h, 3, float, RGB) 提取像素
     *   2. 从 448×96 输出中裁剪有效人脸区域 (face_w × face_h)
     *   3. RGB float → BGR uint8 转换（值域 [0,1] → [0,255]）
     *
     * @param model_output 模型输出 ncnn::Mat，形状 (w, h, 3)，RGB float
     * @param face_w       提取的人脸区域宽度（默认 96）
     * @param face_h       提取的人脸区域高度（默认 96）
     * @return cv::Mat     BGR uint8 图像，空 Mat 表示转换失败
     */
    cv::Mat OutputToMat(const ncnn::Mat& model_output,
                        int face_w = 96, int face_h = 96);

    // ==================== 逆变换 ====================

    /**
     * @brief 将生成的人脸从对齐空间映射回原始图像空间
     *
     * 使用 FaceAligner 的 M_inv 矩阵进行逆仿射变换，
     * 将 96×96 的对齐人脸映射回原始图像的坐标系。
     *
     * @param processed_face 生成的人脸图像 (BGR uint8)
     * @param M_inv          逆仿射变换矩阵 2×3（来自 FaceAligner）
     * @param original_size  原始图像尺寸
     * @return cv::Mat       映射后的图像，空 Mat 表示失败
     */
    cv::Mat InverseTransform(const cv::Mat& processed_face,
                             const cv::Mat& M_inv,
                             const cv::Size& original_size);

    // ==================== 人脸融合 ====================

    /**
     * @brief 将生成的人脸与原始图像进行口唇区域融合
     *
     * 使用 FaceMaskGenerator 生成的口唇遮罩进行 alpha 混合：
     *   result = mask * generated_face + (1 - mask) * original_image
     *
     * @param original_image 原始图像 (BGR uint8)
     * @param generated_face 生成的人脸图像 (BGR uint8)，需与 original 同尺寸
     * @param face_mask      口唇区域遮罩 (CV_32FC1, [0,1])，需与 original 同尺寸
     * @return cv::Mat       融合后的图像，空 Mat 表示失败
     */
    cv::Mat FaceFusion(const cv::Mat& original_image,
                       const cv::Mat& generated_face,
                       const cv::Mat& face_mask);

    // ==================== 后处理 ====================

    /**
     * @brief 图像锐化（USM 非锐化遮罩）
     *
     * unsharp_mask = 原图 + strength × (原图 - 高斯模糊)
     *
     * @param image    输入图像 (BGR uint8)
     * @param strength 锐化强度（默认 1.0，范围 [0, 3]）
     * @return cv::Mat 锐化后的图像
     */
    cv::Mat Sharpen(const cv::Mat& image, float strength = 1.0f);

    /**
     * @brief 色彩融合
     *
     * 将生成图像与原始图像进行加权混合，保留原始色彩信息：
     *   result = alpha × generated + (1 - alpha) × original
     *
     * @param generated 生成图像 (BGR uint8)
     * @param original  原始图像 (BGR uint8)
     * @param alpha     生成图像的权重（默认 0.7，范围 [0, 1]）
     * @return cv::Mat  融合后的图像
     */
    cv::Mat ColorBlend(const cv::Mat& generated, const cv::Mat& original,
                       float alpha = 0.7f);

    /**
     * @brief 完整后处理管道
     *
     * 依次执行锐化和色彩融合。
     *
     * @param fused_image   已融合的图像
     * @param original_face 原始人脸图像（用于色彩参考）
     * @param do_sharpen    是否执行锐化（默认 true）
     * @param do_color_blend 是否执行色彩融合（默认 true）
     * @return cv::Mat      后处理完成的图像
     */
    cv::Mat PostProcess(const cv::Mat& fused_image,
                        const cv::Mat& original_face,
                        bool do_sharpen = true,
                        bool do_color_blend = true);

    // ==================== 全流程管道 ====================

    /**
     * @brief 完整输出处理管道
     *
     * 一键式处理：格式转换 → 逆变换 → 融合 → 后处理
     *
     * @param model_output 模型输出的 ncnn::Mat
     * @param original_face 原始人脸图像 (BGR uint8)
     * @param face_mask     口唇区域遮罩 (CV_32FC1)
     * @param M_inv         逆仿射变换矩阵
     * @return cv::Mat      最终处理完成的图像
     */
    cv::Mat Process(const ncnn::Mat& model_output,
                    const cv::Mat& original_face,
                    const cv::Mat& face_mask,
                    const cv::Mat& M_inv);

    /**
     * @brief 完整输出处理管道（ROI 加速版）
     *
     * 与 Process 功能一致，但逆变换/融合/锐化/色彩混合仅在人脸 ROI
     * 内执行，最后将 ROI 贴回原图。大图（如 1920×1384）下渲染耗时
     * 与全图处理相比可降低一个数量级。
     *
     * ROI 由 M_inv 将 96×96 对齐人脸四角投影到原图坐标后取包围盒，
     * 再向外扩展 margin_ratio 得到；face_rect 提供额外约束（可为空）。
     *
     * @param model_output 模型输出的 ncnn::Mat
     * @param original_face 原始人脸图像 (BGR uint8)
     * @param face_mask     口唇区域遮罩 (CV_32FC1，96×96 对齐空间或全图)
     * @param M_inv         逆仿射变换矩阵
     * @param face_rect     人脸在原图中的矩形（可空，仅作 ROI 参考）
     * @param margin_ratio  ROI 外扩比例（默认 0.25）
     * @return cv::Mat      最终处理完成的图像
     */
    cv::Mat ProcessROI(const ncnn::Mat& model_output,
                       const cv::Mat& original_face,
                       const cv::Mat& face_mask,
                       const cv::Mat& M_inv,
                       const cv::Rect& face_rect = cv::Rect(),
                       float margin_ratio = 0.25f);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace model
}  // namespace digital_human
