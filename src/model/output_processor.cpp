#include "model/output_processor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstring>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace digital_human {
namespace model {

// ============================================================================
// Impl 结构体：封装 OutputProcessor 的所有内部实现细节
// ============================================================================
struct OutputProcessor::Impl {
    // ---- 默认参数 ----
    static constexpr int kDefaultFaceW = 96;   ///< 默认人脸宽度
    static constexpr int kDefaultFaceH = 96;   ///< 默认人脸高度

    // blendLinear 只需要一张单通道逆遮罩。缓存该图像可避免每帧分配，
    // 同时不会为早期手动 float 融合路径保留未使用的全尺寸缓冲。
    cv::Mat inv_mask_1ch_buf_;  ///< CV_32FC1, blendLinear 的背景权重

    void EnsureInverseMaskBuffer(const cv::Size& size) {
        inv_mask_1ch_buf_.create(size, CV_32FC1);
    }

    // ========================================================================
    // ncnn::Mat → cv::Mat 转换
    // ========================================================================

    /**
     * @brief 将 ncnn::Mat (RGB float) 转换为 cv::Mat (BGR uint8)
     *
     * 处理步骤：
     *   1. 从 ncnn::Mat 的三个 float 通道读取 RGB 数据
     *   2. 值域从 [0,1] 缩放到 [0,255] 并 clamp
     *   3. RGB 顺序转 BGR 顺序（OpenCV 默认格式）
     *   4. 返回 CV_8UC3 图像
     *
     * @param src ncnn::Mat, 形状 (w, h, 3), float, RGB
     * @return cv::Mat CV_8UC3, BGR
     */
    cv::Mat ncnnToCvMat(const ncnn::Mat& src) {
        if (src.empty() || src.c != 3) {
            std::cerr << "[OutputProcessor] ncnnToCvMat: 输入无效 (empty="
                      << src.empty() << ", c=" << src.c << ")" << std::endl;
            return cv::Mat();
        }

        int w = src.w;
        int h = src.h;

        // 创建 CV_8UC3 图像
        cv::Mat dst(h, w, CV_8UC3);

        // 从 ncnn::Mat 的 RGB float 通道读取，写入 BGR uint8
        const float* ch_r = src.channel(0);  // R 通道
        const float* ch_g = src.channel(1);  // G 通道
        const float* ch_b = src.channel(2);  // B 通道

        for (int y = 0; y < h; ++y) {
            cv::Vec3b* row = dst.ptr<cv::Vec3b>(y);
            for (int x = 0; x < w; ++x) {
                int idx = y * w + x;
                // 值域 [0,1] → [0,255]，同时处理可能的越界值
                float b = std::clamp(ch_b[idx] * 255.0f, 0.0f, 255.0f);
                float g = std::clamp(ch_g[idx] * 255.0f, 0.0f, 255.0f);
                float r = std::clamp(ch_r[idx] * 255.0f, 0.0f, 255.0f);
                row[x] = cv::Vec3b(
                    static_cast<uchar>(b),
                    static_cast<uchar>(g),
                    static_cast<uchar>(r)
                );
            }
        }

        return dst;
    }

    /**
     * @brief 从模型输出中提取有效人脸区域
     *
     * 模型输出可能是多帧拼接的宽图像（如 448×96），
     * 此方法取输出图像的中心区域（face_w × face_h）。
     *
     * @param src    模型输出转换后的 BGR 图像
     * @param face_w 目标人脸宽度
     * @param face_h 目标人脸高度
     * @return cv::Mat 裁剪后的图像
     */
    cv::Mat extractFaceRegion(const cv::Mat& src, int face_w, int face_h) {
        if (src.empty()) {
            return cv::Mat();
        }

        // 如果尺寸匹配，直接返回（共享引用计数，调用方需要独立数据时自己 clone）
        if (src.cols == face_w && src.rows == face_h) {
            return src;
        }

        // 如果输出比目标大，从中心裁剪
        if (src.cols >= face_w && src.rows >= face_h) {
            int x = (src.cols - face_w) / 2;
            int y = (src.rows - face_h) / 2;
            return src(cv::Rect(x, y, face_w, face_h)).clone();
        }

        // 如果输出比目标小，缩放到目标尺寸
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(face_w, face_h), 0, 0, cv::INTER_LINEAR);
        return resized;
    }

    cv::Mat normalizeAlphaMask(const cv::Mat& mask) {
        if (mask.empty()) {
            return cv::Mat();
        }

        cv::Mat single_channel;
        if (mask.channels() == 1) {
            single_channel = mask;
        } else if (mask.channels() == 3) {
            cv::cvtColor(mask, single_channel, cv::COLOR_BGR2GRAY);
        } else if (mask.channels() == 4) {
            cv::cvtColor(mask, single_channel, cv::COLOR_BGRA2GRAY);
        } else {
            cv::extractChannel(mask, single_channel, 0);
        }

        cv::Mat mask_f;
        if (single_channel.depth() == CV_32F || single_channel.depth() == CV_64F) {
            single_channel.convertTo(mask_f, CV_32FC1);
        } else {
            single_channel.convertTo(mask_f, CV_32FC1, 1.0 / 255.0);
        }

        cv::threshold(mask_f, mask_f, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(mask_f, mask_f, 0.0, 0.0, cv::THRESH_TOZERO);
        return mask_f;
    }

    // ========================================================================
    // 逆变换
    // ========================================================================

    /**
     * @brief 使用逆仿射变换映射回原始坐标系
     *
     * @param face    对齐空间的人脸图像
     * @param M_inv   2×3 逆仿射矩阵
     * @param size    原始图像尺寸
     * @return cv::Mat
     */
    cv::Mat doInverseTransform(const cv::Mat& face, const cv::Mat& M_inv,
                               const cv::Size& size) {
        if (face.empty() || M_inv.empty()) {
            std::cerr << "[OutputProcessor] 逆变换失败：输入为空" << std::endl;
            return cv::Mat();
        }
        if (M_inv.rows != 2 || M_inv.cols != 3) {
            std::cerr << "[OutputProcessor] 逆变换失败：M_inv 必须是 2×3 矩阵"
                      << std::endl;
            return cv::Mat();
        }

        cv::Mat dst(size, CV_8UC3, cv::Scalar(0, 0, 0));  // 预先清零黑色背景
        cv::warpAffine(face, dst, M_inv, size,
                       cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
        return dst;
    }

    cv::Mat doInverseTransformMask(const cv::Mat& mask, const cv::Mat& M_inv,
                                   const cv::Size& size) {
        cv::Mat mask_f = normalizeAlphaMask(mask);
        if (mask_f.empty() || M_inv.empty()) {
            return cv::Mat();
        }
        if (M_inv.rows != 2 || M_inv.cols != 3) {
            return cv::Mat();
        }

        cv::Mat warped(size, CV_32FC1, cv::Scalar(0.0f));
        cv::warpAffine(mask_f, warped, M_inv, size,
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       cv::Scalar(0.0f));
        cv::threshold(warped, warped, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(warped, warped, 0.0, 0.0, cv::THRESH_TOZERO);

        // 在原图空间追加一次高斯模糊:弥补 warpAffine INTER_LINEAR 对 96 空间
        // 渐变的收窄 (实测 4-17px),直接控制原图空间渐变宽度到 ~30px,
        // 让边界色差 (~24) 在足够宽度内自然衰减,消除嘴部边缘割裂
        cv::GaussianBlur(warped, warped, cv::Size(21, 21), 0);
        // blur 可能引入微小越界,二次裁剪保证 [0,1]
        cv::threshold(warped, warped, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(warped, warped, 0.0, 0.0, cv::THRESH_TOZERO);
        return warped;
    }

    cv::Mat prepareFusionMask(const cv::Mat& mask, const cv::Size& aligned_size,
                              const cv::Size& original_size,
                              const cv::Mat& M_inv) {
        if (mask.empty()) {
            return cv::Mat();
        }
        if (mask.size() == original_size) {
            // 原图空间 mask: 归一化后追加一次 blur,确保渐变带足够宽
            // (generateMouthMask 默认 blur_sigma=35 已较大,但各调用路径
            //  可能传不同值,这里统一兜底,消除嘴部边缘割裂)
            cv::Mat m = normalizeAlphaMask(mask);
            cv::GaussianBlur(m, m, cv::Size(21, 21), 0);
            cv::threshold(m, m, 1.0, 1.0, cv::THRESH_TRUNC);
            cv::threshold(m, m, 0.0, 0.0, cv::THRESH_TOZERO);
            return m;
        }
        if (mask.size() == aligned_size) {
            return doInverseTransformMask(mask, M_inv, original_size);
        }

        cv::Mat resized;
        cv::resize(mask, resized, original_size, 0, 0, cv::INTER_LINEAR);
        return normalizeAlphaMask(resized);
    }

    // ========================================================================
    // 人脸融合
    // ========================================================================

    /**
     * @brief 使用 alpha 遮罩进行图像融合（blendLinear 单通道加速）
     *
     * 公式：result = mask × generated + (1 - mask) × original
     * 使用 cv::blendLinear 替代手动 float 转换链（convertTo×2 + merge +
     * subtract + multiply×2 + add + convertTo ≈ 8 次全图遍历），
     * 在 uint8 输入上一次完成，融合耗时降低约 4 倍。
     *
     * @param original 原始图像 (CV_8UC3)
     * @param generated 生成图像 (CV_8UC3)
     * @param mask 遮罩 (CV_32FC1, [0,1])
     * @return cv::Mat
     */
    cv::Mat doFaceFusion(const cv::Mat& original, const cv::Mat& generated,
                         const cv::Mat& mask) {
        if (original.empty() || generated.empty() || mask.empty()) {
            std::cerr << "[OutputProcessor] 融合失败：输入为空" << std::endl;
            return cv::Mat();
        }

        // 确保所有输入尺寸一致
        cv::Mat mask_resized, gen_resized;
        if (mask.size() != original.size()) {
            cv::resize(mask, mask_resized, original.size(), 0, 0, cv::INTER_LINEAR);
        } else {
            mask_resized = mask;
        }
        if (generated.size() != original.size()) {
            cv::resize(generated, gen_resized, original.size(), 0, 0, cv::INTER_LINEAR);
        } else {
            gen_resized = generated;
        }

        // 遮罩归一化到 CV_32FC1 [0,1]
        cv::Mat mask_f = normalizeAlphaMask(mask_resized);
        if (mask_f.empty()) {
            return cv::Mat();
        }

        // result = gen * mask + orig * (1 - mask)
        // blendLinear 要求权重为 CV_32FC1 单通道 float，内部对 8UC3
        // 有 SIMD 优化路径，一次调用替代整个 float 转换链
        EnsureInverseMaskBuffer(original.size());
        cv::subtract(cv::Scalar(1.0f), mask_f, inv_mask_1ch_buf_);

        cv::Mat result;
        cv::blendLinear(gen_resized, original, mask_f, inv_mask_1ch_buf_,
                        result);
        return result;
    }

    // ========================================================================
    // 后处理
    // ========================================================================

    /**
     * @brief USM 非锐化遮罩锐化（uint8 域，避免 float 转换开销）
     *
     * 公式：sharpened = (1 + strength) × image - strength × blurred
     *       = image + strength × (image - blurred)
     * addWeighted 的 saturate_cast 语义与旧 float 路径 clamp 等价，
     * 但全程 uint8 SIMD，大图下快 3-5 倍。
     *
     * @param image    输入图像 (CV_8UC3)
     * @param strength 锐化强度
     * @return cv::Mat
     */
    cv::Mat doSharpen(const cv::Mat& image, float strength) {
        if (image.empty()) {
            return cv::Mat();
        }

        float s = std::clamp(strength, 0.0f, 3.0f);
        if (s < 0.01f) {
            return image;  // 直接共享引用计数，无需 clone
        }

        // 高斯模糊（uint8 域，OpenCV SIMD 优化路径）
        cv::Mat blurred;
        constexpr int ksize = 5;  // 固定奇数，避免死代码
        cv::GaussianBlur(image, blurred, cv::Size(ksize, ksize), 1.5);

        // uint8 域 USM：result = (1+s)*img - s*blur（自动饱和截断）
        cv::Mat result;
        cv::addWeighted(image, 1.0 + s, blurred, -s, 0.0, result);
        return result;
    }

    /**
     * @brief 色彩融合
     *
     * @param gen     生成图像 (CV_8UC3)
     * @param orig    原始图像 (CV_8UC3)
     * @param alpha   生成图像权重
     * @return cv::Mat
     */
    cv::Mat doColorBlend(const cv::Mat& gen, const cv::Mat& orig, float alpha) {
        if (gen.empty() || orig.empty()) {
            return cv::Mat();
        }

        float a = std::clamp(alpha, 0.0f, 1.0f);

        // 确保尺寸一致
        cv::Mat gen_resized;
        if (gen.size() != orig.size()) {
            cv::resize(gen, gen_resized, orig.size(), 0, 0, cv::INTER_LINEAR);
        } else {
            gen_resized = gen;
        }

        cv::Mat result;
        cv::addWeighted(gen_resized, a, orig, 1.0f - a, 0.0, result);
        return result;
    }
};

// ============================================================================
// OutputProcessor 公有接口实现
// ============================================================================

OutputProcessor::OutputProcessor()
    : impl_(std::make_unique<Impl>()) {}

OutputProcessor::~OutputProcessor() = default;

OutputProcessor::OutputProcessor(OutputProcessor&&) noexcept = default;

OutputProcessor& OutputProcessor::operator=(OutputProcessor&&) noexcept = default;

// ==================== 格式转换 ====================

cv::Mat OutputProcessor::OutputToMat(const ncnn::Mat& model_output,
                                      int face_w, int face_h) {
    if (model_output.empty()) {
        std::cerr << "[OutputProcessor] OutputToMat: ncnn::Mat 为空" << std::endl;
        return cv::Mat();
    }
    if (model_output.c != 3) {
        std::cerr << "[OutputProcessor] OutputToMat: 通道数不为 3 (实际 c="
                  << model_output.c << ")" << std::endl;
        return cv::Mat();
    }
    if (face_w <= 0 || face_h <= 0) {
        std::cerr << "[OutputProcessor] OutputToMat: 无效的 face_w/face_h"
                  << std::endl;
        return cv::Mat();
    }

    // 1. ncnn → cv::Mat
    cv::Mat full = impl_->ncnnToCvMat(model_output);
    if (full.empty()) {
        return cv::Mat();
    }

    // 2. 提取人脸区域
    cv::Mat face = impl_->extractFaceRegion(full, face_w, face_h);
    return face;
}

// ==================== 逆变换 ====================

cv::Mat OutputProcessor::InverseTransform(const cv::Mat& processed_face,
                                           const cv::Mat& M_inv,
                                           const cv::Size& original_size) {
    if (processed_face.empty()) {
        std::cerr << "[OutputProcessor] InverseTransform: 输入图像为空" << std::endl;
        return cv::Mat();
    }
    if (M_inv.empty()) {
        std::cerr << "[OutputProcessor] InverseTransform: M_inv 为空" << std::endl;
        return cv::Mat();
    }
    if (original_size.width <= 0 || original_size.height <= 0) {
        std::cerr << "[OutputProcessor] InverseTransform: 无效的原始图像尺寸"
                  << std::endl;
        return cv::Mat();
    }

    return impl_->doInverseTransform(processed_face, M_inv, original_size);
}

// ==================== 人脸融合 ====================

cv::Mat OutputProcessor::FaceFusion(const cv::Mat& original_image,
                                     const cv::Mat& generated_face,
                                     const cv::Mat& face_mask) {
    if (original_image.empty()) {
        std::cerr << "[OutputProcessor] FaceFusion: 原始图像为空" << std::endl;
        return cv::Mat();
    }
    if (generated_face.empty()) {
        std::cerr << "[OutputProcessor] FaceFusion: 生成图像为空" << std::endl;
        return cv::Mat();
    }
    if (face_mask.empty()) {
        std::cerr << "[OutputProcessor] FaceFusion: 遮罩为空" << std::endl;
        return cv::Mat();
    }
    if (original_image.channels() != 3) {
        std::cerr << "[OutputProcessor] FaceFusion: 原始图像通道数不为 3"
                  << std::endl;
        return cv::Mat();
    }

    return impl_->doFaceFusion(original_image, generated_face, face_mask);
}

// ==================== 后处理 ====================

cv::Mat OutputProcessor::Sharpen(const cv::Mat& image, float strength) {
    return impl_->doSharpen(image, strength);
}

cv::Mat OutputProcessor::ColorBlend(const cv::Mat& generated,
                                     const cv::Mat& original,
                                     float alpha) {
    return impl_->doColorBlend(generated, original, alpha);
}

cv::Mat OutputProcessor::PostProcess(const cv::Mat& fused_image,
                                      const cv::Mat& original_face,
                                      bool do_sharpen,
                                      bool do_color_blend) {
    if (fused_image.empty()) {
        std::cerr << "[OutputProcessor] PostProcess: 融合图像为空" << std::endl;
        return cv::Mat();
    }

    cv::Mat result = fused_image;  // 浅拷贝（引用计数共享）

    if (do_sharpen) {
        result = impl_->doSharpen(result, 1.0f);
    }

    if (do_color_blend && !original_face.empty()) {
        result = impl_->doColorBlend(result, original_face, 0.7f);
    }

    return result;  // RVO 确保无额外拷贝
}

// ==================== 全流程管道 ====================

cv::Mat OutputProcessor::Process(const ncnn::Mat& model_output,
                                  const cv::Mat& original_face,
                                  const cv::Mat& face_mask,
                                  const cv::Mat& M_inv) {
    // 1. 格式转换 + 提取人脸
    cv::Mat face = OutputToMat(model_output, Impl::kDefaultFaceW, Impl::kDefaultFaceH);
    if (face.empty()) {
        std::cerr << "[OutputProcessor] Process: 格式转换失败" << std::endl;
        return cv::Mat();
    }

    // 2. 锐化（在对齐空间执行：无黑底污染，blur 核不会跨越 mask 边界）
    cv::Mat face_sharpened = impl_->doSharpen(face, 1.0f);

    // 3. 逆变换
    cv::Mat warped = InverseTransform(face_sharpened, M_inv, original_face.size());
    if (warped.empty()) {
        std::cerr << "[OutputProcessor] Process: 逆变换失败" << std::endl;
        return cv::Mat();
    }

    // 4. 人脸融合
    cv::Mat fusion_mask = impl_->prepareFusionMask(
        face_mask, face.size(), original_face.size(), M_inv);
    if (fusion_mask.empty()) {
        std::cerr << "[OutputProcessor] Process: mask prepare failed" << std::endl;
        return cv::Mat();
    }

    cv::Mat fused = FaceFusion(original_face, warped, fusion_mask);
    if (fused.empty()) {
        std::cerr << "[OutputProcessor] Process: 融合失败" << std::endl;
        return cv::Mat();
    }

    // 5. 色彩混合（在 fused 上执行；mask 外 fused==orig，
    //    0.7*orig+0.3*orig==orig，不会产生边界跳变）
    cv::Mat result = impl_->doColorBlend(fused, original_face, 0.7f);
    return result;
}

// ==================== 全流程管道（ROI 加速版） ====================

cv::Mat OutputProcessor::ProcessROI(const ncnn::Mat& model_output,
                                     const cv::Mat& original_face,
                                     const cv::Mat& face_mask,
                                     const cv::Mat& M_inv,
                                     const cv::Rect& face_rect,
                                     float margin_ratio) {
    if (original_face.empty() || M_inv.empty()) {
        std::cerr << "[OutputProcessor] ProcessROI: 输入为空" << std::endl;
        return cv::Mat();
    }

    // 1. 格式转换 + 提取人脸
    cv::Mat face = OutputToMat(model_output,
                               Impl::kDefaultFaceW, Impl::kDefaultFaceH);
    if (face.empty()) {
        std::cerr << "[OutputProcessor] ProcessROI: 格式转换失败" << std::endl;
        return cv::Mat();
    }

    // 1.5 锐化（在对齐空间执行：无黑底污染，blur 核不会跨越 mask 边界）
    cv::Mat face_sharpened = impl_->doSharpen(face, 1.0f);

    // 2. 由 M_inv 投影对齐人脸四角 → 原图包围盒（+margin，与 face_rect 求并集）
    cv::Mat M64;
    M_inv.convertTo(M64, CV_64F);
    auto project = [&](double x, double y) {
        return cv::Point2d(
            M64.at<double>(0, 0) * x + M64.at<double>(0, 1) * y
                + M64.at<double>(0, 2),
            M64.at<double>(1, 0) * x + M64.at<double>(1, 1) * y
                + M64.at<double>(1, 2));
    };
    std::vector<cv::Point2d> corners = {
        project(0, 0), project(face.cols, 0),
        project(face.cols, face.rows), project(0, face.rows)
    };
    double x0 = corners[0].x, y0 = corners[0].y;
    double x1 = corners[0].x, y1 = corners[0].y;
    for (const auto& p : corners) {
        x0 = std::min(x0, p.x);  y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x);  y1 = std::max(y1, p.y);
    }
    int mx = static_cast<int>((x1 - x0) * margin_ratio);
    int my = static_cast<int>((y1 - y0) * margin_ratio);
    cv::Rect roi(static_cast<int>(std::floor(x0)) - mx,
                 static_cast<int>(std::floor(y0)) - my,
                 static_cast<int>(std::ceil(x1 - x0)) + 2 * mx,
                 static_cast<int>(std::ceil(y1 - y0)) + 2 * my);
    if (face_rect.area() > 0) {
        roi |= face_rect;
    }
    roi &= cv::Rect(0, 0, original_face.cols, original_face.rows);
    if (roi.empty() || roi.size() == original_face.size()) {
        // ROI 无效或等同全图 → 回退全图处理
        return Process(model_output, original_face, face_mask, M_inv);
    }

    // 3. 调整 M_inv → M_roi（平移分量减去 ROI 原点）
    cv::Mat M_roi = M64.clone();
    M_roi.at<double>(0, 2) -= roi.x;
    M_roi.at<double>(1, 2) -= roi.y;

    // 4. 逆变换到 ROI 画布（用锐化后的 face）
    cv::Mat warped = impl_->doInverseTransform(face_sharpened, M_roi, roi.size());
    if (warped.empty()) {
        std::cerr << "[OutputProcessor] ProcessROI: 逆变换失败" << std::endl;
        return cv::Mat();
    }

    // 5. 遮罩 → ROI 空间（全图遮罩先裁剪，96×96 对齐遮罩用 M_roi 逆变换）
    cv::Mat fusion_mask;
    if (face_mask.size() == original_face.size()) {
        cv::Mat cropped(face_mask, roi);
        fusion_mask = impl_->prepareFusionMask(
            cropped, face.size(), roi.size(), M_roi);
    } else {
        fusion_mask = impl_->prepareFusionMask(
            face_mask, face.size(), roi.size(), M_roi);
    }
    if (fusion_mask.empty()) {
        std::cerr << "[OutputProcessor] ProcessROI: mask prepare failed"
                  << std::endl;
        return cv::Mat();
    }

    // 6. ROI 内融合
    cv::Mat orig_roi = original_face(roi);
    cv::Mat fused = FaceFusion(orig_roi, warped, fusion_mask);
    if (fused.empty()) {
        std::cerr << "[OutputProcessor] ProcessROI: 融合失败" << std::endl;
        return cv::Mat();
    }

    // 7. 色彩混合（在 fused 上执行；mask 外 fused==orig_roi，
    //    0.7*orig+0.3*orig==orig，贴回后与 ROI 外像素一致，无矩形 seam）
    cv::Mat blended = impl_->doColorBlend(fused, orig_roi, 0.7f);
    if (blended.empty()) {
        return cv::Mat();
    }

    // 8. 贴回原图
    cv::Mat result = original_face.clone();
    blended.copyTo(result(roi));
    return result;
}

}  // namespace model
}  // namespace digital_human
