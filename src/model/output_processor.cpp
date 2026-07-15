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

        // 如果尺寸匹配，直接返回
        if (src.cols == face_w && src.rows == face_h) {
            return src.clone();
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

        cv::Mat dst;
        cv::warpAffine(face, dst, M_inv, size,
                       cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
        return dst;
    }

    // ========================================================================
    // 人脸融合
    // ========================================================================

    /**
     * @brief 使用 alpha 遮罩进行图像融合
     *
     * 公式：result = mask × generated + (1 - mask) × original
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

        // 转 float 进行计算
        cv::Mat orig_f, gen_f;
        original.convertTo(orig_f, CV_32FC3, 1.0 / 255.0);
        gen_resized.convertTo(gen_f, CV_32FC3, 1.0 / 255.0);

        // 将遮罩转为 CV_32F（单通道 float）
        cv::Mat mask_f;
        if (mask_resized.type() != CV_32F) {
            mask_resized.convertTo(mask_f, CV_32F, 1.0 / 255.0);
        } else {
            mask_f = mask_resized;
        }

        // 逐像素 alpha blend: result = mask * gen + (1 - mask) * orig
        // 直接使用单通道 mask 与 3 通道图像逐元素相乘（OpenCV 自动广播）
        cv::Mat result_f(mask_f.size(), CV_32FC3, cv::Scalar(0, 0, 0));
        cv::Mat inv_mask(mask_f.size(), CV_32F);

        const int total = mask_f.rows * mask_f.cols;
        for (int i = 0; i < total; ++i) {
            float m = mask_f.at<float>(i);  // 遮罩值 [0,1]
            float im = 1.0f - m;            // 逆遮罩

            cv::Vec3f* g = gen_f.ptr<cv::Vec3f>() + i;
            cv::Vec3f* o = orig_f.ptr<cv::Vec3f>() + i;
            cv::Vec3f* r = result_f.ptr<cv::Vec3f>() + i;

            (*r)[0] = m * (*g)[0] + im * (*o)[0];  // B
            (*r)[1] = m * (*g)[1] + im * (*o)[1];  // G
            (*r)[2] = m * (*g)[2] + im * (*o)[2];  // R
        }

        // 转回 uint8
        cv::Mat result;
        result_f.convertTo(result, CV_8UC3, 255.0);
        return result;
    }

    // ========================================================================
    // 后处理
    // ========================================================================

    /**
     * @brief USM 非锐化遮罩锐化
     *
     * 公式：sharpened = image + strength × (image - blurred)
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
            return image.clone();
        }

        // 高斯模糊
        cv::Mat blurred;
        int ksize = 5;
        if (ksize % 2 == 0) ksize++;  // 确保奇数
        cv::GaussianBlur(image, blurred, cv::Size(ksize, ksize), 1.5);

        // float 域计算
        cv::Mat img_f, blur_f;
        image.convertTo(img_f, CV_32FC3, 1.0 / 255.0);
        blurred.convertTo(blur_f, CV_32FC3, 1.0 / 255.0);

        // sharpened = image + strength × (image - blurred)
        cv::Mat diff;
        cv::subtract(img_f, blur_f, diff);
        cv::Mat sharp_f;
        cv::addWeighted(img_f, 1.0, diff, s, 0.0, sharp_f);

        // clamp [0,1] 后转回 uint8
        cv::Mat result;
        cv::threshold(sharp_f, sharp_f, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(sharp_f, sharp_f, 0.0, 0.0, cv::THRESH_TOZERO);
        sharp_f.convertTo(result, CV_8UC3, 255.0);
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

    cv::Mat result = fused_image.clone();

    if (do_sharpen) {
        result = impl_->doSharpen(result, 1.0f);
    }

    if (do_color_blend && !original_face.empty()) {
        result = impl_->doColorBlend(result, original_face, 0.7f);
    }

    return result;
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

    // 2. 逆变换
    cv::Mat warped = InverseTransform(face, M_inv, original_face.size());
    if (warped.empty()) {
        std::cerr << "[OutputProcessor] Process: 逆变换失败" << std::endl;
        return cv::Mat();
    }

    // 3. 人脸融合
    cv::Mat fused = FaceFusion(original_face, warped, face_mask);
    if (fused.empty()) {
        std::cerr << "[OutputProcessor] Process: 融合失败" << std::endl;
        return cv::Mat();
    }

    // 4. 后处理
    cv::Mat result = PostProcess(fused, original_face);
    return result;
}

}  // namespace model
}  // namespace digital_human
