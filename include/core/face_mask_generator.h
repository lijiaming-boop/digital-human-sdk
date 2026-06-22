#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace DigitalHuman {
namespace Core {

class FaceMaskGenerator {
public:
    FaceMaskGenerator();
    ~FaceMaskGenerator();

    FaceMaskGenerator(FaceMaskGenerator&&) noexcept;
    FaceMaskGenerator& operator=(FaceMaskGenerator&&) noexcept;

    FaceMaskGenerator(const FaceMaskGenerator&) = delete;
    FaceMaskGenerator& operator=(const FaceMaskGenerator&) = delete;

    /**
     * @brief 生成嘴部掩码
     * @param image_size 原始图像大小，因为生成的掩码要与原图一致
     * @param landmarks 68个关键点
     * @param dilate_radius 扩展半径（像素），默认 0 表示不扩展
     * @param blur_sigma 羽化程度（高斯模糊核大小），必须是奇数
     * @return cv::Mat 单通道掩码（CV_32FC1，范围 0.0-1.0）
     */
    cv::Mat generateMouthMask(const cv::Size& image_size,
                              const std::vector<cv::Point>& landmarks,
                              int dilate_radius = 5,
                              int blur_sigma = 15);

    /**
     * @brief 生成 96x96 对齐空间下的精细嘴部 alpha mask
     * @param landmarks_96 已经映射到 96x96 空间的 68 点
     * @return CV_32FC1，范围 0.0~1.0
     */
    cv::Mat generatePreciseMouthAlphaMask96(const std::vector<cv::Point2f>& landmarks_96);

    /**
     * @brief 把单通道 alpha mask 转成三通道 mask.
     * 图像融合时，BGR 图像有 3 个通道，所以 alpha mask 也需要扩展成 3 通道，方便逐通道相乘
     */
    cv::Mat to3ChannelMask(const cv::Mat& alpha_mask) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} 
} 