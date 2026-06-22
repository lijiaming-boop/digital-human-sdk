#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/imgproc.hpp>

#include "core/face_mask_generator.h"

namespace DigitalHuman {
namespace Core {

struct FaceMaskGenerator::Impl {

cv::Mat generateMouthMask(const cv::Size& image_size,
                                             const std::vector<cv::Point>& landmarks,
                                             int dilate_radius,
                                             int blur_sigma)
{
    // 1. 先初始化为空白图像
    cv::Mat mask = cv::Mat::zeros(image_size, CV_8UC1);

    // 2. 检查关键点个数
    if (landmarks.size() < 68) {
        std::cerr << "[FaceMaskGenerator] Error: Invalid landmarks count: "
                  << landmarks.size() << std::endl;
        return mask;
    }

    // 3. 提取嘴唇外轮廓关键点 48~67
    std::vector<cv::Point> mouth_points;
    mouth_points.reserve(20);
    for (int i = 48; i <= 67; ++i) {
        mouth_points.push_back(landmarks[i]);
    }

    // 4. 绘制嘴部区域，这里是一个闭合多边形
    const cv::Point* ppt[1] = { mouth_points.data() };
    int npt[] = { static_cast<int>(mouth_points.size()) };
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255));

    // 5. 区域膨胀，增加嘴部覆盖范围
    if (dilate_radius > 0) {
        int k_size = dilate_radius * 2 + 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_size, k_size));
        cv::dilate(mask, mask, kernel);
    }

    // 6. 让边界柔和，无需锐边
    if (blur_sigma > 0) {
        if (blur_sigma % 2 == 0) {
            blur_sigma++;
        }
        cv::GaussianBlur(mask, mask, cv::Size(blur_sigma, blur_sigma), 0);
    }

    // 7. 归一化成浮点型 (0~255 -> 0~1.0)
    cv::Mat mask_float;
    mask.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);
    return mask_float;
}

cv::Mat generatePreciseMouthAlphaMask96(const std::vector<cv::Point2f>& landmarks_96)
{
    const cv::Size mask_size(96, 96);
    cv::Mat mask_u8 = cv::Mat::zeros(mask_size, CV_8UC1);

    if (landmarks_96.size() < 68) {
        std::cerr << "[FaceMaskGenerator] Invalid 96 landmarks count: "
                  << landmarks_96.size() << std::endl;

        cv::Mat empty_mask;
        mask_u8.convertTo(empty_mask, CV_32FC1, 1.0 / 255.0);
        return empty_mask;
    }

    std::vector<cv::Point> mouth_points;
    mouth_points.reserve(20);

    // 初始化嘴部坐标范围
    float mouth_x_min = landmarks_96[48].x;
    float mouth_x_max = landmarks_96[48].x;
    float mouth_y_min = landmarks_96[48].y;
    float mouth_y_max = landmarks_96[48].y;

    // 48~67: 嘴部轮廓关键点
    for (int i = 48; i <= 67; ++i) {
        float x = landmarks_96[i].x;
        float y = landmarks_96[i].y;

        mouth_points.emplace_back(
            static_cast<int>(std::round(x)),
            static_cast<int>(std::round(y))
        );

        mouth_x_min = std::min(mouth_x_min, x);
        mouth_x_max = std::max(mouth_x_max, x);
        mouth_y_min = std::min(mouth_y_min, y);
        mouth_y_max = std::max(mouth_y_max, y);
    }

    // 生成凸包区域，用于更稳定地填充嘴部
    std::vector<cv::Point> hull;
    cv::convexHull(mouth_points, hull);

    if (hull.size() < 3) {
        cv::Mat empty_mask;
        mask_u8.convertTo(empty_mask, CV_32FC1, 1.0 / 255.0);
        return empty_mask;
    }

    cv::fillConvexPoly(mask_u8, hull, cv::Scalar(255));

    // 适当扩张，确保遮罩能覆盖嘴唇周边区域
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(5, 5)
    );
    cv::dilate(mask_u8, mask_u8, kernel, cv::Point(-1, -1), 1);

    // 计算嘴巴区域的范围
    const float mouth_w = std::max(1.0f, mouth_x_max - mouth_x_min);
    const float mouth_h = std::max(1.0f, mouth_y_max - mouth_y_min);

    int x_left   = static_cast<int>(std::floor(mouth_x_min - 0.55f * mouth_w));
    int x_right  = static_cast<int>(std::ceil (mouth_x_max + 0.55f * mouth_w));
    int y_top    = static_cast<int>(std::floor(mouth_y_min - 0.90f * mouth_h));
    int y_bottom  = static_cast<int>(std::ceil (mouth_y_max + 1.00f * mouth_h));

    // 裁剪至 96x96 图像范围内
    x_left   = std::max(0, x_left);
    x_right  = std::min(95, x_right);
    y_top    = std::max(0, y_top);
    y_bottom = std::min(95, y_bottom);

    // 如果嘴巴区域被完全裁剪掉了（极端退化的关键点），返回空掩码
    if (x_left >= x_right || y_top >= y_bottom) {
        cv::Mat empty_mask;
        mask_u8.convertTo(empty_mask, CV_32FC1, 1.0 / 255.0);
        return empty_mask;
    }

    // 清除嘴部区域外的 mask
    if (x_left > 0) {
        mask_u8.colRange(0, x_left).setTo(0);
    }
    if (x_right + 1 < 96) {
        mask_u8.colRange(x_right + 1, 96).setTo(0);
    }
    if (y_top > 0) {
        mask_u8.rowRange(0, y_top).setTo(0);
    }
    if (y_bottom + 1 < 96) {
        mask_u8.rowRange(y_bottom + 1, 96).setTo(0);
    }

    // 清除边缘噪声，防止 mask 贴边
    const int border = 2;
    mask_u8.rowRange(0, border).setTo(0);
    mask_u8.rowRange(96 - border, 96).setTo(0);
    mask_u8.colRange(0, border).setTo(0);
    mask_u8.colRange(96 - border, 96).setTo(0);

    cv::GaussianBlur(mask_u8, mask_u8, cv::Size(13, 13), 0);

    cv::Mat mask_float;
    mask_u8.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);
    return mask_float;
}
cv::Mat to3ChannelMask(const cv::Mat& alpha_mask) const {
    if (alpha_mask.empty()) {
        return cv::Mat();
    }

    cv::Mat alpha_f;

    // 如果输入是 8 位图，先归一化到 0.0-1.0。
    if (alpha_mask.type() != CV_32FC1) {
        alpha_mask.convertTo(alpha_f, CV_32FC1, 1.0 / 255.0);
    } else {
        alpha_f = alpha_mask;
    }

    std::vector<cv::Mat> channels(3, alpha_f);

    cv::Mat mask_3c;
    cv::merge(channels, mask_3c);

    return mask_3c;
}
};

FaceMaskGenerator::FaceMaskGenerator() : pImpl(std::make_unique<Impl>()) {}
FaceMaskGenerator::~FaceMaskGenerator() = default;

FaceMaskGenerator::FaceMaskGenerator(FaceMaskGenerator&&) noexcept = default;
FaceMaskGenerator& FaceMaskGenerator::operator=(FaceMaskGenerator&&) noexcept = default;

// 外部调用接口
cv::Mat FaceMaskGenerator::generateMouthMask(const cv::Size& image_size,
                                             const std::vector<cv::Point>& landmarks,
                                             int dilate_radius,
                                             int blur_kernel_size) {
    return pImpl->generateMouthMask(image_size, landmarks, dilate_radius,
                                    blur_kernel_size);
}

cv::Mat FaceMaskGenerator::generatePreciseMouthAlphaMask96(
    const std::vector<cv::Point2f>& landmarks_96
) {
    return pImpl->generatePreciseMouthAlphaMask96(landmarks_96);
}

cv::Mat FaceMaskGenerator::to3ChannelMask(const cv::Mat& alpha_mask) const {
    return pImpl->to3ChannelMask(alpha_mask);
}

} // namespace Core
} // namespace DigitalHuman