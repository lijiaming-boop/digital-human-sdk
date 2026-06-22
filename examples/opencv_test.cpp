#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 绘制房子：画一个简单的 C++ 房子图形用于测试
void drawCphouse(cv::Mat img) {
    // 1. 绘制轮廓
    cv::line(img, cv::Point(0, 50), cv::Point(600, 550), cv::Scalar(50, 200, 50), 5);

    // 2. 绘制房子主体
    cv::Rect house_body(150, 300, 300, 250);
    cv::rectangle(img, house_body, cv::Scalar(100, 100, 100), -1);  // 填充灰色

    // 3. 绘制屋顶
    std::vector<cv::Point> roof_points;
    roof_points.push_back(cv::Point(150, 300));
    roof_points.push_back(cv::Point(300, 100));
    roof_points.push_back(cv::Point(470, 300));
    roof_points.push_back(cv::Point(150, 300));

    // 注意：fillPoly 的参数类型通常是 vector<vector<Point>>。
    // 但截图里写法是直接传 roof_points，这里按截图意图保留结构。
    cv::fillPoly(img, roof_points, cv::Scalar(50, 50, 200)); // 填充红色/紫色

    // 4. 写字（测试文字渲染）
    cv::putText(img, "C", cv::Point(275, 280), cv::FONT_HERSHEY_DUPLEX, 3.0,
                cv::Scalar(255, 0, 0), 3);
    cv::putText(img, "+", cv::Point(300, 450), cv::FONT_HERSHEY_DUPLEX, 4.0,
                cv::Scalar(0, 250, 0), 2);
}

int main() {
    std::cout << "\n====================================" << std::endl;
    std::cout << "   Digital Human SDK: OpenCV Test  " << std::endl;
    std::cout << "====================================\n" << std::endl;

    // 1. 验证 OpenCV 安装
    std::cout << "[1] Checking OpenCV Installation..." << std::endl;
    std::cout << "   -> OpenCV Version: " << CV_VERSION << std::endl;

    // 2. 生成基础测试图像（Image Write）
    std::cout << "[2] Generating test image in memory..." << std::endl;
    cv::Mat source_img = cv::Mat::zeros(600, 600, CV_8UC3);
    drawCphouse(source_img);

    std::string src_filename = "test_src.jpg";
    if (cv::imwrite(src_filename, source_img)) {
        std::cout << "   -> [SUCCESS] Generated and saved image to: "
                  << src_filename << std::endl;
    } else {
        std::cerr << "   -> [FAILED] Could not save initial image!"
                   << std::endl;
        return -1;
    }

    // 3. 验证图像加载（Image Load）
    std::cout << "[3] Testing imread (Loading image from disk)..."
              << std::endl;

    cv::Mat loaded_img = cv::imread(src_filename);
    if (loaded_img.empty()) {
        std::cerr << "   -> [FAILED] Could not open or find the image: "
                  << src_filename << std::endl;
        return -1;
    } else {
        std::cout << "   -> [SUCCESS] Image loaded!" << std::endl;
        std::cout << "      Resolution: " << loaded_img.cols << "x"
                  << loaded_img.rows << std::endl;
        std::cout << "      Channels: " << loaded_img.channels() << std::endl;
    }

    // 4. 验证颜色空间转换（Color Conversion）
    std::cout << "[4] Testing color space conversion (BGR -> Grayscale)..."
              << std::endl;

    cv::Mat gray_img;
    try {
        cv::cvtColor(loaded_img, gray_img, cv::COLOR_BGR2GRAY);
        std::cout << "   -> [SUCCESS] Converted to Grayscale!"
                  << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "   -> [FAILED] OpenCV Exception: " << e.what()
                   << std::endl;
        return -1;
    }

    // 5. 验证处理结果输出（Result Write）
    std::cout << "[5] Saving processed image..." << std::endl;
    std::string dst_filename = "test_result_gray.jpg";

    if (cv::imwrite(dst_filename, gray_img)) {
        std::cout << "   -> [SUCCESS] Saved grayscale image to: "
                  << dst_filename << std::endl;
    } else {
        std::cerr << "   -> [FAILED] Could not save grayscale image!"
                   << std::endl;
        return -1;
    }

    std::cout << "\n====================================" << std::endl;
    std::cout << "   ALL TESTS PASSED SUCCESSFULLY  " << std::endl;
    std::cout << "====================================\n" << std::endl;

    return 0;
}