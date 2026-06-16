#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "core/image_loader.h"

namespace fs = std::filesystem;
using namespace digital_human::core;

// 辅助函数：生成一张测试用的纯色图片并保存到磁盘
void createTestImage(const std::string& path, const cv::Scalar& color) {
    cv::Mat img(100, 100, CV_8UC3, color);
    cv::imwrite(path, img);
}

int main() {
    std::cout << "========== 开始测试 ImageLoader ==========" << std::endl;

    // 1. 准备测试数据
    std::string valid_path1 = "test_image_1.jpg";
    std::string valid_path2 = "test_image_2.jpg";
    std::string invalid_path = "non_existent_image.jpg";

    createTestImage(valid_path1, cv::Scalar(0, 0, 255)); // 红色图片
    createTestImage(valid_path2, cv::Scalar(0, 255, 0)); // 绿色图片

    // 实例化我们要测试的类
    ImageLoader loader;

    // ==========================================
    // 测试 1：从文件加载图片 (loadImageFromFile)
    // ==========================================
    std::cout << "\n[Test 1] 测试 loadImageFromFile..." << std::endl;
    try {
        cv::Mat img1 = loader.loadImageFromFile(valid_path1);
        std::cout << "  [成功] 读取真实文件: " << valid_path1 << " | 尺寸: " << img1.cols << "x" << img1.rows << std::endl;
        
        // 测试异常：加载不存在的文件
        loader.loadImageFromFile(invalid_path);
    } catch (const ImageLoaderException& e) {
        std::cout << "  [成功] 捕获预期异常 (文件不存在): " << e.what() << std::endl;
    }

    // ==========================================
    // 测试 2：从内存加载图片 (loadImageFromMemory)
    // ==========================================
    std::cout << "\n[Test 2] 测试 loadImageFromMemory..." << std::endl;
    try {
        // 先读取成二进制数据，模拟网络收到的字节流
        std::vector<uint8_t> buffer;
        cv::Mat temp_img = cv::imread(valid_path1);
        cv::imencode(".jpg", temp_img, buffer);

        // 从内存加载
        cv::Mat img_mem = loader.loadImageFromMemory(buffer);
        std::cout << "  [成功] 从内存成功解析图片 | 尺寸: " << img_mem.cols << "x" << img_mem.rows << std::endl;

        // 测试异常：加载空内存
        std::vector<uint8_t> empty_buffer;
        loader.loadImageFromMemory(empty_buffer);
    } catch (const ImageLoaderException& e) {
        std::cout << "  [成功] 捕获预期异常 (内存为空/格式错误): " << e.what() << std::endl;
    }

    // ==========================================
    // 测试 3：批量加载图片 (loadBatch)
    // ==========================================
    std::cout << "\n[Test 3] 测试 loadBatch..." << std::endl;
    try {
        std::vector<std::string> batch_paths = {valid_path1, invalid_path, valid_path2};
        std::vector<cv::Mat> batch_imgs = loader.loadBatch(batch_paths);
        
        std::cout << "  [成功] 批量加载完成。请求列表长度: " << batch_paths.size() 
                  << " | 返回列表长度: " << batch_imgs.size() << std::endl;

        for (size_t i = 0; i < batch_imgs.size(); ++i) {
            if (batch_imgs[i].empty()) {
                std::cout << "    - 第 " << i << " 张图片为空 (原因为加载失败，已放入空矩阵占位)" << std::endl;
            } else {
                std::cout << "    - 第 " << i << " 张图片尺寸: " << batch_imgs[i].cols << "x" << batch_imgs[i].rows << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << "  [失败] 发生未预期的异常: " << e.what() << std::endl;
    }

    // ==========================================
    // 环境清理：删除生成的测试图片
    // ==========================================
    if (fs::exists(valid_path1)) fs::remove(valid_path1);
    if (fs::exists(valid_path2)) fs::remove(valid_path2);

    std::cout << "\n========== 测试结束 ==========" << std::endl;
    return 0;
}