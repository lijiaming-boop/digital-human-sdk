#include "core/image_loader.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

namespace digital_human {
namespace core {

struct ImageLoader::ImageLoaderImpl {
    ImageLoaderImpl() = default;
    ~ImageLoaderImpl() = default;
    ImageLoaderImpl(ImageLoaderImpl&& other) = default;
    ImageLoaderImpl& operator=(ImageLoaderImpl&& other) = default;
    ImageLoaderImpl(const ImageLoaderImpl& other) = delete;
    ImageLoaderImpl& operator=(const ImageLoaderImpl& other) = delete;

    cv::Mat loadImageFromFile(const std::string& path) {
        if (!fs::exists(path)) {
            throw ImageLoaderException("file not exists");
        }
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            throw ImageLoaderException("failed to load image from file");
        }
        return img;
    }

    cv::Mat loadImageFromMemory(const std::vector<uint8_t>& data) {
        if (data.empty()) {
            throw ImageLoaderException("data is empty");
        }
        cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw ImageLoaderException("failed to load image from memory");
        }
        return img;
    }

    std::vector<cv::Mat> loadBatch(const std::vector<std::string>& paths) {
        if (paths.empty()) {
            throw ImageLoaderException("paths is empty");
        }
        std::vector<cv::Mat> imgs;
        imgs.reserve(paths.size());
        for (const auto& path : paths) {
            try {
                imgs.push_back(loadImageFromFile(path));
            } catch (const ImageLoaderException& e) {
                std::cerr << "[ImageLoader] load image from file failed: " << e.what() << std::endl;
                imgs.push_back(cv::Mat());
                continue;
            }
        }
        return imgs;
    }
};

ImageLoader::ImageLoader() : impl_(std::make_unique<ImageLoaderImpl>()) {}

ImageLoader::~ImageLoader() = default;
ImageLoader::ImageLoader(ImageLoader&& other) noexcept = default;
ImageLoader& ImageLoader::operator=(ImageLoader&& other) noexcept = default;

cv::Mat ImageLoader::loadImageFromFile(const std::string& path) {
    return impl_->loadImageFromFile(path);
}

cv::Mat ImageLoader::loadImageFromMemory(const std::vector<uint8_t>& data) {
    return impl_->loadImageFromMemory(data);
}

std::vector<cv::Mat> ImageLoader::loadBatch(const std::vector<std::string>& paths) {
    return impl_->loadBatch(paths);
}

}  // namespace core
}  // namespace digital_human