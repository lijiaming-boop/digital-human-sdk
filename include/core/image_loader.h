#pragma once

#include <string>
#include <vector>
#include <memory>
#include <exception>
#include <opencv2/opencv.hpp>

namespace digital_human {
namespace core {

class ImageLoader {
public:
    ImageLoader();
    ~ImageLoader();
    ImageLoader(const ImageLoader& other) = delete;
    ImageLoader& operator=(const ImageLoader& other) = delete;
    ImageLoader(ImageLoader&& other) noexcept;
    ImageLoader& operator=(ImageLoader&& other) noexcept;

    cv::Mat loadImageFromFile(const std::string& path);
    cv::Mat loadImageFromMemory(const std::vector<uint8_t>& data);
    std::vector<cv::Mat> loadBatch(const std::vector<std::string>& paths);

private:
    struct ImageLoaderImpl;
    std::unique_ptr<ImageLoaderImpl> impl_;
};

class ImageLoaderException : public std::runtime_error {
public:
    explicit ImageLoaderException(const std::string& message) 
        : std::runtime_error(message) {}
};

}  // namespace core
}  // namespace digital_human