#include "test_support.h"

#include <atomic>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "avatar/avatar_image.h"
#include "core/image_loader.h"
#include "dialog/sentence_segmenter.h"
#include "model/model_loader.h"

namespace {

template <typename Function>
bool ThrowsImageLoaderException(Function&& function) {
    try {
        function();
    } catch (const digital_human::core::ImageLoaderException&) {
        return true;
    } catch (...) {
    }
    return false;
}

void TestSentenceSegmenter(TestSuite& test) {
    using digital_human::dialog::SentenceSegmenter;
    using digital_human::dialog::SentenceSegmenterConfig;

    SentenceSegmenter segmenter(SentenceSegmenterConfig{4});
    test.Check(segmenter.Push("  hello").empty(),
               "partial text remains buffered");
    auto strong = segmenter.Push(" world. next");
    test.Equal(strong.size(), size_t{1}, "strong boundary emits one clause");
    test.Equal(strong.front(), std::string("hello world."),
               "strong boundary preserves punctuation and trims whitespace");

    auto weak = segmenter.Push(" abc,");
    test.Equal(weak.size(), size_t{1}, "long weak-boundary clause is emitted");
    test.Equal(weak.front(), std::string("next abc,"),
               "weak-boundary clause includes prior buffered text");
    test.Equal(segmenter.Flush(), std::string{}, "flush after boundary is empty");

    SentenceSegmenter short_weak(SentenceSegmenterConfig{8});
    test.Check(short_weak.Push("abc,").empty(),
               "short weak-boundary clause remains buffered");
    test.Equal(short_weak.Flush(), std::string("abc,"),
               "flush returns a short buffered clause");
    short_weak.Push("discard me");
    short_weak.Reset();
    test.Equal(short_weak.Flush(), std::string{}, "reset clears buffered text");

    SentenceSegmenter utf8(SentenceSegmenterConfig{2});
    test.Check(utf8.Push(u8"你").empty(), "UTF-8 delta remains buffered");
    auto utf8_clause = utf8.Push(",");
    test.Equal(utf8_clause.size(), size_t{1},
               "UTF-8 character count drives weak boundary");
    test.Equal(utf8_clause.front(), std::string(u8"你,"),
               "UTF-8 clause is not split in the middle of a code point");
}

void TestAvatarAndImageLoader(TestSuite& test) {
    using namespace digital_human::avatar;

    cv::Mat source(3, 4, CV_8UC3, cv::Scalar(10, 40, 200));
    std::vector<uint8_t> png;
    test.Check(cv::imencode(".png", source, png),
               "test fixture encodes as PNG");

    AvatarImage image;
    AvatarUploadLimits limits;
    std::string error;
    test.Check(DecodeAvatarUpload(png, "image/png", limits, image, error),
               "valid PNG upload decodes");
    test.Equal(image.format, AvatarImageFormat::PNG, "PNG format is detected");
    test.Equal(image.bgr.cols, 4, "decoded avatar width is retained");
    test.Equal(image.bgr.rows, 3, "decoded avatar height is retained");
    test.Check(image.bgr.data != source.data,
               "decoded avatar owns independent pixel storage");

    test.Check(!DecodeAvatarUpload({}, {}, limits, image, error),
               "empty upload is rejected");
    test.Check(!error.empty(), "empty upload reports an error");
    test.Check(!DecodeAvatarUpload({1, 2, 3}, {}, limits, image, error),
               "unknown image signature is rejected");
    test.Check(!DecodeAvatarUpload(png, "image/jpeg", limits, image, error),
               "content type mismatch is rejected");

    auto byte_limited = limits;
    byte_limited.max_encoded_bytes = png.size() - 1;
    test.Check(!DecodeAvatarUpload(png, "image/png", byte_limited, image, error),
               "encoded byte limit is enforced");
    auto dimension_limited = limits;
    dimension_limited.max_width = 3;
    test.Check(!DecodeAvatarUpload(png, "image/png", dimension_limited,
                                   image, error),
               "decoded dimension limit is enforced");
    auto pixel_limited = limits;
    pixel_limited.max_pixels = 11;
    test.Check(!DecodeAvatarUpload(png, "image/png", pixel_limited,
                                   image, error),
               "decoded pixel limit is enforced");
    auto invalid_limits = limits;
    invalid_limits.max_height = 0;
    test.Check(!DecodeAvatarUpload(png, "image/png", invalid_limits,
                                   image, error),
               "invalid upload limits are rejected");
    test.Check(!LoadAvatarImage("does-not-exist-avatar.png", limits,
                                image, error),
               "missing persisted avatar is rejected");

    digital_human::core::ImageLoader loader;
    auto decoded = loader.loadImageFromMemory(png);
    test.Equal(decoded.cols, 4, "image loader decodes in-memory data");
    test.Check(ThrowsImageLoaderException([&]() {
                   loader.loadImageFromMemory({});
               }),
               "image loader rejects empty memory");
    test.Check(ThrowsImageLoaderException([&]() {
                   loader.loadImageFromMemory({1, 2, 3});
               }),
               "image loader rejects corrupt memory");
    test.Check(ThrowsImageLoaderException([&]() {
                   loader.loadImageFromFile("does-not-exist-image.png");
               }),
               "image loader rejects a missing file");
    test.Check(ThrowsImageLoaderException([&]() {
                   loader.loadBatch({});
               }),
               "batch loader rejects an empty path list");
    auto batch = loader.loadBatch({"does-not-exist-image.png"});
    test.Equal(batch.size(), size_t{1}, "batch loader preserves item count");
    test.Check(batch.front().empty(), "batch loader isolates item failures");

    digital_human::core::ImageLoader moved(std::move(loader));
    test.Equal(moved.loadImageFromMemory(png).rows, 3,
               "moved image loader remains usable");
}

void TestModelLoaderFailureContract(TestSuite& test) {
    using digital_human::model::ModelLoader;

    ModelLoader loader;
    test.Check(!loader.IsLoaded(), "new model loader starts unloaded");
    test.Check(loader.GetNet() == nullptr, "unloaded model has no network");
    test.Near(loader.GetIOCostMs(), 0.0, 0.0, "initial IO cost is zero");
    test.Near(loader.GetWarmupCostMs(), 0.0, 0.0,
              "initial warmup cost is zero");
    loader.Wait();
    loader.SetWarmupShapes(8, 80, 1, 64, 64, 6);

    std::atomic<int> callbacks{0};
    loader.LoadAsync("missing-model-directory", [&](ncnn::Net* net,
                                                      float io_cost,
                                                      float warmup_cost) {
        test.Check(net == nullptr, "missing model callback receives null net");
        test.Near(io_cost, 0.0, 0.0, "missing model IO cost is zero");
        test.Near(warmup_cost, 0.0, 0.0,
                  "missing model warmup cost is zero");
        callbacks.fetch_add(1);
    });
    loader.Wait();
    test.Equal(callbacks.load(), 1, "missing model invokes callback once");
    test.Check(!loader.IsLoaded(), "failed model load remains unloaded");

    ModelLoader explicit_loader;
    explicit_loader.LoadAsync("missing.param", "missing.bin",
                              [&](ncnn::Net* net, float, float) {
        test.Check(net == nullptr,
                   "explicit missing model paths return null net");
        callbacks.fetch_add(1);
    });
    test.Equal(callbacks.load(), 2,
               "explicit missing model paths invoke callback");

    ModelLoader moved(std::move(explicit_loader));
    moved.Wait();
    test.Check(!moved.IsLoaded(), "moved failed loader remains usable");
}

}  // namespace

int main() {
    TestSuite test;
    TestSentenceSegmenter(test);
    TestAvatarAndImageLoader(test);
    TestModelLoaderFailureContract(test);
    return test.Finish("module_contract_coverage_test");
}
