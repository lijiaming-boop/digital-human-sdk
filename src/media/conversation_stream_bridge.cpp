#include "media/conversation_stream_bridge.h"

#include <mutex>
#include <utility>

#include "digital_human_sdk.h"
#include "media/stream_publisher.h"

namespace digital_human {
namespace media {

struct ConversationStreamBridge::Impl {
    DigitalHumanSDK& sdk;
    StreamPublisher& publisher;
    mutable std::mutex mutex;
    std::thread output_thread;
    bool started = false;
    bool finishing = false;
    bool finished = false;
    /// 输出线程发生不可恢复错误后置位，使后续 PushAudio/PushVideo 立即拒绝，
    /// 从而让 ConversationSession 的媒体供料失败并取消当前 turn（P0 错误传播）。
    bool failed = false;
    std::string last_error;

    Impl(DigitalHumanSDK& sdk_ref, StreamPublisher& publisher_ref)
        : sdk(sdk_ref), publisher(publisher_ref) {}

    void SetError(const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex);
        if (last_error.empty()) last_error = error;
        failed = true;  // 任何输出错误都应立即阻断后续供料
    }

    void DrainOutput() {
        while (true) {
            cv::Mat frame;
            int64_t pts_ms = 0;
            const auto result = sdk.GetOutputFrame(frame, pts_ms, 500);
            if (result == SDKError::OK) {
                std::string error;
                if (!publisher.PushVideo(frame, pts_ms, error)) {
                    SetError("publisher video input failed: " + error);
                    break;
                }
                continue;
            }
            if (result == SDKError::TIMEOUT) continue;
            if (result != SDKError::NOT_RUNNING) {
                SetError(std::string("SDK output failed: ")
                         + SDKErrorToString(result) + ": "
                         + sdk.GetLastError());
            }
            break;
        }
    }
};

ConversationStreamBridge::ConversationStreamBridge(
    DigitalHumanSDK& sdk, StreamPublisher& publisher)
    : impl_(std::make_unique<Impl>(sdk, publisher)) {}

ConversationStreamBridge::~ConversationStreamBridge() {
    Finish();
}

bool ConversationStreamBridge::Start(std::string& error) {
    error.clear();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->started || impl_->finished) {
        error = "conversation stream bridge has already been used";
        return false;
    }
    if (impl_->sdk.GetState() != SDKState::RUNNING) {
        error = "DigitalHumanSDK must be running before bridge Start";
        return false;
    }
    if (!impl_->publisher.IsOpen()) {
        error = "StreamPublisher must be open before bridge Start";
        return false;
    }
    impl_->started = true;
    impl_->output_thread = std::thread([this]() { impl_->DrainOutput(); });
    return true;
}

bool ConversationStreamBridge::PushAudio(
    const std::vector<float>& samples,
    int64_t pts_ms,
    std::string& error) {
    error.clear();
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->started || impl_->finishing || impl_->failed) {
            error = impl_->failed
                ? (impl_->last_error.empty()
                    ? "conversation stream bridge output failed"
                    : impl_->last_error)
                : "conversation stream bridge is not accepting audio";
            return false;
        }
    }
    const auto sdk_result = impl_->sdk.PushAudio(samples, pts_ms);
    if (sdk_result != SDKError::OK) {
        error = std::string("SDK PushAudio failed: ")
              + SDKErrorToString(sdk_result) + ": "
              + impl_->sdk.GetLastError();
        impl_->SetError(error);
        return false;
    }
    if (!impl_->publisher.PushAudio(samples, pts_ms, error)) {
        impl_->SetError("publisher audio input failed: " + error);
        return false;
    }
    return true;
}

bool ConversationStreamBridge::PushVideo(const cv::Mat& frame,
                                         int64_t pts_ms,
                                         std::string& error) {
    error.clear();
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->started || impl_->finishing || impl_->failed) {
            error = impl_->failed
                ? (impl_->last_error.empty()
                    ? "conversation stream bridge output failed"
                    : impl_->last_error)
                : "conversation stream bridge is not accepting avatar video";
            return false;
        }
    }
    const auto result = impl_->sdk.PushVideo(frame, pts_ms);
    if (result == SDKError::OK) return true;
    error = std::string("SDK PushVideo failed: ")
          + SDKErrorToString(result) + ": " + impl_->sdk.GetLastError();
    impl_->SetError(error);
    return false;
}

void ConversationStreamBridge::Finish() {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->started || impl_->finished || impl_->finishing) return;
        impl_->finishing = true;
    }

    impl_->sdk.MarkAudioEOS();
    impl_->sdk.MarkVideoEOS();
    if (impl_->output_thread.joinable()) impl_->output_thread.join();

    std::string publisher_error;
    if (!impl_->publisher.Close(true, publisher_error)
        && !publisher_error.empty()) {
        impl_->SetError("publisher close failed: " + publisher_error);
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->started = false;
    impl_->finishing = false;
    impl_->finished = true;
}

std::string ConversationStreamBridge::GetLastError() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->last_error;
}

}  // namespace media
}  // namespace digital_human
