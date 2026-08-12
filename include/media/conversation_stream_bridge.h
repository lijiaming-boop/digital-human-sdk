#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "dialog/conversation_session.h"

namespace digital_human {

class DigitalHumanSDK;

namespace media {

class StreamPublisher;

/// Fans TTS PCM out to both the DigitalHumanSDK and StreamPublisher, while a
/// background thread drains rendered BGR frames from the SDK into the publisher.
class ConversationStreamBridge final : public dialog::IDigitalHumanSink {
public:
    ConversationStreamBridge(DigitalHumanSDK& sdk,
                             StreamPublisher& publisher);
    ~ConversationStreamBridge() override;

    ConversationStreamBridge(const ConversationStreamBridge&) = delete;
    ConversationStreamBridge& operator=(const ConversationStreamBridge&) = delete;

    bool Start(std::string& error);

    bool PushAudio(const std::vector<float>& samples,
                   int64_t pts_ms,
                   std::string& error) override;
    bool PushVideo(const cv::Mat& frame,
                   int64_t pts_ms,
                   std::string& error) override;
    void Finish() override;

    std::string GetLastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace media
}  // namespace digital_human
