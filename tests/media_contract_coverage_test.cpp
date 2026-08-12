#include "test_support.h"

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "digital_human_sdk.h"
#include "media/conversation_stream_bridge.h"
#include "media/stream_publisher.h"

namespace {

digital_human::media::StreamPublisherConfig ValidConfig() {
    digital_human::media::StreamPublisherConfig config;
    config.url = "unused.flv";
    config.protocol = digital_human::media::StreamProtocol::FILE;
    config.width = 64;
    config.height = 64;
    config.fps = 25.0;
    config.max_video_queue = 2;
    config.max_audio_queue = 2;
    return config;
}

void ExpectInvalid(TestSuite& test,
                   digital_human::media::StreamPublisherConfig config,
                   const std::string& case_name) {
    digital_human::media::StreamPublisher publisher;
    std::string error;
    test.Check(!publisher.Open(config, error), case_name);
    test.Equal(error, std::string("invalid stream publisher configuration"),
               case_name + " reports configuration error");
    test.Check(!publisher.IsOpen(), case_name + " leaves publisher closed");
}

void TestPublisherValidation(TestSuite& test) {
    using namespace digital_human::media;

    auto config = ValidConfig();
    config.url.clear();
    ExpectInvalid(test, config, "empty output URL is rejected");
    config = ValidConfig();
    config.width = 0;
    ExpectInvalid(test, config, "zero width is rejected");
    config = ValidConfig();
    config.width = 63;
    ExpectInvalid(test, config, "odd width is rejected");
    config = ValidConfig();
    config.height = 63;
    ExpectInvalid(test, config, "odd height is rejected");
    config = ValidConfig();
    config.fps = 0.0;
    ExpectInvalid(test, config, "zero FPS is rejected");
    config = ValidConfig();
    config.input_audio_sample_rate = 0;
    ExpectInvalid(test, config, "zero input sample rate is rejected");
    config = ValidConfig();
    config.output_audio_channels = 0;
    ExpectInvalid(test, config, "zero output channel count is rejected");
    config = ValidConfig();
    config.max_video_queue = 0;
    ExpectInvalid(test, config, "zero video queue capacity is rejected");
    config = ValidConfig();
    config.max_audio_queue = 0;
    ExpectInvalid(test, config, "zero audio queue capacity is rejected");
    config = ValidConfig();
    config.max_video_queue_latency_ms = -1;
    ExpectInvalid(test, config, "negative video latency limit is rejected");
    config = ValidConfig();
    config.reconnect_initial_backoff_ms = -1;
    ExpectInvalid(test, config, "negative reconnect backoff is rejected");
    config = ValidConfig();
    config.reconnect_initial_backoff_ms = 10;
    config.reconnect_max_backoff_ms = 9;
    ExpectInvalid(test, config, "descending reconnect backoff is rejected");
    config = ValidConfig();
    config.reconnect_window_ms = -1;
    ExpectInvalid(test, config, "negative reconnect window is rejected");
    config = ValidConfig();
    config.max_retained_audio_ms = -1;
    ExpectInvalid(test, config, "negative retained audio limit is rejected");

    StreamPublisher publisher;
    std::string error;
    test.Check(!publisher.PushVideo({}, 0, error),
               "publisher rejects empty video before open");
    cv::Mat invalid_type(2, 2, CV_8UC1);
    test.Check(!publisher.PushVideo(invalid_type, 0, error),
               "publisher rejects non-BGR video");
    cv::Mat bgr(2, 2, CV_8UC3);
    test.Check(!publisher.PushVideo(bgr, -1, error),
               "publisher rejects negative video PTS");
    test.Check(!publisher.PushVideo(bgr, 0, error),
               "publisher rejects valid video while closed");
    test.Check(!publisher.PushAudio({}, 0, error),
               "publisher rejects empty audio before open");
    test.Check(!publisher.PushAudio({0.0f}, -1, error),
               "publisher rejects negative audio PTS");
    test.Check(!publisher.PushAudio({0.0f}, 0, error),
               "publisher rejects valid audio while closed");
    test.Check(publisher.Close(false, error),
               "closing an unopened publisher is idempotent");
    auto metrics = publisher.GetMetrics();
    test.Equal(metrics.video_frames_in, int64_t{0},
               "closed publisher starts with zero video metrics");
    test.Equal(metrics.audio_samples_in, int64_t{0},
               "closed publisher starts with zero audio metrics");
    test.Check(publisher.GetLastError().empty(),
               "new publisher has no asynchronous error");
}

void TestBridgeBoundary(TestSuite& test) {
    digital_human::DigitalHumanSDK sdk;
    digital_human::media::StreamPublisher publisher;
    digital_human::media::ConversationStreamBridge bridge(sdk, publisher);
    std::string error;
    test.Check(!bridge.Start(error),
               "bridge requires an initialized SDK and open publisher");
    test.Check(!error.empty(), "bridge start failure reports an error");
    test.Check(!bridge.PushAudio({}, 0, error),
               "bridge rejects empty audio");
    test.Check(!bridge.PushVideo({}, 0, error),
               "bridge rejects empty video");
    bridge.Finish();
    test.Check(bridge.GetLastError().empty(),
               "pre-start bridge errors remain request-local");
}

}  // namespace

int main() {
    TestSuite test;
    TestPublisherValidation(test);
    TestBridgeBoundary(test);
    return test.Finish("media_contract_coverage_test");
}
