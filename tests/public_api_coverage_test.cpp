#include <cstring>
#include <string>

#include "digital_human/c_api.h"
#include "digital_human_sdk.h"
#include "test_support.h"

int main() {
    using digital_human::DigitalHumanSDK;
    using digital_human::SDKConfig;
    using digital_human::SDKError;
    using digital_human::SDKErrorToString;
    using digital_human::SDKState;

    TestSuite suite;
    suite.Equal(dh_sdk_api_version(), uint32_t{0x00020000},
                "C API version contract");
    suite.Check(std::string(SDKErrorToString(SDKError::SHUTDOWN_TIMEOUT))
                    == "SHUTDOWN_TIMEOUT",
                "C++ error string mapping");
    suite.Check(std::string(SDKErrorToString(static_cast<SDKError>(-1)))
                    == "UNKNOWN",
                "unknown C++ error string mapping");

    dh_error_t error{};
    suite.Check(dh_sdk_create(nullptr, &error) == nullptr,
                "C API rejects null config");
    suite.Equal(error.struct_size, static_cast<uint32_t>(sizeof(dh_error_t)),
                "C API initializes error struct size");
    suite.Check(error.error_code == DH_ERROR_CONFIG
                    && error.message
                    && std::string(error.message).find("null")
                        != std::string::npos,
                "C API reports null config error");

    dh_sdk_config_t undersized{};
    undersized.struct_size = sizeof(dh_sdk_config_t) - 1;
    error = {};
    suite.Check(dh_sdk_create(&undersized, &error) == nullptr,
                "C API rejects undersized config");
    suite.Check(error.error_code == DH_ERROR_CONFIG,
                "undersized config maps to config error");

    dh_error_t too_small{};
    too_small.struct_size = 1;
    too_small.error_code = DH_ERROR_UNKNOWN;
    suite.Check(!dh_sdk_start(nullptr, &too_small),
                "null start fails with undersized error buffer");
    suite.Check(too_small.error_code == DH_ERROR_UNKNOWN,
                "undersized error buffer is not overwritten");

    error = {};
    suite.Check(!dh_sdk_start(nullptr, &error)
                    && error.error_code == DH_ERROR_LIFECYCLE,
                "null start reports lifecycle error");
    error = {};
    suite.Check(!dh_sdk_stop(nullptr, 1, &error)
                    && error.error_code == DH_ERROR_LIFECYCLE,
                "null stop reports lifecycle error");
    error = {};
    suite.Check(!dh_sdk_pause(nullptr, &error)
                    && error.error_code == DH_ERROR_LIFECYCLE,
                "null pause reports lifecycle error");
    error = {};
    suite.Check(!dh_sdk_resume(nullptr, &error)
                    && error.error_code == DH_ERROR_LIFECYCLE,
                "null resume reports lifecycle error");
    float sample = 0.0F;
    error = {};
    suite.Check(!dh_sdk_push_audio(nullptr, &sample, 1, 0, &error)
                    && error.error_code == DH_ERROR_LIFECYCLE,
                "null audio handle reports lifecycle error");
    uint8_t pixel[3] = {};
    error = {};
    suite.Check(!dh_sdk_push_video(nullptr, pixel, 1, 1, 3, 0, &error)
                    && error.error_code == DH_ERROR_LIFECYCLE,
                "null video handle reports lifecycle error");
    suite.Check(dh_sdk_get_state(nullptr) == DH_STATE_UNINITIALIZED,
                "null C API state is UNINITIALIZED");
    suite.Check(std::string(dh_sdk_get_last_error(nullptr)).empty(),
                "null C API last error is empty");
    dh_metrics_t metrics{};
    suite.Check(!dh_sdk_get_metrics(nullptr, &metrics),
                "metrics reject null SDK");

    dh_sdk_config_t config{};
    config.struct_size = sizeof(config);
    config.fps = 25;
    config.audio_sample_rate = 16000;
    config.audio_channels = 1;
    config.inference_threads = 1;
    error = {};
    dh_sdk_t* handle = dh_sdk_create(&config, &error);
    suite.Check(handle != nullptr && error.error_code == DH_ERROR_OK,
                "C API creates model-free initialized SDK");
    if (handle) {
        suite.Check(dh_sdk_get_state(handle) == DH_STATE_INITIALIZED,
                    "created C SDK is initialized");
        dh_metrics_t small_metrics{};
        small_metrics.struct_size = 1;
        suite.Check(!dh_sdk_get_metrics(handle, &small_metrics),
                    "metrics reject undersized output");
        metrics = {};
        suite.Check(dh_sdk_get_metrics(handle, &metrics)
                        && metrics.struct_size == sizeof(metrics),
                    "metrics initialize caller output");
        error = {};
        suite.Check(!dh_sdk_start(handle, &error)
                        && error.error_code == DH_ERROR_MODEL,
                    "C API start reports missing model");
        suite.Check(std::strlen(dh_sdk_get_last_error(handle)) > 0,
                    "C API exposes last model error");
        dh_sdk_mark_audio_eos(handle);
        dh_sdk_mark_video_eos(handle);
        dh_sdk_destroy(handle);
    }

    DigitalHumanSDK sdk;
    suite.Check(sdk.GetState() == SDKState::UNINITIALIZED,
                "C++ SDK initial state");
    suite.Check(sdk.Start() == SDKError::NOT_INITIALIZED,
                "C++ SDK rejects start before Init");
    suite.Check(sdk.PushAudio({}, 0) == SDKError::INVALID_INPUT,
                "C++ SDK validates empty audio before lifecycle state");
    suite.Check(sdk.PushVideo(cv::Mat(), 0) == SDKError::INVALID_INPUT,
                "C++ SDK validates empty video before lifecycle state");

    SDKConfig invalid;
    invalid.audio_sample_rate = 0;
    suite.Check(sdk.Init(invalid) == SDKError::INVALID_CONFIG,
                "C++ SDK rejects invalid configuration");
    SDKConfig valid;
    suite.Check(sdk.Init(valid) == SDKError::OK,
                "C++ SDK initializes without models");
    suite.Check(sdk.Init(valid) == SDKError::ALREADY_RUNNING,
                "C++ SDK rejects repeated Init");
    suite.Check(sdk.Start() == SDKError::MODEL_LOAD_FAILED,
                "C++ SDK start requires lip-sync model");
    suite.Check(sdk.Pause() == SDKError::NOT_RUNNING,
                "C++ SDK rejects pause while initialized");
    suite.Check(sdk.Resume() == SDKError::NOT_RUNNING,
                "C++ SDK rejects resume while initialized");
    suite.Check(sdk.Stop(100) == SDKError::OK,
                "C++ SDK stops initialized pipeline");
    suite.Check(sdk.GetState() == SDKState::STOPPED,
                "C++ SDK reaches terminal stopped state");
    suite.Check(sdk.Start() == SDKError::ALREADY_TERMINATED,
                "C++ SDK cannot restart terminal instance");

    return suite.Finish("public_api_coverage_test");
}
