#include <filesystem>
#include <iostream>
#include <string>

#include "digital_human_sdk.h"

namespace fs = std::filesystem;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "[PASS] " << message << '\n';
    } else {
        std::cerr << "[FAIL] " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const fs::path missing_face_dir = fs::temp_directory_path()
        / "digital_human_sdk_missing_face_models";
    fs::remove_all(missing_face_dir);

    {
        digital_human::DigitalHumanSDK sdk;
        digital_human::SDKConfig config;
        config.face_model_dir = missing_face_dir.string();

        const auto result = sdk.Init(config);
        Check(result == digital_human::SDKError::FACE_MODEL_LOAD_FAILED,
              "Init 传播人脸模型加载错误码");
        Check(sdk.GetLastError().find(missing_face_dir.string())
                  != std::string::npos,
              "Init 错误消息包含失败模型路径");
        Check(sdk.GetState() == digital_human::SDKState::UNINITIALIZED,
              "模型加载失败后 SDK 不进入已初始化状态");
    }

    {
        digital_human::DigitalHumanSDK sdk;
        digital_human::SDKConfig config;
        Check(sdk.Init(config) == digital_human::SDKError::OK,
              "允许先初始化、后手动加载模型");

        const auto result = sdk.LoadFaceModel(missing_face_dir.string());
        Check(result == digital_human::SDKError::FACE_MODEL_LOAD_FAILED,
              "LoadFaceModel 同步返回加载失败");
        Check(!sdk.GetLastError().empty(),
              "LoadFaceModel 失败时保留错误消息");

        const auto start = sdk.Start();
        Check(start == digital_human::SDKError::MODEL_LOAD_FAILED,
              "Start 拒绝在 Wav2Lip 未加载时启动");
        Check(sdk.GetLastError().find("Wav2Lip") != std::string::npos,
              "Start 指明缺失的 Wav2Lip 模型");
    }

    std::cout << "模型错误传播测试: failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
