#pragma once

#include <memory>
#include <string>
#include <functional>

#include <ncnn/net.h>

namespace digital_human {
namespace model {

using LoadCallback = std::function<void(ncnn::Net* net, float io_cost_ms, float warmup_cost_ms)>;

class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;
    ModelLoader(ModelLoader&&) noexcept;
    ModelLoader& operator=(ModelLoader&&) noexcept;

    // Load model from directory (derives .param/.bin paths from known model name)
    void LoadAsync(const std::string& model_dir, LoadCallback callback);

    // Load model with explicit param and bin paths
    void LoadAsync(const std::string& param_path, const std::string& bin_path, LoadCallback callback);

    ncnn::Net* GetNet();
    bool IsLoaded() const;
    float GetIOCostMs() const;
    float GetWarmupCostMs() const;

    // Block until async loading completes
    void Wait();

    // Configure dummy input shapes for warmup inference
    void SetWarmupShapes(int audio_w, int audio_h, int audio_c,
                         int face_w, int face_h, int face_c);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace model
} // namespace digital_human
