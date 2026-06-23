#include "model/model_loader.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <atomic>

namespace fs = std::filesystem;

namespace digital_human {
namespace model {

struct ModelLoader::Impl {
    ncnn::Net net;
    std::atomic<bool> is_loaded{false};
    std::atomic<bool> loading{false};
    std::thread loading_thread;

    float io_cost_ms = 0.0f;
    float warmup_cost_ms = 0.0f;

    // default warmup input shapes
    int audio_w = 80, audio_h = 80, audio_c = 1;
    int face_w = 96, face_h = 96, face_c = 6;

    static constexpr const char* kModelName = "Wav2Lip-SD-GAN-opt";
    static constexpr const char* kAudioInput = "audio_sequences";
    static constexpr const char* kFaceInput  = "face_sequences";
    static constexpr const char* kOutputName = "output";

    ~Impl() {
        if (loading_thread.joinable()) {
            loading_thread.join();
        }
    }

    void derivePaths(const std::string& model_dir,
                     std::string& param_path, std::string& bin_path) {
        fs::path dir(model_dir);
        param_path = (dir / (std::string(kModelName) + ".param")).string();
        bin_path   = (dir / (std::string(kModelName) + ".bin")).string();
    }

    bool verifyFiles(const std::string& param_path, const std::string& bin_path) {
        if (!fs::exists(param_path)) {
            std::cerr << "[ModelLoader] param file not found: " << param_path << std::endl;
            return false;
        }
        if (!fs::exists(bin_path)) {
            std::cerr << "[ModelLoader] bin file not found: " << bin_path << std::endl;
            return false;
        }
        return true;
    }

    void doLoad(const std::string& param_path, const std::string& bin_path,
                LoadCallback callback) {
        // ---- IO timing ----
        auto io_start = std::chrono::steady_clock::now();

        if (net.load_param(param_path.c_str()) != 0) {
            std::cerr << "[ModelLoader] failed to load param: " << param_path << std::endl;
            loading = false;
            if (callback) callback(nullptr, 0.0f, 0.0f);
            return;
        }

        if (net.load_model(bin_path.c_str()) != 0) {
            std::cerr << "[ModelLoader] failed to load model: " << bin_path << std::endl;
            loading = false;
            if (callback) callback(nullptr, 0.0f, 0.0f);
            return;
        }

        auto io_end = std::chrono::steady_clock::now();
        io_cost_ms = std::chrono::duration<float, std::milli>(io_end - io_start).count();
        std::cout << "[ModelLoader] IO cost: " << io_cost_ms << " ms" << std::endl;

        // ---- Warmup ----
        auto warmup_start = std::chrono::steady_clock::now();

        ncnn::Extractor ex = net.create_extractor();

        ncnn::Mat audio_in(audio_w, audio_h, audio_c);
        audio_in.fill(0.0f);
        ncnn::Mat face_in(face_w, face_h, face_c);
        face_in.fill(0.0f);

        ex.input(kAudioInput, audio_in);
        ex.input(kFaceInput, face_in);

        ncnn::Mat out;
        int ret = ex.extract(kOutputName, out);

        auto warmup_end = std::chrono::steady_clock::now();
        warmup_cost_ms = std::chrono::duration<float, std::milli>(warmup_end - warmup_start).count();

        if (ret != 0) {
            std::cerr << "[ModelLoader] warmup inference failed (ret=" << ret
                      << "), model loaded but warmup shapes may be wrong" << std::endl;
        } else {
            std::cout << "[ModelLoader] warmup cost: " << warmup_cost_ms << " ms" << std::endl;
        }

        is_loaded = true;
        loading = false;

        if (callback) {
            callback(&net, io_cost_ms, warmup_cost_ms);
        }
    }
};

ModelLoader::ModelLoader() : impl_(std::make_unique<Impl>()) {}
ModelLoader::~ModelLoader() = default;
ModelLoader::ModelLoader(ModelLoader&&) noexcept = default;
ModelLoader& ModelLoader::operator=(ModelLoader&&) noexcept = default;

void ModelLoader::LoadAsync(const std::string& model_dir, LoadCallback callback) {
    std::string param_path, bin_path;
    impl_->derivePaths(model_dir, param_path, bin_path);
    LoadAsync(param_path, bin_path, std::move(callback));
}

void ModelLoader::LoadAsync(const std::string& param_path, const std::string& bin_path,
                             LoadCallback callback) {
    if (impl_->loading || impl_->is_loaded) {
        return;
    }

    if (!impl_->verifyFiles(param_path, bin_path)) {
        if (callback) callback(nullptr, 0.0f, 0.0f);
        return;
    }

    impl_->loading = true;
    impl_->loading_thread = std::thread(&Impl::doLoad, impl_.get(),
                                         param_path, bin_path, std::move(callback));
}

ncnn::Net* ModelLoader::GetNet() {
    return impl_->is_loaded ? &impl_->net : nullptr;
}

bool ModelLoader::IsLoaded() const {
    return impl_->is_loaded;
}

float ModelLoader::GetIOCostMs() const {
    return impl_->io_cost_ms;
}

float ModelLoader::GetWarmupCostMs() const {
    return impl_->warmup_cost_ms;
}

void ModelLoader::Wait() {
    if (impl_->loading_thread.joinable()) {
        impl_->loading_thread.join();
    }
}

void ModelLoader::SetWarmupShapes(int audio_w, int audio_h, int audio_c,
                                   int face_w, int face_h, int face_c) {
    impl_->audio_w = audio_w; impl_->audio_h = audio_h; impl_->audio_c = audio_c;
    impl_->face_w = face_w; impl_->face_h = face_h; impl_->face_c = face_c;
}

} // namespace model
} // namespace digital_human
