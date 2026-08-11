#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "ncnn/c_api.h"

namespace {

using CreateGpuInstance = int (*)(const char*);
using DestroyGpuInstance = void (*)();
using GetGpuCount = int (*)();
using GetDefaultGpuIndex = int (*)();

template <typename T>
T LoadNcnnSymbol(HMODULE module, const char* name) {
    return reinterpret_cast<T>(GetProcAddress(module, name));
}

double Median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

int Run(const char* param_path, const char* bin_path, bool use_gpu, int requested_gpu,
        const char* output_path) {
    HMODULE ncnn_module = GetModuleHandleA("ncnn.dll");
    if (!ncnn_module) {
        std::fprintf(stderr, "NCNN DLL is not loaded; add its bin directory to PATH.\n");
        return 2;
    }

    const auto create_gpu = LoadNcnnSymbol<CreateGpuInstance>(
        ncnn_module, "?create_gpu_instance@ncnn@@YAHPEBD@Z");
    const auto destroy_gpu = LoadNcnnSymbol<DestroyGpuInstance>(
        ncnn_module, "?destroy_gpu_instance@ncnn@@YAXXZ");
    const auto get_gpu_count = LoadNcnnSymbol<GetGpuCount>(
        ncnn_module, "?get_gpu_count@ncnn@@YAHXZ");
    const auto get_default_gpu_index = LoadNcnnSymbol<GetDefaultGpuIndex>(
        ncnn_module, "?get_default_gpu_index@ncnn@@YAHXZ");

    if (!create_gpu || !destroy_gpu || !get_gpu_count || !get_default_gpu_index) {
        std::fprintf(stderr, "Required NCNN Vulkan exports were not found.\n");
        return 3;
    }

    const int create_result = create_gpu(nullptr);
    const int gpu_count = get_gpu_count();
    const int default_gpu = get_default_gpu_index();
    std::printf("vulkan_instance_result=%d gpu_count=%d default_gpu=%d\n",
                create_result, gpu_count, default_gpu);
    if (use_gpu && (create_result != 0 || gpu_count <= 0)) {
        std::fprintf(stderr, "Vulkan requested but NCNN could not enumerate a GPU.\n");
        destroy_gpu();
        return 4;
    }

    const int device_index = requested_gpu >= 0 ? requested_gpu : default_gpu;
    ncnn_option_t options = ncnn_option_create();
    ncnn_option_set_num_threads(options, 4);
    ncnn_option_set_use_vulkan_compute(options, use_gpu ? 1 : 0);
    ncnn_option_set_use_fp16_packed(options, use_gpu ? 1 : 0);
    ncnn_option_set_use_fp16_storage(options, use_gpu ? 1 : 0);
    ncnn_option_set_use_fp16_arithmetic(options, use_gpu ? 1 : 0);

    ncnn_net_t net = ncnn_net_create();
    ncnn_net_set_option(net, options);
    if (use_gpu) ncnn_net_set_vulkan_device(net, device_index);

    const int param_result = ncnn_net_load_param(net, param_path);
    const int model_result = param_result == 0 ? ncnn_net_load_model(net, bin_path) : -1;
    if (param_result != 0 || model_result != 0) {
        std::fprintf(stderr, "Model load failed: param=%d model=%d\n", param_result, model_result);
        ncnn_net_destroy(net);
        ncnn_option_destroy(options);
        destroy_gpu();
        return 5;
    }

    std::vector<float> audio(16 * 80, 0.0f);
    std::vector<float> face(96 * 96 * 6, 0.0f);
    ncnn_mat_t audio_mat = ncnn_mat_create_external_3d(16, 80, 1, audio.data(), nullptr);
    ncnn_mat_t face_mat = ncnn_mat_create_external_3d(96, 96, 6, face.data(), nullptr);

    std::vector<double> timings;
    int final_result = 0;
    int output_dims = 0, output_w = 0, output_h = 0, output_c = 0;
    size_t output_elemsize = 0;
    int output_elempack = 0;
    double output_sum = 0.0;
    double output_abs_sum = 0.0;
    std::vector<float> output_values;
    for (int iteration = 0; iteration < 5; ++iteration) {
        ncnn_extractor_t extractor = ncnn_extractor_create(net);
        const int input_audio = ncnn_extractor_input(extractor, "audio_sequences", audio_mat);
        const int input_face = ncnn_extractor_input(extractor, "face_sequences", face_mat);
        ncnn_mat_t output = ncnn_mat_create();
        const auto start = std::chrono::steady_clock::now();
        const int extract_result = (input_audio == 0 && input_face == 0)
            ? ncnn_extractor_extract(extractor, "output", &output) : -1;
        const auto end = std::chrono::steady_clock::now();
        const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
        final_result = extract_result;
        if (extract_result == 0) {
            output_dims = ncnn_mat_get_dims(output);
            output_w = ncnn_mat_get_w(output);
            output_h = ncnn_mat_get_h(output);
            output_c = ncnn_mat_get_c(output);
            output_elemsize = ncnn_mat_get_elemsize(output);
            output_elempack = ncnn_mat_get_elempack(output);
            if (output_elemsize == sizeof(float)) {
                const auto* values = static_cast<const float*>(ncnn_mat_get_data(output));
                const size_t count = static_cast<size_t>(output_w) * output_h * output_c * output_elempack;
                output_sum = 0.0;
                output_abs_sum = 0.0;
                for (size_t i = 0; i < count; ++i) {
                    output_sum += values[i];
                    output_abs_sum += std::abs(values[i]);
                }
                output_values.assign(values, values + count);
            }
            if (iteration > 0) timings.push_back(milliseconds);
        }
        ncnn_mat_destroy(output);
        ncnn_extractor_destroy(extractor);
        if (extract_result != 0) break;
    }

    const double average = timings.empty() ? 0.0 : std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    std::printf("mode=%s device_index=%d extract_result=%d output_dims=%d output=%dx%dx%d elemsize=%zu elempack=%d sum=%.9f abs_sum=%.9f runs=%zu avg_ms=%.3f median_ms=%.3f\n",
                use_gpu ? "vulkan" : "cpu", device_index, final_result, output_dims,
                output_w, output_h, output_c, output_elemsize, output_elempack, output_sum, output_abs_sum,
                timings.size(), average, Median(timings));

    if (output_path && !output_values.empty()) {
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<const char*>(output_values.data()),
                          static_cast<std::streamsize>(output_values.size() * sizeof(float)));
        if (!output_file) {
            std::fprintf(stderr, "Could not write output tensor: %s\n", output_path);
            final_result = -1;
        }
    }

    ncnn_mat_destroy(audio_mat);
    ncnn_mat_destroy(face_mat);
    ncnn_net_destroy(net);
    ncnn_option_destroy(options);
    destroy_gpu();

    return final_result == 0 && output_dims > 0 ? 0 : 6;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s <param> <bin> <cpu|vulkan> [gpu-index] [output-f32-bin]\n", argv[0]);
        return 1;
    }
    const bool use_gpu = std::string(argv[3]) == "vulkan";
    const int gpu_index = argc > 4 ? std::atoi(argv[4]) : -1;
    const char* output_path = argc > 5 ? argv[5] : nullptr;
    return Run(argv[1], argv[2], use_gpu, gpu_index, output_path);
}
