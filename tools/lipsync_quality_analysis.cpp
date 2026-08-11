#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

struct RegionMetrics {
    double mae = 0.0;
    double psnr = 0.0;
    double ssim = 0.0;
};

cv::Rect ExpandRect(cv::Rect rect, const cv::Size& bounds, double factor) {
    const int width = static_cast<int>(std::lround(rect.width * factor));
    const int height = static_cast<int>(std::lround(rect.height * factor));
    const int x = rect.x - (width - rect.width) / 2;
    const int y = rect.y - (height - rect.height) / 2;
    return cv::Rect(x, y, width, height) & cv::Rect(0, 0, bounds.width, bounds.height);
}

double GraySsim(const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask) {
    cv::Mat gray_a, gray_b;
    cv::cvtColor(a, gray_a, cv::COLOR_BGR2GRAY);
    cv::cvtColor(b, gray_b, cv::COLOR_BGR2GRAY);
    const cv::Scalar ma = cv::mean(gray_a, mask);
    const cv::Scalar mb = cv::mean(gray_b, mask);
    cv::Mat da, db;
    gray_a.convertTo(da, CV_64F, 1.0, -ma[0]);
    gray_b.convertTo(db, CV_64F, 1.0, -mb[0]);
    cv::Mat aa, bb, ab;
    cv::multiply(da, da, aa);
    cv::multiply(db, db, bb);
    cv::multiply(da, db, ab);
    const double variance_a = cv::mean(aa, mask)[0];
    const double variance_b = cv::mean(bb, mask)[0];
    const double covariance = cv::mean(ab, mask)[0];
    constexpr double c1 = 6.5025;
    constexpr double c2 = 58.5225;
    return ((2.0 * ma[0] * mb[0] + c1) * (2.0 * covariance + c2)) /
           ((ma[0] * ma[0] + mb[0] * mb[0] + c1) * (variance_a + variance_b + c2));
}

RegionMetrics Compare(const cv::Mat& original, const cv::Mat& generated,
                      const cv::Mat& mask) {
    cv::Mat difference;
    cv::absdiff(original, generated, difference);
    const cv::Scalar mean_difference = cv::mean(difference, mask);
    const double mae = (mean_difference[0] + mean_difference[1] + mean_difference[2]) / 3.0;

    cv::Mat original64, generated64, squared;
    original.convertTo(original64, CV_64F);
    generated.convertTo(generated64, CV_64F);
    cv::subtract(original64, generated64, squared);
    cv::multiply(squared, squared, squared);
    const cv::Scalar mse_channels = cv::mean(squared, mask);
    const double mse = (mse_channels[0] + mse_channels[1] + mse_channels[2]) / 3.0;
    const double psnr = mse > 0.0 ? 10.0 * std::log10((255.0 * 255.0) / mse) : 99.0;
    return {mae, psnr, GraySsim(original, generated, mask)};
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <original-face> <frames-dir> <report-dir>\n";
        return 1;
    }
    const fs::path original_path = argv[1];
    const fs::path frames_dir = argv[2];
    const fs::path report_dir = argv[3];
    const cv::Mat original = cv::imread(original_path.string(), cv::IMREAD_COLOR);
    if (original.empty()) {
        std::cerr << "Cannot read original face: " << original_path << "\n";
        return 2;
    }

    std::vector<fs::path> frames;
    for (const auto& entry : fs::directory_iterator(frames_dir)) {
        if (entry.path().extension() == ".jpg") frames.push_back(entry.path());
    }
    std::sort(frames.begin(), frames.end());
    if (frames.empty()) {
        std::cerr << "No output frames in " << frames_dir << "\n";
        return 3;
    }

    // Same ROI emitted by pipeline_lipsync_test for assets/face.jpg.
    const cv::Rect mouth_roi(497, 723, 257, 195);
    const cv::Rect expanded_roi = ExpandRect(mouth_roi, original.size(), 1.5);
    cv::Mat mouth_mask(original.rows, original.cols, CV_8U, cv::Scalar(0));
    mouth_mask(mouth_roi).setTo(255);
    cv::Mat outside_mask(original.rows, original.cols, CV_8U, cv::Scalar(255));
    outside_mask(expanded_roi).setTo(0);

    RegionMetrics mouth_total, outside_total;
    double temporal_mae_total = 0.0;
    double mouth_mae_max = 0.0;
    cv::Mat previous;
    std::map<size_t, cv::Mat> selected;
    const std::vector<size_t> selected_indices = {0, frames.size() / 2, frames.size() - 1};

    for (size_t i = 0; i < frames.size(); ++i) {
        const cv::Mat frame = cv::imread(frames[i].string(), cv::IMREAD_COLOR);
        if (frame.empty() || frame.size() != original.size()) {
            std::cerr << "Invalid generated frame: " << frames[i] << "\n";
            return 4;
        }
        const RegionMetrics mouth = Compare(original, frame, mouth_mask);
        const RegionMetrics outside = Compare(original, frame, outside_mask);
        mouth_total.mae += mouth.mae;
        mouth_total.psnr += mouth.psnr;
        mouth_total.ssim += mouth.ssim;
        outside_total.mae += outside.mae;
        outside_total.psnr += outside.psnr;
        outside_total.ssim += outside.ssim;
        mouth_mae_max = std::max(mouth_mae_max, mouth.mae);
        if (!previous.empty()) {
            cv::Mat diff;
            cv::absdiff(previous(mouth_roi), frame(mouth_roi), diff);
            const cv::Scalar temporal = cv::mean(diff);
            temporal_mae_total += (temporal[0] + temporal[1] + temporal[2]) / 3.0;
        }
        previous = frame;
        if (std::find(selected_indices.begin(), selected_indices.end(), i) != selected_indices.end()) {
            selected.emplace(i, frame(mouth_roi).clone());
        }
    }

    const double count = static_cast<double>(frames.size());
    mouth_total.mae /= count;
    mouth_total.psnr /= count;
    mouth_total.ssim /= count;
    outside_total.mae /= count;
    outside_total.psnr /= count;
    outside_total.ssim /= count;
    const double temporal_mae = frames.size() > 1 ? temporal_mae_total / (count - 1.0) : 0.0;

    std::vector<cv::Mat> tiles = {original(mouth_roi).clone()};
    for (const size_t index : selected_indices) tiles.push_back(selected.at(index));
    for (size_t i = 0; i < tiles.size(); ++i) {
        cv::putText(tiles[i], i == 0 ? "original" : "generated", cv::Point(8, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    cv::Mat comparison;
    cv::hconcat(tiles, comparison);
    fs::create_directories(report_dir);
    cv::imwrite((report_dir / "mouth_quality_comparison.png").string(), comparison);

    std::ofstream report(report_dir / "mouth_quality.json");
    report << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"frames\": " << frames.size() << ",\n"
           << "  \"mouth_roi\": [497, 723, 257, 195],\n"
           << "  \"mouth_mae\": " << mouth_total.mae << ",\n"
           << "  \"mouth_mae_max\": " << mouth_mae_max << ",\n"
           << "  \"mouth_psnr_db\": " << mouth_total.psnr << ",\n"
           << "  \"mouth_gray_ssim\": " << mouth_total.ssim << ",\n"
           << "  \"outside_expanded_mouth_mae\": " << outside_total.mae << ",\n"
           << "  \"outside_expanded_mouth_psnr_db\": " << outside_total.psnr << ",\n"
           << "  \"outside_expanded_mouth_gray_ssim\": " << outside_total.ssim << ",\n"
           << "  \"adjacent_mouth_temporal_mae\": " << temporal_mae << "\n"
           << "}\n";

    report.close();
    std::cout << "frames=" << frames.size()
              << " mouth_mae=" << mouth_total.mae
              << " outside_mae=" << outside_total.mae
              << " mouth_ssim=" << mouth_total.ssim
              << " temporal_mae=" << temporal_mae << "\n";
    return 0;
}
