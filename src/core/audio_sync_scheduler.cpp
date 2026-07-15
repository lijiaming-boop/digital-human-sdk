#include "core/audio_sync_scheduler.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "audio/audio_player.h"

namespace digital_human {
namespace core {

using audio::AudioPlayer;
using audio::AudioPlayerState;

// ============================================================================
// Impl 结构体
// ============================================================================

struct AudioSyncScheduler::Impl {
    // ---- 组合模块 ----
    AudioPlayer      audio_player;           ///< 音频播放器
    AVSync           av_sync;                ///< 音视频同步判定
    FrameScheduler   frame_scheduler;        ///< 帧调度器

    // ---- 配置 ----
    AudioSyncConfig  config;
    bool             initialized   = false;
    bool             audio_loaded  = false;

    // ---- 播放状态 ----
    PlaybackState    play_state    = PlaybackState::STOPPED;

    // ---- 音频帧追踪 ----
    int64_t          total_audio_frames = 0; ///< 音频总帧数
    int64_t          last_consumed_frames = 0; ///< 上次采样时的消耗帧数

    // ---- 上一帧同步结果缓存 ----
    SyncResult       last_sync_result;
    bool             has_sync_result = false;

    // ========================================================================
    // 内部方法
    // ========================================================================

    /**
     * @brief 从 AudioPlayer 获取当前音频时钟并更新 AVSync
     *
     * 将 AudioPlayer 的已消耗帧数增量转换为样本数，更新到 AVSync 的音频时钟。
     * 在 Reset/Stop/Play 循环中自动处理计数归零。
     */
    void updateAudioClock() {
        if (!initialized || !audio_loaded) return;

        int64_t consumedFrames = audio_player.GetConsumedFrames();

        // 计算增量：如果出现回退（重置/重新播放），直接用当前值
        int64_t deltaFrames;
        if (consumedFrames < last_consumed_frames) {
            deltaFrames = consumedFrames;  // 重置后从 0 开始
        } else {
            deltaFrames = consumedFrames - last_consumed_frames;
        }
        last_consumed_frames = consumedFrames;

        if (deltaFrames > 0) {
            int64_t samplesConsumed = deltaFrames * config.audio_channels;
            av_sync.UpdateAudioClock(samplesConsumed);
        }
    }

    /// @brief 生成初始化状态的 SyncConfig
    SyncConfig makeSyncConfig() const {
        SyncConfig sc;
        sc.audio_sample_rate = config.audio_sample_rate;
        sc.sync_threshold_ms = config.sync_threshold_ms;
        sc.max_drift_ms      = config.max_drift_ms;
        return sc;
    }

    /// @brief 生成初始化状态的 SchedulerConfig
    SchedulerConfig makeSchedulerConfig() const {
        SchedulerConfig sc;
        sc.target_fps        = config.target_fps;
        sc.smoothing_factor  = 0.5;
        sc.enable_smoothing  = true;
        sc.max_pending_frames = 10;
        return sc;
    }
};

// ============================================================================
// 构造 / 析构
// ============================================================================

AudioSyncScheduler::AudioSyncScheduler()
    : impl_(std::make_unique<Impl>()) {}

AudioSyncScheduler::~AudioSyncScheduler() {
    Destroy();
}

AudioSyncScheduler::AudioSyncScheduler(AudioSyncScheduler&&) noexcept = default;

AudioSyncScheduler& AudioSyncScheduler::operator=(AudioSyncScheduler&&) noexcept = default;

// ============================================================================
// 初始化
// ============================================================================

bool AudioSyncScheduler::Init(const AudioSyncConfig& config) {
    if (impl_->initialized) {
        std::cerr << "[AudioSyncScheduler] Init: 已初始化，请先调用 Destroy()" << std::endl;
        return false;
    }

    // 校验参数
    if (config.audio_sample_rate <= 0) {
        std::cerr << "[AudioSyncScheduler] Init: 无效的 audio_sample_rate ("
                  << config.audio_sample_rate << ")" << std::endl;
        return false;
    }
    if (config.audio_channels <= 0 || config.audio_channels > 2) {
        std::cerr << "[AudioSyncScheduler] Init: 无效的 audio_channels ("
                  << config.audio_channels << ")" << std::endl;
        return false;
    }
    if (config.target_fps <= 0.0) {
        std::cerr << "[AudioSyncScheduler] Init: 无效的 target_fps ("
                  << config.target_fps << ")" << std::endl;
        return false;
    }

    impl_->config = config;

    // 初始化 AudioPlayer
    if (!impl_->audio_player.Init(
            config.audio_sample_rate,
            config.audio_channels,
            config.audio_frames_per_buffer)) {
        std::cerr << "[AudioSyncScheduler] Init: AudioPlayer 初始化失败: "
                  << impl_->audio_player.GetLastErrorMsg() << std::endl;
        return false;
    }

    // 初始化 AVSync
    impl_->av_sync.Init(impl_->makeSyncConfig());

    // 初始化 FrameScheduler
    impl_->frame_scheduler.Init(impl_->makeSchedulerConfig());

    impl_->initialized = true;
    impl_->play_state  = PlaybackState::STOPPED;

    std::cout << "[AudioSyncScheduler] 初始化成功: "
              << config.audio_sample_rate << "Hz "
              << config.audio_channels << "ch "
              << config.target_fps << "fps"
              << std::endl;

    return true;
}

bool AudioSyncScheduler::IsInitialized() const {
    return impl_->initialized;
}

void AudioSyncScheduler::Destroy() {
    if (!impl_->initialized) return;

    // 停止播放
    if (impl_->play_state != PlaybackState::STOPPED) {
        impl_->audio_player.Stop();
    }

    // 销毁 AudioPlayer
    impl_->audio_player.Destroy();

    // 重置 AVSync 和 FrameScheduler
    impl_->av_sync.Reset();
    impl_->frame_scheduler.Reset();

    impl_->initialized   = false;
    impl_->audio_loaded  = false;
    impl_->play_state    = PlaybackState::STOPPED;
    impl_->total_audio_frames     = 0;
    impl_->last_consumed_frames   = 0;
    impl_->has_sync_result        = false;
}

// ============================================================================
// 音频数据加载
// ============================================================================

bool AudioSyncScheduler::LoadAudio(const float* samples, int numSamples, int channels) {
    if (!impl_->initialized) {
        std::cerr << "[AudioSyncScheduler] LoadAudio: 未初始化" << std::endl;
        return false;
    }
    if (!samples || numSamples <= 0) {
        std::cerr << "[AudioSyncScheduler] LoadAudio: 无效参数" << std::endl;
        return false;
    }

    if (!impl_->audio_player.LoadAudio(samples, numSamples, channels)) {
        std::cerr << "[AudioSyncScheduler] LoadAudio: AudioPlayer 加载失败: "
                  << impl_->audio_player.GetLastErrorMsg() << std::endl;
        return false;
    }

    impl_->total_audio_frames = numSamples / channels;
    impl_->audio_loaded = true;

    // 重置 AVSync 音频时钟
    impl_->av_sync.Reset();
    impl_->frame_scheduler.Reset();
    impl_->has_sync_result = false;

    // 同步帧调度器的帧率到音频时长
    // 总视频帧数 = 音频时长 / 帧间隔
    double audioDurationMs = impl_->audio_player.GetTotalDurationMs();
    double frameIntervalMs = 1000.0 / impl_->config.target_fps;
    int64_t totalVideoFrames = static_cast<int64_t>(audioDurationMs / frameIntervalMs);

    std::cout << "[AudioSyncScheduler] 音频已加载: "
              << numSamples << " samples, "
              << channels << "ch, "
              << audioDurationMs << "ms (约 "
              << totalVideoFrames << " 视频帧 @ "
              << impl_->config.target_fps << "fps)"
              << std::endl;

    return true;
}

bool AudioSyncScheduler::LoadAudio(const std::vector<float>& samples, int channels) {
    return LoadAudio(samples.data(), static_cast<int>(samples.size()), channels);
}

// ============================================================================
// 播放控制
// ============================================================================

bool AudioSyncScheduler::Play() {
    if (!impl_->initialized) {
        std::cerr << "[AudioSyncScheduler] Play: 未初始化" << std::endl;
        return false;
    }
    if (!impl_->audio_loaded) {
        std::cerr << "[AudioSyncScheduler] Play: 未加载音频数据" << std::endl;
        return false;
    }

    if (impl_->play_state == PlaybackState::PLAYING) {
        return true;
    }

    // 如果从 STOPPED 启动，重置所有状态
    if (impl_->play_state == PlaybackState::STOPPED) {
        impl_->av_sync.Reset();
        impl_->frame_scheduler.Reset();
        impl_->has_sync_result      = false;
        impl_->last_consumed_frames = 0;
    }

    // 启动 AudioPlayer
    if (!impl_->audio_player.Play()) {
        std::cerr << "[AudioSyncScheduler] Play: AudioPlayer 播放失败"
                  << std::endl;
        return false;
    }

    impl_->play_state = PlaybackState::PLAYING;
    std::cout << "[AudioSyncScheduler] 开始播放" << std::endl;
    return true;
}

bool AudioSyncScheduler::Pause() {
    if (!impl_->initialized) {
        std::cerr << "[AudioSyncScheduler] Pause: 未初始化" << std::endl;
        return false;
    }
    if (impl_->play_state != PlaybackState::PLAYING) {
        std::cerr << "[AudioSyncScheduler] Pause: 当前不在播放状态" << std::endl;
        return false;
    }

    if (!impl_->audio_player.Pause()) {
        std::cerr << "[AudioSyncScheduler] Pause: AudioPlayer 暂停失败"
                  << std::endl;
        return false;
    }

    impl_->play_state = PlaybackState::PAUSED;
    std::cout << "[AudioSyncScheduler] 已暂停" << std::endl;
    return true;
}

bool AudioSyncScheduler::Resume() {
    if (!impl_->initialized) {
        std::cerr << "[AudioSyncScheduler] Resume: 未初始化" << std::endl;
        return false;
    }
    if (impl_->play_state != PlaybackState::PAUSED) {
        std::cerr << "[AudioSyncScheduler] Resume: 当前不在暂停状态" << std::endl;
        return false;
    }

    if (!impl_->audio_player.Resume()) {
        std::cerr << "[AudioSyncScheduler] Resume: AudioPlayer 恢复失败"
                  << std::endl;
        return false;
    }

    impl_->play_state = PlaybackState::PLAYING;
    std::cout << "[AudioSyncScheduler] 恢复播放" << std::endl;
    return true;
}

bool AudioSyncScheduler::Stop() {
    if (!impl_->initialized) {
        std::cerr << "[AudioSyncScheduler] Stop: 未初始化" << std::endl;
        return false;
    }
    if (impl_->play_state == PlaybackState::STOPPED) {
        return true;
    }

    impl_->audio_player.Stop();
    impl_->av_sync.Reset();
    impl_->frame_scheduler.Reset();
    impl_->play_state = PlaybackState::STOPPED;
    impl_->has_sync_result = false;

    std::cout << "[AudioSyncScheduler] 已停止" << std::endl;
    return true;
}

PlaybackState AudioSyncScheduler::GetPlaybackState() const {
    return impl_->play_state;
}

// ============================================================================
// 视频帧调度
// ============================================================================

ScheduleResult AudioSyncScheduler::ScheduleFrame(int frame_id, double video_pts_ms) {
    // 未初始化或未播放时直接 DISPLAY
    if (!impl_->initialized || impl_->play_state == PlaybackState::STOPPED) {
        ScheduleResult fallback;
        fallback.action   = FrameAction::DISPLAY;
        fallback.frame_id = frame_id;
        fallback.pts_ms   = video_pts_ms;
        return fallback;
    }

    // 暂停期间也返回 DISPLAY（保持最后一帧）
    if (impl_->play_state == PlaybackState::PAUSED) {
        ScheduleResult pauseResult;
        pauseResult.action   = FrameAction::DISPLAY;
        pauseResult.frame_id = frame_id;
        pauseResult.pts_ms   = video_pts_ms;
        return pauseResult;
    }

    // ---- 步骤1: 更新音频时钟 ----
    impl_->updateAudioClock();
    double audioClockMs = impl_->av_sync.GetAudioClockMs();
    (void)audioClockMs;

    // ---- 步骤2: 获取同步判定 ----
    SyncResult syncResult = impl_->av_sync.GetSyncStatus(video_pts_ms);
    impl_->last_sync_result = syncResult;
    impl_->has_sync_result  = true;

    // ---- 步骤3: 根据同步状态调整帧调度 ----
    ScheduleResult frameResult;
    frameResult.frame_id = frame_id;
    frameResult.pts_ms   = video_pts_ms;

    switch (syncResult.status) {
        case SyncStatus::SYNCED:
            // 同步正常：使用 FrameScheduler 做标准调度
            frameResult = impl_->frame_scheduler.ScheduleFrame(frame_id, video_pts_ms);
            break;

        case SyncStatus::VIDEO_AHEAD:
            // 视频超前于音频 → 重复上一帧（等待音频追赶）
            frameResult.action         = FrameAction::DUPLICATE;
            frameResult.scheduled_pts_ms = syncResult.audio_clock_ms;
            frameResult.wait_time_ms   = syncResult.wait_time_ms;
            break;

        case SyncStatus::VIDEO_BEHIND:
            // 视频滞后于音频 → 丢弃帧（快速追赶）
            frameResult.action         = FrameAction::DROP;
            frameResult.scheduled_pts_ms = syncResult.audio_clock_ms;
            break;

        case SyncStatus::SEVERE_OFFSET:
            // 严重偏移 → 丢弃帧（强制同步）
            frameResult.action         = FrameAction::DROP;
            frameResult.scheduled_pts_ms = syncResult.audio_clock_ms;
            break;
    }

    return frameResult;
}

void AudioSyncScheduler::OnFrameDisplayed(double actual_display_time_ms) {
    if (!impl_->initialized) return;
    impl_->frame_scheduler.OnFrameDisplayed(actual_display_time_ms);
}

// ============================================================================
// 同步信息查询
// ============================================================================

double AudioSyncScheduler::GetDriftMs() const {
    if (!impl_->has_sync_result) return 0.0;
    return impl_->last_sync_result.drift_ms;
}

double AudioSyncScheduler::GetAudioClockMs() const {
    return impl_->av_sync.GetAudioClockMs();
}

SyncStatus AudioSyncScheduler::GetSyncStatus() const {
    if (!impl_->has_sync_result) return SyncStatus::SYNCED;
    return impl_->last_sync_result.status;
}

FrameStats AudioSyncScheduler::GetFrameStats() const {
    return impl_->frame_scheduler.GetStats();
}

std::string AudioSyncScheduler::GetStatsString() const {
    std::ostringstream oss;
    oss << "[AudioSyncScheduler 统计]" << std::endl;

    // 音频信息
    oss << "  音频时钟: " << GetAudioClockMs() << " ms";
    if (impl_->audio_loaded) {
        oss << " / " << impl_->audio_player.GetTotalDurationMs() << " ms";
    }
    oss << std::endl;

    // 同步信息
    if (impl_->has_sync_result) {
        oss << "  同步状态: " << impl_->last_sync_result.StatusString() << std::endl;
        oss << "  偏移量: " << impl_->last_sync_result.drift_ms << " ms" << std::endl;
    }

    // 帧统计
    oss << "  帧调度: " << impl_->frame_scheduler.GetStats().ToString() << std::endl;

    // 播放状态
    oss << "  播放状态: ";
    switch (impl_->play_state) {
        case PlaybackState::STOPPED:  oss << "STOPPED";  break;
        case PlaybackState::PLAYING:  oss << "PLAYING";  break;
        case PlaybackState::PAUSED:   oss << "PAUSED";   break;
    }
    oss << std::endl;

    return oss.str();
}

}  // namespace core
}  // namespace digital_human
