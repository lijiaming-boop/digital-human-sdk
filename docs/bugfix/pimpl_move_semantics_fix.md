# PIMPL 模式 Move 语义空指针崩溃修复

## 问题描述

在实现 `AudioPlayer` 和 `AudioSyncScheduler` 模块时，使用 `= default` 声明移动构造函数和移动赋值运算符，导致在 `std::move` 操作后源对象析构时发生 **Segmentation Fault**。

同时，在无音频设备的 WSL2 环境中，`Pa_Terminate()` 调用触发 PortAudio JACK 后端空指针解引用，造成进程退出时二次崩溃。

## 根因分析

### 问题1：PIMPL 移动语义

```cpp
// 错误写法
class AudioPlayer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

AudioPlayer::AudioPlayer(AudioPlayer&&) noexcept = default;
```

`std::unique_ptr` 的移动构造会将源指针置 `nullptr`，这是正确的所有权转移语义。但析构函数调用了 `Destroy()`，而 `Destroy()` 的实现未考虑 `impl_` 为 `nullptr` 的情况：

```cpp
// 崩溃路径
AudioPlayer::~AudioPlayer() {
    Destroy();   // 此时 impl_ 可能为 nullptr
}

void AudioPlayer::Destroy() {
    if (!impl_->initialized) return;  // ❌ 空指针解引用！
    // ...
}
```

### 问题2：`= default` 的隐藏陷阱

`= default` 的移动构造仅逐成员移动 `impl_`（即移动 `unique_ptr`），但不会将源对象的 `impl_` 显式置空（虽然 `unique_ptr` 的移动操作本身会置空源指针）。这里的问题是：

1. `unique_ptr` 被移动 → 源对象 `impl_` 为 `nullptr`
2. 编译器合成的 `= default` 不会执行任何额外的清理逻辑
3. 源对象析构时，`Destroy()` 试图访问 `impl_->initialized` → 段错误

### 问题3：PortAudio JACK 后端空指针（WSL2 无音频设备环境）

在无 ALSA 音频设备的 WSL2 Ubuntu 中，PortAudio 的 JACK 音频后端在进程退出时尝试搜索 `jackd` 可执行文件，期间发生空指针解引用（`si_addr=0x40`，即访问结构体偏移 64 字节处的成员）。

## 修复方案

### 修复1：显式实现移动构造/赋值

```cpp
// 正确写法
AudioPlayer::AudioPlayer(AudioPlayer&& other) noexcept
    : impl_(std::move(other.impl_)) {
    other.impl_ = nullptr;  // 显式置空
}

AudioPlayer& AudioPlayer::operator=(AudioPlayer&& other) noexcept {
    if (this != &other) {
        Destroy();                     // 清理当前资源
        impl_ = std::move(other.impl_);
        other.impl_ = nullptr;         // 显式置空
    }
    return *this;
}
```

### 修复2：Destroy() 增加空指针守卫

```cpp
void AudioPlayer::Destroy() {
    if (!impl_ || !impl_->initialized) return;  // ✅ 双重检查
    // ...
}
```

### 修复3：无音频设备时跳过 Pa_Terminate

```cpp
static std::atomic<bool> gPaDeviceReady{false};

// 仅在成功打开音频流后标记
bool AudioPlayer::Init(...) {
    // ...
    if (err == paNoError) {
        gPaDeviceReady.store(true, std::memory_order_release);
        impl_->initialized = true;
        return true;
    }
    // ...
}

void AudioPlayer::Destroy() {
    // ...
    if (--gPaRefCount == 0 && gPaDeviceReady.load(std::memory_order_acquire)) {
        Pa_Terminate();
        gPaDeviceReady.store(false, std::memory_order_release);
    }
}
```

## 受影响模块

| 模块 | 文件 | 修复内容 |
|------|------|----------|
| `AudioPlayer` | `src/audio/audio_player.cpp` | 显式移动构造/赋值、`Destroy()` 空指针守卫、`gPaDeviceReady` 保护 |
| `AudioSyncScheduler` | `src/core/audio_sync_scheduler.cpp` | 显式移动构造/赋值、`Destroy()` 空指针守卫 |

## 经验教训

1. **PIMPL 模式禁止使用 `= default` 移动构造** — `unique_ptr<Impl>` 的移动操作置空源指针后，析构函数仍会尝试访问源对象，必须显式实现移动构造/赋值并处理源对象状态。

2. **`Destroy()` 方法需考虑部分构造状态** — 当对象在初始化完成前被析构（Move 后源对象、Init 失败等场景），`Destroy()` 必须能够安全处理 `impl_` 为 `nullptr` 的情况。

3. **第三方库的退出清理不可信赖** — PortAudio 在无音频硬件的环境中（WSL2、容器、CI）调用 `Pa_Terminate()` 存在崩溃风险。应使用防护标志延迟或跳过清理调用，让操作系统回收资源。
