---
name: code-style
description: digital-human-sdk 项目编码规范 — 在编写或评审代码时，遵循本规范确保风格一致
---

# digital-human-sdk 编码规范

本文档定义了 `digital-human-sdk` 项目的编码约定。所有新增代码应遵循此规范，代码评审时应以此为标准。

---

## 1. 通用规则

| 规则 | 约定 |
|------|------|
| 缩进 | 4 个空格，**不使用 Tab** |
| 行尾 | LF（Unix），Git 自动转换 CRLF → LF |
| 编码 | UTF-8 |
| 最大行宽 | 120 字符（软限制） |

---

## 2. 命名规范

### 2.1 命名空间
小写 + 下划线，两级嵌套，尾部注释标命名空间名。

```cpp
namespace digital_human {      // 外层：项目名
namespace audio {              // 内层：模块名
// ...
}  // namespace audio
}  // namespace digital_human
```

**模块名对照：**
| 内层命名空间 | 对应目录 |
|-------------|---------|
| `audio`     | `src/audio/`, `include/audio/` |
| `core`      | `src/core/`, `include/core/` |
| `model`     | `src/model/`, `include/model/` |
| `video`     | `src/video/`, `include/video/` |
| `utils`     | `src/utils/`, `include/utils/` |

### 2.2 类名
**PascalCase**，名词或名词短语。

```cpp
class ModelLoader { ... };
class RingBuffer { ... };
class VoiceActivityDetector { ... };
class PreEmphasis { ... };
class InputProcessor { ... };
```

### 2.3 公有方法
**PascalCase**，动词开头。

```cpp
void LoadAsync(const std::string& model_dir, LoadCallback callback);
bool IsLoaded() const;
ncnn::Mat MelToMat(const std::vector<float>& mel_features, int mel_bins, int frames);
```

**例外** — 对于非常底层的"数据流"类（如 RingBuffer、音频处理器的 `process`/`filter`），允许全小写：
```cpp
size_t write(const float* data, size_t count);
size_t read(float* data, size_t count);
void reset();
```

### 2.4 私有/Impl 方法
**camelCase**，在 Impl 结构体中定义。

```cpp
struct ModelLoader::Impl {
    void derivePaths(const std::string& model_dir, ...);
    bool verifyFiles(const std::string& param_path, const std::string& bin_path);
    void doLoad(const std::string& param_path, ...);
};
```

### 2.5 局部变量和参数
**snake_case**。

```cpp
void SetWarmupShapes(int audio_w, int audio_h, int audio_c,
                     int face_w, int face_h, int face_c);
```
```cpp
size_t pos = static_cast<size_t>(r % capacity);
float energy = 0.0f;
int crossings = 0;
```

### 2.6 成员变量
- **主类成员**：`trailing_underscore_`
- **Impl 结构体成员**：plain snake_case（无后缀）

```cpp
// 主类（头文件）
class Foo {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Impl 结构体（实现文件）
struct Foo::Impl {
    ncnn::Net net;
    std::atomic<bool> is_loaded{false};
    float io_cost_ms = 0.0f;
    int audio_w = 80;
};
```

### 2.7 常量
**`k` 前缀 + PascalCase**，`static constexpr`。

```cpp
static constexpr float kEps = 1e-10f;
static constexpr int kFrameSize = 512;
static constexpr int kHopSize = 128;
static constexpr const char* kModelName = "Wav2Lip-SD-GAN-opt";
static constexpr int kDefaultFaceSize = 96;
```

---

## 3. 文件结构

### 3.1 头文件（`.h`）
```
#pragma once                          ← #pragma once 独占

#include <vector>                     ← 标准库头文件
#include <memory>
#include <opencv2/core.hpp>            ← 第三方库头文件
#include <ncnn/net.h>

namespace digital_human {              ← 命名空间
namespace model {

class ClassName {
public:
    ClassName();                        ← 构造/析构
    ~ClassName();
    ClassName(const ClassName&) = delete;       ← 禁止拷贝
    ClassName& operator=(const ClassName&) = delete;
    ClassName(ClassName&&) noexcept;            ← 允许移动
    ClassName& operator=(ClassName&&) noexcept;

    // ---- 公有 API ----
    ncnn::Mat MethodName(params);

private:
    struct Impl;                        ← PIMPL 前向声明
    std::unique_ptr<Impl> impl_;
};

}  // namespace model
}  // namespace digital_human
```

### 3.2 实现文件（`.cpp`）
```cpp
#include "model/class_name.h"           ← 自己的头文件优先

#include <cmath>                        ← 标准库
#include <iostream>
#include <algorithm>
#include <cstring>

namespace digital_human {
namespace model {

struct ClassName::Impl {                ← Impl 结构体定义
    // 实现细节
};

ClassName::ClassName()                   ← 构造
    : impl_(std::make_unique<Impl>()) {}

ClassName::~ClassName() = default;       ← 析构在 .cpp 中 default

ClassName::ClassName(ClassName&&) noexcept = default;
ClassName& ClassName::operator=(ClassName&&) noexcept = default;

ncnn::Mat ClassName::MethodName(params) {  ← 公有方法委托
    return impl_->methodName(params);
}

}  // namespace model
}  // namespace digital_human
```

### 3.3 测试文件（`examples/`）
```cpp
#include <iostream>
#include <vector>
#include "module/header.h"

using namespace digital_human::module;   // 测试文件可用 using namespace

// ==========================================
// Test N: 测试描述
// ==========================================
static void testFeature() {
    // 准备
    // 执行
    // 验证
    std::cout << (cond ? "  [PASS]" : "  [FAIL]") << " 描述" << std::endl;
}
```

---

## 4. PIMPL 模式规范

所有公开类必须使用 PIMPL 模式。

### 4.1 头文件声明
```cpp
class ClassName {
public:
    ClassName();
    ~ClassName();
    ClassName(const ClassName&) = delete;
    ClassName& operator=(const ClassName&) = delete;
    ClassName(ClassName&&) noexcept;
    ClassName& operator=(ClassName&&) noexcept;

    // ... 公有 API ...

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

### 4.2 实现文件
```cpp
struct ClassName::Impl {
    // 所有私有状态和方法放在这里
};

ClassName::ClassName() : impl_(std::make_unique<Impl>()) {}
ClassName::~ClassName() = default;
ClassName::ClassName(ClassName&&) noexcept = default;
ClassName& ClassName::operator=(ClassName&&) noexcept = default;

// 公有方法委托给 impl_
ncnn::Mat ClassName::MethodName(params) {
    return impl_->methodName(params);
}
```

### 4.3 重要规则
- 析构函数必须在 `.cpp` 中 `= default`（不能 inline，因为 Impl 是不完整类型）
- 禁止拷贝构造和拷贝赋值（`= delete`）
- 允许移动构造和移动赋值（`= default`）
- `impl_` 是主类**唯一**的成员变量

---

## 5. 注释规范

### 5.1 注释语言
- **公共 API 和复杂逻辑**：中文（本项目以中文开发者为主）
- **简单内部逻辑**：英文短注释
- **CMakeLists.txt**：中文

### 5.2 公有 API 文档
```cpp
/**
 * @brief 将梅尔频谱特征数据转换为 ncnn::Mat 格式
 *
 * 详细说明数据处理流程和注意事项。
 *
 * @param mel_features 梅尔频谱数据，长度为 mel_bins * frames
 * @param mel_bins     梅尔滤波器组数量
 * @param frames       时间帧数
 * @return ncnn::Mat  形状为 (w=mel_bins, h=frames, c=1)
 */
```

### 5.3 内部注释
```cpp
// ---- 阶段分隔 ----
// 简单说明代码意图
int value = compute();  // 行尾注释
```

### 5.4 节分隔格式
```
// ==================== 一级分隔（区域边界） ====================
// ---- 二级分隔（阶段/步骤） ----
// ==========================================
// Test N: 测试名称（测试文件中）
// ==========================================
```

---

## 6. 错误处理

### 6.1 输入验证失败 → 抛异常
```cpp
if (data.empty()) {
    throw std::invalid_argument("[ModuleName] method: 错误描述");
}
```

### 6.2 运行时失败 → cerr 日志 + 返回空结果
```cpp
if (image.empty()) {
    std::cerr << "[ModuleName] 错误描述" << std::endl;
    return {};  // 或 cv::Mat(), nullptr
}
```

### 6.3 回调安全
```cpp
if (callback) callback(result);
```

### 6.4 参数范围约束
```cpp
impl_->alpha = std::clamp(alpha, 0.0f, 1.0f);
impl_->hangover = std::max(0, hangover);
```

---

## 7. Include 规范

### 7.1 顺序
```
1. 自己的头文件（.cpp 中第一行）： #include "module/header.h"
2. C++ 标准库：                        <vector> <memory> <cmath>
3. OpenCV 头文件：                     <opencv2/core.hpp> <opencv2/imgproc.hpp>
4. ncnn 头文件：                       <ncnn/net.h> <ncnn/mat.h>
5. 其他第三方库：                      dlib, FFmpeg 等
```

### 7.2 格式
- 项目头文件用引号 `""`，路径相对于 `include/`
- 标准库和第三方用尖括号 `<>`

---

## 8. 内存与资源管理

### 8.1 智能指针
- 动态对象优先使用 `std::unique_ptr`（独占所有权）
- PIMPL 模式强制用 `std::unique_ptr<Impl>`
- **不使用裸 `new`/`delete`**

### 8.2 RAII
- 使用 `std::thread` 并在析构中 join
- 使用 `std::lock_guard` 或原子操作保护共享状态

---

## 9. 常量与字面量

### 9.1 浮点数后缀
```cpp
float val = 0.0f;       // float 强制加 f 后缀
double val = 0.0;       // double 不加后缀
```

### 9.2 魔术数字
- 模块内部常量放 `Impl` 的 `static constexpr` 中
- 跨模块常量放在头文件的类作用域内

---

## 10. 现代 C++ 特性

### 10.1 auto
- **允许**：范围 for 循环 `for (const auto& item : container)`
- **允许**：lambda 参数
- **允许**：`make_unique`/`make_shared` 返回值
- **避免**：函数返回类型用 auto
- **避免**：变量类型不清晰时用 auto

### 10.2 nullptr
- 空指针**必须**用 `nullptr`
- 禁止 `NULL`，禁止 `0` 作为空指针

### 10.3 using
- 头文件：允许类型别名 `using Callback = std::function<void(...)>;`
- 实现文件：允许 namespace 别名 `namespace fs = std::filesystem;`
- 实现文件：允许 `using namespace digital_human::audio;`
- **禁止** `using namespace std;`

### 10.4 const 正确性
- 不修改成员的方法必须标记 `const`
- 引用参数尽可能用 `const&`
- 按值传参适用于小对象和会移动的对象

---

## 11. 测试规范

### 11.1 文件位置
- 测试文件放在 `examples/` 目录下
- 测试文件命名：`<module_name>_test.cpp`

### 11.2 测试结构
```cpp
// ==========================================
// Test N: 测试描述
// ==========================================
static void testFeature() {
    // 准备测试数据
    // 调用被测试方法
    // 验证结果
    TEST_CHECK(condition, "描述");
}
```

### 11.3 测试覆盖要求
- 正常输入（Happy path）
- 边界输入（空数据、零值、极值）
- 异常输入（非法参数、不匹配尺寸）
- 批量处理验证（数据隔离、通道一致性）

---

## 12. Git 提交规范

### 12.1 提交信息格式
```
<type>(<scope>): <subject>

<body>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
```

### 12.2 type 类型
| type       | 用途 |
|------------|------|
| `feat`     | 新功能 |
| `fix`      | Bug 修复 |
| `refactor` | 代码重构（不改功能） |
| `docs`     | 文档更新 |
| `test`     | 添加/修改测试 |
| `chore`    | 构建/工具/依赖 |
| `style`    | 代码风格调整（无功能变化） |

### 12.3 scope
- `model` — 模型相关
- `audio` — 音频处理
- `core` — 核心处理
- `video` — 视频处理
- `utils` — 工具函数

### 12.4 文件变更尾部
```
7 files changed, 1189 insertions(+), 7 deletions(-)
```

---

## 13. 禁止模式

- ❌ **裸 `new`/`delete`** — 使用智能指针或容器
- ❌ **`NULL` 或 `0` 代替空指针** — 使用 `nullptr`
- ❌ **`using namespace std`** — 名称空间污染
- ❌ **头文件中定义非内联函数** — 链接重复定义
- ❌ **全局变量** — 使用类封装状态
- ❌ **C 风格数组** — 使用 `std::vector` 或 `std::array`
- ❌ **`#ifndef` 头文件守卫** — 统一用 `#pragma once`
- ❌ **`throw()` 动态异常规范** — C++17 已移除

---

## 14. 代码评审检查清单

- [ ] 命名符合规范（命名空间、类、方法、变量、常量）
- [ ] PIMPL 模式正确（构造/析构/拷贝/移动）
- [ ] `const` 正确性（方法、参数、引用）
- [ ] 所有路径的错误处理（日志/异常/返回值）
- [ ] 智能指针无裸 `new`/`delete`
- [ ] 包含头文件顺序正确
- [ ] 注释完整且与实现一致
- [ ] 无 `using namespace std`
- [ ] 无全局变量
- [ ] 测试覆盖正常/边界/异常场景
- [ ] 批量处理验证了数据隔离

---

## 附录：一致性记录

当前代码库中存在的不一致项（新代码应避免）：

1. 命名空间大小写：`DigitalHuman::core` 过时，应使用 `digital_human::core`
2. Impl 结构体命名：统一使用 `struct ClassName::Impl`，避免 `FaceDetectorImpl` 风格
3. 音频处理器的 `process()`/`filter()` 方法保持全小写以保持命名一致性
4. 测试文件中函数命名保持 snake_case 风格
