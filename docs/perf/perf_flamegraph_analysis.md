# 用 perf + 火焰图定位 digital-human-sdk 性能瓶颈

> 记录一次完整的 perf 采样调试：从环境搭建、踩坑、火焰图生成到根因定位。
> 目标程序：`perf_benchmark`（face.jpg + zw.mp3 跑完整端到端 Pipeline）。
> 采样时间：2026-07-22，WSL2 Ubuntu-22.04，16 核。

---

## 0. TL;DR — 结论先行

| 模块 | 自占 CPU (self) | 性质 |
|------|----------------|------|
| **libgomp（OpenMP 运行时）** | **60.3%** | ❌ **纯浪费**：线程在 barrier 上忙等自旋 |
| libdigital_human_core | 37.9% | 真实计算 |
|  ├─ ncnn 卷积/GEMM（Winograd） | ~25% | Wav2Lip 推理，合理 |
|  └─ OpenCV 混合/平滑（blend/resize/addWeighted） | ~5% | 人脸合成，可优化 |
| 其它（libc/ffmpeg/…） | <2% | 忽略 |

**核心瓶颈：约 60% 的 CPU 没有在算数，而是在 OpenMP 线程屏障上空转自旋。**

**根因**（两处叠加）：
1. `ModelInferencer::doInit()` 中 `net.opt.num_threads = 2` 设在 `load_model()` **之后**，
   ncnn 的 winograd/gemm 卷积层在 load 时已按默认 **8 线程** 定型，运行期改回 2 无效
   → 运行日志实锤：`opt.num_threads 2 changed, convolution winograd will use load-time value 8`。
2. libgomp 默认 **主动自旋等待**（`GOMP_SPINCOUNT` ~30 万）。Wav2Lip 有大量小卷积层，
   每个 `#pragma omp parallel` 区间结束后空闲线程不休眠、持续自旋 → 16 核上被放大成 60% 的无效占用。

**优化方向**：把线程数在 `load_model()` **之前**设定 + 让空闲线程休眠（`OMP_WAIT_POLICY=passive`）。详见 §6。

---

## 1. 环境与可行性判断（第一步永远是"能不能测"）

宿主是 **Windows**，`perf` 是 Linux 内核工具，不能直接用。排查链路：

```
uname -a            → MINGW64（宿主 MSYS，不是真 Linux）
wsl -l -v           → 有 Ubuntu-22.04 (WSL2, Running)   ✅ 用 WSL 跑 perf
```

WSL2 用的是微软定制内核（`6.18-microsoft-standard-WSL2`），必须确认它编译进了 perf 事件支持：

```bash
cat /proc/sys/kernel/perf_event_paranoid   # → 2
```

* 文件**存在** ⇒ 内核带 `CONFIG_PERF_EVENTS`，`perf_event_open` 可用。
* 值 = `2` ⇒ **非 root 也能采样用户态代码**（正好够画用户态火焰图）。

> 判断准则：`paranoid<=2` 且能拿到自己进程的样本，火焰图就能做；不需要 root、不需要改内核。

---

## 2. 工具准备

| 项 | 命令 | 备注 |
|----|------|------|
| perf | `sudo apt install linux-tools-generic linux-tools-common` | 唯一需要 sudo 的一步 |
| FlameGraph | `git clone https://github.com/brendangregg/FlameGraph ~/FlameGraph` | Brendan Gregg 的折叠+画图脚本 |

**版本不匹配的坑**：apt 装的是 `linux-tools-5.15`，而 WSL 内核是 `6.18`。
`/usr/bin/perf` 包装脚本按 `uname -r` 找不到对应版本会报错。**解决**：直接调实体二进制：

```bash
PERF=$(ls /usr/lib/linux-tools/*/perf | sort -V | tail -1)   # /usr/lib/linux-tools/5.15.0-186-generic/perf
```
用户态采样对内核版本不敏感，5.15 的 perf 采 6.18 内核的用户栈完全可用。

---

## 3. 为采样专门做一个"可观测"构建

Release 构建是 `-O3 -DNDEBUG` 且**省略帧指针**，采样栈会断裂、函数名丢失。
新开一个 `build-profile`，加调试信息 + 保留帧指针：

```bash
cmake --preset vcpkg -B build-profile \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -g"
cmake --build build-profile --target perf_benchmark -j$(nproc)
```

* `RelWithDebInfo`（`-O2 -g`）：保持优化后的真实性能，同时带符号。
* `-fno-omit-frame-pointer`：让 perf 用**帧指针**回栈（见 §4 为什么不用 dwarf）。
* 第三方库（ncnn/opencv）是预编译的、无帧指针，但它们的**叶子热点函数名**仍在符号表里，不影响定位。

---

## 4. 采样：踩了 3 个 WSL2 专属的坑

最终稳定命令（封装在 `scripts/profile_flamegraph.sh`）：

```bash
$PERF record -F 999 --call-graph fp -o perf.data -- <程序>
```

踩坑记录（这部分才是"调试思路"的精华）：

| 现象 | 报错 | 原因 | 解决 |
|------|------|------|------|
| ① dwarf 回栈失败 | `failed to write perf data, error: Bad address` | `--call-graph dwarf` 每次采样要拷贝一段用户栈，WSL2 多线程下触发内核 bug | 改用 `--call-graph fp`（已编译帧指针，无需拷栈） |
| ② mmap 失败 | `Permission error mapping pages ... perf_event_mlock_kb` | 非 root 下 `-m 16M` 超过 `perf_event_mlock_kb=4096` 锁页上限 | 去掉 `-m`，用默认缓冲 |
| ③ 采到 0 样本 | perf.data 只有 552B，但程序明明在跑 | 音频 293s，程序要跑好几分钟，`record` 一直等子进程结束才 flush；后台任务被提前中断 → 没落盘 | 用 `timeout -s INT 25` 采够 25s 后发 SIGINT，perf 收到信号会正常 flush |

采样参数说明：
* `-F 999`：999Hz，避开内核 1000Hz tick 谐振，防止采样偏差。
* `--call-graph fp`：帧指针回栈，WSL2 下比 dwarf 稳。
* 25s 采到 **248,508 个样本**（0 丢失），统计上足够。

---

## 5. 生成火焰图 & 读图

```bash
$PERF script -i perf.data                > perf.script
~/FlameGraph/stackcollapse-perf.pl perf.script > out.folded
~/FlameGraph/flamegraph.pl out.folded    > flamegraph.svg
```

产物（都在 `docs/perf/`）：

| 文件 | 说明 |
|------|------|
| `flamegraph.svg` | 完整火焰图，浏览器打开可交互（点击下钻、Ctrl+F 搜索） |
| `flamegraph_no_gomp.svg` | **剔除 libgomp 自旋**后的火焰图，直接看真实计算分布 |
| `out.folded` | 折叠调用栈（文本，可 grep 统计） |
| `perf.data` | 原始采样数据，可 `perf report` 复查 |

**快速定位命令**（不开图也能看热点）：

```bash
# 按库聚合：谁在吃 CPU
$PERF report -g none --sort dso --stdio
#   60.27%  libgomp.so         ← OpenMP 自旋
#   37.90%  libdigital_human_core.so

# 按函数聚合：真实热点
$PERF report -g none --stdio | head
#   33.51% libgomp 0x20602     ← barrier 自旋地址1
#   25.85% libgomp 0x207ba     ← barrier 自旋地址2
#   16.35% ncnn::gemm_transB_packed_tile  ← Winograd 卷积
#    5.31% ncnn::gemm_transB_packed_tile (gemm_AT_x86)
#    2.04% cv::...vlineSmooth5N            ← OpenCV 高斯/缩放
#    1.54% cv::BlendLinearInvoker          ← 人脸 alpha 混合
```

读图要点：**火焰图宽度 = CPU 时间占比**。libgomp 那两根占了近 60% 宽度的"平顶塔"，
且顶端是 libgomp 内部地址（无业务函数）——这是 barrier 忙等的典型形状。

---

## 6. 根因与优化建议

### 6.1 根因链

```
16 核机器
  └─ ncnn 默认 OpenMP 线程数 = 8
       └─ load_model() 时 winograd/gemm 卷积层按 8 线程"定型/打包"
            └─ 之后 net.opt.num_threads=2 对这些层无效（日志已警告）
                 └─ 每层推理开 8 线程，Wav2Lip 小层多、每层计算量小
                      └─ libgomp 默认主动自旋(GOMP_SPINCOUNT~3e5)
                           └─ 空闲线程在 barrier 上忙等不休眠
                                └─ 60% CPU 空转  ❌
```

### 6.2 建议（按性价比排序）

**P0 — 零改代码，先验证收益**：设环境变量让空闲 OpenMP 线程休眠
```bash
OMP_WAIT_POLICY=passive OMP_NUM_THREADS=4 GOMP_SPINCOUNT=0 ./perf_benchmark
```
**已实测验证**（同样 25s 采样，前后对比）：

| DSO | 改前 self-CPU | 改后 self-CPU |
|-----|--------------|--------------|
| libgomp.so（barrier 自旋） | **60.3%** | **0.6%** |
| libdigital_human_core（真实计算） | 37.9% | **94.7%** |

被自旋浪费的 ~60% CPU 被回收给真实计算 → **根因与该修复均已证实**。
建议先用此法量化收益，再决定是否落到代码。

**P1 — 修复线程数设置时机**（`src/model/model_inferencer.cpp`）
把 `net.opt.num_threads = N` 移到 `load_param()` **之前**：
```cpp
net.opt.num_threads = N;         // ← 必须在 load 之前，卷积层才会按 N 打包
net.opt.use_winograd_convolution = true;
net.load_param(param_path);
net.load_model(bin_path);
```
消除 "use load-time value 8" 警告，卷积层真正按目标线程数运行。

**P2 — 线程数与 Pipeline 并发协同调优**
Pipeline 本身有 视频/音频/推理/渲染 多个 worker 线程，叠加 ncnn 的 OpenMP，
容易在 16 核上过订阅。建议 ncnn 推理线程数取 2~4 并实测（配合 P0 的 passive 策略），
`cv::setNumThreads()` 也一并压低，避免 OpenCV 与 ncnn 抢核。

**P3 — OpenCV 合成开销（~5%）**
`vlineSmooth5N / BlendLinearInvoker / addWeighted8u` 来自人脸区域的 resize + alpha 混合。
可考虑：合成分辨率下采样、用 ROI 只处理嘴部区域、或合并 blend 步骤。优先级低于 P0/P1。

---

## 7. 一键复现

```bash
# 在 WSL2 Ubuntu 内，项目根目录
bash scripts/profile_flamegraph.sh build-profile/bin/perf_benchmark 999 fp
# 参数: <目标程序> <采样频率Hz> <回栈方式 fp|dwarf>
# 产物见 docs/perf/flamegraph.svg
```

> 若目标程序长时间运行（如本例音频 293s），脚本内已用 `timeout -s INT` 采样固定时长后
> 让 perf 正常 flush；需要更长/更短样本改脚本里的时长即可。

---

## 附：本次调试方法论小结（可迁移到其它性能问题）

1. **先判可行性**：确认能采样（工具在哪跑、内核支持、权限够不够），不要一上来就 record。
2. **专门构建**：`RelWithDebInfo -fno-omit-frame-pointer`，保真 + 可读栈。
3. **小步验证**：先用一个 2s 的小负载 smoke-test perf 能出样本，再上真实程序。
4. **按坑迭代**：dwarf→fp、去掉超限 mmap、长程序用信号 flush——每个报错都指向一个具体开关。
5. **先聚合后下钻**：`perf report --sort dso`（哪个库）→ `--sort symbol`（哪个函数）→ 火焰图（哪条调用链）。
6. **区分"忙等"与"真算"**：塔顶是运行时/锁/barrier 地址而非业务函数 = 并发浪费，往往比算法本身收益更大。
7. **改前先量化**：P0 用环境变量验证收益，再决定动代码。
