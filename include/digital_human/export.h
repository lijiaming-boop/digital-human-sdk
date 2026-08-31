#pragma once

/// @file export.h
/// @brief 跨平台符号导出宏。
///
/// 在编译 digital_human_core 共享库时定义 DIGITAL_HUMAN_CORE_EXPORTS，
/// 使所有标记 DH_API 的符号被导出（dllexport）；
/// 消费方包含此头文件时不定义该宏，DH_API 展开为 dllimport。
/// Linux/macOS 下使用 visibility 属性控制符号可见性。

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef DIGITAL_HUMAN_CORE_EXPORTS
        #define DH_API __declspec(dllexport)
    #else
        #define DH_API __declspec(dllimport)
    #endif
    #define DH_LOCAL
#else
    #define DH_API __attribute__((visibility("default")))
    #define DH_LOCAL __attribute__((visibility("hidden")))
#endif
