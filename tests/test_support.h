#pragma once

#include <cmath>
#include <iostream>
#include <string>

class TestSuite {
public:
    void Check(bool condition, const std::string& message) {
        if (condition) {
            ++passed_;
            return;
        }
        ++failed_;
        std::cerr << "FAILED: " << message << '\n';
    }

    template <typename Actual, typename Expected>
    void Equal(const Actual& actual, const Expected& expected,
               const std::string& message) {
        Check(actual == expected, message);
    }

    void Near(double actual, double expected, double tolerance,
              const std::string& message) {
        Check(std::isfinite(actual)
                  && std::abs(actual - expected) <= tolerance,
              message);
    }

    int Finish(const char* name) const {
        std::cout << name << ": " << passed_ << " passed, "
                  << failed_ << " failed\n";
        return failed_ == 0 ? 0 : 1;
    }

private:
    int passed_ = 0;
    int failed_ = 0;
};
