#pragma once

#include <chrono>

namespace saqlib::utils {
class StopW {
    std::chrono::steady_clock::time_point time_begin;

  public:
    StopW() { time_begin = std::chrono::steady_clock::now(); }

    float getElapsedTimeSec() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (
            std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin).count());
    }

    float getElapsedTimeMili() {
        return getElapsedTimeMicro() / 1000.0;
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin)
                    .count());
    }

    float getElapsedTimeNano() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin)
                    .count());
    }

    void reset() { time_begin = std::chrono::steady_clock::now(); }
};
} // namespace saqlib::utils
