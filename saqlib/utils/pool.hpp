#pragma once

#include <limits>
#include <stdint.h>
#include <vector>

#include <glog/logging.h>

#include "utils/memory.hpp"

namespace saqlib::utils {
struct ResultPool {
  public:
    ResultPool(size_t capacity, bool greater = false)
        : greater_(greater), ids_(capacity + 1), distances_(capacity + 1), capacity_(capacity) {}

    void insert(PID u, float dist) {
        if (greater_)
            dist = -dist; // Invert distance if greater is true
        if (size_ == capacity_ && dist > distances_[size_ - 1]) {
            return;
        }
        size_t lo = find_bsearch(dist);
        std::memmove(&ids_[lo + 1], &ids_[lo], (size_ - lo) * sizeof(PID));
        ids_[lo] = u;
        std::memmove(&distances_[lo + 1], &distances_[lo], (size_ - lo) * sizeof(float));
        distances_[lo] = dist;
        size_ += (size_ < capacity_);
        return;
    }

    float distk() {
        return (greater_ ? -distk_raw() : distk_raw());
    }

    void copy_results(PID *KNN) { std::copy(ids_.begin(), ids_.end() - 1, KNN); }

    auto get(size_t i) { return std::make_pair(ids_[i], (greater_ ? -1 : 1) * distances_[i]); }

  private:
    bool greater_ = false;
    std::vector<PID, memory::AlignedAllocator<PID>> ids_;
    std::vector<float, memory::AlignedAllocator<float>> distances_;
    size_t size_ = 0, capacity_;

    float distk_raw() {
        return size_ == capacity_ ? distances_[size_ - 1]
                                  : std::numeric_limits<float>::max();
    }

    size_t find_bsearch(float dist) const {
        size_t lo = 0, len = size_;
        size_t half;
        while (len > 1) {
            half = len >> 1;
            len -= half;
            lo += (distances_[lo + half - 1] < dist) * half;
        }
        return (lo < size_ && distances_[lo] < dist) ? lo + 1 : lo;
    }
};

class StatisticsRecorder {
  public:
    StatisticsRecorder()
        : bins_(0) {};

    StatisticsRecorder(double min_val, double max_val, size_t bins = 1500)
        : bins_(bins), min_val_(min_val), max_val_(max_val), histogram_(bins, 0), tot_sum_(0), tot_maximum_(std::numeric_limits<double>::lowest()), cnt_(0) {}

    StatisticsRecorder create_same() const {
        return StatisticsRecorder(min_val_, max_val_, bins_);
    }

    void clean() {
        std::fill(histogram_.begin(), histogram_.end(), 0);
        tot_sum_ = 0;
        tot_maximum_ = std::numeric_limits<double>::lowest();
        cnt_ = 0;
    }

    double gap() const { return (max_val_ - min_val_) / bins_; }

    void insert(double value) {
        CHECK_GT(bins_, 0);
        tot_sum_ += value;
        tot_maximum_ = std::max(tot_maximum_, value);
        // if (value < min_val_ || value > max_val_) return;
        value = std::min(value, max_val_);
        value = std::max(value, min_val_);
        size_t bin = static_cast<size_t>((value - min_val_) / (max_val_ - min_val_) * bins_);
        if (bin >= bins_)
            bin = bins_ - 1;
        histogram_[bin]++;
        cnt_++;
    }

    auto def_min_val() const { return min_val_; }
    auto def_max_val() const { return max_val_; }
    auto def_bins() const { return bins_; }

    double avg() const {
        if (cnt_ == 0)
            return double();
        return tot_sum_ / cnt_;
    }

    double max() const {
        return tot_maximum_;
    }

    auto count() const {
        return cnt_;
    }

    const std::vector<size_t> &histogram() const {
        return histogram_;
    }

    void merge(const StatisticsRecorder &other) {
        if (bins_ != other.bins_ || min_val_ != other.min_val_ || max_val_ != other.max_val_) {
            throw std::invalid_argument("HistogramRecorder bins, min_val, and max_val must match for addition.");
        }
        for (size_t i = 0; i < bins_; ++i) {
            histogram_[i] += other.histogram_[i];
        }
        tot_sum_ += other.tot_sum_;
        tot_maximum_ = std::max(tot_maximum_, other.tot_maximum_);
        cnt_ += other.cnt_;
    }

  private:
    size_t bins_;
    double min_val_, max_val_;
    std::vector<size_t> histogram_;
    double tot_sum_;
    double tot_maximum_;
    size_t cnt_;
};

class AvgMaxRecorder {
  public:
    AvgMaxRecorder() {}

    void clean() {
        tot_sum_ = 0;
        tot_maximum_ = std::numeric_limits<double>::lowest();
        cnt_ = 0;
    }

    void insert(double value) {
        tot_sum_ += value;
        tot_maximum_ = std::max(tot_maximum_, value);
        cnt_++;
    }

    double avg() const {
        if (cnt_ == 0)
            return 0.0;
        return tot_sum_ / cnt_;
    }

    double max() const {
        return tot_maximum_;
    }

    double sum() const {
        return tot_sum_;
    }

    void merge(const AvgMaxRecorder &other) {
        this->tot_sum_ = tot_sum_ + other.tot_sum_;
        this->tot_maximum_ = std::max(tot_maximum_, other.tot_maximum_);
        this->cnt_ = cnt_ + other.cnt_;
    }

    double tot_sum_{0};
    double tot_maximum_{std::numeric_limits<double>::lowest()};
    size_t cnt_{0};
};

class AvgMaxGroup {
  public:
    AvgMaxGroup() {}

    void clean() {
        tot_.clean();
        group_avg_.clean();
    }

    void add(const AvgMaxRecorder &v) {
        tot_.merge(v);
        group_avg_.insert(v.avg());
        group_mx_.insert(v.max());
    }

    auto tot_avg() const {
        return tot_.avg();
    }
    auto tot_max() const {
        return tot_.max();
    }
    auto group_avg_avg() const {
        return group_avg_.avg();
    }
    auto group_mx_avg() const {
        return group_mx_.avg();
    }

    AvgMaxRecorder tot_, group_avg_, group_mx_;
};
} // namespace saqlib::utils
