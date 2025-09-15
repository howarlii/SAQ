#pragma once

#include <climits>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_set>

#include "defines.hpp"
#include "utils/memory.hpp"

namespace saqlib {

/**
 * @brief hash set to record visited vertices
 *
 */
class HashBasedBooleanSet {
  private:
    const uint32_t kPidMax = std::numeric_limits<PID>::max();

    size_t table_size_ = 0;
    PID mask_ = 0;
    std::vector<PID, memory::AlignedAllocator<PID>> table_;
    std::unordered_set<PID> stl_hash_;

    [[nodiscard]] auto hash1(const PID value) const { return value & mask_; }

  public:
    HashBasedBooleanSet() = default;

    HashBasedBooleanSet(const HashBasedBooleanSet &other)
        : table_size_(other.table_size_), mask_(other.mask_), table_(other.table_), stl_hash_(other.stl_hash_) {}

    HashBasedBooleanSet(HashBasedBooleanSet &&other) noexcept
        : table_size_(other.table_size_), mask_(other.mask_), table_(std::move(other.table_)), stl_hash_(std::move(other.stl_hash_)) {}
    HashBasedBooleanSet &operator=(HashBasedBooleanSet &&other) noexcept {
        table_size_ = other.table_size_;
        mask_ = other.mask_;
        table_ = std::move(other.table_);
        stl_hash_ = std::move(other.stl_hash_);

        return *this;
    }

    explicit HashBasedBooleanSet(size_t size) {
        size_t bit_size = 0;
        size_t bit = size;
        while (bit != 0) {
            bit_size++;
            bit >>= 1;
        }
        size_t bucket_size = 0x1 << ((bit_size + 4) / 2 + 3);
        initialize(bucket_size);
    }

    void initialize(const size_t table_size) {
        table_size_ = table_size;
        mask_ = static_cast<PID>(table_size_ - 1);
        const PID check_val = hash1(static_cast<PID>(table_size));
        if (check_val != 0) {
            std::cerr << "[WARN] table size is not 2^N :  " << table_size << '\n';
        }

        table_ = std::vector<PID, memory::AlignedAllocator<PID>>(table_size);
        std::fill(table_.begin(), table_.end(), kPidMax);
        stl_hash_.clear();
    }

    void clear() {
        std::fill(table_.begin(), table_.end(), kPidMax);
        stl_hash_.clear();
    }

    // get if data_id is in the hashset
    [[nodiscard]] bool get(PID data_id) const {
        PID val = this->table_[hash1(data_id)];
        if (val == data_id) {
            return true;
        }
        return (val != kPidMax && stl_hash_.find(data_id) != stl_hash_.end());
    }

    void set(PID data_id) {
        PID &val = table_[hash1(data_id)];
        if (val == data_id) {
            return;
        }
        if (val == kPidMax) {
            val = data_id;
        } else {
            stl_hash_.emplace(data_id);
        }
    }
};

class VisitedListPool {
    std::deque<HashBasedBooleanSet *> pool_;
    std::mutex poolguard_;
    size_t numelements_;

  public:
    VisitedListPool(size_t initpoolsize, size_t max_elements) {
        numelements_ = max_elements / 10;
        for (size_t i = 0; i < initpoolsize; i++) {
            pool_.push_front(new HashBasedBooleanSet(numelements_));
        }
    }

    HashBasedBooleanSet *get_free_vislist() {
        HashBasedBooleanSet *rez;
        {
            std::unique_lock<std::mutex> lock(poolguard_);
            if (pool_.size() > 0) {
                rez = pool_.front();
                pool_.pop_front();
            } else {
                rez = new HashBasedBooleanSet(numelements_);
            }
        }
        rez->clear();
        return rez;
    }

    void release_vis_list(HashBasedBooleanSet *vl) {
        std::unique_lock<std::mutex> lock(poolguard_);
        pool_.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool_.size() > 0) {
            HashBasedBooleanSet *rez = pool_.front();
            pool_.pop_front();
            ::delete rez;
        }
    }
};
} // namespace saqlib