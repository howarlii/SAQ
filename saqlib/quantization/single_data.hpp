#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#include <glog/logging.h>

#include "defines.hpp"
#include "quantization/cluster_data.hpp"
#include "utils/memory.hpp"
#include "utils/tools.hpp"

namespace saqlib {

class SaqSingleDataWrapper;

class CaqSingleDataWrapper {
    friend SaqSingleDataWrapper;

  public:
    static constexpr size_t kNumShortFactors = 2; // factors packed into shortdata

    const size_t num_dim_padded_; // Padded number of dimension (multiple of 64)
    const size_t num_bits_;       // bits

  private:
    size_t shortb_factors_num_; // number of short block factors (in float)
    size_t shortb_code_bytes_;  // bytes of short block code
    size_t longb_code_bytes_;   // bytes of long block code

    bool should_free_ = false;
    float *short_factors_ = nullptr;   // short factors
    uint8_t *short_code_ = nullptr;    // short code
    uint8_t *long_code_ = nullptr;     // long code
    ExFactor *long_factors_ = nullptr; // long factors of vector
    PID id_ = 0;                       // PID of vector

  public:
    /**
     * @brief Construct a new CaqSingleDataWrapper object
     * Data for single vector quantization
     *
     * @param num_dim_padded padded dimension size
     * @param num_bits number of bits for quantization
     */
    explicit CaqSingleDataWrapper(size_t num_dim_padded, size_t num_bits)
        : num_dim_padded_(num_dim_padded),
          num_bits_(num_bits),
          shortb_factors_num_(kNumShortFactors),
          shortb_code_bytes_(num_bits ? num_dim_padded / 8 * sizeof(uint8_t) : 0),
          longb_code_bytes_(num_bits > 1 ? num_dim_padded * (num_bits - 1) / 8 : 0) {
    }

    ~CaqSingleDataWrapper() {
        if (should_free_) {
            std::free(short_factors_);
            std::free(short_code_);
            std::free(long_code_);
            std::free(long_factors_);
        }
    }

    void allocate_data() {
        should_free_ = true;
        if (shortb_factors_num_ > 0) {
            short_factors_ = memory::align_mm<64, float>(shortb_factors_num_);
        }
        if (shortb_code_bytes_ > 0) {
            short_code_ = memory::align_mm<64, uint8_t>(shortb_code_bytes_);
        }
        if (longb_code_bytes_ > 0) {
            long_code_ = memory::align_mm<64, uint8_t>(longb_code_bytes_);
        }
        long_factors_ = memory::align_mm<64, ExFactor>(1);
    }

    /**
     * @brief Set memory base and update pointers (used by SaqSingleDataWrapper)
     */
    void set_pointers(float *short_factors, uint8_t *short_code, uint8_t *long_code, ExFactor *long_factors) {
        short_factors_ = short_factors;
        short_code_ = short_code;
        long_code_ = long_code;
        long_factors_ = long_factors;
    }

    /**
     * @brief Return pointer to short code
     */
    auto short_code() { return short_code_; }
    auto short_code() const { return short_code_; }

    auto &factor_o_l2norm() { return short_factors_[0]; }
    auto factor_o_l2norm() const { return short_factors_[0]; }

    // ip_cent_oa is optional
    auto &factor_ip_cent_oa() { return short_factors_[1]; }
    auto factor_ip_cent_oa() const { return short_factors_[1]; }

    /**
     * @brief Return long code for this vector
     */
    uint8_t *long_code() { return long_code_; }
    uint8_t *long_code() const { return long_code_; }

    /**
     * @brief Return long factor of this vector
     */
    ExFactor &long_factor() { return *long_factors_; }
    const ExFactor &long_factor() const { return *long_factors_; }

    /**
     * @brief Get/Set PID
     */
    PID id() const { return id_; }
    void set_id(PID id) { id_ = id; }

    auto num_dim_padded() const { return num_dim_padded_; }
    auto num_bits() const { return num_bits_; }
};

/**
 * @brief SaqSingleDataWrapper Memory Layout
 *
 * Memory structure for a single quantized vector with multiple segments:
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                        External Memory Block                                │
 * │   (allocated externally, managed by set_memory_base())                      │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 * Memory Layout (code-aligned, padding before short code):
 * ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
 * │ Short        │ Padding      │ Short        │ Long         │ Long         │
 * │ Factors      │ (alignment)  │ Code         │ Code         │ Factors      │
 * │ (floats)     │              │ (uint8_t)    │ (uint8_t)    │ (ExFactor)   │
 * └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
 * ^                             ^              ^              ^
 * │                             │              │              └ long_factors_offset_
 * │                             │              └─────────────── long_code_offset_
 * │                             └────────────────────────────── short_code_offset_
 * └──────────────────────────────────────────────────────────── short_factors_offset_
 *
 * Per-segment data distribution:
 * - Short Factors: segment0[factor0, factor1], segment1[factor0, factor1], ...
 * - Short Code:    segment0[code_data...], segment1[code_data...], ...
 * - Long Code:     segment0[long_data...], segment1[long_data...], ...
 * - Long Factors:  segment0[ExFactor], segment1[ExFactor], ...
 *
 * Usage Pattern:
 * 1. Construct once: SaqSingleDataWrapper wrapper(quant_plan);
 * 2. Reuse multiple times: wrapper.set_memory_base(new_memory_ptr);
 */
class SaqSingleDataWrapper {
  public:
    const size_t num_segments_; // Number of segments

  private:
    std::vector<CaqSingleDataWrapper> segments_;
    size_t shortb_factors_fcnt_ = 0; // total short factors count for all segments
    size_t shortb_code_bytes_ = 0;   // total short code bytes for all segments
    size_t longb_code_bytes_ = 0;    // total long code bytes for all segments
    size_t total_memory_size_ = 0;   // total memory size needed

    // Memory layout offsets
    size_t short_factors_offset_ = 0;
    size_t short_code_offset_ = 0;
    size_t long_code_offset_ = 0;
    size_t long_factors_offset_ = 0;

    // ========================= external memory data below =========================
    uint8_t *memory_base_ = nullptr; // base pointer to external memory
    float *short_factors_;           // short factors (points into memory_base_)
    uint8_t *short_code_;            // short code (points into memory_base_)
    uint8_t *long_code_;             // long code (points into memory_base_)
    ExFactor *long_factors_;         // extra factors of vector (points into memory_base_)

  public:
    /**
     * @brief Construct a new SaqSingleDataWrapper object
     * @param quant_plan quantization plan for each segment. <num_dims, bits>
     *
     * Construction process:
     * 1. Create CaqSingleDataWrapper for each segment based on quant_plan
     * 2. Calculate total memory requirements for all segments
     * 3. Pre-calculate memory section offsets for fast set_memory_base() calls
     */
    explicit SaqSingleDataWrapper(const std::vector<std::pair<size_t, size_t>> &quant_plan)
        : num_segments_(quant_plan.size()) {
        segments_.reserve(quant_plan.size());

        // Step 1: Create segments and accumulate memory requirements
        for (size_t i = 0; i < quant_plan.size(); ++i) {
            auto dim_padded = quant_plan[i].first;
            DCHECK_EQ(0, dim_padded % kDimPaddingSize);
            auto &c = segments_.emplace_back(dim_padded, quant_plan[i].second);

            shortb_factors_fcnt_ += c.shortb_factors_num_;
            shortb_code_bytes_ += c.shortb_code_bytes_;
            longb_code_bytes_ += c.longb_code_bytes_;
        }

        // Step 2: Calculate memory section offsets (code-aligned layout)
        size_t offset = 0;

        // Short factors section (64-byte aligned)
        short_factors_offset_ = offset;
        offset += utils::rd_up_to_multiple_of(shortb_factors_fcnt_ * sizeof(float), 64);

        // Short code section (64-byte aligned for optimal code access)
        short_code_offset_ = offset;
        offset += utils::rd_up_to_multiple_of(shortb_code_bytes_, 64);

        // Long code section (directly follows short code, no extra alignment)
        long_code_offset_ = offset;
        offset += longb_code_bytes_;

        // Long factors section (aligned to ExFactor size)
        long_factors_offset_ = offset;
        offset += num_segments_ * sizeof(ExFactor);

        total_memory_size_ = offset;
    }

    /**
     * @brief Set memory base and update all pointers
     * @param memory_base external memory base pointer
     *
     * Memory assignment process:
     * 1. Update base section pointers using pre-calculated offsets
     * 2. Distribute memory within each section to individual segments
     * 3. Call set_pointers() for each segment to finalize pointer assignment
     */
    void set_memory_base(uint8_t *memory_base) {
        DCHECK_EQ(reinterpret_cast<uintptr_t>(memory_base) % 64, 0) << "memory_base must be 64-byte aligned";
        memory_base_ = memory_base;

        // Update base pointers using pre-calculated offsets
        short_factors_ = reinterpret_cast<float *>(memory_base_ + short_factors_offset_);
        short_code_ = memory_base_ + short_code_offset_;
        long_code_ = memory_base_ + long_code_offset_;
        long_factors_ = reinterpret_cast<ExFactor *>(memory_base_ + long_factors_offset_);

        // Distribute memory within each section to individual segments
        size_t short_factors_offset = 0;
        size_t short_code_offset = 0;
        size_t long_code_offset = 0;

        for (size_t i = 0; i < num_segments_; ++i) {
            auto &c = segments_[i];

            float *segment_short_factors = nullptr;
            uint8_t *segment_short_code = nullptr;
            uint8_t *segment_long_code = nullptr;
            ExFactor *segment_long_factors = &long_factors_[i];

            // Assign short factors (each segment gets kNumShortFactors floats)
            if (c.shortb_factors_num_ > 0) {
                segment_short_factors = short_factors_ + short_factors_offset;
                short_factors_offset += c.shortb_factors_num_;
            }

            // Assign short code (size depends on dim_padded and quantization bits)
            if (c.shortb_code_bytes_ > 0) {
                segment_short_code = short_code_ + short_code_offset;
                short_code_offset += c.shortb_code_bytes_;
            }

            // Assign long code (size depends on dim_padded and (bits-1))
            if (c.longb_code_bytes_ > 0) {
                segment_long_code = long_code_ + long_code_offset;
                long_code_offset += c.longb_code_bytes_;
            }

            // Finalize pointer assignment for this segment
            c.set_pointers(segment_short_factors, segment_short_code, segment_long_code, segment_long_factors);
        }
    }

    /**
     * @brief Calculate required memory size for given quantization plan
     * @param quant_plan quantization plan for each segment. <num_dims, bits>
     * @return required memory size in bytes
     */
    static size_t calculate_memory_size(const std::vector<std::pair<size_t, size_t>> &quant_plan) {
        size_t shortb_factors_fcnt = 0;
        size_t shortb_code_bytes = 0;
        size_t longb_code_bytes = 0;
        size_t num_segments = quant_plan.size();

        for (const auto &plan : quant_plan) {
            auto dim_padded = utils::rd_up_to_multiple_of(plan.first, kDimPaddingSize);
            auto num_bits = plan.second;

            shortb_factors_fcnt += CaqSingleDataWrapper::kNumShortFactors;
            if (num_bits > 0) {
                shortb_code_bytes += dim_padded / 8;
            }
            if (num_bits > 1) {
                longb_code_bytes += dim_padded * (num_bits - 1) / 8;
            }
        }

        size_t total_size = 0;

        // Short factors (64-byte aligned)
        total_size += utils::rd_up_to_multiple_of(shortb_factors_fcnt * sizeof(float), 64);

        // Short code (64-byte aligned for optimal code access)
        total_size += utils::rd_up_to_multiple_of(shortb_code_bytes, 64);

        // Long code (no extra alignment, directly follows short code)
        total_size += longb_code_bytes;

        // Long factors (natural alignment)
        total_size += num_segments * sizeof(ExFactor);

        return total_size;
    }

    ~SaqSingleDataWrapper() = default;

    auto &get_segment(size_t idx) { return segments_[idx]; }
    auto &get_segment(size_t idx) const { return segments_[idx]; }

    auto num_segments() const { return num_segments_; }
    auto total_memory_size() const { return total_memory_size_; }

    /**
     * @brief Clear all data (set memory to zero)
     */
    void clear() {
        if (memory_base_) {
            std::memset(memory_base_, 0, total_memory_size_);
        }
    }

    /**
     * @brief Check if memory layout is valid
     */
    bool is_valid() const {
        return memory_base_ != nullptr;
    }
};

} // namespace saqlib
