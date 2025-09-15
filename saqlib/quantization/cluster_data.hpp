#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <stdlib.h>
#include <vector>

#include <glog/logging.h>

#include "defines.hpp"
#include "utils/memory.hpp"
#include "utils/tools.hpp"

namespace saqlib {
class CaqCode {
  public:
    float v_mx;   // max absolute value of the vector, v_mx = max{|o_i|}
    float v_mi;   // v_mi = -v_mx
    double delta; // (v_mx-v_mi) / 2^b
    Eigen::VectorXi code;
    double oa_l2sqr; // |o_a|^2, squared L2 norm of the quantized vector
    double ip_o_oa;  // <o, o_a>

    float o_l2sqr;     // |o|^2, squared L2 norm of the original vector
    float o_l2norm;    // |o|, L2 norm of the original vector
    float fac_rescale; // |o|^2 / <o, o_a>
    float fac_error;   // |o|^2 * epsilon * sqrt((1 - <o, o_a>^2) / <o, o_a>^2) / sqrt(dim - 1)

    void rescale_vmx_to1() {
        if (!v_mx)
            return;
        const auto scale_rate = 1.0 / v_mx;
        v_mi *= scale_rate;
        v_mx *= scale_rate;
        delta *= scale_rate;
        ip_o_oa *= scale_rate;
        oa_l2sqr *= scale_rate * scale_rate;
    }

    Eigen::VectorXf get_oa() const {
        CHECK(code.size() >= 1) << "CAQSingleData::get_oa() called before encode.";
        return (code.cast<float>() * delta).array() + v_mi;
    }
};

struct ExFactor {
    float rescale = 0;
    float error = 0;
};

class SaqCluData;

class CAQClusterData {
    friend SaqCluData;

  public:
    static constexpr size_t kNumShortFactors = 2; // factors packed into shortdata

    const size_t num_vec_;        // Num of vectors in this cluster
    const size_t num_vec_align_;  // Padded number of vectors (multiple of 32)
    const size_t num_dim_padded_; // Padded number of dimension (multiple of 64)
    const size_t num_bits_;       // bits
    const size_t num_blocks_;     // Num of blocks
  private:
    size_t shortb_factors_num_; // number of short block factors (in float)
    size_t shortb_code_bytes_;  // bytes of short block code
    size_t longb_code_bytes_;   // bytes of long block code

    size_t num_parallel_clusters_ = 1; // number of parallel clusters, that is, segments

    bool should_free_ = false;
    float *short_factors_ = nullptr;   // short factors
    uint8_t *short_code_ = nullptr;    // short code
    uint8_t *long_code_ = nullptr;     // long code
    ExFactor *long_factors_ = nullptr; // long factors of vectors
    PID *ids_ = nullptr;               // PID of vectors
    FloatVec centroid_;                // Rotated centroid of clusters

  public:
    /**
     * @brief Construct a new Cluster:: Cluster object
     * Data in the cluster are mapped to large arrays in memory
     *
     * @param num number of vectors
     * @param short_data blocks of 1-bit codes and corresponding factors
     * @param long_code long code for re-ranking
     * @param ex_factor factors for re-ranking
     * @param ids id for vectors in the cluster
     */
    explicit CAQClusterData(size_t num_vec, size_t num_dim_paded, size_t num_bits)
        : num_vec_(num_vec),
          num_vec_align_(utils::rd_up_to_multiple_of(num_vec, KFastScanSize)),
          num_dim_padded_(num_dim_paded),
          num_bits_(num_bits),
          num_blocks_(utils::div_rd_up(num_vec, KFastScanSize)),
          shortb_factors_num_(KFastScanSize * kNumShortFactors),
          shortb_code_bytes_(num_bits ? num_dim_paded * KFastScanSize / 8 * sizeof(uint8_t) : 0),
          longb_code_bytes_(num_bits ? num_dim_paded * (num_bits - 1) / 8 : 0) {
        centroid_.resize(num_dim_paded);
    }

    ~CAQClusterData() {
        if (should_free_) {
            std::free(short_factors_);
            std::free(short_code_);
            std::free(long_code_);
            std::free(long_factors_);
            std::free(ids_);
        }
    }

    // void allocate_data()
    // {
    //     should_free_ = true;
    //     short_factors_ = memory::align_mm<64, float>(shortb_factors_fcnt_ * num_blocks_);
    //     short_code_ = memory::align_mm<64, uint8_t>(shortb_code_bytes_ * num_blocks_);
    //     long_code_ = memory::align_mm<64, uint8_t>(longb_code_bytes_ * num_vec_align);
    //     EX_FACTOR = memory::align_mm<64, ExFactor>(num_vec_);
    //     IDs_ = memory::align_mm<64, PID>(num_vec_align);
    // }

    /**
     * @brief Return pointer to short code of i-th blocks in this cluster
     */
    auto short_code(size_t block_idx) { return &short_code_[shortb_code_bytes_ * block_idx]; }
    auto short_code(size_t block_idx) const { return &short_code_[shortb_code_bytes_ * block_idx]; }
    auto short_code_single(size_t vec_idx) const {
        auto block_idx = vec_idx / KFastScanSize;
        auto j = vec_idx % KFastScanSize;
        return short_code(block_idx) + num_dim_padded_ / 8 * j;
    }

    auto factor_o_l2norm(size_t block_idx) { return &short_factors_[block_idx * shortb_factors_num_]; }
    auto factor_o_l2norm(size_t block_idx) const { return &short_factors_[block_idx * shortb_factors_num_]; }

    // ip_cent_oa is optional
    auto factor_ip_cent_oa(size_t block_idx) { return factor_o_l2norm(block_idx) + KFastScanSize; }
    auto factor_ip_cent_oa(size_t block_idx) const { return factor_o_l2norm(block_idx) + KFastScanSize; }

    /**
     * @brief Return long code for i-th vector in this cluster
     */
    uint8_t *long_code(size_t vec_idx) {
        DCHECK_LT(vec_idx, num_vec_);
        return &long_code_[vec_idx * longb_code_bytes_];
    }
    uint8_t *long_code(size_t vec_idx) const {
        DCHECK_LT(vec_idx, num_vec_);
        return &long_code_[vec_idx * longb_code_bytes_];
    }

    /**
     * @brief Return long factor of i-th vector in this cluster
     */
    ExFactor &long_factor(size_t vec_idx) {
        return long_factors_[vec_idx * num_parallel_clusters_];
    }
    ExFactor &long_factor(size_t vec_idx) const {
        return long_factors_[vec_idx * num_parallel_clusters_];
    }

    auto &centroid() { return centroid_; }
    auto &centroid() const { return centroid_; }

    /**
     * @brief Return pointer to ids
     */
    PID *ids() { return this->ids_; }
    PID *ids() const { return this->ids_; }

    auto num_vec() const { return num_vec_; }
    auto num_blocks() const { return num_blocks_; }
    auto iter() const { return num_vec_ / KFastScanSize; }
    auto remain() const { return num_vec_ % KFastScanSize; }

    // void load(std::ifstream &input)
    // {
    //     input.read((char *)SHORT_DATA.data(), SHORT_DATA.size() * sizeof(uint8_t));
    //     input.read((char *)LONG_CODE.data(), LONG_CODE.size() * sizeof(uint8_t));
    //     input.read((char *)EX_FACTOR.data(), EX_FACTOR.size() * sizeof(ExFactor));
    //     input.read((char *)IDs_.data(), IDs_.size() * sizeof(PID));
    //     for (auto &centroid : centroids_) {
    //         input.read((char *)centroid.data(), centroid.cols() * sizeof(float));
    //     }
    // }
    // void save(std::ofstream &output) const
    // {
    //     output.write((char *)SHORT_DATA.data(), SHORT_DATA.size() * sizeof(uint8_t));
    //     output.write((char *)LONG_CODE.data(), LONG_CODE.size() * sizeof(uint8_t));
    //     output.write((char *)EX_FACTOR.data(), EX_FACTOR.size() * sizeof(ExFactor));
    //     output.write((char *)IDs_.data(), IDs_.size() * sizeof(PID));
    //     for (auto &centroid : centroids_) {
    //         output.write((char *)centroid.data(), centroid.cols() * sizeof(float));
    //     }
    // }
};

class SaqCluData {
    static constexpr size_t kLongCodeAlignBytes = 16;

  public:
    const size_t num_vec_;       // Num of vectors in this segment
    const size_t num_vec_align_; // Num of vectors in this segment
    const size_t num_blocks_;    // Num of blocks
    const size_t num_segments_;  // Num of segments
  private:
    std::vector<CAQClusterData> segments_;
    size_t shortb_factors_fcnt_ = 0;  // bytes of short factors for all segments
    size_t shortb_code_bytes_ = 0;    // bytes of short code for all segments
    size_t longb_code_bytes_ = 0;     // bytes of long block for all segments
    size_t longb_code_bytes_tot_ = 0; // bytes of long block for all segments

    // ========================= presistence data below =========================
    float *short_factors_;                                    // short factors
    uint8_t *short_code_;                                     // short code
    uint8_t *long_code_;                                      // long code
    ExFactor *long_factors_;                                  // extra factors of vectors
    std::vector<PID, memory::AlignedAllocator<PID, 64>> ids_; // PID of vectors

  public:
    /**
     * @param num number of vectors
     * @param quant_plan_ quantization plan for each segment. <num_dims, bits>
     */
    explicit SaqCluData(size_t num_vec, const std::vector<std::pair<size_t, size_t>> &quant_plan, bool use_compact_layout = false)
        : num_vec_(num_vec),
          num_vec_align_(utils::rd_up_to_multiple_of(num_vec, KFastScanSize)),
          num_blocks_(utils::div_rd_up(num_vec, KFastScanSize)),
          num_segments_(quant_plan.size()) {
        if (num_segments_ == 1)
            use_compact_layout = true;

        segments_.reserve(quant_plan.size());
        for (size_t i = 0; i < quant_plan.size(); ++i) {
            auto dim_padded = quant_plan[i].first;
            DCHECK_EQ(dim_padded % kDimPaddingSize, 0);
            auto &c = segments_.emplace_back(num_vec, dim_padded, quant_plan[i].second);
            c.num_parallel_clusters_ = num_segments_;
            shortb_factors_fcnt_ += c.shortb_factors_num_;
            shortb_code_bytes_ += c.shortb_code_bytes_;

            if (use_compact_layout) {
                longb_code_bytes_ += c.longb_code_bytes_;
                longb_code_bytes_tot_ += utils::rd_up_to_multiple_of(c.longb_code_bytes_ * num_vec, kLongCodeAlignBytes);
            } else {
                longb_code_bytes_ += utils::rd_up_to_multiple_of(c.longb_code_bytes_, kLongCodeAlignBytes);
                longb_code_bytes_tot_ = longb_code_bytes_ * num_vec;
            }
        }

        // assign long code and EX_FACTOR
        if (quant_plan.size() == 1) {
            auto blk_bytes = (shortb_factors_fcnt_ * sizeof(float) + shortb_code_bytes_);
            short_code_ = memory::align_mm<64, uint8_t>(blk_bytes * num_blocks_);
            shortb_code_bytes_ = blk_bytes;
            short_factors_ = nullptr;
            shortb_factors_fcnt_ = 0;
            size_t ptr = 0;
            for (size_t i = 0; i < quant_plan.size(); ++i) {
                auto &c = segments_[i];

                c.short_factors_ = reinterpret_cast<float *>(short_code_ + ptr);
                ptr += c.shortb_factors_num_ * sizeof(float);
                c.shortb_factors_num_ = blk_bytes / sizeof(float);

                c.short_code_ = short_code_ + ptr;
                ptr += c.shortb_code_bytes_;
                c.shortb_code_bytes_ = blk_bytes;
            }
            // CHECK_EQ(ptr, blk_bytes);
            assert(ptr == blk_bytes);
        } else {
            // TODO: optimize layout of short factors and codes
            short_factors_ = memory::align_mm<64, float>(shortb_factors_fcnt_ * num_blocks_);
            short_code_ = memory::align_mm<64, uint8_t>(shortb_code_bytes_ * num_blocks_);
            size_t shortb_factors_begin = 0;
            size_t shortb_code_begin = 0;
            for (size_t i = 0; i < quant_plan.size(); ++i) {
                auto &c = segments_[i];

                c.short_factors_ = short_factors_ + shortb_factors_begin;
                shortb_factors_begin += c.shortb_factors_num_;
                c.shortb_factors_num_ = shortb_factors_fcnt_;

                c.short_code_ = short_code_ + shortb_code_begin;
                shortb_code_begin += c.shortb_code_bytes_;
                c.shortb_code_bytes_ = shortb_code_bytes_;
            }
            assert(shortb_factors_fcnt_ == shortb_factors_begin);
            assert(shortb_code_bytes_ == shortb_code_begin);
        }

        // assign long code and long_factor
        long_code_ = memory::align_mm<64, uint8_t>(longb_code_bytes_tot_);
        long_factors_ = memory::align_mm<64, ExFactor>(num_vec * num_segments_);
        ids_.resize(num_vec, 0);
        size_t longb_begin = 0;
        for (size_t i = 0; i < quant_plan.size(); ++i) {
            auto &c = segments_[i];
            if (use_compact_layout) {
                c.long_code_ = long_code_ + longb_begin;
                longb_begin += utils::rd_up_to_multiple_of(c.longb_code_bytes_ * num_vec, kLongCodeAlignBytes);
            } else {
                c.long_code_ = long_code_ + longb_begin;
                longb_begin += utils::rd_up_to_multiple_of(c.longb_code_bytes_, kLongCodeAlignBytes);
                c.longb_code_bytes_ = longb_code_bytes_;
            }

            c.long_factors_ = long_factors_ + i;
            c.ids_ = ids_.data();
        }
        assert(longb_begin == longb_code_bytes_tot_ || longb_begin == longb_code_bytes_);
    }

    ~SaqCluData() {
        if (short_factors_) {
            std::free(short_factors_);
        }
        std::free(short_code_);
        std::free(long_code_);
        std::free(long_factors_);
    }

    auto &get_segment(size_t idx) { return segments_[idx]; }
    auto &get_segment(size_t idx) const { return segments_[idx]; }

    /**
     * @brief Return pointer to ids
     */
    PID *ids() { return this->ids_.data(); }
    const PID *ids() const { return ids_.data(); }

    auto iter() const { return num_vec_ / KFastScanSize; }
    auto remain() const { return num_vec_ % KFastScanSize; }

    void load(std::ifstream &input) {
        input.read((char *)short_factors_, shortb_factors_fcnt_ * num_blocks_ * sizeof(float));
        input.read((char *)short_code_, shortb_code_bytes_ * num_blocks_);
        input.read((char *)long_code_, longb_code_bytes_ * num_vec_);
        input.read((char *)long_factors_, num_vec_ * num_segments_ * sizeof(ExFactor));
        input.read((char *)ids_.data(), ids_.size() * sizeof(PID));
        for (auto &clu : segments_) {
            input.read((char *)clu.centroid_.data(), clu.centroid_.cols() * sizeof(float));
        }
    }
    void save(std::ofstream &output) const {
        output.write((char *)short_factors_, shortb_factors_fcnt_ * num_blocks_ * sizeof(float));
        output.write((char *)short_code_, shortb_code_bytes_ * num_blocks_);
        output.write((char *)long_code_, longb_code_bytes_ * num_vec_);
        output.write((char *)long_factors_, num_vec_ * num_segments_ * sizeof(ExFactor));
        output.write((char *)ids_.data(), ids_.size() * sizeof(PID));
        for (auto &clu : segments_) {
            output.write((char *)clu.centroid_.data(), clu.centroid_.cols() * sizeof(float));
        }
    }
};
} // namespace saqlib
