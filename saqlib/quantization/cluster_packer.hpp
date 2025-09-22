#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "defines.hpp"
#include "quantization/cluster_data.hpp"
#include "quantization/fastscan/fastscan.hpp"
#include "utils/code_helper.hpp"
#include "utils/memory.hpp"

namespace saqlib {
/**
 * @brief Handles the packing of CAQ codes and factors for cluster data
 */
class ClusterPacker {
  private:
    const size_t num_dim_pad_;
    const size_t num_bits_;
    const size_t shortcode_byte_num_;
    const uint16_t short_bit_;
    const size_t total_blocks_;
    const bool use_fastscan_;
    void (*compacted_code_func_)(uint8_t *o_compact, const uint16_t *o_raw, size_t num_dim);

    // Cluster data reference
    CAQClusterData &clus_;

    // Memory buffers
    memory::UniqueArray<float> fac_o_l2norm_;
    memory::UniqueArray<float> fac_ip_cent_oa_;
    memory::UniqueArray<uint8_t> short_codes_;
    Uint16Vec long_code_;

  public:
    ClusterPacker(size_t num_dim_pad, size_t num_bits, CAQClusterData &clus, bool use_fastscan)
        : num_dim_pad_(num_dim_pad), num_bits_(num_bits),
          shortcode_byte_num_(num_dim_pad / 8),
          short_bit_(num_bits ? (1 << (num_bits - 1)) : 0),
          total_blocks_(clus.num_blocks()),
          use_fastscan_(use_fastscan),
          compacted_code_func_(num_bits ? utils::get_compacted_code16_func(num_bits - 1) : nullptr),
          clus_(clus),
          fac_o_l2norm_(memory::make_unique_array<float>(KFastScanSize * total_blocks_)),
          fac_ip_cent_oa_(memory::make_unique_array<float>(KFastScanSize * total_blocks_)),
          short_codes_(num_bits ? memory::make_unique_array<uint8_t>(
                                      shortcode_byte_num_ * KFastScanSize * total_blocks_)
                                : memory::make_unique_array<uint8_t>(0)),
          long_code_(1, num_dim_pad_) {
    }

    /**
     * @brief Pack a single quant data
     * @param i Index of the data point in the cluster
     * @param base_code CAQ single data containing codes and factors
     */
    void store_and_pack(size_t i, const QuantBaseCode &base_code) {
        fac_o_l2norm_[i] = base_code.o_l2norm;

        if (num_bits_ == 0) {
            return; // No packing needed for 0 bits
        }
        assert(num_dim_pad_ == static_cast<size_t>(base_code.code.size()));

        // Store short data
        fac_ip_cent_oa_[i] = base_code.ip_cent_oa; // Optional
        pack_short_codes(base_code.code, &short_codes_[i * shortcode_byte_num_]);

        // Store long data
        auto &ex_fac = clus_.long_factor(i);
        ex_fac.rescale = base_code.fac_rescale;
        ex_fac.error = base_code.fac_error;
        for (size_t j = 0; j < num_dim_pad_; ++j) {
            long_code_[j] = base_code.code[j] & (short_bit_ - 1);
        }
        compacted_code_func_(clus_.long_code(i), &long_code_(0, 0), num_dim_pad_);
    }

    /**
     * @brief Finalize packing and store all data to cluster
     */
    void finalize_and_store() {
        // Store short data block by block
        for (size_t i = 0; i < total_blocks_; ++i) {
            // copy codes
            if (num_bits_) {
                auto begin_idx = i * shortcode_byte_num_ * KFastScanSize;
                if (use_fastscan_) {
                    fastscan::pack_codes(num_dim_pad_,
                                         &short_codes_[begin_idx],
                                         KFastScanSize, clus_.short_code(i));
                } else {
                    // convert uint8_t to uint64_t for big-endian
                    for (size_t j = 0; j < shortcode_byte_num_ * KFastScanSize; j += 8) {
                        std::swap(short_codes_[begin_idx + j + 0], short_codes_[begin_idx + j + 7]);
                        std::swap(short_codes_[begin_idx + j + 1], short_codes_[begin_idx + j + 6]);
                        std::swap(short_codes_[begin_idx + j + 2], short_codes_[begin_idx + j + 5]);
                        std::swap(short_codes_[begin_idx + j + 3], short_codes_[begin_idx + j + 4]);
                    }
                    std::memcpy(clus_.short_code(i),
                                &short_codes_[begin_idx], shortcode_byte_num_ * KFastScanSize);
                }
            }

            // copy factors
            std::memcpy(clus_.factor_o_l2norm(i), &fac_o_l2norm_[i * KFastScanSize],
                        sizeof(float) * KFastScanSize);

            // ip_cent_oa is optional
            if (auto ip = clus_.factor_ip_cent_oa(i); ip) {
                std::memcpy(ip, &fac_ip_cent_oa_[i * KFastScanSize], sizeof(float) * KFastScanSize);
            }
        }
    }

  private:
    /**
     * @brief Pack short codes from CAQ codes
     * @param code Original CAQ codes (Eigen::VectorXi)
     * @param short_code_begin Output buffer for short codes
     */
    void pack_short_codes(const Eigen::VectorXi &code, uint8_t *short_code_begin) {
        for (size_t j = 0; j < shortcode_byte_num_; ++j) {
            uint8_t byte = 0;
            const size_t base_idx = j * 8;
            byte |= (code[base_idx + 0] & short_bit_) ? 0x80 : 0;
            byte |= (code[base_idx + 1] & short_bit_) ? 0x40 : 0;
            byte |= (code[base_idx + 2] & short_bit_) ? 0x20 : 0;
            byte |= (code[base_idx + 3] & short_bit_) ? 0x10 : 0;
            byte |= (code[base_idx + 4] & short_bit_) ? 0x08 : 0;
            byte |= (code[base_idx + 5] & short_bit_) ? 0x04 : 0;
            byte |= (code[base_idx + 6] & short_bit_) ? 0x02 : 0;
            byte |= (code[base_idx + 7] & short_bit_) ? 0x01 : 0;

            short_code_begin[j] = byte;
        }
    }
};
} // namespace saqlib
