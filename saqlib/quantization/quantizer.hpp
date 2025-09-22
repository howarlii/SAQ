#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>

#include "glog/logging.h"

#include "defines.hpp"
#include "quantization/caq/caq_encoder.hpp"
#include "quantization/cluster_data.hpp"
#include "quantization/cluster_packer.hpp"
#include "quantization/config.h"
#include "quantization/quantizer_data.hpp"
#include "quantization/single_data.hpp"
#include "utils/pool.hpp"
#include "utils/rotator.hpp"

namespace saqlib {
struct QuantMetrics {
    utils::AvgMaxRecorder norm_ip_o_oa; // Inner product relative error metrics
};

class QuantizerCluster {
  public:
    const size_t num_bits_;    // Number of bits for quantization
    const size_t num_dim_pad_; // padded number of dimensions

    const BaseQuantizerData *data_;

  public:
    mutable QuantMetrics metrics_; // Inner product relative error

    QuantizerCluster(const BaseQuantizerData *data)
        : num_bits_(data->num_bits), num_dim_pad_(data->num_dim_pad), data_(data) {
    }

    virtual ~QuantizerCluster() {}

    virtual void quantize(const FloatRowMat &or_vecs, const FloatVec &centroid, CAQClusterData &clus) const {
        CHECK_EQ(or_vecs.cols(), num_dim_pad_) << "Input vector dimension does not match quantizer dimension";
        CHECK_EQ(centroid.cols(), num_dim_pad_) << "Centroid dimension does not match quantizer dimension";

        FloatRowMat o_vecs;
        if (data_->rotator) {
            clus.centroid() = centroid * data_->rotator->get_P(); // rotated centroid
            o_vecs = (or_vecs * data_->rotator->get_P()).rowwise() - clus.centroid();
        } else {
            clus.centroid() = centroid; // centroid
            o_vecs = or_vecs.rowwise() - clus.centroid();
        }

        const size_t num_points = clus.num_vec(); // Num of point in this cluster

        // TODO: Support other quantization types
        CHECK(data_->cfg.quant_type == BaseQuantType::CAQ) << "Only CAQ is supported for DataQuantizer";
        CAQEncoder encoder(num_dim_pad_, num_bits_, data_->cfg);
        ClusterPacker packer(num_dim_pad_, num_bits_, clus, data_->cfg.use_fastscan);

        QuantBaseCode base_code;
        for (size_t i = 0; i < num_points; ++i) {
            const auto &curr_vec = o_vecs.row(i);

            encoder.encode_and_fac(curr_vec, base_code, &centroid);
            packer.store_and_pack(i, base_code);

            // Update metrics
            metrics_.norm_ip_o_oa.insert(base_code.norm_ip_o_oa);
        }

        // Finalize and store all packed data
        packer.finalize_and_store();
    }
};

/**
 * @brief Single vector quantizer for individual vector quantization
 *
 * Unlike QuantizerCluster which processes batches of vectors for a single cluster,
 * QuantizerSingle processes multiple vectors individually, each with its own quantization data.
 */
class QuantizerSingle {
  public:
    const size_t num_bits_;    // Number of bits for quantization
    const size_t num_dim_pad_; // padded number of dimensions

    const BaseQuantizerData *data_;

  public:
    mutable QuantMetrics metrics_; // Inner product relative error

    QuantizerSingle(const BaseQuantizerData *data)
        : num_bits_(data->num_bits), num_dim_pad_(data->num_dim_pad), data_(data) {
        CHECK_EQ(data->cfg.use_fastscan, false) << "Fastscan not supported for single vector quantizer";
        CHECK(data_->cfg.quant_type == BaseQuantType::CAQ) << "Only CAQ is supported for QuantizerSingle";
    }

    virtual ~QuantizerSingle() {}

    /**
     * @brief Quantize a batch of vectors individually
     * @param or_vecs Original vectors to quantize (each row is a vector)
     * @param caq_data Target single data wrapper
     */
    virtual void quantize(const FloatVec &or_vecs, CaqSingleDataWrapper *caq_data) const {
        CHECK_EQ(or_vecs.cols(), num_dim_pad_) << "Input vector dimension does not match quantizer dimension";

        FloatVec o_vecs;

        // Apply rotation if available
        if (data_->rotator) {
            o_vecs = or_vecs * data_->rotator->get_P();
        } else {
            o_vecs = or_vecs;
        }

        CAQEncoder encoder(num_dim_pad_, num_bits_, data_->cfg);
        QuantBaseCode base_code;
        encoder.encode_and_fac(o_vecs, base_code, nullptr);

        // Store quantization results
        store_quantization_result(caq_data, base_code);

        // Update metrics
        metrics_.norm_ip_o_oa.insert(base_code.norm_ip_o_oa);
    }

  private:
    /**
     * @brief Store quantization result into single data wrapper
     * @param single_data Target single data wrapper
     * @param base_code Quantization result to store
     */
    void store_quantization_result(CaqSingleDataWrapper *single_data, const QuantBaseCode &base_code) const {
        // Store L2 norm factor
        single_data->factor_o_l2norm() = base_code.o_l2norm;

        if (num_bits_ == 0) {
            return; // No packing needed for 0 bits
        }

        DCHECK_EQ(num_dim_pad_, static_cast<size_t>(base_code.code.size()));

        // Store ip_cent_oa factor (can be used for other purposes)
        single_data->factor_ip_cent_oa() = 0.0f; // Set to zero since no centroid

        // Pack short codes
        pack_short_codes(base_code.code, single_data->short_code());

        // Store long factors
        auto &ex_fac = single_data->long_factor();
        ex_fac.rescale = base_code.fac_rescale;
        ex_fac.error = base_code.fac_error;

        // Pack long codes
        if (num_bits_ > 1) {
            pack_long_codes(base_code.code, single_data->long_code());
        }
    }

    /**
     * @brief Pack short codes from CAQ codes
     * @param code Original CAQ codes (Eigen::VectorXi)
     * @param short_code_begin Output buffer for short codes
     */
    void pack_short_codes(const Eigen::VectorXi &code, uint8_t *short_code_begin) const {
        const size_t shortcode_byte_num = num_dim_pad_ / 8;
        const uint16_t short_bit = num_bits_ ? (1 << (num_bits_ - 1)) : 0;

        for (size_t j = 0; j < shortcode_byte_num; ++j) {
            uint8_t byte = 0;
            const size_t base_idx = j * 8;

            // Pack 8 bits into one byte (MSB first)
            byte |= (code[base_idx + 0] & short_bit) ? 0x80 : 0;
            byte |= (code[base_idx + 1] & short_bit) ? 0x40 : 0;
            byte |= (code[base_idx + 2] & short_bit) ? 0x20 : 0;
            byte |= (code[base_idx + 3] & short_bit) ? 0x10 : 0;
            byte |= (code[base_idx + 4] & short_bit) ? 0x08 : 0;
            byte |= (code[base_idx + 5] & short_bit) ? 0x04 : 0;
            byte |= (code[base_idx + 6] & short_bit) ? 0x02 : 0;
            byte |= (code[base_idx + 7] & short_bit) ? 0x01 : 0;

            auto j_cov = j + 7 - 2 * (j % 8); // reverse every 8 uint8_t for no-fastscan
            short_code_begin[j_cov] = byte;
        }
    }

    /**
     * @brief Pack long codes from CAQ codes
     * @param code Original CAQ codes (Eigen::VectorXi)
     * @param long_code_begin Output buffer for long codes
     */
    void pack_long_codes(const Eigen::VectorXi &code, uint8_t *long_code_begin) const {
        const uint16_t short_bit = 1 << (num_bits_ - 1);
        auto compacted_code_func = utils::get_compacted_code16_func(num_bits_ - 1);

        Uint16Vec long_code_buffer;
        long_code_buffer.resize(num_dim_pad_);
        for (size_t j = 0; j < num_dim_pad_; ++j) {
            long_code_buffer[j] = code[j] & (short_bit - 1);
        }

        // Compact the long codes
        compacted_code_func(long_code_begin, &long_code_buffer(0, 0), num_dim_pad_);
    }
};

} // namespace saqlib
