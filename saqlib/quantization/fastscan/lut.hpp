#pragma once

#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <stdint.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "defines.hpp"
#include "quantization/fastscan/fastscan.hpp"
#include "quantization/fastscan/fastscan_highacc.hpp"
#include "utils/code_helper.hpp"
#include "utils/memory.hpp"

namespace saqlib
{

class Lut
{
  private:
    static constexpr size_t kNumBits = 8;
    static constexpr size_t kNumBitsHacc = 16;

    const bool use_highacc_ = true;
    const size_t num_dim_padded_;
    const size_t table_length_;
    const float one_over_sqrtD_;
    float (*const IP_FUNC)(const float *__restrict__, const uint8_t *__restrict__, size_t) = nullptr; // Function to get ip between query and long code
    // TODO: WARNING!!! the return type of IP_FUNC is float, but the return value should be double when B>13

    FloatVec query_;
    RowVector<uint8_t> lut_;
    float PORTABLE_ALIGN64 ip_xb_qprime_[KFastScanSize];
    float delta_ = 0;
    float sum_q_ = 0;
    float sum_vl_lut_;
    float q_l2sqr_;
    float q_l2norm_ = 0; // q2c_dist for L2Sqr, or |q| for IP

    void packHighAccLUT();

  public:
    explicit Lut(size_t num_dim_padded, size_t ex_bits)
        : num_dim_padded_(num_dim_padded), table_length_(num_dim_padded / 8 * KFastScanSize),
          one_over_sqrtD_(1.0 / std::sqrt((float)num_dim_padded_)),
          IP_FUNC(utils::get_IP_FUNC(ex_bits))
    {
        lut_ = RowVector<uint8_t>::Zero(table_length_ * (use_highacc_ ? 2 : 1));
    }

    ~Lut() = default;

    float getQL2Sqr() const
    {
        return q_l2sqr_;
    }

    void prepare(FloatVec query)
    {
        query_ = std::move(query);

        sum_q_ = query_.sum();
        q_l2sqr_ = query_.squaredNorm();
        q_l2norm_ = std::sqrt(q_l2sqr_);

        FloatVec lut_float(table_length_);
        fastscan::pack_lut(num_dim_padded_, query_.data(), lut_float.data());
        float vl_lut = lut_float.minCoeff();
        float vr_lut = lut_float.maxCoeff();

        if (use_highacc_) {
            delta_ = (vr_lut - vl_lut) / ((1 << kNumBitsHacc) - 0.01); // prevent the result > (code_max)

            Uint16Vec lut_u16(table_length_);
            lut_u16 = ((lut_float.array() - vl_lut) / delta_).cast<uint16_t>();
            fastscan::transfer_lut_hacc(lut_u16.data(), num_dim_padded_, lut_.data());
        } else {
            delta_ = (vr_lut - vl_lut) / ((1 << kNumBits) - 0.01); // prevent the result > (code_max)
            Eigen::Map<RowVector<uint8_t>> lut_map(lut_.data(), 1, table_length_);
            lut_map = ((lut_float.array() - vl_lut) / delta_).cast<uint8_t>();
        }

        size_t num_table = table_length_ / 16;
        sum_vl_lut_ = (vl_lut + 0.5 * delta_) * static_cast<float>(num_table);
    }

    void compFastIP(
        const float *o_l2norm,
        const uint8_t *short_code,
        __m512 *fst_distances)
    {
        // TODO: support no highacc
        __m512i res[2];
        fastscan::accumulate_hacc(short_code, lut_.data(), res, num_dim_padded_);

        constexpr float const_bound = 0.58;
        // TODO: replace this const_bound with true_error_factor
        constexpr float est_ip_o_oa = 0.8;

        __m512 simd_sum_vl_lut = _mm512_set1_ps(sum_vl_lut_);
        __m512 simd_delta = _mm512_set1_ps(delta_);
        __m512 simd_sumq_const_bound = _mm512_set1_ps(0.5 * sum_q_ - const_bound * q_l2norm_);
        __m512 simd_rescale_over_sqrtD = _mm512_set1_ps(4 / est_ip_o_oa * one_over_sqrtD_);

        for (size_t i = 0; i < 2; i++) {
            __m512 tmp = _mm512_mul_ps(_mm512_cvtepi32_ps(res[i]), simd_delta);
            tmp = _mm512_add_ps(tmp, simd_sum_vl_lut);
            _mm512_store_ps(&ip_xb_qprime_[i * 16], tmp);

            if (fst_distances) {
                tmp = _mm512_mul_ps(_mm512_sub_ps(tmp, simd_sumq_const_bound), simd_rescale_over_sqrtD);
                // unaligned
                tmp = _mm512_mul_ps(tmp, _mm512_loadu_ps(o_l2norm + i * 16));
                fst_distances[i] = tmp;
            }
        }
    }

    float getExtIP(const uint8_t *long_code, float delta, size_t j)
    {
        constexpr double vl = -1;
        double ex_ip = IP_FUNC(query_.data(), long_code, num_dim_padded_);
        return (ip_xb_qprime_[j] + ex_ip * delta + (vl + delta / 2) * sum_q_);
    }
};

} // namespace saqlib
