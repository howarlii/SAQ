#pragma once

#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <stdint.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "defines.hpp"
#include "quantization/cluster_data.hpp"
#include "quantization/config.h"
#include "quantization/fastscan/lut.hpp"
// #include "quantization/fastscan/lut.old.hpp"
#include "quantization/quantizer_data.hpp"
#include "quantization/single_data.hpp"

namespace saqlib {
struct QueryRuntimeMetrics {
    size_t fast_bitsum = 0;
    size_t acc_bitsum = 0;
    size_t total_comp_cnt = 0;
};

template <DistType kDistType = DistType::Any>
class CaqCluEstimator {
  private:
    const size_t num_dim_padded_;
    const uint8_t num_bits_;
    const uint8_t ex_bits_;
    const SearcherConfig cfg_;

    FloatVec query_data_;
    float without_ip_prune_bound_ = 0;
    const double sq_delta_ = 0;
    float ip_q_c_ = 0;
    float q_l2sqr_ = 0;
    Lut lut_;
    const CAQClusterData *curr_cluster_;

    QueryRuntimeMetrics runtime_statics_;

    bool isIpDist() { return kDistType == DistType::IP || (kDistType == DistType::Any && cfg_.dist_type == DistType::IP); }

  public:
    /**
     * @brief Construct a new CaqEstimator object. This estimator can efficiently compute
     * distances between a query vector and a batch of quantized vectors.
     *
     * @param data Pointer to base quantizer data containing quantized vectors and metadata
     * @param cfg Configuration parameters for search behavior and distance type
     * @param query Pointer to query vector segment (FloatVec format)
     */
    explicit CaqCluEstimator(const BaseQuantizerData &data, SearcherConfig cfg, const FloatVec &query)
        : num_dim_padded_(data.num_dim_pad), num_bits_(data.num_bits),
          ex_bits_(num_bits_ ? num_bits_ - 1 : 0), cfg_(std::move(cfg)),
          sq_delta_(2.0 / (1 << num_bits_)),
          lut_(num_dim_padded_, ex_bits_) {
        CHECK(kDistType == DistType::Any || kDistType == cfg_.dist_type) << "distance type mismatch";
        CHECK(data.cfg.use_fastscan) << "CaqEstimator require fastscan enabled. Please use CaqSingleEstimator instead.";
        if (data.rotator) {
            query_data_ = query * data.rotator->get_P();
        } else {
            query_data_ = query;
        }
    }

    ~CaqCluEstimator() = default;

    /**
     * @brief Set variance-based pruning bound
     *
     * Sets the pruning threshold based on variance analysis for early
     * termination of distance computations that exceed the bound.
     *
     * @param vars Variance value used to compute the pruning bound
     */
    void setPruneBound(float vars) {
        without_ip_prune_bound_ = vars * cfg_.searcher_vars_bound_m;
    }

    auto getRuntimeMetrics() const { return runtime_statics_; }

    /**
     * @brief Prepare data before search
     *
     * Initializes lookup tables and precomputes values needed for distance
     * Must call before any distance computation.
     *
     * @param cur_cluster Pointer to current cluster data
     */
    void prepare(const CAQClusterData *cur_cluster) {
        // TODO: prepare only once instead of for each cluster, if factor_ip_cent_oa is set.
        curr_cluster_ = cur_cluster;
        const auto &centroid = cur_cluster->centroid();
        if (isIpDist()) {
            ip_q_c_ = query_data_.dot(centroid);
            lut_.prepare(query_data_);
            q_l2sqr_ = lut_.getQL2Sqr();
        } else {
            lut_.prepare(query_data_ - centroid);
            q_l2sqr_ = lut_.getQL2Sqr();
        }
    }

    /**
     * @brief Compute variance-based distance estimates for a block
     *
     * Computes distance estimates using variance information for early pruning.
     *
     * @param block_idx Index of the block to process (each block contains 32 vectors)
     * @param fst_distances Output array of 2 __m512 vectors containing distance estimates
     */
    void varsEstDist(size_t block_idx, __m512 *fst_distances) {
        if (!fst_distances) {
            return;
        }
        if (isIpDist()) {
            __m512 factor_vec = _mm512_set1_ps(ip_q_c_ - without_ip_prune_bound_);
            for (size_t j = 0; j < KFastScanSize; j += 16) {
                fst_distances[j / 16] = factor_vec;
            }
            return;
        }
        const float *factor_x = curr_cluster_->factor_o_l2norm(block_idx); // (o_r-c)^2, sqr_x
        __m512 factor_vec = _mm512_set1_ps(q_l2sqr_ - 2 * without_ip_prune_bound_);
        __m512 zero_vec = _mm512_setzero_ps();

        for (size_t j = 0; j < KFastScanSize; j += 16) {
            __m512 factor_x_vec = _mm512_load_ps(&factor_x[j]);
            __m512 squared_vec = _mm512_mul_ps(factor_x_vec, factor_x_vec);
            fst_distances[j / 16] = _mm512_max_ps(zero_vec, _mm512_add_ps(squared_vec, factor_vec));
        }
    }

    /**
     * @brief Compute fast distance estimates for a block
     *
     * Computes 1-bit distances using quantized data and lookup tables.
     * Must call compFastDist(block_idx) before compAccurateDist(vec_idx), where vec_idx is inside block_idx.
     * If num_bits_ is 0, falls back to variance estimation.
     *
     * @param block_idx Index of the block to process (each block contains 32 vectors)
     * @param fst_distances Output array of 2 __m512 vectors containing distance estimates (can be nullptr)
     */
    void compFastDist(size_t block_idx, __m512 *fst_distances) {
        if (num_bits_ == 0) {
            varsEstDist(block_idx, fst_distances);
            return;
        }

        const float *o_l2norm = curr_cluster_->factor_o_l2norm(block_idx); // |o_r-c|, |x|
        lut_.compFastIP(o_l2norm, curr_cluster_->short_code(block_idx), fst_distances);

        if (fst_distances == nullptr) {
            return;
        }

        if (isIpDist()) {
            __m512 simd_qc_ip = _mm512_set1_ps(ip_q_c_);
            __m512 naghalf_c = _mm512_set1_ps(0.5);
            fst_distances[0] = _mm512_add_ps(_mm512_mul_ps(fst_distances[0], naghalf_c), simd_qc_ip);
            fst_distances[1] = _mm512_add_ps(_mm512_mul_ps(fst_distances[1], naghalf_c), simd_qc_ip);
        } else {
            __m512 simd_q2c_dist2 = _mm512_set1_ps(q_l2sqr_);
            __m512 simd_x0 = _mm512_loadu_ps(o_l2norm);
            __m512 simd_x1 = _mm512_loadu_ps(o_l2norm + 16);
            fst_distances[0] = _mm512_add_ps(_mm512_mul_ps(simd_x0, simd_x0),
                                             _mm512_sub_ps(simd_q2c_dist2, fst_distances[0]));
            fst_distances[1] = _mm512_add_ps(_mm512_mul_ps(simd_x1, simd_x1),
                                             _mm512_sub_ps(simd_q2c_dist2, fst_distances[1]));
            fst_distances[0] = _mm512_max_ps(_mm512_setzero_ps(), fst_distances[0]);
            fst_distances[1] = _mm512_max_ps(_mm512_setzero_ps(), fst_distances[1]);
        }

        runtime_statics_.fast_bitsum += KFastScanSize * num_dim_padded_;
    }

    /**
     * @brief Compute accurate distance for a specific vector
     *
     * Computes the exact distance using full precision quantized codes.
     * Must call compFastDist(block_idx) before compAccurateDist(vec_idx), where vec_idx is inside block_idx.
     *
     * @param vec_idx Index of the vector within the current cluster data
     * @return float Accurate distance between query and the specified vector
     */
    float compAccurateDist(size_t vec_idx) {
        auto blk_idx = vec_idx / KFastScanSize;
        auto j = vec_idx % KFastScanSize;
        // For L2 distance, we need to compute the squared distance
        const auto o_l2norm = curr_cluster_->factor_o_l2norm(blk_idx)[j];
        const float o_l2sqr = o_l2norm * o_l2norm;
        if (num_bits_ == 0) {
            if (isIpDist()) {
                return ip_q_c_;
            } else {
                return o_l2sqr + q_l2sqr_;
            }
        }

        const uint8_t *long_code = curr_cluster_->long_code(vec_idx);
        const ExFactor &ex_fac = curr_cluster_->long_factor(vec_idx);

        float ip_o_q = ex_fac.rescale * lut_.getExtIP(long_code, sq_delta_, j);

        runtime_statics_.acc_bitsum += num_dim_padded_ * (num_bits_ - 1);

        if (cfg_.dist_type == DistType::IP) {
            return ip_o_q + ip_q_c_;
        } else {
            float est_dist = o_l2sqr + q_l2sqr_ - 2 * ip_o_q;
            return est_dist;
        }
    }
};

template <DistType kDistType = DistType::Any>
class CaqEstimatorSingleImpl {
  protected:
    static constexpr size_t kNumBits = 8;

    const size_t num_dim_padded_;
    const uint8_t num_bits_;
    const uint8_t ex_bits_;
    const float one_over_sqrtD_;
    float (*const IP_FUNC)(const float *__restrict__, const uint8_t *__restrict__, size_t) = nullptr; // Function to get ip between query and long code

    const SearcherConfig cfg_;
    const double caq_delta_ = 0;

    FloatVec curr_query_;
    RowVector<uint16_t> query_sq_;
    RowVector<uint64_t> query_bin_;
    float q_delta_, q_vl_, q_vr_;
    float ip_q_c_ = 0;
    float delta_ = 0;
    float sum_q_ = 0;
    float q_l2sqr_;
    float q_l2norm_ = 0; // q2c_dist for L2Sqr, or |q| for IP

    float without_ip_prune_bound_ = 0;

    QueryRuntimeMetrics runtime_statics_;

    bool isIpDist() { return kDistType == DistType::IP || (kDistType == DistType::Any && cfg_.dist_type == DistType::IP); }

  public:
    /**
     * @brief Construct a new CaqEstimator object. This estimator can efficiently compute
     * distances between a query vector and a quantized vectors.
     *
     * @param data Pointer to base quantizer data containing quantized vectors and metadata
     * @param cfg Configuration parameters for search behavior and distance type
     * @param query Pointer to query vector segment (FloatVec format)
     */
    explicit CaqEstimatorSingleImpl(const BaseQuantizerData &data, SearcherConfig cfg)
        : num_dim_padded_(data.num_dim_pad), num_bits_(data.num_bits),
          ex_bits_(num_bits_ ? num_bits_ - 1 : 0),
          one_over_sqrtD_(1.0 / std::sqrt((float)num_dim_padded_)),
          IP_FUNC(utils::get_IP_FUNC(ex_bits_)),
          cfg_(std::move(cfg)), caq_delta_(2.0 / (1 << num_bits_)) {
        CHECK(kDistType == DistType::Any || kDistType == cfg_.dist_type) << "distance type mismatch";
        CHECK(!data.cfg.use_fastscan) << "CaqSingleEstimator require fastscan disabled. Please use CaqEstimator instead.";
    }

    virtual ~CaqEstimatorSingleImpl() = default;

    /**
     * @brief Set variance-based pruning bound
     *
     * Sets the pruning threshold based on variance analysis for early
     * termination of distance computations that exceed the bound.
     *
     * @param vars Variance value used to compute the pruning bound
     */
    void setPruneBound(float vars) {
        without_ip_prune_bound_ = vars * cfg_.searcher_vars_bound_m;
    }

    auto &getRuntimeMetrics() const { return runtime_statics_; }

    /**
     * @brief Preparing data before search
     *
     * Initializes lookup tables and precomputes values needed for distance
     * Must call before any distance computation.
     *
     */
    void prepare(FloatVec query) {
        curr_query_ = std::move(query);
        q_l2sqr_ = curr_query_.squaredNorm();
        q_l2norm_ = std::sqrt(q_l2sqr_);
        sum_q_ = curr_query_.sum();

        q_vl_ = curr_query_.minCoeff();
        q_vr_ = curr_query_.maxCoeff();
        delta_ = (q_vr_ - q_vl_) / ((1 << kNumBits) - 0.01); // prevent the result > (code_max)
        query_sq_ = ((curr_query_.array() - q_vl_) / delta_).cast<uint16_t>();

        // delta_ = q_l2sqr_ / curr_query_.dot(query_sq_.cast<float>()); // can be remove (better or not?)

        query_bin_.resize(num_dim_padded_ / 64 * kNumBits);
        utils::new_transpose_bin(
            query_sq_.data(), query_bin_.data(), num_dim_padded_, kNumBits);
    }

    float varsEstDist(float o_l2norm) {
        if (isIpDist()) {
            return ip_q_c_ - without_ip_prune_bound_;
        }
        return std::max(0.0f, o_l2norm * o_l2norm + q_l2sqr_ - 2 * without_ip_prune_bound_);
    }

    float compFastDist(float o_l2norm, const uint64_t *short_code) {
        if (num_bits_ == 0) {
            return varsEstDist(o_l2norm);
        }
        constexpr float const_bound = 0.58;
        constexpr float est_error = 0.8;

        float tmp = utils::warmup_ip_x0_q(
            short_code,
            query_bin_.data(),
            delta_,
            q_vl_ + 0.5 * delta_,
            num_dim_padded_,
            kNumBits);
        float ip_oa1_qq = (tmp - (0.5 * sum_q_ - const_bound * q_l2norm_)) * (4 / est_error * one_over_sqrtD_) * o_l2norm;

        runtime_statics_.fast_bitsum += num_dim_padded_;

        if (!isIpDist()) {
            return std::max(q_l2sqr_ + o_l2norm * o_l2norm - ip_oa1_qq, 0.0f);
        } else {
            return ip_oa1_qq * 0.5;
        }
    }

    float compAccurateDist(float o_l2norm, const uint64_t *short_code, const uint8_t *long_code, const ExFactor &ex_fac) {
        const auto o_l2sqr = o_l2norm * o_l2norm;
        if (num_bits_ == 0) {
            if (isIpDist()) {
                return ip_q_c_;
            } else {
                return o_l2sqr + q_l2sqr_;
            }
        }

        float ip_oa1_q = utils::mask_ip_x0_q(curr_query_.data(), short_code, num_dim_padded_);

        constexpr double o_vl = -1;
        double ex_ip = IP_FUNC(curr_query_.data(), long_code, num_dim_padded_);
        float tmp = (ip_oa1_q + ex_ip * caq_delta_ + (o_vl + caq_delta_ / 2) * sum_q_);

        float ip_o_q = ex_fac.rescale * tmp;

        runtime_statics_.acc_bitsum += num_dim_padded_ * (num_bits_ - 1);

        if (cfg_.dist_type == DistType::IP) {
            return ip_o_q + ip_q_c_;
        } else {
            float est_dist = o_l2sqr + q_l2sqr_ - 2 * ip_o_q;
            return est_dist;
        }
    }
};

template <DistType kDistType = DistType::Any>
class CaqCluEstimatorSingle : public CaqEstimatorSingleImpl<kDistType> {
    using Impl = CaqEstimatorSingleImpl<kDistType>;

    FloatVec query_data_;
    const CAQClusterData *curr_cluster_;

  public:
    /**
     * @brief Construct a new CaqEstimator object. This estimator can efficiently compute
     * distances between a query vector and a quantized vectors.
     *
     * @param data Pointer to base quantizer data containing quantized vectors and metadata
     * @param cfg Configuration parameters for search behavior and distance type
     * @param query Pointer to query vector segment (FloatVec format)
     */
    explicit CaqCluEstimatorSingle(const BaseQuantizerData &data, SearcherConfig cfg, const FloatVec &query)
        : Impl(data, std::move(cfg)) {
        if (data.rotator) {
            query_data_ = query * data.rotator->get_P();
        } else {
            query_data_ = query;
        }
    }

    virtual ~CaqCluEstimatorSingle() = default;

    using Impl::getRuntimeMetrics;
    using Impl::isIpDist;
    using Impl::setPruneBound;

    /**
     * @brief Preparing data before search
     *
     * Initializes lookup tables and precomputes values needed for distance
     * Must call before any distance computation.
     *
     * @param cur_cluster Pointer to current cluster data
     */
    void prepare(const CAQClusterData *cur_cluster) {
        curr_cluster_ = cur_cluster;
        const auto &centroid = cur_cluster->centroid();
        if (!isIpDist()) {
            Impl::prepare(query_data_ - centroid);
        } else {
            throw std::runtime_error("not implemented yet");
            Impl::prepare(query_data_);
        }
    }

    float varsEstDist(size_t vec_idx) {
        auto block_idx = vec_idx / KFastScanSize;
        auto j = vec_idx % KFastScanSize;
        auto o_l2norm = curr_cluster_->factor_o_l2norm(block_idx)[j];
        return Impl::varsEstDist(o_l2norm);
    }

    float compFastDist(size_t vec_idx) {
        auto block_idx = vec_idx / KFastScanSize;
        auto j = vec_idx % KFastScanSize;
        auto o_l2norm = curr_cluster_->factor_o_l2norm(block_idx)[j];
        auto short_code = (const uint64_t *)curr_cluster_->short_code_single(vec_idx);

        return Impl::compFastDist(o_l2norm, short_code);
    }

    float compAccurateDist(size_t vec_idx) {
        auto block_idx = vec_idx / KFastScanSize;
        auto j = vec_idx % KFastScanSize;
        auto o_l2norm = curr_cluster_->factor_o_l2norm(block_idx)[j];
        auto short_code = (const uint64_t *)curr_cluster_->short_code_single(vec_idx);
        const uint8_t *long_code = curr_cluster_->long_code(vec_idx);
        const ExFactor &ex_fac = curr_cluster_->long_factor(vec_idx);

        return Impl::compAccurateDist(o_l2norm, short_code, long_code, ex_fac);
    }
};

template <DistType kDistType = DistType::Any>
class CaqSingleEstimator : public CaqEstimatorSingleImpl<kDistType> {
    using Impl = CaqEstimatorSingleImpl<kDistType>;

  public:
    /**
     * @brief Construct a new CaqEstimator object. This estimator can efficiently compute
     * distances between a query vector and a quantized vectors.
     *
     * @param data Pointer to base quantizer data containing quantized vectors and metadata
     * @param cfg Configuration parameters for search behavior and distance type
     * @param query Pointer to query vector segment (FloatVec format)
     */
    explicit CaqSingleEstimator(const BaseQuantizerData &data, SearcherConfig cfg, const FloatVec &query)
        : Impl(data, std::move(cfg)) {
        if (data.rotator) {
            Impl::prepare(query * data.rotator->get_P());
        } else {
            Impl::prepare(query);
        }
    }

    virtual ~CaqSingleEstimator() = default;

    using Impl::getRuntimeMetrics;
    using Impl::isIpDist;
    using Impl::setPruneBound;

    float varsEstDist(const CaqSingleDataWrapper &caq) {
        auto o_l2norm = caq.factor_o_l2norm();
        return Impl::varsEstDist(o_l2norm);
    }

    float compFastDist(const CaqSingleDataWrapper &caq) {
        auto o_l2norm = caq.factor_o_l2norm();
        auto short_code = (const uint64_t *)caq.short_code();
        return Impl::compFastDist(o_l2norm, short_code);
    }

    float compAccurateDist(const CaqSingleDataWrapper &caq) {
        auto o_l2norm = caq.factor_o_l2norm();
        auto short_code = (const uint64_t *)caq.short_code();
        const uint8_t *long_code = caq.long_code();
        const ExFactor &ex_fac = caq.long_factor();
        // DCHECK_EQ(reinterpret_cast<uintptr_t>(long_code) % 64, 0) << "long_code must be 64-byte aligned";

        return Impl::compAccurateDist(o_l2norm, short_code, long_code, ex_fac);
    }
};
} // namespace saqlib
