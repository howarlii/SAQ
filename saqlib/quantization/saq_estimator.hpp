#pragma once

#include <cstring>
#include <immintrin.h>
#include <memory>

#include <glog/logging.h>

#include "defines.hpp"
#include "quantization/caq/caq_estimator.hpp"
#include "quantization/cluster_data.hpp"
#include "quantization/config.h"
#include "quantization/saq_data.hpp"

namespace saqlib {

template <typename CaqEstT>
class SaqEstimatorBase {
  protected:
    std::vector<CaqEstT> estimators_;

  public:
    /**
     * @param data Pointer to SAQ data containing multiple quantized base data structures
     * @param searcher_cfg Configuration parameters for search behavior
     * @param query Pointer to query vector
     */
    SaqEstimatorBase(const SaqData &data, const SearcherConfig &searcher_cfg, const FloatVec &query) {
        auto &data_variance = data.data_variance;
        auto &base_datas = data.base_datas;

        for (size_t i = 0, offset = 0; i < base_datas.size(); ++i) {
            const auto &bdata = base_datas[i];
            FloatVec curr_query;
            if (bdata.num_dim_pad + offset > data.num_dim) {
                curr_query = FloatVec::Zero(bdata.num_dim_pad);
                auto copy_size = query.cols() - offset;
                curr_query.head(copy_size) = query.segment(offset, copy_size);
            } else {
                curr_query = query.segment(offset, bdata.num_dim_pad);
            }
            auto vars2 = (data_variance.segment(offset, bdata.num_dim_pad).array() * curr_query.array().square()).sum();

            estimators_.emplace_back(bdata, searcher_cfg, curr_query).setPruneBound(std::sqrt(vars2));

            offset += bdata.num_dim_pad;
        }
    }

    virtual ~SaqEstimatorBase() = default;

    /**
     * @brief Get aggregated runtime performance metrics
     *
     * Collects and aggregates runtime statistics from all sub-estimators,
     * including bit operation counts for fast and accurate computations.
     *
     * @return QueryRuntimeMetrics Aggregated runtime performance metrics
     */
    auto getRuntimeMetrics() const {
        QueryRuntimeMetrics runtime_metrics;
        for (const auto &estimator : estimators_) {
            auto metrics = estimator.getRuntimeMetrics();
            runtime_metrics.acc_bitsum += metrics.acc_bitsum;
            runtime_metrics.fast_bitsum += metrics.fast_bitsum;
        }
        return runtime_metrics;
    }
};

template <DistType kDistType = DistType::Any>
class SaqCluEstimator : public SaqEstimatorBase<CaqCluEstimator<kDistType>> {
  protected:
    static constexpr size_t FAST_ARRAY = KFastScanSize / 16;
    static_assert(FAST_ARRAY == 2, "KFastScanSize must be 32 for SAQEstimator");

    using Base = SaqEstimatorBase<CaqCluEstimator<kDistType>>;
    using Base::estimators_;

    const SaqCluData *curr_saq_cluster_;

  public:
    /**
     * @brief Construct a new SAQEstimator object
     *
     * Initializes the SAQ estimator by creating individual CAQ estimators for each
     * quantizer in the data. Sets up query segmentation based on dimensional padding
     * and computes variance-based pruning bounds for each sub-quantizer.
     *
     * @param data Pointer to SAQ data containing multiple quantized base data structures
     * @param searcher_cfg Configuration parameters for search behavior
     * @param query Pointer to query vector (Eigen row vector format)
     */
    SaqCluEstimator(const SaqData &data, const SearcherConfig &searcher_cfg, const FloatVec &query) : Base(data, searcher_cfg, query) {
        CHECK(kDistType == DistType::Any || kDistType == searcher_cfg.dist_type) << "distance type mismatch";
    }

    virtual ~SaqCluEstimator() = default;

    auto &getEstimators() const { return estimators_; }

    /**
     * @brief Prepare estimators for search on specific cluster data
     *
     * Initializes all sub-estimators with the provided cluster data structure.
     * Must be called before performing distance computations on the clusters.
     *
     * @param saq_clust Pointer to SAQ cluster data to search within
     */
    void prepare(const SaqCluData *saq_clust) {
        curr_saq_cluster_ = saq_clust;
        DCHECK_EQ(estimators_.size(), saq_clust->num_segments_);
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            estimators_[c_i].prepare(&saq_clust->get_segment(c_i));
        }
    }

    /**
     * @brief Compute variance-based distance estimates for a block
     *
     * Computes distance estimates using variance information for early pruning.
     * Aggregates estimates from all sub-estimators into SIMD vectors.
     *
     * @param block_idx Index of the block to process (each block contains 32 vectors)
     * @param fst_distances Output array of 2 __m512 vectors containing distance estimates
     */
    void varsEstDist(size_t block_idx, __m512 *fst_distances) {
        fst_distances[0] = _mm512_setzero_ps();
        fst_distances[1] = _mm512_setzero_ps();
        __m512 cd[2];
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            estimators_[c_i].varsEstDist(block_idx, cd);

            fst_distances[0] = _mm512_add_ps(fst_distances[0], cd[0]);
            fst_distances[1] = _mm512_add_ps(fst_distances[1], cd[1]);
        }
    }

    /**
     * @brief Compute fast 1-bit distance estimates for a block
     *
     * Aggregates estimates from all sub-estimators into SIMD vectors for
     * efficient parallel processing.
     *
     * @param block_idx Index of the block to process (each block contains 32 vectors)
     * @param fst_distances Output array of 2 __m512 vectors containing distance estimates
     */
    void compFastDist(size_t block_idx, __m512 *fst_distances) {
        fst_distances[0] = _mm512_setzero_ps();
        fst_distances[1] = _mm512_setzero_ps();
        __m512 cd[2];
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            estimators_[c_i].compFastDist(block_idx, cd);

            fst_distances[0] = _mm512_add_ps(fst_distances[0], cd[0]);
            fst_distances[1] = _mm512_add_ps(fst_distances[1], cd[1]);
        }
    }

    /**
     * @brief Compute accurate distance for a specific vector
     *
     * Computes the exact distance using full precision quantized codes.
     * Aggregates accurate distances from all sub-estimators.
     *
     * @param idx Index of the vector within the current cluster data
     * @return float Accurate distance between query and the specified vector
     */
    float compAccurateDist(size_t idx) {
        DCHECK_LT(idx, curr_saq_cluster_->num_vec_);
        float acc_dist = 0;
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            acc_dist += estimators_[c_i].compAccurateDist(idx);
        }
        return acc_dist;
    }

    using Base::getRuntimeMetrics;
};

template <DistType kDistType = DistType::Any>
class SaqCluEstimatorSingle : public SaqEstimatorBase<CaqCluEstimatorSingle<kDistType>> {
  protected:
    using Base = SaqEstimatorBase<CaqCluEstimatorSingle<kDistType>>;
    using Base::estimators_;

    const SaqCluData *curr_saq_cluster_;

  public:
    /**
     * @param data Pointer to SAQ data containing multiple quantized base data structures
     * @param searcher_cfg Configuration parameters for search behavior
     * @param query Pointer to query vector
     */
    SaqCluEstimatorSingle(const SaqData &data, const SearcherConfig &searcher_cfg, const FloatVec &query) : Base(data, searcher_cfg, query) {
        CHECK(kDistType == DistType::Any || kDistType == searcher_cfg.dist_type) << "distance type mismatch";
    }

    virtual ~SaqCluEstimatorSingle() = default;

    auto &getEstimators() const { return estimators_; }

    /**
     * @brief Prepare estimators for search on specific cluster data
     *
     * Initializes all sub-estimators with the provided cluster data structure.
     * Must be called before performing distance computations on the clusters.
     *
     * @param saq_clust Pointer to SAQ cluster data to search within
     */
    void prepare(const SaqCluData *saq_clust) {
        curr_saq_cluster_ = saq_clust;
        DCHECK_EQ(estimators_.size(), saq_clust->num_segments_);
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            estimators_[c_i].prepare(&saq_clust->get_segment(c_i));
        }
    }

    float varsEstDist(size_t vec_idx) {
        float res = 0;
        for (auto &estimator : estimators_) {
            res += estimator.varsEstDist(vec_idx);
        }
        return res;
    }

    float compFastDist(size_t vec_idx) {
        float res = 0;
        for (auto &estimator : estimators_) {
            res += estimator.compFastDist(vec_idx);
        }
        return res;
    }

    float compAccurateDist(size_t vec_idx) {
        DCHECK_LT(vec_idx, curr_saq_cluster_->num_vec_);
        float acc_dist = 0;
        for (auto &estimator : estimators_) {
            acc_dist += estimator.compAccurateDist(vec_idx);
        }
        return acc_dist;
    }

    using Base::getRuntimeMetrics;
};

template <DistType kDistType = DistType::Any>
class SaqSingleEstimator : public SaqEstimatorBase<CaqSingleEstimator<kDistType>> {
  protected:
    using Base = SaqEstimatorBase<CaqSingleEstimator<kDistType>>;
    using Base::estimators_;

    const SaqCluData *curr_saq_cluster_;

  public:
    /**
     * @param data Pointer to SAQ data containing multiple quantized base data structures
     * @param searcher_cfg Configuration parameters for search behavior
     * @param query Pointer to query vector
     */
    SaqSingleEstimator(const SaqData &data, const SearcherConfig &searcher_cfg, const FloatVec &query) : Base(data, searcher_cfg, query) {
        CHECK(kDistType == DistType::Any || kDistType == searcher_cfg.dist_type) << "distance type mismatch";
    }

    virtual ~SaqSingleEstimator() = default;

    auto &getEstimators() const { return estimators_; }

    float varsEstDist(const SaqSingleDataWrapper &wrapper) {
        float res = 0;
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            res += estimators_[c_i].varsEstDist(wrapper.get_segment(c_i));
        }
        return res;
    }

    float compFastDist(const SaqSingleDataWrapper &wrapper) {
        float res = 0;
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            res += estimators_[c_i].compFastDist(wrapper.get_segment(c_i));
        }
        return res;
    }

    float compAccurateDist(const SaqSingleDataWrapper &wrapper) {
        float acc_dist = 0;
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            acc_dist += estimators_[c_i].compAccurateDist(wrapper.get_segment(c_i));
        }
        return acc_dist;
    }

    using Base::getRuntimeMetrics;
};

} // namespace saqlib
