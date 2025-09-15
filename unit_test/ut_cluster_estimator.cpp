#include <memory>
#include <vector>

#include <fmt/core.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "defines.hpp"
#include "quantization/cluster_data.hpp"
#include "quantization/config.h"
#include "quantization/saq_data.hpp"
#include "quantization/saq_estimator.hpp"
#include "quantization/saq_quantizer.hpp"
#include "test_base.hpp"
#include "utils/pool.hpp"

using namespace saqlib;

class CluEstimatorTest : public ::testing::Test, public TestBase {
  protected:
    bool debug_ = true;
    size_t num_data_ = 64;
    size_t num_query_ = 4;
    size_t num_dim_ = 512;
    size_t num_centroids_ = 1;
    float avg_bits_ = 4.0f;

    std::vector<PID> cluster_ids_;
    std::unique_ptr<SaqData> saq_data_;
    std::unique_ptr<SaqCluData> cluster_;
    QuantizeConfig config_;
    SearcherConfig searcher_config_;

    void SetUp() override {
        LOG(INFO) << "CTEST_FULL_OUTPUT";
#ifdef NDEBUG
        debug_ = false;
        num_data_ = 1024;
        num_query_ = 16;
        num_dim_ = 768;
#endif

        // Generate cluster IDs for quantization
        cluster_ids_.resize(num_data_);
        for (size_t i = 0; i < num_data_; ++i) {
            cluster_ids_[i] = i;
        }

        // Setup quantization config
        config_.avg_bits = avg_bits_;
        config_.enable_segmentation = false;

        // Setup searcher config
        searcher_config_.searcher_vars_bound_m = 4.0f;
        searcher_config_.dist_type = DistType::L2Sqr;
    }

    void gen() {
        generateTestData(num_data_, num_query_, num_dim_, num_centroids_);
        LOG(INFO) << fmt::format("Generated synthetic data: {} vectors, {} dims, {} centroids",
                                 num_data_, num_dim_, num_centroids_);
    }

    void load(const std::string &dataset_name, bool use_pca = false) {
        // Load test data from file
        loadTestData(dataset_name, use_pca);

        // Resize loaded data to match desired sizes
        if (data_.rows() > static_cast<Eigen::Index>(num_data_)) {
            data_.conservativeResize(num_data_, data_.cols());
        }
        if (query_.rows() > static_cast<Eigen::Index>(num_query_)) {
            query_.conservativeResize(num_query_, query_.cols());
        }
        if (centroids_.rows() > static_cast<Eigen::Index>(num_centroids_)) {
            centroids_.conservativeResize(num_centroids_, centroids_.cols());
        }

        // Update dimensions based on loaded data
        num_dim_ = data_.cols();

        LOG(INFO) << fmt::format("Loaded {} data: {} vectors, {} dims, {} centroids",
                                 dataset_name, num_data_, num_dim_, num_centroids_);
    }

    void quantize() {
        // Create SAQ data maker and set variance
        SaqDataMaker data_maker(config_, num_dim_);
        data_maker.compute_variance(data_);
        saq_data_ = data_maker.return_data();

        LOG(INFO) << "SAQ data prepared with variance";

        // Print quantization plan
        LOG(INFO) << "Quantization plan:";
        size_t dims_sum = 0;
        std::string log = fmt::format("{}bits: ", config_.avg_bits);
        dims_sum = 0;
        for (auto &[dim_len, bits] : saq_data_->quant_plan) {
            log += fmt::format("| {} -> {} ({}d {}b) ", dims_sum, dims_sum + dim_len, dim_len, bits);
            dims_sum += dim_len;
        }
        LOG(INFO) << log;

        // Create quantizer and quantize one cluster for testing
        SAQuantizer quantizer(saq_data_.get());

        // Create and quantize test cluster
        cluster_ = std::make_unique<SaqCluData>(num_data_, saq_data_->quant_plan, config_.use_compact_layout);
        const FloatVec &centroid = centroids_.row(0);
        quantizer.quantize_cluster(data_, centroid, cluster_ids_, *cluster_);
    }

    // Generic helper function using template and lambda for distance computation
    template <typename EstimatorFactory, typename DistComputer>
    std::tuple<float, float, float> testAllQueries(const std::string &test_name,
                                                   EstimatorFactory create_estimator,
                                                   DistComputer compute_distances) {
        utils::AvgMaxRecorder overall_rela_err_vars;
        utils::AvgMaxRecorder overall_rela_err_fast;
        utils::AvgMaxRecorder overall_rela_err_acc;

        // Test estimator for each query
        for (size_t query_idx = 0; query_idx < num_query_; ++query_idx) {
            const Eigen::RowVectorXf &query = query_.row(query_idx);

            // Create and prepare estimator using factory function
            auto estimator = create_estimator(query, query_idx);

            utils::AvgMaxRecorder rela_err_vars;
            utils::AvgMaxRecorder rela_err_fast;
            utils::AvgMaxRecorder rela_err_acc;

            for (size_t vec_idx = 0; vec_idx < cluster_->num_vec_; ++vec_idx) {
                auto [fast_dist, estimated_dist, vars_dist] = compute_distances(estimator, vec_idx);

                // Compute ground truth distance for comparison
                PID data_id = cluster_->ids()[vec_idx];
                float true_dist = (query - data_.row(data_id)).squaredNorm();

                auto relative_error_vars = (true_dist - vars_dist) / (true_dist + 1e-18);
                auto relative_error_fast = (true_dist - fast_dist) / (true_dist + 1e-18);
                auto relative_error_acc = std::abs(estimated_dist - true_dist) / (true_dist + 1e-18);
                rela_err_vars.insert(relative_error_vars);
                rela_err_fast.insert(relative_error_fast);
                rela_err_acc.insert(relative_error_acc);

                EXPECT_GE(fast_dist, 0.0f) << "Distance should be non-negative";
                EXPECT_GE(estimated_dist, 0.0f) << "Distance should be non-negative";
                EXPECT_GE(vars_dist, 0.0f) << "Distance should be non-negative";

                LOG_IF(INFO, debug_) << fmt::format(
                    "\tVec {}: estimated_dist={:.4f}, vars_dist={:.4f}, true_dist={:.4f} | relative_error: vars={:.4e}, fast={:.4e}, acc={:.4e}",
                    vec_idx, estimated_dist, vars_dist, true_dist, relative_error_vars, relative_error_fast, relative_error_acc);
            }
            overall_rela_err_vars.merge(rela_err_vars);
            overall_rela_err_fast.merge(rela_err_fast);
            overall_rela_err_acc.merge(rela_err_acc);

            EXPECT_LT(rela_err_acc.avg(), 0.2f) << test_name + " estimated distance should be close to true distance";

            LOG(INFO) << fmt::format("{} Query {} | relative_error: vars={:.4f}, fast={:.4f}, acc={:.4e}",
                                     test_name, query_idx, rela_err_vars.avg(), rela_err_fast.avg(), rela_err_acc.avg());
        }

        // Output overall statistics after all queries
        LOG(INFO) << fmt::format("{} Overall | rela_err_vars avg: {:.4f}, rela_err_fast avg: {:.4f}, rela_err_acc avg: {:.4e}",
                                 test_name, overall_rela_err_vars.avg(), overall_rela_err_fast.avg(), overall_rela_err_acc.avg());
        return {overall_rela_err_vars.avg(), overall_rela_err_fast.avg(), overall_rela_err_acc.avg()};
    }
};

TEST_F(CluEstimatorTest, SaqClu) {
    gen();
    config_.single.random_rotation = false;
    quantize();

    // Factory function to create and prepare SAQ estimator
    auto create_saq_estimator = [&](const Eigen::RowVectorXf &query, size_t) {
        auto estimator = std::make_unique<SaqCluEstimator<DistType::L2Sqr>>(*saq_data_.get(), searcher_config_, query);
        estimator->prepare(cluster_.get());
        return estimator;
    };

    // Lambda for SAQ FastScan distance computation
    auto saq_compute_distances = [&](auto &est, size_t vec_idx) -> std::tuple<float, float, float> {
        static thread_local float fast_distances[KFastScanSize];
        static thread_local float vars_distances[KFastScanSize];

        if (vec_idx % KFastScanSize == 0) {
            __m512 t[2];
            est->compFastDist(vec_idx / KFastScanSize, t);
            _mm512_storeu_ps(fast_distances, t[0]);
            _mm512_storeu_ps(fast_distances + 16, t[1]);

            // Get varsEstDist
            __m512 v[2];
            est->varsEstDist(vec_idx / KFastScanSize, v);
            _mm512_storeu_ps(vars_distances, v[0]);
            _mm512_storeu_ps(vars_distances + 16, v[1]);
        }

        float fast_dist = fast_distances[vec_idx % KFastScanSize];
        float estimated_dist = est->compAccurateDist(vec_idx);
        float vars_dist = vars_distances[vec_idx % KFastScanSize];
        return std::make_tuple(fast_dist, estimated_dist, vars_dist);
    };

    testAllQueries("SAQ", create_saq_estimator, saq_compute_distances);
}

TEST_F(CluEstimatorTest, CaqClus) {
    gen();
    config_.single.random_rotation = false;
    quantize();

    // Since enable_segmentation=false, there should be only one segment
    EXPECT_EQ(saq_data_->base_datas.size(), 1) << "Should have only one base data segment";
    const auto &base_data = saq_data_->base_datas[0];

    // Factory function to create and prepare CAQ estimator
    auto create_estimator = [&](const Eigen::RowVectorXf &query, size_t) {
        auto estimator = std::make_unique<CaqCluEstimator<DistType::L2Sqr>>(base_data, searcher_config_, query);

        auto vars2 = (saq_data_->data_variance.head(base_data.num_dim_pad).array() * query.array().square()).sum();
        estimator->setPruneBound(std::sqrt(vars2));

        const auto &caq_cluster = cluster_->get_segment(0);
        estimator->prepare(&caq_cluster);
        return estimator;
    };

    // Lambda for CAQ FastScan distance computation
    auto compute_dist = [&](auto &est, size_t vec_idx) -> std::tuple<float, float, float> {
        static thread_local float fast_distances[KFastScanSize];
        static thread_local float vars_distances[KFastScanSize];

        if (vec_idx % KFastScanSize == 0) {
            __m512 t[2];
            est->compFastDist(vec_idx / KFastScanSize, t);
            _mm512_storeu_ps(fast_distances, t[0]);
            _mm512_storeu_ps(fast_distances + 16, t[1]);

            // Get varsEstDist
            __m512 v[2];
            est->varsEstDist(vec_idx / KFastScanSize, v);
            _mm512_storeu_ps(vars_distances, v[0]);
            _mm512_storeu_ps(vars_distances + 16, v[1]);
        }

        float fast_dist = fast_distances[vec_idx % KFastScanSize];
        float estimated_dist = est->compAccurateDist(vec_idx);
        float vars_dist = vars_distances[vec_idx % KFastScanSize];
        return std::make_tuple(fast_dist, estimated_dist, vars_dist);
    };

    testAllQueries("CAQ", create_estimator, compute_dist);
}

TEST_F(CluEstimatorTest, CaqClusSingle) {
    gen();
    config_.single.random_rotation = false;
    config_.single.use_fastscan = false;
    quantize();

    // Since enable_segmentation=false, there should be only one segment
    EXPECT_EQ(saq_data_->base_datas.size(), 1) << "Should have only one base data segment";
    const auto &base_data = saq_data_->base_datas[0];

    // Factory function to create and prepare CaqClusSingle estimator
    auto create_estimator = [&](const Eigen::RowVectorXf &query, size_t) {
        auto estimator = std::make_unique<CaqCluEstimatorSingle<DistType::L2Sqr>>(base_data, searcher_config_, query);

        // Set variance-based pruning bound
        auto vars2 = (saq_data_->data_variance.head(base_data.num_dim_pad).array() * query.array().square()).sum();
        estimator->setPruneBound(std::sqrt(vars2));

        const auto &caq_cluster = cluster_->get_segment(0);
        estimator->prepare(&caq_cluster);
        return estimator;
    };

    // Lambda for CaqClusSingle distance computation
    auto compute_dist = [&](auto &est, size_t vec_idx) -> std::tuple<float, float, float> {
        float fast_dist = est->compFastDist(vec_idx);
        float estimated_dist = est->compAccurateDist(vec_idx);
        float vars_dist = est->varsEstDist(vec_idx);
        return std::make_tuple(fast_dist, estimated_dist, vars_dist);
    };

    testAllQueries("CaqClusSingle", create_estimator, compute_dist);
}

TEST_F(CluEstimatorTest, CaqClus_GIST) {
    load("gist");
    quantize();

    // Since enable_segmentation=false, there should be only one segment
    EXPECT_EQ(saq_data_->base_datas.size(), 1) << "Should have only one base data segment";
    const auto &base_data = saq_data_->base_datas[0];

    // Factory function to create and prepare CAQ estimator
    auto create_estimator = [&](const Eigen::RowVectorXf &query, size_t) {
        auto estimator = std::make_unique<CaqCluEstimator<DistType::L2Sqr>>(base_data, searcher_config_, query);

        auto vars2 = (saq_data_->data_variance.head(base_data.num_dim_pad).array() * query.array().square()).sum();
        estimator->setPruneBound(std::sqrt(vars2));

        const auto &caq_cluster = cluster_->get_segment(0);
        estimator->prepare(&caq_cluster);
        return estimator;
    };

    // Lambda for CAQ FastScan distance computation
    auto compute_dist = [&](auto &est, size_t vec_idx) -> std::tuple<float, float, float> {
        static thread_local float fast_distances[KFastScanSize];
        static thread_local float vars_distances[KFastScanSize];

        if (vec_idx % KFastScanSize == 0) {
            __m512 t[2];
            est->compFastDist(vec_idx / KFastScanSize, t);
            _mm512_storeu_ps(fast_distances, t[0]);
            _mm512_storeu_ps(fast_distances + 16, t[1]);

            // Get varsEstDist
            __m512 v[2];
            est->varsEstDist(vec_idx / KFastScanSize, v);
            _mm512_storeu_ps(vars_distances, v[0]);
            _mm512_storeu_ps(vars_distances + 16, v[1]);
        }

        float fast_dist = fast_distances[vec_idx % KFastScanSize];
        float estimated_dist = est->compAccurateDist(vec_idx);
        float vars_dist = vars_distances[vec_idx % KFastScanSize];
        return std::make_tuple(fast_dist, estimated_dist, vars_dist);
    };

    auto [vars_err, fast_err, acc_err] = testAllQueries("CAQ_GIST", create_estimator, compute_dist);
    EXPECT_LT(std::abs(vars_err), 1);
    EXPECT_LT(std::abs(fast_err), 0.20);
    EXPECT_LT(std::abs(acc_err), 6.4e-03);
}

TEST_F(CluEstimatorTest, CaqClusSingle_GIST) {
    load("gist");
    config_.single.use_fastscan = false;
    quantize();

    // Since enable_segmentation=false, there should be only one segment
    EXPECT_EQ(saq_data_->base_datas.size(), 1) << "Should have only one base data segment";
    const auto &base_data = saq_data_->base_datas[0];

    // Factory function to create and prepare CaqClusSingle estimator
    auto create_estimator = [&](const Eigen::RowVectorXf &query, size_t) {
        auto estimator = std::make_unique<CaqCluEstimatorSingle<DistType::L2Sqr>>(base_data, searcher_config_, query);

        // Set variance-based pruning bound
        auto vars2 = (saq_data_->data_variance.head(base_data.num_dim_pad).array() * query.array().square()).sum();
        estimator->setPruneBound(std::sqrt(vars2));

        const auto &caq_cluster = cluster_->get_segment(0);
        estimator->prepare(&caq_cluster);
        return estimator;
    };

    // Lambda for CaqClusSingle distance computation
    auto compute_dist = [&](auto &est, size_t vec_idx) -> std::tuple<float, float, float> {
        float fast_dist = est->compFastDist(vec_idx);
        float estimated_dist = est->compAccurateDist(vec_idx);
        float vars_dist = est->varsEstDist(vec_idx);
        return std::make_tuple(fast_dist, estimated_dist, vars_dist);
    };

    auto [vars_err, fast_err, acc_err] = testAllQueries("CaqClusSingle_GIST", create_estimator, compute_dist);
    EXPECT_LT(std::abs(vars_err), 1);
    EXPECT_LT(std::abs(fast_err), 0.20);
    EXPECT_LT(std::abs(acc_err), 6.4e-03);
}

TEST_F(CluEstimatorTest, SaqClu_GIST) {
    load("gist", true);
    config_.enable_segmentation = true;
    config_.single.random_rotation = true;
    quantize();

    // Factory function to create and prepare SAQ estimator
    auto create_estimator = [&](const Eigen::RowVectorXf &query, size_t) {
        auto estimator = std::make_unique<SaqCluEstimator<DistType::L2Sqr>>(*saq_data_.get(), searcher_config_, query);
        estimator->prepare(cluster_.get());
        return estimator;
    };

    // Lambda for CAQ FastScan distance computation
    auto compute_dist = [&](auto &est, size_t vec_idx) -> std::tuple<float, float, float> {
        static thread_local float fast_distances[KFastScanSize];
        static thread_local float vars_distances[KFastScanSize];

        if (vec_idx % KFastScanSize == 0) {
            __m512 t[2];
            est->compFastDist(vec_idx / KFastScanSize, t);
            _mm512_storeu_ps(fast_distances, t[0]);
            _mm512_storeu_ps(fast_distances + 16, t[1]);

            // Get varsEstDist
            __m512 v[2];
            est->varsEstDist(vec_idx / KFastScanSize, v);
            _mm512_storeu_ps(vars_distances, v[0]);
            _mm512_storeu_ps(vars_distances + 16, v[1]);
        }

        float fast_dist = fast_distances[vec_idx % KFastScanSize];
        float estimated_dist = est->compAccurateDist(vec_idx);
        float vars_dist = vars_distances[vec_idx % KFastScanSize];
        return std::make_tuple(fast_dist, estimated_dist, vars_dist);
    };

    auto [vars_err, fast_err, acc_err] = testAllQueries("SaqClu_GIST", create_estimator, compute_dist);
    EXPECT_LT(std::abs(vars_err), 0.2);
    EXPECT_LT(std::abs(fast_err), 0.38);
    EXPECT_LT(std::abs(acc_err), 3.75e-04);
}

TEST_F(CluEstimatorTest, SaqCluSingle_GIST) {
    load("gist", true);
    config_.single.use_fastscan = false;
    config_.enable_segmentation = true;
    config_.single.random_rotation = true;
    quantize();

    // Factory function to create and prepare SAQ estimator
    auto create_saq_estimator = [&](const Eigen::RowVectorXf &query, size_t) {
        auto estimator = std::make_unique<SaqCluEstimatorSingle<DistType::L2Sqr>>(*saq_data_.get(), searcher_config_, query);
        estimator->prepare(cluster_.get());
        return estimator;
    };

    // Lambda for SAQ FastScan distance computation
    auto saq_compute_distances = [&](auto &est, size_t vec_idx) -> std::tuple<float, float, float> {
        float fast_dist = est->compFastDist(vec_idx);
        float estimated_dist = est->compAccurateDist(vec_idx);
        float vars_dist = est->varsEstDist(vec_idx);
        return std::make_tuple(fast_dist, estimated_dist, vars_dist);
    };

    auto [vars_err, fast_err, acc_err] = testAllQueries("SaqCluSingle_GIST", create_saq_estimator, saq_compute_distances);
    EXPECT_LT(std::abs(vars_err), 0.2);
    EXPECT_LT(std::abs(fast_err), 0.38);
    EXPECT_LT(std::abs(acc_err), 3.75e-04);
}
