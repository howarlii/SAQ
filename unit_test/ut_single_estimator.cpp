#include <memory>
#include <vector>

#include <fmt/core.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "defines.hpp"
#include "quantization/config.h"
#include "quantization/saq_data.hpp"
#include "quantization/saq_estimator.hpp"
#include "quantization/saq_quantizer.hpp"
#include "quantization/single_data.hpp"
#include "test_base.hpp"
#include "utils/memory.hpp"
#include "utils/pool.hpp"

using namespace saqlib;

class SingleEstimatorTest : public ::testing::Test, public TestBase {
  protected:
    bool debug_ = true;
    size_t num_data_ = 64;
    size_t num_query_ = 4;
    size_t num_dim_ = 512;
    size_t num_centroids_ = 1;
    float avg_bits_ = 4.0f;

    std::unique_ptr<SaqData> saq_data_;
    std::unique_ptr<SAQuantizerSingle> quantizer_single_;
    QuantizeConfig config_;
    SearcherConfig searcher_config_;

    // Memory management for quantized data
    std::vector<memory::UniqueArray<uint8_t>> quantized_memories_;

    void SetUp() override {
        LOG(INFO) << "CTEST_FULL_OUTPUT";
#ifdef NDEBUG
        debug_ = false;
        num_data_ = 1024;
        num_query_ = 16;
        num_dim_ = 768;
#endif

        // Setup quantization config
        config_.avg_bits = avg_bits_;
        config_.single.use_fastscan = false; // Single quantizer doesn't support fastscan

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

    void prepare_saq_data() {
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

        // Create single quantizer
        quantizer_single_ = std::make_unique<SAQuantizerSingle>(saq_data_.get());
    }

    // Test single quantizer and estimator with individual vectors
    void quant() {
        quantized_memories_.clear();
        quantized_memories_.reserve(num_data_);

        SaqSingleDataWrapper single_wrapper(saq_data_->quant_plan);

        for (size_t data_idx = 0; data_idx < num_data_; ++data_idx) {
            const FloatVec &data_vec = data_.row(data_idx);

            // Calculate required memory size
            size_t memory_size = SaqSingleDataWrapper::calculate_memory_size(saq_data_->quant_plan);

            // Allocate memory
            auto memory = memory::make_unique_array<uint8_t>(memory_size, 64);

            // Set memory base before quantization
            single_wrapper.set_memory_base(memory.get());

            // Quantize the data vector
            quantizer_single_->quantize(data_vec, &single_wrapper);

            // Store the allocated resources
            quantized_memories_.emplace_back(std::move(memory));

            LOG_IF(INFO, debug_ && data_idx < 4) << fmt::format("Quantized vector {} with memory size {}", data_idx, memory_size);
        }

        LOG(INFO) << fmt::format("Quantization completed for {} vectors", num_data_);
    }

    std::tuple<float, float, float> testEst(const std::string &test_name) {
        utils::AvgMaxRecorder overall_rela_err_vars;
        utils::AvgMaxRecorder overall_rela_err_fast;
        utils::AvgMaxRecorder overall_rela_err_acc;

        // Test each query against quantized data vectors
        for (size_t query_idx = 0; query_idx < num_query_; ++query_idx) {
            const Eigen::RowVectorXf &query = query_.row(query_idx);

            // Create SaqSingleEstimator
            auto estimator = std::make_unique<SaqSingleEstimator<DistType::L2Sqr>>(*saq_data_.get(), searcher_config_, query);
            SaqSingleDataWrapper single_wrapper(saq_data_->quant_plan);

            utils::AvgMaxRecorder rela_err_vars;
            utils::AvgMaxRecorder rela_err_fast;
            utils::AvgMaxRecorder rela_err_acc;

            // Test against quantized data vectors
            for (size_t data_idx = 0; data_idx < quantized_memories_.size(); ++data_idx) {
                single_wrapper.set_memory_base(quantized_memories_[data_idx].get());

                // Compute distances using estimator
                float vars_dist = estimator->varsEstDist(single_wrapper);
                float fast_dist = estimator->compFastDist(single_wrapper);
                float estimated_dist = estimator->compAccurateDist(single_wrapper);

                // Compute ground truth distance
                const FloatVec &data_vec = data_.row(data_idx);
                float true_dist = (query - data_vec).squaredNorm();

                // Calculate relative errors
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
                    "\tData {}: vars={:.4f}, fast={:.4f}, acc={:.4f}, true={:.4f} | relative_error: vars={:.4e}, fast={:.4e}, acc={:.4e}",
                    data_idx, vars_dist, fast_dist, estimated_dist, true_dist, relative_error_vars, relative_error_fast, relative_error_acc);
            }

            overall_rela_err_vars.merge(rela_err_vars);
            overall_rela_err_fast.merge(rela_err_fast);
            overall_rela_err_acc.merge(rela_err_acc);

            EXPECT_LT(rela_err_acc.avg(), 0.2f) << test_name + " estimated distance should be close to true distance";

            LOG(INFO) << fmt::format("{} Query {} | relative_error: vars={:.4f}, fast={:.4f}, acc={:.4e}",
                                     test_name, query_idx, rela_err_vars.avg(), rela_err_fast.avg(), rela_err_acc.avg());
        }

        // Output overall statistics after all queries
        LOG(INFO) << fmt::format("{} Overall | avg Err: vars_err={:.4f}, fast_err={:.4f}, acc_err={:.4e}",
                                 test_name, overall_rela_err_vars.avg(), overall_rela_err_fast.avg(), overall_rela_err_acc.avg());
        return {overall_rela_err_vars.avg(), overall_rela_err_fast.avg(), overall_rela_err_acc.avg()};
    }
};

TEST_F(SingleEstimatorTest, CaqSingle) {
    gen();
    config_.single.random_rotation = false;
    config_.enable_segmentation = false;
    prepare_saq_data();
    quant();
    testEst("CaqSingle");
}

TEST_F(SingleEstimatorTest, SaqSingle_GIST) {
    load("gist", true);
    // config_.single.random_rotation = false; // BUG if enable rotation
    prepare_saq_data();
    quant();

    auto [vars_err, fast_err, acc_err] = testEst("SaqSingle_GIST");

    // Verify error bounds for GIST dataset
    EXPECT_LT(std::abs(vars_err), 100.0); // disable var check temporary
    EXPECT_LT(std::abs(fast_err), 1.05 * 0.1773);
    EXPECT_LT(std::abs(acc_err), 1.05 * 4.4427e-04);
}

TEST_F(SingleEstimatorTest, CaqSingle_GIST) {
    load("gist");
    // config_.single.random_rotation = false; //
    config_.enable_segmentation = false;
    prepare_saq_data();
    quant();

    auto [vars_err, fast_err, acc_err] = testEst("CaqSingle_GIST");

    // Verify error bounds for GIST dataset
    EXPECT_LT(std::abs(vars_err), 100.0); // disable var check temporary
    EXPECT_LT(std::abs(fast_err), 1.05 * 0.1653);
    EXPECT_LT(std::abs(acc_err), 1.05 * 5.4499e-03);
    // TODO: the error of SingleEstimator and ClusterEstimator seems like different, why?
}
