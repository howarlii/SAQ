#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <glog/logging.h>

#include "defines.hpp"
#include "index/ivf.hpp"
#include "quantization/config.h"
#include "test_base.hpp"
#include "utils/pool.hpp"

constexpr size_t NPROBE = 200;
constexpr double RelativeErrorTolerance = 1e-3;
const int kNumThread = 64;

template <DistType kDistType = DistType::L2Sqr>
class IvfErrorTest : public TestBase, public ::testing::Test {
  protected:
    SearcherConfig searcher_cfg_;
    std::unique_ptr<IVF> ivf_;

    void SetUp() override {
        LOG(INFO) << "CTEST_FULL_OUTPUT";
        // Set up searcher config
        searcher_cfg_.searcher_vars_bound_m = 4.0;
        searcher_cfg_.dist_type = kDistType;
    }

    void createIndex(const QuantizeConfig &config, size_t K = 4096) {
        size_t N = data_.rows();
        size_t DIM = data_.cols();

        // Create IVF index using unique_ptr
        ivf_ = std::make_unique<IVF>(N, DIM, K, config);
        ivf_->set_variance(data_vars_);
        ivf_->construct(data_, centroids_, cids_.data());
    }

    // Helper function to calculate relative error for a single query
    std::tuple<float, float, float> calculateSingleQueryRelativeError(const Eigen::RowVectorXf &ori_query, const size_t nprobe) {
        std::vector<std::pair<PID, float>> dist_list;
        std::vector<float> fast_dist_list;
        std::vector<float> vars_dist_list;
        ivf_->estimate<kDistType>(ori_query, nprobe, searcher_cfg_, dist_list, &fast_dist_list, &vars_dist_list);
        CHECK_EQ(dist_list.size(), fast_dist_list.size());

        utils::AvgMaxRecorder query_error_recorder;
        utils::AvgMaxRecorder query_fst_error_recorder;
        utils::AvgMaxRecorder query_vars_error_recorder;
        for (size_t i = 0; i < dist_list.size(); i++) {
            auto [data_id, est_dist] = dist_list[i];
            // Compute real distance using Eigen API
            auto data_vec = data_.row(data_id);
            float true_dist;

            if (searcher_cfg_.dist_type == DistType::L2Sqr) {
                // L2 squared distance using Eigen
                true_dist = (ori_query - data_vec).squaredNorm();
            } else if (searcher_cfg_.dist_type == DistType::IP) {
                // Inner product using Eigen
                true_dist = ori_query.dot(data_vec);
            } else {
                throw std::runtime_error("Unsupported distance type for error test");
            }

            // Calculate relative error and insert into recorder
            if (true_dist != 0) {
                float relative_err = std::abs((est_dist - true_dist) / true_dist);
                query_error_recorder.insert(relative_err);
                float fst_relative_err = (true_dist - fast_dist_list[i]) / std::abs(true_dist);
                query_fst_error_recorder.insert(fst_relative_err);
                float vars_relative_err = (true_dist - vars_dist_list[i]) / std::abs(true_dist);
                query_vars_error_recorder.insert(vars_relative_err);
            }
        }

        // For now, return 0 for vars error as we can't easily access it
        return {query_error_recorder.avg(), query_fst_error_recorder.avg(), query_vars_error_recorder.avg()};
    }

    void testDatasetQuantTypeError(const std::string &dataset, QuantizeConfig base_config,
                                   const std::map<int, std::pair<float, float>> &expected_avg_errors) {
        loadTestData(dataset);

        for (auto &[bits, avg_error] : expected_avg_errors) {
            auto [expected_acc_error, expected_fst_error] = avg_error;
            auto config = base_config;
            config.avg_bits = static_cast<float>(bits);

            // Create index using pre-loaded data
            createIndex(config);

            size_t NQ = query_.rows();
            utils::AvgMaxRecorder total_error;
            utils::AvgMaxRecorder total_fst_error;
            utils::AvgMaxRecorder total_vars_error;

            BS::thread_pool pool(kNumThread);
            std::vector<std::future<std::tuple<float, float, float>>> futures;
            for (size_t i = 0; i < NQ; i++) {
                const Eigen::RowVectorXf ori_query = query_.row(i);

                futures.emplace_back(pool.submit_task([this, ori_query]() {
                    return calculateSingleQueryRelativeError(ori_query, NPROBE);
                }));
            }

            for (auto &fut : futures) {
                auto [query_error, query_fst_error, query_vars_error] = fut.get();

                // Merge query error recorder into total recorder
                total_error.insert(query_error);
                total_fst_error.insert(query_fst_error);
                total_vars_error.insert(query_vars_error);
            }

            std::string quant_name = config.enable_segmentation ? "SAQ" : "CAQ";
            // Use relative error for comparison
            float acc_err_err = (total_error.avg() - expected_acc_error) / expected_acc_error;
            float fst_err_err = (total_fst_error.avg() - expected_fst_error) / expected_fst_error;

            // Log test results with relative error including vars
            LOG(INFO) << fmt::format("{} {}-bit ({})\t| Error Acc: {:.5e} (diff={:.2f}%)\t Fast: {:.4e} (diff={:.2f}%)\t Vars: {:.4e}",
                                     quant_name, bits, dataset, total_error.avg(), acc_err_err * 100.0f,
                                     total_fst_error.avg(), fst_err_err * 100.0f, total_vars_error.avg());

            EXPECT_LE(std::abs(acc_err_err), RelativeErrorTolerance)
                << quant_name << " " << bits << "-bit relative error for " << dataset << " dataset";
            EXPECT_LE(std::abs(fst_err_err), RelativeErrorTolerance)
                << quant_name << " " << bits << "-bit fast relative error for " << dataset << " dataset";
        }
    }
};

using IvfErrorTestL2Sqr = IvfErrorTest<DistType::L2Sqr>;
using IvfErrorTestIP = IvfErrorTest<DistType::IP>;

TEST_F(IvfErrorTestL2Sqr, SAQ_OPENAI1536_AllBits) {
    QuantizeConfig config;
    std::map<int, std::pair<float, float>> expected_avg_errors =
        {{1, {7.44806e-03, 1.2469e-01}}, {4, {9.49550e-04, 8.9169e-02}}, {8, {6.74842e-05, 8.9225e-02}}};
    testDatasetQuantTypeError("openai1536", config, expected_avg_errors);
}

TEST_F(IvfErrorTestL2Sqr, CAQ_OPENAI1536_AllBits) {
    QuantizeConfig config;
    config.enable_segmentation = false;
    std::map<int, std::pair<float, float>> expected_avg_errors =
        {{1, {1.50666e-02, 3.6366e-02}}, {4, {2.26619e-03, 3.6340e-02}}, {8, {1.59592e-04, 3.6330e-02}}};
    testDatasetQuantTypeError("openai1536", config, expected_avg_errors);
}

TEST_F(IvfErrorTestL2Sqr, SAQ_GIST_AllBits) {
    QuantizeConfig config;
    std::map<int, std::pair<float, float>> expected_avg_errors =
        {{1, {5.88538e-03, 1.5112e-01}}, {4, {5.77164e-04, 1.4452e-01}}, {8, {4.00324e-05, 1.1412e-01}}};
    testDatasetQuantTypeError("gist", config, expected_avg_errors);
}

TEST_F(IvfErrorTestL2Sqr, CAQ_GIST_AllBits) {
    QuantizeConfig config;
    config.enable_segmentation = false;
    std::map<int, std::pair<float, float>> expected_avg_errors =
        {{1, {1.85407e-02, 4.4969e-02}}, {4, {2.73361e-03, 4.4966e-02}}, {8, {1.87982e-04, 4.4943e-02}}};
    testDatasetQuantTypeError("gist", config, expected_avg_errors);
}

TEST_F(IvfErrorTestIP, SAQ_GIST_AllBits_IP) {
    QuantizeConfig config;
    std::map<int, std::pair<float, float>> expected_avg_errors =
        {{1, {1.89057e-02, -4.2505e-01}}, {4, {1.99794e-03, -4.7931e-01}}, {8, {1.43901e-04, -3.7490e-01}}};
    testDatasetQuantTypeError("gist", config, expected_avg_errors);
}

TEST_F(IvfErrorTestIP, CAQ_GIST_AllBits_IP) {
    QuantizeConfig config;
    config.enable_segmentation = false;
    std::map<int, std::pair<float, float>> expected_avg_errors =
        {{1, {6.46918e-02, -1.4815e-01}}, {4, {9.38642e-03, -1.4744e-01}}, {8, {5.99436e-04, -1.4850e-01}}};
    testDatasetQuantTypeError("gist", config, expected_avg_errors);
}
