#include <atomic>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>

#include <fmt/core.h>
#include <glog/logging.h>
#include <vector>

#include "defines.hpp"
#include "index/ivf.hpp"
#include "quantization/config.h"
#include "test_base.hpp"
#include "utils/BS_thread_pool.hpp"
#include "utils/pool.hpp"

constexpr size_t TOPK = 100;
constexpr size_t NPROBE = 200;
constexpr double TOLERANCE = 3e-3;

class RecallTest : public TestBase, public ::testing::Test {
    const int kNumThread = 64;

  protected:
    SearcherConfig searcher_cfg_;
    std::unique_ptr<IVF> ivf_;

    void SetUp() override {
        LOG(INFO) << "CTEST_FULL_OUTPUT";
        // Set up searcher config
        searcher_cfg_.searcher_vars_bound_m = 4.0;
        searcher_cfg_.dist_type = DistType::L2Sqr;
    }

    void createIndex(const QuantizeConfig &config, size_t K = 4096) {
        size_t N = data_.rows();
        size_t DIM = data_.cols();

        // Create IVF index using unique_ptr
        ivf_ = std::make_unique<IVF>(N, DIM, K, config);
        ivf_->set_variance(data_vars_);
        ivf_->construct(data_, centroids_, cids_.data());
    }

    // Helper function to calculate recall
    std::pair<float, float> calculateRecall(IVF &ivf, const size_t nprobe) {
        size_t NQ = query_.rows();
        size_t total_count = TOPK * NQ;
        std::atomic<size_t> total_correct{0};
        std::vector<QueryRuntimeMetrics> metrics(NQ);

        BS::thread_pool pool(kNumThread);
        pool.detach_loop(0, NQ, [&](size_t i) {
            PID results[TOPK];

            ivf.search<DistType::L2Sqr>(query_.row(i), TOPK, nprobe, searcher_cfg_, results, &metrics[i]);

            // Count correct results
            for (size_t j = 0; j < TOPK; j++) {
                for (size_t k = 0; k < TOPK; k++) {
                    if (gt_(i, k) == results[j]) {
                        total_correct++;
                        break;
                    }
                }
            }
        });
        pool.wait();

        utils::AvgMaxRecorder prune_rate;
        for (auto &m : metrics) {
            prune_rate.insert(1.0 * m.fast_bitsum / m.acc_bitsum);
        }

        return {static_cast<float>(total_correct) / total_count, prune_rate.avg()};
    }

    void testDatasetQuantTypeRecall(const std::string &dataset, QuantizeConfig base_config,
                                    const std::map<int, float> &expected_recalls) {
        loadTestData(dataset);

        for (auto &[bits, expected_recall] : expected_recalls) {
            auto config = base_config;
            config.avg_bits = static_cast<float>(bits);

            // Create index using pre-loaded data
            createIndex(config);

            // Perform search and calculate recall
            // auto [actual_recall, prune_rate] = calculateRecall(*ivf_, NPROBE);
            size_t NQ = query_.rows();
            size_t total_count = TOPK * NQ;
            std::atomic<size_t> total_correct{0};
            std::vector<QueryRuntimeMetrics> metrics(NQ);

            BS::thread_pool pool(kNumThread);
            pool.detach_loop(0, NQ, [&](size_t i) {
                PID results[TOPK];

                ivf_->search<DistType::L2Sqr>(query_.row(i), TOPK, NPROBE, searcher_cfg_, results, &metrics[i]);

                // Count correct results
                for (size_t j = 0; j < TOPK; j++) {
                    for (size_t k = 0; k < TOPK; k++) {
                        if (gt_(i, k) == results[j]) {
                            total_correct++;
                            break;
                        }
                    }
                }
            });
            pool.wait();
            float actual_recall = static_cast<float>(total_correct) / total_count;

            utils::AvgMaxRecorder bits_visit; // pre distance computation
            for (auto &m : metrics) {
                bits_visit.insert(1.0 * (m.fast_bitsum + m.acc_bitsum) / m.total_comp_cnt);
            }
            float prune_rate = 1.0 - bits_visit.avg() / (ivf_->num_dim() * config.avg_bits);
            std::string quant_name = config.enable_segmentation ? "SAQ" : "CAQ";
            EXPECT_NEAR(actual_recall, expected_recall, TOLERANCE)
                << quant_name << " " << bits << "-bit recall for " << dataset << " dataset";

            // Log test results
            LOG(INFO) << fmt::format("{} {}-bit ({})\t| Error: expected={:.4f}, actual={:.4f}, diff={:.4f}\t| Prune: bits_visit={:.1f}, prune_rate:{:.3f}",
                                     quant_name, bits, dataset, expected_recall, actual_recall,
                                     std::abs(actual_recall - expected_recall), bits_visit.avg(), prune_rate);
        }
    }
};

TEST_F(RecallTest, SAQ_OPENAI1536_AllBits) {
    QuantizeConfig config;
    std::map<int, float> expected_recalls = {{1, 0.88058}, {4, 0.94257}, {8, 0.94578}};
    testDatasetQuantTypeRecall("openai1536", config, expected_recalls);
}

TEST_F(RecallTest, CAQ_OPENAI1536_AllBits) {
    QuantizeConfig config;
    config.enable_segmentation = false;
    std::map<int, float> expected_recalls = {{1, 0.83499}, {4, 0.93404}, {8, 0.94296}};
    testDatasetQuantTypeRecall("openai1536", config, expected_recalls);
}

TEST_F(RecallTest, SAQ_GIST_AllBits) {
    QuantizeConfig config;
    std::map<int, float> expected_recalls = {{1, 0.88347}, {4, 0.95118}, {8, 0.95421}};
    testDatasetQuantTypeRecall("gist", config, expected_recalls);
}

TEST_F(RecallTest, CAQ_GIST_AllBits) {
    QuantizeConfig config;
    config.enable_segmentation = false;
    std::map<int, float> expected_recalls = {{1, 0.74049}, {4, 0.93212}, {8, 0.95048}};
    testDatasetQuantTypeRecall("gist", config, expected_recalls);
}
