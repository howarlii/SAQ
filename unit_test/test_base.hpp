#pragma once

#include <random>
#include <string>

#include <fmt/core.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "defines.hpp"
#include "utils/IO.hpp"
#include "utils/tools.hpp"

using namespace saqlib;

class TestBase {
  protected:
    FloatRowMat data_;
    FloatRowMat query_;
    UintRowMat gt_;
    FloatRowMat centroids_;
    UintRowMat cids_;
    FloatRowMat data_vars_;

    // Helper function to load test data
    void loadTestData(const std::string &dataset, bool use_pca = true) {
        std::string path = std::string("./data/") + dataset;

        std::string data_file = fmt::format("{}/{}_base{}.fvecs", path, dataset, use_pca ? "_pca" : "");
        std::string query_file = fmt::format("{}/{}_query{}.fvecs", path, dataset, use_pca ? "_pca" : "");
        std::string centroids_file = fmt::format("{}/{}_centroid_4096{}.fvecs", path, dataset, use_pca ? "_pca" : "");
        std::string cids_file = fmt::format("{}/{}_cluster_id_4096.ivecs", path, dataset);
        std::string gt_file = fmt::format("{}/{}_groundtruth.ivecs", path, dataset);
        std::string data_vars_file = fmt::format("{}/{}_base{}.vars.fvecs", path, dataset, use_pca ? "_pca" : "");

        utils::load_something<float, FloatRowMat>(data_file.c_str(), data_);
        utils::load_something<float, FloatRowMat>(query_file.c_str(), query_);
        utils::load_something<float, FloatRowMat>(centroids_file.c_str(), centroids_);
        utils::load_something<PID, UintRowMat>(cids_file.c_str(), cids_);

        if (utils::file_exists(gt_file.c_str())) {
            utils::load_something<PID, UintRowMat>(gt_file.c_str(), gt_);
        } else {
            LOG(WARNING) << fmt::format("Ground truth file {} not found, skipping.", gt_file);
        }

        if (utils::file_exists(data_vars_file.c_str()))
            utils::load_something<float, FloatRowMat>(data_vars_file.c_str(), data_vars_);
        else {
            LOG(WARNING) << fmt::format("Data variance file {} not found, skipping.", data_vars_file);
        }
    }

    // Helper function to generate synthetic test data
    void generateTestData(size_t num_data, size_t num_query, size_t num_dim, size_t num_centroids = 1, int seed = 42) {
        auto data_dim = utils::rd_up_to_multiple_of(num_dim, kDimPaddingSize); // Ensure data dimension is a multiple of 64
        // Initialize random seed for reproducible results
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(-8, 8);

        // Generate synthetic data
        data_ = FloatRowMat::Zero(num_data, data_dim);
        for (size_t i = 0; i < num_data; ++i) {
            for (size_t j = 0; j < num_dim; ++j) {
                data_(i, j) = dist(gen);
            }
        }

        // Generate synthetic queries
        query_ = FloatRowMat::Zero(num_query, data_dim);
        for (size_t i = 0; i < num_query; ++i) {
            for (size_t j = 0; j < num_dim; ++j) {
                query_(i, j) = dist(gen);
            }
        }

        // Generate centroids
        centroids_ = FloatRowMat::Zero(num_centroids, data_dim);
        for (size_t i = 1; i < num_centroids; ++i) {
            centroids_.row(i) = RowVector<float>::Constant(num_dim, dist(gen));
        }

        // Generate simple cluster assignments - each data point gets its own cluster ID
        cids_.resize(num_data, 1);
        for (size_t i = 0; i < num_data; ++i) {
            cids_(i, 0) = i % num_centroids;
        }
    }
};
