#pragma once

#include <cstring>
#include <fstream>
#include <functional>
#include <stdint.h>
#include <vector>

#include <glog/logging.h>
#include <fmt/core.h>

#include "defines.hpp"
// #include "third/hnswlib/hnswlib.h"

namespace saqlib
{
class Initializer
{
  protected:
    size_t num_dim_;
    size_t num_cluster_;

  public:
    explicit Initializer(size_t d, size_t k)
        : num_dim_(d), num_cluster_(k) {}
    virtual ~Initializer() {}
    // virtual float *centroid(PID, uint8_t) const = 0;
    virtual void set_centroids(FloatRowMat c) = 0;
    virtual void centroids_distances(const FloatVec &, size_t, DistType dist_type, std::vector<Candidate> &)
        const = 0;
    virtual void load(std::ifstream &, const char *) = 0;
    virtual void save(std::ofstream &, const char *) const = 0;
};

class FlatInitializer : public Initializer
{
    FloatRowMat centroids_;

  public:
    explicit FlatInitializer(size_t num_dim, size_t num_cluster)
        : Initializer(num_dim, num_cluster)
    {
    }

    ~FlatInitializer() {}

    void set_centroids(FloatRowMat c) override
    {
        centroids_ = std::move(c);
    }

    void centroids_distances(
        const FloatVec &query, size_t nprobe, DistType dist_type, std::vector<Candidate> &candidates) const override
    {
        std::vector<Candidate> centroid_dist(this->num_cluster_);
        if (dist_type == DistType::L2Sqr) {
            for (PID i = 0; i < num_cluster_; ++i) {
                centroid_dist[i].id = i;
                centroid_dist[i].distance = (centroids_.row(i) - query).squaredNorm();
            }
            std::partial_sort(centroid_dist.begin(), centroid_dist.begin() + nprobe,
                              centroid_dist.end());
        } else if (dist_type == DistType::IP) {
            for (PID i = 0; i < num_cluster_; ++i) {
                centroid_dist[i].id = i;
                centroid_dist[i].distance = query.dot(centroids_.row(i));
            }
            std::partial_sort(centroid_dist.begin(), centroid_dist.begin() + nprobe,
                              centroid_dist.end(), std::greater<>());
        } else {
            throw std::invalid_argument(
                fmt::format("Unsupported distance type: {}", static_cast<int>(dist_type)));
        }

        std::copy(centroid_dist.begin(), centroid_dist.begin() + nprobe, candidates.begin());
    }

    void save(std::ofstream &output, const char *) const override
    {
        CHECK_EQ(centroids_.size(), num_dim_ * num_cluster_) << "Centroids not set";
        output.write(
            reinterpret_cast<const char *>(centroids_.data()),
            static_cast<long>(sizeof(float) * num_dim_ * num_cluster_));
    }

    void load(std::ifstream &input, const char *) override
    {
        if (centroids_.size() < 1) {
            centroids_.resize(num_cluster_, num_dim_);
        }
        input.read(
            reinterpret_cast<char *>(centroids_.data()),
            static_cast<long>(sizeof(float) * num_dim_ * num_cluster_));
    }
};
} // namespace saqlib
