#pragma once

#include <fstream>
#include <memory>

#include "third/Eigen/Dense"
#include "third/Eigen/src/Core/Matrix.h"
#include <glog/logging.h>

#include "defines.hpp"

namespace saqlib::utils {
class Rotator {
  protected:
    size_t D;      // Padded dimension
    FloatRowMat P; // Rotation Matrix
  public:
    explicit Rotator(uint32_t dim)
        : D(dim) {
        this->P = FloatRowMat::Identity(D, D);
    }

    explicit Rotator() {}
    virtual ~Rotator() {}

    void set(FloatRowMat mat) {
        this->P = std::move(mat);
    }

    void orthogonalize() {
        FloatRowMat RAND(FloatRowMat::Random(D, D));
        Eigen::HouseholderQR<FloatRowMat> qr(RAND);
        FloatRowMat Q = qr.householderQ();
        this->P = Q.transpose(); // inverse of Q = Q.T
    }

    size_t size() const { return D; }

    /*
     * Load the rotation matrix from disk
     */
    virtual void load(std::ifstream &input) {
        float element;
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                input.read((char *)&element, sizeof(float));
                P(i, j) = element;
            }
        }
    }

    /*
     * Save the rotation matrix to disk
     */
    virtual void save(std::ofstream &output) const {
        float element;
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                element = P(i, j);
                output.write((char *)&element, sizeof(float));
            }
        }
    }

    /*
     * Rotate Matrix A and store the result in RAND_A
     */
    virtual void rotate(const FloatRowMat &A, FloatRowMat &RAND_A) const {
        // Note that Eigen store Matrix by columns by default
        RAND_A = A * P;
    }
    virtual void rotate(const Eigen::RowVectorXf &A, Eigen::RowVectorXf &RAND_A) const {
        // Note that Eigen store Matrix by columns by default
        RAND_A = A * P;
    }

    auto &get_P() const { return P; }
};

using RotatorPtr = std::unique_ptr<Rotator>;

class PCARotator {
  private:
  public:
    size_t D = 0;  // Padded dimension
    FloatVec mean; // mean vector
    FloatRowMat P; // Rotation Matrix
    void set(FloatRowMat mat, FloatVec mean) {
        this->D = mean.rows();
        this->mean = std::move(mean);
        this->P = std::move(mat);
    }

    // template <typename T>
    // void PCA(const T &A, T &RAND_A) const
    // {
    //     // Note that Eigen store Matrix by columns by default
    //     RAND_A = (A.rowwise() - mean) * P.transpose();
    // }

    void load(std::ifstream &input) {
        input.read((char *)&D, sizeof(size_t));
        mean.resize(D);
        P.resize(D, D);

        input.read((char *)mean.data(), D * sizeof(float));
        P = FloatRowMat::Zero(D, D);
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                input.read((char *)&P(i, j), sizeof(float));
            }
        }
    }

    void save(std::ofstream &output) const {
        output.write((char *)&D, sizeof(size_t));
        output.write((char *)mean.data(), D * sizeof(float));
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                output.write((char *)&P(i, j), sizeof(float));
            }
        }
    }
};
} // namespace saqlib::utils
