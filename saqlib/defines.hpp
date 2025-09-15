#pragma once
#define EIGEN_DONT_PARALLELIZE

#include <stdint.h>

#include "third/Eigen/Dense"
#include "third/Eigen/src/Core/Matrix.h"
#include "third/Eigen/src/Core/util/Constants.h"

namespace saqlib {

// constexpr bool enable_hiaccfs = true;

constexpr size_t KMaxQuantizeBits = 13; // xipnorm will be NaN if > 13
constexpr size_t KFastScanSize = 32;
constexpr size_t kDimPaddingSize = 64;

using PID = uint32_t;

template <typename T>
using RowVector = Eigen::RowVector<T, Eigen::Dynamic>;

using FloatRowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IntRowMat = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using UintRowMat = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Uint8RowMat = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DoubleRowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FloatVec = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using Uint8Vec = Eigen::Matrix<uint8_t, 1, Eigen::Dynamic>;
using Uint16Vec = Eigen::Matrix<uint16_t, 1, Eigen::Dynamic>;

enum class BaseQuantType {
    CAQ,
    RBQ,
    LVQ
};

enum class DistType {
    Any,   // [internal] only for compile optimization
    L2Sqr, // L2 squared distance
    IP,    // inner product
};

struct Candidate {
    PID id;
    float distance;

    Candidate() = default;
    Candidate(PID id, float distance)
        : id(id), distance(distance) {}

    bool operator<(const Candidate &other) const { return distance < other.distance; }

    bool operator>(const Candidate &other) const { return !(*this < other); }
};

} // namespace saqlib