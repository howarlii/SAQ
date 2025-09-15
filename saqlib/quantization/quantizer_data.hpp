#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>

#include "glog/logging.h"

#include "quantization/config.h"
#include "utils/rotator.hpp"

namespace saqlib {
struct BaseQuantizerData {
    size_t num_dim_pad; // Original dimension
    size_t num_bits;
    QuantSingleConfig cfg;     // Quantization configuration
    utils::RotatorPtr rotator; // Vector Rotator

    void init() {
        if (cfg.random_rotation) {
            rotator = std::make_unique<utils::Rotator>(num_dim_pad);
            rotator->orthogonalize();
        }
    }

    void save(std::ofstream &output) const {
        output.write(reinterpret_cast<const char *>(&num_dim_pad), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&num_bits), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&cfg), sizeof(QuantSingleConfig));
        char flags = rotator ? 1 : 0;
        output.write(&flags, sizeof(char));
        if (rotator) {
            rotator->save(output);
        }
    }

    void load(std::ifstream &input) {
        input.read(reinterpret_cast<char *>(&num_dim_pad), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&num_bits), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&cfg), sizeof(QuantSingleConfig));
        char flags;
        input.read(&flags, sizeof(char));
        if (flags) {
            rotator = std::make_unique<utils::Rotator>(num_dim_pad);
            rotator->load(input);
        }
    }
};
} // namespace saqlib
