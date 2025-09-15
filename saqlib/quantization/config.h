#pragma once

#include <cstdlib>
#include <cstring>

#include <fmt/core.h>

#include "defines.hpp"

namespace saqlib {

struct QuantSingleConfig {
    BaseQuantType quant_type = BaseQuantType::CAQ; // quantization type. (CAQ, RBQ)
    bool random_rotation = true;                   // use random rotation or not
    bool use_fastscan = true;                      // use fast scan or not.
    int caq_adj_rd_lmt = 6;                        // adjustment round limit. 0 means no limit
    float caq_adj_eps = 1e-8;                      // adjustment with EPS
    int caq_ori_qB = 0;                            // [Experiment Only] Original quantization bits. 0 means disable.
};

struct QuantizeConfig {
    float avg_bits = 0;              // average bits for quantization.
    int seg_eqseg = 0;               // segment equally into this number of segments. 0 means disable.
    bool enable_segmentation = true; // enable SAQ or not.
    bool use_compact_layout = false; // use compact memory layout for segmentation.

    QuantSingleConfig single; // CAQ configuration

    std::string toString() const {
        std::string args_str;
        args_str += fmt::format("_b{}", avg_bits);

        if (!single.random_rotation) {
            args_str += "_norotate";
        }

        switch (single.quant_type) {
        case BaseQuantType::CAQ:
            args_str += "_caq";
            break;
        case BaseQuantType::RBQ:
            args_str += "_rbq";
            break;
        case BaseQuantType::LVQ:
            args_str += "_lvq";
            break;
        }
        if (single.quant_type == BaseQuantType::CAQ) {
            if (single.caq_adj_rd_lmt) {
                args_str += fmt::format("_adj");
                if (single.caq_adj_rd_lmt != QuantSingleConfig().caq_adj_rd_lmt) {
                    args_str += fmt::format("_rdlmt{}", single.caq_adj_rd_lmt);
                }
                if (single.caq_adj_eps != QuantSingleConfig().caq_adj_eps) {
                    args_str += fmt::format("_eps{:.1e}", single.caq_adj_eps);
                }
            }
            if (single.caq_ori_qB) {
                args_str += fmt::format("_oqb{}", single.caq_ori_qB);
            }
        }

        if (enable_segmentation) {
            args_str += "_seg";
            if (seg_eqseg > 0) {
                args_str += fmt::format("_eqseg{}", seg_eqseg);
            }
        }
        if (use_compact_layout) {
            args_str += "_compactlayout";
        }
        return args_str;
    }
};

struct SearcherConfig {
    float searcher_vars_bound_m = 4;      // searcher variance prune bound m. Larger value means more accurate but slower.
    DistType dist_type = DistType::L2Sqr; // distance type. L2Sqr or IP
};
} // namespace saqlib
