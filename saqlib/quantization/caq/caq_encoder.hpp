#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

#include "glog/logging.h"

#include "defines.hpp"
#include "quantization/cluster_data.hpp"
#include "quantization/config.h"

namespace saqlib {
class CAQEncoder {
    static constexpr float kConstEpsilon = 1.9;

    const size_t num_dim_pad_;     // number of dimensions to paddled
    const size_t num_bits_;        // Number of bits for quantization
    const QuantSingleConfig &cfg_; // Quantization configuration
    uint16_t code_max_;
    const uint16_t code_mi_ = 0;

    void code_adjustment(const float *curr_vec, CaqCode &caq) {
        const auto &v_mi = caq.v_mi;
        const auto &delta = caq.delta;
        auto &code = caq.code;
        double &ip_resi_oa = caq.ip_o_oa;
        double &oa_l2sqr = caq.oa_l2sqr;

        const double re_eps = cfg_.caq_adj_eps * oa_l2sqr;
        [[maybe_unused]] int tot_adj_cnt = 0;
        for (int adj_cnt = 1, round = 1; adj_cnt; round++) {
            CHECK_GE(ip_resi_oa, 0);
            adj_cnt = 0;
            for (size_t j = 0; j < num_dim_pad_; j++) {
                int curr_adj_cnt = 0;
                const auto o = curr_vec[j];
                double oa = (code[j] + 0.5) * delta + v_mi;
                uint16_t c = code[j];
                const auto oa_l2sqr_tmp = oa_l2sqr - oa * oa;
                const auto ip_delta = delta * o;

                // TODO: fix this optimzie bug
                // double target_oa = std::max<double>(v_mi, o * oa_l2sqr_tmp / (ip_resi_oa - oa * o));
                // auto target_code = std::min<int>(code_max_, std::round((target_oa - v_mi) / delta));
                // if (std::abs(target_code - c) > 1 && target_code >= code_mi_) {
                //     auto new_oa = (target_code + 0.5) * delta + v_mi;
                //     ip_resi_oa -= delta * (c * o - target_code * o);
                //     oa_l2sqr -= oa * oa - new_oa * new_oa;
                //     c = target_code;
                // }

                // try ++
                while (c < code_max_) {
                    auto new_q = oa + delta;
                    auto new_length = oa_l2sqr_tmp + new_q * new_q;
                    auto new_ip = ip_resi_oa + ip_delta;
                    if ((ip_resi_oa * ip_resi_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c++;
                    ip_resi_oa = new_ip;
                    oa = new_q;
                    oa_l2sqr = new_length;

                    curr_adj_cnt++;
                }
                // try --
                while (c > code_mi_) {
                    auto new_q = oa - delta;
                    auto new_length = oa_l2sqr_tmp + new_q * new_q;
                    auto new_ip = ip_resi_oa - ip_delta;
                    if ((ip_resi_oa * ip_resi_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c--;
                    ip_resi_oa = new_ip;
                    oa = new_q;
                    oa_l2sqr = new_length;

                    curr_adj_cnt++;
                }
                if (code[j] != c) {
                    code[j] = c;
                    adj_cnt += curr_adj_cnt;
                }
            }
            tot_adj_cnt += adj_cnt;
            if (cfg_.caq_adj_rd_lmt && round >= cfg_.caq_adj_rd_lmt) {
                break;
            }

            // correction
            {
                double check_oa_l2sqr = 0;
                double check_ip = 0;
                for (size_t j = 0; j < num_dim_pad_; ++j) {
                    const auto o = curr_vec[j];
                    auto qc = code[j];
                    auto q = (qc + 0.5) * delta + v_mi;
                    check_ip += q * o;
                    check_oa_l2sqr += q * q;
                }
                // CHECK_LT(std::abs(check_oa_l2sqr - oa_l2sqr), 2e-5);
                // CHECK_LT(std::abs(check_ip - ip_resi_oa), 2e-5);
                oa_l2sqr = check_oa_l2sqr;
                ip_resi_oa = check_ip;
            }
        }
    }

    void downUpSample(const float *vec, CaqCode &caq) {
        auto sampled_rshirt = cfg_.caq_ori_qB - num_bits_;
        caq.delta *= 1 << sampled_rshirt;
        caq.ip_o_oa = 0;
        caq.oa_l2sqr = 0;
        for (size_t j = 0; j < num_dim_pad_; ++j) {
            caq.code[j] >>= sampled_rshirt;
            const auto o = vec[j];
            auto qc = caq.code[j];
            auto q = (qc + 0.5) * caq.delta + caq.v_mi;
            caq.ip_o_oa += q * o;
            caq.oa_l2sqr += q * q;
        }
    }

  public:
    CAQEncoder(size_t num_dim_pad, size_t num_bits, const QuantSingleConfig &cfg)
        : num_dim_pad_(num_dim_pad), num_bits_(num_bits), cfg_(cfg),
          code_max_((1 << num_bits_) - 1) {
        CHECK_LE(num_bits_, KMaxQuantizeBits) << "CAQ only support up to " << KMaxQuantizeBits << " bits";

        if (cfg_.caq_ori_qB) {
            CHECK_GT(cfg_.caq_ori_qB, num_bits_);
            code_max_ = (1 << cfg_.caq_ori_qB) - 1;
            CHECK_NE(cfg_.caq_ori_qB, num_bits_);
        }
    }

    void encode(const FloatVec &o, CaqCode &caq) {
        if (num_bits_ == 0) {
            caq = CaqCode();
            return;
        }
        auto &code = caq.code;
        auto &v_mi = caq.v_mi;
        auto &v_mx = caq.v_mx;
        auto &delta = caq.delta;
        auto &ip_o_oa = caq.ip_o_oa;
        auto &oa_l2sqr = caq.oa_l2sqr;

        v_mx = std::max(o.maxCoeff(), -o.minCoeff());
        v_mi = -v_mx;

        delta = (v_mx - v_mi) / (code_max_ + 1);
        double vec_sum = o.sum();

        Eigen::VectorXf code_f;
        if (delta) {
            code_f = ((o.array() - v_mi) / delta).floor().array().min(code_max_);
        } else {
            code_f = Eigen::VectorXf::Zero(num_dim_pad_);
        }

        double ip_o_code = 0;
        uint64_t code_l2sqr = 0;
        code.resize(num_dim_pad_);
        for (size_t j = 0; j < num_dim_pad_; j++) {
            ip_o_code += code_f[j] * o[j];
            int c = static_cast<int>(code_f[j]);
            code[j] = c;
            code_l2sqr += c * c;
        }
        const uint32_t code_sum = code.sum();

        ip_o_oa = ip_o_code * delta + (v_mi + 0.5 * delta) * vec_sum;
        oa_l2sqr = delta * delta * code_l2sqr + (delta * delta + 2 * delta * v_mi) * code_sum;
        oa_l2sqr += (0.25 * delta * delta + delta * v_mi + v_mi * v_mi) * num_dim_pad_;

        if (cfg_.caq_adj_rd_lmt && oa_l2sqr) {
            DCHECK(delta);
            code_adjustment(o.data(), caq);
        }

        if (cfg_.caq_ori_qB) {
            downUpSample(o.data(), caq);
        }
    }

    void encode_and_fac(const FloatVec &curr_vec, CaqCode &caq) {
        encode(curr_vec, caq);
        caq.rescale_vmx_to1();
        caq.o_l2sqr = curr_vec.squaredNorm();
        caq.o_l2norm = std::sqrt(caq.o_l2sqr);
        caq.fac_rescale = caq.ip_o_oa ? caq.o_l2sqr / caq.ip_o_oa : 0;

        // err = o_l2sqr * epsilon * sqrt((1 - <o, o_a>^2) / <o, o_a>^2) / sqrt(dim - 1)
        caq.fac_error = caq.o_l2sqr * kConstEpsilon *
                        std::sqrt((((caq.o_l2sqr * caq.oa_l2sqr) / (caq.ip_o_oa * caq.ip_o_oa)) - 1) /
                                  (num_dim_pad_ - 1));
    }
};
} // namespace saqlib
