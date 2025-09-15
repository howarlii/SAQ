#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "glog/logging.h"
#include <fmt/core.h>

#include "defines.hpp"
#include "quantization/cluster_data.hpp"
#include "quantization/quantizer.hpp"
#include "quantization/saq_data.hpp"
#include "utils/tools.hpp"

namespace saqlib {
class SAQuantizer {
  protected:
    const size_t num_dim_;                                      // original dimension
    const size_t num_dim_padded_;                               // padded dimension
    std::vector<std::unique_ptr<QuantizerCluster>> data_quans_; // Data quantizer
    const SaqData *data_;

  public:
    explicit SAQuantizer(const SaqData *data)
        : num_dim_(data->num_dim), num_dim_padded_(utils::rd_up_to_multiple_of(num_dim_, kDimPaddingSize)), data_(data) {
        for (auto &bi : data_->base_datas) {
            data_quans_.emplace_back(std::make_unique<QuantizerCluster>(&bi));
        }
    }

    void quantize_cluster(const FloatRowMat &data, const FloatVec &centroid, const std::vector<PID> &IDs,
                          SaqCluData &saq_clus) {
        CHECK_EQ(saq_clus.num_segments_, data_quans_.size());
        std::copy(IDs.begin(), IDs.end(), saq_clus.ids());

        const size_t num_points = saq_clus.num_vec_;
        for (size_t ci = 0, offset = 0; ci < saq_clus.num_segments_; ++ci) {
            auto &clus = saq_clus.get_segment(ci);
            const size_t copy_size = std::min(clus.num_dim_padded_, num_dim_ - offset);

            FloatRowMat vecs(num_points, clus.num_dim_padded_);
            vecs.setZero();
            // Copy data for each row individually
            for (size_t r = 0; r < num_points; ++r) {
                auto id = clus.ids()[r];
                vecs.row(r).head(copy_size) = data.row(id).segment(offset, copy_size);
            }

            FloatVec cen(clus.num_dim_padded_);
            cen.setZero();
            // Copy centroid data
            cen.head(copy_size) = centroid.segment(offset, copy_size);

            data_quans_[ci]->quantize(vecs, cen, clus);
            offset += clus.num_dim_padded_;
        }
    }
};

class SAQuantizerSingle {
  protected:
    const size_t num_dim_;                                     // original dimension
    const size_t num_dim_padded_;                              // padded dimension
    std::vector<std::unique_ptr<QuantizerSingle>> data_quans_; // Data quantizer
    const SaqData *data_;

  public:
    SAQuantizerSingle(const SaqData *data)
        : num_dim_(data->num_dim), num_dim_padded_(utils::rd_up_to_multiple_of(num_dim_, kDimPaddingSize)), data_(data) {
        for (auto &bi : data_->base_datas) {
            data_quans_.emplace_back(std::make_unique<QuantizerSingle>(&bi));
        }
    }

    void quantize(const FloatVec &or_vec, SaqSingleDataWrapper *caq_data) const {
        CHECK_EQ(caq_data->num_segments_, data_quans_.size());

        for (size_t ci = 0, offset = 0; ci < caq_data->num_segments_; ++ci) {
            auto &clus = caq_data->get_segment(ci);

            if (auto rem_sz = num_dim_ - offset; clus.num_dim_padded_ <= rem_sz) {
                data_quans_[ci]->quantize(or_vec.segment(offset, clus.num_dim_padded_), &clus);
            } else {
                FloatVec t = FloatVec::Zero(clus.num_dim_padded_);
                t.head(rem_sz) = or_vec.segment(offset, rem_sz);
                data_quans_[ci]->quantize(t, &clus);
            }

            offset += clus.num_dim_padded_;
        }
    }
};
} // namespace saqlib
