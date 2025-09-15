#include <fstream>
#include <iostream>

#include <fmt/core.h>
#include <string>

#include "define_options.h"

#include "index/ivf.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"

using namespace saqlib;

class IndexCreator {
  private:
    FloatRowMat data_;
    FloatRowMat centroids_;
    UintRowMat cids_;
    FloatRowMat data_vars_;
    std::unique_ptr<IVF> ivf_;

  public:
    void buildIndex(const std::string &dataset, size_t K, const QuantizeConfig &cfg, const std::string &args_str) {
        // Create file paths and load all data needed for index creation
        DataFilePaths paths;
        utils::load_something<float, FloatRowMat>(paths.data_file.c_str(), data_);
        utils::load_something<float, FloatRowMat>(paths.centroids_file.c_str(), centroids_);
        utils::load_something<PID, UintRowMat>(paths.cids_file.c_str(), cids_);
        if (utils::file_exists(paths.data_vars_file.c_str())) {
            utils::load_something<float, FloatRowMat>(paths.data_vars_file.c_str(), data_vars_);
        }

        size_t num_threads = FLAGS_num_threads ? FLAGS_num_threads : 64;

        size_t num_vecs = data_.rows();
        size_t num_dim = data_.cols();

        std::cout << "data loaded\n";
        std::cout << "\tN: " << num_vecs << '\n';
        std::cout << "\tDIM: " << num_dim << '\n';

        utils::StopW stopw;

        // Create IVF index using unique_ptr
        ivf_ = std::make_unique<IVF>(num_vecs, num_dim, K, cfg);

        // Set variance if available
        if (data_vars_.rows() != 0) {
            ivf_->set_variance(std::move(data_vars_));
        }

        ivf_->construct(data_, centroids_, cids_.data(), num_threads, FLAGS_use_1_centroid);
        float tm_sec = stopw.getElapsedTimeMili() / 1000;
        LOG(INFO) << "ivf constructed ";
        ivf_->save(paths.quant_file.c_str());

        std::cout << "index saved at: " << paths.quant_file << '\n';
        std::cout << "Indexing time: " << tm_sec << "seconds\n";

        // === output to csv ===
        auto csv_path = fmt::format("{}/{}_{}.index.csv", paths.result_path, dataset, args_str);
        std::ofstream csv_data(csv_path, std::ios::out);
        csv_data << "index_time_s,ip_err_avg,ip_err_max";
        auto &statis_ip = ivf_->quant_metrics_.norm_ip_o_oa;
        csv_data << std::endl;

        csv_data << tm_sec << ",";
        csv_data << statis_ip.avg() << ",";
        csv_data << statis_ip.max();
        csv_data << std::endl;
        csv_data.close();
        std::cout << "Basic Statistics logged to: " << csv_path << "\n";
    }
};

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    QuantizeConfig cfg;
    auto args_str = parseArgs(&cfg);
    LOG(INFO) << args_str << "\n";

    IndexCreator creator;
    creator.buildIndex(FLAGS_dataset, FLAGS_K, cfg, args_str);

    return 0;
}