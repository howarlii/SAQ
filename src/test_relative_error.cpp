#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

#include <fmt/core.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "define_options.h"

#include "defines.hpp"
#include "index/ivf.hpp"
#include "utils/BS_thread_pool.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"
#include "utils/pool.hpp"

using namespace saqlib;

constexpr double HISTOGRAM_MAX = 0.05;
constexpr double HISTOGRAM_B1_MAX = 0.1;

DEFINE_bool(ALL, false, "use all data");

struct Stats {
    int nprobe{0};
    float avg_tm_ms{0};
    float search_tm_ms{0};
    utils::AvgMaxGroup total_rerr; // rerr means relative error
    // AvgMaxGroup topk_rerr;
    utils::StatisticsRecorder histogram_rela_err_ = utils::StatisticsRecorder(0, FLAGS_B == 1 ? HISTOGRAM_B1_MAX : HISTOGRAM_MAX);
    utils::StatisticsRecorder histogram_real_dist = utils::StatisticsRecorder(0, 0.1);
};

class RelativeErrorTester {
  private:
    FloatRowMat data_;
    FloatRowMat query_;
    std::unique_ptr<IVF> ivf_;
    std::vector<Stats> statistics_;
    SearcherConfig searcher_cfg_;

    void run_error(const size_t nprobe, FloatRowMat &data) {
        size_t NQ = query_.rows();
        utils::AvgMaxRecorder time_recorder_ms;

        auto &curr_stats = statistics_.emplace_back(Stats());

        const auto num_thread = FLAGS_num_threads ? FLAGS_num_threads : 24;
        std::mutex mtx;

        // Use thread pool for parallel processing
        BS::thread_pool pool(num_thread);
        std::vector<std::future<std::tuple<utils::AvgMaxRecorder, Stats>>> futures;

        for (size_t i = 0; i < NQ; i++) {
            futures.emplace_back(pool.submit_task([this, &data, nprobe, i]() {
                const Eigen::RowVectorXf ori_query = query_.row(i);
                QueryRuntimeMetrics runtime_metrics;
                Stats stats;
                std::vector<std::pair<PID, float>> dist_list;
                utils::StopW stopw;
                ivf_->estimate(ori_query, nprobe, searcher_cfg_, dist_list, nullptr, nullptr, &runtime_metrics);
                stats.search_tm_ms = stopw.getElapsedTimeMili();

                utils::AvgMaxRecorder query_error_recorder;
                for (auto [data_id, est_dist] : dist_list) {
                    // Compute real distance using Eigen API
                    auto data_vec = data.row(data_id);
                    float real_dist;

                    if (searcher_cfg_.dist_type == DistType::L2Sqr) {
                        // L2 squared distance using Eigen
                        real_dist = (ori_query - data_vec).squaredNorm();
                    } else if (searcher_cfg_.dist_type == DistType::IP) {
                        // Inner product using Eigen
                        real_dist = ori_query.dot(data_vec);
                    } else {
                        throw std::runtime_error("Unsupported distance type for error test");
                    }

                    // Calculate relative error and insert into recorder
                    if (real_dist > 0) {
                        float relative_err = std::abs(est_dist - real_dist) / real_dist;
                        // LOG(INFO) << "est_dist: " << est_dist << ", real_dist: " << real_dist << ", relative_err: " << relative_err;
                        query_error_recorder.insert(relative_err);
                        stats.histogram_rela_err_.insert(relative_err);
                    }
                    stats.histogram_real_dist.insert(real_dist);
                }
                return std::make_tuple(query_error_recorder, stats);
            }));
        }

        for (auto &fut : futures) {
            auto [query_error_recorder, stats] = fut.get();

            std::lock_guard<std::mutex> lck(mtx);
            time_recorder_ms.insert(stats.search_tm_ms);

            // LOG(INFO) << query_error_recorder.avg();
            // Merge error statistics into total recorder
            curr_stats.total_rerr.add(query_error_recorder);
            curr_stats.histogram_rela_err_.merge(stats.histogram_rela_err_);
            curr_stats.histogram_real_dist.merge(stats.histogram_real_dist);
        }

        curr_stats.nprobe = nprobe;
        curr_stats.avg_tm_ms = time_recorder_ms.avg();
        curr_stats.search_tm_ms = time_recorder_ms.avg(); // Same as avg_tm_ms in this context

        std::cout << "\tnprobe: " << nprobe << "\tq_search_avg_tm: " << curr_stats.search_tm_ms << "ms\n";

        std::cout << "reletive error - visited - total: "
                  << curr_stats.total_rerr.tot_avg() * 100 << "\t"
                  << curr_stats.total_rerr.tot_max() * 100 << std::endl;
        std::cout << "reletive error - visited - query-avg: "
                  << curr_stats.total_rerr.group_avg_avg() * 100 << "\t"
                  << curr_stats.total_rerr.group_mx_avg() * 100 << std::endl;

        // std::cout << "reletive error - topk - total: "
        //           << curr_stats.topk_rerr.tot_avg() * 100 << "\t"
        //           << curr_stats.topk_rerr.tot_max() * 100 << std::endl;
        // std::cout << "reletive error - topk - query-avg: "
        //           << curr_stats.topk_rerr.group_avg_avg() * 100 << "\t"
        //           << curr_stats.topk_rerr.group_mx_avg() * 100 << std::endl;

        std::cout << "Real Dist - total: "
                  << curr_stats.histogram_real_dist.avg() * 100 << "\t"
                  << curr_stats.histogram_real_dist.max() * 100 << std::endl;
    }

  public:
    void loadData(const DataFilePaths &paths, const SearcherConfig &cfg) {
        searcher_cfg_ = cfg;
        utils::load_something<float, FloatRowMat>(paths.data_file.c_str(), data_);
        utils::load_something<float, FloatRowMat>(paths.query_file.c_str(), query_);

        ivf_ = std::make_unique<IVF>();
        ivf_->load(paths.quant_file.data());

        // Output data information
        size_t N = ivf_->num_data();
        size_t DIM = query_.cols();
        size_t NQ = query_.rows();

        std::cout << "data loaded\n";
        std::cout << "\tN: " << N << '\n'
                  << "\tDIM: " << DIM << '\n';
        std::cout << "query loaded\n";
        std::cout << "\tNQ: " << NQ << '\n';
        std::cout << "load index from index file\n";
    }

    void runTest(const std::string &result_file) {
        auto K = FLAGS_K;
        if (K > 200 && !FLAGS_ALL) {
            run_error(200, data_);
            // run_error(100, data_);
            // run_error(50, data_);
        } else {
            run_error(32, data_);
            run_error(16, data_);
            run_error(8, data_);
            run_error(4, data_);
        }

        // === output to csv ===
        auto basic_csv_path = result_file + std::string(".csv");
        std::ofstream csv_data(basic_csv_path, std::ios::out);
        csv_data << "nprobe,avg_tm_ms,search_tm_ms";
        csv_data << ",err_tot_avg,err_tot_max,err_q_avg_avg,err_q_mx_avg";
        // csv_data << ",err_topk_q_tot_avg,err_topk_q_tot_mx,err_topk_q_avg_avg,err_topk_q_mx_avg";
        csv_data << std::endl;

        for (size_t i = 0; i < statistics_.size(); ++i) {
            auto &curr = statistics_[i];
            csv_data << curr.nprobe << ',';
            csv_data << curr.avg_tm_ms << ',';
            csv_data << curr.search_tm_ms << ',';

            csv_data << curr.total_rerr.tot_avg() << ',';
            csv_data << curr.total_rerr.tot_max() << ',';
            csv_data << curr.total_rerr.group_avg_avg() << ',';
            csv_data << curr.total_rerr.group_mx_avg();

            csv_data << std::endl;
        }
        csv_data.close();
        std::cout << "Statistics logged to: " << basic_csv_path << '\n';

        // { // output relative error histogram
        //     auto path = result_file + std::string(".csv.rela_err");
        //     csv_data = std::ofstream(path, std::ios::out);
        //     auto &hist = statistics_[0].histogram_rela_err_;
        //     for (double i = hist.def_min_val(); i < hist.def_max_val(); i += hist.gap()) {
        //         csv_data << i << ",";
        //     }
        //     csv_data.seekp(-1, std::ios_base::cur);
        //     csv_data << std::endl;
        //     for (auto &val : hist.histogram()) {
        //         csv_data << val << ',';
        //     }
        //     csv_data.seekp(-1, std::ios_base::cur);
        //     csv_data << std::endl;
        //     csv_data.close();
        //     std::cout << "Statistics logged to: " << path << "\n";
        // }

        // {
        //     auto varsip_csv_path = result_file + std::string(".csv.varsip");
        //     csv_data = std::ofstream(varsip_csv_path, std::ios::out);
        //     auto &statis_ip = statistics_[0].ip_histogram;
        //     for (double i = statis_ip.def_min_val(); i < statis_ip.def_max_val(); i += statis_ip.gap()) {
        //         csv_data << i << ",";
        //     }
        //     csv_data.seekp(-1, std::ios_base::cur);
        //     csv_data << std::endl;
        //     for (auto &val : statis_ip.histogram()) {
        //         csv_data << val << ',';
        //     }
        //     csv_data.seekp(-1, std::ios_base::cur);
        //     csv_data << std::endl;
        //     csv_data.close();
        //     std::cout << "Statistics logged to: " << varsip_csv_path << "\n";
        // }

        // {
        //     auto varsip_csv_path = result_file + std::string(".csv.dist");
        //     csv_data = std::ofstream(varsip_csv_path, std::ios::out);
        //     auto &statis_dist = statistics_[0].dist_histogram;
        //     for (double i = statis_dist.def_min_val(); i < statis_dist.def_max_val(); i += statis_dist.gap()) {
        //         csv_data << i << ",";
        //     }
        //     csv_data.seekp(-1, std::ios_base::cur);
        //     csv_data << std::endl;
        //     for (auto &val : statis_dist.histogram()) {
        //         csv_data << val << ',';
        //     }
        //     csv_data.seekp(-1, std::ios_base::cur);
        //     csv_data << std::endl;
        //     csv_data.close();
        //     std::cout << "Statistics logged to: " << varsip_csv_path << "\n";
        // }
    }
};

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    QuantizeConfig cfg;
    auto args_str = parseArgs(&cfg);
    LOG(INFO) << args_str << "\n";
    DataFilePaths paths;

    // results file path
    std::string result_file;
    result_file = fmt::format("{}/{}_{}_sm{}", paths.result_path,
                              FLAGS_dataset, args_str, FLAGS_searcher_vars_bound_m);

    SearcherConfig searcher_cfg;
    searcher_cfg.searcher_vars_bound_m = FLAGS_searcher_vars_bound_m;
    if (FLAGS_searcher_dist_type == 0) {
        searcher_cfg.dist_type = DistType::L2Sqr;
    } else if (FLAGS_searcher_dist_type == 1) {
        searcher_cfg.dist_type = DistType::IP;
        result_file += ".ip";
    } else {
        LOG(ERROR) << "Invalid searcher distance type: " << FLAGS_searcher_dist_type;
        return -1;
    }

    RelativeErrorTester tester;
    tester.loadData(paths, searcher_cfg);

    tester.runTest(result_file);

    return 0;
}
