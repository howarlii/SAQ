#include <atomic>
#include <fstream>
#include <iostream>
#include <vector>

#include <fmt/core.h>
#include <gflags/gflags.h>

#include "define_options.h"

#include "defines.hpp"
#include "index/ivf.hpp"
#include "utils/BS_thread_pool.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"
#include "utils/pool.hpp"

using namespace saqlib;

DEFINE_int32(fix_nprobe, 0, "Fixed nprobe value for QPS test. 0 means [5, 4000]");
DEFINE_int32(fix_thread, 24, "Fixed thread value for QPS test. 0 means [1, 48]");

constexpr size_t TOPK = 100;
constexpr size_t ROUND = 10;

struct Stats {
    int num_threads{0};
    float recall{0};
    float avg_tm_ms{0};
    float qps{0};
    float dist_ratio{0};
    float bw_mbps{0};
    float compute_kopps{0}; // computation pre seconds
};

float relative_error(float x, float base) {
    return std::abs(x - base) / base;
}

class QPSTester {
  private:
    FloatRowMat data_;
    FloatRowMat query_;
    UintRowMat gt_;

    IVF ivf_;

  private:
    Stats run_search(const size_t nprobe, SearcherConfig &searcher_cfg, size_t num_threads) {
        size_t NQ = query_.rows();
        size_t total_count = TOPK * NQ;
        std::atomic<size_t> total_correct = 0;

        std::vector<std::vector<PID>> results(NQ, std::vector<PID>(TOPK));
        std::vector<QueryRuntimeMetrics> runtime_metrics(NQ);
        std::vector<float> tm_ms(NQ);
        std::vector<float> dist_ratios(NQ);

        // std::vector<std::thread> threads;
        // utils::StopW tot_stopw;
        // for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        //     threads.emplace_back([&, thread_id]() {
        //         size_t start_index = thread_id * (NQ / num_threads);
        //         size_t end_index = (thread_id == num_threads - 1) ? NQ : (thread_id + 1) * (NQ / num_threads);
        //         utils::StopW stopw;
        //         for (size_t i = start_index; i < end_index; ++i) {
        //             stopw.reset();
        //             ivf_.search(query_.row(i), TOPK, nprobe, searcher_cfg, results[i].data(), &runtime_metrics[i]);
        //             tm_ms[i] = stopw.getElapsedTimeMicro() / 1000.0;
        //         }
        //     });
        // }
        // for (auto &thread : threads) {
        //     thread.join();
        // }

        BS::thread_pool pool(num_threads);
        utils::StopW tot_stopw;
        pool.detach_loop(0, NQ, [&](size_t i) {
            utils::StopW stopw;
            ivf_.search(query_.row(i), TOPK, nprobe, searcher_cfg, results[i].data(), &runtime_metrics[i]);
            tm_ms[i] = stopw.getElapsedTimeMicro() / 1000.0;
        });
        pool.wait();
        auto tot_tm_ms = tot_stopw.getElapsedTimeMili();

        pool.detach_loop(0, NQ, [&](size_t i) {
            dist_ratios[i] = utils::get_ratio(i, query_, data_, gt_, results[i].data(), TOPK, utils::L2Sqr) / TOPK;
            size_t correct_count = 0;
            for (size_t j = 0; j < TOPK; ++j) {
                for (size_t k = 0; k < TOPK; ++k) {
                    if (gt_(i, k) == results[i][j]) {
                        correct_count++;
                        break;
                    }
                }
            }
            total_correct += correct_count;
        });
        pool.wait();

        utils::AvgMaxRecorder time_recorder_ms;
        utils::AvgMaxRecorder dist_ratio;
        size_t bandwith_sum_mb{0};
        size_t comput_sum_kop{0};
        // utils::AvgMaxRecorder bandwith_mbps;
        // utils::AvgMaxRecorder comput_kops;
        Stats curr_stats;
        for (size_t i = 0; i < NQ; ++i) {
            time_recorder_ms.insert(tm_ms[i]);
            dist_ratio.insert(dist_ratios[i]);
            auto &m = runtime_metrics[i];
            // bandwith_mbps.insert((m.fast_bitsum + m.acc_bitsum) / 8.0 / 1024 / (tm_ms[i] / 1000));
            // comput_kops.insert(m.total_comp_cnt / 1000.0 / (tm_ms[i] / 1000));
            bandwith_sum_mb += (m.fast_bitsum + m.acc_bitsum) / 8.0 / 1024 / 1024;
            comput_sum_kop += m.total_comp_cnt / 1000.0;
        }

        float recall = static_cast<float>(total_correct) / total_count;
        curr_stats.num_threads = num_threads;
        curr_stats.recall = recall;
        curr_stats.avg_tm_ms = time_recorder_ms.avg();
        curr_stats.qps = NQ * 1e3 / time_recorder_ms.sum() * num_threads;
        curr_stats.dist_ratio = dist_ratio.avg();
        curr_stats.bw_mbps = bandwith_sum_mb / tot_tm_ms * 1000;
        curr_stats.compute_kopps = comput_sum_kop / tot_tm_ms * 1000;

        std::cout << "num_threads: " << num_threads << "\trecall: " << recall << "\tdist_rate: " << curr_stats.dist_ratio
                  << " \tq_avg_tm: " << time_recorder_ms.avg() << "ms\tqps: " << curr_stats.qps << "\t";

        std::cout << "bw_mbps: " << curr_stats.bw_mbps << "MB/s\t";
        std::cout << "compute_kopps: " << curr_stats.compute_kopps << "KOP/s\t";

        std::cout << std::endl;

        return curr_stats;
    }

    Stats run_search_multi(const size_t nprobe, SearcherConfig &searcher_cfg,
                           size_t num_threads, size_t round) {
        auto sample = run_search(nprobe, searcher_cfg, num_threads);
        utils::AvgMaxRecorder qps;
        utils::AvgMaxRecorder avg_tm_ms;
        qps.insert(sample.qps);
        avg_tm_ms.insert(sample.avg_tm_ms);
        for (size_t i = 1; i < round; i++) {
            auto stats = run_search(nprobe, searcher_cfg, num_threads);
            qps.insert(stats.qps);
            avg_tm_ms.insert(stats.avg_tm_ms);
            auto e = relative_error(stats.recall, sample.recall);
            LOG_IF(WARNING, e > 1e-6) << "!!!!! Unstable! recall error : " << stats.recall << " " << sample.recall << " " << e;
            e = relative_error(stats.qps, qps.avg());
            LOG_IF(WARNING, e > 1e-1) << "!!!!! Unstable!  qps error: " << stats.qps << " " << qps.avg() << " " << e;
            e = relative_error(stats.avg_tm_ms, avg_tm_ms.avg());
            LOG_IF(WARNING, e > 1e-1) << "!!!!! Unstable!  avg_tm_ms error: " << stats.avg_tm_ms << " " << avg_tm_ms.avg() << " " << e;
        }
        sample.qps = qps.avg();
        sample.avg_tm_ms = avg_tm_ms.avg();
        return sample;
    }

  public:
    void loadData(const DataFilePaths &paths) {
        utils::load_something<float, FloatRowMat>(paths.data_file.c_str(), data_);
        utils::load_something<float, FloatRowMat>(paths.query_file.c_str(), query_);
        utils::load_something<PID, UintRowMat>(paths.gt_file.c_str(), gt_);

        size_t N = data_.rows();
        size_t DIM = query_.cols();
        size_t NQ = query_.rows();

        std::cout << "data loaded\n";
        std::cout << "\tN: " << N << '\n'
                  << "\tDIM: " << DIM << '\n';
        std::cout << "query loaded\n";
        std::cout << "\tNQ: " << NQ << '\n';

        std::cout << "load index from " << paths.quant_file << '\n';

        ivf_.load(paths.quant_file.c_str());
    }

    void runQPSTests(const std::string &result_file, SearcherConfig &searcher_cfg) {
        std::vector<size_t> thread_nums_list;
        std::vector<size_t> nprob_list;
        if (FLAGS_fix_thread == 0) {
            thread_nums_list.push_back(1);
            thread_nums_list.push_back(2);
            thread_nums_list.push_back(4);
            thread_nums_list.push_back(6);
            thread_nums_list.push_back(8);
            thread_nums_list.push_back(10);
            thread_nums_list.push_back(12);
            thread_nums_list.push_back(16);
            thread_nums_list.push_back(20);
            thread_nums_list.push_back(24);
            thread_nums_list.push_back(28);
            thread_nums_list.push_back(32);
            thread_nums_list.push_back(40);
            thread_nums_list.push_back(48);
        } else {
            thread_nums_list.push_back(FLAGS_fix_thread);
        }

        if (FLAGS_fix_nprobe == 0) {
            nprob_list.push_back(5);
            for (size_t i = 10; i < 150; i += 10) {
                nprob_list.push_back(i);
            }
            for (size_t i = 150; i < 400; i += 50) {
                nprob_list.push_back(i);
            }
            for (size_t i = 400; i <= 1000; i += 150) {
                nprob_list.push_back(i);
            }
            for (size_t i = 2000; i <= 4000; i += 1000) {
                nprob_list.push_back(i);
            }
        } else {
            nprob_list.push_back(FLAGS_fix_nprobe);
        }

        std::ofstream csv_data(result_file + ".csv", std::ios::out);
        std::string final_result = "nprobe,num_threads,QPS,avg_tm_ms,recall,ratio,bw_mbps,compute_kopps\n";
        csv_data << final_result;

        for (auto num_threads : thread_nums_list) {
            for (auto nprob : nprob_list) {
                auto stats = run_search_multi(nprob, searcher_cfg, num_threads, ROUND);
                auto ts = fmt::format("{},{},{},{},{},{},{},{}\n", nprob, stats.num_threads, stats.qps, stats.avg_tm_ms, stats.recall,
                                      stats.dist_ratio, stats.bw_mbps, stats.compute_kopps);
                csv_data << ts;
                final_result += ts;
            }
        }
        csv_data.close();

        LOG(INFO) << "result log to file: " << result_file << ".csv";
        std::cout << final_result << std::endl;
    }
};

int main(int argc, char *argv[]) {
    [[maybe_unused]] int prio = nice(-10); // Set process priority to high
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string dataset_str = FLAGS_dataset;

    // Parse quantization config and generate args string
    QuantizeConfig cfg;
    auto args_str = parseArgs(&cfg);
    LOG(INFO) << args_str << "\n";
    DataFilePaths paths;

    // Setup searcher config
    SearcherConfig searcher_cfg;
    searcher_cfg.searcher_vars_bound_m = FLAGS_searcher_vars_bound_m;
    if (FLAGS_searcher_dist_type == 0) {
        searcher_cfg.dist_type = DistType::L2Sqr;
    } else if (FLAGS_searcher_dist_type == 1) {
        searcher_cfg.dist_type = DistType::IP;
    } else {
        LOG(ERROR) << "Invalid searcher distance type: " << FLAGS_searcher_dist_type;
        return -1;
    }

    // Setup paths and parameters - use fixed nprobe instead of variable num_threads
    std::string result_file = fmt::format("{}/qps_{}_{}_th{}_np{}", paths.result_path,
                                          dataset_str, args_str.c_str(), FLAGS_fix_thread, FLAGS_fix_nprobe);

    result_file += fmt::format("_sm{}", FLAGS_searcher_vars_bound_m);
    if (FLAGS_searcher_dist_type == 1) {
        result_file += "_ip";
    }

    // Run QPS test with fixed nprobe
    QPSTester tester;
    tester.loadData(paths);
    tester.runQPSTests(result_file, searcher_cfg);

    return 0;
}
