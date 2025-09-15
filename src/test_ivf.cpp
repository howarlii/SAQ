#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <fmt/core.h>

#include "define_options.h"

#include "defines.hpp"
#include "index/initializer.hpp"
#include "quantization/config.h"
#include "utils/BS_thread_pool.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"
#include "utils/pool.hpp"
#include "utils/tools.hpp"

using namespace saqlib;

constexpr size_t TOPK = 100;

struct Stats {
    int nprobe{0};
    float recall{0};
    float avg_tm_ms{0};
    float qps{0};
    utils::AvgMaxGroup visited_rerr;
};

struct Cluster {
    size_t dim;
    std::vector<PID> ids;
    FloatRowMat vecs; // Each row is a vector
    Eigen::VectorXf centroid;
};

class FSearcher {
  public:
    std::vector<std::pair<PID, float>> records;
    size_t q_dim;

    void search1clu(size_t cidx, const Eigen::VectorXf &query, utils::ResultPool &KNNs, std::vector<Cluster> &clusters) {
        auto &clu = clusters[cidx];
        size_t num_vec = clu.ids.size();
        size_t dim = clu.dim;
        CHECK_EQ(clu.vecs.rows() * clu.vecs.cols(), num_vec * dim);

        const Eigen::VectorXf query_vec = query.head(q_dim);
        const Eigen::VectorXf query_remaining = query.segment(q_dim, dim - q_dim);

        for (size_t i = 0; i < num_vec; ++i) {
            auto id = clu.ids[i];

            // Get the i-th vector from the matrix using Eigen
            const Eigen::VectorXf vec = clu.vecs.row(i).head(q_dim);
            const Eigen::VectorXf centroid_remaining = clu.centroid.segment(q_dim, dim - q_dim);

            // Calculate distance using Eigen API
            float dist1 = (vec - query_vec).squaredNorm();
            float dist2 = (centroid_remaining - query_remaining).squaredNorm();
            auto est_dist = dist1 + dist2;

            KNNs.insert(id, est_dist);
            records.emplace_back(id, est_dist);
        }
    }
};
class IVFTester {
  private:
    FloatRowMat data_;
    FloatRowMat query_;
    UintRowMat gt_;
    FloatRowMat centroids_;
    UintRowMat cids_;

    // Move global variables into class
    size_t N_;
    size_t DIM_;
    size_t NQ_;
    size_t K_;
    float compress_ratio_;
    std::vector<Cluster> clusters_;

  public:
    Stats PCAsearch(size_t nprobe, size_t thread_num) {
        FlatInitializer initer(DIM_, K_);
        initer.set_centroids(centroids_);

        clusters_.clear();
        clusters_.resize(K_);
        for (size_t i = 0; i < K_; ++i) {
            clusters_[i].dim = utils::rd_up_to_multiple_of(DIM_, 64);
            clusters_[i].centroid.resize(clusters_[i].dim);
            clusters_[i].centroid.setZero();
            // Copy centroid data using Eigen operations
            clusters_[i].centroid.head(DIM_) = centroids_.row(i).transpose();
        }

        // First pass: count vectors per cluster
        std::vector<size_t> cluster_counts(K_, 0);
        for (size_t i = 0; i < N_; ++i) {
            auto cidx = cids_(i, 0);
            cluster_counts[cidx]++;
        }

        // Initialize cluster matrices
        for (size_t i = 0; i < K_; ++i) {
            clusters_[i].vecs.resize(cluster_counts[i], clusters_[i].dim);
            clusters_[i].vecs.setZero();
        }

        // Second pass: fill the matrices using Eigen operations
        std::vector<size_t> cluster_fill_idx(K_, 0);
        for (size_t i = 0; i < N_; ++i) {
            auto cidx = cids_(i, 0);
            auto &clus = clusters_[cidx];
            size_t row_idx = cluster_fill_idx[cidx]++;

            clus.ids.push_back(i);

            // Copy vector data to the matrix using Eigen
            clus.vecs.row(row_idx).head(DIM_) = data_.row(i);
        }

        BS::thread_pool pool(thread_num);

        // Add timing for QPS calculation
        utils::StopW timer;
        std::vector<std::future<std::pair<float, utils::AvgMaxRecorder>>> futures;
        CHECK_GE(1, compress_ratio_);

        for (size_t i = 0; i < NQ_; i++) {
            auto f = pool.submit_task([&, i]() {
                const Eigen::VectorXf query = query_.row(i).transpose();

                std::vector<Candidate> centroid_dist(nprobe);
                initer.centroids_distances(query_.row(i), nprobe, DistType::L2Sqr, centroid_dist);

                utils::ResultPool KNNs(TOPK);
                FSearcher searcher;
                searcher.q_dim = compress_ratio_ * DIM_;
                for (size_t ci = 0; ci < nprobe; ++ci) {
                    PID cid = centroid_dist[ci].id;
                    searcher.search1clu(cid, query, KNNs, clusters_);
                }

                PID results[TOPK];
                KNNs.copy_results(results);
                size_t total_corrects = 0;

                for (size_t j = 0; j < TOPK; ++j) {
                    for (size_t k = 0; k < TOPK; ++k) {
                        if (gt_(i, k) == results[j]) {
                            total_corrects++;
                            break;
                        }
                    }
                }
                float recall = static_cast<float>(total_corrects) / TOPK;

                utils::AvgMaxRecorder rela_err;
                for (auto [id, est_dist] : searcher.records) {
                    // Use Eigen for distance calculation
                    const Eigen::VectorXf data_vec = data_.row(id).transpose();
                    auto real_dist = (data_vec - query).squaredNorm();
                    auto real_error = std::abs(real_dist - est_dist);
                    float relative_err = 0;
                    relative_err = real_error / real_dist;
                    if (!real_dist)
                        relative_err = 0;
                    if (relative_err != relative_err) { // NaN check
                        LOG(INFO) << "Error ratio: " << relative_err << '\n';
                    }
                    if (relative_err > 1) {
                        LOG_FIRST_N(INFO, 100) << "relative_err too big!  id:" << id << ", err: " << relative_err << "\t real/est dist: " << real_dist << " / " << est_dist << '\n';
                    }
                    rela_err.insert(relative_err);
                }
                return std::make_pair(recall, rela_err);
            });
            futures.emplace_back(std::move(f));
        }

        Stats curr_stats;
        curr_stats.nprobe = nprobe;
        curr_stats.recall = 0;
        for (auto &f : futures) {
            auto [recall, rela_err] = f.get();
            curr_stats.recall += recall;
            curr_stats.visited_rerr.add(rela_err);
        }
        curr_stats.recall /= NQ_;

        // Calculate QPS and timing statistics
        float total_time_ms = timer.getElapsedTimeMili();
        curr_stats.avg_tm_ms = total_time_ms / NQ_;    // Average time per query
        curr_stats.qps = NQ_ * 1000.0 / total_time_ms; // Queries per second

        std::cout << "recall:     " << curr_stats.recall << '\t' << nprobe << '\n';
        std::cout << "avg_query_time: " << curr_stats.avg_tm_ms << "ms\t" << "qps: " << curr_stats.qps << '\n';

        std::cout << "reletive error - visited - total: "
                  << curr_stats.visited_rerr.tot_avg() * 100 << "\t"
                  << curr_stats.visited_rerr.tot_max() * 100 << std::endl;
        std::cout << "reletive error - visited - query-avg: "
                  << curr_stats.visited_rerr.group_avg_avg() * 100 << "\t"
                  << curr_stats.visited_rerr.group_mx_avg() * 100 << std::endl;
        return curr_stats;
    }

  public:
    void runIVFTest(const std::string &dataset, float B, size_t num_threads) {
        compress_ratio_ = B / 32;
        K_ = FLAGS_K;

        QuantizeConfig cfg;
        auto args_str = parseArgs(&cfg);
        LOG(INFO) << args_str << "\n";

        // Create file paths and load all data needed for IVF test
        DataFilePaths paths;
        utils::load_something<float, FloatRowMat>(paths.data_file.c_str(), data_);
        utils::load_something<float, FloatRowMat>(paths.query_file.c_str(), query_);
        utils::load_something<PID, UintRowMat>(paths.gt_file.c_str(), gt_);
        utils::load_something<float, FloatRowMat>(paths.centroids_file.c_str(), centroids_);
        utils::load_something<PID, UintRowMat>(paths.cids_file.c_str(), cids_);

        N_ = data_.rows();
        DIM_ = query_.cols();
        NQ_ = query_.rows();

        std::cout << "data loaded\n";
        std::cout << "\tN: " << N_ << '\n'
                  << "\tDIM: " << DIM_ << '\n';
        std::cout << "query loaded\n";
        std::cout << "\tNQ: " << NQ_ << '\n';
        {
            LOG(INFO) << "B: " << B << ", Compress ratio: " << compress_ratio_;
            LOG(INFO) << fmt::format("=============> Only first {}/{} dimensions are computed", int(compress_ratio_ * DIM_), DIM_);
        }

        std::string result_file = fmt::format("{}/RAWIVF_{}_exhaf{}", paths.result_path, dataset, B);
        runPCASearchTests(result_file, num_threads);
    }

  private:
    void runPCASearchTests(const std::string &result_file, size_t thread_num) {
        std::vector<Stats> statistics;

        statistics.push_back(PCAsearch(200, thread_num));

        // === output to csv ===
        std::ofstream csv_data(result_file + ".csv", std::ios::out);
        std::string final_result = "nprobe,num_threads,QPS,avg_tm_ms,recall,ratio,bw_mbps,err_tot_avg,err_tot_max,err_q_avg_avg,err_q_mx_avg\n";
        csv_data << "nprobe,num_threads,QPS,avg_tm_ms,recall,ratio,bw_mbps,err_tot_avg,err_tot_max,err_q_avg_avg,err_q_mx_avg" << std::endl;

        for (size_t i = 0; i < statistics.size(); ++i) {
            auto &curr = statistics[i];
            // Format: nprobe,num_threads,QPS,avg_tm_ms,recall,ratio,bw_mbps,err_tot_avg,err_tot_max,err_q_avg_avg,err_q_mx_avg
            auto ts = fmt::format("{},{},{},{},{},{},{},{},{},{},{}\n",
                                  curr.nprobe, thread_num, curr.qps, curr.avg_tm_ms, curr.recall, 0, 0,
                                  curr.visited_rerr.tot_avg(), curr.visited_rerr.tot_max(),
                                  curr.visited_rerr.group_avg_avg(), curr.visited_rerr.group_mx_avg());
            csv_data << ts;
            final_result += ts;
        }
        csv_data.close();

        LOG(INFO) << "result log to file: " << result_file << ".csv";
        std::cout << final_result << std::endl;
    }
};

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string DATASET_str = FLAGS_dataset;
    const float B = FLAGS_B;

    IVFTester tester;
    tester.runIVFTest(DATASET_str, B, FLAGS_num_threads);

    return 0;
}
