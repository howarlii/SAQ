#include <cstring>
#include <iostream>
#include <set>

#include "define_options.h"

#include "defines.hpp"
#include "utils/BS_thread_pool.hpp"
#include "utils/IO.hpp"
#include "utils/memory.hpp"
#include "utils/pool.hpp"
#include "utils/space.hpp"

using namespace saqlib;

constexpr size_t TOPK = 1000;
const size_t kNumThread = 100;

size_t N;
size_t DIM;
size_t NQ;
size_t K;

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string DATASET = FLAGS_dataset;
    DataFilePaths paths;

    FloatRowMat data;
    FloatRowMat queries;
    FloatRowMat centroids;
    UintRowMat gt;
    UintRowMat cids;

    utils::load_something<float, FloatRowMat>(paths.data_file.c_str(), data);
    utils::load_something<float, FloatRowMat>(paths.query_file.c_str(), queries);

    N = data.rows();
    DIM = queries.cols();
    NQ = queries.rows();

    std::cout << "data loaded\n";
    std::cout << "\tN: " << N << '\n'
              << "\tDIM: " << DIM << '\n';
    std::cout << "query loaded\n";
    std::cout << "\tNQ: " << NQ << '\n';

    BS::thread_pool pool(kNumThread);
    gt.resize(NQ, TOPK);

    // if (FLAGS_DEBUG) {
    //     NQ = 1;
    // }

    for (size_t qi = 0; qi < NQ; qi++) {
        pool.detach_task([&, qi]() {
            FloatVec query = queries.row(qi);
            utils::ResultPool KNNs(TOPK, FLAGS_searcher_dist_type == 1);
            if (FLAGS_searcher_dist_type == 0) { // L2Sqr
                for (size_t id = 0; id < N; ++id) {
                    auto dist = (data.row(id) - query).squaredNorm();
                    KNNs.insert(id, dist);
                }
            } else { // IP
                for (size_t id = 0; id < N; ++id) {
                    auto dist = query.dot(data.row(id));
                    KNNs.insert(id, dist);
                }
            }

            KNNs.copy_results(gt.row(qi).data());
        });
    }
    pool.wait();

    if (utils::file_exists(paths.gt_file.data())) {
        std::cout << "ground truth file exists\n";
        std::cout << "check if the ground truth is correct\n";
        UintRowMat gt_test;
        utils::load_something<PID, UintRowMat>(paths.gt_file.data(), gt_test);

        for (size_t i = 0; i < NQ; ++i) {
            std::set<PID> gt_set;
            for (size_t j = 0; j < TOPK; ++j) {
                gt_set.insert(gt_test(i, j));
            }
            for (size_t j = 0; j < TOPK; ++j) {
                if (gt_set.find(gt(i, j)) == gt_set.end()) {
                    std::cerr << "ground truth not match\n";
                    std::cerr << "query: " << i << '\n';
                    std::cerr << "gt: " << gt(i, j) << '\n';
                    std::cerr << "gt_test: " << gt_test(i, j) << '\n';
                    std::cerr << "-===========================\n";
                    // return -1;
                }
            }
        }
        std::cout << "ground truth is correct\n";
        return 0;
    }

    // if (FLAGS_DEBUG) {
    //     return 0;
    // }
    utils::save_vecs<float, UintRowMat>(paths.gt_file.data(), gt);
    std::cout << "ground truth saved to " << paths.gt_file << '\n';

    return 0;
}
