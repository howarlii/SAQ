#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <sys/stat.h>
#include <vector>

#include "third/Eigen/Core"

namespace saqlib::utils {
template <typename T>
void save_vector(std::ofstream &output, const std::vector<T> &vec) {
    output.write(reinterpret_cast<const char *>(&vec.size()), sizeof(size_t));
    for (const auto &item : vec) {
        output.write(reinterpret_cast<const char *>(&item), sizeof(T));
    }
}
template <typename T>
void load_vector(std::ifstream &input, std::vector<T> &vec) {
    size_t size;
    input.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    vec.clear();
    vec.resize(size);
    for (size_t i = 0; i < size; ++i) {
        input.read(reinterpret_cast<char *>(&vec[i]), sizeof(T));
    }
}

inline void save_floatvec(std::ofstream &output, const Eigen::RowVectorXf &vec) {
    size_t size = vec.size();
    output.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(vec.data()), sizeof(float) * size);
}

inline size_t load_floatvec(std::ifstream &input, Eigen::RowVectorXf &vec) {
    size_t size;
    input.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    vec.resize(size);
    input.read(reinterpret_cast<char *>(vec.data()), sizeof(float) * size);

    return size;
}

inline size_t get_filesize(const char *filename) {
    struct stat64 stat_buf;
    int rc = stat64(filename, &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

inline bool file_exists(const char *filename) {
    std::ifstream f(filename);
    if (!f.good()) {
        f.close();
        return false;
    }
    f.close();
    return true;
}

template <typename T>
T *load_vecs(const char *filename) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    uint32_t cols;
    size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);

    input.read((char *)&cols, sizeof(uint32_t));

    size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
    T *data = new T[rows * cols];

    input.seekg(0, input.beg);

    for (size_t i = 0; i < rows; i++) {
        input.read((char *)&cols, sizeof(uint32_t));
        input.read((char *)&data[cols * i], sizeof(T) * cols);
    }

    input.close();
    return data;
}

template <typename T, class M>
void load_vecs(const char *filename, M &Mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    assert(typeid(T *) == typeid(Mat.data()));

    uint32_t tmp;
    size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);

    input.read((char *)&tmp, sizeof(uint32_t));

    size_t cols = tmp;
    size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
    Mat = M(rows, cols);

    input.seekg(0, input.beg);

    for (size_t i = 0; i < rows; i++) {
        input.read((char *)&tmp, sizeof(uint32_t));
        input.read((char *)&Mat(i, 0), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded (";
    std::cout << "Rows " << rows << " Cols " << cols << ")\n"
              << std::flush;
    input.close();
}

template <typename T, class M>
void save_vecs(const char *filename, const M &Mat) {
    std::ofstream output(filename, std::ios::binary);

    uint32_t cols = static_cast<uint32_t>(Mat.cols());
    uint32_t rows = static_cast<uint32_t>(Mat.rows());

    // Write initial cols (format header)
    // output.write(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

    // For each row, write cols followed by row data
    for (size_t i = 0; i < rows; i++) {
        output.write(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
        output.write(reinterpret_cast<const char *>(&Mat(i, 0)), sizeof(T) * cols);
    }

    output.close();
    std::cout << "File " << filename << " saved (";
    std::cout << "Rows " << rows << " Cols " << cols << ")\n"
              << std::flush;
}

template <typename T, class M>
void load_bin(const char *filename, M &Mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    assert(typeid(T *) == typeid(Mat.data()));

    uint32_t rows, cols;
    std::ifstream input(filename, std::ios::binary);

    input.read((char *)&rows, sizeof(uint32_t));
    input.read((char *)&cols, sizeof(uint32_t));

    Mat = M(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        input.read((char *)&Mat(i, 0), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded (";
    std::cout << "Rows " << rows << " Cols " << cols << ")\n"
              << std::flush;
    input.close();
}

// load based on file extension
template <typename T, class M>
void load_something(const char *filename, M &row_mat) {
    std::string filename_str(filename);
    if (filename_str.rfind("vecs") == filename_str.size() - 4) {
        load_vecs<T, M>(filename, row_mat);
    } else if (filename_str.rfind("bin") == filename_str.size() - 3) {
        load_bin<T, M>(filename, row_mat);
    } else {
        std::cerr << "Unsupported file format: " << filename << std::endl;
        exit(1);
    }
}
} // namespace saqlib::utils