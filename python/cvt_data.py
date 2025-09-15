# Convert data from fbin/ibin to fvecs/ivecs
import sys
from utils.io import write_fvecs, read_fbin
import numpy as np


def bin_to_fvecs(filename):
    """
    Convert binary file to fvecs format.
    :param filename: Path to the binary file.
    :return: None
    """
    data = read_fbin(filename)
    write_fvecs(filename.replace('.bin', '.fvecs'), data)


def npy_to_fvecs(filename, keep_n=None):
    """
    Convert numpy file to fvecs format.
    :param filename: Path to the numpy file.
    :return: None
    """
    data = np.load(filename)
    print(filename, len(data), data.shape, data.dtype)
    if keep_n is not None:
        data = data[:keep_n]
        print(f"only keep first {keep_n} lines ", data.shape)
    write_fvecs(filename.replace('.npy', '.fvecs'), data)


def merge_npy_to_fvecs(filenames, output_filename, n=None):
    """
    Merge multiple numpy files into a single fvecs file.
    :param filenames: List of paths to the numpy files.
    :param output_filename: Path to the output fvecs file.
    :return: None
    """
    data = []
    tot_n = 0
    for filename in filenames:
        arr = np.load(filename)
        print(filename, len(arr), arr.shape, arr.dtype)
        data.append(arr)
        tot_n += len(arr)
        if (n is not None) and (tot_n >= n):
            print(f"Total lines reached {tot_n}, stopping early.")
            break

    data = np.concatenate(data, axis=0)

    if n is not None:
        data = data[:n]
        print(f"only keep first {n} lines ", data.shape)

    write_fvecs(output_filename, data)
    print(f"Data merged and written to {output_filename}")

if __name__ == "__main__":
    dataset = sys.argv[1]

    # bin_to_fvecs(f"./data/{dataset}/{dataset}_base.bin")
    # bin_to_fvecs(f"./data/{dataset}/{dataset}_query.bin")

    # npy_to_fvecs(f"./data/{dataset}/{dataset}_base.npy")
    # npy_to_fvecs(f"./data/{dataset}/{dataset}_query.npy", 1000)

    paths = []
    for i in range(10):
        paths.append(f"./data/{dataset}/raw/img_emb_{i:04d}.npy")
    merge_npy_to_fvecs(
        paths, f"./data/{dataset}/{dataset}_base.fvecs", 1000000)
