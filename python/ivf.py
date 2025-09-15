import sys
import faiss
import os
from utils.io import write_fvecs, write_ivecs, read_somefiles
import time

SOURCE = "./data/"
PATH = "./data/openai1536"
DATASET = "openai1536"
K = 4096
input_suffix = ""
output_suffix = ""
DISTANCE = "L2"  # Default distance metric

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
        PATH = os.path.join(SOURCE, DATASET)
        if "/" in DATASET:
            PATH = "/".join(DATASET.split("/")[:-1])
            DATASET = DATASET.split("/")[-1]
            print( f"Using dataset: {DATASET} from path: {PATH}")

    # Allow optional parameter to override variance threshold
    if len(sys.argv) > 2:
        K = int(sys.argv[2])

    # Allow optional parameter to set distance metric
    if len(sys.argv) > 3:
        DISTANCE = sys.argv[3].upper()
        if DISTANCE not in ["L2", "IP"]:
            print(f"Invalid distance metric: {DISTANCE}. Using L2 instead.")
            DISTANCE = "L2"

    if DISTANCE == "IP":
        output_suffix = output_suffix + ".ip"

    data_path = os.path.join(PATH, f"{DATASET}_base{input_suffix}.fvecs")
    if not os.path.exists(data_path):
        data_path = os.path.join(PATH, f"{DATASET}_base{input_suffix}.fbin")

    if len(sys.argv) > 4:
        data_path = sys.argv[4]

    print(f"Clustering - {DATASET} ({data_path}) with {K} clusters using {DISTANCE} distance")

    X = read_somefiles(data_path)

    # data_path = os.path.join(path, f"{DATASET}_base.bin")
    # X = read_fbin(data_path)

    D = X.shape[1]
    centroids_path = os.path.join(
        PATH, f"{DATASET}_centroid_{K}{output_suffix}.fvecs")
    dist_to_centroid_path = os.path.join(
        PATH, f"{DATASET}_dist_to_centroid_{K}{output_suffix}.fvecs")
    cluster_id_path = os.path.join(
        PATH, f"{DATASET}_cluster_id_{K}{output_suffix}.ivecs")

    # cluster data vectors
    if DISTANCE == "IP":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, K)
    else:  # L2 distance
        index = faiss.index_factory(D, f"IVF{K},Flat")

    index.verbose = True

    start_time = time.time()
    index.train(X)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    dist_to_centroid, cluster_id = index.quantizer.search(X, 1)

    # For L2 distance, Faiss returns squared distances, so we take square root
    if DISTANCE == "L2":
        dist_to_centroid = dist_to_centroid**0.5
    # For IP distance, higher values mean more similar vectors (opposite of L2)
    # We keep the original values

    # write_fvecs(dist_to_centroid_path, dist_to_centroid)
    write_ivecs(cluster_id_path, cluster_id)
    write_fvecs(centroids_path, centroids)
