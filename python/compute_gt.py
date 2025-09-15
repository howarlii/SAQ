import numpy as np
import sys
import concurrent.futures
from utils.io import read_fvecs, write_ivecs


if __name__ == "__main__":
    dataset = sys.argv[1]
    num_threads = 8
    print(f"Loading dataset {dataset}...")
    base = read_fvecs(f"./data/{dataset}/{dataset}_base.fvecs")
    query = read_fvecs(f"./data/{dataset}/{dataset}_query.fvecs")

    print(f"Computing ground truth for {len(query)} queries using {num_threads} threads...")

    def process_query(q):
        """Process a single query vector and return its top 1000 nearest neighbors."""
        distances = np.linalg.norm(base - q, axis=1)
        return list(np.argsort(distances))[:1000]

    # Prepare tasks for parallel processing
    tasks = [q for q in query]

    # Process queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        gt = list(executor.map(process_query, tasks))

    gt = np.array(gt)

    print(f"Writing results to ./data/{dataset}/{dataset}_groundtruth.ivecs")
    write_ivecs(f"./data/{dataset}/{dataset}_groundtruth.ivecs", gt)
    print("Done!")
