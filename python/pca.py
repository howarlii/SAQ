import sys
import time
import faiss
import os
from utils.io import read_fvecs, write_fvecs
import numpy as np

SOURCE = "./data/"
DATASET = "openai1536"

base_only = False  # Set to True if you only want to process the base data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]

    print(f"Performing PCA on {DATASET}")

    # Path
    path = os.path.join(SOURCE, DATASET)
    base_path = os.path.join(path, f"{DATASET}_base.fvecs")
    query_path = os.path.join(path, f"{DATASET}_query.fvecs")
    centroid_path = os.path.join(path, f"{DATASET}_centroid_4096.fvecs")

    X_base = read_fvecs(base_path)
    print(f"Base data shape: {X_base.shape}")
    if (not base_only):
        X_query = read_fvecs(query_path)
        X_centroid = read_fvecs(centroid_path)
        print(f"Query data shape: {X_query.shape}")
        print(f"Centroid data shape: {X_centroid.shape}")
    D_IN = X_base.shape[1]
    D_OUT = D_IN

    # Initialize PCA with the determined D_OUT
    pca_matrix = faiss.PCAMatrix(D_IN, D_OUT)
    print("Training PCA...")
    start_time = time.time()
    pca_matrix.train(X_base)
    print(f"Training time: {time.time() - start_time:.3f} seconds")

    # Extract PCA components for C++ reproduction
    print("Extracting PCA components...")
    # Get the PCA transformation matrix
    eigen_vecs = faiss.vector_to_array(pca_matrix.eigenvalues)
    pca_matrix_array = faiss.vector_to_array(pca_matrix.PCAMat).reshape(D_OUT, D_IN)
    pca_mean = faiss.vector_to_array(pca_matrix.mean).reshape(1, D_IN)

    print(eigen_vecs)
    # Combine the transformation matrix and mean into a single array for easy export
    # First row is the mean vector, followed by the transformation matrix
    # pca_matrix_array = np.vstack([b.reshape(1, -1), A])

    # Save eigenvalues for variance information
    # eigen_vals_path = os.path.join(path, f"{DATASET}_pca_eigenvals.fvecs")
    # write_fvecs(eigen_vals_path, eigen_vecs.reshape(1, -1))
    # print(f"PCA eigenvalues saved to {eigen_vals_path}")

    print("Applying PCA transformation...")
    start_time = time.time()
    X_base_transformed = pca_matrix.apply(X_base)
    if (not base_only):
        X_query_transformed = pca_matrix.apply(X_query)
        X_centroid_transformed = pca_matrix.apply(X_centroid)
    print(f"Transformation time: {time.time() - start_time:.3f} seconds")

    # PCA matrix path
    pca_mean_path = os.path.join(path, f"{DATASET}_pca_mean.fvecs")
    pca_mat_path = os.path.join(path, f"{DATASET}_pca_matrix.fvecs")
    transformed_base_path = os.path.join(path, f"{DATASET}_base_pca.fvecs")
    transformed_query_path = os.path.join(path, f"{DATASET}_query_pca.fvecs")
    transformed_centroid_path = os.path.join(
        path, f"{DATASET}_centroid_4096_pca.fvecs")

    # Save the transformed data
    write_fvecs(pca_mean_path, pca_mean)
    write_fvecs(pca_mat_path, pca_matrix_array)
    write_fvecs(transformed_base_path, X_base_transformed)
    if (not base_only):
        write_fvecs(transformed_query_path, X_query_transformed)
        write_fvecs(transformed_centroid_path, X_centroid_transformed)

    print(f"PCA mean saved to {pca_mean_path}")
    print(f"PCA matrix saved to {pca_mat_path}")
    print(f"Transformed base data saved to {transformed_base_path}")
    print(f"Transformed query data saved to {transformed_query_path}")
    print(f"Transformed centroid data saved to {transformed_centroid_path}")
    print(f"PCA completed.")

    base_variance_path = os.path.join(path, f"{DATASET}_base_pca.vars.fvecs")
    base_variance = np.var(X_base_transformed, axis=0)
    base_variance = np.array([base_variance])
    print(f"shape of base_variance: {base_variance.shape}")

    print(base_variance)
    # Save the variance data
    write_fvecs(base_variance_path, base_variance)
    print(f"Centroid-adjusted variance of transformed base data saved to {base_variance_path}")