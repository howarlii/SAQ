# Datasets

#### Orgnization
Datasets and corresponding indices are stored in `./data/${dataset}`. For example, `./data/openai1536` contains `openai1536_base.fvecs`, `openai1536_query.fvecs`, `openai1536_groundtruth.ivecs` and indices.


#### Download and pre-processing
Use `python ./python/download_dataset_openai.py` to download openai embeddings. Then run `python ./python/compute_gt.py openai1536` to generate groundtruth for KNN search.

More tested datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html.

For datasets without groundtruth, please refer to `./python/compute_gt.py` or (`src/compute_gt.cpp`) to generate the groundtruth.


#### Clustering
You need to use python and faiss library to train an IVF index.
```
python ./python/ivf.py openai1536 4096
```
where `openai1536` is the dataset name and `4096` is the number of clusters.

Once the process is finished, corresponding data dir will contain centroid vectors of IVF clusters and the cluster id for each data vector. For example, `./data/openai1536/` will contain `openai1536_centroid_4096.fvecs` and `openai1536_cluster_id_4096.ivecs`, where `4096` is `K`. Then, you can use them to build the index.


#### PCA
Before building indices by SAQ, you need to perform PCA projection on the dataset.
```
python ./python/pca.py openai1536
```

The PCA matrix is trained base on the data vectors and apply to the data vectors , centroids and queries, which are stored in `./data/{dataset}/{dataset}_base_pca.fvecs`, `./data/{dataset}/{dataset}_centroid_{K}_pca.fvecs` and `./data/{dataset}/{dataset}_query_pca.fvecs` respectively. The variance of each dimension of data vectors are also saved in `./data/{dataset}/{dataset}_base_pca.vars.fvecs`.