import pandas as pd
import re
from dataset_preprocessor import load_and_preprocess_openml_dataset
from sklearn.metrics import adjusted_rand_score
import time
import statistics
from cohirf import CoHiRF
from sklearn.cluster import KMeans

pattern = r'\b(R|C|q)\s*=\s*(\d+\.?\d*)\b'
def extract_params(s):
    matches = re.findall(pattern, s)
    return {k: int(v) for k, v in matches}

def run_cohirf(X, q, R, C, kmeans_n_init=1):
    model = CoHiRF(components_size=q, repetitions=R, kmeans_n_clusters=C, kmeans_n_init=kmeans_n_init)
    return model.fit_predict(X), model.n_iter_

def run_kmeans(X, C_kmeans):
    k_means_estimator = KMeans(n_clusters=C_kmeans, init='k-means++',
                                n_init=10,
                                max_iter=300, tol=1e-4,
                                verbose=0,
                                random_state=None, algorithm='lloyd')
    return k_means_estimator.fit_predict(X)

def rigorous_benchmark(func, *args, warmup=3, runs=100, **kwargs):
    # Warmup runs (not measured)
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = [None] * runs
    for i in range(runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times[i] = end - start
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'stdev': statistics.stdev(times) if runs > 1 else 0,
    }

df = pd.read_csv('cohirf_realdata.csv')
DATASETS = df.loc[:, ['openml_id', 'dataset']].drop_duplicates()

for DATASET_NAME in DATASETS.dataset:
    print(DATASET_NAME)
    filtered = df.query('dataset == @DATASET_NAME')
    dataset_id = filtered.iloc[0].openml_id.item()
    print(dataset_id)

    # Get parameters
    C_kmeans = extract_params(filtered.query('model == "K-MEANS"').iloc[0].parameters)["C"]
    p = extract_params(filtered.query('model == "COHIRF"').iloc[0].parameters)
    C = p.get('C')
    q = p.get('q')
    R = p.get('R')

    # Load data
    full_dataset = load_and_preprocess_openml_dataset(dataset_id, standardize=True)
    X = full_dataset["X"]
    y = full_dataset["y"]

    # CoHiRF
    print('R = ', R)
    print('C = ', C)
    print('q = ', q)
    
    best_cohirf_ari = 0
    for n in range(10):
        cohirf_pred, n_iter = run_cohirf(X, q, R, C, kmeans_n_init=10)
        cohirf_ari = adjusted_rand_score(cohirf_pred, y)

        if best_cohirf_ari < cohirf_ari:
            best_cohirf_ari = cohirf_ari
    print('CoHiRF ARI = ', best_cohirf_ari)

    # K-MEANS
    kmeans_pred = run_kmeans(X, C_kmeans)
    kmeans_ari = adjusted_rand_score(kmeans_pred, y)
    print('KMeans ARI = ', kmeans_ari)

    # # BENCHMARKING
    stats_cohirf = rigorous_benchmark(run_cohirf, X, q, R, C, kmeans_n_init=10, runs=10)
    print('timings cohirf = ', stats_cohirf['mean'] * 1000, ' ms')
    stats_kmeans = rigorous_benchmark(run_kmeans, X, C_kmeans, runs=10)
    print('timings kmeans = ', stats_kmeans['mean'] * 1000, ' ms')

    print("")