from collections import defaultdict
import gzip
import json

import hdbscan
import jellyfish
from mpire import WorkerPool
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, OPTICS, cluster_optics_dbscan
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.data.filesystem import fopen
from src.models.utils import remove_padding, add_padding

NYSIIS_SCORE = 0.6


def get_sorted_similarities(embeddings, threshold=0.4, batch_size=1024):
    rows = []
    cols = []
    scores = []
    for ix in range(0, len(embeddings), batch_size):
        batch = embeddings[ix:ix + batch_size]
        batch_scores = cosine_similarity(batch, embeddings)
        indices = np.where(batch_scores >= threshold)
        scores.extend(batch_scores[indices])
        batch_rows, batch_cols = indices
        batch_rows += ix
        rows.extend(batch_rows)
        cols.extend(batch_cols)
    score_sorted_ixs = np.array(scores).argsort()[::-1]
    sorted_similarities = np.column_stack((rows, cols, scores))[score_sorted_ixs]
    sorted_similarities = sorted_similarities[sorted_similarities[:, 0] < sorted_similarities[:, 1]]
    return sorted_similarities


def generate_closures(sorted_scores, closure_threshold):
    id2closure = {}
    closure2ids = {}
    next_closure_id = 0
    not_merged = 0
    max_score_not_merged = 0
    for id1, id2, score in sorted_scores:
        id1 = int(id1)
        id2 = int(id2)
        if id1 == id2:
            print(id1)
            continue
        if id1 not in id2closure and id2 not in id2closure:
            closure_id = next_closure_id
            next_closure_id += 1
            id2closure[id1] = closure_id
            id2closure[id2] = closure_id
            closure2ids[closure_id] = {id1, id2}
            continue
        if id1 not in id2closure and id2 in id2closure:
            id1, id2 = id2, id1
        if id1 in id2closure and id2 not in id2closure:
            closure_id = id2closure[id1]
            id2closure[id2] = closure_id
            closure2ids[closure_id].add(id2)
            continue
        if id1 in id2closure and id2 in id2closure:
            closure_id = id2closure[id1]
            closure2_id = id2closure[id2]
            if closure_id == closure2_id:
                continue
            # don't merge closures if the result would be too large
            if len(closure2ids[closure_id]) + len(closure2ids[closure2_id]) > closure_threshold:
                if not_merged == 0:
                    max_score_not_merged = score
                not_merged += 1
                continue
            closure2ids[closure_id].update(closure2ids[closure2_id])
            for _id in closure2ids[closure2_id]:
                id2closure[_id] = closure_id
            del closure2ids[closure2_id]

    return id2closure, closure2ids, not_merged, max_score_not_merged


def _dist_metric(scores):
    """
    Return a function to return the distance between two points
    """
    def distance(x, y):
        x = int(x[0])
        y = int(y[0])
        return 1 - scores[x, y]
    return distance


def _generate_agglomerative_clusters(shared, closure_id, enclosed_ids, closure_embeddings):
    cluster_threshold, linkage = shared
    affinity = "cosine"
    if linkage == "ward":
        affinity = "euclidean"
        closure_embeddings = normalize(closure_embeddings)
        cluster_threshold = cluster_threshold / 10.0  # ward threshold is a different scale
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - cluster_threshold,
                                  affinity=affinity, linkage=linkage)
    cluster_assignments = agg.fit_predict(closure_embeddings)
    return closure_id, enclosed_ids, cluster_assignments


def _convert_neg_to_unique(labels):
    max_cluster = max(labels)
    results = []
    for label in labels:
        if label < 0:
            max_cluster += 1
            label = max_cluster
        results.append(label)
    return results


def _generate_optics_clusters(shared, closure_id, enclosed_ids, closure_embeddings):
    min_samples, eps, max_eps, cluster_method, xi, n_jobs = shared
    run_dbscan = False
    if cluster_method == "xi+dbscan":
        cluster_method = "xi"
        run_dbscan = True
    clust = OPTICS(min_samples=min_samples,
                   cluster_method=cluster_method,
                   xi=xi,
                   max_eps=max_eps,
                   metric="cosine",
                   n_jobs=n_jobs,
                   )
    clust.fit(closure_embeddings)

    if run_dbscan:
        labels = cluster_optics_dbscan(
            reachability=clust.reachability_,
            core_distances=clust.core_distances_,
            ordering=clust.ordering_,
            eps=eps,
        )
    else:
        labels = clust.labels_
        
    return closure_id, enclosed_ids, _convert_neg_to_unique(labels)


def _generate_hdbscan_clusters(shared, closure_id, enclosed_ids, closure_embeddings):
    min_samples, eps, selection_method, min_cluster_size = shared
    clust = hdbscan.HDBSCAN(min_samples=min_samples,
                            cluster_selection_epsilon=eps,
                            cluster_selection_method=selection_method,
                            min_cluster_size=min_cluster_size,
                            metric="euclidean",
                            )
    closure_embeddings = normalize(closure_embeddings)
    clust.fit(closure_embeddings)
    return closure_id, enclosed_ids, _convert_neg_to_unique(clust.labels_)


def generate_clusters(closure2ids, cluster_embeddings,
                      cluster_algo="agglomerative",
                      cluster_linkage="average", cluster_threshold=0.5,  # for agglomerative
                      min_samples=2, eps=0.5,                            # for optics or hdbscan
                      max_eps=1.0, cluster_method="xi+dbscan", xi=0.05,  # for optics
                      selection_method="eom", min_cluster_size=2,        # for hdbscan
                      verbose=True, n_jobs=None):
    id2cluster = {}
    # calculate parameters
    params_list = []
    for closure_id, ids in closure2ids.items():
        enclosed_ids = list(ids)
        closure_embeddings = cluster_embeddings[enclosed_ids]
        params_list.append((closure_id, enclosed_ids, closure_embeddings))
    if n_jobs == 1 or cluster_algo == "optics":
        results_list = []
        if verbose:
            params_list = tqdm(params_list)
        for params in params_list:
            if cluster_algo == "agglomerative":
                results = _generate_agglomerative_clusters((cluster_threshold, cluster_linkage), *params)
            elif cluster_algo == "optics":
                results = _generate_optics_clusters((min_samples, eps, max_eps, cluster_method, xi, n_jobs), *params)
            else:
                results = _generate_hdbscan_clusters((min_samples, eps, selection_method, min_cluster_size), *params)
            results_list.append(results)
    else:
        if cluster_algo == "agglomerative":
            with WorkerPool(shared_objects=(cluster_threshold, cluster_linkage), n_jobs=n_jobs) as pool:
                results_list = pool.map_unordered(_generate_agglomerative_clusters, params_list, progress_bar=verbose)
        else:
            with WorkerPool(shared_objects=(min_samples, eps, selection_method, min_cluster_size), n_jobs=n_jobs) as pool:
                results_list = pool.map_unordered(_generate_hdbscan_clusters, params_list, progress_bar=verbose)

    # process results
    for results in results_list:
        closure_id, enclosed_ids, cluster_assignments = results
        for _id, cluster_id in zip(enclosed_ids, cluster_assignments):
            id2cluster[_id] = f"{closure_id}_{cluster_id}"

    # assign clustered_names not in any cluster to their own (singleton) cluster
    for _id in range(0, len(cluster_embeddings)):
        if _id not in id2cluster:
            id2cluster[_id] = f"{_id}"

    return id2cluster


def get_clusters(all_names, all_embeddings, id2cluster, cluster_embeddings, max_clusters=5, batch_size=1024, k=100, verbose=True):
    """
    For each name in all_names, find the closest clustered names and return a list of (cluster_id, cluster_score) tuples
    :param all_names: all names to assign
    :param all_embeddings: embeddings for names to assign
    :param id2cluster: clustered name id to cluster id
    :param cluster_embeddings: embeddings for clustered names
    :param batch_size:
    :param k:
    :return: for each name in all_names, a list of up to max_clusters (cluster_id, cluster_score) tuples
    and also a dictionary mapping cluster id to the names for which that cluster is closest (that are assigned to the cluster)
    """
    name2clusters = defaultdict(list)
    cluster2names = defaultdict(list)
    ixs = range(0, len(all_embeddings), batch_size)
    if verbose:
        ixs = tqdm(ixs)
    for ix in ixs:
        batch = all_embeddings[ix:ix + batch_size]
        scores = cosine_similarity(batch, cluster_embeddings)
        # ids = np.argsort(-scores)[:, :k]  # slow
        # get the top k
        ids = np.argpartition(-scores, k)[:, :k]
        scores = scores[np.arange(len(batch))[:, None], ids]
        # then sort the top k
        ids_sort = np.argsort(-scores)
        ids = ids[np.arange(len(batch))[:, None], ids_sort]
        scores = scores[np.arange(len(batch))[:, None], ids_sort]
        for name_id, _ids, _scores in zip(list(range(ix, ix + len(batch))), ids, scores):
            name = all_names[name_id]
            embedding = all_embeddings[name_id]
            found_clusters = set()
            # look for nearby clusters only if embedding is not zero
            if np.any(embedding):
                for _id, _score in zip(_ids, _scores):
                    cluster_id = id2cluster[_id]
                    if cluster_id not in found_clusters:
                        if len(found_clusters) == 0:
                            cluster2names[cluster_id].append(name)
                        found_clusters.add(cluster_id)
                        name2clusters[name].append((cluster_id, _score))
                        if len(found_clusters) == max_clusters-1:
                            break
            # add nysiis code
            cluster_id = "_"+jellyfish.nysiis(remove_padding(name)).upper()
            _score = NYSIIS_SCORE
            if len(found_clusters) == 0:
                cluster2names[cluster_id].append(name)
            name2clusters[name].append((cluster_id, _score))
            # re-sort
            name2clusters[name].sort(key=lambda tup: tup[1], reverse=True)

    return name2clusters, cluster2names


def get_best_cluster_matches(name2clusters, cluster2names, input_names, k=256):
    # return [input names, candidates, (names, scores)]
    all_cluster_names = []
    all_scores = []
    for input_name in input_names:
        cluster_names = []
        cluster_scores = []
        for cluster_id, cluster_score in name2clusters[input_name]:
            for name in cluster2names[cluster_id]:
                if len(cluster_names) == k:
                    continue
                cluster_names.append(name)
                cluster_scores.append(cluster_score)
        if len(cluster_names) < k:
            pad_len = k - len(cluster_names)
            cluster_names += [""] * pad_len
            cluster_scores += [0.0] * pad_len
        all_cluster_names.append(np.array(cluster_names, dtype=object))
        all_scores.append(np.array(cluster_scores, dtype=np.float32))
    all_cluster_names = np.vstack(all_cluster_names)
    all_scores = np.vstack(all_scores)
    return np.dstack((all_cluster_names, all_scores))


def merge_name2clusters(name2clusters):
    # merge clusters from all (variant) entries in name2clusters and return a list of (cluster, max score)
    cluster2score = {}
    for cluster_scores in name2clusters.values():
        for cluster_id, score in cluster_scores:
            if cluster_id not in cluster2score or cluster2score[cluster_id] < score:
                cluster2score[cluster_id] = score
    result = list(cluster2score.items())
    result.sort(key=lambda it: it[1], reverse=True)
    return result


def _get_most_frequent_name(names, name2freq):
    freq_name = None
    max_freq = 0
    for name in names:
        if name in name2freq and (freq_name is None or name2freq[name] > max_freq):
            freq_name = name
            max_freq = name2freq[name]
    if freq_name is None:
        freq_name = names[0]
    return freq_name


def prettify_cluster_names(cluster_names, id2cluster, pref_name2freq):
    cluster2names = defaultdict(list)
    for name_id, cluster_id in id2cluster.items():
        cluster2names[cluster_id].append(cluster_names[name_id])
    cluster2pretty = {}
    for cluster_id, names in cluster2names.items():
        cluster2pretty[cluster_id] = remove_padding(_get_most_frequent_name(names, pref_name2freq)).upper()
    return {name_id: cluster2pretty[cluster_id] for name_id, cluster_id in id2cluster.items()}


def write_clusters(path, cluster_names, id2cluster):
    df = pd.DataFrame([{"name": remove_padding(name), "cluster": id2cluster[ix]} for ix, name in enumerate(cluster_names)])
    df.to_csv(path, index=False)


def read_clusters(path):
    df = pd.read_csv(path, na_filter=False)
    return {add_padding(name): cluster for name, cluster in zip(df["name"], df["cluster"])}


def write_cluster_scores(path, name2clusters):
    data = ("\n".join([json.dumps({
        "name": remove_padding(name),
        "clusters": [(cluster[0], float(cluster[1])) for cluster in clusters],
    }) for name, clusters in name2clusters.items()])).encode("utf-8")
    with gzip.open(fopen(path, "wb"), "wb") as f:
        f.write(data)


def read_cluster_scores(path):
    with gzip.open(fopen(path, "rb"), "rb") as f:
        data = f.read()
    name2clusters = {}
    for line in data.decode("utf-8").split("\n"):
        name_clusters = json.loads(line)
        name2clusters[add_padding(name_clusters["name"])] = [(cluster[0], float(cluster[1])) for cluster in name_clusters["clusters"]]
    return name2clusters
