from collections import defaultdict
import gzip
import json

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, OPTICS, cluster_optics_dbscan
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.data.filesystem import fopen
from src.models.ensemble import get_best_ensemble_matches
from src.models.utils import remove_padding, add_padding

NYSIIS_SCORE = 0.6


def get_sorted_similarities(embeddings, threshold=0.4, batch_size=1024):
    # TODO use ensemble score, not cosine_similarity(embeddings)
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


def _generate_agglomerative_clusters(cluster_threshold, linkage, distances):
    print("generating agglomerative clusters", linkage, 1 - cluster_threshold)
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - cluster_threshold,
                                  affinity="precomputed", linkage=linkage)
    return agg.fit_predict(distances)


def _convert_neg_to_unique(labels):
    max_cluster = max(labels)
    results = []
    for label in labels:
        if label < 0:
            max_cluster += 1
            label = max_cluster
        results.append(label)
    return results


def _generate_optics_clusters(min_samples, eps, max_eps, cluster_method, xi, n_jobs, distances):
    """generate clusters using OPTICS"""
    pass





    #  = shared
    # run_dbscan = False
    # if cluster_method == "xi+dbscan":
    #     cluster_method = "xi"
    #     run_dbscan = True
    # clust = OPTICS(min_samples=min_samples,
    #                cluster_method=cluster_method,
    #                xi=xi,
    #                max_eps=max_eps,
    #                metric="cosine",
    #                n_jobs=n_jobs,
    #                )
    # clust.fit(closure_embeddings)
    #
    # if run_dbscan:
    #     labels = cluster_optics_dbscan(
    #         reachability=clust.reachability_,
    #         core_distances=clust.core_distances_,
    #         ordering=clust.ordering_,
    #         eps=eps,
    #     )
    # else:
    #     labels = clust.labels_
    #
    # return closure_id, enclosed_ids, _convert_neg_to_unique(labels)


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


def generate_clusters(distances,
                      cluster_algo="agglomerative",
                      cluster_linkage="average", cluster_threshold=0.5,  # for agglomerative
                      min_samples=2, eps=0.5,                            # for optics or hdbscan
                      max_eps=1.0, cluster_method="xi+dbscan", xi=0.05,  # for optics
                      selection_method="eom", min_cluster_size=2,        # for hdbscan
                      verbose=False, n_jobs=None):
    if cluster_algo == "agglomerative":
        results = _generate_agglomerative_clusters(cluster_threshold, cluster_linkage, distances)
    elif cluster_algo == "optics":
        results = _generate_optics_clusters(min_samples, eps, max_eps, cluster_method, xi, n_jobs, distances)
    else:
        results = _generate_hdbscan_clusters((min_samples, eps, selection_method, min_cluster_size), *params)
    return results


def get_clusters(names_to_cluster,
                 clustered_names,
                 swivel_model,
                 swivel_vocab,
                 encoder_model,
                 ensemble_model,
                 name_freq,
                 max_clusters=5,
                 batch_size=1024,
                 k=5000,
                 is_oov_ensemble_model=True,
                 n_jobs=1,
                 verbose=True):
    """
    For each name in all_names, find the closest clustered names and return a list of (cluster_id, cluster_score) tuples
    :param names_to_cluster: names to assign clusters
    :param clustered_names: clustered name to cluster id
    :param swivel_model: swivel model
    :param swivel_vocab: swivel vocabulary
    :param encoder_model: encoder model
    :param ensemble_model: ensemble model
    :param name_freq: map names to their frequencies
    :param max_clusters: maximum number of clusters to return for each input name
    :param batch_size:
    :param k: number of names to consider in the ensemble model and to get the best clusters
    :param is_oov_ensemble_model: is the ensemble model an oov (out-of-vocab) model
    :return: for each name in input_names, a list of up to max_clusters (cluster_id, cluster_score) tuples
    and also a dictionary mapping cluster id to the names for which that cluster is closest (that are assigned to the cluster)
    """
    name2clusters = defaultdict(list)
    cluster2names = defaultdict(list)

    similar_names_scores = get_best_ensemble_matches(
        model=swivel_model,
        vocab=swivel_vocab,
        input_names=names_to_cluster,
        candidate_names=np.asarray(list(clustered_names.keys())),
        encoder_model=encoder_model,
        ensemble_model=ensemble_model,
        name_freq=name_freq,
        k=k,
        batch_size=batch_size,
        add_context=True,
        n_jobs=n_jobs,
        is_oov_model=is_oov_ensemble_model,
        verbose=verbose,
    )

    for name, names_scores in zip(names_to_cluster, similar_names_scores):
        # currently, never true, because len(names_scores) == k as a result of get_best_ensemble_matches
        if len(names_scores) > k * 4:
            partitioned_ixs = np.argpartition(names_scores[:, 1], -k, axis=0)[-k:]
            sorted_ixs = np.flip(np.argsort(np.take_along_axis(names_scores[:, 1], partitioned_ixs, axis=0), axis=0),
                                 axis=0)
            sorted_ixs = np.take_along_axis(partitioned_ixs, sorted_ixs, axis=0)
        else:
            sorted_ixs = np.flip(np.argsort(names_scores[:, 1], axis=0), axis=0)[:k]
        found_clusters = set()
        for sorted_ix in sorted_ixs:
            similar_name, similar_score = names_scores[sorted_ix]
            similar_cluster_id = clustered_names[similar_name]
            if similar_cluster_id in found_clusters:
                continue
            if len(found_clusters) == 0:
                cluster2names[similar_cluster_id].append(name)
            found_clusters.add(similar_cluster_id)
            name2clusters[name].append((similar_cluster_id, similar_score))
            if len(found_clusters) == max_clusters:
                break

    return name2clusters, cluster2names


def get_best_cluster_matches(name2clusters, cluster2names, input_names):
    # return 3D array: [input name, candidate, (name, score)]
    cluster_names_scores = []
    max_cluster_names = 0
    for input_name in input_names:
        cluster_names = []
        cluster_scores = []
        for cluster_id, cluster_score in name2clusters[input_name]:
            for name in cluster2names[cluster_id]:
                cluster_names.append(name)
                cluster_scores.append(cluster_score)
        if len(cluster_names) > max_cluster_names:
            max_cluster_names = len(cluster_names)
        cluster_names_scores.append((cluster_names, cluster_scores))

    # make sure the second dimension is the same for all input names
    all_cluster_names = []
    all_scores = []
    for cluster_names, cluster_scores in cluster_names_scores:
        if len(cluster_names) < max_cluster_names:
            cluster_names.extend([''] * (max_cluster_names - len(cluster_names)))
            cluster_scores.extend([-1.0] * (max_cluster_names - len(cluster_scores)))
        all_cluster_names.append(np.array(cluster_names, dtype=object))
        all_scores.append(np.array(cluster_scores, dtype=np.float32))

    # return 3D array: [input name, candidate, (name, score)]
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
