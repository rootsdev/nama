import gzip
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from src.data.filesystem import fopen
from src.data.utils import select_frequent_k
from src.eval.metrics import (
    avg_precision_at_threshold,
    avg_weighted_recall_at_threshold,
)
from src.models.ensemble import get_best_ensemble_matches
from src.models.utils import add_padding, remove_padding
from tqdm import tqdm

NYSIIS_SCORE = 0.6


# def get_sorted_similarities(embeddings, threshold=0.4, batch_size=1024):
#     rows = []
#     cols = []
#     scores = []
#     for ix in range(0, len(embeddings), batch_size):
#         batch = embeddings[ix:ix + batch_size]
#         batch_scores = cosine_similarity(batch, embeddings)
#         indices = np.where(batch_scores >= threshold)
#         scores.extend(batch_scores[indices])
#         batch_rows, batch_cols = indices
#         batch_rows += ix
#         rows.extend(batch_rows)
#         cols.extend(batch_cols)
#     score_sorted_ixs = np.array(scores).argsort()[::-1]
#     sorted_similarities = np.column_stack((rows, cols, scores))[score_sorted_ixs]
#     sorted_similarities = sorted_similarities[sorted_similarities[:, 0] < sorted_similarities[:, 1]]
#     return sorted_similarities
#
#
# def generate_closures(sorted_scores, closure_threshold):
#     id2closure = {}
#     closure2ids = {}
#     next_closure_id = 0
#     not_merged = 0
#     max_score_not_merged = 0
#     for id1, id2, score in sorted_scores:
#         id1 = int(id1)
#         id2 = int(id2)
#         if id1 == id2:
#             print(id1)
#             continue
#         if id1 not in id2closure and id2 not in id2closure:
#             closure_id = next_closure_id
#             next_closure_id += 1
#             id2closure[id1] = closure_id
#             id2closure[id2] = closure_id
#             closure2ids[closure_id] = {id1, id2}
#             continue
#         if id1 not in id2closure and id2 in id2closure:
#             id1, id2 = id2, id1
#         if id1 in id2closure and id2 not in id2closure:
#             closure_id = id2closure[id1]
#             id2closure[id2] = closure_id
#             closure2ids[closure_id].add(id2)
#             continue
#         if id1 in id2closure and id2 in id2closure:
#             closure_id = id2closure[id1]
#             closure2_id = id2closure[id2]
#             if closure_id == closure2_id:
#                 continue
#             # don't merge closures if the result would be too large
#             if len(closure2ids[closure_id]) + len(closure2ids[closure2_id]) > closure_threshold:
#                 if not_merged == 0:
#                     max_score_not_merged = score
#                 not_merged += 1
#                 continue
#             closure2ids[closure_id].update(closure2ids[closure2_id])
#             for _id in closure2ids[closure2_id]:
#                 id2closure[_id] = closure_id
#             del closure2ids[closure2_id]
#
#     return id2closure, closure2ids, not_merged, max_score_not_merged


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
    print("generating agglomerative clusters", datetime.now(), linkage, 1 - cluster_threshold)
    agg = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - cluster_threshold, affinity="precomputed", linkage=linkage
    )
    print("generating predictions", datetime.now())
    predictions = agg.fit_predict(distances)
    print("predictions generated", datetime.now())
    return agg, predictions


def _convert_neg_to_unique(labels):
    max_cluster = max(labels)
    results = []
    for label in labels:
        if label < 0:
            max_cluster += 1
            label = max_cluster
        results.append(label)
    return results


# def _generate_optics_clusters(min_samples, eps, max_eps, cluster_method, xi, n_jobs, distances):
#     """generate clusters using OPTICS"""
#     run_dbscan = False
#     if cluster_method == "xi+dbscan":
#         cluster_method = "xi"
#         run_dbscan = True
#     clust = OPTICS(min_samples=min_samples,
#                    cluster_method=cluster_method,
#                    xi=xi,
#                    max_eps=max_eps,
#                    metric="cosine",
#                    n_jobs=n_jobs,
#                    )
#     clust.fit(distances)  # TODO is this correct?
#
#     if run_dbscan:
#         labels = cluster_optics_dbscan(
#             reachability=clust.reachability_,
#             core_distances=clust.core_distances_,
#             ordering=clust.ordering_,
#             eps=eps,
#         )
#     else:
#         labels = clust.labels_
#
#     return _convert_neg_to_unique(labels)  # TODO is this correct?


# def _generate_hdbscan_clusters(min_samples, eps, selection_method, min_cluster_size, distances):
#     clust = hdbscan.HDBSCAN(min_samples=min_samples,
#                             cluster_selection_epsilon=eps,
#                             cluster_selection_method=selection_method,
#                             min_cluster_size=min_cluster_size,
#                             metric="euclidean",
#                             )
#     distances = normalize(distances)
#     clust.fit(distances)  # TODO is this correct?
#     return _convert_neg_to_unique(clust.labels_)


def generate_clusters(
    distances,
    cluster_algo="agglomerative",
    cluster_linkage="average",
    cluster_threshold=0.5,  # for agglomerative
    min_samples=2,
    eps=0.5,  # for optics or hdbscan
    max_eps=1.0,
    cluster_method="xi+dbscan",
    xi=0.05,  # for optics
    selection_method="eom",
    min_cluster_size=2,  # for hdbscan
    verbose=False,
    n_jobs=None,
):
    if cluster_algo == "agglomerative":
        model, results = _generate_agglomerative_clusters(cluster_threshold, cluster_linkage, distances)
    # elif cluster_algo == "optics":
    #     results = _generate_optics_clusters(min_samples, eps, max_eps, cluster_method, xi, n_jobs, distances)
    # else:
    #     results = _generate_hdbscan_clusters(min_samples, eps, selection_method, min_cluster_size, distances)
    return model, results


def get_clusters_from_table(names_to_cluster, clustered_names, verbose=True):
    name2clusters = defaultdict(list)
    cluster2names = defaultdict(list)
    if verbose:
        names_to_cluster = tqdm(names_to_cluster, mininterval=1.0)

    for name in names_to_cluster:
        if name not in clustered_names:
            if verbose:
                print(name)
            cluster = "CLUSTER_" + name
        else:
            cluster = clustered_names[name]
        name2clusters[name].append((cluster, 1.0))
        cluster2names[cluster].append(name)

    return name2clusters, cluster2names


def get_clusters(
    names_to_cluster,
    clustered_names,
    swivel_model,
    swivel_vocab,
    tfidf_vectorizer,
    ensemble_model,
    name_freq,
    max_clusters=5,
    batch_size=1024,
    k=5000,
    search_threshold=None,
    n_jobs=1,
    verbose=True,
):
    """
    For each name in all_names, find the closest clustered names and return a list of (cluster_id, cluster_score) tuples
    :param names_to_cluster: names to assign clusters
    :param clustered_names: clustered name to cluster id
    :param swivel_model: swivel model
    :param swivel_vocab: swivel vocabulary
    :param tfidf_vectorizer: tfidf vectorizer for ensemble model
    :param ensemble_model: ensemble model
    :param name_freq: map names to their frequencies
    :param max_clusters: maximum number of clusters to return for each input name
    :param batch_size:
    :param k: number of names to consider in the ensemble model and to get the best clusters
    :param search_threshold: threshold for searching the clusters
    :return: for each name in input_names, a list of up to max_clusters (cluster_id, cluster_score) tuples
    and also a dictionary mapping cluster id to the names for which that cluster is closest (that are assigned to the cluster)
    """
    name2clusters = defaultdict(list)
    cluster2names = defaultdict(list)

    if verbose:
        print("get_best_ensemble_matches", datetime.now())
    similar_names_scores = get_best_ensemble_matches(
        model=swivel_model,
        vocab=swivel_vocab,
        input_names=names_to_cluster,
        candidate_names=np.asarray(list(clustered_names.keys())),
        tfidf_vectorizer=tfidf_vectorizer,
        ensemble_model=ensemble_model,
        name_freq=name_freq,
        k=k,
        batch_size=batch_size,
        add_context=True,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    if verbose:
        print("get name2clusters", datetime.now())
    for name, names_scores in zip(names_to_cluster, similar_names_scores):
        # currently, never true, because len(names_scores) == k as a result of get_best_ensemble_matches
        if len(names_scores) > k * 4:
            partitioned_ixs = np.argpartition(names_scores[:, 1], -k, axis=0)[-k:]
            sorted_ixs = np.flip(
                np.argsort(np.take_along_axis(names_scores[:, 1], partitioned_ixs, axis=0), axis=0), axis=0
            )
            sorted_ixs = np.take_along_axis(partitioned_ixs, sorted_ixs, axis=0)
        else:
            sorted_ixs = np.flip(np.argsort(names_scores[:, 1], axis=0), axis=0)[:k]
        found_clusters = set()
        for sorted_ix in sorted_ixs:
            similar_name, similar_score = names_scores[sorted_ix]
            if len(found_clusters) > 0 and search_threshold is not None and similar_score < search_threshold:
                break
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


def get_best_cluster_matches(name2clusters, cluster2names, input_names, verbose=False):
    # return 3D array: [input name, candidate, (name, score)]
    if verbose:
        print("get cluster names and scores for input_names", datetime.now(), len(input_names))
        input_names = tqdm(input_names, mininterval=1.0)
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
    if verbose:
        print("expand the second dimension to", datetime.now(), max_cluster_names)
        cluster_names_scores = tqdm(cluster_names_scores, mininterval=1.0)
    all_cluster_names = []
    all_scores = []
    for cluster_names, cluster_scores in cluster_names_scores:
        if len(cluster_names) < max_cluster_names:
            cluster_names.extend([""] * (max_cluster_names - len(cluster_names)))
            cluster_scores.extend([-1.0] * (max_cluster_names - len(cluster_scores)))
        all_cluster_names.append(np.array(cluster_names, dtype=object))
        all_scores.append(np.array(cluster_scores, dtype=np.float32))

    del cluster_names_scores

    # return 3D array: [input name, candidate, (name, score)]
    if verbose:
        print("stack names and scores", datetime.now())
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


def prettify_cluster_names(name_cluster, pref_name2freq):
    cluster2names = defaultdict(list)
    for name, cluster_id in name_cluster.items():
        cluster2names[cluster_id].append(name)
    cluster2pretty = {}
    for cluster_id, names in cluster2names.items():
        cluster2pretty[cluster_id] = remove_padding(_get_most_frequent_name(names, pref_name2freq)).upper()
    return {name: cluster2pretty[cluster_id] for name, cluster_id in name_cluster.items()}


def write_clusters(path, name_cluster):
    df = pd.DataFrame([
        {"name": remove_padding(name), "cluster": cluster_id} for name, cluster_id in name_cluster.items()
    ])
    df.to_csv(path, index=False)


def read_clusters(path):
    df = pd.read_csv(path, na_filter=False)
    return {add_padding(name): cluster for name, cluster in zip(df["name"], df["cluster"])}


def write_cluster_scores(path, name2clusters):
    data = (
        "\n".join([
            json.dumps({
                "name": remove_padding(name),
                "clusters": [(cluster[0], float(cluster[1])) for cluster in clusters],
            })
            for name, clusters in name2clusters.items()
        ])
    ).encode("utf-8")
    with gzip.open(fopen(path, "wb"), "wb") as f:
        f.write(data)


def read_cluster_scores(path):
    with gzip.open(fopen(path, "rb"), "rb") as f:
        data = f.read()
    name2clusters = {}
    for line in data.decode("utf-8").split("\n"):
        name_clusters = json.loads(line)
        name2clusters[add_padding(name_clusters["name"])] = [
            (cluster[0], float(cluster[1])) for cluster in name_clusters["clusters"]
        ]
    return name2clusters


def get_names_to_cluster(name_freq, n_to_cluster):
    return np.array([name for name in name_freq if len(name) > 1][:n_to_cluster])


def get_distances(
    name_freq,
    names_to_cluster,
    swivel_model,
    swivel_vocab,
    tfidf_vectorizer,
    ensemble_model,
    verbose=False,
    num_matches=5000,
    batch_size=256,
    n_jobs=1,
):
    # get ensemble scores
    if verbose:
        print("get ensemble scores", datetime.now(), len(names_to_cluster))
    similar_names_scores = get_best_ensemble_matches(
        model=swivel_model,
        vocab=swivel_vocab,
        input_names=names_to_cluster,
        candidate_names=names_to_cluster,
        tfidf_vectorizer=tfidf_vectorizer,
        ensemble_model=ensemble_model,
        name_freq=name_freq,
        k=num_matches,
        batch_size=batch_size,
        add_context=True,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # create name to index dictionary
    name_index = {}
    for ix, name in enumerate(names_to_cluster):
        name_index[name] = ix

    # create distances array
    # names are initially 2.0 apart; similar names are 1.0 - score apart
    if verbose:
        print("create distances array", datetime.now(), len(names_to_cluster))
    distances = np.full((len(names_to_cluster), len(names_to_cluster)), 2.0, dtype=np.float32)
    for name1, names_scores in zip(names_to_cluster, similar_names_scores):
        name1_ix = name_index[name1]
        for name2, score in names_scores:
            name2_ix = name_index[name2]
            distances[name1_ix, name2_ix] = 1.0 - score
            distances[name2_ix, name1_ix] = 1.0 - score
        distances[name1_ix, name1_ix] = 0.0

    if verbose:
        print("return distances", datetime.now())
    return distances


def generate_clusters_from_distances(
    cluster_algo, cluster_linkage, cluster_threshold, distances, names_to_cluster, verbose=False, n_jobs=1
):
    # generate clusters from distances
    if verbose:
        print("generate clusters", datetime.now())
    model, clusters = generate_clusters(
        distances,
        cluster_algo=cluster_algo,
        # agglomerative options
        cluster_linkage=cluster_linkage,
        cluster_threshold=cluster_threshold,
        # other options
        n_jobs=n_jobs,
        verbose=verbose,
    )
    del distances

    # generate cluster->names and name->cluster
    cluster_names = defaultdict(list)
    name_cluster = {}
    max_cluster_size = 0
    max_cluster_id = None
    for _id, cluster in enumerate(clusters):
        clustered_name = names_to_cluster[_id]
        cluster_names[cluster].append(clustered_name)
        if len(cluster_names[cluster]) > max_cluster_size:
            max_cluster_size = len(cluster_names[cluster])
            max_cluster_id = cluster
        name_cluster[clustered_name] = cluster
    if verbose:
        print("number of clusters", datetime.now(), len(cluster_names))
        print("max cluster size", datetime.now(), max_cluster_size)
        print("max cluster", cluster_names[max_cluster_id])
        cluster_sizes_df = pd.DataFrame([len(names) for names in cluster_names.values()])
        cluster_sizes_df.hist(bins=100)

    return model, name_cluster


def get_validation_results(
    input_names_eval,
    weighted_actual_names_eval,
    candidate_names_eval,
    name_freq,
    name_cluster,
    swivel_model,
    swivel_vocab,
    tfidf_vectorizer,
    ensemble_model,
    search_threshold,
    max_clusters,
    num_matches=5000,
    lookup_mode=False,
    cluster_partition=None,
    n_jobs=1,
    verbose=False,
    sample_size=5000,
    validation_sizes=None,
):
    if validation_sizes is None:
        validation_sizes = [25000, 200000]
    search_thresholds = search_threshold if isinstance(search_threshold, list) else [search_threshold]

    precisions = []
    recalls = []
    avg_partitions = []
    f1s = []
    f2s = []
    all_f1s = []
    all_f2s = []
    for size in validation_sizes:
        if verbose:
            print("validate", datetime.now(), size)
        if size == 0:
            input_names_validate, weighted_actual_names_validate, candidate_names_validate = (
                input_names_eval,
                weighted_actual_names_eval,
                candidate_names_eval,
            )
        else:
            input_names_validate, weighted_actual_names_validate, candidate_names_validate = select_frequent_k(
                input_names_eval, weighted_actual_names_eval, candidate_names_eval, size
            )

        # sample the validation set
        if len(input_names_validate) > sample_size:
            _, input_names_validate, _, weighted_actual_names_validate = train_test_split(
                input_names_validate, weighted_actual_names_validate, test_size=sample_size
            )
            # filter the candidate names to just those in weighted actual names
            # however, we don't do this in notebook 90, so don't do it here either so the numbers a fair comparison
        #             candidate_names_validate = np.array(list(set(
        #                 name for wans in weighted_actual_names_validate for name, _, _ in wans)))

        # get validate names
        all_names_validate = list(set(input_names_validate).union(set(candidate_names_validate)))

        # assign all names to clusters
        if verbose:
            print("get_clusters", datetime.now(), len(all_names_validate))
        if lookup_mode:
            name2clusters, cluster2names = get_clusters_from_table(all_names_validate, name_cluster, verbose=verbose)
        else:
            name2clusters, cluster2names = get_clusters(
                all_names_validate,
                name_cluster,
                swivel_model,
                swivel_vocab,
                tfidf_vectorizer,
                ensemble_model,
                name_freq,
                max_clusters=max_clusters,
                k=num_matches,
                search_threshold=min(search_thresholds),
                n_jobs=n_jobs,
                verbose=verbose,
            )

        #         print("name2clusters", len(name2clusters),
        #               min(len(clusters) for clusters in name2clusters.values()),
        #               max(len(clusters) for clusters in name2clusters.values()))
        #         print("cluster2names", len(cluster2names), \
        #               min(len(names) for names in cluster2names.values()),
        #               max(len(names) for names in cluster2names.values()))
        #         print("maria cluster", name2clusters["<maria>"])
        #         print("maria cluster names", cluster2names[name2clusters["<maria>"][0][0]])
        #         for input_name, wans in zip(input_names_validate, weighted_actual_names_validate):
        #             if input_name != "<maria>":
        #                 continue
        #             print("maria weighted actual names", wans)

        # get best matches
        if verbose:
            print("get_best_cluster_matches", datetime.now())
        best_matches = get_best_cluster_matches(name2clusters, cluster2names, input_names_validate, verbose=verbose)

        # eval f1
        precision_map = {}
        recall_map = {}
        avg_partition_map = {}
        f1_map = {}
        f2_map = {}
        for threshold in search_thresholds:
            # count n_partitions for each distinct partition in cluster_partition for clusters in name2clusters above threshold
            total_partitions = 0
            if cluster_partition is not None:
                for input_name in input_names_validate:
                    partitions = {}
                    for cluster_id, cluster_score in name2clusters[input_name]:
                        if cluster_score >= threshold:
                            start_partition, n_partitions = cluster_partition[cluster_id]
                            partitions[start_partition] = n_partitions
                    total_partitions += sum(partitions.values())  # sum n_partitions for each distinct partition
            avg_partition_lookups = total_partitions / len(input_names_validate)
            # get precision and recall
            precision = avg_precision_at_threshold(weighted_actual_names_validate, best_matches, threshold)
            recall = avg_weighted_recall_at_threshold(weighted_actual_names_validate, best_matches, threshold)
            f1 = 2 * (precision * recall) / (precision + recall)
            f2 = 5 * (precision * recall) / (4 * precision + recall)
            if verbose:
                print(
                    "result",
                    datetime.now(),
                    "threshold",
                    threshold,
                    "precision",
                    precision,
                    "recall",
                    recall,
                    "f1",
                    f1,
                    "f2",
                    f2,
                    "avg_partitions",
                    avg_partition_lookups,
                )
            precision_map[threshold] = precision
            recall_map[threshold] = recall
            avg_partition_map[threshold] = avg_partition_lookups
            f1_map[threshold] = f1
            f2_map[threshold] = f2
            all_f1s.append(f1)
            all_f2s.append(f2)

        precisions.append(precision_map)
        recalls.append(recall_map)
        avg_partitions.append(avg_partition_map)
        f1s.append(f1_map)
        f2s.append(f2_map)

    f1 = (sum(all_f1s) / len(all_f1s)) if len(all_f1s) > 0 else 0
    f2 = (sum(all_f2s) / len(all_f2s)) if len(all_f2s) > 0 else 0

    return {
        "f1": f1,
        "f2": f2,
        "f1s": f1s,
        "f2s": f2s,
        "precisions": precisions,
        "recalls": recalls,
        "avg_partitions": avg_partitions,
    }
