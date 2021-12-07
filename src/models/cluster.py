from collections import defaultdict

from mpire import WorkerPool
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from tqdm import tqdm

from src.models.utils import remove_padding


def get_scores(embeddings, threshold=0.4, batch_size=256):
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
    scores_sparse = csr_matrix((scores, (rows, cols)), shape=(len(embeddings), len(embeddings)), dtype=np.float32)
    score_sorted_ixs = np.array(scores).argsort()[::-1]
    sorted_scores = np.column_stack((rows, cols, scores))[score_sorted_ixs]
    sorted_scores = sorted_scores[sorted_scores[:, 0] < sorted_scores[:, 1]]
    return scores_sparse, sorted_scores


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


def _generate_clusters(shared, closure_id, enclosed_ids, closure_scores):
    cluster_threshold, linkage = shared
    # reshape data
    X = np.arange(len(enclosed_ids)).reshape(-1, 1).astype("int32")
    # precompute distances
    distances = pairwise_distances(X, metric=_dist_metric(closure_scores))
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - cluster_threshold,
                                  affinity="precomputed", linkage=linkage)
    cluster_assignments = agg.fit_predict(distances)
    return closure_id, enclosed_ids, cluster_assignments


def generate_clusters(closure2ids, clustered_scores_sparse, clustered_names, cluster_threshold, cluster_linkage,
                      verbose=True, n_jobs=None):
    id2cluster = {}
    # calculate parameters
    params_list = []
    for closure_id, ids in closure2ids.items():
        enclosed_ids = list(ids)
        closure_scores = clustered_scores_sparse[enclosed_ids][:, enclosed_ids]
        params_list.append((closure_id, enclosed_ids, closure_scores))
    if n_jobs == 1:
        results_list = []
        if verbose:
            params_list = tqdm(params_list)
        for params in params_list:
            results_list.append(_generate_clusters((cluster_threshold, cluster_linkage), *params))
    else:
        with WorkerPool(shared_objects=(cluster_threshold, cluster_linkage), n_jobs=n_jobs) as pool:
            results_list = pool.map_unordered(_generate_clusters, params_list, progress_bar=verbose)

    # process results
    for results in results_list:
        closure_id, enclosed_ids, cluster_assignments = results
        for _id, cluster_id in zip(enclosed_ids, cluster_assignments):
            id2cluster[_id] = f"{closure_id}_{cluster_id}"

    # assign clustered_names not in any cluster to their own (singleton) cluster
    for _id in range(0, len(clustered_names)):
        if _id not in id2cluster:
            id2cluster[_id] = f"{_id}"

    return id2cluster


def assign_names_to_clusters(all_names, all_embeddings, id2cluster, clustered_embeddings, batch_size=256, k=1, verbose=True):
    """
    Find the closest clustered name for each name in all_names and assign the name to the cluster for
    the closest clustered name
    :param all_names: all names
    :param all_embeddings: embeddings for names to assign
    :param id2cluster: clustered name id to cluster id
    :param clustered_embeddings: embeddings for clustered names
    :param batch_size:
    :param k:
    :return:
    """
    name2cluster = {}
    cluster2names = defaultdict(list)
    ixs = range(0, len(all_embeddings), batch_size)
    if verbose:
        ixs = tqdm(ixs)
    for ix in ixs:
        batch = all_embeddings[ix:ix + batch_size]
        scores = cosine_similarity(batch, clustered_embeddings)
        # closest_clustered_ids = np.argmax(scores, axis=1)
        closest_clustered_ids = np.argpartition(scores, -k)[:, -k:]
        for _id, closest_ids in zip(list(range(ix, ix + batch_size)), closest_clustered_ids):
            name = all_names[_id]
            cluster_ids = [id2cluster[closest_clustered_id] for closest_clustered_id in closest_ids]
            # get the mode (most-common) cluster id
            cluster_id = max(set(cluster_ids), key=cluster_ids.count)
            name2cluster[name] = cluster_id
            cluster2names[cluster_id].append(name)
    return name2cluster, cluster2names


def get_best_cluster_matches(name2cluster, cluster2names, input_names, k=256):
    # return [input names, candidates, (names, scores)]
    all_cluster_names = []
    all_scores = []
    for input_name in input_names:
        cluster_id = name2cluster[input_name]
        cluster_names = np.array(cluster2names[cluster_id][0:k], dtype="object")
        scores = np.ones(min(k, len(cluster_names)), dtype="int8")
        if len(cluster_names) < k:
            cluster_names = np.pad(cluster_names, (0, k-len(cluster_names)), constant_values="")
            scores = np.pad(scores, (0, k-len(scores)), constant_values=0)
        all_cluster_names.append(cluster_names)
        all_scores.append(scores)
    all_cluster_names = np.vstack(all_cluster_names)
    all_scores = np.vstack(all_scores)
    return np.dstack((all_cluster_names, all_scores))


def write_clusters(path, clustered_names, id2cluster):
    name2cluster = {remove_padding(clustered_names[id]): cluster_id for id, cluster_id in id2cluster.items()}
    df = pd.DataFrame.from_dict(name2cluster, orient="index", columns=["cluster_id"])
    df.to_csv(path, index_label="name")
