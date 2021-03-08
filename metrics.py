import os
import pandas
import numpy as np
from parastash import models as mds

from typing import Iterable
from scipy.spatial import distance


def _file_basename2uid(path):
    """
    /path/to/image/o_img-0008bf9d613ddfd03be85c85403e24e6-index.jpeg
    """
    filename = os.path.basename(path)
    return filename[len('o_img-'):filename.rfind('-index.jpeg')]


def labeled_collections_precision_recall_at_k(embeddings, dataset, collections: pandas.DataFrame):
    """
    Compute precision and recall at top-K for each collection.

    One entity will be randomly sampled for each collection. Each retrieved entity counted as a hit if it also belongs
    to that collection.
    """

    assert isinstance(embeddings, list), "not a list"
    ks = [1, 5, 10, 50]
    knn = NearestNeighbours(embeddings)

    uid_list = [_file_basename2uid(s) for s, _ in dataset.samples]
    uid_to_idx = {uid: idx for idx, uid in enumerate(uid_list)}

    stats = {
        k: {"total_hits": 0, "total_possible_precision": 0, "total_possible_recall": 0}
        for k in ks
    }

    max_k = max(ks)
    for _, test_indices, test_choice_idx in _get_test_choices_iter(collections, uid_to_idx):
        nn_list, _ = knn.get_neighbours_with_idx(test_choice_idx, max_k)
        nn_list = nn_list[0, :]  # make it hashable :)

        for k, counts in stats.items():
            nearest = nn_list[:k]
            counts["total_hits"] += len(set(nearest) & test_indices)
            counts["total_possible_precision"] += min(len(test_indices), k)
            counts["total_possible_recall"] += len(test_indices)

    for counts in stats.values():
        counts["precision"] = counts["total_hits"] / counts["total_possible_precision"]
        counts["recall"] = counts["total_hits"] / counts["total_possible_recall"]

    return stats


def _get_test_choices_iter(df, uid2idx):
    is_in_dataset = df["uid"].isin(set(uid2idx.keys()))
    collections_ = df[is_in_dataset].groupby('collection_id')['uid']
    for collection_id, collection_uuids in collections_:
        uuids = collection_uuids.unique()
        random_idx = uid2idx[np.random.choice(uuids)]
        ids = set([uid2idx[id_] for id_ in uuids])
        ids.remove(random_idx)
        if len(ids) > 0:
            yield collection_id, ids, random_idx


class NearestNeighbours:
    def __init__(self, embeddings, metric="cosine", **kwargs):
        # embedding is the (1,128) shape, but  nearest-neighbors search requires ndim data types
        self.embeddings = np.vstack(embeddings)
        self.metric_name = metric

    def get_neighbours_with_idx(self, idx_list, n):
        if not isinstance(idx_list, Iterable):
            idx_list = [idx_list]

        index_range = np.arange(0, len(idx_list))
        emb_to_find = self.embeddings[idx_list]
        distances = distance.cdist(emb_to_find, self.embeddings, self.metric_name)
        nearest_idx = np.argsort(distances, axis=1)[:, :n]
        nearest_distances = distances[
            np.repeat(index_range, n).reshape(nearest_idx.shape), nearest_idx
        ]
        return nearest_idx, nearest_distances
