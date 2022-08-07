import numpy as np
import pandas as pd
import numba
from sklearn.neighbors import NearestNeighbors as NearestNeighborsCPU
from collections import defaultdict
from tqdm import tqdm

from foursquare.utils.ml import score_iou

try:
    from cuml.neighbors import NearestNeighbors as NearestNeighborsDefault
    print("imported cuml.neighbors.NearestNeighbors")
except ImportError:
    from sklearn.neighbors import NearestNeighbors as NearestNeighborsDefault
    print("imported sklearn.neighbors.NearestNeighborsDefault")


def build_matches(df_locations):
    df_locations = df_locations[["id", "point_of_interest"]]
    df = df_locations.merge(
        df_locations, on="point_of_interest", suffixes=("_1", "_2"))
    df = df.groupby("id_1")["id_2"].apply(list)
    df = df.loc[df_locations["id"]]
    df = df.reset_index()
    df.columns = ["id", "matches"]
    return df


class MatchesPredictor:
    def __init__(self, n_neighbors, force_cpu=False):
        self.n_neighbors = n_neighbors
        self.force_cpu = force_cpu
        if self.force_cpu:
            self.nn_cls = NearestNeighborsCPU
        else:
            self.nn_cls = NearestNeighborsDefault
        self.indices = None
        self.similarities = None

    def fit(self, x):
        nn = self.nn_cls(n_neighbors=self.n_neighbors, metric="cosine")
        nn.fit(x)
        distances, indices = nn.kneighbors(x)
        self.indices = indices
        self.similarities = 1 - distances

    def get_neighbors(self, threshold):
        return group_indices(self.similarities, self.indices, threshold)


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def normalize_embedding(embedding, dims):
    idx_1 = 0
    embedding_norm = list()
    for idx_2 in np.cumsum(dims):
        emb_norm = normalize(embedding[:, idx_1:idx_2])
        embedding_norm.append(emb_norm)
        idx_1 = idx_2
    embedding_norm = np.hstack(embedding_norm)
    return normalize(embedding_norm)


class MatchesValidator:
    def __init__(self, max_neighbors, blending_stages=0, force_cpu=False, qe=False, verbose=False):
        self.max_neighbors = max_neighbors
        self.blending_stages = blending_stages
        self.predictors = [
            MatchesPredictor(n_neighbors=max_neighbors, force_cpu=force_cpu)
            for _ in range(blending_stages + 1)
        ]
        self.force_cpu = force_cpu
        self.qe = qe
        self.verbose = verbose

    def validate(self, embs, df_true, dims_embedding, return_thresholds=False):
        scores = list()
        thresholds = list()
        for idx, predictor in enumerate(self.predictors):
            embs = normalize_embedding(embs, dims_embedding)

            if self.verbose:
                print(f"[stage={idx}] matching...")

            predictor.fit(embs)
            threshold, score = self._search_threshold(predictor, df_true)
            thresholds.append(threshold)
            scores.append(score)

            if self.verbose:
                print(f"[stage={idx}] score: {score:.4f} at threshold "
                      f"{threshold:.4f}")
                print(f"[stage={idx}] matching, done!")

            if idx < self.blending_stages:
                indices, similarities = predictor.get_neighbors(threshold)
                embs = blend_neighbors(embs, similarities, indices)

        if self.qe:
            indices, similarities = predictor.get_neighbors(threshold)
            indices = [ind.tolist() for ind in indices]
            indices = query_expansion(indices)
            df_pred = df_true.copy()
            indices_map = df_true["id"].values
            df_pred["matches"] = indices
            df_pred["matches"] = df_pred["matches"].apply(lambda ind: indices_map[ind])
            score = score_iou(df_true, df_pred)
            print(f"Query expansion: {score:.4f}")

        if return_thresholds:
            return max(scores), thresholds
        return max(scores)

    def _search_threshold(self, predictor, df_true):
        df_pred = df_true.copy()
        indices_map = df_true["id"].values
        scores, thresholds = list(), list()
        for threshold in np.linspace(0.3, 0.95, 30):
            indices, similarities = predictor.get_neighbors(threshold)
            df_pred["matches"] = indices
            df_pred["matches"] = df_pred["matches"].apply(lambda ind: indices_map[ind])
            score = score_iou(df_true, df_pred)
            if self.verbose:
                print(f"score {score:.4f} at threshold {threshold:.4f}")
            scores.append(score)
            thresholds.append(threshold)
        idx_max = scores.index(max(scores))
        return thresholds[idx_max], scores[idx_max]


class MatchesBlendingPredictor:
    def __init__(
            self,
            max_neighbors,
            blending_stages=0,
            threshold=0.5,
            qe=False,
            force_cpu=False,
            verbose=False):

        if not isinstance(threshold, list):
            threshold = [threshold]
        if not len(threshold) == blending_stages + 1:
            raise ValueError(
                "len of threshold must be equal to blending_stages + 1."
                f" threshold='{threshold}' and "
                f"blending_stages='{blending_stages}'")

        self.max_neighbors = max_neighbors
        self.blending_stages = blending_stages
        self.threshold = threshold
        self.force_cpu = force_cpu
        self.qe = qe
        self.verbose = verbose

    def predict(self, embs, embs_dims):
        indices = None
        for idx in range(self.blending_stages + 1):
            predictor = MatchesPredictor(
                n_neighbors=self.max_neighbors, force_cpu=self.force_cpu)
            embs = normalize_embedding(embs, embs_dims)
            predictor, threshold = predictor, self.threshold[idx]

            predictor.fit(embs)
            indices, similarities = predictor.get_neighbors(threshold)
            if idx < self.blending_stages:
                embs = blend_neighbors(embs, similarities, indices)

        if self.qe:
            indices, similarities = predictor.get_neighbors(threshold)
            indices = query_expansion([ind.tolist() for ind in indices])

        return indices


class MatchesSimplePredictor:
    def __init__(self, max_neighbors, threshold):
        self.max_neighbors = max_neighbors
        self.threshold = threshold

    def predict(self, x):
        nn = NearestNeighborsCPU(n_neighbors=self.max_neighbors)
        nn.fit(x)
        distances, indices = nn.kneighbors(x)
        indices, distances = group_indices(
            distances, indices, self.threshold, distance=True)
        return indices


@numba.jit(nopython=True)
def group_indices(similarities, indices, threshold, distance=False):
    neigh_indices = list()
    neigh_similarities = list()
    if not distance:
        matches = similarities > threshold
    else:
        matches = similarities < threshold
    for k in range(matches.shape[0]):
        idx = np.where(matches[k])
        ids = indices[k]
        ids = ids[idx]
        sim = similarities[k]
        sim = sim[idx]
        neigh_indices.append(ids)
        neigh_similarities.append(sim)
    return neigh_indices, neigh_similarities


def blend_neighbors(embs, similarities, indices):
    embs_blended = np.zeros(embs.shape, dtype="float16")
    for idx in range(len(embs)):
        neigh_sim = similarities[idx].astype("float32")
        neigh_embs = embs[indices[idx]].astype("float32")
        blend = neigh_embs * np.expand_dims(neigh_sim, -1)
        embs_blended[idx] = blend.sum(axis=0)
    return embs_blended


def blend_embeddings(embs, threshold):
    predictor = MatchesPredictor(n_neighbors=min(30, len(embs)))
    predictor.fit(embs)
    indices, similarities = predictor.get_neighbors(threshold)
    embs = blend_neighbors(embs, similarities, indices)
    return embs


def query_expansion(indices):
    for indices_1 in tqdm(indices):
        for ind_1 in indices_1:
            indices_2 = indices[ind_1]
            for ind_2 in indices_2:
                if ind_2 not in indices_1:
                    indices_1.append(ind_2)
    return indices


def validate_pairs_pred(pred, pairs, ids_locations, df_true, verbose=False):
    scores, thresholds = list(), list()
    for threshold in np.linspace(0.3, 0.7, 20):
        df_pred = postprocess_pairs_pred(pred, pairs, ids_locations, threshold)
        score = score_iou(df_true, df_pred)
        scores.append(score)
        thresholds.append(threshold)
        if verbose:
            print(f"score {score:.4f} at threshold {threshold:.4f}")
    idx_max = scores.index(max(scores))
    return thresholds[idx_max], scores[idx_max]


def postprocess_pairs_pred(pred, pairs, ids_locations, threshold):
    # matching based on pairs predicion prob
    matches = defaultdict(list)
    for idx in range(len(pred)):
        if pred[idx] > threshold:
            id_1, id_2 = pairs[idx]
            matches[id_1].append(id_2)
            matches[id_2].append(id_1)

    # complete predicions with same id
    for id_loc in ids_locations:
        matches[id_loc].append(id_loc)

    df_matches = pd.DataFrame(matches.items(), columns=["id", "matches"])
    return df_matches
