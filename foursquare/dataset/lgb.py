import re

import pandas as pd
import numpy as np
from thefuzz import fuzz
import Levenshtein
from unidecode import unidecode
import haversine

from foursquare.utils.matching import (
    MatchesBlendingPredictor,
    MatchesSimplePredictor,
    normalize_embedding,
    blend_embeddings
)


def distance(fn):
    def dist(x1, x2, **kwargs):
        if x1 is np.nan and x2 is np.nan:
            return DiffTextFeatureExtractor.NULL_DISTANCE_BOTH
        elif x1 is np.nan or x2 is np.nan:
            return DiffTextFeatureExtractor.NULL_DISTANCE_ONE
        else:
            if not isinstance(x1, str):
                x1 = str(x1)
            if not isinstance(x2, str):
                x2 = str(x2)
            return fn(x1, x2, **kwargs)
    return dist


def similarity(fn):
    def sim(x1, x2, **kwargs):
        if x1 is np.nan and x2 is np.nan:
            return DiffTextFeatureExtractor.NULL_SIMILARITY_BOTH
        elif x1 is np.nan or x2 is np.nan:
            return DiffTextFeatureExtractor.NULL_SIMILARITY_ONE
        else:
            if not isinstance(x1, str):
                x1 = str(x1)
            if not isinstance(x2, str):
                x2 = str(x2)
            return fn(x1, x2, **kwargs)
    return sim


class DiffTextFeatureExtractor:
    NULL_DISTANCE_ONE = 555555
    NULL_DISTANCE_BOTH = 999999
    NULL_SIMILARITY_ONE = -1
    NULL_SIMILARITY_BOTH = -2

    @staticmethod
    @distance
    def len_diff(x1, x2):
        return abs(len(x1) - len(x2))

    @staticmethod
    @distance
    def len_token_diff(x1, x2):
        return abs(len(x1.split(" ")) - len(x2.split(" ")))

    @staticmethod
    @distance
    def levenshtein(x1, x2):
        return Levenshtein.distance(x1, x2)

    @staticmethod
    @similarity
    def jaro_winkler(x1, x2):
        return Levenshtein.jaro_winkler(x1, x2)

    @staticmethod
    @similarity
    def fuzz_ratio(x1, x2):
        return fuzz.ratio(x1, x2)

    @staticmethod
    @similarity
    def fuzz_partial_ratio(x1, x2):
        return fuzz.partial_ratio(x1, x2)

    @staticmethod
    @similarity
    def fuzz_token_sort_ratio(x1, x2):
        return fuzz.token_sort_ratio(x1, x2, full_process=False)

    @staticmethod
    @similarity
    def fuzz_token_set_ratio(x1, x2):
        return fuzz.token_set_ratio(x1, x2, full_process=False)

    @staticmethod
    @similarity
    def ngrams_set_ratio(x1, x2, n):
        ngrams1 = {x1[i:i + n] for i in range(len(x1) - n + 1)}
        ngrams2 = {x2[i:i + n] for i in range(len(x2) - n + 1)}
        return len(ngrams1 & ngrams2) / (len(ngrams1 | ngrams2) + 1)

    @staticmethod
    @similarity
    def ngrams_ratio(x1, x2, n):
        ngrams1 = [x1[i:i + n] for i in range(len(x1) - n + 1)]
        ngrams2 = [x2[i:i + n] for i in range(len(x2) - n + 1)]
        cnt1 = len([ngram for ngram in ngrams1 if ngram in ngrams2])
        cnt2 = len([ngram for ngram in ngrams2 if ngram in ngrams1])
        return (cnt1 + cnt2) / (len(ngrams1) + len(ngrams2) + 1)


def extract_distance_text_features(x, features_text):
    features = dict()
    fe = DiffTextFeatureExtractor()

    for f in features_text:
        f1, f2 = x[f"{f}_1"], x[f"{f}_2"]
        features[f"{f}_len_diff"] = fe.len_diff(f1, f2)
        features[f"{f}_len_token_diff"] = fe.len_token_diff(f1, f2)
        features[f"{f}_levenshtein"] = fe.levenshtein(f1, f2)
        features[f"{f}_jaro_winkler"] = fe.jaro_winkler(f1, f2)
        features[f"{f}_fuzz_ratio"] = fe.fuzz_ratio(f1, f2)
        features[f"{f}_fuzz_partial_ratio"] = fe.fuzz_partial_ratio(f1, f2)
        features[f"{f}_fuzz_token_sort_ratio"] = fe.fuzz_token_sort_ratio(f1, f2)
        features[f"{f}_fuzz_token_set_ratio"] = fe.fuzz_token_set_ratio(f1, f2)
        features[f"{f}_2_gram_set_ratio"] = fe.ngrams_set_ratio(f1, f2, n=2)
        features[f"{f}_3_gram_set_ratio"] = fe.ngrams_set_ratio(f1, f2, n=3)
        features[f"{f}_4_gram_set_ratio"] = fe.ngrams_set_ratio(f1, f2, n=4)
        features[f"{f}_2_gram_ratio"] = fe.ngrams_ratio(f1, f2, n=2)
        features[f"{f}_3_gram_ratio"] = fe.ngrams_ratio(f1, f2, n=3)
        features[f"{f}_4_gram_ratio"] = fe.ngrams_ratio(f1, f2, n=4)

    return features


def normalize_str_series(series):
    def normalize_unidecode(x):
        return unidecode(x) if x is not np.nan else x
    series = series.apply(normalize_unidecode)
    return series


def clean_str_series(series):
    series = series.str.strip()
    series = series.str.lower()
    return series


def extract_embedding_cosine(pair, embs, embs_dims=None):
    def norm(emb):
        return emb / np.linalg.norm(emb)

    embs_dims = [embs.shape[1]] if embs_dims is None else embs_dims

    features = dict()

    indice_1, indice_2 = pair["indice_1"], pair["indice_2"]
    emb_1, emb_2 = embs[indice_1], embs[indice_2]

    emb_1 = emb_1.astype("float32")
    emb_2 = emb_2.astype("float32")

    # global embedding cosine similarity
    embs_pair = normalize_embedding(np.vstack([emb_1, emb_2]), embs_dims)
    emb_norm_1, emb_norm_2 = embs_pair[0], embs_pair[1]
    features["sim_cosine"] = sum(emb_norm_1 * emb_norm_2)

    if len(embs_dims) > 1:
        # partial embedding cosine similarity
        idx_1 = 0
        for i, idx_2 in enumerate(np.cumsum(embs_dims)):
            features[f"sim_cosine_{i}"] = sum(
                norm(emb_1[idx_1:idx_2]) * norm(emb_2[idx_1:idx_2]))
            idx_1 = idx_2

    return features


def build_pairs(
        embs,
        df_locations,
        embs_dims,
        max_neighbors,
        threshold,
        blending_stages,
        qe,
        add_cosine_distance=True,
        test=False):

    predictor = MatchesBlendingPredictor(
        max_neighbors=min(max_neighbors, len(embs)),
        threshold=threshold,
        blending_stages=blending_stages,
        qe=qe,
        verbose=True)
    matches_emb = predictor.predict(embs, embs_dims)

    predictor = MatchesSimplePredictor(
        max_neighbors=min(200, len(df_locations)),
        threshold=1e-4)
    matches_dist = predictor.predict(
        df_locations[["latitude", "longitude"]].values)

    matches = matches_emb + matches_dist

    pairs = list()
    for match in matches:
        for i in range(1, len(match)):
            if match[0] < match[i]:
                pair = [match[0], match[i]]
            else:
                pair = [match[i], match[0]]
            pairs.append(pair)
    df = pd.DataFrame(pairs, columns=["indice_1", "indice_2"])
    df = df.drop_duplicates()

    # decode indices
    indices_map = df_locations["id"].values
    df["id_1"] = df["indice_1"].apply(lambda ind: indices_map[ind])
    df["id_2"] = df["indice_2"].apply(lambda ind: indices_map[ind])

    if add_cosine_distance:
        df_cosine_dist = df[["indice_1", "indice_2"]].apply(
            extract_embedding_cosine,
            axis=1,
            result_type="expand",
            embs=embs,
            embs_dims=embs_dims)
        df = pd.concat([df, df_cosine_dist], axis=1)

    if not test:
        df = df.merge(
            df_locations[["id", "point_of_interest"]],
            left_on="id_1",
            right_on="id")
        df = df.rename(columns={"point_of_interest": "point_of_interest_1"}) \
            .drop(["id"], axis=1)
        df = df.merge(
            df_locations[["id", "point_of_interest"]],
            left_on="id_2",
            right_on="id")
        df = df.rename(columns={"point_of_interest": "point_of_interest_2"}) \
            .drop(["id"], axis=1)

        df["match"] = df["point_of_interest_1"] == df["point_of_interest_2"]
        df["match"] = df["match"].astype("int32")
        df = df.drop(["point_of_interest_1", "point_of_interest_2"], axis=1)

    return df


def build_pairs_dataset(df_pairs, df_locations):
    features = [
        col for col in df_pairs.columns
        if col not in ["id_1", "id_2", "fold", "match", "indice_1", "indice_2"]]
    print(f"initial features: {features}")

    # features: text difference
    features_text = [
        "name", "address", "phone", "url", "city", "state", "zip", "categories"
    ]
    for f in features_text:
        df_pairs[f"{f}_1"] = normalize_str_series(df_pairs[f"{f}_1"])
        df_pairs[f"{f}_2"] = normalize_str_series(df_pairs[f"{f}_2"])

    df_distance_text_features = df_pairs.apply(
        extract_distance_text_features,
        axis=1,
        result_type="expand",
        features_text=features_text)
    df_distance_text_features = df_distance_text_features.astype("float32")
    df_pairs = df_pairs.drop([f"{f}_1" for f in features_text], axis=1)
    df_pairs = df_pairs.drop([f"{f}_2" for f in features_text], axis=1)
    df_pairs = pd.concat([df_pairs, df_distance_text_features], axis=1)
    features.extend(df_distance_text_features.columns.tolist())

    df_pairs["country_equal"] = (
            df_pairs["country_1"] == df_pairs["country_2"]).astype("int32")
    df_pairs = df_pairs.drop(["country_1", "country_2"], axis=1)
    features.append("country_equal")

    df_pairs["latlon_euclidean"] = np.sqrt(
        (df_pairs["latitude_1"] - df_pairs["latitude_2"]) ** 2
        + (df_pairs["longitude_1"] - df_pairs["longitude_2"]) ** 2)
    df_pairs["latlon_manhattan"] = \
        abs(df_pairs["latitude_1"] - df_pairs["latitude_2"]) \
        + abs(df_pairs["longitude_1"] - df_pairs["longitude_2"])
    df_pairs["latlon_haversine"] = \
        df_pairs[["latitude_1", "longitude_1", "latitude_2", "longitude_2"]] \
        .apply(lambda x: haversine.haversine(
                (x["latitude_1"], x["longitude_1"]),
                (x["latitude_2"], x["longitude_2"])), axis=1)

    features.extend([
        "latlon_euclidean",
        "latlon_manhattan",
        "latlon_haversine",
        "longitude_1",
        "longitude_2",
        "latitude_1",
        "latitude_2"])

    cols_no_drop = features + ["id_1", "id_2", "fold", "match"]
    cols_drop = [col for col in df_pairs.columns if col not in cols_no_drop]
    df_pairs = df_pairs.drop(cols_drop, axis=1)

    return df_pairs, features


def merge_pairs(df_pairs, embeddings, embs_dims):
    df_distances = list()
    for emb in embeddings:
        df_dist = df_pairs[["indice_1", "indice_2"]].apply(
            extract_embedding_cosine,
            axis=1,
            result_type="expand",
            embs=emb,
            embs_dims=embs_dims)
        df_distances.append(df_dist)
    df_distances = sum(df_distances) / len(df_distances)
    df_pairs = pd.concat([df_pairs, df_distances], axis=1)
    return df_pairs


def add_extra_embedding(df_pairs, embs, name, blending=False):
    if blending:
        name += "_blend"
        for idx in range(len(embs)):
            embs[idx] = blend_embeddings(embs[idx], 0.65)

    df_distances = list()
    for emb in embs:
        df_dist = df_pairs[["indice_1", "indice_2"]].apply(
            extract_embedding_cosine,
            axis=1,
            result_type="expand",
            embs=emb,
            embs_dims=None)
        df_distances.append(df_dist)
    df_distances = sum(df_distances) / len(df_distances)
    df_distances.columns = [f"sim_cosine_{name}"]
    df_pairs = pd.concat([df_pairs, df_distances], axis=1)
    return df_pairs


def add_embedding_similarity_features(df):
    features = []
    columns_sim = sorted(
        [col for col in df.columns if re.match("sim_cosine.*", col)])

    for idx, col in enumerate(columns_sim):
        df[f"sim{idx}_max_1"] = df.groupby("id_1")[col].transform("max")
        df[f"sim{idx}_max_2"] = df.groupby("id_2")[col].transform("max")
        df[f"diff_sim{idx}_max_1"] = df[f"sim{idx}_max_1"] - df[col]
        df[f"diff_sim{idx}_max_2"] = df[f"sim{idx}_max_2"] - df[col]
        df[f"sim{idx}_rank_1"] = df.groupby("id_1")[col].rank()
        df[f"sim{idx}_rank_2"] = df.groupby("id_2")[col].rank()

        features.extend([
            f"sim{idx}_max_1",
            f"sim{idx}_max_2",
            f"diff_sim{idx}_max_1",
            f"diff_sim{idx}_max_2",
            f"sim{idx}_rank_1",
            f"sim{idx}_rank_2"
        ])

        df[[f"sim{idx}_rank_1", f"sim{idx}_rank_2"]] = df[
            [f"sim{idx}_rank_1", f"sim{idx}_rank_2"]].rank(pct=True)

    return features
