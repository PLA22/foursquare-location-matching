import argparse
import logging
import os
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

from foursquare.utils.generic import fix_seeds, initialize_logger
from foursquare.utils.io import load_config, load_locations
from foursquare.dataset.lgb import (
    build_pairs_dataset,
    build_pairs,
    add_extra_embedding,
    add_embedding_similarity_features
)
from foursquare.utils.matching import (
    validate_pairs_pred,
    postprocess_pairs_pred,
    score_iou,
    build_matches
)

logger = logging.getLogger('foursquare')
pd.options.mode.chained_assignment = None


def max_iou(df_pairs, df_locations):
    df_pred_fold_maxiou = postprocess_pairs_pred(
        pred=df_pairs.loc[df_pairs["match"] == 1, "match"].values,
        pairs=df_pairs.loc[df_pairs["match"] == 1, ["id_1", "id_2"]].values,
        ids_locations=df_locations["id"].values,
        threshold=0.5)
    df_true_fold = build_matches(df_locations)
    score = score_iou(df_true_fold, df_pred_fold_maxiou)
    return score


def load_embeddings(dir_embeddings, embedding_names, sample_size=None):
    logger.info("loading embeddings...")

    embeddings = dict()

    for dir_emb in sorted(dir_embeddings):
        name_emb = dir_emb.stem
        if name_emb not in embedding_names:
            continue

        for dir_fold_emb in sorted(dir_emb.glob("*")):
            logger.info(f"loading embeddings from {dir_fold_emb}...")
            emb = np.load(dir_fold_emb / "valid_embeddings.npy")
            emb = emb[:sample_size] if sample_size else emb
            logger.info(f"loading embeddings from {dir_fold_emb} with shape "
                        f"{emb.shape}, done!")
            fold = int(dir_fold_emb.stem)

            if fold not in embeddings:
                embeddings[fold] = dict()
            embeddings[fold][name_emb] = emb

    logger.info("loading embeddings, done!")

    return embeddings


def train(
        config,
        path_locations,
        dir_experiment,
        dir_experiment_arcmargin,
        path_pairs=None,
        path_dataset=None):

    logger.info("training...")

    print(config)

    fix_seeds()
    initialize_logger(logger)

    name_experiment = datetime.today().strftime("%Y%m%d_%H%M%S")
    dir_experiment = pathlib.Path(dir_experiment) / name_experiment
    os.makedirs(dir_experiment)

    if not isinstance(dir_experiment_arcmargin, list):
        dir_experiment_arcmargin = [dir_experiment_arcmargin]
    dir_experiment_arcmargin = [pathlib.Path(d) for d in dir_experiment_arcmargin]

    df_locations = load_locations(path_locations)

    folds = [fold for fold in df_locations["fold"].unique() if fold != -1]

    if path_dataset:
        logger.info(f"loading dataset from {path_dataset}...")
        df_pairs = pd.read_csv(path_dataset)
        features = [
            f for f in df_pairs.columns
            if f not in ["id_1", "id_2", "fold", "match"]]
        logger.info(f"loading dataset from {path_dataset}, done!")
    else:
        if path_pairs:
            logger.info(f"loading pairs from {path_pairs}...")
            df_pairs = pd.read_csv(path_pairs)
            logger.info(f"loading pairs from {path_pairs}, done!")
        else:
            logger.info("loading embeddings...")

            embeddings = load_embeddings(
                dir_experiment_arcmargin,
                config["embeddings_neighbors"],
                config["sample_size"])

            embeddings_extra = load_embeddings(
                dir_experiment_arcmargin,
                config["embeddings_extra"],
                config["sample_size"])

            df_pairs_folds = list()
            for fold in sorted(folds):
                df_locations_fold = df_locations[df_locations["fold"] == fold]

                embs_fold = embeddings[fold]
                embs_extra_fold = embeddings_extra.get(fold, [])

                # build location pairs for fold `fold`
                embs = np.hstack(embs_fold.values())
                embs_dims = [emb.shape[1] for emb in embs_fold.values()]
                df_fold = build_pairs(
                    embs=embs,
                    df_locations=df_locations_fold,
                    embs_dims=embs_dims,
                    max_neighbors=config["train_neighbors"]["max"],
                    threshold=config["train_neighbors"]["threshold"],
                    blending_stages=config["train_neighbors"]["blending_stages"],
                    qe=config["train_neighbors"]["qe"],
                    add_cosine_distance=True)

                for emb_name in embs_fold:
                    df_fold = add_extra_embedding(
                        df_fold, [embs_fold[emb_name]], emb_name, blending=True)
                for emb_name in embs_extra_fold:
                    df_fold = add_extra_embedding(
                        df_fold, [embs_extra_fold[emb_name]], emb_name)
                    df_fold = add_extra_embedding(
                         df_fold, [embs_extra_fold[emb_name]], emb_name, blending=True)

                df_fold["fold"] = fold
                df_pairs_folds.append(df_fold)

                score = max_iou(df_fold, df_locations_fold)
                logger.info(f"fold='{fold}': pairs: {len(df_fold)} (max IoU: {score:.4f})")

            df_pairs = pd.concat(df_pairs_folds)
            df_pairs.to_csv(dir_experiment / "pairs.csv", index=False)

        # build location pairs dataset
        logger.info("building dataset features...")
        df_pairs_batches = list()

        bs = 200000
        for idx in tqdm(range(0, len(df_pairs), bs)):
            df_p, features = build_pairs_dataset(
                df_pairs=df_pairs.iloc[idx:idx + bs],
                df_locations=df_locations)
            df_pairs_batches.append(df_p)
        df_pairs = pd.concat(df_pairs_batches).reset_index(drop=True)

        sim_features = add_embedding_similarity_features(df_pairs)
        features += sim_features

        df_pairs.to_csv(dir_experiment / "dataset.csv", index=False)
        logger.info("building dataset features, done!")

    if config.get("only_dataset", None):
        return

    scores_vl = list()
    df_pairs = df_pairs[df_pairs["fold"] != -1]

    features = sorted(features)

    for fold in folds:
        df_tr = df_pairs[df_pairs["fold"] != fold]
        df_vl = df_pairs[df_pairs["fold"] == fold]

        x_tr, y_tr = df_tr[features], df_tr["match"]
        x_vl, y_vl = df_vl[features], df_vl["match"]
        pairs_vl = df_vl[["id_1", "id_2"]].values
        matches_vl = build_matches(df_locations[df_locations["fold"] == fold])

        dset_tr = lgb.Dataset(x_tr, label=y_tr)
        dset_vl = lgb.Dataset(x_vl, label=y_vl)

        model = lgb.train(
            params=config["params"],
            num_boost_round=10000,
            train_set=dset_tr,
            valid_sets=[dset_tr, dset_vl],
            valid_names=["train", "valid"],
            early_stopping_rounds=100,
            verbose_eval=100)

        pred_vl = model.predict(x_vl)
        threshold, score_vl = validate_pairs_pred(
            pred=pred_vl,
            pairs=pairs_vl,
            ids_locations=df_locations.loc[df_locations["fold"] == fold, "id"].values,
            df_true=matches_vl,
            verbose=False)
        scores_vl.append(score_vl)
        logger.info(f"[fold={fold}] score IOU[val]: {score_vl:.4f} at threshold:"
                    f" {threshold:.4f}")

        dir_experiment_fold = dir_experiment / str(fold)
        os.makedirs(dir_experiment_fold)
        model.save_model((dir_experiment_fold / f"{score_vl:.4f}.lgb").as_posix())

    score_vl_avg = np.mean(scores_vl)
    score_vl_std = np.std(scores_vl)
    logger.info(f"score[val]: {score_vl_avg:.4f}+-{score_vl_std:.4f}")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-config", type=str, required=True)
    parser.add_argument("--path-locations", type=str, required=True)
    parser.add_argument("--dir-experiment", type=str, required=False)
    parser.add_argument("--path-dataset", type=str, required=False)
    parser.add_argument("--path-pairs", type=str, required=False)
    parser.add_argument("--dir-experiment-arcmargin", type=str, nargs="+", required=True)

    args = parser.parse_args()

    config = load_config(args.path_config)

    train(
        config=config,
        path_locations=args.path_locations,
        dir_experiment=args.dir_experiment,
        dir_experiment_arcmargin=args.dir_experiment_arcmargin,
        path_pairs=args.path_pairs,
        path_dataset=args.path_dataset)


if __name__ == "__main__":
    run()
