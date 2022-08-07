import logging
import pathlib
import argparse

import numpy as np

from foursquare.utils.generic import initialize_logger, fix_seeds
from foursquare.utils.io import load_config, load_locations
from foursquare.utils.matching import MatchesValidator, build_matches


logger = logging.getLogger('foursquare')


def validate(
        config,
        path_locations,
        dir_experiment,
        project_threshold=False):

    fix_seeds()
    initialize_logger(logger)
    dir_experiment = [pathlib.Path(d) for d in dir_experiment]

    logger.info(f"validating experiment {dir_experiment}...")

    df_locations = load_locations(path_locations)

    folds = [fold for fold in df_locations["fold"].unique() if fold != -1]
    for fold in sorted(folds):
        df_locations_fold = df_locations[df_locations["fold"] == int(fold)]

        # load embeddings from all experiments for the fold `fold`
        embeddings = list()
        dims_embedding = list()
        for d in dir_experiment:
            logger.info(f"loading embeddings from {d / str(fold)}...")
            emb = np.load(d / str(fold) / "valid_embeddings.npy")
            embeddings.append(emb)
            dims_embedding.append(emb.shape[1])
            logger.info(f"loading embeddings from {d / str(fold)}, done!")
        embeddings = np.hstack(embeddings)

        split_df_matches = list()
        split_embeddings = list()
        if project_threshold:
            pois = df_locations_fold["point_of_interest"].unique()
            for size in np.linspace(20000, len(pois), 5, dtype="int32"):
                split_pois = np.random.choice(pois, size, replace=False)
                split_mask = df_locations_fold["point_of_interest"].isin(split_pois)
                split_df_matches.append(build_matches(df_locations_fold[split_mask]))
                split_embeddings.append(embeddings[split_mask])
        else:
            split_df_matches.append(build_matches(df_locations_fold))
            split_embeddings.append(embeddings)

        scores = list()
        thresholds = list()
        for df_matches, embeddings in zip(split_df_matches, split_embeddings):
            validator = MatchesValidator(
                max_neighbors=config["neighbors"]["max"],
                blending_stages=config["neighbors"]["blending_stages"],
                force_cpu=False,
                verbose=True)
            score, threshold = validator.validate(
                embeddings, df_matches, dims_embedding, return_thresholds=True)
            scores.append(score)
            thresholds.append(threshold)
            logger.info(f"size: {len(df_matches)}, score: {score:.4f}, "
                        f"threshold: {threshold}")

        logger.info(f"validation score IOU: {scores[-1]:.4f} at "
                    f"thresholds {thresholds[-1]}")
        logger.info(f"validating experiment {dir_experiment}, done!")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-config", type=str, required=True)
    parser.add_argument("--path-locations", type=str, required=True)
    parser.add_argument("--project-threshold", action="store_true")
    parser.add_argument("--dir-experiment", type=str, nargs="+", required=True)

    args = parser.parse_args()

    logger.info(f"args: {args}")

    config = load_config(args.path_config)

    validate(
        config=config,
        path_locations=args.path_locations,
        project_threshold=args.project_threshold,
        dir_experiment=args.dir_experiment)


if __name__ == "__main__":
    run()
