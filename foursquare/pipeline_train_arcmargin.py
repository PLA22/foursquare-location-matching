import json
import logging
import pathlib
import os
import argparse

import numpy as np
import tensorflow.keras as keras

from foursquare.utils.io import load_locations
from foursquare.dataset.arcmargin import (
    DatasetBuilder
)
from foursquare.model_arcmargin import (
    build_arcmargin_model,
    build_transformer_arcmargin_model
)
from foursquare.utils.generic import initialize_logger, fix_seeds
from foursquare.utils.ml import ValidationLog
from foursquare.utils.matching import MatchesValidator, build_matches
from foursquare.utils.io import load_config


logger = logging.getLogger('foursquare')


def build_sample_weigths(df_locations, exp):
    df = df_locations.groupby(["point_of_interest"]).size()
    df = df.reset_index(name="cnt")
    df = df_locations[["id", "point_of_interest"]].merge(
        df, on=["point_of_interest"], how="left")
    df["weight"] = df["cnt"] ** -exp
    return df["weight"].values


def get_callbacks(
        config,
        model,
        dir_experiment,
        validation_data=None,
        metric_monitor="loss",
        metric_mode="min"):

    callbacks = list()

    validator = MatchesValidator(
        max_neighbors=config["neighbors"]["max"],
        blending_stages=0,
        force_cpu=True,
        verbose=False)
    callbacks.append(
        ValidationLog(
            model=model,
            name_metric=metric_monitor,
            validation_data=validation_data,
            validation_function=validator.validate,
            validation_function_kwargs={
                "dims_embedding": [config["model"]["dim_embedding"]],
                "return_thresholds": False},
            verbose=True))
    callbacks.append(
        keras.callbacks.CSVLogger(
            dir_experiment / "log.log",
            separator=',',
            append=False))

    if config["callbacks"]["checkpoint"]:
        path_ckpt = dir_experiment / "model.ckpt"
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                monitor=metric_monitor,
                mode=metric_mode,
                filepath=path_ckpt,
                save_best_only=True,
                save_weights_only=True))

    return callbacks


def train(
        config,
        path_locations,
        dir_experiment,
        fold,
        path_checkpoint=None):

    logger.info("training...")

    logger.info(config)

    if config["mixed_precision"]:
        keras.mixed_precision.set_global_policy('mixed_float16')

    fix_seeds()
    initialize_logger(logger)

    dir_experiment = pathlib.Path(dir_experiment) / str(fold)
    os.makedirs(dir_experiment, exist_ok=True)
    with open(dir_experiment / "config.json", "w") as f:
        json.dump(config, f)

    df_locations = load_locations(path_locations)

    logger.info("building training dataset...")

    df_locations_tr = df_locations[
        (df_locations["fold"] != fold)
        & (df_locations["fold"] != -1)]
    df_locations_vl = df_locations[df_locations["fold"] == fold]
    df_locations_oof = df_locations[df_locations["fold"] == -1]

    # sample train dataset
    pois_tr = df_locations_tr["point_of_interest"].unique()
    if config["sample_train"] is not None:
        pois_tr = np.random.choice(
            pois_tr, size=config["sample_train"], replace=False)
    df_locations_tr = df_locations_tr[
        df_locations_tr["point_of_interest"].isin(pois_tr)]

    dataset_builder = DatasetBuilder(
        numeric_scale_features=config["features"]["numeric_scale"],
        numeric_no_scale_features=config["features"]["numeric_no_scale"],
        tfidf_features=config["features"]["tfidf"],
        categorical_features=config["features"]["categorical"],
        onehot_features=config["features"]["onehot"],
        transformer_features=config["features"]["transformer"],
        text_clean=config["features"]["text_clean"],
        text_normalize=config["features"]["text_normalize"],
        transformer_url=config["transformer_tokenizer_url"],
        transformer_maxlen=config["transformer_maxlen"],
        embedding_features=config["features"].get("embedding"),
        freq_category_exclude=config["features"]["freq_category_exclude"])
    dataset_builder.fit(df_locations_tr)
    dset_tr = dataset_builder.build(df_locations_tr)
    weights_tr = build_sample_weigths(
        df_locations_tr, config["sample_weight_exp"])

    dir_encoders = dir_experiment / "encoders"
    logger.info(f"saving encoders in {dir_encoders}...")
    os.makedirs(dir_encoders, exist_ok=True)
    dataset_builder.save(dir_encoders)
    logger.info(f"saving encoders in {dir_encoders}, done!")

    logger.info(f"train dataset: {len(df_locations_tr)} locations "
                f"and {len(pois_tr)} POIs")
    logger.info("building training dataset, done!")

    logger.info("building valid dataset...")

    dset_vl = dataset_builder.build(df_locations_vl)
    pois_vl = df_locations_vl["point_of_interest"].unique()

    logger.info(f"validation valid dataset: {len(df_locations_vl)} "
                f"locations and {len(pois_vl)} POIs")
    logger.info("building valid dataset, done!")

    logger.info("building oof dataset...")

    pois_oof = df_locations_oof["point_of_interest"].unique()
    dset_oof = dataset_builder.build(df_locations_oof)
    df_matches_oof = build_matches(df_locations_oof)

    logger.info(f"validation oof dataset: {len(df_locations_oof)} "
                f"locations and {len(pois_oof)} POIs")
    logger.info("building oof dataset, done!")

    logger.info("building models...")

    if config["transformer_tokenizer_url"]:
        model, model_emb = build_transformer_arcmargin_model(
            config=config,
            n_classes=len(np.unique(dset_tr["label"])))
    else:
        model, model_emb = build_arcmargin_model(
            config=config,
            n_classes=len(np.unique(dset_tr["label"])),
            input_dims=dataset_builder.dims)

    if path_checkpoint:
        logger.info(f"loading weights from checkpoint {path_checkpoint}...")
        model.load_weights(path_checkpoint)
        logger.info(f"loading weights from checkpoint {path_checkpoint}, "
                    f"done!")
    model.summary()
    model_emb.summary()

    logger.info("building models, done!")

    callbacks = get_callbacks(
        config=config,
        model=model_emb,
        dir_experiment=dir_experiment,
        validation_data=(dset_oof, df_matches_oof),
        metric_monitor="val_iou",
        metric_mode="max")

    model.fit(
        dset_tr,
        dset_tr["label"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        callbacks=callbacks,
        sample_weight=weights_tr)

    model_emb.save_weights(dir_experiment / f"model_emb.h5")

    logger.info("predicting valid embeddings...")
    path_embs_vl = dir_experiment / "valid_embeddings.npy"
    embs = model_emb.predict(dset_vl, batch_size=128)
    np.save(path_embs_vl, embs)
    logger.info(f"embeddings saved at {path_embs_vl}")
    logger.info("predicting valid embeddings, done!")

    logger.info("predicting oof embeddings...")
    path_embs_oof = dir_experiment / "oof_embeddings.npy"
    embs = model_emb.predict(dset_oof, batch_size=128)
    np.save(path_embs_oof, embs)
    logger.info(f"embeddings saved at {path_embs_oof}")
    logger.info("predicting oof embeddings, done!")

    logger.info("training, done!")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-config", type=str, required=True)
    parser.add_argument("--dir-experiment", type=str, required=True)
    parser.add_argument("--path-locations", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--path-checkpoint", type=str, required=False)

    args = parser.parse_args()

    config = load_config(args.path_config)

    train(
        config=config,
        dir_experiment=args.dir_experiment,
        path_locations=args.path_locations,
        fold=args.fold,
        path_checkpoint=args.path_checkpoint)


if __name__ == "__main__":
    run()
