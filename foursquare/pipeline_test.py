import pathlib

import tensorflow.keras as keras
import pandas as pd
import numpy as np
import lightgbm as lgb
import tqdm

from foursquare.utils.io import load_locations
from foursquare.dataset.arcmargin import DatasetBuilder
from foursquare.dataset.lgb import (
    build_pairs_dataset,
    build_pairs,
    merge_pairs,
    add_extra_embedding,
    add_embedding_similarity_features
)
from foursquare.utils.generic import fix_seeds
from foursquare.utils.matching import postprocess_pairs_pred, blend_embeddings
from foursquare.model_arcmargin import (
    build_transformer_arcmargin_model,
    build_arcmargin_model
)


def extract_embeddings(
        config,
        path_locations,
        dir_model,
        path_embeddings_output):

    fix_seeds()

    keras.mixed_precision.set_global_policy('mixed_float16')

    dir_model = pathlib.Path(dir_model)

    print(f"loading test data from '{path_locations}'...")
    df_locations = load_locations(path_locations)
    print(f"loading test data from '{path_locations}', done!")

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
    dataset_builder.load(dir_model / "encoders")
    dset_test = dataset_builder.build(df_locations, test_mode=True)

    path_model = dir_model / "model_emb.h5"
    print(f"loading model from '{path_model}'...")

    if config["transformer_tokenizer_url"]:
        model = build_transformer_arcmargin_model(
            config=config,
            n_classes=None,
            only_embedding=True)
    else:
        model = build_arcmargin_model(
            config=config,
            n_classes=None,
            input_dims=dataset_builder.dims,
            only_embedding=True)

    model.load_weights(path_model)
    print(f"loading model from '{path_model}, done!")
    model.summary()

    print("extracting embeddings...")
    embs = model.predict(dset_test, batch_size=32)
    print("extracting embeddings, done!")

    print(f"saving embeddings at {path_embeddings_output}...")
    np.save(path_embeddings_output, embs)
    print(f"saving embeddings at {path_embeddings_output}, done!")


def blend_embeddings_neighbors(config, path_embedding):
    dir_embs = path_embedding.parent
    name_embs = path_embedding.stem
    path_output = dir_embs / f"{name_embs}_blend.npy"

    print(f"blending embeddings from {path_embedding}...")
    embs = np.load(path_embedding)

    embs = blend_embeddings(
        embs=embs, threshold=config["lgb"]["embeddings_blending_threshold"])
    np.save(path_output, embs)
    print(f"saved blended embeddings at {path_output}, done!")


def make_pairs(
        config,
        path_locations,
        dir_embeddings,
        path_pairs_output):

    fix_seeds()

    print(f"loading test data from '{path_locations}'...")
    df_locations = load_locations(path_locations)
    print(f"loading test data from '{path_locations}', done!")

    print(f"loading embeddings from {dir_embeddings}...")
    embeddings = list()
    for path in sorted(dir_embeddings.glob("*.npy")):
        if path.stem in config["lgb"]["embeddings_neighbors"]:
            print(f"loading embedding from {path}...")
            embeddings.append(np.load(path))
            print(f"loading embedding from {path}, done!")

    embeddings_dims = [emb.shape[1] for emb in embeddings]
    embeddings = np.hstack(embeddings)
    print(f"loading embeddings from {dir_embeddings} with dims:"
          f" {embeddings_dims}, done!")

    df_pairs = build_pairs(
        embs=embeddings,
        df_locations=df_locations,
        embs_dims=embeddings_dims,
        max_neighbors=config["lgb"]["neighbors"]["max"],
        threshold=config["lgb"]["neighbors"]["threshold"],
        blending_stages=config["lgb"]["neighbors"]["blending_stages"],
        qe=config["lgb"]["neighbors"]["qe"],
        add_cosine_distance=False,
        test=True)

    print(f"writing pairs in '{path_pairs_output}'...")
    df_pairs.to_csv(path_pairs_output, index=False)
    print(f"writing pairs in '{path_pairs_output}', done!")


def merge_pairs_dataset(dir_pairs, path_pairs_output):
    fix_seeds()

    df_pairs = list()
    for path_pairs in sorted(dir_pairs.glob("*")):
        print(f"loading pairs from {path_pairs}...")
        df_pairs.append(pd.read_csv(path_pairs))
        print(f"loading pairs from {path_pairs}, done!")

    df_pairs = pd.concat(df_pairs)
    df_pairs = df_pairs.drop_duplicates()

    print(f"saving pairs merged at {path_pairs_output}...")
    df_pairs.to_csv(path_pairs_output, index=False)
    print(f"saving pairs merged at {path_pairs_output}, done!")


def _load_embeddings(dir_embeddings, name_embeddings):
    embeddings = dict()
    embeddings_dims = dict()
    for dir_fold_embedding in sorted(dir_embeddings.glob("*")):
        fold = int(dir_fold_embedding.stem)
        for path_embedding in sorted(dir_fold_embedding.glob("*")):
            name_embedding = path_embedding.stem
            if name_embedding in name_embeddings:
                print(f"loading embeddings from {path_embedding}...")
                emb = np.load(path_embedding)
                print(f"loading embeddings from {path_embedding}, done!")

                if fold not in embeddings:
                    embeddings[fold] = dict()
                embeddings[fold][name_embedding] = emb
                embeddings_dims[name_embedding] = emb.shape[1]

    return embeddings, embeddings_dims


def add_cosine_distances(config, path_pairs, dir_embeddings, path_pairs_output):
    fix_seeds()

    embeddings, embeddings_dims = _load_embeddings(
        dir_embeddings, config["lgb"]["embeddings_neighbors"])
    embs_stacked = [
        np.hstack(list(embeddings[fold].values()))
        for fold in embeddings]

    print(embeddings_dims)

    df_pairs = list()
    df_iter = pd.read_csv(path_pairs, chunksize=50000)
    for df in df_iter:
        df = merge_pairs(
            df_pairs=df,
            embeddings=embs_stacked,
            embs_dims=list(embeddings_dims.values()))
        df = df[df["sim_cosine"] > config["lgb"]["threshold_merge"]]
        df_pairs.append(df)
    df_pairs = pd.concat(df_pairs)

    print(f"writting pairs in '{path_pairs_output}'...")
    df_pairs.to_csv(path_pairs_output, index=False)
    print(f"writting pairs in '{path_pairs_output}', done!")


def add_cosine_distances_extra(
        path_pairs,
        path_pairs_output,
        dir_embeddings,
        name_embedding):

    fix_seeds()

    embeddings, embeddings_dims = _load_embeddings(dir_embeddings, [name_embedding])
    embs_fold = [embeddings[fold][name_embedding] for fold in embeddings]

    df_pairs = list()
    df_iter = pd.read_csv(path_pairs, chunksize=50000)
    for df in df_iter:
        df = add_extra_embedding(
            df_pairs=df,
            embs=embs_fold,
            name=name_embedding)
        df_pairs.append(df)
    df_pairs = pd.concat(df_pairs)

    print(f"writting pairs in '{path_pairs_output}'...")
    df_pairs.to_csv(path_pairs_output, index=False)
    print(f"writting pairs in '{path_pairs_output}', done!")


def make_pairs_dataset(
        config,
        path_pairs,
        path_locations,
        path_pairs_output):

    fix_seeds()

    print(f"loading test data from '{path_locations}'...")
    df_locations = load_locations(path_locations)
    print(f"loading test data from '{path_locations}', done!")

    dfs = list()
    df_pairs_iter = pd.read_csv(path_pairs, chunksize=50000)
    for df_pairs in tqdm.tqdm(df_pairs_iter):
        df_pairs, features = build_pairs_dataset(df_pairs, df_locations)
        dfs.append(df_pairs)
    df_pairs = pd.concat(dfs)

    _ = add_embedding_similarity_features(df_pairs)

    features = sorted(df_pairs.columns.tolist())
    df_pairs = df_pairs[features]

    print(f"writting pairs dataset at '{path_pairs_output}'...")
    df_pairs.to_csv(path_pairs_output, index=False)
    print(f"writting pairs dataset at '{path_pairs_output}', done!")


def make_submission(
        config,
        path_locations,
        path_submission_output,
        dir_lgb=None,
        path_pairs_dataset=None):

    print("testing...")

    fix_seeds()

    print("predicting...")

    print(f"loading test data from '{path_locations}'...")
    df_locations = load_locations(path_locations, usecols=["id"])
    ids = df_locations["id"].values
    print(f"loading test data from '{path_locations}', done!")

    print(f"loading pairs dataset from '{path_pairs_dataset}'...")
    df_pairs = pd.read_csv(path_pairs_dataset)
    print(f"loading pairs dataset from '{path_pairs_dataset}', done!")

    ids_pairs = df_pairs[["id_1", "id_2"]].values
    df_pairs = df_pairs.drop(["id_1", "id_2"], axis=1)

    paths_model_lgb = list(sorted(dir_lgb.rglob("*.lgb")))
    pred = np.zeros((len(df_pairs), len(paths_model_lgb)))
    for idx, path_model_lgb in enumerate(paths_model_lgb):
        print(f"predicting lgb with {path_model_lgb}...")
        model_lgb = lgb.Booster(model_file=path_model_lgb.as_posix())
        pred[:, idx] = model_lgb.predict(df_pairs)
        print(f"predicting lgb with {path_model_lgb}, done!")
    pred = pred.mean(axis=1)
    df_submission = postprocess_pairs_pred(
        pred=pred,
        pairs=ids_pairs,
        ids_locations=ids,
        threshold=config["lgb"]["threshold_lgb"])

    # format submission
    df_submission["matches"] = df_submission["matches"].apply(
        lambda matches: " ".join(matches))

    print(f"writting submission at '{path_submission_output}'...")
    df_submission.to_csv(path_submission_output, index=False)
    print(f"writting submission at '{path_submission_output}', done!")

    print("testing, done!")

    return df_submission
