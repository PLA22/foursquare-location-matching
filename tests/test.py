from foursquare.utils.io import load_config
from foursquare.pipeline_train_arcmargin import train
from foursquare.pipeline_validate_arcmargin import validate
from foursquare.pipeline_train_lgb import train as train_lgb


def _load_config():
    path_config = "../config/config.json"
    config = load_config(path_config)
    return config


def test_load_config():
    config = _load_config()
    assert config is not None


def test_train_arcface():
    path_config = "../config/config_arcmargin_tfidf.json"
    dir_experiment = "/kaggle/models/arcface/prueba"
    path_locations = "/kaggle/input/foursquare-location-matching/train_fold.csv"
    fold = 1
    path_checkpoint = None

    config = load_config(path_config)

    train(
        config=config,
        dir_experiment=dir_experiment,
        path_locations=path_locations,
        fold=fold,
        path_checkpoint=path_checkpoint)


def test_validation():
    path_config = "../config/config_arcmargin_tfidf.json"
    path_locations = "/kaggle/input/foursquare-location-matching/train_fold.csv"
    dir_experiment = [
        "/kaggle/models/arcface/tfidf_512",
        "/kaggle/models/arcface/xlm_100k",
    ]

    config = load_config(path_config)

    validate(
        config=config,
        path_locations=path_locations,
        dir_experiment=dir_experiment,
        project_threshold=True)


def test_train_lgb():
    path_config = "../config/config_lgb.json"
    path_locations = "/kaggle/input/foursquare-location-matching/train_fold.csv"
    dir_experiment = "/kaggle/models/lgb"
    dir_experiment_arcface = [
        "/kaggle/models/arcface/tfidf_512",
        "/kaggle/models/arcface/tfidf2_512",
        "/kaggle/models/arcface/xlm",
    ]
    path_pairs = "/kaggle/models/lgb/20220706_203539/pairs.csv"

    config = load_config(path_config)

    train_lgb(
        config=config,
        path_locations=path_locations,
        dir_experiment=dir_experiment,
        dir_experiment_arcmargin=dir_experiment_arcface,
        path_pairs=path_pairs)
