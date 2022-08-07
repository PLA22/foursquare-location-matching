import numpy as np
from sklearn.model_selection import KFold

from foursquare.utils.generic import fix_seeds
from foursquare.utils.io import load_locations


def split_dataset(path_locations, path_output, dev_size):
    fix_seeds()

    df_locations = load_locations(path_locations)

    pois_dev = np.random.choice(
        df_locations["point_of_interest"].unique(),
        dev_size,
        replace=False)

    df_locations["validation"] = 0  # cv data
    df_locations.loc[df_locations["point_of_interest"].isin(pois_dev), "validation"] = 1  # oof data

    pois_cv = df_locations.loc[df_locations["validation"] == 0, "point_of_interest"].unique()
    kfold = KFold(n_splits=3, shuffle=True, random_state=2022)
    df_locations["fold"] = -1
    for fold, (idx_tr, idx_vl) in enumerate(kfold.split(pois_cv), start=1):
        df_locations.loc[df_locations["point_of_interest"].isin(pois_cv[idx_vl]), "fold"] = fold

    df_locations.to_csv(path_output, index=False)


if __name__ == "__main__":
    split_dataset(
        path_locations="/kaggle/input/foursquare-location-matching/train.csv",
        path_output="/kaggle/input/foursquare-location-matching/train_fold.csv",
        dev_size=20000)
