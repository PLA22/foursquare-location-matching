import joblib
import urllib.parse
import string

import tldextract
import pandas as pd
import numpy as np
from unidecode import unidecode
from sklearn.preprocessing import (
    LabelEncoder,
    QuantileTransformer,
    OneHotEncoder)
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


def encode_high_precision(x_hp, emb_size=20, precision=1e6):
    m = np.exp(np.log(precision) / emb_size)

    angle_freq = m ** np.arange(emb_size)
    angle_freq = np.expand_dims(angle_freq, axis=0)

    x_hp = np.expand_dims(x_hp, axis=1)

    x_enc = x_hp * angle_freq
    x_enc[:, 0::2] = np.cos(x_enc[:, 0::2])
    x_enc[:, 1::2] = np.sin(x_enc[:, 1::2])

    return x_enc


def extract_features_latlon(df):
    df["latitude"] = np.deg2rad(df["latitude"])
    df["longitude"] = np.deg2rad(df["longitude"])

    df["x"] = 0.5 * np.pi * np.cos(df["latitude"]) * np.cos(df["longitude"])
    df["y"] = 0.5 * np.pi * np.cos(df["latitude"]) * np.sin(df["longitude"])
    df["z"] = 0.5 * np.pi * np.sin(df["latitude"])

    latlon_encoded = list()
    for col in ["x", "y", "z"]:
        col_encoded = encode_high_precision(
            df[col].values, emb_size=20, precision=1e6)
        latlon_encoded.append(col_encoded)

    latlon_encoded = np.hstack(latlon_encoded)
    col_names = [f"latlon_{i}" for i in range(latlon_encoded.shape[1])]
    df_latlon = pd.DataFrame(latlon_encoded, columns=col_names)

    return df_latlon


def extract_features_url(df):
    def _extract_features_url(url):
        if url[0] is np.nan:
            return {
                "url_scheme": np.nan,
                "netloc": np.nan,
                "path": np.nan,
                "params": np.nan,
                "query": np.nan,
                "fragment": np.nan,
                "subdomain": np.nan,
                "url_domain": np.nan,
                "suffix": np.nan,
            }

        url_features = dict()
        url_parsed = urllib.parse.urlparse(url[0])

        url_features["url_scheme"] = url_parsed.scheme
        url_features["url_netloc"] = url_parsed.netloc
        url_features["url_path"] = url_parsed.path
        url_features["url_params"] = url_parsed.params
        url_features["url_query"] = url_parsed.query
        url_features["url_fragment"] = url_parsed.fragment

        url_tld_parsed = tldextract.extract(url_parsed.netloc)
        url_features["url_subdomain"] = url_tld_parsed.subdomain
        url_features["url_domain"] = url_tld_parsed.domain
        url_features["url_suffix"] = url_tld_parsed.suffix

        url_features = {f: v or np.nan for f, v in url_features.items()}

        return url_features

    df_url = df.apply(_extract_features_url, axis=1, result_type="expand")
    return df_url


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        max_length=maxlen,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        add_special_tokens=True)
    return np.array(enc_di['input_ids'])


class DatasetBuilder:
    def __init__(
            self,
            numeric_scale_features,
            numeric_no_scale_features,
            tfidf_features,
            categorical_features,
            onehot_features,
            transformer_features,
            text_clean,
            text_normalize,
            embedding_features=None,
            transformer_url=None,
            transformer_maxlen=None,
            freq_category_exclude=5,
            null_category="null",
            infrequent_or_unknown_category="null"):

        self.numeric_scale_features = numeric_scale_features
        self.numeric_no_scale_features = numeric_no_scale_features
        self.tfidf_features = tfidf_features
        self.categorical_features = categorical_features
        self.onehot_features = onehot_features
        self.transformer_features = transformer_features
        self.embedding_features = embedding_features

        self.text_clean = text_clean
        self.text_normalize = text_normalize

        self.transformer_url = transformer_url
        self.transformer_maxlen = transformer_maxlen

        self.freq_category_exclude = freq_category_exclude
        self.null_category = null_category
        self.infrequent_or_unknown_category = infrequent_or_unknown_category

        self.fitted = False

        self.numeric_encoder = None
        self.tfidf_encoders = dict()
        self.label_encoders = dict()
        self.onehot_encoders = dict()

        self.dims = dict()

    def _normalize(self, series):
        series = series.str.replace(f"[{string.punctuation}]", "", regex=True)
        return series.apply(unidecode)

    def _clean_str(self, series):
        series = series.str.strip()
        series = series.str.lower()
        return series

    def _fillna(self, series):
        return series.fillna(self.null_category)

    def _mask_infrequent_categories(self, series):
        counts = series.value_counts()
        rare_classes = counts[counts < self.freq_category_exclude].index
        series[series.isin(rare_classes)] = self.infrequent_or_unknown_category
        return series

    def _mask_new_categories(self, series, encoder):
        if isinstance(encoder, LabelEncoder):
            mask_unknown = ~series.isin(encoder.classes_)
        elif isinstance(encoder, OneHotEncoder):
            mask_unknown = ~series.isin(encoder.categories_[0])
        else:
            raise ValueError(f"encoder don't known: '{type(encoder)}'")
        series[mask_unknown] = self.infrequent_or_unknown_category
        return series

    def load(self, dir_encoders):
        self.numeric_encoder = joblib.load(dir_encoders / "numeric.joblib")
        for cfg_feat in self.tfidf_features:
            encoder = joblib.load(dir_encoders / f"{cfg_feat['name']}.joblib")
            self.tfidf_encoders[cfg_feat["name"]] = encoder
        for name in self.categorical_features:
            encoder = joblib.load(dir_encoders / f"{name}.joblib")
            self.label_encoders[name] = encoder
            self.dims[name] = len(encoder.classes_)
        for name in self.onehot_features:
            encoder = joblib.load(dir_encoders / f"{name}.joblib")
            self.onehot_encoders[name] = encoder
        self.fitted = True

    def save(self, dir_encoders):
        joblib.dump(self.numeric_encoder, dir_encoders / "numeric.joblib")
        for cfg_feat in self.tfidf_features:
            joblib.dump(
                self.tfidf_encoders[cfg_feat["name"]],
                dir_encoders / f"{cfg_feat['name']}.joblib")
        for name in self.categorical_features:
            joblib.dump(
                self.label_encoders[name], dir_encoders / f"{name}.joblib")
        for name in self.onehot_features:
            joblib.dump(
                self.onehot_encoders[name], dir_encoders / f"{name}.joblib")

    def preprocess(self, df):
        df = df.reset_index(drop=True)

        df_url_features = extract_features_url(df[["url"]])
        df = pd.concat([df, df_url_features], axis=1)

        df["has_phone"] = df["phone"].notnull().astype("int32")
        df["has_url"] = df["url"].notnull().astype("int32")

        df["name_length"] = df["name"].str.len()
        df["address_length"] = df["address"].str.len()
        df["zip_length"] = df["zip"].str.len()
        df["url_length"] = df["url"].str.len()
        df["phone_length"] = df["phone"].str.len()

        df["name_words_n"] = df["name"].str.split().str.len()
        df["address_words_n"] = df["address"].str.split().str.len()
        df["zip_words_n"] = df["zip"].str.split().str.len()

        df["name_num_n"] = df["name"].str.count(r"\d")
        df["adress_num_n"] = df["address"].str.count(r"\d")
        df["zip_num_n"] = df["name"].str.count(r"\d")

        df["name_rare_n"] = df["name"].str.count(r"[^a-zA-Z0-9]")
        df["address_rare_n"] = df["address"].str.count(r"[^a-zA-Z0-9]")
        df["zip_rare_n"] = df["zip"].str.count(r"[^a-zA-Z0-9]")
        df["url_rare_n"] = df["url"].str.count(r"[^a-zA-Z0-9]")

        df[self.numeric_scale_features] = df[self.numeric_scale_features] \
            .fillna(0)

        df["categories"] = self._fillna(df["categories"])
        df["name"] = self._fillna(df["name"])
        df["zip"] = self._fillna(df["zip"])
        df["address"] = self._fillna(df["address"])
        df["country"] = self._fillna(df["country"])
        df["state"] = self._fillna(df["state"])
        df["city"] = self._fillna(df["city"])
        df["url_scheme"] = self._fillna(df["url_scheme"])
        df["url_suffix"] = self._fillna(df["url_suffix"])
        df["url_domain"] = self._fillna(df["url_domain"])
        df["url_subdomain"] = self._fillna(df["url_subdomain"])
        df["url"] = self._fillna(df["url"])
        df["phone"] = self._fillna(df["phone"])

        for f in self.text_normalize:
            df[f] = self._normalize(df[f])

        for f in self.text_clean:
            df[f] = self._clean_str(df[f])

        return df

    def _fit_tfdidf(self, data, params):
        encoder = TfidfVectorizer(**params)
        encoder.fit(data)
        return encoder

    def _fit_label_encoder(self, series):
        series = self._mask_infrequent_categories(series)
        all_values = series.unique().tolist() \
            + [self.infrequent_or_unknown_category]
        encoder = LabelEncoder()
        encoder.fit(all_values)
        return encoder

    def _fit_onehot_encoder(self, series):
        series = self._mask_infrequent_categories(series)
        all_values = series.unique().tolist() \
            + [self.infrequent_or_unknown_category]
        all_values = np.array(all_values)
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(all_values.reshape(-1, 1))
        return encoder

    def fit(self, df):
        df = self.preprocess(df)

        if self.numeric_scale_features:
            self.numeric_encoder = QuantileTransformer()
            self.numeric_encoder.fit(df[self.numeric_scale_features])

        for feature_cfg in self.tfidf_features:
            name = feature_cfg["name"]
            params = feature_cfg["params"]
            encoder = self._fit_tfdidf(df[name], params)
            self.tfidf_encoders[name] = encoder

        for name in self.categorical_features:
            encoder = self._fit_label_encoder(df[name])
            self.label_encoders[name] = encoder
            self.dims[name] = len(encoder.classes_)

        for name in self.onehot_features:
            encoder = self._fit_onehot_encoder(df[name])
            self.onehot_encoders[name] = encoder

        self.fitted = True

    def build(self, df, test_mode=False):
        if not self.fitted:
            raise ValueError("DatasetBuilder must be fitted before call build")

        idx = df.index.values

        df = self.preprocess(df)
        df_latlon = extract_features_latlon(df[["latitude", "longitude"]])

        dataset = dict()

        arr_numeric = list()
        if "latlon" in self.numeric_no_scale_features:
            arr_numeric.append(df_latlon.values)
        if self.numeric_scale_features:
            arr_numeric.append(self.numeric_encoder.transform(
                df[self.numeric_scale_features]))

        if arr_numeric:
            dataset["numeric"] = np.hstack(arr_numeric)
            self.dims["numeric"] = dataset["numeric"].shape[1]

        for feature_name, encoder in self.tfidf_encoders.items():
            arr = encoder.transform(df[feature_name])
            dataset[feature_name] = arr.sorted_indices()
            self.dims[feature_name] = arr.shape[1]

        for feature_name, encoder in self.label_encoders.items():
            series = self._mask_new_categories(
                df[feature_name], encoder=encoder)
            arr = encoder.transform(series)
            dataset[feature_name] = arr
            self.dims[feature_name] = len(encoder.classes_)

        for feature_name, encoder in self.onehot_encoders.items():
            series = self._mask_new_categories(
                df[feature_name], encoder=encoder)
            arr = encoder.transform(series.values.reshape(-1, 1))
            dataset[feature_name] = arr.sorted_indices()
            self.dims[feature_name] = arr.shape[1]

        if self.transformer_features:
            tokenizer = AutoTokenizer.from_pretrained(self.transformer_url)
            encoded_input = df[self.transformer_features[0]]
            for feature_name in self.transformer_features[1:]:
                encoded_input += " [SEP] " + df[feature_name]
            encoded_input = regular_encode(
                encoded_input.values.tolist(),
                tokenizer,
                self.transformer_maxlen)
            dataset["transformer"] = encoded_input

        if self.embedding_features:
            for feature_cfg in self.embedding_features:
                feature_name = f"{feature_cfg['name']}_embedding"
                dataset[feature_name] = np.load(feature_cfg["url"])[idx]
                self.dims[feature_name] = dataset[feature_name].shape[1]

        if not test_mode:
            encoder_label = LabelEncoder()
            dataset["label"] = encoder_label.fit_transform(
                df["point_of_interest"])

        return dataset
