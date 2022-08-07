import math

import transformers
import tensorflow as tf
import tensorflow.keras as keras
from transformers import TFAutoModel

from foursquare.utils.ml import MultiOptimizer


class ArcMarginProductWarmup(tf.keras.layers.Layer):
    """
    Implements large margin arc distance with warm-up.

    Implementation without warmup is based on:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
        https://www.kaggle.com/code/ragnar123/shopee-efficientnetb3-arcmarginproduct
            for Tensorflow implementation
    """
    def __init__(
            self,
            n_classes,
            warmup_iters=2000,
            m_init=0.2,
            s=30,
            m=0.8,
            easy_margin=False,
            ls_eps=0.0,
            l2=0,
            **kwargs):

        super(ArcMarginProductWarmup, self).__init__(**kwargs)

        self.n_classes = n_classes

        self.warmup_iters = warmup_iters
        self.m_init = m_init
        self.m_end = m

        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.l2 = l2

        self._iters = tf.Variable(0, dtype=tf.int32, trainable=False)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProductWarmup, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=keras.regularizers.l2(self.l2))

    def _update_margin_parameters(self):
        self._iters.assign_add(1)

        k = tf.math.minimum(self._iters / self.warmup_iters, 1)
        self.m = tf.cast(
            (self.m_end - self.m_init) * k + self.m_init, tf.float32)
        self.cos_m = tf.math.cos(self.m)
        self.sin_m = tf.math.sin(self.m)
        self.th = tf.math.cos(math.pi - self.m)
        self.mm = tf.math.sin(math.pi - self.m) * self.m

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0))

        self._update_margin_parameters()

        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def build_scheduler(config):
    scheduler = eval(config["name"])(
        **config["params"])
    if "warmup" in config:
        scheduler = transformers.WarmUp(
            **config["warmup"],
            decay_schedule_fn=scheduler)
    return scheduler


def build_transformer_arcmargin_model(config, n_classes, only_embedding=False):
    input_tranformer = keras.layers.Input(
        shape=[config["transformer_maxlen"]],
        dtype=tf.int32,
        name="transformer")
    transformer_layer = TFAutoModel.from_pretrained(
        config["transformer_model_url"], name="transformer_layer")
    sequence_output = transformer_layer(input_tranformer)[0]
    x = sequence_output[:, 0, :]

    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(config["model"]["dim_embedding"])(x)
    embedding = keras.layers.BatchNormalization()(x)

    model_emb = keras.Model(input_tranformer, embedding)

    if only_embedding:
        return model_emb

    input_label = tf.keras.layers.Input(shape=(), name="label")

    arcmargin = ArcMarginProductWarmup(
        n_classes=n_classes,
        s=config["model"]["arcmargin"]["s"],
        m=config["model"]["arcmargin"]["m"],
        m_init=config["model"]["arcmargin"]["m_init"],
        easy_margin=config["model"]["arcmargin"]["easy_margin"],
        ls_eps=config["model"]["arcmargin"]["ls_eps"],
        warmup_iters=config["model"]["arcmargin"]["warmup_iters"],
        l2=config["model"]["arcmargin"]["l2"],
        name="arcmargin_layer",
        dtype="float32")

    output = arcmargin([embedding, input_label])
    output = tf.keras.layers.Softmax(dtype="float32")(output)

    model = keras.Model([input_tranformer, input_label], output)

    optimizers_and_layers = [
        # transformer optimizer
      (tf.keras.optimizers.Adam(
          learning_rate=build_scheduler(config["model"]["scheduler_transformer"])),
       [l for l in model.layers if l.name == "transformer_layer"]),
        # arcmargin optimizer
      (tf.keras.optimizers.Adam(
          clipvalue=1.0,
          learning_rate=build_scheduler(config["model"]["scheduler"])),
       [l for l in model.layers if l.name != "transformer_layer"])
    ]
    optimizer = MultiOptimizer(optimizers_and_layers)

    model.compile(
       loss=keras.losses.SparseCategoricalCrossentropy(),
       metrics=[keras.metrics.SparseCategoricalAccuracy()],
       optimizer=optimizer)

    return model, model_emb


def build_arcmargin_model(
        config,
        n_classes,
        input_dims=None,
        only_embedding=False):

    inputs = list()
    embeddings = list()

    if "embedding" in config["features"] and config["features"]["embedding"]:
        for f_cfg in config["features"]["embedding"]:
            name = f"{f_cfg['name']}_embedding"
            input_embedding = keras.layers.Input(
                shape=(input_dims[name],), name=name)
            inputs.append(input_embedding)
            emb = keras.layers.Dropout(0.3)(input_embedding)
            embeddings.append(emb)

    if len(config["features"]["numeric_scale"] + config["features"]["numeric_no_scale"]) > 0:
        dim = input_dims["numeric"]
        input_numeric = keras.layers.Input(shape=(dim,), name="numeric")
        inputs.append(input_numeric)
        embeddings.append(input_numeric)

    for cfg_feat in config["features"]["tfidf"]:
        dim = input_dims[cfg_feat["name"]]
        input_tfidf = keras.layers.Input(shape=(dim,), name=cfg_feat["name"])
        inputs.append(input_tfidf)
        embeddings.append(input_tfidf)

    for f in config["features"]["onehot"]:
        dim = input_dims[f]
        input_onehot = keras.layers.Input(shape=(dim,), name=f)
        inputs.append(input_onehot)
        embeddings.append(input_onehot)

    for f in config["features"]["categorical"]:
        dim = input_dims[f]
        input_cat = keras.layers.Input(shape=[1], name=f)
        emb_output_dim = int(1.2 * dim ** 0.5)
        emb = keras.layers.Embedding(
            dim,
            emb_output_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-4))(input_cat)
        emb = keras.layers.Reshape([emb_output_dim])(emb)
        inputs.append(input_cat)
        embeddings.append(emb)

    x = keras.layers.Concatenate()(embeddings)

    x = keras.layers.Dense(2048)(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(1024)(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(config["model"]["dim_embedding"])(x)
    embedding = keras.layers.BatchNormalization()(x)

    model_emb = keras.Model(inputs, embedding)

    if only_embedding:
        return model_emb

    input_label = tf.keras.layers.Input(shape=(), name="label")
    inputs_with_label = inputs.copy() + [input_label]

    arcmargin = ArcMarginProductWarmup(
        n_classes=n_classes,
        s=config["model"]["arcmargin"]["s"],
        m=config["model"]["arcmargin"]["m"],
        m_init=config["model"]["arcmargin"]["m_init"],
        easy_margin=config["model"]["arcmargin"]["easy_margin"],
        ls_eps=config["model"]["arcmargin"]["ls_eps"],
        warmup_iters=config["model"]["arcmargin"]["warmup_iters"],
        l2=config["model"]["arcmargin"]["l2"],
        name="arcmargin_layer",
        dtype="float32")

    output = arcmargin([embedding, input_label])
    output = tf.keras.layers.Softmax(dtype="float32")(output)

    model = keras.Model(inputs_with_label, output)

    optimizer = keras.optimizers.Adam(
        learning_rate=build_scheduler(config["model"]["scheduler"]))

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        optimizer=optimizer)

    return model, model_emb
