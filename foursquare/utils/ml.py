from typing import List, Union
from typeguard import typechecked

import tensorflow.keras as keras
import tensorflow as tf


class ValidationLog(keras.callbacks.Callback):
    def __init__(
            self,
            model,
            name_metric,
            validation_data,
            validation_function,
            validation_function_kwargs=None,
            verbose=False):
        super(ValidationLog, self).__init__()
        self._supports_tf_logs = True

        self.model_ = model
        self.name_metric = name_metric
        self.x_valid, self.y_valid = validation_data
        self.validation_function = validation_function
        self.validation_function_kwargs = validation_function_kwargs or dict()
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = self.model_.predict(self.x_valid)
        score = self.validation_function(
            y_pred, self.y_valid, **self.validation_function_kwargs)
        if self.verbose:
            print(f"Epoch {epoch:5d}: validation score {score:.8f}")
        logs.update({self.name_metric: score})


def score_iou(df_true, df_pred):
    def iou(true, pred):
        true = set(true)
        pred = set(pred)
        return len(true & pred) / len(true | pred)

    df = df_true.merge(df_pred, on="id", suffixes=("_true", "_pred"))
    df["iou"] = df[["matches_true", "matches_pred"]].apply(
        lambda x: iou(x[0], x[1]), axis=1)
    return df["iou"].mean()


class MultiOptimizer(tf.keras.optimizers.Optimizer):
    """It's a copy of the class tfa.optimizers.MultiOptimizer with a little fix
    that allow its use with mixed precision.

    The fix is the addition of the keyword argument 'name' to the method
        apply_gradients.
    """
    @typechecked
    def __init__(
        self,
        optimizers_and_layers: Union[list, None] = None,
        optimizer_specs: Union[list, None] = None,
        name: str = "MultiOptimizer",
        **kwargs,
    ):

        super(MultiOptimizer, self).__init__(name, **kwargs)

        if optimizer_specs is None and optimizers_and_layers is not None:
            self.optimizer_specs = [
                self.create_optimizer_spec(optimizer, layers_or_model)
                for optimizer, layers_or_model in optimizers_and_layers
            ]

        elif optimizer_specs is not None and optimizers_and_layers is None:
            self.optimizer_specs = [
                self.maybe_initialize_optimizer_spec(spec) for spec in optimizer_specs
            ]

        else:
            raise RuntimeError(
                "Must specify one of `optimizers_and_layers` or `optimizer_specs`."
            )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """Wrapped apply_gradient method.
        Returns an operation to be executed.
        """

        for spec in self.optimizer_specs:
            spec["gv"] = []

        for grad, var in tuple(grads_and_vars):
            for spec in self.optimizer_specs:
                for name in spec["weights"]:
                    if var.name == name:
                        spec["gv"].append((grad, var))

        return tf.group(
            [
                spec["optimizer"].apply_gradients(spec["gv"], **kwargs)
                for spec in self.optimizer_specs
            ]
        )

    def get_config(self):
        config = super(MultiOptimizer, self).get_config()
        config.update({"optimizer_specs": self.optimizer_specs})
        return config

    @classmethod
    def create_optimizer_spec(
        cls,
        optimizer: tf.keras.optimizers.Optimizer,
        layers_or_model: Union[
            tf.keras.Model,
            tf.keras.Sequential,
            tf.keras.layers.Layer,
            List[tf.keras.layers.Layer],
        ],
    ):
        """Creates a serializable optimizer spec.
        The name of each variable is used rather than `var.ref()` to enable
        serialization and deserialization.
        """
        if isinstance(layers_or_model, list):
            weights = [
                var.name for sublayer in layers_or_model for var in sublayer.weights
            ]
        else:
            weights = [var.name for var in layers_or_model.weights]

        return {
            "optimizer": optimizer,
            "weights": weights,
        }

    @classmethod
    def maybe_initialize_optimizer_spec(cls, optimizer_spec):
        if isinstance(optimizer_spec["optimizer"], dict):
            optimizer_spec["optimizer"] = tf.keras.optimizers.deserialize(
                optimizer_spec["optimizer"]
            )

        return optimizer_spec

    def __repr__(self):
        return "Multi Optimizer with %i optimizer layer pairs" % len(
            self.optimizer_specs
        )
