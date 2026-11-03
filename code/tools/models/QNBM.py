"""Additive quantile neural basis model, adapted for TensorFlow 2.13.

Architecture adapted from Alessandro Brusaferri's ``DQNBM.py`` (full basis):
https://github.com/bruale/PF_DQ-NBM/blob/main/DQNBM.py

This is a project adapter, not a reproduction of the author's experiments.
The deterministic full-basis calculation is evaluated without tiling basis
activations across all hours and quantiles. Dropout therefore shares the
sample/feature basis realization across outputs during training. As in the
author's QR code, loss is applied to raw quantile slots; the old QR-DNN instead
sorts inside its training graph. Public predict/evaluate use sorted outputs.
RevIN and masked bases are deliberately outside this first experiment.

MIT License

Copyright (c) 2025 Alessandro Brusaferri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import tempfile

import numpy as np
import tensorflow as tf

from tools.models.DNN import DNNRegressor


class FeatureShapes(tf.keras.layers.Layer):
    """Give each scalar feature its own combination of the shared bases."""

    def __init__(self, rank, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank

    def build(self, input_shape):
        self.factor_features = self.add_weight(
            name='factor_features', shape=(int(input_shape[-2]), self.rank),
            initializer='glorot_uniform')
        self.factor_bases = self.add_weight(
            name='factor_bases', shape=(self.rank, int(input_shape[-1])),
            initializer='glorot_uniform')
        super().build(input_shape)

    def call(self, inputs):
        weights = tf.matmul(self.factor_features, self.factor_bases)
        return tf.reduce_sum(inputs * weights, axis=-1)


class AdditiveProjection(tf.keras.layers.Layer):
    """Low-rank feature-to-(hour,quantile) projection with output bias."""

    def __init__(self, output_size, rank, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.rank = rank

    def build(self, input_shape):
        self.factor_outputs = self.add_weight(
            name='factor_outputs', shape=(self.output_size, self.rank),
            initializer='glorot_uniform')
        self.factor_features = self.add_weight(
            name='factor_features', shape=(self.rank, int(input_shape[-1])),
            initializer='glorot_uniform')
        self.output_bias = self.add_weight(
            name='output_bias', shape=(self.output_size,), initializer='zeros')
        super().build(input_shape)

    @property
    def kernel(self):
        return tf.matmul(self.factor_outputs, self.factor_features)

    def call(self, inputs):
        # Associate products to retain low-rank compute during training/prediction.
        latent = tf.matmul(inputs, self.factor_features, transpose_b=True)
        return tf.matmul(latent, self.factor_outputs, transpose_b=True) + self.output_bias


class QNBMRegressor(DNNRegressor):
    """QR-only additive adapter using exactly the existing DNN input ordering.

    ``explain`` reconstructs the raw model in standardized target units. Sorting
    can swap quantile labels and is not additive; centered contributions do not
    explain sorted, conformally adjusted, or inverse-scaled predictions by
    themselves. A caller must use one common reference for an ensemble and apply
    the target scale to all contributions, adding location only to the reference.
    """

    DEFAULTS = {
        'hidden_size': 64, 'n_hidden_layers': 2, 'rank_W': 16,
        'dropout': 0.1, 'activation': 'relu', 'lr': 1e-3,
        'perform_RevIN': False,
    }

    def __init__(self, settings, loss):
        self.settings = dict(self.DEFAULTS, **settings)
        if self.settings.get('PF_method') != 'qr':
            raise ValueError('QNBM supports PF_method="qr" only')
        if self.settings['perform_RevIN']:
            raise ValueError('RevIN is disabled in the first QNBM protocol')
        for key in ('input_size', 'pred_horiz', 'hidden_size', 'n_hidden_layers', 'rank_W'):
            value = self.settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(key + ' must be a positive integer')
        quantiles = np.asarray(self.settings['target_quantiles'], dtype=float)
        if (quantiles.ndim != 1 or len(quantiles) == 0 or
                not np.isfinite(quantiles).all() or
                not ((quantiles > 0) & (quantiles < 1)).all() or
                not (np.diff(quantiles) > 0).all()):
            raise ValueError('target_quantiles must be strictly increasing inside (0, 1)')
        if not 0 <= self.settings['dropout'] < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.loss = loss
        self.feature_names = ['feature_' + str(i) for i in range(self.settings['input_size'])]
        self._build_model()

    def _build_model(self):
        x_in = tf.keras.Input(shape=(self.settings['input_size'],), name='x_in')
        bases = tf.keras.layers.Reshape((self.settings['input_size'], 1))(x_in)
        for depth in range(self.settings['n_hidden_layers']):
            bases = tf.keras.layers.Dense(
                self.settings['hidden_size'], activation=self.settings['activation'],
                name='shared_basis_' + str(depth))(bases)
            bases = tf.keras.layers.Dropout(
                self.settings['dropout'], name='basis_dropout_' + str(depth))(bases)
        shapes = FeatureShapes(self.settings['rank_W'], name='features_shape_fun')(bases)
        self.shape_model = tf.keras.Model(x_in, shapes, name='qnbm_feature_shapes')
        output_size = self.settings['pred_horiz'] * len(self.settings['target_quantiles'])
        self.projection = AdditiveProjection(
            output_size, self.settings['rank_W'], name='pred_fun')
        raw = self.projection(shapes)
        raw = tf.keras.layers.Reshape(
            (self.settings['pred_horiz'], len(self.settings['target_quantiles'])),
            name='raw_quantiles')(raw)
        self.model = tf.keras.Model(x_in, raw, name='qnbm')
        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.settings['lr']), loss=self.loss)

    def _flatten(self, x):
        x = np.asarray(x)
        if x.ndim == 3:
            if x.shape[-1] != len(self.settings['x_columns_names']):
                raise ValueError('input columns do not match x_columns_names')
            columns = self.settings['x_columns_names']
            horizon = self.settings['pred_horiz']
            if x.shape[1] <= horizon:
                raise ValueError('a history window before the prediction horizon is required')
            # Derive names through the same converter, including all DE-only slices.
            labels = np.asarray([
                [column + '[t{:+d}]'.format(t - (x.shape[1] - horizon))
                 for column in columns] for t in range(x.shape[1])])[None, ...]
            names = self.build_model_input_from_series(labels, columns, horizon)[0].tolist()
            x = self.build_model_input_from_series(x, columns, horizon)
            if len(names) == self.settings['input_size']:
                self.feature_names = names
        if x.ndim != 2 or x.shape[1] != self.settings['input_size'] or x.shape[0] == 0:
            raise ValueError('input must contain samples with the configured flattened size')
        x = np.asarray(x, dtype='float32')
        if not np.isfinite(x).all():
            raise ValueError('model input must be finite')
        return x

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
        train_x, val_x = self._flatten(train_x), self._flatten(val_x)
        train_y, val_y = np.asarray(train_y), np.asarray(val_y)
        for x, y in ((train_x, train_y), (val_x, val_y)):
            if y.shape != (len(x), self.settings['pred_horiz']) or not np.isfinite(y).all():
                raise ValueError('targets must be finite [sample, horizon] arrays')
        if self.settings['max_epochs'] <= 0:
            raise ValueError('max_epochs must be positive')
        # Every fit owns a private directory, including when pruning raises.
        with tempfile.TemporaryDirectory(prefix='qnbm-checkpoint-',
                                         dir=self.settings.get('checkpoint_root')) as folder:
            path = os.path.join(folder, 'best.weights.h5')
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=self.settings['patience']),
                tf.keras.callbacks.ModelCheckpoint(
                    path, monitor='val_loss', mode='min', save_best_only=True,
                    save_weights_only=True),
            ]
            if pruning_call is not None:
                callbacks.append(pruning_call)
            history = self.model.fit(
                train_x, train_y, validation_data=(val_x, val_y),
                epochs=self.settings['max_epochs'], batch_size=self.settings['batch_size'],
                callbacks=callbacks, verbose=verbose)
            if not os.path.isfile(path):
                raise FloatingPointError('training produced no finite validation checkpoint')
            self.model.load_weights(path)
        return history

    def predict_raw(self, x):
        return self.model(self._flatten(x), training=False)

    def predict(self, x):
        return tf.sort(self.predict_raw(x), axis=-1)

    def evaluate(self, x, y):
        # HPO measures the public sorted forecast, not the raw training objective.
        predictions = self.predict(x)
        targets = tf.cast(tf.convert_to_tensor(y), predictions.dtype)
        return float(self.loss(targets, predictions).numpy())

    def explain(self, x, reference):
        """Center each scalar shape at one shared flat [1,F] reference.

        raw = reference_prediction + contributions.sum(axis=-1), up to float32
        arithmetic. Sorting must be diagnosed and recorded separately.
        """
        x = self._flatten(x)
        reference = np.asarray(reference)
        if reference.ndim != 2 or reference.shape != (1, self.settings['input_size']):
            raise ValueError('reference must be a common flattened array of shape [1,F]')
        reference = self._flatten(reference)
        centered_shapes = (self.shape_model(x, training=False) -
                           self.shape_model(reference, training=False))
        contributions = tf.einsum('bf,of->bof', centered_shapes, self.projection.kernel)
        contributions = tf.reshape(contributions, (
            len(x), self.settings['pred_horiz'],
            len(self.settings['target_quantiles']), self.settings['input_size']))
        return {
            'raw': self.model(x, training=False).numpy(),
            'reference_prediction': self.model(reference, training=False).numpy(),
            'contributions': contributions.numpy(),
            'feature_names': list(self.feature_names),
        }

    @staticmethod
    def get_hyperparams_trial(trial, settings):
        settings['hidden_size'] = trial.suggest_categorical('hidden_size', [32, 64, 128])
        settings['rank_W'] = trial.suggest_categorical('rank_W', [8, 16, 32])
        settings['lr'] = trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2])
        settings['dropout'] = trial.suggest_categorical('dropout', [0., 0.1, 0.3])
        settings.setdefault('n_hidden_layers', 2)
        settings.setdefault('activation', 'relu')
        settings['perform_RevIN'] = False
        return settings

    @staticmethod
    def get_hyperparams_searchspace():
        return {'hidden_size': [32, 64, 128], 'rank_W': [8, 16, 32],
                'lr': [1e-4, 1e-3, 1e-2], 'dropout': [0., 0.1, 0.3]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        return {key: configs.get(key, default) for key, default in QNBMRegressor.DEFAULTS.items()}
