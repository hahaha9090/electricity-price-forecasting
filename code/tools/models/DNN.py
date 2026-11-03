"""
DNN model class
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

# Distributional NNs following: https://github.com/gmarcjasz/distributionalnn.git

from tools.data_utils import features_keys
import numpy as np
import tensorflow as tf
from typing import List
import sys
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import os
import shutil


class DNNRegressor:
    def __init__(self, settings, loss):
        # settings 保存模型结构、训练超参数、预测步长和 PF_method；loss 由外层根据 PF_method 传入。
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):
        # 输入已经由 build_model_input_from_series 拉平成二维特征，因此这里使用普通全连接网络。
        x_in = tf.keras.layers.Input(shape=(self.settings['input_size']))
        x_in = tf.keras.layers.BatchNormalization()(x_in)
        x = (tf.keras.layers.Dense(self.settings['hidden_size'],
                                  activation=self.settings['activation'],
                                  )(x_in))
        for hl in range(self.settings['n_hidden_layers'] - 1):
            x = tf.keras.layers.Dense(self.settings['hidden_size'],
                                        activation=self.settings['activation'],
                                        )(x)
        if self.settings['PF_method'] == 'point':
            # 点预测只输出每个预测小时的一个数，后处理 CP/QRA 会基于它构造区间。
            out_size = 1
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear',
                                          )(x)
            output = tf.reshape(logit, (-1, self.settings['pred_horiz'], out_size))

        elif self.settings['PF_method'] == 'qr':
            # 分位数回归直接输出每个预测小时的全部目标分位数。
            out_size = len(self.settings['target_quantiles'])
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear'
                                          )(x)
            output = tf.reshape(logit, (-1, self.settings['pred_horiz'], out_size))
            # Fix quantile crossing by sorting
            output = tf.keras.layers.Lambda(lambda x: tf.sort(x, axis=-1))(output)

        elif self.settings['PF_method'] == 'Normal':
            # Normal 模型输出每个小时的 loc 和 scale，并包装成 TensorFlow Probability 分布。
            out_size = 2
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear',
                                          )(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :self.settings['pred_horiz']],
                    scale=1e-3 + 3 * tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']:])))(logit)

        elif self.settings['PF_method'] == 'JSU':
            # JohnsonSU 分布可表达偏度和厚尾，适合电价这类非对称、重尾序列。
            out_size = 4
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear'
                                          )(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.JohnsonSU(
                    loc=t[..., :self.settings['pred_horiz']],
                    scale=1e-3 + 3 * tf.math.softplus(t[..., self.settings['pred_horiz']:self.settings['pred_horiz'] * 2]),
                    tailweight=1 + 3 * tf.math.softplus(t[..., self.settings['pred_horiz'] * 2:self.settings['pred_horiz'] * 3]),
                    skewness=t[..., self.settings['pred_horiz'] * 3:]))(logit)
        elif self.settings['PF_method'] == 'STU':
            # StudentT 分布输出 loc、scale、df，自由度 df 控制尾部厚度。
            out_size = 3
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear'
                                          )(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.StudentT(
                    loc=t[..., :self.settings['pred_horiz']],
                    scale=1e-3 + 3 * tf.math.softplus(
                        t[..., self.settings['pred_horiz']:self.settings['pred_horiz'] * 2]),
                    df=1 + 3 * tf.math.softplus(t[..., self.settings['pred_horiz'] * 2:])))(logit)
        else:
            sys.exit('ERROR: unknown PF_method config!')

        # Create model
        self.model= tf.keras.Model(inputs=[x_in], outputs=[output])
        # Compile the model
        # Adam 学习率由已调超参数控制，loss 与 PF_method 对应。
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']),
                           loss=loss
                           )

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
        # Convert the data into the input format using the internal converter
        # 原始窗口张量会在这里转换成 DNN 真正吃的扁平特征。
        train_x = self.build_model_input_from_series(x=train_x,
                                                     col_names=self.settings['x_columns_names'],
                                                     pred_horiz=self.settings['pred_horiz'])
        val_x = self.build_model_input_from_series(x=val_x,
                                                   col_names=self.settings['x_columns_names'],
                                                   pred_horiz=self.settings['pred_horiz'])

        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=self.settings['patience'],
                                              restore_best_weights=False)

        # Create folder to temporally store checkpoints
        # 训练过程中只临时保存验证集最优权重；加载后立即删除 tmp_checkpoints。
        checkpoint_path = os.path.join(os.getcwd(), 'tmp_checkpoints', 'cp.ckpt')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor="val_loss", mode="min",
                                                save_best_only=True,
                                                save_weights_only=True, verbose=0)

        if pruning_call==None:
            callbacks = [es, cp]
        else:
            callbacks = [es, cp, pruning_call]

        history = self.model.fit(train_x,
                                 train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'],
                                 callbacks=callbacks,
                                 verbose=verbose)

        # Load best weights: do not use restore_best_weights from early stop since works only in case it stops training
        self.model.load_weights(checkpoint_path)
        # delete temporary folder
        shutil.rmtree(checkpoint_dir)
        return history

    def predict(self, x):
        # 预测前使用同一套输入构造逻辑，保证训练和测试特征排列一致。
        x = self.build_model_input_from_series(x=x,
                                               col_names=self.settings['x_columns_names'],
                                               pred_horiz=self.settings['pred_horiz'])
        return self.model(x)

    def evaluate(self, x, y):
        # Optuna 调参时用验证集 loss 评估当前候选超参数。
        x = self.build_model_input_from_series(x=x,
                                               col_names=self.settings['x_columns_names'],
                                               pred_horiz=self.settings['pred_horiz'])
        return self.model.evaluate(x=x, y=y)

    @staticmethod
    def build_model_input_from_series(x, col_names: List, pred_horiz: int):
        """
        Defined depending on the task features and model processing
        """
        # get index of target and past features
        # 历史目标和历史外生变量作为 lag 输入，覆盖预测日前的历史窗口。
        past_col_idxs = [index for (index, item) in enumerate(col_names)
                         if features_keys['target'] in item or features_keys['past'] in item]

        # get index of const features
        # const 特征只取预测窗口起点附近的常量信息。
        const_col_idxs = [index for (index, item) in enumerate(col_names)
                          if features_keys['const'] in item]

        # get index of futu features
        # futu 特征表示预测未来 24 小时可提前知道的协变量。
        futu_col_idxs = [index for (index, item) in enumerate(col_names)
                         if features_keys['futu'] in item]

        # Specific features employed only in the DE dataset
        # get index of f_l-1 features
        f_l_1_col_idxs = [index for (index, item) in enumerate(col_names)
                          if features_keys['f_l-1'] in item]
        # get index of f_l-1 features
        const_l_2_col_idxs = [index for (index, item) in enumerate(col_names)
                              if features_keys['const_l-2'] in item]

        # build conditioning variables for past features
        past_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in past_col_idxs]
        # build conditioning variables for futu features
        futu_feat = [x[:, -pred_horiz:, feat_idx] for feat_idx in futu_col_idxs]
        # build conditioning variables for cal features
        c_feat = [x[:, -pred_horiz:-pred_horiz + 1, feat_idx] for feat_idx in const_col_idxs]

        c_l_2_feat = [x[:, 0:1, feat_idx] for feat_idx in const_l_2_col_idxs]
        f_l_1_feat = [x[:, -2 * pred_horiz:, feat_idx] for feat_idx in f_l_1_col_idxs]

        # return flattened input
        # 这些片段沿时间维拼接，形成每个样本的一条长特征向量。
        return np.concatenate(past_feat + futu_feat + c_feat + c_l_2_feat + f_l_1_feat, axis=1)


    @staticmethod
    def get_hyperparams_trial(trial, settings):
        # Optuna 每个 trial 会生成一组候选隐藏层宽度和学习率。
        settings['hidden_size'] = trial.suggest_int('hidden_size', 64, 960, step=64)
        settings['n_hidden_layers'] = 2
        settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        settings['activation'] = 'softplus'
        return settings

    @staticmethod
    def get_hyperparams_searchspace():
        # grid_search 模式下使用这个离散搜索空间。
        return {'hidden_size': [64, 128, 512, 640, 768, 896, 960],
                'lr': [1e-5, 1e-4, 1e-3]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        # 只抽取需要保存/复用的训练超参数，不把完整实验配置写进 tuned_hyperp。
        model_hyperparams = {
            'hidden_size': configs['hidden_size'],
            'n_hidden_layers': configs['n_hidden_layers'],
            'lr': configs['lr'],
            'activation': configs['activation']
        }
        return model_hyperparams
