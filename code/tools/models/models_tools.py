"""
Ensemble model
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import sys
import numpy as np
from typing import List
import tensorflow as tf
import keras
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import tools.prediction_quantiles_tools as qt
from tools.models.DNN import DNNRegressor
from tools.models.QNBM import QNBMRegressor


def get_model_class_from_conf(conf):
    """
    Map the model class depending on the config name
    """
    if conf == 'DNN':
        model_class = DNNRegressor
    elif conf == 'QNBM':
        model_class = QNBMRegressor
    else:
        sys.exit('ERROR: unknown model_class')
    return model_class


class PinballLoss(keras.losses.Loss):
    def __init__(self, quantiles: List, name="pinball_loss"):
        super().__init__(name=name)
        # quantiles 是 QR-DNN 要同时学习的目标分位数列表。
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        # 对每个分位数分别计算 pinball loss，再对所有分位数取平均。
        loss = []
        for i, q in enumerate(self.quantiles):
            error = tf.subtract(y_true, y_pred[:, :, i])
            loss_q = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            loss.append(loss_q)
        L = tf.convert_to_tensor(loss)
        total_loss = tf.reduce_mean(L)
        return total_loss

    def get_config(self):
        return {
            "num_quantiles": self.quantiles,
            "name": self.name,
        }


def regression_model(settings, sample_x):
    """
    Wrapper to the regression model
    :param settings: model configurations, sample_x: input sample to derive the model input shape (first dimension has to be 1)
    :param sample_x: input sample, to derive the model input shape (first dimension has to be 1)
    :return: instantiated model
    """
    # Currently direct link to TF, Future dev pytorch
    # 外部统一调用 regression_model；当前实际返回 TensorflowRegressor。
    return TensorflowRegressor(settings=settings, sample_x=sample_x)


class TensorflowRegressor():
    """
    Implementation of the Tenforflow regressor
    """
    def __init__(self, settings, sample_x):
        self.settings = settings
        self.x_columns_names = settings['x_columns_names']
        self.pred_horiz = settings['pred_horiz']

        tf.keras.backend.clear_session()
        # Map the loss to be used
        # 不同 PF_method 的训练目标不同：QR 用 pinball，point 用 MAE，分布式模型用负对数似然。
        if settings['PF_method'] == 'qr':
            loss = PinballLoss(quantiles=settings['target_quantiles'])
        elif settings['PF_method']=='point':
            loss = 'mae'
        elif (settings['PF_method'] == 'Normal'
              or settings['PF_method'] == 'JSU'
              or settings['PF_method'] == 'STU'
        ):
            loss = lambda y, rv_y: -rv_y.log_prob(y)
        else:
            sys.exit('ERROR: unknown PF_method config!')

        # Instantiate the model
        if settings['model_class'] in ('DNN', 'QNBM'):
            if settings['model_class'] == 'QNBM' and settings['PF_method'] != 'qr':
                raise ValueError('QNBM supports PF_method="qr" only')
            # get input size for the chosen model architecture
            # 用一个样本实际跑输入构造函数，自动推断全连接网络输入维度。
            settings['input_size']=DNNRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1]
            # Build the model architecture
            model_class = get_model_class_from_conf(settings['model_class'])
            self.regressor = model_class(settings, loss)

        else:
            sys.exit('ERROR: unknown model_class')

        # Map handler to convert distributional output to quantiles or distribution parameters
        # 预测输出先被转换成统一的 ensemble 可拼接结构，再由 Ensemble 构造最终分位数。
        if (settings['PF_method'] == 'Normal'):
            self.output_handler = self.__pred_Normal_params__
        elif settings['PF_method'] == 'JSU':
            self.output_handler = self.__pred_JSU_params__
        elif settings['PF_method'] == 'STU':
            self.output_handler = self.__pred_STU_params__
        else:
            self.output_handler =self.__quantiles_out__

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None, plot_history=False):
        # 保留plot_history参数兼容既有调用；训练曲线绘制已清理，完整history仍返回。
        history = self.regressor.fit(train_x, train_y, val_x, val_y,
                                     verbose=verbose, pruning_call=pruning_call)
        return history

    def predict(self, x):
        # 底层模型输出可能是张量或分布对象，这里统一经过 output_handler 转换。
        return self.output_handler(self.regressor.predict(x))

    def evaluate(self, x, y):
        return self.regressor.evaluate(x=x, y=y)

    def __pred_Normal_params__(self, pred_dists: tfp.distributions):
        # Normal 输出 loc/scale，形状扩展为 [样本, 小时, ensemble维, 参数]。
        loc = tf.expand_dims(pred_dists.loc, axis=-1)
        scale = tf.expand_dims(pred_dists.scale, axis=-1)
        # Expand dimension to enable concat in ensemble
        return tf.expand_dims(tf.concat([loc,scale], axis=-1), axis=2)

    def __pred_JSU_params__(self, pred_dists: tfp.distributions):
        # JohnsonSU 输出 loc/scale/tailweight/skewness 四个参数。
        loc = tf.expand_dims(pred_dists.loc, axis=-1)
        scale = tf.expand_dims(pred_dists.scale, axis=-1)
        tailweight = tf.expand_dims(pred_dists.tailweight, axis=-1)
        skewness = tf.expand_dims(pred_dists.skewness, axis=-1)
        # Expand dimension to enable concat in ensemble
        return tf.expand_dims(tf.concat([loc,scale,tailweight,skewness], axis=-1), axis=2)

    def __pred_STU_params__(self, pred_dists: tfp.distributions):
        # StudentT 输出 loc/scale/df 三个参数。
        loc = tf.expand_dims(pred_dists.loc, axis=-1)
        scale = tf.expand_dims(pred_dists.scale, axis=-1)
        df = tf.expand_dims(pred_dists.df, axis=-1)
        # Expand dimension to enable concat in ensemble
        return tf.expand_dims(tf.concat([loc, scale, df], axis=-1), axis=2)

    def __quantiles_out__(self, preds):
        # Expand dimension to enable concat in ensemble
        # point 和 QR 的输出已经是数值预测，只额外添加 ensemble 维。
        return tf.expand_dims(preds, axis=2)


class Ensemble():
    """
    Tensorflow ensemble wrapper
    """
    def __init__(self, settings):
        # store configs for internal use
        self.settings = settings
        # map the methods to use for aggretation and quantile building depending on the configs
        # 点预测/QR 直接聚合数值分位数；概率分布模型先聚合分布参数，再采样生成分位数。
        if (self.settings['PF_method'] == 'point'):
            self.ensemble_aggregator = self.__aggregate_de_quantiles__
            self._build_test_PIs = self.__get_qr_PIs__
        elif (self.settings['PF_method'] == 'qr'):
            self.ensemble_aggregator = self.__aggregate_de_quantiles__
            self._build_test_PIs = self.__get_qr_PIs__
        elif (self.settings['PF_method'] == 'Normal'):
            self.ensemble_aggregator = self.__aggregate_de__
            self._build_test_PIs = self.__build_Normal_PIs__
        elif (self.settings['PF_method'] == 'JSU'):
            self.ensemble_aggregator = self.__aggregate_de__
            self._build_test_PIs = self.__build_JSU_PIs__
        elif (self.settings['PF_method'] == 'STU'):
            self.ensemble_aggregator = self.__aggregate_de__
            self._build_test_PIs = self.__build_STU_PIs__
        else:
            sys.exit('ERROR: Ensemble config not supported!')

    def aggregate_preds(self, ens_comp_preds):
        # link function to the specific aggregator
        # 根据初始化时绑定的聚合函数处理多个 ensemble 组件的预测。
        return self.ensemble_aggregator(ens_comp_preds=ens_comp_preds)

    def get_preds_test_quantiles(self, preds_test):
        # link function to the specific PI builder
        # 聚合后统一转成目标分位数矩阵，供主流程保存。
        return self._build_test_PIs(preds_test=preds_test, settings=self.settings)

    @staticmethod
    def __aggregate_de__(ens_comp_preds):
        # aggregate by concatenation, for point a distributional settings
        # 分布式模型需要保留每个 ensemble 组件的分布参数，因此沿 ensemble 维拼接。
        return np.concatenate(ens_comp_preds, axis=2)

    @staticmethod
    def __aggregate_de_quantiles__(ens_comp_preds):
        # aggregate by a uniform vincentization
        # 对已经是点预测/分位数预测的输出，采用简单平均进行 uniform vincentization。
        return np.mean(np.concatenate(ens_comp_preds, axis=2), axis=2)

    @staticmethod
    def __get_qr_PIs__(preds_test, settings):
        # simply flatten in temporal dimension
        # QR/point 输出已是 [样本, 24小时, 分位数]，只需压平样本和小时维。
        return preds_test.reshape(-1, preds_test.shape[-1])

    @staticmethod
    def __build_Normal_PIs__(preds_test, settings):
        # for each de component, sample, aggregate samples and compute quantiles
        # 对每个 ensemble 组件的 Normal 分布采样，再在混合样本上取目标分位数。
        pred_samples = []
        for k in range(preds_test.shape[2]):
            pred_samples.append(tfd.Normal(
                loc=preds_test[:,:,k,0],
                scale=preds_test[:,:,k,1]).sample(10000).numpy())
        return np.transpose(np.quantile(np.concatenate(pred_samples, axis=0),
                                        q=settings['target_quantiles'], axis=0),
                            axes=(1, 2, 0)).reshape(-1, len(settings['target_quantiles']))

    @staticmethod
    def __build_JSU_PIs__(preds_test, settings):
        # for each de component, sample, aggregate samples and compute quantiles
        # JohnsonSU 同样通过 Monte Carlo 采样得到预测分布的经验分位数。
        pred_samples = []
        for k in range(preds_test.shape[2]):
            pred_samples.append(tfd.JohnsonSU(
                loc=preds_test[:,:,k,0],
                scale=preds_test[:,:,k,1],
                tailweight=preds_test[:,:,k,2],
                skewness=preds_test[:,:,k,3]).sample(10000).numpy())
        return np.transpose(np.quantile(np.concatenate(pred_samples, axis=0),
                                        q=settings['target_quantiles'], axis=0),
                            axes=(1, 2, 0)).reshape(-1, len(settings['target_quantiles']))

    @staticmethod
    def __build_STU_PIs__(preds_test, settings):
        # for each de component, sample, aggregate samples and compute quantiles
        # StudentT 的 ensemble 分位数也通过采样多个组件后的混合分布得到。
        pred_samples = []
        for k in range(preds_test.shape[2]):
            pred_samples.append(tfd.StudentT(
                loc=preds_test[:, :, k, 0],
                scale=preds_test[:, :, k, 1],
                df=preds_test[:, :, k, 2]).sample(10000).numpy())
        return np.transpose(np.quantile(np.concatenate(pred_samples, axis=0),
                                        q=settings['target_quantiles'], axis=0),
                            axes=(1, 2, 0)).reshape(-1, len(settings['target_quantiles']))
