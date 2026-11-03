"""
Utility functions for quantile prediction
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license


import numpy as np
from typing import Dict, List
from sklearn.linear_model import QuantileRegressor
import sys


def build_alpha_quantiles_map(target_alpha: List, target_quantiles: List):
    """
    Build the map between PIs coverage levels and related quantiles
    """
    # alpha 表示失覆盖率；例如 alpha=0.2 对应 80% 区间，左右分位数是 0.1 和 0.9。
    # 返回值保存每个 alpha 对应的下界/上界分位数列索引，同时记录中位数列索引。
    alpha_q = {'med': target_quantiles.index(0.5)}
    for alpha in target_alpha:
        alpha_q[alpha] = {
            'l': target_quantiles.index(alpha / 2),
            'u': target_quantiles.index(1 - alpha / 2),
        }
    return alpha_q

def build_target_quantiles(target_alpha):
    """
    Build target quantiles from the alpha list
    """
    # 所有方法统一输出同一组分位数：先加入中位数，再为每个 alpha 加入上下界分位数。
    target_quantiles = [0.5]
    for alpha in target_alpha:
        target_quantiles.append(alpha / 2)
        target_quantiles.append(1 - alpha / 2)
    target_quantiles.sort()
    return target_quantiles


def fix_quantile_crossing(preds: np.array):
    """
    Fix crossing in the predicted quantiles by means of post-hoc sorting
    """
    # 对最后一维排序，保证低分位数不高于高分位数，避免预测区间上下界交叉。
    return np.sort(preds, axis=-1)




def compute_qra(preds_cali: np.array, y_cali: np.array, preds_test: np.array, settings: Dict):
    """
    Compute PI using Quantile Regression Averaging.

    Python implementation, following: https://ideas.repec.org/c/wuu/hscode/m14003.html

    :param preds_cali: predictions of the ensembles over the calibration set
    :param y_cali: true labels over the calibration set
    :param preds_test: predictions over the test set
    :param settings: configurations
    :return: PIs
    """

    # Prepare the data in the proper format
    # 输入通常是 [样本数, 预测步长, ensemble 数]，这里拉平成逐小时样本用于 sklearn 的分位数回归。
    y_cali = y_cali.flatten()
    ensemble_calib_preds_fl = preds_cali.reshape(-1, preds_cali.shape[-1])
    ensemble_test_preds_fl = preds_test.reshape(-1, preds_test.shape[-1])

    # Compute PIs for each quantile in the target quantile list
    # QRA 对每个目标分位数单独拟合一个线性分位数回归，把多个 ensemble 点预测组合成分位数预测。
    pis = []
    for q_i in settings['target_quantiles']:
        # Instantiate class to compute lower/upper PI components
        # In sklear QR alpha defines the weights in the l1 norm of the regression (set to 0 to obtain standard regression)
        # Here alpha refers to the miscoverage rate, as usually adopted in CP
        qr = QuantileRegressor(quantile=q_i, alpha=0, solver='highs')
        # Fit QR for the current quantile using ensemble predictions (and labels) on the calibration set
        qr.fit(ensemble_calib_preds_fl, y_cali)
        # Predict quantiles for the test sample, including reshape for concatenation
        pis.append(qr.predict(ensemble_test_preds_fl).reshape(-1, preds_test.shape[1], 1))

    # Concatenate the PIs
    # 还原为 [测试样本数, 预测步长, 分位数数]，便于后续与其他方法统一处理。
    test_PIs = np.concatenate(pis, axis=-1)

    # Fix quantile crossing by sorting
    # return prediction flattened in temporal dimension (sample over pred horizon)
    return fix_quantile_crossing(test_PIs.reshape(-1, test_PIs.shape[-1]))


def exec_cqr(preds_cali: np.array, y_cali: np.array, preds_test: np.array, settings: Dict):
    """
       Compute conformalized prediction intervals:
    """

    # Each calibration owns its output: callers may reuse the raw predictions
    # for another calibration method or for the uncalibrated baseline.
    preds_test = np.array(preds_test, dtype=float, copy=True)

    def __asym_mode__(conf_scores: np.array, alpha:float):
        """
        Function to compute asymmetric cqr
        """
        # 非对称 CQR 分别估计下界和上界的校正量，不强制两侧误差使用同一个半径。
        q=(1 - alpha / 2)
        Q_l = np.quantile(a=conf_scores[:, :, 0], q=q, axis=0)
        Q_h = np.quantile(a=conf_scores[:, :, 1], q=q, axis=0)
        return np.stack([Q_l, Q_h], axis=-1)

    def __conform_prediction_intervals__(pred_pi: np.array, q_cp: np.array):
        """
        Function to compute conformalized PIs
        """
        # 下界向下扩张 q_cp[:, 0]，上界向上扩张 q_cp[:, 1]，从而提高覆盖率。
        return np.concatenate([np.expand_dims(pred_pi[:, :, 0] - q_cp[:, 0], axis=2),
                               np.expand_dims(pred_pi[:, :, 1] + q_cp[:, 1], axis=2)], axis=2)

    # map the method to be employed to conformalize
    compute_score_quantiles = __asym_mode__

    # Conformalize predictions for each alpha
    for alpha in settings['target_alpha']:
        # get index of the lower/upper quantiles for the current alpha from the map
        lq_idx = settings['q_alpha_map'][alpha]['l']
        uq_idx = settings['q_alpha_map'][alpha]['u']

        # Compute conformity scores
        # conf_scores 的两列分别衡量“真实值低于下界”和“真实值高于上界”的偏差。
        conf_scores = np.stack([preds_cali[:, :, lq_idx] - y_cali, y_cali - preds_cali[:, :, uq_idx]], axis=-1)
        conformalized_alpha_PIs= __conform_prediction_intervals__(pred_pi=preds_test[:,:,[lq_idx, uq_idx]],
                                                                  q_cp=compute_score_quantiles(conf_scores=conf_scores,
                                                                                               alpha=alpha))
        # replace test_pred PIs columns with conformalized PIs for each alpha
        preds_test[:, :, lq_idx] = conformalized_alpha_PIs[:,:,0]
        preds_test[:, :, uq_idx] = conformalized_alpha_PIs[:,:,1]

    # Fix quantile crossing
    # return prediction flattened in temporal dimension (sample over pred horizon)
    return fix_quantile_crossing(preds_test.reshape(-1, preds_test.shape[-1]))


def exec_cp(preds_cali: np.array, y_cali: np.array, preds_test: np.array, settings: Dict):
    # 普通 CP 从点预测出发，用校准集绝对残差的经验分位数扩张成上下预测区间。
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0]>1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    n=conf_score.shape[0]
    # Stack the quantiles to the point pred for each alpha (next sorted by fixing crossing)
    # 第一列保留点预测/中位数；后续每个 alpha 追加一组下界和上界。
    preds_test_q=[preds_test]
    for alpha in settings['target_alpha']:
        # conformal 分位数采用有限样本校正 ceil((n+1)*(1-alpha))/n。
        q = np.ceil((n + 1) * (1 - alpha)) / n
        Q_1_alpha= np.expand_dims(np.quantile(a=conf_score, q=q, axis=0, method='higher'), axis=(0,-1))
        # Append lower/upper PIs for the current alpha
        preds_test_q.append(preds_test - Q_1_alpha)
        preds_test_q.append(preds_test + Q_1_alpha)
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing
    # return prediction flattened in temporal dimension (sample over pred horizon)
    return fix_quantile_crossing(preds_test_q.reshape(-1, preds_test_q.shape[-1]))
