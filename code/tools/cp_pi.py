# Modified for standalone release: original formulas retained.
import numpy as np
from tqdm import tqdm


"""
Code from https://github.com/aangelopoulos/conformal-time-series.git
"""

# 本文件实现在线共形预测中的 PID/积分校准逻辑，用于根据历史覆盖误差动态调整区间宽度。
# 入口函数是 cts_pid，QuantProc.__compute_pid__ 会按 24 个预测小时分别调用它。

def mytan(x):
    # 对 tan 做边界保护，避免积分项进入 tan 的垂直渐近线后产生数值异常。
    if x >= np.pi/2:
        return np.infty
    elif x <= -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    # 对累计覆盖误差做饱和变换；log(t+1) 会让长期积分项增长更平滑。
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out

def saturation_fn_sqrt(x, t, Csat, KI):
    # 备用的平方根尺度饱和函数，当前主流程使用的是 saturation_fn_log。
    return KI * mytan((x * np.sqrt(t+1))/((Csat * (t+1))))

def quantile_integrator_log(
    scores,
    alpha,
    lr,
    Csat,
    KI,
    ahead,
    T_burnin,
    proportional_lr=True,
    *args,
    **kwargs
):
    # 包一层 log 版本接口，保持与原 conformal-time-series 项目的方法命名一致。
    data = kwargs['data'] if 'data' in kwargs.keys() else None
    results = quantile_integrator(scores, alpha, lr, data, T_burnin, Csat, KI, True, ahead, proportional_lr=proportional_lr)
    results['method'] = "Quantile+Integrator (log)"
    return results


"""
    This is the master method for the quantile, integrator
"""
def quantile_integrator(
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    Csat,
    KI,
    upper,
    ahead,
    integrate=True,
    proportional_lr=True,
    *args,
    **kwargs
):
    # Initialization
    # scores 是每个时间点的 conformity score；q/qts/integrators 分别保存最终分位数、梯度更新分量和积分分量。
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    seasonal_period = kwargs.get('seasonal_period')
    if seasonal_period is None:
        seasonal_period = 1

    # Run the main loop
    # At time t, we observe y_t and make a prediction for y_{t+ahead}
    # We also update the quantile at the next time-step, q[t+1], based on information up to and including t_pred = t - ahead + 1.
    #lr_t = lr * (scores[:T_burnin].max() - scores[:T_burnin].min()) if proportional_lr and T_burnin > 0 else lr
    for t in tqdm(range(T_test)):
        # 使用最近 T_burnin 个 score 的范围缩放学习率，使不同小时/市场的误差尺度更可比。
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived
        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        # 如果上一可评估预测没有覆盖，则梯度为负，下一步会增大 q；覆盖过多时则减小 q。
        grad = alpha if covereds[t_pred] else -(1-alpha)
        integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)

        # Update the next quantile
        # 最终 q = 梯度分量 qts + 积分分量 integrators，用来控制未来区间扩张幅度。
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = integrator if integrate else 0
            qs[t+1] = qts[t+1] + integrators[t+1]
    results = {"method": "Quantile+Integrator (log)", "q" : qs}
    return results


def cts_pid(data, alpha, lr, Csat, KI, T_burnin, score_function_name="cqr-asymmetric", ahead=1, minsize=0):
    # data 需要包含 y 和 forecasts；forecasts 是每个时点的 [下界, 上界]。
    fn = quantile_integrator_log
    kwargs = {'Csat': Csat, 'KI': KI, "T_burnin": T_burnin, "data": data, "seasonal_period": None,
              "ahead": ahead}
    # Initialize the score function
    if score_function_name == "cqr-symmetric":
        # 对称模式用一个 score 同时扩张上下界。
        def score_function(y, forecasts):
            return np.maximum(forecasts[0] - y, y - forecasts[-1])

        def set_function(forecast, q):
            return np.array([forecast[0] - q, forecast[-1] + q])

        asymmetric = False
    elif score_function_name == "cqr-asymmetric":
        # 非对称模式分别计算下界和上界的 score，更适合上下尾误差不对称的电价分布。
        def score_function(y, forecasts):
            return np.array([forecasts[0] - y, y - forecasts[-1]])

        def set_function(forecast, q):
            return np.array([forecast[0] - q[0], forecast[-1] + q[1]])

        asymmetric = True
    else:
        raise ValueError("Invalid score function name")

    # Compute scores
    # 如果调用方没有预先给 scores，就根据真实值和预测区间现场计算 conformity score。
    if 'scores' not in data.columns:
        data['scores'] = [score_function(y, forecast) for y, forecast in zip(data['y'], data['forecasts'])]

    # Compute the results
    results = {}
    if asymmetric:
        # 非对称时，对下尾和上尾分别运行一个在线分位数积分器。
        stacked_scores = np.stack(data['scores'].to_list())
        kwargs['upper'] = False
        q0 = fn(stacked_scores[:, 0], alpha / 2, lr, **kwargs)['q']
        kwargs['upper'] = True
        q1 = fn(stacked_scores[:, 1], alpha / 2, lr, **kwargs)['q']
        q = [np.array([q0[i], q1[i]]) for i in range(len(q0))]
    else:
        # 对称时只运行一次积分器，得到统一扩张半径。
        kwargs['upper'] = True
        q = fn(data['scores'].to_numpy(), alpha, lr, **kwargs)['q']

    sets = [set_function(data['forecasts'].interpolate().to_numpy()[i], q[i]) for i in range(len(q))]
    # Make sure the set size is at least minsize by setting sets[j][0] = min(sets[j][0], sets[j][1]-minsize) and sets[j][1] = max(sets[j][1], sets[j][1]+minsize)
    # minsize 防止区间退化为过窄区间；默认 0 表示不额外限制最小宽度。
    sets = [np.array([np.minimum(sets[j][0], sets[j][1] - minsize), np.maximum(sets[j][1], sets[j][0] + minsize)])
            for j in range(len(sets))]

    return {"q": q, "sets": sets}
