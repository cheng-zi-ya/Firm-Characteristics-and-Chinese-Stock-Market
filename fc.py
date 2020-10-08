
"""
由于并未达到原文中的数值，所以之后会重新检查，看看算式能否进一步提升，因此，并未写出很集成的模块。
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

from factor_test_monthly import compute_num_months, compute_factor_return_series, compute_return_T_test, compute_5_factor_model

from fm import process_bar

import warnings
warnings.filterwarnings("ignore")


def forecast_combination(X, y):
    fc_params = []
    for i in range(X.shape[1]):
        if i == 0:  # 对于常数项
            # result = sm.OLS(y, X[:, i]).fit()
            # fc_params.append(result.params[0])
            fc_params.append(stats.linregress(y, X[:, 1])[1])
            # fc_params.append(stats.linregress(X[:, 1], y)[1])
        else:
            fc_params.append(stats.linregress(y, X[:, i])[0])
            # fc_params.append(stats.linregress(X[:, i], y)[0])
    return fc_params

if __name__ == "__main__":
    data = pd.read_csv('./data.csv')
    begin_month = 200203
    time_length = 190

    months = compute_num_months(begin_month, time_length)

    # 转为按时间排序
    data = data.sort_values(by = "TRDMNT")
    data = data.reset_index(drop = True)

    # 然后对于每一个时间节点，对于75个因子计算一次，得到其参数，
    for i in range(time_length):
        month = months[i]
        data_atmonth = data[data.TRDMNT == month]
        # 线性回归
        X = data_atmonth.iloc[:, 18:92].values
        # X = data_atmonth.iloc[:, 92:166].values
        X = np.column_stack((np.ones(X.shape[0]), X))       #先加上常数看看
        # y = data_atmonth.retx.values
        y = data_atmonth.reta.values
        fc_point = forecast_combination(X, y)
        if i == 0:
            fc_matrix = fc_point
        else:
            fc_matrix = np.vstack((fc_matrix, fc_point)) # 该矩阵和时间的对应关系为: 时间对应的那一行用到了下一个月的回报，
                                                            # 所以应该移动
    T = 12
    dates = data.TRDMNT.tolist()
    data_matrix = data.iloc[:, 18:92].values
    fc_data = []
    for i in range(len(data)):
        date = dates[i]
        if date >= months[T]:     # 因为需要前T个时间点的系数进行计算
            now_pos = int((date - 200200)/100) * 12 + date%100- 3
            fc_params = np.sum(fc_matrix[now_pos - T:now_pos, :], axis=0) / T
            fc_point = np.sum(fc_params * np.array([1] + list(data_matrix[i, :])))
            # fc_point = np.sum(np.array(list(fc_params[0, i] for i in range(74))) * np.array(data.iloc[i, 18:92].tolist()))
            # fc_point = np.sum(np.array(list(fc_params[0, i] for i in range(74))) * np.array(data.iloc[i, 92:166].tolist()))
            fc_data.append(fc_point)
        else:
            fc_data.append(0)
        process_bar(i, len(data))

    data["fc"] = fc_data

    new_panel = data.loc[:, ['stkid', 'TRDMNT', 'retx', 'fc']]
    # new_panel.to_csv('./fc.csv', mode='w', header=True)

    FACTOR = 'fc'
    begin_month = 200303 # 200203
    time_length = 178 # 从200306 到 201612

    months = compute_num_months(begin_month, time_length)
    # 计算该因子对应的多空组合回报率表格
    result = compute_factor_return_series(new_panel, FACTOR, begin_month, time_length)

    print("Factor Name:", FACTOR)
    the_return, t, Minus = compute_return_T_test(result)

    the_return2, t2 = compute_5_factor_model(Minus, months)






