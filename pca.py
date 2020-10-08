
"""
由于并未达到原文中的数值，所以之后会重新检查，看看算式能否进一步提升，因此，并未写出很集成的模块。

"""
import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA

from factor_test_monthly import (compute_num_months, compute_factor_return_series, compute_return_T_test,
                                 compute_5_factor_model)

from fm import process_bar

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    data = pd.read_csv('./data.csv')
    begin_month = 200203
    time_length = 190

    months = compute_num_months(begin_month, time_length)

    # 转为按时间排序
    data = data.sort_values(by = "TRDMNT")
    data = data.reset_index(drop = True)

    # 然后对于每一个时间节点，对于74个因子计算一次，得到其参数，
    # 在这里，每个时间对应的PCA，但在计算回报的时候应该用上一个时间点的数值
    for i in range(time_length):
        month = months[i]
        data_atmonth = data[data.TRDMNT == month]
        # 线性回归
        X = data_atmonth.iloc[:, 18:92].values
        model = IncrementalPCA(n_components=1).fit(X)
        pca_point = model.components_
        if i == 0:
            pca_matrix = pca_point
        else:
            pca_matrix = np.vstack((pca_matrix, pca_point))  # 该矩阵和时间的对应关系为: 时间对应的那一行用到了下一个月的回报，
                                                             # 所以应该移动

    # 对于 data 表，对每一个时间节点的每一个股票计算对应
    T = 12
    dates = data.TRDMNT.tolist()
    data_matrix = data.iloc[:, 18:92].values
    pca_data = []
    for i in range(len(data)):
        date = dates[i]
        if date >= months[T]:     # 因为需要前T个时间点的系数进行计算
            now_pos = int((date - 200200)/100) * 12 + date%100- 3
            pca_params = np.sum(pca_matrix[now_pos - T:now_pos, :], axis=0) / T
            pca_point = np.sum(pca_params *  data_matrix[i, :])
            pca_data.append(pca_point)
        else:
            pca_data.append(0)
        process_bar(i, len(data))

    data["pca"] = pca_data

    new_panel = data.loc[:, ['stkid', 'TRDMNT', 'retx', 'pca']]
    # new_panel.to_csv('./pca.csv', mode='w', header=True)

    FACTOR = 'pca'
    begin_month = 200303 # 200203
    time_length = 178 # 从200306 到 201612

    months = compute_num_months(begin_month, time_length)
    # 计算该因子对应的多空组合回报率表格
    result = compute_factor_return_series(new_panel, FACTOR, begin_month, time_length)

    print("Factor Name:", FACTOR)
    the_return, t, Minus = compute_return_T_test(result)

    the_return2, t2 = compute_5_factor_model(Minus, months)








