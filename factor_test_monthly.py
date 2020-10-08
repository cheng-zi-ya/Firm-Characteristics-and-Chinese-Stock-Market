
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

import time

def compute_num_months(begin_month, time_length):
    """输入起始时间和时间长度，输出起始之后的一段月度的时间序列"""
    begin = begin_month
    months = [begin]
    for i in range(time_length - 1):
        if (begin % 100) % 12 != 0: begin = begin + 1
        else: begin = begin + 89
        months.append(begin)
    return months

def compute_factor_return_series(data, FACTOR, begin_month, time_length):
    months = compute_num_months(begin_month, time_length)

    # 计算价值加权，可以直接用PRC字段，因为其就是t-1的股票价格
    result = pd.DataFrame({'Date': months, 'High': np.zeros(time_length), 'Low': np.zeros(time_length), 'Minus': np.zeros(time_length)})

    for i in range(time_length):
        date = months[i]
        factor, weights, returnrate = [], [], []
        # 对于横截面进行计算（对所有股票计算因子，再按照因子排序进行分组，归并回报）
        data_atdate = data[data.TRDMNT == date]
        factor_value = data_atdate[FACTOR].values
        returns = data_atdate.retx.values
        # returns = data_atdate.reta.values
        for j in range(len(data_atdate)):
            factor.append(factor_value[j])
            weights.append(1)  # 如果是value-weighted,这里是PRC;如果是Equal-weighted，这里是1
            returnrate.append(returns[j])

        factor_table = {'factor': factor, 'weights': weights, 'returnrate': returnrate}
        factor_table = pd.DataFrame(factor_table)
        factor_table = factor_table.sort_values(['factor'], ascending=False)

        len_factor_table = len(factor_table)
        len_group = int(np.floor(len_factor_table / 10))

        group_return, group_weight, final_returns = [], [], []
        for j in range(len_factor_table):
            if (j + 1) % len_group == 0:
                final_returns.append(np.sum(np.array(group_return) * np.array(group_weight)) / np.sum(group_weight))
                group_return, group_weight = [], []
            else:
                group_return.append(factor_table.iloc[j].returnrate)
                group_weight.append(factor_table.iloc[j].weights)

        result.loc[i, 'High'] = final_returns[9]
        result.loc[i, 'Low'] = final_returns[0]
        result.loc[i, 'Minus'] = final_returns[0] - final_returns[9]

    return result

def compute_return_T_test(result_panel):
    # 计算平均回报率，进行t检验。原本是High - Low, 而如果收益反转，则 Low - High
    Minus = np.array((list(result_panel.High)) - np.array(list(result_panel.Low)))
    Minus = Minus * ((np.average(Minus) > 0) *1 *2 -1)
    print("monthly average raw returns(%)", np.average(Minus) * 100)
    print("T-test", stats.ttest_1samp(Minus, 0)[0])
    return np.average(Minus) * 100, stats.ttest_1samp(Minus, 0)[0], Minus

def compute_5_factor_model(Minus, months):
    # 对于五因子模型计算超额收益率，并计算其t检验值
    fivefac = pd.read_csv("./fivefactor_monthly.csv")

    # 然后利用date确定Fama中的对应值
    SMB, HML, MKT, RMW, CMA = [], [], [], [], []

    for i in range(len(fivefac)):
        if fivefac.trdmn[i] in months:
            SMB.append(fivefac.smb[i])
            HML.append(fivefac.hml[i])
            MKT.append(fivefac.mkt_rf[i])
            RMW.append(fivefac.rmw[i])
            CMA.append(fivefac.cma[i])

    X = np.column_stack((np.ones(len(MKT)), np.array(SMB), np.array(HML), np.array(MKT), np.array(RMW), np.array(CMA)))
    y = list(Minus)

    model = sm.OLS(y, X)
    results = model.fit()

    print(results.params[0] * 100)
    print(results.tvalues[0])
    # np.set_printoptions(suppress=True)
    # print(results.summary())
    return results.params[0] * 100, results.tvalues[0]

if __name__ == "__main__":
    # 输入量
    data = pd.read_csv('./data.csv')
    # FACTOR = 'roaq'
    FACTOR = 'BM'
    begin_month = 200203
    time_length = 190

    months = compute_num_months(begin_month, time_length)
    # 计算该因子对应的多空组合回报率表格
    result = compute_factor_return_series(data, FACTOR, begin_month, time_length)

    print("Factor Name:", FACTOR)
    the_return, t, Minus = compute_return_T_test(result)

    the_return2, t2 = compute_5_factor_model(Minus, months)


