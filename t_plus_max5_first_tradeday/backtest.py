import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# 生成持倉
def get_holdings(prediction: pd.Series,
                 data: pd.DataFrame,
                 industry: pd.DataFrame,
                 asset_pool: pd.DataFrame,
                 capital: int, max_w: float):
    """
    constraint:
    - 持倉價位限制 (股票池)
    - 取每周rank排名最前面的一群
    - 排除開盤漲停股票 (不能買進)
    - 取得每周預期持倉張數
    - 在開盤調倉
    - 取得上周持倉預期持倉 (回測用, 第一周的上週持倉為0)
    - 取得下周持倉預期持倉 (回測用，最後一周持倉假設全出, 為0)
    :param prediction:
    :param data:
    :param industry:
    :param asset_pool:
    :param capital:
    :param max_w:
    :return:
    """
    num = int(1 / max_w)
    # print(f'最大持倉金額: {capital}\n每倉最大權重: {max_w}')
    # print(f'每周持倉數: {num}\n最大持倉金額: {capital * max_w}')
    # 用於回測的數據集
    tmp = pd.merge(prediction, data, left_index=True, right_index=True, how='left')
    # 股票池 (給定)
    tmp = tmp.loc[tmp.index.isin(asset_pool.index)]
    # 持倉價位限制 (股票池)
    tmp = tmp[tmp['pre_vwap'] <= capital * max_w / 1000]
    # 取每周rank排名最前面的一群
    tmp_top = tmp.groupby(level='datetime').apply(
        lambda x: x.sort_values('pred', ascending=False).head(num).reset_index())
    tmp_top = tmp_top.drop(columns=['datetime']).reset_index()
    tmp_top.rename(columns={
        'level_1': 'rank'
    }, inplace=True)
    tmp_top = tmp_top.merge(industry, on=['asset'], how='inner')
    tmp_top = tmp_top.sort_values(['datetime', 'rank'])
    tmp_top = tmp_top.set_index(['datetime', 'asset'])
    # 排除開盤漲停股票 (不能買進)
    tmp_top = tmp_top[tmp_top['open_price'] != tmp_top['high_limit']]
    # 取得每周預期持倉張數
    tmp_top['expected_holding_volume(張)'] = (capital * max_w / 1000 // tmp_top['pre_vwap']).astype(int)
    # 開盤調倉
    tmp_top['Market_Value'] = tmp_top['open_price'] * tmp_top['expected_holding_volume(張)'] * 1000

    # 取得上周持倉預期持倉 (回測用)
    datelist = tmp_top.index.get_level_values('datetime').unique().to_list()
    datelist.sort()
    date_ = datelist[0]
    tmp_top.loc[date_, 'previous_expected_holding_volume(張)'] = 0
    for date_ in datelist[1:]:
        # 尋找前一周持倉張數
        idx = datelist.index(date_) - 1
        # 前一周之後的持倉 (前周持倉數)
        previous_date_ = datelist[idx]
        #         print(previous_date_, date_)
        previous_expected_holding = tmp_top.loc[previous_date_]['expected_holding_volume(張)'].to_dict()
        # 尋找下週的持倉 (本周亦持有的)
        test = tmp_top.loc[date_]
        exist_list = [i for i in test.index if i in previous_expected_holding.keys()]
        exist_holding_list = [previous_expected_holding[s] for s in exist_list]
        test.loc[exist_list, 'previous_expected_holding_volume(張)'] = exist_holding_list

    # 取得下周持倉預期持倉 (回測用)
    # 最後一周持倉假設全出 (為 0)
    date_ = datelist[-1]
    tmp_top.loc[date_, 'forward_expected_holding_volume(張)'] = 0
    for date_ in datelist[:-1]:
        # 尋找前一周持倉張數
        idx = datelist.index(date_) + 1
        # 最後一周持之前的持倉 (後周持倉數)
        forward_date_ = datelist[idx]
        #         print(forward_date_, date_)
        forward_expected_holding = tmp_top.loc[forward_date_]['expected_holding_volume(張)'].to_dict()
        # 尋找下週的持倉 (本周亦持有的)
        test = tmp_top.loc[date_]
        exist_list = [i for i in test.index if i in forward_expected_holding.keys()]
        exist_holding_list = [forward_expected_holding[s] for s in exist_list]
        test.loc[exist_list, 'forward_expected_holding_volume(張)'] = exist_holding_list

    tmp_top['previous_expected_holding_volume(張)'] = tmp_top['previous_expected_holding_volume(張)'].fillna(0).astype(
        int)
    tmp_top['forward_expected_holding_volume(張)'] = tmp_top['forward_expected_holding_volume(張)'].fillna(0).astype(int)
    # 新建的倉位
    cond1 = tmp_top['expected_holding_volume(張)'] > tmp_top['previous_expected_holding_volume(張)']
    tmp_top.loc[cond1, 'open_expected_holding_volume(張)'] = tmp_top.loc[cond1, 'expected_holding_volume(張)'] - \
                                                            tmp_top.loc[cond1, 'previous_expected_holding_volume(張)']
    tmp_top['open_expected_holding_volume(張)'] = tmp_top['open_expected_holding_volume(張)'].fillna(0).astype(int)
    # 關倉的倉位數
    cond2 = tmp_top['expected_holding_volume(張)'] > tmp_top['forward_expected_holding_volume(張)']
    tmp_top.loc[cond2, 'close_expected_holding_volume(張)'] = tmp_top.loc[cond2, 'expected_holding_volume(張)'] - \
                                                             tmp_top.loc[cond2, 'forward_expected_holding_volume(張)']
    tmp_top['close_expected_holding_volume(張)'] = tmp_top['close_expected_holding_volume(張)'].fillna(0).astype(int)
    # 開新倉的市值
    tmp_top['open_Market_Value'] = tmp_top['open_price'] * tmp_top['open_expected_holding_volume(張)']
    tmp_top['open_Market_Value'] *= 1000
    # 關舊倉的市值 (forward_open_price = open_price*(1+forward_return))
    tmp_top['close_Market_Value'] = tmp_top['open_price'] * (1 + tmp_top['forward_return']) * tmp_top[
        'close_expected_holding_volume(張)']
    tmp_top['close_Market_Value'] *= 1000
    tmp_top.insert(0, 'weight', max_w)

    return tmp_top

# 回測
def analyze_risk_position_weekly(holdings: pd.DataFrame, show_details=False):
    """
    交易成本計算方法:
    1. 手續費率: 買進與賣出皆需要收取 (0.001425 * 折數)
    2. 證交稅率: 賣出收取 (0.003)
    3. 融資費率: 以年利息6%計算 (年息轉換成周息(年息/52), 以6成金額計算融資)

    計算交易成本分成三部分:
    1. 新增的倉位 (成本: 手續費率(買進))
    2. 需換倉的倉位 (成本: 手續費率(賣出)+證交稅率+融資費率)
    3. 無須換倉的倉位 (成本: 無，假設新增的倉位與需換倉的倉位才支付買進手續費)
    :param holdings:
    :param show_details:
    :return:
    """
    discount = 0.15
    # 手續費率(單邊)
    broker_rate = 0.001425 * discount
    # 證交稅率
    tax_rate = 0.003
    # 融資利息 (年息轉換成周息(年息/52)，以6成金額計算融資)
    loan_rate = 0.06 / 52 * 0.6

    # 1. 新增的倉位 (成本: 手續費率(買進))
    holdings = holdings.copy()
    holdings['open_fee'] = holdings['open_Market_Value'] * broker_rate
    # 2. 需換倉的倉位 (成本: 手續費率(賣出)+證交稅率+融資費率)
    holdings['close_fee'] = holdings['close_Market_Value'] * (broker_rate + tax_rate + loan_rate)
    holdings['total_fee'] = holdings['open_fee'] + holdings['close_fee']
    # Gross Earnings
    holdings['Gross_Earnings'] = holdings['Market_Value'] * holdings['forward_return']
    # Gross Profit
    holdings['Gross_Profit'] = holdings['Gross_Earnings'] - holdings['total_fee']

    pnl = holdings.groupby(level='datetime')[['Market_Value', 'total_fee', 'Gross_Earnings', 'Gross_Profit']].sum()
    pnl['fee_rate'] = pnl['total_fee'] / pnl['Market_Value']
    pnl['earning_rate'] = pnl['Gross_Earnings'] / pnl['Market_Value']
    pnl['return'] = pnl['Gross_Profit'] / pnl['Market_Value']
    factor_returns = pnl['return'].to_list()
    factor_returns = pd.DataFrame(factor_returns)
    f_cum_compound = (1 + factor_returns).cumprod()
    f_cum = 1 + factor_returns.cumsum()

    # 计算累计收益率
    cum_ret_rate = f_cum.iloc[-1] - 1
    cum_ret_rate_compound = f_cum_compound.iloc[-1] - 1
    # 计算年化收益率
    annual_ret_compound = (f_cum_compound.iloc[-1]) ** (52. / float(len(f_cum))) - 1
    annual_ret = (f_cum.iloc[-1] - 1) / float(len(f_cum)) * 52
    mdd = get_max_drawdown(factor_returns)
    mdd_compound = get_max_drawdown_compound(f_cum_compound)
    ac = (annual_ret.iloc[-1] - 0.04) / mdd.iloc[-1]
    ac_compound = (annual_ret_compound.iloc[-1] - 0.04) / mdd_compound.iloc[-1]
    add = get_mean_drawdown(f_cum)
    add_compound = get_mean_drawdown(f_cum_compound)
    ar_md = annual_ret.iloc[-1] / add
    ar_md_compound = annual_ret_compound.iloc[-1] / add_compound
    volatility = np.std(factor_returns) * np.sqrt(52)
    sharpe = (annual_ret - 0.04) / volatility
    sharpe_compound = (annual_ret_compound - 0.04) / volatility
    total_trading_day = factor_returns.shape[0]
    win_trading_day = factor_returns[factor_returns[0] >= 0].shape[0]
    loss_trading_day = factor_returns[factor_returns[0] < 0].shape[0]
    win_rate = win_trading_day / total_trading_day
    ret_compound = f_cum_compound.diff()

    daily_pnl = (cum_ret_rate / len(factor_returns)).iloc[0]
    daily_pnl_compound = (cum_ret_rate_compound / len(factor_returns)).iloc[0]
    daily_win = (factor_returns[factor_returns[0] >= 0].sum() / total_trading_day).iloc[-1]
    daily_win_compound = (ret_compound[ret_compound >= 0].sum() / total_trading_day).iloc[-1]
    daily_loss = (factor_returns[factor_returns[0] < 0].sum() / total_trading_day).iloc[-1]
    daily_loss_compound = (ret_compound[ret_compound[0] < 0].sum() / total_trading_day).iloc[-1]
    pl_ratio = (factor_returns[factor_returns[0] >= 0].sum() / abs(factor_returns[factor_returns[0] < 0].sum())).iloc[
        -1]
    pl_ratio_compound = (ret_compound[ret_compound >= 0].sum() / ret_compound[ret_compound < 0].sum().abs()).iloc[-1]
    max_single_day_win = factor_returns.max().iloc[-1]
    max_single_day_loss = factor_returns.min().iloc[-1]

    df = pd.DataFrame([[annual_ret.iloc[-1], cum_ret_rate.iloc[-1], mdd.iloc[-1],
                        sharpe.iloc[-1], volatility.iloc[-1], total_trading_day, win_trading_day,
                        loss_trading_day, win_rate, daily_pnl, daily_win, daily_loss, pl_ratio,
                        max_single_day_win, max_single_day_loss, ac, ar_md[0]],
                       [annual_ret_compound.iloc[-1], cum_ret_rate_compound.iloc[-1], mdd_compound.iloc[-1],
                        sharpe_compound.iloc[-1], volatility.iloc[-1], total_trading_day, win_trading_day,
                        loss_trading_day, win_rate, daily_pnl_compound, daily_win_compound, daily_loss_compound,
                        pl_ratio_compound, max_single_day_win, max_single_day_loss, ac_compound, ar_md_compound[0]]],
                      columns=['年化收益', '累计收益', '最大回撤', '夏普(r=4%)', '波动率',
                               '總交易周', '凈獲利交易周', '凈虧損交易周', '勝率', '周均凈收益', '周均凈獲利',
                               '周均凈虧損', '總盈虧比', '單周最大交易凈獲利', '單周最大交易凈虧損', '卡爾瑪', '年化收益/平均回撤'],
                      index=['單利', '複利']).T

    if show_details:
        print(pnl[['Market_Value', 'Gross_Profit', 'fee_rate', 'return']].describe())

        plt.figure(figsize=(9, 9))
        plt.subplot(3, 1, 1)
        pnl['return'].cumsum().plot()

        plt.subplot(3, 1, 2)
        plt.bar(pnl.index, pnl['return'], width=3)
        plt.xticks(rotation=25)

        plt.subplot(3, 1, 3)
        return_group_ByTime = pnl['return'].resample('M').sum().reset_index()
        return_group_ByTime['year'] = return_group_ByTime['datetime'].dt.year
        return_group_ByTime['month'] = return_group_ByTime['datetime'].dt.month
        heatmap1_data = pd.pivot_table(return_group_ByTime, values='return', index='year', columns='month')
        sns_plot = sns.heatmap(heatmap1_data, cmap="YlGnBu")

        plt.subplots_adjust(left=0.125,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.2,
                            hspace=0.4)
    return df


def get_max_drawdown(df:pd.DataFrame):
    md = {}
    mdd=0
    last_col=0
    new_dd=0
    for label in df.index:
        col=df.loc[label,0]
        # 最大回撤
        if col<=0 and last_col<=0:
            new_dd+=-col
        elif col<=0 and last_col>0:
            new_dd=-col
        elif col>0 and last_col<=0:
            mdd = max(mdd, new_dd)
            new_dd=0
        else:
            assert new_dd==0
            pass
        md[label] = mdd
    return pd.Series(md)


def get_mean_drawdown(df:pd.DataFrame):
    previous_col=1
    previous_return=0
    l=[]
    count=0
    for index in range(df.shape[0]):
        col = df.iloc[index].values
        if col < previous_col:
            l.append(col-previous_col)
            if previous_return >= 0 or index == 0:
                count += 1
        previous_return = col-previous_col
        previous_col = col
        index+=1
    return -sum(l)/count


def get_max_drawdown_compound(df: pd.DataFrame):
    md = {}
    for label, col in df.items():
        # 最大回撤
        max_nv = np.maximum.accumulate(col)
        mdd = -np.min(col/max_nv-1)
        md[label] = mdd
    return pd.Series(md)
