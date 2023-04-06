import pandas as pd
import numpy as np
from sklearn import preprocessing

def process_industry(industry):
    industry['asset'] = industry['asset'].astype(str)
    return industry


def process_asset_pool(adj_price, liquid_cond=1000*10000):
    adj_price = adj_price[adj_price['trade_date']>=pd.to_datetime('20180101')]
    adj_price.rename(columns={'trade_date': 'datetime'}, inplace=True)
    adj_price = adj_price[['datetime', 'asset', 'amount']]
    amount_ma20 = adj_price.set_index(['datetime']).groupby('asset')['amount'].rolling(20).mean().reset_index()

    amount_ma20 = amount_ma20[amount_ma20['amount']>=liquid_cond]
    amount_ma20 = amount_ma20.set_index(['datetime', 'asset'])
    amount_ma20 = amount_ma20.sort_index()
    amount_ma20.columns = ['amount_ma20']

    asset_pool = amount_ma20
    return asset_pool


def process_factors(daily_factors,data,industry):
    daily_factors = daily_factors.reset_index()

    tmp = pd.DataFrame(np.unique(data.reset_index()['datetime']),columns=['datetime'])
    tmp['belong_datetime'] = tmp['datetime']
    daily_factors = pd.merge(tmp,daily_factors,how='right',on=['datetime'])
    daily_factors['belong_datetime'] = daily_factors['belong_datetime'].fillna(method='bfill')
    daily_factors = daily_factors.dropna(subset=['belong_datetime']).sort_values(['asset','belong_datetime','datetime'])
    daily_factors['datetime'] = (5-(daily_factors['belong_datetime']-daily_factors['datetime']).apply(lambda x: x.days)).astype(str)
    daily_factors['tmp'] = daily_factors['datetime'].apply(lambda x: x in ['1','2','3','4','5'])
    daily_factors = daily_factors[daily_factors.tmp == True].drop(columns=['tmp'])
    daily_factors = daily_factors.rename(columns={'belong_datetime':'datetime','datetime':'weekday'})
    daily_factors = daily_factors.set_index(['asset','datetime','weekday']).reset_index()

    # merge industry
    daily_factors = pd.merge(daily_factors,industry,how='inner',on=['asset']).drop(columns=['name','industry'])

    # change shape
    # daily_factors = daily_factors.set_index(['asset','datetime','weekday']).unstack(2)
    # daily_factors.columns = ["_".join(tuple) for tuple in daily_factors.columns]
    # daily_factors= daily_factors.reset_index()

    daily_factors = daily_factors.drop(columns=['weekday']).groupby(['asset','datetime']).aggregate(['mean'])
    daily_factors.columns = ['_'.join(i) for i in daily_factors.columns]
    daily_factors= daily_factors.reset_index()

    # daily_factors = daily_factors.drop(columns=['weekday']).groupby(['asset','datetime']).last()
    # daily_factors= daily_factors.reset_index()

    # fillna
    columns = daily_factors.columns[2:]
    tmp = daily_factors.groupby('datetime')[columns].transform('mean')
    daily_factors.iloc[:,2:] = daily_factors.iloc[:,2:].fillna(tmp)
    daily_factors = daily_factors.fillna(0)
    daily_factors = daily_factors.replace([np.inf, -np.inf], 0)

    # z_score
    def scale(x):
        x.iloc[:,:] = preprocessing.scale(x)
        return x
    daily_factors.iloc[:,2:] = daily_factors.groupby('datetime')[columns].apply(lambda x: scale(x))
    daily_factors = daily_factors.fillna(0)

    return daily_factors


def process_return(data):
    data = data.reset_index()
    data['return'] = np.log(data['shift_adj_open_price']/data['adj_open_price'])
    return data[['asset','datetime','return']].fillna(0)


def main():
    data = pd.read_pickle('/home/qianshuofu/Yansheng_strategy/data/initial_data/t_plus_max5_first_tradeday_adj_price.pkl')
    industry = pd.read_csv('/home/qianshuofu/Yansheng_strategy/data/initial_data/industry_twse_tpex.csv')
    adj_price = pd.read_feather('/home/qianshuofu/Yansheng_strategy/data/initial_data/adj_daily_TWSE_TPEX.ftr')
    daily_factors = pd.read_pickle('/home/qianshuofu/Yansheng_strategy/data/initial_data/daily_factors_422.pkl')

    industry = process_industry(industry)
    asset_pool = process_asset_pool(adj_price)
    daily_factors = process_factors(daily_factors,data,industry)
    df_return = process_return(data)

    df = pd.merge(daily_factors,df_return,how='inner',on=['asset','datetime']).sort_values(['datetime','asset'])
    df['datetime'] = df['datetime'].astype(str)
    df['return'] = (df.groupby(['datetime'])['return'].rank(pct=True) - 0.5)*3.4624
    # df['return'] = df.groupby(['datetime'])['return'].apply(lambda x: (x-x.mean())/x.std()).fillna(0)

    data_feature = df.iloc[:,2:-1]
    data_label = df.iloc[:,-1]
    data_index = df.iloc[:,:2]
    data_columns = df.columns

    np.save('/home/qianshuofu/Yansheng_strategy/data/generate_data/data_feature.npy',data_feature)
    np.save('/home/qianshuofu/Yansheng_strategy/data/generate_data/data_label.npy',data_label)
    np.save('/home/qianshuofu/Yansheng_strategy/data/generate_data/data_index.npy',data_index)
    np.save('/home/qianshuofu/Yansheng_strategy/data/generate_data/data_columns.npy',data_columns)

    
