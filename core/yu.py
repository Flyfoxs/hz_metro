import lightgbm as lgb
import warnings

import lightgbm as lgb

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from file_cache.utils.util_pandas import *
from file_cache.cache import file_cache
from file_cache.utils.util_log import *
import os

path = 'input'

test_28 = path + '/Metro_testA/testA_record_2019-01-28.csv'


def get_dummy_feature():
    dummy = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')
    # dummy = dummy[dummy.stationID==0]
    dummy['time_ex'] = (pd.to_datetime(dummy['startTime']) - pd.to_datetime('2019-01-29')).dt.seconds // (60 * 10)
    dummy['time'] = pd.to_datetime(dummy['startTime'])

    # df = df_.copy()

    # base time
    # dummy['day']     = (pd.to_datetime(dummy['time'])-pd.to_datetime('2019-01-01')).dt.days+1
    # dummy['week']    = pd.to_datetime(dummy['time']).dt.dayofweek + 1
    # dummy['weekend'] = (pd.to_datetime(dummy.time).dt.weekday >=5).astype(int)
    dummy['hour'] = dummy['time'].dt.hour
    dummy['minute'] = 10 * (dummy['time'].dt.minute // 10)
    # dummy['time'].dt.day=20
    return dummy[['stationID', 'time_ex', 'hour', 'minute']]

    # get_dummy_feature()


    # dummy = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')
    # #pd.to_datetime(dummy['startTime']).dt.date
    # #pd.to_datetime(dummy['startTime'])-pd.to_datetime(dummy['startTime']).dt.date.values
    # (pd.to_datetime(dummy['startTime']) - pd.to_datetime(dummy['startTime']).dt.date.values).dt.seconds // (60 * 10)


@timed()
@file_cache()
def get_base_features(file_path):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    logger.info('Time is done')
    # df['time_ex'] = ( df['time'] - df['time'].dt.date.values).dt.seconds // (60 * 10)
    # logger.info('Time_ex is done')

    # df = df_.copy()

    # base time
    df['day'] = (df['time'] - pd.to_datetime('2019-01-01')).dt.days + 1
    logger.info('Day is done')
    df['week'] = df['time'].dt.dayofweek + 1
    logger.info('Week is done')
    df['weekend'] = (df['time'].dt.weekday >= 5).astype(int)
    logger.info('weekend is done')
    df['hour'] = df['time'].dt.hour
    logger.info('Hour is done')
    df['minute'] = 10 * (df['time'].dt.minute // 10)
    logger.info('Minute is done')

    # count,sum
    dummy = get_dummy_feature()
    result = df.groupby(['stationID', 'week', 'weekend', 'day', 'hour', 'minute']).status.agg(
        ['count', 'sum']).reset_index()
    result = dummy.merge(result, on=['stationID', 'hour', 'minute'], how='left')

    result['inNums'] = result['sum'].fillna(0)
    result['outNums'] = result['count'].fillna(0) - result['sum'].fillna(0)
    result = result.fillna(method='ffill')
    result = result.fillna(method='bfill')

    # nunique
    tmp = df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
    result = result.merge(tmp, on=['stationID'], how='left')
    tmp = df.groupby(['stationID', 'hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
    result = result.merge(tmp, on=['stationID', 'hour'], how='left')
    tmp = df.groupby(['stationID', 'hour', 'minute'])['deviceID'].nunique(). \
        reset_index(name='nuni_deviceID_of_stationID_hour_minute')
    result = result.merge(tmp, on=['stationID', 'hour', 'minute'], how='left')

    # in,out


    #
    result['day_since_first'] = result['day'] - 1
    result.fillna(0, inplace=True)
    del result['sum'], result['count']

    return result


# data_list = os.listdir(path+'/Metro_train/')
# for i in range(0, len(data_list)):
#     if data_list[i].split('.')[-1] == 'csv':
#         print(data_list[i], i)
#         df = get_base_features(path+'/Metro_train/' + data_list[i])
#         data = pd.concat([data, df], axis=0, ignore_index=True)
#     else:
#         continue

@file_cache()
def get_data():
    data = get_base_features(test_28)
    data_list = os.listdir(path + '/Metro_train/')
    for i in range(0, len(data_list)):
        if data_list[i].split('.')[-1] == 'csv':
            logger.info(data_list[i], i)
            # df = pd.read_csv()
            df = get_base_features(path + '/Metro_train/' + data_list[i])
            data = pd.concat([data, df], axis=0, ignore_index=True)
        else:
            continue
    return data




def get_refer_day(d):
    sn = [item for item in range(40) if item%7 not in [5,6] ]
    sn_map = dict(zip(sn[1:], sn[:-1]))
    return sn_map[d]


def get_train_test(data):
    # 剔除周末,并修改为连续时间
    data = data[(data.day != 5) & (data.day != 6)]
    data = data[(data.day != 12) & (data.day != 13)]
    data = data[(data.day != 19) & (data.day != 20)]
    data = data[(data.day != 26) & (data.day != 27)]

    tmp = data.copy()
    # tmp_df = tmp[tmp.day==1]
    # tmp_df['day'] = tmp_df['day'] - 1
    # print(tmp_df.shape)
    # tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True)
    # print(tmp.shape)
    tmp['day'] = tmp['day'].apply(get_refer_day)
    stat_columns = ['inNums', 'outNums']
    for f in stat_columns:
        tmp.rename(columns={f: f + '_next'}, inplace=True)

    tmp = tmp[['stationID', 'day', 'hour', 'minute', 'inNums_next', 'outNums_next']]

    logger.info((data.shape, tmp.shape))
    data = data.merge(tmp, on=['stationID', 'day', 'hour', 'minute'], how='left')
    data.fillna(0, inplace=True)

    tmp = data.groupby(['stationID', 'week', 'hour', 'minute'], as_index=False)['inNums'].agg({
        'inNums_whm_max': 'max',
        'inNums_whm_min': 'min',
        'inNums_whm_mean': 'mean'
    })
    data = data.merge(tmp, on=['stationID', 'week', 'hour', 'minute'], how='left')

    tmp = data.groupby(['stationID', 'week', 'hour', 'minute'], as_index=False)['outNums'].agg({
        'outNums_whm_max': 'max',
        'outNums_whm_min': 'min',
        'outNums_whm_mean': 'mean'
    })
    data = data.merge(tmp, on=['stationID', 'week', 'hour', 'minute'], how='left')

    tmp = data.groupby(['stationID', 'week', 'hour'], as_index=False)['inNums'].agg({
        'inNums_wh_max': 'max',
        'inNums_wh_min': 'min',
        'inNums_wh_mean': 'mean'
    })
    data = data.merge(tmp, on=['stationID', 'week', 'hour'], how='left')

    tmp = data.groupby(['stationID', 'week', 'hour'], as_index=False)['outNums'].agg({
        # 'outNums_wh_max'    : 'max',
        # 'outNums_wh_min'    : 'min',
        'outNums_wh_mean': 'mean'
    })
    data = data.merge(tmp, on=['stationID', 'week', 'hour'], how='left')

    all_days = sorted(data.day.drop_duplicates().values.astype(int))
    logger.info(f'len:{len(all_days)},list:{all_days}')

    last_day = all_days[-1]
    vali_day = 25  # 25 #21
    test_day = 28

    # predict_day = last_day + 1

    all_columns = [f for f in data.columns if f not in ['weekend', 'inNums_next', 'outNums_next', 'time_ex']]
    ### all data
    all_data = data[data.day != test_day]
    # all_data['day'] = all_data['day'].apply(recover_day)
    X_data = all_data[all_columns]#.values

    train = data[~data.day.isin([vali_day, test_day])]
    # train['day'] = train['day'].apply(recover_day)
    X_train = train[all_columns]#.values

    valid = data[data.day == vali_day]
    # valid['day'] = valid['day'].apply(recover_day)
    X_valid = valid[all_columns]#.values

    test = data[data.day == test_day]
    X_test = test[all_columns]#.values

    return all_data, X_data, train, X_train, valid, X_valid, test, X_test

@timed()
def main(sub_model = False):
    data = get_data()
    all_data, X_data, train, X_train, valid, X_valid, test, X_test = \
        get_train_test(data)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_seed': 0,
        'bagging_freq': 1,
        'verbose': 1,
        'reg_alpha': 1,
        'reg_lambda': 2
    }

    ######################################################inNums
    y_train = train['inNums_next']
    y_valid = valid['inNums_next']
    y_data = all_data['inNums_next']

    vali_day = X_valid.day.max()

    logger.info((vali_day, X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_train.columns))

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_evals = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=90000,
                    valid_sets=[lgb_train, lgb_evals],
                    valid_names=['train', 'valid'],
                    early_stopping_rounds=200,
                    verbose_eval=1000,
                    )
    in_score = gbm.best_score['valid']['l1']
    in_iter = gbm.best_iteration

    if sub_model:
        ### all_data
        lgb_train = lgb.Dataset(X_data, y_data)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=gbm.best_iteration,
                        valid_sets=[lgb_train],
                        valid_names=['train'],
                        verbose_eval=1000,
                        )
        test['inNums_next'] = gbm.predict(X_test)

    ######################################################outNums
    y_train = train['outNums_next']
    y_valid = valid['outNums_next']
    y_data = all_data['outNums_next']
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_evals = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=90000,
                    valid_sets=[lgb_train, lgb_evals],
                    valid_names=['train', 'valid'],
                    early_stopping_rounds=200,
                    verbose_eval=1000,
                    )
    out_score = gbm.best_score['valid']['l1']
    out_iter = gbm.best_iteration

    if sub_model:
        ### all_data
        lgb_train = lgb.Dataset(X_data, y_data)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=gbm.best_iteration,
                        valid_sets=[lgb_train],
                        valid_names=['train'],
                        verbose_eval=1000,
                        )
        test['inNums_next'] = gbm.predict(X_test)

    test_day = test.day.max()
    logger.info(f'vali_day:{vali_day},test_day:{test_day}')
    avg = (in_score + out_score) / 2
    logger.info(f'avg:{avg:06.4f}, {in_score:06.4f}@{in_iter}, out_score:{out_score:06.4f}@{out_iter}')

    if sub_model:
        sub = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')
        sub['inNums']   = test['inNums_next'].values
        sub['outNums']  = test['inNums_next'].values
        # 结果修正
        sub.loc[sub.inNums<0 , 'inNums']  = 0
        sub.loc[sub.outNums<0, 'outNums'] = 0

        file_name = f'output/sub_bs_{vali_day:02}_{avg:06.4f}_{in_score:06.4f}_{out_score:06.4f}_{int(time.time() % 10000000)}.csv'
        logger.info(file_name)

        #13.2464/15.0891
        #13.1399/15.054/14.0970
        #14.0970_13.1399_15.0540
        sub[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv(file_name, index=False)


if __name__ == '__main__':
    main(False)