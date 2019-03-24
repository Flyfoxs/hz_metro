import warnings

import lightgbm as lgb
from core.config import *
from sklearn.model_selection import KFold

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
    sn = select_list
    sn_map = dict(zip(sn[1:], sn[:-1]))
    return sn_map[d]

def get_columns(data):
    return [f for f in data.columns if f not in ['weekend', 'inNums_next', 'outNums_next', 'time_ex']]


def attach_label(data):
    tmp = data.copy()
    tmp['day'] = tmp['day'].apply(get_refer_day)
    stat_columns = ['inNums', 'outNums']
    for f in stat_columns:
        tmp.rename(columns={f: f + '_next'}, inplace=True)

    tmp = tmp[['stationID', 'day', 'hour', 'minute', 'inNums_next', 'outNums_next']]

    logger.info((data.shape, tmp.shape))
    data = data.merge(tmp, on=['stationID', 'day', 'hour', 'minute'], how='left')
    data.fillna(0, inplace=True)
    return data


def summary_data(data):
    # 剔除周末,并修改为连续时间
    data = data[(data.day != 5) & (data.day != 6)]
    data = data[(data.day != 12) & (data.day != 13)]
    data = data[(data.day != 19) & (data.day != 20)]
    data = data[(data.day != 26) & (data.day != 27)]

    # tmp_df = tmp[tmp.day==1]
    # tmp_df['day'] = tmp_df['day'] - 1
    # print(tmp_df.shape)
    # tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True)
    # print(tmp.shape)


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
        'outNums_wh_max'    : 'max',
        'outNums_wh_min'    : 'min',
        'outNums_wh_mean': 'mean'
    })
    data = data.merge(tmp, on=['stationID', 'week', 'hour'], how='left')
    return data


def horizontal_data(data, index, length=5 ):
    data = data.copy()
    #print(data.shape)
    day_list = select_list[index:index+length]
    logger.info(f'get {day_list} for {index}, len:{length}')
    #Don't rename the begin day
    def rename_with_index(data,sn):
        cur_day = int(data.day.max() )
        if cur_day != day_list[-1]:
            data.columns = [f'{item}_{sn}'  for item in data.columns]
        return data.reset_index(drop=True)

    feature =  pd.concat([ rename_with_index(data.loc[data.day==item], sn) for sn, item in enumerate(day_list)], axis=1)

    return feature


def get_train_test(data):
    data = summary_data(data.copy())
    all_columns = get_columns(data)
    data = data[all_columns]

    feature_len = len(select_list) - 4
    data = pd.concat( [horizontal_data(data, index, 5)  for index in range(feature_len)])

    logger.info(f'len:{len(select_list)},list:{select_list}')

    test_day = 28


    X_data = data[data.day != test_day]#.values
    all_data = attach_label(X_data)


    X_test = data[data.day == test_day]#[all_columns]#.values

    return all_data, X_data,  X_test



@timed()
def train(X_data,  y_data,  X_test, ):
    all_columns = get_columns(X_data)

    num_fold = 5
    folds = KFold(n_splits=num_fold, shuffle=True, random_state=15)
    oof = np.zeros(len(y_data))
    predictions = np.zeros(len(X_test))
    #start = time.time()
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data.values, y_data.values)):
        logger.info("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(X_data.iloc[trn_idx], y_data.iloc[trn_idx])
        val_data = lgb.Dataset(X_data.iloc[val_idx], y_data.iloc[val_idx], reference=trn_data)

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
        num_round = 10000
        clf = lgb.train(params,
                        trn_data,
                        num_round,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=5000,
                        early_stopping_rounds=200)

        oof[val_idx] = clf.predict(X_data.iloc[val_idx], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] =  all_columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(X_test[all_columns], num_iteration=clf.best_iteration) / folds.n_splits
    predictions = predictions/num_fold
    score = np.abs(oof - y_data.values).mean()
    return predictions, score

@timed()
def main(sub_model = False):
    data = get_data()
    all_data, X_data,  X_test = \
        get_train_test(data)

    y_data = all_data['inNums_next']
    inNums, in_score = train(X_data,  y_data,  X_test)

    y_data = all_data['outNums_next']
    outNums, out_score = train(X_data,  y_data,  X_test)


    test_day = X_test.day.max()
    logger.info(f'test_day:{test_day}')
    avg = (in_score + out_score) / 2
    logger.info(f'avg:{avg:06.4f}, {in_score:06.4f}, out_score:{out_score:06.4f}')

    if sub_model:
        sub = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')
        sub['inNums']   = inNums
        sub['outNums']  = outNums
        sub.loc[sub.inNums<0 , 'inNums']  = 0
        sub.loc[sub.outNums<0, 'outNums'] = 0

        file_name = f'output/sub_kf_{avg:06.4f}_{in_score:06.4f}_{out_score:06.4f}_{int(time.time() % 10000000)}.csv'
        logger.info(file_name)

        #13.2464/15.0891
        #13.1399/15.054/14.0970
        #14.0970_13.1399_15.0540
        sub[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv(file_name, index=False)


if __name__ == '__main__':
    main(True)