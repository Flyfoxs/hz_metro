from core.config import *
import pandas as pd
from file_cache.utils.util_pandas import *
from file_cache.cache import file_cache
from file_cache.utils.util_log import *
from datetime import timedelta
from functools import lru_cache

def get_train_time():
    tmp = pd.date_range(start='1/1/2019', end='1/25/2019')
    return tuple([ str(date)[:10] for date in tmp])


def get_full_time(end='2019-01-29'):
    tmp = pd.date_range(start='2019-01-01', end=end)
    return tuple([str(date)[:10] for date in tmp])

@file_cache()
def summary_to_sub_level(date, gp_list):
    """
        summary_to_sub_level(date, ['time_ex', 'stationID', 'status','lineID', 'deviceID', 'payType'])
        summary_to_sub_level(date, ['time_ex', 'stationID', 'status'])
    :param date:
    :param gp_list:
    :return:
    """
    gp_list =  gp_list.copy()
    try:
        local_file = train_file % date
        sub = pd.read_csv(local_file)
    except Exception as e:
        try:
            local_file = test_file % date
            sub = pd.read_csv(local_file)
        except Exception as e:
            logger.warning(f'Can not find data for:{date}')
            local_file = train_file % '2019-01-01'
            sub = pd.read_csv(local_file, nrows=1).iloc[10:20]

    sub['time'] = pd.to_datetime(sub['time'])

    sub['time_ex'] = (sub['time'] - pd.to_datetime(date)).dt.seconds // (60 * 10)
    #     print(sub.columns)
    #     print(gp_list)
    tmp = sub.groupby(gp_list)['userID'].count().reset_index()

    if 'status' not in tmp:
        tmp['status'] = None

    if 'status' in gp_list:
        gp_list.remove('status')

    tmp_gp = tmp.pivot_table(index=gp_list, columns='status', values='userID',
                             aggfunc=np.sum).reset_index()

    tmp_gp.columns = tmp_gp.columns.droplevel(level=1)
    tmp_gp = tmp_gp.rename({0: 'outNums', 1: 'inNums'}, axis='columns')

    if 'outNums' not in tmp_gp.columns:
        tmp_gp['outNums'] = None
    if 'inNums' not in tmp_gp.columns:
        tmp_gp['inNums'] = None


    tmp_gp['day'] = date

    return tmp_gp


@lru_cache()
def merge_days(days_list):
    submit = pd.read_csv(submit_file)
    submit['startTime'] = pd.to_datetime(submit['startTime'])
    submit.head()
    submit['time_ex'] = (submit['startTime'] - pd.to_datetime('2019-01-29 00:00:00')).dt.seconds // (60 * 10)
    submit = submit.drop(axis=1, columns=['inNums', 'outNums'])

    #gp_list_local = None #gp_list.copy()
    # ['time_ex', 'stationID', 'status','lineID', 'deviceID', 'payType']

    for day in days_list:
        sn = (pd.to_datetime(day) - pd.to_datetime('2019-01-01')).days + 1
        tmp = summary_to_sub_level(day, gp_list, )
        tmp = tmp.fillna(0)
        tmp = tmp.rename({'inNums': f'in_{sn}', 'outNums': f'out_{sn}', }, axis='columns')
        join_list = gp_list.copy()
        if 'status' in join_list:
            join_list.remove('status')
        #
        # print('submit', submit.columns)
        # print('tmp', tmp.columns)
        # print('join', join_list)
        tmp = tmp.drop('day', axis=1)
        submit = pd.merge(submit, tmp, how='left', on=join_list)
        submit = submit.fillna(0)
    return submit


@lru_cache()
def get_test_set(direct):
    return get_feature_set(direct, test_feature_set).copy()


@lru_cache()
def get_train_set(direct):
    tmp1 = get_feature_set(direct, [item - 7 for item in test_feature_set]).copy()

    tmp2 = get_feature_set(direct, [item - 14 for item in test_feature_set]).copy()

    return pd.concat([tmp1, tmp2])


def get_feature_set(direct, feature_loc):
    feature_all = merge_days(get_full_time()).copy()
    # df_list = []

    begin = np.array(feature_loc).min()
    new_colname_list = [item - begin for item in feature_loc]

    col_list = [col for col in feature_all.columns if direct in col]
    train_loc = [f'{direct}_{item}' for item in feature_loc]

    temp = feature_all.loc[:, train_loc]
    temp.columns = new_colname_list

    temp = pd.concat([feature_all.loc[:, ['stationID', 'time_ex']], temp], axis=1)

    return temp


partition_size = 6.0
partition_num = int(np.ceil(144 / partition_size))
print(partition_num)


def get_train_test(stationID, partitionID, direct):
    train = get_train_set(direct).copy()
    # print(train.shape)
    train = train.loc[(train.stationID == stationID) & (train.time_ex // partition_size == partitionID)]

    test = get_test_set(direct).copy()
    # print(test.shape)
    test = test.loc[(test.stationID == stationID) & (test.time_ex // partition_size == partitionID)]

    return train, test


def get_sub_tmp():
    submit = pd.read_csv(submit_file)
    submit['startTime'] = pd.to_datetime(submit['startTime'])
    submit.head()
    submit['time_ex'] = (submit['startTime'] - pd.to_datetime('2019-01-29 00:00:00')).dt.seconds // (60 * 10)

    submit = submit.set_index(['stationID', 'time_ex'])
    # submit = submit.drop(axis=1, columns=['inNums', 'outNums'])
    # submit.head()
    return submit

if __name__ == '__main__':
    print(summary_to_sub_level('2019-01-03'))


