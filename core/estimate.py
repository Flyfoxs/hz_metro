from multiprocessing import Pool as ThreadPool

from statsmodels.tsa.arima_model import ARIMA
import itertools
import warnings

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

from core.db import *

from sklearn.linear_model import Ridge

def get_res(stationID, partition_num, direct):
    train, test = get_train_test(stationID, partition_num, direct)

    logger.info(f'train:{len(train):03}, test:{len(test):03}, for {(stationID, partition_num, direct)}')
    feature_len = len(test_feature_set)

    y = train.iloc[:, -1]
    X = train.iloc[:, -feature_len:-1]
    clf = Ridge(alpha=1.0)

    if len(X) == 0:
        return None
    clf.fit(X, y)

    logger.info(f'{(stationID, partition_num, direct)} Ridge is {clf.coef_}')

    res = clf.predict(test.iloc[:, -feature_len:-1])
    test['res'] = res
    test = test.set_index(['stationID', 'time_ex'])
    return test


def main():
    sub = get_sub_tmp()
    sub['in'] = None
    sub['out'] = None
    print(sub.columns)
    for direct in ['in', 'out']:
        for stationID in range(81):
            logger.info(f'{direct}, {stationID}')
            for partitionid in range(partition_num):
                res = get_res(stationID, partitionid, direct)
                #             print(sub.loc[sub.index.isin(res.index), direct].shape)
                #             print(res.shape)

                if res is not None:
                    sub.loc[sub.index.isin(res.index), direct] = res.res
                    # logger.info(len(res))
                else:
                    logger.error(f'Can not find data for {(stationID, partitionid, direct)}')

    sub.inNums =   sub['in'].astype(float).round()
    sub.outNums =  sub['out'].astype(float).round()
    sub.where(sub>0, 0)
    sub = sub.reset_index()
    sub[sub_col].to_csv(f'./output/Ridge_p_num_{partition_num}_{int(time.time() % 10000000)}.csv',index=None)

if __name__ == '__main__':

    main()