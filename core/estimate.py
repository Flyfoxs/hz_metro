from multiprocessing import Pool as ThreadPool

from statsmodels.tsa.arima_model import ARIMA
import itertools
import warnings

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

from core.db import *

#plt.style.use('fivethirtyeight')

# Define the p, d and q parameters to take any value between 0 and 2



def get_estimate_val():
    time_list = get_full_time('2019-01-19')
    # time_list.append('2019-01-28')
    print(time_list)
    merge_res = merge_days(time_list)
    merge_res['total'] = merge_res.sum(axis=1)
    merge_res.sort_values('total')
    merge_res.head()
    merge_res = merge_res.fillna(0)
    return merge_res.sample(frac=0.1,random_state=1)

def estimate_paras(pdq):

    warnings.filterwarnings("ignore")  # specify to ignore warning messages


    sample = get_estimate_val().sample(10)
    col_list = [col for col in sample.columns if 'in' in col]
    for index, X  in sample[col_list].iterrows():

        # X = series.values
        size = len(X) - 1
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        res = []
        print(f'index:{index}, Total:{len(pdq)},train_len:{len(train)} test_len:{len(test)}')
        for sn, order in enumerate(pdq):
            try:
                #         print()
                predictions = list()
                for t in range(len(test)):
                    model = ARIMA(history, order=order)  # (5,1,0)
                    model_fit = model.fit(disp=0)
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    obs = test[t]
                    history.append(obs)
                    # print('predicted=%f, expected=%f' % (yhat, obs))
                # print(len(test), len(predictions),  )
                error = mean_squared_error(test, predictions)
                res.append((error, order))
                if error < 10:
                    print(f'{sn:03}, Test MSE: {error:6.2f}, {order}')
                    # plot
                    #             pyplot.plot(test)
                    #             pyplot.plot(predictions, color='red')
                    #             pyplot.show()
            except  Exception as e:
                # print(f'{order} process incorrect')
                pass

        res = sorted(res, key=lambda val: val[0])
        print(res)
    return res


@timed()
def process_partition(paras):
    from numpy.linalg import LinAlgError
    day, direct, parition_id = paras
    merge_res = merge_days(get_full_time(day))
    merge_res = merge_res.fillna(0)
    merge_res = merge_res.loc[ merge_res.time_ex%partition_num==parition_id ]

    for index, row in merge_res.iterrows():
        pdq = get_pdq()
        print(f'index:{index}, Total:{len(pdq)}')

        for sn, order in enumerate(pdq):
            res = {}
            res['p'], res['d'], res['q'] = order
            res['day'] = 1+(pd.to_datetime(day) - pd.to_datetime('2019-01-01')).days
            res['day_str'] = day
            res['week_day'] = pd.to_datetime(day).weekday()
            res['stationID'] = row['stationID']
            res['time_ex'] = row['time_ex']
            res['direct'] = direct

            col_list = [col for col in merge_res.columns if direct in col]
            # print(col_list)
            X = row.loc[col_list]
            size = len(X) - 1
            train, test = X[0:size], X[size:len(X)]
            history = [x for x in train]
            # res = []

            try:
                predictions = list()

                model = ARIMA(history, order=order)  # (5,1,0)
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[0]
                # print(direct)
                res[f'obs'] = obs
                res[f'predict'] = round(yhat[0], 2)

                history.append(obs)
                # print('predicted=%f, expected=%f' % (yhat, obs))
                # print(len(test), len(predictions),  )
                error = mean_squared_error(test, predictions)
                # res.append((error, order))
                print(f"{index:06},{direct.rjust(3,' ')},  [{error:04.1f}], {order}, {obs}, {yhat[0]:06.2f}")
            except  ValueError as e:
                logger.debug(f'pdq:{order} incorrect ')
                # logger.exception(e)
                res = {}

            except  LinAlgError as e:
                logger.debug(f'pdq:{order} incorrect, SVD did not converge')
                res = {}
            insert(res)
            # res = sorted(res, key=lambda val: val[0])
            print(index, res)
    return len(merge_res)


def get_pdq():
    p = range(4, 8)
    d = range(0, 3)
    q = range(0, 8)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    return pdq

if __name__ == '__main__':

    day_list = ['2019-01-19']
    direct = ['in','out']
    partition_id = range(partition_num)

    para_list = list(itertools.product(day_list, direct, partition_id))
    try:
        pool = ThreadPool(thred_num)
        logger.info(f'There are {len(para_list)} para need to process  with {thred_num} threds')
        pool.map(process_partition, para_list, chunksize=np.random.randint(1,64))

    except Exception as e:
        logger.exception(e)