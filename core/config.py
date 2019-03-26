import numpy as np
train_file = './input/Metro_train/record_%s.csv'
test_file = './input/Metro_testA/testA_record_%s.csv'
submit_file = './input/Metro_testA/testA_submit_%s.csv' % '2019-01-29'
roadmap_file = './input/Metro_roadMap.csv'

thred_num = 10

partition_size = 6.0 * 4
partition_num = int(np.ceil(144 / partition_size))

select_list = [1,2,3,4 , #5 , 6,
               7,8,9,10,11,  #12, 13,
               14, 15, 16, 17, 18,  #19 , 20
               21, 22, 23, 24, 25,   #26, 27,
               28
               ]

select_list =  [item for item in range(1,29) if item%7 not in [5,6] ]

test_feature_set = [22, 25, 28, 29]


sub_col = ['stationID',	'startTime',	'endTime',	'outNums',	'inNums']


gp_list = ['time_ex', 'stationID', 'status']

model_paras=['week_day','stationID','time_ex', 'bin_id',  'p', 'd','q', ]

for day in [1, 2, 3]:
    for col in [f'out_{day}', f'in_{day}', f'out_{day}_p', f'in_{day}_p',]:
        model_paras.append(col)

print(model_paras)