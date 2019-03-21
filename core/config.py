train_file = './input/Metro_train/record_%s.csv'
test_file = './input/Metro_testA/testA_record_%s.csv'
submit_file = './input/Metro_testA/testA_submit_%s.csv' % '2019-01-29'
roadmap_file = './input/Metro_roadMap.csv'

thred_num = 10

partition_num = 24

test_feature_set = [22, 25, 28, 29]


sub_col = ['stationID',	'startTime',	'endTime',	'outNums',	'inNums']

mysql_pass = 'Had00p!!'
#11664

gp_list = ['time_ex', 'stationID', 'status']

model_paras=['week_day','stationID','time_ex', 'bin_id',  'p', 'd','q', ]

for day in [1, 2, 3]:
    for col in [f'out_{day}', f'in_{day}', f'out_{day}_p', f'in_{day}_p',]:
        model_paras.append(col)

print(model_paras)