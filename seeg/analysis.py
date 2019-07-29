import numpy as np
from seeg.config import PathConfig
from tqdm import tqdm
import matplotlib.pyplot as plt

import pandas as pd


def extract_marks(channels_name, threshold):
    data = np.load(PathConfig.SUBJECT/f'{"_".join(channels_name)}.npy').T
    marks_info = {}
    history_result = None
    for i, data_item in tqdm(enumerate(data), total=data.shape[0]):
        result = ''.join(['1' if _ else '0' for _ in np.where(data_item > threshold, True, False)])
        if result != history_result:
            history_result = result
            if result != '0000':
                if result not in marks_info.keys():
                    marks_info[result] = []
                marks_info[result].append(i)
    total = []
    for k, v in sorted(marks_info.items()):
        print(f'{int(k, 2):02}: {len(v)}')
        total.append(len(v))
    print(f'Total marks in SEEG: {sum(total)}')
    print(marks_info['1000'])
    print(marks_info['1001'])


def reform_eprime_info(eprime_info_path, sample_frequency):
    eprime_info = pd.read_csv(eprime_info_path, sep='\t')
    mark_series = []
    time_series = []
    for i, item in eprime_info.iterrows():
        if not pd.isnull(item['mark_1']):
            mark_series.append(item['mark_1'])
            time_series.append(item['time_1'])
        if not pd.isnull(item['mark_2']):
            mark_series.append(item['mark_2'])
            time_series.append(item['time_2'])
    mark_series = np.array(mark_series, dtype=np.int).T
    time_series = np.array(time_series, dtype=np.int).T
    eprime_info = pd.DataFrame()
    eprime_info['mark'] = mark_series
    eprime_info['time'] = time_series - time_series[0]  # millisecond
    eprime_info['sample'] = eprime_info['time'] / 1000 * sample_frequency
    eprime_info['sample'] = eprime_info['sample'].astype('int')
    print(f'Mark num in Eprime: {eprime_info.shape[0]}')
    print(eprime_info.groupby(eprime_info['mark']).count())

    eprime_channel_info = np.zeros((eprime_info['sample'].values[-1], 4))
    for i in tqdm(range(eprime_info['sample'].values[-1])):
        if i not in eprime_info['sample'].values:
            eprime_channel_info[i] = np.array([0, 0, 0, 0])
        else:
            idx = list(eprime_info['sample'].values).index(i)
            mark = eprime_info['mark'].values[idx]
            template = list('0000')
            temp = list(str(bin(int(mark)))[2:])
            template[4-len(temp):] = temp
            eprime_channel_info[i] = np.array(template).astype(np.int)

    eprime_channel_info = eprime_channel_info.T
    np.save(PathConfig.EPRIME, eprime_channel_info)


def save_marks_seed_data(channels_name, threshold, marks_name):
    data = np.load(PathConfig.SUBJECT/f'{"_".join(channels_name)}.npy').T
    marks_info = {}
    history_result = None
    for i, data_item in tqdm(enumerate(data), total=data.shape[0]):
        result = ''.join(['1' if _ else '0' for _ in np.where(data_item > threshold, True, False)])
        if result != '0000' and result in marks_name:
            if result not in marks_info.keys():
                marks_info[result] = True
                np.save(PathConfig.SUBJECT/f'mark_seed_{result}.npy', data[i-200: i+800, :].T)


def compare_eprime_seeg(channels_name, threshold, marks_name, eprime_info_path, sample_frequency):
    data = np.load(PathConfig.SUBJECT/f'{"_".join(channels_name)}.npy').T
    history_result = None
    eprime_info = pd.read_csv(eprime_info_path, sep='\t')
    mark_series = []
    time_series = []
    for i, item in eprime_info.iterrows():
        if not pd.isnull(item['mark_1']):
            mark_series.append(item['mark_1'])
            time_series.append(item['time_1'])
        if not pd.isnull(item['mark_2']):
            mark_series.append(item['mark_2'])
            time_series.append(item['time_2'])
    mark_series = np.array(mark_series, dtype=np.int).T
    time_series = np.array(time_series, dtype=np.int).T
    eprime_info = pd.DataFrame()
    eprime_info['mark'] = mark_series
    eprime_info['time'] = time_series - time_series[0]

    count = 0
    result_series = []
    flag = False
    count_2 = 0
    value_result_series = []
    for i, data_item in tqdm(enumerate(data),total=data.shape[0]):
        result = ''.join(['1' if _ else '0' for _ in np.where(data_item > threshold, True, False)])
        result_series.append(result)
        if flag:
            if count_2 < 10:
                print(f'{i}, {result}')
                count_2 += 1
            else:
                flag = False
        if result != history_result:
            history_result = result
            if result != '0000':
                value_result_series.append(result)
                eprime_result = eprime_info['mark'][count]
                count += 1
                if int(result, 2) != int(eprime_result):
                    print(f'{i}, N {count}: {int(result, 2)} vs {eprime_result}')
                    print(data_item)
                    print(result_series[-10:])
                    flag = True
                    count_2 = 0
                # else:
                #     print(f'Y {count}: {int(result, 2)} vs {eprime_result}')
                #     print(data_item)

    # print(len(value_result_series))


if __name__ == "__main__":
    # extract_marks(channels_name=['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09'], threshold=2.5)
    # reform_eprime_info(PathConfig.RAW_EPRIME, 1000)
    # save_marks_seed_data(channels_name=['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09'],
    #                      threshold=2.5,
    #                      marks_name=['1000', '1011', '1010', '1110', '0001', '1101', '1100', '1001'])
    compare_eprime_seeg(channels_name=['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09'],
                        threshold=2.8,
                        marks_name=['1000', '1011', '1010', '1110', '0001', '1101', '1100', '1001'],
                        eprime_info_path=PathConfig.RAW_EPRIME,
                        sample_frequency=1000)
    pass
