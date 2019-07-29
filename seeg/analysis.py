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
        if result != '0000':
            if result not in marks_info.keys():
                marks_info[result] = []
            if result != history_result:
                marks_info[result].append(i)
                history_result = result
    total = []
    for k, v in sorted(marks_info.items()):
        print(f'{int(k, 2):02}: {len(v)}')
        total.append(len(v))
    print(f'Total marks: {sum(total)}')
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


if __name__ == "__main__":
    extract_marks(channels_name=['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09'], threshold=2.5)
    reform_eprime_info(PathConfig.RAW_EPRIME, 2000)
    pass
