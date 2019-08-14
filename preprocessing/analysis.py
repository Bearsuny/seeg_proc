import numpy as np
from preprocessing.config import PathConfig, AnalysisConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.cluster

import pandas as pd


def reform_seeg_mark(seeg_mark_channel_path):
    data = np.load(seeg_mark_channel_path)
    data = data.T

    model = sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, p=2)
    seeg_mark = pd.DataFrame()
    step = 10000
    # for sample_start in tqdm(range(0, data.shape[0], step)):
    for sample_start in range(0, data.shape[0], step):
        batch = data[sample_start:sample_start+step]
        predict = model.fit_predict(batch)
        batch = pd.DataFrame(batch)
        batch['predict'] = predict
        batch.index = [i for i in range(sample_start, sample_start+len(batch))]

        mark_group_stat = batch.groupby(batch['predict']).mean()
        for mark in mark_group_stat.iterrows():
            name = mark[1].name
            values = mark[1].values
            label = ['0' if int(value_item) == 0 else '1' for value_item in values]
            label = ''.join(label)
            label = int(label, 2)
            if name != -1:
                batch.loc[batch.predict == name, 'predict'] = label

        seeg_mark = pd.concat([seeg_mark, batch])

    seeg_mark.to_csv(PathConfig.MARK, columns=['predict'])


def build_event(seeg_mark_path):
    predict = pd.read_csv(seeg_mark_path)
    history = None
    event = []
    # for predict_item in tqdm(predict.iterrows(), total=predict.shape[0]):
    for predict_item in predict.iterrows():
        predict_value = predict_item[1].values
        if str(predict_value[1]) != history:
            history = str(predict_value[1])
            if history in AnalysisConfig.MARKS_NAME.keys():
                event.append((AnalysisConfig.MARKS_NAME[str(history)], history, predict_value[0]))
    event = np.array(event)
    event = pd.DataFrame(event, columns=['type', 'value', 'sample'])
    event.to_csv(PathConfig.EVENT)


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
    eprime_info.to_csv(PathConfig.EPRIME_REFORM, columns=['mark', 'time'])

    eprime_channel_info = np.zeros((eprime_info['sample'].values[-1], 4))
    # for i in tqdm(range(eprime_info['sample'].values[-1])):
    for i in range(eprime_info['sample'].values[-1]):
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


def compare_eprime_mark_with_seeg_event():
    seeg_event = pd.read_csv(PathConfig.EVENT)
    eprime_mark = pd.read_csv(PathConfig.EPRIME_REFORM)
    print(seeg_event.groupby(seeg_event['value']).count())
    print(eprime_mark.groupby(eprime_mark['mark']).count())
    for i, (event, mark) in enumerate(zip(seeg_event.iterrows(), eprime_mark.iterrows())):
        if event[0] != mark[0]:
            print(f'{i}th mark error.')
    else:
        print('marks are identical to events.')



if __name__ == "__main__":
    pass
