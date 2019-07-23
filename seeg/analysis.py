import numpy as np
from seeg.config import PathConfig
from tqdm import tqdm


def extract_marks():
    data = np.load(PathConfig.MARK_TEST).T
    print(data.shape)
    marks_mode = {'00': '0000',  # Default
                  '01': '0001',
                  '08': '1000',
                  '09': '1001',
                  '10': '1010',
                  '11': '1011',
                  '12': '1100',
                  '13': '1101',
                  '14': '1110'}
    marks_info = {}
    for i, data_item in tqdm(enumerate(data), total=data.shape[0]):
        result = ''.join(['1' if _ else '0' for _ in np.where(data_item > 2, True, False)])
        if result not in marks_info.keys():
            marks_info[result] = []
        marks_info[result].append(i)
    for k, v in sorted(marks_info.items()):
        print(f'{int(k, 2):02}: {len(v)}')


extract_marks()

