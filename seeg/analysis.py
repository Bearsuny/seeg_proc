import numpy as np
from seeg.config import PathConfig
from tqdm import tqdm


def extract_marks():
    data = np.load(PathConfig.MARK_TEST).T
    print(data.shape)
    marks_info = {}
    for i, data_item in tqdm(enumerate(data), total=data.shape[0]):
        result = ''.join(['1' if _ else '0' for _ in np.where(data_item > 1, True, False)])
        result = result[::-1]
        if result not in marks_info.keys():
            marks_info[result] = []
        marks_info[result].append(i)
    for k, v in sorted(marks_info.items()):
        print(f'{int(k, 2):02}: {len(v)}')


extract_marks()
