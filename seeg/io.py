'''
EDF Format: https://www.edfplus.info/specs/edf.html
MNE Code: https://github.com/mne-tools/mne-python/blob/master/mne/io/edf/edf.py
'''

import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from seeg.config import PathConfig


def output_dict(dict_data):
    screen_width = os.get_terminal_size().columns

    def _func(string):
        string = string.replace('\n', '')
        if len(string) > screen_width-40:
            return string[:screen_width-45] + '...'
        else:
            return string

    for key, value in dict_data.items():
        value_type = re.findall(r'\'(.*)\'', str(type(value)))[0]
        if isinstance(value, np.ndarray):
            print(f'{key:20}|{value_type:20}|{_func(str(value))}')
        else:
            print(f'{key:20}|{value_type:20}|{value}')


class BaseData:
    def __init__(self, record_path):
        self.record_path = record_path
        self.record_file = None
        self.header = {}
        self.data = None

    def read_header(self):
        raise NotImplementedError

    def read_data(self):
        raise NotImplementedError

    def pick_channels(self, channels_name):
        raise NotImplementedError

    def plot_channels(self, channels_name):
        raise NotImplementedError


class EDF(BaseData):
    def __init__(self, record_path):
        super(EDF, self).__init__(record_path)

    def _read(self, n_bytes, n_channels=None):
        if n_channels:
            data = np.array([self.record_file.read(n_bytes).decode().strip() for _ in range(n_channels)])
            return None if (data == '').all() else data
        else:
            return self.record_file.read(n_bytes).decode().strip()

    def read_header(self):
        with open(self.record_path, 'rb') as raw_edf:
            self.record_file = raw_edf

            self.header['version'] = self._read(8)
            self.header['patient_id'] = self._read(80)
            self.header['record_id'] = self._read(80)
            self.header['start_date'] = self._read(8)
            self.header['start_time'] = self._read(8)
            self.header['n_header_bytes'] = int(self._read(8))
            self.header['reserved_area_1'] = self._read(44)
            self.header['n_data_blocks'] = int(self._read(8))
            # sample length (s) in each block for all channels
            self.header['sample_length'] = float(self._read(8))
            self.header['n_channels'] = int(self._read(4))
            n_channels = self.header['n_channels']
            self.header['channels_name'] = self._read(16, n_channels)
            self.header['transducer_type'] = self._read(80, n_channels)
            self.header['physical_dim'] = self._read(8, n_channels)
            self.header['physical_min'] = self._read(8, n_channels).astype(np.float)
            self.header['physical_max'] = self._read(8, n_channels).astype(np.float)
            self.header['digital_min'] = self._read(8, n_channels).astype(np.float)
            self.header['digital_max'] = self._read(8, n_channels).astype(np.float)
            self.header['prefiltering'] = self._read(80, n_channels)
            # n_samples / sample_length (s) = sample frequency (Hz)
            self.header['n_samples'] = self._read(8, n_channels).astype(np.int)
            self.header['reserved_area_2'] = self._read(32, n_channels)

            # extra info
            self.header['sample_frequency'] = self.header['n_samples'][0] // self.header['sample_length']

    def read_data(self):
        with open(self.record_path, 'rb') as raw_edf:
            # move file pointer to the end of file
            raw_edf.seek(0, 2)
            n_record_bytes = raw_edf.tell()
            # dtype_bytes=2, dtype_np=np.int16
            n_data_bytes = (n_record_bytes - self.header['n_header_bytes']) // 2

            # skip header
            raw_edf.seek(self.header['n_header_bytes'], 0)
            self.record_file = raw_edf

            # gain constructor
            physical_ranges = self.header['physical_max'] - self.header['physical_min']
            digital_ranges = self.header['digital_max'] - self.header['digital_min']
            cals = np.atleast_2d(physical_ranges / digital_ranges)
            gains = map(lambda x: 1e-6 if x == 'uV' else 1, self.header['physical_dim'])
            gains = np.atleast_2d([item for item in gains])
            offsets = np.atleast_2d(self.header['physical_min'] - (self.header['digital_min'] * cals)).T

            self.data = np.zeros(n_data_bytes).reshape(self.header['n_channels'], -1).astype(np.float)
            n_block_bytes = n_data_bytes//self.header['n_data_blocks']
            # item: data item for each channel in each block
            n_item_bytes = n_block_bytes // self.header['n_channels']

            for i in tqdm(range(self.header['n_data_blocks'])):
                block = np.fromfile(self.record_file,
                                    dtype=np.int16,
                                    count=n_block_bytes)
                for j in range(self.header['n_channels']):
                    item = block[j*n_item_bytes:(j+1)*n_item_bytes]
                    self.data[j][i*n_item_bytes:(i+1)*n_item_bytes] = item

            self.data *= cals.T
            self.data += offsets
            self.data *= gains.T

    def save_channels(self, channels_name):
        channels_idx = [list(self.header['channels_name']).index(name) for name in channels_name]
        np.save(PathConfig.SUBJECT/f'{"_".join(channels_name)}.npy', self.data[channels_idx])


if __name__ == "__main__":
    edf = EDF(PathConfig.RAW_SEEG)
    edf.read_header()
    output_dict(edf.header)
    edf.read_data()
    edf.save_channels(['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09'])

    pass
