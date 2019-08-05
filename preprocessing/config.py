from pathlib import Path


class PathConfig:
    ROOT = Path().resolve()
    DATA = ROOT/'data'

    subject_no = '024'
    SUBJECT = DATA/f'{subject_no}'

    RAW_SEEG = DATA/SUBJECT/f'LAT.edf'
    RAW_EPRIME = DATA/SUBJECT/f'eprime.csv'
    EPRIME = DATA/SUBJECT/f'eprime.npy'
    EPRIME_REFORM = DATA/SUBJECT/f'eprime_reform.csv'
    MARK = DATA/SUBJECT/f'mark.csv'
    EVENT = DATA/SUBJECT/f'event.csv'


class AnalysisConfig:
    MARK_CHANNELS_NAME = ['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09']
    MARKS_NAME = {'8': 'block', '9': 'block',
                  '10': 'response', '1': 'response',
                  '14': 'condition', '11': 'condition', '12': 'condition', '13': 'condition'}
