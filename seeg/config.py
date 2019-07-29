from pathlib import Path


class PathConfig:
    ROOT = Path().resolve()
    DATA = ROOT/'data'

    subject_no = '024'
    SUBJECT = DATA/f'subject_{subject_no}'

    RAW_SEEG = DATA/SUBJECT/f'{subject_no}_LAT.edf'
    RAW_EPRIME = DATA/SUBJECT/f'{subject_no}_eprime.csv'
    EPRIME = DATA/SUBJECT/f'{subject_no}_eprime.npy'
