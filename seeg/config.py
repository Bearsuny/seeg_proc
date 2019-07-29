from pathlib import Path


class PathConfig:
    ROOT = Path().resolve()
    DATA = ROOT/'data'

    SUBJECT = DATA/'subject_024'

    RAW_SEEG = DATA/SUBJECT/'024_LAT.edf'
    RAW_EPRIME = DATA/SUBJECT/'024_eprime.csv'
    EPRIME = DATA/SUBJECT/'024_eprime.npy'
