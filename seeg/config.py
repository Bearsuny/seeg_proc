from pathlib import Path


class PathConfig:
    ROOT = Path().resolve()
    DATA = ROOT/'data'
    SEEG_TEST = DATA/'subject_024'/'024_LAT.edf'
    MARK_TEST = DATA/'subject_024'/'mark.npy'
