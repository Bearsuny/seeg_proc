from pathlib import Path


class PathConfig:
    ROOT = Path().resolve()
    DATA = ROOT/'data'
    SEEG_TEST = DATA/'subject_021'/'021_LAT.edf'
    MARK_TEST = DATA/'subject_021'/'mark.npy'
