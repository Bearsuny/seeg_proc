from preprocessing.io import EDF, output_dict
from preprocessing.config import PathConfig, AnalysisConfig
from preprocessing.analysis import reform_eprime_info, reform_seeg_mark, build_event, compare_eprime_mark_with_seeg_event
from preprocessing.plot import Plot
import numpy as np
    

if __name__ == "__main__":
    edf = EDF(PathConfig.RAW_SEEG)
    edf.read_header()
    output_dict(edf.header)
    edf.save_channels_name()
    edf.read_data()
    edf.save_channels(AnalysisConfig.MARK_CHANNELS_NAME)

    reform_eprime_info(PathConfig.RAW_EPRIME, 1000)
    reform_seeg_mark(PathConfig.SUBJECT/f'{"_".join(AnalysisConfig.MARK_CHANNELS_NAME)}.npy')
    build_event(PathConfig.MARK)
    compare_eprime_mark_with_seeg_event()
    pass
