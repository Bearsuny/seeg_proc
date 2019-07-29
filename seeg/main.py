from seeg.io import EDF, output_dict
from seeg.config import PathConfig
from seeg.analysis import extract_marks, reform_eprime_info
from seeg.plot import Plot
import numpy as np

if __name__ == "__main__":
    edf = EDF(PathConfig.RAW_SEEG)
    edf.read_header()
    output_dict(edf.header)
    edf.read_data()
    channels_name=['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09']
    edf.save_channels(channels_name)

    extract_marks(channels_name=channels_name, threshold=2.5)
    reform_eprime_info(PathConfig.RAW_EPRIME, 2000)

    eprime_data = np.load(PathConfig.EPRIME)*2
    mark_path = PathConfig.SUBJECT/f'{"_".join(channels_name)}.npy'
    mark_data = np.load(mark_path)[:, 117442:]

    test_plot = Plot()
    test_plot.init_figure(n_rows=4,
                          n_cols=1,
                          figsize=(16, 9),
                          y_lim=[-1, 5],
                          y_labels=['DC12', 'DC11', 'DC10', 'DC09'],
                          line_colors=['b', 'r'])
    test_plot.init_data(data=[eprime_data, mark_data],
                        start_sec=0,
                        step_sec=1,
                        sample_frequency=2000,
                        n_init_steps=10)

    test_plot.init_handler(wait_time=0.1)
    test_plot.show()
    pass
