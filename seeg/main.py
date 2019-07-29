from seeg.io import EDF, output_dict
from seeg.config import PathConfig
from seeg.analysis import extract_marks, reform_eprime_info, save_marks_seed_data
from seeg.plot import Plot
import numpy as np

if __name__ == "__main__":
    # edf = EDF(PathConfig.RAW_SEEG)
    # edf.read_header()
    # output_dict(edf.header)
    # edf.read_data()
    # channels_name=['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09']
    # edf.save_channels(channels_name)

    # extract_marks(channels_name=channels_name, threshold=2.8)
    # reform_eprime_info(PathConfig.RAW_EPRIME, 1000)

    # eprime_data = np.load(PathConfig.EPRIME)*2
    # mark_path = PathConfig.SUBJECT/f'{"_".join(channels_name)}.npy'
    # mark_data = np.load(mark_path)[:, 117442:]

    # test_plot = Plot()
    # test_plot.init_figure(n_rows=4,
    #                       n_cols=1,
    #                       figsize=(16, 9),
    #                       y_lim=[-1, 5],
    #                       y_labels=['DC12', 'DC11', 'DC10', 'DC09'],
    #                       line_colors=['b', 'r'])
    # test_plot.init_data(data=[eprime_data, mark_data],
    #                     start_sec=0,
    #                     step_sec=1,
    #                     sample_frequency=[1000, 2000],
    #                     n_init_steps=10)

    # test_plot.init_handler(wait_time=0.1)
    # test_plot.show()

    # save_marks_seed_data(channels_name=['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09'],
    #                      threshold=3,
    #                      marks_name=['1000', '1011', '1010', '1110', '0001', '1101', '1100', '1001'])

    marks_name = ['1000', '1011', '1010', '1110', '0001', '1101', '1100', '1001']
    subjects_no = ['015', '021', '024']
    data = [np.load(PathConfig.DATA/f'subject_{subject_no}'/f'mark_seed_{marks_name[2]}.npy')
            for subject_no in subjects_no]
    plot = Plot()
    plot.init_figure(n_rows=4,
                     n_cols=1,
                     figsize=(16, 9),
                     y_lim=[-1, 5],
                     y_labels=['DC12', 'DC11', 'DC10', 'DC09'],
                     line_colors=['b', 'r', 'g'])
    plot.init_data(data=data,
                   start_sec=0,
                   step_sec=0.1,
                   sample_frequency=[2000, 2000, 2000],
                   n_init_steps=5)

    plot.init_handler(wait_time=0.1)
    plot.show()
    pass
