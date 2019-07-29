import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from threading import Thread
from matplotlib.widgets import Button
from seeg.config import PathConfig


class ButtonHandler:
    def __init__(self, plot_func, wait_time):
        self.plot_func = plot_func
        self.wait_time = wait_time

        self.seed_flag = False  # False: stop, True: start
        self.direction_flag = True  # False: backward, True: forward

        self.thread = Thread(target=self.update)
        self.thread.setDaemon(True)
        self.thread.start()

    def update(self):
        while True:
            sleep(self.wait_time)
            if self.seed_flag:
                self.plot_func(self.direction_flag)

    def forward(self, event):
        self.seed_flag = True
        self.direction_flag = True

    def backward(self, event):
        self.seed_flag = True
        self.direction_flag = False

    def pause(self, event):
        self.seed_flag = False


class Plot:
    def __init__(self):
        pass

    def init_figure(self, n_rows, n_cols, figsize, y_lim, y_labels, line_colors):
        _, self.axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        self.line_colors = line_colors
        for i, ax_item in enumerate(self.axes):
            ax_item.set(ylim=y_lim, ylabel=y_labels[i])

    def init_data(self, data, start_sec, step_sec, sample_frequency, n_init_steps):
        self.data = data
        self.range_start = start_sec * sample_frequency
        self.step = step_sec * sample_frequency
        self.range_end = self.range_start + self.step * n_init_steps
        self.sample_frequency = sample_frequency

        self.lines = []
        x_data = np.arange(self.range_start, self.range_end) / sample_frequency
        for i, ax_item in enumerate(self.axes):
            for j, data_item in enumerate(self.data):
                y_data = data_item[i, self.range_start:self.range_end]
                line_item = ax_item.plot(x_data, y_data, self.line_colors[j])[0]
                self.lines.append(line_item)
                plt.draw()

    def init_handler(self, wait_time):
        self.callback = ButtonHandler(plot_func=self.plot, wait_time=wait_time)

        self.backward_button = Button(plt.axes([0.55, 0.05, 0.1, 0.05]), 'Backward')
        self.backward_button.on_clicked(self.callback.backward)

        self.forward_button = Button(plt.axes([0.85, 0.05, 0.1, 0.05]), 'Forward')
        self.forward_button.on_clicked(self.callback.forward)

        self.pause_button = Button(plt.axes([0.7, 0.05, 0.1, 0.05]), 'Pause')
        self.pause_button.on_clicked(self.callback.pause)

        plt.subplots_adjust(left=0.075, right=0.95, bottom=0.15, top=0.95, wspace=0.25, hspace=0.25)

    def plot(self, direction):
        if direction:
            self.range_start += self.step
            self.range_end += self.step
        else:
            self.range_start -= self.step
            self.range_end -= self.step
        if self.range_end > self.data[0].shape[1] or self.range_start < 0:
            return
        x_data = np.arange(self.range_start, self.range_end) / self.sample_frequency
        for i, ax_item in enumerate(self.axes):
            for j, data_item in enumerate(self.data):
                y_data = data_item[i, self.range_start:self.range_end]
                self.lines[i*len(self.data)+j].set_xdata(x_data)
                self.lines[i*len(self.data)+j].set_ydata(y_data)
                ax_item.set(xlim=[x_data[0], x_data[-1]])
                plt.draw()

    def show(self):
        plt.show()


if __name__ == "__main__":
    eprime_data = np.load(PathConfig.EPRIME)*2
    channels_name=['POL DC12', 'POL DC11', 'POL DC10', 'POL DC09']
    mark_path = PathConfig.SUBJECT/f'{"_".join(channels_name)}.npy'
    mark_data = np.load(mark_path)[:, 117442:]

    test_plot = Plot()
    test_plot.init_figure(n_rows=4,
                          n_cols=1,
                          figsize=(16, 9),
                          y_lim=[-1, 5],
                          y_labels=['DC12', 'DC11', 'DC10', 'DC09'],
                          line_colors=['b', 'r'])
    test_plot.init_data(data=[eprime_data],
                        start_sec=0,
                        step_sec=1,
                        sample_frequency=2000,
                        n_init_steps=10)

    test_plot.init_handler(wait_time=0.1)
    test_plot.show()
    pass
