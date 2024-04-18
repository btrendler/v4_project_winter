from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import numpy as np


"""
Written by EJ Mercer

This file is under the MIT License. I would love to hear about anything you use this class for.

This is inspired by https://stackoverflow.com/questions/46325447/, but implemented separately
"""


class TimePlayer(FuncAnimation):
    def __init__(self, figure, time_ax, param_list, update, t_per_sec=60, ms_per_frame=200, ctrl_pos=(0.125, 0.92), t_span=600):
        # Save passed-in parameters
        self.figure = figure
        self.param_names = [v[0] for v in param_list]
        self.param_values = [v[1] for v in param_list]
        self.param_func = update
        self.time_ax = time_ax
        self.t_span = t_span

        # Construct initial state information
        self.i = 0
        self.t_per_sec = t_per_sec
        self.interval_gap = t_per_sec * 1000 / ms_per_frame
        self.run_rate = 1
        self.generated = [0, t_span]
        self.axis_limit = np.array([-t_span, 0])
        self.param_changed = False

        # Initialize controls
        self._set_up_ui(ctrl_pos)

        # Call superclass constructor
        FuncAnimation.__init__(self, self.figure, self._anim_func, frames=self._play(), interval=ms_per_frame)

        # Initialize plot & axes
        time_ax.set_xlim(self.axis_limit)
        self.param_func(self.param_values, self.generated)


    def _set_up_ui(self, ctrl_pos):
        # Allocate the space & get a divider for the space
        ax_ctrls = self.figure.add_axes([*ctrl_pos, 0.32, 0.04])
        div = mpl_toolkits.axes_grid1.make_axes_locatable(ax_ctrls)

        # Construct the pause, play, and fast-forward buttons
        self.btn_pause = Button(ax_ctrls, label=u"▌▌", hovercolor='0.975')
        self.btn_pause.on_clicked(self._chng_play_state("pause"))
        self.btn_play = Button(div.append_axes("right", size="80%", pad=0.05), label="▶", hovercolor='0.975')
        self.btn_play.on_clicked(self._chng_play_state("play"))
        self.btn_ffwd = Button(div.append_axes("right", size="80%", pad=0.05), label="▶▶", hovercolor='0.975')
        self.btn_ffwd.on_clicked(self._chng_play_state("ffwd"))

        # Construct a slider for each value
        self.figure.subplots_adjust(bottom=0.25)
        p_sliders = []
        for i, ((n, mi, mx), v) in enumerate(zip(self.param_names, self.param_values)):
            ax = self.figure.add_axes([0.22, 0.1 - (0.03 * i), 0.65, 0.03])
            p_sliders.append(Slider(ax=ax, label=n, valmin=mi, valmax=mx, valinit=v, dragging=False))
            p_sliders[-1].on_changed(self._chng_params(i))
        self.p_sliders = p_sliders

        # Construct a slider for each state


    def _anim_func(self, _):
        # Update the interval
        self.axis_limit = self.axis_limit + self.interval_gap * self.run_rate

        # Call the update function, if necessary according to parameter changes
        if self.param_changed:
            # If we're within 10% of the end of the interval, extend the interval
            if abs(self.generated[1] - self.axis_limit[1]) < 0.1 * self.t_span:
                self.generated[1] += self.t_span

            # Actually regenerate from the current time onwards
            self.param_func(self.param_values, (self.axis_limit[1], self.generated[1]))
            self.param_changed = False

        # Call the update function, if necessary according to the window
        elif self.axis_limit[1] > self.generated[1]:
            self.param_func(self.param_values, [self.generated[1], self.generated[1] + self.t_span])
            self.generated[1] += self.t_span

        # Apply the change to the axes
        self.time_ax.set_xlim(self.axis_limit)

    def _chng_play_state(self, type):
        def _ret(_):
            if type == "pause":
                self.btn_pause.color = "0.55"
                self.run_rate = 0
                self.event_source.stop()
            elif type == "play":
                self.btn_pause.color = "0.85"
                self.run_rate = 1
                self.event_source.start()
            elif type == "ffwd":
                self.btn_pause.color = "0.85"
                if self.run_rate == 0:
                    self.run_rate = 1
                self.run_rate *= 2
                self.run_rate = min(self.run_rate, int(0.8 * self.t_span / self.t_per_sec)) # todo: make sense?
                self.event_source.start()
        return _ret

    def _chng_params(self, idx):
        def _ret(val):
            self.param_changed = self.param_values[idx] != val
            self.param_values[idx] = val
        return _ret

    def _play(self):
        while self.run_rate != 0:
            self.i += 1
            yield self.i
        pass
