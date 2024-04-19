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
    """
    Creates an animated plot that pans to the right through time, Mario-platformer style.
    This allows tuning the model parameters in real-time, and pausing to tune the current state to see results.

    Example usage:
        fig, ax = plt.subplots()
        line, = ax.plot(np.array([]), np.array([]), label="Wave")
        ax.legend(loc="lower right")
        ax.set_ylim((-10, 10))

        params = [(("Amplitude", 0., 10.), 3.), (("Frequency", 0.1, 30.), 3.)]
        state = [(("n", -10., 10.), 0.)]

        Note that this does not correctly account for the start_state, and will produce odd results because of it,
            but I can't come up with a better example function right now.
        def update(params_list, interval, start_state):
            print("Updating on interval", interval, "beginning at", start_state)
            # Extract parameters
            amp, freq = params_list

            # Bound the start_state to the amplitude
            start_state = max(min(amp, start_state[0]), -amp)

            # Get the x and y data currently in the line
            xv = line.get_xdata()
            yv = line.get_ydata()

            # Truncate the data to the input lower bound
            yv = yv[xv < interval[0]]
            xv = xv[xv < interval[0]]

            # Compute the new values
            xv_new = np.linspace(interval[1], interval[0], 1000, endpoint=False)[::-1]
            yv_new = f(xv_new - xv_new[0] + f_inv(start_state, amp, freq), amp, freq)

            # Store the new values into the line
            line.set_xdata(x_full := np.concatenate((xv, xv_new)))
            line.set_ydata(y_full := np.concatenate((yv, yv_new)))

            # Return the computed x and ys
            return x_full, y_full.reshape((1, -1))


        time_player.TimePlayer(fig, ax, params, state, update, t_per_sec=0.001, t_span=2, ms_per_frame=50)
        plt.show()

    """

    def __init__(self, figure, time_ax, param_list, state_list, update, t_per_sec=60, ms_per_frame=200,
                 ctrl_pos=(0.125, 0.92), t_span=600, slider_height=0.45, calc_frac=0.25):
        """
        Initialize a TimePlayer

        The list of parameters should be a list of tuples, where the first element is a tuple containing the
        name of the parameter, the min, and max, and the second element is the initial value.
        This is similar for the list of states.

        The update function should change the line values and return the full x and y points found. It should
        accept the list of parameter values, in the same order as specified in the param_list, the interval
        on which to solve, and the starting state on the left-hand boundary of the interval. The interval should
        be treated as open on the left end, closed on the right end.

        :param figure: The figure to animate
        :param time_ax: The axis or list of axes to pan (x_limit will be changed)
        :param param_list: The list of parameters, as defined above
        :param state_list: The list of states, as defined above
        :param update: The function that updates the lines, as defined above
        :param t_per_sec: The amount of time that passes per second (default is 60 seconds/second)
        :param ms_per_frame: The milliseconds per frame (default is 200)
        :param ctrl_pos: The position of the controls
        :param t_span: The amount of time the window should span, in seconds (default is ten minutes)
        :param slider_height: The percentage (between 0 and 1) of the graph's vertical space that should be allocated to the sliders
        :param calc_frac: The percentage (between 0 and 1) of the horizontal interval to calculate at any given time
        """
        # Save passed-in parameters
        self.figure = figure
        self.state_names = [v[0] for v in state_list]
        self.state_values = [v[1] for v in state_list]
        self.param_names = [v[0] for v in param_list]
        self.param_values = [v[1] for v in param_list]
        self.param_func = update
        self.time_ax = time_ax
        self.t_span = t_span
        self.calc_frac = calc_frac

        # Construct initial state information
        self.i = 0
        self.t_per_sec = t_per_sec
        self.interval_gap = t_per_sec * 1000 / ms_per_frame
        self.run_rate = 1
        self.generated = [0, t_span * calc_frac]
        self.axis_limit = np.array([-t_span, 0])
        self.param_changed = False
        self.state_changed = False

        # Initialize controls
        self._set_up_ui(ctrl_pos, slider_height)

        # Call superclass constructor
        FuncAnimation.__init__(self, self.figure, self._anim_func, frames=self._play(), interval=ms_per_frame,
                               cache_frame_data=False)

        # Initialize plot & axes
        for (t, _) in self.time_ax:
            t.set_xlim(self.axis_limit)
        self.x, self.ys = self.param_func(self.param_values, self.generated, self.state_values)

    def _set_up_ui(self, ctrl_pos, slider_height):
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
        self.figure.subplots_adjust(bottom=slider_height)
        indiv_height = slider_height * 0.9 / (len(self.param_values) + len(self.state_values))

        p_sliders = []
        indiv = 1.5 if (len(self.param_values) + len(self.state_values)) > 4 else 1
        for i, ((n, mi, mx), v) in enumerate(zip(self.param_names, self.param_values)):
            ax = self.figure.add_axes([0.22, slider_height - (indiv_height * (i + indiv)), 0.65, 0.03])
            p_sliders.append(Slider(ax=ax, label=n, valmin=mi, valmax=mx, valinit=v, dragging=False))
            p_sliders[-1].on_changed(self._chng_params(i))
        self.p_sliders = p_sliders
        n_sliders = len(p_sliders)

        # Construct a slider for each state
        s_sliders = []
        for i, ((n, mi, mx), v) in enumerate(zip(self.state_names, self.state_values)):
            ax = self.figure.add_axes([0.22, slider_height - (indiv_height * (i + indiv + n_sliders)), 0.65, 0.03])
            s_sliders.append(Slider(ax=ax, label=n, valmin=mi, valmax=mx, valinit=v, dragging=True, color="grey"))
            s_sliders[-1].on_changed(self._chng_state(i))
            s_sliders[-1].set_active(False)
        self.s_sliders = s_sliders

    def _anim_func(self, _):
        # Update the interval
        self.axis_limit = self.axis_limit + self.interval_gap * self.run_rate

        # Call the update function, if necessary according to parameter changes
        if self.param_changed:
            # If we're within 10% of the end of the interval, extend the interval
            if abs(self.generated[1] - self.axis_limit[1]) < 0.02 * self.t_span:
                self.generated[1] += self.t_span * self.calc_frac

            # Actually regenerate from the current time onwards
            self.x, self.ys = self.param_func(
                self.param_values,
                (self.axis_limit[1], self.generated[1]),
                self.ys[:, self.x < self.axis_limit[1]][:, -1]
            )
            self.param_changed = False

        # Call the update function, if necessary according to the window
        elif self.axis_limit[1] > self.generated[1]:
            self.x, self.ys = self.param_func(
                self.param_values,
                [self.generated[1], self.generated[1] + self.t_span * self.calc_frac],
                self.ys[:, -1]
            )
            self.generated[1] += self.t_span * self.calc_frac

        # Apply the change to the axes
        for (t, l) in self.time_ax:
            t.set_xlim(self.axis_limit)
            if l is not None:
                l, i, n = l
                v = np.abs(float(self.ys[i, self.x < self.axis_limit[1]][-1]))
                l.set_title(f"{n}:{v: 9.4f}")

        # Update the state sliders to match the current values
        for i, sld in enumerate(self.s_sliders):
            v = self.ys[i, self.x < self.axis_limit[1]][-1]
            if sld.label.get_text() == "$\\bf{{L}}$eaving":
                v = np.abs(v)
            sld.set_val(v)

    def _chng_play_state(self, type):
        def _ret(_):
            # Handle the pause button
            if type == "pause":
                # Stop playback
                self.btn_pause.color = "0.55"
                self.run_rate = 0
                self.event_source.stop()

                # Update the state
                self.state_values = self.ys[:, self.x < self.axis_limit[1]][:, -1]

                # Make the sliders for params draggable
                for sld in self.p_sliders:
                    sld.connect_event("motion_notify_event", sld._update)
                    sld.m_cid = sld._cids[-1]

                # Make the sliders for state active
                for sld in self.s_sliders:
                    if sld.label.get_text() == "$\\bf{{L}}$eaving":
                        continue
                    sld.poly.set_facecolor((0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0))
                    sld.set_active(True)
                self.figure.canvas.draw_idle()
                return

            # Handle the play button
            elif type == "play":
                self.run_rate = 1

            # Handle the fast-forward button
            elif type == "ffwd":
                if self.run_rate == 0:
                    self.run_rate = 1
                self.run_rate *= 2
                while (self.interval_gap * self.run_rate) > (self.t_span * self.calc_frac):
                    self.run_rate -= 1

            # Update the solution if necessary
            if self.state_changed:
                self.x, self.ys = self.param_func(
                    self.param_values,
                    (self.axis_limit[1], self.generated[1]),
                    self.state_values
                )

            # Start animation
            self.btn_pause.color = "0.85"
            self.event_source.start()

            # Make the sliders for params not draggable
            for sld in self.p_sliders:
                sld.canvas.mpl_disconnect(sld.m_cid)
                sld._cids.remove(sld.m_cid)

            # Make the sliders for state inactive
            for sld in self.s_sliders:
                sld.poly.set_facecolor("grey")
                sld.set_active(False)
            self.figure.canvas.draw_idle()

        return _ret

    def _chng_params(self, idx):
        def _ret(val):
            self.param_changed = self.param_values[idx] != val
            self.param_values[idx] = val

        return _ret

    def _chng_state(self, idx):
        def _ret(val):
            if not self.run_rate:
                self.state_changed = self.state_values[idx] != val
                self.state_values[idx] = val

        return _ret

    def _play(self):
        while self.run_rate != 0:
            self.i += 1
            yield self.i
        pass
