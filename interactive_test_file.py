from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import numpy as np
import importlib

def f(t, amp, freq):
    return amp * np.sin(2 * np.pi * freq * t)
def f_inv(x, amp, freq):
    return np.arcsin(x / amp) / (2 * np.pi * freq)

import time_player

importlib.reload(time_player)

fig, ax = plt.subplots()
line, = ax.plot(np.array([]), np.array([]), label="Wave")
ax.legend(loc="lower right")
ax.set_ylim((-10, 10))

params = [(("Amplitude", 0., 10.), 3.), (("Frequency", 0.1, 30.), 3.)]
state = [(("n", -10., 10.), 0.)]


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