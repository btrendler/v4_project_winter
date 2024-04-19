from matplotlib import pyplot as plt
import numpy as np
import ext_compl_rd as ecr
import time_player as tp
from scipy.signal import savgol_filter


# See interactive.ipynb for instructions on how to use this file

def configure_model():
    # Create the model segments
    alpha, gamma, delta, beta, rho, cap = 2, 2, 1, 4, 1, 10.0
    begin_segment = ecr.BeginSegment(1.,
                                     ecr.ts(lambda c: (alpha - (alpha * c) / cap, -alpha / cap)),
                                     ecr.ts(lambda c, n: (gamma * c * (1 - n / cap) * (1 - c / cap),
                                                          gamma - 2 * gamma * c / cap - gamma * n / cap + 2 * gamma * c * n / (
                                                                  cap ** 2),
                                                          -gamma * c / cap + gamma * c ** 2 / (cap ** 2))))
    merge_segment = ecr.MergeSegment(2., 5., 1.,
                                     ecr.ts(lambda c, n: (delta * c * (1 - c / cap), delta - 2 * delta * c / cap, 0)),
                                     ecr.ts(lambda c, q: (beta * (1 - c / cap), -beta * c / cap, 0)),
                                     ecr.ts(lambda c, q: (0, 0, 0)))
    end_segment = ecr.EndSegment(2.0)

    # Create the net
    net = ecr.ExtComplRoad()
    net.add(begin_segment)
    net.add(merge_segment)
    net.add(end_segment)

    # Number of intervals to solve on
    num_intervals = 1
    return net, begin_segment, merge_segment, end_segment, num_intervals


l = np.array([])


def controlled():
    global l
    # Configure the model
    net, begin_segment, merge_segment, end_segment, num_intervals = configure_model()

    # Set up the plot, with a subplot for the road states, control, and the queue
    fig, ax = plt.subplots(1, 2, figsize=(7, 9), num="Traffic Model (Controlled)")
    empty = np.array([])
    n, = ax[0].plot(empty, empty, label="I$\\bf{{n}}$put")
    m, = ax[0].plot(empty, empty, label="$\\bf{{M}}$erge")
    q, = ax[0].plot(empty, empty, label="$\\bf{{Q}}$ueue")
    ax[0].legend(loc="upper right")
    ax[0].set_ylim((0, 20))
    u, = ax[1].plot(empty, empty, label="Control")
    leg = ax[1].legend(loc="upper right", alignment="left", title_fontproperties={"family": "monospace"})
    ax[1].set_ylim((0, 20))
    ax = list(ax)
    ls = [None, (leg, 3, "Exit")]
    l = np.array([])

    # Tunable parameters
    params = [
        (("n cost", 0., 5.), 1.),
        (("m cost", 0., 5.), 2.),
        (("q cost", 0., 5.), 5.),
        (("exit reward", 0., 5.), 2.),
        (("control cost", 0., 5.), 1.),
    ]
    state = [
        (("n", 0, 20), 2.),
        (("m", 0, 20), 1.),
        (("q", 0, 20), 1.),
        (("l", 0, 20), 0.),
        (("u", 0, 20), 0.)
    ]

    # Update function
    def update(params, interval, start_state):
        global l
        # Update costs
        c_n, c_m, c_q, c_l, c_u = params
        begin_segment.set_seg_cost(c_n)
        merge_segment.set_seg_cost(c_m)
        merge_segment.set_ramp_cost(c_q)
        merge_segment.set_control_cost(c_u)
        end_segment.set_seg_reward(c_l)

        # Set up initial state
        n0, m0, q0, l0, u0 = start_state
        init_roads = np.array([n0, m0, l0])
        init_queues = np.array([q0])

        # Solve the system
        roads, queues, control = net.multi_step(init_roads, init_queues, interval, num_intervals=num_intervals)
        x_new = np.linspace(interval[1], interval[0], len(control[0]), endpoint=False)[::-1]

        # Get the x and y data currently in the lines
        x_v = n.get_xdata()
        mask = x_v < interval[0]
        n_new = savgol_filter(np.concatenate((n.get_ydata()[mask], roads[0])), 300, 2)
        m_new = savgol_filter(np.concatenate((m.get_ydata()[mask], roads[1])), 300, 2)
        q_new = savgol_filter(np.concatenate((q.get_ydata()[mask], queues[0])), 300, 2)
        u_new = savgol_filter(np.concatenate((u.get_ydata()[mask], control[0])), 300, 2)
        l = np.concatenate((l[mask], roads[2]))
        x_v = x_v[x_v < interval[0]]
        x_full = np.concatenate((x_v, x_new))

        # Update the lines
        n.set_xdata(x_full)
        n.set_ydata(n_new)
        m.set_xdata(x_full)
        m.set_ydata(m_new)
        q.set_xdata(x_full)
        q.set_ydata(q_new)
        u.set_xdata(x_full)
        u.set_ydata(u_new)

        # Return the stacked data
        return x_full, np.vstack((n_new, m_new, q_new, l, u_new))

    _ = tp.TimePlayer(fig, list(zip(ax, ls)), params, state, update, t_per_sec=0.002, t_span=10, ms_per_frame=50,
                      calc_frac=0.25, slider_height=0.45)
    plt.show()
    return _


if __name__ == "__main__":
    # uncontrolled()
    controlled()
