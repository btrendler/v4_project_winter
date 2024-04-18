import complex_road as cr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ext_compl_rd as ecr

if __name__ == '__main__':
    net = ecr.ExtComplRoad()
    net.add(ecr.BeginSegment(0., lambda v: (3, 0), lambda v, v2: (1.5, 0, 0)))
    net.add(ecr.MergeSegment(0., 0.3, 1., lambda c, n: (c * (1 - (c/5)) * 0.05, (1 - (2 * c/5)) * .05, 0), lambda c, q: (2, 0, 0)))
    net.add(ecr.EndSegment(1.0))




    # structure = 'nmrl'
    # gammas = np.array([1.,1.,1.])  # Rates at which cars leave each section
    # cs = np.array([5.])            # Carrying capacity of merge section
    # betas = np.array([1.])         # Rate at which cars enter queue(s)
    # alpha = 1.                     # Rate at which cars enter iNput
    #
    # # Define a complex road object
    # road = cr.ComplexRoad(structure, gammas, cs, betas, alpha)
    #
    # # Initial road values
    ni, mi, li = 15., 15., 15.
    init_roads = np.array([ni, mi, li])
    #
    # # Initial queue values
    q1 = 0.
    init_queues = np.array([q1])

    # Time interval
    t0, tf = 0, 25
    time_span = (t0, tf)

    # Update function
    update_func = None

    # Number of time intervals
    num_intervals = 1000

    roads, queues, control = net.multi_step(init_roads, init_queues, time_span, update_func, num_intervals)

    domain = np.linspace(t0, tf, 1001000)

    fig = plt.figure()
    for i in range(3):
        plt.plot(domain, roads[i], label=["Input", "Merge", "Leaving"][i])
    plt.plot(domain, queues[0], label="Queue")
    plt.plot(domain, control[0], label="Control")
    plt.legend()
    # plt.xlim(0,3)
    # plt.ylim(-5, 15)
    plt.show()

    print(control.shape)
