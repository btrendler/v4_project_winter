import complex_road as cr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    structure = 'nmrl'
    gammas = np.array([1.,1.,1.])  # Rates at which cars leave each section
    cs = np.array([5.])            # Carrying capacity of merge section
    betas = np.array([1.])         # Rate at which cars enter queue(s)
    alpha = 1.                     # Rate at which cars enter iNput

    # Define a complex road object
    road = cr.ComplexRoad(structure, gammas, cs, betas, alpha)

    # Initial road values
    ni, mi, ri, li = 15., 15., 15., 15.
    init_roads = np.array([ni, mi, ri, li])

    # Initial queue values
    q1 = 0.
    init_queues = np.array([q1])

    # Time interval
    t0, tf = 0, 25
    time_span = (t0, tf)

    # Update function
    update_func = None

    # Number of time intervals
    num_intervals = 1000

    roads, queues, control = road.multi_step(init_roads, init_queues, time_span, update_func, num_intervals)

    domain = np.linspace(t0, tf, 1001000)

    fig = plt.figure()
    for i in range(4):
        plt.plot(domain, roads[i])
    plt.legend(["iNput", "Merge", "Road", "Leaving"])
    plt.plot(domain, queues[0])
    plt.plot(domain, control[0])
    plt.legend(["iNput", "Merge", "Road", "Leaving", "Queue", "Control"])
    # plt.xlim(0,3)
    plt.ylim(-5, 15)
    plt.show()

    print(control.shape)
