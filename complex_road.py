import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are
from typing import Callable

"""
Written by EJ Mercer in collaboration with Nathan Schill, Dallin Seyfried, and Brigg Trendler

MIT License. I would love to hear about anything you use this class for.
"""


class ComplexRoad:
    u"""

    Allows the flexible modeling of the optimal control of a simplified road network.
    An example road with structure "nmrl" is of the form:

    ╔───────╗    ╔───────╗    ╔──────╗    ╔─────────╗
    ║ iNput ║ -> ║ Merge ║ -> ║ Road ║ -> ║ Leaving ║
    ╚───────╝    ╚───┬───╝    ╚──────╝    ╚─────────╝
                    ╱
               ╔───┴───╗
               ║ Queue ║
               ╚───────╝
                (Any merge block is automatically given a queue.)

    The blocks have the following parameters:
    Input:
        alpha: the rate at which new cars enter this road segment
        gamma: the rate at which cars pass from this segment into the next segment
    Merge:
        gamma: the rate at which cars pass from this segment into the next segment;
            is constrained by the merge segment's carrying capacity
        c: the merge segment's carrying capacity
      (Queue):
        beta: Specifies the rate at which new cars enter the queue
    Road:
        gamma: the rate at which cars pass from this segment into the next segment

    Structure strings must start with n and end in l, and have any number of m and r's between. For every
    road segment, there must be a gamma value specified for the transition between them - e.g. for a nml-
    structure road, you would need to specify 2 gamma values - the transition from n to m and the transition
    from m to l.

    This class has the following methods:
        - reconstruct_costs: for changing the cost penalties associated with different states
        - single_step: to solve the infinite-horizon version of the optimization problem posed
        - multi_step: to iteratively solve the problem, producing a better solution than one-time linearization

    Future versions may add support for exit ramps and dynamic parameters.

    """

    def __init__(self, structure: str, gammas: np.ndarray, cs: np.array, betas: np.ndarray, alpha: float):
        u"""
        Initialize a complex road comprised of several segments, including multiple entrance ramps and
        road segments.


        :param structure: A string defining the road structure
        :param gammas: The gamma values corresponding to each segment transition
        :param cs: The carrying capacity of every merge segment
        :param betas: The betas for the merge segments (/queues)
        :param alpha: The rate at which cars enter the network
        """
        # Validate inputs
        for i in structure:
            if i not in {'n', 'm', 'r', 'l'}:
                raise ValueError(f"Unknown segment type {i}")
        if structure.find("n") != 0 or structure.find("n", 1) != -1:
            raise ValueError("Cars must be added at the start")
        if structure.find("l") != len(structure) - 1:
            raise ValueError("Cars must leave at the end")
        if len(gammas) != len(structure) - 1:
            raise ValueError(f"Must have {len(structure) - 1} gamma values for {len(structure)} road segments")
        self._m_count = sum(np.array(list(structure)) == "m")
        if self._m_count != len(cs):
            raise ValueError(f"Must have a carrying capacity for each m segment")
        if self._m_count != len(betas):
            raise ValueError(f"Must have a beta for each m segment")

        # Store incoming parameters
        self.structure = np.array(list(structure))
        self.betas = betas
        self.gammas = gammas
        self.cs = cs
        self.alpha = alpha

        # Determine the structure layout
        i = 0
        self._i_alpha = list(range(i, i := i + 1))
        self._i_betas = list(range(i, i := i + len(betas)))
        self._i_kappas = list(range(i, i := i + len(betas)))
        self._i_segments = []
        self._i_queues = []

        # Determine assignments for each road portion
        i -= 1
        for seg in structure:
            if seg == "m":
                self._i_queues.append(i := i + 1)
            self._i_segments.append(i := i + 1)

        # Determine the total number of entries
        self._n_entries = i + 1
        self._n_queues = len(self._i_queues)
        self._n_segments = len(self._i_segments)

        # Compute the default costs
        self.state_costs = None
        self.control_costs = None
        self.reconstruct_costs()

    def _get_evolution(self, merge_init: np.ndarray):
        """
        Get the evolution matrices for the equation x' = Ax + Bu
        :param merge_init: The values of the merge ramps at which to linearize
        :return: A and B
        """

        # Create a matrix of zeros so everything is constant by default
        A = np.zeros((self._n_entries, self._n_entries))
        B = np.zeros((self._n_entries, self._m_count))

        # Loop through each segment and configure it appropriately
        merge_index = 0
        for seg_i, seg in enumerate(self.structure[:-1]):
            # Get this road segment's index and the next index, and the transition rate
            seg_mat_i = self._i_segments[seg_i]
            gamma = self.gammas[seg_i]
            nxt_seg_mat_i = self._i_segments[seg_i + 1]

            # Handle input segments
            if seg == "n":
                A[seg_mat_i, self._i_alpha] = 1
                # Configure term in current segment row
                A[seg_mat_i, seg_mat_i] = -gamma
                # Configure term in next segment row
                A[nxt_seg_mat_i, seg_mat_i] = gamma

            # Handle road segments
            elif seg == "r":
                # Configure term in current segment row
                A[seg_mat_i, seg_mat_i] = -gamma
                # Configure term in next segment row
                A[nxt_seg_mat_i, seg_mat_i] = gamma

            # Handle merge segments
            else:
                # Compute the linearization term
                lt = gamma + (2 * gamma * merge_init[merge_index]) / self.cs[merge_index]

                # Configure terms in the queue
                A[self._i_queues[merge_index], self._i_betas[merge_index]] = 1
                B[self._i_queues[merge_index], merge_index] = -1

                # Configure terms in merge row
                A[seg_mat_i, self._i_kappas[merge_index]] = -1
                A[seg_mat_i, seg_mat_i] = -lt
                B[seg_mat_i, merge_index] = 1

                # Configure terms in next segment
                A[nxt_seg_mat_i, self._i_kappas[merge_index]] = 1
                A[nxt_seg_mat_i, seg_mat_i] = lt

                # Increment merge index
                merge_index += 1

        # Return the computed matrices
        return A, B

    def _get_costs(self):
        """Return the Q and R matrices of the LQR problem using the currently-defined costs"""
        return np.diag(self.state_costs), np.diag(self.control_costs)

    def reconstruct_costs(self, q_penalty: int | float | np.ndarray = 4., n_penalty: float = 3., m_penalty: float = 4.,
                          r_penalty: float = 2., l_penalty: float = 0., seg_penalty: np.ndarray = None,
                          u_penalty: float | np.ndarray = 1.):
        """
        Reconfigure the cost matrices for this optimization problem

        :param q_penalty: The penalty to apply to cars in the queue - either a number applied to all queues, or an array
        :param n_penalty: The penalty to apply to cars in the n segments
        :param m_penalty: The penalty to apply to cars in the m segments
        :param r_penalty: The penalty to apply to cars in the r segments
        :param l_penalty: The penalty to apply to cars in the l segments; should typically be zero to reward this
        :param seg_penalty: The penalty to apply to each segment. If specified, overrides n-l parameters.
        :param u_penalty: The penalty to apply to the control.
        """
        # Compute the cost terms
        costs = [0.] * self._n_entries
        for i in range(self._n_entries):
            # If this is a queue, use the cost for all queues, or its individual cost if specified
            if i in self._i_queues:
                if isinstance(q_penalty, (int, float)):
                    costs[i] = q_penalty
                else:
                    costs[i] = q_penalty[self._i_queues.index(i)]

            # If this is a segment, use the cost for the segment based on its type, or its individual cost if specified
            elif i in self._i_segments:
                if seg_penalty is None:
                    seg_type = self.structure[self._i_segments.index(i)]
                    if seg_type == "m":
                        costs[i] = m_penalty
                    elif seg_type == "n":
                        costs[i] = n_penalty
                    elif seg_type == "r":
                        costs[i] = r_penalty
                    elif seg_type == "l":
                        costs[i] = l_penalty
                else:
                    costs[i] = seg_penalty[self._i_segments.index(i)]

        # Store the computed costs
        self.state_costs = costs
        self.control_costs = u_penalty if not isinstance(u_penalty, (int, float)) else np.ones(
            self._m_count) * u_penalty

    def single_step(self, init_roads: np.ndarray, init_queues: np.ndarray, time_span: np.ndarray,
                    r_inv: np.ndarray = None):
        """
        Perform a one-time evaluation of the infinite-horizon LQR problem defined by this system

        :param init_roads: The initial values for each road segment
        :param init_queues: The initial values for each queue
        :param time_span: The time to evaluate over
        :param r_inv: The inverse of the R matrix from the costs, or None to compute automatically
        :return: Two functions accepting time, the first returns the states, the second returns the control
        """
        # Compute each of the kappas
        ds = self.gammas[self.structure == "m"]
        m0 = init_roads[self.structure == "m"]
        kappas = (ds * m0) + (ds * m0 * m0 / self.cs) - ((ds + 2 * ds * m0 / self.cs) * m0)

        # Construct the initial state vector
        init_state = np.zeros(self._n_entries)
        init_state[self._i_alpha] = self.alpha
        init_state[self._i_betas] = self.betas
        init_state[self._i_kappas] = kappas
        init_state[self._i_queues] = init_queues
        init_state[self._i_segments] = init_roads

        # Get A, B, Q, and R
        A, B = self._get_evolution(init_roads[self.structure == "m"])
        Q, R = self._get_costs()
        r_inv = np.linalg.inv(R) if r_inv is None else r_inv

        # Use the algebraic Ricatti equation to find P
        P = solve_discrete_are(A, B, Q, R)

        # Set up the evolution equation with the optimal control
        def _system(t, y):
            return (A - B @ r_inv @ B.T @ P) @ y

        # Solve the optimal state evolution using the DOP853 solver
        sol = solve_ivp(_system, time_span, init_state, dense_output=True, method="DOP853")

        # Construct return function
        def _get_sol(t):
            res = sol.sol(t)
            return res[self._i_segments], res[self._i_queues]

        # Return the found solution
        return _get_sol, lambda t: (-r_inv @ B.T @ P @ sol.sol(t).reshape(-1, 1))

    def multi_step(self, init_roads: np.ndarray, init_queues: np.ndarray,
                   time_span: tuple[float, float] | tuple[int, int], update_func: Callable = None,
                   num_intervals: int = 10):
        """
        Perform a repeating evaluation of the infinite-horizon LQR problem defined by this system

        The update function should accept 4 parameters:
        - This ComplexRoad
        - The time its being called at
        - The road state
        - The queue state

        :param init_roads: The initial values for each road segment
        :param init_queues: The initial values for each queue
        :param time_span: The time interval on which to evaluate, of the form (t0, tf)
        :param update_func: A function called after every change; can be used to change model parameters over time
        :param num_intervals: The number of intervals to divide the full time span into
        :return: The road states, queue states, and optimal control, in that order
        """

        # Solver parameters
        _, R = self._get_costs()
        R_inv = np.linalg.inv(R)
        n_count = 1001
        time_intervals = np.linspace(*time_span, num_intervals + 1)

        # Create variables to store the found states
        total_entries = n_count * num_intervals
        roads = np.zeros((self._n_segments, total_entries))
        queues = np.zeros((self._n_queues, total_entries))
        control = np.zeros((self._n_queues, total_entries))

        # Call the update function
        update_func(self, time_intervals[0], init_roads, init_queues)

        # Loop through each interval and find the values
        for i in range(len(time_intervals) - 1):
            # Get the interval parameters
            interval = time_intervals[i, i + 1]
            t_space = np.linspace(*interval, n_count)
            i1 = n_count + (i0 := i * n_count)

            # Solve on the interval, storing the states & controls
            sol_poly, ctrl_poly = self.single_step(init_roads, init_queues, interval, r_inv=R_inv)
            roads[:, i0:i1], queues[:, i0:i1] = sol_poly(t_space)
            control[:, i0:i1] = ctrl_poly(t_space)

            # Update the initial conditions for the next interval
            init_roads, init_queues = roads[:, i1], queues[:, i1]

            # Call the update function
            update_func(self, t_space[-1], init_roads, init_queues)

        # Return the computed solutions
        return roads, queues, control
