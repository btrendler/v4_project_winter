import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are
from typing import Callable


def ts(func: Callable[[float], tuple[float, float]] | Callable[[float, float], tuple[float, float, float]]):
    """
    Computes the taylor series of a function & its derivatives

    The function must return first its value, then the derivative w.r.t. the first parameter, then the derivative
    w.r.t. the second parameter, if applicable.

    :param func: The function of one or two variables of which to compute the taylor series
    :return: A function which returns the constant portion, the first parameter term, and the second parameter term,
            if applicable
    """

    try:
        func(0.)

        def _ret(point: float):
            # Compute the constant and first-order terms and return them
            f, fp1 = func(point)
            return (f - point * fp1), fp1

        return _ret
    except TypeError:
        def _ret(p1: float, p2: float):
            # Compute the constant and first-order terms and return them
            f, fp1, fp2 = func(p1, p2)
            return (f - (p1 * fp1) - (p2 * fp2)), fp1, fp2

        return _ret


def const(val: float, dim=2):
    if dim == 3:
        def _ret(p1, p2=0):
            return val, 0, 0

        return _ret
    elif dim == 2:
        def _ret(p1, p2=0):
            return val, 0

        return _ret
    raise ValueError("Dim must be 2 or 3")


def p1_const(val: float, dim=2):
    if dim == 3:
        def _ret(p1, p2=0):
            return 0, val, 0

        return _ret
    elif dim == 2:
        def _ret(p1, p2=0):
            return 0, val

        return _ret
    raise ValueError("Dim must be 2 or 3")


def p2_const(val: float):
    def _ret(p1, p2=0):
        return 0, 0, val

    return _ret


class _AbstractSegment:
    def __init__(self, state_costs: np.ndarray, control_costs: np.ndarray):
        self.state_costs = state_costs
        self.n_indices = len(state_costs)
        self.control_costs = control_costs
        self.n_control = len(control_costs)
        self.state_indices = None
        self.control_indices = None
        self.next = None

    def _set_indices(self, state_indices: np.ndarray):
        if len(state_indices) != self.n_indices:
            raise ValueError(f"Must have at least {self.n_indices} state indices assigned to this node.")
        self.state_indices = state_indices

    def _set_control(self, control_indices: np.ndarray):
        if len(control_indices) != self.n_control:
            raise ValueError(f"Must have at least {self.n_control} control indices assigned to this node.")
        self.control_indices = control_indices

    def _apply_state(self, A, states, n_state, ctrl=True):
        if self.state_indices is None:
            raise ValueError("Must attach to a road network.")

    def _apply_control(self, B, unctrl=False):
        if self.state_indices is None:
            raise ValueError("Must attach to a road network.")


class EndSegment(_AbstractSegment):
    """
    A road segment representing the end of the network.
    """

    def __init__(self, state_reward: float):
        """
        Define a new ending segment.

        :param state_reward: The reward associated with cars in the segment
        """
        # Call superclass
        _AbstractSegment.__init__(self, np.array([state_reward]), np.array([]))

    def _apply_state(self, A, states, n_state, ctrl=True):
        super()._apply_state(A, states, n_state, ctrl)
        # Invert this row of the state matrix, so the cost is a reward
        if ctrl:
            A[self.state_indices, :] *= -1

    def get_seg_reward(self):
        """
        Get the reward associated with this segment
        :return: The reward
        """

        return self.state_costs[0]

    def set_seg_reward(self, value: float) -> None:
        """
        Set the reward associated with this segment
        :param value: The new reward
        """
        self.state_costs[0] = value


class RoadSegment(_AbstractSegment):
    """
    A road segment representing a typical road section of the network.
    """

    def __init__(self, seg_cost: float,
                 leave_func: Callable[[float, float], tuple[float, float, float]]):
        """
        Define a new standard road segment. For information about function structure, see the documentation for the
        ExtComplRoad class.

        :param seg_cost: The cost associated with cars in the segment
        :param leave_func: The function representing how quickly cars leave this segment to the next segment
        """
        # Call superclass
        _AbstractSegment.__init__(self, np.array([seg_cost]), np.array([]))
        # Store passed-in parameters
        self._f = leave_func

    def get_seg_cost(self):
        """
        Get the cost associated with this segment
        :return: The cost
        """
        return self.state_costs[0]

    def set_seg_cost(self, value: float) -> None:
        """
        Set the cost associated with this segment
        :param value: The new cost
        """
        self.state_costs[0] = value

    def _apply_state(self, A, states, n0, ctrl=True):
        super()._apply_state(A, states, n0, ctrl)
        # Extract linearization values
        c0, = states

        # Get indices
        c, = self.state_indices
        n = self.next

        # Evaluate linearized functions
        const, c_term, n_term = self._f(c0, n0)

        # Subtract f from c's state, and add it to the next one
        A[c, 0] = -const
        A[c, c] = -c_term
        A[c, n] = -n_term
        A[n, 0] = const
        A[n, c] = c_term
        A[n, n] = n_term


class BeginSegment(_AbstractSegment):
    """
    A road segment representing the beginning of the network.
    """

    def __init__(self, seg_cost: float,
                 enter_func: Callable[[float], tuple[float, float]],
                 leave_func: Callable[[float, float], tuple[float, float, float]]):
        """
        Define a new beginning segment. For information about function structure, see the documentation for the
        ExtComplRoad class.

        :param seg_cost: The cost associated with cars in the segment
        :param enter_func: The function representing how quickly cars enter the network
        :param leave_func: The function representing how quickly cars leave this segment to the next segment
        """
        # Call superclass
        _AbstractSegment.__init__(self, np.array([seg_cost]), np.array([]))
        # Store passed-in parameters
        self._alpha = enter_func
        self._f = leave_func

    def get_seg_cost(self):
        """
        Get the cost associated with this segment
        :return: The cost
        """
        return self.state_costs[0]

    def set_seg_cost(self, value: float) -> None:
        """
        Set the cost associated with this segment
        :param value: The new cost
        """
        self.state_costs[0] = value

    def _apply_state(self, A, states, n0, ctrl=True):
        super()._apply_state(A, states, n0, ctrl)
        # Extract linearization values
        c0, = states

        # Get indices
        c, = self.state_indices
        n = self.next

        # Evaluate linearized functions
        f_const, f_c_term, f_n_term = self._f(c0, n0)
        a_const, a_c_term = self._alpha(c0)

        # Subtract f from c's state, and add it to the next one, and add alpha to c's state
        A[c, 0] = -f_const + a_const
        A[c, c] = -f_c_term + a_c_term
        A[c, n] = -f_n_term
        A[n, 0] = f_const
        A[n, c] = f_c_term
        A[n, n] = f_n_term


class MergeSegment(_AbstractSegment):
    """
    A road segment representing a merge ramp. Automatically includes a queue.
    """

    def __init__(self, seg_cost: float, ramp_cost: float, control_cost: float,
                 leave_func: Callable[[float, float], tuple[float, float, float]],
                 add_func: Callable[[float, float], tuple[float, float, float]],
                 queue_func: Callable[[float, float], tuple[float, float, float]] = None,
                 kappa: float = 1):
        """
        Define a new merge segment. For information about function structure, see the documentation for the
        ExtComplRoad class.

        :param seg_cost: The cost associated with cars in the segment
        :param ramp_cost: The cost associated with cars on the ramp
        :param control_cost: The cost associated with the control value
        :param leave_func: The function representing how quickly cars leave the merge segment to the next segment
        :param add_func: The function representing how quickly cars enter the queue
        :param queue_func: The function representing the speed at which cars enter the merge segment from the queue,
                when uncontrolled
        :param kappa: The factor on the control value (typically 1 or 0)
        """
        # Call superclass
        _AbstractSegment.__init__(self, np.array([seg_cost, ramp_cost]), np.array([control_cost]))
        # Store passed-in parameters
        self._f = leave_func
        self._beta = add_func
        self._g = queue_func
        self._kappa = kappa

    def get_seg_cost(self):
        """
        Get the cost associated with this segment
        :return: The cost
        """
        return self.state_costs[0]

    def set_seg_cost(self, value: float) -> None:
        """
        Set the cost associated with this segment
        :param value: The new cost
        """
        self.state_costs[0] = value

    def get_ramp_cost(self):
        """
        Get the cost associated with the ramp
        :return: The cost
        """
        return self.state_costs[1]

    def set_ramp_cost(self, value: float) -> None:
        """
        Set the cost associated with the ramp
        :param value: The new cost
        """
        self.state_costs[1] = value

    def get_control_cost(self):
        """
        Get the cost associated with the control
        :return: The cost
        """
        return self.control_costs[0]

    def set_control_cost(self, value: float) -> None:
        """
        Set the cost associated with the control
        :param value: The new cost
        """
        self.control_costs[0] = value

    def _apply_state(self, A, states, n_state, ctrl=True):
        super()._apply_state(A, states, n_state, ctrl)
        # Extract linearization values
        c0, q0, = states

        # Get indices
        c, q, = self.state_indices
        n = self.next

        # Evaluate linearized functions
        f_const, f_c_term, f_n_term = self._f(c0, n_state)
        b_const, b_c_term, b_q_term = self._beta(c0, q0)
        g_const, g_c_term, g_q_term = 0., 0., 0.
        if self._g is not None and not ctrl:
            g_const, g_c_term, g_q_term = self._g(c0, q0)

        # Apply to the state accordingly
        # First, c prime
        A[c, 0] = -f_const + g_const
        A[c, c] = g_c_term - f_c_term
        A[c, n] = -f_n_term
        A[c, q] = g_q_term
        # Then, n prime
        A[n, 0] = f_const
        A[n, n] = f_n_term
        A[n, c] = f_c_term
        # Then q prime
        A[q, 0] = b_const - g_const
        A[q, q] = b_q_term - g_q_term
        A[q, c] = b_c_term - g_c_term

    def _apply_control(self, B, unctrl=False):
        c, q, = self.state_indices
        u, = self.control_indices
        B[c, u] = self._kappa
        B[q, u] = -self._kappa


class ExtComplRoad:
    u"""

    Allows the flexible modeling of the optimal control of a road network.
    A road with a begin, merge, road, and then end block is of the form:

    ╔───────╗    ╔───────╗    ╔──────╗    ╔─────────╗
    ║ Input ║ -> ║ Merge ║ -> ║ Road ║ -> ║ Leaving ║
    ╚───────╝    ╚───┬───╝    ╚──────╝    ╚─────────╝
                    ╱
               ╔───┴───╗
               ║ Queue ║
               ╚───────╝
                (Any merge block is automatically given a queue.)

    This class has the following methods:
        - single_step: to solve the infinite-horizon version of the optimization problem posed
        - multi_step: to iteratively solve the problem, producing a better solution than one-time linearization

    Future versions may add support for exit ramps, loops, and multiple inputs.

    Functions must be specified such that they return the transition rate as the coefficients & constants on the
    equation ret1 + ret2 * p1 + ret3 * p2, where p1 and p2 are variables corresponding to the current state of the
    parameter values. An easy way to do this is to use either ts() or const().

    """

    def __init__(self):
        """
        Initialize a complex road comprised of several segments.
        """

        self.segments = []
        self.c_state_idx = 1
        self.c_ctrl_idx = 0
        self.n_entries = None
        self.n_control = None
        self.n_roads = None
        self.n_queues = None
        self._i_roads = np.array([])
        self._i_queues = np.array([])

    def add(self, seg: _AbstractSegment) -> None:
        """
        Adds a segment to the road network. Must be done in order, beginning with a BeginSegment and finishing with an
        EndSegment. Once an EndSegment has been added, no further segments can be appended to this network

        :param seg: The segment to append to the main road
        """
        # Validate inputs
        if len(self.segments) == 0:
            if seg is BeginSegment:
                raise ValueError("Must start with a BeginSegment.")
        if len(self.segments) != 0:
            if seg is BeginSegment:
                raise ValueError("Cannot have multiple BeginSegments.")
        if self.c_state_idx is None:
            raise ValueError("Cannot append after an EndSegment.")

        # Assign state indexes
        idx = self.c_state_idx
        indices = np.array(range(idx, (idx := idx + seg.n_indices)))
        seg._set_indices(indices)
        self._i_roads = np.concatenate((self._i_roads, indices[:1]))
        self._i_queues = np.concatenate((self._i_queues, indices[1:]))
        self.c_state_idx = idx

        # Assign control indexes
        idx = self.c_ctrl_idx
        seg._set_control(np.array(range(idx, (idx := idx + seg.n_control))))
        self.c_ctrl_idx = idx

        # Set the next element
        seg.next = self.c_state_idx

        # Add it to the list
        self.segments.append(seg)

        # Handle ending segment
        if isinstance(seg, EndSegment):
            seg.next = None
            self.n_entries = self.c_state_idx
            self.n_control = self.c_ctrl_idx
            self.c_state_idx = None
            self.c_ctrl_idx = None
            self.n_roads = len(self._i_roads)
            self.n_queues = len(self._i_queues)
            self._i_roads = self._i_roads.astype(int)
            self._i_queues = self._i_queues.astype(int)

    def uncontrolled_result(self, init_roads: np.ndarray, init_queues: np.ndarray, time_span: tuple):
        """
        Perform a simulation of what would happen on this road network with the given state with zero control

        :param init_roads: The initial values for each road segment
        :param init_queues: The initial values for each queue
        :param time_span: The time to evaluate over
        :return: A function accepting time, returning the road & queue states
        """
        # Construct the state vector
        init_state = np.ones(self.n_entries)
        init_state[self._i_roads] = init_roads
        init_state[self._i_queues] = init_queues

        # Get A, B, Q, and R
        A, _ = self._get_evolution(init_state, ctrl=False)

        # Set up the evolution equation
        def _system(_, y):
            return A @ y

        # Solve the state evolution using DOP853
        sol = solve_ivp(_system, time_span, init_state, dense_output=True, method="DOP853")

        # Construct return function
        def _get_sol(t):
            res = sol.sol(t)
            return res[self._i_roads], res[self._i_queues]

        # Return the found solution
        return _get_sol

    def single_step(self, init_roads: np.ndarray, init_queues: np.ndarray, time_span: tuple,
                    r_inv: np.ndarray = None) -> tuple[Callable, Callable]:
        """
        Perform a one-time evaluation of the infinite-horizon LQR problem defined by this system

        :param init_roads: The initial values for each road segment
        :param init_queues: The initial values for each queue
        :param time_span: The time to evaluate over
        :param r_inv: The inverse of the R matrix from the costs, or None to compute automatically
        :return: Two functions accepting time, the first returns the states, the second returns the control
        """
        # Construct the state vector
        init_state = np.ones(self.n_entries)
        init_state[self._i_roads] = init_roads
        init_state[self._i_queues] = init_queues

        # Get A, B, Q, and R
        A, B = self._get_evolution(init_state)
        Q, R = self._get_costs()
        r_inv = np.linalg.inv(R) if r_inv is None else r_inv

        # Use the algebraic Ricatti equation to find P
        P = solve_discrete_are(A, B, Q, R)

        # Set up the evolution equation with the optimal control
        def _system(_, x):
            u = -r_inv @ B.T @ P @ x
            x_p = A @ x + B @ np.maximum(u, 0)
            x_p[x <= np.finfo(float).eps * 1e8][:-1] = 0
            return x_p

        # Solve the optimal state evolution using the DOP853 solver
        sol = solve_ivp(_system, time_span, init_state, dense_output=True, method="DOP853")

        # Construct return function
        def _get_sol(t):
            res = sol.sol(t)
            return res[self._i_roads], res[self._i_queues]

        # Return the found solution
        return _get_sol, lambda t: np.maximum(-r_inv @ B.T @ P @ sol.sol(t), 0)

    def multi_step(self, init_roads: np.ndarray, init_queues: np.ndarray,
                   time_span: tuple[float, float] | tuple[int, int], update_func: Callable = None,
                   num_intervals: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        r_inv = np.linalg.inv(R)
        n_count = 1001
        time_intervals = np.linspace(*time_span, num_intervals + 1)

        # Create variables to store the found states
        total_entries = n_count * num_intervals
        roads = np.zeros((self.n_roads, total_entries))
        queues = np.zeros((self.n_queues, total_entries))
        control = np.zeros((self.n_queues, total_entries))

        # Call the update function
        if update_func:
            update_func(self, time_intervals[0], init_roads, init_queues)

        # Loop through each interval and find the values
        for i in range(len(time_intervals) - 1):
            # Get the interval parameters
            interval = tuple(time_intervals[i:i + 2])
            t_space = np.linspace(*interval, n_count)
            i1 = n_count + (i0 := i * n_count)

            # Solve on the interval, storing the states & controls
            sol_poly, ctrl_poly = self.single_step(init_roads, init_queues, interval, r_inv=r_inv)
            roads[:, i0:i1], queues[:, i0:i1] = sol_poly(t_space)
            control[:, i0:i1] = ctrl_poly(t_space)

            # Update the initial conditions for the next interval
            init_roads, init_queues = roads[:, i1 - 1], queues[:, i1 - 1]

            # Call the update function
            if update_func:
                update_func(self, t_space[-1], init_roads, init_queues)

        # Return the computed solutions
        return roads, queues, control

    def _get_evolution(self, current_state: np.ndarray, ctrl: bool = True):
        # Create a matrix of zeros so everything is constant by default
        A = np.zeros((self.n_entries, self.n_entries))
        B = np.zeros((self.n_entries, self.n_control))

        # Loop through each segment and apply it to both matrices
        for seg in self.segments:
            seg._apply_state(A, current_state[seg.state_indices],
                             current_state[seg.next] if seg.next is not None else None, ctrl)
            if ctrl:
                seg._apply_control(B)

        # Return the computed matrices
        return A, B

    def _get_costs(self):
        # Create the state cost & control costs using the definitions in the segments
        return (
            np.diag(np.concatenate([np.zeros(1), *(seg.state_costs for seg in self.segments)])),
            np.diag(np.concatenate([seg.control_costs for seg in self.segments]))
        )
