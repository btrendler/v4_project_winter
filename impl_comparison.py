from itertools import zip_longest
import numpy as np
from matplotlib import pyplot as plt
import complex_road as cr
import ext_compl_rd as ecr


def p_ss(elems: list, labels: list[str] = None, sep="\t\t"):
    elems = [e.__repr__().split("\n") for e in elems]
    lines = list(zip_longest(*elems, fillvalue=""))
    max_elems = [max(len(v[i]) for v in lines) for i in range(len(elems))]

    # Print labels
    for i, l in enumerate(labels):
        if i:
            print(sep, end="")
        print(l, end="")
        print(" " * (max_elems[i] - len(l)), end="")
    print()

    # Print lines
    for line in lines:
        for i, l in enumerate(line):
            if i:
                print(sep, end="")
            print(l, end="")
            print(" " * (max_elems[i] - len(l)), end="")
        print()


def const_compl():
    m = cr.ComplexRoad(
        "nml",
        np.array([0.5, 0.3]),
        np.array([3.0]),
        np.array([1.]),
        1.
    )

    A, B = m._get_evolution(np.array([1]))
    Q, R = m._get_costs()

    print("---- ORIGINAL COMPLROAD IMPL ----")
    p_ss([A, B], ["       A:", "       B:"])
    # p_ss([Q, R], ["       Q:", "       R:"])
    print()
    print(["⍺", "β", "Κ", "n", "q", "m", "l"])
    print()
    # Copied from complex_road.py
    ds = m.gammas[m.structure[:-1] == "m"]
    m0 = np.ones(3)[m.structure == "m"]
    kappas = (ds * m0) - (ds * m0 * m0 / m.cs) - (ds - 2 * ds * m0 / m.cs)
    print(1., [1.], kappas)
    print()
    print()


def const_extcomp():
    # Create function, carrying capacity 2.0, delta 0.3
    def merge_func(c, n):
        delta = 0.3
        cap = 3.
        term = delta * c * (1 - c / cap)
        der = delta - (2 * delta * c) / cap
        return term, der, 0.

    net = ecr.ExtComplRoad()
    net.add(ecr.BeginSegment(3., ecr.const(1.), ecr.p1_const(0.5, dim=3)))
    net.add(ecr.MergeSegment(4., 4., 1., ecr.ts(merge_func), ecr.const(1., dim=3)))
    net.add(ecr.EndSegment(0.))

    A, B = net._get_evolution(np.array([1., 1., 1., 1., 1.]))
    Q, R = net._get_costs()

    print("---- ADVANCED COMPLROAD IMPL ----")
    p_ss([A, B], ["       A:", "       B:"])
    # p_ss([Q, R], ["       Q:", "       R:"])
    print()
    print(["_", "n", "m", "q", "l"])
    print()
    print()


if __name__ == "__main__":
    const_compl()
    const_extcomp()
