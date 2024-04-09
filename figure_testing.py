from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 512
plt.rcParams["figure.figsize"] = [3.75, 3]

t_space = np.linspace(0, 10)
print(plt.rcParams["figure.figsize"])
# plt.figure(figsize=(3, 3))
plt.plot(t_space, np.ones(shape=t_space.shape), label=f"m")
plt.plot(t_space, 0.5 * t_space, label=f"l")
plt.plot(t_space, 0.3 * 0.3 * t_space ** 2, label=f"n")
plt.plot(t_space, 0.3 * 0.3 * 0.3 * 0.3 * t_space ** 3, label=f"q")
plt.plot(t_space, 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * t_space ** 4, label=f"u")
plt.ylim(0, 16)
plt.legend()
plt.title(f"Initial Guess: [1 1 1 1 1]\ndelta: 0., gamma: 0., m0: 1, c: 2, q6: 8")
plt.tight_layout()
plt.savefig(f"test.pdf")
plt.show()