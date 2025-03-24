import numpy as np
from lpc.swab import swab
import matplotlib.pyplot as plt
import time

N = 100000
xs = np.linspace(0, N-1, N).astype(int)
lxs = np.sin(np.linspace(0, 20, N))
ys = np.empty(xs.shape[0])

random_lines = 20
split = N//random_lines

np.random.seed(0)

for i in range(random_lines):
    slope = np.random.rand() - 0.5
    intercept = np.random.rand() - 0.5
    ys[i*split:(i+1)*split] = slope*lxs[i*split:(i+1)*split] + intercept + np.random.rand(split)*0.5

# # results = bottom_up_aux(xs, ys, 0, N,1.0)
# # print("aux bottom up", results)
# # bottom_up_results = np.concatenate(results)
# # decompressed_data = np.interp(xs, xs[bottom_up_results], ys[bottom_up_results])
#
# plt.plot(xs, ys)
# plt.plot(xs, decompressed_data)
# plt.show()

# results = bottom_up(xs, ys, 1.0)
# print("full bottom up", np.where(results)[0])

# start = time.time()
# py_swab_samples = bup(xs, ys, 1.0)
# py_swab_samples = np.concatenate(py_swab_samples)
# print("Python took", time.time() - start, "seconds")
# print("CR", N/(len(py_swab_samples)))
#
# decompressed_data = np.interp(xs, xs[py_swab_samples], ys[py_swab_samples])
# plt.plot(xs, ys)
# plt.plot(xs, decompressed_data)
# plt.show()
#
start = time.time()
swab_results = swab(xs, ys, 1.0)
swab_results = np.where(swab_results)[0]
print(swab_results)
print("Python took", time.time() - start, "seconds")
print("CR", N/(len(swab_results)))

decompressed_data = np.interp(xs, xs[swab_results], ys[swab_results])

plt.plot(xs, ys)
plt.plot(xs, decompressed_data)
plt.show()


