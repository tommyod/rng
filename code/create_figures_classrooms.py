# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import random
import collections.abc
import math
import operator
import os
from scipy import linalg

# %%
SAVE_DIR = os.path.join("..", "presentation", "figures")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

FIGSIZE = (6, 3)
COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# %% [markdown]
# ## Simulated annealing

# %%
np.random.seed(123)


# %%
def random_permutation(n):
    """Create a random permutation matrix."""
    I = np.eye(n)
    seq = np.arange(n)

    rand_seq = np.argsort(np.random.randn(n))
    I[seq, :] = I[rand_seq, :]

    rand_seq = np.argsort(np.random.randn(n))
    I[:, seq] = I[:, rand_seq]

    assert np.all(np.sum(I, axis=0) == np.sum(I, axis=0))
    assert np.all(np.sum(I, axis=0) == np.ones(n))
    return I


# %% [markdown]
# ## Utility function
#
# - For each pupil, see how happy they are
# - Maximize the sum of (1) how happy the least happy pupil is, plus (2) average happyiness

# %%
import numba


def utility(x, preferences):
    """88.7 µs ± 3.43 µs per loop"""
    # U_i is a vector of utilities per pupil
    # If each pupil has 3 wishes, then the
    # utility is between 0 and 3. Simply a count
    # of the number of wishes satisfied
    U_i = np.diag(x @ (x.T @ preferences.T))

    # Utility is the sum of how happy the last happy pupil is,
    # and how happy they are on average, scaled between 0 and 1
    return (np.min(U_i) + np.mean(U_i)) / 6


def utility_fast(x, preferences):
    """76.8 µs ± 4.33 µs per loop"""
    U_i = ((x @ x.T) * preferences).sum(axis=1)
    return (np.min(U_i) + np.mean(U_i)) / 6


@numba.jit
def utility_fast2(x, preferences):
    """8.86 µs ± 193 ns per loop"""

    U_i = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j]:
                students_in_class_of_i = x[:, j]
                break

        the_sum = 0
        for k in range(x.shape[0]):
            the_sum += preferences[i, k] * students_in_class_of_i[k]
        U_i[i] = the_sum

    return (np.min(U_i) + np.mean(U_i)) / 6


# %%
num_students = 100
num_classes = 5
class_size = num_students // num_classes


def generate_class_prefs(class_size, num_prefs):
    """Preferences within classroom, without preference for self."""
    base_perms = []
    for i in range(class_size):
        while True:
            base = [1] * num_prefs + [0] * (class_size - num_prefs)
            base_perm = np.random.permutation(base)

            if base_perm[i] == 0:
                base_perms.append(base_perm)
                break

    return np.array(base_perms)


# Create block diagonal
preferences = linalg.block_diag(
    *[generate_class_prefs(class_size, 3) for _ in range(num_classes)]
)

assert preferences.shape == (num_students, num_students)

# %%
# Create optimal assignment
x = np.zeros(shape=(num_students, num_classes), dtype=np.bool_)
for i in range(num_students):
    for j in range(num_classes):
        if np.sum(x[:, j]) < class_size:
            x[i, j] = 1
            break

x_optimal = x.copy()
optimal_value = utility(x_optimal, preferences)


# Permute the rows for a random start
x = np.array(random_permutation(num_students) @ x, dtype=np.bool_)
x_initial = x.copy()

assert np.all(np.sum(x, axis=0) == np.ones(num_classes) * class_size)

# %%
print("How many possible solutions are there?")
from scipy.special import comb, factorial

sum(
    [comb(num_students - class_size * i, class_size, exact=True) for i in range(5)]
) / factorial(5)

# %%
assert np.allclose(
    utility(x_initial, preferences), utility_fast(x_initial, preferences)
)

# %%
assert np.allclose(
    utility(x_initial, preferences), utility_fast2(x_initial, preferences)
)


# %%
def temperature(iteration):
    return 0.998**iteration


# %%
# %#%time

iterations = list(range(10**5))
objective_values = []
temps = []

all_time_iters = []
all_time_objectives = []
all_time_x = []

best_objective = utility(x, preferences)

for iteration in iterations:
    # Get two random classrooms
    class1, class2 = np.random.choice(
        a=np.arange(num_classes), size=2, replace=False, p=None
    )

    # Get two students
    students_class1 = np.arange(num_students)[x[:, class1]]
    (student1,) = np.random.choice(students_class1, size=1, replace=False, p=None)
    students_class2 = np.arange(num_students)[x[:, class2]]
    (student2,) = np.random.choice(students_class2, size=1, replace=False, p=None)

    # Swap two students
    x[student1, class1], x[student1, class2] = x[student1, class2], x[student1, class1]
    x[student2, class1], x[student2, class2] = x[student2, class2], x[student2, class1]

    new_objective = utility_fast2(x, preferences)

    temp = temperature(iteration)
    temps.append(temp)

    if new_objective > best_objective:
        all_time_iters.append(iteration)
        all_time_objectives.append(new_objective)
        all_time_x.append(x.copy())

    if new_objective > best_objective or (np.random.rand() < temp):
        best_objective = new_objective

    else:
        # Swap two students
        x[student1, class1], x[student1, class2] = (
            x[student1, class2],
            x[student1, class1],
        )
        x[student2, class1], x[student2, class2] = (
            x[student2, class2],
            x[student2, class1],
        )

    objective_values.append(best_objective)

# %%
plt.figure(figsize=FIGSIZE)
plt.title(
    f"Simulated annealing on {num_classes} classrooms and {num_students} students"
)

plt.semilogx(iterations, objective_values, zorder=20, lw=2, label="Objective value")
plt.semilogx(iterations, temps, zorder=10, lw=2, label="Temperature")
plt.axhline(y=optimal_value, ls="--", label="Optimal value", color=COLORS[2], zorder=8)

plt.grid(True, ls="--", zorder=5, alpha=0.8)
# plt.ylabel("Objective function value")
plt.xlabel("Iterations")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "classroom_sim_annealing.pdf"))
plt.show()

# %%
plt.figure(figsize=(5, 4.5))
plt.title(
    f"Preference matrix $P$ ({num_students} students, {3} preferences)", fontsize=12
)
plt.imshow(preferences, cmap="BuGn")
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "classroom_matrix_structures.pdf"))

plt.show()

# %%
plt.figure(figsize=(8, 4.0))
plt.title(f"Circulant $P$ ({40} students, {3} preferences)", fontsize=12)
plt.imshow(linalg.circulant([0] + [1] * 3 + [0] * 36).T, cmap="BuGn")
plt.xticks([])
plt.yticks([])

plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "matrix_structure_circulant.png"), dpi=300)
# plt.savefig(os.path.join(SAVE_DIR, "classroom_matrix_structures.pdf"))
plt.show()

# %%
iters_show = 20

plt.figure(figsize=(10, 4.5))
plt.suptitle(
    f"Convergence of student-classroom assignments under simulated annealing ({num_students} students)",
    y=0.99,
    fontsize=14,
)

iters_inds = np.array(
    np.linspace(0, 1, num=iters_show) ** (1 / 10) * (len(all_time_iters) - 1), dtype=int
)[1::]

print(iters_inds)
for i in range(len(iters_inds)):
    plt.subplot(1, len(iters_inds), 1 + i)
    # plt.title("$X^{" + str(all_time_iters[iters_inds[i]] - 1) + "}$")
    plt.imshow(all_time_x[iters_inds[i]], cmap="BuGn")
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        plt.ylabel("Students")

plt.tight_layout()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, "classroom_convergence.pdf"))
plt.show()

# %%
