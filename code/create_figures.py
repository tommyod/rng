# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
import matplotlib as mpl

# update matplotlibrc
mpl.rcParams["font.family"] = "Open Sans"

# %%
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
from scipy import stats

# %matplotlib inline

# %%
import random

random.gauss(0, 1)

# %%
SAVE_DIR = os.path.join("..", "presentation", "figures")

# %% [markdown]
# # General figures

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

# Plot PDF
distribution = stats.uniform()
x = np.linspace(-0.05, 1.05, num=2**10)
ax.plot(x, distribution.pdf(x))
ax.set_ylim([0, 1.2])

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "uniform.pdf"))

# Draw samples and plot them
samples = distribution.rvs(1000, random_state=4)

ax.scatter(
    samples,
    np.ones_like(samples) * stats.uniform().rvs(len(samples), random_state=1) * 0.2,
    color="black",
    s=1,
)

fig.savefig(os.path.join(SAVE_DIR, "uniform_samples.pdf"))

ax.hist(samples, bins="auto", density=True, zorder=0)

fig.savefig(os.path.join(SAVE_DIR, "uniform_samples_hist.pdf"))

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

# Plot PDF
distribution = stats.norm()
x = np.linspace(-4, 4, num=2**10)
ax.plot(x, distribution.pdf(x))
ax.set_ylim([0, 0.45])

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "normal.pdf"))

# Draw samples and plot them
samples = distribution.rvs(1000, random_state=6)

ax.scatter(
    samples,
    np.ones_like(samples) * stats.uniform().rvs(len(samples), random_state=1) * 0.1,
    color="black",
    s=1,
)

fig.savefig(os.path.join(SAVE_DIR, "normal_samples.pdf"))

ax.hist(samples, bins="auto", density=True, zorder=0)

fig.savefig(os.path.join(SAVE_DIR, "normal_samples_hist.pdf"))


# %% [markdown]
# # Mutual fund saving

# %%
def simulate(years, yearly, interest):
    yield (saved := 0)
    for year in range(years):
        saved = saved * interest + yearly
        yield saved


years = 18
yearly = 12
interest = 1.05

list(simulate(years, yearly, interest))

# %%
import random


def simulate_rng(years, yearly, interest):
    yield (saved := 0)
    for year in range(years):
        saved = saved * random.gauss(*interest) + yearly
        yield saved


years = 18
yearly = 12
interest = (1.05, 0.1)

# %%
fig, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))

ax.plot(list(simulate(years, yearly, interest[0])), zorder=99, lw=3)
ax.set_xticks(np.arange(0, years + 1, 2))
ax.grid(True, ls="--", alpha=0.5, zorder=0)
ax.set_xlabel("Years")
ax.set_ylabel("Money")
ax.set_ylim([0, 500])
ax.set_xlim([0, 18])

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "mutual_fund.pdf"))

for simulation in range(99):
    ax.plot(
        list(simulate_rng(years, yearly, interest)), color="black", alpha=0.1, zorder=9
    )

fig.savefig(os.path.join(SAVE_DIR, "mutual_fund_simulations.pdf"))

# %% [markdown]
# # Groceries

# %%
import pandas as pd

# %%
money_per_day = pd.Series(
    [
        0.0,
        0.0,
        471.9,
        784.22,
        0.0,
        355.29,
        0.0,
        600.76,
        399.7,
        0.0,
        266.7,
        238.6,
        433.0,
        0.0,
        1214.8,
        470.5,
        0.0,
        0.0,
        306.4,
        341.8,
        0.0,
        0.0,
        0.0,
        358.89,
        0.0,
        715.2,
        1049.02,
        0.0,
        0.0,
        536.7,
        367.7,
    ],
    index=pd.DatetimeIndex(
        [
            "2023-05-01",
            "2023-05-02",
            "2023-05-03",
            "2023-05-04",
            "2023-05-05",
            "2023-05-06",
            "2023-05-07",
            "2023-05-08",
            "2023-05-09",
            "2023-05-10",
            "2023-05-11",
            "2023-05-12",
            "2023-05-13",
            "2023-05-14",
            "2023-05-15",
            "2023-05-16",
            "2023-05-17",
            "2023-05-18",
            "2023-05-19",
            "2023-05-20",
            "2023-05-21",
            "2023-05-22",
            "2023-05-23",
            "2023-05-24",
            "2023-05-25",
            "2023-05-26",
            "2023-05-27",
            "2023-05-28",
            "2023-05-29",
            "2023-05-30",
            "2023-05-31",
        ],
        dtype="datetime64[ns]",
        freq="D",
    ),
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))

ax.plot(money_per_day, '-o', zorder=9)
#ax.set_xticks(np.arange(len(money_per_day)))
ax.grid(True, ls="--", alpha=0.5, zorder=0)
ax.set_ylabel("NOK")

fig.autofmt_xdate(rotation=90)
fig.tight_layout()

fig.savefig(os.path.join(SAVE_DIR, "groceries_data.pdf"))

# %%
money_per_day.sum()

# %%
money_per_day = money_per_day.values

# %%
money_per_day

# %%
import numpy as np

resamples = np.random.choice(
    money_per_day, size=(9999, len(money_per_day)), replace=True, p=None
)


np.mean(resamples.sum(axis=1))

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.25))


ax1.set_title("Distribution of resampled means")
ax1.hist(resamples.sum(axis=1), bins="auto", density=True)
ax1.grid(True, ls="--", alpha=0.8, zorder=0, color="black")
ax1.set_yticklabels([])

ax2.set_title("Cumulative distribution of resampled means")
ax2.hist(resamples.sum(axis=1), bins="auto", density=True, cumulative=True)
ax2.grid(True, ls="--", alpha=0.8, zorder=0, color="black")

fig.tight_layout()

fig.savefig(os.path.join(SAVE_DIR, "groceries_data_resampled.pdf"))

# %%

# %%

# %%

# %%

# %%

# %%
