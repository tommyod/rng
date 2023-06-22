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
import matplotlib.pyplot as plt

# %% [markdown]
# # Mutual fund saving

# %%
years = 18
initial = 0
yearly = 12
interest = 1.05

def simulate(years, initial, yearly, interest):
    saved = initial
    for year in range(years):
        saved = saved * interest
        saved = saved + yearly
        yield saved

list(simulate(years, initial, yearly, interest))

# %%
import random

years = 18
initial = 0
yearly = 12
interest = (1.05, 0.1)

def simulate(years, initial, yearly, interest):
    saved = initial
    for year in range(years):
        saved = saved * random.gauss(*interest)
        saved = saved + yearly
        yield saved

list(simulate(years, initial, yearly, interest))

# %%
for simulation in range(999):
    
    plt.plot(list(simulate(years, initial, yearly, interest)), color="black", alpha=0.01)
    
plt.show()

# %%

# %%

# %%
