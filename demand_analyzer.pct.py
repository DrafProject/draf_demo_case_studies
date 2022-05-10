# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # screenshots for draf_demo_paper

# %%
import draf
import pandas as pd

demand = pd.read_csv("P_el_2020_15min.csv").squeeze("columns")
sc = draf.Scenario(year=2020, freq="15min")
da = sc.analyze_demand(demand)

# %%
pla = da.show_peaks(target_percentile=97, c_EG_peak=50.0)

# %%
sc.plot.heatmap(demand);

# %%
sc.plot.heatmap(demand.where(demand>pla.target_peakload));
