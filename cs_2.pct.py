# -*- coding: utf-8 -*-
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
# # DRAF demo Case Study 2: : Design Optimization of a Multi-Use BES and PV System

# %% init_cell=true
import pandas as pd

import draf
from draf.components import *

# %% [markdown]
# ## Modeling

# %%
coords = (49.01, 8.39)  # Random place in Karlsruhe
p_el = pd.read_csv("P_el_2020_15min.csv").squeeze("columns")
cs = draf.CaseStudy(
    name="draf_demo_cs2", year=2020, freq="15min", coords=coords, consider_invest=True
)
# cs.set_time_horizon(start="Jun-1 00:00", steps=cs.steps_per_day * 30)
sc = cs.add_REF_scen(
    components=[
        eDem(p_el=cs.match_dtindex(p_el, resample=True)),
        BES(allow_new=False),
        EG(c_buyPeak=100),
        PV(P_CAPx=300, A_avail_=1000, allow_new=False),
        Main,
    ]
).update_params(
    c_EG_addon_=0.06228,  # Source: sc.prep.c_EG_addon_(EEG_surcharge=0))
    c_BES_inv_=209,  # Source: Forecast for 2022 in @Vartiainen_2019
    c_PV_inv_=384,  # Source: Forecast for 2022 in @Vartiainen_2019
)
cs.add_scen("optBes").update_params(z_BES_=1)
cs.add_scen("optPV").update_params(z_PV_=1)
cs.add_scen("optBesPv").update_params(z_BES_=1, z_PV_=1)

# %%
# Uncomment to run optimization:
# cs.optimize(parallel=True).save()

# %%
cs = draf.open_latest_casestudy("draf_demo_cs2")

# %% [markdown]
# ## Results

# %%
cs.plot.tables()

# %%
cs.plot.sankey_interact()

# %%
cs.plot.heatmap_interact("v")

# %%
import plotly.io as pio
pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469#issuecomment-994907721

fig = cs.scens.optBesPv.plot.ts_balance(
    data={
        "pos": ["P_PV_OC_T", "P_BES_out_T", "P_EG_buy_T"],
        "neg": ["P_eDem_T", "P_BES_in_T", "P_PV_FI_T"],
    },
    data_ylabel="Electric Power<br>(MW)",
    data_conversion_factor=1e-3,
    addon_ts="c_EG_RTP_T",
    addon_conversion_factor=1e3,
    addon_ts_ylabel="Price<br>(€/MWh)",
    colors=dict(  # https://coolors.co/8ecae6-219ebc-023047-ffb703-fb8500
        PV="#ffb703",
        EG="#023047",
        BES="#dd4477",
        eDem="#8ecae6",
    ),
    ts_slicer=slice("2020-06-15", "2020-06-21"),
)
fig.update_layout(legend=dict(tracegroupgap=30, y=0.95))
fig

# %%
cs.scens.REF.plot.violin_T(show_zero=True)

# %%
cs.scens.optBesPv.plot.violin_T(show_zero=True)

# %%
cs.plot.pareto()
