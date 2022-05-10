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
# # DRAF demo Case Study 3: Multi-Objective Design and Operational Optimization of Thermal-Electric Sector Coupling

# %% init_cell=true
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np

import draf
from draf.components import *

# %% [markdown]
# ## Modeling

# %%
coords = (49.01, 8.39)  # Random place in Karlsruhe
cs = draf.CaseStudy("draf_demo_cs3", year=2020, freq="60min", coords=coords, consider_invest=True)
# cs.set_time_horizon(start="Jun-20 00:00", steps=cs.steps_per_day * 5)

hp_config = dict(n=3, ambient_as_source=False, ambient_as_sink=True, heating_levels=[])
sc = cs.add_REF_scen(components=[cDem, eDem, hDem, EG, Fuel, H2H1, HOB, Main, HP(**hp_config)])
new_thermal_demands = dict(
    dQ_hDem_TH=sc.params.dQ_hDem_TH + 2e3, dQ_cDem_TN=sc.params.dQ_cDem_TN + 2e3
)
sc.update_params(**new_thermal_demands)


def no_bio_fuel(sc, m, d, p, v, c):
    m.addConstr(v.F_fuel_F["bio"] == 0)


cs.add_scen(
    "new",
    based_on=None,
    components=[
        cDem,
        eDem(),
        hDem,
        BES(allow_new=True),
        CHP(allow_new=True),
        EG,
        Fuel,
        H2H1,
        HOB(allow_new=True),
        HP(n=3, allow_new=True),
        P2H(allow_new=True),
        PV(A_avail_=2e4),
        TES(allow_new=True),
        Main,
    ],
    custom_model=no_bio_fuel,
).update_params(**new_thermal_demands)
cs.add_scens(nParetoPoints=6, based_on="new", remove_base=True)
cs.improve_pareto_norm_factors()

print(
    "Maximum demands for cooling = {:.0f} MW, heat = {:.0f} MW, electricity = {:.0f} MW".format(
        sc.params.dQ_cDem_TN.unstack().sum(1).max(),
        sc.params.dQ_hDem_TH.unstack().sum(1).max(),
        sc.params.P_eDem_T.max(),
    )
)

# %%
# Uncomment to run optimization:
# cs.optimize(parallel=True).save()

# %%
cs = draf.open_latest_casestudy("draf_demo_cs3")
cs.scens.sc5.doc = cs.scens.sc5.doc[:16]
cs.scens.sc5.name = cs.scens.sc5.name[:4]
cs.scens

# %% [markdown]
# ## Plot Results

# %%
cs.plot.tables()

# %%
fig = cs.plot.capas(include_capx=False, subplot_x_anchors=(0.6, 0.8))
fig.update_layout(template="plotly_white")

# %%
df = cs.get_collector_values("C_TOT_")[::-1] / 1e3
df = df.rename(index=dict(inv="ann_inv"))
fig = px.bar(
    df.T,
    orientation="h",
    labels=dict(variable="Cost type:", index="Scenario", value="TAC (M€/a, cut at 10 M€/a)"),
    color_discrete_sequence=px.colors.qualitative.Set2,
    barmode="group",
    width=400,
    height=200,
    template="plotly_white",
)
fig.update_layout(
    margin=dict(b=0, l=0, r=0, t=0),
    yaxis_categoryorder="category descending",
    xaxis_range=[0, 10],
    legend=dict(traceorder="reversed", orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
)

# %%
df = pd.DataFrame(cs.get_entity_dict("Q_TES_CAPn_L")) / 1e3
fig = px.bar(
    df.T,
    orientation="h",
    labels=dict(value="TES capacity (MWh)", index="Scenario", L="Temp. level:"),
    color_discrete_sequence=["#6C8EBF", "#accdfc", "#B85450", "#ed9f9d"],
    width=400,
    height=200,
    template="plotly_white",
)
fig.update_layout(
    margin=dict(b=0, l=0, r=0, t=0),
    yaxis_categoryorder="category descending",
    legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
)

# %%
df = cs.get_ent("P_EG_buy_T").loc[:, :"sc7"] - cs.get_ent("P_EG_sell_T").loc[:, :"sc7"]
df /= 1e3
# df = df.rename(columns=lambda x: f"{x} ($\\alpha=${getattr(cs.scens, x).params.k_PTO_alpha_:.1f})")
fig, ax = plt.subplots(1, figsize=(9.5, 2))
fig.subplots_adjust(bottom=0.23, left=0.06, right=0.99, top=0.965)
sns.violinplot(data=df, orient="h", scale="width", ax=ax, cut=0, width=0.75, palette="Set2")
plt.margins(0.02)
ax.set_xlabel("Electrical power from grid (MW$_\mathrm{el}$)", fontsize=12)
ax.set_ylabel("Scenario", fontsize=12)
sns.despine()

# %%
fig, axes = plt.subplots(1, len(cs.scens_list), figsize=(10, 2), sharey=True, sharex=True)
fig.subplots_adjust(bottom=0.25, left=0.06, right=0.99, top=0.68)

for sc, ax in zip(cs.scens_list, axes):
    p_diff = (sc.res.P_EG_buy_T - sc.res.P_EG_sell_T) / 1e3
    incentive = sc.params.ce_EG_T * 1e3

    y = incentive
    x = p_diff

    ax.scatter(x, y, s=0.02)

    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, "k:", lw=2)

    ax.set_title(f"{sc.id}\n$\\rho_{{x,y}}$={x.corr(y):.2f}\nslope={m:.0f}")
axes[0].set_ylabel("CEFs (kg$_\mathrm{CO2eq}$/MWh)")
axes[3].set_xlabel("P$^\mathrm{buy}_t$ - P$^\mathrm{sell}_t$ (MW)")
sns.despine()

# %% [markdown]
# ### Pareto

# %% [raw]
# cs.plot.pareto()

# %%
# Pareto with broken axis for paper
import matplotlib.ticker as ticker

df = cs.pareto.copy()
x = df.CE_TOT_ / 1e6
y = df.C_TOT_ / 1e3

fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(6, 3), gridspec_kw={"height_ratios": [1, 4]}
)
fig.subplots_adjust(hspace=0.30, bottom=0.15, top=0.98, right=0.98)
sns.despine()

# REF point:
ax2.plot(x[0], y[0], "ko")

# guide lines of REF point:
ax2.plot([0, x[0]], [y[0], y[0]], "k", lw=1, alpha=0.4)
ax2.plot([x[0], x[0]], [y[0], 0], "k", lw=1, alpha=0.4)

# Pareto frontier and points:
for ax in (ax1, ax2):
    ax.plot(x[1:], y[1:], "ko", linestyle=":", linewidth=2)
    ax.set_xlim(0)
    ax.grid(True, axis="y", c="k", lw=1, alpha=0.15)

ax1.set_ylim(149.99, 155)
ax2.set_ylim(0, 20)
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.tick_params(labeltop=False, bottom=False)
ax2.xaxis.tick_bottom()
ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))

d = 0.2
kw = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax1.plot([0], [0], transform=ax1.transAxes, **kw)
ax2.plot([0], [1], transform=ax2.transAxes, **kw)

fig.text(0.03, 0.58, "TAC (M€/a)", va="center", rotation="vertical")
ax2.set_xlabel("Annual carbon emissions (kt$_\mathrm{CO2eq} / a$)")

for i in df.index:
    text = "REF" if i == "REF" else f"{i} ({getattr(cs.scens, i).name})".replace("a", r"$\alpha$=")
    ax1.annotate(
        text, xy=(x[i], y[i]), xytext=(x[i] + 0.2, y[i]), rotation=0, ha="left", va="center"
    )
    ax2.annotate(
        text, xy=(x[i], y[i]), xytext=(x[i] + 0.1, y[i] + 0.1), rotation=35, ha="left", va="bottom"
    )

factor = cs.pareto.C_TOT_["sc7"] / cs.pareto.C_TOT_["REF"]
print(f"sc7 has {factor:.1f} times higher comparative costs than REF")

# %% [markdown]
# ### Interactive

# %%
cs.plot.tables()

# %%
cs.plot.sankey_interact()

# %%
cs.plot.heatmap_interact("v")

# %% [markdown]
# ### Collectors

# %%
cs.scens.REF.plot.collectors(filter_etype="C", auto_convert_units=False)

# %%
cs.scens.sc3.plot.collectors(filter_etype="C", auto_convert_units=False)

# %% [markdown]
# ### combined heatmap and line

# %%
fig = sc.plot.heatmap_line_py(sc.params.dQ_cDem_TN[:, 1], colorbar_label="kW_th")
fig.update_layout(margin={"l": 0, "r": 0, "t": 20, "b": 0}, width=800, height=400)

# %%
fig = sc.plot.heatmap_line_py(sc.params.dQ_cDem_TN[:, 2], colorbar_label="kW_th")
fig.update_layout(margin={"l": 0, "r": 0, "t": 20, "b": 0}, width=800, height=400)

# %%
fig = sc.plot.heatmap_line_py(sc.params.dQ_hDem_TH[:, 1], colorbar_label="kW_th")
fig.update_layout(margin={"l": 0, "r": 0, "t": 20, "b": 0}, width=800, height=400)

# %%
sc.plot.heatmap_line_py(sc.params.dQ_hDem_TH[:, 2])

# %% [markdown]
# ### TS balance

# %%
fig = cs.scens.sc3.plot.ts_balance(
    data={
        "pos": ["P_PV_OC_T", "P_BES_out_T", "P_EG_buy_T"],
        "neg": ["P_eDem_T", "P_BES_in_T", "P_PV_FI_T"],
    },
    data_ylabel="Power<br>[MW]",
    data_conversion_factor=1e-3,
    addon_ts="c_EG_RTP_T",
    addon_conversion_factor=1e3,
    addon_ts_ylabel="Price<br>[€/MWh]",
    colors=dict(PV="#ffb703", EG="#023047", BES="#dd4477", eDem="#8ecae6"),
    ts_slicer=slice("2020-06-15", "2020-06-21"),
)
fig.update_layout(legend=dict(tracegroupgap=30, y=0.95))

# %% [markdown]
# ## Correlations

# %%
cs.plot.correlations("P_EG_buy_T", "c_EG_RTP_T")

# %%
cs.plot.correlations("P_EG_buy_T", "ce_EG_T")

# %%
plt.scatter(
    cs.scens.sc7.res.P_EG_buy_T - cs.scens.sc7.res.P_EG_sell_T, cs.scens.sc7.params.ce_EG_T, s=1
)

# %%
fig, ax = plt.subplots(1, len(cs.scens_list), figsize=(10, 2), sharex=True, sharey=True)
for sc, ax in zip(cs.scens_list, ax):
    p_diff = sc.res.P_EG_buy_T - sc.res.P_EG_sell_T
    incentive = sc.params.c_EG_RTP_T
    ax.scatter(p_diff, incentive, s=1)

    x = p_diff
    y = incentive

    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, "k:", lw=2)

# %%
fig, ax = plt.subplots(1, len(cs.scens_list), figsize=(10, 2), sharey=True, sharex=True)
for sc, ax in zip(cs.scens_list, ax):
    p_diff = (sc.res.P_EG_buy_T - sc.res.P_EG_sell_T) / 1e3
    incentive = sc.params.c_EG_RTP_T

    y = incentive
    x = p_diff

    ax.scatter(x, y, s=0.02)

    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, "k:", lw=2)

    ax.set_title(f"{sc.id}\n{x.corr(y):.2f}\n{m:.3f}")
    sns.despine()

# %%
plt.figure(figsize=(3, 3))
for i, sc in enumerate(cs.scens_list[::-1]):
    p_diff = sc.res.P_EG_buy_T - sc.res.P_EG_sell_T
    incentive = sc.params.ce_EG_T

    y = incentive
    x = p_diff

    plt.scatter(x, y, s=3, alpha=0.1)
    plt.title(f"{x.corr(y):.2f}")

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, lw=10 - i, label=sc.id, alpha=1)
    plt.legend()

