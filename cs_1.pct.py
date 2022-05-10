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
# # DRAF demo Case Study 1: Price-Based DR Potential of Industrial Production Process

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import holidays

import draf
from draf.components import *


# %% [markdown]
# ## Prepare Data

# %%
def make_sort_demand_TS(cs):
    avg_demand = {
        1: 1310,
        2: 2030,
        3: 2810,
        4: 3120,
        5: 3430,
        6: 3750,
        7: 3280,
        8: 3040,
        9: 3430,
        10: 3430,
        11: 3280,
        12: 1800,
    }

    ser = pd.Series(data=cs.dtindex, index=range(len(cs.dtindex)))
    is_demand = ser.apply(
        lambda x: True
        if (8 <= x.hour < 20) and x.dayofweek < 5 and x not in holidays.Germany(prov="BW")
        else False
    )
    ser.loc[is_demand] = ser.dt.month.map(avg_demand) / 12
    ser.loc[~is_demand] = 0

    shares = pd.Series([1, 1, 2], index=range(1, 4))
    shares = shares / shares.sum()

    return (ser.to_frame() @ shares.to_frame().T).stack()


def remove_invalid_sort_machine_combinations(sc):
    for sm in [(2, 1), (1, 2)]:
        sc.params.y_PP_compat_SM[sm] = 0


def get_eta_PP_SM():
    sort_factors = (pd.Series([0.019, 0.020, 0.021], index=[1, 2, 3])).to_frame()
    machine_factors = pd.Series([0.95, 1.0], index=[1, 2]).to_frame()
    return (sort_factors @ machine_factors.T).stack()


def add_machine_revisions(sc):
    # (March 15th until March 16th):
    sc.params.y_PP_avail_TM.loc[sc.timeslice("3-15", "3-16"), 1] = 0
    sc.params.y_PP_avail_TM.loc[sc.timeslice("2-15", "2-16"), 1] = 0


sort_colors = ["#6c8ebf", "#d79b00", "#b85450"]

# %% [markdown]
# ## Modeling

# %%
coords = (49.01, 8.39)  # Random place in Karlsruhe
cs = draf.CaseStudy("draf_demo_cs1", year=2019, freq="60min", coords=coords, consider_invest=False)
# cs.set_time_horizon(start="Jun-01 00:00", steps=30 * 24)
sorts = (1, 2, 3)
machines = (1, 2)
sc = cs.add_REF_scen(components=[eDem, EG, PP, PS, pDem(sorts=sorts, machines=machines), Main])
sc.update_params(
    c_EG_buyPeak_=50,
    P_PP_CAPx_M=3500,
    k_PP_minPL_M=1.0,
    y_PP_compat_SM=1,
    G_PS_CAPx_S=5000,
    k_PS_min_S=0.2,
    c_PP_SU_=10,
    c_PP_SC_=10,
)
remove_invalid_sort_machine_combinations(sc)
add_machine_revisions(sc)
sc.params.eta_PP_SM.update(get_eta_PP_SM())
sc.params.dG_pDem_TS.update(make_sort_demand_TS(cs))

cs.add_scens([("c_EG_T", "t", [f"c_EG_{s}_T" for s in ["TOU", "RTP"]])], remove_base=True)

# %%
# Uncomment to run optimization:
# cs.optimize(parallel=True).save(name=f"{cs.year}")

# %%
cs = draf.open_latest_casestudy("draf_demo_cs1")

# %% [markdown]
# ## Plot Parameters

# %% [markdown]
# ### Plot Production Efficiency

# %%
(get_eta_PP_SM().unstack() * 1e3).plot(style=[":o", ":x"], figsize=(2.5, 2), color=["k", "grey"])
plt.ylim(16)
plt.xticks([1, 2, 3])
plt.ylabel("Production efficiency\n(t/MWh)")
plt.xlabel("Cement sort")
plt.legend(loc="lower right", title="Cement mill")
sns.despine()

# %% [markdown]
# ### Plot Sort Demand

# %%
demand = make_sort_demand_TS(cs)
fig, ax = plt.subplots(2, figsize=(6, 3), sharex=True)
df = cs.dated(demand.unstack()).sort_index(axis=1, ascending=False)
df.plot.area(linewidth=0, ax=ax[0], color=sort_colors[::-1])
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(
    handles[::-1],
    labels[::-1],
    title="Sort",
    fontsize="small",
    frameon=True,
    loc="lower center",
    ncol=3,
)
plt.margins(0)
ax[0].set_ylabel("Cement demand\n(t/h)")
cs.dated(cs.REF_scen.params.P_eDem_T).plot(ax=ax[1], color="grey", linewidth=0.1)
ax[1].set_ylabel("$P_{el,fixed}$ (kW)")
sns.despine()

# %% [markdown]
# ## Plot Results

# %% [markdown]
# ### Interactive

# %%
cs.plot.tables()

# %%
cs.plot.heatmap_interact("v")

# %%
cs.plot.heatmap_interact("v", dim="TSM")

# %%
cs.scens.tc_EG_RTP_T.plot.collectors(filter_etype="P")

# %%
cs.scens.tc_EG_RTP_T.plot.collectors(filter_etype="C")

# %% [markdown]
# ### Correlations

# %%
cs.plot.correlations("c_EG_RTP_T", "P_EG_buy_T")

# %%
cs.plot.correlations("ce_EG_T", "P_EG_buy_T")

# %%
fig, ax = plt.subplots(1, 2, figsize=(6, 2), sharey=True)

for i, sc in enumerate(cs.scens_list):
    c = sc.params.c_EG_RTP_T * 1e3
    p = sc.res.P_EG_buy_T / 1e3
    corr = c.corr(p)
    sns.regplot(
        y=p, x=c, ax=ax[i], scatter_kws={"s": 1, "color": "black"}, line_kws={"linewidth": 1}
    )
    pric_scheme = sc.id.split("_")[2]
    ax[i].set_title(f"{pric_scheme} scenario\n$r={corr:.2f}$")
    ax[i].set_xlabel("Real-time-price (€/MWh)")
    ax[i].set_ylabel("Purchased electricity\n(MWh)")

ax[1].set_ylabel("")
sns.despine()

# %% [markdown]
# ### Production Planning

# %%
from matplotlib.patches import Patch


def plot_res(tariff_type, timeslice, sc, patch_legend=False, save=False):
    fig, ax = plt.subplots(3, figsize=(10, 2.5))
    sns.despine()

    axis = ax[0]
    dated = True
    c_EG_T = sc.dated(sc.params.c_EG_T.loc[timeslice], dated) * 1e3
    c_EG_T.plot(ax=axis, drawstyle="steps", linewidth=1, c="black", alpha=1, label=tariff_type)
    axis.margins(0, 0.05)
    axis.set_ylabel("El. price\n(€/MWh)", rotation="horizontal", ha="right", va="center")
    axis.axes.get_xaxis().set_visible(False)

    axis = ax[1]
    ser = sc.res.dG_PP_TSM.loc[timeslice, :, :]
    ser[ser > 0] = ser.index.get_level_values(1)
    df = ser.groupby(level=(0, 2)).sum().unstack().astype(int).T
    cbar_labels = ["off", "sort 1", "sort 2", "sort 3"]
    off_color = "#eeeeee"
    colors = [off_color] + sort_colors
    cmap = sns.color_palette(colors)
    sns.heatmap(df, cmap=cmap, ax=axis, cbar=False)
    axis.axes.get_xaxis().set_visible(False)
    axis.set_ylabel("Machine\nactivity", rotation="horizontal", ha="right", va="center")
    axis.tick_params(axis="y", labelrotation=0)

    axis = ax[2]
    sc.dated(sc.res.G_PS_TS.loc[timeslice, :].unstack() / 1e3, dated).plot.area(
        ax=axis, color=sort_colors, legend=False, linewidth=0
    )
    axis.margins(0, 0)
    axis.set_ylabel("Silo filling\nlevel (kt)", rotation="horizontal", ha="right", va="center")
    if patch_legend:
        legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors, cbar_labels)]
        axis.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.4),
            ncol=4,
            frameon=False,
        )

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=0.2)
    fig.suptitle(tariff_type, fontsize=12, fontweight="bold", x=-0.11, ha="left", va="center")
    fullyear_str = "_fullyear" if timeslice == slice(None, None) else ""


ts = cs.timeslice(4, 4)
plot_res("TOU", ts, cs.scens.tc_EG_TOU_T, patch_legend=False)
plot_res("RTP", ts, cs.scens.tc_EG_RTP_T, patch_legend=True)
