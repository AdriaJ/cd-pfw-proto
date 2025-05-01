import os
from time import time
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

dbdir = "/home/jarret/PycharmProjects/cd-pfw-proto/scaling_exp/database"

fig_path = "/home/jarret/Documents/EPFL/PhD/thesis/manuscript/figures/cd-pfw/scaling"

if __name__=="__main__":
    # list the file paths
    list_paths = []
    for dirpath, dirnames, filenames in os.walk(dbdir):
        if len(dirnames) == 0:
            list_paths += [os.path.join(dirpath, f) for f in filenames if f.endswith(".npz")]

    tmp = {"path": [], "peaks": [], "seed": [], "type": []}
    for path in list_paths:
        l = path.split('/')
        type = l[-1][:-4]
        if type != "source":
            peaks = int(l[-3])
            seed = int(l[-2])
            tmp["path"].append(path)
            tmp["peaks"].append(peaks)
            tmp["seed"].append(seed)
            tmp["type"].append(type)
    df = pd.DataFrame(tmp)

    durations = []
    flat_costs0 = []
    flat_costs1 = []
    loss = []
    t_candidates = []
    t_corrections = []
    t_slidings = []
    for row in df.itertuples():
        data = np.load(row[1], allow_pickle=True)
        durations.append(data["t"].item())
        flat_costs0.append(data["flat_costs"][0].item())
        flat_costs1.append(data["flat_costs"][1].item())
        loss.append(data["loss"].item())
        t_candidates.append(data["t_candidate"].item())
        t_corrections.append(data["t_correction"].item())
        t_slidings.append(data["t_sliding"].item())
    df["duration"] = durations
    df["flat_002"] = flat_costs0
    df["flat_01"] = flat_costs1
    df["loss"] = loss
    df["t_candidate"] = t_candidates
    df["t_correction"] = t_corrections
    df["t_sliding"] = t_slidings


    def subtract_type_loss(group, base_type='SFW'):
        base_loss = group.loc[group['type'] == base_type, 'loss']
        if base_loss.empty:
            return pd.Series([None] * len(group), index=group.index)  # fallback if base_type not in group
        return (group['loss'] - base_loss.values[0]) / base_loss.values[0]


    df['relative_loss'] = df.groupby(['seed', 'peaks']).apply(subtract_type_loss).reset_index(level=[0, 1], drop=True)

    # Filter for more than 5 peaks only

    df = df.loc[df["peaks"] > 6]

    types = ["SFW", "SFW_PSO", "PFW", "SlidingPFW"]
    t_times = {}
    break_times = {}
    metrics = {}
    for type in types:
        # t_times[type] = df[df['type'] == "SFW"].groupby('peaks')['duration'].apply(lambda x: np.array(x)).to_dict()
        t_times[type] = df[df['type'] == type][["peaks", "duration"]]
        metrics[type] = df[df['type'] == type][["peaks", "flat_002", "flat_01", "relative_loss"]]
        break_times[type] = df[df['type'] == type][["peaks", "t_candidate", "t_correction", "t_sliding"]]

    # plt.figure()
    # for t in types:
    #     plt.scatter(t_times[t]["peaks"], t_times[t]["duration"], marker='+', label=t)
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # for t in types:
    #     idx = t_times[t]["peaks"].unique()
    #     val = t_times[t].groupby("peaks").median()
    #     plt.scatter(idx, val.loc[idx]["duration"], marker='+', label=t)
    # plt.legend()
    # plt.show()

    # draw the same plot but using interquartile spread
    plt.figure(figsize=(8, 5))
    for t in types:
        idx = t_times[t]["peaks"].unique()
        idx.sort()
        median_val = t_times[t].groupby("peaks").median()
        quartile1_val = t_times[t].groupby("peaks").quantile(0.25)
        quartile3_val = t_times[t].groupby("peaks").quantile(0.75)
        plt.plot(idx, median_val["duration"].values, marker='+', label=t)
        plt.fill_between(idx, quartile1_val["duration"].values, quartile3_val["duration"].values, alpha=.5)
    plt.legend()
    plt.yscale('log')
    plt.xticks(idx)
    plt.xlabel("Number of impulses")
    plt.ylabel("Time (s)")
    plt.suptitle("Comparative reconstruction time")
    # plt.savefig(fig_path + "/" + "total_time" + ".pdf")
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=True)
    for ax, t in zip(axes.flat, types):
        for lab in ["t_candidate", "t_correction", "t_sliding"]:
            # if lab == "t_sliding" and t=="PFW":
            #     break
            idx = break_times[t]["peaks"].unique()
            idx.sort()
            median_val = break_times[t].groupby("peaks")[lab].median().values
            quartile1_val = break_times[t].groupby("peaks")[lab].quantile(0.25).values
            quartile3_val = break_times[t].groupby("peaks")[lab].quantile(0.75).values
            ax.plot(idx, median_val, marker='+', label=lab)
            ax.set_title(t)
            ax.fill_between(idx, quartile1_val, quartile3_val, alpha=.5)
        ax.legend()
        ax.set_yscale('log')
        ax.set_xticks(idx)
        ax.set_xlabel("Number of impulses")
        ax.set_ylabel("Time (s)")
        ax.label_outer()
    plt.suptitle("Reconstruction time for each method, split view")
    # plt.savefig(fig_path + "/" + "split_time" + ".pdf")
    plt.show()



    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lab in zip(axes, ["flat_002", "flat_01"]):
        for t in types:
            # if lab == "relative_loss" and t=="SFW":
            #     break
            idx = metrics[t]["peaks"].unique()
            idx.sort()
            median_val = metrics[t].groupby("peaks")[lab].median().values
            quartile1_val = metrics[t].groupby("peaks")[lab].quantile(0.25).values
            quartile3_val = metrics[t].groupby("peaks")[lab].quantile(0.75).values
            ax.plot(idx, median_val, marker='+', label=t)
            # ax.fill_between(idx, quartile1_val, quartile3_val, alpha=.5)
        ax.legend()
        ax.set_yscale('log')
        ax.set_xticks(idx)
    plt.show()

    plt.figure()
    lab="relative_loss"
    for t in types:
        idx = t_times[t]["peaks"].unique()
        idx.sort()
        median_val = metrics[t].groupby("peaks")[lab].median().values
        quartile1_val = metrics[t].groupby("peaks")[lab].quantile(0.25).values
        quartile3_val = metrics[t].groupby("peaks")[lab].quantile(0.75).values
        plt.plot(idx, median_val, marker='+', label=t)
    plt.legend()
    # plt.yscale('log')
    plt.xticks(idx)
    plt.show()

    # df[["seed", "peaks", "type", "relative_loss"]]