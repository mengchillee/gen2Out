####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Email	: mengchil@cs.cmu.edu      #
####################################


import os

import numpy as np
from scipy.spatial.distance import cityblock

import matplotlib.pyplot as plt


### Sizes of the four data segments: background disk, two dense group
### anomalies, and the scattered point anomalies. The X-ray plot indices are
### derived from these, so the demo stays consistent if the sizes change.
DISK_SIZE = 10000
GROUP_A_SIZE = 100
GROUP_B_SIZE = 200
N_POINT_ANOMALIES = 4

def uni_disk(n, low=0, high=1):
    r = np.random.uniform(low=low, high=high, size=n)  # radius
    theta = np.random.uniform(low=0, high=2*np.pi, size=n)  # angle
    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)
    return x, y

def sythetic_group_anomaly(seed=0):
    np.random.seed(seed)

    x1, y1 = uni_disk(DISK_SIZE)
    x1 *= 5
    y1 *= 5

    x2, y2 = uni_disk(GROUP_A_SIZE)
    x2 = x2 * 1.5 + 10
    y2 = y2 * 1.5 + 5

    x3, y3 = uni_disk(GROUP_B_SIZE)
    x3 = x3 * 6 + 3
    y3 = y3 - 10

    x4 = [11, -2, 13, 14]
    y4 = [0, 9, -10, 10]

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    X_norm = np.array([x, y]).T

    return X_norm

def uni_ball(n, dim, radius=1):
    ### Uniformly sample n points inside a `dim`-dimensional ball of `radius`
    directions = np.random.normal(size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    r = radius * np.random.uniform(size=(n, 1)) ** (1 / dim)
    return directions * r

def sythetic_group_anomaly_4d(seed=0):
    ### 4D analogue of sythetic_group_anomaly: one large background ball plus
    ### two dense group anomalies and four scattered point anomalies. Segment
    ### sizes match the 2D version so the demo indices stay consistent.
    np.random.seed(seed)

    bg = uni_ball(DISK_SIZE, 4, radius=5)
    ga1 = uni_ball(GROUP_A_SIZE, 4, radius=1.5) + np.array([10, 5, 10, 5])
    ga2 = uni_ball(GROUP_B_SIZE, 4, radius=2.0) + np.array([3, -10, -8, 6])

    pa = np.array([[11, 0, 12, -3],
                   [-2, 9, -11, 7],
                   [13, -10, 10, 10],
                   [14, 10, -12, -11]])

    return np.concatenate([bg, ga1, ga2, pa])

def _demo_plot_indices(sample_per_group=300):
    ### Sample indices from the start of each data segment for the X-ray plots
    sizes = [DISK_SIZE, GROUP_A_SIZE, GROUP_B_SIZE, N_POINT_ANOMALIES]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    idx = [np.arange(offsets[k], offsets[k] + min(sample_per_group, sizes[k]))
           for k in range(len(sizes))]
    return np.concatenate(idx)

def load_csv(path):
    ### Load a 2D point dataset from a CSV file (one point per row)
    return np.genfromtxt(path, delimiter=',', skip_header=1)

def results_dataframe(X, model, point_scores=None):
    ### Build a per-point table of the group anomaly detection results.
    ### Requires model.group_anomaly_scores(X) to have been run first.
    import pandas as pd

    xrays = np.max(np.mean(model.scores, axis=1), axis=0)

    ### Map each point's cluster to its group anomaly severity score (NaN if
    ### the point is not part of any detected generalized anomaly).
    group_score = np.full(len(X), np.nan)
    for cid in np.unique(model.labels):
        if cid != -1:
            group_score[model.labels == cid] = model.ga_scores[cid - 1]

    ### A point above the threshold is an anomaly. If it landed in a cluster it
    ### is a group anomaly, otherwise it is a (standalone) point anomaly.
    above_threshold = xrays >= model.threshold
    is_group_anomaly = above_threshold & (model.labels != -1)
    is_point_anomaly = above_threshold & (model.labels == -1)

    ### One column per feature: x, y for 2D, else feature_0, feature_1, ...
    if X.shape[1] == 2:
        df = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1]})
    else:
        df = pd.DataFrame(X, columns=['feature_%d' % d for d in range(X.shape[1])])
    if point_scores is not None:
        df['point_score'] = point_scores           # point anomaly score
    df['xray_score'] = xrays                        # group anomaly x-ray score
    df['is_point_anomaly'] = is_point_anomaly       # above threshold, not clustered
    df['is_group_anomaly'] = is_group_anomaly       # above threshold, in a cluster
    df['cluster_id'] = model.labels                 # DBSCAN group (-1 = ungrouped)
    df['group_score'] = group_score                 # group anomaly severity (per cluster)
    return df

def save_results(X, model, path, point_scores=None):
    ### Save the per-point results table to a Parquet file.
    df = results_dataframe(X, model, point_scores=point_scores)
    df.to_parquet(path)
    return df

def plot_xray(X, model, idx_arr, line=False):
    plt.scatter(1, 1, s=100, c='k', marker='*')
    xline = 1 / (2 ** np.arange(0, model.min_rate))

    for idx in idx_arr:
        s = model.scores.T[idx].T
        std, mean = np.std(s, axis=1), np.mean(s, axis=1)
        if line:
            plt.plot(xline, mean, c='k', alpha=0.7)
            plt.fill_between(xline, mean-std, mean+std, color='grey', alpha=0.2)

        max_idx = np.argmax(mean)
        plt.scatter(xline[max_idx], mean[max_idx], s=20, c='k')
    
    plt.plot([2 ** (-(model.min_rate - 0.7)), 1.2], [model.threshold, model.threshold], '--', label='Mean + 3 * Std', alpha=0.8, c='r')
    plt.ylim(-0.05, 1.05)
    plt.xlim(2 ** (-(model.min_rate - 0.7)), 1.2)
    plt.xscale('log', base=2)
    plt.xlabel('Qualification Rate', fontsize=20)
    plt.ylabel('Anomaly Score', fontsize=20)
    plt.legend(fontsize=12)

def plot_results(X, model, x_ideal=1, y_ideal=1, out_dir='results'):
    if X.shape[1] != 2:
        print('Skipping plots: figures are only supported for 2D data (got %dD).' % X.shape[1])
        return
    os.makedirs(out_dir, exist_ok=True)
    ### Sample points from each data segment when plotting
    idx_arr = _demo_plot_indices()

    ### Plot heatmap
    plt.figure(figsize=(4.8, 4))
    plt.hexbin(X[:, 0], X[:, 1], cmap='cool', gridsize=30, bins='log', mincnt=1)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'step0_heatmap.png'))

    ### Step 1: X-ray plot
    plt.figure(figsize=(4, 4))
    plot_xray(X, model, idx_arr, line=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'step1_xray_plot.png'))

    ### Step 2: Apex extraction
    plt.figure(figsize=(4, 4))
    plot_xray(X, model, idx_arr, line=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'step2_apex_extraction.png'))

    ### Step 3: Outlier grouping 
    c_arr = ['', 'b', 'r', 'y', 'm', 'g', 'c']
    plt.figure(figsize=(4, 4))
    plt.scatter(X[:, 0], X[:, 1], c='lightgrey', alpha=0.5)

    for l in np.unique(model.labels):
        if l != -1:
            idx = np.where(model.labels == l)[0]
            plt.scatter(X[idx, 0], X[idx, 1], c=c_arr[l], label='GA ' + str(l))

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'step3_outlier_grouping.png'))

    ### Step 4: Anomaly iso-curves
    man_x, man_y, man_dis = [], [], []
    for i in np.arange(0, model.min_rate, 0.01):
        for j in np.arange(0, 1.01, 0.01):
            ix = 1 / (2 ** i)
            man_x.append(ix)
            man_y.append(j)
            man_dis.append(cityblock([np.log2(ix) / 10, j], [x_ideal, y_ideal]))
    man_x, man_y, man_dis = np.array(man_x), np.array(man_y), np.array(man_dis)

    plt.figure(figsize=(4.8, 4))
    plt.scatter(man_x, man_y, c=man_dis, cmap='gist_rainbow', alpha=0.1)
    plt.colorbar()
    plt.scatter(x_ideal, y_ideal, s=100, c='k', marker='*')

    xline = 1 / (2 ** np.arange(0, model.min_rate))
    for idx in idx_arr:
        if model.labels[idx] != -1:
            c = c_arr[model.labels[idx]]
            s = model.scores.T[idx].T
            std, mean = np.std(s, axis=1), np.mean(s, axis=1)
            plt.plot(xline, mean, c=c, alpha=0.05)
            max_idx = np.argmax(mean)
            plt.scatter(xline[max_idx], mean[max_idx], s=20, c=c)
    for l in np.unique(model.labels):
        if l != -1:
            plt.plot([], [], '-o', c=c_arr[l], label='GA ' + str(l))

    plt.xscale('log', base=2)
    plt.ylim(-0.05, 1.05)
    plt.xlim(2 ** (-(model.min_rate - 0.7)), 1.2)
    plt.xlabel('Qualification Rate', fontsize=20)
    plt.ylabel('Anomaly Score', fontsize=20)
    plt.legend(fontsize=12, loc=4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'step4_anomaly_isocurves.png'))

    ### Step 5: Scoring
    plt.figure(figsize=(4.4, 4))

    for idx, s in enumerate(model.s_arr):
        ymin, ymax = np.min(s), np.max(s)
        Q1, Q3 = np.percentile(s, 25), np.percentile(s, 75)
        m = np.median(s)
        plt.scatter([idx, idx], [ymin, ymax], facecolors='none', edgecolors='lightgrey')
        plt.plot([idx, idx], [Q1, Q3], c='grey', linewidth=0.9)
        plt.plot([idx-0.12, idx+0.12], [Q1, Q1], c='grey', linewidth=0.9)
        plt.plot([idx-0.12, idx+0.12], [Q3, Q3], c='grey', linewidth=0.9)
        plt.plot([idx-0.24, idx+0.24], [m, m], c='r', linewidth=3)

    plt.xticks(np.arange(len(model.s_arr)), ['GA '+str(i+1) for i in range(len(model.s_arr))], fontsize=12)
    plt.xlabel('Generalized Anomaly ID', fontsize=20)
    plt.ylabel('Distribution of\nAnomaly Score', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'step5_scoring.png'))
