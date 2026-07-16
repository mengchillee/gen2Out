# gen<sup>2</sup>Out: Detecting and Ranking Generalized Anomalies

------------

Lee, MC., Shekhar, S., Faloutsos, C., Hutson, TN., and Iasemidis, L., gen2Out: Detecting and Ranking Generalized Anomalies. *IEEE International Conference on Big Data (Big Data)*, 2021.

https://ieeexplore.ieee.org/abstract/document/9671550

Please cite the paper as:

    @inproceedings{lee2021gen2out,
      title={{gen2Out:} Detecting and Ranking Generalized Anomalies},
      author={Lee, Meng-Chieh and Shekhar, Shubhranshu and Faloutsos, Christos and Hutson, T Noah and Iasemidis, Leon},
      booktitle={2021 IEEE International Conference on Big Data (Big Data)},
      year={2021},
      organization={IEEE},
    }
    
## Installation

**With pip:**
```bash
pip install .
```

**With conda (alternative):**
```bash
bash run_install.sh conda
conda activate gen2out
```

**With uv (alternative):**
```bash
bash run_install.sh uv
source .venv/bin/activate
```

The code is written in Python 3 and built on a number of Python packages:
- matplotlib==3.5.0
- numpy==1.21.2
- scipy==1.7.3
- scikit_learn==1.0.2
- joblib==1.1.0
- tqdm==4.62.0

## Usage and Sample Output
Experiments of Fig. 6 in the paper could be reproduced by running the demo directly.
After installing the package, execute the code by

```bash
gen2out-demo [OPTIONS]
# or: gen2out [OPTIONS]
# or: python -m gen2out.main [OPTIONS]
# or: make demo
```
![image](https://user-images.githubusercontent.com/14501754/162111146-3e04009a-b443-43c9-8c9b-2c654a7e1a02.png)

### Output Figures

The demo runs point and group anomaly detection on a synthetic dataset, then
generates six figures in the output directory (`results/` by default), one per
step of the gen2Out pipeline:

| File | Description |
|---|---|
| `step0_heatmap.png` | Log-scaled hexbin density of the input data |
| `step1_xray_plot.png` | X-ray plot: anomaly score vs. qualification rate, with mean ± std bands and the `mean + 3 * std` threshold |
| `step2_apex_extraction.png` | Apex (max-score) point extracted from each X-ray curve |
| `step3_outlier_grouping.png` | Detected generalized anomalies, grouped by DBSCAN and colored per group |
| `step4_anomaly_isocurves.png` | Anomaly iso-curves, colored by distance to the ideal point |
| `step5_scoring.png` | Distribution of anomaly scores per generalized anomaly |

The figures are produced by `plot_results()` in `gen2out/utils.py` using the
model state populated during detection. They are tuned for the built-in
`sythetic_group_anomaly()` dataset.

### Dataset

The synthetic dataset used by the demo is saved at
[`data/synthetic_group_anomaly.csv`](data/synthetic_group_anomaly.csv) so you
can inspect it directly. It contains 10,304 2D points (one per row, `x,y`
header): a large uniform disk of normal points plus two planted group anomalies
(dense clusters) and a handful of scattered point anomalies.

Regenerate it at any time with:

```python
import numpy as np
from gen2out import sythetic_group_anomaly

X = sythetic_group_anomaly()
np.savetxt('data/synthetic_group_anomaly.csv', X, delimiter=',',
           header='x,y', comments='', fmt='%.6f')
```

Or run the demo on your own CSV of 2D points:

```bash
gen2out-demo --data data/synthetic_group_anomaly.csv
```

### Using gen2Out in your own code

```python
from gen2out import gen2Out, sythetic_group_anomaly, load_csv, plot_results

model = gen2Out(random_state=0)

# Use the built-in synthetic data, or load points from a CSV:
X = sythetic_group_anomaly()
# X = load_csv('data/synthetic_group_anomaly.csv')

point_scores = model.point_anomaly_scores(X)

# Group anomaly detection runs its trials in parallel across all cores.
# Tune `eps` (DBSCAN radius) and `n_jobs` as needed.
group_scores, group_indices = model.group_anomaly_scores(X, eps=1.0, n_jobs=-1)

# Generate the six step-by-step figures into results/
plot_results(X, model, out_dir='results')
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--lower_bound` | int | `9` | Lower bound of sampling (2^i) |
| `--upper_bound` | int | `12` | Upper bound of sampling (2^i) |
| `--max_depth` | int | `7` | Maximum depth of each tree |
| `--rotate` | bool | `True` | Whether to use the rotated IF or not |
| `--contamination` | str | `auto` | Contamination rate of the dataset |
| `--random_state` | int | `0` | Control the randomness |
| `--out` | str | `results` | Directory to save the output plots |
| `--data` | str | `None` | Path to a CSV of 2D points; if omitted, the built-in synthetic dataset is used |
| `--eps` | float | `1.0` | DBSCAN neighborhood radius for grouping anomalies |
| `--n_jobs` | int | `-1` | Number of parallel jobs for group anomaly detection (`-1` uses all cores) |

## Acknowledgement
One part of our code is based on scikit-learn IsolationForest, downloaded from https://github.com/scikit-learn/scikit-learn/.

This implementation is according to the following paper:

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In 2008 8th IEEE International Conference on Data Mining (pp. 413-422). IEEE.
