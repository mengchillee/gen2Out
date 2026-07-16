####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Email	: mengchil@cs.cmu.edu      #
####################################


import os
import time
import argparse

from .gen2out import gen2Out
from .utils import sythetic_group_anomaly, load_csv, plot_results, save_results


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for gen2Out')
    parser.add_argument('--lower_bound', default=9, type=int, help='Lower bound of sampling (2^i)')
    parser.add_argument('--upper_bound', default=12, type=int, help='Upper bound of sampling (2^i)')
    parser.add_argument('--max_depth', default=7, type=int, help='Maximum depth of each tree')
    parser.add_argument('--rotate', default=True, type=bool, help='Whether to use the rotated IF or not')
    parser.add_argument('--contamination', default='auto', type=str, help='Contamination rate of the dataset')
    parser.add_argument('--random_state', default=0, type=int, help='Control the randomness')
    parser.add_argument('--out', default='results', type=str, help='Directory to save the output plots')
    parser.add_argument('--data', default=None, type=str, help='Path to a CSV of 2D points; if omitted, the built-in synthetic dataset is used')
    parser.add_argument('--eps', default=1.0, type=float, help='DBSCAN neighborhood radius for grouping anomalies')
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of parallel jobs for group anomaly detection (-1 uses all cores)')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    model = gen2Out(lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    max_depth=args.max_depth,
                    rotate=args.rotate,
                    contamination=args.contamination,
                    random_state=args.random_state)

    if args.data is not None:
        print('Loading data from %s' % args.data)
        X = load_csv(args.data)
    else:
        X = sythetic_group_anomaly()

    print('Start point anomaly detection:')
    t1 = time.time()
    pscores = model.point_anomaly_scores(X)
    t2 = time.time()
    print('Finish in %.1f seconds!\n' % (t2 - t1))

    print('Start group anomaly detection:')
    t1 = time.time()
    model.group_anomaly_scores(X, eps=args.eps, n_jobs=args.n_jobs)
    t2 = time.time()
    print('Finish in %.1f seconds!\n' % (t2 - t1))

    print('Generating plots...')
    plot_results(X, model, out_dir=args.out)

    out_path = os.path.join(args.out, 'microclusters.parquet')
    print('Saving results to %s...' % out_path)
    save_results(X, model, out_path, point_scores=pscores)

    print('Finish!')


if __name__ == '__main__':
    main()
