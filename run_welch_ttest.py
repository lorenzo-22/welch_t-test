#!/usr/bin/env python

"""
Simple Welch t-test Omnibenchmark module.

Loads:
  --data         csv dataframe with proteins as rows and samples as columns. 

Outputs:
  A CSV-like matrix:
      header: ID, effect_size, p_value
      rows:   per-protein results
"""

import argparse
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import warnings


def load_dataset(data_file):
    data = (
    pd.read_csv(data_file, index_col=0)
    .select_dtypes(include=['number']) # Select only proteomics data
    .T #transform
)
    if data.ndim != 2:
        raise ValueError("Invalid data structure: data.matrix must be 2D")
    return data

def load_labels(labels_file):
    data = pd.read_csv(labels_file, index_col=0, header=None)
    labels = data.iloc[:, 0].to_numpy()
    labels = labels.astype(int) 
    return labels


def welch_ttest_df(data, labels):
    """
    data: Pandas dataframe with proteins as columns and samples as rows
    labels: 1D array of 0/1 group labels, length = n_samples = n_rows in data
    
    Returns:
        pandas DataFrame with t-statistics and p-values for each column
    """

    X = np.asarray(data)
    labels = np.asarray(labels)
    
    results = []

    for col_idx in range(data.shape[1]):
        col = X[:, col_idx]
        
        # Split into groups
        group0 = col[labels == 0]
        group1 = col[labels == 1]
        
        # Run Welch's t-test (unequal variance)
        t_stat, p_val = ttest_ind(group0, group1, equal_var=False)
        
        results.append({'ID': f'{data.columns[col_idx]}', 'effect_size': t_stat, 'p_value': p_val})

    df = pd.DataFrame(results)
    return df



def main():
    parser = argparse.ArgumentParser(description='Welch t-test benchmark runner')

    parser.add_argument('--data.matrix', type=str,
                        help='csv dataframe with proteins as rows and samples as columns.', required = True)
    parser.add_argument('--data.true_labels', type=str,
                        help='csv file with true labels.', required = True)
    parser.add_argument('--output_dir', type=str,
                        help='output directory to store data files.', 
                        required=True)

    parser.add_argument('--name', type=str, help='dataset name (ignored, for compatibility)', required=False)
    # parser.add_argument('--method', type=str,
    #                     help='sklearn method',
    #                     required = True)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    name = getattr(args, 'name')

    print('Loading data')
    data = load_dataset(getattr(args, 'data.matrix'))
    labels = load_labels(getattr(args, 'data.true_labels'))

    print('Running Welch t-test')
    results = welch_ttest_df(data, labels)

    output_file = os.path.join(args.output_dir, f"{args.name}_results.csv")
    results.to_csv(output_file, index=False)
    print(f'Welch t-test results stored in {output_file}')

if __name__ == "__main__":
    main()
