# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Extract 5% and 95% ranges of dxx, dxy, and pval from distribution shift logs.
import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file', required=True)
args = vars(parser.parse_args())

with open(args['file'], 'r') as f:
    reader = csv.reader(f)
    logs = list(reader)

# Get ranges of subsample tests.
data = {}
data_names = np.unique([l[0] for l in logs])
for data_name in data_names:
    data[data_name] = {'subsample': {}}

for line in logs:
    data_name = line[0]
    split_name = line[1]
    test_name = line[2]
    dxx = np.float(line[3])
    dxy = np.float(line[4])
    pval = np.float(line[5])
    decision_str = line[6]
    time_single_run = np.float(line[7])

    if split_name not in data[data_name]['subsample']:
        data[data_name]['subsample'][split_name] = {
            'dxx': [], 'dxy': [], 'pval': [], 'time_single_run': [],
            'decisions': {True: 0, False: 0},
        }

    if test_name == 'subsample':
        data[data_name]['subsample'][split_name]['dxx'].append(dxx)
        data[data_name]['subsample'][split_name]['dxy'].append(dxy)
        data[data_name]['subsample'][split_name]['pval'].append(pval)
        data[data_name]['subsample'][split_name]['time_single_run'].append(time_single_run)
        if 'No shift' in decision_str:
            data[data_name]['subsample'][split_name]['decisions'][False] += 1
        else:
            data[data_name]['subsample'][split_name]['decisions'][True] += 1

for data_name in data_names:
    subsample_info = data[data_name]['subsample']
    for split_name in subsample_info:
        measures = data[data_name]['subsample'][split_name]
        d0 = measures['decisions'][False]
        d1 = measures['decisions'][True]
        fit_score = max(d0, d1) / (d0 + d1)
        
        print(f'\ndata_name: {data_name}')
        print(f'    split_name: {split_name}')
        print(f'        time (single run, sec): {np.mean(measures["time_single_run"]):.3f}')
        print(f'        fit score: {fit_score:.2f}')
        print(f'        shift decisions: {measures["decisions"]}')
        print(f'        fit score, decisions (T:F): {fit_score:.2f} \,\, ({d1}:{d0})')
        #print(f'        dxx 5,95%: {np.percentile(measures["dxx"], 5):.3f},{np.percentile(measures["dxx"], 95):.3f}')
        #print(f'        dxy 5,95%: {np.percentile(measures["dxy"], 5):.3f},{np.percentile(measures["dxy"], 95):.3f}')
        print(f'        dxx, dxy: [{np.percentile(measures["dxx"], 5):.2f} \, \, {np.percentile(measures["dxx"], 95):.2f}] : [{np.percentile(measures["dxy"], 5):.3f} \, \, {np.percentile(measures["dxy"], 95):.3f}]')
        print(f'        pval: [{np.percentile(measures["pval"], 5):.2f} \, \, {np.percentile(measures["pval"], 95):.2f}]')
        print()
        print(
            f'{np.mean(measures["time_single_run"]):.2f} & '
            f'$[{np.percentile(measures["dxx"], 5):.2f} \, \, {np.percentile(measures["dxx"], 95):.2f}] : [{np.percentile(measures["dxy"], 5):.2f} \, \, {np.percentile(measures["dxy"], 95):.2f}]$ & '
            f'$[{np.percentile(measures["pval"], 5):.2f} \, \, {np.percentile(measures["pval"], 95):.2f}]$ & '
            f'${fit_score:.2f} \,\, ({d1}:{d0})$'
        )
