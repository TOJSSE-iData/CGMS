import os
import pickle


import matplotlib.pyplot as plt
import torch
import json
import random
import numpy as np

from collections import Counter


def save_best_model(state_dict, model_dir: str, best_epoch: int, keep: int):
    save_to = os.path.join(model_dir, '{}.pkl'.format(best_epoch))
    torch.save(state_dict, save_to)
    model_files = [f for f in os.listdir(model_dir) if os.path.splitext(f)[-1] == '.pkl']
    epochs = [int(os.path.splitext(f)[0]) for f in model_files if str.isdigit(f[0])]
    outdated = sorted(epochs, reverse=True)[keep:]
    for n in outdated:
        os.remove(os.path.join(model_dir, '{}.pkl'.format(n)))


def find_best_model(model_dir: str):
    model_files = [f for f in os.listdir(model_dir) if os.path.splitext(f)[-1] == '.pkl']
    epochs = [int(os.path.splitext(f)[0]) for f in model_files if str.isdigit(f[0])]
    best_epoch = max(epochs)
    return os.path.join(model_dir, '{}.pkl'.format(best_epoch))


def save_and_visual_loss(loss: list, loss_file: str,
                         title: str = None, xlabel: str = None, ylabel: str = None, step: int = 1):
    with open(loss_file, 'wb') as f:
        pickle.dump(loss, f)

    plt.figure()
    plt.plot(range(step, step * len(loss) + 1, step), loss)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.savefig(loss_file[:-3] + 'png')


def save_args(args, save_to: str):
    args_dict = args.__dict__
    with open(save_to, 'w') as f:
        json.dump(args_dict, f, indent=2)


def load_args(args, load_from: str):
    args_dict = args.__dict__
    with open(load_from, 'r') as f:
        tmp_dict = json.load(f)
    for k, v in tmp_dict.items():
        args_dict[k] = v


def read_map(map_file):
    d = {}
    with open(map_file, 'r') as f:
        f.readline()
        for line in f:
            k, v = line.rstrip().split('\t')
            d[k] = int(v)
    return d


def window_smooth_loss(losses, ws):
    ws = max(ws, 3)
    half_ws = ws // 2
    new_losses = [losses[0]] * half_ws
    new_losses.extend(losses)
    new_losses.extend([losses[-1]] * half_ws)
    smoothed = []
    for i in range(half_ws, half_ws + len(losses)):
        low = i - half_ws
        high = low + ws
        val = sum(new_losses[low:high]) / ws
        smoothed.append(val)
    return arg_min(smoothed)


def arg_min(lst):
    m = float('inf')
    idx = 0
    for i, v in enumerate(lst):
        if v < m:
            m = v
            idx = i
    return m, idx


def calc_stat(numbers):
    mu = sum(numbers) / len(numbers)
    sigma = (sum([(x - mu) ** 2 for x in numbers]) / len(numbers)) ** 0.5
    return mu, sigma


def conf_inv(mu, sigma, n):
    delta = 2.776 * sigma / (n ** 0.5)  # 95%
    return mu - delta, mu + delta


def random_split_indices(dataset, test_size: int = None, test_rate: float = None):
    u_keys = dataset.keys.drop_duplicates()
    n_keys = u_keys.shape[0]
    if test_size is None and test_rate is None:
        raise ValueError("Either train_rate or test_rate should be given.")
    elif test_size is not None:
        if test_size <= 0:
            raise ValueError("test size should be larger than 0, found {}".format(test_size))
        train_size = n_keys - test_size
    elif test_rate is not None:
        if test_rate < 0 or test_rate > 1:
            raise ValueError("test rate should be in [0, 1], found {}".format(test_rate))
        train_size = int(n_keys * (1 - test_rate))
    else:
        Warning("Both test_size and test_rate are given, use test_size.")
        train_size = n_keys - test_size

    u_keys = u_keys.sample(frac=1).reset_index(drop=True)
    u_keys['flag'] = -1
    u_keys.iloc[:train_size, 2] = 0
    u_keys.iloc[train_size:, 2] = 1
    k2f = u_keys.set_index(['drug_row_idx', 'drug_col_idx'])['flag'].to_dict()
    train_indices = []
    test_indices = []
    for i, row in dataset.keys.iterrows():
        if k2f[(int(row['drug_row_idx']), int(row['drug_col_idx']))] == 0:
            train_indices.append(i)
        else:
            test_indices.append(i)
    return train_indices, test_indices



def real_random_split_indices(n_samples, test_size: int = None, test_rate: float = None):
    if test_size is None and test_rate is None:
        raise ValueError("Either train_rate or test_rate should be given.")
    elif test_size is not None:
        if test_size <= 0:
            raise ValueError("test size should be larger than 0, found {}".format(test_size))
        train_size = n_samples - test_size
    elif test_rate is not None:
        if test_rate < 0 or test_rate > 1:
            raise ValueError("test rate should be in [0, 1], found {}".format(test_rate))
        train_size = int(n_samples * (1 - test_rate))
    else:
        Warning("Both test_size and test_rate are given, use test_size.")
        train_size = n_samples - test_size
    evidence = list(range(n_samples))
    random.shuffle(evidence)
    train_indices = evidence[:train_size]
    test_indices = evidence[train_size:]
    return train_indices, test_indices



def stratified_split_indices(samples, groupby, test_rate: float = None):
    n_samples = len(samples)
    counter = Counter()
    for sample in samples:
        ks = []
        for gp in groupby:
            ks.append(sample[gp])
        ks = tuple(ks) if len(ks) > 1 else ks[0]
        counter[ks] += 1
    keys = list(counter.keys())
    random.shuffle(keys)
    test_keys = set()
    test_count = 0
    cur = 0
    while n_samples * test_rate > test_count:
        test_keys.add(keys[cur])
        test_count += counter[keys[cur]]
        cur += 1
    train_indices = []
    test_indices = []
    for i in range(n_samples):
        sample = samples[i]
        ks = []
        for gp in groupby:
            ks.append(sample[gp])
        ks = tuple(ks) if len(ks) > 1 else ks[0]
        if ks in test_keys:
            test_indices.append(i)
        else:
            train_indices.append(i)
    return train_indices, test_indices


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)