#!/usr/bin/env python
# coding: utf-8

import glob
import logging
import os
import random
from math import floor
from typing import Tuple, List

import pandas as pd


def select_files(rootdir, datasource, labels, num_rows, is_validation):
    selected = []
    rows = 0
    for act in labels[datasource]:
        apath = f"{rootdir}/data/{'validation' if is_validation else 'train_test'}/{datasource}/{act}/*.csv"
        f = glob.glob(apath)[0]
        df = pd.read_csv(f)
        rows += len(df.index)
        selected.append(f)
    files = [f for f in glob.glob(f"{rootdir}/data/{'validation' if is_validation else 'train_test'}/{datasource}/*/*.csv")]
    random.seed(72)
    random.shuffle(files)
    for f in files:
        if f not in selected:
            df = pd.read_csv(f)
            rows += len(df.index)
            selected.append(f)
            if rows >= num_rows:
                return selected
    return selected


def get_file_list_from_dir_uci_har(rootdir, datasource, num_rows, is_validation):
    # Below are the labels before limiting experiments to 4 labels
    # labels_lst = ["sitting", "standing", "walking_upstairs", "walking_downstairs", "walking", "laying"]
    labels_lst = ["sitting", "standing", "walking_upstairs", "walking_downstairs"]
    fs = {}
    for datatype in ["train", "test"]:
        apath = f"{rootdir}/data/{'validation' if is_validation else 'train_test'}/{datasource}_{datatype}/*/*.csv"
        if num_rows is None:
            fs[datatype] = [f for f in glob.glob(apath)]
        else:
            labels = {f"UCI-HAR_{datatype}": labels_lst}
            fs[datatype] = select_files(rootdir, f"{datasource}_{datatype}", labels, num_rows, is_validation)
    return fs


def get_file_list_from_dir(rootdir, datasource, num_rows, is_validation):
    labels = {
        # Below are the labels before limiting experiments to 4 labels
        # "KU-HAR": ["sit", "stand", "stair_down", "stair_up", "walk", "walk_circle", "lay"],
        # "PAMAP2": ["sitting", "standing", "ascending", "descending", "walking", "lying"]
        "KU-HAR": ["sit", "stand", "stair_down", "stair_up"],
        "PAMAP2": ["sitting", "standing", "ascending", "descending"]
    }
    if num_rows is None:
        apath = f"{rootdir}/data/{'validation' if is_validation else 'train_test'}/{datasource}/*/*.csv"
        return [f for f in glob.glob(apath)]
    else:
        fs = select_files(rootdir, datasource, labels, num_rows, is_validation)
        return fs


def randomize_files(allfiles):
    random.seed(72)
    random.shuffle(allfiles)
    return allfiles


def create_folds(xs, n):
    k, m = divmod(len(xs), n)
    lst = list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return lst


def create_train_test_uci(train_folds, test_folds, index):
    training = []
    test = []
    for i, fold in enumerate(train_folds):
        if i == index:
            test = test_folds[i]
        else:
            training.extend(train_folds[i])
    return training, test


def create_train_test(folds, index):
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training.extend(fold)
    return training, test


def csv2df(files):
    df1 = pd.read_csv(files[0])
    if len(files) == 1:
        return df1
    else:
        for f in files[1:]:
            if os.path.exists(f):
                df2 = pd.read_csv(f)
                concat = pd.concat([df1, df2], axis=0, ignore_index=True)
                assert len(concat) == len(df1) + len(df2)
                assert len(concat.columns) == len(df1.columns) == len(df2.columns)
                df1 = concat.copy()
            else:
                logging.error(f"can't find file {f} to convert to df")
                raise ValueError
        return concat


def load_filenames(filepath):
    allfiles = get_file_list_from_dir(filepath)
    allfiles = randomize_files(allfiles)
    return allfiles
