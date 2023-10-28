import logging
import random
from typing import List, Tuple

import pandas as pd

import cross_validation as cv
import data_imputations
from blstm_models import SequenceDataset
from dim_reduction import decompose_dims


def gen_source_domain_test_train_ds_from_folds(idx: int, target_label: str, folds: List, sequence_length: int,
                                               device: str, reduce_dims: bool, reduced_dim_num: int,
                                               datasource: str, uci_test_fold: List) -> \
        tuple[SequenceDataset, SequenceDataset, int, int, int, int]:
    # load source domain files into dataframes
    logging.debug("reading source domain csv files")
    if uci_test_fold is None:
        train, test = cv.create_train_test(folds=folds, index=idx)
    else:
        train, test = cv.create_train_test_uci(train_folds=folds, test_folds=uci_test_fold, index=idx)
    random.seed(72)
    random.shuffle(train)
    random.shuffle(test)
    train_df = cv.csv2df(train)
    test_df = cv.csv2df(test)
    # do on-the-fly imputations required by datasource
    if datasource == "PAMAP2":
        logging.debug(f"source knn imputation: {datasource}")
        train_df, test_df = data_imputations.knn_impute_missing(train_df, test_df, target_label)
    if datasource == "PAMAP2" or datasource == "KU-HAR":
        logging.debug(f"source min-max normalization: {datasource}")
        train_df, test_df = data_imputations.minmax_normalization(train_df, test_df, target_label)
    # perform PCA if model type is BLSTMGauss
    # fit ONLY on train data
    # transform test and train
    if reduce_dims:
        logging.debug(f"source pca: {datasource}")
        train_df, test_df = decompose_dims(reduced_dim_num, train_df, test_df, target_label)
    features = train_df.columns.difference([target_label])
    num_input_features = len(features)
    logging.debug(f"source domain train unique classes: {train_df.activity_label.unique()}")
    logging.debug(f"source domain test unique classes: {test_df.activity_label.unique()}")
    num_train_classes = len(train_df.activity_label.unique())
    num_train_rows = len(train_df.index)
    num_test_rows = len(test_df.index)
    logging.debug(f"SOURCE len train: {num_train_rows}")
    logging.debug(f"SOURCE len test: {num_test_rows}")
    # get train and test data
    train_dataset = SequenceDataset(
        dataframe=train_df,
        target=target_label,
        features=features,
        sequence_length=sequence_length,
        device=device
    )
    test_dataset = SequenceDataset(
        dataframe=test_df,
        target=target_label,
        features=features,
        sequence_length=sequence_length,
        device=device
    )
    return train_dataset, test_dataset, num_input_features, num_train_rows, num_test_rows, num_train_classes


def gen_target_domain_train_test_ds_from_folds(idx: int, target_label: str, folds: List, sequence_length: int,
                                               num_train_rows: int, num_test_rows: int, device: str,
                                               reduce_dims: bool, reduced_dim_num: int,
                                               datasource: str, uci_test_fold: List) -> tuple[
    SequenceDataset, SequenceDataset]:
    # load target domain files into dataframes
    logging.debug("reading target domain csv files")
    if uci_test_fold is None:
        train, test = cv.create_train_test(folds=folds, index=idx)
    else:
        train, test = cv.create_train_test_uci(train_folds=folds, test_folds=uci_test_fold, index=idx)
    random.seed(72)
    random.shuffle(train)
    random.shuffle(test)
    train_df = cv.csv2df(train)
    test_df = cv.csv2df(test)
    # do on-the-fly imputations required by datasource
    if datasource == "PAMAP2":
        logging.debug(f"target knn imputation: {datasource}")
        train_df, test_df = data_imputations.knn_impute_missing(train_df, test_df, target_label)
    if datasource == "PAMAP2" or datasource == "KU-HAR":
        logging.debug(f"target min-max normalization: {datasource}")
        train_df, test_df = data_imputations.minmax_normalization(train_df, test_df, target_label)
    # perform PCA if model type is BLSTMGauss
    # fit ONLY on train data
    # transform test and train
    if reduce_dims:
        logging.debug(f"source pca: {datasource}")
        logging.debug("performing PCA dim reduction")
        train_df, test_df = decompose_dims(reduced_dim_num, train_df, test_df, target_label)

    # here i combine train and test dataframes because for target, you can train and test on the whole fold since we don't know the labels
    concat_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    assert len(concat_df.index) == len(train_df.index) + len(test_df.index)
    train_df = concat_df.copy()
    test_df = concat_df.copy()
    # doing below so that if the target df is smaller than the source dataframe then we can continue iterating
    # as zip stops at the min row for either source or target
    if len(train_df.index) < num_train_rows:
        logging.debug("sourcedata is bigger than target dataset, so concatenating target datset to itself...")
        while len(train_df.index) < num_train_rows:
            newdf = pd.concat([train_df, train_df], axis=0, ignore_index=True)
            train_df = newdf

    features = train_df.columns.difference([target_label])
    logging.debug(f"target domain train unique classes: {train_df.activity_label.unique()}")
    logging.debug(f"target domain test unique classes: {test_df.activity_label.unique()}")
    logging.debug(f"TARGET len train: {len(train_df.index)}")
    logging.debug(f"TARGET len test: {len(test_df.index)}")
    # get train and test data
    train_dataset = SequenceDataset(
        dataframe=train_df,
        target=target_label,
        features=features,
        sequence_length=sequence_length,
        device=device
    )
    test_dataset = SequenceDataset(
        dataframe=test_df,
        target=target_label,
        features=features,
        sequence_length=sequence_length,
        device=device
    )
    return train_dataset, test_dataset
