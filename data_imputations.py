import logging
from typing import Tuple

import pandas as pd
from numpy import isnan
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


def knn_impute_missing(traindf: pd.DataFrame, testdf: pd.DataFrame, activity_label: str) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    features = list(traindf.columns.difference([activity_label]))
    # split into input and output dfs
    new_traindf = traindf[features].copy()
    y_traindf = traindf[[activity_label]].copy()
    new_testdf = testdf[features].copy()
    y_testdf = testdf[[activity_label]].copy()
    # get feature values
    train_data = new_traindf.values
    test_data = new_testdf.values
    # print total missing
    logging.debug('Missing pre-processing train: %d' % sum(isnan(train_data).flatten()))
    logging.debug('Missing pre-processing test: %d' % sum(isnan(test_data).flatten()))
    # define imputer
    imputer = KNNImputer(n_neighbors=3)
    # fit on the train dataset
    imputer.fit(train_data)
    # transform the train and test dataset
    train_data_trans = imputer.transform(train_data)
    test_data_trans = imputer.transform(test_data)
    # print total missing
    logging.debug('Missing after processing: %d' % sum(isnan(train_data_trans).flatten()))
    logging.debug('Missing after processing: %d' % sum(isnan(test_data_trans).flatten()))
    # concat datasets with the target labels
    train_concat = pd.concat([pd.DataFrame(train_data_trans, columns=features, index=None), y_traindf], axis=1,
                             ignore_index=True)
    test_concat = pd.concat([pd.DataFrame(test_data_trans, columns=features, index=None), y_testdf], axis=1,
                            ignore_index=True)
    newf = [f for f in features]
    newf.append(activity_label)
    train_concat.columns = newf
    test_concat.columns = newf
    assert len(train_concat.columns) == len(traindf.columns)
    assert len(test_concat.columns) == len(testdf.columns)
    assert len(train_concat.index) == len(traindf.index)
    assert len(test_concat.index) == len(testdf.index)
    return train_concat, test_concat


def minmax_normalization(traindf: pd.DataFrame, testdf: pd.DataFrame, activity_label: str) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    features = list(traindf.columns.difference([activity_label]))
    # split into input and output dfs
    new_traindf = traindf[features].copy()
    y_traindf = traindf[[activity_label]].copy()
    new_testdf = testdf[features].copy()
    y_testdf = testdf[[activity_label]].copy()
    # get feature values
    train_data = new_traindf.values
    test_data = new_testdf.values
    # define scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # fit on the train dataset
    scaler.fit(train_data)
    # transform the train and test dataset
    train_data_trans = scaler.transform(train_data)
    test_data_trans = scaler.transform(test_data)
    # concat datasets with the target labels
    train_concat = pd.concat([pd.DataFrame(train_data_trans, columns=features, index=None), y_traindf], axis=1,
                             ignore_index=True)
    test_concat = pd.concat([pd.DataFrame(test_data_trans, columns=features, index=None), y_testdf], axis=1,
                            ignore_index=True)
    newf = [f for f in features]
    newf.append(activity_label)
    train_concat.columns = newf
    test_concat.columns = newf
    assert len(train_concat.columns) == len(traindf.columns)
    assert len(test_concat.columns) == len(testdf.columns)
    assert len(train_concat.index) == len(traindf.index)
    assert len(test_concat.index) == len(testdf.index)
    return train_concat, test_concat
