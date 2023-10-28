import pandas as pd
from sklearn.decomposition import PCA
from typing import Tuple


def concatdf(df1: pd.DataFrame, df2: pd.DataFrame)-> pd.DataFrame:
    assert len(df1.index) == len(df2.index)
    lst = []
    for v1, v2, in zip(df1.values, df2.values):
        row = []
        row.extend(v1)
        row.extend(v2)
        lst.append(row)
    assert len(lst) == len(df1.index) == len(df2.index)
    return pd.DataFrame(lst, index=None)


def decompose_dims(num_components: int, train_df: pd.DataFrame, test_df: pd.DataFrame, target_label: str)-> Tuple[pd.DataFrame, pd.DataFrame]:
    pca = PCA(n_components=num_components)
    features_train_df = train_df[train_df.columns.difference([target_label])].copy()
    target_train_df = train_df[[target_label]].copy()
    pca.fit(features_train_df)
    features_test_df = test_df[test_df.columns.difference([target_label])].copy()
    target_test_df = test_df[[target_label]].copy()
    transform_train = pca.transform(features_train_df)
    transform_test = pca.transform(features_test_df)
    concat_train = concatdf(pd.DataFrame(transform_train, index=None), target_train_df)
    concat_test = concatdf(pd.DataFrame(transform_test), target_test_df)
    targetcol_idx = len(concat_train.columns) - 1
    concat_train.rename(columns={targetcol_idx: f"{target_label}"}, inplace=True)
    concat_test.rename(columns={targetcol_idx: f"{target_label}"}, inplace=True)
    assert (len(concat_train.index)) == len(train_df.index)
    assert (len(concat_test.index)) == len(test_df.index)
    assert (len(concat_train.columns)) == num_components + 1
    assert (len(concat_test.columns)) == num_components + 1
    return concat_train, concat_test
