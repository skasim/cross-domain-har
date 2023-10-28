import logging
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from datetime import datetime
from typing import List, Tuple

import cross_validation as cv
import dataset_generators as data_gen
from blstm_models import BLSTMBase, BLSTMJoint
from loss_func import JointLoss
from param_classes import DataLoaderParams, BLSTMParams, TrainTestParams
from train_test_helpers import plot_charts, train_model, test_model_for_target, write_actual_and_expected_results


def get_generic_folds(rootdir: str, datasource: str, num_rows: int, num_folds: int, is_validation: bool)-> List[List]:
    domain_files = cv.get_file_list_from_dir(rootdir, datasource, num_rows, is_validation)
    domain_files = cv.randomize_files(domain_files)
    folds = cv.create_folds(domain_files, n=num_folds)
    return folds


def get_uci_folds(rootdir: str, datasource: str, num_rows: int, num_folds: int, is_validation: bool)-> Tuple[List[List], List[List]]:
    out = cv.get_file_list_from_dir_uci_har(rootdir, datasource, num_rows, is_validation)
    train_files = out["train"]
    test_files = out["test"]
    train_files = cv.randomize_files(train_files)
    test_files = cv.randomize_files(test_files)
    train_folds = cv.create_folds(train_files, n=num_folds)
    test_folds = cv.create_folds(test_files, n=num_folds)
    return train_folds, test_folds


def train_test(dl_params: DataLoaderParams, blstm_params: BLSTMParams, traintest_params: TrainTestParams,
               device: str) -> None:
    if blstm_params.model_type == "BLSTMGauss" and not dl_params.reduce_dims:
        raise ValueError("BLSTMGauss requires dimension reduction")
    if blstm_params.model_type != "BLSTMGauss" and dl_params.reduce_dims:
        logging.debug(f"Doing dim reduction for {blstm_params.model_type}. Is this what you intend?")
    if blstm_params.model_type == "BLSTMCCA":
        logging.debug(f"CCA kernalization is set to: {blstm_params.cca_is_kernalized}")

    logging.debug(f"model type: {blstm_params.model_type}")
    random.seed(72)
    torch.manual_seed(101)
    loss_alpha = traintest_params.joint_loss_alpha
    if dl_params.source_datasource in ["KU-HAR", "PAMAP2"]:
        source_folds = get_generic_folds(dl_params.rootdir, dl_params.source_datasource, dl_params.num_source_rows,
                                         traintest_params.num_folds, dl_params.is_validation)
        uci_source_test_fold = None
    elif dl_params.source_datasource == "UCI-HAR":
        source_folds, uci_source_test_fold = get_uci_folds(dl_params.rootdir, dl_params.source_datasource,
                                                           dl_params.num_source_rows, traintest_params.num_folds,
                                                           dl_params.is_validation)
    else:
        raise ValueError(f"Unknown source datasource provided: {dl_params.source_datasource}")

    if dl_params.target_datasource in ["KU-HAR", "PAMAP2"]:
        target_folds = get_generic_folds(dl_params.rootdir, dl_params.target_datasource, dl_params.num_target_rows,
                                         traintest_params.num_folds, dl_params.is_validation)
        uci_target_test_fold = None
    elif dl_params.target_datasource == "UCI-HAR":
        target_folds, uci_target_test_fold = get_uci_folds(dl_params.rootdir, dl_params.target_datasource,
                                                           dl_params.num_target_rows, traintest_params.num_folds,
                                                           dl_params.is_validation)
    else:
        raise ValueError(f"Unknown target datasource provided: {dl_params.target_datasource}")
    test_results_by_fold = []
    for i, fold in enumerate(source_folds):
        source_train_dataset, source_test_dataset, num_source_domain_features, num_source_train_rows, \
        num_source_test_rows, num_source_domain_classes = data_gen.gen_source_domain_test_train_ds_from_folds(
            idx=i,
            target_label=dl_params.source_domain_activity_label,
            folds=source_folds,
            sequence_length=dl_params.sequence_length,
            device=device,
            reduce_dims=dl_params.reduce_dims,
            reduced_dim_num=dl_params.reduced_dim_num,
            datasource=dl_params.source_datasource,
            uci_test_fold=uci_source_test_fold
        )
        target_train_dataset, target_test_dataset = data_gen.gen_target_domain_train_test_ds_from_folds(
            idx=i,
            target_label=dl_params.source_domain_activity_label,
            folds=target_folds,
            sequence_length=dl_params.sequence_length,
            num_train_rows=num_source_train_rows,
            num_test_rows=num_source_test_rows,
            device=device,
            reduce_dims=dl_params.reduce_dims,
            reduced_dim_num=dl_params.reduced_dim_num,
            datasource=dl_params.target_datasource,
            uci_test_fold=uci_target_test_fold
        )
        # load data into loaders
        logging.debug("loading source and data")
        source_train_loader = DataLoader(source_train_dataset, batch_size=traintest_params.batch_size, shuffle=False)
        # source_test_loader = DataLoader(source_test_dataset, batch_size=traintest_params.batch_size, shuffle=False)
        target_train_loader = DataLoader(target_train_dataset, batch_size=traintest_params.batch_size, shuffle=False)
        target_test_loader = DataLoader(target_test_dataset, batch_size=traintest_params.batch_size, shuffle=False)

        # instantiate BLSTM model
        if blstm_params.model_type == "BLSTMBase":
            # init model and loss function
            blstm_model = BLSTMBase(input_size=num_source_domain_features, num_classes=num_source_domain_classes,
                                    blstm_params=blstm_params, device=device)
            loss_function = nn.NLLLoss(reduction="mean")
        else:
            blstm_model = BLSTMJoint(input_size=num_source_domain_features, num_classes=num_source_domain_classes,
                                     blstm_params=blstm_params, device=device)
            loss_function = JointLoss()

        # define optimizer
        optimizer = torch.optim.SGD(blstm_model.parameters(), lr=traintest_params.learning_rate, momentum=0.9)
        # define scheduler
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        results = []
        logging.debug(f"\nFOLD: {i}\n==========")
        logging.debug(f"{blstm_model} => {dl_params.source_datasource} v {dl_params.target_datasource}")

        for ix_epoch in range(traintest_params.num_epochs):
            logging.debug(f"\nEpoch {ix_epoch}\n---------")
            train_results = train_model(source_train_loader, target_train_loader, blstm_model, loss_function,
                                        optimizer, loss_alpha, blstm_params, device)

            # gather metrics only for training by epoch
            results.append([i, ix_epoch, train_results.avg_loss, train_results.f1, train_results.acc])
            logging.debug(f"learning rate: {scheduler.get_last_lr()}")
            scheduler.step()
        if dl_params.is_validation:
            # plot training result only during validation to save some time
            results_df = pd.DataFrame(results, columns=["fold", "epoch", "train_loss", "train_f1_score", "train_acc"], index=None)
            # plot charts for every fold
            plot_charts(results_df, traintest_params, i, blstm_params.model_type, dl_params)
        # test once after training for a fold
        test_target_results, actuals, expecteds = test_model_for_target(target_test_loader, blstm_model, blstm_params.model_type)
        write_actual_and_expected_results(actuals, expecteds, path=f"{traintest_params.actexpdir}/{blstm_params.model_type}_{dl_params.source_datasource}v{dl_params.target_datasource}_fold{i}_{str(datetime.now()).strip()}.csv")
        test_results_by_fold.append([i, test_target_results.f1, test_target_results.acc])
    test_results_df = pd.DataFrame(test_results_by_fold, columns=["fold",  "test_target_f1", "test_target_acc"], index=None)
    test_results_df.to_csv(
        f"{traintest_params.resultsdir}/TEST_{blstm_params.model_type}_{dl_params.source_datasource}v{dl_params.target_datasource}_results_{str(datetime.now()).strip()}.csv",
        index=False)
