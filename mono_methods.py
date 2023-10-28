from typing import Tuple, List, Any

import pandas as pd
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from datetime import datetime

from blstm_models import BLSTMBase
from param_classes import BLSTMParams, DataLoaderParams, TrainTestParams
import logging
import random
import torch
import torch.nn.functional as F

from train_test_helpers import plot_charts, Metrics, train_base_model, write_actual_and_expected_results
from training import get_generic_folds, get_uci_folds
import dataset_generators as data_gen


def test_mono_model(domain_data_loader: DataLoader, model: BLSTMBase, data_name: str)-> tuple[
    Metrics, list[Any], list[Any]]:
    metrics = Metrics()
    model.eval()
    actuals = [] # yhat
    expecteds = [] # y
    with torch.no_grad():
        for domain in domain_data_loader:
            X, y = domain
            yhat_probs = model(X)
            loss = F.nll_loss(yhat_probs, y.long())
            _, yhat = torch.max(yhat_probs, dim=1)
            metrics.update_metrics(yhat=yhat, y=y, latest_loss=loss.item())
            if type(actuals) == torch.Tensor:
                actuals.extend(yhat.detach().cpu().numpy())
            else:
                actuals.extend(yhat.detach().numpy())
            if type(y) == torch.Tensor:
                expecteds.extend(y.detach().cpu().numpy())
            else:
                expecteds.extend(y.detach().numpy())
    metrics.calc_epoch_metrics(num_batches=len(domain_data_loader))
    metrics.print_metrics(f"MonoTEST_{data_name}")
    return metrics, actuals, expecteds


def mono_dataset_train_test(dl_params: DataLoaderParams, blstm_params: BLSTMParams, traintest_params: TrainTestParams, device: str):
    logging.debug(f"mono dataset run")
    random.seed(72)
    torch.manual_seed(101)

    if dl_params.source_datasource in ["KU-HAR", "PAMAP2"]:
        source_folds = get_generic_folds(dl_params.rootdir, dl_params.source_datasource, dl_params.num_source_rows,
                                         traintest_params.num_folds, dl_params.is_validation)
        uci_source_test_fold = None
    elif dl_params.source_datasource == "UCI-HAR":
        source_folds, uci_source_test_fold = get_uci_folds(dl_params.rootdir, dl_params.source_datasource,
                                                           dl_params.num_source_rows, traintest_params.num_folds,
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
        logging.debug("loading source data in mono ds")
        source_train_loader = DataLoader(source_train_dataset, batch_size=traintest_params.batch_size, shuffle=False)
        source_test_loader = DataLoader(source_test_dataset, batch_size=traintest_params.batch_size, shuffle=False)

        blstm_model = BLSTMBase(input_size=num_source_domain_features, num_classes=num_source_domain_classes,
                                blstm_params=blstm_params, device=device)
        loss_function = nn.NLLLoss(reduction="mean")

        # define optimizer
        optimizer = torch.optim.SGD(blstm_model.parameters(), lr=traintest_params.learning_rate, momentum=0.9)
        # define scheduler
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        results = []
        logging.debug(f"\nFOLD: {i}\n==========")
        logging.debug(f"{blstm_model} => {dl_params.source_datasource} v {dl_params.target_datasource}")

        for ix_epoch in range(traintest_params.num_epochs):
            logging.debug(f"\nEpoch {ix_epoch}\n---------")
            train_results = train_base_model(source_train_loader, blstm_model, loss_function, optimizer,
                                             dl_params.source_datasource, device)

            # gather metrics only for training by epoch
            results.append([i, ix_epoch, train_results.avg_loss, train_results.f1, train_results.acc])
            logging.debug(f"learning rate: {scheduler.get_last_lr()}")
            scheduler.step()
        if dl_params.is_validation:
            # plot training result only during validation to save some time
            results_df = pd.DataFrame(results, columns=["fold", "epoch", "train_loss", "train_f1_score", "train_acc"],
                                      index=None)
            # plot charts for every fold
            plot_charts(results_df, traintest_params, i, blstm_params.model_type, dl_params)
        # test once after training for a fold
        test_target_results, actuals, expecteds = test_mono_model(source_test_loader, blstm_model, dl_params.source_datasource)
        write_actual_and_expected_results(actuals, expecteds,
                                          path=f"{traintest_params.actexpdir}/{blstm_params.model_type}_{dl_params.source_datasource}v{dl_params.target_datasource}_fold{i}_{str(datetime.now()).strip()}.csv")
        test_results_by_fold.append([i, test_target_results.f1, test_target_results.acc])
    test_results_df = pd.DataFrame(test_results_by_fold, columns=["fold", "test_target_f1", "test_target_acc"],
                                   index=None)
    test_results_df.to_csv(
        f"{traintest_params.resultsdir}/TEST_{blstm_params.model_type}_{dl_params.source_datasource}_mono_results_fold{i}_{str(datetime.now()).strip()}.csv",
        index=False)