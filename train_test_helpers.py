import logging
from datetime import datetime
from typing import Tuple, Union, List, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F


from blstm_models import BLSTMBase, BLSTMJoint
from loss_func import JointLoss
from metric_calcs import calc_accuracy, calc_f1
from param_classes import TrainTestParams, BLSTMParams, DataLoaderParams


class Metrics:
    def __init__(self):
        self.loss = 0
        self.acc = 0
        self.f1 = 0
        self.yhats = []
        self.ys = []
        self.avg_loss = 0.0

    def update_metrics(self, yhat, y, latest_loss):
        self.loss += latest_loss
        self.acc += calc_accuracy(yhat, y.long())
        self.f1 += calc_f1(yhat, y.long())
        if type(yhat) == torch.Tensor:
            self.yhats.extend(yhat.detach().cpu().numpy())
        else:
            self.yhats.extend(yhat.detach().numpy())
        if type(y) == torch.Tensor:
            self.ys.extend(y.detach().cpu().numpy())
        else:
            self.ys.extend(y.detach().numpy())

    def calc_epoch_metrics(self, num_batches):
        self.acc = calc_accuracy(self.yhats, self.ys)
        self.f1 = calc_f1(self.yhats, self.ys)
        self.avg_loss = self.loss/num_batches

    def print_metrics(self, type: str)-> None:
        yhats = self.yhats
        ys = self.ys
        correct = sum([1 for yhat, y in zip(yhats, ys) if yhat == y])
        logging.debug(f"\n{type} METRICS")
        logging.debug(f"correct instances: {correct}")
        logging.debug(f"total instances: {len(ys)}")
        logging.debug(f"total loss: {self.loss}")
        logging.debug(f"average loss: {self.avg_loss}")
        logging.debug(f"f1 score: {round(self.f1, 6)}")
        logging.debug(f"classification accuracy: {round(self.acc, 6) * 100}%")


def plot_charts(df: pd.DataFrame, traintest_params: TrainTestParams, fold_num: int, model_type: str,
                dl_params: DataLoaderParams) -> None:
    # create loss and f1 dataframes from results dataframe
    losses_pltdf = df[["epoch", "train_loss"]].copy()
    f1s_pltdf = df[["epoch", "train_f1_score"]].copy()
    # plot test train crossentropy loss
    plt.figure()
    lossdfm = losses_pltdf.melt("epoch", var_name="type", value_name="loss")
    sns.lineplot(x="epoch", y="loss", hue="type", data=lossdfm, palette="muted")
    plt.savefig(f"{traintest_params.plotdir}/losses/{model_type}_{dl_params.source_datasource}v{dl_params.target_datasource}_{fold_num}_loss_{str(datetime.now()).strip()}.png")
    # plot test train f1s
    plt.figure()
    f1dfm = f1s_pltdf.melt("epoch", var_name="type", value_name="f1_score")
    sns.lineplot(x="epoch", y="f1_score", hue="type", data=f1dfm, palette="muted")
    plt.savefig(f"{traintest_params.plotdir}/f1_scores/{model_type}_{dl_params.source_datasource}v{dl_params.target_datasource}_{fold_num}_f1_{str(datetime.now()).strip()}.png")


def get_equal_row_matrices(X: Tensor, y: Tensor, target_X: Tensor, target_y: Tensor) -> Tuple[
    Tensor, Tensor, Tensor, Tensor]:
    min_row = min(X.shape[0], target_X.shape[0])
    X = X[:min_row]
    target_X = target_X[:min_row]
    y = y[:min_row]
    target_y = target_y[:min_row]
    return X, y, target_X, target_y


def train_base_model(source_domain_data_loader: DataLoader, model: BLSTMBase, loss_function: nn.NLLLoss,
                     optimizer: SGD, model_type: str, device: str) -> Metrics:
    metrics = Metrics()
    model = model.to(device)
    model.train()  # tells model that i am training
    for source_domain in source_domain_data_loader:
        X, y = source_domain
        X = X.to(device)
        optimizer.zero_grad()
        yhat_probs = model(X)
        _, yhat = torch.max(yhat_probs, dim=1)
        assert yhat.shape == y.shape
        loss = loss_function(yhat_probs, y.long())
        metrics.update_metrics(yhat=yhat, y=y, latest_loss=loss.item())
        loss.backward()
        optimizer.step()
    metrics.calc_epoch_metrics(num_batches=len(source_domain_data_loader))
    metrics.print_metrics(f"TRAIN_{model_type}")
    return metrics


def train_joint_model(source_domain_data_loader: DataLoader, target_domain_data_loader, model: BLSTMJoint,
                         loss_function: JointLoss, optimizer: SGD, alpha: float, model_name: str, device: str) -> Metrics:
    loss_types = {
        "BLSTMGauss": "mmd",
        "BLSTMCCA": "cca",
        "BLSTMKCCA": "cca",
        "BLSTMCos": "cos",
        "BLSTMEuc": "euc"
    }
    metrics = Metrics()
    model = model.to(device)
    model.train()  # tells model that i am training
    for source_domain, target_domain in zip(source_domain_data_loader,
                                            target_domain_data_loader):  # TODO: data loaders are iterables not indexed. So if one loader is less than other, with zip it will only iter the lenght of the smaller len
        X, y = source_domain
        X = X.to(device)
        target_X, target_y = target_domain
        target_X = target_X.to(device)
        if X.shape != target_X.shape:
            # if one dataset has more rows than the other dataset, then pick the min. number of rows
            X, y, target_X, target_y = get_equal_row_matrices(X, y, target_X, target_y)
        assert X.shape == target_X.shape
        assert y.shape == target_y.shape
        optimizer.zero_grad()
        loss_type = loss_types[model_name]
        model_output = model(X, target_X, loss_type)
        yhat_probs = model_output["source_out"]
        custom_loss = model_output["custom_loss"]
        _, yhat = torch.max(yhat_probs, dim=1)
        assert yhat.shape == y.shape
        loss = loss_function(yhat_probs, y.long(), custom_loss, alpha)
        metrics.update_metrics(yhat=yhat, y=y, latest_loss=loss.item())
        loss.backward()
        optimizer.step()
    metrics.calc_epoch_metrics(num_batches=len(source_domain_data_loader))
    metrics.print_metrics(f"TRAIN_{model_name}")
    return metrics


def train_model(source_domain_data_loader: DataLoader, target_domain_data_loader: DataLoader,
                model: Union[BLSTMBase, BLSTMJoint], loss_function: Union[nn.NLLLoss, JointLoss],
                optimizer: SGD, joint_loss_alpha, blstm_params: BLSTMParams, device: str) -> Metrics:
    if blstm_params.model_type == "BLSTMBase" or blstm_params.model_type == "BLSTMSourceMono":
        return train_base_model(source_domain_data_loader, model, loss_function, optimizer, blstm_params.model_type, device)
    else:
        return train_joint_model(source_domain_data_loader, target_domain_data_loader, model, loss_function,
                                    optimizer, joint_loss_alpha, blstm_params.model_type, device)


def test_model_for_target(domain_data_loader: DataLoader, model: Union[BLSTMBase, BLSTMJoint], model_type: str)-> tuple[
    Metrics, list[Any], list[Any]]:
    metrics = Metrics()
    loss_types = {
        "BLSTMGauss": "mmd",
        "BLSTMCCA": "cca",
        "BLSTMKCCA": "cca",
        "BLSTMCos": "cos",
        "BLSTMEuc": "euc"
    }
    model.eval()
    actuals = [] # yhat
    expecteds = [] # y
    with torch.no_grad():
        for domain in domain_data_loader:
            X, y = domain
            if model_type == "BLSTMBase":
                yhat_probs = model(X)
                loss = F.nll_loss(yhat_probs, y.long())
                _, yhat = torch.max(yhat_probs, dim=1)
            else:
                model_output = model(None, X, None)
                yhat_probs = model_output["target_out"]
                loss = F.nll_loss(yhat_probs, y.long())
                _, yhat = torch.max(yhat_probs, dim=1)
            metrics.update_metrics(yhat=yhat, y=y, latest_loss=loss.item())
            actuals.extend(yhat.detach().cpu().numpy())
            expecteds.extend(y.detach().cpu().numpy())
    metrics.calc_epoch_metrics(num_batches=len(domain_data_loader))
    metrics.print_metrics(f"TEST_{model_type}")
    return metrics, actuals, expecteds


def write_actual_and_expected_results(actuals: List, expecteds: List, path: str)->None:
    actdf = pd.DataFrame(actuals, columns=["actuals"])
    expdf = pd.DataFrame(expecteds, columns=["expecteds"])
    concat = pd.concat([actdf, expdf], ignore_index=True, axis=1)
    out = pd.DataFrame(concat.values, columns=["actuals", "expecteds"])
    out.to_csv(path, index=False)


# TODO: may be deprecated so might need to be removed
def test_model_for_training(source_domain_data_loader: DataLoader, target_domain_data_loader, model: Union[BLSTMBase, BLSTMJoint],
               loss_function: Union[nn.NLLLoss, JointLoss], joint_loss_alpha: float, model_type: str) -> Metrics:
    loss_types = {
        "BLSTMGauss": "mmd",
        "BLSTMCCA": "cca",
        "BLSTMKCCA": "cca",
        "BLSTMCos": "cos",
        "BLSTMEuc": "euc"
    }

    target_metrics = Metrics()
    model.eval()  # tells model that i am testing
    with torch.no_grad():
        for source_domain, target_domain in zip(source_domain_data_loader, target_domain_data_loader):
            X, y = source_domain
            target_X, target_y = target_domain
            if X.shape != target_X.shape:
                # if one dataset has more rows than the other dataset, then pick the min. number of rows
                X, y, target_X, target_y = get_equal_row_matrices(X, y, target_X, target_y)
            assert y.shape == target_y.shape
            if model_type == "BLSTMBase":
                yhat_probs_target = model(target_X)
                loss_target = loss_function(yhat_probs_target, target_y.long())
            else:
                loss_type = loss_types[model_type]
                model_output = model(X, target_X, loss_type)
                yhat_probs_target = model_output["target_out"]
                custom_loss = model_output["custom_loss"]
                if joint_loss_alpha is None or custom_loss is None:
                    raise ValueError(
                        f"custom loss [{custom_loss}] and/or joint loss alpha [{joint_loss_alpha}] not provided")
                loss_target = loss_function(yhat_probs_target, target_y.long(), custom_loss, joint_loss_alpha)
            _, yhat_target = torch.max(yhat_probs_target, dim=1)
            assert yhat_target.shape == target_y.shape
            # test on target data
            target_metrics.update_metrics(yhat=yhat_target, y=target_y, latest_loss=loss_target.item())
        target_metrics.calc_epoch_metrics(num_batches=len(target_domain_data_loader))
        target_metrics.print_metrics(f"TEST_TARGET_{model_type}")
        return target_metrics


