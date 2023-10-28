import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from similarity_measures import cca_similarity_loss, MMD, cosine_similarity, euclidean_distance
from param_classes import BLSTMParams


# Example: https://www.crosstab.io/articles/time-series-pytorch-lstm/


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length, device):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values, device=device).float()
        self.X = torch.tensor(dataframe[self.features].values, device=device).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]


def concat_hidden_layer_output(x: torch.Tensor, direction: int, num_layers: int, hidden_size: int,
                               model: nn.LSTM, device: str) -> torch.Tensor:
    batch_size = x.shape[0]
    h_0 = torch.zeros((direction * num_layers), batch_size, hidden_size)
    x = x.to(device)
    h_0 = h_0.to(device)
    c_0 = torch.zeros((direction * num_layers, batch_size, hidden_size))
    c_0 = c_0.to(device)
    _, (h_n, _) = model(x, (h_0, c_0))  # lstm returns: output, (h_n, c_n)
    assert (h_n.shape[0], h_n.shape[1], h_n.shape[2]) == (direction * num_layers, batch_size, hidden_size)
    h_n_reshaped = h_n.view(num_layers, direction, batch_size, hidden_size)
    assert (h_n_reshaped.shape[0], h_n_reshaped.shape[1], h_n_reshaped.shape[2], h_n_reshaped.shape[3]) \
           == (num_layers, direction, batch_size, hidden_size)
    h_n_fwd = h_n_reshaped[:, 0, :, :]
    h_n_bwd = h_n_reshaped[:, 1, :, :]
    assert (h_n_fwd.shape[0], h_n_fwd.shape[1], h_n_fwd.shape[2]) \
           == (h_n_bwd.shape[0], h_n_bwd.shape[1], h_n_bwd.shape[2]) \
           == (num_layers, batch_size, hidden_size)
    concat_hn = torch.cat((h_n_fwd, h_n_bwd), dim=2)  # concat on the third dimension, i.e., hidden_size
    assert (concat_hn.shape[0], concat_hn.shape[1], concat_hn.shape[2]) \
           == (num_layers, batch_size, hidden_size * 2)
    return concat_hn


class BLSTMBase(nn.Module):
    def __init__(self, input_size: int, num_classes: int, blstm_params: BLSTMParams, device: str):
        super().__init__()
        self.blstm_params = blstm_params
        self.device = device
        self.input_size = input_size
        self.hidden_size = blstm_params.num_hidden_units
        self.num_layers = blstm_params.num_lstm_layers
        self.direction = 2 if blstm_params.bidirectional else 1
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=blstm_params.num_hidden_units,
                            num_layers=blstm_params.num_lstm_layers,
                            batch_first=True,
                            bidirectional=blstm_params.bidirectional)
        self.fc1 = nn.Linear(in_features=self.hidden_size * 2 if blstm_params.bidirectional else self.hidden_size,
                             out_features=self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concat_hn = concat_hidden_layer_output(x, self.direction, self.num_layers, self.hidden_size, self.lstm, self.device)
        out = self.fc1(concat_hn[-1])  # -1 to get the last concatenated layer logits from h_n
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)  # outputs log probability values
        return out


class BLSTMJoint(nn.Module):
    def __init__(self, input_size: int, num_classes: int, blstm_params: BLSTMParams, device: str):
        super().__init__()
        self.blstm_params = blstm_params
        self.device = device
        self.input_size = input_size
        self.hidden_size = blstm_params.num_hidden_units
        self.num_layers = blstm_params.num_lstm_layers
        self.direction = 2 if blstm_params.bidirectional else 1
        self.num_classes = num_classes
        self.lstm_source = nn.LSTM(input_size=input_size,
                                   hidden_size=blstm_params.num_hidden_units,
                                   num_layers=blstm_params.num_lstm_layers,
                                   batch_first=True,
                                   bidirectional=blstm_params.bidirectional)
        self.fc1_source = nn.Linear(
            in_features=self.hidden_size * 2 if blstm_params.bidirectional else self.hidden_size,
            out_features=self.hidden_size)
        self.relu_source = nn.ReLU()
        self.fc2_source = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)
        self.lstm_target = nn.LSTM(input_size=input_size,
                                   hidden_size=blstm_params.num_hidden_units,
                                   num_layers=blstm_params.num_lstm_layers,
                                   batch_first=True,
                                   bidirectional=blstm_params.bidirectional)
        self.fc1_target = nn.Linear(
            in_features=self.hidden_size * 2 if blstm_params.bidirectional else self.hidden_size,
            out_features=self.hidden_size)
        self.relu_target = nn.ReLU()
        self.fc2_target = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)
        self.softmax_source = nn.LogSoftmax(dim=1)
        self.softmax_target = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, x_target: torch.Tensor, loss_type:str) -> Dict:
        if x is None:
            target_concat_hn = concat_hidden_layer_output(x_target, self.direction, self.num_layers, self.hidden_size,
                                                          self.lstm_target, self.device)

            # target NN
            target_out = self.fc1_target(target_concat_hn[-1])  # -1 to get the last concatenated layer logits from h_n
            target_out = self.relu_target(target_out)
            target_out = self.fc2_target(target_out)
            target_out = self.softmax_target(target_out)

            return {
                "target_out": target_out
            }
        else:
            source_concat_hn = concat_hidden_layer_output(x, self.direction, self.num_layers, self.hidden_size,
                                                          self.lstm_source, self.device)
            target_concat_hn = concat_hidden_layer_output(x_target, self.direction, self.num_layers, self.hidden_size,
                                                          self.lstm_target, self.device)
            if loss_type == "mmd":
                custom_loss = MMD(source_concat_hn[-1], target_concat_hn[-1], sigma=self.blstm_params.mmd_sigmas,
                          device=self.device)
            elif loss_type == "cca":
                custom_loss = cca_similarity_loss(source_concat_hn[-1], target_concat_hn[-1],
                                          is_kcca=self.blstm_params.cca_is_kernalized)
            elif loss_type == "cos":
                custom_loss = cosine_similarity(source_concat_hn[-1], target_concat_hn[-1])
            elif loss_type == "euc":
                custom_loss = euclidean_distance(source_concat_hn[-1], target_concat_hn[-1])
            else:
                raise ValueError(f"incorrect loss type given: {loss_type}")

            # source NN
            source_out = self.fc1_source(source_concat_hn[-1])  # -1 to get the last concatenated layer logits from h_n
            source_out = self.relu_source(source_out)
            source_out = self.fc2_source(source_out)
            source_out = self.softmax_source(source_out)

            # target NN
            target_out = self.fc1_target(target_concat_hn[-1])  # -1 to get the last concatenated layer logits from h_n
            target_out = self.relu_target(target_out)
            target_out = self.fc2_target(target_out)
            target_out = self.softmax_target(target_out)

            return {
                "source_out": source_out,
                "target_out": target_out,
                "custom_loss": custom_loss
            }


