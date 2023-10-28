#!/usr/bin/env python
# coding: utf-8

import logging
from datetime import datetime
import torch

import training
from interactive import print_params, choose_model
from param_classes import DataLoaderParams, TrainTestParams, BLSTMParams

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.debug(f"device: {device}")

# configuring logger
filename = datetime.now().strftime("%Y%m%d%H%M_log_file.log")
filepath = f"./logs/{filename}"
handler = logging.FileHandler(filepath, "w", "utf-8")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)  # Pass handler as a parameter, not assign
root_logger.addHandler(handler)

# key parameters
# ROOTDIR = "/Users/safr/School/Independent_Study/neural_nets" # personal
ROOTDIR = "/home/ec2-user/neural_nets" # server
SOURCE_FILEPATH = "/Users/safr/School/Independent_Study/neural_nets"
TARGET_FILEPATH = "/Users/safr/School/Independent_Study/neural_nets"
PLOTDIR = f"{ROOTDIR}/plots"
LOGDIR = f"{ROOTDIR}/logs"
RESULTSDIR = f"{ROOTDIR}/results"
ACTEXPDIR = f"{ROOTDIR}/actuals_expecteds"
IS_VALIDATION = False

SOURCE_DATASOURCE = "UCI-HAR"  # KU-HAR, UCI-HAR, PAMAP2
TARGET_DATASOURCE = "PAMAP2"  # KU-HAR, UCI-HAR, PAMAP2
NUM_SOURCE_ROWS = None # None if you don't want to do iterative parsing or limit # of source and target instances
NUM_TARGET_ROWS = None
BATCH_SIZE = 64
SEQUENCE_LENGTH = 5  # number of previous timesteps to consider (in timeseries this is the # of previous rows), i.e., seq_len-1 rows + the curr row
NUM_HIDDEN_UNITS = 128
NUM_EPOCHS = 40
LEARNING_RATE = .9

# probably remain unchanged for now
NUM_LSTM_LAYERS = 1
NUM_FOLDS = 10
BIDIRECTIONAL = True
SOURCE_LABEL = "activity_label"
TARGET_LABEL = "activity_label"

# BLSTMGauss, BLSTMCCA, BLSTMCos params
JOINT_LOSS_ALPHA = .5
# BLSTMGauss params
MMD_SIGMA = .01 # provide only one value
REDUCE_DIM = True
REDUCE_DIM_NUM = 39
# BLSTMCCA params
BLSTM_CCA_IS_KERNALIZED = True

def ascii_text():
    logging.debug("╭━━━━╮╱╱╱╱╱╱╱╱╱╱╭━╮╱╱╱╱╱╭╮")
    logging.debug("┃╭╮╭╮┃╱╱╱╱╱╱╱╱╱╱┃╭╯╱╱╱╱╱┃┃")
    logging.debug("╰╯┃┃┣┻┳━━┳━╮╭━━┳╯╰┳━━┳━╮┃┃╱╱╭━━┳━━┳━┳━╮╭┳━╮╭━━╮")
    logging.debug("╱╱┃┃┃╭┫╭╮┃╭╮┫━━╋╮╭┫┃━┫╭╯┃┃╱╭┫┃━┫╭╮┃╭┫╭╮╋┫╭╮┫╭╮┃")
    logging.debug("╱╱┃┃┃┃┃╭╮┃┃┃┣━━┃┃┃┃┃━┫┃╱┃╰━╯┃┃━┫╭╮┃┃┃┃┃┃┃┃┃┃╰╯┃")
    logging.debug("╱╱╰╯╰╯╰╯╰┻╯╰┻━━╯╰╯╰━━┻╯╱╰━━━┻━━┻╯╰┻╯╰╯╰┻┻╯╰┻━╮┃")
    logging.debug("╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╭━╯┃")
    logging.debug("╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╰━━╯")


for mod in ["2", "4", "5", "1"]:
    # Select model
    model_choice = mod  # input(f"pick a model:\n[1]BLSTMBase\n[2]BLSTMGauss\n[3]BLSTMCCA\n[4]BLSTMKCCA\n[5]BLSTMCos\n[6]BLSTMEuc\n[7]BLSTMSourceMono")

    BLSTM_MODEL = choose_model(model_choice)

    # init params
    dl_params = DataLoaderParams(SEQUENCE_LENGTH, ROOTDIR, SOURCE_LABEL, TARGET_LABEL,
                                 REDUCE_DIM, REDUCE_DIM_NUM, SOURCE_DATASOURCE, TARGET_DATASOURCE, NUM_SOURCE_ROWS,
                                 NUM_TARGET_ROWS, IS_VALIDATION)
    blstm_params = BLSTMParams(NUM_HIDDEN_UNITS, NUM_LSTM_LAYERS, BIDIRECTIONAL, MMD_SIGMA, BLSTM_CCA_IS_KERNALIZED,
                               BLSTM_MODEL)
    traintest_params = TrainTestParams(BATCH_SIZE, NUM_EPOCHS, NUM_FOLDS, LEARNING_RATE,
                                       JOINT_LOSS_ALPHA, PLOTDIR, LOGDIR, RESULTSDIR, ACTEXPDIR)
    ascii_text()
    print_params(blstm_params, traintest_params, dl_params)

    # begin experiments
    logging.debug("Beginning experiment...")
    training.train_test(dl_params, blstm_params, traintest_params, device)
    logging.debug("Experiment Complete")
