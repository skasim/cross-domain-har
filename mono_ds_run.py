import logging
import torch

from interactive import print_params
from param_classes import DataLoaderParams, TrainTestParams, BLSTMParams
import mono_methods

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.debug(f"device: {device}")

def ascii_text_mm():
    logging.debug("╭━╮╭━┳━━━┳━╮╱╭┳━━━╮╭━╮╭━┳━━━┳━━━┳━━━┳╮")
    logging.debug("┃┃╰╯┃┃╭━╮┃┃╰╮┃┃╭━╮┃┃┃╰╯┃┃╭━╮┣╮╭╮┃╭━━┫┃")
    logging.debug("┃╭╮╭╮┃┃╱┃┃╭╮╰╯┃┃╱┃┃┃╭╮╭╮┃┃╱┃┃┃┃┃┃╰━━┫┃")
    logging.debug("┃┃┃┃┃┃┃╱┃┃┃╰╮┃┃┃╱┃┃┃┃┃┃┃┃┃╱┃┃┃┃┃┃╭━━┫┃╱╭╮")
    logging.debug("┃┃┃┃┃┃╰━╯┃┃╱┃┃┃╰━╯┃┃┃┃┃┃┃╰━╯┣╯╰╯┃╰━━┫╰━╯┃")
    logging.debug("╰╯╰╯╰┻━━━┻╯╱╰━┻━━━╯╰╯╰╯╰┻━━━┻━━━┻━━━┻━━━╯")


# key parameters
ROOTDIR = "/Users/safr/School/Independent_Study/neural_nets" # personal
# ROOTDIR = "/home/ec2-user/neural_nets" # server
#ROOTDIR = "/Users/safr/Desktop/independent_study/neural_nets" # zahra
SOURCE_FILEPATH = "/Users/safr/School/Independent_Study/neural_nets"
PLOTDIR = f"{ROOTDIR}/plots"
LOGDIR = f"{ROOTDIR}/logs"
RESULTSDIR = f"{ROOTDIR}/results"
ACTEXPDIR = f"{ROOTDIR}/actuals_expecteds"
IS_VALIDATION = False

SOURCE_DATASOURCE = "UCI-HAR"  # KU-HAR, UCI-HAR, PAMAP2
NUM_SOURCE_ROWS = None # None if you don't want to do iterative parsing or limit # of source and target instances
BATCH_SIZE = 64
SEQUENCE_LENGTH = 5  # number of previous timesteps to consider (in timeseries this is the # of previous rows), i.e., seq_len-1 rows + the curr row
NUM_HIDDEN_UNITS = 128
NUM_EPOCHS = 3
LEARNING_RATE = .9

# probably remain unchanged for now
NUM_LSTM_LAYERS = 1
NUM_FOLDS = 10
BIDIRECTIONAL = True
SOURCE_LABEL = "activity_label"

REDUCE_DIM = False
REDUCE_DIM_NUM = 39


# init params
dl_params = DataLoaderParams(SEQUENCE_LENGTH, ROOTDIR, SOURCE_LABEL, None,
                             REDUCE_DIM, REDUCE_DIM_NUM, SOURCE_DATASOURCE, None, NUM_SOURCE_ROWS,
                             None, IS_VALIDATION)
blstm_params = BLSTMParams(NUM_HIDDEN_UNITS, NUM_LSTM_LAYERS, BIDIRECTIONAL, None, None,
                           "BLSTMSourceMono")
traintest_params = TrainTestParams(BATCH_SIZE, NUM_EPOCHS, NUM_FOLDS, LEARNING_RATE,
                                   None, PLOTDIR, LOGDIR, RESULTSDIR, ACTEXPDIR)
print_params(blstm_params, traintest_params, dl_params)

# begin experiments
ascii_text_mm()
logging.debug("Beginning mono ds experiment...")
mono_methods.mono_dataset_train_test(dl_params, blstm_params, traintest_params, device)
logging.debug("Mono ds Experiment Complete")