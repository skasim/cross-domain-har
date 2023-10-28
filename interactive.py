import logging
from param_classes import BLSTMParams, TrainTestParams, DataLoaderParams

def basic_params(blstm_params: BLSTMParams, tt_params: TrainTestParams, dl_params: DataLoaderParams):
    logging.debug("BASIC PARAMS")
    logging.debug(f"BATCH_SIZE: {tt_params.batch_size}")
    logging.debug(f"SEQUENCE_LENGTH: {dl_params.sequence_length}")
    logging.debug(f"NUM_LSTM_LAYERS: {blstm_params.num_lstm_layers}")
    logging.debug(f"NUM_HIDDEN_UNITS: {blstm_params.num_hidden_units}")
    logging.debug(f"BIDIRECTIONAL: {blstm_params.bidirectional}")
    logging.debug(f"NUM_FOLDS: {tt_params.num_folds}")
    logging.debug(f"LEARNING_RATE: {tt_params.learning_rate}")
    logging.debug(f"NUM_EPOCHS: {tt_params.num_epochs}")
    logging.debug(f"SOURCE_LABEL: {dl_params.source_domain_activity_label}")
    logging.debug(f"TARGET_LABEL: {dl_params.source_domain_activity_label}")
    logging.debug(f"SOURCE_DATASOURCE: {dl_params.source_datasource}")
    logging.debug(f"TARGET_DATASOURCE: {dl_params.target_datasource}")


def blstmgauss_params(blstm_params: BLSTMParams, tt_params: TrainTestParams, dl_params: DataLoaderParams):
    logging.debug("BLSTMGauss PARAMS")
    logging.debug(f"JOINT_LOSS_ALPHA: {tt_params.joint_loss_alpha}")
    logging.debug(f"MMD_SIGMAS: {blstm_params.mmd_sigmas}")
    logging.debug(f"REDUCE_DIM: {dl_params.reduce_dims}")
    logging.debug(f"REDUCE_DIM_NUM: {dl_params.reduced_dim_num}")
    logging.debug(f"NUM_SOURCE_ROWS: {dl_params.num_source_rows}")


def blstmcca_params(blstm_params: BLSTMParams):
    logging.debug("BLSTMCCA PARAMS")
    logging.debug(f"BLSTM_CCA_IS_KERNALIZED: {blstm_params.cca_is_kernalized}")


# pick model
def choose_model(model_choice: str):
    if model_choice == "1":
        return "BLSTMBase"
    elif model_choice == "2":
        return "BLSTMGauss"
    elif model_choice == "3":
        return "BLSTMCCA"
    elif model_choice == "4":
        return "BLSTMKCCA"
    elif model_choice == "5":
        return "BLSTMCos"
    elif model_choice == "6":
        return "BLSTMEuc"
    elif model_choice == "7":
        return "BLSTMSourceMono"
    else:
        raise ValueError(f"menu choice entered incorrectly: {model_choice}")


def print_params(blstm_params: BLSTMParams, tt_params: TrainTestParams, dl_params: DataLoaderParams):
    basic_params(blstm_params, tt_params, dl_params)
    if blstm_params.model_type == "BLSTMGauss" or blstm_params.model_type == "BLSTMCos" or blstm_params.model_type == "BLSTMEuc" or blstm_params.model_type == "BLSTMSourceMono":
        blstmgauss_params(blstm_params, tt_params, dl_params)
    if blstm_params.model_type == "BLSTMCCA" or blstm_params.model_type == "BLSTMKCCA":
        blstmcca_params(blstm_params)