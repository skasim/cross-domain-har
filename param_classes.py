

class BLSTMParams:
    def __init__(self, num_hidden_units: int, num_lstm_layers: int, bidirectional: bool, mmd_sigmas: float,
                 cca_is_kernalized: bool, model_type: str):
        self.num_hidden_units = num_hidden_units
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional
        self.mmd_sigmas = mmd_sigmas
        self.cca_is_kernalized = cca_is_kernalized
        self.model_type = model_type


class TrainTestParams:
    def __init__(self, batch_size: int, num_epochs: int, num_folds: int, learning_rate: float, joint_loss_alpha: float,
                 plotdir: str, logdir: str, resultsdir: str, actexpdir: str):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_folds = num_folds
        self.learning_rate = learning_rate
        self.joint_loss_alpha = joint_loss_alpha
        self.plotdir = plotdir
        self.logdir = logdir
        self.resultsdir = resultsdir
        self.actexpdir = actexpdir


class DataLoaderParams:
    def __init__(self, sequence_length: int, rootdir: str,
                 source_domain_activity_label: str, target_domain_activity_label: str, reduce_dims: bool,
                 reduced_dim_num: int, source_datasource: str, target_datasource: str, num_source_rows: int,
                 num_target_rows: int, is_validation: bool):
        self.sequence_length = sequence_length
        self.rootdir = rootdir
        self.source_domain_activity_label = source_domain_activity_label
        self.target_domain_activity_label = target_domain_activity_label
        self.reduce_dims = reduce_dims
        self.reduced_dim_num = reduced_dim_num
        self.source_datasource = source_datasource
        self.target_datasource = target_datasource
        self.num_source_rows = num_source_rows
        self.num_target_rows = num_target_rows
        self.is_validation = is_validation

