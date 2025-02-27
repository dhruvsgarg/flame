import logging
from flame.mode.horizontal.syncfl.fwdllm_trainer import Trainer
import torch
import time
import io

logger = logging.getLogger(__name__)


class FedSGDTrainer(Trainer):

    def __init__(
        self,
        trainer_id,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
        config=None,
        client_index=None,
    ):
        self.trainer = model_trainer
        self.trainer_id = trainer_id
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        # NRL most of this is reduntant since our dict is of size=1. But keeping this code for consistency
        logger.info(
            f"train_data_local_dict keys: {train_data_local_dict.keys()}, client idx: {args.client_idx}, {type(args.client_idx)}"
        )

        self.train_local = [self.train_data_local_dict[args.client_idx]]

        # this will return -1
        self.test_local = self.test_data_local_dict[args.client_idx]

        self.train_local_list = [
            [data for data in self.train_local[i]] for i in range(len(self.train_local))
        ]
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num

        self.local_sample_number = None
        # logger.info(f"self.device: {self.device}, torch.cuda.device_count(): {torch.cuda.device_count()}")

        # self.train_data_local_dict.to(self.device)

        self.args = args
        self.accumulated_error = None
        self.config = config
        self.device = device

        # abstract attributes
        self.loss_fn = None
        self.dataset_size = None
        self.model = model_trainer.model
        # NRL adding new variables
        self.data_id = None
        self.total_data_bins = None

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        logger.info(f"self.device: {self.device}")
        self.total_data_bins = len(self.train_local[0])
        # loading data to gpu
        # NRL TODO: This didnt work. Error: expected all tensors to be on the same device. Needed to load them on gpu again during train_model
        for each_train_local in self.train_local[0]:
            train_data = tuple(t for t in each_train_local)
            # logger.info(f"train data: {len(train_data)}")
            train_data[1].to(self.device)
            train_data[4].to(self.device)
        logger.info(
            f"Task_id: {self.trainer_id} initialize completed at timestamp: "
            f"{time.time()}"
        )

    def update_model(self, weights):
        # logger.info(f"NRL: Updated model weights: {weights}")
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        logger.info(f"NRL: Updated client index: {client_index}")
        self.client_index = client_index
        self.train_local = [self.train_data_local_dict[id] for id in client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index[0]]

        self.test_local = self.test_data_local_dict[client_index[0]]

        self.train_local_list = [
            [data for data in self.train_local[i]] for i in range(len(self.train_local))
        ]

    def train(self, round_idx=None):
        logger.info("entered train where weights = params and not grad")
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    def train_with_data_id(self):
        logger.info(
            f"starting training for trainer id: {self.trainer_id}, data_id = {self.data_id}"
        )
        logger.info(
            f"train_local_list[0][0]: {len(self.train_local_list[0][0])}, {len(self.train_local_list)}"
        )
        self.trainer.train(
            [self.train_local_list[0][self.data_id]], self.device, self.args
        )

        logger.info(
            f"completed training for trainer id: {self.trainer_id}, data_id = {self.data_id}"
        )

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = (
            test_metrics["test_correct"],
            test_metrics["test_total"],
            test_metrics["test_loss"],
        )

        return (
            train_tot_correct,
            train_loss,
            train_num_sample,
            test_tot_correct,
            test_loss,
            test_num_sample,
        )

    def load_data(self) -> None:
        pass

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def check_and_sleep(self) -> None:
        pass
