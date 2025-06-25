from typing import Dict, List, Optional
import torch

class AggregatorToTrainerDTO:
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        grad_pool: Optional[List[torch.Tensor]],
        round: int,
        model_version: int,
        task_to_perform: str,
        data_id: int,
        iteration_per_data_id: int,
    ):
        self.weights = weights
        self.grad_pool = grad_pool
        self.round = round
        self.model_version = model_version          # usually same as round
        self.task_to_perform = task_to_perform      # train or eval
        self.data_id = data_id
        self.iteration_per_data_id = iteration_per_data_id

class TrainerToAggregatorDTO:
    def __init__(
        self,
        gradients: Dict[str, torch.Tensor],
        gradients_for_var_check: List[torch.Tensor],
        dataset_size: int,
        model_version: int,
        datasampler_metadata: Any,
        total_data_bins: int,
        stat_utility: float,
    ):
        self.gradients = gradients
        self.gradients_for_var_check = gradients_for_var_check
        self.dataset_size = dataset_size
        self.model_version = model_version
        self.datasampler_metadata = datasampler_metadata
        self.total_data_bins = total_data_bins
        self.stat_utility = stat_utility            # for OORT, only when task_to_perform == eval