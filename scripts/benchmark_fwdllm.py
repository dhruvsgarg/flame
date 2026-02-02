import os
import sys
import time
import logging
import argparse
import torch
import numpy as np
import gc
from datetime import datetime
from torch.nn import CrossEntropyLoss
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix

# Add workspace roots to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "lib/python")))

from examples.fwdllm.data_preprocessing.text_classification_preprocessor import TLMPreprocessor
from examples.fwdllm.trainer.model.transformer.model_args import ClassificationArgs
from examples.fwdllm.data_manager.text_classification_data_manager import TextClassificationDataManager
from examples.fwdllm.data_manager.base_data_manager import BaseDataManager
from examples.fwdllm.expts.initializer import set_seed, create_model
from flame.config import Config

logger = logging.getLogger(__name__)

# Reconstructing the timer decorator as requested
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        # Adding cuda synchronize before timing if cuda is used
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        duration = end_time - start_time
        # Format consistent with historical logs: [GJD] In timer_decorator wrapper
        logger.info(f"[GJD] In timer_decorator wrapper - {func.__name__} took {duration:.4f}s")
        return result
    return wrapper

class MockAggregator:
    def __init__(self, model, test_data_global, num_labels, device, args):
        self.model = model
        self.test_global = test_data_global
        self.num_labels = num_labels
        self.device = device
        self.args = args # Should contain eval_batch_size
    
    def log_memory(self, msg, device):
        if device.type == 'cuda':
            logger.debug(f"{msg} - CUDA Memory Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    @timer_decorator
    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        logger.info(f"device inside eval_model() is set to: {device}")
        self.log_memory("start eval_model", self.device)

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_global)
        test_sample_len = len(self.test_global.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        logger.info(
            f"Created n_batches: {n_batches}, test_sample_len: {test_sample_len} and preds.shape: {preds.shape}, location of model: {next(self.model.parameters()).device}"
        )

        out_label_ids = np.empty(test_sample_len)
        # Move model to device before performing the eval
        self.model.to(device)
        self.model.eval()
        
        # Original code uses functional call: self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)
        try:
            import functorch as fc
            self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)
        except ImportError:
            try:
                import torch.func as fc
                self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)
            except ImportError:
                logger.warning("functorch or torch.func not found, skipping make_functional_with_buffers")

        # Metering for granular analysis
        data_movement_times = []
        forward_pass_times = []

        for i, batch in enumerate(self.test_global):
            with torch.no_grad():
                # Metering: Data Movement Start
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.time()
                
                batch = tuple(t for t in batch)
                x = batch[1].to(device)
                labels = batch[4].to(device)
                
                # Metering: Data Movement End
                if device.type == 'cuda': torch.cuda.synchronize()
                t1 = time.time()
                data_movement_times.append(t1 - t0)

                # Metering: Forward Pass Start
                output = self.model(x)
                logits = output[0]
                
                # Metering: Forward Pass End
                if device.type == 'cuda': torch.cuda.synchronize()
                t2 = time.time()
                forward_pass_times.append(t2 - t1)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()

            nb_eval_steps += 1
            # Note: start_index calculation in original code: start_index = self.args.eval_batch_size * i
            # Using eval_batch_size from args as it might vary
            current_batch_size = x.size(0)
            start_index = self.args.eval_batch_size * i

            end_index = (
                start_index + current_batch_size
                if i != (n_batches - 1)
                else test_sample_len
            )
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(
            preds, out_label_ids, self.test_global.examples
        )
        result["eval_loss"] = eval_loss
        results.update(result)

        # Log granular metrics
        logger.info(f"Granular Performance (Avg per batch): Data Movement: {np.mean(data_movement_times)*1000:.2f}ms, Forward Pass: {np.mean(forward_pass_times)*1000:.2f}ms")

        # self.results.update(result)
        logging.info(f"results after eval are: {results}, len(wrong) is: {len(wrong)}")

        # TODO: Check if model needs to be moved back to cpu? Do we need to keep
        # moving the model between CPU and GPU repeatedly?
        del x, labels, output, logits, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        self.log_memory("end eval_model", self.device)

        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)
        self.log_memory("start compute_metrics", self.device)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        # mismatched = labels != preds

        # if eval_examples:
        #     wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        # else:
        #     wrong = ["NA"]
        wrong = [] # Simplified for benchmark

        mcc = matthews_corrcoef(labels, preds)

        # confusion_matrix logic
        try:
             tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        except ValueError:
             tn, fp, fn, tp = 0, 0, 0, 0

        self.log_memory("end compute_metrics", self.device)

        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--batch_sizes", type=str, default="8,256", help="Comma separated batch sizes to test")
    parser.add_argument("--test_cut_off", type=int, default=7600, help="Number of samples from test set")
    args = parser.parse_args()

    # Log setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(funcName)s(): %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    config = Config(args.config)
    set_seed(config.hyperparameters.manual_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load attributes to get num_labels
    attributes = BaseDataManager.load_attributes(config.hyperparameters.data_file_path)
    num_labels = len(attributes["label_vocab"])

    # Create original model and tokenizer
    model_args = ClassificationArgs()
    model_args.model_name = config.hyperparameters.model_name
    model_args.model_type = config.hyperparameters.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.update_from_dict(vars(config.hyperparameters))
    model_args.config["num_labels"] = num_labels
    
    # Ensure eval_batch_size is set (will be overridden in loop)
    model_args.eval_batch_size = 8

    model_config, client_model, tokenizer = create_model(
        model_args, formulation="classification"
    )

    # Data management
    preprocessor = TLMPreprocessor(
        args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer
    )
    
    batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",")]
    
    for bs in batch_sizes:
        logger.info(f"\n{'='*20} Testing Batch Size: {bs} {'='*20}")
        model_args.eval_batch_size = bs
        
        dm = TextClassificationDataManager(
            config.hyperparameters,
            model_args,
            preprocessor,
            0,
            1
        )
        # Use load_federated_data(process_id=0) for server-side global test data
        (
            _, _, test_data_global,
            _, _, _, _
        ) = dm.load_federated_data(process_id=0, test_cut_off=args.test_cut_off)

        logger.info(f"Dataset size: {len(test_data_global.dataset)}")
        
        agg = MockAggregator(client_model, test_data_global, num_labels, device, model_args)
        
        # Warmup
        if device.type == 'cuda':
            logger.info("Warming up...")
            agg.eval_model() 
        
        logger.info(f"Running timed eval for BS={bs}...")
        agg.eval_model()

if __name__ == "__main__":
    main()
