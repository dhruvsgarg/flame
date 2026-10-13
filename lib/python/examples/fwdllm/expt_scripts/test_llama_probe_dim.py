"""`p` for llama checkpoints, read off `probe_port_init.py` ([ProbeDim]); the 360M row is the default."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from examples.fwdllm.expts.dataset_registry import probe_dim  # noqa: E402


def test_smollm2_360m_is_default():
    assert probe_dim(4, 8, "llama") == 823040
    assert probe_dim(4, 8, "llama", "HuggingFaceTB/SmolLM2-360M") == 823040


def test_smollm2_1_7b_lora_r4():
    assert probe_dim(4, 4, "llama", "HuggingFaceTB/SmolLM2-1.7B") == 794624
    assert probe_dim(4, 8, "llama", "HuggingFaceTB/SmolLM2-1.7B") == 1581056
