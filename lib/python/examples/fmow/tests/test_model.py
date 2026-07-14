import pytest
import torch

from fmow.dependencies.model import build_model

def test_build_model_shape():
    model = build_model()
    model.eval()

    dummy_input = torch.randn(2,3,224,224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2,62)

def test_frozen_layers():
    model = build_model()

    assert not model.features.conv0.weight.requires_grad
    assert model.classifier.weight.requires_grad