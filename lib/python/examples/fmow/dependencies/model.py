import torch.nn as nn
from torchvision.models import DenseNet161_Weights, densenet161

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_GROUPS = 32

def find_num_groups(num_channels: int, max_groups: int) -> int:
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0: 
            return num_groups
    return 1

def use_group_normalization(module: nn.Module, max_groups: int) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = find_num_groups(num_channels, max_groups)
            setattr(module, name, nn.GroupNorm(num_groups, num_channels))
        else:
            use_group_normalization(child, max_groups)

def freeze_layers(model: nn.Module) -> None:
    # Layers used by FedSpace
    trainable_layers = ("features.denseblock4", "features.norm5", "classifier")
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(trainable_layers)

def build_model(num_classes: int=62) -> nn.Module:
    model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    use_group_normalization(model, MAX_GROUPS)
    freeze_layers(model)

    return model