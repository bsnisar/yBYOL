import io
import logging

import torch
from torch import nn
from torchvision.models import resnet101, resnet50

logger = logging.getLogger(__name__)


def _load_instagram_resnet():
    output_dim = 2048
    resnet = torch.hub.load(
        "facebookresearch/WSL-Images", "resnext101_32x8d_wsl", progress=False,
    )
    model = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
    return model, output_dim


_MODELS = {
    "instagram": _load_instagram_resnet,
}


class ExternalModel:

    def __init__(self, name) -> None:
        super().__init__()
        net, out_dims = find(name)
        self._net = net
        self._id = name
        self._out_dims = out_dims

    @property
    def net(self):
        return self._net

    @property
    def dims(self):
        return self._out_dims

    @property
    def id(self):
        return self._id


def find(name):
    return _MODELS[name]()
