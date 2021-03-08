import copy
import json
import logging
import os
import sys
import typing
from datetime import datetime
from datetime import timedelta

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from parastash.models_hub import ExternalModel

logger = logging.getLogger(__name__)


def _flatten(t):
    return t.reshape(t.shape[0], -1)


class MLP(nn.Module):
    """
    A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN).
    """

    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, out_features),
        )

    def forward(self, x):
        return self.network(x)


class BYOL:
    """
           BYOL: Bootstrap Your Own Latent


           The goal of BYOL is similar to contrastive learning, but with one big difference.
           BYOL does not worry about whether dissimilar samples have dissimilar representations
           (the contrastive part of contrastive learning). We only care that similar samples have
           similar representations.

           BYOL minimizes the distance between representations of each sample and a transformation of that sample.
           Examples of transformations include: translation, rotation, blurring, color inversion, color jitter,
           gaussian noise, etc.

           BYOL cocnsist from two identical encoders. Each encoder represented as a sequential container that wrap
           base net + layer that projecting base model's features into a lower-dimensional, latent space.

               1. The first is trained as usual, and its weights are updated with each training batch.
               2. The second (referred to as the “target” network) is updated using a running average of the first
                  Encoder’s weights.

           During training, the target network is provided a raw training batch, and the other Encoder is given a
           transformed version of the same batch. Each network generates a low-dimensional, latent representation
           for their respective data.

           Then, we attempt to predict the output of the target network using a multi-layer perceptron.
           BYOL maximizes the similarity between this prediction and the target network’s output.
           """

    def __init__(
            self,
            external_net: ExternalModel,
            input_shape,
            output_dimension,
            mlp_hidden_size=4096,
            freeze_base_network=True,
            learning_rate=1e-3,
            beta=0.996
    ):
        super().__init__()
        self.net_id = external_net.id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.beta = beta
        self.input_shape = input_shape
        self.mlp_hidden_size = mlp_hidden_size
        self.output_dimension = output_dimension
        self.learning_rate = learning_rate
        self._net = external_net.net
        self._net_out_dim = external_net.dims

        if freeze_base_network:
            # Each parameters of the model have requires_grad flag this loop freeze all layers see:
            # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the
            # -training/7088/2
            for x in self._net.parameters():
                x.requires_grad = False

        # Projector: a linear layer, which projects outputs down lower dimensions.
        projector = MLP(
            self._net_out_dim,
            hidden_units=mlp_hidden_size,
            out_features=output_dimension
        )
        # The first encoder
        online_network = nn.Sequential(self._net, projector)

        # The second encoder.
        target_network = copy.deepcopy(online_network)

        # Multi-layer perceptron that will learn to predict the output of the target network
        predictor = MLP(
            output_dimension,
            hidden_units=mlp_hidden_size,
            out_features=output_dimension
        )

        # freeze all parameters for target net.
        for x in target_network.parameters():
            x.requires_grad = False

        if freeze_base_network:
            parameters = list(predictor.parameters()) + list(projector.parameters())
        else:
            parameters = list(predictor.parameters()) + list(online_network.parameters())

        self._optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        self._online_network = online_network
        self._predictor = predictor
        self._target_network = target_network

        self._current_net = self._online_network
        self._scaler = amp.GradScaler()

    def train(self, loader: DataLoader, epochs, log_interval=80, return_embedding=False):
        step = 0
        total_steps = epochs * len(loader)
        pbar = progressbar(total_steps, log_iter=log_interval)
        logger.info(
            f"prepare the loop: device={self.device}, batches={len(loader)}, epochs={epochs}, steps={total_steps}")

        for net in (self._online_network, self._target_network, self._predictor):
            net.to(self.device)
            net.train()

        scaler = self._scaler
        for epoch in range(epochs):
            for (left, right), _ in loader:
                image_one = left.to(self.device)
                image_two = right.to(self.device)

                with amp.autocast():
                    loss = self._loss(image_one, image_two)

                if step % log_interval == 0:
                    pbar.print_next(loss=loss.item())

                self._optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(self._optimizer)
                scaler.update()

                self._update_target_network(step, total_steps)

                step += 1

    def _loss(self, image_one, image_two) -> torch.Tensor:
        # BYOL contains two identical Encoder networks.
        #
        # The first is trained as usual, and its weights are updated with each training batch.
        # The second (referred to as the “target” network) is updated using a running average of the
        # first Encoder’s weights.
        #
        # During training, the target network is provided a raw training batch, and the other Encoder is given
        # a transformed version of the same batch. Each network generates a low-dimensional, latent
        #  representation
        # for their respective data.
        online_proj_one = self._online_network(image_one)
        online_proj_two = self._online_network(image_two)

        # Then, we attempt to predict the output of the target network using a multi-layer perceptron.
        # BYOL maximizes the similarity between this prediction and the target network’s output.
        online_pred_one = self._predictor(online_proj_one)
        online_pred_two = self._predictor(online_proj_two)

        with torch.no_grad():
            left_target = self._target_network(image_one)
            right_target = self._target_network(image_two)

        loss_one = self._loss_fn(online_pred_one, left_target.detach())
        loss_two = self._loss_fn(online_pred_two, right_target.detach())
        return (loss_one + loss_two).mean()

    def _update_target_network(self, step, total_steps):
        for online, target in zip(self._online_network.parameters(), self._target_network.parameters()):
            # for p, pt in
            #   pt.data = self.beta * pt.data + (1 - self.beta) * p.data
            #   target.data += (1 - tau) * (online - target)
            target.data = self.beta * target + (1 - self.beta) * online

    @staticmethod
    def _loss_fn(x, y):
        x = nn.functional.normalize(x, dim=1)
        y = nn.functional.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def load(self):
        pass

    def save(self, dir, suffix=None):
        logger.info(f"saving model and manifest to {dir}")
        suffix = suffix or f"train_{datetime.isoformat(datetime.now())}"
        meta_path = os.path.join(dir, f"model.{suffix}.json")
        with open(meta_path, "w") as f:
            json.dump(self.manifest, f, indent=4)

        model_path = os.path.join(dir, f"model.{suffix}.pth")
        torch.save(self.state, model_path)

    @property
    def state(self):
        online_state_dict = self._online_network.cpu().state_dict()
        target_state_dict = self._target_network.cpu().state_dict()
        predictor_state_dict = self._predictor.cpu().state_dict()
        return {
            "optimizer": self._optimizer.state_dict(),
            "scaller": self._scaler.state_dict(),
            "epoch": 1,
            "online": online_state_dict,
            "target": target_state_dict,
            "predictor": predictor_state_dict,
        }

    @property
    def manifest(self):
        return {
            "net": self.net_id,
            "net_dim": self._net_out_dim,
            "input_shape": self.input_shape,
            "mlp_hidden_size": self.mlp_hidden_size,
            "output_dimension": self.output_dimension,
            "learning_rate": self.learning_rate,
            "beta": self.beta
        }

    def predict_from_loader(self, loader: DataLoader):
        self._current_net.to(self.device)
        self._current_net.eval()
        with torch.no_grad():
            embeddings = [
                self._current_net(inputs.to(self.device)).cpu().numpy()
                for (inputs, _) in loader
            ]
        return embeddings


# noinspection PyPep8Naming
class progressbar(object):

    def __init__(self, steps, size=40, log_iter=80):
        self.bar_size = size
        self.steps = steps
        self.step = 0
        self._last = None
        self.log_iter = log_iter

    def print_next(self, loss=None):
        j = self.step + 1
        self.step = j
        x = int(self.bar_size * j / self.steps)
        dur = None
        if self._last:
            dur = self._last - datetime.now()
        self._last = datetime.now()

        if j % self.log_iter == 0:
            logger.info("%i/%i [%s%s] - %s/s - loss=%s" % (j, self.steps, "#" * x, "." * (self.bar_size - x),
                                                           dur or "?", loss or "?"))
