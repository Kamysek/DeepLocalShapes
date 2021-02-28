#!/usr/bin/env python3
# Based on: https://github.com/facebookresearch/DeepSDF using MIT LICENSE (https://github.com/facebookresearch/DeepSDF/blob/master/LICENSE)
# Copyright 2021-present Philipp Friedrich, Josef Kamysek. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims) #10
        self.norm_layers = norm_layers # []
        self.latent_in = latent_in #[]
        self.latent_dropout = latent_dropout # false
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all # false
        self.weight_norm = weight_norm # false

        for layer in range(0, self.num_layers - 1):
            # latent_in is not used in DeepLS
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            # weight_norm is not used in DeepLS
            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))
            
            # Batch Norm is not used in DeepLS
            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        # tanh in between the layers is not used in DeepLS
        self.use_tanh = use_tanh # false
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        # Dropout not used in DeepLS (Always else)
        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            # Not used in DeepLS
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
                # Not used in DeepLS
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                # No batch/weight norm is used in DeepLS
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                # Dropout not used in DeepLS
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.th(x)

        return x
