# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

import torch.nn as nn
import torch

from . import layers

default_initializer = layers.default_init


class tabularUnet(nn.Module):
    def __init__(
        self,
        input_size,
        cond_in,
        embed_dim=64,
        encoder_dim=[64, 128, 256],
    ):
        super().__init__()

        self.embed_dim = embed_dim  # FLAGS.nf
        tdim = self.embed_dim * 4
        self.act = nn.ReLU()
        self.is_cond = True
        self.d_in = input_size

        modules = []
        modules.append(nn.Linear(self.embed_dim, tdim))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(tdim, tdim))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        cond_in = cond_in
        cond_out = (input_size) // 2
        if cond_out < 2:
            cond_out = input_size
        modules.append(nn.Linear(cond_in, cond_out))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        self.all_modules = nn.ModuleList(modules)

        dim_in = input_size + cond_out
        dim_out = list(encoder_dim)[0]
        self.inputs = nn.Linear(dim_in, dim_out)  # input layer

        self.encoder = layers.Encoder(encoder_dim, tdim)  # encoder

        dim_in = encoder_dim[-1]
        dim_out = encoder_dim[-1]
        self.bottom_block = nn.Linear(dim_in, dim_out)  # bottom_layer

        self.decoder = layers.Decoder(list(reversed(encoder_dim)), tdim)  # decoder

        dim_in = encoder_dim[0]
        dim_out = input_size
        self.outputs = nn.Linear(dim_in, dim_out)  # output layer

    def forward(self, x, time_cond, z_cond):
        modules = self.all_modules
        m_idx = 0

        # time embedding
        temb = layers.get_timestep_embedding(time_cond, self.embed_dim)
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = self.act(temb)
        temb = modules[m_idx](temb)
        m_idx += 1

        # condition layer
        cond = modules[m_idx](z_cond)
        m_idx += 1

        x = torch.cat([x, cond], dim=1).float()
        inputs = self.inputs(x)  # input layer
        skip_connections, encoding = self.encoder(inputs, temb)
        encoding = self.bottom_block(encoding)
        encoding = self.act(encoding)
        x = self.decoder(skip_connections, encoding, temb)
        outputs = self.outputs(x)

        return outputs
