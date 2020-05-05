#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OOP wrapper implementation for highway network
"""

import torch
import torch.nn as nn

class Highway(nn.Module):

    def __init__(self, embedding_size, dropout_prob):
        """ Init Highway network object

        @param input_embed_size (int): Embedding size (dimensionality) at word level

        @param dropout_prob (float): dropout rate
                    this is needed at the last step of forward

        # Input: (batch_size, embedding_size)

        """
        super(Highway, self).__init__()
        self.embed_size = embedding_size
        self.dropout_prob = dropout_prob
        self.proj_conv_weight = nn.Parameter(torch.zeros(self.embed_size, self.embed_size))
        self.proj_bias = nn.Parameter(torch.zeros(self.embed_size))
        self.gate_conv_weight = nn.Parameter(torch.zeros(self.embed_size, self.embed_size))
        self.gate_bias = nn.Parameter(torch.zeros(self.embed_size))
        self.dropout = nn.Dropout(self.dropout_prob)


    def forward(self, inputs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.

        @param inputs (Tensor): (batch_size, embeding size)
                This is the output from the convolution network's forward step

        # Output: (batch_size, embedding_size)
        """
        x_proj_linear = torch.add(torch.matmul(inputs, self.proj_conv_weight), self.proj_bias)
        x_proj = nn.functional.relu(x_proj_linear)

        x_gate_linear = torch.add(torch.matmul(inputs, self.gate_conv_weight), self.gate_bias)
        x_gate = nn.functional.sigmoid(x_gate_linear)

        x_highway = torch.add( x_gate*x_proj, (1 - x_gate) * inputs)
        output = self.dropout(x_highway)
        return output