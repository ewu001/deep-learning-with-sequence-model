#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OOP wrapper for convolutional network
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    
    def __init__(self, input_embed_size, kernel_size, num_filter):
        """ Init Convolutional Neural Network object

        @param input_embed_size (int): Embedding size (dimensionality) of each character
        @param kernel_size (int): Size of the kernel to use for convolution (dimensionality)
        @param num_filter (int): How many filters to use, 
                    this directly decides the output channel dimension of convolution

        # Input: (batch_size, embedding_size, maximum_word_length)
        """
        super(CNN, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.input_embed_size = input_embed_size

        self.conv1d = nn.Conv1d(input_embed_size, num_filter, kernel_size)

    def forward(self, inputs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.

        @param inputs (Tensor): (batch_size, char embeding size, maximum word character size)
                This is the output from character embedding look up plus padding step

        @output x_conv_out (Tensor): (batch_size, embedding_size)
        """
        
        max_word_length = inputs.size(2)
        x_conv = self.conv1d(inputs)
        maxpool1d_layer = nn.MaxPool1d(max_word_length-self.kernel_size+1)
        x_conv_out = maxpool1d_layer(nn.functional.relu(x_conv)).squeeze()

        return x_conv_out



