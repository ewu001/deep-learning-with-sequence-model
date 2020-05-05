#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_embeddings.py: Embeddings for the NMT model
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object.

        self.vocab.char2id is useful when create the embedding

        #Input tensor: (batch_size*sentence_length, max_word_length)
        """
        super(ModelEmbeddings, self).__init__()
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        # Hardcode a vew parameters
        self.char_embed_size = 50
        self.dropout_rate = 0.3
        self.kernel_size = 5
        
        src_char_pad_token_id = vocab.char_pad
        self.source_embedding = nn.Embedding(len(self.vocab.char2id), self.char_embed_size, padding_idx=src_char_pad_token_id)

        self.cnn = CNN(self.char_embed_size, self.kernel_size, word_embed_size)
        self.highway = Highway(self.word_embed_size, self.dropout_rate)
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """

        # Unroll parameters size from input tensor
        batch_size = input.size(1)
        sentence_length = input.size(0)
        max_word_length = input.size(2)

        input_reshape = input.contiguous().view(batch_size*sentence_length, max_word_length)
        source_embed_output = self.source_embedding(input_reshape)
        # Here we must permute because dimension of embed_size is added at the end by nn.embedding object
        # However, our CNN definition requires this dimension to be in the middle of its input
        cnn_output = self.cnn.forward(source_embed_output.permute(0, 2, 1))
        highway_output = self.highway(cnn_output)
        dropout_word_embed = self.dropout_layer(highway_output)
        # Reshape to required output dimension
        word_embed = dropout_word_embed.contiguous().view(sentence_length, batch_size, self.word_embed_size)

        return word_embed



