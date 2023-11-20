import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from data_loader import CustomTokenize as ct
import torch
from model.glove import glove as gl

class LSTM_glove(BaseModel):

    def __init__(self, max_length, num_class):
        super(LSTM_glove, self).__init__()

        vocab_size, embedding_matrix = gl.glove_embedding(self)

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm1 = nn.LSTM(max_length*embedding_matrix.shape[1], 600)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(600, num_class)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, start_dim=1)
        lstm_out, hidden_states = self.lstm1(x)
        # final hidden state
        x = self.linear1(lstm_out)
        return x
