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

class glove(BaseModel):

    def __init__(self, max_length, num_class):
        super(glove, self).__init__()

        vocab_size, embedding_matrix = self.glove_embedding()

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.linear1 = nn.Linear(max_length*embedding_matrix.shape[1], 600)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(600, num_class)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x
    
    def create_emb_layer(self, weights_matrix, non_trainable=True):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer

    def glove_embedding(self):
        x = pd.read_csv('Dataset/Complete/data_pymed_train.csv')

        docs = ct.process_samples2(x)

        # prepare tokenizer
        t = Tokenizer()
        t.fit_on_texts(docs)
        vocab_size = len(t.word_index) + 1

        embeddings_index = dict()
        f = open('data/glove.6B/glove.6B.100d.txt', mode='rt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, 100))
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return vocab_size, embedding_matrix
