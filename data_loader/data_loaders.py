# from torchvision import datasets, transforms
from base import BaseDataLoader
import pandas as pd
from torch.utils.data import Dataset
from keras.utils import to_categorical
from data_loader import CustomTokenize as ct
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from transformers import AutoTokenizer
import torch
import itertools
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
import sys


class CustomPaperDataset(Dataset):
    """
    Custom dataset for papers
    """
    def __init__(self, path, label_encoder, type, num_classes, tokenizer, model=None, max_length = 512, tokenize = False, content=None):
        # Load data
        data = pd.read_csv(path)
        # Custom Tokenization
        if tokenize:
            docs = ct.process_samples(data)
        # Different input data to use Customized HuggingFace Tokenizer
        elif content:
            docs = ct.process_samples3(data, content)
        # All input data to use Customized HuggingFace Tokenizer
        else:
            docs = ct.process_samples2(data)

        # Label encoding
        labels = label_encoder.transform(data['journal'])
        labels = to_categorical(labels, num_classes=num_classes)

        # Tokenizer
        self.tokenizer = tokenizer
        if type == 'bert-base':
            self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=True)
        
        self.type = type
        self.max_length = max_length
        
        self.papers = docs
        self.papers_labels = labels

    def __len__(self):
        return len(self.papers_labels)
    
    # Return the paper embedding and its label
    def __getitem__(self, index):
        paper = self.embedding(self.type, self.papers[index])
        label = torch.tensor(self.papers_labels[index])
        return paper, label
    
    # Return the embedding of the paper
    def embedding(self, type, doc):
        if type == 'binary':
            return torch.tensor(self.binary_bow(doc))
        if type == 'glove':
            text = self.tokenizer.texts_to_sequences([doc])
            padded_text = pad_sequences(text, maxlen=self.max_length, padding='post')
            text = list(padded_text[0])
            return torch.LongTensor(text)
        if type == 'bert-base':
            encoding = self.tokenizer(doc,truncation=True,padding="max_length",max_length=self.max_length)
            return {k: torch.tensor(v) for k, v in encoding.items()}

    # Return the binary bag of words of the paper
    def binary_bow(self,x):
        return self.tokenizer.texts_to_matrix([x], mode='binary').astype('float32')[0]

class Papers(BaseDataLoader):
    """
    Papers data loading
    """
    def __init__(self, data_dir, dir, batch_size, type, model=None, max_length=512, num_classes = 469, shuffle=True,
                 validation_split=0.0, num_workers=1, tokenize = False, content=None):
        # Label encoding train set
        x = pd.read_csv(dir)
        self.label_encoder = LabelEncoder()
        self.label_encoder = self.label_encoder.fit(x['journal'])
        self.max_length = max_length

        # Embedding train set
        if tokenize:
            docs = ct.process_samples(x)
        else:
            docs = ct.process_samples2(x)

        tokenizer = self.create_tokenizer(docs)
        del docs
        del x

        self.dataset = CustomPaperDataset(data_dir, self.label_encoder, type, num_classes, tokenizer,
                                          model=model, max_length=self.max_length, tokenize=tokenize, content=content)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def create_tokenizer(self, lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
