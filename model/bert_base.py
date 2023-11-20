import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import AutoModelForSequenceClassification
from transformers import BertForSequenceClassification
import sys
import torch

class bert_base(BaseModel):

    def __init__(self, model, hidden_size, num_class):
        super(bert_base, self).__init__()

        self.model = model
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.bert = AutoModelForSequenceClassification.from_pretrained(model, num_labels=self.num_class).to("cuda")


    def forward(self, x, mask):
        output = self.bert(x, mask)

        return output['logits']