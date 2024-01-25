from model.bow import bow
from model.glove import glove
from model.LSTM_glove import LSTM_glove
from model.bert_base import bert_base
from model.bert_gnn import bert_gnn
from model.bert_gnn_PT import bert_gnn_PT

class bowModel(bow):
    def __init__(self, hidden_size, num_class):
        super().__init__(hidden_size, num_class)

class gloveModel(glove):
    def __init__(self, max_length, num_class):
        super().__init__(max_length, num_class)

class LSTM_gloveModel(LSTM_glove):
    def __init__(self, max_length, num_class):
        super().__init__(max_length, num_class)

class bert_baseModel(bert_base):
    def __init__(self, model, hidden_size, num_class):
        super().__init__(model, hidden_size, num_class)

class bert_gnnModel(bert_gnn):
    def __init__(self, model, hidden_size, num_class, max_length=512):
        super().__init__(model, hidden_size, num_class, max_length=512)

class bert_gnn_PTModel(bert_gnn_PT):
    def __init__(self, model, hidden_size, num_class):
        super().__init__(model, hidden_size, num_class)