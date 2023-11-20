from model.bow import bow
from model.glove import glove
from model.LSTM_glove import LSTM_glove
from model.bert_base import bert_base
from model.bert_gnn_final import bert_gnn_final
from model.bert_gnn_final_ST import bert_gnn_final_ST

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

class bert_gnn_finalModel(bert_gnn_final):
    def __init__(self, model, hidden_size, num_class):
        super().__init__(model, hidden_size, num_class)

class bert_gnn_finalModel_ST(bert_gnn_final_ST):
    def __init__(self, model, hidden_size, num_class):
        super().__init__(model, hidden_size, num_class)