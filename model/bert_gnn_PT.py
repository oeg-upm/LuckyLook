import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
from dgl.nn.pytorch import GATv2Conv
from transformers import AutoModel, AutoTokenizer
from base import BaseModel




class bert_gnn_PT(BaseModel):

    def __init__(self, model, hidden_size, num_class, max_length=512, device='cuda'):
        super(bert_gnn_PT, self).__init__()
        self.top_rate = 0.05
        self.model = model
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.device = device

        # Initialize the BERT model with specified configuration
        self.bert = AutoModel.from_pretrained(
            self.model,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
            max_length=max_length
        ).to(self.device)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, do_lower_case=True)

        # Initialize the GNN Module
        self.attentionGNN = AttentionModule(self.top_rate, device=device).to(device)

        # Initialize the Gating modules for lexical and semantic features
        self.word_gate = GateModule(384, device=device).to(device)
        self.semantic_gate = GateModule(384, device=device).to(device)
        
        # Define the fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(1152, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, num_class),
        ).to(device)

        # Further transformation layer for BERT pooler output
        self.bert_trans = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
        ).to(device)

    def forward(self, x, mask):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input data.
            mask (Tensor): Attention mask.

        Returns:
            Tensor: Model output.
        """
        # Pass input through the BERT model
        output = self.bert(x, mask)
        bert_out = output['pooler_output']

        # Apply a linear transformation to the BERT output
        bert_out = self.bert_trans(bert_out)

        # Apply the GNN Module to obtain word and semantic features
        word_output, semantic_output = self.attentionGNN(output, x, mask)

        # Apply gating mechanisms to filter GNN outputs
        word_gnn_output = self.word_gate(bert_out, word_output)
        semantic_gnn_output = self.semantic_gate(bert_out, semantic_output)
        
        # Concatenate BERT and GNN outputs
        gnn_out = torch.cat((word_gnn_output, semantic_gnn_output), dim=1)
        gnn_out = torch.cat((bert_out, gnn_out), dim=1)

        # Pass through the fully connected layers for classification
        out = self.fc(gnn_out)

        return out
    
class AttentionModule(BaseModel):
    """
    Description of the AttentionModule class.

    Args:
        top_rate (float): The top rate percentage used for selecting the top percentage of relations between nodes to use.
    """

    def __init__(self, top_rate):
        super(AttentionModule, self).__init__()
        self.word_gnn = AttentionGNNModule(top_rate).to(self.device)
        self.semantic_gnn = AttentionGNNModule(top_rate).to(self.device)
        self.hs_word_trans = nn.Linear(768, 768).to(self.device)
        self.hs_semantic_trans = nn.Linear(768, 768).to(self.device)

    def forward(self, output, encoded_inputs, mask):
        """
        Forward pass of the AttentionModule.

        Args:
            output (dict): Output from BERT model.
            encoded_inputs (Tensor): Encoded input BERT model data.
            mask (Tensor): Attention mask.

        Returns:
            Tensor: Lexical representation output.
            Tensor: Semantic representation output.
        """
        # Extract word-level attention and hidden states from the first 3 layers (Lexical Representation)
        word_attention = torch.stack(output['attentions'][:3], dim=4).max(dim=4)[0].mean(dim=1)
        word_hidden_state = self.hs_word_trans(torch.stack(output['hidden_states'][:4], dim=3).transpose(-2, -1)
                                        ).transpose(-2, -1).max(dim=3)[0]
        #word_hidden_state = torch.stack(output['hidden_states'][:3], dim=3).max(dim=3)[0]
        
        # Extract semantic-level attention and hidden states from the last 3 layers except final output layer (Semantic Representation)
        semantic_attention = torch.stack(output['attentions'][9:12], dim=4).max(dim=4)[0].mean(dim=1)
        semantic_hidden_state = self.hs_semantic_trans(torch.stack(output['hidden_states'][9:12], dim=3).transpose(-2, -1)
                                                ).transpose(-2, -1).max(dim=3)[0]
        #semantic_hidden_state = torch.stack(output['hidden_states'][9:12], dim=3).max(dim=3)[0]

        # Apply GNN modules to obtain word and semantic representations
        word_output = self.word_gnn(word_hidden_state, word_attention, encoded_inputs, 'word', mask)
        semantic_output = self.semantic_gnn(semantic_hidden_state, semantic_attention, encoded_inputs, 'semantic', mask)

        return word_output, semantic_output

class AttentionGNNModule(nn.Module):
    """
    Description of the AttentionGNNModule class.

    Args:
        top_rate (float): The top rate percentage used for selecting the top percentage of relations between nodes to use.
        device (str): Device for computation.
    """

    def __init__(self, top_rate, device):
        super(AttentionGNNModule, self).__init__()
        self.device = device
        self.top_rate = top_rate

        # GATv2 convolution layer with LeakyReLU activation and residual connection
        self.conv1 = GATv2Conv(768, 768, num_heads=1, activation=nn.LeakyReLU(), residual=True).to(device)
        #self.conv2 = GATv2Conv(768,768, num_heads=1, activation=nn.LeakyReLU(), residual=True).to(self.device)
        
        # Global Attention Pooling
        self.gate_nn = nn.Linear(768, 1).to(device)
        self.gap = dgl.nn.GlobalAttentionPooling(self.gate_nn).to(device)
        
        # Dropout and normalization layers
        self.dropout = nn.Dropout(0.1).to(device)
        self.activation = nn.ReLU().to(device)
        self.ln = nn.LayerNorm(768).to(device)
        self.fc = nn.Linear(768, 384).to(device)
        self.ln2 = nn.LayerNorm(384).to(device)

    def forward(self, hidden_state, attention, encoded_inputs, type, mask):
        """
        Forward pass of the AttentionGNNModule.

        Args:
            hidden_state (Tensor): Hidden state of the input.
            attention (Tensor): Attention scores.
            encoded_inputs (Tensor): Encoded input data.
            type (str): Type of representation (e.g., 'word', 'semantic').
            mask (Tensor): Attention mask.

        Returns:
            Tensor: Output representation.
        """
        batch_size = hidden_state.size(0)
        length = hidden_state.size(1)

        # Select top values and indices based on attention scores
        top_result = torch.topk(attention, round(self.top_rate * length), dim=-1)
        top_values = top_result.values
        top_indices = top_result.indices

        # Convert selected values and indices to sub-graphs
        sub_graphs = [self.seq_to_graph(top_values[i], top_indices[i], hidden_state[i], mask[i]) for i in
                      range(batch_size)]

        batch_graph = dgl.batch(sub_graphs).to(self.device)

        # Apply GATv2 convolution layer with LeakyReLU activation
        result_node_embedding = self.conv1(batch_graph, batch_graph.ndata['h'])
        result_node_embedding = torch.flatten(result_node_embedding, start_dim=1)

        # Update Embedding
        batch_graph.ndata['h'] = result_node_embedding

        ## Apply 2 GATv2 convolution layer with LeakyReLU activation
        #result_node_embedding = self.conv2(batch_graph, batch_graph.ndata['h'])
        #result_node_embedding = torch.flatten(result_node_embedding, start_dim=1)

        ##Update Embedding
        #batch_graph.ndata['h'] = result_node_embedding

        # Apply Graph Attention Pooling
        out, node_attention = self.gap(batch_graph, batch_graph.ndata['h'], get_attention=True)

         # Apply dropout, activation, normalization, and linear layers
        out = self.fc(self.dropout(self.ln(self.activation(out))))
        out = self.ln2(out)

        return out

    def seq_to_graph(self, topk_value, topk_indice, hidden_state):
        """
        Converts a sequence to a graph based on values and indices.

        Args:
            values (Tensor): Values for sub-graph selection.
            indices (Tensor): Indices for sub-graph selection.
            hidden_state (Tensor): Hidden state of the input.

        Returns:
            DGLGraph: The constructed DGL graph.
        """
        # Filter out values and indices based on the attention mask excluding the padding tokens
        topk_values = topk_value[:]
        topk_indices = topk_indice[:]
        num_edges = topk_values.size(-1)
        length = topk_values.size(0)

        # Create target nodes for the edges
        target_nodes = torch.tensor([[i] * num_edges for i in range(length)]).to(self.device)

        # Apply softmax to mask_edge_value while handling values less than or equal to 0
        mask_edge_value = F.softmax(torch.where(topk_values > 0.0, topk_values,
                                            torch.tensor(-1e9, dtype=torch.float).to(self.device)), dim=-1)

        # Find positive sign positions in mask_edge_value
        pos_sign = torch.nonzero(torch.flatten(mask_edge_value), as_tuple=True)[0].to(self.device)

        # Extract source and target nodes based on positive sign positions
        source_nodes = torch.flatten(topk_indices)[pos_sign].to(self.device)
        target_nodes = torch.flatten(target_nodes)[pos_sign].to(self.device)
        edge_tuple = (source_nodes, target_nodes)
        edge_weight = torch.flatten(mask_edge_value)[pos_sign]
        
        # Create a DGL sub-graph
        sub_graph = dgl.graph(edge_tuple).to(self.device)
        sub_graph.ndata['h'] = hidden_state[:].to(self.device)
        sub_graph.ndata['index'] = torch.tensor(range(length)).to(self.device)
       
        sub_graph.edata['w'] = edge_weight.to(self.device)

        return sub_graph

class GateModule(nn.Module):
    """
    Description of the GateModule class.

    Args:
        dim_model (int): Dimensionality of the input models.
        device (str): Device for computation.
    """

    def __init__(self, dim_model, device):
        super(GateModule, self).__init__()
        
        # Linear transformations for BERT and GNN inputs
        self.bert_trans = nn.Linear(dim_model, dim_model).to(device)
        self.gnn_trans = nn.Linear(dim_model, dim_model).to(device)
        
        # Sigmoid activation for gating
        self.activation = nn.Sigmoid()

    def forward(self, bert, gnn):
        """
        Forward pass of the GateModule.

        Args:
            bert (Tensor): BERT input.
            gnn (Tensor): GNN input.

        Returns:
            Tensor: Gated output.
        """
        alpha = self.activation(self.bert_trans(bert) + self.gnn_trans(gnn))
        return alpha * gnn