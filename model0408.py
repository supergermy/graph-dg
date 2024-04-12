import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import Linear, ReLU, Dropout, Sequential, ModuleList
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GATConv, LayerNorm
from torch_geometric.data import Batch

class CrossAttention(nn.Module):
    def __init__(self, embed_size=256):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.value = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.key = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.query = nn.Linear(self.embed_size, self.embed_size, bias=False)
        
    def forward(self, value_seq, key_seq, query_seq):
        # Compute the value, key, and query representations
        values = self.value(value_seq)
        keys = self.key(key_seq)
        queries = self.query(query_seq)
        
        # Calculate the attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.embed_size ** 0.5
        attention = F.softmax(attention_scores, dim=-1)
        
        # Apply the attention scores to the values
        out = torch.matmul(attention, values)
        return query_seq + out

class GCN(nn.Module):
    def __init__(self, in_channels=1025, hidden_channels=512, out_dim=256,):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = LayerNorm(hidden_channels)
        
        self.pool = global_mean_pool
        
        self.readout = nn.Linear(hidden_channels, hidden_channels//2)
        self.hidden = nn.Linear(hidden_channels//2, hidden_channels//4)
        self.output = nn.Linear(hidden_channels//4, out_dim)
        
    def forward(self, x, edge_index, batch):
        
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.gelu(x)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.gelu(x)
        
        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        x = self.pool(x, batch)
        
        x = self.readout(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        
        x = self.hidden(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        x = self.output(x)
        
        return x


class GATfly3(nn.Module):
    def __init__(self, in_channels=1025, hidden_channels=512, out_dim=256, heads=3, pool_rate=0.5, ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1, dropout=0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(hidden_channels)
        
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, dropout=0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(hidden_channels)
        
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=heads, dropout=0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = LayerNorm(hidden_channels * heads)
        
        self.pool = global_mean_pool
        
        self.readout = nn.Linear(hidden_channels * heads, hidden_channels)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels//2)
        self.linear2 = nn.Linear(hidden_channels//2, hidden_channels//4)
        self.output = nn.Linear(hidden_channels//4, out_dim)
        
    def forward(self, x, edge_index, batch, edge_attr):
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x, batch)
        x = F.gelu(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x, batch)
        x = F.gelu(x)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x, batch)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        x = self.pool(x, batch)
        
        x = self.readout(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        
        return x

class DistogramMLP(nn.Module):
    def __init__(self, in_dim=256, out_dim=25, loss_type="JSD"):
        super().__init__()
        self.loss_type = loss_type
        self.l1 = nn.Linear(in_dim, in_dim//2)
        self.l2 = nn.Linear(in_dim//2, in_dim//4)
        self.output = nn.Linear(in_dim//4, out_dim)
        
    def forward(self, x):
        x = F.gelu(self.l1(x))
        x = F.gelu(self.l2(x))
        x = self.output(x)
        
        if self.loss_type == "JSD":
            return F.softmax(x, dim=1) # JSD
        elif self.loss_type == "KLD":
            return F.log_softmax(x, dim=1) # KLD

class GATfly3_pair_distogram(nn.Module):
    def __init__(self, loss_type="JSD"):
        super().__init__()
        
        # self.gcn_input = GCN()       # G -> [256,]
        # self.gcn_template = GCN()    # G -> [256,]
        self.gcn_double = GCN()
        self.cross_attention = CrossAttention()
        self.distogram = DistogramMLP(loss_type=loss_type)  # [256,] -> [25,]
        
    def forward(self, in_data, tp_data): # in_data: input sequence data, tp_data: corresponding template data
        # or: tp_data = self.get_template(in_data) ?
        in_x, in_edge_index, in_batch, in_edge_attr = in_data.x, in_data.edge_index, in_data.batch, in_data.edge_attr
        tp_x, tp_edge_index, tp_batch, tp_edge_attr = tp_data.x, tp_data.edge_index, tp_data.batch, tp_data.edge_attr
        
        # on-the-fly calculations. initial edge_attr is 25 inter atom distances
        # in_edge_attr = self._rbf(in_edge_attr).view(in_edge_index.shape[1],-1)
        # tp_edge_attr = self._rbf(tp_edge_attr).view(tp_edge_index.shape[1],-1)
        
        # in_rep = self.gatfly_input(in_x, in_edge_index, in_batch, in_edge_attr)
        # tp_rep = self.gatfly_template(tp_x, tp_edge_index, tp_batch, tp_edge_attr)
        
        # siamese network style
        in_rep = self.gcn_double(in_x, in_edge_index, in_batch)
        tp_rep = self.gcn_double(tp_x, tp_edge_index, tp_batch)
        
        in_rep = self.cross_attention(tp_rep, tp_rep, in_rep)
        
        distogram = self.distogram(in_rep) # expected to be bare probabilities for JSD (log-probs for KLD)
        
        return distogram
    
    def _rbf(self, D):
        D_min, D_max, D_count = 2., 22., 16
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF


class GCNfly3_pair_distogram_dual(nn.Module):
    def __init__(self, loss_type="JSD"):
        super().__init__()
        
        self.gcn_input = GCN()       # G -> [256,]
        self.gcn_template = GCN()    # G -> [256,]
        
        self.cross_attention = CrossAttention()
        self.distogram = DistogramMLP(loss_type=loss_type)  # [256,] -> [25,]
        
    def forward(self, in_data, tp_data): # in_data: input sequence data, tp_data: corresponding template data
        # or: tp_data = self.get_template(in_data) ?
        in_x, in_edge_index, in_batch = in_data.x, in_data.edge_index, in_data.batch
        tp_x, tp_edge_index, tp_batch = tp_data.x, tp_data.edge_index, tp_data.batch
        
        in_rep = self.gcn_input(in_x, in_edge_index, in_batch)
        tp_rep = self.gcn_template(tp_x, tp_edge_index, tp_batch)
        
        in_rep = self.cross_attention(tp_rep, tp_rep, in_rep)
        
        distogram = self.distogram(in_rep) # expected to be bare probabilities for JSD (log-probs for KLD)
        
        return distogram
    