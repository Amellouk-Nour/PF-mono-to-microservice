import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class MonolithGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, use_attention=True):
        super(MonolithGNN, self).__init__()
        self.use_attention = use_attention
        
        # Choose between GAT or GCN layers
        if use_attention:
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
            self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
            final_hidden = hidden_dim * 4
        else:
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            final_hidden = hidden_dim

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(final_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(final_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def get_embeddings(self, x, edge_index):
        if self.use_attention:
            x = F.elu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.elu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = self.conv3(x, edge_index)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = self.conv3(x, edge_index)
        return x
    
    def forward(self, x, edge_index, batch=None):
        # Get node embeddings
        x = self.get_embeddings(x, edge_index)
        
        # Apply classification head
        logits = self.classifier(x)
        
        # Global pooling if batch information is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        return F.log_softmax(logits, dim=1), self.projection(x)
