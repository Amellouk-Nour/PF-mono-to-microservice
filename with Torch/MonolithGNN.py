import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn

class MonolithGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(MonolithGNN, self).__init__()
        
        # Calculate dimensions for each layer
        self.dim1 = hidden_dim * 4  # First layer output (4 heads)
        self.dim2 = hidden_dim * 2  # Second layer output (2 heads)
        self.dim3 = hidden_dim      # Third layer output (1 head)
        
        # Multi-head attention layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATConv(self.dim1, hidden_dim, heads=2, dropout=dropout)
        self.conv3 = GATConv(self.dim2, hidden_dim, heads=1, dropout=dropout)
        
        # Batch normalization layers with correct dimensions
        self.batch_norm1 = nn.BatchNorm1d(self.dim1)  # hidden_dim * 4
        self.batch_norm2 = nn.BatchNorm1d(self.dim2)  # hidden_dim * 2
        self.batch_norm3 = nn.BatchNorm1d(self.dim3)  # hidden_dim
        
        # MLPs for feature processing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification layer
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x, edge_index):
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Third GAT layer
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.elu(x)
        
        # Feature processing
        embeddings = self.mlp(x)
        
        # Classification
        logits = self.classifier(embeddings)
        
        return F.log_softmax(logits, dim=1), embeddings