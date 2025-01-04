import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn
import math

class AdaptiveMonolithGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_size=None, min_heads=1, max_heads=8, dropout=0.3):
        """
        Architecture GNN adaptative qui s'ajuste à la taille du projet
        
        Args:
            input_dim: Dimension des features d'entrée
            hidden_dim: Dimension de base des couches cachées
            output_dim: Nombre de classes (clusters) en sortie
            graph_size: Nombre de nœuds dans le graphe
            min_heads: Nombre minimum de têtes d'attention
            max_heads: Nombre maximum de têtes d'attention
            dropout: Taux de dropout
        """
        super(AdaptiveMonolithGNN, self).__init__()
        
        # Calcul automatique du nombre de couches basé sur la taille du graphe
        self.num_layers = self._calculate_num_layers(graph_size) if graph_size else 3
        
        # Calcul automatique de la configuration des têtes d'attention
        self.heads_config = self._calculate_heads_config(
            self.num_layers, 
            min_heads=min_heads, 
            max_heads=max_heads
        )
        
        # Création des couches GAT dynamiques
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Dimensions des couches
        current_dim = input_dim
        self.layer_dims = []
        
        # Construction des couches
        for layer_idx, num_heads in enumerate(self.heads_config):
            layer_out_dim = hidden_dim
            total_out_dim = layer_out_dim * num_heads
            
            self.gat_layers.append(
                GATConv(
                    current_dim,
                    layer_out_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            
            self.batch_norms.append(nn.BatchNorm1d(total_out_dim))
            self.layer_dims.append(total_out_dim)
            
            current_dim = total_out_dim
        
        # MLP adaptatif
        mlp_dims = self._calculate_mlp_dims(
            self.layer_dims[-1],
            output_dim,
            graph_size
        )
        
        self.mlp = self._build_adaptive_mlp(mlp_dims, dropout)
        self.classifier = nn.Linear(mlp_dims[-1], output_dim)
        
    def _calculate_num_layers(self, graph_size):
        """Calcule le nombre optimal de couches basé sur la taille du graphe"""
        if graph_size is None:
            return 3
        
        # Utilise une échelle logarithmique pour déterminer le nombre de couches
        num_layers = max(2, min(5, int(math.log2(graph_size/10))))
        return num_layers
    
    def _calculate_heads_config(self, num_layers, min_heads=1, max_heads=8):
        """Calcule une configuration progressive des têtes d'attention"""
        # Distribution progressive du nombre de têtes
        heads = []
        for i in range(num_layers):
            # Plus de têtes au début, moins à la fin
            num_heads = max(min_heads, 
                          min(max_heads, 
                              max_heads - (i * (max_heads - min_heads) // (num_layers - 1))))
            heads.append(num_heads)
        return heads
    
    def _calculate_mlp_dims(self, input_dim, output_dim, graph_size):
        """Calcule les dimensions optimales pour le MLP"""
        if graph_size is None:
            return [input_dim, input_dim // 2, input_dim // 4]
            
        # Ajuste la complexité du MLP en fonction de la taille du graphe
        num_layers = max(2, min(4, int(math.log2(graph_size/20))))
        dims = [input_dim]
        
        for i in range(num_layers - 1):
            next_dim = dims[-1] // 2
            if next_dim < output_dim * 2:
                break
            dims.append(next_dim)
            
        return dims
    
    def _build_adaptive_mlp(self, dims, dropout):
        """Construit un MLP avec les dimensions spécifiées"""
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        return nn.Sequential(*layers)
    
    def forward(self, x, edge_index):
        """Propagation avant avec résidus conditionnels"""
        # Liste pour stocker les sorties intermédiaires
        intermediate_outputs = []
        
        # Propagation à travers les couches GAT
        for i, (conv, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_prev = x
            
            # Couche GAT
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            
            # Connexions résiduelles conditionnelles
            if i > 0 and x.size(-1) == x_prev.size(-1):
                x = x + x_prev
                
            x = F.dropout(x, p=0.3, training=self.training)
            intermediate_outputs.append(x)
        
        # Traitement des features avec le MLP
        embeddings = self.mlp(x)
        
        # Classification
        logits = self.classifier(embeddings)
        
        return F.log_softmax(logits, dim=1), embeddings