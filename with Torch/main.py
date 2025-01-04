import torch
from MonolithDataPreparer import MonolithDataPreparer
from MonolithGNN import MonolithGNN
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from torch_geometric.data import Data
import os

def initialize_weights(model):
    """Initialise les poids du modèle avec Xavier initialization"""
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)

def train_gnn(model, data, optimizer, mask, epochs=500):
    """
    Entraîne le modèle GNN avec affichage des métriques
    """
    model.train()
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out, embeddings = model(data.x, data.edge_index)
        loss = F.nll_loss(out[mask], data.y[mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_state)
            break
            
        if epoch % 50 == 0:
            print(f"Époque {epoch + 1}/{epochs}, Perte : {loss.item():.4f}")
            pred = out.argmax(dim=1)
            masked_correct = (pred[mask] == data.y[mask]).sum()
            masked_total = mask.sum()
            accuracy = masked_correct / masked_total
            print(f"Accuracy sur données étiquetées : {accuracy:.4f}")

def evaluate_clustering(model, data, original_graph):
    """
    Évalue la qualité du clustering proposé
    """
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
    # Calcul du score silhouette
    embeddings = model.get_embeddings(data.x, data.edge_index).detach().numpy()
    if len(np.unique(pred)) > 1:  # Silhouette score needs at least 2 clusters
        silhouette = silhouette_score(embeddings, pred)
        print(f"Score Silhouette : {silhouette:.4f}")
    
    # Analyse de la cohésion des clusters
    clusters = {}
    for i, p in enumerate(pred):
        p = p.item()
        if p not in clusters:
            clusters[p] = []
        clusters[p].append(i)
    
    print("\nAnalyse des clusters proposés:")
    for cluster_id, nodes in clusters.items():
        print(f"\nCluster {cluster_id}:")
        
        # Compte des types de composants
        component_types = {}
        for node_idx in nodes:
            node_name = list(original_graph.nodes())[node_idx]
            node_type = original_graph.nodes[node_name].get('type', 'Unknown')
            component_types[node_type] = component_types.get(node_type, 0) + 1
        
        print("Composition:")
        for comp_type, count in component_types.items():
            print(f"- {comp_type}: {count}")
            
        # Analyse des dépendances internes vs externes
        internal_edges = 0
        external_edges = 0
        for node_idx in nodes:
            node_name = list(original_graph.nodes())[node_idx]
            for neighbor in original_graph.neighbors(node_name):
                neighbor_idx = list(original_graph.nodes()).index(neighbor)
                if neighbor_idx in nodes:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        if internal_edges + external_edges > 0:
            cohesion = internal_edges / (internal_edges + external_edges)
            print(f"Cohésion du cluster: {cohesion:.2f}")
            print(f"Dépendances internes: {internal_edges}")
            print(f"Dépendances externes: {external_edges}")

def save_model_and_predictions(model, data, output_dir="output"):
    """
    Sauvegarde le modèle et les prédictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde du modèle
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    # Sauvegarde des prédictions
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        torch.save(pred, os.path.join(output_dir, "predictions.pt"))

def main():
    # Configuration
    source_dir = "projet_SI_gestion_ECM-main"  # À modifier selon votre projet
    num_clusters = 5  # Nombre de microservices souhaités
    epochs = 500
    hidden_dim = 64
    learning_rate = 0.001
    
    # Préparation des données
    print("\n=== Préparation des données ===")
    data_preparer = MonolithDataPreparer(source_dir)
    data_preparer.parse_project()
    data = data_preparer.prepare_for_gnn()
    
    # Génération des labels semi-supervisés
    labels = data_preparer.generate_labels(n_clusters=num_clusters)
    if labels is None or len(labels) == 0:
        raise ValueError("Les labels n'ont pas été générés correctement.")
    
    # Création du masque pour l'apprentissage supervisé
    mask = labels != -1
    supervised_indices = torch.where(mask)[0]
    
    if supervised_indices.numel() == 0:
        raise ValueError("Aucun nœud supervisé trouvé. Vérifiez la génération des labels.")
    
    # Équilibrage des données supervisées
    balanced_indices = supervised_indices[torch.randperm(len(supervised_indices))[:min(100, len(supervised_indices))]]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[balanced_indices] = True
    data.y = labels
    
    print(f"\nStatistiques des données:")
    print(f"Nombre total de nœuds: {data.x.size(0)}")
    print(f"Nombre de caractéristiques: {data.x.size(1)}")
    print(f"Nombre d'arêtes: {data.edge_index.size(1)}")
    print(f"Nombre de nœuds supervisés: {mask.sum().item()}")
    print(f"Classes supervisées: {set(labels[mask].numpy())}")
    
    # Initialisation du modèle
    print("\n=== Initialisation du modèle ===")
    input_dim = data.x.size(1)
    output_dim = len(torch.unique(labels[labels != -1]))
    model = MonolithGNN(input_dim, hidden_dim, output_dim)
    initialize_weights(model)
    
    # Optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Entraînement
    print("\n=== Entraînement du modèle ===")
    train_gnn(model, data, optimizer, mask, epochs=epochs)
    
    # Évaluation
    print("\n=== Évaluation du modèle ===")
    evaluate_clustering(model, data, data_preparer.graph)
    
    # Visualisation
    print("\n=== Génération des visualisations ===")
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        data_preparer.visualize_clusters(data_preparer.graph, pred.numpy())
    
    # Sauvegarde
    print("\n=== Sauvegarde des résultats ===")
    save_model_and_predictions(model, data)
    data_preparer.export_graph("output/monolith.graphml")
    
    print("\nAnalyse terminée ! Les résultats ont été sauvegardés dans le dossier 'output'.")

if __name__ == "__main__":
    main()