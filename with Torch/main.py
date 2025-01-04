import torch
import numpy as np
import logging
from pathlib import Path
from MonolithDataPreparer import MonolithDataPreparer
from MonolithGNN import MonolithGNN
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import networkx as nx
import os

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_project_path(source_dir):
    """Validates project path and checks for Java files"""
    path = Path(source_dir)
    
    if not path.exists():
        raise ValueError(f"Directory {source_dir} does not exist")
        
    java_files = list(path.rglob("*.java"))
    if not java_files:
        raise ValueError(f"No Java files found in {source_dir}")
        
    logger.info(f"Found Java files ({len(java_files)}):")
    for file in java_files[:5]:
        logger.info(f"- {file}")
    if len(java_files) > 5:
        logger.info(f"... and {len(java_files)-5} more files")
    
    return True

def setup_training(data_preparer, num_clusters, min_labeled_ratio=0.1):
    """Prépare les données d'entraînement avec validation"""
    data = data_preparer.prepare_for_gnn()
    if data is None:
        raise ValueError("Échec de la préparation des données pour le GNN")
    
    # Génération des labels semi-supervisés
    labels = data_preparer.generate_labels(n_clusters=num_clusters)
    if labels is None:
        raise ValueError("Échec de la génération des labels")
        
    # Vérification de la distribution des labels
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    logger.info(f"Distribution des labels supervisés: {dict(zip(unique_labels, counts))}")
    
    # Vérification du ratio minimum de données étiquetées
    labeled_ratio = (labels != -1).sum() / len(labels)
    if labeled_ratio < min_labeled_ratio:
        logger.warning(f"Ratio de données étiquetées ({labeled_ratio:.2f}) inférieur au minimum recommandé ({min_labeled_ratio})")
    
    data.y = labels
    mask = labels != -1
    
    return data, mask

def train_model(model, data, mask, num_epochs=1000, early_stopping_patience=20):
    """Entraîne le modèle avec early stopping"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    best_loss = float('inf')
    patience_counter = 0
    training_losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out, embeddings = model(data.x, data.edge_index)
        loss = F.nll_loss(out[mask], data.y[mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        training_losses.append(loss.item())
        
        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_state)
            break
            
        if epoch % 50 == 0:
            # Calculate metrics
            pred = out.argmax(dim=1)
            accuracy = (pred[mask] == data.y[mask]).sum() / mask.sum()
            logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    return model, training_losses

def evaluate_clusters(model, data, graph):
    """Évalue la qualité des clusters proposés"""
    model.eval()
    with torch.no_grad():
        out, embeddings = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
    embeddings_np = embeddings.numpy()
    predictions_np = pred.numpy()
    
    # Calcul des métriques de clustering
    if len(np.unique(predictions_np)) > 1:
        silhouette = silhouette_score(embeddings_np, predictions_np)
        calinski = calinski_harabasz_score(embeddings_np, predictions_np)
        logger.info(f"Score de Silhouette: {silhouette:.4f}")
        logger.info(f"Score de Calinski-Harabasz: {calinski:.4f}")
    
    # Analyse des clusters
    clusters = {}
    for i, p in enumerate(pred):
        p = p.item()
        if p not in clusters:
            clusters[p] = []
        clusters[p].append(i)
        
    nodes = list(graph.nodes())
    cluster_analysis = []
    
    for cluster_id, node_indices in clusters.items():
        cluster_nodes = [nodes[i] for i in node_indices]
        
        # Analyse des types de composants
        component_types = {}
        for node in cluster_nodes:
            node_type = graph.nodes[node]['type']
            component_types[node_type] = component_types.get(node_type, 0) + 1
            
        # Analyse des dépendances
        internal_edges = 0
        external_edges = 0
        for node in cluster_nodes:
            for neighbor in graph.neighbors(node):
                if neighbor in cluster_nodes:
                    internal_edges += 1
                else:
                    external_edges += 1
                    
        cluster_info = {
            'id': cluster_id,
            'size': len(node_indices),
            'composition': component_types,
            'internal_edges': internal_edges,
            'external_edges': external_edges,
            'cohesion': internal_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0
        }
        cluster_analysis.append(cluster_info)
        
    return cluster_analysis, pred.numpy()

def save_results(model, data_preparer, predictions, output_dir):
    """Sauvegarde les résultats de l'analyse"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Sauvegarde du modèle
    torch.save(model.state_dict(), output_dir / "model.pt")
    
    # Sauvegarde du graphe
    nx.write_graphml(data_preparer.graph, output_dir / "monolith.graphml")
    
    # Génération de la visualisation
    data_preparer.visualize_clusters(predictions)
    plt.savefig(output_dir / "clusters.png")
    
    # Sauvegarde des prédictions
    np.save(output_dir / "predictions.npy", predictions)

def main():
    # Configuration
    config = {
        'source_dir': './projet_SI_gestion_ECM-main',  # Chemin vers votre projet Java
        'num_clusters': 3,
        'hidden_dim': 64,
        'num_epochs': 1000,
        'min_labeled_ratio': 0.1,
        'output_dir': 'output'
    }
    
    try:
        # Validation du chemin du projet
        logger.info(f"Validation du chemin du projet: {config['source_dir']}")
        validate_project_path(config['source_dir'])
        
        # Préparation des données
        logger.info("Début de l'analyse du projet...")
        data_preparer = MonolithDataPreparer(config['source_dir'])
        if not data_preparer.parse_project():
            raise ValueError("Aucun composant n'a été détecté dans le projet")
        
        # Configuration du modèle et des données
        data, mask = setup_training(data_preparer, config['num_clusters'])
        
        # Initialisation du modèle
        model = MonolithGNN(
            input_dim=data.x.size(1),
            hidden_dim=64,
            output_dim=config['num_clusters']
        )
        
        # Entraînement
        logger.info("Début de l'entraînement...")
        model, losses = train_model(
            model, 
            data, 
            mask, 
            num_epochs=config['num_epochs']
        )
        
        # Évaluation
        logger.info("Évaluation des clusters...")
        cluster_analysis, predictions = evaluate_clusters(
            model, 
            data, 
            data_preparer.graph
        )
        
        # Affichage des résultats
        logger.info("\nRésultats de l'analyse des clusters:")
        for cluster in cluster_analysis:
            logger.info(f"\nCluster {cluster['id']}:")
            logger.info(f"Taille: {cluster['size']} composants")
            logger.info(f"Composition: {cluster['composition']}")
            logger.info(f"Cohésion: {cluster['cohesion']:.2f}")
            logger.info(f"Dépendances internes/externes: {cluster['internal_edges']}/{cluster['external_edges']}")
        
        # Sauvegarde des résultats
        save_results(model, data_preparer, predictions, config['output_dir'])
        
        logger.info("\nAnalyse terminée avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()