# MonolithDataPreparer.py
import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class MonolithDataPreparer:
    def __init__(self, source_dir):
        self.source_dir = Path(source_dir)
        self.graph = nx.DiGraph()
        self.node_features = None
        self.edge_index = None
        self.node_mapping = {}
        try:
            # Tentative avec la nouvelle version de scikit-learn
            self.feature_encoder = OneHotEncoder(sparse_output=False)
        except TypeError:
            # Fallback pour les anciennes versions
            self.feature_encoder = OneHotEncoder(sparse=False)

    def parse_project(self):
        """
        Analyse le projet pour construire un graphe à partir des classes et des relations.
        """
        for file_path in self.source_dir.rglob("*.java"):
            self._process_file(file_path)
        
        # Débogage : Lister les nœuds sans type
        for node in self.graph.nodes:
            if "type" not in self.graph.nodes[node]:
                print(f"Nœud sans type : {node}")

    def _process_file(self, file_path):
        """
        Analyse un fichier Java pour extraire les nœuds et les relations.
        """
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Extraire les classes
        class_match = re.search(r"class\s+(\w+)", content)
        if class_match:
            class_name = class_match.group(1)
            print(f"Classe détectée : {class_name}")

            # Identifier le type de classe en fonction des annotations
            if "@Controller" in content:
                node_type = "Controller"
            elif "@Service" in content:
                node_type = "Service"
            elif "@Repository" in content:
                node_type = "Repository"
            elif "@Component" in content:
                node_type = "Component"
            elif "@Entity" in content:
                node_type = "Entity"
            else:
                node_type = "Class"

            # Ajouter le nœud au graphe
            self.graph.add_node(class_name, type=node_type, file=str(file_path))
            print(f"Ajouté au graphe : {class_name} ({node_type})")

            # Extraire les dépendances (Autowired, Inject)
            dependencies = re.findall(r"@(Autowired|Inject)\s+.*\s+(\w+);", content)
            for _, dependency in dependencies:
                self.graph.add_edge(class_name, dependency, relation="dependency")
                print(f"Dépendance détectée : {class_name} -> {dependency}")

            # Extraire les appels directs
            method_calls = re.findall(r"(\w+)\.\w+\(", content)
            for callee in method_calls:
                if callee != class_name:
                    self.graph.add_edge(class_name, callee, relation="call")
                    print(f"Appel direct détecté : {class_name} -> {callee}")

    def generate_labels(self, n_clusters=5):
        """
        Générer des labels semi-supervisés basés sur des règles métier.
        Assure qu'au moins quelques nœuds sont étiquetés.
        """
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        labels = torch.full((n_nodes,), -1, dtype=torch.long)
        
        labeled_count = 0
        
        # Première passe : étiqueter les composants clés
        for idx, node in enumerate(nodes):
            node_data = self.graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            
            # Règle 1: Regrouper les contrôleurs et leurs services associés
            if node_type == 'Controller':
                # Trouver les services associés
                cluster = labeled_count % n_clusters
                labels[idx] = cluster
                labeled_count += 1
                
                # Étiqueter les services directement connectés
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor].get('type') == 'Service':
                        neighbor_idx = nodes.index(neighbor)
                        labels[neighbor_idx] = cluster
                        labeled_count += 1

            # Règle 2: Regrouper les entités et leurs repositories
            elif node_type == 'Entity':
                # Si pas déjà étiqueté
                if labels[idx] == -1:
                    cluster = labeled_count % n_clusters
                    labels[idx] = cluster
                    labeled_count += 1
                    
                    # Trouver le repository associé
                    for neighbor in self.graph.neighbors(node):
                        if self.graph.nodes[neighbor].get('type') == 'Repository':
                            neighbor_idx = nodes.index(neighbor)
                            labels[neighbor_idx] = cluster
                            labeled_count += 1

        # Si nous n'avons pas assez de nœuds étiquetés, ajouter des étiquettes supplémentaires
        if labeled_count < n_nodes * 0.1:  # Au moins 10% des nœuds devraient être étiquetés
            print("Ajout d'étiquettes supplémentaires pour assurer une supervision suffisante...")
            unlabeled = [idx for idx, label in enumerate(labels) if label == -1]
            additional_labels = min(len(unlabeled), n_nodes // 5)  # Étiqueter jusqu'à 20% supplémentaires
            
            for idx in unlabeled[:additional_labels]:
                labels[idx] = labeled_count % n_clusters
                labeled_count += 1

        print(f"Nombre total de nœuds étiquetés : {labeled_count}")
        print(f"Distribution des labels : {torch.bincount(labels[labels != -1])}")
        
        if labeled_count == 0:
            print("ATTENTION : Aucun nœud n'a été étiqueté. Vérification du graphe :")
            print(f"Nombre total de nœuds : {len(nodes)}")
            for node in nodes:
                print(f"Nœud : {node}, Type : {self.graph.nodes[node].get('type', 'Unknown')}")
        
        return labels

    def prepare_for_gnn(self):
        """
        Prépare les données pour un modèle GNN avec des caractéristiques enrichies.
        Ajoute une vérification des types de nœuds.
        """
        # Créer un mapping entre les classes et des indices
        nodes = list(self.graph.nodes())
        self.node_mapping = {node: i for i, node in enumerate(nodes)}

        # Afficher les statistiques des types de nœuds
        node_type_stats = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'Unknown')
            node_type_stats[node_type] = node_type_stats.get(node_type, 0) + 1
        
        print("\nStatistiques des types de nœuds:")
        for node_type, count in node_type_stats.items():
            print(f"{node_type}: {count}")

        # Construire edge_index pour le GNN
        edges = list(self.graph.edges())
        self.edge_index = torch.tensor(
            [[self.node_mapping[src], self.node_mapping[dst]] for src, dst in edges],
            dtype=torch.long
        ).t()

        # Préparer les caractéristiques des nœuds
        node_types = []
        node_features_list = []
        
        for node in nodes:
            # Type de base du nœud
            node_type = self.graph.nodes[node].get('type', 'Unknown')
            node_types.append(node_type)
            
            # Caractéristiques structurelles
            degree = self.graph.degree(node)
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            clustering_coeff = nx.clustering(self.graph, node)
            
            # Caractéristiques spécifiques aux microservices
            is_entry_point = 1.0 if node_type == "Controller" else 0.0
            is_data_layer = 1.0 if node_type in ["Repository", "Entity"] else 0.0
            is_service = 1.0 if node_type == "Service" else 0.0
            
            # Caractéristiques de connectivité
            neighbors = list(self.graph.neighbors(node))
            has_controller = 1.0 if any(self.graph.nodes[n].get('type') == 'Controller' for n in neighbors) else 0.0
            has_service = 1.0 if any(self.graph.nodes[n].get('type') == 'Service' for n in neighbors) else 0.0
            has_repository = 1.0 if any(self.graph.nodes[n].get('type') == 'Repository' for n in neighbors) else 0.0
            
            # Combiner toutes les caractéristiques
            node_features = [
                degree,
                in_degree,
                out_degree,
                clustering_coeff,
                is_entry_point,
                is_data_layer,
                is_service,
                has_controller,
                has_service,
                has_repository
            ]
            node_features_list.append(node_features)

        # Encoder les types de nœuds avec OneHotEncoder
        node_types = np.array(node_types).reshape(-1, 1)
        self.feature_encoder.fit(node_types)
        type_encoded = self.feature_encoder.transform(node_types)
        
        # Normaliser les caractéristiques structurelles
        node_features_array = np.array(node_features_list)
        if node_features_array.size > 0:  # Vérifier qu'il y a des features
            # Normaliser chaque colonne séparément
            node_features_mean = np.mean(node_features_array, axis=0)
            node_features_std = np.std(node_features_array, axis=0)
            node_features_std[node_features_std == 0] = 1  # Éviter la division par zéro
            node_features_normalized = (node_features_array - node_features_mean) / node_features_std
        else:
            node_features_normalized = node_features_array

        # Combiner les caractéristiques encodées avec les caractéristiques normalisées
        self.node_features = torch.tensor(
            np.hstack([type_encoded, node_features_normalized]),
            dtype=torch.float
        )

        print("\nStatistiques des données préparées:")
        print(f"Nombre de nœuds: {len(nodes)}")
        print(f"Nombre d'arêtes: {len(edges)}")
        print(f"Dimensions des features: {self.node_features.shape}")
        print(f"Dimensions de edge_index: {self.edge_index.shape}")

        # Créer l'objet Data de PyTorch Geometric
        data = Data(x=self.node_features, edge_index=self.edge_index)
        
        # Vérifier la validité des données
        if data.validate():
            print("Les données sont valides pour PyTorch Geometric")
        else:
            print("ATTENTION: Les données pourraient avoir des problèmes de format")

        return data

    def visualize_clusters(self, graph, labels):
        """
        Visualise les clusters proposés.
        """
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Créer une palette de couleurs pour les clusters
        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))
        
        # Dessiner les nœuds avec leurs couleurs de cluster
        for label in unique_labels:
            node_list = [node for i, node in enumerate(graph.nodes()) if labels[i] == label]
            nx.draw_networkx_nodes(graph, pos, 
                                 nodelist=node_list,
                                 node_color=[color_map[label]],
                                 node_size=500)
        
        # Dessiner les arêtes
        nx.draw_networkx_edges(graph, pos, alpha=0.2)
        
        # Ajouter les labels
        nx.draw_networkx_labels(graph, pos, font_size=8)
        
        plt.title("Clusters de Microservices Proposés")
        plt.axis("off")
        plt.show()

    def export_graph(self, output_path):
        """
        Exporte le graphe au format GraphML pour analyse ultérieure.
        """
        nx.write_graphml(self.graph, output_path)