import os
import re
import networkx as nx
import torch
from torch_geometric.data import Data
from pathlib import Path
import matplotlib.pyplot as plt

class MonolithDataPreparer:
    def __init__(self, source_dir):
        """
        Initialise l'instance pour préparer les données d'un projet monolithique.
        :param source_dir: Répertoire contenant les fichiers source du monolithe.
        """
        self.source_dir = Path(source_dir)
        self.graph = nx.DiGraph()  # Graphe orienté des dépendances
        self.node_mapping = {}  # Mapping des nœuds vers des indices
        self.node_features = None  # Features des nœuds
        self.edge_index = None  # Arêtes dans un format adapté à PyTorch Geometric
        self.labels = None  # Labels des microservices (si supervisé)

    def parse_project(self):
        """
        Analyse le projet Java pour extraire les classes et leurs relations.
        """
        for file_path in self.source_dir.rglob("*.java"):
            self._process_file(file_path)

    def _process_file(self, file_path):
        """
        Analyse un fichier Java pour extraire les informations de classes et relations.
        :param file_path: Chemin du fichier Java.
        """
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Extraire le nom de la classe
        class_match = re.search(r"class\s+(\w+)", content)
        if not class_match:
            return
        class_name = class_match.group(1)

        # Identifier le type de la classe
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

        # Extraire les dépendances (Autowired, Inject)
        dependencies = re.findall(r"@(Autowired|Inject)\s+.*\s+(\w+);", content)
        for _, dependency in dependencies:
            self.graph.add_edge(class_name, dependency, relation="dependency")

        # Extraire les appels directs
        method_calls = re.findall(r"(\w+)\.\w+\(", content)
        for callee in method_calls:
            if callee != class_name:
                self.graph.add_edge(class_name, callee, relation="call")

    def prepare_for_gnn(self):
        """
        Prépare les données pour un modèle GNN.
        Retourne un objet `torch_geometric.data.Data`.
        """
        # Créer un mapping entre les nœuds et leurs indices
        nodes = list(self.graph.nodes())
        self.node_mapping = {node: i for i, node in enumerate(nodes)}

        # Construire edge_index
        edges = list(self.graph.edges())
        self.edge_index = torch.tensor(
            [[self.node_mapping[src], self.node_mapping[dst]] for src, dst in edges],
            dtype=torch.long
        ).t()

        # Construire les features des nœuds (encodage one-hot des types de classes)
        node_types = [self.graph.nodes[node].get("type", "Unknown") for node in nodes]
        type_mapping = {t: i for i, t in enumerate(set(node_types))}
        self.node_features = torch.tensor(
            [type_mapping[node_type] for node_type in node_types],
            dtype=torch.float
        ).unsqueeze(1)  # Ajout d'une dimension pour PyTorch Geometric

        # Construire l'objet Data
        data = Data(x=self.node_features, edge_index=self.edge_index)
        return data

    def export_graph(self, output_path):
        """
        Exporte le graphe sous forme d'un fichier GraphML.
        :param output_path: Chemin du fichier GraphML de sortie.
        """
        nx.write_graphml(self.graph, output_path)
        print(f"Graphe exporté vers {output_path}")

    def visualize_graph(self):
        """
        Visualise le graphe avec des couleurs selon le type de nœud, avec des nœuds plus espacés.
        """
        color_map = {
            "Controller": "red",
            "Service": "blue",
            "Repository": "green",
            "Component": "orange",
            "Entity": "purple",
            "Class": "gray",
            "Unknown": "black"  # Ajouter une couleur pour les nœuds inconnus
        }

        # Assurez-vous que tous les nœuds ont un type
        for node in self.graph.nodes:
            if "type" not in self.graph.nodes[node]:
                self.graph.nodes[node]["type"] = "Unknown"

        node_colors = [
            color_map[self.graph.nodes[node]["type"]]
            for node in self.graph.nodes
        ]

        plt.figure(figsize=(12, 12))

        # Ajuster le layout avec une valeur de `k` plus élevée pour espacer les nœuds
        pos = nx.spring_layout(self.graph, seed=42, k=0.5, iterations=100)  # Augmenter `k` pour plus d'espacement

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, edge_color="black")
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        plt.title("Graphe du Monolithe Spring Boot")
        plt.axis("off")
        plt.show()
