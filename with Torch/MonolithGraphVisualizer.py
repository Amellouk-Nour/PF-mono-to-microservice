import networkx as nx
import torch
from torch_geometric.data import Data
import re
from pathlib import Path
import matplotlib.pyplot as plt

class MonolithGraphPreparer:
    def __init__(self, source_dir):
        """
        Initialise l'instance pour préparer un projet monolithique.
        :param source_dir: Répertoire contenant les fichiers source du monolithe.
        """
        self.source_dir = Path(source_dir)  # Utilisation de Pathlib
        self.graph = nx.DiGraph()  # Graphe orienté (Dépendances)
        self.node_mapping = {}
        self.node_features = None
        self.edge_index = None

    def parse_project(self):
        """
        Analyse le projet pour construire un graphe à partir des classes et des relations.
        """
        for file_path in self.source_dir.rglob("*.java"):  # Parcourir récursivement les fichiers .java
            self._process_file(file_path)
        
        # Débogage : Lister les nœuds sans type
        for node in self.graph.nodes:
            if "type" not in self.graph.nodes[node]:
                print(f"Nœud sans type : {node}")

    def _process_file(self, file_path):
        """
        Analyse un fichier Java pour extraire les nœuds et les relations.
        :param file_path: Chemin du fichier Java (Pathlib object).
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

            # Extraire les appels directs à d'autres classes ou entités
            method_calls = re.findall(r"(\w+)\.\w+\(", content)
            for callee in method_calls:
                if callee != class_name:  # Éviter les auto-appels
                    # Associer l'appel si la classe appelée existe ou est une entité
                    self.graph.add_edge(class_name, callee, relation="call")
                    print(f"Appel direct détecté : {class_name} -> {callee}")

    def prepare_for_gnn(self):
        """
        Prépare les données pour un modèle GNN.
        Retourne un objet `torch_geometric.data.Data`.
        """
        # Créer un mapping entre les classes et des indices
        nodes = list(self.graph.nodes())
        self.node_mapping = {node: i for i, node in enumerate(nodes)}

        # Construire edge_index pour le GNN
        edges = list(self.graph.edges())
        self.edge_index = torch.tensor(
            [[self.node_mapping[src], self.node_mapping[dst]] for src, dst in edges], 
            dtype=torch.long
        ).t()

        # Construire les features des nœuds (encodage one-hot des types de classes)
        node_types = []
        for node in nodes:
            node_type = self.graph.nodes[node].get("type", "Unknown")  # Défaut "Unknown" si le type est absent
            node_types.append(node_type)
        
        type_mapping = {t: i for i, t in enumerate(set(node_types))}
        self.node_features = torch.tensor(
            [type_mapping[node_type] for node_type in node_types], 
            dtype=torch.float
        ).unsqueeze(1)  # Ajouter une dimension pour PyTorch Geometric

        # Construire l'objet Data
        data = Data(x=self.node_features, edge_index=self.edge_index)
        return data

    def visualize_dependency_graph(self):
        """
        Visualise uniquement le graphe des dépendances.
        """
        # Filtrer les arêtes de dépendance
        dependency_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) 
            if d.get("relation") == "dependency"
        ]

        # Créer un sous-graphe pour les dépendances
        dependency_graph = self.graph.edge_subgraph(dependency_edges).copy()

        # Couleurs pour les nœuds
        color_map = {
            "Controller": "red",
            "Service": "blue",
            "Repository": "green",
            "Component": "orange",
            "Class": "gray",
            "Unknown": "black"  # Ajouter une couleur pour les nœuds inconnus
        }

        node_colors = [
            color_map.get(dependency_graph.nodes[node].get("type", "Unknown"), "gray") 
            for node in dependency_graph.nodes
        ]

        plt.figure(figsize=(12, 12))

        # Ajuster le layout pour espacer les nœuds
        pos = nx.spring_layout(dependency_graph, seed=42, k=0.5, iterations=100)

        # Dessiner les nœuds et arêtes
        nx.draw_networkx_nodes(dependency_graph, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(dependency_graph, pos, alpha=0.5, edge_color="black")
        nx.draw_networkx_labels(dependency_graph, pos, font_size=8)

        plt.title("Graphe de Dépendance du Monolithe")
        plt.axis("off")
        plt.show()

    def visualize_graph(self):
        """
        Visualise le graphe en mettant en évidence les entités.
        """
        color_map = {
            "Controller": "red",
            "Service": "blue",
            "Repository": "green",
            "Component": "orange",
            "Entity": "purple",  # Couleur pour les entités
            "Class": "gray",
            "Unknown": "black"
        }

        # S'assurer que tous les nœuds ont un type, sinon assigner "Unknown"
        for node in self.graph.nodes:
            if "type" not in self.graph.nodes[node]:
                self.graph.nodes[node]["type"] = "Unknown"

        node_colors = [
            color_map[self.graph.nodes[node]["type"]]
            for node in self.graph.nodes
        ]

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.graph, seed=42, k=0.5, iterations=100)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, edge_color="black")
        nx.draw_networkx_labels(self.graph, pos, font_size=8)

        plt.title("Graphe avec entités (@Entity) mises en évidence")
        plt.axis("off")
        plt.show()




# Exemple d'utilisation
source_dir = "projet_SI_gestion_ECM-main"  # Remplacez par le chemin de votre projet
monolith_preparer = MonolithGraphPreparer(source_dir)

# Étape 1 : Analyse et extraction des données
monolith_preparer.parse_project()

# Étape 2 : Préparation des données pour le GNN
data = monolith_preparer.prepare_for_gnn()
print(f"Features des nœuds (x): {data.x.shape}")
print(f"Index des arêtes (edge_index): {data.edge_index.shape}")

# Étape 3 : Visualisation
monolith_preparer.visualize_graph()

