import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
import re
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import logging

class MonolithDataPreparer:
    def __init__(self, source_dir):
        self.source_dir = Path(source_dir)
        self.graph = nx.DiGraph()
        self.scaler = RobustScaler()
        self.logger = logging.getLogger(__name__)
        
    def parse_project(self):
        """Parse le projet Java et construit le graphe de dépendances"""
        self.logger.info(f"Analyse du projet dans: {self.source_dir}")
        java_files = list(self.source_dir.rglob("*.java"))
        self.logger.info(f"Fichiers Java trouvés: {len(java_files)}")
        
        for file_path in java_files:
            self._process_file(file_path)
            
        self.logger.info(f"Analyse terminée. {len(self.graph.nodes)} composants détectés")
        return len(self.graph.nodes) > 0
            
    def _process_file(self, file_path):
        """Traite un fichier Java individuel"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extraction du nom de la classe
            class_pattern = r"(?:public\s+)?(?:class|interface|enum)\s+(\w+)"
            class_match = re.search(class_pattern, content)
            if not class_match:
                return
                
            class_name = class_match.group(1)
            
            # Package extraction
            package_pattern = r"package\s+([\w.]+);"
            package_match = re.search(package_pattern, content)
            package = package_match.group(1) if package_match else "default"
            
            # Type detection with hierarchy
            type_patterns = [
                (r"@RestController|@Controller", "Controller"),
                (r"@Service", "Service"),
                (r"@Repository", "Repository"),
                (r"@Entity", "Entity"),
                (r"@Component", "Component"),
                (r"@Configuration", "Configuration")
            ]
            
            component_type = None
            for pattern, type_name in type_patterns:
                if re.search(pattern, content):
                    component_type = type_name
                    break
                    
            if not component_type:
                # Fallback to naming conventions
                if class_name.endswith("Controller"):
                    component_type = "Controller"
                elif class_name.endswith("Service"):
                    component_type = "Service"
                elif class_name.endswith("Repository"):
                    component_type = "Repository"
                elif class_name.endswith("Entity"):
                    component_type = "Entity"
                else:
                    component_type = "Component"
            
            # Add node with metadata
            self.graph.add_node(class_name, 
                              type=component_type,
                              file=str(file_path),
                              package=package)
                              
            # Extract dependencies
            self._extract_dependencies(content, class_name)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de {file_path}: {str(e)}")
            
    def _extract_dependencies(self, content, class_name):
        """Extrait les dépendances du code"""
        # Spring annotations
        dependencies = set()
        
        # Autowired dependencies
        autowired_pattern = r'@Autowired\s+(?:private\s+)?(\w+)\s+\w+'
        dependencies.update(re.findall(autowired_pattern, content))
        
        # Constructor injection
        constructor_pattern = r'public\s+\w+\(((?:[^)]+))\)'
        constructor_matches = re.findall(constructor_pattern, content)
        for params in constructor_matches:
            param_types = re.findall(r'(\w+)\s+\w+(?:\s*,\s*|$)', params)
            dependencies.update(param_types)
            
        # Import statements
        import_pattern = r'import\s+[\w.]+\.(\w+);'
        dependencies.update(re.findall(import_pattern, content))
        
        # Add edges
        for dep in dependencies:
            if dep != class_name and dep in self.graph:
                self.graph.add_edge(class_name, dep, type="dependency")

    def generate_labels(self, n_clusters=3):
        """Génère des labels semi-supervisés équilibrés"""
        if len(self.graph.nodes) == 0:
            return None
            
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        labels = torch.full((n_nodes,), -1, dtype=torch.long)
        
        # Calculate node metrics
        metrics = {}
        for node in nodes:
            # Basic metrics
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            
            # Calculate coupling and cohesion
            coupling = in_deg + out_deg
            neighbors = list(self.graph.predecessors(node)) + list(self.graph.successors(node))
            same_type_neighbors = sum(1 for n in neighbors 
                                   if self.graph.nodes[n]['type'] == self.graph.nodes[node]['type'])
            cohesion = same_type_neighbors / len(neighbors) if neighbors else 0
            
            # Package cohesion
            same_package_neighbors = sum(1 for n in neighbors 
                                      if self.graph.nodes[n]['package'] == self.graph.nodes[node]['package'])
            package_cohesion = same_package_neighbors / len(neighbors) if neighbors else 0
            
            metrics[node] = {
                'coupling': coupling,
                'cohesion': cohesion,
                'package_cohesion': package_cohesion,
                'type': self.graph.nodes[node]['type'],
                'package': self.graph.nodes[node]['package']
            }
        
        # Select representative seeds
        packages = set(nx.get_node_attributes(self.graph, 'package').values())
        component_types = set(nx.get_node_attributes(self.graph, 'type').values())
        
        # Ensure balanced representation
        seeds_per_cluster = max(2, n_nodes // (n_clusters * 5))  # At least 2 seeds per cluster
        seeds = []
        
        # Select seeds based on coupling and cohesion
        nodes_metrics = [(node, metrics[node]) for node in nodes]
        nodes_metrics.sort(key=lambda x: (-x[1]['cohesion'], x[1]['coupling']))
        
        cluster_counts = {i: 0 for i in range(n_clusters)}
        type_counts = {t: 0 for t in component_types}
        
        for node, node_metrics in nodes_metrics:
            if len(seeds) >= n_clusters * seeds_per_cluster:
                break
                
            # Find best cluster for this node
            best_cluster = min(cluster_counts.items(), key=lambda x: x[1])[0]
            
            if type_counts[node_metrics['type']] < (len(seeds) // len(component_types) + 1):
                seeds.append((node, best_cluster))
                cluster_counts[best_cluster] += 1
                type_counts[node_metrics['type']] += 1
        
        # Assign initial labels
        for node, cluster in seeds:
            idx = nodes.index(node)
            labels[idx] = cluster
            
        return labels

    def prepare_for_gnn(self):
        """Prépare les données pour le GNN"""
        if len(self.graph.nodes) == 0:
            return None
            
        # Prepare features
        nodes = list(self.graph.nodes())
        features = []
        
        # Get unique values for one-hot encoding
        types = list(set(nx.get_node_attributes(self.graph, 'type').values()))
        packages = list(set(nx.get_node_attributes(self.graph, 'package').values()))
        
        for node in nodes:
            # Basic metrics
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            
            # Type one-hot encoding
            type_idx = types.index(self.graph.nodes[node]['type'])
            type_onehot = [0] * len(types)
            type_onehot[type_idx] = 1
            
            # Package one-hot encoding
            package_idx = packages.index(self.graph.nodes[node]['package'])
            package_onehot = [0] * len(packages)
            package_onehot[package_idx] = 1
            
            # Structural metrics
            neighbors = list(self.graph.neighbors(node))
            clustering_coef = nx.clustering(self.graph, node)
            
            feature_vector = [
                in_deg / self.graph.number_of_nodes(),
                out_deg / self.graph.number_of_nodes(),
                clustering_coef
            ] + type_onehot + package_onehot
            
            features.append(feature_vector)
            
        x = torch.tensor(features, dtype=torch.float)
        
        # Create edge_index
        edge_index = []
        for edge in self.graph.edges():
            src_idx = nodes.index(edge[0])
            dst_idx = nodes.index(edge[1])
            edge_index.append([src_idx, dst_idx])
            
        edge_index = torch.tensor(edge_index).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)

    def visualize_clusters(self, predictions):
        """Visualise les clusters proposés"""
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(self.graph, k=1.5, iterations=50)
        
        # Color mapping
        unique_clusters = len(set(predictions))
        colors = plt.cm.rainbow(np.linspace(0, 1, unique_clusters))
        color_map = dict(zip(range(unique_clusters), colors))
        
        # Draw nodes by cluster
        nodes = list(self.graph.nodes())
        for cluster_id in range(unique_clusters):
            node_list = [nodes[i] for i, pred in enumerate(predictions) if pred == cluster_id]
            nx.draw_networkx_nodes(self.graph, pos,
                                 nodelist=node_list,
                                 node_color=[color_map[cluster_id]],
                                 node_size=500)
        
        # Draw edges and labels
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2)
        nx.draw_networkx_labels(self.graph, pos)
        
        plt.title("Proposed Microservices Clusters")
        plt.axis("off")
        plt.savefig("clusters.png")
        plt.close()