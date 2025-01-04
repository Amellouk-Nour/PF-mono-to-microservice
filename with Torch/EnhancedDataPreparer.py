import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
import re
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import logging

class EnhancedMonolithDataPreparer:
    def __init__(self, source_dir):
        self.source_dir = Path(source_dir)
        self.graph = nx.DiGraph()
        self.scaler = RobustScaler()
        self.logger = logging.getLogger(__name__)
        self.config_files = {}  # Stocke les fichiers de configuration par package
        self.isolated_entities = set()  # Stocke les entités isolées

    def parse_project(self):
        """Parse le projet Java et construit le graphe de dépendances"""
        self.logger.info(f"Analyse du projet dans: {self.source_dir}")
        java_files = list(self.source_dir.rglob("*.java"))
        self.logger.info(f"Fichiers Java trouvés: {len(java_files)}")
        
        for file_path in java_files:
            self._process_file(file_path)
            
        self.logger.info(f"Analyse terminée. {len(self.graph.nodes)} composants détectés")
        return len(self.graph.nodes) > 0

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
        
    def _process_file(self, file_path):
        """Traite un fichier Java individuel avec analyse améliorée"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extraction du nom de la classe
            class_pattern = r"(?:public\s+)?(?:class|interface|enum)\s+(\w+)"
            class_match = re.search(class_pattern, content)
            if not class_match:
                return
                
            class_name = class_match.group(1)
            
            # Package extraction avec gestion des sous-packages
            package_pattern = r"package\s+([\w.]+);"
            package_match = re.search(package_pattern, content)
            package = package_match.group(1) if package_match else "default"
            
            # Détection améliorée des types
            type_patterns = [
                (r"@RestController|@Controller", "Controller"),
                (r"@Service", "Service"),
                (r"@Repository", "Repository"),
                (r"@Entity", "Entity"),
                (r"@Configuration", "Configuration"),
                (r"@Component", "Component"),
                (r"@FeatureConfiguration", "FeatureConfig"),  # Configuration spécifique aux features
                (r"@DataConfiguration", "DataConfig"),        # Configuration des données
                (r"@SecurityConfiguration", "SecurityConfig") # Configuration de sécurité
            ]
            
            component_type = None
            for pattern, type_name in type_patterns:
                if re.search(pattern, content):
                    component_type = type_name
                    break
                    
            if not component_type:
                # Analyse approfondie basée sur le nom et le contenu
                if class_name.endswith("Configuration") or "application.properties" in str(file_path) or "application.yml" in str(file_path):
                    component_type = "Configuration"
                elif class_name.endswith("Controller"):
                    component_type = "Controller"
                elif class_name.endswith("Service"):
                    component_type = "Service"
                elif class_name.endswith("Repository"):
                    component_type = "Repository"
                elif class_name.endswith("Entity") or self._has_jpa_annotations(content):
                    component_type = "Entity"
                else:
                    component_type = "Component"
            
            # Gestion spéciale des fichiers de configuration
            if component_type == "Configuration":
                if package not in self.config_files:
                    self.config_files[package] = []
                self.config_files[package].append({
                    'name': class_name,
                    'path': file_path,
                    'content': content
                })
            
            # Ajout du nœud avec métadonnées enrichies
            self.graph.add_node(class_name, 
                              type=component_type,
                              file=str(file_path),
                              package=package,
                              is_config=component_type == "Configuration",
                              dependencies_count=0)
                              
            # Extraction des dépendances
            dependencies = self._extract_enhanced_dependencies(content, class_name)
            
            # Si c'est une entité sans dépendances, on la marque comme isolée
            if component_type == "Entity" and not dependencies:
                self.isolated_entities.add(class_name)
                
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de {file_path}: {str(e)}")
            
    def _extract_enhanced_dependencies(self, content, class_name):
        """Extraction améliorée des dépendances avec analyse du contexte"""
        dependencies = set()
        
        # Analyse des annotations de dépendance
        dependency_patterns = [
            (r'@Autowired\s+(?:private\s+)?(\w+)\s+\w+', 1),
            (r'@Resource\s+(?:private\s+)?(\w+)\s+\w+', 1),
            (r'@Inject\s+(?:private\s+)?(\w+)\s+\w+', 1),
            (r'@ManyToOne\s+(?:private\s+)?(\w+)\s+\w+', 1),
            (r'@OneToMany\s+(?:private\s+)?(\w+)\s+\w+', 1),
            (r'@OneToOne\s+(?:private\s+)?(\w+)\s+\w+', 1),
            (r'@ManyToMany\s+(?:private\s+)?(\w+)\s+\w+', 1)
        ]
        
        for pattern, group in dependency_patterns:
            dependencies.update(re.findall(pattern, content))
        
        # Analyse du constructeur et des méthodes
        constructor_pattern = r'public\s+\w+\(((?:[^)]+))\)'
        constructor_matches = re.findall(constructor_pattern, content)
        for params in constructor_matches:
            param_types = re.findall(r'(\w+)\s+\w+(?:\s*,\s*|$)', params)
            dependencies.update(param_types)
        
        # Analyse des imports avec contexte
        imports = re.findall(r'import\s+([\w.]+\.(\w+));', content)
        for full_import, class_import in imports:
            if class_import not in dependencies and class_import != class_name:
                # Vérifie si la classe importée est effectivement utilisée
                if re.search(fr'\b{class_import}\b', content):
                    dependencies.add(class_import)
        
        # Ajout des arêtes avec métadonnées
        for dep in dependencies:
            if dep != class_name and dep in self.graph:
                self.graph.add_edge(class_name, dep, 
                                  type="dependency",
                                  is_config_dependency=self.graph.nodes[dep].get('is_config', False))
                self.graph.nodes[class_name]['dependencies_count'] += 1
        
        return dependencies
    
    def _has_jpa_annotations(self, content):
        """Vérifie la présence d'annotations JPA"""
        jpa_patterns = [
            r'@Entity\b',
            r'@Table\b',
            r'@Column\b',
            r'@Id\b',
            r'@GeneratedValue\b',
            r'@ManyToOne\b',
            r'@OneToMany\b',
            r'@OneToOne\b',
            r'@ManyToMany\b'
        ]
        return any(re.search(pattern, content) for pattern in jpa_patterns)
    
    def analyze_clustering_constraints(self):
        """Analyse les contraintes de clustering pour les microservices"""
        constraints = {
            'config_dependencies': {},  # Configuration dependencies by package
            'isolated_entities': list(self.isolated_entities),
            'strongly_coupled_components': self._find_strongly_coupled_components(),
            'package_boundaries': self._analyze_package_boundaries(),
            'data_dependencies': self._analyze_data_dependencies()
        }
        
        # Analyse des dépendances de configuration par package
        for package, configs in self.config_files.items():
            related_components = set()
            for config in configs:
                for node in self.graph.nodes():
                    if self.graph.has_edge(node, config['name']):
                        related_components.add(node)
            constraints['config_dependencies'][package] = list(related_components)
        
        return constraints
    
    def _find_strongly_coupled_components(self):
        """Identifie les composants fortement couplés"""
        strongly_coupled = []
        for node in self.graph.nodes():
            deps_count = self.graph.nodes[node]['dependencies_count']
            if deps_count > 5:  # Seuil arbitraire, à ajuster selon le contexte
                strongly_coupled.append({
                    'component': node,
                    'dependencies': deps_count,
                    'type': self.graph.nodes[node]['type']
                })
        return strongly_coupled
    
    def _analyze_package_boundaries(self):
        """Analyse les frontières des packages"""
        package_boundaries = {}
        for node in self.graph.nodes():
            package = self.graph.nodes[node]['package']
            if package not in package_boundaries:
                package_boundaries[package] = {'components': [], 'external_deps': 0}
            package_boundaries[package]['components'].append(node)
            
            # Compte les dépendances externes
            for neighbor in self.graph.neighbors(node):
                if self.graph.nodes[neighbor]['package'] != package:
                    package_boundaries[package]['external_deps'] += 1
        
        return package_boundaries
    
    def _analyze_data_dependencies(self):
        """Analyse les dépendances de données entre entités"""
        data_deps = {}
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == 'Entity':
                data_deps[node] = {
                    'related_entities': [],
                    'accessed_by': []
                }
                
                # Cherche les relations avec d'autres entités
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]['type'] == 'Entity':
                        data_deps[node]['related_entities'].append(neighbor)
                    else:
                        data_deps[node]['accessed_by'].append(neighbor)
        
        return data_deps

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