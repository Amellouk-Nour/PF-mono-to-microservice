from MonolithDataPreparer import MonolithDataPreparer

# Exemple de chemin vers le projet Java
source_dir = "projet_SI_gestion_ECM-main"

# Initialisation
data_preparer = MonolithDataPreparer(source_dir)

# Étape 1 : Analyse et extraction des données
data_preparer.parse_project()

# Étape 2 : Préparation pour le GNN
data = data_preparer.prepare_for_gnn()
print(f"Features des nœuds : {data.x.shape}")
print(f"Index des arêtes : {data.edge_index.shape}")

# Étape 3 : Export et visualisation
data_preparer.export_graph("input-graph/monolith.graphml")
data_preparer.visualize_graph()
