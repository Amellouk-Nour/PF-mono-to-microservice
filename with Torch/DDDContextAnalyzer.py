from collections import defaultdict
from pathlib import Path
import re
import logging
from typing import Dict, Set, List

class DDDContextAnalyzer:
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.bounded_contexts = {}
        self.ubiquitous_language = {}
        self.aggregates = {}
        self.domain_events = {}
        
    def analyze_domain_contexts(self):
        """Analyse les bounded contexts à partir du code source"""
        for file_path in Path(self.source_dir).rglob("*.java"):
            self._analyze_file_for_ddd_patterns(file_path)

    def _analyze_package_context(self, package_name):
        """
        Analyse le nom du package pour identifier le contexte métier potentiel.
        
        Args:
            package_name (str): Le nom complet du package
            
        Returns:
            str: Le nom du package possiblement modifié pour refléter le contexte métier
        """
        # Liste de mots-clés indiquant un contexte métier
        business_contexts = {
            'order': 'sales.orders',
            'payment': 'finance.payments',
            'customer': 'crm',
            'product': 'catalog',
            'inventory': 'warehouse',
            'user': 'identity',
            'auth': 'security',
            'notification': 'communication',
            'shipping': 'logistics',
            'billing': 'finance.billing'
        }
        
        # Normalisation du nom de package
        package_parts = package_name.lower().split('.')
        
        # Recherche de contextes métier dans le nom du package
        for part in package_parts:
            for context_key, context_value in business_contexts.items():
                if context_key in part:
                    # Si on trouve un contexte métier, on le met en avant dans le nom du package
                    return f"{context_value}.{package_name}"
                    
        return package_name
            
    def _extract_class_name(self, content):
        """Extrait le nom de la classe du contenu Java"""
        pattern = r"(?:public\s+)?(?:class|interface|enum)\s+(\w+)"
        match = re.search(pattern, content)
        return match.group(1) if match else None

    def _find_entities(self, content):
        """Trouve les entités associées"""
        pattern = r"@Entity\s+class\s+(\w+)"
        return set(re.findall(pattern, content))

    def _find_value_objects(self, content):
        """Trouve les objets valeur"""
        pattern = r"@Value\s+class\s+(\w+)"
        return set(re.findall(pattern, content))

    def _find_domain_events(self, content):
        """Trouve les événements domaine"""
        pattern = r"@DomainEvent\s+class\s+(\w+)"
        return set(re.findall(pattern, content))

    def _extract_package_name(self, content):
        """
        Extrait le nom du package d'un fichier Java en prenant en compte
        différentes conventions de nommage et structures de packages.
        
        Args:
            content (str): Le contenu du fichier Java
            
        Returns:
            str: Le nom du package, ou 'default' si non trouvé
        """
        # Patterns possibles pour la déclaration de package
        patterns = [
            # Pattern standard
            r'package\s+([\w.]+);',
            # Pattern avec commentaires possibles entre package et nom
            r'package\s+(?:/\*.*?\*/\s+)?([\w.]+);',
            # Pattern avec annotations
            r'@[\w.]+\s+package\s+([\w.]+);'
        ]
        
        try:
            # Recherche ligne par ligne pour éviter les faux positifs dans les commentaires
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("package ") or line.startswith("@"):
                    for pattern in patterns:
                        match = re.search(pattern, line)
                        if match:
                            package_name = match.group(1)
                            # Analyse du package pour détection du contexte métier
                            return self._analyze_package_context(package_name)
            
            # Essai de déduction à partir du chemin du fichier si pas de déclaration explicite
            file_path = self.source_dir
            if file_path:
                relative_path = str(file_path).replace(str(self.source_dir), '').strip('/')
                if relative_path and relative_path.endswith('.java'):
                    path_parts = relative_path.split('/')
                    if len(path_parts) > 1:
                        # Exclure le nom du fichier et convertir le chemin en nom de package
                        package_parts = path_parts[:-1]
                        return '.'.join(package_parts)
            
            return 'default'
            
        except Exception as e:
            logging.warning(f"Erreur lors de l'extraction du package: {str(e)}")
            return 'default'
            
            
    def _analyze_file_for_ddd_patterns(self, file_path):
        """Analyse un fichier pour les patterns DDD"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Détection des agrégats
        if '@Aggregate' in content or 'implements AggregateRoot' in content:
            aggregate_name = self._extract_class_name(content)
            self.aggregates[aggregate_name] = {
                'entities': self._find_entities(content),
                'value_objects': self._find_value_objects(content),
                'domain_events': self._find_domain_events(content)
            }
            
        # Détection des bounded contexts via les packages et commentaires
        package_name = self._extract_package_name(content)
        context_hints = self._analyze_context_hints(content, package_name)
        
        if context_hints:
            if package_name not in self.bounded_contexts:
                self.bounded_contexts[package_name] = {
                    'ubiquitous_language': set(),
                    'aggregates': set(),
                    'domain_events': set(),
                    'entities': set(),
                }
            self.bounded_contexts[package_name].update(context_hints)
            
    def _analyze_context_hints(self, content, package_name):
        """Analyse les indices de contexte métier"""
        hints = {
            'ubiquitous_language': set(),
            'aggregates': set(),
            'domain_events': set(),
            'entities': set()
        }
        
        # Analyse des commentaires pour le vocabulaire métier
        domain_terms = self._extract_domain_terms(content)
        hints['ubiquitous_language'].update(domain_terms)
        
        # Détection des patterns d'agrégats
        aggregate_pattern = r'@Aggregate\s+class\s+(\w+)'
        aggregates = re.findall(aggregate_pattern, content)
        hints['aggregates'].update(aggregates)
        
        # Détection des événements domaine
        event_pattern = r'@DomainEvent\s+class\s+(\w+)'
        events = re.findall(event_pattern, content)
        hints['domain_events'].update(events)
        
        return hints
        
    def _extract_domain_terms(self, content):
        """Extrait les termes du domaine des commentaires et noms de classe"""
        domain_terms = set()
        
        # Analyse des commentaires Javadoc pour les termes métier
        javadoc_pattern = r'/\*\*\s*(.*?)\s*\*/'
        javadocs = re.findall(javadoc_pattern, content, re.DOTALL)
        
        for javadoc in javadocs:
            # Recherche de termes métier dans les tags @DomainTerm ou @BusinessTerm
            domain_term_pattern = r'@(?:Domain|Business)Term\s+([^@\n]*)'
            terms = re.findall(domain_term_pattern, javadoc)
            domain_terms.update(terms)
            
        # Analyse des noms de classes et méthodes pour les termes métier
        class_name = self._extract_class_name(content)
        if class_name:
            parts = re.findall('[A-Z][a-z]*', class_name)
            domain_terms.update(parts)
            
        return domain_terms
        
    def enhance_clustering_constraints(self, existing_constraints):
        """Enrichit les contraintes de clustering avec les aspects DDD"""
        ddd_constraints = {
            'bounded_contexts': self.bounded_contexts,
            'aggregates': self.aggregates,
            'ubiquitous_language': self.ubiquitous_language,
            'domain_events': self.domain_events
        }
        
        # Fusion avec les contraintes existantes
        enhanced_constraints = {
            **existing_constraints,
            'ddd': ddd_constraints
        }
        
        # Ajout de règles de cohésion basées sur DDD
        enhanced_constraints['cohesion_rules'] = self._generate_cohesion_rules()
        
        return enhanced_constraints
        
    def _generate_cohesion_rules(self):
        """Génère des règles de cohésion basées sur les patterns DDD"""
        rules = []
        
        # Règle 1: Les agrégats et leurs entités doivent rester ensemble
        for aggregate, details in self.aggregates.items():
            rules.append({
                'type': 'must_be_together',
                'components': [aggregate] + list(details['entities']),
                'reason': 'Aggregate integrity'
            })
            
        # Règle 2: Les bounded contexts suggèrent des frontières de services
        for context, details in self.bounded_contexts.items():
            rules.append({
                'type': 'preferred_boundary',
                'context': context,
                'components': list(details['aggregates']),
                'reason': 'Bounded context separation'
            })
            
        # Règle 3: Les composants partageant le même vocabulaire métier
        shared_terms = defaultdict(list)
        for context, details in self.bounded_contexts.items():
            for term in details['ubiquitous_language']:
                shared_terms[term].append(context)
                
        for term, contexts in shared_terms.items():
            if len(contexts) > 1:
                rules.append({
                    'type': 'semantic_coupling',
                    'term': term,
                    'contexts': contexts,
                    'reason': 'Shared ubiquitous language'
                })
                
        return rules