# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import shuffle
from scipy.spatial.distance import pdist, cdist
from scipy.stats import entropy
import re
import gc
import psutil
import os
from collections import Counter, defaultdict
import warnings
from sentence_transformers.util import cos_sim
import itertools
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Import sentence transformers - required for job skills clustering
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ SentenceTransformers available - perfect for job skills clustering")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("❌ SentenceTransformers required for job skills clustering!")
    print("   Install with: pip install sentence-transformers")
    raise ImportError("SentenceTransformers is required for job skills clustering. Install with: pip install sentence-transformers")

# Try to import transformers for additional models
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available for additional embedding options")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available (optional). Install with: pip install transformers torch")

class GridSearchJobSkillsClusterer:
    def __init__(self, memory_limit_gb=4, batch_size=1000, 
                 embedding_models=['auto'], clustering_algorithms=['kmeans'],
                 evaluation_sample_size=5000):
        """
        Enhanced clusterer with grid search capabilities
        
        Args:
            embedding_models: List of embedding models to test
            clustering_algorithms: List of clustering algorithms to test
            evaluation_sample_size: Sample size for evaluation metrics
        """
        self.memory_limit_gb = memory_limit_gb
        self.batch_size = batch_size
        self.embedding_models = embedding_models
        self.clustering_algorithms = clustering_algorithms
        self.evaluation_sample_size = evaluation_sample_size
        
        # Initialize embedding models
        self.embedders = {}
        self._initialize_embedding_models()

        self.skill_categories = self._define_skill_categories()
        self.skill_synonyms = self._define_skill_synonyms()
        
        # Results storage for grid search
        self.grid_search_results = []
        self.best_model = None
        self.best_score = -1
        self.evaluation_metrics = {}
        
    def _initialize_embedding_models(self):
        """Initialize multiple embedding models for grid search"""
        
        model_configs = {
            'auto': [
                'all-MiniLM-L6-v2',        # Fast, excellent for skills (384 dim)
                'all-mpnet-base-v2',       # Best overall quality (768 dim)
            ],
            'fast': ['all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2'],
            'quality': ['all-mpnet-base-v2', 'all-distilroberta-v1'],
            'multilingual': ['paraphrase-multilingual-MiniLM-L12-v2'],
            'jina': ['jinaai/jina-embeddings-v3'],
            'specialized': ['sentence-transformers/all-roberta-large-v1']
        }
        
        models_to_load = []
        for model_spec in self.embedding_models:
            if model_spec in model_configs:
                models_to_load.extend(model_configs[model_spec])
            else:
                models_to_load.append(model_spec)
        
        # Remove duplicates while preserving order
        models_to_load = list(dict.fromkeys(models_to_load))
        
        print(f"Loading {len(models_to_load)} embedding models for grid search...")
        
        for model_name in models_to_load:
            try:
                print(f"Loading {model_name}...")
                embedder = SentenceTransformer(model_name, trust_remote_code=True)
                embedding_dim = embedder.get_sentence_embedding_dimension()
                self.embedders[model_name] = embedder
                print(f"✅ Loaded {model_name} (dim: {embedding_dim})")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        if not self.embedders:
            raise RuntimeError("Could not initialize any embedding models")
        
        print(f"Successfully loaded {len(self.embedders)} embedding models")

    def _define_skill_synonyms(self):
        """Define skill synonyms and variations for better matching"""
        return {
            'javascript': ['js', 'ecmascript', 'es6', 'es2015'],
            'machine learning': ['ml', 'artificial intelligence', 'ai'],
            'user interface': ['ui', 'frontend', 'front-end'],
            'user experience': ['ux', 'usability', 'user-centered design'],
            'database': ['db', 'data storage', 'data management'],
            'application programming interface': ['api', 'rest api', 'restful'],
            'continuous integration': ['ci', 'continuous deployment', 'cd'],
            'search engine optimization': ['seo', 'organic search'],
            'customer relationship management': ['crm'],
            'enterprise resource planning': ['erp']
        }

    def _define_skill_categories(self):
        """Define comprehensive job skill categories"""
        return {
            'Programming Languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift',
                'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'typescript', 'perl', 'shell'
            ],
            'Web Development': [
                'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
                'laravel', 'rails', 'bootstrap', 'jquery', 'webpack', 'sass', 'less'
            ],
            'Data Science & Analytics': [
                'machine learning', 'deep learning', 'data analysis', 'statistics', 'pandas',
                'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'tableau', 'power bi',
                'excel', 'data visualization', 'predictive modeling', 'regression', 'classification'
            ],
            'Database Technologies': [
                'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server', 'redis', 'elasticsearch',
                'cassandra', 'dynamodb', 'sqlite', 'database design', 'data modeling'
            ],
            'Cloud & DevOps': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins',
                'git', 'ci/cd', 'devops', 'microservices', 'serverless', 'infrastructure'
            ],
            'Mobile Development': [
                'ios', 'android', 'react native', 'flutter', 'xamarin', 'swift', 'kotlin',
                'mobile app development', 'app store', 'mobile ui/ux'
            ],
            'Design & UX': [
                'ui/ux design', 'graphic design', 'photoshop', 'illustrator', 'figma', 'sketch',
                'user experience', 'user interface', 'wireframing', 'prototyping', 'design thinking'
            ],
            'Project Management': [
                'agile', 'scrum', 'kanban', 'project management', 'jira', 'confluence', 'trello',
                'stakeholder management', 'risk management', 'budget management'
            ],
            'Business & Strategy': [
                'business analysis', 'strategic planning', 'market research', 'competitive analysis',
                'financial modeling', 'roi analysis', 'stakeholder engagement', 'process improvement'
            ],
            'Communication & Leadership': [
                'leadership', 'team management', 'communication', 'presentation', 'public speaking',
                'mentoring', 'coaching', 'negotiation', 'conflict resolution', 'collaboration'
            ],
            'Security & Compliance': [
                'cybersecurity', 'information security', 'compliance', 'risk assessment', 'penetration testing',
                'security auditing', 'gdpr', 'hipaa', 'encryption', 'firewall'
            ],
            'Quality Assurance': [
                'testing', 'qa', 'test automation', 'selenium', 'unit testing', 'integration testing',
                'performance testing', 'bug tracking', 'test planning'
            ]
        }
    
    def get_memory_usage(self):
        """Monitor current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024 / 1024  # GB
        except:
            return 0.0
    
    def preprocess_skills(self, skills):
        """Specialized preprocessing for job skills"""
        processed = []
        
        for skill in skills:
            if pd.isna(skill) or skill == '':
                processed.append('')
                continue
            
            # Convert to string and lowercase
            skill = str(skill).lower().strip()
            
            # Remove common prefixes/suffixes
            skill = re.sub(r'^(experience (with|in)|knowledge of|proficiency in|skilled in)', '', skill)
            skill = re.sub(r'(skills?|experience|knowledge|proficiency)$', '', skill)
            
            # Clean up punctuation and extra spaces
            skill = re.sub(r'[^\w\s\-\+\#\.]', ' ', skill)  # Keep - + # . for tech terms
            skill = re.sub(r'\s+', ' ', skill).strip()
            
            # Handle empty after cleaning
            if not skill:
                processed.append('')
                continue
            
            # Expand common abbreviations and synonyms
            skill = self._expand_skill_synonyms(skill)
            
            processed.append(skill)
        
        return processed
    
    def _expand_skill_synonyms(self, skill):
        """Expand skill abbreviations and synonyms"""
        # Check for exact matches first
        for full_form, abbreviations in self.skill_synonyms.items():
            if skill in abbreviations:
                return full_form
        
        # Check for partial matches
        for category, terms in self.skill_synonyms.items():
            for term in terms:
                if term in skill:
                    skill = skill.replace(term, category)
                    break
        
        return skill
    
    def create_skill_embeddings(self, skills, model_name, batch_size=None):
        """Create embeddings for a specific model"""
        if batch_size is None:
            batch_size = min(self.batch_size, 64)
        
        print(f"Creating embeddings using {model_name} for {len(skills)} skills...")
        
        embedder = self.embedders[model_name]
        embeddings = []
        
        for i in range(0, len(skills), batch_size):
            batch_skills = skills[i:i + batch_size]
            
            try:
                if 'jina' in model_name:
                    task = "text-matching"
                    batch_embeddings = embedder.encode(
                        batch_skills,
                        task=task,
                        prompt_name='passage',
                        batch_size=min(batch_size, 64),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                else:
                    batch_embeddings = embedder.encode(
                        batch_skills,
                        batch_size=min(batch_size, 64),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                embeddings.append(batch_embeddings)
                
                if i % (batch_size * 20) == 0:
                    current_memory = self.get_memory_usage()
                    print(f"Embedded {min(i + batch_size, len(skills))}/{len(skills)} skills - Memory: {current_memory:.2f} GB")
                    gc.collect()
                    
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                continue
        
        final_embeddings = np.vstack(embeddings)
        print(f"Created embeddings shape: {final_embeddings.shape}")
        
        return final_embeddings
    
    def smart_dimensionality_reduction(self, X, target_dim=50):
        """Dimensionality reduction optimized for job skills"""
        n_samples, n_features = X.shape
        
        print(f"Input shape: {n_samples} x {n_features}")
        
        # Only reduce if very high dimensional
        if n_features <= 512:
            print("Keeping original embedding dimensions")
            return X, None
        
        target_dim = min(target_dim, n_features // 3)
        
        if n_samples > 50000:
            print(f"Using IncrementalPCA for dimensionality reduction to {target_dim}...")
            reducer = IncrementalPCA(n_components=target_dim, batch_size=min(1000, n_samples//10))
            
            chunk_size = min(2000, n_samples)
            for i in range(0, n_samples, chunk_size):
                chunk = X[i:i+chunk_size]
                reducer.partial_fit(chunk)
                gc.collect()
            
            X_reduced = reducer.transform(X)
        else:
            print(f"Using PCA for dimensionality reduction to {target_dim}...")
            reducer = PCA(n_components=target_dim, random_state=42)
            X_reduced = reducer.fit_transform(X)
        
        print(f"Reduced to shape: {X_reduced.shape}")
        return X_reduced, reducer
    
    def estimate_optimal_clusters(self, X, max_k=50, sample_size=10000, algo='kmeans'):
        """Estimate optimal clusters with different algorithms"""
        n_samples = X.shape[0]
        np.random.seed(42)
        
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        print(f"Estimating optimal clusters using {X_sample.shape[0]} samples with {algo}...")
        
        max_k = min(max_k, X_sample.shape[0] // 5)
        k_values = list(range(5, min(max_k + 1, 51), 2))  # Test 5, 7, 9, ..., up to 50
        
        best_score = -1
        best_k = 10  # Default for skills
        scores = []
        
        for k in k_values:
            try:
                if algo == 'kmeans':
                    clusterer = KMeans(n_clusters=k, random_state=42, n_init=5)
                elif algo == 'gmm':
                    clusterer = GaussianMixture(n_components=k, random_state=42, n_init=3)
                elif algo == 'agglomerative':
                    clusterer = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
                
                if algo == 'gmm':
                    labels = clusterer.fit_predict(X_sample)
                else:
                    labels = clusterer.fit_predict(X_sample)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X_sample, labels, sample_size=min(2000, X_sample.shape[0]))
                    scores.append((k, score))
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
            except Exception as e:
                print(f"Error with k={k}: {e}")
                continue
        
        print(f"Optimal clusters for {algo}: {best_k} (score: {best_score:.3f})")
        return best_k, scores
    
    def perform_clustering(self, X, n_clusters, algo='kmeans'):
        """Perform clustering with specified algorithm"""
        print(f"Clustering with {algo} into {n_clusters} clusters...")
        
        if algo == 'kmeans':
            if X.shape[0] > 100000:
                clusterer = MiniBatchKMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    batch_size=min(10000, X.shape[0] // 2),
                    max_iter=200,
                    n_init=5
                )
            else:
                clusterer = KMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    n_init=10,
                    max_iter=300
                )
        elif algo == 'gmm':
            clusterer = GaussianMixture(
                n_components=n_clusters, 
                random_state=42,
                n_init=5
            )
        elif algo == 'agglomerative':
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        labels = clusterer.fit_predict(X)
        return labels, clusterer
    
    def comprehensive_evaluation(self, X, labels, skills=None, run_id=None):
        """Comprehensive evaluation of clustering quality"""
        print(f"\\n=== COMPREHENSIVE EVALUATION {run_id or ''} ===")
        
        evaluation_results = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(labels),
            'n_clusters': len(set(labels[labels >= 0])),
            'n_noise': np.sum(labels == -1)
        }
        
        # Sample for evaluation if too large
        n_samples = X.shape[0]
        if n_samples > self.evaluation_sample_size:
            indices = np.random.choice(n_samples, self.evaluation_sample_size, replace=False)
            X_eval = X[indices]
            labels_eval = labels[indices]
        else:
            X_eval = X
            labels_eval = labels
        
        # 1. Internal Metrics
        print("1. Internal Evaluation Metrics:")
        try:
            if len(set(labels_eval)) > 1:
                sil_score = silhouette_score(X_eval, labels_eval, sample_size=min(2000, len(X_eval)))
                ch_score = calinski_harabasz_score(X_eval, labels_eval)
                db_score = davies_bouldin_score(X_eval, labels_eval)
                
                evaluation_results.update({
                    'silhouette_score': sil_score,
                    'calinski_harabasz_score': ch_score,
                    'davies_bouldin_score': db_score
                })
                
                print(f"   Silhouette Score: {sil_score:.4f}")
                print(f"   Calinski-Harabasz Score: {ch_score:.2f}")
                print(f"   Davies-Bouldin Score: {db_score:.4f}")
            else:
                print("   Cannot compute internal metrics: only one cluster")
                evaluation_results.update({
                    'silhouette_score': -1,
                    'calinski_harabasz_score': -1,
                    'davies_bouldin_score': float('inf')
                })
        except Exception as e:
            print(f"   Error computing internal metrics: {e}")
            evaluation_results.update({
                'silhouette_score': -1,
                'calinski_harabasz_score': -1,
                'davies_bouldin_score': float('inf')
            })
        
        # 2. Distance Analysis
        print("\\n2. Distance Analysis:")
        distance_results = self._analyze_distances(X_eval, labels_eval)
        evaluation_results.update(distance_results)
        
        # 3. Cluster Balance Analysis
        print("\\n3. Cluster Balance Analysis:")
        balance_results = self._analyze_cluster_balance(labels)
        evaluation_results.update(balance_results)
        
        # 4. Qualitative Analysis (if skills provided)
        if skills is not None:
            print("\\n4. Qualitative Analysis:")
            qualitative_results = self._analyze_cluster_quality(skills, labels)
            evaluation_results.update(qualitative_results)
        
        # 5. Stability Analysis (simplified)
        print("\\n5. Stability Metrics:")
        stability_results = self._analyze_stability(X_eval, labels_eval)
        evaluation_results.update(stability_results)
        
        # 6. Composite Score
        composite_score = self._calculate_composite_score(evaluation_results)
        evaluation_results['composite_score'] = composite_score
        
        print(f"\\n📊 COMPOSITE SCORE: {composite_score:.4f}")
        
        return evaluation_results
    
    def _analyze_distances(self, X, labels):
        """Analyze intra-cluster vs inter-cluster distances"""
        try:
            unique_clusters = np.unique(labels[labels >= 0])
            if len(unique_clusters) < 2:
                return {'separation_ratio': 0, 'avg_intra_distance': 0, 'avg_inter_distance': 0}
            
            intra_distances = []
            inter_distances = []
            
            # Calculate intra-cluster distances
            for cluster in unique_clusters:
                cluster_mask = labels == cluster
                cluster_points = X[cluster_mask]
                
                if len(cluster_points) > 1:
                    distances = pdist(cluster_points, metric='cosine')
                    intra_distances.extend(distances)
            
            # Calculate inter-cluster distances (sample to avoid memory issues)
            if len(unique_clusters) > 1:
                for i, cluster1 in enumerate(unique_clusters):
                    for cluster2 in unique_clusters[i+1:]:
                        mask1 = labels == cluster1
                        mask2 = labels == cluster2
                        
                        points1 = X[mask1]
                        points2 = X[mask2]
                        
                        # Sample points if clusters are large
                        if len(points1) > 100:
                            points1 = points1[np.random.choice(len(points1), 100, replace=False)]
                        if len(points2) > 100:
                            points2 = points2[np.random.choice(len(points2), 100, replace=False)]
                        
                        distances = cdist(points1, points2, metric='cosine')
                        inter_distances.extend(distances.flatten())
            
            avg_intra = np.mean(intra_distances) if intra_distances else 0
            avg_inter = np.mean(inter_distances) if inter_distances else 0
            separation_ratio = avg_inter / avg_intra if avg_intra > 0 else 0
            
            print(f"   Average intra-cluster distance: {avg_intra:.4f}")
            print(f"   Average inter-cluster distance: {avg_inter:.4f}")
            print(f"   Separation ratio: {separation_ratio:.4f}")
            
            return {
                'separation_ratio': separation_ratio,
                'avg_intra_distance': avg_intra,
                'avg_inter_distance': avg_inter
            }
        except Exception as e:
            print(f"   Error in distance analysis: {e}")
            return {'separation_ratio': 0, 'avg_intra_distance': 0, 'avg_inter_distance': 0}
    
    def _analyze_cluster_balance(self, labels):
        """Analyze cluster size distribution"""
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        
        if len(counts) == 0:
            return {'cluster_balance_ratio': float('inf'), 'largest_cluster_ratio': 1, 'smallest_cluster_size': 0}
        
        balance_ratio = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else float('inf')
        largest_cluster_ratio = np.max(counts) / len(labels)
        smallest_cluster_size = np.min(counts)
        
        print(f"   Cluster sizes: {dict(zip(unique, counts))}")
        print(f"   Balance ratio (lower=better): {balance_ratio:.4f}")
        print(f"   Largest cluster ratio: {largest_cluster_ratio:.4f}")
        print(f"   Smallest cluster size: {smallest_cluster_size}")
        
        return {
            'cluster_balance_ratio': balance_ratio,
            'largest_cluster_ratio': largest_cluster_ratio,
            'smallest_cluster_size': smallest_cluster_size,
            'cluster_sizes': dict(zip(unique.tolist(), counts.tolist()))
        }
    
    def _analyze_cluster_quality(self, skills, labels):
        """Analyze qualitative aspects of clusters"""
        try:
            unique_clusters = np.unique(labels[labels >= 0])
            coherence_scores = []
            
            for cluster in unique_clusters[:10]:  # Analyze top 10 clusters
                cluster_mask = labels == cluster
                cluster_skills = [skills[i] for i in range(len(skills)) if cluster_mask[i]]
                
                if len(cluster_skills) < 2:
                    continue
                
                # Simple coherence based on word overlap
                all_words = []
                for skill in cluster_skills:
                    words = str(skill).lower().split()
                    all_words.extend(words)
                
                word_freq = Counter(all_words)
                total_words = len(all_words)
                unique_words = len(word_freq)
                
                # Coherence proxy: ratio of repeated words
                coherence = 1 - (unique_words / total_words) if total_words > 0 else 0
                coherence_scores.append(coherence)
            
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
            
            print(f"   Average cluster coherence: {avg_coherence:.4f}")
            
            return {'cluster_coherence': avg_coherence}
        except Exception as e:
            print(f"   Error in qualitative analysis: {e}")
            return {'cluster_coherence': 0}
    
    def _analyze_stability(self, X, labels):
        """Analyze clustering stability"""
        try:
            # Simple stability: resample and check consistency
            n_samples = min(1000, X.shape[0])
            if X.shape[0] <= n_samples:
                return {'stability_score': 1.0}
            
            # Take two random samples
            indices1 = np.random.choice(X.shape[0], n_samples, replace=False)
            indices2 = np.random.choice(X.shape[0], n_samples, replace=False)
            
            # Calculate overlap in cluster assignments for common indices
            common_indices = np.intersect1d(indices1, indices2)
            if len(common_indices) < 10:
                return {'stability_score': 0.5}
            
            # Simple stability metric
            stability = len(common_indices) / n_samples
            
            print(f"   Stability score: {stability:.4f}")
            
            return {'stability_score': stability}
        except Exception as e:
            print(f"   Error in stability analysis: {e}")
            return {'stability_score': 0.5}
    
    def _calculate_composite_score(self, results):
        """Calculate a composite score for ranking different runs"""
        try:
            # Normalize and weight different metrics
            weights = {
                'silhouette_score': 0.25,      # Internal quality
                'separation_ratio': 0.20,      # Cluster separation
                'cluster_balance': 0.15,       # Balance (inverted)
                'cluster_coherence': 0.15,     # Qualitative
                'stability_score': 0.10,       # Stability
                'size_penalty': 0.15           # Penalty for very small/large clusters
            }
            
            score = 0
            
            # Silhouette score (0 to 1, higher better)
            sil = max(0, min(1, results.get('silhouette_score', 0)))
            score += weights['silhouette_score'] * sil
            
            # Separation ratio (normalize to 0-1, cap at 3)
            sep = min(1, results.get('separation_ratio', 0) / 3)
            score += weights['separation_ratio'] * sep
            
            # Cluster balance (inverted, lower is better)
            balance = results.get('cluster_balance_ratio', float('inf'))
            balance_score = max(0, 1 - min(1, balance / 2))  # Cap at 2, invert
            score += weights['cluster_balance'] * balance_score
            
            # Coherence (0 to 1, higher better)
            coherence = max(0, min(1, results.get('cluster_coherence', 0)))
            score += weights['cluster_coherence'] * coherence
            
            # Stability (0 to 1, higher better)
            stability = max(0, min(1, results.get('stability_score', 0.5)))
            score += weights['stability_score'] * stability
            
            # Size penalty (penalize very unbalanced cluster sizes)
            largest_ratio = results.get('largest_cluster_ratio', 0.5)
            smallest_size = results.get('smallest_cluster_size', 10)
            size_penalty = 1 - max(0, largest_ratio - 0.5) * 2  # Penalize if >50% in one cluster
            size_penalty *= min(1, smallest_size / 5)  # Penalize very small clusters
            score += weights['size_penalty'] * size_penalty
            
            return max(0, min(1, score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            print(f"Error calculating composite score: {e}")
            return 0.0
    
    def grid_search_clustering(self, skills, n_clusters=None, auto_optimize=True):
        """
        Perform grid search over embedding models and clustering algorithms
        """
        print("🔍 STARTING GRID SEARCH CLUSTERING")
        print("=" * 50)
        
        start_time = time.time()
        
        # Preprocess skills once
        print("Preprocessing skills...")
        processed_skills = self.preprocess_skills(skills)
        
        # Track valid skills
        valid_skills = []
        original_to_processed = {}
        
        for i, skill in enumerate(processed_skills):
            if skill.strip():
                original_to_processed[i] = len(valid_skills)
                valid_skills.append(skill)
        
        print(f"Valid skills: {len(valid_skills)}/{len(skills)}")
        
        if len(valid_skills) < 10:
            print("❌ Too few valid skills for meaningful clustering")
            return None
        
        # Grid search parameters
        embedding_models = list(self.embedders.keys())
        clustering_algorithms = self.clustering_algorithms
        
        total_combinations = len(embedding_models) * len(clustering_algorithms)
        print(f"Testing {total_combinations} combinations:")
        print(f"  Embedding models: {embedding_models}")
        print(f"  Clustering algorithms: {clustering_algorithms}")
        
        # Store all results
        all_results = []
        run_counter = 1
        
        # Grid search loop
        for model_name in embedding_models:
            print(f"\\n🤖 EMBEDDING MODEL: {model_name}")
            print("-" * 40)
            
            # Create embeddings for this model
            try:
                embeddings = self.create_skill_embeddings(valid_skills, model_name)
                
                # Dimensionality reduction
                X_reduced, reducer = self.smart_dimensionality_reduction(embeddings)
                
                # Test each clustering algorithm
                for algo in clustering_algorithms:
                    print(f"\\n🎯 RUN {run_counter}/{total_combinations}: {model_name} + {algo}")
                    print("-" * 30)
                    
                    try:
                        # Optimize cluster number if requested
                        if auto_optimize and n_clusters is None:
                            optimal_k, _ = self.estimate_optimal_clusters(X_reduced, algo=algo)
                        else:
                            optimal_k = n_clusters or min(15, max(5, len(valid_skills) // 50))
                        
                        # Perform clustering
                        labels, clusterer = self.perform_clustering(X_reduced, optimal_k, algo)
                        
                        # Map results back to original indices
                        full_labels = np.full(len(skills), -1, dtype=int)
                        for original_idx, processed_idx in original_to_processed.items():
                            if processed_idx < len(labels):
                                full_labels[original_idx] = labels[processed_idx]
                        
                        # Comprehensive evaluation
                        run_id = f"{model_name}_{algo}"
                        evaluation = self.comprehensive_evaluation(
                            X_reduced, labels, valid_skills, run_id
                        )
                        
                        # Store complete results
                        result = {
                            'run_id': run_id,
                            'run_number': run_counter,
                            'embedding_model': model_name,
                            'clustering_algorithm': algo,
                            'n_clusters': optimal_k,
                            'labels': full_labels,
                            'clusterer': clusterer,
                            'embeddings': embeddings,
                            'reduced_features': X_reduced,
                            'evaluation': evaluation,
                            'processing_time': time.time() - start_time
                        }
                        
                        all_results.append(result)
                        
                        print(f"✅ Run {run_counter} completed - Composite Score: {evaluation['composite_score']:.4f}")
                        
                    except Exception as e:
                        print(f"❌ Run {run_counter} failed: {e}")
                        continue
                    
                    run_counter += 1
                    gc.collect()  # Clean up memory
                
                # Clean up embeddings after testing all algorithms
                del embeddings, X_reduced
                gc.collect()
                
            except Exception as e:
                print(f"❌ Failed to create embeddings for {model_name}: {e}")
                continue
        
        # Store results and find best model
        self.grid_search_results = all_results
        
        if not all_results:
            print("❌ No successful runs completed")
            return None
        
        # Find best model based on composite score
        best_result = max(all_results, key=lambda x: x['evaluation']['composite_score'])
        self.best_model = best_result
        self.best_score = best_result['evaluation']['composite_score']
        
        total_time = time.time() - start_time
        
        print(f"\\n🏆 GRID SEARCH COMPLETED!")
        print("=" * 50)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Successful runs: {len(all_results)}/{total_combinations}")
        print(f"Best model: {best_result['embedding_model']} + {best_result['clustering_algorithm']}")
        print(f"Best score: {self.best_score:.4f}")
        
        return best_result['labels']
    
    def print_detailed_results(self):
        """Print detailed comparison of all grid search results"""
        if not self.grid_search_results:
            print("No results to display")
            return
        
        print("\\n📊 DETAILED GRID SEARCH RESULTS")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison_data = []
        for result in self.grid_search_results:
            eval_data = result['evaluation']
            comparison_data.append({
                'Run': result['run_number'],
                'Model': result['embedding_model'],
                'Algorithm': result['clustering_algorithm'],
                'Clusters': result['n_clusters'],
                'Composite Score': eval_data['composite_score'],
                'Silhouette': eval_data.get('silhouette_score', 0),
                'Separation': eval_data.get('separation_ratio', 0),
                'Balance': eval_data.get('cluster_balance_ratio', float('inf')),
                'Coherence': eval_data.get('cluster_coherence', 0),
                'Stability': eval_data.get('stability_score', 0),
                'Time (s)': result['processing_time']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Composite Score', ascending=False)
        
        print("\\n🥇 TOP 5 BEST PERFORMING COMBINATIONS:")
        print(df.head().to_string(index=False, float_format='%.4f'))
        
        print("\\n📈 PERFORMANCE BY EMBEDDING MODEL:")
        model_performance = df.groupby('Model')['Composite Score'].agg(['mean', 'std', 'max']).round(4)
        print(model_performance)
        
        print("\\n🎯 PERFORMANCE BY CLUSTERING ALGORITHM:")
        algo_performance = df.groupby('Algorithm')['Composite Score'].agg(['mean', 'std', 'max']).round(4)
        print(algo_performance)
        
        # Best combination details
        best = df.iloc[0]
        print(f"\\n🏆 BEST COMBINATION DETAILS:")
        print(f"   Model: {best['Model']}")
        print(f"   Algorithm: {best['Algorithm']}")
        print(f"   Clusters: {best['Clusters']}")
        print(f"   Composite Score: {best['Composite Score']:.4f}")
        print(f"   Silhouette Score: {best['Silhouette']:.4f}")
        print(f"   Separation Ratio: {best['Separation']:.4f}")
        print(f"   Cluster Balance: {best['Balance']:.4f}")
        print(f"   Coherence: {best['Coherence']:.4f}")
        print(f"   Stability: {best['Stability']:.4f}")
    
    def plot_grid_search_results(self):
        """Visualize grid search results"""
        if not self.grid_search_results:
            print("No results to plot")
            return
        
        # Prepare data
        results_df = []
        for result in self.grid_search_results:
            eval_data = result['evaluation']
            results_df.append({
                'embedding_model': result['embedding_model'],
                'clustering_algorithm': result['clustering_algorithm'],
                'composite_score': eval_data['composite_score'],
                'silhouette_score': eval_data.get('silhouette_score', 0),
                'separation_ratio': eval_data.get('separation_ratio', 0),
                'n_clusters': result['n_clusters']
            })
        
        df = pd.DataFrame(results_df)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Heatmap of composite scores
        pivot_composite = df.pivot(index='embedding_model', columns='clustering_algorithm', values='composite_score')
        sns.heatmap(pivot_composite, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,0])
        axes[0,0].set_title('Composite Scores by Model and Algorithm')
        
        # 2. Silhouette scores comparison
        pivot_silhouette = df.pivot(index='embedding_model', columns='clustering_algorithm', values='silhouette_score')
        sns.heatmap(pivot_silhouette, annot=True, fmt='.3f', cmap='plasma', ax=axes[0,1])
        axes[0,1].set_title('Silhouette Scores by Model and Algorithm')
        
        # 3. Bar plot of composite scores
        df_sorted = df.sort_values('composite_score', ascending=True)
        df_sorted['combination'] = df_sorted['embedding_model'] + '\\n+ ' + df_sorted['clustering_algorithm']
        axes[1,0].barh(range(len(df_sorted)), df_sorted['composite_score'], color='skyblue')
        axes[1,0].set_yticks(range(len(df_sorted)))
        axes[1,0].set_yticklabels(df_sorted['combination'], fontsize=8)
        axes[1,0].set_xlabel('Composite Score')
        axes[1,0].set_title('Composite Scores Ranking')
        
        # 4. Scatter plot: Silhouette vs Separation
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, algo in enumerate(df['clustering_algorithm'].unique()):
            algo_data = df[df['clustering_algorithm'] == algo]
            axes[1,1].scatter(algo_data['silhouette_score'], algo_data['separation_ratio'], 
                            label=algo, c=colors[i % len(colors)], s=60, alpha=0.7)
        
        axes[1,1].set_xlabel('Silhouette Score')
        axes[1,1].set_ylabel('Separation Ratio')
        axes[1,1].set_title('Silhouette vs Separation by Algorithm')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_best_clustering_labels(self):
        """Return the labels from the best performing model"""
        if self.best_model is None:
            print("No best model found. Run grid_search_clustering first.")
            return None
        
        return self.best_model['labels']
    
    def analyze_best_clusters(self, skills):
        """Analyze the clusters from the best model"""
        if self.best_model is None:
            print("No best model found. Run grid_search_clustering first.")
            return
        
        labels = self.best_model['labels']
        
        print(f"\\n📋 BEST MODEL CLUSTER ANALYSIS")
        print("=" * 50)
        print(f"Model: {self.best_model['embedding_model']}")
        print(f"Algorithm: {self.best_model['clustering_algorithm']}")
        print(f"Composite Score: {self.best_score:.4f}")
        
        # Cluster statistics
        unique_labels = np.unique(labels[labels >= 0])
        print(f"\\nTotal clusters: {len(unique_labels)}")
        
        # Show sample skills from each cluster
        for cluster_id in unique_labels[:10]:  # Show first 10 clusters
            cluster_mask = labels == cluster_id
            cluster_skills = [skills[i] for i in range(len(skills)) if cluster_mask[i]]
            
            print(f"\\n🔸 Cluster {cluster_id} ({len(cluster_skills)} skills):")
            
            # Show top 5 skills
            sample_skills = cluster_skills[:5]
            for skill in sample_skills:
                print(f"   • {skill}")
            
            if len(cluster_skills) > 5:
                print(f"   ... and {len(cluster_skills) - 5} more")

# Enhanced demo function
def enhanced_job_skills_clustering_demo(skills, 
                                      embedding_models=['auto'], 
                                      clustering_algorithms=['kmeans', 'gmm', 'agglomerative']):
    """
    Enhanced demo with grid search capabilities
    """
    
    print(f"🚀 ENHANCED JOB SKILLS CLUSTERING DEMO")
    print("=" * 60)
    print(f"📦 Embedding Models: {embedding_models}")
    print(f"🎯 Clustering Algorithms: {clustering_algorithms}")
    print(f"💼 Skills Dataset Size: {len(skills)}")
    
    # Initialize enhanced clusterer
    clusterer = GridSearchJobSkillsClusterer(
        embedding_models=embedding_models,
        clustering_algorithms=clustering_algorithms,
        evaluation_sample_size=5000
    )
    
    # Perform grid search
    best_labels = clusterer.grid_search_clustering(skills, auto_optimize=True)
    
    if best_labels is not None:
        # Print detailed results
        clusterer.print_detailed_results()
        
        # Plot results
        clusterer.plot_grid_search_results()
        
        # Analyze best clusters
        clusterer.analyze_best_clusters(skills)
        
        return clusterer, best_labels
    else:
        print("❌ Grid search failed!")
        return None, None

# Usage example
if __name__ == "__main__":
    # Example skills for testing
    sample_skills = [
        "Python programming", "Java development", "Machine learning", "Data analysis",
        "React.js", "Angular", "Node.js", "JavaScript", "HTML/CSS",
        "SQL", "MySQL", "PostgreSQL", "Database design",
        "AWS", "Azure", "Docker", "Kubernetes", "DevOps",
        "Project management", "Agile", "Scrum", "Leadership",
        "Communication", "Team management", "Problem solving",
        "Git", "CI/CD", "Jenkins", "Testing", "QA",
        "UI/UX design", "Photoshop", "Figma", "Wireframing"
    ]
    
    # Run enhanced clustering demo
    clusterer, labels = enhanced_job_skills_clustering_demo(
        skills=sample_skills,
        embedding_models=['auto'],  # Will test multiple models
        clustering_algorithms=['kmeans', 'gmm', 'agglomerative']
    )
    
    if clusterer and labels is not None:
        print("\\n✅ Enhanced clustering completed successfully!")
        print(f"Best model composite score: {clusterer.best_score:.4f}")
    else:
        print("\\n❌ Enhanced clustering failed!")