# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import re
import gc
import psutil
import os
from collections import Counter, defaultdict
import warnings
from sentence_transformers.util import cos_sim
warnings.filterwarnings('ignore')

# Import sentence transformers - required for job skills clustering
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ SentenceTransformers available - perfect for job skills clustering")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ SentenceTransformers required for job skills clustering!")
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
    print("⚠️ Transformers not available (optional). Install with: pip install transformers torch")

class JobSkillsClusterer:
    def __init__(self, memory_limit_gb=4, batch_size=1000, model_names=['auto'], embedders: dict=None):
        self.memory_limit_gb = memory_limit_gb
        self.batch_size = batch_size
        self.model_names = model_names
        
        # Initialize embedding model
        self.embedders = embedders
        if self.embedders is None:
            self.embedders = {}
            self._initialize_embedding_model()

        self.skill_categories = self._define_skill_categories()
        self.skill_synonyms = self._define_skill_synonyms()
        
        # Results storage
        self.best_models = []
        self.best_score = -1
        self.results = []
        self.ensemble_results = None
        self.solid_clusters = None
        self.final_labels = None
    
    def _initialize_embedding_model(self):
        """Initialize the best embedding model for job skills"""
        
        try:
            for model_name in self.model_names:
                if model_name == 'auto':
                    # Models ranked by effectiveness for job skills
                    model_options = [
                        'jinaai/jina-embeddings-v3',
                        'all-MiniLM-L6-v2',        # Fast, excellent for skills (384 dim)
                        'all-mpnet-base-v2',       # Best overall quality (768 dim)
                        'paraphrase-MiniLM-L6-v2', # Good for skill variations
                        'all-distilroberta-v1',    # Alternative option
                    ]
                    
                    for name in model_options:
                        try:
                            print(f"Loading SentenceTransformer model for job skills: {name}")
                            embedder = SentenceTransformer(name, trust_remote_code=True)
                            embedding_dim = embedder.get_sentence_embedding_dimension()
                            self.embedders[name] = embedder
                            print(f"✅ Successfully loaded {name} (dim: {embedding_dim})")
                            return
                        except Exception as e:
                            print(f"Failed to load {name}: {e}")
                            continue
                else:
                    print(f"Loading specified model for job skills: {model_name}")
                    embedder = SentenceTransformer(model_name, trust_remote_code=True)
                    embedding_dim = embedder.get_sentence_embedding_dimension()
                    self.embedders[model_name] = embedder
                    print(f"✅ Successfully loaded {model_name} (dim: {embedding_dim})")
            return
                    
        except Exception as e:
            print(f"Failed to initialize SentenceTransformer: {e}")
            raise RuntimeError("Could not initialize any SentenceTransformer model for job skills clustering")
        

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
    
    def create_skill_embeddings(self, skills, batch_size=None):
        """Create embeddings optimized for job skills"""
        if batch_size is None:
            batch_size = min(self.batch_size, 64)  # Larger batches for skills (usually short)
        
        print(f"Creating embeddings for {len(skills)} skills...")
        
        if len(self.embedders.keys()) > 0:
            return self._create_sentence_transformer_embeddings(skills, batch_size)
        else:
            raise RuntimeError("No embedding model available for job skills")
    
    def _create_sentence_transformer_embeddings(self, skills, batch_size):
        """Create embeddings using SentenceTransformers optimized for skills"""
        embeddings = {}
        for model_name, embedder in self.embedders.items():
            print(f"Using SentenceTransformer for skills: {model_name}")
            model_embeddings = []
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
                            normalize_embeddings=True  # L2 normalization for better clustering
                            )
                    elif 'Qwen' in model_name:
                        batch_embeddings = embedder.encode(
                            batch_skills,
                            prompt='Instruct: Given a skill, use it for context matching in a skill taxonomy framework \nSkill:',
                            batch_size=min(batch_size, 64),
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True  # L2 normalization for better clustering
                            )
                    else:
                        batch_embeddings = embedder.encode(
                            batch_skills,
                            batch_size=min(batch_size, 64),
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True  # L2 normalization for better clustering
                        )
                    model_embeddings.append(batch_embeddings)
                    
                    if i % (batch_size * 20) == 0:
                        current_memory = self.get_memory_usage()
                        print(f"Embedded {min(i + batch_size, len(skills))}/{len(skills)} skills - Memory: {current_memory:.2f} GB")
                        gc.collect()
                        
                except Exception as e:
                    print(f"Error in batch {i//batch_size}: {e}")
                    # Create zero embeddings for failed batch
                    # batch_embeddings = np.zeros((len(batch_skills), embedding_dim))
                    # embeddings.append(batch_embeddings)
            
            embeddings[model_name] = np.vstack(model_embeddings)
            print(f"Created skill embeddings (model: {model_name}) shape: {embeddings[model_name].shape}")
            
        return embeddings
    
    def smart_dimensionality_reduction(self, X, target_dim=50):
        """Dimensionality reduction optimized for job skills"""
        n_samples, n_features = X.shape
        
        print(f"Skill embeddings shape: {n_samples} x {n_features}")
        
        # For skills, we typically don't need aggressive reduction
        # Skills embeddings are already well-structured
        # if n_features <= 512:  # Most models are 384-768 dim
        #     print("Keeping original embedding dimensions - optimal for skills")
        #     return X, None
        
        # Only reduce if very high dimensional
        if n_features > 512:
            target_dim = min(target_dim, n_features // 3)
            
        if n_samples > 500000:
            print(f"Using IncrementalPCA for skill dimensionality reduction to {target_dim} dimensions...")
            reducer = IncrementalPCA(n_components=target_dim, batch_size=min(1000, n_samples//10))
            
            chunk_size = min(2000, n_samples)
            for i in range(0, n_samples, chunk_size):
                chunk = X[i:i+chunk_size]
                reducer.partial_fit(chunk)
                gc.collect()
            
            X_reduced = reducer.transform(X)
        else:
            print(f"Using PCA for skill dimensionality reduction to {target_dim} dimensions...")
            # reducer = TruncatedSVD(n_components=target_dim, random_state=42)
            # X_reduced = reducer.fit_transform(X)

            reducer = PCA(n_components=target_dim, random_state=42)
            X_reduced = reducer.fit_transform(X)
        
        print(f"Reduced skill embeddings to shape: {X_reduced.shape}")
        return X_reduced, reducer
        
        # return X, None
    
    def estimate_optimal_clusters(self, X, max_k=30, sample_size=20000, algo='kmeans'):
        """Estimate optimal clusters specifically for job skills"""
        n_samples = X.shape[0]
        np.random.seed(42)
        # Use larger sample for skills since they're more diverse
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        print(f"Estimating optimal skill clusters using {X_sample.shape[0]} samples...")
        
        # Skills typically have more clusters than general text
        max_k = min(max_k, X_sample.shape[0] // 5, 25)
        
        # Test more k values for skills
        if max_k > 15:
            # k_values = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
            k_values = [x for x in range(10, 200, 1)]
        else:
            k_values = list(range(2, max_k + 1))
        
        k_values = [k for k in k_values if k < X_sample.shape[0]]
        print(f"k values: {k_values}")
        
        best_score = -1
        best_k = 8  # Default for skills
        
        for k in k_values:
            try:
                if algo == 'kmeans':
                    # Use regular KMeans for skills (usually not huge datasets)
                    if X_sample.shape[0] < 50000:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    else:
                        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=min(self.batch_size, X_sample.shape[0]//2))
                    
                    labels = kmeans.fit_predict(X_sample)
                elif algo == 'gmm':
                    gmm = GaussianMixture(k, n_init=10).fit(X_sample) 
                    labels = gmm.predict(X_sample)
                elif algo == 'agglomerative':
                    clustering = AgglomerativeClustering(
                                    n_clusters=k,
                                    metric='cosine',
                                    linkage='average'
                                )
                    labels = clustering.fit_predict(X_sample)

                if len(set(labels)) > 1:
                    score = silhouette_score(X_sample, labels, sample_size=min(self.batch_size, X_sample.shape[0]))
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
            except Exception as e:
                print(f"Error with k={k}: {e}")
                continue
        
        print(f"Estimated optimal skill clusters: {best_k} (score: {best_score:.3f})")
        return best_k
    
    def scalable_clustering(self, X, n_clusters, algo='kmeans'):
        """Clustering optimized for job skills"""
        print(f"Clustering {X.shape[0]} skills into {n_clusters} clusters using {algo}...")
        
        if algo == 'kmeans':
            if X.shape[0] > 100000:
                # Use MiniBatchKMeans for large skill datasets
                batch_size = min(10000, X.shape[0] // 2)
                clusterer = MiniBatchKMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    batch_size=batch_size,
                    max_iter=200,  # More iterations for skills
                    n_init=5
                )
            else:
                # Use regular KMeans for better quality on smaller datasets
                clusterer = KMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    n_init=10,  # More initializations for skills
                    max_iter=300
                )
        elif algo == 'gmm':
            clusterer = GaussianMixture(n_clusters, n_init=10, random_state=42)
        elif algo == 'agglomerative':
            clusterer = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            metric='cosine',
                            linkage='average'
                        )
        elif algo == 'dbscan':
            # For DBSCAN, we need to estimate eps parameter
            eps = self._estimate_dbscan_eps(X)
            clusterer = DBSCAN(eps=eps, min_samples=5, metric='cosine')
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        labels = clusterer.fit_predict(X)
        return labels, clusterer
    
    def _estimate_dbscan_eps(self, X, k=5):
        """Estimate eps parameter for DBSCAN using k-distance graph"""
        if X.shape[0] > 10000:
            # Sample for large datasets
            indices = np.random.choice(X.shape[0], 10000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Find k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
        neighbors_fit = neighbors.fit(X_sample)
        distances, indices = neighbors_fit.kneighbors(X_sample)
        
        # Get k-th nearest neighbor distances
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)
        
        # Find the elbow point (simple heuristic)
        diffs = np.diff(k_distances)
        elbow_idx = np.argmax(diffs)
        eps = k_distances[elbow_idx]
        
        print(f"Estimated DBSCAN eps: {eps:.4f}")
        return eps
    
    def evaluate_clustering_sample(self, X, labels, sample_size=20000):
        """Evaluate skill clustering quality"""
        if len(set(labels)) <= 1:
            return -1, -1
        
        n_samples = X.shape[0]
        
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        try:
            silhouette = silhouette_score(X_sample, labels_sample, sample_size=min(2000, len(X_sample)))
            calinski = calinski_harabasz_score(X_sample, labels_sample)
            return silhouette, calinski
        except:
            return -1, -1
    
    def ensemble_clustering_with_solid_clusters(self, X, algorithms=['kmeans', 'gmm', 'agglomerative'], n_clusters=None):
        """
        Perform ensemble clustering with multiple algorithms and identify solid clusters.
        Solid clusters are data points that are consistently assigned to the same cluster across all algorithms.
        """
        print(f"\n🔄 Starting ensemble clustering with algorithms: {algorithms}")
        
        clustering_results = {}
        all_labels = []
        
        # Run clustering with each algorithm
        for algo in algorithms:
            print(f"\n📊 Running {algo} clustering...")
            
            # Get optimal clusters for this algorithm if not specified
            if n_clusters is None:
                optimal_k = self.estimate_optimal_clusters(X, algo=algo)
            else:
                optimal_k = n_clusters
            
            try:
                labels, clusterer = self.scalable_clustering(X, optimal_k, algo=algo)
                
                # Handle DBSCAN noise labels
                if algo == 'dbscan':
                    # Reassign noise points (-1) to nearest cluster
                    labels = self._reassign_noise_points(X, labels)
                
                clustering_results[algo] = {
                    'labels': labels,
                    'clusterer': clusterer,
                    'n_clusters': len(set(labels[labels != -1])) if -1 in labels else len(set(labels))
                }
                all_labels.append(labels)
                
                # Evaluate this clustering
                silhouette, calinski = self.evaluate_clustering_sample(X, labels)
                print(f"✅ {algo} completed - Clusters: {clustering_results[algo]['n_clusters']}, "
                      f"Silhouette: {silhouette:.3f}")
                
            except Exception as e:
                print(f"❌ Error with {algo}: {e}")
                continue
        
        if len(all_labels) < 2:
            print("❌ Need at least 2 successful clustering results for ensemble")
            return None, None, None
        
        # Convert to numpy array for easier processing
        all_labels = np.array(all_labels)
        n_samples = all_labels.shape[1]
        
        print(f"\n🔍 Identifying solid clusters from {len(algorithms)} clustering results...")
        
        # Find solid clusters (points that are consistently clustered together)
        solid_clusters, solid_labels = self._identify_solid_clusters(all_labels, X)
        
        print(f"✅ Found {len(solid_clusters)} solid clusters containing "
              f"{np.sum(solid_labels != -1)} data points")
        
        # Assign remaining points to closest solid cluster
        final_labels = self._assign_remaining_points(X, solid_clusters, solid_labels)
        
        # Store results
        self.ensemble_results = clustering_results
        self.solid_clusters = solid_clusters
        self.final_labels = final_labels
        
        return clustering_results, solid_clusters, final_labels
    
    def _reassign_noise_points(self, X, labels):
        """Reassign DBSCAN noise points (-1) to nearest cluster"""
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        
        # Get unique non-noise cluster labels
        cluster_labels = np.unique(labels[~noise_mask])
        if len(cluster_labels) == 0:
            return labels
        
        # Calculate cluster centroids
        centroids = []
        for cluster_id in cluster_labels:
            cluster_mask = labels == cluster_id
            centroid = np.mean(X[cluster_mask], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Assign noise points to nearest centroid
        noise_points = X[noise_mask]
        distances = cdist(noise_points, centroids, metric='cosine')
        nearest_clusters = np.argmin(distances, axis=1)
        
        # Update labels
        new_labels = labels.copy()
        new_labels[noise_mask] = cluster_labels[nearest_clusters]
        
        return new_labels
    
    def _identify_solid_clusters(self, all_labels, X, min_consensus=None):
        """
        Identify solid clusters based on consensus across all clustering algorithms.
        """
        n_algorithms, n_samples = all_labels.shape
        
        if min_consensus is None:
            min_consensus = n_algorithms  # Require unanimous consensus
        
        # Build co-occurrence matrix
        print("Building co-occurrence matrix...")
        co_occurrence = np.zeros((n_samples, n_samples), dtype=int)
        
        for algo_labels in all_labels:
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if algo_labels[i] == algo_labels[j] and algo_labels[i] != -1:
                        co_occurrence[i, j] += 1
                        co_occurrence[j, i] += 1
        
        # Find groups of points that are consistently clustered together
        print("Finding consensus groups...")
        solid_clusters = []
        processed = set()
        solid_labels = np.full(n_samples, -1, dtype=int)
        
        for i in range(n_samples):
            if i in processed:
                continue
            
            # Find all points that consistently cluster with point i
            consensus_group = {i}
            for j in range(i + 1, n_samples):
                if co_occurrence[i, j] >= min_consensus:
                    consensus_group.add(j)
            
            # Verify that all points in the group consistently cluster together
            valid_group = True
            group_list = list(consensus_group)
            
            for idx1 in range(len(group_list)):
                for idx2 in range(idx1 + 1, len(group_list)):
                    point1, point2 = group_list[idx1], group_list[idx2]
                    if co_occurrence[point1, point2] < min_consensus:
                        valid_group = False
                        break
                if not valid_group:
                    break
            
            # If group is valid and large enough, add it as a solid cluster
            if valid_group and len(consensus_group) >= 3:  # Minimum cluster size
                cluster_id = len(solid_clusters)
                solid_clusters.append(list(consensus_group))
                
                for point_idx in consensus_group:
                    solid_labels[point_idx] = cluster_id
                    processed.add(point_idx)
        
        return solid_clusters, solid_labels
    
    def _assign_remaining_points(self, X, solid_clusters, solid_labels):
        """
        Assign remaining points to the closest solid cluster.
        """
        if len(solid_clusters) == 0:
            print("⚠️ No solid clusters found, returning original labels")
            return solid_labels
        
        final_labels = solid_labels.copy()
        unassigned_mask = solid_labels == -1
        
        if not np.any(unassigned_mask):
            print("✅ All points are already in solid clusters")
            return final_labels
        
        print(f"📍 Assigning {np.sum(unassigned_mask)} remaining points to closest solid clusters...")
        
        # Calculate centroids of solid clusters
        cluster_centroids = []
        for cluster_points in solid_clusters:
            centroid = np.mean(X[cluster_points], axis=0)
            cluster_centroids.append(centroid)
        
        cluster_centroids = np.array(cluster_centroids)
        
        # Find unassigned points
        unassigned_points = X[unassigned_mask]
        
        # Calculate distances to all cluster centroids
        distances = cdist(unassigned_points, cluster_centroids, metric='cosine')
        
        # Assign each unassigned point to nearest cluster
        nearest_clusters = np.argmin(distances, axis=1)
        
        # Update final labels
        unassigned_indices = np.where(unassigned_mask)[0]
        for i, cluster_id in enumerate(nearest_clusters):
            final_labels[unassigned_indices[i]] = cluster_id
        
        return final_labels
    
    def ensemble_clustering(self, clustering_results, 
                          method: str = 'voting') -> np.ndarray:
        """
        Combine clustering results using ensemble methods.
        """
        
        n_samples = len(clustering_results[list(clustering_results.keys())[0]]['labels'])
        all_labels = []
        
        # Collect all clustering results
        for algo_name, results in clustering_results.items():
            all_labels.append(results["labels"])
        
        if not all_labels:
            raise ValueError("No valid clustering results found")
        
        all_labels = np.array(all_labels)
        
        if method == 'voting':
            # Majority voting (simplified)
            ensemble_labels = self._majority_voting(all_labels)
        elif method == 'consensus':
            # Consensus-based clustering
            ensemble_labels = self._consensus_clustering(all_labels)
        else:
            # Default to first clustering result
            ensemble_labels = all_labels[0]
        
        return ensemble_labels
    
    def _majority_voting(self, all_labels: np.ndarray) -> np.ndarray:
        """
        Simple majority voting for cluster assignment.
        """
        n_samples = all_labels.shape[1]
        ensemble_labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Get all cluster assignments for sample i
            sample_labels = all_labels[:, i]
            # Find most common cluster (excluding noise labels -1)
            valid_labels = sample_labels[sample_labels >= 0]
            if len(valid_labels) > 0:
                unique, counts = np.unique(valid_labels, return_counts=True)
                ensemble_labels[i] = unique[np.argmax(counts)]
            else:
                ensemble_labels[i] = -1 # Noise
        
        return ensemble_labels
    
    def _consensus_clustering(self, all_labels: np.ndarray) -> np.ndarray:
        """
        Consensus clustering using co-association matrix.
        """
        n_samples = all_labels.shape[1]
        n_clusterings = all_labels.shape[0]
        
        # Build co-association matrix
        coassoc_matrix = np.zeros((n_samples, n_samples))
        
        for clustering in all_labels:
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if clustering[i] == clustering[j] and clustering[i] >= 0:
                        coassoc_matrix[i, j] += 1
                        coassoc_matrix[j, i] += 1
        
        # Normalize co-association matrix
        coassoc_matrix /= n_clusterings
        
        # Apply clustering to co-association matrix
        distance_matrix = 1 - coassoc_matrix
        clusterer = AgglomerativeClustering(
            n_clusters=len(np.unique(all_labels[0])), 
            metric='precomputed', 
            linkage='average'
        )
        consensus_labels = clusterer.fit_predict(distance_matrix)
        
        return consensus_labels
    
    def fit_predict(self, skills, n_clusters=None, auto_optimize=True, algorithms=['kmeans', 'gmm', 'agglomerative']):
        """Main method to cluster job skills using ensemble approach with solid clusters"""
        start_memory = self.get_memory_usage()
        print(f"Starting ensemble job skills clustering of {len(skills)} skills...")
        print(f"Initial memory usage: {start_memory:.2f} GB")
        print(f"Algorithms to use: {algorithms}")
        
        if len(skills) < 2:
            print("Error: Need at least 2 skills for clustering")
            return None
        
        # Preprocess skills
        print("\n📝 Preprocessing job skills...")
        processed_skills = []
        
        for i in range(0, len(skills), self.batch_size):
            batch = skills[i:i + self.batch_size]
            processed_batch = self.preprocess_skills(batch)
            processed_skills.extend(processed_batch)
            
            if i % (self.batch_size * 10) == 0:
                current_memory = self.get_memory_usage()
                print(f"Processed {min(i + self.batch_size, len(skills))}/{len(skills)} skills - Memory: {current_memory:.2f} GB")
                gc.collect()
        
        # Track valid skills
        valid_skills = []
        original_to_processed = {}
        
        for i, skill in enumerate(processed_skills):
            if skill.strip():  # Non-empty after processing
                original_to_processed[i] = len(valid_skills)
                valid_skills.append(skill)
        
        print(f"Total skills: {len(skills)}")
        print(f"Valid skills after preprocessing: {len(valid_skills)}")
        print(f"Empty/invalid skills: {len(skills) - len(valid_skills)}")
        
        if len(valid_skills) < 5:
            print("Error: Too few valid skills for meaningful clustering")
            return None
        
        # Create skill embeddings
        print("\n🔢 Creating skill embeddings...")
        X_embeddings = self.create_skill_embeddings(valid_skills)
        similarities = None
        coef = 1/len(X_embeddings.keys())
        for model_name, embed in X_embeddings.items():
            print(f"Model: {model_name}, embedding matrix shape: {embed.shape}")
            if similarities is None:
                similarities = coef * cos_sim(embed, embed).numpy()
            else:
                similarities += coef * cos_sim(embed, embed).numpy()
        
        # Dimensionality reduction (conservative for skills)
        target_dim = min(100, similarities.shape[1])
        X_reduced, _ = self.smart_dimensionality_reduction(similarities, target_dim=target_dim)
        
        del X_embeddings
        gc.collect()
        
        # Perform ensemble clustering with solid clusters
        clustering_results, solid_clusters, final_labels = self.ensemble_clustering_with_solid_clusters(
            X_reduced, algorithms=algorithms, n_clusters=n_clusters
        )
        
        if final_labels is None:
            print("❌ Ensemble clustering failed!")
            return None
        
        # Create full output array
        full_labels = np.full(len(skills), -1, dtype=int)
        
        # Map cluster results back
        for original_idx, processed_idx in original_to_processed.items():
            if processed_idx < len(final_labels):
                full_labels[original_idx] = final_labels[processed_idx]
        
        # Store results
        self.best_model = {
            'labels': full_labels,
            'ensemble_labels': final_labels,
            'solid_clusters': solid_clusters,
            'clustering_results': clustering_results,
            'n_clusters': len(set(final_labels)),
            'processed_skills': valid_skills,
            'original_indices': list(original_to_processed.keys()),
            'feature_matrix': X_reduced,
            'algorithms_used': algorithms
        }
        
        # Validation
        assert len(full_labels) == len(skills), f"Output size {len(full_labels)} != input size {len(skills)}"
        
        final_memory = self.get_memory_usage()
        print(f"\n🎉 Ensemble Job Skills Clustering Completed!")
        print(f"Final memory usage: {final_memory:.2f} GB (Peak increase: {final_memory - start_memory:.2f} GB)")
        print(f"Input skills: {len(skills)}")
        print(f"Output labels: {len(full_labels)}")
        print(f"Successfully clustered skills: {np.sum(full_labels != -1)}")
        print(f"Invalid/empty skills: {np.sum(full_labels == -1)}")
        print(f"Number of Solid Clusters: {len(solid_clusters) if solid_clusters else 0}")
        print(f"Final Number of Clusters: {len(set(final_labels))}")
        print(f"Algorithms used: {algorithms}")
        
        return full_labels
    
    def analyze_solid_clusters(self, skills):
        """Analyze the quality and composition of solid clusters"""
        if self.solid_clusters is None or self.best_model is None:
            print("❌ No solid clusters found. Run clustering first.")
            return
        
        print(f"\n📊 Solid Clusters Analysis")
        print(f"=" * 50)
        
        valid_skills = self.best_model['processed_skills']
        solid_clusters = self.solid_clusters
        final_labels = self.final_labels
        
        print(f"Total solid clusters: {len(solid_clusters)}")
        print(f"Points in solid clusters: {sum(len(cluster) for cluster in solid_clusters)}")
        print(f"Points assigned to nearest solid cluster: {np.sum(final_labels != -1) - sum(len(cluster) for cluster in solid_clusters)}")
        
        # Analyze each solid cluster
        for i, cluster_points in enumerate(solid_clusters):
            print(f"\n🔸 Solid Cluster {i} ({len(cluster_points)} skills):")
            
            # Show sample skills from this cluster
            sample_skills = []
            for point_idx in cluster_points[:10]:  # Show first 10
                if point_idx < len(valid_skills):
                    sample_skills.append(valid_skills[point_idx])
            
            print(f"   Sample skills: {sample_skills}")
            
        # Show cluster size distribution
        cluster_sizes = [len(cluster) for cluster in solid_clusters]
        if cluster_sizes:
            print(f"\n📈 Solid Cluster Size Statistics:")
            print(f"   Min size: {min(cluster_sizes)}")
            print(f"   Max size: {max(cluster_sizes)}")
            print(f"   Average size: {np.mean(cluster_sizes):.1f}")
            print(f"   Median size: {np.median(cluster_sizes):.1f}")
    
    def plot_ensemble_results(self, skills, sample_size=3000):
        """Visualize ensemble clustering results including solid clusters"""
        if self.best_model is None:
            print("❌ No clustering results to plot")
            return
        
        X = self.best_model['feature_matrix']
        final_labels = self.final_labels
        solid_clusters = self.solid_clusters
        clustering_results = self.best_model['clustering_results']
        
        # Sample for visualization if too large
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_plot = X[indices]
            final_labels_plot = final_labels[indices]
            
            # Map solid clusters to sampled indices
            solid_indices_plot = []
            for cluster_points in solid_clusters:
                solid_in_sample = [i for i, orig_i in enumerate(indices) if orig_i in cluster_points]
                if solid_in_sample:
                    solid_indices_plot.append(solid_in_sample)
        else:
            X_plot = X
            final_labels_plot = final_labels
            solid_indices_plot = solid_clusters
            indices = np.arange(len(X))
        
        # Create comprehensive plots
        n_algorithms = len(clustering_results)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 2D visualization for final ensemble result
        if X_plot.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_plot)
        else:
            X_2d = X_plot
        
        # Plot 1: Final ensemble clustering result
        scatter = axes[0, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=final_labels_plot, 
                                   cmap='tab20', alpha=0.7, s=30)
        
        # Highlight solid cluster points
        for cluster_points in solid_indices_plot:
            if len(cluster_points) > 0:
                cluster_points = [p for p in cluster_points if p < len(X_2d)]
                if cluster_points:
                    axes[0, 0].scatter(X_2d[cluster_points, 0], X_2d[cluster_points, 1], 
                                     s=100, facecolors='none', edgecolors='red', linewidths=2)
        
        axes[0, 0].set_title(f'Ensemble Clustering Result\n(Red circles = Solid clusters)')
        axes[0, 0].set_xlabel('Component 1')
        axes[0, 0].set_ylabel('Component 2')
        
        # Plot 2: Individual algorithm results (show up to 2)
        algo_names = list(clustering_results.keys())
        for i, algo_name in enumerate(algo_names[:2]):
            col = i + 1
            algo_labels = clustering_results[algo_name]['labels']
            
            if len(indices) < len(algo_labels):
                algo_labels_plot = algo_labels[indices]
            else:
                algo_labels_plot = algo_labels[:len(indices)]
            
            scatter = axes[0, col].scatter(X_2d[:, 0], X_2d[:, 1], c=algo_labels_plot, 
                                         cmap='tab20', alpha=0.7, s=30)
            axes[0, col].set_title(f'{algo_name.upper()} Clustering')
            axes[0, col].set_xlabel('Component 1')
            axes[0, col].set_ylabel('Component 2')
        
        # Plot 3: Cluster size distributions
        final_cluster_counts = pd.Series(final_labels).value_counts().sort_index()
        axes[1, 0].bar(final_cluster_counts.index, final_cluster_counts.values, 
                      color='skyblue', alpha=0.7)
        axes[1, 0].set_title('Final Cluster Sizes')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Skills')
        
        # Plot 4: Solid vs Non-solid points
        solid_points_mask = np.zeros(len(final_labels_plot), dtype=bool)
        for cluster_points in solid_indices_plot:
            solid_points_mask[cluster_points] = True
        
        colors = ['lightcoral' if is_solid else 'lightblue' for is_solid in solid_points_mask]
        axes[1, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.7, s=30)
        axes[1, 1].set_title('Solid vs Assigned Points\n(Red = Solid, Blue = Assigned)')
        axes[1, 1].set_xlabel('Component 1')
        axes[1, 1].set_ylabel('Component 2')
        
        # Plot 5: Algorithm agreement heatmap
        if len(solid_clusters) > 0:
            solid_sizes = [len(cluster) for cluster in solid_clusters]
            axes[1, 2].bar(range(len(solid_sizes)), solid_sizes, color='green', alpha=0.7)
            axes[1, 2].set_title('Solid Cluster Sizes')
            axes[1, 2].set_xlabel('Solid Cluster ID')
            axes[1, 2].set_ylabel('Number of Skills')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Solid Clusters Found', 
                           ha='center', va='center', fontsize=12)
            axes[1, 2].set_title('Solid Clusters')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Ensemble Clustering Summary:")
        print(f"   Total algorithms used: {len(clustering_results)}")
        print(f"   Solid clusters found: {len(solid_clusters)}")
        print(f"   Points in solid clusters: {sum(len(cluster) for cluster in solid_clusters)}")
        print(f"   Final number of clusters: {len(set(final_labels))}")
        print(f"   Successfully clustered skills: {np.sum(final_labels != -1)}")

# Enhanced demo function for ensemble clustering
def job_skills_ensemble_clustering(skills, clusterer, algorithms=['kmeans', 'gmm', 'agglomerative']):
    """Demonstrate ensemble clustering with solid clusters on job skills dataset"""
    
    print(f"\n🚀 Ensemble Job Skills Clustering Configuration:")
    print(f"📦 Embedding Model: {clusterer.model_names}")
    print(f"🤖 Algorithms: {algorithms}")
    print(f"💼 Optimized for: Job Skills from Job Descriptions")
    print(f"🎯 Strategy: Identify solid clusters with consensus, assign rest to nearest")
    
    # Perform ensemble skills clustering
    predicted_labels = clusterer.fit_predict(skills, algorithms=algorithms)
    
    if predicted_labels is not None:
        print(f"\n✅ Ensemble Skills Clustering Results:")
        print(f"Input skills: {len(skills)}")
        print(f"Output labels: {len(predicted_labels)}")
        print(f"Dimensions match: {len(skills) == len(predicted_labels)}")
        
        if len(skills) == len(predicted_labels):
            print(f"Final clusters: {len(set(predicted_labels[predicted_labels != -1]))}")
            
            # Analyze solid clusters
            clusterer.analyze_solid_clusters(skills)
            
            # Visualize ensemble results
            clusterer.plot_ensemble_results(skills)
            
            return clusterer
        else:
            print("❌ ERROR: Dimension mismatch!")
            return None
    else:
        print("❌ Ensemble skills clustering failed!")
        return None

# Example usage
if __name__ == "__main__":
    # Example job skills for demonstration
    sample_skills = [
        "Python programming", "Machine learning", "Data analysis", "SQL", "JavaScript",
        "React", "Node.js", "AWS", "Docker", "Kubernetes", "Project management",
        "Agile methodology", "Scrum", "Leadership", "Communication", "Problem solving",
        "Deep learning", "TensorFlow", "PyTorch", "Data visualization", "Tableau",
        "Power BI", "Excel", "Statistics", "R programming", "Java", "Spring Boot",
        "Microservices", "REST API", "Database design", "PostgreSQL", "MongoDB",
        "Git", "CI/CD", "Jenkins", "Terraform", "Ansible", "Linux", "Bash scripting",
        "UI/UX design", "Figma", "Photoshop", "HTML", "CSS", "Bootstrap", "Vue.js",
        "Angular", "TypeScript", "GraphQL", "Redis", "Elasticsearch", "Apache Kafka"
    ]
    
    # Initialize clusterer
    clusterer = JobSkillsClusterer(
        memory_limit_gb=8,
        batch_size=500,
        model_names=['all-MiniLM-L6-v2']
    )
    
    # Run ensemble clustering
    algorithms_to_use = ['kmeans', 'gmm', 'agglomerative']
    result_clusterer = job_skills_ensemble_clustering(
        sample_skills, 
        clusterer, 
        algorithms=algorithms_to_use
    )