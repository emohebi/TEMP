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

class JobSkillsClusterer:
    def __init__(self, memory_limit_gb=4, batch_size=1000, model_names=['auto'], embedders: dict=None):
        self.memory_limit_gb = memory_limit_gb
        self.batch_size = batch_size
        self.model_names = model_names
        
        # Initialize embedding model
        self.embedders = embedders
        if self.embedders is None:
            self._initialize_embedding_model()

        self.skill_categories = self._define_skill_categories()
        self.skill_synonyms = self._define_skill_synonyms()
        
        # Results storage
        self.best_models = []
        self.best_score = -1
        self.results = []
    
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
                            print(f"✅ Successfully loaded {model_name} (dim: {embedding_dim})")
                            return
                        except Exception as e:
                            print(f"Failed to load {model_name}: {e}")
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
        
        print(f"Creating embeddings for {len(skills)}...")
        
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
        print(f"Clustering {X.shape[0]} skills into {n_clusters} clusters...")
        
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
        elif algo == 'kmeans':
            # Use regular KMeans for better quality on smaller datasets
            clusterer = KMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                n_init=10,  # More initializations for skills
                max_iter=300
            )
        elif algo == 'gmm':
            clusterer = GaussianMixture(n_clusters, n_init=10)
        elif algo == 'agglomerative':
            clusterer = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            metric='cosine',
                            linkage='average'
                        )
        
        labels = clusterer.fit_predict(X)
        return labels, clusterer
    
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
        
    def ensemble_clustering(self, clustering_results, 
                          method: str = 'voting') -> np.ndarray:
        """
        Combine clustering results using ensemble methods.
        """
        
        n_samples = len(clustering_results[self.model_names[0]]['labels'])
        all_labels = []
        
        # Collect all clustering results
        for model_name, results in clustering_results.items():
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
        
        # self.ensemble_labels = ensemble_labels
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
    
    def fit_predict(self, skills, n_clusters=None, auto_optimize=True, algo='kmeans'):
        """Main method to cluster job skills"""
        start_memory = self.get_memory_usage()
        print(f"Starting job skills clustering of {len(skills)} skills...")
        print(f"Initial memory usage: {start_memory:.2f} GB")
        
        if len(skills) < 2:
            print("Error: Need at least 2 skills for clustering")
            return None
        
        # Preprocess skills
        print("Preprocessing job skills...")
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
        target_dim = min(100, embed.shape[1])
        X_reduced, _ = self.smart_dimensionality_reduction(similarities, target_dim=target_dim)
        
        del X_embeddings
        gc.collect()
 
            # Optimize cluster number for skills
        if auto_optimize and n_clusters is None:
            n_clusters = self.estimate_optimal_clusters(X_reduced, algo=alo)
        elif n_clusters is None:
            # Default clustering strategy for skills
            n_clusters = min(15, max(5, len(valid_skills) // 50, int(np.sqrt(len(valid_skills)))))
        
        print(f"Using {n_clusters} clusters for skills")
        
        # Cluster skills
        cluster_labels, clusterer = self.scalable_clustering(X_reduced, n_clusters, algo=algo)

        
        # Evaluate
        silhouette, calinski = self.evaluate_clustering_sample(X_reduced, cluster_labels)

        # Create full output array
        full_labels = np.full(len(skills), -1, dtype=int)
        
        # Map cluster results back
        for original_idx, processed_idx in original_to_processed.items():
            if processed_idx < len(cluster_labels):
                full_labels[original_idx] = cluster_labels[processed_idx]
        
        # Store results
        self.best_model = {
            'labels': full_labels,
            'n_clusters': len(set(full_labels)),
            # 'silhouette': silhouette,
            # 'calinski_harabasz': calinski,
            'clusterer': clusterer,
            'processed_skills': valid_skills,
            'original_indices': list(original_to_processed.keys()),
            'feature_matrix': X_reduced,
            # 'embedding_model': self.embedding_model,
            'model_name': getattr(self, 'model_name', 'unknown')
        }
        
        # Validation
        assert len(full_labels) == len(skills), f"Output size {len(full_labels)} != input size {len(skills)}"
        
        final_memory = self.get_memory_usage()
        print(f"\nJob Skills Clustering Completed!")
        print(f"Final memory usage: {final_memory:.2f} GB (Peak increase: {final_memory - start_memory:.2f} GB)")
        print(f"Input skills: {len(skills)}")
        print(f"Output labels: {len(full_labels)}")
        print(f"Successfully clustered skills: {np.sum(full_labels != -1)}")
        print(f"Invalid/empty skills: {np.sum(full_labels == -1)}")
        # print(f"Silhouette Score: {silhouette:.3f}")
        # print(f"Calinski-Harabasz Score: {calinski:.1f}")
        print(f"Number of Skill Clusters: {len(set(cluster_labels))}")
        
        return full_labels
    
    
    def plot_skill_clusters(self, sample_size=3000):
        """Visualize skill clusters"""
        if self.best_model is None:
            print("No clustering results to plot")
            return
        
        X = self.best_model['feature_matrix']
        labels = self.best_model['labels']
        
        # Sample for visualization if too large
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_plot = X[indices]
            labels_plot = labels[indices]
        else:
            X_plot = X
            labels_plot = labels
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 2D skill cluster visualization
        if X_plot.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_plot)
        else:
            X_2d = X_plot
        
        scatter = axes[0, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_plot, 
                                   cmap='tab20', alpha=0.7, s=20)
        axes[0, 0].set_title(f'Job Skills Clusters ({len(X_plot)} skills)')
        axes[0, 0].set_xlabel('Component 1')
        axes[0, 0].set_ylabel('Component 2')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Cluster size distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        axes[0, 1].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
        axes[0, 1].set_title('Skill Cluster Sizes')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Number of Skills')
        
        # Cluster size histogram
        axes[1, 0].hist(cluster_counts.values, bins=min(15, len(cluster_counts)), 
                       color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution of Cluster Sizes')
        axes[1, 0].set_xlabel('Skills per Cluster')
        axes[1, 0].set_ylabel('Number of Clusters')
        
        # Top 10 largest clusters
        top_clusters = cluster_counts.head(10)
        axes[1, 1].barh(range(len(top_clusters)), top_clusters.values, color='coral')
        axes[1, 1].set_yticks(range(len(top_clusters)))
        axes[1, 1].set_yticklabels([f'Cluster {idx}' for idx in top_clusters.index])
        axes[1, 1].set_title('Top 10 Largest Skill Clusters')
        axes[1, 1].set_xlabel('Number of Skills')
        
        plt.tight_layout()
        plt.show()

# Demo function specifically for job skills
def job_skills_clustering(skills, clusterer, algo='kmeans'):
    """Demonstrate clustering on realistic job skills dataset"""
    
    print(f"\n🚀 Job Skills Clustering Configuration:")
    print(f"📦 Embedding Model: {clusterer.model_names}")
    # print(f"📐 Embedding Dimension: {clusterer.embedding_dim}")
    print(f"💼 Optimized for: Job Skills from Job Descriptions")
    
    # Perform skills clustering
    predicted_labels = clusterer.fit_predict(skills, n_clusters=None, algo=algo)
    
    if predicted_labels is not None:
        print(f"\n✅ Skills Clustering Results:")
        print(f"Input skills: {len(skills)}")
        print(f"Output labels: {len(predicted_labels)}")
        print(f"Dimensions match: {len(skills) == len(predicted_labels)}")
        
        if len(skills) == len(predicted_labels):

            print(f"Discovered clusters: {len(set(predicted_labels[predicted_labels != -1]))}")
            
            # Visualize skill clusters
            # clusterer.plot_skill_clusters()
            
            return clusterer
        else:
            print("❌ ERROR: Dimension mismatch!")
            return None
    else:
        print("❌ Skills clustering failed!")
        return None
