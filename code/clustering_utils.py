import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances
from sklearn.metrics import davies_bouldin_score
import json
import os
from sklearn.preprocessing import normalize



def load_mllama_embedding(embedding_file, text, images):
    """
    Load embeddings for a given text and one or multiple images from a preloaded dictionary.

    Args:
        embedding_file (dict): Dictionary containing embeddings with keys as (text, image) tuples.
        text (str): A single text label (verb).
        images (str or list of str): A single image filename or a list of image filenames.

    Returns:
        np.ndarray: The retrieved or averaged embedding.
    """
    if isinstance(images, str):  # Single image case
        images = [images]  # Convert to list for consistency
    images = [image.split('.')[0] for image in images]
    embeddings = []
    for image in images:
        key = (text, image)
        if key in embedding_file:
            embeddings.append(embedding_file[key])
        else:
            print(f"Warning: Embedding not found for ({text}, {image}), using zeros.")
            embeddings.append(np.zeros_like(next(iter(embedding_file.values()))))  # Use a zero vector

    # Return the average if multiple images were provided
    return np.mean(embeddings, axis=0)





def cluster_embeddings(embeddings, method='agglomerative', **kwargs):
    """
    Perform clustering on a list of image embeddings.

    Args:
        embeddings (list or np.ndarray): List of image embeddings.
        method (str): Clustering method ('agglomerative', 'kmeans').
        **kwargs: Additional arguments for the clustering method.

    Returns:
        np.ndarray: Cluster labels.
    """
    # Convert list of embeddings to a NumPy array if needed
    embeddings = np.array(embeddings)

    # Check if metric is cosine and normalize embeddings
    metric = kwargs.get('metric', 'euclidean')
    if method == 'kmeans' and metric == 'cosine':
        embeddings = normalize(embeddings, norm='l2')  # Normalize for cosine similarity

    # Choose clustering method
    if method == 'agglomerative':
        cluster_model = AgglomerativeClustering(
            n_clusters=kwargs.get('n_clusters', None),
            metric=metric,
            linkage=kwargs.get('linkage', 'complete'),
            distance_threshold=kwargs.get('distance_threshold', None)
        )
    elif method == 'kmeans':
        cluster_model = KMeans(
            n_clusters=kwargs.get('n_clusters', 8),
            random_state=kwargs.get('random_state', 42),
            n_init=10  # Ensuring better convergence
        )
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    return cluster_model.fit_predict(embeddings)




def calculate_silhouette_score(embeddings, labels, metric='euclidean'):
    """
    Calculate the silhouette score for clustering results.

    Args:
        embeddings (np.ndarray): Array of image embeddings.
        labels (np.ndarray): Cluster labels assigned to each embedding.
        metric (str): Distance metric to use ('euclidean' or 'cosine').

    Returns:
        float: Silhouette score (higher is better).
    """
    if metric == 'cosine':
        # Convert cosine similarity to cosine distance (1 - similarity)
        distance_matrix = cosine_distances(embeddings)
        return silhouette_score(distance_matrix, labels, metric='precomputed')
    elif metric == 'euclidean':
        return silhouette_score(embeddings, labels, metric='euclidean')
    else:
        raise ValueError("Invalid metric. Choose either 'euclidean' or 'cosine'.")
    

def calculate_davies_bouldin_index(embeddings, labels, metric='euclidean'):
    """
    Calculate the Davies-Bouldin Index (DBI) for clustering results.

    Args:
        embeddings (np.ndarray): Array of image embeddings.
        labels (np.ndarray): Cluster labels assigned to each embedding.
        metric (str): Distance metric to use ('euclidean' or 'cosine').

    Returns:
        float: Davies-Bouldin Index (lower is better).
    """
    if len(set(labels)) < 2:
        raise ValueError("Davies-Bouldin Index requires at least 2 clusters.")

    if metric == 'cosine':
        # Convert cosine similarity to cosine distance (1 - similarity)
        distance_matrix = cosine_distances(embeddings)
        return davies_bouldin_score(distance_matrix, labels)
    elif metric == 'euclidean':
        return davies_bouldin_score(embeddings, labels)
    else:
        raise ValueError("Invalid metric. Choose either 'euclidean' or 'cosine'.")



def calculate_within_cluster_similarity(embeddings, labels, metric='euclidean'):
    """
    Compute the average within-cluster similarity for K-Means clustering.

    Args:
        embeddings (np.ndarray): Array of image embeddings (shape: (n_samples, n_features)).
        labels (np.ndarray): Cluster labels assigned by K-Means.
        metric (str): Distance metric to use ('euclidean' or 'cosine').

    Returns:
        float: Average within-cluster similarity (higher is better for cosine, lower is better for Euclidean).
    """
    unique_clusters = np.unique(labels)
    similarities = []

    for cluster in unique_clusters:
        # Get points belonging to the current cluster
        cluster_points = embeddings[labels == cluster]

        if len(cluster_points) > 1:  # Avoid empty or singleton clusters
            if metric == 'cosine':
                # Compute cosine similarity matrix within the cluster
                sim_matrix = cosine_similarity(cluster_points)
                # Take the mean of all similarities excluding self-similarities (diagonal elements)
                mean_similarity = (np.sum(sim_matrix) - np.trace(sim_matrix)) / (len(cluster_points) * (len(cluster_points) - 1))
            elif metric == 'euclidean':
                # Compute Euclidean distance matrix within the cluster
                dist_matrix = euclidean_distances(cluster_points)
                # Take the mean of all distances excluding self-distances (diagonal)
                mean_similarity = np.mean(dist_matrix)
            else:
                raise ValueError("Invalid metric. Choose either 'euclidean' or 'cosine'.")

            similarities.append(mean_similarity)

    # Return the average similarity across all clusters
    return np.mean(similarities) if similarities else None



# Convert NumPy types to Python types before dumping JSON
def convert_to_serializable(obj):
    """
    Convert NumPy types (float32, int64, ndarray) to JSON serializable types.
    """
    if isinstance(obj, np.ndarray):  # Convert arrays to lists
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):  # Convert floats to Python float
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):  # Convert ints to Python int
        return int(obj)
    elif isinstance(obj, list):  # Recursively process lists
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):  # Recursively process dictionaries
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj  # Return as is if no conversion needed
