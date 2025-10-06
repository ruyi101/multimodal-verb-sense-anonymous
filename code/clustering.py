from clustering_utils import *
import json
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import random





def run_step1_clustering(embedding_file, df_rows, dataset_name, results_folder, clustering_params):
    """
    Runs Step 1 clustering (image embeddings only) and saves results for reuse in Step 2.
    """
    
    # Ensure results folder exists
    os.makedirs(results_folder, exist_ok=True)

    step1_filename = f"{results_folder}/{dataset_name}_step1_{clustering_params['method']}_{clustering_params['metric']}.csv"
    if os.path.exists(step1_filename):
        print(f"Loading existing step 1 results from: {step1_filename}")
        df_clustered = pd.read_csv(step1_filename)
        return df_clustered

    # Extract unique verbs
    verbs = df_rows['verbs'].unique()

    all_clustering_results = []

    for verb in tqdm(verbs, desc=f'Running Step 1: {clustering_params["method"]} clustering'):
        record = {'verbs': verb}

        # Select data for the current verb
        data = df_rows[df_rows['verbs'] == verb].copy()


        if dataset_name == 'llama_90b_top5' and verb == 'standing':  # 38155 images, out of memory
            data = data.sample(frac=0.5, random_state=1108)


        # Get image names
        image_names = data['image'].tolist()
        texts = data['verbs'].tolist()



#############################################################################################################
        # Step 1: Load precomputed embeddings
        embeddings = np.array([
            load_mllama_embedding(embedding_file, text=text, images=img) for text, img in zip(texts, image_names)
        ])
#############################################################################################################


        # Find optimal number of clusters using silhouette score
        best_silhouette = -1
        best_num_clusters = clustering_params['min_clusters']

        for num_clusters in range(clustering_params['min_clusters'], clustering_params['max_clusters'] + 1):
            clusters = cluster_embeddings(
                embeddings, 
                method=clustering_params['method'], 
                metric=clustering_params['metric'], 
                linkage=clustering_params.get('linkage', None), 
                n_clusters=num_clusters
            )
            score = silhouette_score(embeddings, clusters, metric=clustering_params['metric'])
            
            if score > best_silhouette:
                best_silhouette = score
                best_num_clusters = num_clusters

        # Perform final clustering with the optimal cluster count
        clusters = cluster_embeddings(
            embeddings, 
            method=clustering_params['method'], 
            metric=clustering_params['metric'], 
            linkage=clustering_params.get('linkage', None), 
            n_clusters=best_num_clusters
        )

        # Store clustering results
        data['cluster'] = clusters
        all_clustering_results.append(data)

    # Save first clustering results with a unique name
    df_clustered = pd.concat(all_clustering_results)
    step1_filename = f"{results_folder}/{dataset_name}_step1_{clustering_params['method']}_{clustering_params['metric']}.csv"
    df_clustered.to_csv(step1_filename, index=False)

    print(f"Step 1 results saved: {step1_filename}")
    return df_clustered




def run_step2_clustering(df_clustered, embedding_file, dataset_name, results_folder, clustering_params):
    """
    Runs Step 2 clustering using the saved results from Step 1 and different embedding combination strategies.
    """

    step2_filename = f"{results_folder}/{dataset_name}_step2_{clustering_params['method']}_{clustering_params['metric']}.csv"
    if os.path.exists(step2_filename):
        print(f"Loading existing step 2 results from: {step2_filename}")
        grouped_data = pd.read_csv(step2_filename)
        return grouped_data

    # Step 1.5: Group Data for Next Step
    grouped_data = df_clustered.groupby(['verbs', 'cluster']).agg(list).reset_index()

    # Step 2: Process grouped images & texts
    images = grouped_data['image'].tolist()
    texts = grouped_data['verbs'].tolist()



#############################################################################################################
    # Step 2: Load embeddings and apply combination strategy using embed_images_with_text_clip
    embeddings_step2 = np.array([
         load_mllama_embedding(embedding_file, text=text, images=images) 
         for images, text in tqdm(zip(images, texts), total=len(images), desc=f"Embedding for Step 2")
    ])
#############################################################################################################


    # Step 2: Determine number of clusters using silhouette score
    num_verbs = len(df_clustered['verbs'].unique())
    cluster_candidates = [int(num_verbs * scale) for scale in np.arange(0.6, 1.7, 0.1)]
    cluster_candidates = sorted(set(cluster_candidates))  # Ensure unique values

    best_silhouette_step2 = -1
    best_num_clusters_step2 = max(2, cluster_candidates[0])  # Ensure at least 2 clusters

    for num_clusters in tqdm(cluster_candidates, desc=f'Running Step 2: clustering'):
        clusters_step2 = cluster_embeddings(
            embeddings_step2, 
            method=clustering_params['method'], 
            metric=clustering_params['metric'], 
            linkage=clustering_params.get('linkage', None), 
            n_clusters=num_clusters
        )
        score = silhouette_score(embeddings_step2, clusters_step2, metric=clustering_params['metric'])

        if score > best_silhouette_step2:
            best_silhouette_step2 = score
            best_num_clusters_step2 = num_clusters

    # Perform final clustering with the optimal cluster count
    clusters_step2 = cluster_embeddings(
        embeddings_step2, 
        method=clustering_params['method'], 
        metric=clustering_params['metric'], 
        linkage=clustering_params.get('linkage', None), 
        n_clusters=best_num_clusters_step2
    )

    grouped_data['final_cluster'] = clusters_step2

    # Save final clustering results with a unique name
    step2_filename = f"{results_folder}/{dataset_name}_step2_{clustering_params['method']}_{clustering_params['metric']}.csv"
    grouped_data.to_csv(step2_filename, index=False)

    print(f"Step 2 results saved: {step2_filename}")
    return grouped_data
