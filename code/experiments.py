from clustering import run_step1_clustering, run_step2_clustering
from clustering_utils import *
import json
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# Read in data


test = json.load(open("Data/test.json"))
train = json.load(open("Data/train.json"))
val = json.load(open('Data/dev.json'))

imsitu = train|test|val

verbs = [val[img]['verb'] for img in val]
verbs = list(set(verbs))


gpt_top5_data = json.load(open('Data/gpt_top_5_rows.json'))
gpt_top5_data = pd.DataFrame.from_dict(gpt_top5_data)
gpt_top5_data = gpt_top5_data[gpt_top5_data['verbs'].isin(verbs)].copy()




llama_90b_504verbs_data = json.load(open('Data/llama_90b_504_verbs_rows.json'))
llama_90b_504verbs_data = pd.DataFrame.from_dict(llama_90b_504verbs_data)
llama_90b_504verbs_data = llama_90b_504verbs_data[llama_90b_504verbs_data['verbs'].isin(verbs)].copy()


llama_90b_top5_data = json.load(open('Data/llama_90b_top_5_rows.json'))
llama_90b_top5_data = pd.DataFrame.from_dict(llama_90b_top5_data)
llama_90b_top5_data = llama_90b_top5_data[llama_90b_top5_data['verbs'].isin(verbs)].copy()


# Set experiment parameters
results_folder = "clustering_results_llama"       # Output directory
datasets = [
    ("llama_90b_504verbs", llama_90b_504verbs_data),
    ("gpt_top5", gpt_top5_data),
    ("llama_90b_top5", llama_90b_top5_data)
    ]

###############################################
embedding_file = ...
###############################################




# Define clustering configurations
clustering_methods = [
    {'method': 'kmeans', 'metric': 'cosine', 'linkage': None},
    {'method': 'agglomerative', 'metric': 'cosine', 'linkage': 'complete'}
]


# Ensure clustering results folder exists
os.makedirs(results_folder, exist_ok=True)
for dataset_name, df_rows in datasets:
    # Run Step 1 once for each clustering method
    for clustering_config in clustering_methods:
        print(f"\nRunning Step 1: {clustering_config['method']} - {clustering_config['metric']}")
        
        clustering_config.update({
            'min_clusters': 2,
            'max_clusters': 16
        })

        # Run Step 1 and save results
        df_clustered = run_step1_clustering(
            embedding_file=embedding_file,
            df_rows=df_rows,
            dataset_name=dataset_name,
            results_folder=results_folder,
            clustering_params=clustering_config
        )

        # Run Step 2 for each combination option
        print(f"\nRunning Step 2: {clustering_config['method']} - {clustering_config['metric']}")
        
        # Run Step 2
        df_results = run_step2_clustering(
            df_clustered=df_clustered,
            embedding_file=embedding_file,
            dataset_name=dataset_name,
            results_folder=results_folder,
            clustering_params=clustering_config,
        )

            # print(df_results.head())