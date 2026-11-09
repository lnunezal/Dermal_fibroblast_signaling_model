import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Load edge list directly from Excel (parses Rule column)
def load_edge_list_from_excel(filepath, sheet="reactions"):
    df = pd.read_excel(filepath, sheet_name=sheet, header=1)
    df.columns = df.columns.str.lower().str.strip()

    # Skip first 10 rows (inputs module)
    df = df.iloc[10:]

    edges = []
    for rule in df['rule']:
        regulators, target = rule.split("=>")
        regulators = regulators.strip()
        target = target.strip()
        edges.append((regulators, target))

    return edges

# Build dataframe for clustering
def build_edge_dataframe(edges):
    df_edges = pd.DataFrame(edges, columns=['source', 'target'])
    df_edges['interaction'] = 'unknown'  # No interaction column, we treat all equally for now
    return df_edges

# Clean interaction values (placeholder — can be extended)
def clean_interaction(val):
    try:
        return int(val)
    except:
        return 0

# Clustering function
def cluster_edges(df_edges, n_clusters):
    df_edges['interaction_clean'] = df_edges['interaction'].apply(clean_interaction)
    le_source = LabelEncoder()
    le_target = LabelEncoder()
    df_edges['source_enc'] = le_source.fit_transform(df_edges['source'])
    df_edges['target_enc'] = le_target.fit_transform(df_edges['target'])
    X = df_edges[['source_enc', 'target_enc', 'interaction_clean']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_edges['family'] = kmeans.fit_predict(X)
    return df_edges

# Save/load mapping functions
def build_edge_family_map(df_edges):
    return {(row['source'], row['target']): int(row['family']) for _, row in df_edges.iterrows()}

def save_edge_family_csv(df_edges, filename):
    df_edges.to_csv(filename, index=False)

def load_edge_family_map(filename):
    df_edges = pd.read_csv(filename)
    return {(row['source'], row['target']): int(row['family']) for _, row in df_edges.iterrows()}

