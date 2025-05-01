import pandas as pd
import numpy as np
import random

# Reproducibility (optional)
np.random.seed(42)
random.seed(42)

# Hyperparameters
p = 0.2    # Probability of inserting a synthetic removal
q = 0.01   # Scale factor for Exponential noise

# Step 1: Read real entries, record min/max ts
real_entries = []
min_ts = float('inf')
max_ts = float('-inf')

with open("../data/out.opsahl-ucsocial.txt", "r") as raw_data:
    for idx, line in enumerate(raw_data):
        if idx < 2:
            continue  # Skip first two lines
        if line.strip() == "":
            continue  # Skip empty lines
        parts = line.strip().split()
        if len(parts) != 4:
            continue  # Skip malformed lines
        
        u, i, f, ts = map(int, parts)
        real_entries.append([u, i, f, ts])
        min_ts = min(min_ts, ts)
        max_ts = max(max_ts, ts)

# Step 2: Compute r
r = max_ts - min_ts
if r == 0:
    r = 1  # Safeguard against zero division

# Step 3: Shift timestamps in real entries
real_entries_shifted = [[u, i, f, ts - min_ts] for u, i, f, ts in real_entries]

# Step 4: Build real DataFrame
df_real = pd.DataFrame(real_entries_shifted, columns=["u", "i", "f", "ts"])
df_real.insert(0, "idx", range(len(df_real)))  # Add idx starting from 0

# Save real entries to CSV
df_real.to_csv("../data/ml_UCI-Msg.csv", index=False)

# Step 5: Inject synthetic removals (after ts-shifting)
synthetic_entries = []
for u, i, f, ts in real_entries_shifted:
    if random.random() < p:
        delta = np.random.exponential(q * r)
        synthetic_ts = ts + int(round(delta))
        synthetic_entries.append([u, i, -1, synthetic_ts])

# Step 6: Combine real and synthetic entries
combined_entries = real_entries_shifted + synthetic_entries

# Step 7: Build combined DataFrame
df_combined = pd.DataFrame(combined_entries, columns=["u", "i", "f", "ts"])

# Step 8: Sort by ts
df_combined = df_combined.sort_values(by="ts").reset_index(drop=True)

# Step 9: Assign new idx after sorting
df_combined.insert(0, "idx", range(len(df_combined)))

# Step 10: Save combined DataFrame
df_combined.to_csv("../data/ml_UCI-Msg-del.csv", index=False)

# ---------------------
# Now generate .npy files
# ---------------------

def generate_edge_features(df, save_path):
    # Generate a single shared vector for all edges
    shared_edge_feat = np.random.randn(1, 128).astype(np.float32)  # Single 1x128 vector
    edge_feats = np.repeat(shared_edge_feat, repeats=len(df), axis=0)  # Repeat it n times
    np.save(save_path, edge_feats)

def generate_node_features(df, save_path):
    # Collect all unique nodes from src and dst columns
    unique_nodes = set(df['u']).union(set(df['i']))
    unique_nodes = sorted(unique_nodes)  # Sort to keep consistent order
    n_nodes = len(unique_nodes)

    # Generate a single shared node feature vector
    shared_node_feat = np.random.randn(1, 128).astype(np.float32)  # Single 1x128 vector
    node_feats = np.repeat(shared_node_feat, repeats=n_nodes, axis=0)  # Repeat it n_nodes times

    np.save(save_path, node_feats)

# Generate edge and node features for real dataset
generate_edge_features(df_real, "../data/ml_ml_UCI-Msg.npy")
generate_node_features(df_real, "../data/ml_ml_UCI-Msg_node.npy")

# Generate edge and node features for combined dataset
generate_edge_features(df_combined, "../data/ml_ml_UCI-Msg-del.npy")
generate_node_features(df_combined, "../data/ml_ml_UCI-Msg-del_node.npy")