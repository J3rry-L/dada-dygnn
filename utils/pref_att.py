import pandas as pd
import numpy as np
import random
from collections import defaultdict

def generate_preferential_graph(num_events=50000, p1=0.05, p2=0.9, beta_ts=100, seed=42):
    assert p1 + p2 < 1, "p1 + p2 must be less than 1 so that p3 < 1/2"
    p3 = 1 - p1 - p2
    np.random.seed(seed)
    random.seed(seed)

    edges = set()
    degrees = defaultdict(int)
    nodes = set()
    data = []

    current_time = 0
    next_node_id = 0

    def weighted_random_node():
        total_degree = sum(degrees[n] for n in nodes)
        if total_degree == 0:
            return random.choice(list(nodes))
        r = random.uniform(0, total_degree)
        cum = 0
        for n in nodes:
            cum += degrees[n]
            if cum >= r:
                return n

    for _ in range(num_events):
        op = np.random.choice(['p1', 'p2', 'p3'], p=[p1, p2, p3])
        current_time += int(np.random.exponential(beta_ts))

        if op == 'p1':
            # Add a new node and attach it to an existing node (preferential)
            new_node = next_node_id
            next_node_id += 1
            if nodes:
                existing_node = weighted_random_node()
                edges.add((new_node, existing_node))
                degrees[new_node] += 1
                degrees[existing_node] += 1
                data.append([new_node, existing_node, 1, current_time])
            nodes.add(new_node)

        elif op == 'p2':
            # Add an edge between two existing nodes
            if len(nodes) >= 2:
                u = weighted_random_node()
                v = random.choice(list(nodes - {u}))
                if (u, v) not in edges and (v, u) not in edges:
                    edges.add((u, v))
                    degrees[u] += 1
                    degrees[v] += 1
                    data.append([u, v, 1, current_time])

        elif op == 'p3':
            # Delete a random edge from a high-degree node
            if edges:
                u = weighted_random_node()
                # find an edge involving u
                candidate_edges = [e for e in edges if u in e]
                if candidate_edges:
                    e = random.choice(candidate_edges)
                    edges.remove(e)
                    u1, u2 = e
                    degrees[u1] -= 1
                    degrees[u2] -= 1
                    data.append([u1, u2, -1, current_time])
    
        # Update nodes set
        for u in degrees:
            if degrees[u] > 0:
                nodes.add(u)

    df = pd.DataFrame(data, columns=['u', 'i', 'f', 'ts'])
    return df

df = generate_preferential_graph()
df.insert(0, "idx", range(len(df)))  # Add idx starting from 0
df.to_csv("../data/ml_pref-att.csv", index=False)

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
generate_edge_features(df, "../data/ml_pref-att.npy")
generate_node_features(df, "../data/ml_pref-att_node.npy")