import numpy as np
# import random
# from typing import List, Tuple, Set, Dict

class EarlyStopMonitor(object):
	def __init__(self, max_round=20, higher_better=True, tolerance=1e-10):
		self.max_round = max_round
		self.num_round = 0

		self.epoch_count = 0
		self.best_epoch = 0

		self.last_best = None
		self.higher_better = higher_better
		self.tolerance = tolerance

	def early_stop_check(self, curr_val):
		if not self.higher_better:
			curr_val *= -1
		if self.last_best is None:
			self.last_best = curr_val
		elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
			self.last_best = curr_val
			self.num_round = 0
			self.best_epoch = self.epoch_count
		else:
			self.num_round += 1

		self.epoch_count += 1

		return self.num_round >= self.max_round


def get_neighbor_finder(data, uniform):
    max_node_idx = max(data.src.max(), data.dst.max())
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for src, dst, edge_idx, timestamp, flag in zip(data.src, data.dst, data.edge_idxs, data.timestamps, data.flags):
        if flag == 1:
            adj_list[src].append((dst, edge_idx, timestamp, np.inf))
            adj_list[dst].append((src, edge_idx, timestamp, np.inf))
        else:
            # If the edge is deleted, set deletion time to timestamp
            for i, (neighbor, idx, ins_time, _del_time) in enumerate(adj_list[src]):
                if neighbor == dst and idx == edge_idx:
                    adj_list[src][i] = (neighbor, idx, ins_time, timestamp)
            for i, (neighbor, idx, ins_time, _del_time) in enumerate(adj_list[dst]):
                if neighbor == src and idx == edge_idx:
                    adj_list[dst][i] = (neighbor, idx, ins_time, timestamp)
                
    return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        """
        adj_list: list of lists, where each sublist contains tuples of the form
        (neighbor, edge_idx, insertion_time, deletion_time)
        """
        if not isinstance(adj_list, list):
            raise TypeError("adj_list must be a list of lists.")
        self.num_nodes = len(adj_list) - 1
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_ins_time_stamps = []
        self.node_to_edge_del_time_stamps = []
        for neighbors in adj_list:
            if not isinstance(neighbors, list):
                raise TypeError("Each element of adj_list must be a list.")
            # format of neighbors: (neighbor, edge_idx, insertion_time, deletion_time)
            # sorted based on insertion time
            sorted_neighbor = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbor]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbor]))
            self.node_to_edge_ins_time_stamps.append(np.array([x[2] for x in sorted_neighbor]))
            self.node_to_edge_del_time_stamps.append(np.array([x[3] for x in sorted_neighbor]))

        self.uniform = uniform
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)
        else:
            self.random_state = np.random

    def reset_random_state(self, seed=None):
        """
        Resets the random state to a new seed or the original seed.
        If seed is None, it resets to the original seed.
        """
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)
        else:
            self.random_state = np.random

    def find_before(self, src_idx, cut_time, include_deleted=False):
        """
        Extracts all the interactions (insertions) happening before cut_time
        for user src_idx, and their corresponding deletion times.
        Returns 4 arrays: neighbors, edge_idxs, insertion_timestamps, deletion_timestamps

        If include_deleted is False (default): only returns neighbors not deleted before cut_time.
        If include_deleted is True: returns all neighbors inserted before cut_time, regardless of deletion.
        """
        if src_idx < 0 or src_idx >= len(self.node_to_neighbors):
            raise IndexError(f"src_idx {src_idx} is out of bounds.")
        i = np.searchsorted(self.node_to_edge_ins_time_stamps[src_idx], cut_time)
        neighbors = self.node_to_neighbors[src_idx][:i]
        edge_idxs = self.node_to_edge_idxs[src_idx][:i]
        ins_times = self.node_to_edge_ins_time_stamps[src_idx][:i]
        del_times = self.node_to_edge_del_time_stamps[src_idx][:i]
        # For edges not deleted before cut_time, set deletion time to np.inf
        del_times = np.where(del_times < cut_time.item(), del_times, np.inf)
        if not include_deleted:
            mask = (del_times == np.inf)
            neighbors = neighbors[mask]
            edge_idxs = edge_idxs[mask]
            ins_times = ins_times[mask]
            del_times = del_times[mask]
        return neighbors, edge_idxs, ins_times, del_times
        
    def get_active_edges(self, t):
        """
        Returns a set of (src, neighbor) pairs for all nodes, where the edge is active 
        (inserted before t, not deleted before t).
        """
        active_edges = set()
        for src in range(len(self.node_to_neighbors)):
            neighbors, _, _, _ = self.find_before(src, t, include_deleted=False)
            for n in neighbors:
                edge = (min(src, int(n)), max(src, int(n)))
                active_edges.add(edge)
        return active_edges

    def get_temporal_neighbor(self, src_nodes, timestamps, n_neighbors=200, include_deleted=False):
        """
        For each src_node and cut_time, extract up to n_neighbors neighbors
        whose insertion time is before cut_time, along with their edge idxs,
        insertion and deletion times.

        If include_deleted is False (default): only returns neighbors not deleted before cut_time.
        If include_deleted is True: returns all neighbors inserted before cut_time, regardless of deletion.
        """

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1

        # Use 0 for missing neighbors, np.inf for missing times
        neighbors = np.full((len(src_nodes), tmp_n_neighbors), 0, dtype=np.float32)
        edge_idxs = np.full((len(src_nodes), tmp_n_neighbors), 0, dtype=np.int32)
        ins_times = np.full((len(src_nodes), tmp_n_neighbors), np.inf, dtype=np.float32)
        del_times = np.full((len(src_nodes), tmp_n_neighbors), np.inf, dtype=np.float32)

        for i, (src_node, timestamp) in enumerate(zip(src_nodes, timestamps)):
            if src_node < 0 or src_node >= len(self.node_to_neighbors):
                raise IndexError(f"src_node {src_node} at position {i} is out of bounds.")
            src_neighbors, src_edge_idxs, src_ins_times, src_del_times = \
                self.find_before(src_node, timestamp, include_deleted=include_deleted)

            if len(src_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:
                    sample_idx = self.random_state.randint(0, len(src_neighbors), n_neighbors)
                    neighbors[i, :] = src_neighbors[sample_idx]
                    edge_idxs[i, :] = src_edge_idxs[sample_idx]
                    ins_times[i, :] = src_ins_times[sample_idx]
                    del_times[i, :] = src_del_times[sample_idx]
                    # sort by insertion time
                    pos = ins_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                    ins_times[i, :] = ins_times[i, :][pos]
                    del_times[i, :] = del_times[i, :][pos]
                else:
                    # Take most recent insertions
                    source_neighbors = src_neighbors[-n_neighbors:]
                    source_edge_idxs = src_edge_idxs[-n_neighbors:]
                    source_ins_times = src_ins_times[-n_neighbors:]
                    source_del_times = src_del_times[-n_neighbors:]
                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
                    ins_times[i, n_neighbors - len(source_ins_times):] = source_ins_times
                    del_times[i, n_neighbors - len(source_del_times):] = source_del_times
        return neighbors, edge_idxs, ins_times, del_times
    
    def negative_sample(self, src_list, dst_list, timestamps, flags):
        """
        Samples negative edges for link prediction tasks.
        Returns a list of tuples (src, dst) representing negative samples.
        """

        if len(src_list) != len(dst_list):
            raise ValueError("src_list and dst_list must have the same length.")
        
        negative_samples = []

        for src, dst, timestamp, flag in zip(src_list, dst_list, timestamps, flags):
            if flag == 1:  # Only sample negatives for insertions
                # Get all neighbors of src at the given timestamp
                neighbors, _, _, _ = self.find_before(src, timestamp)
                if len(neighbors) > 0:
                    # Sample a random neighbor to create a negative sample
                    neg_idx = self.random_state.choice(len(neighbors))
                    neg_dst = neighbors[neg_idx]
                else:
                    # If no neighbors, sample a random dst from the dst_list
                    neg_dst = self.random_state.randint(1, self.num_nodes)
                negative_samples.append(neg_dst)
            elif flag == -1:
                # For deletions, we can sample a random edge that exists at the timestamp
                active_edges = self.get_active_edges(timestamp)
                if len(active_edges) > 0:
                    active_edges_list = list(active_edges)
                    idx = self.random_state.randint(0, len(active_edges_list))
                    neg_edge = active_edges_list[idx]
                    neg_dst = neg_edge[0] if neg_edge[0] != src else neg_edge[1]
                else:
                    neg_dst = self.random_state.randint(1, self.num_nodes)
                negative_samples.append(neg_dst)
        return negative_samples