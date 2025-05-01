import numpy as np
import torch
import math

def eval_edge_prediction(model, neighbor_finder, data, bs, mlp=None, flags=None):
	"""
	If mlp is provided, use tanh(mlp) output for ranking; otherwise, use model.compute_sim.
	flags: array of +1 (insertion) or -1 (deletion) for each event in data.
	Returns:
		- aggregate mean MRR, recall@20, recall@50
		- per-class (insertion, deletion) mean MRR, recall@20, recall@50
	"""
	neighbor_finder.reset_random_state()
	val_mrr, val_recall_20, val_recall_50 = [], [], []
	mrr_insert, recall20_insert, recall50_insert = [], [], []
	mrr_delete, recall20_delete, recall50_delete = [], [], []
	with torch.no_grad():
		num_batch = math.ceil(len(data.src) / bs)
		for batch_idx in range(num_batch):
			st_idx = batch_idx * bs
			ed_idx = min((batch_idx + 1) * bs, len(data.src))
			flag_batch = None
			if flags is not None:
				flag_batch = flags[st_idx:ed_idx]
			src_batch = data.src[st_idx:ed_idx]
			dst_batch = data.dst[st_idx:ed_idx]
			edge_batch = data.edge_idxs[st_idx:ed_idx]
			timestamps_batch = data.timestamps[st_idx:ed_idx]

			negative_batch = neighbor_finder.negative_sample(dst_batch, src_batch, timestamps_batch, flag_batch)

			if mlp is not None:
				# Use MLPLinkPredictor for scoring
				emb_all = model.memory.emb
				mrrs, recalls_20, recalls_50 = [], [], []
				mrrs_ins, recalls_20_ins, recalls_50_ins = [], [], []
				mrrs_del, recalls_20_del, recalls_50_del = [], [], []
				for i in range(len(src_batch)):
					s = src_batch[i]
					d = dst_batch[i]
					flag = flag_batch[i] if flag_batch is not None else 1
					emb_s = emb_all[s].unsqueeze(0).repeat(emb_all.shape[0], 1)
					emb_all_nodes = emb_all
					# Score for all possible dsts
					scores = mlp(emb_s, emb_all_nodes).squeeze(-1).cpu().numpy()
					# For insertions: sort descending, for deletions: sort ascending
					if flag == 1:
						ranking = np.argsort(-scores)
					else:
						ranking = np.argsort(scores)
					mrr_val = compute_mrr_single(d, ranking)
					recall20_val = compute_recall_single(d, ranking, 20)
					recall50_val = compute_recall_single(d, ranking, 50)
					mrrs.append(mrr_val)
					recalls_20.append(recall20_val)
					recalls_50.append(recall50_val)
					if flag == 1:
						mrrs_ins.append(mrr_val)
						recalls_20_ins.append(recall20_val)
						recalls_50_ins.append(recall50_val)
					else:
						mrrs_del.append(mrr_val)
						recalls_20_del.append(recall20_val)
						recalls_50_del.append(recall50_val)
				# Repeat for dst->src direction
				for i in range(len(dst_batch)):
					d = dst_batch[i]
					s = src_batch[i]
					flag = flag_batch[i] if flag_batch is not None else 1
					emb_d = emb_all[d].unsqueeze(0).repeat(emb_all.shape[0], 1)
					emb_all_nodes = emb_all
					scores = mlp(emb_d, emb_all_nodes).squeeze(-1).cpu().numpy()
					if flag == 1:
						ranking = np.argsort(-scores)
					else:
						ranking = np.argsort(scores)
					mrr_val = compute_mrr_single(s, ranking)
					recall20_val = compute_recall_single(s, ranking, 20)
					recall50_val = compute_recall_single(s, ranking, 50)
					mrrs.append(mrr_val)
					recalls_20.append(recall20_val)
					recalls_50.append(recall50_val)
					if flag == 1:
						mrrs_ins.append(mrr_val)
						recalls_20_ins.append(recall20_val)
						recalls_50_ins.append(recall50_val)
					else:
						mrrs_del.append(mrr_val)
						recalls_20_del.append(recall20_val)
						recalls_50_del.append(recall50_val)
				val_mrr.append(np.mean(mrrs))
				val_recall_20.append(np.mean(recalls_20))
				val_recall_50.append(np.mean(recalls_50))
				if mrrs_ins:
					mrr_insert.append(np.mean(mrrs_ins))
					recall20_insert.append(np.mean(recalls_20_ins))
					recall50_insert.append(np.mean(recalls_50_ins))
				if mrrs_del:
					mrr_delete.append(np.mean(mrrs_del))
					recall20_delete.append(np.mean(recalls_20_del))
					recall50_delete.append(np.mean(recalls_50_del))
			else:
				src_cos_sim, src_idx = model.compute_sim(src_batch)
				dst_cos_sim, dst_idx = model.compute_sim(dst_batch)
				recall_20 = recall(dst_batch, src_idx, 20) + recall(src_batch, dst_idx, 20)
				recall_50 = recall(dst_batch, src_idx, 50) + recall(src_batch, dst_idx, 50)
				mrr = MRR(dst_batch, src_idx) + MRR(src_batch, dst_idx)
				val_recall_20.append(recall_20 / 2)
				val_recall_50.append(recall_50 / 2)
				val_mrr.append(mrr / 2)
				model(src_batch, dst_batch, src_batch, negative_batch, edge_batch, timestamps_batch)

	results = {
		"agg_mrr": np.mean(val_mrr),
		"agg_recall_20": np.mean(val_recall_20),
		"agg_recall_50": np.mean(val_recall_50),
		"insertion_mrr": np.mean(mrr_insert) if mrr_insert else 0,
		"insertion_recall_20": np.mean(recall20_insert) if recall20_insert else 0,
		"insertion_recall_50": np.mean(recall50_insert) if recall50_insert else 0,
		"deletion_mrr": np.mean(mrr_delete) if mrr_delete else 0,
		"deletion_recall_20": np.mean(recall20_delete) if recall20_delete else 0,
		"deletion_recall_50": np.mean(recall50_delete) if recall50_delete else 0,
	}
	return results


def compute_mrr_single(target, ranking):
	"""Compute MRR for a single target given a ranking (np.array of indices, descending order)."""
	rank = np.where(ranking == target)[0]
	if len(rank) == 0:
		return 0.0
	return 1.0 / (rank[0] + 1)

def compute_recall_single(target, ranking, top_k):
	"""Compute recall@k for a single target given a ranking."""
	return float(target in ranking[:top_k])

def recall(dst_idxs, idx, top_k):
	bs = idx.shape[0]
	idx = idx[:, :top_k]
	rec = np.array([a in idx[i].cpu() for i, a in enumerate(dst_idxs)])
	rec = rec.sum() / rec.size
	return rec

def MRR(dst_idxs, idx):
	bs = idx.shape[0]
	mrr = np.array([float(np.where(idx[i].cpu() == a)[0] + 1) for i, a in enumerate(dst_idxs)])
	mrr = (1 / mrr).mean()
	return mrr
