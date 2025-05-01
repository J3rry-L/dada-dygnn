import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from modules.memory import Memory
from torch.distributions import Bernoulli


class DadaDyGNN(nn.Module):
	def __init__(self, n_nodes, n_neighbors, n_update_neighbors,
				 edge_dim, emb_dim, message_dim,
				 neighbor_finder, edge_feat, tot_time, device='cpu'):
		super(DadaDyGNN, self).__init__()
		torch.manual_seed(0)
		# graph
		self.damp_list = []
		self.n_nodes = n_nodes
		# Dimensions
		self.edge_dim = edge_dim
		self.emb_dim = emb_dim
		self.message_dim = message_dim
		# memory
		self.edge_feat = torch.tensor(edge_feat, dtype=torch.float, device=device)
		self.memory = Memory(self.n_nodes + 1, self.emb_dim, device)
		# other
		self.tot_time = tot_time
		self.n_neighbors = n_neighbors
		self.n_update_neighbors = n_update_neighbors
		self.neighbor_finder = neighbor_finder
		self.device = device
		self.eps = 1e-10

		# GAT
		self.W_g = nn.Parameter(torch.zeros((self.emb_dim, self.message_dim), device=self.device))  # doubled for insert/delete
		self.a = nn.Parameter(torch.zeros(3 * self.message_dim, device=self.device))  # 3x for [insert, delete, center]
		self.dropout = nn.Dropout(p=0.5)
		nn.init.xavier_normal_(self.W_g)

		self.W_e = nn.Parameter(torch.zeros((self.edge_dim, 4 * self.message_dim), device=self.device))  # 4x for new message
		nn.init.xavier_normal_(self.W_e)

		self.W_uc = nn.Parameter(torch.zeros((self.emb_dim + 8 * self.message_dim, self.emb_dim), device=self.device))  # 8x for new message
		self.W_un = nn.Parameter(torch.zeros((2 * self.emb_dim + 1, self.emb_dim), device=self.device))
		nn.init.xavier_normal_(self.W_uc)
		nn.init.xavier_normal_(self.W_un)

		self.W_p = nn.Parameter(torch.zeros((8 * self.message_dim, self.emb_dim), device=self.device))  # 8x for new message
		nn.init.xavier_normal_(self.W_p)

		self.W_1 = nn.Parameter(torch.zeros((self.emb_dim * 2 + 1, self.emb_dim), device=self.device))  # +1 for flag
		self.W_2 = nn.Parameter(torch.zeros((self.emb_dim, 1), device=self.device))
		nn.init.xavier_normal_(self.W_1)
		nn.init.xavier_normal_(self.W_2)
		self.beta = 0.1

	def forward(self, src_idxs, dst_idxs, neg_src_idxs, neg_dst_idxs, edge_idxs, timestamps, flags=None):
		"""
		flags: +1 for insertion, -1 for deletion, shape [batch]
		neg_src_idxs, neg_dst_idxs: for each event, the negative sample (src, neg) or (neg, dst) for link prediction
		"""
		# Time-aware attention and message passing for insertions and deletions
		message = []
		for i, idxs in enumerate([src_idxs, dst_idxs]):
			# The neighbor_finder returns both insertion and deletion times for each neighbor
			neighbors, _, ins_times, del_times = self.neighbor_finder.get_temporal_neighbor(
				idxs, timestamps, self.n_neighbors
			)

			current_time = timestamps.unsqueeze(1).repeat(1, self.n_neighbors)
			delta_ins = (current_time - ins_times) / self.tot_time
			delta_del = (current_time - del_times) / self.tot_time
			phi_ins = 1 / (1 + delta_ins)
			phi_del = 1 / (1 + delta_del)
			phi_ins = torch.where(torch.isinf(delta_ins), torch.zeros_like(phi_ins), phi_ins)
			phi_del = torch.where(torch.isinf(delta_del), torch.zeros_like(phi_del), phi_del)

			neighbors = torch.from_numpy(neighbors).long().to(self.device)
			bs = neighbors.shape[0]
			neighbor_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_neighbors, self.emb_dim)

			# Prepare insert and delete messages
			phi_ins = phi_ins.unsqueeze(2).repeat(1, 1, self.emb_dim)
			phi_del = phi_del.unsqueeze(2).repeat(1, 1, self.emb_dim)

			ins_msg = phi_ins * neighbor_emb
			del_msg = phi_del * (-neighbor_emb)  # Negate for deletion

			# Concatenate for attention: [center | insert | delete]
			h_c = torch.matmul(self.memory.emb[idxs], self.W_g).unsqueeze(dim=1).repeat(1, self.n_neighbors, 1)
			h_ins = torch.matmul(ins_msg, self.W_g)
			h_del = torch.matmul(del_msg, self.W_g)
			h_in = torch.cat((h_c, h_ins, h_del), dim=2)
			h_in = self.dropout(h_in)
			
			att = F.leaky_relu(torch.matmul(h_in, self.a), negative_slope=0.2)
			att = att.softmax(dim=1).unsqueeze(dim=2).repeat(1, 1, 2 * self.message_dim)
			h = torch.cat((h_ins, h_del), dim=2) * att  # Combine insert and delete messages
			message.append(h)

		h = torch.cat((message[0], message[1]), dim=2).sum(dim=1).relu()
		h_e = torch.matmul(self.edge_feat[edge_idxs], self.W_e).relu()
		h = torch.cat((h, h_e), dim=1)  # h now in R^{8d_m}

		# Update memory with new message size
		to_updated_src = torch.matmul(torch.cat((self.memory.emb[src_idxs], h), dim=1), self.W_uc).tanh()
		to_updated_dst = torch.matmul(torch.cat((self.memory.emb[dst_idxs], h), dim=1), self.W_uc).tanh()
		self.memory.emb[src_idxs] = to_updated_src
		self.memory.emb[dst_idxs] = to_updated_dst

		reward = 0
		policy_loss = 0
		
		# RNSM with flag concatenation
		for idxs in [src_idxs, dst_idxs]:
			neighbors, _, ins_times, del_times = self.neighbor_finder.get_temporal_neighbor(
				idxs, timestamps, self.n_update_neighbors
			)
			current_time = timestamps.unsqueeze(1).repeat(1, self.n_update_neighbors)
			delta_ins = (current_time - ins_times) / self.tot_time
			delta_del = (current_time - del_times) / self.tot_time
			ins_times = torch.tensor(ins_times, dtype=torch.float32, device=self.device)
			del_times = torch.tensor(del_times, dtype=torch.float32, device=self.device)

			phi_ins = 1 / (1 + delta_ins)
			phi_del = 1 / (1 + delta_del)
			phi_ins = torch.where(torch.isinf(delta_ins), torch.zeros_like(phi_ins), phi_ins)
			phi_del = torch.where(torch.isinf(delta_del), torch.zeros_like(phi_del), phi_del)

			neighbors = torch.from_numpy(neighbors).long().to(self.device)
			bs = neighbors.shape[0]
			neighbor_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_update_neighbors, self.emb_dim)
			phi_ins = phi_ins.unsqueeze(2).repeat(1, 1, self.emb_dim)
			phi_del = phi_del.unsqueeze(2).repeat(1, 1, self.emb_dim)
			ins_msg = phi_ins * neighbor_emb
			del_msg = phi_del * (-neighbor_emb)

			# Concatenate for message

			h1 = torch.matmul(h, self.W_p)
			h1 = h1.unsqueeze(dim=1).repeat(1, self.n_update_neighbors, 1)
			score_ins = (h1 * ins_msg).sum(dim=2)  # [bs, n_update_neighbors]
			score_del = (h1 * del_msg).sum(dim=2)
			score = score_ins + score_del + self.eps
			att = torch.softmax(score, dim=1)  # [bs, n_update_neighbors]
			changed_emb = h1 * att.unsqueeze(dim=2).repeat(1, 1, self.emb_dim)
			current_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_update_neighbors, self.emb_dim)

			# Add flag for RNSM state
			if flags is not None:
				flag = flags.unsqueeze(1).repeat(1, self.n_update_neighbors).unsqueeze(2)
				flag = flag.to(self.device)
				x = torch.cat((changed_emb, current_emb, flag), dim=2)  # [bs, n_update_neighbors, 2*emb_dim+1]
			else:
				x = torch.cat((changed_emb, current_emb), np.ones_like(current_emb[:, :, 0:1]), dim=2)  # [bs, n_update_neighbors, 2*emb_dim+1]

			x.detach_()
			x = torch.matmul(x, self.W_1)
			x = x.relu()
			x = torch.matmul(x, self.W_2)
			probs = x.sigmoid()

			changed_emb = torch.matmul(
				torch.cat((
					self.memory.emb[neighbors.flatten()],                    # [B, emb_dim]
					changed_emb.flatten().view(-1, self.emb_dim),            # [B, emb_dim]
					flags.unsqueeze(1).repeat(1, self.n_update_neighbors).view(-1, 1)  # shape: [B, 1]
				), dim=1),
				self.W_un
			).relu()
			
			policy_map = probs.detach().clone()
			policy_map[policy_map < 0.5] = 0.0
			policy_map[policy_map >= 0.5] = 1.0
			policy_map = policy_map
			distr = Bernoulli(probs)
			policy = distr.sample()
			if not self.training:
				mask = policy_map.repeat(1, 1, self.emb_dim).flatten().view(-1, self.emb_dim)
				with torch.no_grad():
					self.memory.emb[neighbors.flatten()] = mask * changed_emb + (1 - mask) * self.memory.emb[
						neighbors.flatten()]
			else:
				policy_map = policy_map.repeat(1, 1, self.emb_dim).flatten().view(-1, self.emb_dim)
				policy_sample = policy.repeat(1, 1, self.emb_dim).flatten().view(-1, self.emb_dim)
				ori_emb = self.memory.emb[neighbors.flatten()]
				updated_emb_map = policy_map * changed_emb + (1 - policy_map) * ori_emb
				updated_emb_sample = policy_sample * changed_emb + (1 - policy_sample) * ori_emb
				reward_map = self.get_reward(idxs, updated_emb_map).detach()
				reward_sample = self.get_reward(idxs, updated_emb_sample).detach()
				advantage = reward_sample - reward_map
				loss = -distr.log_prob(policy) * advantage.expand_as(policy)
				loss = loss.sum()
				probs = torch.clamp(probs, 1e-15, 1 - 1e-15)
				entropy_loss = -probs * torch.log(probs)
				entropy_loss = self.beta * entropy_loss.sum()
				loss = (loss - entropy_loss) / bs / self.n_update_neighbors
				self.memory.emb[neighbors.flatten()] = updated_emb_sample
				policy_loss += loss
				reward += reward_sample

		# compute loss
		pos_score, neg_score = self.compute_score(src_idxs, dst_idxs, neg_src_idxs, neg_dst_idxs)
		pos_prob = torch.sigmoid(pos_score)
		neg_prob = torch.sigmoid(neg_score)
		return pos_prob, neg_prob, policy_loss, reward

	def get_reward(self, idxs, neighbor_emb):
		central_emb = self.memory.emb[idxs].repeat(1, self.n_update_neighbors).view(-1, self.emb_dim)
		central_emb_norm = F.normalize(central_emb, p=2, dim=1).detach()
		neighbor_emb_norm = F.normalize(neighbor_emb, p=2, dim=1)
		cos_sim = torch.matmul(central_emb_norm, neighbor_emb_norm.t())
		return cos_sim.mean()

	def compute_sim(self, src_idxs):
		src_norm = F.normalize(self.memory.emb[src_idxs], p=2, dim=1)
		emb_norm = F.normalize(self.memory.emb, p=2, dim=1)
		cos_sim = torch.matmul(src_norm, emb_norm.t())
		sorted_cos_sim, idx = cos_sim.sort(descending=True)
		return sorted_cos_sim, idx

	def compute_score(self, src_idxs, dst_idxs, neg_src_idxs, neg_dst_idxs):
		pos_score = (self.memory.emb[src_idxs] * self.memory.emb[dst_idxs]).sum(dim=1)
		neg_score = (self.memory.emb[neg_src_idxs] * self.memory.emb[neg_dst_idxs]).sum(dim=1)
		return pos_score, neg_score

	def reset_graph(self):
		self.memory.__init_memory__()

	def set_neighbor_finder(self, neighbor_finder):
		self.neighbor_finder = neighbor_finder

	def detach_memory(self):
		self.memory.detach_memory()

	def back_up_memory(self):
		return self.memory.emb.clone()

	def restore_memory(self, back_up):
		self.memory.emb = nn.Parameter(back_up)