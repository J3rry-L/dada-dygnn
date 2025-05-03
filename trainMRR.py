import torch
import numpy as np
import math
import os
import time
from tqdm import tqdm
import pickle
import argparse
from pathlib import Path
from models.DadaDyGNN import DadaDyGNN
from models.MLPLinkPredictor import MLPLinkPredictor
from utils.data_processing import get_data, computer_time_statics
from utils.utils import get_neighbor_finder, EarlyStopMonitor
from utils.evaluation import eval_edge_prediction
from utils.log_and_checkpoints import set_logger, get_checkpoint_path
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ModelName = 'DadaDyGNN'
parser = argparse.ArgumentParser('DadaDyGNN')
parser.add_argument('-d', '--data', type=str, default='pref-att')
parser.add_argument('--bs', type=int, default=16, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=200, help='Number of neighbors to sample')
parser.add_argument('--n_update_degree', type=int, default=200, help='Number of neighbors to sample')
parser.add_argument('--n_epoch', type=int, default=30, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--alpha', type=float, default=1, help='Loss balance')
parser.add_argument('--mlp_epochs', type=int, default=1000, help='Epochs for MLP classifier')
parser.add_argument('--mlp_lr', type=float, default=0.001, help='Learning rate for MLP classifier')
log_to_file = True
args = parser.parse_args()
dataset = args.data
Epoch = args.n_epoch
Batchsize = args.bs
n_neighbors = args.n_degree
n_update_neighbors = args.n_update_degree
lr = args.lr
alpha = args.alpha
mlp_epochs = args.mlp_epochs
mlp_lr = args.mlp_lr
logger, time_now = set_logger(ModelName, dataset, "", log_to_file)
curr_dir = Path.cwd()
Path("log/{}/{}/checkpoints".format(ModelName, time_now)).mkdir(parents=True, exist_ok=True)
Path("{}/MRR-result/".format(curr_dir)).mkdir(parents=True, exist_ok=True)
f = open("{}/MRR-result/{}.txt".format(curr_dir, dataset), "a+")
f.write("bs = {}, degree = {}, up_degree = {}, lr = {}".format(Batchsize, n_neighbors, n_update_neighbors, lr))
f.write("\n")

# data processing
node_features, edge_features, full_data, train_data, \
val_data, test_data, new_node_val_data, new_node_test_data, tot_time = get_data(dataset)
# initialize temporal graph
train_neighbor_finder = get_neighbor_finder(train_data, False)
full_neighbor_finder = get_neighbor_finder(full_data, False)
new_val_neighbor_finder = get_neighbor_finder(new_node_val_data, False)
new_test_neighbor_finder = get_neighbor_finder(new_node_test_data, False)


# Build edge history for deletion-aware negative sampling
def build_edge_history(src, dst, timestamps):
    """
    Returns a dict: time -> set of (u, v) edges active at that time.
    Assumes undirected edges (u, v) with u < v.
    """
    edge_history = dict()
    active_edges = set()
    time_order = np.argsort(timestamps)
    src, dst, timestamps = src[time_order], dst[time_order], timestamps[time_order]
    for i in range(len(src)):
        t = timestamps[i]
        u, v = int(src[i]), int(dst[i])
        edge = (min(u, v), max(u, v))
        # Insert edge
        active_edges.add(edge)
        edge_history[t] = set(active_edges)
    return edge_history


# Check if flags are available in the dataset (indicating insertion/deletion events)
flags_available = hasattr(train_data, 'flags')
flags_array = train_data.flags if flags_available else np.ones_like(train_data.src)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DadaDyGNN(node_features.shape[0], n_neighbors=n_neighbors, n_update_neighbors=n_update_neighbors,
				 edge_dim=edge_features.shape[1], emb_dim=64, message_dim=16, neighbor_finder=train_neighbor_finder,
				 edge_feat=edge_features, tot_time=tot_time, device=device)
mlp_emb_dim = model.emb_dim
mlp = MLPLinkPredictor(mlp_emb_dim).to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
LOSS = []
Policy_LOSS = []

train_metrics = {
	"agg_mrr": [],
	"agg_recall_20": [],
	"agg_recall_50": [],
	"insertion_mrr": [],
	"insertion_recall_20": [],
	"insertion_recall_50": [],
	"deletion_mrr": [],
	"deletion_recall_20": [],
	"deletion_recall_50": [],
}

train_n_metrics = {
	"agg_mrr": [],
	"agg_recall_20": [],
	"agg_recall_50": [],
	"insertion_mrr": [],
	"insertion_recall_20": [],
	"insertion_recall_50": [],
	"deletion_mrr": [],
	"deletion_recall_20": [],
	"deletion_recall_50": [],
}

start = time.time()

logger.info("Training DadaDyGNN model...")
check_path = []
early_stopper = EarlyStopMonitor(max_round=100)
for e in tqdm(range(Epoch)):
	logger.debug('Start {} epoch'.format(e))
	num_batch = math.ceil(len(train_data.src) / Batchsize)
	Loss = 0
	Policy_Loss = 0
	Reward = 0
	cnt = 0
	sum_size1 = 0
	sum_size2 = 0
	model.reset_graph()
	model.set_neighbor_finder(train_neighbor_finder)
	model.train()
	for i in range(num_batch):
		st_idx = i * Batchsize
		ed_idx = min((i + 1) * Batchsize, len(train_data.src))
		src_batch = train_data.src[st_idx:ed_idx]
		dst_batch = train_data.dst[st_idx:ed_idx]
		edge_batch = train_data.edge_idxs[st_idx:ed_idx]
		timestamp_batch = train_data.timestamps[st_idx:ed_idx]
		flags_batch = flags_array[st_idx:ed_idx] if flags_available else np.ones(ed_idx - st_idx, dtype=int)
		size = len(src_batch)
		loss = 0
		optimizer.zero_grad()
		
		# Per-event negative sampling
		neg_dst = train_neighbor_finder.negative_sample(src_batch, dst_batch, timestamp_batch, flags_batch)

		# Convert to tensors
		neg_src = torch.LongTensor(src_batch).to(device)
		neg_dst = torch.LongTensor(neg_dst).to(device)
		src_batch = torch.LongTensor(src_batch).to(device)
		dst_batch = torch.LongTensor(dst_batch).to(device)
		edge_batch = torch.LongTensor(edge_batch).to(device)
		timestamp_batch = torch.FloatTensor(timestamp_batch).to(device)
		flags_batch = torch.LongTensor(flags_batch).to(device)
		
		with torch.no_grad():
			pos_label = (flags_batch == 1).float()
			neg_label = (flags_batch == -1).float()
		try:
			pos_prob, neg_prob, reinforce_loss, reward = model(src_batch, dst_batch, neg_src, neg_dst, edge_batch, timestamp_batch, flags=flags_batch)
		except Exception as e:
			logger.error(f"Model call failed: {e}")
			raise
		loss += criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
		loss /= size
		Loss += loss.item()
		Policy_Loss += alpha * (reinforce_loss.item() if hasattr(reinforce_loss, 'item') else reinforce_loss)
		Reward += reward if isinstance(reward, float) else reward.item()
		loss += reinforce_loss
		loss.backward()
		optimizer.step()
		model.detach_memory()
	LOSS.append(Loss)
	Policy_LOSS.append(Policy_Loss)
	logger.debug("Loss in whole dataset = {}".format(Loss))
	logger.debug("Policy Loss = {}".format(Policy_Loss))


	# Validation
	train_memory_backup = model.back_up_memory()
	model.eval()
	model.set_neighbor_finder(full_neighbor_finder)
	val_metrics = eval_edge_prediction(model, full_neighbor_finder, val_data, Batchsize, mlp=mlp, flags=val_data.flags)
	val_mrr = val_metrics["agg_mrr"]
	for k in val_metrics:
		logger.debug("In validation, {} = {}".format(k, val_metrics[k]))
		train_metrics[k].append(val_metrics[k])
	val_memory_backup = model.back_up_memory()

	model.restore_memory(train_memory_backup)
	val_n_metrics = eval_edge_prediction(model, new_val_neighbor_finder, new_node_val_data, Batchsize, mlp=mlp, flags=new_node_val_data.flags)
	for k in val_n_metrics:
		logger.debug("In new node validation, {} = {}".format(k, val_n_metrics[k]))
		train_n_metrics[k].append(val_n_metrics[k])

	model.restore_memory(val_memory_backup)
	path = get_checkpoint_path(ModelName, time_now, e)
	check_path.append(path)
	if early_stopper.early_stop_check(val_mrr):
		logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
		logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
		best_model_path = get_checkpoint_path(ModelName, time_now, early_stopper.best_epoch)
		model.load_state_dict(torch.load(best_model_path))
		logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
		model.eval()
		break
	else:
		# Save state_dict instead of full model
		torch.save(model.state_dict(), path)

end = time.time()
train_time = end - start
logger.info(f"Training complete in {train_time:.2f} seconds.")
print(f"Training complete in {train_time:.2f} seconds.")

# Train the MLPLinkPredictor on top of the frozen GNN
from torch.utils.data import DataLoader, TensorDataset

logger.info("Starting MLPLinkPredictor training...")

# Freeze GNN parameters
for param in model.parameters():
	param.requires_grad = False

# Prepare positive and negative pairs for MLP training
mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=mlp_lr)
mlp_criterion = torch.nn.BCEWithLogitsLoss()

# Gather all positive and negative samples from train_data
src_nodes = torch.LongTensor(train_data.src).to(device)
dst_nodes = torch.LongTensor(train_data.dst).to(device)
flags = torch.LongTensor(flags_array).to(device)
edge_idxs = torch.LongTensor(train_data.edge_idxs).to(device)
timestamps = torch.FloatTensor(train_data.timestamps).to(device)

neg_nodes = train_neighbor_finder.negative_sample(src_nodes, dst_nodes, timestamps, flags)
neg_nodes = torch.LongTensor(neg_nodes).to(device)

# Get node embeddings from the frozen GNN memory
with torch.no_grad():
	emb_src = model.memory.emb[src_nodes]
	emb_dst = model.memory.emb[dst_nodes]
	emb_neg = model.memory.emb[neg_nodes]

# Prepare labels: 1 for insertion, 0 for deletion
mlp_labels = (flags == 1).float()

mlp_dataset = TensorDataset(emb_src, emb_dst, emb_neg, mlp_labels)
mlp_loader = DataLoader(mlp_dataset, batch_size=Batchsize, shuffle=True)

for epoch in range(mlp_epochs):
	mlp.train()
	mlp_loss_epoch = 0
	for emb_i, emb_j, emb_k, label in mlp_loader:
		mlp_optimizer.zero_grad()
		mlp_pos = mlp(emb_i, emb_j)
		mlp_neg = mlp(emb_i, emb_k)
		# BCEWithLogitsLoss expects targets in {0,1}
		loss = mlp_criterion(mlp_pos, label) + mlp_criterion(mlp_neg, 1-label)
		loss.backward()
		mlp_optimizer.step()
		mlp_loss_epoch += loss.item()
	logger.info(f"MLP Epoch {epoch+1}/{mlp_epochs}, Loss: {mlp_loss_epoch/len(mlp_loader):.4f}")

logger.info("MLPLinkPredictor training complete.")


# Evaluate the model on the test set
memory_backup = model.back_up_memory()
model.eval()
model.set_neighbor_finder(full_neighbor_finder)
test_metrics = eval_edge_prediction(model, full_neighbor_finder, test_data, Batchsize, mlp=mlp, flags=test_data.flags)
test_mrr = test_metrics["agg_mrr"]

for k in test_metrics:
	logger.info("In test, {} = {}".format(k, test_metrics[k]))
	print("In test, {} = {}".format(k, test_metrics[k]))

model.restore_memory(memory_backup)

test_n_metrics = eval_edge_prediction(model, new_test_neighbor_finder, new_node_test_data, Batchsize, mlp=mlp, flags=new_node_test_data.flags)
test_n_mrr = test_n_metrics["agg_mrr"]
for k in test_n_metrics:
	logger.info("In new node test, {} = {}".format(k, test_n_metrics[k]))
	print("In new node test, {} = {}".format(k, test_n_metrics[k]))

f.write("test_mrr = {:.4f}, new node test mrr = {:.4f} ".format(test_mrr, test_n_mrr))

# Plot policy loss
plt.figure(figsize=(10, 5))
plt.plot(LOSS, label='Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss versus Epochs')
plt.legend()
plt.savefig(f"log/{ModelName}/{time_now}/loss_plot.png")
plt.close()

# Plot training metrics
plt.figure(figsize=(10, 5))
plt.plot(train_metrics["agg_mrr"], label='Aggregate MRR', color='blue')
plt.plot(train_metrics["insertion_mrr"], label='Insertion MRR', color='orange')
plt.plot(train_metrics["deletion_mrr"], label='Deletion MRR', color='green')
plt.xlabel('Epoch')
plt.ylabel('MRR')
plt.title('Training Metrics - MRR')
plt.legend()
plt.savefig(f"log/{ModelName}/{time_now}/train_mrr_plot.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(train_n_metrics["agg_mrr"], label='Aggregate MRR', color='blue')
plt.plot(train_n_metrics["insertion_mrr"], label='Insertion MRR', color='orange')
plt.plot(train_n_metrics["deletion_mrr"], label='Deletion MRR', color='green')
plt.xlabel('Epoch')
plt.ylabel('MRR')
plt.title('Training Metrics, New Nodes - MRR')
plt.legend()
plt.savefig(f"log/{ModelName}/{time_now}/train_n_mrr_plot.png")
plt.close()