import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLinkPredictor(nn.Module):
    """
    MLP link prediction head
    Takes entrywise product of node embeddings, passes through a 2-layer MLP, and outputs raw value
    """

    def __init__(self, emb_dim):
        super().__init__()
        # W_2^MLP = I, W_1^MLP = 1^T (init)
        self.W2 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W1 = nn.Linear(emb_dim, 1, bias=True)
        # W2 = identity, W1 = all-ones
        nn.init.eye_(self.W2.weight)
        nn.init.constant_(self.W1.weight, 1.0)
        nn.init.zeros_(self.W1.bias)

    def forward(self, emb_i, emb_j):
        """
        emb_i: [batch, emb_dim]
        emb_j: [batch, emb_dim]
        Returns: [batch] values in [-1, 1]
        """
        x = emb_i * emb_j  # entrywise product
        x = F.relu(self.W2(x))
        x = self.W1(x).squeeze(-1)
        return x

    def predict_prob(self, emb_i, emb_j):
        """
        Returns probability-like scores in [0, 1] for binary classification.
        """
        with torch.no_grad():
            prob = torch.tanh(self.forward(emb_i, emb_j))
        return prob

    def loss(self, emb_i, emb_j, emb_k, flag):
        """
        emb_i, emb_j: positive pair
        emb_k: negative node (for negative sample)
        flag: +1 for insertion, -1 for deletion (shape [batch])
        Returns: scalar loss
        """
        mlp_pos = self.forward(emb_i, emb_j)
        mlp_neg = self.forward(emb_i, emb_k)
        # Use sigmoid for cross-entropy
        # For insertions: positive label=1, negative label=0
        # For deletions: positive label=0, negative label=1
        pos_label = (flag == 1).to(dtype=torch.float)
        neg_label = (flag == -1).to(dtype=torch.float)
        criterion = nn.BCELoss()
        # Apply sigmoid to get probabilities
        x_pos = mlp_pos.sigmoid()
        x_neg = mlp_neg.sigmoid()
        loss = criterion(x_pos, pos_label) + criterion(x_neg, neg_label)
        return loss