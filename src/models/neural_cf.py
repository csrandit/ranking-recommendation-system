import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)

        x = torch.cat([user_vec, item_vec], dim=1)
        return self.mlp(x).squeeze()


class NeuralCFRecommender:
    def __init__(self, epochs=5, lr=0.001, embedding_dim=32):
        self.epochs = epochs
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}

    def fit(self, train_df: pd.DataFrame):

        users = train_df["user_id"].unique()
        items = train_df["item_id"].unique()

        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}

        num_users = len(users)
        num_items = len(items)

        self.model = NeuralCF(num_users, num_items, self.embedding_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        user_tensor = torch.tensor(
            train_df["user_id"].map(self.user_map).values
        )
        item_tensor = torch.tensor(
            train_df["item_id"].map(self.item_map).values
        )
        rating_tensor = torch.tensor(
            train_df["rating"].values, dtype=torch.float32
        )

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()

            predictions = self.model(user_tensor, item_tensor)
            loss = loss_fn(predictions, rating_tensor)

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss.item():.4f}")

    def predict(self, user_id, item_id):
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0

        self.model.eval()

        user_tensor = torch.tensor([self.user_map[user_id]])
        item_tensor = torch.tensor([self.item_map[item_id]])

        with torch.no_grad():
            return self.model(user_tensor, item_tensor).item()

    def recommend(self, user_id, k=10):

        if user_id not in self.user_map:
            return []

        self.model.eval()

        user_idx = self.user_map[user_id]
        user_tensor = torch.tensor([user_idx] * len(self.item_map))

        item_indices = list(self.item_map.values())
        item_tensor = torch.tensor(item_indices)

        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor)

        top_indices = torch.topk(scores, k).indices.numpy()

        return [
            self.reverse_item_map[item_indices[i]]
            for i in top_indices
        ]