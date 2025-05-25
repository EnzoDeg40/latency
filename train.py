import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np 

# Chargement
with open("dataset.json", "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} messages.")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def encode_time(t):
    return [math.sin(2 * math.pi * t / 1440), math.cos(2 * math.pi * t / 1440)]

class LatencyDataset(Dataset):
    def __init__(self, data):
        self.X = []
        self.y = []
        for d in data:
            text_vec = embedder.encode(d["message"])
            time_vec = encode_time(d["time"])
            full_vec = np.concatenate([text_vec, time_vec])
            self.X.append(torch.tensor(full_vec, dtype=torch.float32))
            self.y.append(torch.tensor([d["latency"]], dtype=torch.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = LatencyDataset(data)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Modèle
class LatencyModel(nn.Module):
    def __init__(self, input_dim=386):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

model = LatencyModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Entraînement
for epoch in range(200):
    for x_batch, y_batch in loader:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "latency_model_torch.pt")
