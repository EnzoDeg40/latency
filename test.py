import math
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def encode_time(t):
    return [math.sin(2 * math.pi * t / 1440), math.cos(2 * math.pi * t / 1440)]

class LatencyModel(torch.nn.Module):
    def __init__(self, input_dim=386):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = LatencyModel()
model.load_state_dict(torch.load("latency_model_torch.pt"))
model.eval()

# Test
message = "Toi aussi tu vas à la brocante ?"
time = 234
text_vec = embedder.encode(message)
time_vec = encode_time(time)
full_vec = np.concatenate([text_vec, time_vec])
x = torch.tensor(full_vec, dtype=torch.float32)
with torch.no_grad():
    pred = model(x)
    print(f"Latence prédite : {round(pred.item())} minutes")
