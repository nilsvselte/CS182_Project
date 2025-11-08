import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


#Generate function
def hermite_linear_data(n_tasks=16, n_points=30, n_dims=1, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(n_tasks, n_dims, 1)
    x = torch.randn(n_tasks, n_points, n_dims) 
    t = (x @ w)[:, :, 0]
    y = t.clone()                       
    return x, y, w


#GPT-style transformer
class TransformerICL(nn.Module):
    def __init__(self, n_dims=1, n_positions=64, n_embd=64, n_layer=2, n_head=2):
        super().__init__()
        cfg = GPT2Config(
            n_positions=n_positions * 2,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.read_in  = nn.Linear(n_dims, n_embd)
        self.backbone = GPT2Model(cfg)
        self.read_out = nn.Linear(n_embd, 1)
        self.n_dims = n_dims

    @staticmethod
    def combine(xs, ys):
        bsize, points, dim = xs.shape
        ys_wide = torch.cat([ys.unsqueeze(2), torch.zeros(bsize, points, dim-1)], dim=2)
        z = torch.stack((xs, ys_wide), dim=2).view(bsize, 2*points, dim)
        return z

    def forward(self, xs, ys):
        z = self.combine(xs, ys)
        emb = self.read_in(z)
        out = self.backbone(inputs_embeds=emb).last_hidden_state
        pred = self.read_out(out)
        return pred[:, ::2, 0]


#Train on linear
device = "cuda" if torch.cuda.is_available() else "cpu"

n_tasks, n_points = 32, 30
xs, ys, _ = hermite_linear_data(n_tasks=n_tasks, n_points=n_points)
xs, ys = xs.to(device), ys.to(device)

model = TransformerICL(n_dims=1, n_positions=n_points,
                       n_embd=64, n_layer=2, n_head=2).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

loss_trace = []
for step in range(2000):
    opt.zero_grad()
    preds = model(xs, ys)
    loss = loss_fn(preds, ys)
    loss.backward()
    opt.step()
    loss_trace.append(loss.item())
    if step % 200 == 0:
        print(f"Step {step:4d} | loss = {loss.item():.6f}")

#plot MSE
plt.plot(loss_trace)
plt.xlabel("Step")
plt.ylabel("MSE Loss")
plt.title("Tiny Transformer on Linear Hermite Tasks (lr=1e-4)")
plt.show()
