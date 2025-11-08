from src.models import TransformerModel
import torch

# small toy model / sizes to keep it quick
m = TransformerModel(n_dims=5, n_positions=12, n_embd=32, n_layer=2, n_head=4, seed=0)

# small synthetic batch: bsize=2, points=5, dims=5
bsize = 2
n_points = 5
n_dims = 5

xs = torch.randn(bsize, n_points, n_dims)    # shape [bsize, points, dims]
ys = torch.randn(bsize, n_points)            # shape [bsize, points]

# run forward (prompt_type='standard' and prompt indices 0)
embeds, att, pred = m(xs, ys, prompt_type='standard', prompt_row=0, prompt_col=0)
print("pred shape:", pred.shape)  # expected [bsize, n_points]
print("pred sample:", pred[0, :3])