import torch
import torch.nn.functional as F

from src.models import TransformerModel
from src.tasks import HermiteRegression

# small toy model / sizes to keep it quick
# HI! HI
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


def data_gen(f_class, number_of_points):
    if f_class == "linear":
        print("hi")
    
    elif f_class == "quadratic":
        print("quadratic hi")

def hyperparameter_train():
    #train a model with mixed, sweep over a few hyperparameter types
    #1. define the model
    n_dims   = 20
    n_layer  = 12
    n_head   = 8
    n_embd   = 128

    T = 10                
    batch_size = 64       
    n_samples = int(200000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TransformerModel(
        n_dims=n_dims,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        seed=0,
        n_positions=T,
    ).to(device)
    # Define the optimizer ()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train 
    task = HermiteRegression(
        n_dims=n_dims,
        batch_size=batch_size,
        degree=2,          # 1, 2, 3, 4 ...
        scale=1.0,
        weight_type="standard",
    )
    steps = n_samples // (batch_size * T)

    for step in range(steps):
        # 1) sample inputs: xs_b ~ N(0, I)
        xs_b = torch.randn(batch_size, T, n_dims, device=device)

        # 2) generate targets from the Hermite task
        ys_b = task.evaluate(xs_b).to(device)     # shape [batch_size, T]

        # 3) forward pass through transformer
        _, _, preds = model(xs_b, ys_b, prompt_type="standard", prompt_row=0, prompt_col=0)
                     # e.g. [batch_size, T] or [batch_size, T, 1]
        if preds.dim() == 3 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)             # shape [batch_size, T]

        # 4) loss and update
        loss = F.mse_loss(preds, ys_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step}/{steps}, loss {loss.item():.4f}")

if __name__ == "__main__":
    hyperparameter_train()

