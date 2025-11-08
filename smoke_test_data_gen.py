import torch

from src.tasks import HermiteRegression

def smoke_test_hermite():
    n_dims = 1        # 2D (x,y)
    batch_size = 100    # 3 tasks
    T = 10            # 10 points per task (shots)

    task = HermiteRegression(
        n_dims=n_dims,
        batch_size=batch_size,
        degree=2,
        scale=1,
        weight_type='standard',
    )

    # xs_b ~ N(0, I)
    xs_b = torch.randn(batch_size, T, n_dims)

    ys_b = task.evaluate(xs_b)

    print("xs_b shape:", xs_b.shape)   # expect [3, 10, 1]
    print("ys_b shape:", ys_b.shape)   # expect [3, 10]
    print("w_b shape:", task.w_b.shape) # expect [3, 1, 1]
    print("xs_b[0, :, 0]:", xs_b[0, :, 0])
    print("ys_b[0]:", ys_b[0])

if __name__ == "__main__":
    smoke_test_hermite()
