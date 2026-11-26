from train import train

if __name__ == "__main__":
    betas = [1, 3, 5, 7, 10]
    seeds = [1, 42, 12345]
    max_steps = 300000
    for beta in betas:
        for seed in seeds:
            train(beta=beta, seed=seed, max_steps=max_steps, save_path=f"runs/sweep/beta{beta}_seed{seed}")

