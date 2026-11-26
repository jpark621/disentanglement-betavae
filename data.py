import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DSpritesDataset(Dataset):
    def __init__(self, path="dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"):
        self.dataset = np.load(path)
        self.imgs = self.dataset['imgs']
        self.factors = self.dataset['latents_classes']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return torch.tensor(self.imgs[idx], dtype=torch.float32), torch.tensor(self.factors[idx, 1:])

if __name__ == "__main__":
    dataset = DSpritesDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for X, _ in dataloader:
        print(X.shape)
        break
