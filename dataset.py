from typing import Tuple
import torch.utils.data
import torch


class FooDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int) -> None:
        super().__init__()
        self.samples = samples
        self.X = torch.randn(self.samples, 10)
        self.Y = torch.randn(self.samples, 5)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.samples
