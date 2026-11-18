import torch
from torch import nn

#只包含一种 (X0 Xu Xs X)
class Partitions(nn.Module):
    # This is nn.Module to allow ".to(device)"
    def __init__(self, bounds):
        super().__init__()

        lower, upper = bounds

        self.register_buffer('lower', lower)
        self.register_buffer('upper', upper)

    # 所有cell的体积和
    @property
    def volume(self):
        return self.volumes.sum()

    # 每个cell的体积
    @property
    def volumes(self):
        return self.width.prod(dim=-1)

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def diff(self):
        return (self.upper - self.lower) / 2

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    def sample(self, num_samples):
        perm = torch.randperm(len(self))
        idx = perm[:num_samples]

        return Partitions((self.lower[idx], self.upper[idx]))

    def __getitem__(self, idx):
        return Partitions(
            (self.lower[idx], self.upper[idx])
        )

    def __len__(self):
        return self.lower.size(0)

# 分区类 包含四种
class Partitioning(nn.Module):
    def __init__(self, initial, safe, unsafe, state_space=None):
        super().__init__()
        # 以initial举例，initial包括 initial的lower和upper
        # 每个都是一个Partitions
        self.initial = self.convert(initial)
        self.safe = self.convert(safe)
        self.unsafe = self.convert(unsafe)
        self.state_space = self.convert(state_space)

    @staticmethod
    def convert(partitions):
        if partitions is None:
            return None
        elif isinstance(partitions, Partitions):
            return partitions
        else:
            return Partitions(partitions)

    def __len__(self):
        return len(self.initial) + \
               len(self.safe) + \
               len(self.unsafe) + \
               (len(self.state_space) if self.state_space else 0)

    def __getitem__(self, idx):
        initial_idx, safe_idx, unsafe_idx, state_space_idx = idx

        return Partitioning(
            self.initial[initial_idx],
            self.safe[safe_idx],
            self.unsafe[unsafe_idx],
            self.state_space[state_space_idx] if self.state_space else None,
        )