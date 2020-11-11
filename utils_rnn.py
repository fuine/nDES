import argparse
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence


class DummyDataset(Dataset):
    def __init__(self, dataset, sizes, gt):
        self.dataset = dataset
        self.sizes = sizes
        self.gt = gt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.gt[idx]


class DummyDataGenerator:
    def __init__(self, dataset, sizes, gts, device):
        self.pack = pack_sequence(dataset, enforce_sorted=False)
        self.pack = self.pack.to(device)
        self.gts = gts.to(device)

    def __iter__(self):
        return self

    def __next__(self):
        return 0, (self.pack, self.gts)

    def __len__(self):
        return 1


def pad_collate(batch):
    (xx, y) = zip(*batch)
    dev = y[0].device
    pack = pack_sequence(xx, enforce_sorted=False)
    y = torch.tensor(y).to(dev)
    return pack, y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cuda", type=int, default=0, choices=list(range(torch.cuda.device_count()))
    )
    parser.add_argument(dest="sequence_length", type=int)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(device)
    return args.sequence_length, device
