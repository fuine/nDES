from timeit import default_timer as timer

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from ndes_optimizer import RNNnDESOptimizer
from utils_rnn import DummyDataGenerator, DummyDataset, pad_collate, parse_args

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)


class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(1, 4, batch_first=True)
        self.output = nn.Linear(4, 1)

    def forward(self, x, hidden=None):
        out, _ = self.rnn(x, hidden)
        seq_unpacked, lens_unpacked = pad_packed_sequence(out, batch_first=True)
        lens_unpacked -= 1
        lens = lens_unpacked.unsqueeze(-1)
        indices = lens.repeat(1, 4)
        self.indices = indices.unsqueeze(1).to(DEVICE)
        out = torch.gather(seq_unpacked, 1, self.indices)
        return self.output(out).flatten()

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        # tensorboard_logs = {'train_loss': loss}
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def dataset_generator(num_samples, max_sample_length):
    dataset = []
    gts = []
    min_sample_length = max_sample_length // 2
    sizes = (
        np.random.default_rng()
        .uniform(min_sample_length, max_sample_length, num_samples)
        .astype(int)
    )
    for size in sizes:
        sample = np.random.default_rng().standard_t(5, size=size)
        sample = (sample > 0).astype(int)
        gt = (sample.sum() % 2) == 1
        dataset.append(torch.tensor((sample * 2) - 1).unsqueeze(1).float().to(DEVICE))
        gts.append(gt * 2 - 1)
    #  dataset = torch.tensor(dataset).unsqueeze(1).permute(0, 2, 1).float()
    gts = torch.tensor(gts).float().to(DEVICE)
    return dataset, sizes.tolist(), gts


def test_ndes(sequence_length):
    data_generator = DummyDataGenerator(
        *dataset_generator(5000, sequence_length), DEVICE
    )

    net = Net().to(DEVICE)

    cost_function = F.mse_loss

    ndes = RNNnDESOptimizer(
        model=net,
        criterion=cost_function,
        data_gen=data_generator,
        budget=1000000,
        history=16,
        lower=-2,
        upper=2,
        tol=1e-6,
        worst_fitness=3,
        device=DEVICE,
        log_dir=f"rnn_parity_{sequence_length}",
    )

    best = ndes.run()


def test_adam(sequence_length):
    dataset, sizes, gts = dataset_generator(5000, sequence_length)
    ds = DummyDataset(dataset, sizes, gts)
    train_loader = DataLoader(ds, batch_size=64, collate_fn=pad_collate)

    net = Net().to(DEVICE)

    trainer = pl.Trainer(
        gpus=0,
        precision=32,
        max_epochs=100,
        early_stop_callback=EarlyStopping(monitor="loss", patience=20, min_delta=1e-3),
    )
    trainer.fit(net, train_loader)

    pack = pack_sequence(dataset, enforce_sorted=False)
    pack = pack.to(DEVICE)
    gts = gts.to(DEVICE)

    loss = F.mse_loss(net(pack), gts)
    print(loss.item())
    timestamp = timer()
    with open(f"rnn_parity_{sequence_length}/adam_result_{timestamp}.csv", "w+") as fh:
        fh.writelines(["best_found\r\n", str(loss.item())])


if __name__ == "__main__":
    sequence_length, device = parse_args()
    DEVICE = device
    #  test_adam(sequence_length)
    test_ndes(sequence_length)
