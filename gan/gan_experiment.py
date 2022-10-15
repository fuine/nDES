import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from ndes_optimizer import BasenDESOptimizer
from ndes import SecondaryMutation
from utils import seed_everything, train_via_ndes, stratify

from datasource import get_train_images_dataset, create_loader_for_dataset

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 200)
EPOCHS = int(POPULATION) * 10
NDES_TRAINING = True

DEVICE = torch.device("cuda:0")
BOOTSTRAP = True
MODEL_NAME = "fashion_ndes_bootstrapped"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 64
VALIDATION_SIZE = 10000
STRATIFY = True


class Generator(pl.LightningModule):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc_1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.fc_1(x))
        h = self.LeakyReLU(self.fc_2(h))

        x_hat = torch.sigmoid(self.fc_3(h))
        x_hat = x_hat.view([-1, 1, 28, 28])
        return x_hat


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.LeakyReLU(self.fc_1(x))
        x = self.LeakyReLU(self.fc_2(x))
        x = self.fc_out(x)
        return x


def create_base_ndes_parametizer(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    return BasenDESOptimizer(
        model=model,
        criterion=criterion,
        data_gen=train_loader,
        use_fitness_ewma=True,
        log_dir="ndes_logs/",
        lr=1e-3, #a cziemu?
        secondary_mutation=SecondaryMutation.Gradient
    )

if __name__ == "__main__":
    seed_everything(SEED_OFFSET)

    train_loader_config = {
        'batch_size': 64,
        'shuffle': True,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 4
    }

    train_dataset = get_train_images_dataset()
    train_loader = create_loader_for_dataset(train_dataset, **train_loader_config)

    if LOAD_WEIGHTS:
        raise Exception("Not yet implemented")

    discriminator = Discriminator(input_dim=784, hidden_dim=256)
    if NDES_TRAINING:
        if STRATIFY:
            raise Exception("Not yet implemented")
        if BOOTSTRAP:
            raise Exception("Not yet implemented")
        # generate fixed noise
        # 1. teach discriminator via ndes
        # 2. teach generator via ndes


    else:
        raise Exception("Not yet implemented")
#
# class Net(pl.LightningModule):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 32, 5)
#         self.fc1 = nn.Linear(4 * 4 * 32, 64)
#         self.fc2 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = F.softsign(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.softsign(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 32)
#         x = F.softsign(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
#
#
#     def prepare_data(self):
#         mean = (0.2860405969887955,)
#         std = (0.3530242445149223,)
#         train_dataset = datasets.FashionMNIST(
#             "../data",
#             train=True,
#             download=True,
#             transform=transforms.Compose(
#                 [transforms.ToTensor(), transforms.Normalize(mean, std)]
#             ),
#         )
#         self.test_dataset = datasets.FashionMNIST(
#             "../data",
#             train=False,
#             transform=transforms.Compose(
#                 [transforms.ToTensor(), transforms.Normalize(mean, std)]
#             ),
#         )
#
#         for x, y in torch.utils.data.DataLoader(
#             train_dataset, batch_size=len(train_dataset), shuffle=True
#         ):
#             x_train, y_train = x.to(DEVICE), y.to(DEVICE)
#
#         train_idx, val_idx = train_test_split(
#             np.arange(0, len(train_dataset)),
#             test_size=VALIDATION_SIZE,
#             stratify=y_train.cpu().numpy(),
#         )
#
#         x_val = x_train[val_idx, :]
#         y_val = y_train[val_idx]
#         x_train = x_train[train_idx, :]
#         y_train = y_train[train_idx]
#
#         self.train_dataset = TensorDataset(x_train, y_train)
#         self.val_dataset = TensorDataset(x_val, y_val)
#
#         return x_train, y_train, x_val, y_val
