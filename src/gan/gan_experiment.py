import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.ndes import SecondaryMutation
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset, train_via_ndes, shuffle_dataset
from src.classic.fashion_mnist_experiment import MyDatasetLoader
from src.data_loaders.datasource import show_images_from_tensor

from src.data_loaders.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_loaders.datasets.generated_fake_dataset import GeneratedFakeDataset

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 50)
EPOCHS = int(POPULATION) * 10
NDES_TRAINING = True

DEVICE = torch.device("cuda:0")
BOOTSTRAP = False
MODEL_NAME = "gan_ndes"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 64
BATCH_NUM = 50
VALIDATION_SIZE = 10000
STRATIFY = False


def show_sample_predictions(discriminator, my_data_loader_batch):
    # TODO to można podzielić na funkcje
    show_images_from_tensor(my_data_loader_batch[1][0].cpu())
    predictions = discriminator(my_data_loader_batch[1][0].to(DEVICE)).cpu()
    print(f"Loss: {criterion(predictions.cuda(), my_data_loader_batch[1][1].cuda())}")
    # print(f"Predictions: {predictions}")
    # print(f"Targets: {my_data_loader_batch[1][1]}")


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

    def get_latent_dim(self):
        return self.fc_1.in_features


class Discriminator(pl.LightningModule):
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


if __name__ == "__main__":
    seed_everything(SEED_OFFSET)

    train_loader_config = {
        'batch_size': 64,
        'shuffle': True,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 4
    }
    ndes_config = {
        'history': 16,
        'worst_fitness': 3,
        'Ft': 1,
        'ccum': 0.96,
        # 'cp': 0.1,
        'lower': -50.0,
        'upper': 50.0,
        'log_dir': "ndes_logs/",
        'tol': 1e-6,
        'budget': EPOCHS,
        'device': DEVICE
    }
    wandb.init(project="gan-nDES", entity="mmatak", config={**train_loader_config, **ndes_config})

    criterion = nn.MSELoss()

    discriminator = Discriminator(hidden_dim=256, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=256, output_dim=784).to(DEVICE)

    fashionMNIST = FashionMNISTDataset()
    train_data_real = FashionMNISTDataset().train_data

    generated_fake_dataset = GeneratedFakeDataset(generator, len(train_data_real))
    train_data_fake = generated_fake_dataset.train_dataset

    train_data_merged = torch.cat([train_data_fake, train_data_real], 0)
    train_targets_merged = torch.cat(
        [generated_fake_dataset.get_train_set_targets(), fashionMNIST.get_train_set_targets()], 0).unsqueeze(1)
    train_data_merged, train_targets_merged = shuffle_dataset(train_data_merged, train_targets_merged)
    train_loader = MyDatasetLoader(
        x_train=train_data_merged.to(DEVICE),
        y_train=train_targets_merged.to(DEVICE),
        batch_size=BATCH_SIZE
    )

    if LOAD_WEIGHTS:
        raise Exception("Not yet implemented")

    if NDES_TRAINING:
        if STRATIFY:
            raise Exception("Not yet implemented")
        if BOOTSTRAP:
            raise Exception("Not yet implemented")
        discriminator_ndes_optim = BasenDESOptimizer(
            model=discriminator,
            criterion=criterion,
            data_gen=train_loader,
            ndes_config=ndes_config,
            use_fitness_ewma=True,
            restarts=None,
            lr=0.001,
            secondary_mutation=SecondaryMutation.Gradient,
            lambda_=POPULATION,
            device=DEVICE,
        )
        # train_via_ndes(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)
        # print(discriminator(train_loader.get_sample_images_gpu()))
        show_sample_predictions(discriminator, next(iter(train_loader)))
        train_via_ndes_without_test_dataset(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)
        show_sample_predictions(discriminator, next(iter(train_loader)))
        # print(discriminator(train_loader.get_sample_images_gpu()))
        # generate noise
        # 1. teach discriminator via ndes
        # 2. teach generator via ndes


    else:
        raise Exception("Not yet implemented")
    wandb.finish()
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
