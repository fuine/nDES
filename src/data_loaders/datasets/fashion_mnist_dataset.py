import torch
from torchvision import datasets, transforms
import torch.utils.data


class FashionMNISTDataset:
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = datasets.FashionMNIST(
            "../data",
            train=True,
            download=True,
            transform=transform,
        )
        self.test_dataset = datasets.FashionMNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor()]
            ),
        )

        for x, y in torch.utils.data.DataLoader(
            self.train_dataset, batch_size=len(self.train_dataset), shuffle=True
        ):
            self.train_data, self.train_targets = x, y

    def get_train_set_targets(self):
        return torch.ones_like(self.train_dataset.targets)

    def get_train_images(self):
        return self.train_data
