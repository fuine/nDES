import torch
from torchvision import datasets, transforms
from src.data_loaders.datasource import show_images_from_tensor


class FashionMNISTDataset:
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = datasets.FashionMNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transform]
            ),
        )
        self.test_dataset = datasets.FashionMNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToTensor()]
            ),
        )

    def get_train_set_target(self):
        return torch.ones_like(self.train_dataset.targets)

    def get_train_images_displayable(self):
        return torch.unsqueeze(self.train_dataset.data, 1)

#
# if __name__ == "__main__":
#     fashionMNIST = FashionMNISTDataset()
#     # print(tensor_to_list_of_images(fashionMNIST.get_train_set()))
#     result = fashionMNIST.get_train_images_displayable()
#     # 64, 1, 28, 28
#     show_images_from_tensor(result[0:10])
