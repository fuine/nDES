import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import transforms

# from gan_experiment import Generator
import torch.utils.data


def get_noise_for_nn(latent_dim, n_examples, device):
    return get_fixed_noise(n_examples, latent_dim, device)


def get_fixed_noise(x_size, y_size, device):
    return torch.randn(x_size, y_size, device=device)


def show_images_from_tensor(images, n_row=8):
    grid = torchvision.utils.make_grid(images, nrow=n_row)
    grid = grid.permute(1, 2, 0)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


def get_train_images_dataset():
    from torchvision.datasets import FashionMNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = FashionMNIST(root="data/", train=True, transform=transform, download=True)
    return train_dataset


def create_loader_for_dataset(dataset, **kwargs):
    return torch.utils.data.DataLoader(
        dataset,
        **kwargs
    )


def get_batch_from_data_loader(data_loader):
    return next(iter(data_loader))[0]


if __name__ == "__main__":
    # device = torch.device("cuda:0")
    # latent_dim = 32
    #
    # train_loader_config = {
    #     'batch_size': 64,
    #     'shuffle': True,
    #     'drop_last': True,
    #     'pin_memory': True,
    #     'num_workers': 4
    # }
    #
    # train_dataset = get_train_images_dataset()
    # train_loader = create_loader_for_dataset(train_dataset, **train_loader_config)
    #
    # sample_images = get_batch_from_data_loader(train_loader)
    # show_images_from_tensor(sample_images)
    #
    # generator = Generator(latent_dim=latent_dim, hidden_dim=256, output_dim=784).to(device)
    # fake = generator(get_noise_for_nn(latent_dim, 25, device)).detach().cpu()
    # show_images_from_tensor(fake)
    sth = (5,)
    print(type(sth))
    print(int((5)))