from generated_fake_dataset import GeneratedFakeDataset
from fashion_mnist_dataset import FashionMNISTDataset
from src.gan.gan_experiment import Generator
import torch.utils.data
from src.data_loaders.datasource import show_images_from_tensor


if __name__ == "__main__":
    device = torch.device("cuda:0")
    latent_dim = 32
    generator = Generator(latent_dim=latent_dim, hidden_dim=256, output_dim=784).to(device)
    fashionMNIST = FashionMNISTDataset()
    train_real = FashionMNISTDataset().train_data

    generated_fake_dataset = GeneratedFakeDataset(generator, len(train_real))
    train_fake = generated_fake_dataset.train_dataset

    merged_datasets = torch.cat([train_fake, train_real], 0)

    print(merged_datasets.shape)
    train_loader_config = {
        'batch_size': 64,
        'shuffle': True,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 4
    }
    loader = torch.utils.data.DataLoader(merged_datasets, **train_loader_config)
    sample_batch = next(iter(loader))
    show_images_from_tensor(sample_batch)