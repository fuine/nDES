from generated_fake_dataset import GeneratedFakeDataset
from fashion_mnist_dataset import FashionMNISTDataset
from src.gan.discriminator_experiment import Generator
import torch.utils.data
from src.classic.utils import shuffle_dataset
from src.data_loaders.datasource import show_images_from_tensor
from src.data_loaders.my_data_set_loader import MyDatasetLoader


if __name__ == "__main__":
    device = torch.device("cuda:0")
    latent_dim = 32
    generator = Generator(latent_dim=latent_dim, hidden_dim=256, output_dim=784).to(device)
    fashionMNIST = FashionMNISTDataset()
    train_data_real = FashionMNISTDataset().train_data

    generated_fake_dataset = GeneratedFakeDataset(generator, len(train_data_real))
    train_data_fake = generated_fake_dataset.train_dataset

    train_data_merged = torch.cat([train_data_fake, train_data_real], 0)
    train_targets_merged = torch.cat([generated_fake_dataset.get_train_set_targets(), fashionMNIST.get_train_set_targets()], 0)
    train_data_merged, train_targets_merged = shuffle_dataset(train_data_merged, train_targets_merged)
    loader = MyDatasetLoader(
        x_train=train_data_merged,
        y_train=train_targets_merged,
        batch_size=100
    )

    sample_batch = next(iter(loader))
    show_images_from_tensor(sample_batch[1][0])
    print(sample_batch[1][1].unsqueeze(1).shape)
