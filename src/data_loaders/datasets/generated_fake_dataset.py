import torch
from src.gan.gan_experiment import Generator
from src.data_loaders.datasource import show_images_from_tensor


def get_noise_for_nn(latent_dim, n_examples, device):
    return get_noise(n_examples, latent_dim, device)


def get_noise(x_size, y_size, device):
    return torch.randn(x_size, y_size, device=device)


class GeneratedFakeDataset:

    def __init__(self, generator, num_of_samples):
        self.generator = generator
        self.num_of_samples = num_of_samples
        self.train_dataset = self.generator(get_noise_for_nn(self.generator.get_latent_dim(), self.num_of_samples, self.generator.device)).cpu()

    def set_generator(self, generator):
        self.generator = generator  # TODO cpu and gpu

    def get_train(self):
        return self.train_dataset

    def get_train_set_targets(self):
        return torch.ones(self.num_of_samples)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    latent_dim = 32
    generator = Generator(latent_dim=latent_dim, hidden_dim=256, output_dim=784).to(device)
    generated_fake_dataset = GeneratedFakeDataset(generator, 1000)
    fake_dataset = generated_fake_dataset.get_train()
    show_images_from_tensor(fake_dataset[0:10])
    print(generated_fake_dataset.get_train_set_targets().shape)