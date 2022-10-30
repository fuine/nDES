import torch


def get_noise_for_nn(latent_dim, n_examples, device):
    return get_noise(n_examples, latent_dim, device)


def get_noise(x_size, y_size, device):
    return torch.randn(x_size, y_size, device=device)


class GeneratedFakeDataset:

    def __init__(self, generator, num_of_samples):
        self.generator = generator
        self.generator.eval()
        self.num_of_samples = num_of_samples
        self.train_dataset = self.generator(get_noise_for_nn(self.generator.get_latent_dim(), self.num_of_samples, self.generator.device)).detach().cpu()

    def set_generator(self, generator):
        self.generator = generator  # TODO cpu and gpu

    def get_train(self):
        return self.train_dataset

    def get_train_set_targets(self):
        return torch.zeros(self.num_of_samples)