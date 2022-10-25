import torch

# from src.gan.gan_experiment import Generator
# from src.data_loaders.datasource import show_images_from_tensor
# from src.gan.gan_experiment import Discriminator


class GeneratorDataLoader:
    def __init__(self, generator, batch_size, device, num_batches):
        self.generator = generator
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device
        self.i = 0

    def __iter__(self):
        return self

    def set_generator(self, generator):
        self.generator = generator  # TODO cpu and gpu

    def __next__(self):
        # wyniki zwracane są na cuda
        # brzydkie
        self.generator.zero_grad()
        i = self.i
        self.i += 1
        output = i, (self.generator(self.get_noise_for_nn(self.generator.get_latent_dim(), self.batch_size, self.generator.device)),
                     self.get_classes_for_images_gpu())
        if i == self.num_batches - 1:
            self.i = 0
            return output
        else:
            return output

    def __len__(self):
        return self.num_batches  # TODO check with ewma logger

    def get_noise_for_nn(self, latent_dim, n_examples, device):
        return self.get_noise(n_examples, latent_dim, device)

    def get_classes_for_images_gpu(self):
        return torch.zeros(self.batch_size, 1).to(self.device)

    def get_sample_images_gpu(self):
        return next(iter(self))[1][0]

    @staticmethod
    def get_noise(x_size, y_size, device):
        return torch.randn(x_size, y_size, device=device)


# if __name__ == "__main__":
#     device = torch.device("cuda:0")
#     latent_dim = 32
#     generator = Generator(latent_dim=latent_dim, hidden_dim=256, output_dim=784).to(device)
#     loader = GeneratorDataLoader(generator, 25, device, num_batches=20)
#     for batch_idx, (b_x, y) in loader:
#         print(batch_idx)
#     sample_images_gpu = loader.get_sample_images_gpu()
#     show_images_from_tensor(sample_images_gpu.cpu())

    # fake = generator(get_noise_for_nn(latent_dim, 25, device)).detach().cpu()