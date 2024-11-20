import os
import cv2
import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from abc import abstractmethod
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple

Tensor = TypeVar('torch.tensor')


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class ConditionalVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 final_img_size: int = 4,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.final_img_size = final_img_size

        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            # hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.final_img_size * self.final_img_size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.final_img_size * self.final_img_size, latent_dim)
        self.fc_state = nn.Linear(3, 3)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.final_img_size * self.final_img_size)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

        self.final_channels = hidden_dims[0]

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.final_channels, self.final_img_size, self.final_img_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        x = self.embed_data(input)

        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = (args[0] + 1) * 127.5
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = args[4]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]


def evaluate_model(model, dataloader, save_imgs_path, kld_weight=0.00025):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img = torch.FloatTensor(data).to(device)
            img = img.unsqueeze(0)

            with torch.no_grad():
                output = model(img)
            loss = model.loss_function(output[0], output[1], output[2], output[2], kld_weight)
            total_loss += loss['loss']

            reconstructed_img = (output[0].squeeze(0).cpu().numpy() + 1) * 127.5
            reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            # reconstructed_pil = Image.fromarray(reconstructed_img, "RGB")
            # reconstructed_pil.save(os.path.join(save_imgs_path, f'recon_imgs/{i}.png'))
            cv2.imwrite(os.path.join(save_imgs_path, f'recon_imgs/{i}.png'), reconstructed_img)

    avg_loss = total_loss / len(dataloader)

    return avg_loss


if __name__ == '__main__':
    ptr = 0
    size = 0
    alpha = 0.8
    lr = 0.0005
    weight_decay = 0.00025
    kld_weight = 0.00025
    seed = 1265
    batch_size = 128
    max_size = int(1e5)
    save_img_path = './results/VAE'
    save_model_path = "./checkpoints/VAE"
    dataloader = np.zeros((max_size, 3, 128, 128), dtype=np.uint8)
    device = 'cuda:0'

    # image_encoder_path = r'/home/wht/automatic-search/checkpoints/VAE/best_8.pt'

    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from automatic_search_env import AutomaticSearch, SeedEnvWrapper

    env = AutomaticSearch()
    env = SeedEnvWrapper(env, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = ConditionalVAE(
        in_channels=3,
        latent_dim=256,
        hidden_dims=None,
        img_size=128
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 1.collect training data
    for episode in tqdm(range(2471)):
        obs, info = env.reset()
        action = env.action_space.sample()
        for t in range(200):
            next_obs, reward, done, _, _ = env.step(action)

            dataloader[ptr] = obs
            ptr = int((ptr + 1) % max_size)
            size = min(size + 1, max_size)

            obs = next_obs

            if np.random.uniform(0, 1) >= alpha:
                action = env.action_space.sample()
            else:
                current_pos = np.array(env.current_pos)
                goal_pos = np.array(env.goal_pos)
                distances = np.linalg.norm(goal_pos - current_pos, axis=1)
                nearest_index = np.argmin(distances)

                distance = goal_pos[nearest_index] - current_pos

                if abs(distance[0]) > abs(distance[1]):
                    if distance[0] < 0:
                        action = 3
                    else:
                        action = 4
                else:
                    if distance[1] < 0:
                        action = 1
                    else:
                        action = 2

            if done:
                break

    index = np.random.choice(size, size=32, replace=False)
    for i, ind in enumerate(index):
        original_img = dataloader[ind].astype(np.uint8).transpose(1, 2, 0)
        # original_pil = Image.fromarray(original_img, "RGB")
        # original_pil.save(os.path.join(save_img_path, f'imgs/{i}.png'))
        cv2.imwrite(os.path.join(save_img_path, f'imgs/{i}.png'), original_img)

    print(size, len(dataloader))
    # 2.training vae model
    min_loss = 500
    for epoch in tqdm(range(35000)):
        ind = np.random.randint(50, size - 1, size=batch_size)
        imgs = torch.FloatTensor(dataloader[ind]).to(device)

        model.zero_grad()
        output = model(imgs)
        loss = model.loss_function(output[0], output[1], output[2], output[3], kld_weight)
        loss['loss'].backward()
        optimizer.step()

        if epoch % 500 == 0:
            eval_loss = evaluate_model(model, dataloader[index], save_img_path, kld_weight)

            print(
                f"train_loss:{loss['loss']}, Reconstruction_Loss:{loss['Reconstruction_Loss']}, eval_loss:{eval_loss}")

            torch.save(
                {"encoder": model.encoder,
                 "fc_state": model.fc_state,
                 "fc_mu": model.fc_mu,
                 "fc_var": model.fc_var,
                 "decoder_input": model.decoder_input,
                 "decoder": model.decoder,
                 "final_layer": model.final_layer
                 },
                os.path.join(save_model_path, f"checkpoint_{epoch}.pt")
            )

            if loss['loss'] < min_loss:
                min_loss = loss['loss']
                print(f"min_loss:{min_loss}")
                torch.save(
                    {"encoder": model.encoder,
                     "fc_state": model.fc_state,
                     "fc_mu": model.fc_mu,
                     "fc_var": model.fc_var,
                     "decoder_input": model.decoder_input,
                     "decoder": model.decoder,
                     "final_layer": model.final_layer
                     },
                    os.path.join(save_model_path, f"best_4.pt")
                )
