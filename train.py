import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from model import Denoiser, Diffusion

# Model Hyperparameters

dataset_path = '~/datasets'

cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")

dataset = 'MNIST'
img_size = (32, 32, 3) if dataset == "CIFAR10" else (28, 28, 1)  # (width, height, channels)

timestep_embedding_dim = 256
n_layers = 8
hidden_dim = 256
n_timesteps = 1000
beta_minmax = [1e-4, 2e-2]

train_batch_size = 128
inference_batch_size = 64
lr = 5e-5
epochs = 100

seed = 1234

hidden_dims = [hidden_dim for _ in range(n_layers)]
torch.manual_seed(seed)
np.random.seed(seed)


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if dataset == 'CIFAR10':
        train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
        test_dataset = CIFAR10(dataset_path, transform=transform, train=False, download=True)
    else:
        train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
        test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    model = Denoiser(image_resolution=img_size,
                     hidden_dims=hidden_dims,
                     diffusion_time_embedding_dim=timestep_embedding_dim,
                     n_times=n_timesteps).to(DEVICE)
    print(f"Model has {count_parameters(model):,} trainable parameters")
    diffusion = Diffusion(model, image_resolution=img_size, n_times=n_timesteps,
                          beta_minmax=beta_minmax, device=DEVICE).to(DEVICE)

    optimizer = Adam(diffusion.parameters(), lr=lr)
    denoising_loss = nn.MSELoss()

    train_loader, _ = load_dataset()

    print("Start training DDPMs...")
    model.train()

    for epoch in range(epochs):
        noise_prediction_loss = 0
        for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            x = x.to(DEVICE)

            noisy_input, epsilon, pred_epsilon = diffusion(x)
            loss = denoising_loss(pred_epsilon, epsilon)

            noise_prediction_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "complete!", "\t Denoising Loss: ", noise_prediction_loss / batch_idx)
    print("Finish!!")
    # Save the model
    torch.save(diffusion.state_dict(), "diffusion.pt")
    print("Model saved!")


def draw_sample_image(x, postfix):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(f"{postfix}.png")


def sample():
    model = Denoiser(image_resolution=img_size,
                     hidden_dims=hidden_dims,
                     diffusion_time_embedding_dim=timestep_embedding_dim,
                     n_times=n_timesteps).to(DEVICE)
    model.eval()
    diffusion = Diffusion(model, image_resolution=img_size, n_times=n_timesteps,
                          beta_minmax=beta_minmax, device=DEVICE).to(DEVICE)

    diffusion.load_state_dict(torch.load("diffusion.pt"))
    diffusion.eval()

    with torch.no_grad():
        generated_images = diffusion.sample(N=inference_batch_size)

    _, test_loader = load_dataset()

    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.to(DEVICE)
        perturbed_images, epsilon, pred_epsilon = diffusion(x)
        perturbed_images = diffusion.reverse_scale_to_zero_to_one(perturbed_images)
        break
    draw_sample_image(perturbed_images, "Perturbed Images")
    draw_sample_image(generated_images, "Generated Images")
    draw_sample_image(x[:inference_batch_size], "Ground-truth Images")
    print("Sample complete!")


if __name__ == "__main__":
    train()
    sample()
